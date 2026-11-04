"""Derive's statistics in groups and fit, without a core: moments against raw arrays.

A live route (events one by one) and a sealed route (one whole dataset) are the
same increments, so their groups agree to rounding. ``ZLC_DERIVE_FILE`` names a
real dataset to do that on (``data.npz`` with ``info``, ``data``, ``data.validity``).
Runs directly with Python; ``ctx.fit`` is a small numpy fit with the same signature.
"""
import json
import os
import pathlib
import sys
import unittest
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
import zlc_plugin.fit
from zlc_data import DatasetSchema, DomainSpec, SCALAR_DOMAIN, ValidityContract, ValueSchema, owned_snapshot_from_arrays
from zlc_data.axis import REPEAT, SCAN_POINT, SITE, AxisId, AxisRoleId, AxisSpec
from zlc_data.codec import dataset_schema_from_tree
from zlc_atom.nodes.derive.expression import Operand, evaluate, typed_variables, published
from zlc_atom.nodes.derive.groups import restricted
from zlc_atom.nodes.derive.logic_node import DERIVE_SCHEMA, _ROWS
from zlc_atom.nodes.derive.editor import derive_editor_facts
from zlc_atom.nodes.derive.processor import DeriveProcessor

zlc_plugin.fit._MODELS = {'line': {'id': 'line', 'label': 'Line', 'parameters': [
    {'name': 'slope', 'unit_relation': 'inverse_axis_0'}, {'name': 'offset', 'unit_relation': 'value'}]}}


def fake_fit(model, x, y, sigma=None, warm=None):
    """A straight line per row; a row with under three points fails."""
    assert model == 'line' and y.shape[1] == x.size and (sigma is None or sigma.shape == y.shape)
    cells = y.shape[0]
    values, errors, success = np.full((cells, 2), np.nan), np.full((cells, 2), np.nan), np.zeros(cells, bool)
    for cell in range(cells):
        keep = np.isfinite(y[cell])
        if keep.sum() < 3:
            continue
        (values[cell], covariance) = np.polyfit(x[keep], y[cell][keep], 1, cov='unscaled')
        errors[cell] = np.sqrt(np.diag(covariance))
        success[cell] = True
    return types.SimpleNamespace(values=values, errors=errors, success=success, warm=warm)


class Context:
    def __init__(self): self.warms = []
    def fit(self, model, x, y, sigma=None, warm=None):
        self.warms.append(warm)
        return fake_fit(model, x, y, sigma, warm)


def rows_of(*variables):
    return DERIVE_SCHEMA.project_values({'variables': list(variables)})['variables']


def schema_of(repeats, points, cells, *, dtype=np.float64, time=False):
    """repeat (with a time beside it) x duration x Site (and, with two cells, pixel)."""
    repeat = AxisSpec(AxisId('scan.repeat'), 'repeat', REPEAT, repeats, coordinates=np.arange(repeats))
    axes = [repeat]
    if time:
        axes.append(AxisSpec(AxisId('scan.time'), 'time', REPEAT, repeats, coordinates=np.arange(repeats) * 0.5, unit='s',
                             coordinate_of=repeat.axis_id))
    repeat_domain = DomainSpec((repeats,), tuple(axes), tuple(range(repeats) for _axis in axes))
    duration = AxisSpec(AxisId('scan.duration'), 'duration', SCAN_POINT, points, coordinates=np.arange(points) * 2.0 + 1.0, unit='µs')
    point_domain = DomainSpec((points,), (duration,), (range(points),))
    site = tuple(AxisSpec(AxisId(f'site{k}'), f'Site{k}', SITE, size, coordinates=np.arange(size)) for k, size in enumerate(cells))
    cell_domain = DomainSpec(tuple(cells), site)
    return DatasetSchema(repeat_domain, point_domain, cell_domain,
                         ValueSchema(ValidityContract.components(*(axis.axis_id for axis in site)), np.dtype(dtype), 'V', name='counts'))


class Value:
    """A delivered member, as ``signal_value`` makes it."""

    def __init__(self, schema, values, valid, canonical, origin, ordinal=None):
        self.snapshot = owned_snapshot_from_arrays(schema, values, 1, validity=valid, stream_generation='source')
        self.canonical_schema, self.cell_origin = canonical, origin
        self.key, self.name, self.event_record = {'ordinal': ordinal}, '@logic/source/counts', None


def live_route(schema, values, valid, *, member='counts', order=None, block=1):
    """One delivery per (repeat, point block), in row order unless ``order`` says."""
    repeats, points = schema.repeat_domain.size, schema.point_domain.size
    places = [(r, p) for r in range(repeats) for p in range(0, points, block)]
    if order is not None:
        places = order(places)
    for ordinal, (r, p) in enumerate(places):
        size = min(block, points - p)
        event = restricted(schema, (r, p), (1, size))
        yield {member: Value(event, values[r:r + 1, p:p + size], valid[r:r + 1, p:p + size], schema, (r, p), ordinal)}


def sealed_route(schema, values, valid, *, member='counts'):
    yield {member: Value(schema, values, valid, schema, (0, 0))}


class Written:
    """What Runtime holds of each finite output: every placement write-once."""

    def __init__(self): self.planes, self.declared, self.commits, self.chunks = {}, {}, 0, {}

    def take(self, commit):
        self.commits += 1
        for name, output in commit.items():
            if output.canonical_schema is None:
                self.chunks.setdefault(name, []).append(output)
                continue
            canonical = output.canonical_schema
            shape = canonical.physical_shape
            plane = self.planes.setdefault(name, {
                'values': np.full(shape, np.nan), 'valid': np.zeros(shape, bool), 'sigma': np.full(shape, np.nan), 'written': np.zeros(shape[:2], bool)})
            block = output.snapshot.materialize().block
            r, p = output.cell_origin
            rows, columns = block.schema.repeat_domain.size, block.schema.point_domain.size
            assert block.schema.cell_domain == canonical.cell_domain and block.schema.value_schema == canonical.value_schema
            assert not plane['written'][r:r + rows, p:p + columns].any(), f'{name} wrote a cell twice'
            plane['written'][r:r + rows, p:p + columns] = True
            plane['values'][r:r + rows, p:p + columns] = block.values
            plane['valid'][r:r + rows, p:p + columns] = output.snapshot.expanded_validity()
            if block.sigma is not None:
                plane['sigma'][r:r + rows, p:p + columns] = block.sigma


def run(variables, deliveries, context=None):
    processor = DeriveProcessor(variables=variables, primary_output='counts')
    processor.bind_run(context or Context())
    written = Written()
    each = []
    for inputs in deliveries:
        commits = processor.evaluate_inputs(inputs)
        each.append(len(commits))
        for commit in commits:
            written.take(commit)
    for commit in processor.finish_inputs():
        written.take(commit)
    return written, each, processor


def direct(values, valid, group, axis_over, operation):
    """The statistic over raw arrays: ``group`` per row of axis 0 or 1 by ``axis_over``."""
    source = values.astype(np.float64) if values.dtype.kind != 'b' else values
    answer = {}
    for out in np.unique(group):
        chosen = np.compress(group == out, source, axis=axis_over[0])
        kept = np.compress(group == out, valid, axis=axis_over[0])
        answer[int(out)] = _reduced(chosen, kept, axis_over[1:], axis_over[0], operation)
    return answer


def _reduced(values, valid, whole, by_axis, operation):
    axes = tuple(sorted((by_axis, *whole)))
    count = valid.sum(axis=axes)
    ok = count > 0
    numeric = np.where(valid, values, 0).astype(np.float64)
    total = numeric.sum(axis=axes)
    mean = np.divide(total, count, out=np.full(count.shape, np.nan), where=ok)
    if operation == 'count':
        return count, ok, None
    if operation == 'sum':
        return total, ok, None
    if operation == 'mean' or operation == 'std':
        expanded = np.expand_dims(mean, axes)
        spread = np.where(valid, (values.astype(np.float64) - expanded) ** 2, 0).sum(axis=axes)
        if operation == 'std':
            return np.sqrt(np.divide(spread, count, out=np.full(count.shape, np.nan), where=ok)), ok, None
        sem = np.sqrt(np.divide(spread, count * (count - 1.0), out=np.full(count.shape, np.nan), where=count > 1))
        return mean, ok, sem
    if operation == 'min':
        return np.where(valid, values, np.inf).min(axis=axes), ok, None
    if operation == 'max':
        return np.where(valid, values, -np.inf).max(axis=axes), ok, None
    if operation == 'any':
        return np.where(valid, values, False).any(axis=axes), ok, None
    return np.where(valid, values, True).all(axis=axes), ok, None


def close(left, right, valid):
    scale = max(1.0, float(np.nanmax(np.abs(right[valid])))) if valid.any() else 1.0
    np.testing.assert_allclose(left[valid].astype(np.float64), right[valid].astype(np.float64), rtol=0, atol=2e-11 * scale)


class Moments(unittest.TestCase):
    def test_every_zero_is_todays_processor(self):
        """No groups, no fit: one mapping per delivery, no sigma, and the event's own schema."""
        schema = schema_of(4, 3, (5,))
        rng = np.random.default_rng(1)
        values, valid = rng.normal(size=schema.physical_shape), rng.random(schema.physical_shape) > 0.2
        rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['Site0']})
        processor = DeriveProcessor(variables=rows, primary_output='counts')
        self.assertFalse(hasattr(processor.dataset_output_declarations[0], 'storage'))
        for inputs in live_route(schema, values, valid):
            outputs = processor.evaluate_inputs(inputs)
            self.assertIsInstance(outputs, dict)
            output = outputs['a']
            self.assertIsNone(output.canonical_schema)
            self.assertIsNone(output.snapshot.block.sigma)
            value = inputs['counts']
            event = Operand(value.snapshot.block.schema, value.snapshot.materialize().block.values, value.snapshot.expanded_validity())
            np.testing.assert_array_equal(output.snapshot.materialize().block.values, evaluate(rows, {'counts': event})['a'].values)
        self.assertEqual(processor.run_outputs(), {})
        self.assertEqual(processor.finish_inputs(), ())

    def test_randomized_against_raw_arrays(self):
        rng = np.random.default_rng(20261104)
        for trial in range(120):
            repeats, points = int(rng.integers(1, 13)), int(rng.integers(1, 9))
            cells = tuple(int(size) for size in rng.integers(1, 5, size=int(rng.integers(1, 3))))
            operation = str(rng.choice(('mean', 'sum', 'count', 'min', 'max', 'std', 'any', 'all')))
            boolean = operation in ('any', 'all') or (operation in ('mean', 'sum', 'count') and rng.random() < 0.3)
            schema = schema_of(repeats, points, cells, dtype=bool if boolean else np.float64, time=bool(rng.integers(2)))
            values = rng.random(schema.physical_shape) > 0.5 if boolean else rng.normal(3.0, 2.0, schema.physical_shape)
            valid = rng.random(schema.physical_shape) > float(rng.choice((0.0, 0.3, 0.8)))
            by_repeat = bool(rng.integers(2))
            by = 'repeat' if by_repeat else 'duration'
            size = repeats if by_repeat else points
            every = int(rng.integers(1, size + 2))
            over = [by]
            if rng.random() < 0.5:
                over.append('Site0')
            other_point = by_repeat and rng.random() < 0.3
            if other_point:
                over.append('duration')
            rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': operation, 'over': over, 'by': by, 'every': every})
            with self.subTest(trial=trial, operation=operation, by=by, every=every, shape=schema.physical_shape, over=over):
                axes = {'repeat': 0, 'duration': 1, 'Site0': 2}
                d = axes[by]
                whole = tuple(axes[name] for name in over if name != by)
                group = np.arange(size) // every
                expected = direct(values, valid, group, (d, *whole), operation)
                for route, name in ((live_route(schema, values, valid), 'live'), (sealed_route(schema, values, valid), 'sealed')):
                    written, _each, _processor = run(rows, route)
                    plane = written.planes['a']
                    self.assertTrue(plane['written'].all(), name)
                    for out, (want, ok, sem) in expected.items():
                        cut = (slice(None),) * d + (out,)
                        got, got_valid = plane['values'][cut], plane['valid'][cut]
                        # The cell's reductions leave a placeholder dimension when no cell is left.
                        want, ok = np.reshape(want, got.shape), np.reshape(ok, got.shape)
                        np.testing.assert_array_equal(got_valid, ok, name)
                        close(got, want, ok)
                        if sem is not None and not boolean:
                            finite = ok & np.isfinite(np.reshape(sem, got.shape))
                            close(plane['sigma'][cut], np.reshape(sem, got.shape), finite)

    def test_axis_stays_with_mean_coordinates(self):
        schema = schema_of(7, 2, (3,), time=True)
        values, valid = np.ones(schema.physical_shape), np.ones(schema.physical_shape, bool)
        rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['repeat', 'Site0'], 'by': 'repeat', 'every': 3})
        answer = typed_variables(rows, {'counts': schema})[0]
        grouped = answer['schema'].repeat_domain
        by_name = {axis.name: axis for axis in grouped.axes}
        np.testing.assert_allclose(np.asarray(by_name['repeat']), [1.0, 4.0, 6.0])
        np.testing.assert_allclose(np.asarray(by_name['time']), [0.5, 2.0, 3.0])
        self.assertEqual(by_name['time'].unit, 's')
        self.assertEqual(by_name['time'].coordinate_of, by_name['repeat'].axis_id)
        self.assertNotEqual(by_name['repeat'].axis_id, AxisId('scan.repeat'))
        self.assertEqual(answer['schema'].value_schema.value_unit, 'V')

    def test_post_selection_rows_read_as_before(self):
        from zlc_atom.nodes.occupancy.post_selection import _MASKED, _rows
        frame = AxisSpec(AxisId('frame'), 'frame', AxisRoleId('readout-event'), 2, coordinates=np.arange(2))
        site = AxisSpec(AxisId('site'), 'Site', SITE, 4, coordinates=np.arange(4))
        def schema(dtype, name):
            return DatasetSchema(DomainSpec((1,), (), ()), DomainSpec((2,), (frame,), (range(2),)), DomainSpec((4,), (site,)),
                                 ValueSchema(ValidityContract.components(site.axis_id), np.dtype(dtype), '1', name=name))
        rng = np.random.default_rng(8)
        occupied = rng.random((1, 2, 4)) > 0.5
        members = {member: Operand(schema(bool, member), occupied if member == 'occupied' else rng.random((1, 2, 4)) > 0.5) for member in _MASKED}
        results = evaluate(_rows('frame', ('0', '1'), 'relation', '>'), members)
        kept = occupied[0, 0] & ~occupied[0, 1]
        self.assertEqual(set(results), {'kept', *(f'{member}_kept' for member in _MASKED)})
        for member in _MASKED:
            self.assertIsNone(results[f'{member}_kept'].sigma)
            np.testing.assert_array_equal(results[f'{member}_kept'].valid[0], np.tile(kept, (2, 1)))
            np.testing.assert_array_equal(results[f'{member}_kept'].values, members[member].values)

    def test_cell_group_closes_inside_one_event(self):
        schema = schema_of(2, 2, (7, 3))
        rng = np.random.default_rng(2)
        values, valid = rng.normal(size=schema.physical_shape), rng.random(schema.physical_shape) > 0.3
        rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['Site0', 'Site1'], 'by': 'Site0', 'every': 3})
        event = Operand(schema, values, valid)
        result = evaluate(rows, {'counts': event})['a']
        self.assertEqual(result.values.shape, (2, 2, 3))
        for out, chosen in enumerate((range(0, 3), range(3, 6), range(6, 7))):
            cut = (slice(None), slice(None), list(chosen))
            count = valid[cut].sum(axis=(2, 3))
            mean = np.where(valid[cut], values[cut], 0).sum(axis=(2, 3)) / np.where(count, count, 1)
            np.testing.assert_allclose(result.values[..., out][count > 0], mean[count > 0])
        # Every other statistic here: a group is the same Operand path, and nothing waits.
        written, each, _processor = run(rows, live_route(schema, values, valid))
        self.assertEqual(each, [1] * 4)

    def test_groups_close_at_full_later_and_seal(self):
        schema = schema_of(5, 2, (3,))
        rng = np.random.default_rng(3)
        values, valid = rng.normal(size=schema.physical_shape), np.ones(schema.physical_shape, bool)
        rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['repeat', 'Site0'], 'by': 'repeat', 'every': 2})
        written, each, _processor = run(rows, live_route(schema, values, valid))
        # Full at the 4th and 8th of ten deliveries; the last group has one repeat to expect.
        self.assertEqual(each, [0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertEqual(written.commits, 3)
        # A source that sealed early: the seal gives the partial group, n smaller.
        cut = [(r, p) for r in range(5) for p in (0, 1)][:7]
        written, each, _processor = run(rows, live_route(schema, values, valid, order=lambda _all: cut))
        self.assertEqual(each, [0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(written.commits, 2)
        self.assertEqual(written.planes['a']['written'][:, 0].tolist(), [True, True, False])
        np.testing.assert_allclose(written.planes['a']['values'][1, 1, 0], values[2, 1].mean())
        # A missing repeat: the next one closes the group it left open, n smaller.
        places = [(r, p) for r in (0, 2, 3, 4) for p in (0, 1)]
        written, each, _processor = run(rows, live_route(schema, values, valid, order=lambda _all: places))
        self.assertEqual(each, [0, 0, 1, 0, 0, 1, 0, 1])
        np.testing.assert_allclose(written.planes['a']['values'][0], values[0].mean(axis=-1)[None].reshape(2, -1)[:, :])
        # Repeats run in order, and a sample after its group closed is refused loudly.
        backwards = [(2, 0), (2, 1), (0, 0)]
        with self.assertRaisesRegex(ValueError, 'after its group closed'):
            run(rows, live_route(schema, values, valid, order=lambda _all: backwards))

    def test_a_shuffled_point_scan_waits(self):
        schema = schema_of(2, 6, (2,))
        rng = np.random.default_rng(4)
        values, valid = rng.normal(size=schema.physical_shape), np.ones(schema.physical_shape, bool)
        rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'sum', 'over': ['duration'], 'by': 'duration', 'every': 3})
        shuffled = lambda places: [places[k] for k in rng.permutation(len(places))]
        written, _each, _processor = run(rows, live_route(schema, values, valid, order=shuffled))
        sealed, each, _processor = run(rows, sealed_route(schema, values, valid))
        self.assertEqual(each, [1])
        self.assertEqual(sealed.commits, 1)
        close(written.planes['a']['values'], sealed.planes['a']['values'], np.ones(written.planes['a']['values'].shape, bool))
        np.testing.assert_allclose(written.planes['a']['values'][1, 0, 1], values[1, :3, 1].sum())

    def test_helper_and_pick_and_where(self):
        """A mask helper, a pick that drops a Point axis' rows: placed as typing placed them."""
        repeats, frames, sites = 4, 3, 2
        repeat = AxisSpec(AxisId('scan.repeat'), 'repeat', REPEAT, repeats, coordinates=np.arange(repeats))
        frame = AxisSpec(AxisId('frame'), 'frame', AxisRoleId('readout-event'), frames, coordinates=np.arange(frames))
        site = AxisSpec(AxisId('site'), 'Site', SITE, sites, coordinates=np.arange(sites))
        schema = DatasetSchema(DomainSpec((repeats,), (repeat,), (range(repeats),)), DomainSpec((frames,), (frame,), (range(frames),)),
                               DomainSpec((sites,), (site,)), ValueSchema(ValidityContract.components(site.axis_id), np.dtype(float), 'V', name='counts'))
        rng = np.random.default_rng(5)
        values, valid = rng.normal(size=schema.physical_shape), np.ones(schema.physical_shape, bool)
        rows = rows_of(
            {'name': 'mask', 'publish': False, 'terms': [{'of': 'counts'}, {'op': '>', 'number': 0.0}]},
            {'name': 'a', 'terms': [{'of': 'counts', 'at': 'frame', 'labels': ['1']}, {'op': 'where', 'of': 'mask', 'at': 'frame', 'labels': ['1']}],
             'statistic': 'mean', 'over': ['repeat', 'Site'], 'by': 'repeat', 'every': 2})
        live, _each, _processor = run(rows, live_route(schema, values, valid, block=frames))
        sealed, _each, _processor = run(rows, sealed_route(schema, values, valid))
        plane = live.planes['a']
        self.assertEqual(plane['values'].shape, (2, 1, 1))
        close(plane['values'], sealed.planes['a']['values'], plane['valid'])
        chosen = values[:2, 1]
        np.testing.assert_allclose(plane['values'][0, 0, 0], chosen[chosen > 0].mean())

    def test_monitor_one_latest_event_per_group(self):
        repeat = AxisSpec(AxisId('shot'), 'shot', REPEAT, 1, coordinates=np.arange(1))
        site = AxisSpec(AxisId('site'), 'Site', SITE, 3, coordinates=np.arange(3))
        event = DatasetSchema(DomainSpec((1,), (repeat,), (range(1),)), DomainSpec((1,), (), ()),
                              DomainSpec((3,), (site,)), ValueSchema(ValidityContract.components(site.axis_id), np.dtype(float), 'V', name='counts'))
        rng = np.random.default_rng(6)
        values = rng.normal(size=(7, 1, 1, 3))
        deliveries = [{'counts': Value(event, values[k], np.ones((1, 1, 3), bool), None, None, k)} for k in range(7)]
        rows = rows_of({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['shot'], 'by': 'shot', 'every': 3})
        written, each, processor = run(rows, deliveries)
        self.assertEqual(each, [0, 0, 1, 0, 0, 1, 0])
        outputs = written.chunks['a']
        self.assertEqual([output.ordinal for output in outputs], [0, 1, 2])
        schema, storage, _index = processor.run_outputs()['a']
        self.assertEqual(storage, 'latest')
        for k, output in enumerate(outputs):
            self.assertEqual(output.snapshot.block.schema, schema)
            np.testing.assert_allclose(output.snapshot.materialize().block.values[0, 0], values[3 * k:3 * k + 3, 0, 0].mean(axis=0))


class Fit(unittest.TestCase):
    def setUp(self):
        self.schema = schema_of(6, 8, (3,))
        rng = np.random.default_rng(7)
        x = np.arange(8) * 2.0 + 1.0
        self.values = 2.0 * x[None, :, None] + rng.normal(0, 0.01, self.schema.physical_shape)
        self.valid = rng.random(self.schema.physical_shape) > 0.1
        self.rows = rows_of(
            {'name': 'a', 'publish': False, 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['repeat'], 'by': 'repeat', 'every': 2},
            {'name': 'rabi', 'terms': [{'of': 'a'}], 'statistic': 'fit', 'over': ['duration'], 'model': 'line'},
            {'name': 'twice', 'terms': [{'of': 'rabi_slope'}, {'op': '*', 'number': 2.0}]})

    def test_typing_answers_before_data(self):
        answers = typed_variables(self.rows, {'counts': self.schema})
        self.assertIsNone(answers[1]['error'])
        self.assertEqual(tuple(answers[1]['fit']), ('rabi_slope', 'rabi_offset'))
        self.assertEqual(answers[1]['fit']['rabi_slope'].value_schema.value_unit, 'MHz')
        self.assertEqual(answers[1]['fit']['rabi_offset'].value_schema.value_unit, 'V')
        self.assertEqual(answers[1]['fit']['rabi_offset'].repeat_domain.size, 3)
        self.assertEqual(answers[2]['schema'].value_schema.value_unit, 'MHz')
        self.assertEqual(published(self.rows), ('rabi_slope', 'rabi_offset', 'twice'))

    def test_fit_per_group_once_each_and_routes_agree(self):
        live_context, sealed_context = Context(), Context()
        live, _each, _p = run(self.rows, live_route(self.schema, self.values, self.valid), live_context)
        sealed, _each, _p = run(self.rows, sealed_route(self.schema, self.values, self.valid), sealed_context)
        self.assertEqual(len(live_context.warms), 3)
        self.assertEqual(len(sealed_context.warms), 3)
        self.assertIsNone(live_context.warms[0])
        self.assertEqual(live_context.warms[1].shape, (3, 2))
        for name in ('rabi_slope', 'rabi_offset', 'twice'):
            close(live.planes[name]['values'], sealed.planes[name]['values'], live.planes[name]['valid'])
        np.testing.assert_allclose(live.planes['rabi_slope']['values'], 2.0, atol=0.05)
        self.assertTrue(np.isfinite(live.planes['rabi_slope']['sigma']).all())
        np.testing.assert_allclose(live.planes['twice']['values'], 4.0, atol=0.1)
        # Arithmetic has no rule for a sigma.
        self.assertTrue(np.isnan(live.planes['twice']['sigma']).all())

    def test_failed_cell_is_invalid_and_others_stand(self):
        valid = self.valid.copy()
        valid[:2, :, 1] = False
        written, _each, _p = run(self.rows, sealed_route(self.schema, self.values, valid))
        plane = written.planes['rabi_slope']
        self.assertFalse(plane['valid'][0, 0, 1])
        self.assertTrue(plane['valid'][0, 0, 0])
        self.assertTrue(plane['valid'][1, 0, 1])

    def test_an_undetermined_parameter_is_invalid_alone(self):
        class Undetermined(Context):
            def fit(self, model, x, y, sigma=None, warm=None):
                answer = fake_fit(model, x, y, sigma, warm)
                answer.errors[:, 0] = np.nan
                return answer
        written, _each, _p = run(self.rows, sealed_route(self.schema, self.values, self.valid), Undetermined())
        self.assertFalse(written.planes['rabi_slope']['valid'].any())
        self.assertTrue(written.planes['rabi_offset']['valid'].all())

    def test_a_fit_on_a_whole_dataset_needs_no_groups(self):
        schema = schema_of(1, 8, (3,))
        rows = rows_of({'name': 'rabi', 'terms': [{'of': 'counts'}], 'statistic': 'fit', 'over': ['duration'], 'model': 'line'})
        outputs = None
        processor = DeriveProcessor(variables=rows, primary_output='counts')
        processor.bind_run(Context())
        values, valid = self.values[:1], self.valid[:1]
        commits = processor.evaluate_inputs(next(sealed_route(schema, values, valid)))
        outputs = commits[0]
        self.assertEqual(set(outputs), {'rabi_slope', 'rabi_offset'})
        self.assertIsNone(outputs['rabi_slope'].canonical_schema)

    def test_refusals(self):
        schema = self.schema
        def refused(*variables, changing=None):
            rows = _ROWS.draft_values({'variables': list(variables)})['variables']
            answers = typed_variables(rows, {'counts': schema}, changing)
            return [str(answer['error']) for answer in answers if answer['error'] is not None]
        fit = {'terms': [{'of': 'counts'}], 'statistic': 'fit', 'model': 'line'}
        self.assertIn('exactly one axis', refused({'name': 'r', **fit, 'over': ['duration', 'repeat']})[0])
        self.assertIn('no fit model', refused({'name': 'r', **fit, 'model': 'nope', 'over': ['duration']})[0])
        self.assertIn('choose a model', refused({'name': 'r', **fit, 'model': '', 'over': ['duration']})[0])
        self.assertIn('no groups', refused({'name': 'r', **fit, 'over': ['duration'], 'by': 'duration', 'every': 2})[0])
        self.assertIn('not among', refused({'name': 'a', 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['Site0'], 'by': 'repeat', 'every': 2})[0])
        changing = {axis.axis_id: () for axis in schema.point_domain.axes}
        self.assertIn('changes with each publication', refused({'name': 'r', **fit, 'over': ['duration']}, changing=changing)[0])
        group = {'name': 'a', 'publish': False, 'terms': [{'of': 'counts'}], 'statistic': 'mean', 'over': ['repeat'], 'by': 'repeat', 'every': 2}
        self.assertIn('closes group by group', refused(group, {'name': 'r', **fit, 'terms': [{'of': 'a'}], 'over': ['repeat']})[0])
        self.assertIn('closed groups and a publication', refused(group, {'name': 'b', 'terms': [{'of': 'a'}, {'op': '+', 'of': 'counts'}]})[0])
        other = {**group, 'name': 'c', 'every': 3}
        self.assertTrue(any('different times' in text for text in refused(group, other, {'name': 'b', 'terms': [{'of': 'a'}, {'op': '+', 'of': 'c'}]})))

    def test_editor_facts(self):
        from zlc_data.codec import dataset_schema_to_tree
        bundle = [{'name': 'counts', 'schema': dataset_schema_to_tree(self.schema), 'event_schema': None}]
        facts = derive_editor_facts(self.rows, '', bundle, None)
        cards = facts['cards']
        self.assertEqual(cards[0]['statistic']['by'], 'repeat')
        self.assertEqual(cards[0]['statistic']['every'], 2)
        self.assertIn('each group of 2 repeat', cards[0]['preview'])
        self.assertEqual(cards[0]['statistic']['models'], [])
        statistic = cards[1]['statistic']
        self.assertEqual(statistic['model'], 'line')
        self.assertEqual(statistic['models'], [['line', 'Line']])
        self.assertEqual(statistic['publishes'], [['slope', 'MHz'], ['offset', 'V']])
        self.assertIn('duration', statistic['fit_axes'])
        self.assertEqual(cards[1]['preview'], '→ publishes slope (MHz), offset (V)')
        self.assertTrue(any(item['value'] == 'rabi_slope' for item in cards[2]['lines'][0]['operands']))
        self.assertIn('fit', [choice[0] for choice in statistic['choices']])
        self.assertEqual(facts['defaults']['variable']['every'], 0)


@unittest.skipUnless(os.environ.get('ZLC_DERIVE_FILE'), 'ZLC_DERIVE_FILE names a real dataset')
class RealFile(unittest.TestCase):
    def test_live_and_sealed_agree_and_memory_stays_small(self):
        z = np.load(os.environ['ZLC_DERIVE_FILE'])
        values, valid = z['data'], z['data.validity']
        schema = dataset_schema_from_tree(json.loads(str(z['info']))['sections']['dataset']['data']['schema'])
        self.assertEqual(values.shape, schema.physical_shape)
        rows = rows_of({'name': 'a', 'terms': [{'of': 'survival'}], 'statistic': 'mean', 'over': ['repeat', 'Site'], 'by': 'repeat', 'every': 50},
                       {'name': 'rabi', 'terms': [{'of': 'a'}], 'statistic': 'fit', 'over': ['rabi.duration'], 'model': 'line'})
        sealed, _each, _p = run_member(rows, sealed_route(schema, values, valid, member='survival'))
        live, each, processor = run_member(rows, live_route(schema, values, valid, member='survival'))
        a, b = live.planes['a'], sealed.planes['a']
        self.assertEqual(a['values'].shape, (20, 50, 1))
        np.testing.assert_array_equal(a['valid'], b['valid'])
        close(a['values'], b['values'], a['valid'])
        close(a['sigma'], b['sigma'], a['valid'] & np.isfinite(b['sigma']))
        k, n = (values & valid)[:50, 3].sum(), valid[:50, 3].sum()
        self.assertAlmostEqual(a['values'][0, 3, 0], k / n, places=12)
        close(live.planes['rabi_slope']['values'], sealed.planes['rabi_slope']['values'], live.planes['rabi_slope']['valid'])
        self.assertEqual(live.planes['rabi_slope']['values'].shape, (20, 1, 1))
        # One open group: n, mean, m2 of each kept cell (50 points x 1), never a sample.
        self.assertEqual(processor.peak_open, 3 * 50 * 8)
        print(f'\nreal file: {sum(each)} commits while live; state held at most {processor.peak_open} bytes')


def run_member(rows, deliveries):
    processor = DeriveProcessor(variables=rows, primary_output='survival')
    processor.bind_run(Context())
    written = Written()
    each = []
    processor.peak_open = 0
    for inputs in deliveries:
        commits = processor.evaluate_inputs(inputs)
        each.append(len(commits))
        for commit in commits:
            written.take(commit)
        for accumulator in processor._accumulation.accumulators.values():
            held = sum(plane.nbytes for tally in accumulator._tallies.values() for plane in tally.values())
            processor.peak_open = max(processor.peak_open, held)
    for commit in processor.finish_inputs():
        written.take(commit)
    return written, each, processor


if __name__ == '__main__':
    unittest.main()
