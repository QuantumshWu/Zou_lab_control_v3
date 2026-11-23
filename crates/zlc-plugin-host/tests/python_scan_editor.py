"""The scan plan editor when a lent device cannot be read: no instrument, no core."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.nodes.scan.plan import (PortGrid, ScanPlan, api_scan_ports_for, bind_plan, plan_from_authored, plan_input_rows,
                                     scan_ports_for_devices)
from zlc_atom.nodes.scan.editor import scan_editor_facts
from zlc_atom.nodes.seamless_scan.logic_node import SEAMLESS_SCAN_SCHEMA, _admit_resources, _build, _editor_facts, _estimate_run
from zlc_atom.nodes.scan.plan import SEAMLESS_PULSE_RESOURCE
from zlc_plugin import PluginError

VISA = 'VI_ERROR_SYSTEM_ERROR (-1073807360): Unknown system error'

#: A pulse whose bracket's repeat count is an API parameter, as pulse_describe states it
#: (zlc-pulse's own test checks the grid it states for every kind of field).
REPEATS = {'path': 'repeats.json', 'name': 'repeats', 'scan_fields': [], 'unscannable': {}, 'periods': [],
           'api_fields': [{'field_id': 'bracket_count:loop', 'label': 'Loop.repeats', 'kind': 'bracket_count', 'unit': 'count',
                           'value': 2.0, 'is_dac': False, 'limit_lo': 1.0, 'limit_hi': 4294967295.0,
                           'grid': {'step': 1.0, 'ties': 'even'}}]}


class Reads:
    role = 'dds'

    def tunable_fields(self):
        field = AuthoringField('frequency', 'float', 'Frequency', None, minimum=1.0, maximum=2.0, unit='Hz')
        return (TunableField(field, 1.5, True, ('frequency',)),)


class Counts:
    """A device whose knob is an int field in hertz (no device in the tree has one)."""
    role = 'counter'

    def tunable_fields(self):
        field = AuthoringField('frequency', 'int', 'Frequency', None, minimum=0, maximum=1_000_000, unit='Hz')
        return (TunableField(field, 1000, True, ('frequency',)),)


class Unconvertible:
    """A device that reads its knob in its own unit and fails to read it in another."""
    role = 'flaky'

    def tunable_fields(self):
        field = AuthoringField('frequency', 'float', 'Frequency', None, minimum=1.0, maximum=1e6, unit='Hz')
        return (TunableField(field, 1000.0, True, ('frequency',)),)

    def read_tunable_in_unit(self, name, unit=''):
        raise PluginError('failed', VISA)


class Fails:
    role = 'rigol'

    def tunable_fields(self):
        raise PluginError('failed', VISA)


def facts(devices):
    plan = {'axes': [{'port': 'device:rigol:frequency', 'unit': 'Hz', 'values': [1.0, 2.0]}]}
    return _editor_facts({'plan': json.dumps(plan), 'api_values': ''}, {}, devices, None, None, None)


class ScanEditorFacts(unittest.TestCase):
    def test_a_device_that_cannot_be_read_leaves_the_editor_and_the_others(self):
        answer = facts({'tunable:rigol': Fails(), 'tunable:dds': Reads()})
        self.assertEqual(answer['editor'], 'scan_plan')
        self.assertEqual([port['port'] for port in answer['ports']], ['device:dds:frequency'])
        self.assertEqual(answer['unoffered']['device:rigol:frequency']['leaf'], 'frequency')
        self.assertEqual(answer['unread'], f'rigol could not be read: {VISA}')

    def test_every_device_read_says_nothing(self):
        self.assertEqual(facts({'tunable:rigol': Reads()})['unread'], '')

    def test_a_run_still_refuses_a_device_it_cannot_read(self):
        with self.assertRaisesRegex(PluginError, 'VI_ERROR_SYSTEM_ERROR'):
            scan_ports_for_devices({'rigol': Fails()})

    def test_native_preview_and_execution_share_compact_range_and_keep_inactive_banks(self):
        plan = {'axes': [{'port': 'pulse:param:duration', 'unit': 'us', 'values': [],
                         'range': {'start': 0.1, 'stop': 0.3, 'count': 3}, 'mode': 'range',
                         'value_text': 'unfinished,', 'link_enabled': False,
                         'links': [{'port': 'api:delay', 'values': [0, 1], 'mode': 'formula',
                                    'formula': 'unfinished(', 'value_text': 'kept'}]}]}
        answer = scan_editor_facts({'plan': json.dumps(plan), 'api_values': ''}, None, ())
        execution = ScanPlan.from_tree(plan)
        self.assertEqual(execution.axes[0].values, (0.1, 0.2, 0.3))
        self.assertIn('3 points', answer['summary'])
        self.assertEqual(answer['problem'], '')
        self.assertEqual(answer['rows'][0]['range'], plan['axes'][0]['range'])
        self.assertEqual(answer['rows'][0]['value_text'], 'unfinished,')
        self.assertEqual(answer['rows'][0]['links'][0]['formula'], 'unfinished(')
        self.assertTrue(answer['axis_facts'][0]['range_uniform'])
        self.assertEqual(answer['axis_facts'][0]['active_bounds'], [0.1, 0.3])
        plan['axes'][0]['value_text'] = '5, 7'
        answer = scan_editor_facts({'plan': json.dumps(plan), 'api_values': ''}, None, ())
        self.assertEqual(answer['axis_facts'][0]['active_bounds'], [0.1, 0.3])
        self.assertEqual(answer['axis_facts'][0]['values_bounds'], [5.0, 7.0],
                         'the inactive Values bank still has its own limit feedback')
        self.assertEqual(answer['plan_source'], json.dumps(plan))

    def test_invalid_values_stay_editable_and_report_the_same_native_refusal_as_start(self):
        plan = {'axes': [{'port': 'pulse:param:duration', 'values': [1, 2, 3],
                         'mode': 'values', 'value_text': '1,'}]}
        answer = scan_editor_facts({'plan': json.dumps(plan), 'api_values': ''}, None, ())
        with self.assertRaises(ValueError) as failure:
            ScanPlan.from_tree(plan)
        self.assertEqual(answer['problem'], str(failure.exception))
        self.assertEqual(answer['rows'][0]['value_text'], '1,')
        self.assertEqual(plan_input_rows(plan)[0]['values'], [1, 2, 3])
        self.assertEqual(answer['summary'], '')

    def test_a_repeat_count_swept_through_fractions_plays_and_shows_whole_counts(self):
        plan = {'axes': [{'port': 'api:bracket_count:loop', 'unit': 'count', 'values': [],
                          'range': {'start': 1, 'stop': 3, 'count': 5}, 'mode': 'range', 'value_text': ''}]}
        answer = scan_editor_facts({'plan': json.dumps(plan), 'api_values': ''}, REPEATS, ())
        self.assertEqual(answer['axis_facts'][0]['played'],
                         {'first': 1.0, 'last': 3.0, 'count': 3, 'written': 5, 'moved': 2, 'distinct': 3})
        self.assertIn('3 = 3 points', answer['summary'])
        self.assertEqual(answer['problem'], '')
        ports = api_scan_ports_for(REPEATS)
        self.assertEqual(ports[0].grid, PortGrid(1.0, 'even', whole=True))
        played = plan_from_authored(json.dumps(plan), ports)
        self.assertEqual(played.axes[0].values, (1.0, 2.0, 3.0))
        self.assertEqual(bind_plan(played, ports), ports)
        self.assertEqual(plan_from_authored(json.dumps(plan)).axes[0].values, (1.0, 1.5, 2.0, 2.5, 3.0),
                         'a draft read without its ports is the plan as written')

    def test_an_int_field_read_in_another_unit_keeps_its_whole_numbers(self):
        port = 'device:counter:frequency'
        own, = scan_ports_for_devices({'counter': Counts()})
        start, = scan_ports_for_devices({'counter': Counts()}, units={port: 'kHz'})
        self.assertEqual((own.unit, own.grid), ('Hz', PortGrid(1.0, whole=True)))
        self.assertEqual((start.unit, start.grid), ('kHz', PortGrid(0.001)), 'one hertz, read in kHz')
        plan = json.dumps({'axes': [{'port': port, 'unit': 'kHz', 'mode': 'values', 'value_text': '1.0004, 2.5', 'values': []}]})
        # The editor reads the knob in its own unit, a Start in the plan's: both play 1 kHz and 2.5 kHz.
        self.assertEqual(plan_from_authored(plan, (own,)).axes[0].values, (1.0, 2.5))
        self.assertEqual(plan_from_authored(plan, (start,)).axes[0].values, (1.0, 2.5))

    def test_the_draft_and_the_estimate_read_the_plan_as_a_start_plays_it(self):
        # A count following a manual knob by x: at 1.0 and 1.4 it is a count of 1 both times.
        plan = json.dumps({'axes': [{'port': 'manual:power', 'unit': '', 'values': [1.0, 1.4, 1.6], 'link_enabled': True,
                                     'links': [{'port': 'api:bracket_count:loop', 'mode': 'formula', 'formula': 'x', 'unit': 'count'}]}]})
        values = {'plan': plan, 'api_values': '', 'repeats': 1, 'shots_per_point': 1}
        resources = {SEAMLESS_PULSE_RESOURCE.field_name: REPEATS}
        SEAMLESS_SCAN_SCHEMA.validator(values)  # as written, the plan reads
        issues = _admit_resources(values, resources)
        self.assertEqual(len(issues), 1, issues)
        self.assertIn("its knob's grid made them one", issues[0])
        estimate = _estimate_run(values, {}, resources)
        self.assertEqual((estimate['status'], estimate['reason']), ('unknown', issues[0]))
        fine = {**values, 'plan': json.dumps({'axes': [{'port': 'api:bracket_count:loop', 'unit': 'count', 'values': [1.4, 2.6]}]})}
        self.assertEqual(_admit_resources(fine, resources), ())

    def test_a_repeat_count_is_a_whole_number_box_and_a_fraction_written_for_it_is_a_draft_issue(self):
        resources = {SEAMLESS_PULSE_RESOURCE.field_name: REPEATS}
        field, = scan_editor_facts({'plan': '', 'api_values': ''}, REPEATS, ())['api']['fields']
        self.assertEqual({key: field[key] for key in ('kind', 'minimum', 'maximum', 'authored', 'value', 'unit')},
                         {'kind': 'integer', 'minimum': 1, 'maximum': 4294967295, 'authored': 2, 'value': 2, 'unit': 'count'})
        self.assertIsInstance(field['value'], int)
        draft = {'plan': json.dumps({'axes': [{'port': 'manual:power', 'values': [1.0]}]}), 'repeats': 1, 'shots_per_point': 1}
        for written, issue in (('bracket_count:loop = 2.5', "API value 'bracket_count:loop' is 2.5: bracket_count:loop takes whole numbers"),
                               ('Loop.repeats = 2.5', "API value 'Loop.repeats' is 2.5: bracket_count:loop takes whole numbers"),
                               ('bracket_count:loop = 3', None)):
            issues = _admit_resources({**draft, 'api_values': written}, resources)
            self.assertEqual(issues, () if issue is None else (issue,), written)
        # Its box holds the 2.5 as the draft does, said wrong under it -- not a 2 an edit of another box writes back.
        field, = scan_editor_facts({'plan': '', 'api_values': 'bracket_count:loop = 2.5'}, REPEATS, ())['api']['fields']
        self.assertEqual((field['value'], field['problem']),
                         (2.5, "API value 'bracket_count:loop' is 2.5: bracket_count:loop takes whole numbers"))
        field, = scan_editor_facts({'plan': '', 'api_values': 'bracket_count:loop = 3'}, REPEATS, ())['api']['fields']
        self.assertEqual((field['value'], field['problem']), (3, ''))

    def test_a_clock_grid_is_no_whole_number_box_however_whole_its_tick_reads(self):
        # An output delay in ns on a 20 ns clock: its step reads 20, a whole number, and it is still a time.
        delay = {'field_id': 'delay:cooling', 'label': 'cooling.delay', 'kind': 'delay', 'unit': 'ns', 'value': 0.0,
                 'is_dac': False, 'limit_lo': -100000.0, 'limit_hi': 100000.0, 'grid': {'step': 20.0, 'ties': 'away'}}
        delayed = {**REPEATS, 'api_fields': [*REPEATS['api_fields'], delay]}
        fields = {field['name']: field for field in scan_editor_facts({'plan': '', 'api_values': ''}, delayed, ())['api']['fields']}
        self.assertEqual((fields['delay:cooling']['kind'], fields['bracket_count:loop']['kind']), ('number', 'integer'))
        self.assertEqual(fields['delay:cooling']['units'][-1][0], 'ns', 'read in any of its spellings')
        draft = {'plan': json.dumps({'axes': [{'port': 'manual:power', 'values': [1.0]}]}), 'repeats': 1, 'shots_per_point': 1,
                 'api_values': 'delay:cooling = 30.5'}
        self.assertEqual(_admit_resources(draft, {SEAMLESS_PULSE_RESOURCE.field_name: delayed}), (), 'the binding layer aligns a delay')
        # A description that states no grid offers none: no whole-number box, no refusal.
        bare = {**REPEATS, 'api_fields': [{key: value for key, value in REPEATS['api_fields'][0].items() if key != 'grid'}]}
        self.assertEqual(api_scan_ports_for(bare)[0].grid, None)
        self.assertEqual(scan_editor_facts({'plan': '', 'api_values': ''}, bare, ())['api']['fields'][0]['kind'], 'number')

    def test_a_device_that_cannot_be_read_in_the_plans_unit_leaves_the_estimate_without_its_knob(self):
        port = 'device:flaky:frequency'
        unread = {}
        self.assertEqual(scan_ports_for_devices({'flaky': Unconvertible()}, units={port: 'kHz'}, unread=unread), ())
        self.assertEqual(unread, {'flaky': str(PluginError('failed', VISA))}, 'the device says why')
        with self.assertRaisesRegex(PluginError, 'VI_ERROR_SYSTEM_ERROR'):
            scan_ports_for_devices({'flaky': Unconvertible()}, units={port: 'kHz'})
        values = {'plan': json.dumps({'axes': [{'port': port, 'unit': 'kHz', 'values': [1.0, 2.0]}]}), 'api_values': '',
                  'repeats': 1, 'shots_per_point': 1}
        estimate = _estimate_run(values, {'tunable:flaky': Unconvertible()}, {SEAMLESS_PULSE_RESOURCE.field_name: REPEATS})
        self.assertEqual(estimate['status'], 'unknown')

    def test_a_fraction_kept_for_a_count_the_plan_sweeps_refuses_nothing_until_its_axis_goes(self):
        resources = {SEAMLESS_PULSE_RESOURCE.field_name: REPEATS}
        draft = {'api_values': 'bracket_count:loop = 2.5', 'repeats': 1, 'shots_per_point': 1}
        swept = json.dumps({'axes': [{'port': 'api:bracket_count:loop', 'unit': 'count', 'values': [1.0, 2.0]}]})
        self.assertEqual(_admit_resources({**draft, 'plan': swept}, resources), (), 'swept: the fixed value plays nowhere')
        field, = scan_editor_facts({**draft, 'plan': swept}, REPEATS, ())['api']['fields']
        self.assertEqual((field['value'], field['problem']), (2.5, ''), 'kept as written, nothing said wrong')
        self.assertIn('kept for when its axis is taken out', field['unavailable_reason'])
        measurement = _build(source_signal='camera/source', pulse_resource=REPEATS, plan=swept, api_values=draft['api_values'])
        self.assertEqual(measurement.api_values['bracket_count:loop'], (2.5, 'count'), 'Start: kept, each point overriding it')
        self.assertEqual(measurement.plan.axes[0].values, (1.0, 2.0))
        fixed = json.dumps({'axes': [{'port': 'manual:power', 'values': [1.0]}]})
        problem = "API value 'bracket_count:loop' is 2.5: bracket_count:loop takes whole numbers"
        self.assertEqual(_admit_resources({**draft, 'plan': fixed}, resources), (problem,), 'its axis gone: said again')
        self.assertEqual(scan_editor_facts({**draft, 'plan': fixed}, REPEATS, ())['api']['fields'][0]['problem'], problem)


if __name__ == '__main__':
    unittest.main()
