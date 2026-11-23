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
from zlc_atom.nodes.seamless_scan.logic_node import SEAMLESS_SCAN_SCHEMA, _admit_resources, _editor_facts, _estimate_run
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
        self.assertEqual(ports[0].grid, PortGrid(1.0, 'even'))
        played = plan_from_authored(json.dumps(plan), ports)
        self.assertEqual(played.axes[0].values, (1.0, 2.0, 3.0))
        self.assertEqual(bind_plan(played, ports), ports)
        self.assertEqual(plan_from_authored(json.dumps(plan)).axes[0].values, (1.0, 1.5, 2.0, 2.5, 3.0),
                         'a draft read without its ports is the plan as written')

    def test_an_int_field_read_in_another_unit_keeps_its_whole_numbers(self):
        port = 'device:counter:frequency'
        own, = scan_ports_for_devices({'counter': Counts()})
        start, = scan_ports_for_devices({'counter': Counts()}, units={port: 'kHz'})
        self.assertEqual((own.unit, own.grid), ('Hz', PortGrid(1.0)))
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


if __name__ == '__main__':
    unittest.main()
