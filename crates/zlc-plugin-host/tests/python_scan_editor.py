"""The scan plan editor when a lent device cannot be read: no instrument, no core."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.nodes.scan.plan import ScanPlan, plan_input_rows, scan_ports_for_devices
from zlc_atom.nodes.scan.editor import scan_editor_facts
from zlc_atom.nodes.seamless_scan.logic_node import _editor_facts
from zlc_plugin import PluginError

VISA = 'VI_ERROR_SYSTEM_ERROR (-1073807360): Unknown system error'


class Reads:
    role = 'dds'

    def tunable_fields(self):
        field = AuthoringField('frequency', 'float', 'Frequency', None, minimum=1.0, maximum=2.0, unit='Hz')
        return (TunableField(field, 1.5, True, ('frequency',)),)


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


if __name__ == '__main__':
    unittest.main()
