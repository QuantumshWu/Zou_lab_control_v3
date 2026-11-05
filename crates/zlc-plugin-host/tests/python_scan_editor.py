"""The scan plan editor when a lent device cannot be read: no instrument, no core."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.nodes.scan.plan import scan_ports_for_devices
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


if __name__ == '__main__':
    unittest.main()
