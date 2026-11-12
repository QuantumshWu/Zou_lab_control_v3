"""A device's description is read from the instrument once per change and answered after."""
import pathlib
import queue
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host
from zlc_plugin.declare import DeviceType
from zlc_atom.authoring import AuthoringField, TunableField


class Source:
    """An instrument whose every hardware read is counted; a tune of `limit` moves the frequency's upper limit."""

    def __init__(self):
        self.reads = 0
        self.conversions = 0
        self.limit = 10.0
        self.frequency = 1.0

    def _fields(self):
        return (TunableField(AuthoringField('frequency', 'float', 'Frequency', None, minimum=0.0, maximum=self.limit, unit='Hz'),
                             self.frequency, True, ('frequency',)),
                TunableField(AuthoringField('limit', 'float', 'Limit', None, unit='Hz'), self.limit, False, ('limit',)))

    def refresh_tunable_fields(self):
        self.reads += 1
        return self._fields()

    def tunable_fields(self):
        return self._fields()

    def tune(self, name, value):
        setattr(self, name, float(value))
        return float(value)

    def tunable_values(self):
        self.reads += 1
        return {'frequency': self.frequency, 'limit': self.limit}

    def settings_provenance(self):
        return {}

    def convert_tunable_value(self, name, value, source, target):
        self.conversions += 1
        return tuple(float(item) * 1e-6 for item in value)

    def read_tunable_in_unit(self, name, unit=''):
        from zlc_plugin.tunable import convert_tunable_field
        return convert_tunable_field(next(field for field in self._fields() if field.metadata.name == name), unit)

    def close(self):
        pass


class Owner:
    def __init__(self):
        self.instances = {}
        self.answers = {}
        self._releases = queue.SimpleQueue()

    def decode(self, value):
        return value

    def reply(self, identity, result=None, **kwargs):
        self.answers[identity] = (result, kwargs)


class Description(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.instance = host.Instance(self.owner, 'rf')
        self.source = Source()
        self.instance.declared = DeviceType('rf.test', 'rf', None, (), lambda key, values: self.source)
        self.instance.adapter = self.source
        self.identity = 0

    def tearDown(self):
        self.instance.lane.put(None)
        self.instance._thread.join(2)
        self.instance.shutdown()

    def ask(self, op, params):
        self.identity += 1
        self.instance._handle({'id': self.identity, 'op': op, 'p': params})
        result, kwargs = self.owner.answers[self.identity]
        self.assertNotIn('error', kwargs, kwargs)
        return result

    def maximum(self, rows):
        return next(row['field']['maximum'] for row in rows if row['field']['name'] == 'frequency')

    def test_fields_are_read_once_and_answered_until_the_device_changes(self):
        spi = lambda method, *args: self.ask('spi', {'method': method, 'args': list(args), 'kwargs': {}})
        for _ in range(23):
            rows = spi('tunable_fields')
            spi('read_tunable_in_unit', 'frequency', '')
        self.assertEqual(self.source.reads, 1, 'every editor and run answered from one reading')
        self.assertEqual(self.maximum(rows), 10.0)
        # Readings are not the description: they stay live and leave it standing.
        self.ask('values', {})
        spi('tunable_values')
        reads = self.source.reads
        spi('tunable_fields')
        self.assertEqual(self.source.reads, reads)
        # A tune changes the device: what it says is read again, once.
        self.ask('tune', {'field': 'limit', 'value': 20.0})
        for _ in range(5):
            rows = spi('tunable_fields')
        self.assertEqual(self.maximum(rows), 20.0)
        self.assertEqual(self.source.reads, reads + 1)
        # So does a tune through a run's proxy (a non-read SPI call).
        spi('tune', 'limit', 30.0)
        self.assertEqual(self.maximum(spi('tunable_fields')), 30.0)
        self.assertEqual(self.source.reads, reads + 2)

    def test_conversions_hold_until_the_device_changes(self):
        convert = lambda: self.ask('spi', {'method': 'convert_tunable_value', 'args': ['frequency', [1.0, 2.0], 'Hz', 'MHz'], 'kwargs': {}})
        for _ in range(12):
            self.assertEqual(list(convert()), [1e-6, 2e-6])
        self.assertEqual(self.source.conversions, 1)
        self.ask('spi', {'method': 'tune', 'args': ['frequency', 3.0], 'kwargs': {}})
        convert()
        self.assertEqual(self.source.conversions, 2)

    def test_device_control_refresh_reads_the_instrument_again(self):
        self.ask('fields', {})
        self.ask('spi', {'method': 'tunable_fields', 'args': [], 'kwargs': {}})
        self.assertEqual(self.source.reads, 1, "a node's ask is answered from Device Control's reading")
        self.source.limit = 40.0  # turned on the front panel: only the operator's Refresh can know
        self.assertEqual(self.maximum(self.ask('fields', {})), 40.0)
        self.assertEqual(self.source.reads, 2)
        self.assertEqual(self.maximum(self.ask('spi', {'method': 'tunable_fields', 'args': [], 'kwargs': {}})), 40.0)
        self.assertEqual(self.source.reads, 2)


if __name__ == '__main__':
    unittest.main()
