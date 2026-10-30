"""Scalar device-unit boundaries: no VISA session or real device is opened."""
import pathlib
import sys
import threading
import types
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.devices.rf.contract import FREQUENCY_FIELD, POWER_FIELD
from zlc_atom.devices.rf.rigol_dg4000.source import RigolDg4000RfSource
from zlc_plugin.tunable import convert_tunable_field, read_tunable_in_unit, tune_in_unit


def field(unit='Hz', current=1001000.0, step=1000.0):
    return TunableField(AuthoringField('frequency', 'float', 'Frequency', 1001000.0,
        minimum=1000.0, maximum=200000000.0, step=step, unit=unit), current, True,
        ('frequency',), device_limits=(1000.0, 200000000.0))


class Device:
    def __init__(self, entry): self.entry = entry; self.writes = []; self.readback = None
    def tunable_fields(self): return (self.entry,)
    def tune(self, name, value):
        self.writes.append(value)
        return value if self.readback is None else self.readback


class NumericUnits(unittest.TestCase):
    def test_shared_scalar_conversion_moves_reading_step_and_bounds_together(self):
        device = Device(field())
        shown = read_tunable_in_unit(device, 'frequency', 'MHz')
        self.assertEqual((shown.current, shown.metadata.step), (1.001, 0.001))
        self.assertEqual((shown.metadata.minimum, shown.metadata.maximum), (0.001, 200.0))
        self.assertEqual(shown.device_limits, (0.001, 200.0))
        self.assertEqual(tune_in_unit(device, 'frequency', 1.001, 'MHz'), 1.001)
        self.assertEqual(device.writes, [1001000.0])

    def test_real_quantized_readback_and_high_precision_step_are_not_rounded(self):
        device = Device(field(step=0.12345678901234566))
        device.readback = 179999999.9997
        self.assertEqual(tune_in_unit(device, 'frequency', 180, 'MHz'), 179.9999999997)
        self.assertEqual(read_tunable_in_unit(device, 'frequency', 'MHz').metadata.step, 1.2345678901234566e-7)

    def test_nonlinear_display_has_no_constant_step_and_does_not_convert_zero(self):
        entry = TunableField(AuthoringField('power', 'float', 'Power', 1.0,
            minimum=0.1, maximum=100.0, step=0.1, unit='mW'), 1.0, True, ('power',))
        shown = convert_tunable_field(entry, 'dBm')
        self.assertIsNone(shown.metadata.step)
        self.assertEqual(shown.current, 0.0)

    def rigol(self):
        # Bypass only transport setup. Invoke the shipped native unit overrides
        # with the same field/readback seam that their SCPI implementation uses.
        device = object.__new__(RigolDg4000RfSource)
        device._condition = threading.Condition(threading.RLock())
        device._routing = {'frequency': ('ch1', FREQUENCY_FIELD), 'power': ('ch1', POWER_FIELD)}
        entry = field()
        device.tunable_fields = lambda: (entry,)
        writes = []
        device.tune = lambda name, value: writes.append(value) or value
        return device, writes

    def test_rigol_native_frequency_override_uses_the_shared_field_conversion(self):
        device, writes = self.rigol()
        shown = device.read_tunable_in_unit('frequency', 'MHz')
        self.assertEqual(shown.metadata.step, 0.001)
        self.assertEqual(shown.current, 1.001)
        self.assertEqual(device.convert_tunable_value('frequency', 1.001, 'MHz', 'Hz'), 1001000.0)
        self.assertEqual(device.tune_in_unit('frequency', 1.001, 'MHz'), 1.001)
        self.assertEqual(writes, [1001000.0])

    def test_rigol_prefixes_never_take_a_power_or_binary_round_trip(self):
        device, _ = self.rigol()
        self.assertEqual(device._convert_amplitude('ch1', 0.043, 'Vpp', 'mVpp'), 43.0)
        self.assertEqual(device._convert_amplitude('ch1', 0.017, 'mW', 'W'), 0.000017)
        self.assertEqual(device._convert_amplitude('ch1', 0.1799999999997, 'Vpp', 'mVpp'), 179.9999999997)

    def test_rigol_scpi_amplitude_writes_and_readbacks_keep_prefix_precision(self):
        device, _ = self.rigol()
        commands = []
        device._amplitudes = {'ch1': {'unit': 'VPP', 'current': 0.1, 'minimum': 0.001, 'maximum': 1.0}}
        device._link = types.SimpleNamespace(write=commands.append, query=lambda _: '0.043')
        self.assertEqual(device._write_power_in_unit('ch1', 43.0, 'mVpp'), 43.0)
        self.assertEqual(float(commands[0].split()[-1].removesuffix('VPP')), 0.043)
        device._link.query = lambda _: '0.1799999999997'
        self.assertEqual(device._write_power_in_unit('ch1', 180, 'mVpp'), 179.9999999997)

    def test_scan_records_applied_settings_and_each_readback_but_restores_original_value(self):
        import numpy as np
        from zlc_atom.nodes.scan.plan import ScanAxis, ScanPlan, ScanPort
        from zlc_atom.nodes.scan.seamless import SeamlessScanMeasurement
        from zlc_data import AxisId, AxisSpec, DomainSpec, SPATIAL_X, ValueSchema, ValidityContract, ValidityMode, owned_snapshot_from_arrays

        class Knob(Device):
            def __init__(self):
                super().__init__(field(current=1234000.0)); self.epoch = 0; self.snapshot_reads = []
            def tune(self, name, value):
                self.writes.append(value); actual = round(value / 1000) * 1000.0
                self.entry = replace(self.entry, current=actual); self.epoch += 1
                return actual
            def tunable_values(self):
                self.snapshot_reads.append(self.entry.current)
                return {'frequency': self.entry.current}
            def settings_provenance(self): return {'device_session_id': 'rf-test', 'settings_epoch': self.epoch}

        schema = ValueSchema(ValidityContract(ValidityMode.VALUE), np.dtype('<f8'), 'count')
        source = types.SimpleNamespace(snapshot=owned_snapshot_from_arrays(schema, np.ones((1, 1, 1)), 0,
                                                                         cell_domain=DomainSpec((1,), (AxisSpec(AxisId('x'), 'x', SPATIAL_X, 1),))),
                                       event_record={'source_fact': 'kept'})
        for fail_second in (False, True):
            with self.subTest(fail_second=fail_second):
                knob = Knob(); outputs = []; ordering = []; run_record = {}
                def core(command, **params):
                    ordering.append((command, params.get('stage', 'run')))
                    if command == 'pulse_run':
                        if fail_second and params.get('stage') is None: raise RuntimeError('second fire failed')
                        return {'command_id': 1, 'applied': {'config': {'clock': 50e6}}}
                    return {'done': True, 'status': 'done'}
                context = types.SimpleNamespace(core=core, cancel_requested=lambda: False,
                    discard_inputs=lambda: None, next_input=lambda timeout: source,
                    acquire=lambda node, **kwargs: ordering.append(('acquire', knob.entry.current)),
                    report_progress=lambda *args, **kwargs: None,
                    record_devices=lambda: ordering.append(('snapshot', knob.entry.current)),
                    set_run_record=lambda record: run_record.update(record),
                    commit_live=lambda records, **kwargs: outputs.append(records['scan']))
                scan = SeamlessScanMeasurement(pulse={'path': 'test.zpulse', 'api_fields': []},
                    plan=ScanPlan((ScanAxis('device:rf:frequency', (1.0004, 2.0006), 'MHz'),)),
                    ports=(ScanPort('device:rf:frequency', 'RF frequency', 'Hz', 1e3, 2e8),),
                    source_signal='source', tunables={'rf': knob}, repeats=1, shots_per_point=1, sequencer_key='board',
                    acquisition_logic='camera')
                if fail_second:
                    with self.assertRaisesRegex(RuntimeError, 'second fire failed'): scan.execute(context)
                else:
                    self.assertEqual(scan.execute(context), {'points': 2, 'shots': 2})
                baseline = run_record['device_snapshots']['tunable:rf']
                self.assertEqual(baseline, {'values': {'frequency': 1000000.0}, 'units': {'frequency': 'Hz'},
                    'provenance': {'device_session_id': 'rf-test', 'settings_epoch': 1}})
                self.assertEqual(knob.snapshot_reads, [1000000.0])
                self.assertEqual(knob.entry.current, 1234000.0)
                self.assertEqual(knob.writes, [1000400.0, 2000600.0, 1234000.0])
                self.assertLess(ordering.index(('pulse_run', 'load')), ordering.index(('acquire', 1000000.0)))
                self.assertLess(ordering.index(('acquire', 1000000.0)), ordering.index(('snapshot', 1000000.0)))
                self.assertLess(ordering.index(('snapshot', 1000000.0)), ordering.index(('pulse_run', 'fire')))
                self.assertEqual(sum(op == 'acquire' for op, _ in ordering), 1)
                self.assertEqual(run_record['plan']['axes'][0]['values'], [1.0004, 2.0006])
                for index, output in enumerate(outputs):
                    recorded = output.event_record['actual_devices']['rf']
                    self.assertEqual(recorded['values']['frequency'], [1.0, 2.001][index])
                    self.assertEqual(recorded['units']['frequency'], 'MHz')
                    self.assertEqual(recorded['provenance']['settings_epoch'], index + 1)
                    self.assertEqual(output.cell_origin, (0, index))
                    self.assertEqual(output.event_record['source_fact'], 'kept')


class LinkedScan(unittest.TestCase):
    def plan(self, enabled=True):
        from zlc_atom.nodes.scan.plan import ScanPlan
        return ScanPlan.from_tree({'axes': [
            {'port':'pulse:param:B','values':[1,2,3],'unit':'us','link_enabled':enabled,'links':[
                {'port':'pulse:param:A','values':[0,1],'unit':'ms','mode':'formula','formula':'5 - x'},
                {'port':'api:delay','values':[10,30],'unit':'us','mode':'range'}]},
            {'port':'pulse:param:D','values':[7,8],'unit':'us'}]})

    def test_numeric_formula_pairing_and_disabled_memory(self):
        from zlc_atom.nodes.scan.plan import formula_values, split_outer_axes, scan_rows
        plan=self.plan()
        self.assertEqual(plan.shape,(3,2));self.assertEqual(plan.point_count,6)
        self.assertEqual(plan.rows()[0],(1,4,10,7))
        self.assertEqual(plan.rows()[-1],(3,2,30,8))
        outer,inner=split_outer_axes(plan)
        self.assertEqual(len(scan_rows(outer)),3);self.assertEqual(len(scan_rows(inner)),2)
        self.assertEqual(self.plan(False).shape,(3,2))
        self.assertEqual(len(self.plan(False).axes),2)
        self.assertEqual(formula_values('120 - x * 2',(1,3)),(118,114))
        for bad in ('api.C - x','__import__("os")','1 / (x-x)','x ** 100000'):
            with self.assertRaises(ValueError):formula_values(bad,(1,2))

    def test_linked_schema_uses_one_coordinate_dimension(self):
        import numpy as np
        from zlc_data import AxisSpec,AxisId,SPATIAL_X,DomainSpec,owned_snapshot_from_arrays,ValueSchema,ValidityContract,ValidityMode
        from zlc_atom.nodes.scan.dataset import scan_dataset_schema
        plan=self.plan();source=owned_snapshot_from_arrays(ValueSchema(ValidityContract(ValidityMode.VALUE),np.dtype('<f8'),'1'),np.ones((1,1,1)),0,cell_domain=DomainSpec((1,),(AxisSpec(AxisId("x"),"x",SPATIAL_X,1),))).block.schema
        schema=scan_dataset_schema(source,plan.rows(),[(axis.port,axis.unit) for axis in plan.axes],coordinate_of=[None,0,0,None])
        axes=schema.point_domain.axes
        self.assertEqual(schema.point_domain.size,6)
        self.assertEqual(axes[1].coordinate_of,axes[0].axis_id)
        self.assertEqual(axes[2].coordinate_of,axes[0].axis_id)
        self.assertEqual(tuple(schema.point_domain.codes(axes[1].axis_id)),tuple(schema.point_domain.codes(axes[0].axis_id)))

    def test_mixed_group_loads_paired_slots_and_api_before_fire(self):
        import numpy as np
        from zlc_data import AxisSpec,AxisId,SPATIAL_X,DomainSpec,ValueSchema,ValidityContract,ValidityMode,owned_snapshot_from_arrays
        from zlc_atom.nodes.scan.plan import ScanPort
        from zlc_atom.nodes.scan.seamless import SeamlessScanMeasurement
        plan=self.plan();calls=[];records=[];outputs=[];held={}
        source=types.SimpleNamespace(snapshot=owned_snapshot_from_arrays(ValueSchema(ValidityContract(ValidityMode.VALUE),np.dtype('<f8'),'1'),np.ones((1,1,1)),0,cell_domain=DomainSpec((1,),(AxisSpec(AxisId("x"),"x",SPATIAL_X,1),))),event_record={})
        def core(command,**params):
            calls.append((command,params))
            if command=='pulse_run':
                if 'scan' in params:held.update(scan=params['scan'])
                return {'command_id':1,'applied':{'config':{}},'scan':held['scan']}
            return {'done':True,'status':'done'}
        context=types.SimpleNamespace(core=core,cancel_requested=lambda:False,discard_inputs=lambda:None,next_input=lambda t:source,
            report_progress=lambda *a,**k:None,record_devices=lambda:None,set_run_record=records.append,commit_live=lambda data,**k:outputs.append(data['scan']))
        scan=SeamlessScanMeasurement(pulse={'path':'test.json','api_fields':[]},plan=plan,
            ports=tuple(ScanPort(a.port,a.port,a.unit,0,1000) for a in plan.axes),source_signal='source',repeats=1,shots_per_point=1)
        self.assertEqual(scan.execute(context),{'points':6,'shots':6})
        programs=[p for command,p in calls if command=='pulse_run' and p.get('stage')!='fire']
        self.assertEqual([p['api_values']['delay'][0] for p in programs],[10,20,30])
        self.assertEqual(programs[0]['scan']['fields'],['D','B','A'])
        self.assertEqual(programs[0]['scan']['rows'],[[7,1,4],[8,1,4]])
        self.assertEqual(programs[-1]['scan']['rows'],[[7,3,2],[8,3,2]])
        self.assertEqual([o.cell_origin for o in outputs],[(0,i) for i in range(6)])

if __name__ == '__main__': unittest.main()
