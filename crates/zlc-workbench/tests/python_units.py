"""Canonical unit adapter parity: NumPy dtype/coercion and native registry state."""
import pathlib
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from decimal import Decimal
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_plugin.services import call
from zlc_data.units import (Scaled, Offset, Decibel, VoltageIntoLoad, Prefixed, Unit, UnitError, UnitRegistry,
                            DEFAULT_UNITS, PREFIXES, NO_PREFIX, format_quantity, parse_quantity, prefix_for)


def reference(conversion, values, inverse=False):
    """Independent NumPy oracle for the established public conversion contract."""
    if isinstance(conversion, Scaled):
        return np.asarray(values) / conversion.factor if inverse else np.asarray(values) * conversion.factor
    if isinstance(conversion, Offset):
        return np.asarray(values, dtype=float) - conversion.offset if inverse else np.asarray(values, dtype=float) + conversion.offset
    if isinstance(conversion, Decibel):
        return (conversion.factor * np.log10(np.asarray(values, dtype=float) / conversion.reference) if inverse
                else conversion.reference * np.power(10., np.asarray(values) / conversion.factor))
    if isinstance(conversion, VoltageIntoLoad):
        values = np.asarray(values, dtype=float)
        scale = conversion.mean_square_divisor * conversion.load_ohms
        return np.sqrt(scale * values) if inverse else np.where(values < 0, np.nan, values * values) / scale
    if inverse:
        return np.asarray(reference(conversion.inner, values, True), dtype=float) / conversion.factor
    return reference(conversion.inner, np.asarray(values, dtype=float) * conversion.factor)


class Units(unittest.TestCase):
    def test_dto_construction_and_reading_need_no_engine_or_host(self):
        root = pathlib.Path(__file__).resolve().parents[3]
        script = r'''
import copy, dataclasses, sys
from unittest.mock import patch
with patch("subprocess.Popen", side_effect=AssertionError("constructor launched a process")):
    import numpy as np
    import zlc_data
    from zlc_data.units import Scaled, Offset, Decibel, VoltageIntoLoad, Prefixed, Unit, UnitError
    conversions = (Scaled(1), Offset(273.15), Decibel(.001), VoltageIntoLoad(50), Prefixed(Scaled(2), .001))
    unit = Unit("widget", "custom", conversions[0], prefixes=("k",), aliases=("widgets",))
    assert unit.scale == 1 and unit.is_linear and unit.is_base
    assert [p.symbol for p in unit.ladder] == ["k", ""]
    assert unit == copy.deepcopy(unit)
    assert hash(unit) == hash(copy.copy(unit))
    assert "widget" in repr(unit)
    assert dataclasses.asdict(unit)["symbol"] == "widget"
    value = zlc_data.ValueSchema(zlc_data.ValidityContract.value(), np.dtype("<f8"), unit.symbol)
    assert value.value_unit == "widget"
    invalid = (lambda: Scaled(0), lambda: Scaled(float("nan")), lambda: Offset(float("inf")),
               lambda: Decibel(0), lambda: Decibel(1, 0), lambda: VoltageIntoLoad(-1),
               lambda: VoltageIntoLoad(50, 0), lambda: Prefixed(Scaled(1), -1),
               lambda: Unit("bad ", "custom"), lambda: Unit("x", "custom", aliases=("x",)))
    for build in invalid:
        try: build()
        except UnitError: pass
        else: raise AssertionError("invalid DTO was accepted")
    assert "zlc_plugin.host" not in sys.modules
    service = sys.modules.get("zlc_plugin.services")
    assert service is None or service._helper is None
try: unit.to_base(np.array([1.]))
except RuntimeError as error:
    assert "not installed" in str(error)
else: raise AssertionError("conversion bypassed native service")
'''
        env = dict(os.environ, PYTHONPATH=str(root / "science" / "src"),
                   ZLC_NATIVE_EXECUTABLE=str(pathlib.Path(tempfile.gettempdir()) / "zlc-no-unit-engine" / "missing.exe"))
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_validation_and_lazy_decade_do_not_affect_value_identity(self):
        import zlc_data.units as units_module
        value = Unit("local-milli", "custom", Scaled(.001))
        before = hash(value)
        self.assertFalse(value._decade_known)
        self.assertEqual(value.decade, -3)
        self.assertTrue(value._decade_known)
        self.assertEqual(hash(value), before)
        self.assertEqual(value, Unit("local-milli", "custom", Scaled(.001)))
        with mock.patch.object(units_module, "_unit_native", side_effect=AssertionError("already bound")):
            self.assertEqual(value.decade, -3)
            self.assertEqual(value.coordinate_scale, ("custom", .001))
            self.assertIs(value.validate(), value)
        invalid = Unit("stacked", "custom", Scaled(.001), prefixes=("m",))
        with self.assertRaises(UnitError):
            UnitRegistry().register(invalid)

    def test_bulk_conversion_dtypes_and_stage_rounding_match_numpy(self):
        conversions = (Scaled(.001), Scaled(3.3), Offset(273.15), Decibel(.001), VoltageIntoLoad(50),
                       Prefixed(VoltageIntoLoad(50), .001), Prefixed(Scaled(3.3), .01))
        for dtype in (np.int16, np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128):
            values = np.array([0, 1, 2.5, 12.3], dtype=dtype)
            if np.issubdtype(dtype, np.complexfloating):
                values = values + np.array([0, .1j, .2j, .3j], dtype=dtype)
            for conversion in conversions:
                for inverse in (False, True):
                    with self.subTest(dtype=dtype, conversion=conversion, inverse=inverse), np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                        expected = reference(conversion, values, inverse)
                        actual = conversion.from_base(values) if inverse else conversion.to_base(values)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.shape, expected.shape)
                        tolerance = 3e-3 if expected.dtype == np.float16 else 4e-6 if expected.dtype in (np.float32, np.complex64) else 2e-14
                        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-15, equal_nan=True)
                        if isinstance(conversion, Scaled) and dtype not in (np.complex64, np.complex128):
                            np.testing.assert_array_equal(actual, expected)

    def test_whole_unit_conversion_retains_two_rounding_steps(self):
        values = np.array([1.2345, 33.333, -44.44], np.float32)
        source, target = DEFAULT_UNITS.resolve("ms"), DEFAULT_UNITS.resolve("µs")
        expected = reference(target.conversion, reference(source.conversion, values), True)
        actual = source.convert_value_to(values, target)
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(actual, expected)
        integer = np.array([2**63 + 1, 2**64 - 1], np.uint64)
        self.assertIs(DEFAULT_UNITS.convert(integer, "count", "count"), integer)
        empty = np.array([], np.float32)
        with self.assertRaises(UnitError):
            DEFAULT_UNITS.convert(empty, "m", "Hz")

    def test_scalar_and_complex_coercion_contract(self):
        for value in (1, np.float16(1.25), np.float32(1.25), 1.25):
            expected = reference(Scaled(.001), value)
            actual = Scaled(.001).to_base(value)
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(actual.dtype, expected.dtype)
        with self.assertRaises(TypeError):
            Offset(1).to_base([1 + 2j])
        with self.assertWarns(np.exceptions.ComplexWarning):
            actual = Offset(1).to_base(np.array([1 + 2j]))
        np.testing.assert_array_equal(actual, [2.])
        source, target = Unit("origin", "custom", Scaled(2)), Unit("shifted", "custom", Offset(1))
        with self.assertWarns(np.exceptions.ComplexWarning):
            actual = source.convert_value_to(np.array([1 + 2j]), target)
        np.testing.assert_array_equal(actual, [1.])

    def test_nonlinear_rf_negative_amplitudes_and_exact_decimal_prefixes(self):
        values = DEFAULT_UNITS.convert(np.array([-1., 0., .04]), "Vpp", "W")
        self.assertTrue(np.isnan(values[0])); self.assertEqual(values[1], 0)
        self.assertEqual(values[2], .04 * .04 / 400)
        dbm = DEFAULT_UNITS.convert(np.array([0., .001, -.001]), "W", "dBm")
        self.assertTrue(np.isneginf(dbm[0])); self.assertEqual(dbm[1], 0.); self.assertTrue(np.isnan(dbm[2]))
        self.assertEqual(DEFAULT_UNITS.convert_decimal(Decimal("0.1799999999997"), "Vpp", "mVpp"), Decimal("179.9999999997"))
        self.assertEqual(DEFAULT_UNITS.convert_decimal(Decimal("0.1"), "ns", "s"), Decimal("1e-10"))
        self.assertEqual(parse_quantity("0.1 ns", "s"), 1e-10)
        self.assertEqual(parse_quantity("5m", "s"), .005)
        self.assertEqual(parse_quantity("2G", "Hz"), 2e9)
        self.assertEqual(parse_quantity("1e999", "Hz"), float("inf"))

    def test_formatting_and_prefix_metadata_share_native_owner(self):
        native = call("units.prefixes")
        self.assertEqual([(p.symbol, p.exponent, list(p.accepts)) for p in PREFIXES],
                         [(p["symbol"], p["exponent"], p["accepts"]) for p in native])
        self.assertEqual(format_quantity(2**100 + 1, "count", prefix=NO_PREFIX), f"{2**100 + 1} count")
        self.assertEqual(format_quantity(.000001, "s"), "1 µs")
        self.assertEqual(prefix_for([1e6, .1], "Hz").symbol, "M")
        self.assertEqual(format_quantity(2000, "ms"), "2000 ms")

    def test_registry_replacement_invalidates_bounded_caches_and_old_aliases(self):
        registry = UnitRegistry([Unit("metre", "custom", Scaled(1), prefixes=("m",), aliases=("old",))])
        previous = registry.revision
        self.assertEqual(registry.resolve("mmetre").scale, .001)
        self.assertEqual(registry.family_of("old").symbol, "metre")
        registry.register(Unit("metre", "custom", Scaled(1), prefixes=("k",), aliases=("new",)), replace=True)
        self.assertEqual(registry.revision, previous + 1)
        for symbol in ("old", "mmetre"):
            with self.assertRaises(UnitError): registry.resolve(symbol)
        self.assertEqual(registry.resolve("kmetre").scale, 1000)
        self.assertEqual(registry.base_for("custom").aliases, ("new",))
        for exponent in range(1, 530): registry.resolve(f"metre^{exponent}")
        self.assertLessEqual(len(registry._cache), 512)
        self.assertLessEqual(len(registry._metadata_cache), 512)


if __name__ == "__main__":
    unittest.main()
