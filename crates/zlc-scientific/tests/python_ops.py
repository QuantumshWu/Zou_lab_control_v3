"""Native scientific operations against NumPy, using the real offline protocol."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_plugin.native import call


class Scientific(unittest.TestCase):
    def test_named_geometry_compiles_groups_and_selection_in_the_native_owner(self):
        from zlc_data import DatasetSchema, DomainSpec, ValueSchema, ValidityContract
        from zlc_data.axis import AxisId, AxisSpec, REPEAT, SCAN_POINT, SITE
        from zlc_data.codec import dataset_schema_to_tree, dataset_schema_from_tree
        repeat = AxisSpec(AxisId("r"), "repeat", REPEAT, 5, tuple(range(5)))
        time = AxisSpec(AxisId("time"), "time", REPEAT, 5, tuple(i * .1 for i in range(5)), unit="s", coordinate_of=repeat.axis_id)
        point = AxisSpec(AxisId("p"), "point", SCAN_POINT, 2, (1, 2))
        site = AxisSpec(AxisId("site"), "site", SITE, 3, (10, 20, 30))
        schema = DatasetSchema(DomainSpec((5,), (repeat, time), (range(5), range(5))),
                               DomainSpec((2,), (point,), (range(2),)), DomainSpec((3,), (site,)),
                               ValueSchema(ValidityContract.components(site.axis_id), np.dtype("<i2"), "V"))
        answer = call("scientific.reduce_plan", schema=dataset_schema_to_tree(schema), statistic="mean",
                      names=["repeat"], by="repeat", every=2)
        result = dataset_schema_from_tree(answer["schema"])
        self.assertEqual(result.physical_shape, (3, 2, 3))
        self.assertEqual(result.value_schema.dtype, np.dtype("<f8"))
        np.testing.assert_array_equal(result.repeat_domain.axes[0].coordinates, [.5, 2.5, 4.])
        np.testing.assert_allclose(result.repeat_domain.axes[1].coordinates, [.05, .25, .4])
        self.assertEqual(result.repeat_domain.axes[1].coordinate_of.value, "r/every-2")
        np.testing.assert_array_equal(answer["plans"][0]["codes"], [0, 0, 1, 1, 2])
        selected = call("scientific.pick_plan", schema=dataset_schema_to_tree(schema), domain=2,
                        axes=["site"], indexes=np.array([0, 2], np.int64), scalar=False, domain_pick=False)
        picked = dataset_schema_from_tree(selected["schema"])
        self.assertEqual(picked.physical_shape, (5, 2, 2))
        self.assertEqual(picked.value_schema.dtype, np.dtype("<i2"))
        np.testing.assert_array_equal(picked.cell_domain.axes[0].coordinates, [10, 30])
        with self.assertRaisesRegex(ValueError, "another coordinate"):
            call("scientific.reduce_plan", schema=dataset_schema_to_tree(schema), statistic="mean", names=["time"], by="", every=0)

    def test_complex_arithmetic_is_refused_as_the_existing_derive_contract_requires(self):
        with self.assertRaisesRegex(ValueError, "real numeric"):
            call("scientific.reduce", values=np.array([1 + 2j]), valid=True,
                 plans=[{"kind": "dense", "axes": [0], "expand": False}], statistic="sum")

    def test_typed_binary_operators_keep_integer_bits_and_float_precision(self):
        cases = [
            (np.array([2**63 + 1, 2**64 - 1], np.uint64), np.array([1, 2], np.uint64)),
            (np.array([-128, 127], np.int8), np.array([3, 2], np.int8)),
            (np.array([1.0001, 12.3], np.float16), np.array([0.02, 3.2], np.float16)),
            (np.array([1.0001, 12.3], np.float32), np.array([0.02, 3.2], np.float32)),
            (np.array([1.0001, 12.3], np.float64), np.array([0.02, 3.2], np.float64)),
            (np.array([2**63 + 1, 4], np.uint64), np.array([-1, 2], np.int64)),
        ]
        operations = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
                      "==": np.equal, ">": np.greater, "<=": np.less_equal}
        for a, b in cases:
            for op, ufunc in operations.items():
                with self.subTest(dtype=(a.dtype, b.dtype), op=op), np.errstate(all="ignore"):
                    expected = ufunc(a, b)
                    compute = np.result_type(a, b) if expected.dtype.kind == "b" else expected.dtype
                    actual = call("scientific.binary", op=op, left=a, right=b, shape=list(a.shape),
                                  valid=np.ones(a.shape, bool), dtype=expected.dtype.str, compute_dtype=compute.str)
                    self.assertEqual(actual["values"].dtype, expected.dtype)
                    np.testing.assert_array_equal(actual["values"], expected)

    def test_selection_retains_u64_and_all_planes_in_one_operation(self):
        values = (np.arange(24, dtype=np.uint64) + 2**63).reshape(2, 3, 4)
        valid = values % 2 == 0
        sigma = np.ones(values.shape, np.float64)
        answer = call("scientific.select", planes=[values, valid, sigma, None],
                      steps=[{"axis": 1, "indexes": np.array([0, 2], np.int64), "drop": False},
                             {"axis": 2, "indexes": np.array([1], np.int64), "drop": True}], expand=True)
        for got, original in zip(answer["planes"][:3], [values, valid, sigma]):
            np.testing.assert_array_equal(got, original[:, [0, 2], 1, None])
        self.assertIsNone(answer["planes"][3])

    def test_masked_dense_reductions_preserve_sem_and_dtype(self):
        rng = np.random.default_rng(3)
        for kind in (np.int16, np.uint64, np.float16, np.float32, np.float64):
            for axes in ([0], [1], [2], [0, 2], [1, 2]):
                data = (rng.random((2, 3, 4)) * 20).astype(kind)
                flags = rng.random(data.shape) > .3
                sigma = np.full(data.shape, .2)
                count = flags.sum(axis=tuple(axes))
                total = np.where(flags, data, 0).astype(np.float64).sum(axis=tuple(axes))
                with np.errstate(all="ignore"):
                    mean = total / count
                for operation in ("sum", "count", "mean", "min", "max", "std"):
                    with self.subTest(dtype=kind, axes=axes, operation=operation):
                        actual = call("scientific.reduce", values=data, valid=flags, sigma=sigma,
                                      plans=[{"kind": "dense", "axes": axes, "expand": False}],
                                      statistic=operation, errors=True)
                        if operation == "count":
                            expected = count
                        elif operation == "sum":
                            expected = total
                        elif operation == "mean":
                            expected = mean
                        elif operation in ("min", "max"):
                            minimum = operation == "min"
                            expected = (np.min if minimum else np.max)(np.where(flags, data, np.inf if minimum else -np.inf), axis=tuple(axes))
                        else:
                            expanded = mean
                            for axis in sorted(axes):
                                expanded = np.expand_dims(expanded, axis)
                            with np.errstate(all="ignore"):
                                expected = np.sqrt(np.where(flags, (data.astype(float) - expanded) ** 2, 0).sum(axis=tuple(axes)) / count)
                        np.testing.assert_allclose(actual["values"], expected, rtol=2e-14, atol=1e-14)
                        np.testing.assert_array_equal(actual["valid"], count > 0)
                        if operation == "mean":
                            np.testing.assert_array_equal(actual["dof"], np.where(count > 1, count - 1., np.inf))
                            self.assertTrue(np.isnan(actual["sigma"][count == 0]).all())
                            np.testing.assert_allclose(actual["sigma"][count == 1], .2)

    def test_all_boolean_statistics_and_nonfinite_values(self):
        data = np.array([[[True, False, True]], [[False, True, True]]])
        valid = np.array([[[True, False, True]], [[False, False, False]]])
        expected = {"count": [2, 0], "sum": [2, 0], "mean": [1., np.nan], "any": [True, False], "all": [True, True]}
        for operation, values in expected.items():
            answer = call("scientific.reduce", values=data, valid=valid, plans=[{"kind": "dense", "axes": [1, 2], "expand": False}], statistic=operation)
            np.testing.assert_array_equal(answer["values"], values)
            np.testing.assert_array_equal(answer["valid"], [True, False])
        answer = call("scientific.reduce", values=np.array([np.nan, np.inf, 5.]), valid=True,
                      plans=[{"kind": "dense", "axes": [0], "expand": False}], statistic="mean", errors=True)
        self.assertEqual(answer["values"], 5.)
        self.assertTrue(np.isnan(answer["sigma"]))
        answer = call("scientific.reduce", values=np.array([1e308, 1e308]), valid=True,
                      plans=[{"kind": "dense", "axes": [0], "expand": False}], statistic="sum")
        self.assertFalse(answer["valid"])
        answer = call("scientific.binary", left=np.array([np.nan, 2.]), right=np.array([1., 1.]),
                      shape=[2], op=">", dtype="|b1", compute_dtype="<f8")
        np.testing.assert_array_equal(answer["valid"], [False, True])

    def test_mapped_reduction_preserves_first_seen_group_indexes(self):
        data = np.array([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]])
        answer = call("scientific.reduce", values=data, valid=True,
                      plans=[{"kind": "mapped", "axis": 1, "codes": np.array([1, 0, 1, 0], np.int64), "size": 2}], statistic="sum")
        np.testing.assert_array_equal(answer["values"], [[[10., 12.], [6., 8.]]])
        with self.assertRaisesRegex(ValueError, "cover"):
            call("scientific.reduce", values=data, valid=True,
                 plans=[{"kind": "mapped", "axis": 1, "codes": np.array([1, 0], np.int64), "size": 2}], statistic="sum")


if __name__ == "__main__":
    unittest.main()
