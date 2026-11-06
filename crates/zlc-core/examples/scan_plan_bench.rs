//! Run with `cargo run --release -p zlc-core --example scan_plan_bench`.
//! Reports each operation separately; stored coordinate bytes are not process RSS.
use std::hint::black_box;
use std::time::Instant;

use zlc_core::DType;
use zlc_core::schema::{DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::scan::{AxisSpec, Plan, PlanSpec, Scope};

fn bench(width: usize, length: usize) {
    let spec = PlanSpec {
        axes: (0..width).map(|index| AxisSpec {
            port: format!("axis{index}"), identity: format!("axis{index}"), label: format!("axis{index}"), unit: "Hz".into(),
            values: (0..length).map(|i| i as f64).collect(), coordinate_of: None, host_advanced: index == 0,
        }).collect(),
        scan_repeats: 1, run_repeats: 1,
    };
    let scalar = Domain::new(vec![1], vec![], vec![]).unwrap();
    let source = DatasetSchema::new(scalar.clone(), scalar.clone(), scalar,
        ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
    let plan = Plan::new(spec.clone()).unwrap();
    let started = Instant::now();
    for _ in 0..1000 { black_box(Plan::new(black_box(spec.clone())).unwrap()); }
    let plan_us = started.elapsed().as_secs_f64() * 1e6 / 1000.0;
    let started = Instant::now();
    for _ in 0..1000 { black_box(plan.dataset_schema(black_box(&source)).unwrap()); }
    let schema_us = started.elapsed().as_secs_f64() * 1e6 / 1000.0;
    let started = Instant::now();
    for i in 0..100_000 { black_box(plan.row(Scope::All, black_box(i * 997 % plan.point_count())).unwrap()); }
    let row_ns = started.elapsed().as_secs_f64() * 1e9 / 100_000.0;
    println!("axes={width} values_per_axis={length} points={} authored_coordinate_bytes={} build_us={plan_us:.3} schema_us={schema_us:.3} random_row_ns={row_ns:.3}",
        plan.point_count(), plan.axes().iter().map(|axis| axis.values.len() * size_of::<f64>()).sum::<usize>());
}

fn main() {
    bench(2, 1000);
    bench(3, 100);
}
