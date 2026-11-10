//! The near light's tables (model/tabulate.rs, SPEC §2.3) against the exact elimination: R2_RECOIL's σ⁺ probe near, its responses
//! a function of the tweezer's intensity alone over the atoms' reach (the probe a plane wave: its unit's intensity the same
//! everywhere), tabulated on that one axis.
use atom_motion::field::Light;
use atom_motion::linalg::C64;
use atom_motion::model::tabulate::{Coordinate, Tables};
use atom_motion::model::{Component, Model, Point, Run, Sample, Unit, GROUND};
use atom_motion::request::Request;
use atom_motion::step::Dynamics;
use serde_json::Value;
use std::path::PathBuf;

fn scene(name: &str) -> Request {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    let line: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// R2_RECOIL with its probe near, the atom in |2,2⟩, its reach at the request's temperature and its tables.
fn r2() -> (Model, [f64; GROUND], Tables) {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let beam = request.beams.iter().position(|b| b.id == "probe").unwrap();
    let probe = light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap();
    let model = Model::new(&request, &light, &[Unit::Far { component: Component { frequency: probe, counter: false } }]).unwrap();
    let mut p = [0.0; GROUND];
    p[7] = 1.0;
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let reach = model.reach(&trap, &p, request.temperature_uk, 2e-3);
    let tables = model.tabulate(&reach, 1 << 16);
    (model, p, tables)
}

/// Points through the reach's core: along the tweezer and across it, a few at an angle.
fn points() -> Vec<[f64; 3]> {
    let mut out = vec![];
    for i in 0..9 {
        let u = -0.4 + 0.1 * i as f64;
        out.extend([[u, 0.0, 0.0], [0.0, u * 0.6, 0.0], [0.0, 0.0, u * 0.6], [u * 0.5, u * 0.3, -u * 0.2]]);
    }
    out
}

/// The largest relative difference of the generators at a point (every part's matrix and force), tabulated against exact.
fn difference(model: &Model, tables: &Tables, p: [f64; GROUND], cluster: usize, at: &Point) -> (f64, f64) {
    let (exact, fast) = (Run::new(model, p), Run::tabulated(model, tables, p));
    let (a, b) = (exact.generator(cluster, at), fast.generator(cluster, at));
    assert_eq!(a.parts.len(), b.parts.len());
    let (mut value, mut force, mut scale, mut scale_force) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.parts.iter().zip(&b.parts) {
        let mut d = x.generator.clone();
        d.add_scaled(-C64::ONE, &y.generator);
        value = value.max(d.frobenius());
        scale = scale.max(x.generator.frobenius());
        for i in 0..3 {
            let mut d = x.force[i].clone();
            d.add_scaled(-C64::ONE, &y.force[i]);
            force = force.max(d.frobenius());
            scale_force = scale_force.max(x.force[i].frobenius());
        }
    }
    (value / scale, force / scale_force)
}

/// The tweezer's intensity is the one axis (its weight the line's move in half-widths over the reach; the probe's plane wave weighs
/// nothing), and the tabulated generator -- its matrices and its force through the chain rule -- is the exact one's to the
/// interpolation's orders: halving the axis's spacing divides the largest value error over the points by about 2³ and the force's
/// by about 2² (within a factor two of each: the points meet the cubic's error shape at different places in its intervals).
#[test]
fn a_table_is_the_elimination_to_its_interpolation_s_orders() {
    let (model, p, mut tables) = r2();
    let cluster = model.cluster_of(7);
    let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).expect("a table for the running lights");
    let t = &tables.tables[index];
    assert_eq!(t.axes(), 1, "{:?} {:?}", t.layout.coordinates, t.layout.weights);
    assert!(matches!(t.layout.coordinates[0], Coordinate::Chart { index: 0, .. }), "the tweezer's intensity: {:?}", t.layout.coordinates[0]);
    let errors = |tables: &Tables| -> (f64, f64) {
        points().iter().map(|&r| difference(&model, tables, p, cluster, &Point { r, kappa: [2.0, -1.0, 0.5], t: 0.0 })).fold((0.0, 0.0), |a, b| (a.0.max(b.0), a.1.max(b.1)))
    };
    let mut levels = vec![errors(&tables)];
    for _ in 0..2 {
        tables.refine(&model, index, &[1], usize::MAX).unwrap();
        levels.push(errors(&tables));
    }
    eprintln!("value and force errors at three spacings: {levels:?}");
    let (coarse, fine) = (levels[1], levels[2]);
    let (pv, pf) = ((coarse.0 / fine.0).log2(), (coarse.1 / fine.1).log2());
    assert!((pv - 3.0).abs() <= 1.0, "value order {pv}: {levels:?}");
    assert!((pf - 2.0).abs() <= 1.0, "force order {pf}: {levels:?}");
    assert!(fine.0 < 1e-6, "the refined table's value error {}", fine.0);
}

/// The table's error reads at samples (`Model::table_reads`): with no coordinate held at the focus that moves, the whole error is
/// the interpolation's, and the axis's read is never below it on average over the samples (the read is conservative, crate::table)
/// and within a factor four of it (the end intervals' and the error's shape at the samples).
#[test]
fn a_table_s_axis_read_holds_its_whole_error() {
    let (model, _, tables) = r2();
    let cluster = model.cluster_of(7);
    let n = model.clusters()[cluster].states.len();
    let samples: Vec<Sample> = points()
        .iter()
        .map(|&r| Sample { cluster, at: Point { r, kappa: [1.0, 0.0, 0.0], t: 0.0 }, psi: (0..n).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect() })
        .collect();
    let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).unwrap();
    let reads = model.table_reads(&tables, &samples);
    let read = &reads[index];
    let mean = |v: &[atom_motion::model::UnitError]| v.iter().map(|e| e.parameter).sum::<f64>() / v.len() as f64;
    let (axis, whole) = (mean(&read.axes[0]), mean(&read.whole));
    eprintln!("mean read {axis:.3e}, mean whole error {whole:.3e}");
    assert!(whole > 0.0 && axis >= whole && axis <= 4.0 * whole, "read {axis} against the whole error {whole}");
}
