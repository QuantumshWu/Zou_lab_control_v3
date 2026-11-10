// TEMPORARY timing harness (not to be committed): the closure, the near tables, the assembly's time and count exact and tabulated.
use atom_motion::field::Light;
use atom_motion::model::{Model, Point, Run, Unit, GROUND};
use atom_motion::request::Request;
use atom_motion::step::Dynamics;
use serde_json::Value;
use std::path::PathBuf;
fn scene(name: &str) -> Request {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    let line: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}
fn closed(request: &Request, light: &Light, p: &[f64; GROUND]) -> Model {
    let share = atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64);
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(request, light, &kept).unwrap();
        let trap = model.trap(p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, p, request.temperature_uk, share);
        let units = model.units();
        let gates = model.gates(&units, &reach);
        let failing: Vec<Unit> = units.iter().zip(&gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            return model;
        }
        kept.extend(failing);
    }
}
#[test]
fn closure_timing() {
    let name = std::env::var("SCENE").unwrap();
    let request = scene(&name);
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = {
        let mut p = [0.0; GROUND];
        p[b] = 1.0;
        p
    };
    let model = closed(&request, &light, &p);
    let s = std::time::Instant::now();
    let reach = { let t = model.trap(&p, &model.prepared()).unwrap(); model.reach(&t, &p, request.temperature_uk, atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64)) };
    let tables = model.tabulate(&reach, 1 << 20);
    eprintln!("{name}: tabulate {:?}, work {:.3e} complex MA", s.elapsed(), tables.work);
    for t in &tables.tables {
        eprintln!("  cluster {} lights {:?}: axes {:?} nodes {:?} width {}, weights {:?}", t.layout.cluster, t.layout.lights, &t.layout.coordinates[..t.axes()], t.table.axes().iter().map(|a| a.nodes).collect::<Vec<_>>(), t.table.width(), t.layout.weights.iter().map(|w| format!("{w:.2e}")).collect::<Vec<_>>());
    }
    let exact = Run::new(&model, p);
    let fast = Run::tabulated(&model, &tables, p);
    let cluster = model.cluster_of(b);
    let at = Point { r: [0.05, -0.03, 0.02], kappa: [1.0, 0.0, 0.0], t: 0.0 };
    let n = 200;
    let time = |run: &Run| {
        let s = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(run.generator(cluster, &at));
        }
        s.elapsed() / n
    };
    let (te, tf) = (time(&exact), time(&fast));
    let (ge, gf) = (exact.generator(cluster, &at), fast.generator(cluster, &at));
    let mut diff = 0.0_f64;
    let mut size = 0.0_f64;
    for (a, b) in ge.parts.iter().zip(&gf.parts) {
        let mut d = a.generator.clone();
        d.add_scaled(atom_motion::linalg::C64::new(-1.0, 0.0), &b.generator);
        diff = diff.max(d.frobenius());
        size = size.max(a.generator.frobenius());
    }
    eprintln!("{name}: assembly exact {te:?} ({:.3e} counted), tabulated {tf:?} ({:.3e} counted); parts {} {}, largest part difference {diff:.3e} of {size:.3e}", exact.assembly_work(), fast.assembly_work(), ge.parts.len(), gf.parts.len());
}
