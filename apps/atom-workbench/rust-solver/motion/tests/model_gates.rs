//! The existence gates over the reach on the solve's own closure (SPEC §1.5, §5.2): a scene's model closed as solve.rs `attempt`
//! closes it -- every unit whose gate fails kept, the model made again until none fails --, each pass's unit counts, the time of
//! each of its calls, and the digest of its gates: every candidate unit, whether its fold exists, and its two bounds to the bit.
//! The digest is what a change of the gates' evaluation must leave as it is.
use atom_motion::estimate::unseen;
use atom_motion::field::Light;
use atom_motion::model::{Model, Unit, GROUND};
use atom_motion::request::Request;
use atom_motion::units::MOST_TRAJECTORIES;
use std::fmt::Write;
use std::path::Path;
use std::time::Instant;

mod model_common;
use model_common::scene;

/// The time (s) of a fixed arithmetic loop -- 2²⁶ steps of the logistic map, one dependent chain -- on one core, and on every
/// core of the pool at once: the machine's own pace beside each time printed, as other work shares it.
fn yardstick() -> [f64; 2] {
    let chain = || {
        let mut x = 0.5_f64;
        for _ in 0..1 << 26 {
            x = std::hint::black_box(3.7 * x * (1.0 - x));
        }
        std::hint::black_box(x);
    };
    let time = |run: &dyn Fn()| {
        let s = Instant::now();
        run();
        s.elapsed().as_secs_f64()
    };
    [time(&chain), time(&|| drop(rayon::broadcast(|_| chain())))]
}

/// The reference state's ground populations as solve.rs `attempt` takes them: the initial density's largest ground component
/// (of equal ones the last), alone.
fn reference(request: &Request) -> [f64; GROUND] {
    let diagonal: Vec<f64> = (0..GROUND).map(|s| request.initial_density[(s, s)].re).collect();
    let total: f64 = diagonal.iter().sum();
    let state = (0..GROUND).filter(|&s| diagonal[s] > 0.0).map(|s| (s, diagonal[s] / total)).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 })
}

/// A scene's closure (solve.rs `attempt`, from no unit kept, the reach at the request's temperature): its digest, and on stderr
/// each pass's times.
fn closure(name: &str) -> String {
    let request = scene(name);
    let light = Light::new(&request);
    let reference = reference(&request);
    let mut kept: Vec<Unit> = vec![];
    let mut digest = String::new();
    for pass in 0.. {
        let mut times = vec![];
        let mut timed = |what: &'static str, from: Instant| times.push(format!("{what} {:.3} s", from.elapsed().as_secs_f64()));
        let s = Instant::now();
        let model = Model::new(&request, &light, &kept).unwrap();
        timed("new", s);
        let s = Instant::now();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        timed("trap", s);
        let s = Instant::now();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        timed("reach", s);
        let s = Instant::now();
        let units = model.units();
        timed("units", s);
        let s = Instant::now();
        let gates = model.gates(&units, &reach);
        timed("gates", s);
        let failing: Vec<Unit> = units.iter().zip(&gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        let clusters: Vec<(usize, usize)> = model.clusters().iter().map(|c| (c.ground, c.states.len() - c.ground)).collect();
        let [one, all] = yardstick();
        eprintln!("{name} pass {pass}: kept {}, folded {}, failing {}; {}; yardstick {one:.3} s on one core, {all:.3} s on all", kept.len(), units.len(), failing.len(), times.join(", "));
        writeln!(digest, "pass {pass}: kept {}, folded {}, failing {}; clusters (ground, explicit) {clusters:?}", kept.len(), units.len(), failing.len()).unwrap();
        for (u, g) in units.iter().zip(&gates) {
            writeln!(digest, "{} | {u:?} | exists {} | S {:e} {:016x} | a {:e} {:016x}", model.label(u), g.exists(), g.saturation, g.saturation.to_bits(), g.sweep, g.sweep.to_bits()).unwrap();
        }
        if failing.is_empty() {
            break;
        }
        kept.extend(failing);
    }
    digest
}

/// The closures of the scenes `GATES_SCENES` names (reference scenes by name or native requests by path, `;` between them):
/// each one's digest written to the folder `GATES_DIGESTS` as `<the scene's file stem>.txt`, the passes' times on stderr.
#[test]
#[ignore]
fn the_closure_s_gates() {
    let scenes = std::env::var("GATES_SCENES").expect("GATES_SCENES: the scenes, `;` between them");
    let folder = std::env::var("GATES_DIGESTS").expect("GATES_DIGESTS: the folder the digests are written to");
    for name in scenes.split(';').filter(|s| !s.is_empty()) {
        let digest = closure(name);
        let stem = Path::new(name).file_stem().unwrap().to_str().unwrap();
        std::fs::write(Path::new(&folder).join(format!("{stem}.txt")), digest).unwrap();
    }
}
