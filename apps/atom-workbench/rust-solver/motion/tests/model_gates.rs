//! The existence gates over the reach on the solve's own closure (SPEC §1.5, §5.2): a scene's model closed as solve.rs `attempt`
//! closes it -- every unit whose gate fails kept, the model made again until none fails --, each pass's unit counts, the time of
//! each of its calls, and the digest of its gates: every candidate unit, whether its fold exists, and its two bounds to the bit;
//! then the closed model's units' errors at a few states of each cluster, to the bit -- the gates' bounds on a point are the
//! couplings' measured parameters (units.rs). The digests are what a change of the gates' evaluation must leave as it is.
use atom_motion::estimate::unseen;
use atom_motion::field::Light;
use atom_motion::linalg::C64;
use atom_motion::model::{Model, Point, Sample, Unit, GROUND};
use atom_motion::request::Request;
use atom_motion::units::{KT_PER_UK, MOST_TRAJECTORIES};
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

/// The closed model's units' errors at three states of each cluster -- the trap's centre at rest, and two points off it moving
/// at the request's thermal momentum's scale, every state of the cluster equally held with phases apart --, as a digest: per
/// unit and state its parameter and its errors on the populations and on T, to the bit; the time per state on stderr.
fn errors(name: &str, model: &Model, units: &[Unit], centre: [f64; 3], temperature_uk: f64) -> String {
    let thermal = (KT_PER_UK * temperature_uk / model.hbar_over_mass()).sqrt();
    let samples: Vec<Sample> = (0..model.clusters().len())
        .flat_map(|cluster| {
            let n = model.clusters()[cluster].states.len();
            let psi: Vec<C64> = (0..n).map(|x| C64::cis(x as f64) * (1.0 / n as f64).sqrt()).collect();
            (0..3).map(move |k| {
                let s = k as f64;
                let r = [centre[0] + 0.03 * s, centre[1] - 0.02 * s, centre[2] + 0.11 * s];
                Sample { cluster, at: Point { r, kappa: [0.6 * s * thermal, -0.5 * s * thermal, 0.4 * s * thermal], t: 0.0 }, psi: psi.clone() }
            })
        })
        .collect();
    let s = Instant::now();
    let errors = model.errors(units, &samples);
    eprintln!("{name}: errors of {} units at {} states, {:.4} s a state", units.len(), samples.len(), s.elapsed().as_secs_f64() / samples.len() as f64);
    let mut digest = String::new();
    for (u, of) in units.iter().zip(&errors) {
        write!(digest, "{u:?}").unwrap();
        for e in of {
            write!(digest, " | {:e} {:016x} {:016x} {:016x}", e.parameter, e.parameter.to_bits(), e.population.to_bits(), e.temperature.to_bits()).unwrap();
        }
        writeln!(digest).unwrap();
    }
    digest
}

/// A scene's closure (solve.rs `attempt`, from no unit kept, the reach at the request's temperature): its gates' digest and its
/// closed model's errors' digest, and on stderr each pass's times. With `GATES_REACH_UK` the closure then goes on with the
/// reach at that temperature (µK), from the units kept so far -- an attempt of the solve after its reach has grown.
fn closure(name: &str) -> [String; 2] {
    let request = scene(name);
    let light = Light::new(&request);
    let reference = reference(&request);
    let mut kept: Vec<Unit> = vec![];
    let mut digest = String::new();
    let grown: Option<f64> = std::env::var("GATES_REACH_UK").ok().map(|uk| uk.parse().expect("GATES_REACH_UK: µK"));
    let mut at = request.temperature_uk;
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
        let reach = model.reach(&trap, &reference, at, unseen(MOST_TRAJECTORIES as f64));
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
        if failing.is_empty() && grown.is_none_or(|uk| uk == at) {
            return [digest, errors(name, &model, &units, trap.centre, request.temperature_uk)];
        }
        if failing.is_empty() {
            at = grown.expect("a grown reach");
        }
        kept.extend(failing);
    }
    unreachable!("the closure ends in a pass without a failing gate")
}

/// A scene's closed model (the closure of solve.rs `attempt`, as `closure`): the units kept and its clusters, each as
/// (ground states, explicit excited states).
fn closed(name: &str) -> (Vec<Unit>, Vec<(usize, usize)>) {
    let request = scene(name);
    let light = Light::new(&request);
    let reference = reference(&request);
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(&request, &light, &kept).unwrap();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let gates = model.gates(&units, &reach);
        let failing: Vec<Unit> = units.iter().zip(&gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            return (kept, model.clusters().iter().map(|c| (c.ground, c.states.len() - c.ground)).collect());
        }
        kept.extend(failing);
    }
}

/// A light far from its lines stays folded beside a line that is not, whatever its intensity (SPEC §1.5): in the
/// lin-perp-lin scenes at 40, 80 and 160 mW/cm² a beam, 20 linewidths red, the closure makes explicit the repump light's
/// drives F=1 → F′=2 alone (resonant: their gate fails, S 0.6 to 1.4) with the three F′=2 states they reach from F=1
/// m=±1 and the two from m=0 -- the same clusters at every intensity. No cooling drive (from F=2, the ground states from
/// the fourth on) is explicit, and no coupling joins F=1 to F=2: the cooling light's own saturation is at most 0.12
/// there, and its shift of a level is its coupling across its own 121 MHz, a hundredth of the two-photon detuning.
#[test]
fn a_far_detuned_light_stays_folded_beside_an_explicit_line() {
    for name in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
        let (kept, clusters) = closed(name);
        assert_eq!(clusters, [(2, 3), (1, 2), (3, 0), (2, 0)], "{name}: the closed model's clusters");
        for unit in &kept {
            match unit {
                Unit::Drive { ground, .. } => assert!(*ground < 3, "{name}: a cooling drive is explicit: {unit:?}"),
                Unit::Term { a, b, .. } => assert!((*a < 3) == (*b < 3) || *a >= GROUND, "{name}: F=1 joined to F=2: {unit:?}"),
                _ => {}
            }
        }
    }
}

/// The closures of the scenes `GATES_SCENES` names (reference scenes by name or native requests by path, `;` between them):
/// each one's digests written to the folder `GATES_DIGESTS` as `<the scene's file stem>.txt` (the gates) and `.errors.txt` (the
/// closed model's errors), the times on stderr; with `GATES_REACH_UK` each closure goes on over a reach grown to it (`closure`).
#[test]
#[ignore]
fn the_closure_s_gates() {
    let scenes = std::env::var("GATES_SCENES").expect("GATES_SCENES: the scenes, `;` between them");
    let folder = std::env::var("GATES_DIGESTS").expect("GATES_DIGESTS: the folder the digests are written to");
    for name in scenes.split(';').filter(|s| !s.is_empty()) {
        let [gates, errors] = closure(name);
        let stem = Path::new(name).file_stem().unwrap().to_str().unwrap();
        std::fs::write(Path::new(&folder).join(format!("{stem}.txt")), gates).unwrap();
        std::fs::write(Path::new(&folder).join(format!("{stem}.errors.txt")), errors).unwrap();
    }
}
