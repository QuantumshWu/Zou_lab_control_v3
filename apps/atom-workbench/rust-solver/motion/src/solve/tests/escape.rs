//! LOSSCAP's check of the loss at the escape value (SPEC §1.7, a measurement, ignored): whether losing an atom where its
//! loss energy first passes the far value, wherever it is, changes a scene's readings against the loss that also asked
//! the atom to be outside the trap's region (until 2026-11-21). The run's own model, ensemble, plan and step (the
//! product's trajectories by their numbers) through the solve's `run`; a model that hands every call on to the run's
//! notes each loss: its time and whether it was inside the region -- a loss the old test would not have made then. Where
//! there are none before the last output's window, no reading changes: the two tests agree on every other state (an
//! atom outside the region above the far value is lost by both), and the trajectories are the same to the bit up to
//! their first disagreement.
//!
//! LOSSCAP_SCENE one of the six scenes (pgc3d, r4, ...; V3D_REQ: their requests), LOSSCAP_GROUPS "first..last" (group
//! numbers; trajectory = group·8 + member), LOSSCAP_TAU the step (µs, rounded to the run's end over an even number).
use super::super::*;
use crate::linalg::C64;
use crate::model::selected::SelectedMiddle;
use crate::model::{Cluster, Point, Quadratic};
use crate::step::{Emitted, Middle};
use std::cell::RefCell;

/// A scene's request by its name (V3D_REQ, the outside forms' transitions filled in, as select/tests.rs reads them).
fn request_of(name: &str) -> Request {
    let dir = std::path::PathBuf::from(std::env::var("V3D_REQ").expect("V3D_REQ: the six scenes' requests"));
    let file = match name {
        "gmd1c" => "GMD1C_1ms.json",
        "pgc3d" => "PGC3D_1ms.json",
        "r4" => "R4_LIN_4.json",
        "tet" => "TET_1ms.json",
        "tilt0" => "tilt0.json",
        "tilt2" => "tilt2.json",
        other => panic!("unknown scene {other}"),
    };
    let p = dir.join(file);
    let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap();
    line["options"]["device"] = serde_json::json!("cpu");
    for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
        form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
    }
    Request::from_native(&line).unwrap()
}

/// The run's model with its loss noted (time, inside the region): the same model to the bit, each call handed on --
/// or, `old`, with the loss of until 2026-11-21, outside the trap's region as well.
struct Watched<'a> {
    inner: &'a Selected,
    loss: RefCell<Option<(f64, bool)>>,
    old: bool,
}

impl crate::step::Dynamics for Watched<'_> {
    type Middle<'b>
        = SelectedMiddle<'b>
    where
        Self: 'b;

    fn hbar_over_mass(&self) -> f64 {
        self.inner.hbar_over_mass()
    }
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.inner.reference(r)
    }
    fn cluster(&self, cluster: usize) -> &Cluster {
        self.inner.cluster(cluster)
    }
    fn cluster_of(&self, ground: usize) -> usize {
        self.inner.cluster_of(ground)
    }
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> SelectedMiddle<'_> {
        self.inner.middle(cluster, r, t)
    }
    fn emit<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Self::Middle<'m>, &<Self::Middle<'m> as Middle>::Path, f64)>, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        self.inner.emit(cluster, at, on, psi, draw)
    }
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        self.inner.energy(cluster, at, psi)
    }
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        let lost = self.inner.lost(cluster, at, psi) && !(self.old && !self.inner.light().outside(at.r));
        if lost && self.loss.borrow().is_none() {
            *self.loss.borrow_mut() = Some((at.t, !self.inner.light().outside(at.r)));
        }
        lost
    }
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.inner.readout(cluster, at, psi)
    }
}

/// The scene's trajectories (LOSSCAP_GROUPS) at the run's step through its plan, each loss noted: per trajectory its
/// initial loss energy over the depth U₀, and where lost its time and whether inside the region. Says the counts, the
/// largest initial energy, and every loss inside the region with its time against the last output's.
#[test]
#[ignore]
fn losscap_the_escape_value_against_the_region() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
    let name = var("LOSSCAP_SCENE");
    let groups: Vec<u64> = {
        let g = var("LOSSCAP_GROUPS");
        let (a, b) = g.split_once("..").unwrap();
        (a.parse().unwrap()..b.parse().unwrap()).collect()
    };
    let clock = Instant::now();
    let request = request_of(&name);
    let b = chief(&request);
    let reference = alone(b);
    let run = run_of(&request, reference).unwrap();
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(&request);
    let at = initial.iter().position(|&(s, _)| s == b).unwrap();
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s)).unwrap()).collect();
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
    let trap = reported(&run, &reference).unwrap_or_else(|_| wells[at].trap.clone());
    let mut plan = Plan::new(&request.times_us, trap.lab);
    let pairs = (plan.end / (2. * var("LOSSCAP_TAU").parse::<f64>().unwrap())).round();
    let tau = plan.end / (2. * pairs);
    plan.cell = 4. * tau;
    let domain = Domain { widest: Widest::of(&request) };
    let depth = -run.prepared(&reference).at(wells[at].trap.centre).value;
    let last = *request.times_us.last().unwrap();
    eprintln!("{name}: P{} τ {tau} µs, run to {} µs (last output {last} µs), U₀ {depth} rad/µs, groups {}..{}; built in {:.1} s", run.cluster(0).states.len(), plan.end, groups[0], groups.last().unwrap() + 1, clock.elapsed().as_secs_f64());
    let indices: Vec<u64> = groups.iter().flat_map(|&g| (0..GROUP).map(move |m| g * GROUP + m)).collect();
    let started = Instant::now();
    let one = |index: u64, old: bool| {
        let watched = Watched { inner: &run, loss: RefCell::new(None), old };
        let record = super::super::run(&watched, &domain, ensemble.draw(SEED, index), index, tau, &plan, false);
        (record, watched.loss.into_inner())
    };
    let results: Vec<(u64, f64, Option<(f64, bool)>)> = indices
        .par_iter()
        .map(|&index| {
            let start = ensemble.draw(SEED, index);
            let t0 = Trajectory::new(&run, start.state, start.r, start.kappa, SEED, index);
            let e0 = run.energy(t0.cluster, &t0.point(), &t0.psi) / depth;
            (index, e0, one(index, false).1)
        })
        .collect();
    let highest = results.iter().map(|r| r.1).fold(f64::NEG_INFINITY, crate::linalg::larger);
    let lost: Vec<(u64, f64, bool)> = results.iter().filter_map(|r| r.2.map(|(t, inside)| (r.0, t, inside))).collect();
    let inside: Vec<(u64, f64)> = lost.iter().filter(|l| l.2).map(|l| (l.0, l.1)).collect();
    let before = inside.iter().filter(|l| l.1 <= last).count();
    eprintln!(
        "{name}: {} trajectories in {:.1} s; initial loss energy at most {highest:.4} U₀ ({} above 0); {} lost, {} of them inside the region (the old test held them then) at {:?} (trajectory, µs), {before} before the last output",
        results.len(),
        started.elapsed().as_secs_f64(),
        results.iter().filter(|r| r.1 > 0.).count(),
        lost.len(),
        inside.len(),
        inside
    );
    // Each one lost inside the region, again under the old test: its fate and window means at every output, both ways.
    for &(index, at) in &inside {
        let ((new, _), (old, then)) = (one(index, false), one(index, true));
        for (k, (a, b)) in new.outputs.iter().zip(&old.outputs).enumerate() {
            if a != b {
                let kinetic = |f: &Fate| if let Fate::Held(h) = f { format!("held, Σ_a ⟨κ_a²⟩ {:.6e} /µm², Σ_a ⟨κ_a⟩² {:.6e}", h.kappa2.iter().sum::<f64>(), h.kappa.iter().map(|x| x * x).sum::<f64>()) } else { format!("{f:?}") };
                eprintln!("{name} trajectory {index} (lost at {at} µs; under the old test {then:?}): output {k} at {} µs -- {} against the old {}", request.times_us[k], kinetic(a), kinetic(b));
            }
        }
    }
}
