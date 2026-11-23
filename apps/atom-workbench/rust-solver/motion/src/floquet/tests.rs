//! The scene and the frame on a request of the reference set (refs/scenes/R4_LIN_4.json: cooling and repump beams in
//! pairs, a tweezer on an axis of its own).

use super::*;
use crate::linalg::Mat;
use crate::request::{Request, STATES};
use crate::scene::{Components, Scene};

fn request(name: &str) -> Request {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// The reference where the tweezer is dark: the fold has nothing to fold (the tweezer's sidebands are zero), so the
/// reference is the components themselves with each one's conjugate, and the Cartesian emission as three channels.
fn dark_reference(scene: &Scene, parts: &Components) -> Hilbert {
    let mut hamiltonian = Vec::new();
    for c in &parts.parts {
        hamiltonian.push(Part { key: c.key.clone(), value: c.value.clone(), gradient: c.gradient.clone() });
        if c.key.iter().any(|&k| k != 0) {
            hamiltonian.push(Part { key: c.key.iter().map(|k| -k).collect(), value: c.value.adjoint(), gradient: c.gradient.clone().map(|g| g.adjoint()) });
        }
    }
    let levels = scene.request.levels().unwrap();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut seen = Vec::new();
    for (s, l) in levels.iter().enumerate() {
        let tag = (l.manifold as u8, l.f);
        match seen.iter().position(|&t| t == tag) {
            Some(g) => groups[g].push(s),
            None => {
                seen.push(tag);
                groups.push(vec![s]);
            }
        }
    }
    let zero = || std::array::from_fn(|_| Mat::zeros(STATES, STATES));
    Hilbert {
        states: (0..STATES).collect(),
        hamiltonian,
        jumps: scene.jumps.emission.iter().map(|c| vec![Part { key: vec![0; scene.generators.len()], value: c.clone(), gradient: zero() }]).collect(),
        groups,
        lift: None,
    }
}

/// D1/D2 only for the light other than the tweezer: a 700 nm cooling beam is refused with its wavelength named; the
/// tweezer's 852 nm is not checked.
#[test]
fn a_beam_off_the_d_lines_is_refused() {
    let mut r = request("R4_LIN_4");
    assert!(r.validate().is_ok());
    assert!(!(0.765..=0.810).contains(&(std::f64::consts::TAU / r.beams[r.trap_beam].wave_number)));
    r.beams[0].wave_number = std::f64::consts::TAU / 0.700;
    let why = r.validate().unwrap_err();
    assert!(why.contains("700.0 nm") && why.contains("D1/D2"), "{why}");
}

/// Each component's gradient is the derivative of its value: central differences at h = 1e-5 µm (error h²k³/6 ≈ 1e-8 of
/// the value, k ≈ 8 rad/µm; rounding u/h ≈ 1e-11) at a point off every axis, within 1e-7 of |∇value| + 10|value|.
#[test]
fn the_scene_s_gradients_are_its_values_derivatives() {
    let scene = Scene::new(&request("R4_LIN_4")).unwrap();
    let w = vec![0.0; scene.request.beams.len()];
    let r = [0.31, -0.22, 0.57];
    let at = |r: [f64; 3]| scene.components(r, &w, 1.0);
    let parts = at(r);
    let h = 1e-5;
    for i in 0..3 {
        let (mut p, mut m) = (r, r);
        p[i] += h;
        m[i] -= h;
        let (up, down) = (at(p), at(m));
        for (k, c) in parts.parts.iter().enumerate() {
            let scale = c.gradient[i].max_abs() + 10.0 * c.value.max_abs();
            let mut worst: f64 = 0.0;
            for e in 0..STATES * STATES {
                let fd = (up.parts[k].value.data[e] - down.parts[k].value.data[e]) * (0.5 / h);
                worst = crate::linalg::larger(worst, (fd - c.gradient[i].data[e]).abs());
            }
            assert!(worst <= 1e-7 * scale, "component {k} axis {i}: {worst:e} of {scale:e}");
        }
    }
}

/// The frame's exact split at the dark edge: the tweezer's co-rotating key is one tweezer frequency (n_tw = 1), its
/// counter-rotating key −1 tweezer frequency and −2 carriers; a repump beam's key one ground hyperfine (|n_φ| = 1), a
/// cooling beam's none; the tweezer's sector is (0, 1). (Until 2026-11-23 also the static Liouvillian's and the
/// assembled system's trace, with the assembly, floquet/mod.rs.)
#[test]
fn the_dark_edge_s_frame_splits_exactly() {
    let scene = Scene::new(&request("R4_LIN_4")).unwrap();
    let w = vec![0.0; scene.request.beams.len()];
    let parts = scene.components([0.0; 3], &w, 0.0);
    let reference = dark_reference(&scene, &parts);
    let frame = Frame::new(&scene, &reference, &w);
    let tw = scene.tweezer();
    let carrier = scene.generators.len() - 1;
    let unit = |g: usize| scene.unit(g);
    assert_eq!(frame.fast_index(&unit(tw)), [0, 1, 0]);
    let counter: Vec<i32> = unit(tw).iter().zip(unit(carrier)).map(|(a, b)| -a - 2 * b).collect();
    assert_eq!(frame.fast_index(&counter), [0, -1, -2]);
    let id = |name: &str| scene.request.beams.iter().position(|b| b.id == name).unwrap();
    assert_eq!(frame.fast_index(&unit(id("repump-plus")))[0].abs(), 1);
    assert_eq!(frame.fast_index(&unit(id("cool-plus"))), [0, 0, 0]);
    let sectors = frame.sectors();
    assert_eq!(sectors.generators[tw], [0, 1]);
}
