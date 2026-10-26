//! Benchmark and regression harness for the population-core kernels.
//!
//! The test is ignored by default. It loads real `rb87-full/2` requests,
//! prepares the model exactly like `population_core_adapter`, builds graph-ball
//! domains of the drive charges and then
//!   (a) times the E/F pencil, the shifted pencil M, P0 and the REAL map, and
//!   (b) saves or compares the outputs for a fixed pseudo-random input.
//!
//! Environment:
//!   ATOM_KERNEL_INPUTS   directory holding `<case>.request.json` (required)
//!   ATOM_KERNEL_CASES    comma-separated cases (default threeNearStrong,fourNear,fiveNearPrepared)
//!   ATOM_KERNEL_ORDERS   comma-separated domain orders (default 1,2)
//!   ATOM_KERNEL_RHS      comma-separated right-hand-side counts (default 1,4)
//!   ATOM_KERNEL_SECONDS  minimum timed seconds per kernel (default 0.25; 0 disables timing)
//!   ATOM_KERNEL_SAVE     directory: write the raw outputs (reference run)
//!   ATOM_KERNEL_COMPARE  directory: compare against previously saved outputs
//!   ATOM_KERNEL_REPORT   file: append one JSON line per (case, order, rhs)
use crate::{harmonics, Model, Request};
use atom_population_core::owned_prepared::b0_inverse::P0Policy;
use atom_population_core::owned_prepared::coordinates::InitialView;
use atom_population_core::owned_prepared::finite_driver::support_domain;
use atom_population_core::owned_prepared::runtime_core::RuntimeCore;
use atom_population_core::owned_prepared::KernelPolicy;
use atom_population_core::preparation::{
    Complex64, DriveView, FrequencyParts, GaugePolicy, MatrixView, ModelView, PhaseBasis,
};
use atom_population_core::{STAGE_COUNT, STAGE_NAMES};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Pole used for the shifted pencil and the P0 shifts (production uses
/// (samples - 1) / duration, about 10 per microsecond for 200 samples / 20 us).
const POLE: f64 = 10.0;

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn load_model(path: &Path) -> Model {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let request: Request = serde_json::from_str(&text).expect("request JSON");
    request.model.expect("request has a model")
}

/// Same preparation as `population_core_adapter::solve_inner`.
fn prepare_core(model: &Model) -> RuntimeCore {
    let ground: Vec<bool> = model.states.iter().map(|label| label.split('-').next() == Some("ground")).collect();
    let declared = model.frequency_coordinates.as_ref().expect("declared phase coordinates");
    let axes = declared.frequencies.len();
    let identity: Vec<Vec<i32>> =
        (0..axes).map(|i| (0..axes).map(|j| i32::from(i == j)).collect()).collect();
    let parts = harmonics::model_phase_frequency_parts(model, &identity).unwrap();
    let parts: Vec<_> = parts.into_iter().map(|(hi, lo)| FrequencyParts { hi, lo }).collect();
    let phases = PhaseBasis::from_parts(&parts, &declared.drive_indices).unwrap();
    let collapse: Vec<_> = model.collapse.iter().map(|x| MatrixView { real: &x.real, imag: &x.imag }).collect();
    let drives: Vec<_> = model
        .drives
        .iter()
        .map(|x| DriveView {
            omega: x.omega,
            cosine: MatrixView { real: &x.cos.real, imag: &x.cos.imag },
            sine: MatrixView { real: &x.sin.real, imag: &x.sin.imag },
        })
        .collect();
    let view = ModelView {
        dimension: model.dimension,
        hamiltonian: MatrixView { real: &model.h_real, imag: &model.h_imag },
        collapse: &collapse,
        drives: &drives,
        ground: &ground,
        initial_index: model.initial_index,
    };
    let initial = InitialView {
        index: model.initial_index,
        real: model.initial_real.as_deref(),
        imag: model.initial_imag.as_deref(),
    };
    RuntimeCore::prepare(
        &view,
        initial,
        phases,
        GaugePolicy::UnsafeComponentAligned,
        KernelPolicy::HybridBlas,
        P0Policy::EigenPreferred,
    )
    .unwrap()
}

/// Deterministic values in [-1, 1) (64-bit LCG, top 53 bits).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn complex(re: f64, im: f64) -> Complex64 {
    Complex64 { re, im }
}

/// (median, minimum) wall time in milliseconds of `action`, repeated for at
/// least `seconds` (and at least three times) after one warm-up call. The
/// minimum is the robust figure on a machine shared with other jobs.
fn time_ms(seconds: f64, mut action: impl FnMut()) -> (f64, f64) {
    action();
    let mut samples = Vec::new();
    let began = Instant::now();
    while samples.len() < 3 || (began.elapsed().as_secs_f64() < seconds && samples.len() < 2000) {
        let start = Instant::now();
        action();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2], samples[0])
}

fn as_f64(values: &[Complex64]) -> Vec<f64> {
    values.iter().flat_map(|z| [z.re, z.im]).collect()
}

fn write_f64(path: &Path, values: &[f64]) {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()
}

/// (||new - old|| / ||old||, max |new - old| / max |old|).
fn relative_difference(new: &[f64], old: &[f64]) -> (f64, f64) {
    assert_eq!(new.len(), old.len(), "output length changed");
    let (mut diff, mut norm, mut diff_max, mut max) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (a, b) in new.iter().zip(old) {
        diff += (a - b) * (a - b);
        norm += b * b;
        diff_max = diff_max.max((a - b).abs());
        max = max.max(b.abs());
    }
    (diff.sqrt() / norm.sqrt().max(f64::MIN_POSITIVE), diff_max / max.max(f64::MIN_POSITIVE))
}

struct Outputs(Vec<(&'static str, Vec<f64>)>);

fn run_case(name: &str, inputs: &Path, orders: &[usize], rhs_list: &[usize], seconds: f64) {
    let began = Instant::now();
    let model = load_model(&inputs.join(format!("{name}.request.json")));
    let mut core = prepare_core(&model);
    println!("[{name}] prepared in {:.0} ms", began.elapsed().as_secs_f64() * 1e3);
    let source = &core.prepared().primitives().hilbert.source;
    let axes = source.phases.axes.len();
    let steps = source.phases.drive_indices.clone();
    let phases = source.phases.clone();
    let max_rhs = rhs_list.iter().copied().max().unwrap();
    for &order in orders {
        let modes = support_domain(axes, &steps, order).unwrap();
        let shifts: Vec<Complex64> = modes
            .iter()
            .map(|q| phases.frequency(q).map(|f| complex(POLE, f.hi + f.lo)))
            .collect::<Result<_, _>>()
            .unwrap();
        let setup = Instant::now();
        let (owned, request_p0) = core.operator_context();
        let map = owned.real_fourier_map(&modes).unwrap();
        let mut pencil = owned.pencil(&modes, &modes, max_rhs).unwrap();
        let pencil_ms = setup.elapsed().as_secs_f64() * 1e3;
        let setup = Instant::now();
        let mut p0 = request_p0.prepare_batch(&shifts, max_rhs).unwrap();
        let p0_setup_ms = setup.elapsed().as_secs_f64() * 1e3;
        let rank = request_p0.factor().rank();
        let diagnostics = pencil.diagnostics().clone();
        println!(
            "[{name} order {order}] modes {} D {} J {} Z {} rank {rank}; pencil setup {pencil_ms:.1} ms, P0 setup {p0_setup_ms:.1} ms",
            modes.len(),
            diagnostics.first_intermediate_modes,
            diagnostics.j_intermediate_modes,
            diagnostics.z_intermediate_modes
        );
        for &rhs in rhs_list {
            let rows = map.dimension();
            let length = rows * rhs;
            let mut random = Lcg(0x5eed_0000 + (order as u64) * 131 + rhs as u64);
            let real: Vec<f64> = (0..length).map(|_| random.next()).collect();
            let complex_coordinates: Vec<Complex64> =
                (0..length).map(|_| complex(random.next(), random.next())).collect();
            let zero = Complex64::default();
            let mut signed = vec![zero; length];
            let mut signed_complex = vec![zero; length];
            let mut e = vec![zero; length];
            let mut f = vec![zero; length];
            let mut m = vec![zero; length];
            let mut m_complex = vec![zero; length];
            let mut p = vec![zero; length];
            let mut p_complex = vec![zero; length];
            let mut pulled_e = vec![zero; length];
            let mut pulled_f = vec![zero; length];
            let mut pulled_m = vec![zero; length];
            let mut proposal = vec![0.0; length];
            let shift = complex(POLE, 0.0);
            let complex_shift = complex(POLE, 0.37 * POLE);

            // Reference outputs.
            map.push_into(&real, rhs, &mut signed).unwrap();
            map.push_complex_into(&complex_coordinates, rhs, &mut signed_complex).unwrap();
            pencil.apply_into(&signed, rhs, &mut e, &mut f).unwrap();
            pencil.apply_shifted_into(&signed, rhs, shift, &mut m).unwrap();
            pencil.apply_shifted_into(&signed_complex, rhs, complex_shift, &mut m_complex).unwrap();
            p0.apply_into(&signed, rhs, &mut p).unwrap();
            p0.apply_into(&signed_complex, rhs, &mut p_complex).unwrap();
            map.pull_complex_into(&e, rhs, &mut pulled_e).unwrap();
            map.pull_complex_into(&f, rhs, &mut pulled_f).unwrap();
            map.pull_complex_into(&m, rhs, &mut pulled_m).unwrap();
            map.pull_real_proposal_into(&p, rhs, &mut proposal).unwrap();
            let real_part = |values: &[Complex64]| values.iter().map(|z| z.re).collect::<Vec<_>>();
            let outputs = Outputs(vec![
                ("signed", as_f64(&signed)),
                ("E", as_f64(&e)),
                ("F", as_f64(&f)),
                ("M", as_f64(&m)),
                ("Mcomplex", as_f64(&m_complex)),
                ("P0", as_f64(&p)),
                ("P0complex", as_f64(&p_complex)),
                ("pullE", real_part(&pulled_e)),
                ("pullF", real_part(&pulled_f)),
                ("pullM", real_part(&pulled_m)),
                ("proposalP0", proposal.clone()),
            ]);
            let tag = format!("{name}_o{order}_r{rhs}");
            let mut differences = serde_json::Map::new();
            if let Ok(dir) = std::env::var("ATOM_KERNEL_SAVE") {
                std::fs::create_dir_all(&dir).unwrap();
                for (label, values) in &outputs.0 {
                    write_f64(&PathBuf::from(&dir).join(format!("{tag}_{label}.bin")), values);
                }
            }
            if let Ok(dir) = std::env::var("ATOM_KERNEL_COMPARE") {
                let mut line = format!("[{tag}] rel diff:");
                for (label, values) in &outputs.0 {
                    let old = read_f64(&PathBuf::from(&dir).join(format!("{tag}_{label}.bin")));
                    let (relative, maximum) = relative_difference(values, &old);
                    line.push_str(&format!(" {label}={relative:.1e}/{maximum:.1e}"));
                    differences.insert(label.to_string(), json!([relative, maximum]));
                }
                println!("{line}");
            }

            if seconds <= 0.0 {
                continue;
            }
            let push_ms = time_ms(seconds, || map.push_into(&real, rhs, &mut signed).unwrap());
            pencil.reset_stage_profile();
            let apply_ms = time_ms(seconds, || pencil.apply_into(&signed, rhs, &mut e, &mut f).unwrap());
            let profile = pencil.stage_profile().clone();
            let shifted_ms = time_ms(seconds, || pencil.apply_shifted_into(&signed, rhs, shift, &mut m).unwrap());
            let p0_ms = time_ms(seconds, || p0.apply_into(&signed, rhs, &mut p).unwrap());
            let pull_ms = time_ms(seconds, || map.pull_complex_into(&e, rhs, &mut pulled_e).unwrap());
            let proposal_ms =
                time_ms(seconds, || drop(map.pull_real_proposal_into(&p, rhs, &mut proposal).unwrap()));
            let calls = profile.applications.max(1) as f64;
            let stages: Vec<f64> = profile.nanos.iter().map(|&ns| ns as f64 / calls / 1e6).collect();
            println!(
                "[{tag}] ms (median/min): push {:.3}/{:.3} apply {:.3}/{:.3} shifted {:.3}/{:.3} P0 {:.3}/{:.3} pull {:.3}/{:.3} proposal {:.3}/{:.3}",
                push_ms.0, push_ms.1, apply_ms.0, apply_ms.1, shifted_ms.0, shifted_ms.1, p0_ms.0, p0_ms.1, pull_ms.0, pull_ms.1, proposal_ms.0, proposal_ms.1
            );
            let stage_text: Vec<String> =
                (0..STAGE_COUNT).map(|i| format!("{}={:.3}", STAGE_NAMES[i], stages[i])).collect();
            println!("[{tag}] stages ms: {}", stage_text.join(" | "));
            if let Ok(path) = std::env::var("ATOM_KERNEL_REPORT") {
                let record = json!({
                    "case": name, "order": order, "rhs": rhs, "modes": modes.len(),
                    "threads": std::env::var("ATOM_PENCIL_THREADS").unwrap_or_default(),
                    "pushMs": push_ms.0, "applyMs": apply_ms.0, "shiftedMs": shifted_ms.0, "p0Ms": p0_ms.0,
                    "pullMs": pull_ms.0, "proposalMs": proposal_ms.0,
                    "pushMin": push_ms.1, "applyMin": apply_ms.1, "shiftedMin": shifted_ms.1, "p0Min": p0_ms.1,
                    "pullMin": pull_ms.1, "proposalMin": proposal_ms.1,
                    "stageNames": STAGE_NAMES, "stageMs": stages, "differences": differences,
                    "pencilSetupMs": pencil_ms, "p0SetupMs": p0_setup_ms,
                });
                let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                writeln!(file, "{record}").unwrap();
            }
        }
    }
}

#[test]
#[ignore = "benchmark harness; needs ATOM_KERNEL_INPUTS and an OpenBLAS DLL"]
fn kernel_bench() {
    let inputs = PathBuf::from(std::env::var("ATOM_KERNEL_INPUTS").expect("set ATOM_KERNEL_INPUTS"));
    let cases = env_list("ATOM_KERNEL_CASES", "threeNearStrong,fourNear,fiveNearPrepared");
    let orders: Vec<usize> = env_list("ATOM_KERNEL_ORDERS", "1,2").iter().map(|s| s.parse().unwrap()).collect();
    let rhs: Vec<usize> = env_list("ATOM_KERNEL_RHS", "1,4").iter().map(|s| s.parse().unwrap()).collect();
    let seconds: f64 = std::env::var("ATOM_KERNEL_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.25);
    for case in &cases {
        run_case(case, &inputs, &orders, &rhs, seconds);
    }
}
