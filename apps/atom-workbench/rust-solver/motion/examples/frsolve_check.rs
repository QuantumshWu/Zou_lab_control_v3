//! FRSOLVE check: the krylov solver on the systems the Python prototype exported (`frsolve\exp.py`), against
//! Python's `fr.solve` readouts and ρ, with the clock and the process's peak memory.
//!
//! frsolve_check <data dir> [filter] [--repeat N] [--level route|exact|direct] [--zero]
//! - route (default): `krylov::solve`, the clock the least of N runs;
//! - exact / direct: that level alone (the split, the exact S or P = A, GMRES), forced -- the fallbacks' cost;
//! - --zero: the same systems with b = 0 (x = 0 certified, no iteration).
//! Measures as td_s.cmp: F by max(|F|, kR), R relative, populations absolute; ρ by max |Δρ| / max |ρ|.

use atom_motion::krylov::{gmres, solve, split, Csr, Readouts, Row, Rows, Schur, System};
use atom_motion::linalg::C64;
use std::io::Read;
use std::time::Instant;

fn u64_at(b: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}
fn u32s(b: &[u8], p: &mut usize, n: usize) -> Vec<usize> {
    let v = (0..n).map(|k| u32::from_le_bytes(b[*p + 4 * k..*p + 4 * k + 4].try_into().unwrap()) as usize).collect();
    *p += 4 * n;
    v
}
fn c64s(b: &[u8], p: &mut usize, n: usize) -> Vec<C64> {
    let f = |k: usize| f64::from_le_bytes(b[*p + 8 * k..*p + 8 * k + 8].try_into().unwrap());
    let v = (0..n).map(|k| C64::new(f(2 * k), f(2 * k + 1))).collect();
    *p += 16 * n;
    v
}

fn load(path: &str) -> System {
    let mut b = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    assert_eq!(&b[..8], b"FRSYS001");
    let mut p = 8;
    let n = u64_at(&b, &mut p) as usize;
    let nnz = u64_at(&b, &mut p) as usize;
    let start = u32s(&b, &mut p, n + 1);
    let column = u32s(&b, &mut p, nnz);
    let value = c64s(&b, &mut p, nnz);
    let rhs = c64s(&b, &mut p, n);
    let harmonic = u32s(&b, &mut p, n);
    let rows = u64_at(&b, &mut p) as usize;
    let mut readouts = Readouts::default();
    for _ in 0..rows {
        let len = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let name = String::from_utf8(b[p..p + len].to_vec()).unwrap();
        p += len;
        let count = u64_at(&b, &mut p) as usize;
        let row = Row { index: u32s(&b, &mut p, count), value: c64s(&b, &mut p, count) };
        match name.as_str() {
            "F" => readouts.force = row,
            "R" => readouts.rate = row,
            _ => readouts.populations.push((name[4..].to_string(), row)),
        }
    }
    System { matrix: Csr { start, column, value }, rhs, harmonic, readouts }
}

#[repr(C)]
#[derive(Default)]
struct Counters {
    cb: u32,
    page_faults: u32,
    peak_working_set: usize,
    working_set: usize,
    quota_peak_paged: usize,
    quota_paged: usize,
    quota_peak_nonpaged: usize,
    quota_nonpaged: usize,
    pagefile: usize,
    peak_pagefile: usize,
}
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
}
/// (current private commit, peak private commit, peak working set) in MB.
fn memory() -> (f64, f64, f64) {
    let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    let mb = |v: usize| v as f64 / 1048576.0;
    (mb(c.pagefile), mb(c.peak_pagefile), mb(c.peak_working_set))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let mut filter = String::new();
    let (mut repeat, mut level, mut zero) = (1usize, "route".to_string(), false);
    let mut k = 2;
    while k < args.len() {
        match args[k].as_str() {
            "--repeat" => {
                repeat = args[k + 1].parse().unwrap();
                k += 1;
            }
            "--level" => {
                level = args[k + 1].clone();
                k += 1;
            }
            "--zero" => zero = true,
            f => filter = f.to_string(),
        }
        k += 1;
    }
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|f| f.ends_with(".sys") && f.contains(&filter))
        .map(|f| f.trim_end_matches(".sys").to_string())
        .collect();
    names.sort();
    println!("threads {}", rayon::current_num_threads());
    let mut worst = [0.0f64; 4];
    for tag in &names {
        let mut system = load(&format!("{dir}/{tag}.sys"));
        let py: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{tag}.json")).unwrap()).unwrap();
        let rho: Vec<C64> = {
            let mut b = Vec::new();
            std::fs::File::open(format!("{dir}/{tag}.rho")).unwrap().read_to_end(&mut b).unwrap();
            let mut p = 0;
            c64s(&b, &mut p, b.len() / 16)
        };
        if zero {
            system.rhs.iter_mut().for_each(|v| *v = C64::ZERO);
        }
        let (mem0, _, _) = memory();
        let a = &system.matrix;
        let mut best = f64::INFINITY;
        let mut x = Vec::new();
        let mut report = String::new();
        for _ in 0..repeat {
            let clock = Instant::now();
            if level == "route" {
                let out = solve(&system).unwrap();
                let t = clock.elapsed().as_secs_f64();
                if t < best {
                    best = t;
                    report = out
                        .route
                        .iter()
                        .map(|at| match &at.run {
                            Some(r) => format!(
                                "{:?} slow {} blocks {} it {} best {} cert {:?} zeta {:.1e} {:?} [setup {:.3} iter {:.3}]",
                                at.level, at.slow, at.blocks, r.iterations, r.best, r.first_certified, r.zeta, r.stop,
                                at.seconds_setup, at.seconds_iterate
                            ),
                            None => format!("{:?} unfactorised", at.level),
                        })
                        .collect::<Vec<_>>()
                        .join(" -> ")
                        + &format!(" | split {:.3} slow {} tests {} | certified {}", out.seconds_split, out.slow, out.split_tests, out.certified);
                    x = out.x;
                }
            } else {
                let rows = Rows::new(a);
                let (slow, _) = split(a);
                let n = a.dim();
                let mask = if level == "direct" { vec![true; n] } else { slow };
                let p = Schur::new(a, &mask, &vec![0usize; n]).ok().expect("unfactorised");
                let t_setup = clock.elapsed().as_secs_f64();
                let r = gmres(a, &system.rhs, &p, &rows);
                let t = clock.elapsed().as_secs_f64();
                if t < best {
                    best = t;
                    report = format!(
                        "{level} slow {} it {} best {} cert {:?} zeta {:.1e} {:?} [setup {:.3} iter {:.3}]",
                        p.shape().0, r.iterations, r.best, r.first_certified, r.zeta, r.stop, t_setup, t - t_setup
                    );
                    x = r.x;
                }
            }
        }
        let (_, peak, peak_ws) = memory();
        let got = system.readouts.read(&x);
        let kph = py["kph"].as_f64().unwrap();
        let r = &py["readouts"];
        let (f0, r0) = (r["F"].as_f64().unwrap(), r["R"].as_f64().unwrap());
        let (d_f, d_r, d_p, d_x);
        if zero {
            d_f = got.force.abs();
            d_r = got.rate.abs();
            d_p = got.populations.iter().map(|(_, v)| v.abs()).fold(0.0, f64::max);
            d_x = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
        } else {
            d_f = (got.force - f0).abs() / f0.abs().max(kph * r0);
            d_r = (got.rate - r0).abs() / r0.abs();
            d_p = got
                .populations
                .iter()
                .map(|(g, v)| (v - r["pops"][g].as_f64().unwrap()).abs())
                .fold(0.0, f64::max);
            let scale = rho.iter().map(|v| v.abs()).fold(0.0, f64::max);
            d_x = rho.iter().zip(&x).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max) / scale;
        }
        for (w, d) in worst.iter_mut().zip([d_f, d_r, d_p, d_x]) {
            *w = w.max(d);
        }
        let pr = &py["rho"];
        println!(
            "{tag:<24} n {:>6} nnz {:>8} | rust {best:.4} s | {report} | py it {} best {} cert {} zeta {:.1e} slow {} | dF {d_f:.1e} dR {d_r:.1e} dP {d_p:.1e} drho {d_x:.1e} | mem MB before {mem0:.0} peak {peak:.0} ws {peak_ws:.0}",
            a.dim(),
            a.nnz(),
            pr["iters"],
            pr["kbest"],
            pr["first"],
            pr["zeta"].as_f64().unwrap(),
            py["split"]["nslow"],
        );
    }
    println!("worst dF {:.1e} dR {:.1e} dP {:.1e} drho {:.1e}", worst[0], worst[1], worst[2], worst[3]);
}
