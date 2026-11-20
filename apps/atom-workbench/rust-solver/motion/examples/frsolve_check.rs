//! FRSOLVE check: `krylov::solve` on the systems the Python prototype exported (`frsolve\exp.py`), against Python's
//! `fr.solve` (its ρ, readouts and route), with the clock and the process's peak memory.
//!
//! frsolve_check <data dir> [filter] [--repeat N] [--level route|exact|direct] [--select]
//! - route (default): `krylov::solve` (ρ), the clock the least of N runs;
//! - exact / direct: ρ on that route alone (the split, the exact S or P = A, GMRES), forced -- the fallbacks' cost;
//! - --select: `select::decide` per scene (the filter names the scenes) on the full references and the fold2 models
//!   of every P on the scene's path (`<scene>_manifest.json`); F⃗ from the rows 'Fx', 'Fy', 'Fz' ('F' = z alone).
//! Readouts (the exported rows, as mdlat.readouts): F = Re(f·ρ), R = Re(r·ρ), populations Re(p_g·ρ) over their sum.
//! Measures as td_s.cmp: F by max(|F|, kR), R relative, populations absolute; ρ by max |Δ| / max |Python's|.

use atom_motion::floquet::{Elements, Readouts, Sparse, System};
use atom_motion::krylov::{gmres, solve, split, Rows, Schur};
use atom_motion::select::decide;
use atom_motion::linalg::C64;
use std::io::Read;
use std::time::Instant;

fn bytes(path: &str) -> Vec<u8> {
    let mut b = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    b
}
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
fn csr(b: &[u8], p: &mut usize) -> Sparse {
    let n = u64_at(b, p) as usize;
    let nnz = u64_at(b, p) as usize;
    let indptr = u32s(b, p, n + 1);
    let indices = u32s(b, p, nnz);
    let values = c64s(b, p, nnz);
    Sparse { rows: n, cols: n, indptr, indices, values }
}

/// A readout row: Re(row · ρ).
struct Row(Vec<usize>, Vec<C64>);
impl Row {
    fn read(&self, x: &[C64]) -> f64 {
        self.0.iter().zip(&self.1).map(|(&i, &v)| v * x[i]).sum::<C64>().re
    }
}

/// The system and its readout rows (F⃗ -- 'F' is z alone --, R, the populations by group).
fn load(dir: &str, tag: &str) -> (System, [Row; 3], Row, Vec<(String, Row)>) {
    let b = bytes(&format!("{dir}/{tag}.sys"));
    assert_eq!(&b[..8], b"FRSYS001");
    let mut p = 8;
    let matrix = csr(&b, &mut p);
    let n = matrix.rows;
    let _rhs = c64s(&b, &mut p, n);
    let harmonic = u32s(&b, &mut p, n);
    let rows = u64_at(&b, &mut p) as usize;
    let empty_row = || Row(vec![], vec![]);
    let (mut force, mut rate, mut pops) = ([empty_row(), empty_row(), empty_row()], empty_row(), Vec::new());
    for _ in 0..rows {
        let len = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let name = String::from_utf8(b[p..p + len].to_vec()).unwrap();
        p += len;
        let count = u64_at(&b, &mut p) as usize;
        let row = Row(u32s(&b, &mut p, count), c64s(&b, &mut p, count));
        match name.as_str() {
            "Fx" => force[0] = row,
            "Fy" => force[1] = row,
            "F" | "Fz" => force[2] = row,
            "R" => rate = row,
            _ => pops.push((name[4..].to_string(), row)),
        }
    }
    let empty = || Sparse { rows: n - 1, cols: n - 1, indptr: vec![0; n], indices: vec![], values: vec![] };
    (System { elements: Elements::default(), matrix, k1: [empty(), empty(), empty()], trace: vec![], harmonic }, force, rate, pops)
}

fn vector(path: &str) -> Vec<C64> {
    let b = bytes(path);
    let mut p = 0;
    c64s(&b, &mut p, b.len() / 16)
}

/// max |a − b| / max |b|.
fn relative(a: &[C64], b: &[C64]) -> f64 {
    let scale = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let d = a.iter().zip(b).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        d
    } else {
        d / scale
    }
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

/// The readouts of ρ from the exported rows (levels over their sum).
fn readouts(rho: &[C64], force: &[Row; 3], rate: &Row, pops: &[(String, Row)]) -> Readouts {
    let raw: Vec<f64> = pops.iter().map(|(_, row)| row.read(rho)).collect();
    let total: f64 = raw.iter().sum();
    Readouts {
        levels: raw.iter().map(|v| v / total).collect(),
        rate: rate.read(rho),
        force: [force[0].read(rho), force[1].read(rho), force[2].read(rho)],
        ..Default::default()
    }
}

/// `select::decide` on a scene's exported systems; prints the chosen P and every candidate's worst error per point.
fn select_scene(dir: &str, scene: &str) {
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{scene}_manifest.json")).unwrap()).unwrap();
    let k = man["kph"].as_f64().unwrap();
    let points = ["centre0", "edge"];
    let models = |p: &str| man["systems"][p]["models"].as_array().unwrap().clone();
    let sizes: Vec<usize> = models("centre0").iter().map(|m| m["P"].as_u64().unwrap() as usize).collect();
    let systems: Vec<Vec<_>> = points
        .iter()
        .map(|p| {
            let mut tags = vec![man["systems"][*p]["full"].as_str().unwrap().to_string()];
            tags.extend(models(p).iter().map(|m| m["fold2"].as_str().unwrap().to_string()));
            tags.iter().map(|t| load(dir, t)).collect()
        })
        .collect();
    let candidates: Vec<Vec<usize>> = sizes.iter().map(|&n| (0..n).collect()).collect();
    let out = decide(&candidates, k, |p, c| {
        let (system, force, rate, pops) = &systems[p][c.map_or(0, |c| c + 1)];
        let solved = solve(system).map_err(|e| format!("{e:?}"))?;
        Ok((readouts(&solved.rho, force, rate, pops), solved.route))
    })
    .unwrap();
    let errors: Vec<String> = out
        .candidates
        .iter()
        .map(|c| {
            let e = |p: &atom_motion::select::PointError| format!("{:.16e},{:.16e},{:.16e}", p.force, p.rate, p.populations);
            format!("P{} {:.2e}/{:.2e} [c {} e {}]", c.keep.len(), c.centre.worst(), c.edge.worst(), e(&c.centre), e(&c.edge))
        })
        .collect();
    println!("{scene:<6} P {} | {:.3} s | {}", out.keep.len(), out.seconds, errors.join(" | "));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let mut filter = String::new();
    let (mut repeat, mut level, mut select) = (1usize, "route".to_string(), false);
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
            "--select" => select = true,
            f => filter = f.to_string(),
        }
        k += 1;
    }
    if select {
        println!("threads {}", rayon::current_num_threads());
        for scene in filter.split(',') {
            select_scene(dir, scene);
        }
        return;
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
        let (system, force, rate, pops) = load(dir, tag);
        let py: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{tag}.json")).unwrap()).unwrap();
        let rho_py = vector(&format!("{dir}/{tag}.rho"));
        let (mem0, _, _) = memory();
        let a = &system.matrix;
        let mut best = f64::INFINITY;
        let mut rho = Vec::new();
        let mut report = String::new();
        for _ in 0..repeat {
            let clock = Instant::now();
            if level == "route" {
                let out = solve(&system).unwrap();
                let t = clock.elapsed().as_secs_f64();
                if t < best {
                    best = t;
                    report = out
                        .attempts
                        .iter()
                        .map(|at| match &at.run {
                            Some(r) => format!(
                                "{:?} slow {} blocks {} it {} best {} cert {:?} zeta {:.1e} {:?} [setup {:.3} iter {:.3}]",
                                at.route, at.slow, at.blocks, r.iterations, r.best, r.first_certified, r.zeta, r.stop,
                                at.seconds_setup, at.seconds_iterate
                            ),
                            None => format!("{:?} unfactorised: {}", at.route, at.unfactorised.as_deref().unwrap_or("")),
                        })
                        .collect::<Vec<_>>()
                        .join(" -> ")
                        + &format!(
                            " | split {:.3} slow {} tests {} | route {:?} certified {}",
                            out.seconds_split, out.slow, out.split_tests, out.route, out.certified
                        );
                    rho = out.rho;
                }
            } else {
                let rows = Rows::new(a);
                let (slow, _) = split(a);
                let n = a.rows;
                let p = match if level == "direct" { Schur::exact(a, &slow) } else { Schur::new(a, &slow, &vec![0usize; n]) } {
                    Ok(p) => p,
                    Err(u) => panic!("unfactorised after {:.1} s: {}", clock.elapsed().as_secs_f64(), u.reason),
                };
                let t_setup = clock.elapsed().as_secs_f64();
                let mut b = vec![C64::ZERO; n];
                b[n - 1] = C64::ONE;
                let r = gmres(a, &b, &p, &rows);
                let t = clock.elapsed().as_secs_f64();
                if t < best {
                    best = t;
                    report = format!(
                        "{level} slow {} it {} best {} cert {:?} zeta {:.1e} {:?} [setup {:.3} iter {:.3}]",
                        p.shape().0, r.iterations, r.best, r.first_certified, r.zeta, r.stop, t_setup, t - t_setup
                    );
                    rho = r.x[..n - 1].to_vec();
                }
            }
        }
        let (_, peak, peak_ws) = memory();
        let kph = py["kph"].as_f64().unwrap();
        let r = &py["readouts"];
        let (f0, r0) = (r["F"].as_f64().unwrap(), r["R"].as_f64().unwrap());
        let raw: Vec<f64> = pops.iter().map(|(_, row)| row.read(&rho)).collect();
        let total: f64 = raw.iter().sum();
        let d_f = (force[2].read(&rho) - f0).abs() / f0.abs().max(kph * r0);
        let d_r = (rate.read(&rho) - r0).abs() / r0.abs();
        let d_p = pops
            .iter()
            .zip(&raw)
            .map(|((g, _), v)| (v / total - r["pops"][g].as_f64().unwrap()).abs())
            .fold(0.0, f64::max);
        let d_x = relative(&rho, &rho_py);
        for (w, d) in worst.iter_mut().zip([d_f, d_r, d_p, d_x]) {
            *w = w.max(d);
        }
        let pr = &py["rho"];
        println!(
            "{tag:<24} n {:>6} nnz {:>8} | rust {best:.4} s | {report} | py it {} best {} cert {} zeta {:.1e} slow {} | dF {d_f:.1e} dR {d_r:.1e} dP {d_p:.1e} drho {d_x:.1e} | mem MB before {mem0:.0} peak {peak:.0} ws {peak_ws:.0}",
            a.rows,
            a.indices.len(),
            pr["iters"],
            pr["kbest"],
            pr["first"],
            pr["zeta"].as_f64().unwrap(),
            py["split"]["nslow"],
        );
    }
    println!("worst dF {:.1e} dR {:.1e} dP {:.1e} drho {:.1e}", worst[0], worst[1], worst[2], worst[3]);
}
