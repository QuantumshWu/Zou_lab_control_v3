//! Finite-interval full-density Fourier normal-form propagation.
//!
//! P(t) and B satisfy x(t)=R P(t) exp(Bt) z0 up to the explicitly
//! evaluated full Fourier residual, including outside Fourier indices.
//! Every original state, coherent mode and input harmonic remains represented
//! either algebraically or by a finite-time numerical bound. This is not a
//! steady-state claim; atomic dispatch performs separate frame and density
//! refinement checks before accepting this finite-time result.
//! Harmonic-basis projection and static spectral propagation roundoff require
//! separate verification; the reported Fourier bound does not certify those.
use super::{
    cluster_action::{self, ModalBasis, Sparse},
    lapack::{Complex64, Lapack},
    modal_defect,
};
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;
type Z = Complex64;
type Index = Vec<i32>;
type Blocks = BTreeMap<(usize, usize), Vec<Z>>;
#[derive(Clone)]
pub struct Harmonic {
    pub index: Index,
    pub cos: Sparse,
    pub sin: Sparse,
}
pub struct Input {
    pub base: Sparse,
    pub harmonics: Vec<Harmonic>,
    pub frequencies: Vec<f64>,
    pub bounded_generator_tail: f64,
}
#[derive(Clone, Debug)]
pub struct Iteration {
    pub coefficients: usize,
    pub evaluated_residual: Option<f64>,
    pub product_bound: f64,
    pub rounding_bound: f64,
    pub generator_tail: f64,
    pub total_residual: f64,
    pub boundary_residual: f64,
    pub products: usize,
    pub elapsed_ms: f64,
    pub norm_ms: f64,
    pub product_ms: f64,
    pub algebraic_candidate: Option<f64>,
    pub used_algebraic_bound: bool,
}
#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub dimension: usize,
    pub component_dimensions: Vec<usize>,
    pub modal_condition: f64,
    pub modal_defect_norm: f64,
    pub iterations: Vec<Iteration>,
    pub constant_correction_norm: f64,
    pub modal_representation_error: f64,
    pub preparation_ms: f64,
    pub modal_preparation_ms: f64,
    pub harmonic_projection_ms: f64,
    pub normal_form_ms: f64,
    pub physical_transform_ms: f64,
    pub constant_preparation_ms: f64,
    pub physical_reference: bool,
}
pub struct Prepared {
    pub diagnostics: Diagnostics,
    n: usize,
    frequencies: Vec<f64>,
    maximum_time: f64,
    tolerance: f64,
    components: Vec<Component>,
    modal: ModalBasis,
    polynomial: BTreeMap<Index, Blocks>,
    constant: Vec<(Vec<usize>, cluster_action::Prepared)>,
    initial_lu: Vec<f64>,
    initial_pivots: Vec<i32>,
    initial_transform: Vec<f64>,
    constant_matrix: Sparse,
    residual: f64,
    growth_rate: f64,
    vector_norm: f64,
    transformation_norm_bound: f64,
    cptp_generator_tail: f64,
    physical_reference: bool,
}
#[derive(Debug)]
pub struct ActionResult {
    pub states: Vec<Vec<f64>>,
    pub fourier_error_bound: f64,
    pub initial_mismatch: f64,
    pub imaginary_residue: f64,
    pub elapsed_ms: f64,
    pub constant_initial_reconstruction: f64,
    pub cptp_generator_tail_bound: f64,
}
#[derive(Debug)]
pub struct InitialAudit {
    pub fourier_bound: f64,
    pub uniform_growth_bound: f64,
    pub initial_mismatch: f64,
    pub constant_initial_reconstruction: f64,
    pub integrated_modal_norm: f64,
    pub accepted: bool,
}
#[derive(Debug)]
pub struct ConstantAudit {
    pub maximum_coordinate_difference: f64,
    pub maximum_hilbert_schmidt_difference: f64,
    pub sampled_trace_norm_bound: f64,
    pub maximum_at_time: f64,
    pub alternate_initial_error: f64,
    pub alternative_partition: f64,
    pub alternative_maximum_real_eigenvalue: f64,
    pub samples: usize,
    pub elapsed_ms: f64,
}
struct Component {
    coordinates: Vec<usize>,
    offset: usize,
    vectors: Vec<Z>,
    inverse: Vec<Z>,
    identity: bool,
}
#[inline]
fn add(a: Z, b: Z) -> Z {
    Z {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}
#[inline]
fn sub(a: Z, b: Z) -> Z {
    Z {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}
#[inline]
fn mul(a: Z, b: Z) -> Z {
    Z {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}
#[inline]
fn scale(a: Z, v: f64) -> Z {
    Z {
        re: a.re * v,
        im: a.im * v,
    }
}
#[inline]
fn abs(a: Z) -> f64 {
    // The normal-form matrices are ordinarily far from overflow/underflow.
    // Hardware sqrt avoids an out-of-line hypot call in hundreds of millions
    // of norm entries. Extreme values retain hypot's scaled implementation;
    // a small outward allowance keeps every use here an upper magnitude bound.
    let largest = a.re.abs().max(a.im.abs());
    let value = if largest >= 1e-150 && largest <= 1e150 {
        (a.re * a.re + a.im * a.im).sqrt()
    } else {
        a.re.hypot(a.im)
    };
    value * (1. + 4. * f64::EPSILON)
}
fn divide(a: Z, b: Z) -> Z {
    let largest = b.re.abs().max(b.im.abs());
    let a = scale(a, 1. / largest);
    let b = scale(b, 1. / largest);
    let d = b.re * b.re + b.im * b.im;
    Z {
        re: (a.re * b.re + a.im * b.im) / d,
        im: (a.im * b.re - a.re * b.im) / d,
    }
}
fn exponential_integral(rate: f64, time: f64) -> f64 {
    if rate == 0. {
        time
    } else {
        (rate * time).exp_m1() / rate
    }
}
fn remainder_integral(delta: f64, time: f64) -> f64 {
    if delta == 0. || time == 0. {
        return 0.;
    }
    let x = delta * time;
    if x < 0.5 {
        let mut term = x * x * 0.5;
        let mut sum = term;
        for k in 3..80 {
            term *= x / k as f64;
            sum += term;
            if term <= sum * f64::EPSILON {
                break;
            }
        }
        sum / delta * (1. + 64. * f64::EPSILON)
    } else {
        (x.exp_m1() - x) / delta * (1. + 64. * f64::EPSILON)
    }
}
fn norm(a: &[Z], rows: usize) -> f64 {
    a.chunks(rows)
        .map(|c| c.iter().copied().map(abs).sum::<f64>())
        .fold(0., f64::max)
}
/// Verified rank-one decomposition bound, not density normalization. For any
/// computed U,lambda, ||U diag(lambda) U†||tr <= sum |lambda_j| ||u_j||².
/// The measured reconstruction defect plus a conservative multiplication /
/// summation allowance bounds the remaining Hermitian matrix in trace norm.
fn density_trace_upper(lap: &Lapack, x: &[f64]) -> f64 {
    let h = (x.len() as f64).sqrt() as usize;
    let n = h * h;
    let frobenius = x
        .iter()
        .enumerate()
        .map(|(i, &v)| if i < h { v * v } else { 2. * v * v })
        .sum::<f64>()
        .sqrt();
    let fallback = (h as f64).sqrt() * frobenius;
    let rho = super::expand_density(&super::coordinates(h), x, h);
    let matrix: Vec<_> = (0..n)
        .map(|k| {
            let z = rho[k % h * h + k / h];
            Z { re: z.re, im: z.im }
        })
        .collect();
    let Ok((lambda, u)) = lap.hermitian_eigen(&matrix, h) else {
        return fallback;
    };
    let mut rank_sum = 0.;
    let lambda_sum = lambda.iter().map(|v| v.abs()).sum::<f64>();
    let umax = u.iter().copied().map(abs).fold(0., f64::max);
    let mut squared = 0.;
    for j in 0..h {
        let norm = u[j * h..(j + 1) * h]
            .iter()
            .map(|z| z.re * z.re + z.im * z.im)
            .sum::<f64>();
        rank_sum += lambda[j].abs() * norm;
    }
    for col in 0..h {
        for row in 0..h {
            let mut value = Z::default();
            for k in 0..h {
                let a = u[row + k * h];
                let b = u[col + k * h];
                value = add(
                    value,
                    scale(
                        mul(
                            a,
                            Z {
                                re: b.re,
                                im: -b.im,
                            },
                        ),
                        lambda[k],
                    ),
                );
            }
            let delta = sub(matrix[row + col * h], value);
            squared += delta.re * delta.re + delta.im * delta.im;
        }
    }
    let eps = f64::EPSILON;
    let gamma = 16. * h as f64 * eps / (1. - 16. * h as f64 * eps);
    let scale = lambda_sum * umax * umax;
    let evaluation = h as f64 * (gamma * scale + 2. * eps * (frobenius + scale));
    let defect = (squared / (1. - 4. * n as f64 * eps)).sqrt() + evaluation;
    let bound = rank_sum / (1. - 8. * n as f64 * eps) + (h as f64).sqrt() * defect;
    if bound.is_finite() {
        bound.min(fallback * (1. + 16. * n as f64 * eps))
    } else {
        fallback
    }
}
fn finite(a: &[Z]) -> Result<(), String> {
    if a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        Err("Non-finite multi-frequency matrix".into())
    } else {
        Ok(())
    }
}
fn keyadd(a: &[i32], b: &[i32]) -> Result<Index, String> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            a.checked_add(b)
                .ok_or("Multi-frequency index overflow".into())
        })
        .collect()
}
fn canonical(q: &[i32]) -> bool {
    q.iter().find(|&&v| v != 0).map_or(true, |v| *v > 0)
}
fn negative(q: &[i32]) -> Index {
    q.iter().map(|&v| -v).collect()
}
fn conjugate_blocks(blocks: &Blocks, components: &[Component], pairs: &[usize]) -> Blocks {
    blocks
        .iter()
        .map(|(&(i, j), matrix)| {
            let row = &components[i];
            let col = &components[j];
            let rows = row.coordinates.len();
            let columns = col.coordinates.len();
            let mut value = Vec::with_capacity(rows * columns);
            for b in 0..columns {
                let source = (pairs[col.offset + b] - col.offset) * rows;
                for a in 0..rows {
                    let z = matrix[source + pairs[row.offset + a] - row.offset];
                    value.push(Z {
                        re: z.re,
                        im: -z.im,
                    });
                }
            }
            ((i, j), value)
        })
        .collect()
}
fn complete_pairs(
    polynomial: &mut BTreeMap<Index, Blocks>,
    components: &[Component],
    pairs: &[usize],
) {
    let keys: Vec<_> = polynomial
        .keys()
        .filter(|q| canonical(q) && q.iter().any(|&v| v != 0))
        .cloned()
        .collect();
    for q in keys {
        let value = conjugate_blocks(&polynomial[&q], components, pairs);
        polynomial.insert(negative(&q), value);
    }
}
fn frequency(q: &[i32], frequencies: &[f64]) -> f64 {
    q.iter()
        .zip(frequencies)
        .fold(0., |sum, (&q, &w)| (q as f64).mul_add(w, sum))
}
fn phase(q: &[i32], frequencies: &[f64], time: f64) -> Result<Z, String> {
    let mut out = Z { re: 1., im: 0. };
    for (&index, &omega) in q.iter().zip(frequencies) {
        let hi = omega * time;
        let lo = omega.mul_add(time, -hi);
        let integer = index as f64;
        let angle = hi * integer;
        let tail = hi.mul_add(integer, -angle) + lo * integer;
        if !angle.is_finite() || !tail.is_finite() {
            return Err("Multi-frequency phase overflow".into());
        }
        let (s, c) = angle.sin_cos();
        let (sl, cl) = tail.sin_cos();
        out = mul(
            out,
            Z {
                re: c * cl - s * sl,
                im: s * cl + c * sl,
            },
        );
    }
    Ok(out)
}
fn add_blocks(destination: &mut Blocks, source: &Blocks, factor: f64) {
    for (&key, values) in source {
        let target = destination
            .entry(key)
            .or_insert_with(|| vec![Z::default(); values.len()]);
        for (i, &v) in values.iter().enumerate() {
            target[i] = add(target[i], scale(v, factor));
        }
    }
}
fn update_blocks(destination: &mut Blocks, source: &Blocks) -> Blocks {
    let mut change = Blocks::new();
    for (&key, values) in source {
        let target = destination
            .entry(key)
            .or_insert_with(|| vec![Z::default(); values.len()]);
        let mut delta = vec![Z::default(); values.len()];
        for (i, &v) in values.iter().enumerate() {
            let old = target[i];
            target[i] = add(old, v);
            delta[i] = sub(target[i], old);
        }
        change.insert(key, delta);
    }
    change
}
fn block_norm(blocks: &Blocks, components: &[Component], n: usize) -> f64 {
    let mut sums = vec![0.; n];
    for (&(i, j), value) in blocks {
        let rows = components[i].coordinates.len();
        let col = &components[j];
        for k in 0..col.coordinates.len() {
            sums[col.offset + k] += value[k * rows..(k + 1) * rows]
                .iter()
                .copied()
                .map(abs)
                .sum::<f64>();
        }
    }
    sums.into_iter().fold(0., f64::max)
}
fn absolute_column_sums(matrix: &[Z], rows: usize) -> Vec<f64> {
    let gamma = 4. * (rows + 1) as f64 * f64::EPSILON;
    matrix
        .chunks(rows)
        .map(|col| col.iter().map(|z| z.re.abs() + z.im.abs()).sum::<f64>() / (1. - gamma))
        .collect()
}
fn product_magnitude_one(left_column_sums: &[f64], right: &[Z], inner: usize) -> f64 {
    let bound = right
        .chunks(inner)
        .map(|col| {
            left_column_sums
                .iter()
                .zip(col)
                .map(|(&a, b)| a * (b.re.abs() + b.im.abs()))
                .sum::<f64>()
        })
        .fold(0., f64::max);
    bound / (1. - 8. * (inner + 1) as f64 * f64::EPSILON)
}
fn block_product(
    lap: &Lapack,
    a: &Blocks,
    b: &Blocks,
    components: &[Component],
    matrix_norms: &BTreeMap<usize, f64>,
    column_sums: &BTreeMap<usize, Vec<f64>>,
) -> Result<(Blocks, f64), String> {
    let mut result = Blocks::new();
    let mut round = 0.;
    for (&(i, j), left) in a {
        for (&(k, l), right) in b {
            if j != k {
                continue;
            }
            let m = components[i].coordinates.len();
            let inner = components[j].coordinates.len();
            let n = components[l].coordinates.len();
            let value = lap.multiply_complex(left, right, m, n, inner)?;
            let x = 8. * inner as f64 * f64::EPSILON;
            let gamma = x / (1. - x);
            let coarse =
                matrix_norms[&(left.as_ptr() as usize)] * matrix_norms[&(right.as_ptr() as usize)];
            let sharp =
                product_magnitude_one(&column_sums[&(left.as_ptr() as usize)], right, inner);
            round += gamma * coarse.min(sharp);
            // Transfer the first GEMM result directly. Creating an equally
            // large zero matrix and copying it added gigabytes of traffic in
            // a full preparation; later products keep their original order.
            match result.entry((i, l)) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(value);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    for (a, b) in entry.get_mut().iter_mut().zip(value) {
                        *a = add(*a, b);
                    }
                }
            }
        }
    }
    Ok((result, round))
}
fn components(
    lap: &Lapack,
    base: &Sparse,
    physical_reference: bool,
) -> Result<(ModalBasis, Vec<Component>, Blocks, f64), String> {
    let n = base.len();
    let mut edges = vec![vec![]; n];
    for (col, entries) in base.iter().enumerate() {
        for &(row, value) in entries {
            if row >= n || !value.is_finite() {
                return Err("Invalid multi-frequency base generator".into());
            }
            if value != 0. && row != col {
                edges[col].push(row);
                edges[row].push(col);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut groups = vec![];
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut members = vec![];
        while let Some(i) = queue.pop_front() {
            members.push(i);
            for &j in &edges[i] {
                if !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        members.sort_unstable();
        groups.push(members);
    }
    let mut vectors = vec![Z::default(); n * n];
    let mut inverse = vec![Z::default(); n * n];
    let mut eigenvalues = vec![];
    let mut origin = vec![];
    let mut saved = vec![];
    let mut defect = Blocks::new();
    let mut offset = 0;
    let mut representation_error = 0_f64;
    for (ci, group) in groups.into_iter().enumerate() {
        let size = group.len();
        let mut map = vec![usize::MAX; n];
        for (i, &j) in group.iter().enumerate() {
            map[j] = i;
        }
        let local: Sparse = group
            .iter()
            .map(|&j| {
                base[j]
                    .iter()
                    .filter_map(|&(i, v)| (v != 0.).then_some((map[i], v)))
                    .collect()
            })
            .collect();
        let modal = if physical_reference {
            let mut vectors = vec![Z::default(); size * size];
            for i in 0..size {
                vectors[i + i * size].re = 1.;
            }
            ModalBasis {
                dimension: size,
                eigenvalues: vec![Z::default(); size],
                inverse: vectors.clone(),
                vectors,
                condition: 1.,
                origin: vec![cluster_action::ModeOrigin::PhysicalCoordinate; size],
            }
        } else {
            cluster_action::prepare(lap, &local, 1e5)?.modal_basis(lap)?
        };
        if physical_reference {
            let mut full = vec![Z::default(); size * size];
            for (col, entries) in local.iter().enumerate() {
                for &(row, value) in entries {
                    full[row + col * size].re += value;
                }
            }
            defect.insert((ci, ci), full);
        } else {
            let delta = modal_defect::compute(
                lap,
                &local,
                &modal.vectors,
                &modal.inverse,
                &modal.eigenvalues,
            )?;
            let inverse_error = delta.diagnostics.inverse_residual_one_bound;
            // W R=I+E implies R^-1 F=(I+E)^-1 W F. This accounts for
            // the supplied numerical inverse without multiplying by ||A||.
            representation_error = representation_error.max(
                (delta.diagnostics.modal_roundoff_one
                    + inverse_error * delta.diagnostics.modal_defect_one)
                    / (1. - inverse_error),
            );
            defect.insert((ci, ci), delta.matrix);
        }
        for j in 0..size {
            for i in 0..size {
                vectors[group[i] + (offset + j) * n] = modal.vectors[i + j * size];
                inverse[offset + i + group[j] * n] = modal.inverse[i + j * size];
            }
        }
        eigenvalues.extend(modal.eigenvalues);
        origin.extend(modal.origin);
        saved.push(Component {
            coordinates: group,
            offset,
            vectors: modal.vectors,
            inverse: modal.inverse,
            identity: physical_reference,
        });
        offset += size;
    }
    let condition = norm(&vectors, n) * norm(&inverse, n);
    if !condition.is_finite() || condition > 1e9 {
        return Err("Multi-frequency modal basis is too ill-conditioned".into());
    }
    Ok((
        ModalBasis {
            dimension: n,
            eigenvalues,
            vectors,
            inverse,
            condition,
            origin,
        },
        saved,
        defect,
        representation_error,
    ))
}
fn modal_operator(
    lap: &Lapack,
    cos: &Sparse,
    sin: &Sparse,
    components: &[Component],
    n: usize,
) -> Result<Blocks, String> {
    if cos.len() != n || sin.len() != n {
        return Err("Invalid harmonic dimension".into());
    }
    let mut map = vec![(0, 0); n];
    for (ci, c) in components.iter().enumerate() {
        for (i, &row) in c.coordinates.iter().enumerate() {
            map[row] = (ci, i);
        }
    }
    let mut physical = Blocks::new();
    for col in 0..n {
        let (cj, j) = map[col];
        for (part, entries) in [(false, &cos[col]), (true, &sin[col])] {
            for &(row, value) in entries {
                if row >= n || !value.is_finite() {
                    return Err("Invalid harmonic coordinate".into());
                }
                if value == 0. {
                    continue;
                }
                let (ci, i) = map[row];
                let rows = components[ci].coordinates.len();
                let cols = components[cj].coordinates.len();
                let entry = physical
                    .entry((ci, cj))
                    .or_insert_with(|| vec![Z::default(); rows * cols]);
                if part {
                    entry[i + j * rows].im -= value * 0.5;
                } else {
                    entry[i + j * rows].re += value * 0.5;
                }
            }
        }
    }
    let mut out = Blocks::new();
    for ((i, j), a) in physical {
        if components[i].identity && components[j].identity {
            out.insert((i, j), a);
            continue;
        }
        let rows = components[i].coordinates.len();
        let cols = components[j].coordinates.len();
        let rhs = lap.multiply_complex(&a, &components[j].vectors, rows, cols, cols)?;
        let value = lap.multiply_complex(&components[i].inverse, &rhs, rows, cols, rows)?;
        out.insert((i, j), value);
    }
    Ok(out)
}
struct Residual {
    coefficients: BTreeMap<Index, Blocks>,
    skipped: f64,
    rounding: f64,
    products: usize,
    norm_ms: f64,
    product_ms: f64,
}
fn residual(
    lap: &Lapack,
    polynomial: &BTreeMap<Index, Blocks>,
    drives: &BTreeMap<Index, Blocks>,
    defect: &Blocks,
    constant: &Blocks,
    modal: &ModalBasis,
    components: &[Component],
    pairs: &[usize],
    frequencies: &[f64],
    budget: f64,
) -> Result<Residual, String> {
    let norm_start = Instant::now();
    let zero = vec![0; frequencies.len()];
    let n = modal.dimension;
    let physical_reference = components.iter().all(|c| c.identity);
    let mut output = BTreeMap::<Index, Blocks>::new();
    let mut products = vec![];
    let dn = block_norm(defect, components, n);
    let cn = block_norm(constant, components, n);
    let mut arithmetic = 0.;
    // Every source matrix is immutable for this residual evaluation. Reusing
    // its norm avoids O(number_of_pairs * n^2) scalar hypot scans; the bound
    // and the products are exactly the same as recomputing each norm.
    let polynomial_norms: BTreeMap<_, _> = polynomial
        .iter()
        .map(|(q, p)| (q.clone(), block_norm(p, components, n)))
        .collect();
    let mut matrix_norms = BTreeMap::<usize, f64>::new();
    let mut column_sums = BTreeMap::<usize, Vec<f64>>::new();
    for blocks in polynomial
        .values()
        .chain(drives.values())
        .chain(std::iter::once(defect))
        .chain(std::iter::once(constant))
    {
        for (&(i, _), matrix) in blocks {
            matrix_norms
                .entry(matrix.as_ptr() as usize)
                .or_insert_with(|| norm(matrix, components[i].coordinates.len()));
            column_sums
                .entry(matrix.as_ptr() as usize)
                .or_insert_with(|| absolute_column_sums(matrix, components[i].coordinates.len()));
        }
    }
    let norm_ms = norm_start.elapsed().as_secs_f64() * 1000.;
    for (q, blocks) in polynomial {
        if !canonical(q) {
            continue;
        }
        let pn = polynomial_norms[q];
        let entry = output.entry(q.clone()).or_default();
        if q != &zero {
            let freq = frequency(q, frequencies);
            for (&(i, j), value) in blocks {
                let row = &components[i];
                let col = &components[j];
                let rows = row.coordinates.len();
                let mut matrix = vec![Z::default(); value.len()];
                let driving = if physical_reference {
                    drives.get(q).and_then(|blocks| blocks.get(&(i, j)))
                } else {
                    None
                };
                let frequency_low = if physical_reference {
                    (q[0] as f64).mul_add(frequencies[0], -freq)
                } else {
                    0.
                };
                for b in 0..col.coordinates.len() {
                    for a in 0..rows {
                        if physical_reference {
                            let k = a + b * rows;
                            let p = value[k];
                            let e = driving.map_or(Z::default(), |v| v[k]);
                            // E_q-i q.Omega P_q is a large cancellation in a
                            // converged normal form. FMA keeps its residual,
                            // instead of estimating two separate large terms.
                            matrix[k] = Z {
                                re: freq.mul_add(p.im, e.re) + frequency_low * p.im,
                                im: (-freq).mul_add(p.re, e.im) - frequency_low * p.re,
                            };
                            continue;
                        }
                        let mut d = sub(
                            modal.eigenvalues[row.offset + a],
                            modal.eigenvalues[col.offset + b],
                        );
                        d.im -= freq;
                        matrix[a + b * rows] = mul(d, value[a + b * rows]);
                    }
                }
                arithmetic += if physical_reference {
                    8. * f64::EPSILON
                        * (norm(&matrix, rows) + frequency_low.abs() * norm(value, rows))
                        + 32. * f64::EPSILON * f64::EPSILON * freq.abs() * norm(value, rows)
                } else {
                    32. * f64::EPSILON * norm(&matrix, rows)
                };
                entry.insert((i, j), matrix);
            }
            products.push((2. * dn * pn, q.clone(), defect, blocks, 1.));
            products.push((2. * dn * pn, q.clone(), blocks, defect, -1.));
            if !constant.is_empty() {
                products.push((2. * cn * pn, q.clone(), blocks, constant, -1.));
            }
        } else {
            add_blocks(entry, constant, -1.);
        }
    }
    for (q, drive) in drives {
        let en = block_norm(drive, components, n);
        if canonical(q) {
            if physical_reference {
                let target = output.entry(q.clone()).or_default();
                for (&key, values) in drive {
                    if !polynomial.get(q).is_some_and(|p| p.contains_key(&key)) {
                        target.insert(key, values.clone());
                    }
                }
            } else {
                add_blocks(output.entry(q.clone()).or_default(), drive, 1.);
                arithmetic += 8. * f64::EPSILON * en;
            }
        }
        for (k, p) in polynomial {
            if k != &zero {
                let target = keyadd(q, k)?;
                if canonical(&target) {
                    let multiplicity = if target == zero { 1. } else { 2. };
                    products.push((
                        multiplicity * en * polynomial_norms[k],
                        target,
                        drive,
                        p,
                        1.,
                    ));
                }
            }
        }
    }
    products.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut skipped = 0.;
    let mut computed = 0;
    let mut rounding = arithmetic;
    let product_start = Instant::now();
    for (bound, q, a, b, sign) in products {
        if skipped + bound <= budget {
            skipped += bound;
            continue;
        }
        let multiplicity = if q == zero { 1. } else { 2. };
        let (value, error) = block_product(lap, a, b, components, &matrix_norms, &column_sums)?;
        rounding += multiplicity * error + 4. * f64::EPSILON * bound;
        add_blocks(output.entry(q).or_default(), &value, sign);
        computed += 1;
    }
    let product_ms = product_start.elapsed().as_secs_f64() * 1000.;
    // q=0 still includes every signed product. Its exact J-conjugate symmetry
    // is restored only with the measured correction added to the error bound.
    if let Some(mean) = output.get_mut(&zero) {
        let conjugate = conjugate_blocks(mean, components, pairs);
        let mut difference = Blocks::new();
        for (&key, value) in mean.iter_mut() {
            let other = &conjugate[&key];
            let delta: Vec<_> = value
                .iter()
                .zip(other)
                .map(|(&a, &b)| scale(sub(a, b), 0.5))
                .collect();
            for (i, v) in value.iter_mut().enumerate() {
                *v = scale(add(*v, other[i]), 0.5);
            }
            difference.insert(key, delta);
        }
        rounding += block_norm(&difference, components, n);
    }
    complete_pairs(&mut output, components, pairs);
    Ok(Residual {
        coefficients: output,
        skipped,
        rounding,
        products: computed,
        norm_ms,
        product_ms,
    })
}
fn physical_blocks(
    lap: &Lapack,
    blocks: &Blocks,
    components: &[Component],
    n: usize,
) -> Result<Vec<Z>, String> {
    let mut result = vec![Z::default(); n * n];
    for (&(i, j), matrix) in blocks {
        let row = &components[i];
        let col = &components[j];
        let rows = row.coordinates.len();
        let columns = col.coordinates.len();
        let a = lap.multiply_complex(&row.vectors, matrix, rows, columns, rows)?;
        let value = lap.multiply_complex(&a, &col.inverse, rows, columns, columns)?;
        for b in 0..columns {
            for a in 0..rows {
                result[row.coordinates[a] + col.coordinates[b] * n] = value[a + b * rows];
            }
        }
    }
    finite(&result)?;
    Ok(result)
}
fn prepare_constant(
    lap: &Lapack,
    a: &Sparse,
) -> Result<Vec<(Vec<usize>, cluster_action::Prepared)>, String> {
    let n = a.len();
    let mut adjacent = vec![vec![]; n];
    for (col, entries) in a.iter().enumerate() {
        for &(row, value) in entries {
            if value != 0. && row != col {
                adjacent[col].push(row);
                adjacent[row].push(col);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut result = vec![];
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut members = vec![];
        while let Some(i) = queue.pop_front() {
            members.push(i);
            for &j in &adjacent[i] {
                if !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        members.sort_unstable();
        let mut map = vec![usize::MAX; n];
        for (i, &j) in members.iter().enumerate() {
            map[j] = i;
        }
        let local: Sparse = members
            .iter()
            .map(|&col| {
                a[col]
                    .iter()
                    .filter_map(|&(row, v)| (v != 0.).then_some((map[row], v)))
                    .collect()
            })
            .collect();
        let prepared = cluster_action::prepare(lap, &local, 1e5)?;
        result.push((members, prepared));
    }
    Ok(result)
}
pub fn prepare(
    lap: &Lapack,
    input: &Input,
    maximum_time: f64,
    tolerance: f64,
) -> Result<Prepared, String> {
    prepare_internal(lap, input, None, maximum_time, tolerance, false)
}
pub fn prepare_for_initial(
    lap: &Lapack,
    input: &Input,
    initial: &[f64],
    maximum_time: f64,
    tolerance: f64,
) -> Result<Prepared, String> {
    prepare_internal(lap, input, Some(initial), maximum_time, tolerance, false)
}
/// Only callers that constructed Input from a COMPLETE time-dependent
/// Lindblad generator may use this separate CPTP omitted-product bound.
pub fn prepare_lindblad_for_initial(
    lap: &Lapack,
    input: &Input,
    initial: &[f64],
    maximum_time: f64,
    tolerance: f64,
) -> Result<Prepared, String> {
    prepare_internal(lap, input, Some(initial), maximum_time, tolerance, true)
}
fn prepare_internal(
    lap: &Lapack,
    input: &Input,
    initial: Option<&[f64]>,
    maximum_time: f64,
    tolerance: f64,
    complete_lindblad: bool,
) -> Result<Prepared, String> {
    let started = Instant::now();
    let n = input.base.len();
    let axes = input.frequencies.len();
    let zero = vec![0; axes];
    let hilbert = (n as f64).sqrt() as usize;
    let cptp_generator_tail = if complete_lindblad {
        input.bounded_generator_tail
    } else {
        0.
    };
    let modal_generator_tail = if complete_lindblad {
        0.
    } else {
        input.bounded_generator_tail
    };
    if hilbert.checked_mul(hilbert) != Some(n) {
        return Err("Multi-frequency input must contain the complete square Hermitian density coordinate space".into());
    }
    if n == 0
        || axes == 0
        || axes > 3
        || input.frequencies.iter().any(|v| !v.is_finite() || *v <= 0.)
        || !maximum_time.is_finite()
        || maximum_time < 0.
        || !tolerance.is_finite()
        || tolerance <= 0.
        || !input.bounded_generator_tail.is_finite()
        || input.bounded_generator_tail < 0.
    {
        return Err("Invalid finite-interval multi-frequency input".into());
    }
    let base_norm = input
        .base
        .iter()
        .map(|c| c.iter().map(|(_, v)| v.abs()).sum::<f64>())
        .fold(0., f64::max);
    // This is a numerical reference basis, not spectral averaging: Lambda=0,
    // Delta0=L0 retains the ENTIRE base generator in every commutator and B.
    // The carrier separation makes the full homological iteration contractive.
    let physical_reference =
        complete_lindblad && axes == 1 && base_norm < 1e5 && input.frequencies[0] > 4. * base_norm;
    let stage = Instant::now();
    let (modal, components, mut defect, mut representation_error) =
        components(lap, &input.base, physical_reference)?;
    let modal_preparation_ms = stage.elapsed().as_secs_f64() * 1000.;
    let stage = Instant::now();
    let mut drives = BTreeMap::<Index, Blocks>::new();
    let initial_modal_norm = if let Some(initial) = initial {
        if initial.len() != n || initial.iter().any(|v| !v.is_finite()) {
            return Err(
                "Invalid initial coordinates for a multi-frequency preparation budget".into(),
            );
        }
        let z: Vec<_> = initial.iter().map(|&re| Z { re, im: 0. }).collect();
        Some(
            lap.multiply_complex(&modal.inverse, &z, n, 1, n)?
                .iter()
                .copied()
                .map(abs)
                .sum::<f64>(),
        )
    } else {
        None
    };
    let initial_trace_upper = if physical_reference {
        initial.map(|x| density_trace_upper(lap, x))
    } else {
        None
    };
    let mut pairs: Vec<_> = (0..n).collect();
    let mut i = 0;
    while i < n {
        if modal.eigenvalues[i].im > 0. {
            if i + 1 >= n
                || modal.eigenvalues[i + 1].im != -modal.eigenvalues[i].im
                || modal.eigenvalues[i + 1].re != modal.eigenvalues[i].re
            {
                return Err("Incomplete multi-frequency complex modal pair".into());
            }
            pairs[i] = i + 1;
            pairs[i + 1] = i;
            i += 2;
        } else {
            i += 1;
        }
    }
    let paired_defect = conjugate_blocks(&defect, &components, &pairs);
    let mut symmetry_change = Blocks::new();
    for (&key, value) in &mut defect {
        let other = &paired_defect[&key];
        let delta: Vec<_> = value
            .iter()
            .zip(other)
            .map(|(&a, &b)| scale(sub(a, b), 0.5))
            .collect();
        for (i, v) in value.iter_mut().enumerate() {
            *v = scale(add(*v, other[i]), 0.5);
        }
        symmetry_change.insert(key, delta);
    }
    representation_error += block_norm(&symmetry_change, &components, n);
    for harmonic in &input.harmonics {
        if harmonic.index.len() != axes
            || harmonic.index == zero
            || harmonic.index.iter().any(|&q| q == i32::MIN)
        {
            return Err("Invalid multi-frequency harmonic index".into());
        }
        let positive = modal_operator(lap, &harmonic.cos, &harmonic.sin, &components, n)?;
        add_blocks(
            drives.entry(harmonic.index.clone()).or_default(),
            &positive,
            1.,
        );
        let negative_operator = conjugate_blocks(&positive, &components, &pairs);
        add_blocks(
            drives.entry(negative(&harmonic.index)).or_default(),
            &negative_operator,
            1.,
        );
    }
    let harmonic_projection_ms = stage.elapsed().as_secs_f64() * 1000.;
    let stage = Instant::now();
    let mut identity = Blocks::new();
    for (i, c) in components.iter().enumerate() {
        let size = c.coordinates.len();
        let mut a = vec![Z::default(); size * size];
        for j in 0..size {
            a[j + j * size].re = 1.;
        }
        identity.insert((i, i), a);
    }
    let mut polynomial = BTreeMap::from([(zero.clone(), identity)]);
    let mut correction = Blocks::new();
    let mut iterations = vec![];
    let product_budget = (tolerance / (maximum_time.max(1.) * modal.condition * 64.)).min(1e-13);
    let mut r = residual(
        lap,
        &polynomial,
        &drives,
        &defect,
        &correction,
        &modal,
        &components,
        &pairs,
        &input.frequencies,
        product_budget,
    )?;
    for iteration in 0..if physical_reference { 8 } else { 4 } {
        let begin = Instant::now();
        let old_pnorm = polynomial
            .values()
            .map(|p| block_norm(p, &components, n))
            .sum::<f64>();
        let old_cnorm = block_norm(&correction, &components, n);
        let mut change = BTreeMap::<Index, Blocks>::new();
        let mut mean_change = Blocks::new();
        if let Some(mean) = r.coefficients.get(&zero) {
            if iteration >= 2 {
                mean_change = update_blocks(&mut correction, mean);
            } else {
                add_blocks(&mut correction, mean, 1.);
            }
        }
        for (q, blocks) in &r.coefficients {
            if q == &zero || !canonical(q) || q.iter().any(|q| q.unsigned_abs() > 4) {
                continue;
            }
            let freq = frequency(q, &input.frequencies);
            let mut delta = Blocks::new();
            for (&(i, j), value) in blocks {
                let row = &components[i];
                let col = &components[j];
                let rows = row.coordinates.len();
                let mut matrix = vec![Z::default(); value.len()];
                for b in 0..col.coordinates.len() {
                    for a in 0..rows {
                        let mut denominator = sub(
                            modal.eigenvalues[row.offset + a],
                            modal.eigenvalues[col.offset + b],
                        );
                        denominator.im -= freq;
                        if (denominator.re == 0. && denominator.im == 0.)
                            || !denominator.re.is_finite()
                            || !denominator.im.is_finite()
                        {
                            return Err("Multi-frequency Sylvester resonance; use the general full Fourier solver".into());
                        }
                        matrix[a + b * rows] = scale(divide(value[a + b * rows], denominator), -1.);
                    }
                }
                finite(&matrix)?;
                delta.insert((i, j), matrix);
            }
            if iteration >= 2 {
                let actual_change = update_blocks(polynomial.entry(q.clone()).or_default(), &delta);
                change.insert(q.clone(), actual_change);
            } else {
                add_blocks(polynomial.entry(q.clone()).or_default(), &delta, 1.);
            }
        }
        complete_pairs(&mut polynomial, &components, &pairs);
        complete_pairs(&mut change, &components, &pairs);
        if polynomial.len() > 343 {
            return Err(
                "Multi-frequency polynomial lattice exceeds this experimental solver budget".into(),
            );
        }
        let pnorm = polynomial
            .values()
            .map(|b| block_norm(b, &components, n))
            .sum::<f64>();
        if pnorm >= 2. {
            return Err(
                "The multi-frequency change of variables is not bounded near identity".into(),
            );
        }
        let generator_tail =
            (modal.condition * modal_generator_tail + representation_error) * pnorm;
        let mut candidate = None;
        if iteration >= 2 {
            let delta_norm = change
                .values()
                .map(|p| block_norm(p, &components, n))
                .sum::<f64>();
            let mean_norm = block_norm(&mean_change, &components, n);
            let mut cancellation = 0.;
            for (q, blocks) in &r.coefficients {
                let mut leftover = blocks.clone();
                if q == &zero {
                    add_blocks(&mut leftover, &mean_change, -1.);
                } else if let Some(delta) = change.get(q) {
                    let freq = frequency(q, &input.frequencies);
                    for (&(i, j), values) in delta {
                        let row = &components[i];
                        let col = &components[j];
                        let rows = row.coordinates.len();
                        let target = leftover
                            .entry((i, j))
                            .or_insert_with(|| vec![Z::default(); values.len()]);
                        for b in 0..col.coordinates.len() {
                            for a in 0..rows {
                                let la = modal.eigenvalues[row.offset + a];
                                let lb = modal.eigenvalues[col.offset + b];
                                let mut d = sub(la, lb);
                                d.im -= freq;
                                let k = a + b * rows;
                                target[k] = add(target[k], mul(d, values[k]));
                            }
                        }
                        // Denominator construction, quotient cancellation and the
                        // stored P increment's subtraction roundoff are included.
                        let scale_bound = modal
                            .eigenvalues
                            .iter()
                            .copied()
                            .map(abs)
                            .fold(0., f64::max)
                            * 2.
                            + freq.abs();
                        cancellation += 32. * f64::EPSILON * scale_bound * norm(values, rows);
                    }
                }
                cancellation += block_norm(&leftover, &components, n);
            }
            let dn = block_norm(&defect, &components, n);
            let en = drives
                .values()
                .map(|d| block_norm(d, &components, n))
                .sum::<f64>();
            let algebra = (2. * dn + en + old_cnorm) * delta_norm
                + (old_pnorm - 1. + delta_norm) * mean_norm
                + cancellation;
            let total = algebra + r.skipped + r.rounding + generator_tail;
            candidate = Some(total);
            if let Some(initial_norm) = initial_modal_norm {
                let vector_norm = norm(&modal.vectors, n);
                let rate = modal.eigenvalues.iter().map(|v| v.re).fold(0., f64::max)
                    + dn
                    + block_norm(&correction, &components, n)
                    + representation_error;
                let prospective = if physical_reference {
                    let h = (n as f64).sqrt();
                    let delta = h * (2. * h).sqrt() * block_norm(&correction, &components, n);
                    let trace_upper = initial_trace_upper.unwrap();
                    let inverse_gap = 1. - h * (2. * h).sqrt() * (pnorm - 1.);
                    if inverse_gap > 0. {
                        h * (2. * h).sqrt() * trace_upper / inverse_gap
                            * exponential_integral(delta, maximum_time)
                            * total
                    } else {
                        f64::INFINITY
                    }
                } else {
                    (2. * (n as f64).sqrt()).sqrt() * vector_norm * initial_norm / (2. - pnorm)
                        * maximum_time
                        * (rate * maximum_time).exp()
                        * total
                };
                if prospective.is_finite() && prospective <= 0.995 * tolerance {
                    let boundary = r
                        .coefficients
                        .iter()
                        .filter(|(q, _)| !polynomial.contains_key(*q))
                        .map(|(_, b)| block_norm(b, &components, n))
                        .sum::<f64>();
                    iterations.push(Iteration {
                        coefficients: polynomial.len(),
                        evaluated_residual: None,
                        product_bound: r.skipped,
                        rounding_bound: r.rounding,
                        generator_tail,
                        total_residual: total,
                        boundary_residual: boundary,
                        products: 0,
                        elapsed_ms: begin.elapsed().as_secs_f64() * 1000.,
                        norm_ms: 0.,
                        product_ms: 0.,
                        algebraic_candidate: candidate,
                        used_algebraic_bound: true,
                    });
                    break;
                }
            }
        }
        r = residual(
            lap,
            &polynomial,
            &drives,
            &defect,
            &correction,
            &modal,
            &components,
            &pairs,
            &input.frequencies,
            product_budget,
        )?;
        let evaluated = r
            .coefficients
            .values()
            .map(|b| block_norm(b, &components, n))
            .sum::<f64>();
        let total = evaluated + r.skipped + r.rounding + generator_tail;
        let boundary = r
            .coefficients
            .iter()
            .filter(|(q, _)| !polynomial.contains_key(*q))
            .map(|(_, b)| block_norm(b, &components, n))
            .sum::<f64>();
        iterations.push(Iteration {
            coefficients: polynomial.len(),
            evaluated_residual: Some(evaluated),
            product_bound: r.skipped,
            rounding_bound: r.rounding,
            generator_tail,
            total_residual: total,
            boundary_residual: boundary,
            products: r.products,
            elapsed_ms: begin.elapsed().as_secs_f64() * 1000.,
            norm_ms: r.norm_ms,
            product_ms: r.product_ms,
            algebraic_candidate: candidate,
            used_algebraic_bound: false,
        });
        if iteration >= 2 && !physical_reference {
            break;
        }
    }
    let residual_bound = iterations.last().unwrap().total_residual;
    if !residual_bound.is_finite() {
        return Err("Multi-frequency residual bound overflowed".into());
    }
    let normal_form_ms = stage.elapsed().as_secs_f64() * 1000.;
    let stage = Instant::now();
    let mut p0 = Blocks::new();
    for p in polynomial.values() {
        add_blocks(&mut p0, p, 1.);
    }
    let transform = physical_blocks(lap, &p0, &components, n)?;
    let t_imag = transform.iter().map(|z| z.im.abs()).fold(0., f64::max);
    if t_imag > tolerance {
        return Err(format!(
            "Initial multi-frequency transform is not real to tolerance: {t_imag:e}"
        ));
    }
    let initial_transform: Vec<_> = transform.iter().map(|z| z.re).collect();
    let (initial_lu, initial_pivots) = lap.factor(&initial_transform, n)?;
    let physical_correction = physical_blocks(lap, &correction, &components, n)?;
    let imaginary = physical_correction
        .iter()
        .map(|z| z.im.abs())
        .fold(0., f64::max);
    if imaginary > tolerance {
        return Err(format!(
            "Constant multi-frequency correction is not real: {imaginary:e}"
        ));
    }
    let mut constant_matrix: Sparse = (0..n)
        .map(|col| {
            (0..n)
                .filter_map(|row| {
                    let v = physical_correction[row + col * n].re;
                    (v != 0.).then_some((row, v))
                })
                .collect()
        })
        .collect();
    for col in 0..n {
        constant_matrix[col].extend_from_slice(&input.base[col]);
    }
    let physical_transform_ms = stage.elapsed().as_secs_f64() * 1000.;
    let stage = Instant::now();
    let constant = prepare_constant(lap, &constant_matrix)?;
    let constant_preparation_ms = stage.elapsed().as_secs_f64() * 1000.;
    let vector_norm = norm(&modal.vectors, n);
    let correction_norm = block_norm(&correction, &components, n);
    let defect_norm = block_norm(&defect, &components, n);
    let growth_rate = if physical_reference {
        hilbert as f64 * (2. * hilbert as f64).sqrt() * correction_norm
    } else {
        modal.eigenvalues.iter().map(|v| v.re).fold(0., f64::max)
            + defect_norm
            + correction_norm
            + representation_error
    };
    let diagnostics = Diagnostics {
        dimension: n,
        component_dimensions: components.iter().map(|c| c.coordinates.len()).collect(),
        modal_condition: modal.condition,
        modal_defect_norm: defect_norm,
        iterations,
        constant_correction_norm: correction_norm,
        modal_representation_error: representation_error,
        preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        modal_preparation_ms,
        harmonic_projection_ms,
        normal_form_ms,
        physical_transform_ms,
        constant_preparation_ms,
        physical_reference,
    };
    let transformation_norm_bound = modal.condition
        * polynomial
            .values()
            .map(|p| block_norm(p, &components, n))
            .sum::<f64>();
    Ok(Prepared {
        diagnostics,
        n,
        frequencies: input.frequencies.clone(),
        maximum_time,
        tolerance,
        components,
        modal,
        polynomial,
        constant,
        constant_matrix,
        initial_lu,
        initial_pivots,
        initial_transform,
        residual: residual_bound,
        growth_rate,
        vector_norm,
        transformation_norm_bound,
        cptp_generator_tail,
        physical_reference,
    })
}
impl Prepared {
    fn complete_generator_tail_bound(&self, initial: &[f64], time: f64) -> f64 {
        let hilbert = (self.n as f64).sqrt();
        let h = hilbert as usize;
        let frobenius = (initial
            .iter()
            .enumerate()
            .map(|(i, &v)| if i < h { v * v } else { 2. * v * v })
            .sum::<f64>())
        .sqrt();
        // Hermitian x: ||x||1<=N||rho||tr, ||rho||tr<=sqrt(2N)||x||1.
        // The FULL C(t) generator is CPTP; its omitted numerical product tail
        // need not be. Duhamel/Gronwall uses the true propagator's contraction.
        hilbert.sqrt()
            * frobenius
            * (hilbert * (2. * hilbert).sqrt() * self.cptp_generator_tail * time).exp_m1()
    }
    fn map_constant_states(
        &self,
        lap: &Lapack,
        physical: &[Z],
        times: &[f64],
    ) -> Result<Vec<Z>, String> {
        let n = self.n;
        let count = times.len();
        let modal = if count == 0 {
            vec![]
        } else {
            lap.multiply_complex(&self.modal.inverse, physical, n, count, n)?
        };
        let mut corrected = vec![Z::default(); n * count];
        for (q, blocks) in &self.polynomial {
            for (&(ci, cj), matrix) in blocks {
                let row = &self.components[ci];
                let col = &self.components[cj];
                let rows = row.coordinates.len();
                let cols = col.coordinates.len();
                let packed: Vec<_> = (0..count)
                    .flat_map(|t| {
                        modal[t * n + col.offset..t * n + col.offset + cols]
                            .iter()
                            .copied()
                    })
                    .collect();
                if count == 0 {
                    continue;
                }
                let value = lap.multiply_complex(matrix, &packed, rows, count, cols)?;
                for (ti, &time) in times.iter().enumerate() {
                    let ph = phase(q, &self.frequencies, time)?;
                    for i in 0..rows {
                        let k = row.offset + i + ti * n;
                        corrected[k] = add(corrected[k], mul(ph, value[i + ti * rows]));
                    }
                }
            }
        }
        let output = if count == 0 {
            vec![]
        } else {
            lap.multiply_complex(&self.modal.vectors, &corrected, n, count, n)?
        };
        finite(&output)?;
        Ok(output)
    }
    /// Independent numerical cross-check at the requested times, NOT a
    /// uniform-in-time theorem. The exact same final B is represented through
    /// a signed permutation and a different invariant-cluster partition.
    pub fn audit_constant(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
        primary_states: &[Vec<f64>],
    ) -> Result<ConstantAudit, String> {
        let start = Instant::now();
        let n = self.n;
        let count = times.len();
        let hilbert = (n as f64).sqrt() as usize;
        if initial.len() != n
            || initial.iter().any(|v| !v.is_finite())
            || times
                .iter()
                .any(|&t| !t.is_finite() || t < 0. || t > self.maximum_time)
            || primary_states.len() != count
            || primary_states
                .iter()
                .any(|v| v.len() != n || v.iter().any(|x| !x.is_finite()))
        {
            return Err("Invalid complete-density constant audit request".into());
        }
        let transformed = lap.solve_factored(&self.initial_lu, &self.initial_pivots, initial, n)?;
        let mut physical = vec![Z::default(); n * count];
        let mut alternative_initial_l1 = 0.;
        let mut maximum_real = f64::NEG_INFINITY;
        for (members, _) in &self.constant {
            let m = members.len();
            let permutation: Vec<_> = (0..m).rev().collect();
            let mut inverse_permutation = vec![0; m];
            for (i, &p) in permutation.iter().enumerate() {
                inverse_permutation[p] = i;
            }
            let signs: Vec<_> = (0..m)
                .map(|i| {
                    if (i * 1664525usize + 1013904223) & 4 == 0 {
                        1.
                    } else {
                        -1.
                    }
                })
                .collect();
            let mut global_to_local = vec![usize::MAX; n];
            for (i, &row) in members.iter().enumerate() {
                global_to_local[row] = i;
            }
            if members.iter().any(|&col| {
                self.constant_matrix[col]
                    .iter()
                    .any(|&(row, v)| v != 0. && global_to_local[row] == usize::MAX)
            }) {
                return Err(
                    "Exact constant-generator component has an unexpected external coupling".into(),
                );
            }
            let alternate: Sparse = (0..m)
                .map(|j| {
                    let original_col = members[permutation[j]];
                    self.constant_matrix[original_col]
                        .iter()
                        .filter_map(|&(row, value)| {
                            if value == 0. {
                                None
                            } else {
                                let local = global_to_local[row];
                                if local == usize::MAX {
                                    return None;
                                }
                                let i = inverse_permutation[local];
                                Some((i, signs[i] * signs[j] * value))
                            }
                        })
                        .collect()
                })
                .collect();
            let source: Vec<_> = (0..m)
                .map(|i| signs[i] * transformed[members[permutation[i]]])
                .collect();
            let solver = cluster_action::prepare(lap, &alternate, 100.)?;
            let modes = solver.modal_basis(lap)?;
            maximum_real = maximum_real.max(
                modes
                    .eigenvalues
                    .iter()
                    .map(|z| z.re)
                    .fold(f64::NEG_INFINITY, f64::max),
            );
            let result = solver.apply(lap, &source, times)?;
            alternative_initial_l1 += m as f64 * result.initial_error;
            for (t, state) in result.states.iter().enumerate() {
                for i in 0..m {
                    physical[members[permutation[i]] + t * n].re = signs[i] * state[i];
                }
            }
        }
        let alternate = self.map_constant_states(lap, &physical, times)?;
        let mut maximum = 0_f64;
        let mut frobenius = 0_f64;
        let mut at = 0.;
        for (t, &time) in times.iter().enumerate() {
            let mut squared = 0.;
            for i in 0..n {
                let delta = sub(
                    alternate[i + t * n],
                    Z {
                        re: primary_states[t][i],
                        im: 0.,
                    },
                );
                let difference = abs(delta);
                if difference > maximum {
                    maximum = difference;
                    at = time;
                }
                squared += if i < hilbert { 1. } else { 2. } * difference * difference;
            }
            frobenius = frobenius.max(squared.sqrt());
        }
        let trace_bound = (hilbert as f64).sqrt() * frobenius;
        let alternative_initial_error =
            (2. * hilbert as f64).sqrt() * self.transformation_norm_bound * alternative_initial_l1;
        if !trace_bound.is_finite()
            || trace_bound > self.tolerance
            || !alternative_initial_error.is_finite()
            || alternative_initial_error > self.tolerance
        {
            return Err(format!("Independent static-B full-density cross-check exceeds tolerance: sampled trace {trace_bound:e}, alternate initial {alternative_initial_error:e}; no uniform floating certificate claimed"));
        }
        Ok(ConstantAudit {
            maximum_coordinate_difference: maximum,
            maximum_hilbert_schmidt_difference: frobenius,
            sampled_trace_norm_bound: trace_bound,
            maximum_at_time: at,
            alternate_initial_error: alternative_initial_error,
            alternative_partition: 100.,
            alternative_maximum_real_eigenvalue: maximum_real,
            samples: count,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
        })
    }
    fn modal_integral(&self, z0: &[Z], time: f64) -> f64 {
        if self.physical_reference {
            let h = (self.n as f64).sqrt();
            // L0 is the complete zero-harmonic Lindblad generator. Treat only
            // the small full correction M as a trace-norm perturbation. The
            // Hamiltonian part of L0 contributes no fictitious exponential
            // growth. The reference R is only a coordinate permutation.
            return h
                * (2. * h).sqrt()
                * z0.iter().copied().map(abs).sum::<f64>()
                * exponential_integral(self.growth_rate, time)
                * (1. + 64. * self.n as f64 * f64::EPSILON);
        }
        let alpha = self
            .modal
            .eigenvalues
            .iter()
            .map(|v| v.re)
            .fold(0., f64::max);
        let delta = self.diagnostics.modal_defect_norm
            + self.diagnostics.constant_correction_norm
            + self.diagnostics.modal_representation_error;
        let diagonal = z0
            .iter()
            .zip(&self.modal.eigenvalues)
            .map(|(&z, &l)| abs(z) * exponential_integral(l.re, time))
            .sum::<f64>();
        let norm = z0.iter().copied().map(abs).sum::<f64>();
        (diagonal + norm * (alpha * time).exp() * remainder_integral(delta, time))
            * (1. + 64. * self.n as f64 * f64::EPSILON)
    }
    fn physical_integral(&self, lap: &Lapack, transformed: &[f64], time: f64) -> f64 {
        let h = (self.n as f64).sqrt();
        h * density_trace_upper(lap, transformed)
            * exponential_integral(self.growth_rate, time)
            * (1. + 64. * self.n as f64 * f64::EPSILON)
    }
    pub fn audit_initial(
        &self,
        lap: &Lapack,
        initial: &[f64],
        time: f64,
    ) -> Result<InitialAudit, String> {
        if initial.len() != self.n
            || initial.iter().any(|v| !v.is_finite())
            || !time.is_finite()
            || time < 0.
            || time > self.maximum_time
        {
            return Err("Invalid initial-state numerical audit request".into());
        }
        let n = self.n;
        let transformed = lap.solve_factored(&self.initial_lu, &self.initial_pivots, initial, n)?;
        let rebuilt = lap.multiply(&self.initial_transform, &transformed, n, 1, n);
        let mismatch = rebuilt
            .iter()
            .zip(initial)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        let z: Vec<_> = transformed.iter().map(|&re| Z { re, im: 0. }).collect();
        let z0 = lap.multiply_complex(&self.modal.inverse, &z, n, 1, n)?;
        let trace_factor = (2. * (n as f64).sqrt()).sqrt();
        let integral = if self.physical_reference {
            self.physical_integral(lap, &transformed, time)
        } else {
            self.modal_integral(&z0, time)
        };
        let fourier_bound = trace_factor * (mismatch + self.vector_norm * self.residual * integral)
            + self.complete_generator_tail_bound(initial, time);
        let uniform_growth_bound = trace_factor
            * (mismatch
                + self.vector_norm
                    * self.residual
                    * z0.iter().copied().map(abs).sum::<f64>()
                    * time
                    * (self.growth_rate * time).exp())
            + self.complete_generator_tail_bound(initial, time);
        let mut constant_initial_l1 = 0.;
        for (members, solver) in &self.constant {
            let local: Vec<_> = members.iter().map(|&i| transformed[i]).collect();
            let result = solver.apply(lap, &local, &[0.])?;
            constant_initial_l1 += members.len() as f64 * result.initial_error;
        }
        let constant_initial_reconstruction =
            trace_factor * self.transformation_norm_bound * constant_initial_l1;
        let total = fourier_bound + constant_initial_reconstruction;
        Ok(InitialAudit {
            fourier_bound,
            uniform_growth_bound,
            initial_mismatch: trace_factor * mismatch,
            constant_initial_reconstruction,
            integrated_modal_norm: integral,
            accepted: total.is_finite() && total <= self.tolerance,
        })
    }
    /// Static spectral propagation and harmonic projection roundoff need
    /// independent checks; the Fourier residual bound is not their substitute.
    pub fn apply(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
    ) -> Result<ActionResult, String> {
        self.apply_interval(lap, initial, times, false)
    }
    /// The stored Fourier polynomial and its uniform residual are time
    /// independent. A longer interval is accepted only after recomputing the
    /// COMPLETE residual integral, CPTP tail and initial-reconstruction gates
    /// at its actual endpoint; no old interval certificate is reused.
    pub fn apply_checked_interval(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
    ) -> Result<ActionResult, String> {
        self.apply_interval(lap, initial, times, true)
    }
    fn apply_interval(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
        allow_extension: bool,
    ) -> Result<ActionResult, String> {
        let started = Instant::now();
        let n = self.n;
        let count = times.len();
        let maximum = times.iter().copied().fold(0., f64::max);
        if initial.len() != n
            || initial.iter().any(|v| !v.is_finite())
            || times.iter().any(|&t| !t.is_finite() || t < 0.)
            || (!allow_extension && maximum > self.maximum_time)
        {
            return Err("Multi-frequency request exceeds its finite prepared interval".into());
        }
        let transformed = lap.solve_factored(&self.initial_lu, &self.initial_pivots, initial, n)?;
        let rebuilt = lap.multiply(&self.initial_transform, &transformed, n, 1, n);
        let mismatch = rebuilt
            .iter()
            .zip(initial)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        let source: Vec<_> = transformed.iter().map(|&re| Z { re, im: 0. }).collect();
        let z0 = lap.multiply_complex(&self.modal.inverse, &source, n, 1, n)?;
        let trace_factor = (2. * (n as f64).sqrt()).sqrt();
        let cptp_generator_tail_bound = self.complete_generator_tail_bound(initial, maximum);
        let integral = if self.physical_reference {
            self.physical_integral(lap, &transformed, maximum)
        } else {
            self.modal_integral(&z0, maximum)
        };
        let bound = trace_factor * (mismatch + self.vector_norm * self.residual * integral)
            + cptp_generator_tail_bound;
        if !bound.is_finite() || bound > self.tolerance {
            return Err(format!("Multi-frequency finite-time Fourier certificate {bound:e} exceeds {:e}; use the general full solver",self.tolerance));
        }
        let mut physical = vec![Z::default(); n * count];
        let mut constant_initial_l1 = 0.;
        for (members, solver) in &self.constant {
            let local: Vec<_> = members.iter().map(|&i| transformed[i]).collect();
            let base = solver.apply(lap, &local, times)?;
            constant_initial_l1 += members.len() as f64 * base.initial_error;
            for (t, state) in base.states.iter().enumerate() {
                for (i, &row) in members.iter().enumerate() {
                    physical[row + t * n].re = state[i];
                }
            }
        }
        // The constant solver reports its actual pre-override reconstruction.
        // Keep it visible and gate it after the full micromotion transform;
        // setting the displayed t=0 point to x0 must not hide a bad basis.
        let constant_initial_reconstruction =
            trace_factor * self.transformation_norm_bound * constant_initial_l1;
        if !constant_initial_reconstruction.is_finite()
            || bound + constant_initial_reconstruction > self.tolerance
        {
            return Err(format!("Static basis initial reconstruction {constant_initial_reconstruction:e} plus Fourier estimate {bound:e} exceeds {:e}",self.tolerance));
        }
        let output = self.map_constant_states(lap, &physical, times)?;
        let imaginary_residue = output.iter().map(|v| v.im.abs()).fold(0., f64::max);
        if imaginary_residue > self.tolerance {
            return Err("Multi-frequency full density lost Hermiticity".into());
        }
        let states = times
            .iter()
            .enumerate()
            .map(|(ti, &t)| {
                if t == 0. {
                    initial.to_vec()
                } else {
                    output[ti * n..(ti + 1) * n].iter().map(|z| z.re).collect()
                }
            })
            .collect();
        Ok(ActionResult {
            states,
            fourier_error_bound: bound,
            initial_mismatch: trace_factor * mismatch,
            constant_initial_reconstruction,
            cptp_generator_tail_bound,
            imaginary_residue,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fast_magnitude_is_an_outward_bound_including_extreme_scales() {
        for power in [-300, -160, -150, -50, 0, 50, 150, 160, 300] {
            for (re, im) in [(1., 0.), (0., 1.), (1., 1.), (-0.73, 0.29), (1., 1e-30)] {
                let scale = 10_f64.powi(power);
                let z = Z {
                    re: re * scale,
                    im: im * scale,
                };
                let exact = z.re.hypot(z.im);
                let bound = abs(z);
                assert!(bound >= exact, "{power}: {bound:e} < {exact:e}");
                assert!(bound <= exact * (1. + 8. * f64::EPSILON));
            }
        }
        assert_eq!(abs(Z::default()), 0.);
    }
    #[test]
    fn spectral_trace_bound_covers_pure_mixed_and_indefinite_hermitian_inputs() {
        let lap = Lapack::load().unwrap();
        for (x, exact) in [
            (vec![1., 0., 0., 0.], 1.),
            (vec![0.5, 0.5, 0., -0.5], 1.),
            (vec![0.4, 0.6, 0., 0.], 1.),
            (vec![2., -1., 0., 0.], 3.),
        ] {
            let bound = density_trace_upper(&lap, &x);
            assert!(bound >= exact, "{bound:e} < {exact}");
            assert!(
                bound < exact + 1e-10,
                "Unnecessarily loose trace bound {bound:e}"
            );
        }
        assert_eq!(density_trace_upper(&lap, &[0.; 4]), 0.);
    }
    #[test]
    fn physical_reference_keeps_full_time_dependent_collapse_and_checks_new_intervals() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/generator-mean-periodic-collapse.json"
        ))
        .unwrap();
        let matrix = |value: &serde_json::Value| -> Sparse {
            let x = value.as_array().unwrap();
            (0..4)
                .map(|j| {
                    (0..4)
                        .filter_map(|i| {
                            let value = x[i * 4 + j].as_f64().unwrap();
                            (value != 0.).then_some((i, value))
                        })
                        .collect()
                })
                .collect()
        };
        let omega = fixture["frequency"].as_f64().unwrap();
        let times: Vec<_> = fixture["oracle"]["phases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() / omega)
            .collect();
        let initial: Vec<_> = fixture["oracle"]["initial"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let input = Input {
            base: matrix(&fixture["base"]),
            harmonics: fixture["drives"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| Harmonic {
                    index: vec![h["index"][0].as_i64().unwrap() as i32],
                    cos: matrix(&h["cos"]),
                    sin: matrix(&h["sin"]),
                })
                .collect(),
            frequencies: vec![omega],
            bounded_generator_tail: 0.,
        };
        let lap = Lapack::load().unwrap();
        let prepared = prepare_lindblad_for_initial(
            &lap,
            &input,
            &initial,
            *times.last().unwrap() / 1.001,
            1e-9,
        )
        .unwrap();
        assert!(prepared.diagnostics.physical_reference);
        assert!(
            prepared.apply(&lap, &initial, &times).is_err(),
            "Original API preserves its explicit prepared interval"
        );
        let result = prepared
            .apply_checked_interval(&lap, &initial, &times)
            .unwrap();
        let mut difference = 0_f64;
        for (x, expected) in result
            .states
            .iter()
            .zip(fixture["oracle"]["densityCoordinates"].as_array().unwrap())
        {
            for i in 0..4 {
                difference = difference.max((x[i] - expected[i].as_f64().unwrap()).abs());
            }
        }
        assert!(
            difference < 1e-9,
            "Full complex-C(t) DOP853 difference {difference:e}"
        );
        assert!(
            prepared
                .apply_checked_interval(&lap, &initial, &[0., 1e9])
                .is_err(),
            "Extending an interval must recompute and enforce its actual error gate"
        );
    }
    fn decay(rate: f64) -> Sparse {
        vec![
            vec![],
            vec![(0, rate), (1, -rate)],
            vec![(2, -rate / 2.)],
            vec![(3, -rate / 2.)],
        ]
    }
    #[test]
    fn two_incommensurate_modulations_keep_complete_density() {
        let lap = Lapack::load().unwrap();
        let w1 = 1e5;
        let w2 = 2_f64.sqrt() * 1e5;
        let input = Input {
            base: decay(1.),
            frequencies: vec![w1, w2],
            harmonics: vec![
                Harmonic {
                    index: vec![1, 0],
                    cos: decay(0.06),
                    sin: decay(-0.02),
                },
                Harmonic {
                    index: vec![0, 1],
                    cos: decay(0.04),
                    sin: decay(0.01),
                },
            ],
            bounded_generator_tail: 0.,
        };
        let prepared = prepare(&lap, &input, 5., 1e-8).unwrap();
        let initial = [0.5, 0.5, 0.4, -0.1];
        let times = [0., 0.1, 0.5, 2., 5.];
        let result = prepared.apply(&lap, &initial, &times).unwrap();
        let mut max_error = 0_f64;
        for (x, t) in result.states.iter().zip(times) {
            let integrated = t + 0.06 / w1 * (w1 * t).sin() - 0.02 / w1 * (1. - (w1 * t).cos())
                + 0.04 / w2 * (w2 * t).sin()
                + 0.01 / w2 * (1. - (w2 * t).cos());
            let e = (-integrated).exp();
            let c = (-integrated / 2.).exp();
            for (a, b) in x.iter().zip([1. - 0.5 * e, 0.5 * e, 0.4 * c, -0.1 * c]) {
                max_error = max_error.max((a - b).abs());
            }
        }
        assert!(
            max_error < result.fourier_error_bound + 1e-12,
            "error {max_error:e}, bound {:e}",
            result.fourier_error_bound
        );
        assert!(max_error < 2e-12);
        assert_eq!(result.states[0], initial);
        assert!(prepared.apply(&lap, &initial, &[5.01]).is_err());
    }
    #[test]
    fn supplied_original_fourier_tail_is_never_silently_ignored() {
        let lap = Lapack::load().unwrap();
        let input = Input {
            base: decay(1.),
            frequencies: vec![1e5, 1e4],
            harmonics: vec![],
            bounded_generator_tail: 1e-4,
        };
        let prepared = prepare(&lap, &input, 20., 1e-8).unwrap();
        assert!(prepared
            .apply(&lap, &[0., 1., 0., 0.], &[20.])
            .unwrap_err()
            .contains("certificate"));
    }
    #[test]
    fn algebraic_final_certificate_keeps_the_identical_updated_solution() {
        let lap = Lapack::load().unwrap();
        let input = Input {
            base: decay(1.),
            frequencies: vec![1e5, 2_f64.sqrt() * 1e5],
            harmonics: vec![
                Harmonic {
                    index: vec![1, 0],
                    cos: decay(0.06),
                    sin: decay(-0.02),
                },
                Harmonic {
                    index: vec![0, 1],
                    cos: decay(0.04),
                    sin: decay(0.01),
                },
            ],
            bounded_generator_tail: 0.,
        };
        let initial = [0.5, 0.5, 0.4, -0.1];
        let times = [0., 0.1, 1., 5.];
        let complete = prepare(&lap, &input, 5., 1e-8).unwrap();
        let fast = prepare_for_initial(&lap, &input, &initial, 5., 1e-8).unwrap();
        let full_last = complete.diagnostics.iterations.last().unwrap();
        let fast_last = fast.diagnostics.iterations.last().unwrap();
        assert!(fast_last.used_algebraic_bound);
        assert!(fast_last.evaluated_residual.is_none());
        assert!(full_last.evaluated_residual.unwrap() <= fast_last.total_residual);
        assert_eq!(
            complete.apply(&lap, &initial, &times).unwrap().states,
            fast.apply(&lap, &initial, &times).unwrap().states
        );
    }
    #[test]
    fn decay_weighted_integral_bounds_a_coupled_positive_semigroup() {
        let lap = Lapack::load().unwrap();
        let augmented = [-0.08, 0.01, 1., 0.03, -6.98, 1., 0., 0., 0.];
        let time = 5.;
        let actual = super::super::dense_expm::action(&lap, &augmented, &[0.2, 1., 0.], &[time])
            .unwrap()[0][2];
        let bound = 0.2 * exponential_integral(-0.1, time)
            + exponential_integral(-7., time)
            + 1.2 * remainder_integral(0.05, time);
        assert!(bound >= actual);
        assert!(bound < 1.2 * time * (0.05 * time).exp());
        assert!(remainder_integral(1e-12, 20.) >= 1e-12 * 20. * 20. * 0.5);
    }
    #[test]
    fn independent_static_coordinate_gauge_checks_the_complete_micromotion() {
        let lap = Lapack::load().unwrap();
        let input = Input {
            base: decay(1.),
            frequencies: vec![1e5, 2_f64.sqrt() * 1e5],
            harmonics: vec![Harmonic {
                index: vec![1, 0],
                cos: decay(0.06),
                sin: decay(-0.02),
            }],
            bounded_generator_tail: 0.,
        };
        let initial = [0.5, 0.5, 0.4, -0.1];
        let times = [0., 0.1, 1., 5.];
        let prepared = prepare(&lap, &input, 5., 1e-8).unwrap();
        let result = prepared.apply(&lap, &initial, &times).unwrap();
        let audit = prepared
            .audit_constant(&lap, &initial, &times, &result.states)
            .unwrap();
        assert!(audit.maximum_coordinate_difference < 2e-14);
        assert_eq!(audit.samples, times.len());
    }
}
