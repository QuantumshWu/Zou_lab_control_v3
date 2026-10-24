//! Error-gated full-density evolution of a weak periodic dissipator.
//!
//! All real Fourier generator coefficients are inputs. In a complete modal
//! basis of L0, S_q=-V_q/(lambda_a-lambda_b-i q omega) gives the exact identity
//! y=(I+S(t))z, z'=[Lambda+(I+S(t))^-1 E(t)S(t)]z. We retain all S_q and compute
//! the complete constant convolution F0=sum_q V_q S_-q. The optional interval
//! preparation may bound higher Fourier products without evaluating them;
//! their original operators are retained and a Duhamel bound is added on each
//! apply. Plain prepare always evaluates every harmonic. Oscillating second-
//! order terms are bounded using their spectral gaps; inverse-series and
//! Dyson remainders have separate bounds. No state/coherence, beam, decay
//! channel, or collapse Fourier cross term is removed from the physical model.
//!
//! This is a finite-time numerical integrator. Its tail bound does NOT include
//! floating-point error of the supplied Floquet frame/modal basis, and is not
//! a steady-state or t->infinity guarantee. An unmet error budget fails so the
//! caller can use the general Fourier method. It never silently returns D0.
use super::{
    cluster_action::{self, ModalBasis, Sparse},
    lapack::{Complex64, Lapack},
    modal_defect,
};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

type Z = Complex64;
#[derive(Clone)]
pub struct Harmonic {
    pub index: i32,
    pub cos: Sparse,
    pub sin: Sparse,
}
#[derive(Clone, Debug)]
pub struct BoundedHarmonic {
    pub index: i32,
    pub raw_norm_sum: f64,
    pub modal_norm_bound: f64,
}
#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub dimension: usize,
    pub component_dimensions: Vec<usize>,
    pub modal_condition: f64,
    pub harmonic_indices: Vec<i32>,
    pub summed_drive_norm: f64,
    pub summed_transform_norm: f64,
    pub mean_correction_norm: f64,
    pub minimum_sylvester_gap: Option<f64>,
    pub oscillatory_convolution_bound: f64,
    pub inverse_generator_bound: f64,
    pub dense_harmonic_products: usize,
    pub bounded_harmonics: Vec<BoundedHarmonic>,
    pub requested_interval: Option<f64>,
    pub requested_tolerance: Option<f64>,
    pub preparation_ms: f64,
    pub raw_modal_components: usize,
    pub base_defect_norm: f64,
    pub base_representation_error: f64,
    pub raw_candidate_rejection: Option<String>,
    pub modal_basis_reused: bool,
}
#[derive(Clone, Debug)]
pub struct TailBounds {
    pub inverse_initial: f64,
    pub inverse_generator: f64,
    pub oscillatory_convolution: f64,
    pub dyson_remainder: f64,
    pub total_coordinate_l1: f64,
    /// Duhamel bound for original higher Fourier generators whose matrix
    /// products were not evaluated. Recomputed for every actual initial/time.
    pub bounded_fourier_tail: f64,
    /// Propagated measured reconstruction error of the raw numerical basis.
    pub modal_initial_reconstruction: f64,
}
#[derive(Debug)]
pub struct ActionResult {
    pub states: Vec<Vec<f64>>,
    pub bounds: TailBounds,
    pub initial_error: f64,
    pub imaginary_residue: f64,
    pub elapsed_ms: f64,
}
/// Coefficients of a real linear observable O(t)=sum_l O_l exp(i l omega t).
/// Each matrix is column-major rows x density_dimension. Supply all signed
/// indices; complex coefficients are necessary even for real observables.
#[derive(Clone)]
pub struct ObservableHarmonic {
    pub index: i32,
    pub rows: usize,
    pub matrix: Vec<Z>,
}
pub struct ObservablePrepared {
    pub rows: usize,
    pub maximum_time: f64,
    pub preparation_ms: f64,
    source_id: u64,
    indices: Vec<i32>,
    projection: Vec<Z>,
    quotient: Vec<Block>,
    close: Vec<(usize, usize, Z, Z)>,
    zero_operator: Vec<Z>,
    row_norms: Vec<f64>,
}
#[derive(Debug)]
pub struct ObservableResult {
    pub values: Vec<Vec<f64>>,
    pub bounds: TailBounds,
    pub row_error_bounds: Vec<f64>,
    pub initial_error: f64,
    pub imaginary_residue: f64,
    pub elapsed_ms: f64,
}
#[derive(Clone)]
struct Component {
    coordinates: Vec<usize>,
    offset: usize,
    vectors: Vec<Z>,
    inverse: Vec<Z>,
}
#[derive(Clone)]
struct Block {
    row: usize,
    col: usize,
    matrix: Vec<Z>,
}
struct ModalHarmonic {
    index: i32,
    transform: Vec<Block>,
    drive_norm: f64,
    transform_norm: f64,
}
pub struct Prepared {
    pub diagnostics: Diagnostics,
    id: u64,
    n: usize,
    omega: f64,
    modal: ModalBasis,
    components: Vec<Component>,
    conjugate: Vec<usize>,
    harmonics: Vec<ModalHarmonic>,
    mean: Vec<Block>,
    transform_norm: f64,
    drive_norm: f64,
    oscillation_tail: f64,
    inverse_tail: f64,
    alpha: f64,
    vector_norm: f64,
    bounded_norm: f64,
    bounded_operators: Vec<Harmonic>,
    base_defect_norm: f64,
    base_representation_error: f64,
    base: Sparse,
    base_defects: Vec<Block>,
    lindblad_dimension: Option<usize>,
    bounded_raw_norm: f64,
}
static PREPARED_ID: AtomicU64 = AtomicU64::new(1);

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
fn scale(a: Z, b: f64) -> Z {
    Z {
        re: a.re * b,
        im: a.im * b,
    }
}
#[inline]
fn conj(a: Z) -> Z {
    Z {
        re: a.re,
        im: -a.im,
    }
}
#[inline]
fn abs(a: Z) -> f64 {
    a.re.hypot(a.im)
}
fn divide(a: Z, b: Z) -> Z {
    let largest = b.re.abs().max(b.im.abs());
    let normalized = scale(b, 1. / largest);
    let numerator = mul(scale(a, 1. / largest), conj(normalized));
    scale(
        numerator,
        1. / (normalized.re * normalized.re + normalized.im * normalized.im),
    )
}
fn norm_one(a: &[Z], n: usize) -> f64 {
    a.chunks(n)
        .map(|col| col.iter().map(|&z| abs(z)).sum::<f64>())
        .fold(0., f64::max)
}
fn finite(a: &[Z], label: &str) -> Result<(), String> {
    if a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        Err(format!(
            "Non-finite {label} in periodic full-density integration"
        ))
    } else {
        Ok(())
    }
}
fn validate(a: &Sparse, n: usize) -> Result<(), String> {
    if a.len() != n
        || a.iter()
            .any(|col| col.iter().any(|&(row, v)| row >= n || !v.is_finite()))
    {
        Err("Invalid real Fourier generator dimensions/entries".into())
    } else {
        Ok(())
    }
}
fn real_norm_one(a: &Sparse) -> f64 {
    a.iter()
        .map(|col| col.iter().map(|&(_, v)| v.abs()).sum::<f64>())
        .fold(0., f64::max)
}

/// Preserve the low product bits before trigonometric argument reduction.
fn phase(frequency: f64, time: f64, index: i32) -> Result<Z, String> {
    let main = frequency * time;
    let low = frequency.mul_add(time, -main);
    let q = index as f64;
    let argument = main * q;
    let correction = main.mul_add(q, -argument) + low * q;
    if !argument.is_finite() || !correction.is_finite() {
        return Err("Periodic phase overflowed".into());
    }
    let (s, c) = argument.sin_cos();
    // The low multiplication bits are tiny for ordinary physical times. A
    // bounded small-angle polynomial avoids a second transcendental call;
    // at 1e-4 its first omitted sine term is below 2e-32.
    let (ds, dc) = if correction.abs() <= 1e-4 {
        let x2 = correction * correction;
        (
            correction * (1. - x2 / 6. + x2 * x2 / 120.),
            1. - x2 / 2. + x2 * x2 / 24. - x2 * x2 * x2 / 720.,
        )
    } else {
        correction.sin_cos()
    };
    Ok(Z {
        re: c * dc - s * ds,
        im: s * dc + c * ds,
    })
}
fn exponential(value: Z, time: f64) -> Result<Z, String> {
    let result = if value.im == 0. {
        Z {
            re: (value.re * time).exp(),
            im: 0.,
        }
    } else {
        scale(phase(value.im, time, 1)?, (value.re * time).exp())
    };
    finite(&[result], "modal exponential")?;
    Ok(result)
}
fn exprel_small(z: Z) -> Z {
    // Used only for |z|<1e-3; the omitted ninth term is below 3e-34.
    let mut result = Z { re: 1., im: 0. };
    let mut term = result;
    for k in 1..=8 {
        term = scale(mul(term, z), 1. / (k + 1) as f64);
        result = add(result, term);
    }
    result
}

fn direct_sum_modes(
    lap: &Lapack,
    base: &Sparse,
    allow_raw: bool,
) -> Result<(ModalBasis, Vec<Component>, Vec<Block>, f64, usize), String> {
    let n = base.len();
    let mut adjacent = vec![vec![]; n];
    for (col, entries) in base.iter().enumerate() {
        for &(row, value) in entries {
            if value != 0. && row != col {
                adjacent[col].push(row);
                adjacent[row].push(col);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut components = vec![];
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut component = vec![];
        while let Some(i) = queue.pop_front() {
            component.push(i);
            for &j in &adjacent[i] {
                if !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    let mut vectors = vec![Z::default(); n * n];
    let mut inverse = vec![Z::default(); n * n];
    let mut eigenvalues = vec![];
    let mut origin = vec![];
    let mut offset = 0;
    let mut saved = vec![];
    let mut defects = vec![];
    let mut representation_error = 0_f64;
    let mut raw_components = 0;
    for component in &components {
        let m = component.len();
        let mut map = vec![usize::MAX; n];
        for (i, &global) in component.iter().enumerate() {
            map[global] = i;
        }
        let local: Sparse = component
            .iter()
            .map(|&col| {
                base[col]
                    .iter()
                    .filter_map(|&(row, value)| (value != 0.).then_some((map[row], value)))
                    .collect()
            })
            .collect();
        // A matrix norm limits the complete component's spectral range. This
        // is only a numerical candidate selector: every mode is retained and
        // its ORIGINAL-generator defect is carried into the evolution.
        let candidate = if allow_raw && real_norm_one(&local) <= 1e5 {
            modal_defect::raw_modal_basis(lap, &local)
                .and_then(|modal| {
                    let delta = modal_defect::compute(
                        lap,
                        &local,
                        &modal.vectors,
                        &modal.inverse,
                        &modal.eigenvalues,
                    )?;
                    Ok((modal, delta))
                })
                .ok()
        } else {
            None
        };
        let modal = if let Some((modal, delta)) = candidate {
            let error = (delta.diagnostics.modal_roundoff_one
                + delta.diagnostics.inverse_residual_one_bound
                    * delta.diagnostics.modal_defect_one)
                / (1. - delta.diagnostics.inverse_residual_one_bound);
            representation_error = representation_error.max(error);
            let ci = saved.len();
            defects.push(Block {
                row: ci,
                col: ci,
                matrix: delta.matrix,
            });
            raw_components += 1;
            modal
        } else {
            let prepared = cluster_action::prepare(lap, &local, 1e5)?;
            prepared.modal_basis(lap)?
        };
        for j in 0..m {
            for i in 0..m {
                vectors[component[i] + (offset + j) * n] = modal.vectors[i + j * m];
                inverse[offset + i + component[j] * n] = modal.inverse[i + j * m];
            }
        }
        eigenvalues.extend(modal.eigenvalues);
        origin.extend(modal.origin);
        saved.push(Component {
            coordinates: component.clone(),
            offset,
            vectors: modal.vectors,
            inverse: modal.inverse,
        });
        offset += m;
    }
    let condition = norm_one(&vectors, n) * norm_one(&inverse, n);
    if !condition.is_finite() || condition > 1e10 {
        return Err(
            "Constant full-density modal basis is ill-conditioned; use the general solver".into(),
        );
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
        defects,
        representation_error,
        raw_components,
    ))
}

fn block_norm(blocks: &[Block], components: &[Component], n: usize) -> f64 {
    let mut sums = vec![0.; n];
    for block in blocks {
        let rows = components[block.row].coordinates.len();
        let col = &components[block.col];
        for j in 0..col.coordinates.len() {
            sums[col.offset + j] += block.matrix[j * rows..(j + 1) * rows]
                .iter()
                .map(|&z| abs(z))
                .sum::<f64>();
        }
    }
    sums.into_iter().fold(0., f64::max)
}
fn block_conjugate(block: &Block, components: &[Component], conjugate: &[usize]) -> Block {
    let row = &components[block.row];
    let col = &components[block.col];
    let rows = row.coordinates.len();
    let columns = col.coordinates.len();
    Block {
        row: block.row,
        col: block.col,
        matrix: (0..rows * columns)
            .map(|k| {
                conj(
                    block.matrix[(conjugate[row.offset + k % rows] - row.offset)
                        + (conjugate[col.offset + k / rows] - col.offset) * rows],
                )
            })
            .collect(),
    }
}
fn block_action(
    lap: &Lapack,
    blocks: &[Block],
    components: &[Component],
    input: &[Z],
    n: usize,
    count: usize,
) -> Result<Vec<Z>, String> {
    let mut output = vec![Z::default(); n * count];
    let mut packed = BTreeMap::<usize, Vec<Z>>::new();
    for block in blocks {
        let row = &components[block.row];
        let col = &components[block.col];
        let rows = row.coordinates.len();
        let columns = col.coordinates.len();
        let source = packed.entry(block.col).or_insert_with(|| {
            (0..count)
                .flat_map(|t| {
                    input[t * n + col.offset..t * n + col.offset + columns]
                        .iter()
                        .copied()
                })
                .collect()
        });
        let value = lap.multiply_complex(&block.matrix, source, rows, count, columns)?;
        for t in 0..count {
            for i in 0..rows {
                output[row.offset + i + t * n] =
                    add(output[row.offset + i + t * n], value[i + t * rows]);
            }
        }
    }
    Ok(output)
}
fn physical_action(
    lap: &Lapack,
    components: &[Component],
    input: &[Z],
    n: usize,
    count: usize,
) -> Result<Vec<Z>, String> {
    let mut output = vec![Z::default(); n * count];
    for component in components {
        let size = component.coordinates.len();
        let source: Vec<_> = (0..count)
            .flat_map(|t| {
                input[t * n + component.offset..t * n + component.offset + size]
                    .iter()
                    .copied()
            })
            .collect();
        let value = lap.multiply_complex(&component.vectors, &source, size, count, size)?;
        for t in 0..count {
            for (i, &row) in component.coordinates.iter().enumerate() {
                output[row + t * n] = value[i + t * size];
            }
        }
    }
    Ok(output)
}
fn observable_right_action(
    lap: &Lapack,
    left: &[Z],
    rows: usize,
    blocks: &[Block],
    components: &[Component],
    n: usize,
) -> Result<Vec<Z>, String> {
    let mut output = vec![Z::default(); rows * n];
    for block in blocks {
        let row = &components[block.row];
        let col = &components[block.col];
        let inner = row.coordinates.len();
        let columns = col.coordinates.len();
        let product = lap.multiply_complex(
            &left[row.offset * rows..(row.offset + inner) * rows],
            &block.matrix,
            rows,
            columns,
            inner,
        )?;
        for j in 0..columns {
            for i in 0..rows {
                let k = i + (col.offset + j) * rows;
                output[k] = add(output[k], product[i + j * rows]);
            }
        }
    }
    Ok(output)
}

pub fn prepare(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
) -> Result<Prepared, String> {
    prepare_internal(lap, base, harmonics, omega, None, false, None, None)
}
/// Optional finite-interval numerical optimization. q=1,2 are always explicit.
/// Higher original operators are retained and may be bounded without computing
/// their modal products. This path is never an infinity/steady-state shortcut.
pub fn prepare_for_interval(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    maximum_time: f64,
    tolerance: f64,
) -> Result<Prepared, String> {
    prepare_refined_for_interval(lap, base, harmonics, omega, maximum_time, tolerance)
}
/// Explicit fallback keeps the original refined modal preparation.
pub fn prepare_refined_for_interval(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    maximum_time: f64,
    tolerance: f64,
) -> Result<Prepared, String> {
    if !maximum_time.is_finite() || maximum_time < 0. || !tolerance.is_finite() || tolerance <= 0. {
        return Err("Invalid finite-interval preparation budget".into());
    }
    prepare_internal(
        lap,
        base,
        harmonics,
        omega,
        Some((maximum_time, tolerance)),
        false,
        None,
        None,
    )
}
/// Try a complete raw basis only on low-range components. Its FULL Delta,
/// including off-diagonal entries, is retained. The actual initial vector and
/// finite interval must pass the same gate; otherwise rebuild the refined path.
pub fn prepare_for_initial(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    initial: &[f64],
    maximum_time: f64,
    tolerance: f64,
) -> Result<Prepared, String> {
    prepare_initial_internal(
        lap,
        base,
        harmonics,
        omega,
        initial,
        maximum_time,
        tolerance,
        None,
    )
}
/// Only for coefficients constructed from a complete Hermitian H(t) and
/// complete C_q(t) dissipator. The caller must establish that Lindblad form;
/// this is not inferred from arbitrary real matrices or small residuals.
pub(super) fn prepare_lindblad_for_initial(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    initial: &[f64],
    maximum_time: f64,
    tolerance: f64,
    hilbert_dimension: usize,
) -> Result<Prepared, String> {
    if hilbert_dimension == 0
        || hilbert_dimension.checked_mul(hilbert_dimension) != Some(base.len())
    {
        return Err("Invalid full Lindblad Hermitian coordinate dimension".into());
    }
    prepare_initial_internal(
        lap,
        base,
        harmonics,
        omega,
        initial,
        maximum_time,
        tolerance,
        Some(hilbert_dimension),
    )
}
fn prepare_initial_internal(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    initial: &[f64],
    maximum_time: f64,
    tolerance: f64,
    lindblad_dimension: Option<usize>,
) -> Result<Prepared, String> {
    if !maximum_time.is_finite() || maximum_time < 0. || !tolerance.is_finite() || tolerance <= 0. {
        return Err("Invalid finite-interval preparation budget".into());
    }
    let candidate = prepare_internal(
        lap,
        base,
        harmonics,
        omega,
        Some((maximum_time, tolerance)),
        true,
        None,
        lindblad_dimension,
    )
    .and_then(|prepared| {
        let modal = prepared.initial_modal(lap, initial)?;
        prepared.checked_bounds(lap, initial, &modal, maximum_time, tolerance)?;
        Ok(prepared)
    });
    match candidate {
        Ok(prepared) => Ok(prepared),
        Err(reason) => {
            let mut refined = prepare_internal(
                lap,
                base,
                harmonics,
                omega,
                Some((maximum_time, tolerance)),
                false,
                None,
                lindblad_dimension,
            )?;
            let modal = refined.initial_modal(lap, initial)?;
            refined.checked_bounds(lap, initial, &modal, maximum_time, tolerance)?;
            refined.diagnostics.raw_candidate_rejection = Some(reason);
            Ok(refined)
        }
    }
}
fn prepare_internal(
    lap: &Lapack,
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    interval: Option<(f64, f64)>,
    allow_raw: bool,
    reuse: Option<&Prepared>,
    lindblad_dimension: Option<usize>,
) -> Result<Prepared, String> {
    let started = Instant::now();
    let n = base.len();
    if n == 0 || !omega.is_finite() || omega <= 0. {
        return Err("Invalid periodic generator dimension or beat".into());
    }
    validate(base, n)?;
    let (modal, components, base_defects, base_representation_error, raw_modal_components) =
        if let Some(old) = reuse {
            if old.base != *base || old.omega != omega || old.diagnostics.raw_modal_components != 0
            {
                return Err(
                    "Periodic modal cache does not match the complete refined base generator"
                        .into(),
                );
            }
            let modal = ModalBasis {
                dimension: old.modal.dimension,
                eigenvalues: old.modal.eigenvalues.clone(),
                vectors: old.modal.vectors.clone(),
                inverse: old.modal.inverse.clone(),
                condition: old.modal.condition,
                origin: old.modal.origin.clone(),
            };
            (
                modal,
                old.components.clone(),
                old.base_defects.clone(),
                old.base_representation_error,
                0,
            )
        } else {
            direct_sum_modes(lap, base, allow_raw)?
        };
    let base_defect_norm = block_norm(&base_defects, &components, n);
    let original_base_defects = base_defects.clone();
    let component_dimensions = components.iter().map(|c| c.coordinates.len()).collect();
    let mut coordinate_map = vec![(0, 0); n];
    for (ci, component) in components.iter().enumerate() {
        for (local, &global) in component.coordinates.iter().enumerate() {
            coordinate_map[global] = (ci, local);
        }
    }
    let mut conjugate: Vec<_> = (0..n).collect();
    let mut i = 0;
    while i < n {
        if modal.eigenvalues[i].im > 0. {
            if i + 1 >= n
                || modal.eigenvalues[i + 1].im != -modal.eigenvalues[i].im
                || modal.eigenvalues[i + 1].re != modal.eigenvalues[i].re
            {
                return Err("Incomplete complex modal pair".into());
            }
            conjugate[i] = i + 1;
            conjugate[i + 1] = i;
            i += 2;
        } else {
            i += 1;
        }
    }
    let mut grouped: BTreeMap<i32, (Sparse, Sparse)> = BTreeMap::new();
    for harmonic in harmonics {
        if harmonic.index == 0 || harmonic.index == i32::MIN {
            return Err("Zero/overflow Fourier index belongs in the constant generator".into());
        }
        validate(&harmonic.cos, n)?;
        validate(&harmonic.sin, n)?;
        let (cos, sin) = grouped
            .entry(harmonic.index.abs())
            .or_insert_with(|| (vec![vec![]; n], vec![vec![]; n]));
        for col in 0..n {
            cos[col].extend_from_slice(&harmonic.cos[col]);
            sin[col].extend(
                harmonic.sin[col]
                    .iter()
                    .map(|&(row, v)| (row, v * harmonic.index.signum() as f64)),
            );
        }
    }
    let raw_bounds: BTreeMap<_, _> = grouped
        .iter()
        .map(|(&q, (c, s))| {
            let raw = real_norm_one(c) + real_norm_one(s);
            (q, (raw, modal.condition * raw))
        })
        .collect();
    let alpha = modal.eigenvalues.iter().map(|v| v.re).fold(0., f64::max);
    let mut output: Vec<ModalHarmonic> = vec![];
    let mut mean_map: BTreeMap<(usize, usize), Vec<Z>> = base_defects
        .into_iter()
        .map(|b| ((b.row, b.col), b.matrix))
        .collect();
    let mut minimum_gap = f64::INFINITY;
    let mut dense_harmonic_products = 0;
    let mut bounded_norm = 0.;
    let mut bounded_raw_norm = 0.;
    let mut bounded_operators = vec![];
    let mut bounded_harmonics = vec![];
    for (index, (cos, sin)) in grouped {
        if index >= 3 {
            if let Some((time, tolerance)) = interval {
                let (raw_norm_sum, modal_norm_bound) = raw_bounds[&index];
                let candidate = bounded_norm + modal_norm_bound;
                let explicit = 2. * output.iter().map(|h| h.drive_norm).sum::<f64>();
                let future = raw_bounds
                    .range((std::ops::Bound::Excluded(index), std::ops::Bound::Unbounded))
                    .map(|(_, (_, bound))| *bound)
                    .sum::<f64>();
                // ||R|| ||R^-1|| bound for unit coordinate-l1 initial vectors.
                // The actual initial vector is checked again by apply(); no cache
                // may reuse this estimate for a longer/stronger request unchecked.
                let unit_bound = if let Some(h) = lindblad_dimension {
                    // ||x||_1 <= h ||rho||_tr and ||rho||_tr <= sqrt(2h)||x||_1.
                    // For a unit coordinate-l1 initial vector, the full CPTP
                    // propagator gives h sqrt(2h) [exp(delta*t)-1], with
                    // delta=h sqrt(2h)||omitted original generator||_1.
                    let conversion = h as f64 * (2. * h as f64).sqrt();
                    conversion * (conversion * (bounded_raw_norm + raw_norm_sum) * time).exp_m1()
                } else {
                    modal.condition
                        * ((alpha
                            + base_defect_norm
                            + base_representation_error
                            + explicit
                            + future)
                            * time)
                            .exp()
                        * (candidate * time).exp_m1()
                };
                if unit_bound.is_finite() && unit_bound <= 0.25 * tolerance {
                    bounded_norm = candidate;
                    bounded_raw_norm += raw_norm_sum;
                    bounded_harmonics.push(BoundedHarmonic {
                        index,
                        raw_norm_sum,
                        modal_norm_bound,
                    });
                    bounded_operators.push(Harmonic { index, cos, sin });
                    continue;
                }
            }
        }
        let frequency = index as f64 * omega;
        if !frequency.is_finite() {
            return Err("Fourier frequency overflow".into());
        }
        // (Ac-i As)R/2; the negative coefficient follows by the exact modal
        // conjugate-pair permutation, not by a physical approximation.
        let mut physical_blocks = BTreeMap::<(usize, usize), Vec<Z>>::new();
        for col in 0..n {
            let (cj, j) = coordinate_map[col];
            for &(row, value) in &cos[col] {
                if value == 0. {
                    continue;
                }
                let (ci, i) = coordinate_map[row];
                let rows = components[ci].coordinates.len();
                let columns = components[cj].coordinates.len();
                physical_blocks
                    .entry((ci, cj))
                    .or_insert_with(|| vec![Z::default(); rows * columns])[i + j * rows]
                    .re += 0.5 * value;
            }
            for &(row, value) in &sin[col] {
                if value == 0. {
                    continue;
                }
                let (ci, i) = coordinate_map[row];
                let rows = components[ci].coordinates.len();
                let columns = components[cj].coordinates.len();
                physical_blocks
                    .entry((ci, cj))
                    .or_insert_with(|| vec![Z::default(); rows * columns])[i + j * rows]
                    .im -= 0.5 * value;
            }
        }
        let mut drives = vec![];
        let mut transforms = vec![];
        for ((ci, cj), plus) in physical_blocks {
            if plus.iter().all(|z| z.re == 0. && z.im == 0.) {
                continue;
            }
            let rows = components[ci].coordinates.len();
            let columns = components[cj].coordinates.len();
            dense_harmonic_products += 1;
            let rhs =
                lap.multiply_complex(&plus, &components[cj].vectors, rows, columns, columns)?;
            let drive = lap.multiply_complex(&components[ci].inverse, &rhs, rows, columns, rows)?;
            let mut transform = vec![Z::default(); rows * columns];
            for col in 0..columns {
                for row in 0..rows {
                    let mut denominator = sub(
                        modal.eigenvalues[components[ci].offset + row],
                        modal.eigenvalues[components[cj].offset + col],
                    );
                    denominator.im -= frequency;
                    let gap = abs(denominator);
                    if gap == 0. || !gap.is_finite() {
                        return Err(
                            "Periodic Sylvester equation is resonant; use the general solver"
                                .into(),
                        );
                    }
                    minimum_gap = minimum_gap.min(gap);
                    transform[row + col * rows] =
                        scale(divide(drive[row + col * rows], denominator), -1.);
                }
            }
            finite(&transform, "periodic transformation")?;
            drives.push(Block {
                row: ci,
                col: cj,
                matrix: drive,
            });
            transforms.push(Block {
                row: ci,
                col: cj,
                matrix: transform,
            });
        }
        let negatives: Vec<_> = transforms
            .iter()
            .map(|b| block_conjugate(b, &components, &conjugate))
            .collect();
        for a in &drives {
            for b in &negatives {
                if a.col != b.row {
                    continue;
                }
                let rows = components[a.row].coordinates.len();
                let columns = components[b.col].coordinates.len();
                let inner = components[a.col].coordinates.len();
                let product = Block {
                    row: a.row,
                    col: b.col,
                    matrix: lap.multiply_complex(&a.matrix, &b.matrix, rows, columns, inner)?,
                };
                let paired = block_conjugate(&product, &components, &conjugate);
                let entry = mean_map
                    .entry((a.row, b.col))
                    .or_insert_with(|| vec![Z::default(); rows * columns]);
                for i in 0..entry.len() {
                    entry[i] = add(entry[i], add(product.matrix[i], paired.matrix[i]));
                }
            }
        }
        let drive_norm = block_norm(&drives, &components, n);
        let transform_norm = block_norm(&transforms, &components, n);
        output.push(ModalHarmonic {
            index,
            transform: transforms,
            drive_norm,
            transform_norm,
        });
    }
    let mean: Vec<_> = mean_map
        .into_iter()
        .filter(|(_, v)| v.iter().any(|z| z.re != 0. || z.im != 0.))
        .map(|((row, col), matrix)| Block { row, col, matrix })
        .collect();
    for block in &mean {
        finite(&block.matrix, "constant convolution")?;
    }
    let transform_norm = 2. * output.iter().map(|h| h.transform_norm).sum::<f64>();
    let drive_norm = 2. * output.iter().map(|h| h.drive_norm).sum::<f64>();
    if !transform_norm.is_finite() || transform_norm >= 1. {
        return Err(format!("The periodic near-identity inverse is not bounded (norm {transform_norm:e}); use the general solver"));
    }
    let mut convolution_bounds = BTreeMap::<i32, f64>::new();
    for a in &output {
        for b in &output {
            for sa in [-1_i32, 1] {
                for sb in [-1_i32, 1] {
                    let index = a
                        .index
                        .checked_mul(sa)
                        .and_then(|x| b.index.checked_mul(sb).and_then(|y| x.checked_add(y)))
                        .ok_or("Convolution harmonic index overflow")?;
                    *convolution_bounds.entry(index).or_default() +=
                        a.drive_norm * b.transform_norm;
                }
            }
        }
    }
    let maximum_imag = modal
        .eigenvalues
        .iter()
        .map(|v| v.im)
        .fold(f64::NEG_INFINITY, f64::max);
    let minimum_imag = modal
        .eigenvalues
        .iter()
        .map(|v| v.im)
        .fold(f64::INFINITY, f64::min);
    let span = maximum_imag - minimum_imag;
    let mut oscillation_tail = 0.;
    for (index, bound) in convolution_bounds {
        if index == 0 {
            continue;
        }
        let gap = index.unsigned_abs() as f64 * omega - span;
        if gap <= 0. {
            return Err(
                "Oscillating convolution gap is not separated; use explicit Fourier propagation"
                    .into(),
            );
        }
        oscillation_tail += 2. * bound / gap;
    }
    // For A=Lambda+Delta+E and y=(I+S)z, the extra transformed generator is
    // (I+S)^-1([Delta,S]+E S). Delta itself is in the retained Frechet term.
    let inverse_tail = (drive_norm * transform_norm * transform_norm
        + 2. * base_defect_norm * transform_norm
        + base_representation_error * (1. + transform_norm))
        / (1. - transform_norm);
    let vector_norm = norm_one(&modal.vectors, n);
    let diagnostics = Diagnostics {
        dimension: n,
        component_dimensions,
        modal_condition: modal.condition,
        harmonic_indices: output.iter().map(|h| h.index).collect(),
        summed_drive_norm: drive_norm,
        summed_transform_norm: transform_norm,
        mean_correction_norm: block_norm(&mean, &components, n),
        minimum_sylvester_gap: minimum_gap.is_finite().then_some(minimum_gap),
        oscillatory_convolution_bound: oscillation_tail,
        inverse_generator_bound: inverse_tail,
        dense_harmonic_products,
        bounded_harmonics,
        requested_interval: interval.map(|v| v.0),
        requested_tolerance: interval.map(|v| v.1),
        preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        raw_modal_components,
        base_defect_norm,
        base_representation_error,
        raw_candidate_rejection: None,
        modal_basis_reused: reuse.is_some(),
    };
    Ok(Prepared {
        id: PREPARED_ID.fetch_add(1, Ordering::Relaxed),
        diagnostics,
        n,
        omega,
        modal,
        components,
        conjugate,
        harmonics: output,
        mean,
        transform_norm,
        drive_norm,
        oscillation_tail,
        inverse_tail,
        alpha,
        vector_norm,
        bounded_norm,
        bounded_operators,
        base_defect_norm,
        base_representation_error,
        base: base.clone(),
        base_defects: original_base_defects,
        lindblad_dimension,
        bounded_raw_norm,
    })
}

impl Prepared {
    /// Re-evaluate every Fourier-product budget for a changed interval while
    /// reusing only an exactly matching complete refined constant eigensystem.
    /// A rejected raw-basis candidate is rebuilt through the refined path.
    pub fn reprepare_for_interval(
        &self,
        lap: &Lapack,
        base: &Sparse,
        harmonics: &[Harmonic],
        maximum: f64,
        tolerance: f64,
    ) -> Result<Prepared, String> {
        if !maximum.is_finite() || maximum < 0. || !tolerance.is_finite() || tolerance <= 0. {
            return Err("Invalid periodic re-preparation budget".into());
        }
        if self.diagnostics.raw_modal_components > 0 {
            return prepare_internal(
                lap,
                base,
                harmonics,
                self.omega,
                Some((maximum, tolerance)),
                false,
                None,
                self.lindblad_dimension,
            );
        }
        prepare_internal(
            lap,
            base,
            harmonics,
            self.omega,
            Some((maximum, tolerance)),
            false,
            Some(self),
            self.lindblad_dimension,
        )
    }
    pub fn bounded_operators(&self) -> &[Harmonic] {
        &self.bounded_operators
    }
    fn zero_phase_action(&self, lap: &Lapack, vector: &[Z]) -> Result<Vec<Z>, String> {
        let mut out = vec![Z::default(); self.n];
        for h in &self.harmonics {
            let value = block_action(lap, &h.transform, &self.components, vector, self.n, 1)?;
            for i in 0..self.n {
                out[i] = add(out[i], add(value[i], conj(value[self.conjugate[i]])));
            }
        }
        Ok(out)
    }
    fn initial_modal(&self, lap: &Lapack, initial: &[f64]) -> Result<Vec<Z>, String> {
        if initial.len() != self.n || initial.iter().any(|v| !v.is_finite()) {
            return Err("Invalid periodic initial coordinates".into());
        }
        let mut modal = vec![Z::default(); self.n];
        for component in &self.components {
            let size = component.coordinates.len();
            let source: Vec<_> = component
                .coordinates
                .iter()
                .map(|&i| Z {
                    re: initial[i],
                    im: 0.,
                })
                .collect();
            let values = lap.multiply_complex(&component.inverse, &source, size, 1, size)?;
            modal[component.offset..component.offset + size].copy_from_slice(&values);
        }
        Ok(modal)
    }
    fn numerical_bounds(
        &self,
        initial: &[f64],
        modal: &[Z],
        maximum: f64,
        tolerance: f64,
    ) -> Result<TailBounds, String> {
        if !maximum.is_finite() || maximum < 0. || !tolerance.is_finite() || tolerance <= 0. {
            return Err("Invalid periodic numerical error budget".into());
        }
        let norm = modal.iter().map(|&z| abs(z)).sum::<f64>();
        let s = self.transform_norm;
        let factor = self.vector_norm * (1. + s) * norm / (1. - s);
        let b = (self.drive_norm * s
            + (self.base_defect_norm + self.base_representation_error) * (1. + s))
            / (1. - s);
        let growth = ((self.alpha + b) * maximum).exp();
        let inverse_initial = factor * growth * s.powi(3);
        let inverse_generator = factor * growth * maximum * self.inverse_tail;
        let oscillatory_convolution = factor * growth * self.oscillation_tail;
        let dyson_remainder = factor * growth * 0.5 * (maximum * b).powi(2);
        // Variation of constants bounds every effect of the unevaluated
        // original Fourier operators, including their cross convolutions.
        let bounded_fourier_tail = if self.bounded_norm == 0. {
            0.
        } else if let Some(h) = self.lindblad_dimension {
            // The complete time-dependent Lindblad propagator is contractive
            // on Hermitian trace norm. Variation of constants against that
            // exact propagator and Gronwall give ||error||_tr <=
            // ||rho0||_tr [exp(delta*t)-1], even though omitting individual
            // Fourier terms need not leave a Lindblad approximate generator.
            // Bound the actual initial trace norm by sqrt(h)||rho0||_F.
            let frobenius = (initial[..h].iter().map(|v| v * v).sum::<f64>()
                + 2. * initial[h..].iter().map(|v| v * v).sum::<f64>())
            .sqrt();
            let conversion = h as f64 * (2. * h as f64).sqrt();
            h as f64
                * (h as f64).sqrt()
                * frobenius
                * (conversion * self.bounded_raw_norm * maximum).exp_m1()
        } else {
            self.vector_norm
                * norm
                * ((self.alpha
                    + self.base_defect_norm
                    + self.base_representation_error
                    + self.drive_norm)
                    * maximum)
                    .exp()
                * (self.bounded_norm * maximum).exp_m1()
        };
        let total_coordinate_l1 = inverse_initial
            + inverse_generator
            + oscillatory_convolution
            + dyson_remainder
            + bounded_fourier_tail;
        if !total_coordinate_l1.is_finite() || total_coordinate_l1 > tolerance {
            return Err(format!("Periodic full-dissipator numerical tail {total_coordinate_l1:e} exceeds {tolerance:e}; rebuild with explicit bounded harmonics or use the general solver, no averaged result returned"));
        }
        Ok(TailBounds {
            inverse_initial,
            inverse_generator,
            oscillatory_convolution,
            dyson_remainder,
            bounded_fourier_tail,
            total_coordinate_l1,
            modal_initial_reconstruction: 0.,
        })
    }
    fn checked_bounds(
        &self,
        lap: &Lapack,
        initial: &[f64],
        modal: &[Z],
        maximum: f64,
        tolerance: f64,
    ) -> Result<TailBounds, String> {
        let mut bounds = self.numerical_bounds(initial, modal, maximum, tolerance)?;
        if self.diagnostics.raw_modal_components > 0 {
            let reconstructed = physical_action(lap, &self.components, modal, self.n, 1)?;
            let mismatch = reconstructed
                .iter()
                .zip(initial)
                .map(|(a, b)| (a.re - b).hypot(a.im))
                .sum::<f64>();
            let growth = ((self.alpha
                + self.base_defect_norm
                + self.base_representation_error
                + self.drive_norm
                + self.bounded_norm)
                * maximum)
                .exp();
            bounds.modal_initial_reconstruction = self.modal.condition * growth * mismatch;
            bounds.total_coordinate_l1 += bounds.modal_initial_reconstruction;
            if !bounds.total_coordinate_l1.is_finite() || bounds.total_coordinate_l1 > tolerance {
                return Err(format!("Raw-modal reconstruction and numerical tail {} exceed {tolerance:e}; rebuild the refined basis",bounds.total_coordinate_l1));
            }
        }
        Ok(bounds)
    }
    fn mean_quotient(&self, maximum: f64) -> (Vec<Block>, Vec<(usize, usize, Z, Z)>) {
        let mut quotient = vec![];
        let mut close = vec![];
        for block in &self.mean {
            let row_component = &self.components[block.row];
            let col_component = &self.components[block.col];
            let rows = row_component.coordinates.len();
            let columns = col_component.coordinates.len();
            let mut matrix = vec![Z::default(); rows * columns];
            for col in 0..columns {
                for row in 0..rows {
                    let global_row = row_component.offset + row;
                    let global_col = col_component.offset + col;
                    let delta = sub(
                        self.modal.eigenvalues[global_col],
                        self.modal.eigenvalues[global_row],
                    );
                    let value = block.matrix[row + col * rows];
                    if abs(delta) * maximum >= 1e-3 {
                        matrix[row + col * rows] = divide(value, delta);
                    } else if value.re != 0. || value.im != 0. {
                        close.push((global_row, global_col, delta, value));
                    }
                }
            }
            quotient.push(Block {
                row: block.row,
                col: block.col,
                matrix,
            });
        }
        (quotient, close)
    }
    /// Cache exact linear observations of the same numerical full-density
    /// solution as apply(). No dynamical modes are removed. The supplied
    /// interval fixes only the stable divided-difference classification.
    pub fn prepare_observables(
        &self,
        lap: &Lapack,
        operators: &[ObservableHarmonic],
        maximum_time: f64,
    ) -> Result<ObservablePrepared, String> {
        let started = Instant::now();
        let n = self.n;
        if operators.is_empty() || !maximum_time.is_finite() || maximum_time < 0. {
            return Err("Invalid observable preparation interval/operators".into());
        }
        let rows = operators[0].rows;
        if rows == 0 {
            return Err("Empty observable rows".into());
        }
        let mut grouped = BTreeMap::<i32, Vec<Z>>::new();
        for operator in operators {
            if operator.rows != rows || operator.matrix.len() != rows * n {
                return Err("Inconsistent observable Fourier dimensions".into());
            }
            finite(&operator.matrix, "observable Fourier matrix")?;
            let destination = grouped
                .entry(operator.index)
                .or_insert_with(|| vec![Z::default(); rows * n]);
            for (i, &value) in operator.matrix.iter().enumerate() {
                destination[i] = add(destination[i], value);
            }
        }
        let mut zero_operator = vec![Z::default(); rows * n];
        let mut row_norms = vec![0.; rows];
        let mut modal_rows = BTreeMap::<i32, Vec<Z>>::new();
        for (&index, matrix) in &grouped {
            for row in 0..rows {
                row_norms[row] += (0..n)
                    .map(|col| abs(matrix[row + col * rows]))
                    .fold(0., f64::max);
            }
            for (i, &value) in matrix.iter().enumerate() {
                zero_operator[i] = add(zero_operator[i], value);
            }
            let mut projected = vec![Z::default(); rows * n];
            for component in &self.components {
                let size = component.coordinates.len();
                let input: Vec<_> = component
                    .coordinates
                    .iter()
                    .flat_map(|&col| matrix[col * rows..(col + 1) * rows].iter().copied())
                    .collect();
                let value = lap.multiply_complex(&input, &component.vectors, rows, size, size)?;
                projected[component.offset * rows..(component.offset + size) * rows]
                    .copy_from_slice(&value);
            }
            modal_rows.insert(index, projected);
        }
        let mut effective = modal_rows.clone();
        for (&index, left) in &modal_rows {
            for harmonic in &self.harmonics {
                for sign in [-1_i32, 1] {
                    let target = index
                        .checked_add(
                            harmonic
                                .index
                                .checked_mul(sign)
                                .ok_or("Observable Fourier index overflow")?,
                        )
                        .ok_or("Observable Fourier index overflow")?;
                    let negatives;
                    let blocks = if sign == 1 {
                        &harmonic.transform
                    } else {
                        negatives = harmonic
                            .transform
                            .iter()
                            .map(|b| block_conjugate(b, &self.components, &self.conjugate))
                            .collect::<Vec<_>>();
                        &negatives
                    };
                    let product =
                        observable_right_action(lap, left, rows, blocks, &self.components, n)?;
                    let output = effective
                        .entry(target)
                        .or_insert_with(|| vec![Z::default(); rows * n]);
                    for i in 0..output.len() {
                        output[i] = add(output[i], product[i]);
                    }
                }
            }
        }
        let (quotient, close) = self.mean_quotient(maximum_time);
        let indices: Vec<_> = effective.keys().copied().collect();
        let stacked_rows = rows * indices.len();
        // One batch product evaluates both W_eff*u and (W_eff*Q)*v.
        let mut projection = vec![Z::default(); stacked_rows * 2 * n];
        for (ki, index) in indices.iter().enumerate() {
            let left = &effective[index];
            let right = observable_right_action(lap, left, rows, &quotient, &self.components, n)?;
            for j in 0..n {
                for i in 0..rows {
                    projection[ki * rows + i + j * stacked_rows] = left[i + j * rows];
                    projection[ki * rows + i + (n + j) * stacked_rows] = right[i + j * rows];
                }
            }
        }
        finite(&projection, "projected observables")?;
        Ok(ObservablePrepared {
            rows,
            maximum_time,
            preparation_ms: started.elapsed().as_secs_f64() * 1000.,
            source_id: self.id,
            indices,
            projection,
            quotient,
            close,
            zero_operator,
            row_norms,
        })
    }
    /// tolerance is the density-coordinate l1 budget, as in apply(); each
    /// observable's dimensional error bound is returned separately.
    pub fn apply_observables(
        &self,
        lap: &Lapack,
        observables: &ObservablePrepared,
        initial: &[f64],
        times: &[f64],
        tolerance: f64,
    ) -> Result<ObservableResult, String> {
        let started = Instant::now();
        let n = self.n;
        let rows = observables.rows;
        let count = times.len();
        let maximum = times.iter().copied().fold(0., f64::max);
        if observables.source_id != self.id {
            return Err("Observable cache belongs to another periodic generator".into());
        }
        if times.iter().any(|&t| !t.is_finite() || t < 0.) || maximum > observables.maximum_time {
            return Err("Observable time request exceeds its prepared interval; rebuild the observable cache".into());
        }
        let modal = self.initial_modal(lap, initial)?;
        let bounds = self.checked_bounds(lap, initial, &modal, maximum, tolerance)?;
        let row_error_bounds: Vec<_> = observables
            .row_norms
            .iter()
            .map(|norm| norm * bounds.total_coordinate_l1)
            .collect();
        let first = self.zero_phase_action(lap, &modal)?;
        let second = self.zero_phase_action(lap, &first)?;
        let z0: Vec<_> = (0..n)
            .map(|i| add(sub(modal[i], first[i]), second[i]))
            .collect();
        let q_initial = block_action(lap, &observables.quotient, &self.components, &z0, n, 1)?;
        let mut exponentials = vec![Z::default(); n * count];
        let mut input = vec![Z::default(); 2 * n * count];
        for (ti, &time) in times.iter().enumerate() {
            for i in 0..n {
                let e = if self.conjugate[i] < i {
                    conj(exponentials[self.conjugate[i] + ti * n])
                } else {
                    exponential(self.modal.eigenvalues[i], time)?
                };
                exponentials[i + ti * n] = e;
                let base = mul(e, z0[i]);
                input[i + ti * 2 * n] = sub(base, mul(e, q_initial[i]));
                input[n + i + ti * 2 * n] = base;
            }
        }
        for &(row, col, delta, coefficient) in &observables.close {
            for (ti, &time) in times.iter().enumerate() {
                let value = scale(
                    mul(
                        mul(coefficient, z0[col]),
                        mul(exponentials[row + ti * n], exprel_small(scale(delta, time))),
                    ),
                    time,
                );
                input[row + ti * 2 * n] = add(input[row + ti * 2 * n], value);
            }
        }
        let stacked_rows = rows * observables.indices.len();
        let projected = if count == 0 {
            vec![]
        } else {
            lap.multiply_complex(&observables.projection, &input, stacked_rows, count, 2 * n)?
        };
        finite(&projected, "observable trajectory")?;
        let mut values = vec![vec![0.; rows]; count];
        let mut imaginary_residue = 0_f64;
        let mut initial_error = 0_f64;
        let initial_z: Vec<_> = initial.iter().map(|&re| Z { re, im: 0. }).collect();
        let exact_initial =
            lap.multiply_complex(&observables.zero_operator, &initial_z, rows, 1, n)?;
        for (ti, &time) in times.iter().enumerate() {
            let mut observed = vec![Z::default(); rows];
            for (ki, &index) in observables.indices.iter().enumerate() {
                let p = phase(self.omega, time, index)?;
                for row in 0..rows {
                    observed[row] = add(
                        observed[row],
                        mul(p, projected[ki * rows + row + ti * stacked_rows]),
                    );
                }
            }
            for row in 0..rows {
                imaginary_residue = imaginary_residue.max(observed[row].im.abs());
                if observed[row].im.abs() > tolerance * observables.row_norms[row].max(1.) {
                    return Err(format!(
                        "Observable row {row} has a non-real residue {:e}",
                        observed[row].im
                    ));
                }
                values[ti][row] = if time == 0. {
                    initial_error = initial_error.max(abs(sub(observed[row], exact_initial[row])));
                    exact_initial[row].re
                } else {
                    observed[row].re
                };
            }
        }
        Ok(ObservableResult {
            values,
            bounds,
            row_error_bounds,
            initial_error,
            imaginary_residue,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
        })
    }
    pub fn apply(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
    ) -> Result<ActionResult, String> {
        self.apply_with_tolerance(lap, initial, times, 1e-8)
    }
    pub fn apply_with_tolerance(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
        tolerance: f64,
    ) -> Result<ActionResult, String> {
        let started = Instant::now();
        let n = self.n;
        let count = times.len();
        if initial.len() != n
            || initial.iter().any(|v| !v.is_finite())
            || times.iter().any(|v| !v.is_finite() || *v < 0.)
            || !tolerance.is_finite()
            || tolerance <= 0.
        {
            return Err("Invalid periodic action initial vector/times/tolerance".into());
        }
        let maximum = times.iter().copied().fold(0., f64::max);
        if maximum == 0. {
            return Ok(ActionResult {
                states: vec![initial.to_vec(); count],
                bounds: TailBounds {
                    inverse_initial: 0.,
                    inverse_generator: 0.,
                    oscillatory_convolution: 0.,
                    dyson_remainder: 0.,
                    bounded_fourier_tail: 0.,
                    total_coordinate_l1: 0.,
                    modal_initial_reconstruction: 0.,
                },
                initial_error: 0.,
                imaginary_residue: 0.,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
            });
        }
        let modal = self.initial_modal(lap, initial)?;
        let bounds = self.checked_bounds(lap, initial, &modal, maximum, tolerance)?;
        let first = self.zero_phase_action(lap, &modal)?;
        let second = self.zero_phase_action(lap, &first)?;
        let z0: Vec<_> = (0..n)
            .map(|i| add(sub(modal[i], first[i]), second[i]))
            .collect();
        let mut exponentials = vec![Z::default(); n * count];
        let mut base = vec![Z::default(); n * count];
        for (ti, &time) in times.iter().enumerate() {
            for i in 0..n {
                let e = if self.conjugate[i] < i {
                    conj(exponentials[self.conjugate[i] + ti * n])
                } else {
                    exponential(self.modal.eigenvalues[i], time)?
                };
                exponentials[i + ti * n] = e;
                base[i + ti * n] = mul(e, z0[i]);
            }
        }
        let (quotient, close) = self.mean_quotient(maximum);
        let q_initial = block_action(lap, &quotient, &self.components, &z0, n, 1)?;
        let q_base = block_action(lap, &quotient, &self.components, &base, n, count)?;
        let mut state = base.clone();
        for ti in 0..count {
            for i in 0..n {
                state[i + ti * n] = add(
                    state[i + ti * n],
                    sub(
                        q_base[i + ti * n],
                        mul(exponentials[i + ti * n], q_initial[i]),
                    ),
                );
            }
        }
        for (row, col, delta, coefficient) in close {
            for (ti, &time) in times.iter().enumerate() {
                let value = scale(
                    mul(
                        mul(coefficient, z0[col]),
                        mul(exponentials[row + ti * n], exprel_small(scale(delta, time))),
                    ),
                    time,
                );
                state[row + ti * n] = add(state[row + ti * n], value);
            }
        }
        let mut corrected = state.clone();
        for h in &self.harmonics {
            let transformed = block_action(lap, &h.transform, &self.components, &state, n, count)?;
            for (ti, &time) in times.iter().enumerate() {
                let p = phase(self.omega, time, h.index)?;
                for i in 0..n {
                    corrected[i + ti * n] = add(
                        corrected[i + ti * n],
                        add(
                            mul(p, transformed[i + ti * n]),
                            mul(conj(p), conj(transformed[self.conjugate[i] + ti * n])),
                        ),
                    );
                }
            }
        }
        let physical = physical_action(lap, &self.components, &corrected, n, count)?;
        finite(&physical, "trajectory")?;
        let imaginary_residue = physical.iter().map(|v| v.im.abs()).fold(0., f64::max);
        if imaginary_residue > tolerance {
            return Err(format!(
                "Periodic real-density coordinate residue {imaginary_residue:e} exceeds tolerance"
            ));
        }
        let transformed_initial = add_vectors(&z0, &self.zero_phase_action(lap, &z0)?);
        let rebuilt = physical_action(lap, &self.components, &transformed_initial, n, 1)?;
        let initial_error = rebuilt
            .iter()
            .zip(initial)
            .map(|(&a, &b)| abs(sub(a, Z { re: b, im: 0. })))
            .fold(0., f64::max);
        let states = times
            .iter()
            .enumerate()
            .map(|(ti, &time)| {
                if time == 0. {
                    initial.to_vec()
                } else {
                    physical[ti * n..(ti + 1) * n]
                        .iter()
                        .map(|v| v.re)
                        .collect()
                }
            })
            .collect();
        Ok(ActionResult {
            states,
            bounds,
            initial_error,
            imaginary_residue,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
        })
    }
}
fn add_vectors(a: &[Z], b: &[Z]) -> Vec<Z> {
    a.iter().zip(b).map(|(&a, &b)| add(a, b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decay(rate: f64) -> Sparse {
        vec![
            vec![],
            vec![(0, rate), (1, -rate)],
            vec![(2, -rate / 2.)],
            vec![(3, -rate / 2.)],
        ]
    }
    #[test]
    fn raw_full_defect_matches_complex_jump_oracle_and_observables() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/periodic-raw-complex.json"))
                .unwrap();
        let lap = Lapack::load().unwrap();
        let base: Sparse = serde_json::from_value(fixture["base"].clone()).unwrap();
        let initial: Vec<f64> = serde_json::from_value(fixture["initial"].clone()).unwrap();
        let times: Vec<f64> = serde_json::from_value(fixture["times"].clone()).unwrap();
        let reference: Vec<Vec<f64>> =
            serde_json::from_value(fixture["reference"].clone()).unwrap();
        let omega = fixture["omega"].as_f64().unwrap();
        let end = *times.last().unwrap();
        let harmonic = Harmonic {
            index: 1,
            cos: serde_json::from_value(fixture["cos"].clone()).unwrap(),
            sin: serde_json::from_value(fixture["sin"].clone()).unwrap(),
        };
        let prepared =
            prepare_for_initial(&lap, &base, &[harmonic.clone()], omega, &initial, end, 5e-9)
                .unwrap();
        assert!(
            prepared.diagnostics.raw_modal_components > 0,
            "{:?}",
            prepared.diagnostics.raw_candidate_rejection
        );
        let refined =
            prepare_refined_for_interval(&lap, &base, &[harmonic], omega, end, 5e-9).unwrap();
        assert_eq!(refined.diagnostics.raw_modal_components, 0);
        let raw = prepared
            .apply_with_tolerance(&lap, &initial, &times, 5e-9)
            .unwrap();
        let old = refined
            .apply_with_tolerance(&lap, &initial, &times, 5e-9)
            .unwrap();
        let oracle_error = raw
            .states
            .iter()
            .zip(&reference)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
            .fold(0_f64, f64::max);
        let difference = raw
            .states
            .iter()
            .zip(&old.states)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
            .fold(0_f64, f64::max);
        assert!(oracle_error < raw.bounds.total_coordinate_l1 + 3e-12);
        assert!(oracle_error < 2e-10);
        assert!(difference < 2e-11);
        let n = initial.len();
        let mut identity = vec![Z::default(); n * n];
        for i in 0..n {
            identity[i + i * n].re = 1.;
        }
        let observed = prepared
            .prepare_observables(
                &lap,
                &[ObservableHarmonic {
                    index: 0,
                    rows: n,
                    matrix: identity,
                }],
                end,
            )
            .unwrap();
        let output = prepared
            .apply_observables(&lap, &observed, &initial, &times, 5e-9)
            .unwrap();
        for (a, b) in output
            .values
            .iter()
            .flatten()
            .zip(raw.states.iter().flatten())
        {
            assert!((a - b).abs() < 2e-13);
        }
        assert_eq!(
            output.bounds.total_coordinate_l1,
            raw.bounds.total_coordinate_l1
        );
        assert!(prepared
            .apply_with_tolerance(&lap, &initial, &times, 1e-18)
            .is_err());
    }
    #[test]
    fn raw_candidate_rebuilds_refined_when_its_finite_tail_exceeds_budget() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/periodic-raw-complex-fallback.json"
        ))
        .unwrap();
        let lap = Lapack::load().unwrap();
        let base: Sparse = serde_json::from_value(f["base"].clone()).unwrap();
        let initial: Vec<f64> = serde_json::from_value(f["initial"].clone()).unwrap();
        let times: Vec<f64> = serde_json::from_value(f["times"].clone()).unwrap();
        let reference: Vec<Vec<f64>> = serde_json::from_value(f["reference"].clone()).unwrap();
        let h = Harmonic {
            index: 1,
            cos: serde_json::from_value(f["cos"].clone()).unwrap(),
            sin: serde_json::from_value(f["sin"].clone()).unwrap(),
        };
        let p = prepare_for_initial(
            &lap,
            &base,
            &[h],
            f["omega"].as_f64().unwrap(),
            &initial,
            *times.last().unwrap(),
            5e-9,
        )
        .unwrap();
        assert_eq!(p.diagnostics.raw_modal_components, 0);
        assert!(p
            .diagnostics
            .raw_candidate_rejection
            .as_deref()
            .unwrap()
            .contains("tail"));
        let output = p
            .apply_with_tolerance(&lap, &initial, &times, 5e-9)
            .unwrap();
        let error = output
            .states
            .iter()
            .zip(reference)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
            .fold(0_f64, f64::max);
        assert!(error < 2e-10);
    }
    #[test]
    fn all_density_coordinates_match_exact_modulated_lindblad_decay() {
        let lap = Lapack::load().unwrap();
        let epsilon = 0.03;
        let omega = 1e5;
        for angle in [0_f64, 0.7] {
            let base = decay(1. + epsilon * epsilon);
            let h = Harmonic {
                index: 1,
                cos: decay(2. * epsilon * angle.cos()),
                sin: decay(-2. * epsilon * angle.sin()),
            };
            let prepared = prepare(&lap, &base, &[h], omega).unwrap();
            assert_eq!(prepared.diagnostics.component_dimensions, vec![2, 1, 1]);
            let times = [0., 0.1, 0.7, 5.];
            let result = prepared.apply(&lap, &[0.5, 0.5, 0.5, 0.], &times).unwrap();
            let mut error = 0_f64;
            let mut average_error = 0_f64;
            for (x, &t) in result.states.iter().zip(&times) {
                let integrated = (1. + epsilon * epsilon) * t
                    + 2. * epsilon / omega * ((omega * t + angle).sin() - angle.sin());
                let pe = 0.5 * (-integrated).exp();
                let coherence = 0.5 * (-integrated / 2.).exp();
                for (a, b) in x.iter().zip([1. - pe, pe, coherence, 0.]) {
                    error = error.max((a - b).abs());
                }
                average_error =
                    average_error.max((x[1] - 0.5 * (-(1. + epsilon * epsilon) * t).exp()).abs());
                assert!((x[0] + x[1] - 1.).abs() < 1e-12);
            }
            assert!(
                error <= result.bounds.total_coordinate_l1 + 3e-13,
                "error={error:e}, bound={:?}",
                result.bounds
            );
            assert!(
                average_error > 1e-9,
                "Averaging alone must visibly fail this test"
            );
            assert!(result.imaginary_residue < 1e-14);
            assert_eq!(result.states[0], vec![0.5, 0.5, 0.5, 0.]);
        }
    }
    #[test]
    fn insufficient_numerical_error_budget_fails_instead_of_returning_an_average() {
        let lap = Lapack::load().unwrap();
        let base = decay(1.0009);
        let harmonic = Harmonic {
            index: 1,
            cos: decay(0.06),
            sin: decay(0.),
        };
        let prepared = prepare(&lap, &base, &[harmonic], 50.).unwrap();
        assert!(prepared
            .apply_with_tolerance(&lap, &[0.5, 0.5, 0.5, 0.], &[0., 5.], 1e-15)
            .unwrap_err()
            .contains("tail"));
    }
    #[test]
    fn absent_periodic_terms_preserve_the_constant_full_density_equation() {
        let lap = Lapack::load().unwrap();
        let prepared = prepare(&lap, &decay(1.), &[], 100.).unwrap();
        let result = prepared
            .apply(&lap, &[0., 1., 0., 0.], &[0., 0.2, 1.])
            .unwrap();
        for (x, t) in result.states.iter().zip([0_f64, 0.2, 1.]) {
            assert!((x[1] - (-t).exp()).abs() < 1e-14);
        }
        assert_eq!(result.bounds.total_coordinate_l1, 0.);
    }
    #[test]
    fn signed_harmonic_indices_use_the_same_physical_sine_convention() {
        let lap = Lapack::load().unwrap();
        let base = decay(1.0001);
        let positive = Harmonic {
            index: 1,
            cos: decay(0.01),
            sin: decay(0.017),
        };
        let negative = Harmonic {
            index: -1,
            cos: decay(0.01),
            sin: decay(-0.017),
        };
        let a = prepare(&lap, &base, &[positive], 1e5)
            .unwrap()
            .apply(&lap, &[0.5, 0.5, 0.5, 0.], &[0.1, 0.3])
            .unwrap();
        let b = prepare(&lap, &base, &[negative], 1e5)
            .unwrap()
            .apply(&lap, &[0.5, 0.5, 0.5, 0.], &[0.1, 0.3])
            .unwrap();
        assert_eq!(a.states, b.states);
    }
    #[test]
    fn harmonic_couplings_between_distinct_constant_components_are_retained() {
        let lap = Lapack::load().unwrap();
        let amplitude = 0.03;
        let omega = 1e5;
        // H(t)=amplitude*cos(omega*t)*sigma_x, four independent L0=0
        // coordinates. The periodic term couples population/coherence blocks.
        let cosine = vec![
            vec![(3, amplitude)],
            vec![(3, -amplitude)],
            vec![],
            vec![(0, -2. * amplitude), (1, 2. * amplitude)],
        ];
        let prepared = prepare(
            &lap,
            &vec![vec![]; 4],
            &[Harmonic {
                index: 1,
                cos: cosine,
                sin: vec![vec![]; 4],
            }],
            omega,
        )
        .unwrap();
        assert_eq!(prepared.diagnostics.component_dimensions, vec![1, 1, 1, 1]);
        let times = [0., 0.1, 0.3];
        let result = prepared.apply(&lap, &[0., 1., 0., 0.], &times).unwrap();
        for (x, t) in result.states.iter().zip(times) {
            let angle = amplitude / omega * (omega * t).sin();
            let expected = [
                angle.sin().powi(2),
                angle.cos().powi(2),
                0.,
                -angle.sin() * angle.cos(),
            ];
            for (a, b) in x.iter().zip(expected) {
                assert!((a - b).abs() <= result.bounds.total_coordinate_l1 + 2e-14);
            }
        }
    }
    #[test]
    fn bounded_fourier_optimization_retains_operators_and_rechecks_longer_times() {
        let lap = Lapack::load().unwrap();
        let base = decay(1.0009);
        let harmonics = [
            Harmonic {
                index: 1,
                cos: decay(0.06),
                sin: decay(0.),
            },
            Harmonic {
                index: 3,
                cos: decay(1e-15),
                sin: decay(0.),
            },
        ];
        let all = prepare(&lap, &base, &harmonics, 1e5).unwrap();
        let limited = prepare_for_interval(&lap, &base, &harmonics, 1e5, 20., 1e-8).unwrap();
        assert_eq!(limited.bounded_operators().len(), 1);
        assert_eq!(limited.bounded_operators()[0].index, 3);
        let initial = [0.5, 0.5, 0.5, 0.];
        let times = [0., 0.1, 2., 20.];
        let a = all.apply(&lap, &initial, &times).unwrap();
        let b = limited.apply(&lap, &initial, &times).unwrap();
        let error = a
            .states
            .iter()
            .zip(&b.states)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
            .fold(0., f64::max);
        assert!(error <= b.bounds.bounded_fourier_tail + 1e-14);
        assert!(b.bounds.bounded_fourier_tail > 0.);
        assert!(limited.apply(&lap, &initial, &[0., 1e6]).is_err());
        assert!(all.diagnostics.bounded_harmonics.is_empty());
    }
    #[test]
    fn lindblad_contractivity_bounds_original_omitted_terms_for_changed_intervals() {
        let lap = Lapack::load().unwrap();
        let base = decay(1.);
        let omega = 1e5;
        // Gamma(t)=1+0.06 cos(wt)+1e-12 cos(3wt)>0 is exactly a complete
        // amplitude-damping Lindblad generator at every time.
        let harmonics = [
            Harmonic {
                index: 1,
                cos: decay(0.06),
                sin: decay(0.),
            },
            Harmonic {
                index: 3,
                cos: decay(1e-12),
                sin: decay(0.),
            },
        ];
        let initial = [0.5, 0.5, 0.5, 0.];
        let prepared =
            prepare_lindblad_for_initial(&lap, &base, &harmonics, omega, &initial, 20., 1e-8, 2)
                .unwrap();
        assert_eq!(prepared.bounded_operators().len(), 1);
        let times = [0., 0.137, 2., 20., 30.];
        let result = prepared.apply(&lap, &initial, &times).unwrap();
        for (row, &t) in result.states.iter().zip(&times) {
            let integral = t
                + 0.06 / omega * (omega * t).sin()
                + 1e-12 / (3. * omega) * (3. * omega * t).sin();
            let excited = 0.5 * (-integral).exp();
            let expected = [1. - excited, excited, 0.5 * (-0.5 * integral).exp(), 0.];
            let error = row
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            assert!(error <= result.bounds.total_coordinate_l1 + 1e-13);
        }
        assert!(result.bounds.bounded_fourier_tail > 0.);
        assert!(prepared.apply(&lap, &initial, &[0., 1e6]).is_err());
        let refined = prepared
            .reprepare_for_interval(&lap, &base, &harmonics, 30., 1e-8)
            .unwrap();
        assert_eq!(refined.diagnostics.raw_modal_components, 0);
        let refreshed = refined
            .reprepare_for_interval(&lap, &base, &harmonics, 31., 1e-8)
            .unwrap();
        assert!(refreshed.diagnostics.modal_basis_reused);
        assert!(refreshed
            .reprepare_for_interval(&lap, &decay(1.01), &harmonics, 30., 1e-8)
            .is_err());
    }
    #[test]
    fn observable_fourier_projection_matches_every_full_density_coordinate() {
        let lap = Lapack::load().unwrap();
        let base = decay(1.0009);
        let prepared = prepare(
            &lap,
            &base,
            &[Harmonic {
                index: 1,
                cos: decay(0.06),
                sin: decay(-0.017),
            }],
            1e5,
        )
        .unwrap();
        let rows = 4;
        let mut constant = vec![Z::default(); 16];
        for i in 0..4 {
            constant[i + i * 4].re = 1.;
        }
        // A complex Fourier row deliberately measures both a population and
        // a coherence; retaining signed phase and every S convolution matters.
        let mut positive = vec![Z::default(); 16];
        positive[1].re = 0.3;
        positive[2 + 2 * 4].im = 0.2;
        let negative: Vec<_> = positive.iter().copied().map(conj).collect();
        let operators = [
            ObservableHarmonic {
                index: 0,
                rows,
                matrix: constant,
            },
            ObservableHarmonic {
                index: 2,
                rows,
                matrix: positive,
            },
            ObservableHarmonic {
                index: -2,
                rows,
                matrix: negative,
            },
        ];
        let observed = prepared.prepare_observables(&lap, &operators, 5.).unwrap();
        let times = [0., 0.00001, 0.2, 0.7, 5.];
        let initial = [0.5, 0.5, 0.4, -0.1];
        let full = prepared.apply(&lap, &initial, &times).unwrap();
        let result = prepared
            .apply_observables(&lap, &observed, &initial, &times, 1e-8)
            .unwrap();
        for (ti, &time) in times.iter().enumerate() {
            let mut expected = vec![Z::default(); rows];
            for operator in &operators {
                let p = phase(1e5, time, operator.index).unwrap();
                for row in 0..rows {
                    for col in 0..4 {
                        expected[row] = add(
                            expected[row],
                            scale(
                                mul(p, operator.matrix[row + col * rows]),
                                full.states[ti][col],
                            ),
                        );
                    }
                }
            }
            for row in 0..rows {
                assert!((result.values[ti][row] - expected[row].re).abs() < 2e-14);
            }
        }
        assert!(result.initial_error < 1e-14);
        assert_eq!(
            result.bounds.total_coordinate_l1,
            full.bounds.total_coordinate_l1
        );
        assert!(prepared
            .apply_observables(&lap, &observed, &initial, &[5.1], 1e-8)
            .is_err());
        let other = prepare(&lap, &base, &[], 1e5).unwrap();
        assert!(other
            .apply_observables(&lap, &observed, &initial, &times, 1e-8)
            .unwrap_err()
            .contains("another"));
    }
    #[test]
    fn observable_projection_keeps_cross_component_transform_and_degenerate_frechet_terms() {
        let lap = Lapack::load().unwrap();
        let amplitude = 0.03;
        let omega = 1e5;
        let cosine = vec![
            vec![(3, amplitude)],
            vec![(3, -amplitude)],
            vec![],
            vec![(0, -2. * amplitude), (1, 2. * amplitude)],
        ];
        let prepared = prepare(
            &lap,
            &vec![vec![]; 4],
            &[Harmonic {
                index: 1,
                cos: cosine,
                sin: vec![vec![]; 4],
            }],
            omega,
        )
        .unwrap();
        let mut matrix = vec![Z::default(); 16];
        for i in 0..4 {
            matrix[i + i * 4].re = 1.;
        }
        let observed = prepared
            .prepare_observables(
                &lap,
                &[ObservableHarmonic {
                    index: 0,
                    rows: 4,
                    matrix,
                }],
                5.,
            )
            .unwrap();
        let initial = [0., 1., 0., 0.];
        let times = [0., 0.1, 1., 5.];
        let full = prepared.apply(&lap, &initial, &times).unwrap();
        let result = prepared
            .apply_observables(&lap, &observed, &initial, &times, 1e-8)
            .unwrap();
        for (a, b) in result
            .values
            .iter()
            .flatten()
            .zip(full.states.iter().flatten())
        {
            assert!((a - b).abs() < 2e-14);
        }
    }
}
