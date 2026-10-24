//! Stationary Fourier density for a complete Hermitian-coordinate generator.
//!
//! Arbitrary Liouvillian Fourier coefficients are accepted, including all
//! cross terms of time-dependent collapse operators. Every retained harmonic
//! contains the COMPLETE operator algebra on the supplied invariant Hilbert
//! support. Initial-coordinate reachability never establishes uniqueness.
//! No finite-time normal-form remainder is used to claim an infinite-time mean.
use super::{coordinates, expand_density, floquet, harmonics, positivity, Coordinate, Lapack};
use faer::sparse::{SparseColMat, Triplet};
use harmonics::{FourierCoefficient, FourierDrive, FrequencyBasis, RealSparse as Sparse};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

pub struct Input {
    pub n: usize,
    pub base: Sparse,
    pub drives: Vec<FourierDrive>,
    pub frequencies: Vec<f64>,
    pub initial: Vec<f64>,
    pub hilbert_support: Vec<usize>,
    pub bounded_generator_tail: f64,
}
#[derive(Clone, Copy)]
pub struct Options {
    /// Maximum residual-driven refinement rounds after the central solve.
    pub max_order: i32,
    pub max_coordinates: usize,
    pub max_entries: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_order: 4,
            max_coordinates: 32768,
            max_entries: 32_000_000,
        }
    }
}
#[derive(Debug)]
pub struct Output {
    pub coefficients: Vec<FourierCoefficient>,
    pub frequencies: Vec<f64>,
    /// Map original frequency axes to the exact reduced axes of coefficients.
    pub input_axis_indices: Vec<Vec<i32>>,
    pub diagnostics: Value,
}
pub struct BoundaryEvaluation {
    pub coefficients: Vec<FourierCoefficient>,
    /// Uniform coordinate error allowance added to the existing boundary gate.
    pub roundoff_bound: f64,
    pub diagnostics: Value,
}
pub type BoundaryCallback<'a> = dyn FnMut(&[FourierCoefficient], &[Vec<i32>], &[Vec<i32>]) -> Result<BoundaryEvaluation, String>
    + 'a;
struct Group {
    index: Vec<i32>,
    members: Vec<(Vec<i32>, i32)>,
}
struct Harmonic {
    index: Vec<i32>,
    offset: usize,
    imaginary: bool,
}
struct Candidate {
    coefficients: Vec<FourierCoefficient>,
    outside: Vec<FourierCoefficient>,
    diagnostics: Value,
    trace_error: f64,
    psd_error: f64,
    boundary: f64,
    residual: f64,
}
fn outside_rank(outside: &[FourierCoefficient]) -> Vec<(usize, f64)> {
    let mut ranked: Vec<_> = outside
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                i,
                2. * c
                    .real
                    .iter()
                    .zip(&c.imag)
                    .map(|(&r, &im)| r.hypot(im))
                    .fold(0_f64, f64::max),
            )
        })
        .collect();
    ranked.sort_by(|&(a, x), &(b, y)| {
        y.total_cmp(&x)
            .then_with(|| norm_index(&outside[a].index).cmp(&norm_index(&outside[b].index)))
            .then_with(|| outside[a].index.cmp(&outside[b].index))
    });
    ranked
}
struct Refinement {
    lattice: Vec<Vec<i32>>,
    added: Vec<Vec<i32>>,
    remaining: f64,
}
fn refine_lattice(
    retained: &[Vec<i32>],
    outside: &[FourierCoefficient],
    operator_dimension: usize,
    coordinate_budget: usize,
) -> Result<Refinement, String> {
    let capacity = coordinate_budget / operator_dimension;
    let axes = retained[0].len();
    let mut nodes: BTreeSet<_> = retained.iter().cloned().collect();
    let mut remaining = vec![0.; outside.first().map_or(0, |c| c.real.len())];
    for c in outside {
        for (i, (&r, &im)) in c.real.iter().zip(&c.imag).enumerate() {
            remaining[i] += 2. * r.hypot(im);
        }
    }
    let mut added = Vec::new();
    for (index, _) in outside_rank(outside) {
        if !added.is_empty() && remaining.iter().copied().fold(0_f64, f64::max) < 5e-8 {
            break;
        }
        if nodes.len() + 2 > capacity {
            break;
        }
        let c = &outside[index];
        if nodes.contains(&c.index) {
            continue;
        }
        let negative: Vec<_> = c.index.iter().map(|q| -q).collect();
        nodes.insert(c.index.clone());
        nodes.insert(negative);
        added.push(c.index.clone());
        for (i, (&r, &im)) in c.real.iter().zip(&c.imag).enumerate() {
            remaining[i] = (remaining[i] - 2. * r.hypot(im)).max(0.);
        }
    }
    // Even an exactly polynomial solution needs a genuinely larger trial
    // space for the existing adjacent-solution convergence gate. No repeated
    // solve on an unchanged lattice can manufacture a zero difference.
    if added.is_empty() && nodes.len() + 2 <= capacity && axes > 0 {
        for radius in 1..=64 {
            let mut choices = lattice(axes, radius);
            choices.sort_by(|a, b| norm_index(a).cmp(&norm_index(b)).then_with(|| a.cmp(b)));
            if let Some(q) = choices.into_iter().find(|q| {
                q.iter().any(|&v| v != 0)
                    && canonical(q).is_ok_and(|(_, s)| s > 0)
                    && !nodes.contains(q)
            }) {
                let negative = q.iter().map(|q| -q).collect();
                nodes.insert(q.clone());
                nodes.insert(negative);
                added.push(q);
                break;
            }
        }
    }
    if added.is_empty() {
        return Err(format!("Adaptive Fourier refinement cannot add a complete +/- operator pair within {coordinate_budget} coordinates; retained {} signed harmonics",nodes.len()));
    }
    Ok(Refinement {
        lattice: nodes.into_iter().collect(),
        added,
        remaining: remaining.iter().copied().fold(0_f64, f64::max),
    })
}

fn canonical(index: &[i32]) -> Result<(Vec<i32>, i32), String> {
    let sign = index
        .iter()
        .find(|&&i| i != 0)
        .map_or(1, |&i| if i > 0 { 1 } else { -1 });
    Ok((
        index
            .iter()
            .map(|&i| i.checked_mul(sign).ok_or("Fourier index overflow"))
            .collect::<Result<_, _>>()?,
        sign,
    ))
}
fn norm_index(index: &[i32]) -> u64 {
    index.iter().map(|i| i.unsigned_abs() as u64).sum()
}
fn index_add(a: &[i32], b: &[i32], sign: i32) -> Result<Vec<i32>, String> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            b.checked_mul(sign)
                .and_then(|b| a.checked_add(b))
                .ok_or("Fourier index overflow".into())
        })
        .collect()
}
fn compress(matrix: &mut Sparse) {
    for column in matrix {
        column.sort_by_key(|x| x.0);
        let mut result: Vec<(usize, f64)> = Vec::with_capacity(column.len());
        for &(row, value) in column.iter() {
            if let Some(last) = result.last_mut() {
                if last.0 == row {
                    last.1 += value;
                    continue;
                }
            }
            result.push((row, value));
        }
        result.retain(|&(_, v)| v != 0.);
        *column = result;
    }
}
fn append(target: &mut Sparse, source: &Sparse, factor: f64) {
    for (a, b) in target.iter_mut().zip(source) {
        a.extend(b.iter().map(|&(i, v)| (i, v * factor)));
    }
}
fn validate_operator(matrix: &Sparse, d: usize, n: usize, allowed: &[bool]) -> Result<(), String> {
    if matrix.len() != d {
        return Err("Generator coefficient has the wrong density dimension".into());
    }
    for (j, column) in matrix.iter().enumerate() {
        let mut trace = 0.;
        let mut magnitude = 0.;
        for &(i, value) in column {
            if i >= d || !value.is_finite() {
                return Err("Invalid generator coefficient entry".into());
            }
            if allowed[j] && !allowed[i] && value != 0. {
                return Err(
                    "Supplied Hilbert support is not invariant under the complete generator".into(),
                );
            }
            if i < n {
                trace += value;
                magnitude += value.abs();
            }
        }
        if trace.abs() > 1e-10 * (1. + magnitude) {
            return Err(format!("Generator Fourier coefficient is not trace preserving: column {j}, defect {trace:e}"));
        }
    }
    Ok(())
}
fn frequency(index: &[i32], frequencies: &[f64]) -> f64 {
    let mut hi = 0.;
    let mut lo = 0.;
    for (&i, &w) in index.iter().zip(frequencies) {
        let a = i as f64;
        let p = a * w;
        let tail = a.mul_add(w, -p);
        let sum = hi + p;
        let z = sum - hi;
        lo += (hi - (sum - z)) + (p - z) + tail;
        hi = sum;
    }
    hi + lo
}
fn lattice(axes: usize, order: i32) -> Vec<Vec<i32>> {
    fn visit(
        axis: usize,
        axes: usize,
        budget: i32,
        current: &mut Vec<i32>,
        out: &mut Vec<Vec<i32>>,
    ) {
        if axis == axes {
            out.push(current.clone());
            return;
        }
        for x in -budget..=budget {
            current.push(x);
            visit(axis + 1, axes, budget - x.abs(), current, out);
            current.pop();
        }
    }
    let mut values = Vec::new();
    visit(0, axes, order, &mut Vec::new(), &mut values);
    values.sort();
    values
}
fn groups(indices: &[Vec<i32>], basis: &FrequencyBasis) -> Result<Vec<Group>, String> {
    let mut seen = BTreeSet::new();
    let mut groups: BTreeMap<Vec<i32>, Vec<(Vec<i32>, i32)>> = BTreeMap::new();
    for index in indices {
        if index.len() != basis.drive_indices.len()
            || !seen.insert(index.clone())
            || index.iter().all(|&i| i == 0)
        {
            return Err("Fourier catalog must contain distinct nonzero indices in the original frequency axes".into());
        }
        let mut mapped = vec![0_i32; basis.frequencies.len()];
        for (&coefficient, axis) in index.iter().zip(&basis.drive_indices) {
            for (i, &value) in axis.iter().enumerate() {
                mapped[i] = coefficient
                    .checked_mul(value)
                    .and_then(|v| mapped[i].checked_add(v))
                    .ok_or("Exact frequency-basis index overflow")?;
            }
        }
        let (mapped, sign) = canonical(&mapped)?;
        groups
            .entry(mapped)
            .or_default()
            .push((index.clone(), sign));
    }
    Ok(groups
        .into_iter()
        .map(|(index, members)| Group { index, members })
        .collect())
}
fn fetch(
    group: &Group,
    d: usize,
    provider: &mut impl FnMut(&[i32]) -> Result<FourierDrive, String>,
) -> Result<FourierDrive, String> {
    let mut cos = vec![Vec::new(); d];
    let mut sin = vec![Vec::new(); d];
    for (index, sign) in &group.members {
        let value = provider(index)?;
        if value.index != *index || value.cos.len() != d || value.sin.len() != d {
            return Err("Fourier provider returned a different index or dimension".into());
        }
        append(&mut cos, &value.cos, 1.);
        append(&mut sin, &value.sin, *sign as f64);
    }
    compress(&mut cos);
    compress(&mut sin);
    Ok(FourierDrive {
        index: group.index.clone(),
        cos,
        sin,
    })
}
fn add(
    matrix: &mut Sparse,
    row: usize,
    col: usize,
    value: f64,
    count: &mut usize,
    limit: usize,
) -> Result<(), String> {
    if value != 0. {
        *count = count.checked_add(1).ok_or("Fourier matrix size overflow")?;
        if *count > limit {
            return Err("Full stationary Fourier matrix exceeds the entry budget; no stationary mean was selected".into());
        }
        matrix[col].push((row, value));
    }
    Ok(())
}
fn product(matrix: &Sparse, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.; matrix.len()];
    for (j, column) in matrix.iter().enumerate() {
        for &(i, value) in column {
            y[i] += value * x[j];
        }
    }
    y
}

pub fn solve(lap: &Lapack, input: &Input) -> Result<Output, String> {
    solve_with_options(lap, input, Options::default())
}
pub fn solve_with_options(lap: &Lapack, input: &Input, options: Options) -> Result<Output, String> {
    let catalog: Vec<_> = input
        .drives
        .iter()
        .map(|drive| drive.index.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    solve_internal(
        lap,
        input,
        &catalog,
        |index| {
            let d = input.n * input.n;
            let mut cos = vec![Vec::new(); d];
            let mut sin = vec![Vec::new(); d];
            for drive in input.drives.iter().filter(|drive| drive.index == index) {
                if drive.cos.len() != d || drive.sin.len() != d {
                    return Err("Stored Fourier coefficient has the wrong dimension".into());
                }
                append(&mut cos, &drive.cos, 1.);
                append(&mut sin, &drive.sin, 1.);
            }
            Ok(FourierDrive {
                index: index.to_vec(),
                cos,
                sin,
            })
        },
        options,
        None,
    )
}
pub fn solve_streamed(
    lap: &Lapack,
    input: &Input,
    indices: &[Vec<i32>],
    provider: impl FnMut(&[i32]) -> Result<FourierDrive, String>,
) -> Result<Output, String> {
    solve_streamed_with_options(lap, input, indices, provider, Options::default())
}
pub fn solve_streamed_with_options(
    lap: &Lapack,
    input: &Input,
    indices: &[Vec<i32>],
    provider: impl FnMut(&[i32]) -> Result<FourierDrive, String>,
    options: Options,
) -> Result<Output, String> {
    if !input.drives.is_empty() {
        return Err("Streamed mean input must not also contain stored drives".into());
    }
    solve_internal(lap, input, indices, provider, options, None)
}
pub fn solve_streamed_with_boundary(
    lap: &Lapack,
    input: &Input,
    indices: &[Vec<i32>],
    provider: impl FnMut(&[i32]) -> Result<FourierDrive, String>,
    boundary: &mut BoundaryCallback<'_>,
    options: Options,
) -> Result<Output, String> {
    if !input.drives.is_empty() {
        return Err("Streamed mean input must not also contain stored drives".into());
    }
    solve_internal(lap, input, indices, provider, options, Some(boundary))
}
fn solve_internal(
    lap: &Lapack,
    input: &Input,
    indices: &[Vec<i32>],
    mut provider: impl FnMut(&[i32]) -> Result<FourierDrive, String>,
    options: Options,
    mut boundary: Option<&mut BoundaryCallback<'_>>,
) -> Result<Output, String> {
    let started = Instant::now();
    let n = input.n;
    let d = n.checked_mul(n).ok_or("Density dimension overflow")?;
    if n == 0
        || n > 64
        || input.initial.len() != d
        || input.initial.iter().any(|v| !v.is_finite())
        || options.max_order < 1
        || options.max_order > 12
    {
        return Err("Invalid stationary density dimensions/options".into());
    }
    if !input.bounded_generator_tail.is_finite() || input.bounded_generator_tail != 0. {
        return Err("Infinite-time mean requires ALL original Fourier generator coefficients; a finite-time or unbuilt generator-tail bound cannot certify it".into());
    }
    let basis = harmonics::frequency_basis(&input.frequencies, 16)?;
    if !basis.exact
        || basis.rounded_relations != 0
        || basis.reconstruction_errors.iter().any(|&v| v != 0.)
    {
        return Err(
            "Rounded/ambiguous frequency relations cannot establish an infinite-time mean".into(),
        );
    }
    if basis.frequencies.len() > 3 {
        return Err(
            "Stationary Fourier lattice supports at most three independent frequency axes".into(),
        );
    }
    let mut support = vec![false; n];
    for &i in &input.hilbert_support {
        if i >= n || support[i] {
            return Err("Invalid or duplicate invariant Hilbert support index".into());
        }
        support[i] = true;
    }
    if !support.iter().any(|&v| v) {
        return Err("Empty invariant Hilbert support".into());
    }
    let coords = coordinates(n);
    let operator_indices: Vec<_> = coords
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| match c {
            Coordinate::Population(a) => support[a].then_some(i),
            Coordinate::Real(a, b) | Coordinate::Imag(a, b) => {
                (support[a] && support[b]).then_some(i)
            }
        })
        .collect();
    let mut allowed = vec![false; d];
    for &i in &operator_indices {
        allowed[i] = true;
    }
    if input
        .initial
        .iter()
        .enumerate()
        .any(|(i, &v)| v != 0. && !allowed[i])
    {
        return Err("Initial density lies outside the supplied invariant Hilbert support".into());
    }
    if (input.initial[..n].iter().sum::<f64>() - 1.).abs() > 1e-8
        || positivity(lap, &expand_density(&coords, &input.initial, n), n)? > 1e-10
    {
        return Err("Initial density is not normalized and positive".into());
    }
    let mut base = input.base.clone();
    compress(&mut base);
    validate_operator(&base, d, n, &allowed)?;
    let grouped = groups(indices, &basis)?;
    let mut nonzero = Vec::new();
    for group in grouped {
        if group.index.iter().all(|&i| i == 0) {
            let coefficient = fetch(&group, d, &mut provider)?;
            append(&mut base, &coefficient.cos, 1.);
        } else {
            nonzero.push(group);
        }
    }
    compress(&mut base);
    validate_operator(&base, d, n, &allowed)?;
    let mut previous: Option<Vec<FourierCoefficient>> = None;
    let mut history = Vec::new();
    let mut last_reason = String::new();
    let mut retained = lattice(basis.frequencies.len(), 0);
    for order in 0..=options.max_order {
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
            eprintln!("mean Fourier start K={order}, Hilbert={n}, axes={}, complete operators={}, drives={}",basis.frequencies.len(),operator_indices.len(),nonzero.len());
        }
        let attempt = solve_order(
            lap,
            n,
            &base,
            &nonzero,
            &basis.frequencies,
            &operator_indices,
            &allowed,
            order,
            &mut provider,
            options,
            &basis.drive_indices,
            &mut boundary,
            &retained,
        );
        let candidate = match attempt {
            Ok(value) => value,
            Err(reason) => {
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                    eprintln!("mean Fourier K={order} unresolved: {reason}");
                }
                last_reason = reason;
                history.push(json!({"order":order,"unresolved":last_reason}));
                previous = None;
                let fallback = lattice(basis.frequencies.len(), order + 1);
                if fallback == retained
                    || fallback.len().saturating_mul(operator_indices.len())
                        > options.max_coordinates
                {
                    break;
                }
                retained = fallback;
                continue;
            }
        };
        let difference = if nonzero.is_empty() {
            0.
        } else if let Some(old) = &previous {
            harmonics::uniform_difference(old, &candidate.coefficients)?.max_coordinate
        } else {
            f64::INFINITY
        };
        let mut diagnostic = candidate.diagnostics;
        diagnostic["uniformFourierDifference"] =
            json!(difference.is_finite().then_some(difference));
        history.push(diagnostic.clone());
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
            eprintln!("mean Fourier result {}", diagnostic);
        }
        if difference < 1e-8
            && candidate.residual < 1e-5
            && candidate.boundary < 1e-7
            && candidate.trace_error < 1e-8
            && candidate.psd_error < 1e-8
        {
            diagnostic["elapsedMs"] = json!(started.elapsed().as_secs_f64() * 1000.);
            diagnostic["history"] = json!(history);
            diagnostic["status"] = json!("unique-periodic-density");
            diagnostic["builtGeneratorTail"] = json!(0.);
            diagnostic["frequencyReconstructionErrors"] = json!(basis.reconstruction_errors);
            diagnostic["inputAxisIndices"] = json!(basis.drive_indices);
            return Ok(Output {
                coefficients: candidate.coefficients,
                frequencies: basis.frequencies,
                input_axis_indices: basis.drive_indices,
                diagnostics: diagnostic,
            });
        }
        last_reason=format!("K={order}: uniform difference {difference:e}, full residual {:e}, boundary {:e}, trace {:e}, positivity {:e}",candidate.residual,candidate.boundary,candidate.trace_error,candidate.psd_error);
        if order < options.max_order {
            match refine_lattice(
                &retained,
                &candidate.outside,
                operator_indices.len(),
                options.max_coordinates,
            ) {
                Ok(refinement) => {
                    if let Some(last) = history.last_mut() {
                        last["nextAdaptiveAddedCanonicalIndices"] = json!(refinement.added);
                        last["nextAdaptiveEstimatedUnselectedBoundary"] =
                            json!(refinement.remaining);
                        last["nextAdaptiveSignedHarmonics"] = json!(refinement.lattice.len());
                    }
                    if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                        eprintln!("mean adaptive refinement: added {:?}, next signed {}, estimated remaining {:e}",refinement.added,refinement.lattice.len(),refinement.remaining);
                    }
                    retained = refinement.lattice;
                }
                Err(reason) => {
                    last_reason = format!("{last_reason}; {reason}");
                    break;
                }
            }
        }
        previous = Some(candidate.coefficients);
    }
    Err(format!("Initial-dependent or unresolved stationary Fourier density; no arbitrary mixture selected. {last_reason}"))
}

fn solve_order(
    lap: &Lapack,
    n: usize,
    base: &Sparse,
    groups: &[Group],
    frequencies: &[f64],
    operator_indices: &[usize],
    allowed: &[bool],
    order: i32,
    provider: &mut impl FnMut(&[i32]) -> Result<FourierDrive, String>,
    options: Options,
    input_axis_indices: &[Vec<i32>],
    boundary_callback: &mut Option<&mut BoundaryCallback<'_>>,
    retained_lattice: &[Vec<i32>],
) -> Result<Candidate, String> {
    let started = Instant::now();
    let d = n * n;
    let s = operator_indices.len();
    let signed = retained_lattice.to_vec();
    let size = s
        .checked_mul(signed.len())
        .ok_or("Fourier allocation overflow")?;
    if size > options.max_coordinates {
        return Err(format!("Stationary K={order} requires {size} complete coordinates, beyond the configured budget"));
    }
    let zero = vec![0; frequencies.len()];
    let mut harmonics = vec![Harmonic {
        index: zero.clone(),
        offset: 0,
        imaginary: false,
    }];
    let mut offsets = BTreeMap::from([(zero, 0)]);
    let mut cursor = s;
    for index in &signed {
        if index.iter().all(|&v| v == 0) || canonical(index)?.1 < 0 {
            continue;
        }
        offsets.insert(index.clone(), cursor);
        harmonics.push(Harmonic {
            index: index.clone(),
            offset: cursor,
            imaginary: true,
        });
        cursor += 2 * s;
    }
    let mut position = vec![usize::MAX; d];
    for (i, &global) in operator_indices.iter().enumerate() {
        position[global] = i;
    }
    let mut matrix = vec![Vec::new(); size];
    let mut count = 0;
    for h in &harmonics {
        for part in 0..if h.imaginary { 2 } else { 1 } {
            for (j, &global) in operator_indices.iter().enumerate() {
                for &(row, value) in &base[global] {
                    add(
                        &mut matrix,
                        h.offset + part * s + position[row],
                        h.offset + part * s + j,
                        value,
                        &mut count,
                        options.max_entries,
                    )?;
                }
            }
        }
        if h.imaginary {
            let omega = frequency(&h.index, frequencies);
            if !omega.is_finite() || omega == 0. {
                return Err(
                    "Unresolved zero/overflow frequency on a nonzero retained harmonic".into(),
                );
            }
            for i in 0..s {
                add(
                    &mut matrix,
                    h.offset + i,
                    h.offset + s + i,
                    omega,
                    &mut count,
                    options.max_entries,
                )?;
                add(
                    &mut matrix,
                    h.offset + s + i,
                    h.offset + i,
                    -omega,
                    &mut count,
                    options.max_entries,
                )?;
            }
        }
    }
    let mut required_drives = BTreeSet::new();
    for a in &signed {
        for b in &signed {
            let difference = index_add(a, b, -1)?;
            if difference.iter().any(|&q| q != 0) {
                required_drives.insert(canonical(&difference)?.0);
            }
        }
    }
    let mut matrix_drives = 0;
    for group in groups {
        if !required_drives.contains(&group.index) {
            continue;
        }
        let drive = fetch(group, d, provider)?;
        validate_operator(&drive.cos, d, n, allowed)?;
        validate_operator(&drive.sin, d, n, allowed)?;
        matrix_drives += 1;
        for h in &harmonics {
            for sign in [-1, 1] {
                let source = index_add(&h.index, &drive.index, sign)?;
                let (key, conjugate) = canonical(&source)?;
                let Some(&input) = offsets.get(&key) else {
                    continue;
                };
                let imaginary = source.iter().any(|&v| v != 0);
                for (j, &global) in operator_indices.iter().enumerate() {
                    for &(row, value) in &drive.cos[global] {
                        let row = position[row];
                        add(
                            &mut matrix,
                            h.offset + row,
                            input + j,
                            0.5 * value,
                            &mut count,
                            options.max_entries,
                        )?;
                        if h.imaginary && imaginary {
                            add(
                                &mut matrix,
                                h.offset + s + row,
                                input + s + j,
                                0.5 * value * conjugate as f64,
                                &mut count,
                                options.max_entries,
                            )?;
                        }
                    }
                    for &(row, value) in &drive.sin[global] {
                        let row = position[row];
                        let v = 0.5 * sign as f64 * value;
                        if imaginary {
                            add(
                                &mut matrix,
                                h.offset + row,
                                input + s + j,
                                -v * conjugate as f64,
                                &mut count,
                                options.max_entries,
                            )?;
                        }
                        if h.imaginary {
                            add(
                                &mut matrix,
                                h.offset + s + row,
                                input + j,
                                v,
                                &mut count,
                                options.max_entries,
                            )?;
                        }
                    }
                }
            }
        }
    }
    compress(&mut matrix);
    let construction_ms = started.elapsed().as_secs_f64() * 1000.;
    if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
        eprintln!("mean Fourier K={order} matrix built: n={size}, entries={count}, elapsed={construction_ms:.2}ms");
    }
    let trace = operator_indices
        .iter()
        .position(|&i| i < n)
        .ok_or("No trace coordinate in physical support")?;
    let mut scales = vec![0_f64; size];
    for column in &matrix {
        for &(row, value) in column {
            if row != trace {
                scales[row] = scales[row].max(value.abs());
            }
        }
    }
    scales[trace] = 1.;
    if scales.iter().any(|&v| v == 0.) {
        return Err("Complete stationary operator algebra has a zero equation; initial dependence is unresolved".into());
    }
    let mut entries = Vec::with_capacity(count + n);
    let mut one = 0_f64;
    for (j, column) in matrix.iter().enumerate() {
        let mut norm = 0.;
        for &(row, value) in column {
            if row != trace {
                let v = value / scales[row];
                entries.push(Triplet::new(row, j, v));
                norm += v.abs();
            }
        }
        if j < s && operator_indices[j] < n {
            entries.push(Triplet::new(trace, j, 1.));
            norm += 1.;
        }
        one = one.max(norm);
    }
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(size, size, &entries)
        .map_err(|e| format!("Stationary sparse construction failed: {e:?}"))?;
    drop(entries);
    let before = Instant::now();
    let symbolic =
        faer::sparse::linalg::lu::factorize_symbolic_lu(a.as_ref().symbolic(), Default::default())
            .map_err(|e| format!("Stationary symbolic LU: {e:?}"))?;
    let symbolic_ms = before.elapsed().as_secs_f64() * 1000.;
    let solved = floquet::stationary_linear_solve(&symbolic, a.as_ref(), trace, one)?;
    let remaining = product(&matrix, &solved.values);
    let residual = remaining.iter().map(|v| v.abs()).fold(0_f64, f64::max);
    drop(a);
    drop(matrix);
    let coefficients: Vec<_> = harmonics
        .iter()
        .map(|h| {
            let mut real = vec![0.; d];
            let mut imag = vec![0.; d];
            for (i, &global) in operator_indices.iter().enumerate() {
                real[global] = solved.values[h.offset + i];
                if h.imaginary {
                    imag[global] = solved.values[h.offset + s + i];
                }
            }
            FourierCoefficient {
                index: h.index.clone(),
                real,
                imag,
            }
        })
        .collect();
    let boundary_started = Instant::now();
    if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
        eprintln!(
            "mean Fourier K={order} solve finished, checking {} boundary operators",
            groups.len()
        );
    }
    let evaluated = if let Some(callback) = boundary_callback.as_deref_mut() {
        callback(&coefficients, &signed, input_axis_indices)?
    } else {
        let mut outside: BTreeMap<Vec<i32>, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        // Stream every original generator coefficient. Sum vectors from all paths
        // at each outside harmonic BEFORE taking absolute values, preserving
        // coherent cancellations. No matrix from the provider is kept here.
        for group in groups {
            let drive = fetch(group, d, provider)?;
            validate_operator(&drive.cos, d, n, allowed)?;
            validate_operator(&drive.sin, d, n, allowed)?;
            let boundary = harmonics::boundary_forcing(&[drive], &coefficients, &signed)?;
            for value in boundary.coefficients {
                let target = outside
                    .entry(value.index)
                    .or_insert_with(|| (vec![0.; d], vec![0.; d]));
                for i in 0..d {
                    target.0[i] += value.real[i];
                    target.1[i] += value.imag[i];
                }
            }
        }
        let outside: Vec<_> = outside
            .into_iter()
            .map(|(index, (real, imag))| FourierCoefficient { index, real, imag })
            .collect();
        BoundaryEvaluation {
            coefficients: outside,
            roundoff_bound: 0.,
            diagnostics: json!({"method":"complete streamed Liouvillian coefficients"}),
        }
    };
    if !evaluated.roundoff_bound.is_finite() || evaluated.roundoff_bound < 0. {
        return Err("Invalid boundary roundoff allowance".into());
    }
    let retained: BTreeSet<_> = signed.iter().cloned().collect();
    if evaluated.coefficients.iter().any(|c| {
        c.real.len() != d
            || c.imag.len() != d
            || c.index.len() != frequencies.len()
            || retained.contains(&c.index)
    }) {
        return Err(
            "Boundary callback returned a retained index or inconsistent dimensions".into(),
        );
    }
    let boundary_raw = harmonics::uniform_difference(&evaluated.coefficients, &[])?.max_coordinate;
    let boundary = boundary_raw + evaluated.roundoff_bound;
    let boundary_ms = boundary_started.elapsed().as_secs_f64() * 1000.;
    let mut trace_error = 0.;
    for coefficient in &coefficients {
        let re = coefficient.real[..n].iter().sum::<f64>();
        let im = coefficient.imag[..n].iter().sum::<f64>();
        trace_error += if coefficient.index.iter().all(|&v| v == 0) {
            (re - 1.).abs()
        } else {
            2. * re.hypot(im)
        };
    }
    let coords = coordinates(n);
    let mut psd_error = 0_f64;
    for sample in 0..64 {
        let phase: Vec<_> = (0..frequencies.len())
            .map(|axis| {
                let base = [2, 3, 5][axis];
                let mut index = sample + 1;
                let mut fraction = 1.;
                let mut result = 0.;
                while index > 0 {
                    fraction /= base as f64;
                    result += (index % base) as f64 * fraction;
                    index /= base;
                }
                result * 2. * std::f64::consts::PI
            })
            .collect();
        let mut x = vec![0.; d];
        for c in &coefficients {
            let angle = c
                .index
                .iter()
                .zip(&phase)
                .map(|(&q, &p)| q as f64 * p)
                .sum::<f64>();
            let (si, co) = angle.sin_cos();
            let factor = if c.index.iter().all(|&v| v == 0) {
                1.
            } else {
                2.
            };
            for i in 0..d {
                x[i] += factor * (co * c.real[i] - si * c.imag[i]);
            }
        }
        psd_error = psd_error.max(positivity(lap, &expand_density(&coords, &x, n), n)?);
    }
    let largest:Vec<_>=outside_rank(&evaluated.coefficients).into_iter().take(20).map(|(i,score)|json!({"index":evaluated.coefficients[i].index,"uniformMaxCoordinateContribution":score})).collect();
    let maximum_index = signed.iter().map(|q| norm_index(q)).max().unwrap_or(0);
    let diagnostics = json!({"harmonicOrder":maximum_index,"refinementIteration":order,"latticeStrategy":"residual-adaptive full-operator Galerkin","retainedCanonicalIndices":harmonics.iter().map(|h|&h.index).collect::<Vec<_>>(),"largestOutsideHarmonics":largest,"operatorDimension":size,"physicalOperatorDimension":s,"retainedSignedHarmonics":signed.len(),"matrixDriveCount":matrix_drives,"fullInputDriveCount":groups.len(),"matrixEntriesBeforeCoalescing":count,"conditionEstimate":solved.condition,"residual":residual,"boundaryForcingUniformMax":boundary,"boundaryRawUniformMax":boundary_raw,"boundaryRoundoffBound":evaluated.roundoff_bound,"boundaryMethodDiagnostics":evaluated.diagnostics,"traceUniformError":trace_error,"sampledPositivityError":psd_error,"positivitySamples":64,"constructionMs":construction_ms,"symbolicMs":symbolic_ms,"numericMs":solved.numeric_ms,"conditionMs":solved.condition_ms,"boundaryMs":boundary_ms,"orderElapsedMs":started.elapsed().as_secs_f64()*1000.,"numericWorkspaceReused":solved.workspace_reused,"uniquenessScope":"complete Hermitian operator algebra on supplied invariant Hilbert support, all retained harmonics"});
    Ok(Candidate {
        coefficients,
        outside: evaluated.coefficients,
        diagnostics,
        trace_error,
        psd_error,
        boundary,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore]
    fn export_three_beam_radiative_frames_for_independent_reference() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let source =
            root.join("artifacts/v4/corrected-bath/f2a6-three-beam/repump-mean-request.json");
        let bytes = std::fs::read(&source).unwrap();
        let request: crate::Request = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(request.model_schema.as_deref(), Some("rb87-dipole/2"));
        let model = request.model.unwrap();
        let radiation = request.radiation.unwrap();
        let lap = Lapack::load().unwrap();
        let original = crate::build_generator(&model).unwrap();
        let rho = crate::expand_density(&original.coords, &original.initial, model.dimension);
        let gauges = crate::multitone_frame::resonant_gauges(&model).unwrap();
        let destination = root.join("artifacts/v4/three-beam-corrected");
        std::fs::create_dir_all(&destination).unwrap();
        std::fs::write(destination.join("request.json"), bytes).unwrap();
        let complex = |values: &[crate::C]| json!({"real":values.iter().map(|z|z.re).collect::<Vec<_>>(),"imag":values.iter().map(|z|z.im).collect::<Vec<_>>()});
        for order in [2, 3] {
            let frame = crate::multitone_frame::prepare_with_radiation(
                &lap,
                &model,
                order,
                &gauges,
                Some(&radiation),
            )
            .unwrap();
            let fourier = |map: &std::collections::BTreeMap<Vec<i32>, Vec<crate::C>>| {
                map.iter()
                    .map(|(index, value)| json!({"index":index,"matrix":complex(value)}))
                    .collect::<Vec<_>>()
            };
            let record = json!({"scope":"Complete native H/P/C Fourier frame only; no Liouvillian matrices or stationary result supplied. Independent reference must build its own generator.","modelSchema":"rb87-dipole/2","dimension":model.dimension,"matrixOrder":"row-major","frameHarmonicOrder":order,"frequencies":frame.frequencies,"gauges":frame.gauges,"hamiltonian":complex(&frame.hamiltonian),"modes":fourier(&frame.modes),"physicalModes":fourier(&frame.physical_modes()),"collapse":frame.collapse.iter().map(&fourier).collect::<Vec<_>>(),"initialFrameDensity":complex(&frame.into_frame(&rho)),"unitarityBound":frame.unitarity_bound,"schrodingerResidual":frame.schrodinger_residual,"schrodingerResidualSum":frame.schrodinger_residual_sum,"preparationMs":frame.preparation_ms});
            let path = destination.join(format!("frame-M{order}.json"));
            std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
            println!("exported {}", path.display());
        }
    }
    fn thermal(down: f64, up: f64) -> Sparse {
        vec![
            vec![(0, -up), (1, up)],
            vec![(0, down), (1, -down)],
            vec![(2, -0.5 * (down + up))],
            vec![(3, -0.5 * (down + up))],
        ]
    }
    fn input(base: Sparse) -> Input {
        Input {
            n: 2,
            base,
            drives: vec![],
            frequencies: vec![],
            initial: vec![1., 0., 0., 0.],
            hilbert_support: vec![0, 1],
            bounded_generator_tail: 0.,
        }
    }
    #[test]
    fn unique_constant_mean_uses_full_density_algebra() {
        let lap = Lapack::load().unwrap();
        let out = solve(&lap, &input(thermal(2., 1.))).unwrap();
        assert_eq!(out.coefficients.len(), 1);
        assert!((out.coefficients[0].real[0] - 2. / 3.).abs() < 1e-14);
    }
    #[test]
    fn nonunique_space_never_selects_an_arbitrary_mixture() {
        let lap = Lapack::load().unwrap();
        assert!(solve(&lap, &input(vec![vec![]; 4]))
            .unwrap_err()
            .contains("no arbitrary mixture"));
    }
    #[test]
    fn nonzero_unbuilt_tail_is_rejected_for_infinite_time() {
        let lap = Lapack::load().unwrap();
        let mut model = input(thermal(2., 1.));
        model.bounded_generator_tail = 1e-30;
        assert!(solve(&lap, &model).unwrap_err().contains("ALL original"));
    }
    #[test]
    fn streamed_mixed_harmonics_preserve_all_boundary_terms() {
        let lap = Lapack::load().unwrap();
        let mut model = input(thermal(2., 1.));
        model.frequencies = vec![100., 2_f64.sqrt() * 100.];
        let drive = FourierDrive {
            index: vec![1, -1],
            cos: thermal(0.03, 0.015),
            sin: thermal(-0.02, -0.01),
        };
        let mut calls = 0;
        let streamed = solve_streamed(&lap, &model, &[drive.index.clone()], |_| {
            calls += 1;
            Ok(drive.clone())
        })
        .unwrap();
        assert!(calls >= 2);
        assert_eq!(streamed.diagnostics["fullInputDriveCount"], 1);
        assert!((streamed.coefficients[0].real[0] - 2. / 3.).abs() < 1e-14);
        model.drives.push(drive);
        let stored = solve(&lap, &model).unwrap();
        assert!(
            harmonics::uniform_difference(&stored.coefficients, &streamed.coefficients)
                .unwrap()
                .max_coordinate
                < 1e-14
        );
    }
    #[test]
    fn exact_frequency_reduction_returns_mapping_for_caller_reconstruction() {
        let lap = Lapack::load().unwrap();
        let mut model = input(thermal(2., 1.));
        model.frequencies = vec![100., 200.];
        model.drives.push(FourierDrive {
            index: vec![2, -1],
            cos: thermal(1., 0.5),
            sin: vec![vec![]; 4],
        });
        let out = solve(&lap, &model).unwrap();
        assert_eq!(out.frequencies.len(), 1);
        assert_eq!(out.input_axis_indices, vec![vec![1], vec![2]]);
        assert!((out.coefficients[0].real[1] - 1. / 3.).abs() < 1e-14);
    }
    #[test]
    fn mixed_harmonic_boundary_prevents_false_constant_mean() {
        // Exact Lindblad solution p_e=.3+epsilon*cos((w1-w2)t).
        // The total rate is 3; modulating its up/down partition gives
        // p_e'=up(t)-3*p_e. All instantaneous rates remain positive.
        let lap = Lapack::load().unwrap();
        let mut model = input(thermal(2.1, 0.9));
        model.frequencies = vec![100., 2_f64.sqrt() * 100.];
        let omega = model.frequencies[0] - model.frequencies[1];
        let epsilon = 0.001;
        model.drives.push(FourierDrive {
            index: vec![1, -1],
            cos: thermal(-3. * epsilon, 3. * epsilon),
            sin: thermal(omega * epsilon, -omega * epsilon),
        });
        let out = solve(&lap, &model).unwrap();
        let dc = out.coefficients.iter().find(|c| c.index == [0, 0]).unwrap();
        let side = out
            .coefficients
            .iter()
            .find(|c| c.index == [1, -1])
            .unwrap();
        assert!((dc.real[1] - 0.3).abs() < 1e-14);
        assert!((side.real[1] - epsilon / 2.).abs() < 1e-14);
        assert!(side.imag[1].abs() < 1e-14);
        assert!(
            out.diagnostics["history"][0]["boundaryForcingUniformMax"]
                .as_f64()
                .unwrap()
                > 0.01
        );
        assert!(
            out.diagnostics["retainedCanonicalIndices"]
                .as_array()
                .unwrap()
                .iter()
                .any(|q| q == &json!([1, -1])),
            "The forced mixed harmonic must actually enter the full-operator trial space"
        );
        for phase in [0_f64, 0.7, 2.1, 4.7] {
            let p = dc.real[1] + 2. * (phase.cos() * side.real[1] - phase.sin() * side.imag[1]);
            assert!((p - (0.3 + epsilon * phase.cos())).abs() < 1e-14);
        }
    }
    #[test]
    fn periodic_complex_collapse_mean_matches_independent_monodromy() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/generator-mean-periodic-collapse.json"
        ))
        .unwrap();
        let sparse = |value: &Value| -> Sparse {
            let values = value.as_array().unwrap();
            (0..4)
                .map(|j| {
                    (0..4)
                        .filter_map(|i| {
                            let value = values[i * 4 + j].as_f64().unwrap();
                            (value != 0.).then_some((i, value))
                        })
                        .collect()
                })
                .collect()
        };
        let mut model = input(sparse(&fixture["base"]));
        model.frequencies = vec![fixture["frequency"].as_f64().unwrap()];
        model.drives = fixture["drives"]
            .as_array()
            .unwrap()
            .iter()
            .map(|drive| FourierDrive {
                index: drive["index"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap() as i32)
                    .collect(),
                cos: sparse(&drive["cos"]),
                sin: sparse(&drive["sin"]),
            })
            .collect();
        let lap = Lapack::load().unwrap();
        let out = solve(&lap, &model).unwrap();
        let mut error = 0_f64;
        for (phase, expected) in fixture["oracle"]["phases"]
            .as_array()
            .unwrap()
            .iter()
            .zip(fixture["oracle"]["densityCoordinates"].as_array().unwrap())
        {
            let phase = phase.as_f64().unwrap();
            let mut value = [0.; 4];
            for coefficient in &out.coefficients {
                let k = coefficient.index[0];
                let (sin, cos) = (k as f64 * phase).sin_cos();
                let factor = if k == 0 { 1. } else { 2. };
                for i in 0..4 {
                    value[i] += factor * (cos * coefficient.real[i] - sin * coefficient.imag[i]);
                }
            }
            for i in 0..4 {
                error = error.max((value[i] - expected[i].as_f64().unwrap()).abs());
            }
        }
        assert!(
            error < 5e-11,
            "complete density mismatch {error:e}; {}",
            out.diagnostics
        );
        assert!(out
            .coefficients
            .iter()
            .any(|c| c.index == [2] && c.real.iter().chain(&c.imag).any(|v| v.abs() > 1e-8)));
    }
    #[test]
    #[ignore]
    fn export_default_radiative_frame_for_independent_mean() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let path = root.join("artifacts/v4/bath-independent/default-mean-request.json");
        let request: crate::Request =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let model = request.model.unwrap();
        let radiation = request.radiation.unwrap();
        let lap = Lapack::load().unwrap();
        let original = crate::build_generator(&model).unwrap();
        let rho = crate::expand_density(&original.coords, &original.initial, model.dimension);
        let gauges = crate::multitone_frame::resonant_gauges(&model).unwrap();
        let complex = |values: &[crate::C]| json!({"real":values.iter().map(|z|z.re).collect::<Vec<_>>(),"imag":values.iter().map(|z|z.im).collect::<Vec<_>>()});
        for order in 1..=3 {
            let frame = crate::multitone_frame::prepare_with_radiation(
                &lap,
                &model,
                order,
                &gauges,
                Some(&radiation),
            )
            .unwrap();
            let base_model = frame.base_model(&model, &rho);
            let base = crate::hilbert_floquet::build_density_generator(&lap, &base_model).unwrap();
            let initial = frame.into_frame(&rho);
            let input = Input {
                n: model.dimension,
                base: base.matrix,
                drives: vec![],
                frequencies: frame.frequencies.clone(),
                initial: base.initial,
                hilbert_support: frame.invariant_hilbert_support(&initial),
                bounded_generator_tail: 0.,
            };
            let mean = solve_streamed(&lap, &input, &frame.generator_indices(), |index| {
                let h = frame.generator_harmonic(&lap, index)?;
                Ok(FourierDrive {
                    index: h.index,
                    cos: h.cos,
                    sin: h.sin,
                })
            })
            .unwrap();
            let fourier = |map: &std::collections::BTreeMap<Vec<i32>, Vec<crate::C>>| {
                map.iter()
                    .map(|(index, value)| json!({"index":index,"matrix":complex(value)}))
                    .collect::<Vec<_>>()
            };
            let record = json!({"scope":"Native physical Fourier frame and jumps only: independent reference must build its own complex Liouvillian from these matrices. No native generator matrices exported.","dimension":model.dimension,"matrixOrder":"row-major","frameHarmonicOrder":order,"frequencies":frame.frequencies,"gauges":frame.gauges,"hamiltonian":complex(&frame.hamiltonian),"modes":fourier(&frame.modes),"physicalModes":fourier(&frame.physical_modes()),"collapse":frame.collapse.iter().map(&fourier).collect::<Vec<_>>(),"initialFrameDensity":complex(&initial),"unitarityBound":frame.unitarity_bound,"schrodingerResidual":frame.schrodinger_residual,"schrodingerResidualSum":frame.schrodinger_residual_sum,"nativeMean":{"coefficients":mean.coefficients.iter().map(|c|json!({"index":c.index,"real":c.real,"imag":c.imag})).collect::<Vec<_>>(),"diagnostics":mean.diagnostics}});
            let target = root.join(format!(
                "artifacts/v4/bath-independent/default-frame-M{order}.json"
            ));
            std::fs::write(&target, serde_json::to_vec(&record).unwrap()).unwrap();
            println!("exported {}", target.display());
        }
    }
}
