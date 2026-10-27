//! Retained Fourier domain and the append-only trial space of the finite-time
//! population engine.
//!
//! The engine evolves the exact gauge pencil `E x' = F x` (see `lib.rs`
//! `execute`) in REAL Fourier coordinates of Hermitian-symmetric envelope
//! fields (`real_fourier.rs`), projected on an orthonormal trial basis `Q`.
//! `Q`, `EQ` and `FQ` are stored once, column-major `[rows, m]`; the projected
//! matrices `Q^T E Q` and `Q^T F Q` and the population readout coefficients
//! grow incrementally when columns are appended. Nothing here depends on the
//! output time grid.

use super::*;
use super::b0_inverse::P0Batch;
use super::readout::Readout;
use super::real_fourier::RealFourierMap;
use super::runtime_core::RuntimeCore;

/// Kernel calls process at most this many columns at once.
const KERNEL_BLOCK: usize = 4;
/// A new direction whose component outside the current space is smaller than
/// this fraction of its input norm carries no resolvable information.
const DEFLATION: f64 = 1e-8;

/// E/F pencil, shifted operator and P0 preconditioner of one retained domain,
/// all acting on REAL coordinate columns.
pub(crate) struct DomainOperator {
    modes: Vec<Charge>,
    map: RealFourierMap,
    pencil: OwnedPencil,
    poles: Vec<f64>,
    p0: Vec<P0Batch>,
    rows: usize,
    interleaved: Vec<f64>,
    signed: Vec<C>,
    first: Vec<C>,
    second: Vec<C>,
    pulled: Vec<C>,
}

/// Copies `width` columns of a column-major block into `[row, width]` order.
fn interleave(columns: &[f64], rows: usize, start: usize, width: usize, out: &mut [f64]) {
    for j in 0..width {
        let column = &columns[(start + j) * rows..(start + j + 1) * rows];
        for (i, &value) in column.iter().enumerate() {
            out[i * width + j] = value;
        }
    }
}

/// Writes the real part of an interleaved `[row, width]` block into columns.
fn deinterleave_real(pulled: &[C], rows: usize, start: usize, width: usize, out: &mut [f64]) {
    for j in 0..width {
        let column = &mut out[(start + j) * rows..(start + j + 1) * rows];
        for (i, value) in column.iter_mut().enumerate() {
            *value = pulled[i * width + j].re;
        }
    }
}

impl DomainOperator {
    pub fn new(core: &mut RuntimeCore, modes: &[Charge], poles: &[f64]) -> Result<Self, String> {
        if poles.is_empty() || poles.iter().any(|p| !p.is_finite() || *p <= 0.) {
            return Err("Rational poles must be positive and finite".into());
        }
        let map = core.prepared().real_fourier_map(modes)?;
        let pencil = core.prepared().pencil(modes, modes, KERNEL_BLOCK)?;
        let frequencies = {
            let phases = &core.prepared().primitives().hilbert.source.phases;
            modes.iter().map(|q| phases.frequency(q).map(|f| f.hi + f.lo)).collect::<Result<Vec<_>, _>>()?
        };
        let mut p0 = Vec::with_capacity(poles.len());
        for &pole in poles {
            let shifts: Vec<C> = frequencies.iter().map(|&w| C { re: pole, im: w }).collect();
            p0.push(core.p0_mut().prepare_batch(&shifts, KERNEL_BLOCK)?);
        }
        let rows = map.dimension();
        let length = rows * KERNEL_BLOCK;
        Ok(Self {
            modes: modes.to_vec(),
            map,
            pencil,
            poles: poles.to_vec(),
            p0,
            rows,
            interleaved: vec![0.; length],
            signed: vec![C::default(); length],
            first: vec![C::default(); length],
            second: vec![C::default(); length],
            pulled: vec![C::default(); length],
        })
    }

    pub fn modes(&self) -> &[Charge] {
        &self.modes
    }
    pub fn map(&self) -> &RealFourierMap {
        &self.map
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn poles(&self) -> &[f64] {
        &self.poles
    }

    /// Pushes `width` REAL columns into the signed complex buffer.
    fn push(&mut self, columns: &[f64], start: usize, width: usize) -> Result<usize, String> {
        let length = self.rows * width;
        interleave(columns, self.rows, start, width, &mut self.interleaved[..length]);
        self.map.push_into(&self.interleaved[..length], width, &mut self.signed[..length])?;
        Ok(length)
    }

    /// Signed complex form `[row, k]` of REAL columns (used by the readout).
    pub fn signed_block(&mut self, columns: &[f64], k: usize) -> Result<Vec<C>, String> {
        let mut out = vec![C::default(); self.rows * k];
        let mut packed = vec![0.; self.rows * k];
        interleave(columns, self.rows, 0, k, &mut packed);
        self.map.push_into(&packed, k, &mut out)?;
        Ok(out)
    }

    /// Exact E and F images of `k` REAL columns.
    pub fn images(&mut self, x: &[f64], k: usize, ex: &mut [f64], fx: &mut [f64]) -> Result<(), String> {
        for start in (0..k).step_by(KERNEL_BLOCK) {
            let width = KERNEL_BLOCK.min(k - start);
            let length = self.push(x, start, width)?;
            self.pencil.apply_into(&self.signed[..length], width, &mut self.first[..length], &mut self.second[..length])?;
            self.map.pull_complex_into(&self.first[..length], width, &mut self.pulled[..length])?;
            deinterleave_real(&self.pulled[..length], self.rows, start, width, ex);
            self.map.pull_complex_into(&self.second[..length], width, &mut self.pulled[..length])?;
            deinterleave_real(&self.pulled[..length], self.rows, start, width, fx);
        }
        Ok(())
    }

    /// `(pole_j E - F) x_j` for `k` REAL columns, `pole[j]` indexing the poles.
    pub fn shifted(&mut self, x: &[f64], k: usize, pole: &[usize], out: &mut [f64]) -> Result<(), String> {
        let rows = self.rows;
        for start in (0..k).step_by(KERNEL_BLOCK) {
            let width = KERNEL_BLOCK.min(k - start);
            let length = self.push(x, start, width)?;
            self.pencil.apply_into(&self.signed[..length], width, &mut self.first[..length], &mut self.second[..length])?;
            for i in 0..rows {
                for j in 0..width {
                    let sigma = self.poles[pole[start + j]];
                    let (e, f) = (self.first[i * width + j], self.second[i * width + j]);
                    self.first[i * width + j] = C { re: sigma * e.re - f.re, im: sigma * e.im - f.im };
                }
            }
            self.map.pull_complex_into(&self.first[..length], width, &mut self.pulled[..length])?;
            deinterleave_real(&self.pulled[..length], rows, start, width, out);
        }
        Ok(())
    }

    /// P0 preconditioner of each column for its own pole, an approximate
    /// inverse of `pole_j E - F`.
    ///
    /// Columns that share a pole go through its P0 together, up to
    /// `KERNEL_BLOCK` at a time (one batched application per block).
    pub fn precondition(&mut self, x: &[f64], k: usize, pole: &[usize], out: &mut [f64]) -> Result<(), String> {
        let rows = self.rows;
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&j| pole[j]);
        for group in order.chunk_by(|&a, &b| pole[a] == pole[b]) {
            for block in group.chunks(KERNEL_BLOCK) {
                let width = block.len();
                let length = rows * width;
                for (c, &j) in block.iter().enumerate() {
                    for (i, &value) in x[j * rows..(j + 1) * rows].iter().enumerate() {
                        self.interleaved[i * width + c] = value;
                    }
                }
                self.map.push_into(&self.interleaved[..length], width, &mut self.signed[..length])?;
                self.p0[pole[block[0]]].apply_into(&self.signed[..length], width, &mut self.first[..length])?;
                self.map.pull_complex_into(&self.first[..length], width, &mut self.pulled[..length])?;
                for (c, &j) in block.iter().enumerate() {
                    for (i, value) in out[j * rows..(j + 1) * rows].iter_mut().enumerate() {
                        *value = self.pulled[i * width + c].re;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Removes the trace of every noncentral Fourier quadrature block. The
/// physical trace is `sum_q e^{i q.theta} tr(x_q)`; trial columns therefore
/// keep `tr(x_q) = 0` for every `q != 0`, and the reduced trace test row
/// conserves the central trace exactly.
pub(crate) fn remove_noncentral_traces(map: &RealFourierMap, column: &mut [f64]) {
    let n = map.physical_dimension();
    let d = n * n;
    for block in map.harmonics() {
        if !block.has_imaginary_block {
            continue;
        }
        for offset in [block.offset, block.offset + d] {
            let populations = &mut column[offset..offset + n];
            let mean = populations.iter().sum::<f64>() / n as f64;
            for value in populations {
                *value -= mean;
            }
        }
    }
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Grows an `m x m` column-major projection `A = Q^T W` to `(m+k) x (m+k)`
/// given the stored `Q`, `W` (rows x m) and the new columns `V`, `WV`.
#[allow(clippy::too_many_arguments)]
fn enlarge(
    lapack: &Lapack, old: &[f64], q: &[f64], w: &[f64], v: &[f64], wv: &[f64], rows: usize, m: usize, k: usize,
) -> Result<Vec<f64>, String> {
    let size = m + k;
    let mut out = vec![0.; size * size];
    for j in 0..m {
        out[j * size..j * size + m].copy_from_slice(&old[j * m..(j + 1) * m]);
    }
    // Top-right Q^T WV, bottom-left V^T W, bottom-right V^T WV.
    if m > 0 {
        lapack.gemm(true, false, m, k, rows, 1., q, rows, wv, rows, 0., &mut out[m * size..], size)?;
        lapack.gemm(true, false, k, m, rows, 1., v, rows, w, rows, 0., &mut out[m..], size)?;
    }
    lapack.gemm(true, false, k, k, rows, 1., v, rows, wv, rows, 0., &mut out[m * size + m..], size)?;
    Ok(out)
}

/// Orthonormal trial basis with its exact E/F images and projections.
pub(crate) struct TrialSpace {
    pub op: DomainOperator,
    lapack: Arc<Lapack>,
    n: usize,
    initial: Vec<f64>,
    q: Vec<f64>,
    eq: Vec<f64>,
    fq: Vec<f64>,
    mass: Vec<f64>,
    pencil: Vec<f64>,
    initial_coefficients: Vec<f64>,
    trace_q: Vec<f64>,
    trace_e: Vec<f64>,
    trace_f: Vec<f64>,
    readout: Readout,
    m: usize,
    image_columns: usize,
    pub timing: SpaceTiming,
}

/// Wall time spent inside trial-space updates, in milliseconds.
#[derive(Clone, Debug, Default)]
pub struct SpaceTiming {
    pub orthogonalize_ms: f64,
    pub images_ms: f64,
    pub projections_ms: f64,
    pub readout_ms: f64,
    pub transfer_ms: f64,
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}

impl SpaceTiming {
    /// Adds the times of an earlier (retired) space.
    pub fn absorb(&mut self, earlier: &SpaceTiming) {
        self.orthogonalize_ms += earlier.orthogonalize_ms;
        self.images_ms += earlier.images_ms;
        self.projections_ms += earlier.projections_ms;
        self.readout_ms += earlier.readout_ms;
        self.transfer_ms += earlier.transfer_ms;
    }
}

impl TrialSpace {
    /// New trial space on `modes` whose first column is the initial state.
    pub fn new(core: &mut RuntimeCore, modes: &[Charge], poles: &[f64]) -> Result<Self, String> {
        let op = DomainOperator::new(core, modes, poles)?;
        let initial = core.initial_real(op.map(), modes)?;
        let readout = Readout::new(core.observer(), core.prepared(), modes)?;
        let mut space = Self {
            n: op.map().physical_dimension(),
            op,
            lapack: Arc::clone(core.lapack()),
            initial: initial.clone(),
            q: vec![],
            eq: vec![],
            fq: vec![],
            mass: vec![],
            pencil: vec![],
            initial_coefficients: vec![],
            trace_q: vec![],
            trace_e: vec![],
            trace_f: vec![],
            readout,
            m: 0,
            image_columns: 0,
            timing: SpaceTiming::default(),
        };
        if space.append(initial, 1)?.len() != 1 {
            return Err("The initial state has no representable component".into());
        }
        Ok(space)
    }

    pub fn rows(&self) -> usize {
        self.op.rows()
    }
    pub fn columns(&self) -> usize {
        self.m
    }
    pub fn modes(&self) -> &[Charge] {
        self.op.modes()
    }
    pub fn q(&self) -> &[f64] {
        &self.q
    }
    pub fn eq(&self) -> &[f64] {
        &self.eq
    }
    pub fn fq(&self) -> &[f64] {
        &self.fq
    }
    pub fn mass(&self) -> &[f64] {
        &self.mass
    }
    pub fn pencil(&self) -> &[f64] {
        &self.pencil
    }
    pub fn initial_coefficients(&self) -> &[f64] {
        &self.initial_coefficients
    }
    pub fn traces(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.trace_q, &self.trace_e, &self.trace_f)
    }
    pub fn readout(&self) -> &Readout {
        &self.readout
    }
    pub fn lapack(&self) -> &Arc<Lapack> {
        &self.lapack
    }
    /// Number of columns whose full E/F images were evaluated.
    pub fn image_columns(&self) -> usize {
        self.image_columns
    }
    /// Bytes held by the large `rows x m` arrays.
    pub fn bytes(&self) -> usize {
        (self.q.capacity() + self.eq.capacity() + self.fq.capacity()) * 8 + self.readout.bytes()
    }

    /// Orthogonalizes `k` raw REAL columns (column-major) against the space
    /// and each other, evaluates their exact images and appends the resolved
    /// ones. Returns the input indices of the appended columns, in order.
    pub fn append(&mut self, mut v: Vec<f64>, k: usize) -> Result<Vec<usize>, String> {
        let rows = self.rows();
        let m = self.m;
        if k == 0 || v.len() != rows * k {
            return Err("Trial columns must be a complete [rows, k] block".into());
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite trial column".into());
        }
        let clock = Instant::now();
        let original: Vec<f64> = (0..k).map(|j| norm(&v[j * rows..(j + 1) * rows])).collect();
        for j in 0..k {
            remove_noncentral_traces(self.op.map(), &mut v[j * rows..(j + 1) * rows]);
        }
        // Classical Gram-Schmidt against Q, twice, as two GEMM pairs.
        if m > 0 {
            let mut c = vec![0.; m * k];
            for _ in 0..2 {
                self.lapack.gemm(true, false, m, k, rows, 1., &self.q, rows, &v, rows, 0., &mut c, m)?;
                self.lapack.gemm(false, false, rows, k, m, -1., &self.q, rows, &c, m, 1., &mut v, rows)?;
            }
        }
        // Then among the new columns, dropping unresolved directions.
        let mut kept: Vec<f64> = Vec::with_capacity(rows * k);
        let mut sources = Vec::with_capacity(k);
        for j in 0..k {
            let mut column = v[j * rows..(j + 1) * rows].to_vec();
            let count = kept.len() / rows;
            for _ in 0..2 {
                for i in 0..count {
                    let previous = &kept[i * rows..(i + 1) * rows];
                    let c = dot(previous, &column);
                    for (x, p) in column.iter_mut().zip(previous) {
                        *x -= c * p;
                    }
                }
            }
            let length = norm(&column);
            if !length.is_finite() {
                return Err("Nonfinite trial orthogonalization".into());
            }
            if length <= DEFLATION * original[j] || length == 0. {
                continue;
            }
            if m + count + 1 > rows {
                break;
            }
            for x in column.iter_mut() {
                *x /= length;
            }
            kept.extend(column);
            sources.push(j);
        }
        let added = sources.len();
        self.timing.orthogonalize_ms += ms(clock);
        if added == 0 {
            return Ok(sources);
        }
        let clock = Instant::now();
        let mut ek = vec![0.; rows * added];
        let mut fk = vec![0.; rows * added];
        self.op.images(&kept, added, &mut ek, &mut fk)?;
        self.timing.images_ms += ms(clock);
        let clock = Instant::now();
        if ek.iter().chain(&fk).any(|x| !x.is_finite()) {
            return Err("Nonfinite E/F image of a trial column".into());
        }
        self.image_columns += added;
        let mass = enlarge(&self.lapack, &self.mass, &self.q, &self.eq, &kept, &ek, rows, m, added)?;
        let pencil = enlarge(&self.lapack, &self.pencil, &self.q, &self.fq, &kept, &fk, rows, m, added)?;
        let mut coefficients = vec![0.; added];
        self.lapack.gemm(true, false, added, 1, rows, 1., &kept, rows, &self.initial, rows, 0., &mut coefficients, added)?;
        let n = self.n;
        for j in 0..added {
            self.trace_q.push(kept[j * rows..j * rows + n].iter().sum());
            self.trace_e.push(ek[j * rows..j * rows + n].iter().sum());
            self.trace_f.push(fk[j * rows..j * rows + n].iter().sum());
        }
        self.timing.projections_ms += ms(clock);
        let clock = Instant::now();
        let signed = self.op.signed_block(&kept, added)?;
        self.readout.append(&signed, added)?;
        self.timing.readout_ms += ms(clock);
        self.initial_coefficients.extend(coefficients);
        self.mass = mass;
        self.pencil = pencil;
        self.q.extend(kept);
        self.eq.extend(ek);
        self.fq.extend(fk);
        self.m += added;
        Ok(sources)
    }

    /// Residual operator `R = F Q - E Q G` for a reduced generator `G`
    /// (m x m, column-major, unrotated). The physical residual of the reduced
    /// solution `x(t) = Q a(t)` with `a' = G a` is `R a(t)`.
    pub fn residual(&self, generator: &[f64]) -> Result<Vec<f64>, String> {
        let rows = self.rows();
        let m = self.m;
        if generator.len() != m * m {
            return Err("Reduced generator does not match the trial space".into());
        }
        let mut r = self.fq.clone();
        self.lapack.gemm(false, false, rows, m, m, -1., &self.eq, rows, generator, m, 1., &mut r, rows)?;
        Ok(r)
    }

    /// Moves the space into a larger domain. Old columns, their projections
    /// and readout coefficients are reused exactly; only the new image rows on
    /// the added modes are evaluated.
    pub fn transfer(self, core: &mut RuntimeCore, new_modes: &[Charge]) -> Result<Self, String> {
        let old_modes = self.op.modes().to_vec();
        let old: BTreeSet<&Charge> = old_modes.iter().collect();
        if !old.iter().all(|q| new_modes.contains(q)) {
            return Err("A domain transfer must keep every retained mode".into());
        }
        let shell: Vec<Charge> = new_modes.iter().filter(|q| !old.contains(q)).cloned().collect();
        if shell.is_empty() {
            return Ok(self);
        }
        let clock = Instant::now();
        let poles = self.op.poles().to_vec();
        let op = DomainOperator::new(core, new_modes, &poles)?;
        let new_rows = op.rows();
        let old_rows = self.rows();
        let m = self.m;
        let d = self.n * self.n;
        // Copy every retained harmonic block into the new REAL layout.
        let old_offsets: BTreeMap<&Charge, usize> =
            self.op.map().harmonics().iter().map(|b| (&b.index, b.offset)).collect();
        let lift = |columns: &[f64], count: usize| -> Vec<f64> {
            let mut out = vec![0.; new_rows * count];
            for block in op.map().harmonics() {
                let Some(&from) = old_offsets.get(&block.index) else { continue };
                let width = if block.has_imaginary_block { 2 * d } else { d };
                for j in 0..count {
                    out[j * new_rows + block.offset..j * new_rows + block.offset + width]
                        .copy_from_slice(&columns[j * old_rows + from..j * old_rows + from + width]);
                }
            }
            out
        };
        let q = lift(&self.q, m);
        let mut eq = lift(&self.eq, m);
        let mut fq = lift(&self.fq, m);
        let initial = lift(&self.initial, 1);
        // Exact shell rows of EQ and FQ: the pencil from the old modes onto
        // the added modes. Q itself is zero on the added modes, so the
        // projections Q^T E Q and Q^T F Q are unchanged.
        let mut shell_pencil = core.prepared().pencil(&old_modes, &shell, KERNEL_BLOCK)?;
        let new_index: BTreeMap<&Charge, usize> = new_modes.iter().enumerate().map(|(i, q)| (q, i)).collect();
        let shell_set: BTreeSet<&Charge> = shell.iter().collect();
        let shell_blocks: Vec<_> =
            op.map().harmonics().iter().filter(|b| shell_set.contains(&b.index)).cloned().collect();
        let shell_rows = shell.len() * d;
        let mut packed = vec![0.; old_rows * KERNEL_BLOCK];
        let mut signed = vec![C::default(); old_rows * KERNEL_BLOCK];
        let mut e = vec![C::default(); shell_rows * KERNEL_BLOCK];
        let mut f = vec![C::default(); shell_rows * KERNEL_BLOCK];
        let mut assembled = vec![C::default(); new_rows * KERNEL_BLOCK];
        let mut pulled = vec![C::default(); new_rows * KERNEL_BLOCK];
        for start in (0..m).step_by(KERNEL_BLOCK) {
            let width = KERNEL_BLOCK.min(m - start);
            interleave(&self.q, old_rows, start, width, &mut packed[..old_rows * width]);
            self.op.map().push_into(&packed[..old_rows * width], width, &mut signed[..old_rows * width])?;
            shell_pencil.apply_into(
                &signed[..old_rows * width],
                width,
                &mut e[..shell_rows * width],
                &mut f[..shell_rows * width],
            )?;
            for (values, output) in [(&e, &mut eq), (&f, &mut fq)] {
                assembled[..new_rows * width].fill(C::default());
                for (source, q) in shell.iter().enumerate() {
                    let target = new_index[q];
                    assembled[target * d * width..(target + 1) * d * width]
                        .copy_from_slice(&values[source * d * width..(source + 1) * d * width]);
                }
                op.map().pull_complex_into(&assembled[..new_rows * width], width, &mut pulled[..new_rows * width])?;
                for block in &shell_blocks {
                    let count = if block.has_imaginary_block { 2 * d } else { d };
                    for row in block.offset..block.offset + count {
                        for j in 0..width {
                            output[(start + j) * new_rows + row] = pulled[row * width + j].re;
                        }
                    }
                }
            }
        }
        if eq.iter().chain(&fq).any(|x| !x.is_finite()) {
            return Err("Nonfinite transferred shell image".into());
        }
        let readout = self.readout.transferred(new_modes)?;
        let mut timing = self.timing.clone();
        timing.transfer_ms += ms(clock);
        Ok(Self {
            op,
            lapack: self.lapack,
            n: self.n,
            initial,
            q,
            eq,
            fq,
            mass: self.mass,
            pencil: self.pencil,
            initial_coefficients: self.initial_coefficients,
            trace_q: self.trace_q,
            trace_e: self.trace_e,
            trace_f: self.trace_f,
            readout,
            m,
            image_columns: self.image_columns,
            timing,
        })
    }
}
