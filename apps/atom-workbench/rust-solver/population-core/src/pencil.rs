//! Complete E/F pencil action on a finite Fourier domain.
//!
//! For a signed Fourier field x (modes q, one n x n matrix per mode and
//! right-hand side) the pencil computes
//!   J   = x + [S, x] + [S, [S, x]] / 2,      z = K vec(J_EE),
//!   E x = J + embed_GG(z),
//!   F x = NL * J + J * NR + R vec(J_EE) + L_GG z + L_OG z - i omega E x,
//! where `*` is the convolution over charges, (A * X)_q = sum_p A_p X_{q-p}.
//! The charge pairs (source mode, target mode) of every coefficient are its
//! edges. Hilbert stages are evaluated per target mode: for each ground or
//! excited output part, the nonzero blocks of all incoming coefficients are
//! concatenated along the contraction and the matching source rows/columns
//! stacked, so one ZGEMM accumulates the whole convolution into the target.
//! Thin stages (K, R, L_GG, L_OG) use one dense GEMM or a CSR pass per charge.
//!
//! Field layout ("mode blocks"): mode m stores the matrices of all rhs
//! columns as one column-major n x (rhs*n) block whose column index is
//! b + rhs*j, so entry (i, j) of column b sits at i + n*(b + rhs*j). Read with
//! leading dimension n the block is the column-interleaved [X_0 | X_1 | ...]
//! (left products A*X); read with leading dimension n*rhs it is the vertical
//! stack of the X_b (right products X*A).
//!
//! Hermitian half: when every mode set is closed under q -> -q and the input
//! satisfies x_{-q} = x_q^dagger exactly, only the canonical modes (central
//! mode and one of each +-q pair) of every intermediate and of the output are
//! computed; their partners are mirrored. The prepared coefficients preserve
//! this symmetry up to the bounded preparation defect (see DomainDiagnostics),
//! so the mirrored modes agree with a full evaluation to that defect.
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use crate::lapack::{ComplexGemm, Lapack};
use crate::parallel::{Execution, MINIMUM_WORK_PER_TASK};
use crate::preparation::{Charge, FourierMatrices};
use crate::{Coefficient, Finite, C};

const ZERO: C = C { re: 0., im: 0. };
const ONE: C = C { re: 1., im: 0. };
/// Task-sizing weights in complex multiply-add units (scheduling only):
/// one element of a memory-bound conversion pass, and one scalar CSR term.
const COPY_WORK: usize = 16;
const CSR_WORK: usize = 4;

// ---------------------------------------------------------------------------
// Stage profile
// ---------------------------------------------------------------------------

pub const STAGE_COUNT: usize = 9;
pub const STAGE_NAMES: [&str; STAGE_COUNT] = [
    "input",
    "dx=[S,x]",
    "J=x+dx+[S,dx]/2",
    "z=K vec(J_EE)",
    "E assembly",
    "F left NL*J",
    "F right J*NR",
    "F thin+transport",
    "output",
];

/// Wall-clock time spent in each pencil stage, summed over applications.
/// Diagnostic only; it never influences arithmetic or scheduling.
#[derive(Clone, Debug, Default)]
pub struct StageProfile {
    pub applications: u64,
    pub nanos: [u64; STAGE_COUNT],
}

pub(crate) struct StageClock {
    last: Instant,
}

impl StageClock {
    pub(crate) fn start() -> Self {
        Self { last: Instant::now() }
    }

    pub(crate) fn mark(&mut self, profile: &mut StageProfile, stage: usize) {
        let now = Instant::now();
        profile.nanos[stage] += (now - self.last).as_nanos() as u64;
        self.last = now;
    }
}

// ---------------------------------------------------------------------------
// GEMM backend
// ---------------------------------------------------------------------------

/// Column-major C = alpha*A*B + beta*C. `Reference` is a plain triple loop
/// used by small validation fixtures that run without BLAS.
#[derive(Clone, Copy)]
pub(crate) enum Gemm<'a> {
    Blas(ComplexGemm<'a>),
    Reference,
}

impl Gemm<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn multiply(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: C,
        a: &[C],
        lda: usize,
        b: &[C],
        ldb: usize,
        beta: C,
        c: &mut [C],
        ldc: usize,
    ) -> Result<(), String> {
        match self {
            Gemm::Blas(blas) => blas.multiply(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
            Gemm::Reference => {
                for column in 0..n {
                    for row in 0..m {
                        let mut sum = ZERO;
                        for p in 0..k {
                            sum = add(sum, mul(a[row + lda * p], b[p + ldb * column]));
                        }
                        let at = row + ldc * column;
                        let previous = if beta.re == 0. && beta.im == 0. { ZERO } else { mul(beta, c[at]) };
                        c[at] = add(previous, mul(alpha, sum));
                    }
                }
                Ok(())
            }
        }
    }
}

#[inline]
fn add(a: C, b: C) -> C {
    C { re: a.re + b.re, im: a.im + b.im }
}

#[inline]
fn mul(a: C, b: C) -> C {
    C { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }
}

#[inline]
fn conj(a: C) -> C {
    C { re: a.re, im: -a.im }
}

#[inline]
fn add_slice(target: &mut [C], source: &[C]) {
    for (t, s) in target.iter_mut().zip(source) {
        t.re += s.re;
        t.im += s.im;
    }
}

/// Grow `buffer` to at least `length` entries without growth slack (never
/// shrinks; existing entries are kept, new ones are zero).
fn ensure(buffer: &mut Vec<C>, length: usize) {
    if buffer.len() < length {
        buffer.reserve_exact(length - buffer.len());
        buffer.resize(length, ZERO);
    }
}

fn consecutive(indices: &[usize]) -> bool {
    indices.windows(2).all(|w| w[1] == w[0] + 1)
}

// ---------------------------------------------------------------------------
// Coefficient library (shared by every domain of one request)
// ---------------------------------------------------------------------------

/// One block product of an n x n coefficient: `output` rows (left product)
/// or columns (right product) of the result, contracting over `input`.
#[derive(Clone, Debug)]
struct Block {
    output: Range<usize>,
    input: Range<usize>,
}

/// S, NL or NR at one charge, with its nonzero ground/excited block pattern.
pub(crate) struct HilbertCoefficient {
    /// Column-major n x n.
    dense: Vec<C>,
    /// Blocks of A*X: for each output row part, the contracted columns.
    left: Vec<Block>,
    /// Blocks of X*A: for each output column part, the contracted rows.
    right: Vec<Block>,
    left_macs: usize,
    right_macs: usize,
}

impl HilbertCoefficient {
    /// `matrix` is row-major n x n; `ground` is the size of the leading
    /// ground block. Exactly-zero blocks are skipped, nothing else.
    pub(crate) fn new(matrix: &[C], n: usize, ground: usize) -> Self {
        let nonzero = |z: &C| z.re != 0. || z.im != 0.;
        let mut dense = vec![ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                dense[i + n * j] = matrix[i * n + j];
            }
        }
        let parts = [0..ground, ground..n];
        let occupied = |rows: &Range<usize>, cols: &Range<usize>| {
            rows.clone().any(|i| cols.clone().any(|j| nonzero(&matrix[i * n + j])))
        };
        let contraction = |used: Vec<&Range<usize>>| -> Option<Range<usize>> {
            match used.as_slice() {
                [] => None,
                [one] => Some((*one).clone()),
                _ => Some(0..n),
            }
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        for output in &parts {
            let used = parts.iter().filter(|input| occupied(output, input)).collect();
            if let Some(input) = contraction(used) {
                left.push(Block { output: output.clone(), input });
            }
            let used = parts.iter().filter(|input| occupied(input, output)).collect();
            if let Some(input) = contraction(used) {
                right.push(Block { output: output.clone(), input });
            }
        }
        let macs = |blocks: &[Block]| blocks.iter().map(|b| b.output.len() * b.input.len() * n).sum();
        Self { dense, left_macs: macs(&left), right_macs: macs(&right), left, right }
    }
}

/// K, R, L_GG or L_OG at one charge.
pub(crate) struct ThinCoefficient {
    rows: usize,
    cols: usize,
    sparse: Coefficient,
    /// Column-major copy when at least 15% of the entries are nonzero.
    dense: Option<Vec<C>>,
}

impl ThinCoefficient {
    /// `matrix` is row-major rows x cols; `column_order[c]` is the internal
    /// column of original column c.
    pub(crate) fn new(matrix: &[C], rows: usize, cols: usize, column_order: Option<&[usize]>) -> Self {
        let mut permuted = vec![ZERO; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let to = column_order.map_or(c, |order| order[c]);
                permuted[r * cols + to] = matrix[r * cols + c];
            }
        }
        let mut sparse = Coefficient { rows, cols, row: vec![0], col: Vec::new(), value: Vec::new() };
        for r in 0..rows {
            for c in 0..cols {
                let z = permuted[r * cols + c];
                if z.re != 0. || z.im != 0. {
                    sparse.col.push(c);
                    sparse.value.push(z);
                }
            }
            sparse.row.push(sparse.value.len());
        }
        let dense = (rows.min(cols) >= 8 && sparse.value.len() as f64 >= 0.15 * (rows * cols) as f64).then(|| {
            let mut dense = vec![ZERO; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    dense[r + rows * c] = permuted[r * cols + c];
                }
            }
            dense
        });
        Self { rows, cols, sparse, dense }
    }

    fn bytes(&self) -> usize {
        (self.sparse.row.len() + self.sparse.col.len()) * std::mem::size_of::<usize>()
            + (self.sparse.value.len() + self.dense.as_ref().map_or(0, Vec::len)) * std::mem::size_of::<C>()
    }
}

/// All prepared coefficients of one request, shared by every domain plan.
#[derive(Default)]
pub(crate) struct Library {
    pub(crate) hilbert: Vec<HilbertCoefficient>,
    pub(crate) thin: Vec<ThinCoefficient>,
}

impl Library {
    pub(crate) fn bytes(&self) -> usize {
        self.hilbert.iter().map(|a| a.dense.len() * std::mem::size_of::<C>()).sum::<usize>()
            + self.thin.iter().map(ThinCoefficient::bytes).sum::<usize>()
    }
}

/// Charge -> library index for one coefficient family.
pub(crate) type Family = BTreeMap<Charge, usize>;

// ---------------------------------------------------------------------------
// Mode orders and edges
// ---------------------------------------------------------------------------

/// Internal order of one set of Fourier modes. When the set is closed under
/// q -> -q the internal order is [central] [positive q] [matching -q], so
/// the canonical modes are a prefix and partners differ by a fixed offset.
#[derive(Clone, Debug)]
pub(crate) struct ModeOrder {
    charges: Vec<Charge>,
    /// Caller index of each internal mode.
    external: Vec<usize>,
    /// Internal index of each caller mode.
    internal: Vec<usize>,
    pairing: Option<Pairing>,
}

#[derive(Clone, Copy, Debug)]
struct Pairing {
    /// Modes [0, central) are self-conjugate (the zero charge).
    central: usize,
    /// Modes [0, canonical) are computed in Hermitian-half mode.
    canonical: usize,
}

fn negate(q: &[i32]) -> Option<Charge> {
    q.iter().map(|&x| x.checked_neg()).collect()
}

fn is_positive(q: &[i32]) -> bool {
    q.iter().copied().find(|&x| x != 0).is_some_and(|x| x > 0)
}

impl ModeOrder {
    pub(crate) fn new(modes: &[Charge]) -> Self {
        let position: BTreeMap<&Charge, usize> = modes.iter().enumerate().map(|(i, q)| (q, i)).collect();
        let partner = |q: &Charge| negate(q).and_then(|p| position.get(&p).copied());
        let closed = modes.iter().all(|q| partner(q).is_some());
        let (external, pairing) = if closed {
            let central: Vec<usize> = (0..modes.len()).filter(|&i| modes[i].iter().all(|&x| x == 0)).collect();
            let positive: Vec<usize> = (0..modes.len()).filter(|&i| is_positive(&modes[i])).collect();
            let negative: Vec<usize> = positive.iter().map(|&i| partner(&modes[i]).unwrap()).collect();
            let pairing = Pairing { central: central.len(), canonical: central.len() + positive.len() };
            ([central, positive, negative].concat(), Some(pairing))
        } else {
            ((0..modes.len()).collect(), None)
        };
        let mut internal = vec![0; modes.len()];
        for (i, &e) in external.iter().enumerate() {
            internal[e] = i;
        }
        Self { charges: modes.to_vec(), external, internal, pairing }
    }

    /// The modes in the caller's order, without Hermitian pairing.
    pub(crate) fn given(modes: &[Charge]) -> Self {
        let identity: Vec<usize> = (0..modes.len()).collect();
        Self { charges: modes.to_vec(), external: identity.clone(), internal: identity, pairing: None }
    }

    pub(crate) fn len(&self) -> usize {
        self.charges.len()
    }

    fn charge(&self, internal: usize) -> &Charge {
        &self.charges[self.external[internal]]
    }

    /// Number of leading internal modes that are computed.
    fn computed(&self, half: bool) -> usize {
        match (half, self.pairing) {
            (true, Some(p)) => p.canonical,
            _ => self.len(),
        }
    }

    /// Internal modes whose value is the conjugate transpose of the canonical
    /// mode `index - offset`: (first mirrored mode, offset).
    fn mirrored(&self) -> Option<(usize, usize)> {
        self.pairing.map(|p| (p.canonical, p.canonical - p.central))
    }

    fn index(&self) -> BTreeMap<&Charge, usize> {
        (0..self.len()).map(|i| (self.charge(i), i)).collect()
    }
}

/// Edges (source, target) of one coefficient, sorted by target.
struct Link {
    coefficient: usize,
    sources: Vec<usize>,
    targets: Vec<usize>,
}

impl Link {
    /// Edges whose target lies in `targets`.
    fn within(&self, targets: &Range<usize>) -> Range<usize> {
        self.targets.partition_point(|&t| t < targets.start)..self.targets.partition_point(|&t| t < targets.end)
    }
}

fn add_charge(a: &[i32], b: &[i32]) -> Result<Charge, String> {
    a.iter().zip(b).map(|(&x, &y)| x.checked_add(y).ok_or_else(|| "Fourier charge overflow".to_string())).collect()
}

fn links(family: &Family, input: &ModeOrder, output: &ModeOrder) -> Result<Vec<Link>, String> {
    let index = output.index();
    let mut result = Vec::new();
    for (charge, &coefficient) in family {
        let mut edges = Vec::new();
        for source in 0..input.len() {
            if let Some(&target) = index.get(&add_charge(input.charge(source), charge)?) {
                edges.push((target, source));
            }
        }
        if !edges.is_empty() {
            edges.sort_unstable();
            result.push(Link {
                coefficient,
                sources: edges.iter().map(|e| e.1).collect(),
                targets: edges.iter().map(|e| e.0).collect(),
            });
        }
    }
    Ok(result)
}

/// Edges of one Hilbert stage grouped by target: target t receives
/// (coefficient, source) pairs k in offsets[t]..offsets[t+1], in charge order.
struct Adjacency {
    offsets: Vec<usize>,
    coefficient: Vec<usize>,
    source: Vec<usize>,
}

impl Adjacency {
    fn new(links: &[Link], targets: usize) -> Self {
        let mut offsets = vec![0; targets + 1];
        for link in links {
            for &t in &link.targets {
                offsets[t + 1] += 1;
            }
        }
        for t in 0..targets {
            offsets[t + 1] += offsets[t];
        }
        let mut next = offsets.clone();
        let mut coefficient = vec![0; offsets[targets]];
        let mut source = vec![0; offsets[targets]];
        for link in links {
            for (&s, &t) in link.sources.iter().zip(&link.targets) {
                coefficient[next[t]] = link.coefficient;
                source[next[t]] = s;
                next[t] += 1;
            }
        }
        Self { offsets, coefficient, source }
    }

    fn edges(&self, target: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let range = self.offsets[target]..self.offsets[target + 1];
        self.coefficient[range.clone()].iter().copied().zip(self.source[range].iter().copied())
    }

    fn widest(&self) -> usize {
        self.offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }
}

/// Same-charge identity embedding, (source, target) sorted by target.
fn embedding(input: &ModeOrder, output: &ModeOrder) -> Vec<(usize, usize)> {
    let index = output.index();
    let mut pairs: Vec<_> =
        (0..input.len()).filter_map(|s| index.get(input.charge(s)).map(|&t| (s, t))).collect();
    pairs.sort_unstable_by_key(|p| p.1);
    pairs
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// Families and index sets for one rectangular domain (input -> output).
pub(crate) struct PlanInput<'a> {
    pub n: usize,
    pub ground: usize,
    pub library: Arc<Library>,
    pub commutator: &'a Family,
    pub left: &'a Family,
    pub right: &'a Family,
    pub selected_k: &'a Family,
    pub recycling: &'a Family,
    pub ground_hamiltonian: &'a Family,
    pub optical_hamiltonian: &'a Family,
    pub input: &'a [Charge],
    pub first: &'a [Charge],
    pub j: &'a [Charge],
    pub z: &'a [Charge],
    pub output: &'a [Charge],
    /// (row, column) of each selected ground coordinate.
    pub gg: Vec<(usize, usize)>,
    /// (row, column) of each optical coordinate.
    pub optical: Vec<(usize, usize)>,
    /// Transport frequency of each output mode, caller order.
    pub frequency: Vec<f64>,
    pub max_rhs: usize,
    pub blas: Option<Arc<Lapack>>,
}

pub(crate) struct Plan {
    n: usize,
    ground: usize,
    rank: usize,
    max_rhs: usize,
    input: ModeOrder,
    first: ModeOrder,
    j: ModeOrder,
    z: ModeOrder,
    output: ModeOrder,
    first_commutator: Adjacency,
    second_commutator: Adjacency,
    left_no_jump: Adjacency,
    right_no_jump: Adjacency,
    selected_k: Vec<Link>,
    recycling: Vec<Link>,
    ground_hamiltonian: Vec<Link>,
    optical_hamiltonian: Vec<Link>,
    input_to_j: Vec<(usize, usize)>,
    first_to_j: Vec<(usize, usize)>,
    j_to_output: Vec<(usize, usize)>,
    z_to_output: Vec<(usize, usize)>,
    gg: Vec<(usize, usize)>,
    /// gg index of the transposed coordinate, when the set is closed.
    gg_transpose: Option<Vec<usize>>,
    optical: Vec<(usize, usize)>,
    /// Transport frequency per internal output mode.
    frequency: Vec<f64>,
    half_capable: bool,
    library: Arc<Library>,
    blas: Option<Arc<Lapack>>,
    execution: Execution,
}

impl Plan {
    pub(crate) fn new(spec: PlanInput<'_>) -> Result<Self, String> {
        let input = ModeOrder::new(spec.input);
        let first = ModeOrder::new(spec.first);
        let j = ModeOrder::new(spec.j);
        let z = ModeOrder::new(spec.z);
        let output = ModeOrder::new(spec.output);
        let gg_position: BTreeMap<(usize, usize), usize> = spec.gg.iter().enumerate().map(|(k, &p)| (p, k)).collect();
        let gg_transpose: Option<Vec<usize>> =
            spec.gg.iter().map(|&(a, b)| gg_position.get(&(b, a)).copied()).collect();
        let families_closed = [
            spec.commutator,
            spec.left,
            spec.right,
            spec.selected_k,
            spec.recycling,
            spec.ground_hamiltonian,
            spec.optical_hamiltonian,
        ]
        .iter()
        .all(|family| family.keys().all(|q| negate(q).is_some_and(|p| family.contains_key(&p))));
        let half_capable = families_closed
            && gg_transpose.is_some()
            && [&input, &first, &j, &z, &output].iter().all(|order| order.pairing.is_some());
        let frequency = (0..output.len()).map(|t| spec.frequency[output.external[t]]).collect();
        Ok(Self {
            n: spec.n,
            ground: spec.ground,
            rank: spec.gg.len(),
            max_rhs: spec.max_rhs,
            first_commutator: Adjacency::new(&links(spec.commutator, &input, &first)?, first.len()),
            second_commutator: Adjacency::new(&links(spec.commutator, &first, &j)?, j.len()),
            left_no_jump: Adjacency::new(&links(spec.left, &j, &output)?, output.len()),
            right_no_jump: Adjacency::new(&links(spec.right, &j, &output)?, output.len()),
            selected_k: links(spec.selected_k, &j, &z)?,
            recycling: links(spec.recycling, &j, &output)?,
            ground_hamiltonian: links(spec.ground_hamiltonian, &z, &output)?,
            optical_hamiltonian: links(spec.optical_hamiltonian, &z, &output)?,
            input_to_j: embedding(&input, &j),
            first_to_j: embedding(&first, &j),
            j_to_output: embedding(&j, &output),
            z_to_output: embedding(&z, &output),
            gg: spec.gg,
            gg_transpose,
            optical: spec.optical,
            frequency,
            half_capable,
            library: spec.library,
            blas: spec.blas,
            execution: Execution::new()?,
            input,
            first,
            j,
            z,
            output,
        })
    }

    pub(crate) fn input_modes(&self) -> usize {
        self.input.len()
    }

    pub(crate) fn output_modes(&self) -> usize {
        self.output.len()
    }

    pub(crate) fn intermediate_modes(&self) -> (usize, usize, usize) {
        (self.first.len(), self.j.len(), self.z.len())
    }

    pub(crate) fn max_rhs(&self) -> usize {
        self.max_rhs
    }

    pub(crate) fn hermitian_half_capable(&self) -> bool {
        self.half_capable
    }

    #[cfg(test)]
    pub(crate) fn library(&self) -> &Arc<Library> {
        &self.library
    }

    /// Bytes of the index tables owned by this plan (coefficients excluded).
    pub(crate) fn index_bytes(&self) -> usize {
        let word = std::mem::size_of::<usize>();
        let hilbert = [&self.first_commutator, &self.second_commutator, &self.left_no_jump, &self.right_no_jump];
        let thin = [&self.selected_k, &self.recycling, &self.ground_hamiltonian, &self.optical_hamiltonian];
        let edges: usize = hilbert.iter().map(|s| s.offsets.len() + 2 * s.source.len()).sum::<usize>()
            + thin.iter().flat_map(|s| s.iter()).map(|l| 2 * l.sources.len()).sum::<usize>();
        let maps = self.input_to_j.len() + self.first_to_j.len() + self.j_to_output.len() + self.z_to_output.len();
        let orders = [&self.input, &self.first, &self.j, &self.z, &self.output]
            .iter()
            .map(|o| 2 * o.len() + o.charges.iter().map(Vec::len).sum::<usize>() / 2)
            .sum::<usize>();
        (edges + 2 * maps + orders + 2 * (self.gg.len() + self.optical.len())) * word + self.frequency.len() * 8
    }

    /// Upper bound of the workspace bytes needed per right-hand side.
    pub(crate) fn workspace_bytes_per_rhs(&self) -> usize {
        let d = self.n * self.n;
        let e = self.n - self.ground;
        let fields = (self.input.len() + self.first.len() + self.j.len() + 2 * self.output.len()) * d
            + self.j.len() * e * e
            + self.z.len() * self.rank;
        // Hilbert tasks stack at most all sources and coefficient blocks of
        // one target; thin tasks pack one coefficient's edges.
        let hilbert = [&self.first_commutator, &self.second_commutator, &self.left_no_jump, &self.right_no_jump];
        let widest = hilbert.iter().map(|s| s.widest()).max().unwrap_or(0);
        let thin = [&self.selected_k, &self.recycling, &self.ground_hamiltonian, &self.optical_hamiltonian];
        let thin_widest = thin
            .iter()
            .flat_map(|s| s.iter())
            .map(|l| {
                let a = &self.library.thin[l.coefficient];
                l.sources.len() * (a.rows + a.cols)
            })
            .max()
            .unwrap_or(0);
        let scratch = (2 * widest * d).max(thin_widest) * self.execution.threads();
        (fields + scratch) * std::mem::size_of::<C>()
    }

    fn gemm(&self) -> Gemm<'_> {
        self.blas.as_ref().map_or(Gemm::Reference, |lapack| Gemm::Blas(lapack.complex_gemm()))
    }

    /// Compute E and F of `input` (caller layout [mode][(i*n+j)*rhs+b]) into
    /// the workspace. Returns whether the Hermitian half was used.
    pub(crate) fn apply(
        &self,
        w: &mut Workspace,
        input: &[C],
        rhs: usize,
        profile: &mut StageProfile,
    ) -> Result<bool, String> {
        let n = self.n;
        let block = n * n * rhs;
        let gemm = self.gemm();
        let mut clock = StageClock::start();

        // Input in mode-block layout; detect exact Hermitian symmetry.
        let half = self.load_input(input, rhs, w)?;
        clock.mark(profile, 0);

        // dx = [S, x].
        let first_count = self.first.computed(half);
        ensure(&mut w.dx, self.first.len() * block);
        w.dx[..first_count * block].fill(ZERO);
        let dx = &mut w.dx[..first_count * block];
        self.hilbert_stage(gemm, &self.first_commutator, &w.x, dx, rhs, Some(1.), Some(-1.), &mut w.scratch)?;
        if half {
            mirror(&mut w.dx, &self.first, n, rhs);
        }
        clock.mark(profile, 1);

        // J = x + dx + [S, dx] / 2.
        let j_count = self.j.computed(half);
        ensure(&mut w.j, self.j.len() * block);
        let j_field = &mut w.j[..j_count * block];
        j_field.fill(ZERO);
        for map in [(&self.input_to_j, &w.x), (&self.first_to_j, &w.dx)] {
            for &(s, t) in map.0.iter().take_while(|p| p.1 < j_count) {
                add_slice(&mut j_field[t * block..(t + 1) * block], &map.1[s * block..(s + 1) * block]);
            }
        }
        self.hilbert_stage(gemm, &self.second_commutator, &w.dx, j_field, rhs, Some(0.5), Some(-0.5), &mut w.scratch)?;
        if half {
            mirror(&mut w.j, &self.j, n, rhs);
        }
        clock.mark(profile, 2);

        // z = K vec(J_EE) on the selected ground coordinates.
        let e = n - self.ground;
        let ee_width = e * e * rhs;
        ensure(&mut w.ee, self.j.len() * ee_width);
        extract_excited(&w.j, &mut w.ee, self.j.len(), n, self.ground, rhs);
        let z_width = self.rank * rhs;
        let z_count = self.z.computed(half);
        ensure(&mut w.z, self.z.len() * z_width);
        w.z[..z_count * z_width].fill(ZERO);
        let z = &mut w.z[..z_count * z_width];
        self.thin_stage(gemm, &self.selected_k, &w.ee, z, Rows::Vector, rhs, &mut w.scratch)?;
        if half {
            mirror_ground(&mut w.z, &self.z, self.gg_transpose.as_deref().unwrap_or(&[]), self.rank, rhs);
        }
        clock.mark(profile, 3);

        // E = J + embed_GG(z).
        let output_count = self.output.computed(half);
        ensure(&mut w.e, self.output.len() * block);
        let e_field = &mut w.e[..output_count * block];
        e_field.fill(ZERO);
        for &(s, t) in self.j_to_output.iter().take_while(|p| p.1 < output_count) {
            e_field[t * block..(t + 1) * block].copy_from_slice(&w.j[s * block..(s + 1) * block]);
        }
        for &(s, t) in self.z_to_output.iter().take_while(|p| p.1 < output_count) {
            for b in 0..rhs {
                for (k, &(row, col)) in self.gg.iter().enumerate() {
                    let at = t * block + row + n * (b + rhs * col);
                    e_field[at] = add(e_field[at], w.z[s * z_width + k + self.rank * b]);
                }
            }
        }
        clock.mark(profile, 4);

        // F = NL * J + J * NR + R vec(J_EE) + L_GG z + L_OG z - i omega E.
        ensure(&mut w.f, self.output.len() * block);
        let f_field = &mut w.f[..output_count * block];
        f_field.fill(ZERO);
        self.hilbert_stage(gemm, &self.left_no_jump, &w.j, f_field, rhs, Some(1.), None, &mut w.scratch)?;
        clock.mark(profile, 5);
        self.hilbert_stage(gemm, &self.right_no_jump, &w.j, f_field, rhs, None, Some(1.), &mut w.scratch)?;
        clock.mark(profile, 6);
        self.thin_stage(gemm, &self.recycling, &w.ee, f_field, Rows::Matrix(&self.gg), rhs, &mut w.scratch)?;
        self.thin_stage(gemm, &self.ground_hamiltonian, &w.z, f_field, Rows::Matrix(&self.gg), rhs, &mut w.scratch)?;
        let optical = Rows::Matrix(&self.optical);
        self.thin_stage(gemm, &self.optical_hamiltonian, &w.z, f_field, optical, rhs, &mut w.scratch)?;
        for t in 0..output_count {
            let omega = self.frequency[t];
            let range = t * block..(t + 1) * block;
            for (f, e) in f_field[range.clone()].iter_mut().zip(&w.e[range]) {
                f.re += omega * e.im;
                f.im -= omega * e.re;
            }
        }
        clock.mark(profile, 7);
        Ok(half)
    }

    /// Write E and F in caller layout.
    pub(crate) fn store(
        &self,
        w: &Workspace,
        half: bool,
        rhs: usize,
        mass: &mut [C],
        pencil: &mut [C],
    ) -> Result<(), String> {
        let read = |field: &[C], at: usize, mirrored: bool| if mirrored { conj(field[at]) } else { field[at] };
        self.store_output(half, rhs, mass, |at, mirrored| read(&w.e, at, mirrored))?;
        self.store_output(half, rhs, pencil, |at, mirrored| read(&w.f, at, mirrored))
    }

    /// Write shift*E - F in caller layout (E and F are mirrored before the
    /// shift is applied, so complex shifts are exact as well).
    pub(crate) fn store_shifted(
        &self,
        w: &Workspace,
        shift: C,
        half: bool,
        rhs: usize,
        out: &mut [C],
    ) -> Result<(), String> {
        self.store_output(half, rhs, out, |at, mirrored| {
            let (e, f) = if mirrored { (conj(w.e[at]), conj(w.f[at])) } else { (w.e[at], w.f[at]) };
            let value = mul(shift, e);
            C { re: value.re - f.re, im: value.im - f.im }
        })
    }

    /// Caller layout [mode][(i*n+j)*rhs+b] -> mode blocks in internal order,
    /// in parallel over modes. Rejects nonfinite input. Returns whether the
    /// Hermitian half applies: the plan allows it and x_{-q} = x_q^dagger
    /// holds exactly for every pair (the central mode is Hermitian).
    fn load_input(&self, input: &[C], rhs: usize, w: &mut Workspace) -> Result<bool, String> {
        let (n, order) = (self.n, &self.input);
        let block = n * n * rhs;
        ensure(&mut w.x, order.len() * block);
        let x = &mut w.x[..order.len() * block];
        let work = x.len() * COPY_WORK;
        let minimum = MINIMUM_WORK_PER_TASK;
        self.execution.for_mode_chunks(x, block, work, minimum, &mut Vec::<()>::new(), |first, chunk, _| {
            for (k, target) in chunk.chunks_exact_mut(block).enumerate() {
                let external = order.external[first + k];
                let source = &input[external * block..(external + 1) * block];
                if !source.iter().all(|z| z.finite()) {
                    return Err("Nonfinite owned pencil input".into());
                }
                for j in 0..n {
                    for b in 0..rhs {
                        for i in 0..n {
                            target[i + n * (b + rhs * j)] = source[(i * n + j) * rhs + b];
                        }
                    }
                }
            }
            Ok(())
        })?;
        let Some(pairing) = order.pairing.filter(|_| self.half_capable) else { return Ok(false) };
        let pairs = pairing.canonical - pairing.central;
        let Workspace { x, symmetric, .. } = w;
        let x = &x[..order.len() * block];
        symmetric.resize(pairing.canonical, false);
        let work = pairing.canonical * block * COPY_WORK;
        self.execution.for_mode_chunks(symmetric, 1, work, minimum, &mut Vec::<()>::new(), |first, chunk, _| {
            for (k, flag) in chunk.iter_mut().enumerate() {
                let c = first + k;
                let partner = if c < pairing.central { c } else { c + pairs };
                let (a, b) = (&x[c * block..(c + 1) * block], &x[partner * block..(partner + 1) * block]);
                *flag = (0..n).all(|j| {
                    (0..rhs).all(|r| {
                        (0..n).all(|i| {
                            let (u, v) = (a[i + n * (r + rhs * j)], b[j + n * (r + rhs * i)]);
                            u.re == v.re && u.im == -v.im
                        })
                    })
                });
            }
            Ok(())
        })?;
        Ok(symmetric.iter().all(|&flag| flag))
    }

    /// Write every caller-order output mode, in parallel over modes; in half
    /// mode a mirrored mode is the conjugate transpose of its canonical
    /// partner. `value(at, mirrored)` reads internal entry `at` (conjugated
    /// when `mirrored`). Rejects nonfinite results.
    fn store_output(
        &self,
        half: bool,
        rhs: usize,
        out: &mut [C],
        value: impl Fn(usize, bool) -> C + Sync,
    ) -> Result<(), String> {
        let (n, order) = (self.n, &self.output);
        let block = n * n * rhs;
        let mirrored = order.mirrored().filter(|_| half);
        let work = out.len() * COPY_WORK;
        let minimum = MINIMUM_WORK_PER_TASK;
        self.execution.for_mode_chunks(out, block, work, minimum, &mut Vec::<()>::new(), |first, chunk, _| {
            let mut finite = true;
            for (k, target) in chunk.chunks_exact_mut(block).enumerate() {
                let internal = order.internal[first + k];
                // A mirrored mode reads its partner's block transposed.
                let (base, transposed) = match mirrored {
                    Some((start, offset)) if internal >= start => ((internal - offset) * block, true),
                    _ => (internal * block, false),
                };
                for i in 0..n {
                    for j in 0..n {
                        let at = if transposed { base + j + n * rhs * i } else { base + i + n * rhs * j };
                        for b in 0..rhs {
                            let z = value(at + n * b, transposed);
                            finite &= z.finite();
                            target[(i * n + j) * rhs + b] = z;
                        }
                    }
                }
            }
            if finite { Ok(()) } else { Err("Nonfinite complete native E/F result".into()) }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn hilbert_stage(
        &self,
        gemm: Gemm<'_>,
        stage: &Adjacency,
        source: &[C],
        target: &mut [C],
        rhs: usize,
        left: Option<f64>,
        right: Option<f64>,
        scratch: &mut Vec<Scratch>,
    ) -> Result<(), String> {
        let products = StageProducts { gemm, library: &self.library.hilbert, n: self.n, ground: self.ground, stage, rhs };
        products.apply(&self.execution, source, target, left, right, scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn thin_stage(
        &self,
        gemm: Gemm<'_>,
        links: &[Link],
        source: &[C],
        target: &mut [C],
        rows: Rows<'_>,
        rhs: usize,
        scratch: &mut Vec<Scratch>,
    ) -> Result<(), String> {
        let n = self.n;
        let library = &self.library.thin;
        let Some(first_link) = links.first() else { return Ok(()) };
        let target_width = match rows {
            Rows::Vector => library[first_link.coefficient].rows * rhs,
            Rows::Matrix(_) => n * n * rhs,
        };
        let modes = 0..target.len() / target_width;
        let work: usize = links
            .iter()
            .map(|link| {
                let a = &library[link.coefficient];
                let per_edge = if a.dense.is_some() { a.rows * a.cols } else { CSR_WORK * a.sparse.value.len() };
                link.within(&modes).len() * per_edge * rhs
            })
            .sum();
        let minimum = MINIMUM_WORK_PER_TASK;
        self.execution.for_mode_chunks(target, target_width, work, minimum, scratch, |first, chunk, scratch| {
            let targets = first..first + chunk.len() / target_width;
            for link in links {
                let edges = link.within(&targets);
                if edges.is_empty() {
                    continue;
                }
                let a = &library[link.coefficient];
                let sources = &link.sources[edges.clone()];
                let destinations = &link.targets[edges];
                let position = |row: usize, b: usize| match rows {
                    Rows::Vector => row + a.rows * b,
                    Rows::Matrix(at) => at[row].0 + n * (b + rhs * at[row].1),
                };
                let width = a.cols * rhs;
                match &a.dense {
                    Some(dense) if sources.len() * rhs >= 8 => {
                        let columns = sources.len() * rhs;
                        let Scratch { packed, product, .. } = scratch;
                        let input: &[C] = if consecutive(sources) {
                            &source[sources[0] * width..(sources[0] + sources.len()) * width]
                        } else {
                            ensure(packed, sources.len() * width);
                            for (k, &s) in sources.iter().enumerate() {
                                packed[k * width..(k + 1) * width].copy_from_slice(&source[s * width..(s + 1) * width]);
                            }
                            &packed[..sources.len() * width]
                        };
                        ensure(product, a.rows * columns);
                        let (rows, cols) = (a.rows, a.cols);
                        gemm.multiply(rows, columns, cols, ONE, dense, rows, input, cols, ZERO, product, rows)?;
                        for (k, &t) in destinations.iter().enumerate() {
                            let out = &mut chunk[(t - first) * target_width..(t - first + 1) * target_width];
                            for b in 0..rhs {
                                let column = &product[a.rows * (k * rhs + b)..a.rows * (k * rhs + b + 1)];
                                for (row, &value) in column.iter().enumerate() {
                                    let at = position(row, b);
                                    out[at] = add(out[at], value);
                                }
                            }
                        }
                    }
                    _ => {
                        let csr = &a.sparse;
                        for (&s, &t) in sources.iter().zip(destinations) {
                            let x = &source[s * width..(s + 1) * width];
                            let out = &mut chunk[(t - first) * target_width..(t - first + 1) * target_width];
                            for b in 0..rhs {
                                let x = &x[a.cols * b..a.cols * (b + 1)];
                                for row in 0..a.rows {
                                    let mut sum = ZERO;
                                    for at in csr.row[row]..csr.row[row + 1] {
                                        sum = add(sum, mul(csr.value[at], x[csr.col[at]]));
                                    }
                                    let at = position(row, b);
                                    out[at] = add(out[at], sum);
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

/// Where a thin coefficient's rows land in a target mode.
#[derive(Clone, Copy)]
enum Rows<'a> {
    /// Row r of rhs b at r + rows*b (z field).
    Vector,
    /// Row r of rhs b at matrix entry at[r] of the mode block.
    Matrix(&'a [(usize, usize)]),
}

/// One Hilbert stage on explicit data: for every target mode t,
/// target_t += left * sum A_p X_{t-p} + right * sum X_{t-p} A_p over the
/// stage edges, one task per contiguous chunk of target modes.
struct StageProducts<'a> {
    gemm: Gemm<'a>,
    library: &'a [HilbertCoefficient],
    n: usize,
    ground: usize,
    stage: &'a Adjacency,
    rhs: usize,
}

impl StageProducts<'_> {
    fn apply(
        &self,
        execution: &Execution,
        source: &[C],
        target: &mut [C],
        left: Option<f64>,
        right: Option<f64>,
        scratch: &mut Vec<Scratch>,
    ) -> Result<(), String> {
        let (gemm, library, n, stage, rhs) = (self.gemm, self.library, self.n, self.stage, self.rhs);
        let block = n * n * rhs;
        let work: usize = (0..target.len() / block)
            .flat_map(|t| stage.edges(t))
            .map(|(c, _)| (left.map_or(0, |_| library[c].left_macs) + right.map_or(0, |_| library[c].right_macs)) * rhs)
            .sum();
        let parts = [0..self.ground, self.ground..n];
        execution.for_mode_chunks(target, block, work, MINIMUM_WORK_PER_TASK, scratch, |first, chunk, scratch| {
            for (k, y) in chunk.chunks_exact_mut(block).enumerate() {
                let products = TargetProducts { gemm, n, rhs, library, source, stage, target: first + k };
                for output in &parts {
                    if let Some(alpha) = left {
                        products.left(alpha, output, y, scratch)?;
                    }
                    if let Some(alpha) = right {
                        products.right(alpha, output, y, scratch)?;
                    }
                }
            }
            Ok(())
        })
    }
}

/// Commutator convolution between two mode sets given in the caller's order:
/// target_t += alpha * sum_p (A_p X_{t-p} - X_{t-p} A_p), with the fields in
/// mode-block layout. It runs on the same blocked kernel as the pencil's
/// Hilbert stages (zero ground/excited blocks of A_p are skipped).
pub(crate) struct Commutator {
    n: usize,
    ground: usize,
    library: Vec<HilbertCoefficient>,
    stage: Adjacency,
}

impl Commutator {
    /// `family` maps charge p to A_p (row-major n x n).
    pub(crate) fn new(family: &FourierMatrices, input: &[Charge], output: &[Charge], n: usize, ground: usize) -> Result<Self, String> {
        let mut library = Vec::new();
        let mut charges = Family::new();
        for (p, a) in family {
            if a.len() != n * n {
                return Err("Commutator coefficient is not n x n".into());
            }
            charges.insert(p.clone(), library.len());
            library.push(HilbertCoefficient::new(a, n, ground));
        }
        let links = links(&charges, &ModeOrder::given(input), &ModeOrder::given(output))?;
        Ok(Self { n, ground, library, stage: Adjacency::new(&links, output.len()) })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &self,
        execution: &Execution,
        gemm: Gemm<'_>,
        input: &[C],
        output: &mut [C],
        rhs: usize,
        alpha: f64,
        scratch: &mut Vec<Scratch>,
    ) -> Result<(), String> {
        let products = StageProducts { gemm, library: &self.library, n: self.n, ground: self.ground, stage: &self.stage, rhs };
        products.apply(execution, input, output, Some(alpha), Some(-alpha), scratch)
    }
}

/// All Hilbert products landing in one target mode. For each output part
/// (ground or excited rows/columns) the coefficient blocks of every edge are
/// concatenated along the contraction and the matching source rows/columns
/// stacked, so one GEMM accumulates the whole convolution into the target.
struct TargetProducts<'a> {
    gemm: Gemm<'a>,
    n: usize,
    rhs: usize,
    library: &'a [HilbertCoefficient],
    source: &'a [C],
    stage: &'a Adjacency,
    target: usize,
}

impl TargetProducts<'_> {
    /// Total contraction length of the edges' blocks that write `output`.
    fn depth(&self, blocks: impl Fn(&HilbertCoefficient) -> &[Block], output: &Range<usize>) -> usize {
        self.stage
            .edges(self.target)
            .filter_map(|(c, _)| blocks(&self.library[c]).iter().find(|b| b.output == *output))
            .map(|b| b.input.len())
            .sum()
    }

    /// y[output rows, :] += alpha * [A_1 | A_2 | ...] [X_1; X_2; ...].
    fn left(&self, alpha: f64, output: &Range<usize>, y: &mut [C], scratch: &mut Scratch) -> Result<(), String> {
        let (n, height) = (self.n, self.n * self.rhs);
        let depth = self.depth(|a| &a.left, output);
        if depth == 0 {
            return Ok(());
        }
        let m = output.len();
        let Scratch { packed, stacked, .. } = scratch;
        ensure(stacked, m * depth);
        ensure(packed, depth * height);
        let mut row = 0;
        for (c, s) in self.stage.edges(self.target) {
            let a = &self.library[c];
            let Some(block) = a.left.iter().find(|b| b.output == *output) else { continue };
            let input = &block.input;
            for (k, col) in input.clone().enumerate() {
                let entries = &a.dense[output.start + n * col..output.end + n * col];
                stacked[m * (row + k)..m * (row + k + 1)].copy_from_slice(entries);
            }
            let x = &self.source[s * n * height..(s + 1) * n * height];
            for column in 0..height {
                packed[row + depth * column..row + input.len() + depth * column]
                    .copy_from_slice(&x[input.start + n * column..input.end + n * column]);
            }
            row += input.len();
        }
        let alpha = C { re: alpha, im: 0. };
        self.gemm.multiply(m, height, depth, alpha, stacked, m, packed, depth, ONE, &mut y[output.start..], n)
    }

    /// y[:, output columns] += alpha * [X_1 | X_2 | ...] [A_1; A_2; ...].
    fn right(&self, alpha: f64, output: &Range<usize>, y: &mut [C], scratch: &mut Scratch) -> Result<(), String> {
        let (n, height) = (self.n, self.n * self.rhs);
        let depth = self.depth(|a| &a.right, output);
        if depth == 0 {
            return Ok(());
        }
        let m = output.len();
        let Scratch { packed, stacked, .. } = scratch;
        ensure(packed, height * depth);
        ensure(stacked, depth * m);
        let mut row = 0;
        for (c, s) in self.stage.edges(self.target) {
            let a = &self.library[c];
            let Some(block) = a.right.iter().find(|b| b.output == *output) else { continue };
            let input = &block.input;
            let x = &self.source[s * n * height..(s + 1) * n * height];
            let columns = &x[height * input.start..height * input.end];
            packed[height * row..height * (row + input.len())].copy_from_slice(columns);
            for (k, col) in output.clone().enumerate() {
                let entries = &a.dense[input.start + n * col..input.end + n * col];
                stacked[row + depth * k..row + input.len() + depth * k].copy_from_slice(entries);
            }
            row += input.len();
        }
        let alpha = C { re: alpha, im: 0. };
        let target = &mut y[height * output.start..];
        self.gemm.multiply(height, m, depth, alpha, packed, height, stacked, depth, ONE, target, height)
    }
}

// ---------------------------------------------------------------------------
// Workspace and layout conversion
// ---------------------------------------------------------------------------

/// Per-task packing buffers.
#[derive(Default)]
pub(crate) struct Scratch {
    packed: Vec<C>,
    product: Vec<C>,
    stacked: Vec<C>,
}

/// Reusable buffers of one plan; they only grow.
#[derive(Default)]
pub(crate) struct Workspace {
    x: Vec<C>,
    dx: Vec<C>,
    j: Vec<C>,
    ee: Vec<C>,
    z: Vec<C>,
    pub(crate) e: Vec<C>,
    pub(crate) f: Vec<C>,
    symmetric: Vec<bool>,
    scratch: Vec<Scratch>,
}

impl Workspace {
    pub(crate) fn bytes(&self) -> usize {
        let fields = [&self.x, &self.dx, &self.j, &self.ee, &self.z, &self.e, &self.f];
        let scratch: usize =
            self.scratch.iter().map(|s| s.packed.capacity() + s.product.capacity() + s.stacked.capacity()).sum();
        (fields.iter().map(|v| v.capacity()).sum::<usize>() + scratch) * std::mem::size_of::<C>()
    }
}

/// Fill the non-canonical modes of a mode-block field from their partners.
fn mirror(field: &mut [C], order: &ModeOrder, n: usize, rhs: usize) {
    let Some((start, offset)) = order.mirrored() else { return };
    let block = n * n * rhs;
    let (canonical, partners) = field[..order.len() * block].split_at_mut(start * block);
    for (k, target) in partners.chunks_exact_mut(block).enumerate() {
        let source = &canonical[(start + k - offset) * block..(start + k - offset + 1) * block];
        for i in 0..n {
            for b in 0..rhs {
                for j in 0..n {
                    target[i + n * (b + rhs * j)] = conj(source[j + n * (b + rhs * i)]);
                }
            }
        }
    }
}

/// Mirror the selected ground coordinates z_{-q} = (z_q)^dagger.
fn mirror_ground(field: &mut [C], order: &ModeOrder, transpose: &[usize], rank: usize, rhs: usize) {
    let Some((start, offset)) = order.mirrored() else { return };
    let width = rank * rhs;
    let (canonical, partners) = field[..order.len() * width].split_at_mut(start * width);
    for (k, target) in partners.chunks_exact_mut(width).enumerate() {
        let source = &canonical[(start + k - offset) * width..(start + k - offset + 1) * width];
        for b in 0..rhs {
            for (row, &from) in transpose.iter().enumerate() {
                target[row + rank * b] = conj(source[from + rank * b]);
            }
        }
    }
}

/// vec(J_EE) of every mode, column-major (i + e*j), rhs-major.
fn extract_excited(j: &[C], ee: &mut [C], modes: usize, n: usize, ground: usize, rhs: usize) {
    let e = n - ground;
    let block = n * n * rhs;
    for m in 0..modes {
        for b in 0..rhs {
            for col in 0..e {
                let from = m * block + ground + n * (b + rhs * (ground + col));
                let to = (m * rhs + b) * e * e + e * col;
                ee[to..to + e].copy_from_slice(&j[from..from + e]);
            }
        }
    }
}
