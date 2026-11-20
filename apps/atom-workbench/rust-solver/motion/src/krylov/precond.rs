//! The preconditioner (FASTREF.md §§ 2.1, 8; `fr.PrecondH`, `fr.Precond`).
//!
//! With the unknowns split into slow s and fast f, P = A except (i) the fast-fast block is replaced by its diagonal
//! D_f and (ii) the slow Schur complement S = A_ss − A_sf D_f⁻¹ A_fs is replaced by its diagonal blocks over a
//! partition of the slow unknowns, each block factorised exactly:
//!
//! ```text
//!   P⁻¹ r:  t = D_f⁻¹ r_f;   S_blocks z_s = r_s − A_sf t;   z_f = D_f⁻¹ (r_f − A_fs z_s).
//! ```
//!
//! The partition is the one decision: the harmonics (S_h = blockdiag_j S_jj, the adopted first level), one block (the
//! exact S, the second level), or -- with every unknown slow -- one block of A itself (P = A: the direct LU, the
//! third). A − P = diag(S − S_blocks, A_ff − D_f), so P⁻¹A − I is set by S_blocks⁻¹ S_cross (a harmonic's response to
//! another's source against its own gap) and D_f⁻¹ O_f (fast-fast coupling against the fast gap); the slow-fast
//! coupling (light shifts, pumping through fast elements) is exact in P. S is S's own second-order fold of the fast
//! elements, with dissipation.
//!
//! Only the entries S keeps are formed: a dropped cross-block entry is never used (GMRES multiplies by A itself), so
//! forming it would be work for nothing; whether any exists is recorded (`cross`), since without one the exact S is
//! this same operator. Rows of S and the blocks' factorisations are independent and run in parallel; each row's sums
//! run in the prototype's order (T = (A_sf D_f⁻¹) A_fs first, then A_ss − T), so the result does not depend on the
//! thread count.
//!
//! The sparse LU is faer's (COLAMD column order, partial row pivoting) -- see Cargo.toml -- in its left-looking
//! (Gilbert–Peierls, "simplicial") form: L and U grow with the fill the pivots actually make, as SuperLU's do. faer's
//! supernodal form instead sizes its storage from the structure of AᵀA, an upper bound over every pivot sequence; on
//! tilt2's edge reference (30097 unknowns, 110 entries a row) that bound is near dense and the factorisation runs out
//! of memory (8 GB), where the actual fill is a few hundred MB. A block whose factorisation fails makes the
//! preconditioner unavailable (`Unfactorised`, with faer's error); a numerically zero pivot shows as a non-finite
//! application, which GMRES stops on.

use super::{diagonal, row};
use crate::floquet::Sparse;
use crate::linalg::C64;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuRef, LuSymbolicParams, NumericLu, SymbolicLu};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{c64, Conj, MatMut, Par};
use rayon::prelude::*;

/// One diagonal block of the slow Schur complement, factorised: its slow-local members (ascending) and the LU.
struct Block {
    members: Vec<usize>,
    symbolic: SymbolicLu<usize>,
    numeric: NumericLu<usize, c64>,
}

/// A block's LU failed (faer's error, e.g. a structurally singular block); `cross`: S had entries outside the blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct Unfactorised {
    pub cross: bool,
    pub reason: String,
}

/// The preconditioner P for one partition of the slow unknowns.
pub struct Schur {
    slow: Vec<usize>,
    fast: Vec<usize>,
    d_fast: Vec<C64>,
    a_sf: Sparse,
    a_fs: Sparse,
    blocks: Vec<Block>,
    /// S has an entry outside the blocks (then the blocks are not the exact S).
    pub cross: bool,
    /// Entries of S formed (inside the blocks).
    pub nnz_kept: usize,
    /// Complex multiply-adds of one application (2 n_f + nnz A_sf + nnz A_fs + the block solves' factors are not
    /// counted: faer does not report them; the solve's own clock does).
    pub apply_madds: f64,
}

impl Schur {
    /// P for the slow mask `slow` (true = slow) and the slow unknowns' partition `group` (one label per unknown; only
    /// the slow ones' are read).
    pub fn new(a: &Sparse, slow: &[bool], group: &[usize]) -> Result<Schur, Unfactorised> {
        let n = a.rows;
        let s: Vec<usize> = (0..n).filter(|&i| slow[i]).collect();
        let f: Vec<usize> = (0..n).filter(|&i| !slow[i]).collect();
        let mut local = vec![0usize; n];
        for (k, &i) in s.iter().enumerate() {
            local[i] = k;
        }
        for (k, &i) in f.iter().enumerate() {
            local[i] = k;
        }
        let diag = diagonal(a);
        let d_fast: Vec<C64> = f.iter().map(|&i| diag[i]).collect();
        // the four blocks of A by rows: A_ss and A_sf from the slow rows, A_fs from the fast rows
        let pick = |rows: &[usize], want_slow: bool| -> Sparse {
            let mut indptr = Vec::with_capacity(rows.len() + 1);
            let (mut indices, mut values) = (Vec::new(), Vec::new());
            indptr.push(0);
            for &i in rows {
                let (c, v) = row(a, i);
                for (&j, &x) in c.iter().zip(v) {
                    if slow[j] == want_slow {
                        indices.push(local[j]);
                        values.push(x);
                    }
                }
                indptr.push(indices.len());
            }
            let cols = if want_slow { s.len() } else { f.len() };
            Sparse { rows: rows.len(), cols, indptr, indices, values }
        };
        let a_ss = pick(&s, true);
        let a_sf = pick(&s, false);
        let a_fs = pick(&f, true);
        let label: Vec<usize> = s.iter().map(|&i| group[i]).collect();
        let reciprocal: Vec<C64> = d_fast.iter().map(|&d| C64::ONE / d).collect();
        let ns = s.len();
        // rows of S inside the blocks: (columns, values, any entry outside them)
        let rows: Vec<(Vec<usize>, Vec<C64>, bool)> = (0..ns)
            .into_par_iter()
            .map_init(
                || (vec![C64::ZERO; ns], vec![usize::MAX; ns], vec![C64::ZERO; ns], vec![usize::MAX; ns]),
                |(product, in_product, own, in_own), i| {
                    let g = label[i];
                    let mut cols = Vec::new();
                    let mut cross = false;
                    // T_i = Σ_k (A_ik D_k⁻¹) A_k·
                    let (cf, vf) = row(&a_sf, i);
                    for (&k, &x) in cf.iter().zip(vf) {
                        let coeff = x * reciprocal[k];
                        let (cs, vs) = row(&a_fs, k);
                        for (&j, &y) in cs.iter().zip(vs) {
                            if label[j] != g {
                                cross = true;
                                continue;
                            }
                            if in_product[j] != i {
                                in_product[j] = i;
                                product[j] = C64::ZERO;
                                if in_own[j] != i {
                                    cols.push(j);
                                }
                            }
                            product[j] += coeff * y;
                        }
                    }
                    let (cs, vs) = row(&a_ss, i);
                    for (&j, &x) in cs.iter().zip(vs) {
                        if label[j] != g {
                            cross = true;
                            continue;
                        }
                        if in_own[j] != i && in_product[j] != i {
                            cols.push(j);
                        }
                        in_own[j] = i;
                        own[j] = x;
                    }
                    // S_i = A_ss,i − T_i
                    let value = cols
                        .iter()
                        .map(|&j| match (in_own[j] == i, in_product[j] == i) {
                            (true, true) => own[j] - product[j],
                            (true, false) => own[j],
                            _ => -product[j],
                        })
                        .collect();
                    (cols, value, cross)
                },
            )
            .collect();
        let cross = rows.iter().any(|r| r.2);
        let nnz_kept = rows.iter().map(|r| r.0.len()).sum();
        // the blocks: members by label, in ascending slow-local order
        let mut labels: Vec<usize> = label.clone();
        labels.sort_unstable();
        labels.dedup();
        let members: Vec<Vec<usize>> =
            labels.iter().map(|&g| (0..ns).filter(|&i| label[i] == g).collect()).collect();
        let mut position = vec![0usize; ns];
        for m in &members {
            for (k, &i) in m.iter().enumerate() {
                position[i] = k;
            }
        }
        let blocks: Result<Vec<Block>, String> = members
            .into_par_iter()
            .map(|m| {
                // compressed columns of the block, rows ascending within each column
                let size = m.len();
                let mut count = vec![0usize; size + 1];
                for &i in &m {
                    for &j in &rows[i].0 {
                        count[position[j] + 1] += 1;
                    }
                }
                for c in 0..size {
                    count[c + 1] += count[c];
                }
                let mut fill = count.clone();
                let mut row_index = vec![0usize; count[size]];
                let mut values = vec![c64::new(0.0, 0.0); count[size]];
                for &i in &m {
                    for (&j, &x) in rows[i].0.iter().zip(&rows[i].1) {
                        let c = position[j];
                        row_index[fill[c]] = position[i];
                        values[fill[c]] = c64::new(x.re, x.im);
                        fill[c] += 1;
                    }
                }
                let symbolic = SymbolicSparseColMatRef::new_checked(size, size, &count, None, &row_index);
                let matrix = SparseColMatRef::new(symbolic, &values);
                let params = LuSymbolicParams {
                    supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
                    ..Default::default()
                };
                let symbolic = factorize_symbolic_lu(matrix.symbolic(), params).map_err(|e| format!("{e:?}"))?;
                let mut numeric = NumericLu::new();
                let scratch = symbolic.factorize_numeric_lu_scratch::<c64>(Par::Seq, Default::default());
                let mut memory = MemBuffer::try_new(scratch).map_err(|e| format!("{e:?}"))?;
                symbolic
                    .factorize_numeric_lu(&mut numeric, matrix, Par::Seq, MemStack::new(&mut memory), Default::default())
                    .map_err(|e| format!("{e:?}"))?;
                Ok(Block { members: m, symbolic, numeric })
            })
            .collect();
        let blocks = blocks.map_err(|reason| Unfactorised { cross, reason })?;
        let apply_madds = (2 * f.len() + a_sf.indices.len() + a_fs.indices.len()) as f64;
        Ok(Schur { slow: s, fast: f, d_fast, a_sf, a_fs, blocks, cross, nnz_kept, apply_madds })
    }

    /// Number of slow unknowns and of blocks.
    pub fn shape(&self) -> (usize, usize) {
        (self.slow.len(), self.blocks.len())
    }

    /// z = P⁻¹ r.
    pub fn apply(&self, r: &[C64], z: &mut [C64]) {
        let t: Vec<C64> = self.fast.iter().zip(&self.d_fast).map(|(&i, &d)| r[i] / d).collect();
        let mut rs: Vec<C64> = self.slow.iter().map(|&i| r[i]).collect();
        for (k, ri) in rs.iter_mut().enumerate() {
            let (c, v) = row(&self.a_sf, k);
            *ri -= c.iter().zip(v).map(|(&j, &x)| x * t[j]).sum::<C64>();
        }
        let solved: Vec<Vec<c64>> = self
            .blocks
            .par_iter()
            .map(|b| {
                let mut buf: Vec<c64> = b.members.iter().map(|&i| c64::new(rs[i].re, rs[i].im)).collect();
                let m = buf.len();
                let mut memory = MemBuffer::new(b.symbolic.solve_in_place_scratch::<c64>(1, Par::Seq));
                LuRef::new_unchecked(&b.symbolic, &b.numeric).solve_in_place_with_conj(
                    Conj::No,
                    MatMut::from_column_major_slice_mut(&mut buf, m, 1),
                    Par::Seq,
                    MemStack::new(&mut memory),
                );
                buf
            })
            .collect();
        for (b, buf) in self.blocks.iter().zip(&solved) {
            for (&i, x) in b.members.iter().zip(buf) {
                z[self.slow[i]] = C64::new(x.re, x.im);
            }
        }
        for (k, (&i, &d)) in self.fast.iter().zip(&self.d_fast).enumerate() {
            let (c, v) = row(&self.a_fs, k);
            let coupled: C64 = c.iter().zip(v).map(|(&j, &x)| x * z[self.slow[j]]).sum();
            z[i] = (r[i] - coupled) / d;
        }
    }
}
