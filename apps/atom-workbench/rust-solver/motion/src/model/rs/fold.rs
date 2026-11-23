//! The one fold, linear in the light field (model/rs.rs, # The fold; lin.status.md, LIN's `lin_fold`): its structure made
//! once (`Plan`, and per grouping of its components `Grouping`), its numbers at a point (`Local`), its contraction by a
//! grouping into the model's operators (`Out`).
//!
//! # Structure once
//! Which entries count, which components and which products, which elements are slow and which channel an element
//! falls in, are read once off the fixed matrices (their rounding rule: rs.rs `fixed`) and the frame's integer fast
//! parts. So are the parts of G₀ and of X no point changes -- the entries of a factor that does not vary, at the first
//! coefficient (one) -- summed once into dense blocks (`Block::fixed`, `Batch::fixed`): a point adds the rest. A point puts in the factors (the beams' amplitudes, the tweezer's intensity, its Doppler series' powers) and
//! the frequencies; it makes the folded blocks' spectra at their local generator and the products. A block whose
//! generator varies with the tweezer's intensity s alone (its Doppler series' powers and the other beams' own
//! intensities about their focus' one to first order) takes its spectrum and the intensity's T̂ off its table in s, made
//! once at the plan's first evaluation where a node's counted work off it is below the work of making them there
//! (`table`, fold_table.rs), and each beam's own intensity's T̂ on the table's eigenbasis.
//!
//! # At a point, per component m and kept state a
//! - every folded block, Q's and K's sideband parts alike: M̂ = D ∘ (V⁻¹X), D[k, a] = 1/(((E_a + f_m) − c_k) − μ′_k)
//!   (the block's decoupled spectrum, spectrum.rs), ∂M̂ = D ∘ (V⁻¹∂X + Ĝ M̂ − M̂ ∂E), Ĝ = V⁻¹∂G₀V = Σ_σ ∂σ T̂_σ over the
//!   generator's varying factors σ (the tweezer's intensity, a Gaussian beam's), T̂_σ = V⁻¹G_σV: one product per
//!   varying factor, not per axis; M = V M̂ in the contraction (until 2026-11-21 a sideband part's M was the Neumann
//!   series about its diagonal, in the states' basis: one formula now, the spectrum's);
//! - the out-legs W = X_KQ.
//!
//! # Contraction
//! Per block the amplitudes' and the legs' sums by (class, group, the fast part of the kept state): a pair of sums is
//! slow or not as a whole (its elements' fast index γ + γ′ − n_b + n_a is one number), so each pair is one product, kept
//! whole (H_eff, ½ and the h.c.) or put whole in the channel of its fast index (the jumps).

use std::collections::BTreeMap;

use super::spectrum::Spectrum;
use crate::floquet::Sectors;

/// The folded blocks tabled in the tweezer's intensity (fold_table.rs).
#[path = "fold_table.rs"]
pub(crate) mod table;
use crate::linalg::{self, C64, Mat};
use crate::scene::Key;
use crate::units::UNIT_ROUNDOFF;

/// A complex number moved, zeroed or added into another, and a real number's operation, in complex multiply-adds
/// (decide.rs, # Work): the fold's and the contraction's element-wise loops are counted in them.
const MOVED: f64 = crate::decide::MOVED / crate::decide::COMPLEX;
const REAL: f64 = 1. / crate::decide::COMPLEX;
/// A complex reciprocal 1/(z − w − κ) (`linalg::reciprocals`): the two differences, |z|², its division, the two products.
const RECIPROCAL: f64 = (7. + crate::decide::DIVISION) / crate::decide::COMPLEX;

/// A term: `scale` × factor × Σ coefficient · matrix (each matrix's adjoint where `adjoint`).
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub factor: usize,
    pub adjoint: bool,
    pub scale: C64,
    pub matrices: Vec<(usize, usize)>,
}

impl Term {
    pub fn one(factor: usize, matrix: usize) -> Term {
        Term { factor, adjoint: false, scale: C64::ONE, matrices: vec![(matrix, 0)] }
    }
}

/// A frequency component of a source operator: Σ terms at `key`'s frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub key: Key,
    pub terms: Vec<Term>,
}

/// A source jump channel: its emission axis, its fast index, its components and its photons' forms.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceChannel {
    pub emission: usize,
    pub fast: [i64; 2],
    pub parts: Vec<Piece>,
    pub kicks: Vec<Piece>,
}

/// What a fold folds, by structure, on the 32 states (indices into a library of fixed matrices): H's components (each
/// nonzero key both ways), the jump channels, the terms of the generator the fold is taken at (G − (i/2)Γ), and whether
/// each factor can vary along a path (the generator's varying factors make Ĝ).
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub hamiltonian: Vec<Piece>,
    pub jumps: Vec<SourceChannel>,
    pub generator: Vec<Term>,
    pub varying: Vec<bool>,
    pub coefficients: usize,
    /// The factors that are the tweezer's intensity s (its unit fold's `Power` first, then its own outside form's
    /// |a_tw|²): the folded blocks' tables are in s (fold_table.rs). None in a check's source.
    pub intensity: Vec<usize>,
    /// The factors that are another beam's own intensity |a_l|² and vary (a Gaussian beam's outside form's, one at
    /// its focus): the tables take each as a small part about one (fold_table.rs, # The small parts). None in a
    /// check's source.
    pub own: Vec<usize>,
}

impl Source {
    /// Every piece in the fold's order: H's, then each channel's parts and kicks; with (channel, which) of a jump's.
    pub fn pieces(&self) -> impl Iterator<Item = (&Piece, Option<(usize, usize)>)> {
        self.hamiltonian.iter().map(|p| (p, None)).chain(self.jumps.iter().enumerate().flat_map(|(c, ch)| ch.parts.iter().map(move |p| (p, Some((c, 0)))).chain(ch.kicks.iter().map(move |p| (p, Some((c, 1)))))))
    }
}

/// The numbers a source takes at a point: each factor's value and gradient, each coefficient (coefficient 0 is one),
/// each piece's frequency (`Source::pieces`' order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Values {
    pub factors: Vec<(C64, [C64; 3])>,
    pub coefficients: Vec<f64>,
    pub frequencies: Vec<f64>,
}

/// A routed entry: row, column (positions in what it falls in), its term, coefficient and value.
#[derive(Clone, Copy, Debug)]
struct Entry {
    row: u32,
    col: u32,
    term: u32,
    coefficient: u32,
    value: C64,
}

/// A routed sparse matrix: its rows and columns (positions in K or in a block) and its entries by those indices; the
/// single factor of its terms where they share one.
#[derive(Clone, Debug, Default)]
pub(crate) struct Routed {
    block: usize,
    piece: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    entries: Vec<Entry>,
    single: Option<usize>,
    /// The entries a point puts in (`Batch::fixed` holds the others, made once): a coefficient other than the first,
    /// or a factor that varies along a path under a component of more than one factor.
    rest: Vec<Entry>,
}

/// A block's amplitudes in one batch (`Plan::batches`): their indices, per one its column offsets in the batch's values
/// and in its X (one factor: |cols| columns; else 4|cols|, X and ∂X), and the batch's widths.
#[derive(Clone, Debug, Default)]
struct Batch {
    amplitudes: Vec<usize>,
    offsets: Vec<(usize, usize)>,
    total: usize,
    wide: usize,
    /// X's part no point changes (|B| × wide, as `Local::evaluate` lays X), made once (module documentation, # Structure
    /// once): of a component of one factor its entries at the first coefficient (the factor comes after V⁻¹X), of one of
    /// more its entries of a factor that does not vary, at the first coefficient, times the term's scale.
    fixed: Vec<C64>,
}

/// A folded block: Q's (its rows Q's indices) or a sideband part of K (its rows K's positions, its fast part).
#[derive(Clone, Debug)]
struct Block {
    rows: Vec<usize>,
    sideband: Option<[i64; 2]>,
    /// The generator's entries on the block (row, column in the block), and those of its varying factors apart.
    generator: Vec<Entry>,
    varying: Vec<(usize, Vec<Entry>)>,
    /// G₀'s part no point changes (|B| × |B|), made once: the entries of a factor that does not vary at the first
    /// coefficient, times the term's scale (such a factor is one, `rs.rs` `Factor::varying`); and the entries a point
    /// puts in, the others.
    fixed: Vec<C64>,
    rest: Vec<Entry>,
    /// A sideband part of K left out (`Plan::drop_sideband`): its spectrum, amplitudes and pairs not made.
    dropped: bool,
}

/// The fold's structure for a kept set K (module documentation, # Structure once).
#[derive(Clone, Debug)]
pub struct Plan {
    pub n: usize,
    pub nq: usize,
    /// K, Q and the levels over the 32 (a test's model, rs.rs `model_of`).
    #[cfg(test)]
    pub keep: Vec<usize>,
    #[cfg(test)]
    pub folded: Vec<usize>,
    #[cfg(test)]
    pub groups: Vec<Vec<usize>>,
    parts: Vec<[i64; 2]>,
    /// Per piece (`Source::pieces`' order): its key, its class (fast index of its key), its terms' factors and scales,
    /// and (channel, which) of a jump's.
    keys: Vec<Key>,
    classes: Vec<[i64; 2]>,
    terms: Vec<Vec<(usize, C64)>>,
    jump: Vec<Option<(usize, usize)>>,
    channels: Vec<(usize, [i64; 2])>,
    hamiltonian: usize,
    generator_terms: Vec<(usize, C64)>,
    /// Per factor whether it can vary along a path (`Source::varying`): the generator's varying factors are the
    /// intensities the reach's bound ranges over (`Plan::reach`).
    varying: Vec<bool>,
    /// Per piece where its terms start in the flat list of every piece's terms.
    offsets: Vec<usize>,
    blocks: Vec<Block>,
    /// The generator's entries on K's diagonal: (position, entry) -- E_a = Re of their sum.
    diagonal: Vec<Entry>,
    pub(crate) amplitudes: Vec<Routed>,
    /// Per block its amplitudes in one batch, and per amplitude (block, column offset in its batch).
    batches: Vec<Batch>,
    place: Vec<(usize, usize)>,
    legs: Vec<Routed>,
    /// Per piece its K × K entries: H's slow ones, a jump's all with their fast index.
    direct: Vec<Vec<(Entry, [i64; 2])>>,
    /// The factors that are the tweezer's intensity (`Source::intensity`) and the beams' own (`Source::own`), and the
    /// blocks' tables in s, made at the plan's first evaluation (`Plan::table`).
    intensity: Vec<usize>,
    own: Vec<usize>,
    table: std::sync::OnceLock<std::sync::Arc<table::Table>>,
}

fn element(class: [i64; 2], row: [i64; 2], col: [i64; 2]) -> [i64; 2] {
    Sectors::element(class, row, col)
}

impl Plan {
    /// The structure of `source`'s fold keeping `keep` (sorted) by `sectors` on the fast coordinates `axes`, Q the rest;
    /// `splits` the library's fixed matrices split for this K and Q (rs.rs `Split::all`).
    pub fn new(splits: &[super::Split], keep: &[usize], folded: &[usize], source: &Source, sectors: &Sectors, axes: [bool; 2], #[cfg_attr(not(test), allow(unused_variables))] groups: &[Vec<usize>]) -> Plan {
        let (n, nq) = (keep.len(), folded.len());
        let parts: Vec<[i64; 2]> = keep.iter().map(|&s| sectors.state(s, axes)).collect();
        let mut keys = Vec::new();
        let mut classes = Vec::new();
        let mut terms = Vec::new();
        let mut jump = Vec::new();
        for (piece, which) in source.pieces() {
            keys.push(piece.key.clone());
            classes.push(sectors.key(&piece.key, axes));
            terms.push(piece.terms.iter().map(|t| (t.factor, t.scale)).collect::<Vec<_>>());
            jump.push(which);
        }
        let hamiltonian = source.hamiltonian.len();
        let channels = source.jumps.iter().map(|c| (c.emission, c.fast)).collect();
        // Each entry of a term's matrices, transposed and conjugated for an adjoint term: (i, j, coefficient, value) by
        // where it falls: K × K by δ = n_i − n_j, Q × K, K × Q, Q × Q.
        let each = |t: &Term, f: &mut dyn FnMut(Fall, usize, usize, usize, C64)| {
            for &(m, c) in &t.matrices {
                let s = &splits[m];
                let conj = |v: C64| if t.adjoint { v.conj() } else { v };
                for (delta, list) in &s.kk {
                    let delta = if t.adjoint { [-delta[0], -delta[1]] } else { *delta };
                    for &(i, j, v) in list {
                        let (i, j) = if t.adjoint { (j, i) } else { (i, j) };
                        f(Fall::KK(delta), i, j, c, conj(v));
                    }
                }
                let (qk, kq) = if t.adjoint { (&s.kq, &s.qk) } else { (&s.qk, &s.kq) };
                for &(i, j, v) in qk {
                    let (i, j) = if t.adjoint { (j, i) } else { (i, j) };
                    f(Fall::QK, i, j, c, conj(v));
                }
                for &(i, j, v) in kq {
                    let (i, j) = if t.adjoint { (j, i) } else { (i, j) };
                    f(Fall::KQ, i, j, c, conj(v));
                }
                for &(i, j, v) in &s.qq {
                    let (i, j) = if t.adjoint { (j, i) } else { (i, j) };
                    f(Fall::QQ, i, j, c, conj(v));
                }
            }
        };
        // The blocks: Q's connected parts under the generator's pattern; K's states by fast part where some class has a
        // fast element into them, each split into its connected parts under the generator's K × K pattern.
        let (mut qq, mut kk) = (Vec::new(), Vec::new());
        for t in &source.generator {
            each(t, &mut |fall, i, j, _, _| match fall {
                Fall::QQ => qq.push((i, j)),
                Fall::KK(_) => kk.push((i, j)),
                _ => {}
            });
        }
        let mut blocks: Vec<Block> = Vec::new();
        let (mut q_of, mut s_of) = (vec![(usize::MAX, usize::MAX); nq], vec![(usize::MAX, usize::MAX); n]);
        for members in connected_parts(nq, &qq) {
            for (r, &i) in members.iter().enumerate() {
                q_of[i] = (blocks.len(), r);
            }
            blocks.push(Block { rows: members, sideband: None, generator: Vec::new(), varying: Vec::new(), fixed: Vec::new(), rest: Vec::new(), dropped: false });
        }
        let piece_classes: Vec<[i64; 2]> = classes[..hamiltonian].to_vec();
        let mut by_part: BTreeMap<[i64; 2], Vec<usize>> = BTreeMap::new();
        for (a, &s) in parts.iter().enumerate() {
            by_part.entry(s).or_default().push(a);
        }
        for (s, rows) in &by_part {
            if !piece_classes.iter().any(|&k| parts.iter().any(|&na| element(k, *s, na) != [0; 2])) {
                continue;
            }
            let place: BTreeMap<usize, usize> = rows.iter().enumerate().map(|(i, &a)| (a, i)).collect();
            let within: Vec<(usize, usize)> = kk.iter().filter_map(|&(i, j)| Some((*place.get(&i)?, *place.get(&j)?))).collect();
            for members in connected_parts(rows.len(), &within) {
                let sub: Vec<usize> = members.iter().map(|&i| rows[i]).collect();
                for (r, &a) in sub.iter().enumerate() {
                    s_of[a] = (blocks.len(), r);
                }
                blocks.push(Block { rows: sub, sideband: Some(*s), generator: Vec::new(), varying: Vec::new(), fixed: Vec::new(), rest: Vec::new(), dropped: false });
            }
        }
        // The generator's entries by block, and K's diagonal.
        let mut diagonal = Vec::new();
        for (ti, t) in source.generator.iter().enumerate() {
            let varying = source.varying[t.factor];
            each(t, &mut |fall, i, j, c, v| {
                let entry = |row: usize, col: usize| Entry { row: row as u32, col: col as u32, term: ti as u32, coefficient: c as u32, value: v };
                match fall {
                    Fall::QQ => {
                        let ((b, r), (b2, c2)) = (q_of[i], q_of[j]);
                        debug_assert_eq!(b, b2);
                        blocks[b].generator.push(entry(r, c2));
                        if varying {
                            push_varying(&mut blocks[b].varying, t.factor, entry(r, c2));
                        }
                    }
                    Fall::KK(_) => {
                        if i == j {
                            diagonal.push(entry(i, i));
                        }
                        let ((b, r), (b2, c2)) = (s_of[i], s_of[j]);
                        if b != usize::MAX && b == b2 {
                            blocks[b].generator.push(entry(r, c2));
                            if varying {
                                push_varying(&mut blocks[b].varying, t.factor, entry(r, c2));
                            }
                        }
                    }
                    _ => {}
                }
            });
        }
        let generator_terms: Vec<(usize, C64)> = source.generator.iter().map(|t| (t.factor, t.scale)).collect();
        // The amplitudes (H's pieces) and the legs (every piece) by block, the direct K × K entries.
        let order = |a: usize| (parts[a], a);
        let mut amplitudes = Vec::new();
        let mut legs = Vec::new();
        let mut direct = Vec::new();
        for (index, (piece, which)) in source.pieces().enumerate() {
            let class = classes[index];
            let is_h = which.is_none();
            let mut amp: BTreeMap<usize, Vec<(usize, usize, usize, usize, C64)>> = BTreeMap::new();
            let mut leg: BTreeMap<usize, Vec<(usize, usize, usize, usize, C64)>> = BTreeMap::new();
            let mut own = Vec::new();
            for (ti, t) in piece.terms.iter().enumerate() {
                each(t, &mut |fall, i, j, c, v| match fall {
                    Fall::KK(delta) => {
                        let fast = delta != class;
                        if is_h && !fast {
                            own.push((Entry { row: i as u32, col: j as u32, term: ti as u32, coefficient: c as u32, value: v }, [0; 2]));
                        }
                        if !is_h {
                            own.push((Entry { row: i as u32, col: j as u32, term: ti as u32, coefficient: c as u32, value: v }, element(class, parts[i], parts[j])));
                        }
                        // A fast entry into a sideband part (an H piece's amplitude); a leg out of one (H's fast entries,
                        // a jump's every entry).
                        if is_h && fast {
                            let (b, r) = s_of[i];
                            if b != usize::MAX {
                                amp.entry(b).or_default().push((r, j, ti, c, v));
                            }
                        }
                        if !is_h || fast {
                            let (b, r) = s_of[j];
                            if b != usize::MAX {
                                leg.entry(b).or_default().push((i, r, ti, c, v));
                            }
                        }
                    }
                    Fall::QK => {
                        if is_h {
                            let (b, r) = q_of[i];
                            amp.entry(b).or_default().push((r, j, ti, c, v));
                        }
                    }
                    Fall::KQ => {
                        let (b, r) = q_of[j];
                        leg.entry(b).or_default().push((i, r, ti, c, v));
                    }
                    Fall::QQ => {}
                });
            }
            let single = piece.terms.first().map(|t| t.factor).filter(|&f| piece.terms.iter().all(|t| t.factor == f && t.scale == piece.terms[0].scale));
            for (b, list) in amp {
                let mut cols: Vec<usize> = list.iter().map(|e| e.1).collect();
                cols.sort_by_key(|&a| order(a));
                cols.dedup();
                let entries = list.iter().map(|&(r, a, t, c, v)| Entry { row: r as u32, col: cols.iter().position(|&x| x == a).unwrap() as u32, term: t as u32, coefficient: c as u32, value: v }).collect();
                amplitudes.push(Routed { block: b, piece: index, rows: Vec::new(), cols, entries, single, rest: Vec::new() });
            }
            for (b, list) in leg {
                let mut rows: Vec<usize> = list.iter().map(|e| e.0).collect();
                rows.sort_by_key(|&a| order(a));
                rows.dedup();
                let entries = list.iter().map(|&(i, r, t, c, v)| Entry { row: rows.iter().position(|&x| x == i).unwrap() as u32, col: r as u32, term: t as u32, coefficient: c as u32, value: v }).collect();
                legs.push(Routed { block: b, piece: index, rows, cols: Vec::new(), entries, single, rest: Vec::new() });
            }
            direct.push(own);
        }
        let offsets = terms.iter().scan(0, |at, ts: &Vec<(usize, C64)>| {
            let here = *at;
            *at += ts.len();
            Some(here)
        }).collect();
        let mut batches: Vec<Batch> = (0..blocks.len()).map(|_| Batch::default()).collect();
        let mut place = Vec::with_capacity(amplitudes.len());
        for (i, r) in amplitudes.iter().enumerate() {
            let batch = &mut batches[r.block];
            let count = r.cols.len();
            place.push((r.block, batch.total));
            batch.amplitudes.push(i);
            batch.offsets.push((batch.total, batch.wide));
            batch.total += count;
            batch.wide += if r.single.is_some() { count } else { 4 * count };
        }
        // The parts no point changes, made once (`Block::fixed`, `Batch::fixed`): G₀'s and X's entries of a factor that
        // does not vary at the first coefficient; the rest a point puts in.
        for block in blocks.iter_mut() {
            let size = block.rows.len();
            let mut fixed = vec![C64::ZERO; size * size];
            for e in &block.generator {
                let (f, scale) = generator_terms[e.term as usize];
                if e.coefficient == 0 && !source.varying[f] {
                    fixed[e.row as usize * size + e.col as usize] += scale * e.value;
                } else {
                    block.rest.push(*e);
                }
            }
            block.fixed = fixed;
        }
        for (b, batch) in batches.iter_mut().enumerate() {
            let size = blocks[b].rows.len();
            let mut fixed = vec![C64::ZERO; size * batch.wide];
            for (&i, &(_, xoff)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &mut amplitudes[i];
                let mut rest = Vec::new();
                for e in &r.entries {
                    let at = e.row as usize * batch.wide + xoff + e.col as usize;
                    let (f, scale) = terms[r.piece][e.term as usize];
                    if e.coefficient == 0 && r.single.is_some() {
                        fixed[at] += e.value;
                    } else if e.coefficient == 0 && !source.varying[f] {
                        fixed[at] += scale * e.value;
                    } else {
                        rest.push(*e);
                    }
                }
                r.rest = rest;
            }
            batch.fixed = fixed;
        }
        Plan {
            n,
            nq,
            #[cfg(test)]
            keep: keep.to_vec(),
            #[cfg(test)]
            folded: folded.to_vec(),
            #[cfg(test)]
            groups: groups.to_vec(),
            parts, keys, classes, terms, jump, channels, hamiltonian, generator_terms, varying: source.varying.clone(), offsets, blocks, diagonal, amplitudes, batches, place, legs, direct, intensity: source.intensity.clone(), own: source.own.clone(), table: std::sync::OnceLock::new() }
    }

    /// The folded blocks' sizes and which are Q's.
    pub fn blocks(&self) -> Vec<(usize, bool)> {
        self.blocks.iter().map(|b| (b.rows.len(), b.sideband.is_none())).collect()
    }

    /// K's states' fast parts (`Sectors::state` on the folding coordinates), in K's order.
    pub(crate) fn parts(&self) -> &[[i64; 2]] {
        &self.parts
    }

    /// K's sideband parts (`Block::sideband`): their blocks' indices.
    pub(crate) fn sidebands(&self) -> Vec<usize> {
        (0..self.blocks.len()).filter(|&b| self.blocks[b].sideband.is_some()).collect()
    }

    /// K's sideband part `b` left out of the fold (`out`) or put back: what it adds -- its pairs into H_eff and the
    /// channels, values, gradients and photon forms -- is then not made (`Local::evaluate`, `Out::contract`), nor
    /// bounded (`Out::bounds`); its caller books it (model/selected.rs, # The sideband parts dropped).
    pub(crate) fn drop_sideband(&mut self, b: usize, out: bool) {
        assert!(self.blocks[b].sideband.is_some(), "only K's sideband parts are left out");
        self.blocks[b].dropped = out;
    }

    /// Whether K's sideband part `b` is left out (`drop_sideband`; the tests').
    #[cfg(test)]
    pub(crate) fn dropped(&self, b: usize) -> bool {
        self.blocks[b].dropped
    }

    /// Whether the block `b` is made where `model` (K's sideband parts only where a model is made, and not left out).
    fn made(&self, b: usize, model: bool) -> bool {
        let block = &self.blocks[b];
        block.sideband.is_none() || (model && !block.dropped)
    }

    #[cfg(test)]
    /// The count of pieces (H's and the jumps').
    pub fn pieces(&self) -> usize {
        self.keys.len()
    }

    /// The pieces' keys (`Source::pieces`' order).
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// The folded blocks' tables in the tweezer's intensity (fold_table.rs), made where first asked for, at the values
    /// of that point (`table::BlockTable::new`: their active small parts count a node's work).
    pub(crate) fn table(&self, values: &Values) -> &table::Table {
        self.table.get_or_init(|| std::sync::Arc::new(table::Table::new(self, values)))
    }
}

/// Where an entry of a term falls in a fold of K and Q.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fall {
    KK([i64; 2]),
    QK,
    KQ,
    QQ,
}

fn push_varying(list: &mut Vec<(usize, Vec<Entry>)>, factor: usize, e: Entry) {
    match list.iter_mut().find(|x| x.0 == factor) {
        Some(x) => x.1.push(e),
        None => list.push((factor, vec![e])),
    }
}

/// The connected parts of `size` rows under the pattern of the entries (i, j).
pub fn connected_parts(size: usize, entries: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut root: Vec<usize> = (0..size).collect();
    fn find(root: &mut [usize], mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    for &(i, k) in entries {
        if i != k {
            let (a, b) = (find(&mut root, i), find(&mut root, k));
            root[a.max(b)] = a.min(b);
        }
    }
    let mut parts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..size {
        let r = find(&mut root, i);
        parts.entry(r).or_default().push(i);
    }
    parts.into_values().collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// Groupings.

/// A sum of amplitudes (Z, |B| × |cols|) or legs (W, |rows| × |B|) of one block, class, group and fast part of its kept
/// states: its K positions and its members (amplitude or leg, and per member index its place in the sum).
#[derive(Clone, Debug)]
struct Sum {
    block: usize,
    class: [i64; 2],
    group: usize,
    part: [i64; 2],
    index: Vec<usize>,
    members: Vec<(usize, Vec<(usize, usize)>)>,
    /// A jump's leg sum: its (channel, which).
    jump: Option<(usize, usize)>,
    /// Its place in its list.
    at: usize,
}

/// A grouping of a fold's components (rs.rs, # Contractions): each piece's group; the sums; the pairs, each with the
/// output it writes; the outputs (H's by key, the channels' by (channel, which, key)), and their structural supports.
#[derive(Clone, Debug)]
pub struct Grouping {
    /// Per group its label (a key) and the piece whose frequency it carries.
    pub labels: Vec<(Key, usize)>,
    zs: Vec<Sum>,
    ws: Vec<Sum>,
    /// H's pairs and the jumps'.
    h_pairs: Vec<HPair>,
    j_pairs: Vec<JPair>,
    /// H's outputs: key, frequency's pieces (summed); the channels' outputs: (channel's emission, fast index, which, key,
    /// frequency's pieces).
    pub outputs: Vec<(Key, Vec<(usize, f64)>)>,
    pub channel_outputs: Vec<(usize, [i64; 2], usize, Key, Vec<usize>)>,
    /// Per H piece its direct entries' output and their places in it; per jump piece per direct entry its channel
    /// output and place.
    h_direct: Vec<usize>,
    h_direct_places: Vec<Vec<u32>>,
    j_direct: Vec<Vec<(usize, u32)>>,
    /// Each output's and channel output's structural elements (places b·n + a, in order): its numbers' order.
    pub lists: Vec<Vec<u32>>,
    pub channel_lists: Vec<Vec<u32>>,
    /// The lift's sums: per Q block and group the amplitude sums there.
    lift: Vec<(usize, usize, Vec<usize>)>,
    /// Per leg sum its members (legs) with each member row's row in the sum (u32::MAX: another sum's).
    wmaps: Vec<Vec<(usize, Vec<u32>)>>,
    /// Per block its amplitude sums side by side (`Out`'s M): their indices and their columns together C; per sum its
    /// first column there.
    layout: Vec<(Vec<usize>, usize)>,
    z_offset: Vec<usize>,
    /// The direct entries spanned per output (`Grouping::spanned`), taken in place of the entries one by one where there
    /// are any.
    pub(crate) spans: Vec<Span>,
}

/// One output's direct entries spanned (`Grouping::spanned`): H's output (`h`) or a channel's (`which` its value 0 or its
/// photon form 1), its index, its places; per atom -- a piece's term at a coefficient, the fixed vector of its entries
/// over the places -- its coordinates on the basis; the basis.
#[derive(Clone, Debug)]
pub(crate) struct Span {
    pub h: bool,
    pub out: usize,
    pub which: usize,
    pub places: Vec<usize>,
    pub atoms: Vec<(usize, usize, usize, Vec<C64>)>,
    pub basis: Vec<Vec<C64>>,
}

/// A span of the direct entries (`Grouping::spanned`): per output its span, and per output (H's or a channel's, its
/// index) the bound of what it leaves out over the reach -- Σ ‖r‖₂ b over the atoms whose residual r was left, b the
/// atom's point coefficient's bound: the dropped entries' Frobenius norm, which bounds their sup norm --; its counted work at a node (complex multiply-adds: the coordinates and the basis, values and gradients)
/// against the entries' one by one.
pub(crate) struct Spanned {
    pub spans: Vec<Span>,
    pub dropped: Vec<(bool, usize, f64)>,
    pub work: [f64; 2],
}

/// An H pair (`Grouping`): its sums, its output and its conjugate's, and per element (i, j) of W Z its place in each.
#[derive(Clone, Debug)]
struct HPair {
    w: usize,
    z: usize,
    out: usize,
    conjugate: usize,
    places: Vec<u32>,
    conjugate_places: Vec<u32>,
}

/// A jump pair (`Grouping`): its sums, its channel output, and per element its place there.
#[derive(Clone, Debug)]
struct JPair {
    w: usize,
    z: usize,
    out: usize,
    places: Vec<u32>,
}

/// The structural elements of a support (n² flags) in order, and each place's index among them.
fn listed(support: &[bool]) -> (Vec<u32>, Vec<u32>) {
    let mut index = vec![u32::MAX; support.len()];
    let mut list = Vec::new();
    for (x, &s) in support.iter().enumerate() {
        if s {
            index[x] = list.len() as u32;
            list.push(x as u32);
        }
    }
    (list, index)
}

fn sum_keys(a: &Key, b: &Key) -> Key {
    let length = a.len().max(b.len());
    (0..length).map(|i| a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0)).collect()
}

fn negated(key: &Key) -> Key {
    key.iter().map(|k| -k).collect()
}

impl Grouping {
    /// The grouping that puts each piece (`Plan`'s order) in the group of its label, groups by equal labels in the order
    /// of their first piece.
    pub fn new(plan: &Plan, label: &[Key]) -> Grouping {
        let mut labels: Vec<(Key, usize)> = Vec::new();
        let of: Vec<usize> = label
            .iter()
            .enumerate()
            .map(|(p, l)| match labels.iter().position(|x| &x.0 == l) {
                Some(i) => i,
                None => {
                    labels.push((l.clone(), p));
                    labels.len() - 1
                }
            })
            .collect();
        let parts = &plan.parts;
        let place = |sums: &mut Vec<Sum>, r: &Routed, index: &[usize], jump: Option<(usize, usize)>| {
            let (class, group) = (plan.classes[r.piece], of[r.piece]);
            let mut by: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
            for (k, &a) in index.iter().enumerate() {
                let part = parts[a];
                let at = match sums.iter().position(|s| s.block == r.block && s.class == class && s.group == group && s.part == part && s.jump == jump) {
                    Some(i) => i,
                    None => {
                        sums.push(Sum { block: r.block, class, group, part, index: Vec::new(), members: Vec::new(), jump, at: sums.len() });
                        sums.len() - 1
                    }
                };
                let s = &mut sums[at];
                let slot = match s.index.iter().position(|&x| x == a) {
                    Some(i) => i,
                    None => {
                        s.index.push(a);
                        s.index.len() - 1
                    }
                };
                match by.iter_mut().find(|x| x.0 == at) {
                    Some(x) => x.1.push((k, slot)),
                    None => by.push((at, vec![(k, slot)])),
                }
            }
            by
        };
        let mut zs: Vec<Sum> = Vec::new();
        for (i, r) in plan.amplitudes.iter().enumerate() {
            for (at, map) in place(&mut zs, r, &r.cols, None) {
                zs[at].members.push((i, map));
            }
        }
        let mut ws: Vec<Sum> = Vec::new();
        for (i, r) in plan.legs.iter().enumerate() {
            for (at, map) in place(&mut ws, r, &r.rows, plan.jump[r.piece]) {
                ws[at].members.push((i, map));
            }
        }
        // Sort each sum's index (K positions) and remap its members.
        for s in zs.iter_mut().chain(ws.iter_mut()) {
            let mut order: Vec<usize> = (0..s.index.len()).collect();
            order.sort_by_key(|&i| s.index[i]);
            let mut new_of = vec![0; order.len()];
            for (new, &old) in order.iter().enumerate() {
                new_of[old] = new;
            }
            s.index = order.iter().map(|&i| s.index[i]).collect();
            for (_, map) in s.members.iter_mut() {
                for x in map.iter_mut() {
                    x.1 = new_of[x.1];
                }
            }
        }
        let n = plan.n;
        let mut outputs: Vec<(Key, Vec<(usize, f64)>)> = Vec::new();
        let mut support: Vec<Vec<bool>> = Vec::new();
        let output = |outputs: &mut Vec<(Key, Vec<(usize, f64)>)>, support: &mut Vec<Vec<bool>>, key: Key, pieces: Vec<(usize, f64)>| -> usize {
            match outputs.iter().position(|o| o.0 == key) {
                Some(i) => i,
                None => {
                    outputs.push((key, pieces));
                    support.push(vec![false; n * n]);
                    outputs.len() - 1
                }
            }
        };
        let mut h_pairs = Vec::new();
        for (wi, w) in ws.iter().enumerate().filter(|(_, w)| w.jump.is_none()) {
            for (zi, z) in zs.iter().enumerate().filter(|(_, z)| z.block == w.block) {
                if element([w.class[0] + z.class[0], w.class[1] + z.class[1]], w.part, z.part) != [0; 2] {
                    continue;
                }
                let key = sum_keys(&labels[z.group].0, &labels[w.group].0);
                let pieces = vec![(labels[z.group].1, 1.0), (labels[w.group].1, 1.0)];
                let o = output(&mut outputs, &mut support, key.clone(), pieces);
                let c = output(&mut outputs, &mut support, negated(&key), vec![(labels[z.group].1, -1.0), (labels[w.group].1, -1.0)]);
                for &b in &w.index {
                    for &a in &z.index {
                        support[o][b * n + a] = true;
                        support[c][a * n + b] = true;
                    }
                }
                h_pairs.push((wi, zi, o, c));
            }
        }
        let mut h_direct = Vec::new();
        for p in 0..plan.hamiltonian {
            let o = output(&mut outputs, &mut support, labels[of[p]].0.clone(), vec![(labels[of[p]].1, 1.0)]);
            for (e, _) in &plan.direct[p] {
                support[o][e.row as usize * n + e.col as usize] = true;
            }
            h_direct.push(o);
        }
        let mut channel_outputs: Vec<(usize, [i64; 2], usize, Key, Vec<usize>)> = Vec::new();
        let mut channel_support: Vec<Vec<bool>> = Vec::new();
        let channel = |outs: &mut Vec<(usize, [i64; 2], usize, Key, Vec<usize>)>, sup: &mut Vec<Vec<bool>>, emission: usize, fast: [i64; 2], which: usize, key: Key, pieces: Vec<usize>| -> usize {
            match outs.iter().position(|o| o.0 == emission && o.1 == fast && o.2 == which && o.3 == key) {
                Some(i) => i,
                None => {
                    outs.push((emission, fast, which, key, pieces));
                    sup.push(vec![false; n * n]);
                    outs.len() - 1
                }
            }
        };
        let mut j_pairs = Vec::new();
        for (wi, w) in ws.iter().enumerate() {
            let Some((ch, which)) = w.jump else { continue };
            let emission = plan.channels[ch].0;
            for (zi, z) in zs.iter().enumerate().filter(|(_, z)| z.block == w.block) {
                let fast = element([w.class[0] + z.class[0], w.class[1] + z.class[1]], w.part, z.part);
                let key = sum_keys(&labels[w.group].0, &labels[z.group].0);
                let o = channel(&mut channel_outputs, &mut channel_support, emission, fast, which, key, vec![labels[w.group].1, labels[z.group].1]);
                for &b in &w.index {
                    for &a in &z.index {
                        channel_support[o][b * n + a] = true;
                    }
                }
                j_pairs.push((wi, zi, o));
            }
        }
        let mut j_direct = Vec::new();
        for p in plan.hamiltonian..plan.keys.len() {
            let (ch, which) = plan.jump[p].expect("a jump's piece");
            let emission = plan.channels[ch].0;
            let mut list = Vec::new();
            for (e, fast) in &plan.direct[p] {
                let o = channel(&mut channel_outputs, &mut channel_support, emission, *fast, which, labels[of[p]].0.clone(), vec![labels[of[p]].1]);
                channel_support[o][e.row as usize * n + e.col as usize] = true;
                list.push(o);
            }
            j_direct.push(list);
        }
        // The outputs' elements in order, and every write's place among them.
        let (lists, index): (Vec<Vec<u32>>, Vec<Vec<u32>>) = support.iter().map(|s| listed(s)).unzip();
        let (channel_lists, channel_index): (Vec<Vec<u32>>, Vec<Vec<u32>>) = channel_support.iter().map(|s| listed(s)).unzip();
        let h_pairs = h_pairs
            .into_iter()
            .map(|(w, z, out, conjugate)| {
                let (wi, zi) = (&ws[w].index, &zs[z].index);
                let places = wi.iter().flat_map(|&b| zi.iter().map(move |&a| (b, a))).map(|(b, a)| index[out][b * n + a]).collect();
                let conjugate_places = wi.iter().flat_map(|&b| zi.iter().map(move |&a| (b, a))).map(|(b, a)| index[conjugate][a * n + b]).collect();
                HPair { w, z, out, conjugate, places, conjugate_places }
            })
            .collect();
        let j_pairs = j_pairs
            .into_iter()
            .map(|(w, z, out)| {
                let (wi, zi) = (&ws[w].index, &zs[z].index);
                let places = wi.iter().flat_map(|&b| zi.iter().map(move |&a| (b, a))).map(|(b, a)| channel_index[out][b * n + a]).collect();
                JPair { w, z, out, places }
            })
            .collect();
        let h_direct_places = (0..plan.hamiltonian).map(|p| plan.direct[p].iter().map(|(e, _)| index[h_direct[p]][e.row as usize * n + e.col as usize]).collect()).collect();
        let j_direct = j_direct
            .into_iter()
            .enumerate()
            .map(|(j, outs)| outs.into_iter().zip(&plan.direct[plan.hamiltonian + j]).map(|(o, (e, _))| (o, channel_index[o][e.row as usize * n + e.col as usize])).collect())
            .collect();
        let mut lift = Vec::new();
        for (b, block) in plan.blocks.iter().enumerate().filter(|(_, b)| b.sideband.is_none()) {
            let _ = block;
            for group in 0..labels.len() {
                let mine: Vec<usize> = (0..zs.len()).filter(|&i| zs[i].block == b && zs[i].group == group).collect();
                if !mine.is_empty() {
                    lift.push((b, group, mine));
                }
            }
        }
        let wmaps = ws
            .iter()
            .map(|s| {
                s.members
                    .iter()
                    .map(|(member, map)| {
                        let mut dense = vec![u32::MAX; plan.legs[*member].rows.len()];
                        for &(from, to) in map {
                            dense[from] = to as u32;
                        }
                        (*member, dense)
                    })
                    .collect()
            })
            .collect();
        let mut layout: Vec<(Vec<usize>, usize)> = vec![(Vec::new(), 0); plan.blocks.len()];
        let mut z_offset = vec![0; zs.len()];
        for (zi, z) in zs.iter().enumerate() {
            let l = &mut layout[z.block];
            z_offset[zi] = l.1;
            l.0.push(zi);
            l.1 += z.index.len();
        }
        Grouping { labels, zs, ws, h_pairs, j_pairs, outputs, channel_outputs, h_direct, h_direct_places, j_direct, lists, channel_lists, lift, wmaps, layout, z_offset, spans: Vec::new() }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The numbers at a point.

/// The fold's numbers at a point (module documentation, # At a point), kept for the contractions; its buffers made once
/// for a plan and reused.
#[derive(Clone, Debug, Default)]
pub struct Local {
    pub values: Values,
    spectra: Vec<Spectrum>,
    /// Per block: G₀; per varying factor σ moving at the point (σ, T̂_σ = V⁻¹G_σV).
    g: Vec<Mat>,
    hats: Vec<Vec<(usize, Mat)>>,
    energies: Vec<f64>,
    slopes: [Vec<f64>; 3],
    /// Per block its batch's M̂ (|B| × total) and ∂M̂ (|B| × 3 total: each row the three axes' rows of the batch in
    /// turn).
    m: Vec<Mat>,
    dm: Vec<Mat>,
    /// Scratch: the generator's and the pieces' terms' factors, a block's denominators, products' operands.
    gen: Vec<(C64, [C64; 3])>,
    terms: Vec<(C64, [C64; 3])>,
    z: Vec<f64>,
    zc: Vec<C64>,
    d: Vec<C64>,
    x: Mat,
    y: Mat,
    t: Mat,
    gs: Mat,
    tables: table::Scratch,
    /// The counted complex multiply-adds of the last evaluation.
    pub work: f64,
}

/// STEP's Doppler census (BRIEF_IMPL line STEP item 8, a prototype's measurement; tests alone): after an evaluation
/// at a moving point, per amplitude piece of every block that holds amplitudes, the shift δ of the piece's frequency
/// against `rest` (the same pieces' frequencies at rest, `Values::frequencies` there) and, over its columns a and its
/// block's eigenvalues λ_k, the largest ratio |δ| / |E_a + f_rest − λ_k| at which a weight of the amplitude stands --
/// the Doppler series Σ_j δ^j (z − λ)^{−(j+1)} converges at that ratio, its tail after the order J within
/// ρ^{J+1}/(1 − ρ) of the leading term -- with that weight |(V⁻¹X)_k| |1/(z − λ_k)| (the term's size in M̂). Returns
/// per piece (block, piece, |δ|, [ρ at the term, its size]) of every term, and the block's size.
#[cfg(test)]
impl Local {
    pub fn doppler(&self, plan: &Plan, rest: &[f64]) -> Vec<(usize, usize, f64, Vec<(f64, f64)>)> {
        let mut out = Vec::new();
        for (b, batch) in plan.batches.iter().enumerate() {
            let s = &self.spectra[b];
            if s.centre.is_empty() || batch.amplitudes.is_empty() {
                continue;
            }
            let size = plan.blocks[b].rows.len();
            for (&i, &(off, _)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &plan.amplitudes[i];
                let delta = self.values.frequencies[r.piece] - rest[r.piece];
                let mut terms = Vec::new();
                for (j, &a) in r.cols.iter().enumerate() {
                    let z = self.energies[a] + rest[r.piece];
                    for k in 0..size {
                        let w = C64::real(z - s.centre[k]) - s.values[k];
                        let at = k * batch.total + off + j;
                        let m = self.m[b].data.get(at).copied().unwrap_or(C64::ZERO);
                        if m != C64::ZERO {
                            terms.push((delta.abs() / w.abs(), m.abs()));
                        }
                    }
                }
                out.push((b, r.piece, delta.abs(), terms));
            }
        }
        out
    }

    /// The pieces' frequencies of the last evaluation (`Values::frequencies`).
    pub fn frequencies(&self) -> &[f64] {
        &self.values.frequencies
    }
}

/// out (r × c) = a b.
fn product(a: &Mat, b: &Mat, out: &mut Mat) {
    assert_eq!(a.cols, b.rows);
    out.rows = a.rows;
    out.cols = b.cols;
    out.data.resize(a.rows * b.cols, C64::ZERO);
    linalg::fused_product(&a.data, &b.data, (a.rows, a.cols, b.cols), &mut out.data);
}

fn sized(m: &mut Mat, rows: usize, cols: usize) {
    m.rows = rows;
    m.cols = cols;
    m.data.clear();
    m.data.resize(rows * cols, C64::ZERO);
}

impl Local {
    /// The fold's numbers at a point: its `values`, the source's there (put in by the caller, `Rs::evaluate`), what
    /// `making` asks for: K's sideband parts where it makes the model, the gradients where it makes them.
    pub fn evaluate(&mut self, plan: &Plan, making: Making) {
        assert!(making.gradients || !making.model, "a model is made with its gradients");
        let mut work = 0.0;
        let Local { values: v, spectra, g, hats, energies, slopes, m: ms, dm: dms, gen, terms, z, zc, d, x, y, t, gs, tables, .. } = self;
        let v = &*v;
        // Each term's factor with its scale: the generator's, then every piece's (flat, `Plan::offsets`).
        let factor = |f: usize, s: C64| -> (C64, [C64; 3]) {
            let (x, g) = v.factors[f];
            (s * x, g.map(|d| s * d))
        };
        gen.clear();
        gen.extend(plan.generator_terms.iter().map(|&(f, s)| factor(f, s)));
        terms.clear();
        terms.extend(plan.terms.iter().flat_map(|ts| ts.iter().map(|&(f, s)| factor(f, s))));
        // Each term's factor and its three gradients times its scale.
        work += 4.0 * (gen.len() + terms.len()) as f64;
        let nb = plan.blocks.len();
        spectra.resize_with(nb, Spectrum::default);
        g.resize_with(nb, Mat::default);
        hats.resize_with(nb, Vec::new);
        // G₀ on each block, its spectrum and T̂_σ: a tabled block's off its table (fold_table.rs), its T̂ the intensity's
        // whole, at the intensity's gradient.
        // (A check's: `table::MODE` 1 and 2 take no table, and make none.)
        #[cfg(test)]
        let mode = table::MODE.with(|m| m.get());
        #[cfg(not(test))]
        let mode = 0;
        let table = (mode == 0).then(|| plan.table(v));
        let s = plan.intensity.first().map_or(0.0, |&f| v.factors[f].0.re);
        for (b, block) in plan.blocks.iter().enumerate() {
            if !plan.made(b, making.model) {
                continue;
            }
            if let Some(tabled) = table.and_then(|t| t.blocks[b].as_deref()) {
                let moving = making.gradients && plan.intensity.first().is_some_and(|&f| v.factors[f].1.iter().any(|d| *d != C64::ZERO));
                if hats[b].is_empty() {
                    hats[b].push((0, Mat::default()));
                }
                if let Some(w) = tabled.fill(s, v, moving, &mut spectra[b], &mut hats[b][0].1, tables) {
                    work += w;
                    let mut used = 0;
                    if moving {
                        hats[b][0].0 = plan.intensity[0];
                        used = 1;
                    }
                    // A beam's own intensity's T̂ on the table's eigenbasis, as the node makes every other varying
                    // factor's (below).
                    if making.gradients {
                        let size = block.rows.len();
                        let sp = &spectra[b];
                        for (f, entries) in block.varying.iter().filter(|(f, _)| plan.own.contains(f)) {
                            if v.factors[*f].1.iter().all(|d| *d == C64::ZERO) {
                                continue;
                            }
                            sized(gs, size, size);
                            for e in entries {
                                let (_, scale) = plan.generator_terms[e.term as usize];
                                gs.data[e.row as usize * size + e.col as usize] += scale * e.value * v.coefficients[e.coefficient as usize];
                            }
                            if hats[b].len() <= used {
                                hats[b].push((0, Mat::default()));
                            }
                            product(gs, &sp.vectors, t);
                            product(&sp.inverse, t, &mut hats[b][used].1);
                            hats[b][used].0 = *f;
                            used += 1;
                            work += 2.0 * (size * size * size) as f64 + MOVED * (size * size) as f64 + entries.len() as f64;
                        }
                    }
                    hats[b].truncate(used);
                    continue;
                }
            }
            let size = block.rows.len();
            (g[b].rows, g[b].cols) = (size, size);
            g[b].data.clear();
            g[b].data.extend_from_slice(&block.fixed);
            for e in &block.rest {
                let (s, _) = gen[e.term as usize];
                let c = e.value * v.coefficients[e.coefficient as usize];
                g[b].data[e.row as usize * size + e.col as usize] += s * c;
            }
            work += MOVED * (size * size) as f64 + block.rest.len() as f64;
            // (A check's: mode 2 moves every entry of G₀ by one unit round-off, its sign by the entry's place -- the
            // rounding's floor of what follows.)
            if mode == 2 {
                for (i, z) in g[b].data.iter_mut().enumerate() {
                    let sign = if (i.wrapping_mul(2654435761) >> 7) & 1 == 0 { 1.0 } else { -1.0 };
                    *z = *z * (1.0 + sign * UNIT_ROUNDOFF);
                }
            }
            let s = &mut spectra[b];
            assert!(s.decompose(&g[b]), "a folded block's G₀ has no eigendecomposition (defective)");
            work += s.work;
            if !making.gradients {
                continue;
            }
            let mut used = 0;
            for (f, entries) in &block.varying {
                if v.factors[*f].1.iter().all(|d| *d == C64::ZERO) {
                    continue;
                }
                sized(gs, size, size);
                for e in entries {
                    let (_, scale) = plan.generator_terms[e.term as usize];
                    gs.data[e.row as usize * size + e.col as usize] += scale * e.value * v.coefficients[e.coefficient as usize];
                }
                if hats[b].len() <= used {
                    hats[b].push((0, Mat::default()));
                }
                product(gs, &s.vectors, t);
                product(&s.inverse, t, &mut hats[b][used].1);
                hats[b][used].0 = *f;
                used += 1;
                work += 2.0 * (size * size * size) as f64 + MOVED * (size * size) as f64 + entries.len() as f64;
            }
            hats[b].truncate(used);
        }
        // K's energies and their slopes.
        energies.clear();
        energies.resize(plan.n, 0.0);
        for s in slopes.iter_mut() {
            s.clear();
            s.resize(plan.n, 0.0);
        }
        for e in &plan.diagonal {
            let (s, ds) = gen[e.term as usize];
            let c = e.value * v.coefficients[e.coefficient as usize];
            energies[e.row as usize] += (s * c).re;
            for (sl, dd) in slopes.iter_mut().zip(ds) {
                sl[e.row as usize] += (dd * c).re;
            }
        }
        // The energies and slopes zeroed; per entry four real parts of a product, each two multiply-adds.
        work += REAL * (4 * plan.n) as f64 + 2.0 * plan.diagonal.len() as f64;
        // The amplitudes, block by block, all of a block's in one batch (`Plan::batches`): X of every component side by
        // side, one product V⁻¹X, the gradients likewise.
        ms.resize_with(nb, Mat::default);
        dms.resize_with(nb, Mat::default);
        for (b, batch) in plan.batches.iter().enumerate() {
            let block = &plan.blocks[b];
            if !plan.made(b, making.model) {
                continue;
            }
            let size = block.rows.len();
            let (total, wide) = (batch.total, batch.wide);
            let (m, dm) = (&mut ms[b], &mut dms[b]);
            sized(m, size, total);
            if making.gradients {
                sized(dm, size, 3 * total);
            }
            work += MOVED * (size * total * if making.gradients { 4 } else { 1 }) as f64;
            if batch.amplitudes.is_empty() {
                continue;
            }
            // X: per component its columns (one factor: X without it; else X and ∂X side by side), and each column's z;
            // its part no point changes made once (`Batch::fixed`).
            (x.rows, x.cols) = (size, wide);
            x.data.clear();
            x.data.extend_from_slice(&batch.fixed);
            z.clear();
            z.resize(total, 0.0);
            for (&i, &(off, xoff)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &plan.amplitudes[i];
                let count = r.cols.len();
                let ts = &terms[plan.offsets[r.piece]..];
                let f = v.frequencies[r.piece];
                for (j, &a) in r.cols.iter().enumerate() {
                    z[off + j] = energies[a] + f;
                }
                for e in &r.rest {
                    let k = v.coefficients[e.coefficient as usize];
                    if k == 0.0 {
                        continue;
                    }
                    let c = e.value * k;
                    let at = e.row as usize * wide + xoff + e.col as usize;
                    if r.single.is_some() {
                        x.data[at] += c;
                    } else {
                        let (s, ds) = ts[e.term as usize];
                        x.data[at] += s * c;
                        if making.gradients {
                            for (k, dd) in ds.iter().enumerate() {
                                x.data[at + (k + 1) * count] += *dd * c;
                            }
                        }
                    }
                }
                // Its columns' z, its entries (a factor's: the value and, where the gradients are made, three more).
                let per = if r.single.is_some() { MOVED } else if making.gradients { 4.0 } else { 1.0 };
                work += REAL * count as f64 + per * r.rest.len() as f64;
            }
            // X's fixed part copied.
            work += MOVED * (size * wide) as f64;
            let s = &spectra[b];
            product(&s.inverse, x, y);
            work += (size * size * wide) as f64;
            // D, M̂ = D ∘ X̂, the gradient's ∂X̂ − M̂∂E.
            d.clear();
            d.resize(size * total, C64::ZERO);
            zc.clear();
            zc.extend(z.iter().map(|&x| C64::real(x)));
            for k in 0..size {
                linalg::reciprocals(zc, s.centre[k], s.values[k], &mut d[k * total..(k + 1) * total]);
            }
            // D zeroed and made, a reciprocal an entry; then per entry of M̂ its product and its place, a single
            // component's factor on X̂ and its three gradients, and the three gradients ∂X̂ − M̂∂E in their places.
            let per = 1.0 + MOVED + if making.gradients { 3.0 + 3.0 * MOVED } else { 0.0 };
            let singles: usize = batch.amplitudes.iter().filter(|&&i| plan.amplitudes[i].single.is_some()).map(|&i| plan.amplitudes[i].cols.len()).sum();
            work += (MOVED + RECIPROCAL + per) * (size * total) as f64 + MOVED * total as f64 + 4.0 * (size * singles) as f64;
            for (&i, &(off, xoff)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &plan.amplitudes[i];
                let count = r.cols.len();
                let single = r.single.map(|_| terms[plan.offsets[r.piece]]);
                for k in 0..size {
                    for j in 0..count {
                        let yv = y.data[k * wide + xoff + j];
                        let (xv, dx): (C64, [C64; 3]) = match single {
                            Some((sv, sd)) => (sv * yv, sd.map(|g| g * yv)),
                            None => (yv, std::array::from_fn(|a| y.data[k * wide + xoff + (a + 1) * count + j])),
                        };
                        let mv = d[k * total + off + j] * xv;
                        m.data[k * total + off + j] = mv;
                        if making.gradients {
                            for a in 0..3 {
                                dm.data[k * 3 * total + a * total + off + j] = dx[a] - mv * slopes[a][r.cols[j]];
                            }
                        }
                    }
                }
            }
            if !making.gradients {
                continue;
            }
            // Σ_σ ∂σ T̂_σ M̂: one product per varying factor for the whole batch, each axis's share onto that axis's
            // row of ∂M̂ at once; then D ∘.
            for (sigma, hat) in &hats[b] {
                product(hat, m, t);
                work += (size * size * total) as f64;
                let ds = v.factors[*sigma].1;
                for k in 0..size {
                    for (a, da) in ds.iter().enumerate() {
                        if *da != C64::ZERO {
                            let to = k * 3 * total + a * total;
                            linalg::mul_add(false, &mut dm.data[to..to + total], *da, &t.data[k * total..(k + 1) * total]);
                        }
                    }
                }
                work += (ds.iter().filter(|d| **d != C64::ZERO).count() * size * total) as f64;
            }
            for k in 0..size {
                let dk = &d[k * total..(k + 1) * total];
                for row in dm.data[k * 3 * total..(k + 1) * 3 * total].chunks_exact_mut(total) {
                    for (x, dd) in row.iter_mut().zip(dk) {
                        *x = *dd * *x;
                    }
                }
            }
            work += (3 * size * total) as f64;
        }
        self.work = work;
    }
}

impl Local {
    /// Per block the rounding scales of its amplitudes M̂ and their gradients ∂M̂ as `evaluate` made them (`Out::scales`,
    /// the model's structure: model/selected.rs # The structure's rounding): each the same sums and products on its
    /// operands' moduli -- a block's M̂ = D ∘ (V⁻¹X) by |D| ∘ (|V⁻¹| |X|), |X| its entries' moduli summed, and
    /// ∂M̂ = D ∘ (V⁻¹∂X − M̂∂E + Σ_σ ∂σ T̂_σ M̂) by |D| ∘ (|V⁻¹| |∂X| + M̂_s |∂E| + Σ_σ |∂σ| |T̂_σ| M̂_s). Rows × total and
    /// rows × 3 total, as `m` and `dm`.
    pub fn shadows(&self, plan: &Plan) -> Vec<(Vec<f64>, Vec<f64>)> {
        let v = &self.values;
        let factor = |f: usize, s: C64| -> (f64, [f64; 3]) {
            let (x, g) = v.factors[f];
            ((s * x).abs(), g.map(|d| (s * d).abs()))
        };
        let terms: Vec<(f64, [f64; 3])> = plan.terms.iter().flat_map(|ts| ts.iter().map(|&(f, s)| factor(f, s))).collect();
        let mut out = Vec::with_capacity(plan.batches.len());
        for (b, batch) in plan.batches.iter().enumerate() {
            let block = &plan.blocks[b];
            let size = block.rows.len();
            let (total, wide) = (batch.total, batch.wide);
            let (m, dm) = (&self.m[b], &self.dm[b]);
            if !plan.made(b, true) {
                out.push((Vec::new(), Vec::new()));
                continue;
            }
            if batch.amplitudes.is_empty() {
                out.push((m.data.iter().map(|z| z.abs()).collect(), dm.data.iter().map(|z| z.abs()).collect()));
                continue;
            }
            // |X| as `evaluate` lays X: per component its columns, X and ∂X side by side where it has more than one factor.
            let mut x = vec![0.0; size * wide];
            let mut z = vec![0.0; total];
            for (&i, &(off, xoff)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &plan.amplitudes[i];
                let count = r.cols.len();
                let ts = &terms[plan.offsets[r.piece]..];
                let f = v.frequencies[r.piece];
                for (j, &a) in r.cols.iter().enumerate() {
                    z[off + j] = self.energies[a] + f;
                }
                for e in &r.entries {
                    let k = v.coefficients[e.coefficient as usize];
                    if k == 0.0 {
                        continue;
                    }
                    let c = (e.value * k).abs();
                    let at = e.row as usize * wide + xoff + e.col as usize;
                    if r.single.is_some() {
                        x[at] += c;
                    } else {
                        let (s, ds) = ts[e.term as usize];
                        x[at] += s * c;
                        for (k, dd) in ds.iter().enumerate() {
                            x[at + (k + 1) * count] += dd * c;
                        }
                    }
                }
            }
            let s = &self.spectra[b];
            let inverse: Vec<f64> = s.inverse.data.iter().map(|z| z.abs()).collect();
            let mut y = vec![0.0; size * wide];
            for k in 0..size {
                for q in 0..size {
                    let w = inverse[k * size + q];
                    if w != 0.0 {
                        for (o, xq) in y[k * wide..(k + 1) * wide].iter_mut().zip(&x[q * wide..(q + 1) * wide]) {
                            *o += w * xq;
                        }
                    }
                }
            }
            let reciprocal = |k: usize, j: usize| 1.0 / (C64::real(z[j]) - C64::real(s.centre[k]) - s.values[k]).abs();
            let mut ms = vec![0.0; size * total];
            let mut dms = vec![0.0; size * 3 * total];
            for (&i, &(off, xoff)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &plan.amplitudes[i];
                let count = r.cols.len();
                let single = r.single.map(|_| terms[plan.offsets[r.piece]]);
                for k in 0..size {
                    for j in 0..count {
                        let yv = y[k * wide + xoff + j];
                        let (xv, dx): (f64, [f64; 3]) = match single {
                            Some((sv, sd)) => (sv * yv, sd.map(|g| g * yv)),
                            None => (yv, std::array::from_fn(|a| y[k * wide + xoff + (a + 1) * count + j])),
                        };
                        let d = reciprocal(k, off + j);
                        let mv = d * xv;
                        ms[k * total + off + j] = mv;
                        for a in 0..3 {
                            dms[k * 3 * total + a * total + off + j] = dx[a] + mv * self.slopes[a][r.cols[j]].abs();
                        }
                    }
                }
            }
            // Σ_σ |∂σ| |T̂_σ| M̂_s, then |D| ∘.
            for (sigma, hat) in &self.hats[b] {
                let hat: Vec<f64> = hat.data.iter().map(|z| z.abs()).collect();
                let ds = v.factors[*sigma].1;
                for (&i, &(off, _)) in batch.amplitudes.iter().zip(&batch.offsets) {
                    let count = plan.amplitudes[i].cols.len();
                    for (a, da) in ds.iter().enumerate() {
                        if *da == C64::ZERO {
                            continue;
                        }
                        for k in 0..size {
                            for q in 0..size {
                                let w = da.abs() * hat[k * size + q];
                                if w != 0.0 {
                                    for j in 0..count {
                                        dms[k * 3 * total + a * total + off + j] += w * ms[q * total + off + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (&i, &(off, _)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let count = plan.amplitudes[i].cols.len();
                for k in 0..size {
                    for a in 0..3 {
                        for j in 0..count {
                            dms[k * 3 * total + a * total + off + j] *= reciprocal(k, off + j);
                        }
                    }
                }
            }
            out.push((ms, dms));
        }
        out
    }
}

/// The Neumann series' workspace (`Series::resolve`).
#[derive(Clone, Debug, Default)]
pub struct Series {
    off: Vec<(usize, usize, C64)>,
    spread: Vec<f64>,
    inverse: Vec<C64>,
    tail: Vec<f64>,
    w: Vec<C64>,
    next: Vec<C64>,
    zc: Vec<C64>,
    wmax: Vec<f64>,
    omax: Vec<f64>,
    /// Where the series does not converge: the block's decomposition and its products.
    eigen: linalg::Eigen,
    product: Mat,
    result: Mat,
}

impl Series {
    /// out (|B| × |cols|) = (z_j − G₀)⁻¹ y_j on each column j of y, the Neumann series about the diagonal (rs.rs, # At a
    /// point): every column summed until its remainder's bound ‖w_k‖ρ/(1 − ρ) is below the rounding of its sum; where
    /// it does not converge (ρ ≥ 1), through the block's decomposition. Returns the counted complex multiply-adds.
    pub fn resolve(&mut self, g: &Mat, z: &[f64], y: &Mat, out: &mut Mat) -> f64 {
        let (size, count) = (y.rows, y.cols);
        self.off.clear();
        self.spread.clear();
        self.spread.resize(size, 0.0);
        for i in 0..size {
            for k in 0..size {
                if i != k && g[(i, k)] != C64::ZERO {
                    self.spread[i] += g[(i, k)].abs();
                    self.off.push((i, k, g[(i, k)]));
                }
            }
        }
        self.inverse.clear();
        self.inverse.resize(size * count, C64::ZERO);
        self.zc.clear();
        self.zc.extend(z.iter().take(count).map(|&x| C64::real(x)));
        for i in 0..size {
            linalg::reciprocals(&self.zc, 0.0, g[(i, i)], &mut self.inverse[i * count..(i + 1) * count]);
        }
        // Per column ρ = max_i spread_i |(z_j − g_ii)⁻¹| and the remainder's factor (ρ/(1 − ρ))², compared in squares.
        self.tail.clear();
        self.tail.resize(count, 0.0);
        for i in 0..size {
            for (t, x) in self.tail.iter_mut().zip(&self.inverse[i * count..(i + 1) * count]) {
                *t = linalg::larger(*t, self.spread[i] * self.spread[i] * x.norm_sqr());
            }
        }
        let converges = self.tail.iter().all(|&r2| r2 < 1.0);
        for t in self.tail.iter_mut() {
            let rho = t.sqrt();
            *t = (rho / (1.0 - rho)).powi(2);
        }
        sized(out, size, count);
        let mut work = 0.0;
        if converges {
            self.w.clear();
            self.w.extend(y.data.iter().zip(&self.inverse).map(|(a, b)| *a * *b));
            out.data.copy_from_slice(&self.w);
            let floor = UNIT_ROUNDOFF * UNIT_ROUNDOFF;
            loop {
                // Per column the largest |w|² and |out|²: done when no column's remainder is known to be above its
                // rounding (a column of NaN is done: the NaN is in `out`, and goes on with it).
                self.wmax.clear();
                self.wmax.resize(count, 0.0);
                self.omax.clear();
                self.omax.resize(count, 0.0);
                for i in 0..size {
                    for j in 0..count {
                        self.wmax[j] = linalg::larger(self.wmax[j], self.w[i * count + j].norm_sqr());
                        self.omax[j] = linalg::larger(self.omax[j], out.data[i * count + j].norm_sqr());
                    }
                }
                if !(0..count).any(|j| self.wmax[j] * self.tail[j] > floor * self.omax[j]) {
                    break;
                }
                self.next.clear();
                self.next.resize(size * count, C64::ZERO);
                for &(i, k, v) in &self.off {
                    linalg::mul_add(false, &mut self.next[i * count..(i + 1) * count], v, &self.w[k * count..(k + 1) * count]);
                }
                work += (self.off.len() * count) as f64;
                std::mem::swap(&mut self.w, &mut self.next);
                for (a, b) in self.w.iter_mut().zip(&self.inverse) {
                    *a *= *b;
                }
                for (o, a) in out.data.iter_mut().zip(&self.w) {
                    *o += *a;
                }
            }
            return work;
        }
        let Series { eigen, product: w, result, .. } = self;
        assert!(eigen.decompose(g), "a sideband part's G₀ has no eigendecomposition (defective)");
        plain_product(&eigen.inverse, y, w);
        for k in 0..size {
            for (j, &zj) in z.iter().enumerate().take(count) {
                w[(k, j)] = w[(k, j)] / (C64::real(zj) - eigen.values[k]);
            }
        }
        plain_product(&eigen.vectors, w, result);
        out.data.copy_from_slice(&result.data);
        work + eigen.work + 2.0 * (size * size * count) as f64
    }
}

/// out = (z_j − G₀)⁻¹ y_j (`Series::resolve` on a workspace of its own).
pub fn neumann_into(g: &Mat, z: &[f64], y: &Mat, out: &mut Mat) -> f64 {
    Series::default().resolve(g, z, y, out)
}

/// The rounding scale of `neumann_into`'s result (rs.rs `Scaled`; model/classify.rs, # The tiers): per entry the sum
/// of the moduli of the terms it is the sum of, the same series on the moduli -- |w₀| = |D⁻¹| ∘ S_y, |w_{k+1}| =
/// |D⁻¹| ∘ (|N| |w_k|), summed until a term no longer moves the sum, S_y the scale of y (|B| × columns, row by row) --
/// or where the series does not converge the decomposition's, |V| ((|V⁻¹| S_y) ⊘ |z − λ|); and the operations' count
/// behind an entry, the terms' chains (one row of N and its product a term).
pub fn neumann_scale(g: &Mat, z: &[f64], y: &[f64]) -> (Vec<f64>, usize) {
    let size = g.rows;
    let count = if size == 0 { 0 } else { y.len() / size };
    let mut off: Vec<(usize, usize, f64)> = Vec::new();
    let mut spread = vec![0.0; size];
    for i in 0..size {
        for k in 0..size {
            if i != k && g[(i, k)] != C64::ZERO {
                spread[i] += g[(i, k)].abs();
                off.push((i, k, g[(i, k)].abs()));
            }
        }
    }
    let inverse: Vec<f64> = (0..size).flat_map(|i| (0..count).map(move |j| (i, j))).map(|(i, j)| 1.0 / (C64::real(z[j]) - g[(i, i)]).abs()).collect();
    let converges = (0..count).all(|j| (0..size).map(|i| spread[i] * inverse[i * count + j]).fold(0.0, linalg::larger) < 1.0);
    let longest = (0..size).map(|i| off.iter().filter(|e| e.0 == i).count()).max().unwrap_or(0) + 2;
    if converges {
        let mut w: Vec<f64> = y.iter().zip(&inverse).map(|(a, b)| a * b).collect();
        let mut out = w.clone();
        let mut steps = 1;
        loop {
            let mut next = vec![0.0; size * count];
            for &(i, k, v) in &off {
                for j in 0..count {
                    next[i * count + j] += v * w[k * count + j];
                }
            }
            w = next.iter().zip(&inverse).map(|(a, b)| a * b).collect();
            let mut moved = false;
            for (o, a) in out.iter_mut().zip(&w) {
                let before = *o;
                *o += a;
                moved |= *o != before;
            }
            steps += 1;
            if !moved {
                return (out, steps * longest);
            }
        }
    }
    let mut eigen = linalg::Eigen::default();
    assert!(eigen.decompose(g), "a block's G₀ has no eigendecomposition (defective)");
    let (v, vi): (Vec<f64>, Vec<f64>) = (eigen.vectors.data.iter().map(|z| z.abs()).collect(), eigen.inverse.data.iter().map(|z| z.abs()).collect());
    let mut w = vec![0.0; size * count];
    for k in 0..size {
        for q in 0..size {
            for j in 0..count {
                w[k * count + j] += vi[k * size + q] * y[q * count + j];
            }
        }
        for j in 0..count {
            w[k * count + j] /= (C64::real(z[j]) - eigen.values[k]).abs();
        }
    }
    let mut out = vec![0.0; size * count];
    for i in 0..size {
        for k in 0..size {
            for j in 0..count {
                out[i * count + j] += v[i * size + k] * w[k * count + j];
            }
        }
    }
    (out, 4 * size)
}

// ---------------------------------------------------------------------------------------------------------------------
// The contraction.

/// What an evaluation and its contraction make (rs.rs, # Contractions): the model (H_eff and the channels, and with it
/// K's sideband parts) -- of it H_eff where `effective` --, the channels' photon forms, the lift, and their gradients
/// along r. A model is made with its gradients; a readout makes the lift's values alone (`READOUT`: Q's blocks, no
/// gradient); a jump's photon the channels with their gradients and photon forms alone (`PHOTONS`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Making {
    pub model: bool,
    pub effective: bool,
    pub kicks: bool,
    pub lift: bool,
    pub gradients: bool,
}

impl Making {
    #[cfg(test)]
    pub const ALL: Making = Making { model: true, effective: true, kicks: true, lift: true, gradients: true };
    /// A node's: the model with its gradients, the photon forms where `kicks`.
    pub const fn node(kicks: bool) -> Making {
        Making { model: true, effective: true, kicks, lift: false, gradients: true }
    }
    /// A readout's: the lift's values (model/selected.rs `Selected::rows`).
    pub const READOUT: Making = Making { model: false, effective: false, kicks: false, lift: true, gradients: false };
    /// A jump's photon's: the channels with their gradients and photon forms, no H_eff (model/selected.rs
    /// `SelectedMiddle::emitted`).
    pub const PHOTONS: Making = Making { model: true, effective: false, kicks: true, lift: false, gradients: true };
}

/// An output's numbers on its structural elements (`Grouping::lists`' order): values and the three gradients; and where
/// asked (`Out::scales`), each element's rounding scale -- the sum of its terms' moduli, the value's and each
/// gradient's, and the terms' count (the operations of the longest chain that made it).
#[derive(Clone, Debug, Default)]
pub struct Compact {
    pub value: Vec<C64>,
    pub gradient: [Vec<C64>; 3],
    pub scale: Vec<f64>,
    pub gradient_scale: [Vec<f64>; 3],
    pub terms: Vec<usize>,
}

impl Compact {
    fn zeroed(&mut self, size: usize) {
        for v in std::iter::once(&mut self.value).chain(self.gradient.iter_mut()) {
            v.clear();
            v.resize(size, C64::ZERO);
        }
    }

    #[cfg(test)]
    /// The output as dense n × n matrices on its places.
    pub fn dense(&self, list: &[u32], n: usize) -> (Mat, [Mat; 3]) {
        let mut value = Mat::zeros(n, n);
        let mut gradient: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(n, n));
        for (k, &x) in list.iter().enumerate() {
            value.data[x as usize] = self.value[k];
            for (g, d) in gradient.iter_mut().zip(&self.gradient) {
                g.data[x as usize] = d[k];
            }
        }
        (value, gradient)
    }
}

/// A contraction's outputs (`Grouping`'s): per H output and per channel output its numbers on its structural elements,
/// per group the lift (|Q| × n) and its gradient; their buffers made once for a grouping and reused.
#[derive(Clone, Debug, Default)]
pub struct Out {
    pub h: Vec<Compact>,
    pub channels: Vec<Compact>,
    pub lift: Vec<(Mat, [Mat; 3])>,
    /// Per block its amplitude sums side by side (`Grouping::layout`), Ẑ in the block's basis and M = V Ẑ in the
    /// states': the values (|B| × C, C their columns together) and the three gradients (|B| × 3C, side by side).
    zv: Vec<Mat>,
    zd: Vec<Mat>,
    mv: Vec<Mat>,
    md: Vec<Mat>,
    /// A leg sum at the point, dense on its rows × the block: X, then ∂X along each axis (4|rows| × |B|, stacked), and
    /// whether it moves (∂X ≠ 0); its products [X; ∂X] M (4|rows| × C) and X ∂M (|rows| × 3C).
    x: Mat,
    moving: bool,
    p: Mat,
    pd: Mat,
    /// A piece's terms' factors at the point with their scales and weight, and whether each moves.
    tf: Vec<(C64, [C64; 3], bool)>,
    /// A spanned output's coordinates at the point, values and gradients (`Grouping::spanned`).
    span: Vec<(C64, [C64; 3])>,
    made: usize,
    pub work: f64,
    /// The moduli `scales` reads: a leg sum's |X| and |∂X| (4|rows| × |B|), its block's |M| and |∂M|, their products.
    xa: Vec<f64>,
    ma: Vec<f64>,
    mda: Vec<f64>,
    pa: Vec<f64>,
    pda: Vec<f64>,
}

fn zero(m: &mut Mat, rows: usize, cols: usize) {
    sized(m, rows, cols);
}

/// A piece's direct entries' counted complex multiply-adds (`Out::contract`): its terms' factors with their scales and
/// weight (a value and three gradients each), and per entry its value and, where its factor moves, three gradients.
fn direct_work(factors: &[(C64, [C64; 3], bool)], entries: &[(Entry, [i64; 2])]) -> f64 {
    4.0 * factors.len() as f64 + entries.iter().map(|(e, _)| if factors[e.term as usize].2 { 4.0 } else { 1.0 }).sum::<f64>()
}

/// out (r × c) = a b by `Mat::product`'s own operations (each row a's entries' multiples of b's rows in turn, the
/// product rounded and then the sum), on out's memory.
fn plain_product(a: &Mat, b: &Mat, out: &mut Mat) {
    assert_eq!(a.cols, b.rows);
    sized(out, a.rows, b.cols);
    for (row, x) in out.data.chunks_exact_mut(b.cols.max(1)).zip(a.data.chunks_exact(a.cols.max(1))) {
        for (x, b) in x.iter().zip(b.data.chunks_exact(b.cols.max(1))) {
            linalg::mul_add(false, row, *x, b);
        }
    }
}

impl Out {
    /// The contraction of `local` by `grouping` (module documentation, # Contraction): each piece's sums with its weight
    /// (`weights`, one per piece, or one for all where None), what `making` asks for. The amplitudes' sums are taken to
    /// the states' basis (M = V Ẑ) and every pair is its legs' sparse fixed entries times M: H⁽ᶜ′⁾_KQ M_c.
    pub fn contract(&mut self, plan: &Plan, grouping: &Grouping, local: &Local, weights: Option<&[C64]>, making: Making) {
        let n = plan.n;
        let weight = |p: usize| weights.map_or(C64::ONE, |w| w[p]);
        let mut work = 0.0;
        self.h.resize_with(grouping.outputs.len(), Default::default);
        self.channels.resize_with(grouping.channel_outputs.len(), Default::default);
        self.lift.resize_with(grouping.labels.len(), Default::default);
        for (c, list) in self.h.iter_mut().zip(&grouping.lists) {
            c.zeroed(list.len());
        }
        for (c, list) in self.channels.iter_mut().zip(&grouping.channel_lists) {
            c.zeroed(list.len());
        }
        // The outputs zeroed: each element its value and three gradients.
        work += 4.0 * MOVED * grouping.lists.iter().chain(&grouping.channel_lists).map(Vec::len).sum::<usize>() as f64;
        self.made = usize::MAX;
        // The amplitudes' sums, block by block side by side, then M = V Ẑ: two products a block.
        let nb = plan.blocks.len();
        for list in [&mut self.zv, &mut self.zd, &mut self.mv, &mut self.md] {
            list.resize_with(nb, Default::default);
        }
        for (b, (list, width)) in grouping.layout.iter().enumerate() {
            if !plan.made(b, making.model) {
                continue;
            }
            let (size, width) = (plan.blocks[b].rows.len(), *width);
            let (zv, zd) = (&mut self.zv[b], &mut self.zd[b]);
            zero(zv, size, width);
            if making.gradients {
                zero(zd, size, 3 * width);
            }
            let sides = if making.gradients { 4 } else { 1 };
            work += MOVED * (sides * size * width) as f64;
            if list.is_empty() {
                continue;
            }
            for &zi in list {
                let (s, o) = (&grouping.zs[zi], grouping.z_offset[zi]);
                for (member, map) in &s.members {
                    let (mb, off) = plan.place[*member];
                    let (a, da) = (&local.m[mb], &local.dm[mb]);
                    let total = a.cols;
                    let w = weight(plan.amplitudes[*member].piece);
                    for k in 0..size {
                        let (row, av) = (&mut zv.data[k * width..(k + 1) * width], &a.data[k * total + off..]);
                        for &(from, to) in map {
                            row[o + to] += w * av[from];
                        }
                        if making.gradients {
                            let (drow, dv) = (&mut zd.data[k * 3 * width..(k + 1) * 3 * width], &da.data[k * 3 * total + off..]);
                            for &(from, to) in map {
                                for d in 0..3 {
                                    drow[d * width + o + to] += w * dv[d * total + from];
                                }
                            }
                        }
                    }
                    // Each of its columns' weight on a row and, with the gradients, on three more.
                    work += (sides * size * map.len()) as f64;
                }
            }
            let vectors = &local.spectra[b].vectors;
            product(vectors, &self.zv[b], &mut self.mv[b]);
            work += (size * size * width) as f64;
            if making.gradients {
                product(vectors, &self.zd[b], &mut self.md[b]);
                work += (size * size * 3 * width) as f64;
            }
        }
        let v = &local.values;
        // A piece's terms' factors with their scales and the piece's weight, once a piece.
        let factors = |tf: &mut Vec<(C64, [C64; 3], bool)>, p: usize| {
            let w = weight(p);
            tf.clear();
            tf.extend(plan.terms[p].iter().map(|&(f, s)| {
                let (x, g) = v.factors[f];
                let c = s * w;
                (x * c, g.map(|d| d * c), g.iter().any(|d| *d != C64::ZERO))
            }));
        };
        // The direct entries spanned (`Grouping::spanned`): per output its coordinates at the point -- each atom's factor,
        // scale, weight and coefficient on its coordinates --, then the basis at them.
        for d in &grouping.spans {
            if (d.h && !(making.model && making.effective)) || (!d.h && (!making.model || (d.which == 1 && !making.kicks))) {
                continue;
            }
            self.span.clear();
            self.span.resize(d.basis.len(), (C64::ZERO, [C64::ZERO; 3]));
            for (piece, term, coefficient, coords) in &d.atoms {
                let k = v.coefficients[*coefficient];
                if k == 0.0 {
                    continue;
                }
                let (f, scale) = plan.terms[*piece][*term];
                let (x, g) = v.factors[f];
                let c = scale * weight(*piece) * k;
                let (x, dx) = (x * c, g.map(|d| d * c));
                for (sum, z) in self.span.iter_mut().zip(coords) {
                    sum.0 += *z * x;
                    for a in 0..3 {
                        sum.1[a] += *z * dx[a];
                    }
                }
            }
            // The coordinates zeroed; per atom its factor, scale, weight and coefficient (four products with the gradients).
            work += 4.0 * MOVED * d.basis.len() as f64 + 4.0 * d.atoms.len() as f64;
            let o = if d.h { &mut self.h[d.out] } else { &mut self.channels[d.out] };
            for (sum, b) in self.span.iter().zip(&d.basis) {
                for (&place, z) in d.places.iter().zip(b) {
                    o.value[place] += sum.0 * *z;
                    for a in 0..3 {
                        o.gradient[a][place] += sum.1[a] * *z;
                    }
                }
            }
            // Each coordinate an atom's value and three gradients, each place a basis vector's.
            work += 4.0 * (d.atoms.len() * d.basis.len() + d.basis.len() * d.places.len()) as f64;
        }
        let spanned = !grouping.spans.is_empty();
        if making.model && making.effective {
            // H_eff: the direct slow entries, then the pairs, ½ and the conjugate.
            for p in (0..plan.hamiltonian).filter(|_| !spanned) {
                factors(&mut self.tf, p);
                let o = &mut self.h[grouping.h_direct[p]];
                for ((e, _), &k) in plan.direct[p].iter().zip(&grouping.h_direct_places[p]) {
                    let (x, dx, moving) = self.tf[e.term as usize];
                    let c = e.value * v.coefficients[e.coefficient as usize];
                    let k = k as usize;
                    o.value[k] += x * c;
                    if moving {
                        for (g, d) in o.gradient.iter_mut().zip(dx) {
                            g[k] += d * c;
                        }
                    }
                }
                work += direct_work(&self.tf, &plan.direct[p]);
            }
            for pair in &grouping.h_pairs {
                if !plan.made(grouping.ws[pair.w].block, true) {
                    continue;
                }
                if self.made != pair.w {
                    work += self.legs(plan, grouping, v, &weight, pair.w);
                    work += self.product(grouping, pair.w);
                    self.made = pair.w;
                }
                let products = Products::of(&self.p, &self.pd, self.moving, grouping, pair.w, pair.z);
                let (rows, cols) = (grouping.ws[pair.w].index.len(), grouping.zs[pair.z].index.len());
                // Per element: its gradients' two parts added where the leg sum moves, the half of it and its gradients,
                // and its value and gradients into two outputs (the conjugate's).
                work += (rows * cols) as f64 * (if self.moving { 3.0 * MOVED } else { 0.0 } + 2.0 + 8.0 * MOVED);
                for i in 0..rows {
                    let (places, conjugates) = (&pair.places[i * cols..(i + 1) * cols], &pair.conjugate_places[i * cols..(i + 1) * cols]);
                    for (j, (&k, &kc)) in places.iter().zip(conjugates).enumerate() {
                        let (x, dx) = products.element(i, j);
                        let (x, dx) = (x * 0.5, dx.map(|d| d * 0.5));
                        let (k, kc) = (k as usize, kc as usize);
                        let out = &mut self.h[pair.out];
                        out.value[k] += x;
                        for a in 0..3 {
                            out.gradient[a][k] += dx[a];
                        }
                        let out = &mut self.h[pair.conjugate];
                        out.value[kc] += x.conj();
                        for a in 0..3 {
                            out.gradient[a][kc] += dx[a].conj();
                        }
                    }
                }
            }
        }
        if making.model {
            // The jumps: direct entries, then the pairs, each whole in its channel.
            for p in plan.hamiltonian..plan.keys.len() {
                let (_, which) = plan.jump[p].unwrap();
                if which == 1 && !making.kicks {
                    continue;
                }
                if spanned && which == 0 {
                    continue;
                }
                factors(&mut self.tf, p);
                for ((e, _), &(o, k)) in plan.direct[p].iter().zip(&grouping.j_direct[p - plan.hamiltonian]) {
                    let (x, dx, moving) = self.tf[e.term as usize];
                    let c = e.value * v.coefficients[e.coefficient as usize];
                    let (o, k) = (&mut self.channels[o], k as usize);
                    o.value[k] += x * c;
                    if moving {
                        for (g, d) in o.gradient.iter_mut().zip(dx) {
                            g[k] += d * c;
                        }
                    }
                }
                work += direct_work(&self.tf, &plan.direct[p]);
            }
            for pair in &grouping.j_pairs {
                if (grouping.ws[pair.w].jump.is_some_and(|(_, which)| which == 1) && !making.kicks) || !plan.made(grouping.ws[pair.w].block, true) {
                    continue;
                }
                if self.made != pair.w {
                    work += self.legs(plan, grouping, v, &weight, pair.w);
                    work += self.product(grouping, pair.w);
                    self.made = pair.w;
                }
                let products = Products::of(&self.p, &self.pd, self.moving, grouping, pair.w, pair.z);
                let (rows, cols) = (grouping.ws[pair.w].index.len(), grouping.zs[pair.z].index.len());
                // Per element: its gradients' two parts added where the leg sum moves, its value and gradients added out.
                work += (rows * cols) as f64 * (if self.moving { 3.0 * MOVED } else { 0.0 } + 4.0 * MOVED);
                let out = &mut self.channels[pair.out];
                for i in 0..rows {
                    for (j, &k) in pair.places[i * cols..(i + 1) * cols].iter().enumerate() {
                        let (x, dx) = products.element(i, j);
                        let k = k as usize;
                        out.value[k] += x;
                        for a in 0..3 {
                            out.gradient[a][k] += dx[a];
                        }
                    }
                }
            }
        }
        // The lift: M on Q's blocks, by group.
        if making.lift {
            for (l, g) in self.lift.iter_mut() {
                zero(l, plan.nq, n);
                for d in g.iter_mut() {
                    zero(d, if making.gradients { plan.nq } else { 0 }, n);
                }
            }
            let sides = if making.gradients { 4 } else { 1 };
            work += MOVED * (sides * plan.nq * n * self.lift.len()) as f64;
            for (b, group, mine) in &grouping.lift {
                let (lv, lg) = &mut self.lift[*group];
                let (mv, md, width) = (&self.mv[*b], &self.md[*b], grouping.layout[*b].1);
                for &i in mine {
                    let (s, o) = (&grouping.zs[i], grouping.z_offset[i]);
                    work += MOVED * (sides * plan.blocks[*b].rows.len() * s.index.len()) as f64;
                    for (k, &row) in plan.blocks[*b].rows.iter().enumerate() {
                        for (j, &a) in s.index.iter().enumerate() {
                            lv.data[row * n + a] += mv.data[k * width + o + j];
                            if making.gradients {
                                for d in 0..3 {
                                    lg[d].data[row * n + a] += md.data[k * 3 * width + d * width + o + j];
                                }
                            }
                        }
                    }
                }
            }
        }
        self.work = work;
    }

    /// The leg sum wi at the point, dense: X and ∂X (`x`), its members' fixed entries at their factors, weights and
    /// coefficients, each on its row in the sum (`Grouping::wmaps`) and its column in the block.
    fn legs(&mut self, plan: &Plan, grouping: &Grouping, v: &Values, weight: &dyn Fn(usize) -> C64, wi: usize) -> f64 {
        let s = &grouping.ws[wi];
        let (rows, size) = (s.index.len(), plan.blocks[s.block].rows.len());
        zero(&mut self.x, 4 * rows, size);
        self.moving = false;
        // X and ∂X zeroed; per entry its value (four reals' product with the coefficient, the scale and the weight) and
        // its moving factors' gradients.
        let mut work = MOVED * (4 * rows * size) as f64;
        for (member, map) in &grouping.wmaps[s.at] {
            let r = &plan.legs[*member];
            let f = weight(r.piece);
            let ts = &plan.terms[r.piece];
            for e in &r.entries {
                let (k, to) = (v.coefficients[e.coefficient as usize], map[e.row as usize]);
                if to == u32::MAX || k == 0.0 {
                    continue;
                }
                let (fi, sc) = ts[e.term as usize];
                let (x, g) = v.factors[fi];
                let c = e.value * k * sc * f;
                let at = to as usize * size + e.col as usize;
                self.x.data[at] += x * c;
                work += 1.0 + 3.0 * REAL;
                for (a, d) in g.iter().enumerate() {
                    if *d != C64::ZERO {
                        self.x.data[(a + 1) * rows * size + at] += *d * c;
                        self.moving = true;
                        work += 1.0;
                    }
                }
            }
        }
        work
    }

    /// The products of the leg sum wi (made, `legs`) with its block's M, by the product kernel: [X; ∂X] M_v -- X M_v
    /// alone where the sum does not move -- and X ∂M; every pair of the sum is a slice of them (`Grouping::z_offset`,
    /// `element`). Returns the counted complex multiply-adds.
    fn product(&mut self, grouping: &Grouping, wi: usize) -> f64 {
        let s = &grouping.ws[wi];
        let (rows, width) = (s.index.len(), grouping.layout[s.block].1);
        let (mv, md) = (&self.mv[s.block], &self.md[s.block]);
        let size = mv.rows;
        let stacked = if self.moving { 4 * rows } else { rows };
        sized(&mut self.p, stacked, width);
        linalg::fused_product(&self.x.data[..stacked * size], &mv.data, (stacked, size, width), &mut self.p.data);
        sized(&mut self.pd, rows, 3 * width);
        linalg::fused_product(&self.x.data[..rows * size], &md.data, (rows, size, 3 * width), &mut self.pd.data);
        ((stacked + 3 * rows) * size * width) as f64
    }

}

/// A pair's slices of its leg sum's products (`Out::product`): [X; ∂X] M and X ∂M, the sum's rows, the block's width,
/// the amplitude sum's first column, whether the leg sum moves.
struct Products<'a> {
    p: &'a [C64],
    pd: &'a [C64],
    rows: usize,
    width: usize,
    at: usize,
    moving: bool,
}

impl Products<'_> {
    /// The pair of the leg sum wi (its products `p`, `pd` made, `Out::product`; `moving` its) and the amplitude sum zi.
    fn of<'a>(p: &'a Mat, pd: &'a Mat, moving: bool, grouping: &Grouping, wi: usize, zi: usize) -> Products<'a> {
        let (rows, width) = (grouping.ws[wi].index.len(), grouping.layout[grouping.ws[wi].block].1);
        Products { p: &p.data, pd: &pd.data, rows, width, at: grouping.z_offset[zi], moving }
    }

    /// The pair's element (i, j): X M and its three gradients ∂X M + X ∂M.
    #[inline]
    fn element(&self, i: usize, j: usize) -> (C64, [C64; 3]) {
        let (width, at) = (self.width, self.at + j);
        let value = self.p[i * width + at];
        let gradient = std::array::from_fn(|a| {
            let held = self.pd[i * 3 * width + a * width + at];
            if self.moving {
                held + self.p[((a + 1) * self.rows + i) * width + at]
            } else {
                held
            }
        });
        (value, gradient)
    }
}

impl Out {
    /// Each output's elements' rounding scales of the contraction just made (`contract` with the same `plan`,
    /// `grouping`, `local` and `weights`, the model made): the sum of the moduli of the terms each element is the sum
    /// of, and their count -- a direct entry one term |x c|; a pair ½ Σ_q |X_iq| |M_qj| (a jump's pair without the ½),
    /// |X| its leg sum's members' moduli summed and |M| the block's amplitudes' own scale (|V| Σ|w| |M̂|_s, the
    /// amplitudes' by `Local::shadows`), |B| + the members' count of terms (the inner product's and the leg sum's); a
    /// gradient's ½ Σ_q (|∂X_iq| |M_qj| + |X_iq| |∂M_qj|) likewise. A computed element within Higham's γ_K of
    /// its scale is that computation's rounding (units.rs `rounding`): what the model's structure is read by
    /// (model/selected.rs `Selected::structure`). Made where asked, never on a trajectory's nodes.
    pub fn scales(&mut self, plan: &Plan, grouping: &Grouping, local: &Local, weights: Option<&[C64]>) {
        let weight = |p: usize| weights.map_or(C64::ONE, |w| w[p]);
        let v = &local.values;
        // Each block's |M| and |∂M|: the amplitude sums Z on the amplitudes' shadows with |w|, then |V| Z (a Q block's).
        let shadows = local.shadows(plan);
        let mut blocks: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(grouping.layout.len());
        for (b, (list, width)) in grouping.layout.iter().enumerate() {
            let (size, width) = (plan.blocks[b].rows.len(), *width);
            let (mut zv, mut zd) = (vec![0.0; size * width], vec![0.0; size * 3 * width]);
            if !plan.made(b, true) {
                blocks.push((zv, zd));
                continue;
            }
            for &zi in list {
                let (s, o) = (&grouping.zs[zi], grouping.z_offset[zi]);
                for (member, map) in &s.members {
                    let (mb, off) = plan.place[*member];
                    let (a, da) = &shadows[mb];
                    let total = local.m[mb].cols;
                    let w = weight(plan.amplitudes[*member].piece).abs();
                    for k in 0..size {
                        for &(from, to) in map {
                            zv[k * width + o + to] += w * a[k * total + off + from];
                            for d in 0..3 {
                                zd[k * 3 * width + d * width + o + to] += w * da[k * 3 * total + d * total + off + from];
                            }
                        }
                    }
                }
            }
            if !list.is_empty() {
                let vectors: Vec<f64> = local.spectra[b].vectors.data.iter().map(|z| z.abs()).collect();
                let (mut mv, mut md) = (vec![0.0; size * width], vec![0.0; size * 3 * width]);
                for q in 0..size {
                    for k in 0..size {
                        let x = vectors[q * size + k];
                        if x != 0.0 {
                            for (o, z) in mv[q * width..(q + 1) * width].iter_mut().zip(&zv[k * width..(k + 1) * width]) {
                                *o += x * z;
                            }
                            for (o, z) in md[q * 3 * width..(q + 1) * 3 * width].iter_mut().zip(&zd[k * 3 * width..(k + 1) * 3 * width]) {
                                *o += x * z;
                            }
                        }
                    }
                }
                blocks.push((mv, md));
            } else {
                blocks.push((zv, zd));
            }
        }
        for (c, list) in self.h.iter_mut().zip(&grouping.lists).chain(self.channels.iter_mut().zip(&grouping.channel_lists)) {
            for x in std::iter::once(&mut c.scale).chain(c.gradient_scale.iter_mut()) {
                x.clear();
                x.resize(list.len(), 0.0);
            }
            c.terms.clear();
            c.terms.resize(list.len(), 0);
        }
        // The direct entries: one term each.
        for p in 0..plan.keys.len() {
            let w = weight(p);
            let terms = &plan.terms[p];
            let is_jump = p >= plan.hamiltonian;
            for (i, (e, _)) in plan.direct[p].iter().enumerate() {
                let (fi, sc) = terms[e.term as usize];
                let (x, dx) = v.factors[fi];
                let c = (e.value * v.coefficients[e.coefficient as usize] * sc * w).abs();
                let (o, k) = if is_jump {
                    let (o, k) = grouping.j_direct[p - plan.hamiltonian][i];
                    (&mut self.channels[o], k as usize)
                } else {
                    (&mut self.h[grouping.h_direct[p]], grouping.h_direct_places[p][i] as usize)
                };
                o.scale[k] += x.abs() * c;
                for (g, d) in o.gradient_scale.iter_mut().zip(dx) {
                    g[k] += d.abs() * c;
                }
                o.terms[k] += 1;
            }
        }
        // The pairs: the leg sums' moduli times their blocks' |M|, per leg sum once.
        let count = grouping.h_pairs.len() + grouping.j_pairs.len();
        let mut made = usize::MAX;
        for k in 0..count {
            let (wi, zi, is_jump) = if k < grouping.h_pairs.len() {
                let p = &grouping.h_pairs[k];
                (p.w, p.z, false)
            } else {
                let p = &grouping.j_pairs[k - grouping.h_pairs.len()];
                (p.w, p.z, true)
            };
            let s = &grouping.ws[wi];
            if !plan.made(s.block, true) {
                continue;
            }
            let (rows, size, width) = (s.index.len(), plan.blocks[s.block].rows.len(), grouping.layout[s.block].1);
            if made != wi {
                // |X| and |∂X| (4 rows × |B|): each member's entry's modulus.
                self.xa.clear();
                self.xa.resize(4 * rows * size, 0.0);
                for (member, map) in &grouping.wmaps[s.at] {
                    let r = &plan.legs[*member];
                    let f = weight(r.piece);
                    let ts = &plan.terms[r.piece];
                    for e in &r.entries {
                        let (kc, to) = (v.coefficients[e.coefficient as usize], map[e.row as usize]);
                        if to == u32::MAX || kc == 0.0 {
                            continue;
                        }
                        let (fi, sc) = ts[e.term as usize];
                        let (x, g) = v.factors[fi];
                        let c = (e.value * kc * sc * f).abs();
                        let at = to as usize * size + e.col as usize;
                        self.xa[at] += x.abs() * c;
                        for (a, d) in g.iter().enumerate() {
                            self.xa[(a + 1) * rows * size + at] += d.abs() * c;
                        }
                    }
                }
                // The block's |M| and |∂M| (their scales); the products |X| |M| (the value's and ∂X's 4 rows) and |X| |∂M|.
                let (mv, md) = &blocks[s.block];
                self.ma.clear();
                self.ma.extend_from_slice(mv);
                self.mda.clear();
                self.mda.extend_from_slice(md);
                self.pa.clear();
                self.pa.resize(4 * rows * width, 0.0);
                self.pda.clear();
                self.pda.resize(rows * 3 * width, 0.0);
                for i in 0..4 * rows {
                    for q in 0..size {
                        let x = self.xa[i * size + q];
                        if x != 0.0 {
                            for (o, m) in self.pa[i * width..(i + 1) * width].iter_mut().zip(&self.ma[q * width..(q + 1) * width]) {
                                *o += x * m;
                            }
                        }
                    }
                }
                for i in 0..rows {
                    for q in 0..size {
                        let x = self.xa[i * size + q];
                        if x != 0.0 {
                            for (o, m) in self.pda[i * 3 * width..(i + 1) * 3 * width].iter_mut().zip(&self.mda[q * 3 * width..(q + 1) * 3 * width]) {
                                *o += x * m;
                            }
                        }
                    }
                }
                made = wi;
            }
            let members = grouping.wmaps[s.at].len();
            let half = if is_jump { 1.0 } else { 0.5 };
            let (cols, at0) = (grouping.zs[zi].index.len(), grouping.z_offset[zi]);
            for i in 0..rows {
                for j in 0..cols {
                    let at = at0 + j;
                    let value = half * self.pa[i * width + at];
                    let gradient: [f64; 3] = std::array::from_fn(|a| half * (self.pda[i * 3 * width + a * width + at] + self.pa[((a + 1) * rows + i) * width + at]));
                    let add = |o: &mut Compact, place: usize| {
                        o.scale[place] += value;
                        for (g, d) in o.gradient_scale.iter_mut().zip(gradient) {
                            g[place] += d;
                        }
                        o.terms[place] += size + members;
                    };
                    if is_jump {
                        let p = &grouping.j_pairs[k - grouping.h_pairs.len()];
                        add(&mut self.channels[p.out], p.places[i * cols + j] as usize);
                    } else {
                        let p = &grouping.h_pairs[k];
                        add(&mut self.h[p.out], p.places[i * cols + j] as usize);
                        add(&mut self.h[p.conjugate], p.conjugate_places[i * cols + j] as usize);
                    }
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// The reach's bounds.

/// The range a run's atom reaches, as the fold's numbers take it (model/classify.rs, # The drop tier): every beam's
/// amplitude of modulus at most one (its focus value, field.rs), every intensity of the generator -- the tweezer's
/// |a_tw|² = s, a Gaussian beam's |a_l|² -- in [0, 1], the momentum within κ_max(s) = κ √s (a held atom's kinetic
/// energy below s U₀), and the tweezer's Doppler powers within their largest there.
#[derive(Clone, Debug)]
pub struct Reach {
    /// The tweezer's intensity among the values' factors (`Factor::Power`).
    pub power: Option<usize>,
    /// κ (1/µm): the largest momentum at the focus, κ_max(s) = κ √s.
    pub kappa: f64,
    /// Per piece (`Plan::keys`' order) the change of its frequency per unit momentum, (ħ/m)|Σ_g c_g k_g| (rad/µs·µm).
    pub doppler: Vec<f64>,
    /// Per coefficient its largest modulus over the reach (coefficient 0 is one).
    pub coefficients: Vec<f64>,
}

/// The most boxes of the intensities one column's bound takes (`Plan::reach`): beyond them the column is unbounded.
const BOXES: usize = 1 << 12;

/// x ← N x + b for N n × n (row by row, nonnegative) until no entry changes (`Plan::reach`'s Neumann sum).
fn neumann_sum(n: &[f64], size: usize, b: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend_from_slice(b);
    let mut term = b.to_vec();
    let mut next = vec![0.0; size];
    loop {
        for i in 0..size {
            next[i] = (0..size).map(|k| n[i * size + k] * term[k]).sum();
        }
        let mut moved = false;
        for (o, t) in out.iter_mut().zip(&next) {
            let before = *o;
            *o += *t;
            moved |= *o != before;
        }
        std::mem::swap(&mut term, &mut next);
        if !moved {
            return;
        }
    }
}

impl Plan {
    /// Per folded block the entrywise bound over the reach (`Reach`) of its amplitudes M = (z − G₀)⁻¹X in the states'
    /// basis, |B| × its batch's columns as `Local::m` lays them (model/classify.rs, # The drop tier): per column j, z =
    /// E_a + f_c of its kept state a and its piece c, X its piece's entries' moduli at their largest coefficients (every
    /// factor's modulus at most one). The generator and E_a are affine in the varying intensities σ (`varying`); on a
    /// box of them about σ^c of half-widths h, A = A_c + Δ with A_c = z^c − G₀(σ^c) at the box's centre and
    ///   |Δ| ≤ E = Σ_σ h_σ |e_aσ − G_σ| + δ_c(s_hi) + Σ_{k ≥ 1} c̄_k (|g_ak| + |G_k|),
    /// G_σ, e_aσ the intensity's part of the block's generator and of E_a, δ_c(s) = (ħ/m)|K_c| κ √s the column's
    /// Doppler change at the box's largest tweezer intensity, the coefficients' parts by their largest; where
    /// N = E|R_c| has row sums below ½, |R| ≤ |R_c| Σ_k N^k (R_c = A_c⁻¹; the Neumann series of (1 + ΔR_c)⁻¹), and the
    /// column's bound is |R_c| Σ_k N^k |X|. A box whose N is not so is halved along the intensity of the largest share;
    /// the column's bound is the largest over the boxes (`BOXES` of them at most: beyond, the column is unbounded).
    /// `values` are the structure point's (the frequencies at rest).
    pub fn reach(&self, values: &Values, reach: &Reach) -> Vec<Vec<f64>> {
        let coefficient = |c: u32| -> f64 { if c == 0 { 1.0 } else { reach.coefficients[c as usize] } };
        let mut out = Vec::with_capacity(self.blocks.len());
        for (block, batch) in self.blocks.iter().zip(&self.batches) {
            let size = block.rows.len();
            let total = batch.total;
            let mut bound = vec![0.0_f64; size * total];
            if batch.amplitudes.is_empty() {
                out.push(bound);
                continue;
            }
            // The block's generator: the fixed part (every factor at its value, the coefficient one), each varying
            // intensity's part, the coefficients' moduli.
            let mut fixed = Mat::zeros(size, size);
            let mut parts: Vec<(usize, Mat)> = Vec::new();
            let mut wobble = vec![0.0_f64; size * size];
            for e in &block.generator {
                let (f, scale) = self.generator_terms[e.term as usize];
                let at = e.row as usize * size + e.col as usize;
                if e.coefficient != 0 {
                    wobble[at] += (scale * e.value).abs() * coefficient(e.coefficient);
                } else if self.varying[f] {
                    let i = parts.iter().position(|x| x.0 == f).unwrap_or_else(|| {
                        parts.push((f, Mat::zeros(size, size)));
                        parts.len() - 1
                    });
                    parts[i].1.data[at] += scale * e.value;
                } else {
                    fixed.data[at] += values.factors[f].0 * scale * e.value;
                }
            }
            for (&i, &(off, _)) in batch.amplitudes.iter().zip(&batch.offsets) {
                let r = &self.amplitudes[i];
                for (j, &a) in r.cols.iter().enumerate() {
                    // The column's X, E_a's parts, its frequency and its Doppler change.
                    let mut x = vec![0.0_f64; size];
                    for e in r.entries.iter().filter(|e| e.col as usize == j) {
                        let (_, scale) = self.terms[r.piece][e.term as usize];
                        x[e.row as usize] += (e.value * scale).abs() * coefficient(e.coefficient);
                    }
                    let (mut energy, mut slope, mut drift) = (0.0_f64, vec![0.0_f64; parts.len()], 0.0_f64);
                    for e in self.diagonal.iter().filter(|e| e.row as usize == a) {
                        let (f, scale) = self.generator_terms[e.term as usize];
                        if e.coefficient != 0 {
                            drift += (scale * e.value).abs() * coefficient(e.coefficient);
                        } else if let Some(i) = parts.iter().position(|x| x.0 == f) {
                            slope[i] += (scale * e.value).re;
                        } else if self.varying[f] {
                            // A varying intensity of the kept state alone: E_a moves within its range.
                            drift += (scale * e.value).abs();
                        } else {
                            energy += (values.factors[f].0 * scale * e.value).re;
                        }
                    }
                    let z = energy + values.frequencies[r.piece];
                    let column = self.column(size, &fixed, &parts, &slope, &wobble, z, drift, reach.doppler[r.piece] * reach.kappa, reach.power, &x);
                    for k in 0..size {
                        bound[k * total + off + j] = column[k];
                    }
                }
            }
            out.push(bound);
        }
        out
    }

    /// One column's bound over the boxes of the intensities (`reach`): the fixed generator, the intensities' parts
    /// and the kept state's slopes along them, the coefficients' moduli on the block and on E_a (`drift`, with a kept
    /// state's own intensities), z at rest, the Doppler change at s = 1, the tweezer's intensity, |X|.
    #[allow(clippy::too_many_arguments)]
    fn column(&self, size: usize, fixed: &Mat, parts: &[(usize, Mat)], slope: &[f64], wobble: &[f64], z: f64, drift: f64, doppler: f64, power: Option<usize>, x: &[f64]) -> Vec<f64> {
        let mut most = vec![0.0_f64; size];
        let mut boxes: Vec<Vec<(f64, f64)>> = vec![vec![(0.0, 1.0); parts.len()]];
        let mut made = 0usize;
        let (mut a, mut inverse, mut pivots) = (vec![C64::ZERO; size * size], vec![C64::ZERO; size * size], vec![0usize; size]);
        let mut n = vec![0.0_f64; size * size];
        let mut y = Vec::new();
        while let Some(bx) = boxes.pop() {
            made += 1;
            if made > BOXES {
                return vec![f64::INFINITY; size];
            }
            // A_c = z^c − G₀(σ^c) and its inverse.
            let mut centre = z;
            for (i, s) in slope.iter().enumerate() {
                centre += 0.5 * (bx[i].0 + bx[i].1) * s;
            }
            for r in 0..size {
                for c in 0..size {
                    let mut g = fixed.data[r * size + c];
                    for (i, (_, part)) in parts.iter().enumerate() {
                        g += part.data[r * size + c] * (0.5 * (bx[i].0 + bx[i].1));
                    }
                    a[r * size + c] = -g;
                }
                a[r * size + r] += C64::real(centre);
            }
            if !linalg::factor(&mut a, &mut pivots, size) {
                return vec![f64::INFINITY; size];
            }
            inverse.fill(C64::ZERO);
            for k in 0..size {
                inverse[k * size + k] = C64::ONE;
            }
            linalg::solve(&a, &pivots, size, &mut inverse, size);
            let r_abs: Vec<f64> = inverse.iter().map(|z| z.abs()).collect();
            if r_abs.iter().any(|v| !v.is_finite()) {
                return vec![f64::INFINITY; size];
            }
            // E and N = E|R_c|, each intensity's share apart.
            let s_high = power.and_then(|p| parts.iter().position(|x| x.0 == p)).map_or(1.0, |i| bx[i].1);
            let scalar = drift + doppler * s_high.sqrt();
            let mut e = wobble.to_vec();
            for k in 0..size {
                e[k * size + k] += scalar;
            }
            let mut shares = vec![0.0_f64; parts.len()];
            for (i, (_, part)) in parts.iter().enumerate() {
                let h = 0.5 * (bx[i].1 - bx[i].0);
                let mut d = vec![0.0_f64; size * size];
                for r in 0..size {
                    for c in 0..size {
                        let mut v = part.data[r * size + c];
                        if r == c {
                            v -= C64::real(slope[i]);
                        }
                        d[r * size + c] = h * v.abs();
                    }
                }
                // The share's row sums of d |R_c|.
                let mut worst = 0.0_f64;
                for r in 0..size {
                    let mut row = 0.0;
                    for c in 0..size {
                        row += (0..size).map(|k| d[r * size + k] * r_abs[k * size + c]).sum::<f64>();
                    }
                    worst = linalg::larger(worst, row);
                }
                shares[i] = worst;
                for (o, v) in e.iter_mut().zip(&d) {
                    *o += v;
                }
            }
            let mut q = 0.0_f64;
            for r in 0..size {
                let mut row = 0.0;
                for c in 0..size {
                    let v: f64 = (0..size).map(|k| e[r * size + k] * r_abs[k * size + c]).sum();
                    n[r * size + c] = v;
                    row += v;
                }
                q = linalg::larger(q, row);
            }
            if !(q < 0.5) {
                let Some(widest) = (0..parts.len()).filter(|&i| shares[i] > 0.0).max_by(|&i, &j| shares[i].total_cmp(&shares[j])) else {
                    return vec![f64::INFINITY; size];
                };
                let (lo, hi) = bx[widest];
                let mid = 0.5 * (lo + hi);
                if !(lo < mid && mid < hi) {
                    return vec![f64::INFINITY; size];
                }
                let (mut low, mut high) = (bx.clone(), bx);
                low[widest].1 = mid;
                high[widest].0 = mid;
                boxes.push(low);
                boxes.push(high);
                continue;
            }
            // |R_c| Σ_k N^k |X|.
            neumann_sum(&n, size, x, &mut y);
            for (k, m) in most.iter_mut().enumerate() {
                let v: f64 = (0..size).map(|c| r_abs[k * size + c] * y[c]).sum();
                *m = linalg::larger(*m, v);
            }
        }
        most
    }
}

impl Out {
    /// Each output's elements' bounds over the reach (model/classify.rs, # The drop tier), the values alone: the sum of
    /// the moduli of the terms each element is the sum of, every factor at most one and every coefficient at its
    /// largest (`Reach`), the blocks' amplitudes at their bounds `moduli` (`Plan::reach`, the states' basis) -- a direct
    /// entry |x c|; a pair ½ Σ_q |X_iq| M̄_qj (a jump's pair without the ½). The weights are one (the lab's grouping).
    /// Per H output and per channel output (the photon forms' outputs left at zero), on their structural elements.
    pub fn bounds(plan: &Plan, grouping: &Grouping, moduli: &[Vec<f64>], coefficients: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        Out::bounds_of(plan, grouping, moduli, coefficients, &|b| b.is_none_or(|b| plan.made(b, true)))
    }

    /// `bounds` of the parts `take` takes: the direct entries (None), the pairs of a block (Some(b)).
    pub fn bounds_of(plan: &Plan, grouping: &Grouping, moduli: &[Vec<f64>], coefficients: &[f64], take: &dyn Fn(Option<usize>) -> bool) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let coefficient = |c: u32| -> f64 { if c == 0 { 1.0 } else { coefficients[c as usize] } };
        let mut h: Vec<Vec<f64>> = grouping.lists.iter().map(|l| vec![0.0; l.len()]).collect();
        let mut channels: Vec<Vec<f64>> = grouping.channel_lists.iter().map(|l| vec![0.0; l.len()]).collect();
        // The amplitude sums' bounds per block, side by side (`Grouping::layout`).
        let mut sums: Vec<Vec<f64>> = Vec::with_capacity(grouping.layout.len());
        for (b, (list, width)) in grouping.layout.iter().enumerate() {
            let size = plan.blocks[b].rows.len();
            let mut z = vec![0.0; size * width];
            for &zi in list {
                let (s, o) = (&grouping.zs[zi], grouping.z_offset[zi]);
                for (member, map) in &s.members {
                    let (mb, off) = plan.place[*member];
                    let total = plan.batches[mb].total;
                    for k in 0..size {
                        for &(from, to) in map {
                            z[k * width + o + to] += moduli[mb][k * total + off + from];
                        }
                    }
                }
            }
            sums.push(z);
        }
        // The direct entries.
        for p in (0..plan.keys.len()).filter(|_| take(None)) {
            let terms = &plan.terms[p];
            if p >= plan.hamiltonian && plan.jump[p].is_some_and(|(_, which)| which == 1) {
                continue;
            }
            for (i, (e, _)) in plan.direct[p].iter().enumerate() {
                let (_, sc) = terms[e.term as usize];
                let c = (e.value * sc).abs() * coefficient(e.coefficient);
                if p >= plan.hamiltonian {
                    let (o, k) = grouping.j_direct[p - plan.hamiltonian][i];
                    channels[o][k as usize] += c;
                } else {
                    h[grouping.h_direct[p]][grouping.h_direct_places[p][i] as usize] += c;
                }
            }
        }
        // The pairs: per leg sum its |X| once, then its products with the amplitude sums' bounds.
        let count = grouping.h_pairs.len() + grouping.j_pairs.len();
        let (mut made, mut xa) = (usize::MAX, Vec::new());
        for k in 0..count {
            let (wi, zi, is_jump) = if k < grouping.h_pairs.len() {
                let p = &grouping.h_pairs[k];
                (p.w, p.z, false)
            } else {
                let p = &grouping.j_pairs[k - grouping.h_pairs.len()];
                (p.w, p.z, true)
            };
            let s = &grouping.ws[wi];
            if s.jump.is_some_and(|(_, which)| which == 1) || !take(Some(s.block)) {
                continue;
            }
            let (rows, size, width) = (s.index.len(), plan.blocks[s.block].rows.len(), grouping.layout[s.block].1);
            if made != wi {
                xa.clear();
                xa.resize(rows * size, 0.0);
                for (member, map) in &grouping.wmaps[s.at] {
                    let r = &plan.legs[*member];
                    let ts = &plan.terms[r.piece];
                    for e in &r.entries {
                        let to = map[e.row as usize];
                        if to == u32::MAX {
                            continue;
                        }
                        let (_, sc) = ts[e.term as usize];
                        xa[to as usize * size + e.col as usize] += (e.value * sc).abs() * coefficient(e.coefficient);
                    }
                }
                made = wi;
            }
            let half = if is_jump { 1.0 } else { 0.5 };
            let (cols, at0) = (grouping.zs[zi].index.len(), grouping.z_offset[zi]);
            let z = &sums[s.block];
            for i in 0..rows {
                for j in 0..cols {
                    let v = half * (0..size).map(|q| xa[i * size + q] * z[q * width + at0 + j]).sum::<f64>();
                    if is_jump {
                        let p = &grouping.j_pairs[k - grouping.h_pairs.len()];
                        channels[p.out][p.places[i * cols + j] as usize] += v;
                    } else {
                        let p = &grouping.h_pairs[k];
                        h[p.out][p.places[i * cols + j] as usize] += v;
                        h[p.conjugate][p.conjugate_places[i * cols + j] as usize] += v;
                    }
                }
            }
        }
        (h, channels)
    }
}

impl Grouping {
    /// The direct entries spanned per output (model/selected.rs, # The direct entries spanned): per output -- H's, a
    /// channel's value or photon form -- its atoms, each a piece's term at a coefficient with the fixed vector of its
    /// entries over the output's places and the bound b of its point coefficient (`bound`: |scale| times the
    /// coefficient's largest over the reach `coefficients`, every factor at most one), orthonormalized in the order of
    /// ‖v‖b (two passes of Gram–Schmidt): an atom's residual r is a new direction where ‖r‖b exceeds `tolerance` times
    /// its kind's largest ‖v‖b (channels, H), else left out, its ‖r‖b booked. Returns the spans, what each output
    /// leaves out, and the work.
    pub(crate) fn spanned(&self, plan: &Plan, coefficients: &[f64], tolerance: f64) -> Spanned {
        let bound = |p: usize, t: usize, c: usize| plan.terms[p][t].1.abs() * if c == 0 { 1.0 } else { coefficients[c] };
        // The outputs' entries: per output its places and per atom its entries (place, value).
        let mut outs: Vec<(bool, usize, usize, Vec<usize>, Vec<(usize, usize, usize)>, Vec<Vec<(usize, C64)>>)> = Vec::new();
        let mut plain = 0.0;
        for p in 0..plan.keys.len() {
            // The photon forms' (a jump's kick) are taken one by one: their error is the kick's, not the model's.
            let which = plan.jump[p].map_or(0, |x| x.1);
            if which == 1 {
                continue;
            }
            for (i, (e, _)) in plan.direct[p].iter().enumerate() {
                let (h, o, k) = if p < plan.hamiltonian { (true, self.h_direct[p], self.h_direct_places[p][i] as usize) } else { (false, self.j_direct[p - plan.hamiltonian][i].0, self.j_direct[p - plan.hamiltonian][i].1 as usize) };
                plain += 4.0;
                let g = outs.iter().position(|x| x.0 == h && x.1 == o && x.2 == which).unwrap_or_else(|| {
                    outs.push((h, o, which, Vec::new(), Vec::new(), Vec::new()));
                    outs.len() - 1
                });
                let out = &mut outs[g];
                let place = out.3.iter().position(|&x| x == k).unwrap_or_else(|| {
                    out.3.push(k);
                    out.3.len() - 1
                });
                let key = (p, e.term as usize, e.coefficient as usize);
                let a = out.4.iter().position(|x| *x == key).unwrap_or_else(|| {
                    out.4.push(key);
                    out.5.push(Vec::new());
                    out.4.len() - 1
                });
                out.5[a].push((place, e.value));
            }
        }
        let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y);
        // Each kind's largest atom (H's outputs and the channels' are apart by orders).
        let mut top = [0.0_f64; 2];
        for (h, _, _, places, keys, entries) in &outs {
            for (&(p, t, c), e) in keys.iter().zip(entries) {
                let mut v = vec![C64::ZERO; places.len()];
                for &(i, x) in e {
                    v[i] += x;
                }
                top[usize::from(*h)] = top[usize::from(*h)].max(dot(&v, &v).re.sqrt() * bound(p, t, c));
            }
        }
        let (mut spans, mut dropped, mut work) = (Vec::new(), Vec::new(), 0.0);
        for (h, out, which, places, keys, entries) in outs {
            let vectors: Vec<Vec<C64>> = entries
                .iter()
                .map(|e| {
                    let mut v = vec![C64::ZERO; places.len()];
                    for &(i, x) in e {
                        v[i] += x;
                    }
                    v
                })
                .collect();
            let bounds: Vec<f64> = keys.iter().map(|&(p, t, c)| bound(p, t, c)).collect();
            let size: Vec<f64> = vectors.iter().zip(&bounds).map(|(v, b)| dot(v, v).re.sqrt() * b).collect();
            let mut order: Vec<usize> = (0..vectors.len()).collect();
            order.sort_by(|&a, &b| size[b].total_cmp(&size[a]));
            let (mut basis, mut coords, mut left): (Vec<Vec<C64>>, Vec<Vec<C64>>, f64) = (Vec::new(), vec![Vec::new(); vectors.len()], 0.0);
            for &t in &order {
                let mut r = vectors[t].clone();
                let mut cs = vec![C64::ZERO; basis.len()];
                for _ in 0..2 {
                    for (k, u) in basis.iter().enumerate() {
                        let q = dot(u, &r);
                        cs[k] += q;
                        for (x, y) in r.iter_mut().zip(u) {
                            *x -= q * *y;
                        }
                    }
                }
                let rest = dot(&r, &r).re.sqrt();
                if rest * bounds[t] > tolerance * top[usize::from(h)] {
                    let inverse = 1.0 / rest;
                    basis.push(r.iter().map(|x| *x * inverse).collect());
                    cs.push(C64::real(rest));
                } else {
                    left += rest * bounds[t];
                }
                coords[t] = cs;
            }
            for c in coords.iter_mut() {
                c.resize(basis.len(), C64::ZERO);
            }
            work += 4.0 * (keys.len() * basis.len() + basis.len() * places.len()) as f64;
            dropped.push((h, out, left));
            spans.push(Span { h, out, which, places, atoms: keys.into_iter().zip(coords).map(|((p, t, c), cs)| (p, t, c, cs)).collect(), basis });
        }
        Spanned { spans, dropped, work: [plain, work] }
    }

    #[cfg(test)]
    /// The structure's sizes (a measurement's): Z sums, W sums, H pairs, jump pairs, and their products' summed
    /// rows × |B| × cols.
    pub fn sizes(&self, plan: &Plan) -> [usize; 6] {
        let size = |b: usize| plan.blocks[b].rows.len();
        let h: usize = self.h_pairs.iter().map(|p| self.ws[p.w].index.len() * size(self.ws[p.w].block) * self.zs[p.z].index.len()).sum();
        let j: usize = self.j_pairs.iter().map(|p| self.ws[p.w].index.len() * size(self.ws[p.w].block) * self.zs[p.z].index.len()).sum();
        [self.zs.len(), self.ws.len(), self.h_pairs.len(), self.j_pairs.len(), h, j]
    }
}
