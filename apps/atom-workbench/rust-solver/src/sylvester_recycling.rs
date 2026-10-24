//! Exact static Lindblad resolvents using a small Hilbert-space Schur factor
//! and the algebraic range of spontaneous-emission recycling.
//!
//! L(X) = F X + X F^H + E V(X), F = -iH - sum(C^H C)/2.
//! A shifted Sylvester solve supplies S^-1; Woodbury adds the COMPLETE E V
//! term. The range is discovered from exact nonzero collapse entries, never
//! from a small-rate or detuning threshold. This is an optional preconditioner
//! experiment; no production dispatch or physical-model approximation.
use super::{add_superterm, build_generator, complex_input, coordinate_map, coordinates, Coordinate, Lapack, Model, C};
use crate::lapack::{Complex64 as Z, SylvesterKernel};
use rayon::prelude::*;
use std::{collections::{BTreeSet, BTreeMap}, sync::{Arc, Mutex, OnceLock}};

type Sparse = Vec<Vec<(usize, f64)>>;

// Reuse workers, never model factors or computed results. A pool is selected
// once when an adapter is built, not by repeated environment reads in GMRES.
static HARMONIC_POOLS: OnceLock<Mutex<BTreeMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
fn harmonic_pool(lap:&Lapack, dimension:usize, blocks:usize)->Result<Option<Arc<rayon::ThreadPool>>,String>{
    let threads=std::env::var("ATOM_HARMONIC_THREADS").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(1);
    if threads<2||dimension<256||blocks<3||lap.configured_blas_threads()!=1{return Ok(None);}
    if threads>16{return Err("ATOM_HARMONIC_THREADS must be between 1 and 16".into());}
    let mut pools=HARMONIC_POOLS.get_or_init(||Mutex::new(BTreeMap::new())).lock()
        .map_err(|_|"Harmonic worker-pool lock was poisoned")?;
    if let Some(pool)=pools.get(&threads){return Ok(Some(pool.clone()));}
    let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads)
        .thread_name(|i|format!("atom-harmonic-{i}"))
        .build().map_err(|error|format!("Cannot create harmonic worker pool: {error}"))?);
    pools.insert(threads,pool.clone());Ok(Some(pool))
}

pub struct Prepared {
    n: usize,
    coords: Vec<Coordinate>,
    t: Vec<Z>,
    q: Vec<Z>,
    qh: Vec<Z>,
    adjoint_t: Vec<Z>,
    adjoint_q: Vec<Z>,
    adjoint_qh: Vec<Z>,
    recycling_indices: Vec<usize>,
    recycling_rows: Sparse,
    original: Sparse,
}

pub struct Resolvent {
    prepared: Arc<Prepared>,
    shift: C,
    shifted_t: Vec<Z>,
    shifted_adjoint_t: Vec<Z>,
    // Column-major density-coordinate S^-1 E, including complex Fourier shifts.
    correction_columns: Vec<C>,
    correction_lu: Vec<Z>,
    pivots: Vec<i32>,
    correction_transpose_lu: Vec<Z>,
    transpose_pivots: Vec<i32>,
}

/// Block-diagonal PRECONDITIONER for the complete Fourier operator. Only the
/// inverse device separates harmonic blocks; the outer operator still retains
/// every inter-harmonic Hamiltonian entry and checks its original residual.
pub struct HarmonicBlocks {
    physical_dimension: usize,
    blocks: Vec<(usize, bool, Resolvent)>,
    indices: Vec<usize>,
    weights: Vec<f64>,
    pool: Option<Arc<rayon::ThreadPool>>,
}

struct ComplexBlock {
    offset: usize,
    plus: Arc<Resolvent>,
    minus: Option<Arc<Resolvent>>,
}

/// Complex resolvent represented as a real 2N system. The external Krylov
/// imaginary part and each Fourier sine block are independent coordinates.
/// Diagonalizing the 2x2 harmonic pair produces the two shifts eta +/- omega.
pub struct ComplexHarmonicBlocks {
    physical_dimension: usize,
    blocks: Vec<ComplexBlock>,
    indices: Vec<usize>,
    weights: Vec<f64>,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl ComplexHarmonicBlocks {
    pub fn new(
        lap: &Lapack, model: &Model, shift: f64, pole_frequency: f64,
        blocks: &[(usize, f64, bool)], physical_dimension: usize,
        indices: &[usize], weights: &[f64],
    ) -> Result<Self, String> {
        if physical_dimension != model.dimension*model.dimension || !pole_frequency.is_finite()
            || weights.iter().any(|v| !v.is_finite() || *v <= 0.)
            || indices.iter().any(|&i| i >= weights.len()) {
            return Err("Invalid augmented Fourier preconditioner mapping".into());
        }
        let prepared = Prepared::new(lap, model)?;
        // Identical shifts within this one factorization share their factors.
        // No input model, trajectory, or prior-request result is cached.
        let mut factors = std::collections::HashMap::<u64, Arc<Resolvent>>::new();
        let mut factor = |frequency: f64| -> Result<Arc<Resolvent>, String> {
            let key = (if frequency == 0. { 0. } else { frequency }).to_bits();
            if let Some(inverse) = factors.get(&key) { return Ok(inverse.clone()); }
            let inverse = Arc::new(prepared.shifted(lap, shift, frequency)?);
            factors.insert(key, inverse.clone()); Ok(inverse)
        };
        let mut result = Vec::with_capacity(blocks.len());
        for &(offset, frequency, imaginary) in blocks {
            let end = offset.checked_add(physical_dimension*(if imaginary { 2 } else { 1 }))
                .ok_or("Augmented Fourier preconditioner dimension overflow")?;
            if end > weights.len() || (!imaginary && frequency != 0.) {
                return Err("Invalid augmented Fourier harmonic block".into());
            }
            result.push(ComplexBlock { offset, plus: factor(pole_frequency+frequency)?,
                minus: if imaginary { Some(factor(pole_frequency-frequency)?) } else { None } });
        }
        let pool=harmonic_pool(lap,physical_dimension,result.len())?;
        Ok(Self { physical_dimension, blocks: result, indices: indices.to_vec(), weights: weights.to_vec(), pool })
    }

    pub fn apply(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        self.apply_inner(lap, b, true)
    }
    pub fn apply_preconditioner(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        self.apply_inner(lap, b, false)
    }
    fn apply_inner(&self, lap: &Lapack, b: &[f64], verify_inverse: bool) -> Result<Vec<f64>, String> {
        let r = self.indices.len(); let d = self.physical_dimension;
        if b.len() != 2*r { return Err("Invalid augmented Fourier RHS dimension".into()); }
        let mut rhs = vec![C::default(); self.weights.len()];
        for (local, &global) in self.indices.iter().enumerate() {
            rhs[global] = C::new(b[local]/self.weights[global], b[r+local]/self.weights[global]);
        }
        let mut output = vec![C::default(); self.weights.len()];
        let kernel=lap.sylvester_kernel();
        let solve=|block:&ComplexBlock|->Result<(Vec<C>,Option<Vec<C>>),String>{
            let offset = block.offset;
            if let Some(minus) = &block.minus {
                let plus_rhs: Vec<_> = (0..d).map(|i| {
                    let cosine = rhs[offset+i]; let sine = rhs[offset+d+i];
                    cosine + C::new(-sine.im, sine.re)
                }).collect();
                let minus_rhs: Vec<_> = (0..d).map(|i| {
                    let cosine = rhs[offset+i]; let sine = rhs[offset+d+i];
                    cosine + C::new(sine.im, -sine.re)
                }).collect();
                let (plus,minus)=if self.pool.is_some(){
                    rayon::join(||block.plus.apply_complex_inner(&kernel,&plus_rhs,verify_inverse),
                        ||minus.apply_complex_inner(&kernel,&minus_rhs,verify_inverse))
                }else{(block.plus.apply_complex_inner(&kernel,&plus_rhs,verify_inverse),
                    minus.apply_complex_inner(&kernel,&minus_rhs,verify_inverse))};
                Ok((plus?,Some(minus?)))
            } else {
                Ok((block.plus.apply_complex_inner(&kernel,&rhs[offset..offset+d],verify_inverse)?,None))
            }
        };
        let solved:Result<Vec<_>,String>=if let Some(pool)=&self.pool {
            pool.install(||self.blocks.par_iter().map(solve).collect())
        }else{self.blocks.iter().map(solve).collect()};
        // Scatter in the original block order. Workers never share an output
        // or scratch buffer, and no parallel reduction changes floating sums.
        for (block,(plus,minus)) in self.blocks.iter().zip(solved?) {
            let offset=block.offset;
            if let Some(minus)=minus {
                for i in 0..d {
                    output[offset+i] = (plus[i]+minus[i])*(0.5*self.weights[offset+i]);
                    let difference = plus[i]-minus[i];
                    output[offset+d+i] = C::new(difference.im, -difference.re)*(0.5*self.weights[offset+d+i]);
                }
            } else {
                for (i, value) in plus.into_iter().enumerate() { output[offset+i] = value*self.weights[offset+i]; }
            }
        }
        let mut result = Vec::with_capacity(2*r);
        result.extend(self.indices.iter().map(|&global| output[global].re));
        result.extend(self.indices.iter().map(|&global| output[global].im));
        if result.iter().any(|x| !x.is_finite()) { return Err("Non-finite augmented Fourier preconditioner output".into()); }
        Ok(result)
    }
}

impl HarmonicBlocks {
    pub fn new(
        lap: &Lapack, model: &Model, shift: f64, blocks: &[(usize, f64, bool)],
        physical_dimension: usize, indices: &[usize], weights: &[f64],
    ) -> Result<Self, String> {
        if physical_dimension != model.dimension*model.dimension
            || weights.iter().any(|v| !v.is_finite() || *v <= 0.)
            || indices.iter().any(|&i| i >= weights.len()) {
            return Err("Invalid Hilbert/Fourier preconditioner coordinate mapping".into());
        }
        let prepared = Prepared::new(lap, model)?;
        let mut result = Vec::with_capacity(blocks.len());
        for &(offset, frequency, imaginary) in blocks {
            let end = offset.checked_add(physical_dimension*(if imaginary { 2 } else { 1 }))
                .ok_or("Fourier preconditioner dimension overflow")?;
            if end > weights.len() || (!imaginary && frequency != 0.) {
                return Err("Invalid Fourier harmonic block".into());
            }
            // z_n'=(L0-i*n.omega) z_n, hence the shifted inverse has +i*n.omega.
            result.push((offset, imaginary, prepared.shifted(lap, shift, frequency)?));
        }
        let pool=harmonic_pool(lap,physical_dimension,result.len())?;
        Ok(Self { physical_dimension, blocks: result, indices: indices.to_vec(), weights: weights.to_vec(), pool })
    }

    pub fn apply(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        self.apply_inner(lap, b, true, false)
    }

    /// The outer GMRES still validates the COMPLETE original operator at its
    /// original tolerance. An approximate inverse used solely as a
    /// preconditioner does not itself need to solve each block to that tolerance.
    pub fn apply_preconditioner(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        self.apply_inner(lap, b, false, false)
    }

    pub fn apply_transpose(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        self.apply_inner(lap, b, true, true)
    }

    pub fn apply_preconditioner_transpose(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        self.apply_inner(lap, b, false, true)
    }

    fn apply_inner(&self, lap: &Lapack, b: &[f64], verify_inverse: bool, transpose: bool) -> Result<Vec<f64>, String> {
        if b.len() != self.indices.len() { return Err("Invalid Fourier preconditioner RHS dimension".into()); }
        let mut rhs = vec![0.; self.weights.len()];
        for (&global, &value) in self.indices.iter().zip(b) {
            rhs[global] = value*if transpose { self.weights[global] } else { 1./self.weights[global] };
        }
        let mut output = vec![0.; self.weights.len()];
        let d = self.physical_dimension;
        let kernel=lap.sylvester_kernel();
        let solve=|(offset,imaginary,inverse):&(usize,bool,Resolvent)|->Result<Vec<C>,String>{
            let coefficients: Vec<_> = (0..d).map(|i| C::new(rhs[offset+i],
                if *imaginary { rhs[offset+d+i]*if transpose {-1.}else{1.} } else { 0. })).collect();
            if transpose { inverse.apply_complex_transpose_inner(&kernel,&coefficients,verify_inverse) }
            else { inverse.apply_complex_inner(&kernel,&coefficients,verify_inverse) }
        };
        let solved:Result<Vec<_>,String>=if let Some(pool)=&self.pool {
            pool.install(||self.blocks.par_iter().map(solve).collect())
        }else{self.blocks.iter().map(solve).collect()};
        for ((offset,imaginary,_),values) in self.blocks.iter().zip(solved?) {
            for (i, value) in values.into_iter().enumerate() {
                output[offset+i] = value.re*if transpose {1./self.weights[offset+i]}else{self.weights[offset+i]};
                if *imaginary { output[offset+d+i] = value.im*if transpose {-1./self.weights[offset+d+i]}else{self.weights[offset+d+i]}; }
            }
        }
        Ok(self.indices.iter().map(|&global| output[global]).collect())
    }
}

fn z(v: C) -> Z { Z { re: v.re, im: v.im } }
fn c(v: Z) -> C { C::new(v.re, v.im) }
fn negative(v: C) -> C { C::new(-v.re, -v.im) }

impl Prepared {
    pub fn new(lap: &Lapack, model: &Model) -> Result<Arc<Self>, String> {
        // This also verifies dimensions and Hermiticity against the original
        // model builder. Its sparse matrix remains the final residual oracle.
        let original = build_generator(model)?.matrix;
        let n = model.dimension;
        let d = n*n;
        let h = complex_input(&model.h_real, &model.h_imag, d, "Sylvester Hamiltonian")?;
        let coords = coordinates(n);
        let map = coordinate_map(&coords, n);
        let mut recycling: Sparse = vec![Vec::new(); d];
        let mut diagonal_imag = vec![0.; n*d];
        let mut gram = vec![C::default(); d];
        for operator in &model.collapse {
            let values = complex_input(&operator.real, &operator.imag, d, "Sylvester collapse")?;
            let entries: Vec<_> = values.iter().enumerate()
                .filter(|(_, v)| v.re != 0. || v.im != 0.)
                .map(|(index, &v)| (index/n, index%n, v)).collect();
            for &(a, i, v) in &entries {
                for &(b, j, w) in &entries {
                    add_superterm(&mut recycling, &mut diagonal_imag, &map,
                        a*n+b, i*n+j, v*w.conj(), n);
                    if a == b { gram[i*n+j] += v.conj()*w; }
                }
            }
        }
        if diagonal_imag.iter().any(|v| v.abs() > 1e-9) {
            return Err("Recycling map does not preserve Hermiticity".into());
        }
        for column in &mut recycling {
            column.sort_by_key(|&(row, _)| row);
            let mut summed: Vec<(usize, f64)> = Vec::new();
            for &(row, value) in column.iter() {
                if let Some(last) = summed.last_mut().filter(|last| last.0 == row) {
                    last.1 += value;
                } else { summed.push((row, value)); }
            }
            summed.retain(|&(_, value)| value != 0.);
            *column = summed;
        }
        let recycling_indices: Vec<_> = recycling.iter().flatten()
            .map(|&(row, _)| row).collect::<BTreeSet<_>>().into_iter().collect();
        if recycling_indices.len() > 128 {
            return Err("Recycling range exceeds this optional resolvent's 128-coordinate budget".into());
        }
        let mut positions = vec![usize::MAX; d];
        for (i, &row) in recycling_indices.iter().enumerate() { positions[row] = i; }
        let mut recycling_rows = vec![Vec::new(); recycling_indices.len()];
        for (column, entries) in recycling.iter().enumerate() {
            for &(row, value) in entries { recycling_rows[positions[row]].push((column, value)); }
        }
        let mut f = vec![Z::default(); d];
        for i in 0..n {
            for j in 0..n {
                let hij = h[i*n+j];
                let gij = gram[i*n+j];
                f[i+j*n] = Z { re: hij.im - 0.5*gij.re, im: -hij.re - 0.5*gij.im };
            }
        }
        let (t, q) = lap.complex_schur(&f, n)?;
        let qh = (0..d).map(|index| {
            let v = q[index/n + (index%n)*n]; Z { re: v.re, im: -v.im }
        }).collect();
        // If F=Q T Q^H, then F^H=(Q J)(J T^H J)(Q J)^H,
        // where J reverses coordinates. J T^H J is upper triangular, so the
        // SAME mature triangular Sylvester kernel solves the Hilbert adjoint.
        // This is not a replacement C -> C^H Lindblad construction.
        let adjoint_t=(0..d).map(|index|{
            let i=index%n;let j=index/n;let value=t[n-1-j+(n-1-i)*n];
            Z{re:value.re,im:-value.im}
        }).collect();
        let adjoint_q:Vec<_>=(0..d).map(|index|q[index%n+(n-1-index/n)*n]).collect();
        let adjoint_qh=(0..d).map(|index|{
            let value=adjoint_q[index/n+(index%n)*n];Z{re:value.re,im:-value.im}
        }).collect();
        Ok(Arc::new(Self { n, coords, t, q, qh, adjoint_t, adjoint_q, adjoint_qh,
            recycling_indices, recycling_rows, original }))
    }

    pub fn recycling_dimension(&self) -> usize { self.recycling_indices.len() }

    /// Positive real part gives a bounded resolvent of a Lindblad generator.
    /// A nonzero imaginary part handles paired Fourier harmonic blocks.
    pub fn shifted(self: &Arc<Self>, lap: &Lapack, real: f64, frequency: f64) -> Result<Resolvent, String> {
        if !real.is_finite() || real <= 0. || !frequency.is_finite() {
            return Err("Sylvester recycling resolvent requires a positive finite real shift".into());
        }
        let mut shifted_t: Vec<_> = self.t.iter().map(|v| Z { re: -v.re, im: -v.im }).collect();
        let mut shifted_adjoint_t: Vec<_> = self.adjoint_t.iter().map(|v| Z { re: -v.re, im: -v.im }).collect();
        for i in 0..self.n { shifted_t[i+i*self.n].re += real; shifted_t[i+i*self.n].im += frequency; }
        for i in 0..self.n { shifted_adjoint_t[i+i*self.n].re += real; shifted_adjoint_t[i+i*self.n].im += frequency; }
        let rank = self.recycling_indices.len();
        let d = self.n*self.n;
        let mut resolver = Resolvent { prepared: self.clone(), shift: C::new(real, frequency), shifted_t, shifted_adjoint_t,
            correction_columns: Vec::with_capacity(d*rank), correction_lu: Vec::new(), pivots: Vec::new(),
            correction_transpose_lu:Vec::new(),transpose_pivots:Vec::new() };
        let mut correction = vec![Z::default(); rank*rank];
        for (j, &coordinate) in self.recycling_indices.iter().enumerate() {
            let mut basis = vec![C::default(); d]; basis[coordinate].re = 1.;
            let column = resolver.no_jump(&lap.sylvester_kernel(), &basis)?;
            for (i, v) in self.recycle(&column).iter().enumerate() {
                correction[i+j*rank] = Z { re: (if i == j { 1. } else { 0. }) - v.re, im: -v.im };
            }
            resolver.correction_columns.extend(column);
        }
        if rank != 0 {
            (resolver.correction_lu, resolver.pivots) = lap.complex_factor(&correction, rank)?;
            let transposed:Vec<_>=(0..rank*rank).map(|index|correction[index/rank+(index%rank)*rank]).collect();
            (resolver.correction_transpose_lu,resolver.transpose_pivots)=lap.complex_factor(&transposed,rank)?;
        }
        Ok(resolver)
    }

    fn recycle(&self, x: &[C]) -> Vec<C> {
        self.recycling_rows.iter().map(|row| {
            let mut value = C::default();
            for &(i, scale) in row { value += x[i]*scale; }
            value
        }).collect()
    }

    // Coefficients multiply the real Hermitian basis, but can themselves be
    // complex for Fourier shifts. Do not discard either coefficient part.
    fn matrix(&self, values: &[C]) -> Vec<Z> {
        let mut matrix = vec![Z::default(); self.n*self.n];
        for (&coordinate, &value) in self.coords.iter().zip(values) {
            match coordinate {
                Coordinate::Population(i) => matrix[i+i*self.n] = z(value),
                Coordinate::Real(i, j) => { matrix[i+j*self.n] = z(value); matrix[j+i*self.n] = z(value); }
                Coordinate::Imag(i, j) => {
                    let iv = C::new(-value.im, value.re);
                    let upper = i+j*self.n; let lower = j+i*self.n;
                    matrix[upper] = z(c(matrix[upper]) + iv);
                    matrix[lower] = z(c(matrix[lower]) + negative(iv));
                }
            }
        }
        matrix
    }
    fn coefficients(&self, matrix: &[Z]) -> Vec<C> {
        self.coords.iter().map(|&coordinate| match coordinate {
            Coordinate::Population(i) => c(matrix[i+i*self.n]),
            Coordinate::Real(i, j) => (c(matrix[i+j*self.n]) + c(matrix[j+i*self.n]))*0.5,
            Coordinate::Imag(i, j) => {
                let difference = c(matrix[i+j*self.n]) - c(matrix[j+i*self.n]);
                C::new(difference.im*0.5, -difference.re*0.5)
            }
        }).collect()
    }
}

impl Resolvent {
    fn no_jump(&self, lap: &SylvesterKernel, rhs: &[C]) -> Result<Vec<C>, String> {
        self.no_jump_basis(lap,rhs,&self.shifted_t,&self.prepared.t,&self.prepared.q,&self.prepared.qh)
    }
    fn no_jump_basis(&self,lap:&SylvesterKernel,rhs:&[C],shifted_t:&[Z],t:&[Z],q:&[Z],qh:&[Z])->Result<Vec<C>,String>{
        let p = &self.prepared; let n = p.n;
        let physical = p.matrix(rhs);
        let transformed = lap.multiply_complex(&lap.multiply_complex(qh, &physical, n, n, n)?, q, n, n, n)?;
        let solved = lap.complex_sylvester(shifted_t,t,&transformed,n)?;
        let physical = lap.multiply_complex(&lap.multiply_complex(q, &solved, n, n, n)?, qh, n, n, n)?;
        Ok(p.coefficients(&physical))
    }

    fn no_jump_transpose(&self,lap:&SylvesterKernel,rhs:&[C])->Result<Vec<C>,String>{
        let p=&self.prepared;
        // The real Hermitian coordinate basis has Gram diag(1 populations,
        // 2 coherences). Coordinate transpose = G * Hilbert-adjoint * G^-1.
        let scaled:Vec<_>=rhs.iter().enumerate().map(|(i,&v)|v*if i<p.n{1.}else{0.5}).collect();
        let mut result=self.no_jump_basis(lap,&scaled,&self.shifted_adjoint_t,&p.adjoint_t,&p.adjoint_q,&p.adjoint_qh)?;
        for(i,v)in result.iter_mut().enumerate(){if i>=p.n{*v=*v*2.;}}
        Ok(result)
    }

    fn apply_complex_transpose_inner(&self,lap:&SylvesterKernel,rhs:&[C],verify_inverse:bool)->Result<Vec<C>,String>{
        let p=&self.prepared;let d=p.n*p.n;let rank=p.recycling_indices.len();
        if rhs.len()!=d||rhs.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){
            return Err("Invalid transposed recycling-resolvent RHS".into());
        }
        // A=S-EV, Y=S^-1 E, M=I-VY. A^-T b=S^-T[b+V^T M^-T Y^T b].
        // These are coefficient transposes: NO complex conjugation in Y^T/M^T.
        let mut corrected=rhs.to_vec();
        if rank!=0{
            let reduced:Vec<_>=(0..rank).map(|j|{
                let mut value=C::default();for(i,&b)in rhs.iter().enumerate(){value+=self.correction_columns[i+j*d]*b;}z(value)
            }).collect();
            let amount=lap.complex_solve_factored(&self.correction_transpose_lu,&self.transpose_pivots,&reduced,rank)?;
            for(j,row)in p.recycling_rows.iter().enumerate(){for &(i,value)in row{corrected[i]+=c(amount[j])*value;}}
        }
        let result=self.no_jump_transpose(lap,&corrected)?;
        if result.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){return Err("Non-finite transposed recycling inverse".into());}
        if verify_inverse{
            let mut residual:Vec<_>=result.iter().zip(rhs).map(|(&x,&b)|self.shift*x-b).collect();
            let mut magnitude:Vec<_>=result.iter().zip(rhs).map(|(&x,&b)|self.shift.abs()*x.abs()+b.abs()).collect();
            for(j,column)in p.original.iter().enumerate(){for &(i,value)in column{
                residual[j]=residual[j]-result[i]*value;magnitude[j]+=result[i].abs()*value.abs();
            }}
            let scale=rhs.iter().map(|v|v.abs()).fold(0.,f64::max).max(1e-30);
            if residual.iter().zip(magnitude).any(|(r,bound)|!r.abs().is_finite()||r.abs()>(1e-10*scale).max(64.*f64::EPSILON*bound)){
                return Err("Transposed Sylvester recycling inverse failed original-system residual".into());
            }
        }
        Ok(result)
    }

    pub fn apply(&self, lap: &Lapack, rhs: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let values: Vec<_> = rhs.iter().map(|&re| C::new(re, 0.)).collect();
        let result = self.apply_complex(lap, &values)?;
        Ok((result.iter().map(|z| z.re).collect(), result.iter().map(|z| z.im).collect()))
    }

    fn apply_complex(&self, lap: &Lapack, rhs: &[C]) -> Result<Vec<C>, String> {
        self.apply_complex_inner(&lap.sylvester_kernel(), rhs, true)
    }

    fn apply_complex_inner(&self, lap: &SylvesterKernel, rhs: &[C], verify_inverse: bool) -> Result<Vec<C>, String> {
        let p = &self.prepared; let d = p.n*p.n; let rank = p.recycling_indices.len();
        if rhs.len() != d || rhs.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err("Invalid recycling-resolvent right-hand side".into());
        }
        let mut result = self.no_jump(lap, rhs)?;
        if rank != 0 {
            let reduced: Vec<_> = p.recycle(&result).into_iter().map(z).collect();
            let correction = lap.complex_solve_factored(&self.correction_lu, &self.pivots, &reduced, rank)?;
            for (j, &amount) in correction.iter().enumerate() {
                for (i, value) in result.iter_mut().enumerate() {
                    *value += self.correction_columns[i+j*d]*c(amount);
                }
            }
        }
        if result.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
            return Err("Non-finite recycling inverse application".into());
        }
        if !verify_inverse { return Ok(result); }
        // A unitary Schur factor avoids eigenvector conditioning; still verify
        // the reconstructed solution against EVERY original generator entry.
        let mut residual: Vec<_> = result.iter().zip(rhs).map(|(&x, &b)| self.shift*x - b).collect();
        let mut magnitude: Vec<_> = result.iter().zip(rhs).map(|(&x, &b)| self.shift.abs()*x.abs() + b.abs()).collect();
        for (j, column) in p.original.iter().enumerate() {
            for &(i, value) in column {
                residual[i] = residual[i] - result[j]*value;
                magnitude[i] += result[j].abs()*value.abs();
            }
        }
        let scale = rhs.iter().map(|v| v.abs()).fold(0., f64::max).max(1e-30);
        if residual.iter().zip(magnitude).any(|(r, bound)| !r.abs().is_finite()
            || r.abs() > (1e-10*scale).max(64.*f64::EPSILON*bound)) {
            return Err("Sylvester recycling inverse failed the full original-system residual".into());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Operator;

    fn model() -> Model {
        Model { frequency_coordinates:None, dimension: 3, states: vec![], h_real: vec![0., 0.3, 0.2, 0.3, 1.2, -0.4, 0.2, -0.4, 3.4],
            h_imag: vec![0., 0.1, -0.2, -0.1, 0., 0.25, 0.2, -0.25, 0.],
            collapse: vec![Operator { real: vec![0., 0., 0.7, 0., 0., 0.2, 0., 0., 0.],
                imag: vec![0., 0., 0., 0., 0., 0.3, 0., 0., 0.] },
                Operator { real: vec![0., 0.2, 0., 0., 0., 0., 0., 0., 0.], imag: vec![0.; 9] }],
            initial_index: 0, initial_real: None, initial_imag: None, time_dependent: false, drives: vec![] }
    }
    #[test]
    fn parallel_harmonic_applications_match_serial_and_original_matrix_residual() {
        fn assert_send_sync<T:Send+Sync>(){}
        assert_send_sync::<SylvesterKernel<'static>>();
        assert_send_sync::<Prepared>();assert_send_sync::<Resolvent>();
        let lap=Lapack::load().unwrap();let model=model();let g=build_generator(&model).unwrap();
        let d=model.dimension*model.dimension;let r=5*d;let shift=0.7;let pole=1.1;
        let blocks=[(0,0.,false),(d,2.3,true),(3*d,5.1,true)];
        let indices:Vec<_>=(0..r).collect();
        let weights:Vec<_>=indices.iter().map(|&i|((if i%d<model.dimension{1_f64}else{2.})*if i>=d{2.}else{1.}).sqrt()).collect();
        let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap());
        // Tiny deterministic fixture explicitly bypasses the production work
        // threshold; ordinary small models stay on the serial path.
        let mut real=HarmonicBlocks::new(&lap,&model,shift,&blocks,d,&indices,&weights).unwrap();
        real.pool=None;
        let mut dense=vec![0.;r*r];
        for i in 0..r{dense[i+i*r]=shift;}
        for &(offset,frequency,imaginary) in &blocks{
            for part in 0..if imaginary{2}else{1}{for(j,col)in g.matrix.iter().enumerate(){for&(i,a)in col{
                let row=offset+part*d+i;let column=offset+part*d+j;
                dense[row+column*r]-=a*weights[row]/weights[column];
            }}}
            if imaginary{for i in 0..d{dense[offset+i+(offset+d+i)*r]-=frequency;dense[offset+d+i+(offset+i)*r]+=frequency;}}
        }
        let expected:Vec<_>=(0..r).map(|i|(i as f64+0.4).cos()).collect();
        let rhs:Vec<_>=(0..r).map(|i|(0..r).map(|j|dense[i+j*r]*expected[j]).sum()).collect();
        let serial=real.apply(&lap,&rhs).unwrap();real.pool=Some(pool.clone());
        let parallel=real.apply(&lap,&rhs).unwrap();assert_eq!(parallel,serial);
        assert_eq!(real.apply_preconditioner(&lap,&rhs).unwrap(),serial);
        for i in 0..r{assert!(((0..r).map(|j|dense[i+j*r]*parallel[j]).sum::<f64>()-rhs[i]).abs()<1e-10);}

        let mut complex=ComplexHarmonicBlocks::new(&lap,&model,shift,pole,&blocks,d,&indices,&weights).unwrap();
        complex.pool=None;let n=2*r;let mut augmented=vec![0.;n*n];
        for part in 0..2{for i in 0..r{for j in 0..r{augmented[part*r+i+(part*r+j)*n]=dense[i+j*r];}}}
        for i in 0..r{augmented[i+(r+i)*n]=-pole;augmented[r+i+i*n]=pole;}
        let expected:Vec<_>=(0..n).map(|i|(i as f64+0.7).sin()).collect();
        let rhs:Vec<_>=(0..n).map(|i|(0..n).map(|j|augmented[i+j*n]*expected[j]).sum()).collect();
        let serial=complex.apply(&lap,&rhs).unwrap();complex.pool=Some(pool);
        let parallel=complex.apply(&lap,&rhs).unwrap();assert_eq!(parallel,serial);
        assert_eq!(complex.apply_preconditioner(&lap,&rhs).unwrap(),serial);
        for i in 0..n{assert!(((0..n).map(|j|augmented[i+j*n]*parallel[j]).sum::<f64>()-rhs[i]).abs()<1e-10);}
    }

    #[test]
    fn sylvester_recycling_matches_original_dense_complex_resolvent() {
        let lap = Lapack::load().unwrap();
        let model = model(); let g = build_generator(&model).unwrap(); let d = 9;
        let prepared = Prepared::new(&lap, &model).unwrap();
        assert_eq!(prepared.recycling_dimension(), 4);
        for (real, frequency) in [(0.03, 0.), (0.2, 1.7), (7., -5.2)] {
            let inverse = prepared.shifted(&lap, real, frequency).unwrap();
            let rhs: Vec<_> = (0..d).map(|i| C::new((i as f64+0.2).sin(), (i as f64+0.7).cos())).collect();
            let result = inverse.apply_complex(&lap, &rhs).unwrap();
            let mut dense = vec![Z::default(); d*d];
            for i in 0..d { dense[i+i*d] = Z { re: real, im: frequency }; }
            for (j, column) in g.matrix.iter().enumerate() {
                for &(i, value) in column { dense[i+j*d].re -= value; }
            }
            let (lu, pivots) = lap.complex_factor(&dense, d).unwrap();
            let expected = lap.complex_solve_factored(&lu, &pivots, &rhs.iter().copied().map(z).collect::<Vec<_>>(), d).unwrap();
            let error = result.iter().zip(expected).map(|(&a, b)| (a-c(b)).abs()).fold(0., f64::max);
            assert!(error < 2e-10, "shift={real}+i{frequency}, error={error:e}");
        }
    }
    #[test]
    fn coefficient_transpose_preserves_complex_rhs_and_complete_recycling(){
        let lap=Lapack::load().unwrap();let model=model();let g=build_generator(&model).unwrap();let d=9;
        let prepared=Prepared::new(&lap,&model).unwrap();
        for(real,frequency)in[(0.001,0.),(0.2,1.7),(7.,-5.2)]{
            let inverse=prepared.shifted(&lap,real,frequency).unwrap();
            let rhs:Vec<_>=(0..d).map(|i|C::new((i as f64+0.2).sin(),(i as f64+0.7).cos())).collect();
            let actual=inverse.apply_complex_transpose_inner(&lap.sylvester_kernel(),&rhs,true).unwrap();
            let mut transposed=vec![Z::default();d*d];
            for i in 0..d{transposed[i+i*d]=Z{re:real,im:frequency};}
            for(j,column)in g.matrix.iter().enumerate(){for &(i,value)in column{transposed[j+i*d].re-=value;}}
            let(lu,pivots)=lap.complex_factor(&transposed,d).unwrap();
            let expected=lap.complex_solve_factored(&lu,&pivots,&rhs.iter().copied().map(z).collect::<Vec<_>>(),d).unwrap();
            let error=actual.iter().zip(expected).map(|(&a,b)|(a-c(b)).abs()).fold(0.,f64::max);
            assert!(error<2e-8,"shift={real}+i{frequency}, transpose error={error:e}");
        }
    }
    #[test]
    fn sylvester_zero_recycling_and_zero_shift_guard() {
        let lap = Lapack::load().unwrap(); let mut model = model(); model.collapse.clear();
        let prepared = Prepared::new(&lap, &model).unwrap();
        assert_eq!(prepared.recycling_dimension(), 0);
        assert!(prepared.shifted(&lap, 0., 1.).is_err());
        let inverse = prepared.shifted(&lap, 0.5, 0.).unwrap();
        let (real, imag) = inverse.apply(&lap, &[1., 0., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
        assert!(imag.iter().all(|v| v.abs() < 1e-12));
        assert!((real[..3].iter().sum::<f64>()-2.).abs() < 1e-11);
    }
    #[test]
    fn sylvester_harmonic_adapter_preserves_weighted_real_and_imaginary_blocks() {
        let lap = Lapack::load().unwrap(); let model = model();
        let g = build_generator(&model).unwrap(); let d = model.dimension*model.dimension;
        let blocks = [(0, 0., false), (d, 2.3, true)]; let shift = 0.7;
        let indices: Vec<_> = (0..3*d).collect();
        let weights: Vec<_> = indices.iter().map(|&i|
            ((if i%d < model.dimension { 1_f64 } else { 2. }) * if i >= d { 2. } else { 1. }).sqrt()).collect();
        let inverse = HarmonicBlocks::new(&lap, &model, shift, &blocks, d, &indices, &weights).unwrap();
        let expected: Vec<_> = indices.iter().map(|&i| (i as f64 + 0.4).cos()).collect();
        let raw: Vec<_> = expected.iter().zip(&weights).map(|(&x, &w)| x/w).collect();
        let mut rhs: Vec<_> = raw.iter().map(|&x| shift*x).collect();
        for (offset, frequency, imaginary) in blocks {
            for part in 0..if imaginary { 2 } else { 1 } {
                for (j, col) in g.matrix.iter().enumerate() {
                    for &(i, a) in col { rhs[offset+part*d+i] -= a*raw[offset+part*d+j]; }
                }
            }
            if imaginary { for i in 0..d { rhs[offset+i] -= frequency*raw[offset+d+i]; rhs[offset+d+i] += frequency*raw[offset+i]; } }
        }
        for (x, &w) in rhs.iter_mut().zip(&weights) { *x *= w; }
        let result = inverse.apply(&lap, &rhs).unwrap();
        assert!(result.iter().zip(expected).all(|(a, b)| (a-b).abs() < 1e-10));
        let expected:Vec<_>=(0..3*d).map(|i|(i as f64+0.8).cos()).collect();
        let mut rhs=vec![0.;3*d];
        for &(offset,frequency,imaginary)in &blocks{
            for part in 0..if imaginary{2}else{1}{
                let base=offset+part*d;
                for i in 0..d{rhs[base+i]+=shift*expected[base+i];}
                for(j,column)in g.matrix.iter().enumerate(){for &(i,value)in column{
                    rhs[base+j]-=value*weights[base+i]/weights[base+j]*expected[base+i];
                }}
            }
            if imaginary{for i in 0..d{
                rhs[offset+d+i]-=frequency*expected[offset+i];
                rhs[offset+i]+=frequency*expected[offset+d+i];
            }}
        }
        let result=inverse.apply_transpose(&lap,&rhs).unwrap();
        assert!(result.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));
    }
    #[test]
    fn approximate_sylvester_preconditioner_preserves_original_outer_residual_gate() {
        let lap = Lapack::load().unwrap(); let model = model();
        let g = build_generator(&model).unwrap(); let d = model.dimension*model.dimension; let shift = 0.3;
        let mut inverse = HarmonicBlocks::new(&lap, &model, shift, &[(0, 0., false)], d,
            &(0..d).collect::<Vec<_>>(), &vec![1.; d]).unwrap();
        // Deliberately degrade only the optional inverse device. It must fail
        // standalone inverse validation, yet can still precondition a solve
        // whose final acceptance uses every entry of the untouched generator.
        for value in &mut inverse.blocks[0].2.correction_columns { *value = *value*1.01; }
        let mut matrix: Sparse = g.matrix.iter().map(|col| col.iter().map(|&(i, a)| (i, -a)).collect()).collect();
        for (i, col) in matrix.iter_mut().enumerate() { col.push((i, shift)); }
        let expected: Vec<_> = (0..d).map(|i| (i as f64+0.2).sin()).collect();
        let mut rhs = vec![0.; d];
        for (j, col) in matrix.iter().enumerate() { for &(i, a) in col { rhs[i] += a*expected[j]; } }
        assert!(inverse.apply(&lap, &rhs).is_err());
        assert!(inverse.apply_preconditioner(&lap, &rhs).unwrap().iter().all(|x| x.is_finite()));
        let outer = crate::block_solve::Prepared::with_custom(&matrix,
            Box::new(move |lap, rhs| inverse.apply_preconditioner(lap, rhs))).unwrap();
        let actual = outer.solve(&lap, &rhs).unwrap();
        assert!(actual.iter().zip(expected).all(|(a, b)| (a-b).abs() < 1e-10));
    }
    #[test]
    fn complex_harmonic_adapter_keeps_external_and_fourier_imaginary_parts_independent() {
        let lap = Lapack::load().unwrap(); let model = model();
        let g = build_generator(&model).unwrap(); let d = model.dimension*model.dimension; let r = 3*d;
        let blocks = [(0, 0., false), (d, 2.3, true)]; let shift = 0.7;
        let indices: Vec<_> = (0..r).collect();
        let weights: Vec<_> = indices.iter().map(|&i|
            ((if i%d < model.dimension { 1_f64 } else { 2. }) * if i >= d { 2. } else { 1. }).sqrt()).collect();
        for pole in [0., 1.1, -3.7] {
            let inverse = ComplexHarmonicBlocks::new(&lap, &model, shift, pole, &blocks, d, &indices, &weights).unwrap();
            let expected: Vec<_> = (0..2*r).map(|i| (i as f64+0.4).cos()).collect();
            let mut dense = vec![0.; 4*r*r]; let n = 2*r;
            for outer_part in 0..2 {
                for i in 0..r { dense[outer_part*r+i+(outer_part*r+i)*n] = shift; }
                for &(offset, frequency, imaginary) in &blocks {
                    for part in 0..if imaginary { 2 } else { 1 } {
                        for (j, col) in g.matrix.iter().enumerate() { for &(i, a) in col {
                            let row = offset+part*d+i; let column = offset+part*d+j;
                            dense[outer_part*r+row+(outer_part*r+column)*n] -= a*weights[row]/weights[column];
                        } }
                    }
                    if imaginary { for i in 0..d {
                        let re = outer_part*r+offset+i; let im = outer_part*r+offset+d+i;
                        dense[re+im*n] -= frequency; dense[im+re*n] += frequency;
                    } }
                }
            }
            for i in 0..r { dense[i+(r+i)*n] = -pole; dense[r+i+i*n] = pole; }
            let rhs: Vec<_> = (0..n).map(|i| (0..n).map(|j| dense[i+j*n]*expected[j]).sum()).collect();
            let actual = inverse.apply(&lap, &rhs).unwrap();
            let dense_reference = lap.solve(&dense, &rhs, n).unwrap();
            assert!(actual.iter().zip(&expected).all(|(a, b)| (a-b).abs() < 2e-10));
            assert!(actual.iter().zip(dense_reference).all(|(a, b)| (a-b).abs() < 2e-10));
        }
    }
    #[test]
    #[ignore = "Focused fixed32 optional-resolvent timing and residual experiment"]
    fn sylvester_rb87_positive_shift_profile() {
        let request: crate::Request = serde_json::from_str(include_str!("../tests/fixtures/strong-two-carrier.json")).unwrap();
        let model = request.model.unwrap(); let lap = Lapack::load().unwrap();
        let start = std::time::Instant::now();
        let prepared = Prepared::new(&lap, &model).unwrap();
        let setup_ms = start.elapsed().as_secs_f64()*1000.;
        let d = model.dimension*model.dimension;
        assert_eq!(prepared.recycling_dimension(), 34);
        let g = build_generator(&model).unwrap();
        for (shift, frequency) in [(0.0016, 0.), (7.95, 0.), (7.95, model.drives[0].omega)] {
            let before = std::time::Instant::now();
            let inverse = prepared.shifted(&lap, shift, frequency).unwrap();
            let factor_ms = before.elapsed().as_secs_f64()*1000.;
            let expected: Vec<_> = (0..d).map(|i| (0.2+i as f64).sin()).collect();
            let mut rhs: Vec<_> = expected.iter().map(|&v| C::new(shift*v, frequency*v)).collect();
            for (j, col) in g.matrix.iter().enumerate() { for &(i, a) in col { rhs[i].re -= a*expected[j]; } }
            let before = std::time::Instant::now();
            let result = inverse.apply_complex(&lap, &rhs).unwrap();
            let apply_ms = before.elapsed().as_secs_f64()*1000.;
            let difference = result.iter().zip(&expected).map(|(&a, &b)| (a-C::new(b, 0.)).abs()).fold(0., f64::max);
            println!("Sylvester fixed32 rank={} shift={shift} frequency={frequency} setupMs={setup_ms:.3} factorMs={factor_ms:.3} applyMs={apply_ms:.3} manufacturedSolutionError={difference:e}", prepared.recycling_dimension());
            assert!(difference < 1e-4, "The preconditioner must resolve all static coordinates, not populations alone");
        }
    }
}
