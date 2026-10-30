//! The a-priori amplitude rule: what x0 alone says each charge carries,
//! before any solve. One rule decides the run's lattice completion, its
//! translations and the far shell's depth (`pipeline`).
//!
//! The transfers: in the frame's eigen-coordinates of L0' (c0 = E^-1 x0),
//! the first-order Schrieffer-Wolff kernels X~_s (`kernels`) carry x0 to
//! the charge s, X~_s c0, and the second order carries that on to s + s',
//! X~_s' X~_s c0: the slip each charge's family starts with. A charge's
//! amplitude is that slip propagated by L0''s own decay rates,
//! sum_j e^{lambda_j t} (slip)_j (Re lambda_j clipped at zero), read as
//! populations through E's population rows and doubled (the mirror charge
//! carries the conjugate), the largest over the run's sample times (a
//! trajectory's grid sampled geometrically, a mean's horizons). A charge is
//! carried when its amplitude is above GATE x tolerance (`charge_set::GATE`,
//! the translation gate: a tenth of the tolerance).
//!
//! What a carried charge gets: on the lattice it is a translation of the
//! space, its slip a start vector of the block (`space`); off the lattice it
//! joins it (the completion). The far shell's depth reads the same rule: a
//! ring's landing on x0's a-priori field (x0 on the centre and every lattice
//! charge's slip, `field`) is a source on the lattice, its amplitude the
//! response that source drives through each mode's own block, sum_j
//! (e^{mu_j t} - 1) / mu_j (c_q)_j with mu_j = lambda_j - i nu_q (Re clipped at
//! zero) and c_q = E^-1 (the landing on q), read as populations
//! (`response`).
use super::*;
use super::eigenbasis::BlockEigenbasis;
use super::kernels::SwKernels;

/// A trajectory's sample times for the amplitudes: grid index 1, then each
/// index GEOMETRIC x the last (at least one more), to the grid's end.
const GEOMETRIC:f64=1.5;

/// The a-priori transfers of x0 and their amplitudes, per charge.
pub struct Apriori{
    /// Per charge (not the centre): its first- and second-order slip in the
    /// eigen-coordinates of L0' and its amplitude over the sample times.
    pub slips:BTreeMap<Charge,Vec<C>>,pub amplitudes:BTreeMap<Charge,f64>,
}
/// The sample times of a run: a trajectory's grid geometrically from its
/// first nonzero time, or the horizons.
pub fn sample_times(times:&[f64],horizons:&[f64])->Vec<f64>{
    if times.is_empty(){return horizons.to_vec();}
    let mut out=Vec::new();let mut i=1usize;
    while i<times.len(){out.push(times[i]);i=((i as f64*GEOMETRIC)as usize).max(i+1);}
    out
}
impl Apriori{
    /// The transfers of rho0 (row-major n x n) through every explicit step's
    /// kernel and their amplitudes at `samples` (us).
    pub fn new(prepared:&LatticePrepared,kernels:&SwKernels,eigen:&BlockEigenbasis,initial:&[C],samples:&[f64])->Result<Self,String>{
        let n=prepared.generator.n;let d=n*n;let zero=&prepared.generator.zero;
        let c0=eigen.coefficients(&[initial])?;
        // X~_s c.
        let apply=|s:usize,c:&[C]|->Vec<C>{let mut o=vec![ZERO;d];kernels.subtract_kernel(s,c,&mut o);for v in &mut o{v.re=-v.re;v.im=-v.im;}o};
        let mut slips:BTreeMap<Charge,Vec<C>>=BTreeMap::new();
        let first:Vec<(Charge,Vec<C>)>=kernels.steps.iter().enumerate().map(|(s,st)|(st.charge.clone(),apply(s,&c0))).collect();
        for(q,v)in &first{let slip=slips.entry(q.clone()).or_insert_with(||vec![ZERO;d]);for(a,b)in slip.iter_mut().zip(v){add(a,*b);}}
        for(q1,v1)in &first{for(s2,st2)in kernels.steps.iter().enumerate(){
            let q2=charge_plus(q1,&st2.charge)?;if &q2==zero{continue;}
            let v=apply(s2,v1);let slip=slips.entry(q2).or_insert_with(||vec![ZERO;d]);for(a,b)in slip.iter_mut().zip(&v){add(a,*b);}}}
        let rows=eigen.population_rows();
        let amplitudes=slips.iter().map(|(q,slip)|{let mut amplitude=0_f64;
            for &t in samples{let decayed:Vec<C>=(0..d).map(|j|mul(slip[j],cexp(C{re:eigen.lambda[j].re.min(0.),im:eigen.lambda[j].im},t))).collect();
                for i in 0..n{let mut v=ZERO;for j in 0..d{mul_add(&mut v,rows[i+n*j],decayed[j]);}amplitude=amplitude.max(2.*magnitude(v));}}
            (q.clone(),amplitude)}).collect();
        Ok(Self{slips,amplitudes})
    }
    /// The amplitude of a charge (zero for one no transfer reaches).
    pub fn amplitude(&self,q:&Charge)->f64{self.amplitudes.get(q).copied().unwrap_or(0.)}
    /// The charges carried above `gate` that `modes` misses, one of each
    /// pair (the larger), largest amplitude first.
    pub fn missing(&self,modes:&[Charge],gate:f64)->Vec<Charge>{
        let inside:BTreeSet<&Charge>=modes.iter().collect();
        let mut out:Vec<(Charge,f64)>=self.amplitudes.iter().filter(|(q,&a)|a>gate&&!inside.contains(q)&&**q>opposite(q)).map(|(q,&a)|(q.clone(),a)).collect();
        out.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));out.into_iter().map(|x|x.0).collect()
    }
    /// The slip of lattice mode q as a field on the modes (its block E slip
    /// on q and the mirror's on -q).
    pub fn slip_field(&self,eigen:&BlockEigenbasis,modes:&[Charge],mirror:&[usize],q:usize)->Vec<C>{
        let d=eigen.dimension();let n=(d as f64).sqrt().round()as usize;let mut f=vec![ZERO;modes.len()*d];
        let Some(slip)=self.slips.get(&modes[q])else{return f};
        let blocks=eigen.eigenvector_blocks();
        for j in 0..d{let a=slip[j];if !nonzero(a){continue;}for e in 0..d{mul_add(&mut f[q*d+e],a,blocks[j*d+e]);}}
        let m=mirror[q];if m!=q{for i in 0..n{for k in 0..n{f[m*d+i*n+k]=conj(f[q*d+k*n+i]);}}}
        f
    }
    /// The a-priori field on the modes: x0 on the centre and every mode's
    /// slip (each canonical one with its mirror).
    pub fn field(&self,eigen:&BlockEigenbasis,modes:&[Charge],canonical:&[bool],mirror:&[usize],centre:usize,initial:&[C])->Vec<C>{
        let d=eigen.dimension();let mut f=vec![ZERO;modes.len()*d];
        f[centre*d..(centre+1)*d].copy_from_slice(initial);
        for q in(0..modes.len()).filter(|&q|q!=centre&&canonical[q]){for(a,b)in f.iter_mut().zip(self.slip_field(eigen,modes,mirror,q)){add(a,b);}}
        f
    }
    /// The amplitude of a source field f on the lattice modes (`modes` with
    /// their frequencies `nu`, the canonical ones read, a pair's weight 2):
    /// the largest over the sample times and states of sum_q w_q |sum_j
    /// rows_ij (e^{mu_j t} - 1) / mu_j (c_q)_j|, mu_j = lambda_j - i nu_q (Re
    /// clipped at zero), c_q = E^-1 f_q: the populations the constant source
    /// drives through each mode's own block.
    pub fn response(eigen:&BlockEigenbasis,nu:&[f64],canonical:&[bool],mirror:&[usize],field:&[C],samples:&[f64])->Result<f64,String>{
        let d=eigen.dimension();let n=(d as f64).sqrt().round()as usize;let rows=eigen.population_rows();
        let read:Vec<usize>=(0..nu.len()).filter(|&q|canonical[q]&&field[q*d..(q+1)*d].iter().any(|z|nonzero(*z))).collect();
        if read.is_empty(){return Ok(0.);}
        let blocks:Vec<&[C]>=read.iter().map(|&q|&field[q*d..(q+1)*d]).collect();let c=eigen.coefficients(&blocks)?;
        let mut amplitude=0_f64;
        for &t in samples{let mut populations=vec![0_f64;n];
            for(k,&q)in read.iter().enumerate(){let w=if mirror[q]==q{1.}else{2.};
                let r:Vec<C>=(0..d).map(|j|{let mu=C{re:eigen.lambda[j].re.min(0.),im:eigen.lambda[j].im-nu[q]};
                    let factor=if magnitude(mu)*t<1e-8{C{re:t,im:0.}}else{let e=cexp(mu,t);let a=C{re:e.re-1.,im:e.im};let m=mu.re*mu.re+mu.im*mu.im;mul(a,C{re:mu.re/m,im:-mu.im/m})};
                    mul(factor,c[k*d+j])}).collect();
                for(i,p)in populations.iter_mut().enumerate(){let mut v=ZERO;for j in 0..d{mul_add(&mut v,rows[i+n*j],r[j]);}*p+=w*magnitude(v);}}
            amplitude=amplitude.max(populations.into_iter().fold(0.,f64::max));}
        Ok(amplitude)
    }
}
