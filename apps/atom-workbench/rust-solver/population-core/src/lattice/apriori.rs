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
//! joins it (the completion).
//!
//! The fold's pieces (`Duhamel`, `fold`): a static piece D the operator might
//! leave out acts on x0's own evolution as the first-order Duhamel response
//! dx(t) = int_0^t e^{(t - s) L0'} D e^{s L0'} x0 ds, closed form in L0''s
//! eigenbasis: (e^{mu_i t} - e^{lambda_j t}) / (mu_i - lambda_j) (t e^{mu t}
//! when they coincide), mu_i = lambda_i - i nu_q on the target mode q; read
//! as populations at the run's sample times. L0' = L0 + K_0^(0) is the folded
//! centre generator: the slow optical pumping and the trap's weak scattering
//! that amplify small pieces at long T are inside it.
use super::*;
use super::eigenbasis::BlockEigenbasis;
use super::kernels::SwKernels;

/// x0's eigen-components below this share of the largest (|c_j| times the
/// eigenvector's largest entry) are not sources of the Duhamel response: the
/// responses are compared with a budget, not summed to roundoff. Measured on
/// three lasers at 20 us: the dropped pieces' responses 1.236e-5 with this
/// floor (17 sources) against 1.249e-5 at 1e-6 (182 sources, the fold's
/// build 8x slower).
const SOURCE_FLOOR:f64=1e-3;

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
}

/// The first-order Duhamel response of x0's evolution under L0' to a static
/// piece (module documentation): x0's significant eigen-components (their
/// indices, coefficients and eigenvector blocks) and the sample times.
pub struct Duhamel<'e>{eigen:&'e BlockEigenbasis,columns:Vec<usize>,c:Vec<C>,blocks:Vec<C>,samples:Vec<f64>}
impl<'e> Duhamel<'e>{
    pub fn new(eigen:&'e BlockEigenbasis,initial:&[C],samples:&[f64])->Result<Self,String>{
        let d=eigen.dimension();let c0=eigen.coefficients(&[initial])?;let all=eigen.eigenvector_blocks();
        let size:Vec<f64>=(0..d).map(|j|magnitude(c0[j])*all[j*d..(j+1)*d].iter().map(|z|magnitude(*z)).fold(0_f64,f64::max)).collect();
        let top=size.iter().copied().fold(0_f64,f64::max);
        let columns:Vec<usize>=(0..d).filter(|&j|size[j]>SOURCE_FLOOR*top).collect();
        let blocks=columns.iter().flat_map(|&j|all[j*d..(j+1)*d].iter().copied()).collect();
        Ok(Self{eigen,c:columns.iter().map(|&j|c0[j]).collect(),columns,blocks,samples:samples.to_vec()})
    }
    /// The source blocks: the eigenvectors E_j of x0's significant components.
    pub fn sources(&self)->std::slice::ChunksExact<'_,C>{self.blocks.chunks_exact(self.eigen.dimension())}
    pub fn count(&self)->usize{self.columns.len()}
    /// The response of a piece on a target mode of frequency nu (a mirror
    /// pair's `weight` 2): `images` holds D E_j on the target block for every
    /// source (`sources`, in order); the largest population over the sample
    /// times and states, weight x |rows (sum_j E^-1 D E_j c_j phi_ij(t))|.
    pub fn response(&self,nu:f64,weight:f64,images:&[C])->Result<f64,String>{
        let d=self.eigen.dimension();let n=(d as f64).sqrt().round()as usize;let count=self.columns.len();
        if count==0||images.iter().all(|z|!nonzero(*z)){return Ok(0.);}
        let blocks:Vec<&[C]>=images.chunks_exact(d).collect();let m=self.eigen.coefficients(&blocks)?;
        let(lambda,rows)=(&self.eigen.lambda,self.eigen.population_rows());
        let mu:Vec<C>=lambda.iter().map(|l|C{re:l.re,im:l.im-nu}).collect();
        let mut amplitude=0_f64;
        for &t in &self.samples{
            let ea:Vec<C>=mu.iter().map(|&z|cexp(z,t)).collect();
            let eb:Vec<C>=self.columns.iter().map(|&j|cexp(lambda[j],t)).collect();
            let delta:Vec<C>=(0..d).map(|i|{let mut v=ZERO;
                for jj in 0..count{let a=m[i+d*jj];if !nonzero(a){continue;}
                    let k=lambda[self.columns[jj]];let gap=C{re:mu[i].re-k.re,im:mu[i].im-k.im};
                    let phi=if magnitude(gap)*t<1e-6{let mid=C{re:0.5*(mu[i].re+k.re),im:0.5*(mu[i].im+k.im)};scaled(cexp(mid,t),t)}
                        else{let num=C{re:ea[i].re-eb[jj].re,im:ea[i].im-eb[jj].im};let g2=gap.re*gap.re+gap.im*gap.im;mul(num,C{re:gap.re/g2,im:-gap.im/g2})};
                    mul_add(&mut v,mul(a,self.c[jj]),phi);}
                v}).collect();
            for i in 0..n{let mut v=ZERO;for j in 0..d{mul_add(&mut v,rows[i+n*j],delta[j]);}amplitude=amplitude.max(weight*magnitude(v));}
        }
        Ok(amplitude)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn duhamel_response_is_the_first_order_population_change(){
        // A small static piece D added to the folded centre generator L0': the
        // populations' change at t against dense RK4 propagation of L0' + D and
        // L0', to first order in D.
        let f=fixture();let p=prepared(&f);let(n,d)=(4,16);let eigen=p.eigenbasis().unwrap();
        let l0=p.centre_generator();let scale=1e-6;
        let piece:Vec<C>=(0..d*d).map(|k|C{re:scale*(((k*37+11)%97)as f64/97.-0.5),im:scale*(((k*53+7)%89)as f64/89.-0.5)}).collect();
        let mut x0=vec![ZERO;d];x0[0]=ONE;
        for t in[0.5,3.,40.]{
            let duhamel=Duhamel::new(eigen,&x0,&[t]).unwrap();
            let images:Vec<C>=duhamel.sources().flat_map(|e|(0..d).map(|r|(0..d).fold(ZERO,|v,c|{let mut v=v;mul_add(&mut v,piece[r*d+c],e[c]);v})).collect::<Vec<C>>()).collect();
            let predicted=duhamel.response(0.,1.,&images).unwrap();
            let propagate=|m:&[C]|->Vec<f64>{let steps=(t*400.)as usize;let h=t/steps as f64;let mut x=x0.clone();
                let rate=|x:&[C]|->Vec<C>{(0..d).map(|r|(0..d).fold(ZERO,|v,c|{let mut v=v;mul_add(&mut v,m[r*d+c],x[c]);v})).collect()};
                let axpy=|x:&[C],k:&[C],s:f64|->Vec<C>{x.iter().zip(k).map(|(a,b)|C{re:a.re+s*b.re,im:a.im+s*b.im}).collect()};
                for _ in 0..steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
                    for e in 0..d{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}}
                (0..n).map(|i|x[i*n+i].re).collect()};
            let perturbed:Vec<C>=l0.iter().zip(&piece).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
            let(a,b)=(propagate(&perturbed),propagate(&l0));
            let measured=a.iter().zip(&b).map(|(u,v)|(u-v).abs()).fold(0.,f64::max);
            assert!(measured<=predicted*(1.+1e-3)+1e-15&&predicted<=measured*1.5+1e-12,"t {t}: predicted {predicted:e}, measured {measured:e}");
        }
    }
}
