//! The slow spectrum of the folded centre generator L0' = L0 + K_0^(0): the
//! conserved quantities (uniqueness of the steady state) and the slowest
//! relaxation.
//!
//! The slow modes live on the ground-ground block G (populations and ground
//! coherences); every other element X (optical coherences, excited block) is
//! damped at least at the optical rates. Eliminating X exactly at rate
//! lambda, S(lambda) = L_GG - lambda - L_GX (L_XX - lambda)^-1 L_XG, and
//! expanding to first order in lambda gives the generalized problem
//!   S a = lambda N a,  S = L_GG - L_GX L_XX^-1 L_XG,  N = I + L_GX L_XX^-2 L_XG,
//! exact at lambda = 0 (so the conserved count is exact) and accurate to
//! lambda / (optical rate) for the slow rates. S is small (the optical
//! pumping and light-shift scale), so rates far below eps ||L0|| (the 1.8 s
//! dark states, ||L0|| ~ 5e7 rad/us) are resolved, which a direct
//! eigensolve of L0' cannot do.
//!
//! A mode is conserved when |lambda| <= floor, with floor 1000 times the
//! numerical size of the exactly conserved trace mode (at least
//! eps ||S||_F); every other mode with -Re lambda > floor relaxes, the
//! slowest setting the relaxation time. Undamped modes at nonzero frequency
//! average out of the mean and are neither.
use super::*;

/// Eigenvalues returned for diagnostics.
const REPORTED:usize=6;

#[derive(Clone,Debug)]
pub struct SlowSpectrum{
    /// Conserved quantities, the trace included.
    pub conserved:usize,
    /// Slowest relaxation rate (1/us) of the decaying modes.
    pub slowest_rate:Option<f64>,
    pub floor:f64,
    /// The slowest eigenvalues (rad/us), by magnitude.
    pub slowest:Vec<C>,
    pub milliseconds:f64,
}

pub fn slow_spectrum(prepared:&LatticePrepared)->Result<SlowSpectrum,String>{
    let clock=Instant::now();let lapack=&prepared.lapack;let generator=&prepared.generator;
    let n=generator.n;let d=n*n;
    let mut l=vec![ZERO;d*d];generator.zero_component().op.dense_into(n,ONE,&mut l);
    if let Some(t)=prepared.fold.targets.first().filter(|t|t.charge==generator.zero){t.dense_into(n,0,ONE,&mut l);}
    let ground=|p:usize|generator.ground[p/n]&&generator.ground[p%n];
    let g:Vec<usize>=(0..d).filter(|&p|ground(p)).collect();let x:Vec<usize>=(0..d).filter(|&p|!ground(p)).collect();
    let(r,m)=(g.len(),x.len());
    if r==0{return Err("The slow spectrum needs ground states".into());}
    // Column-major L_XX and L_XG.
    let mut lxx=vec![ZERO;m*m];for(j,&c)in x.iter().enumerate(){for(i,&row)in x.iter().enumerate(){lxx[i+m*j]=l[row*d+c];}}
    let mut lxg=vec![ZERO;m*r];for(k,&c)in g.iter().enumerate(){for(i,&row)in x.iter().enumerate(){lxg[i+m*k]=l[row*d+c];}}
    let(lu,pivots)=lapack.complex_factor(&lxx,m)?;drop(lxx);
    let y=lapack.complex_solve_matrix_factored(&lu,&pivots,&lxg,m)?;
    let z=lapack.complex_solve_matrix_factored(&lu,&pivots,&y,m)?;
    // S and N (column-major r x r).
    let(mut s,mut nm)=(vec![ZERO;r*r],vec![ZERO;r*r]);
    for(i,&row)in g.iter().enumerate(){
        let gx:Vec<(usize,C)>=x.iter().enumerate().filter_map(|(k,&c)|{let v=l[row*d+c];nonzero(v).then_some((k,v))}).collect();
        for j in 0..r{
            let(mut a,mut b)=(l[row*d+g[j]],if i==j{ONE}else{ZERO});
            for &(k,v)in &gx{let(yy,zz)=(y[k+m*j],z[k+m*j]);a.re-=v.re*yy.re-v.im*yy.im;a.im-=v.re*yy.im+v.im*yy.re;mul_add(&mut b,v,zz);}
            s[i+r*j]=a;nm[i+r*j]=b;
        }
    }
    let(lun,pn)=lapack.complex_factor(&nm,r)?;let reduced=lapack.complex_solve_matrix_factored(&lun,&pn,&s,r)?;
    let(values,_)=lapack.complex_eigen_right(&reduced,r)?;
    let size=s.iter().fold(0_f64,|a,z|a.hypot(magnitude(*z)));
    let trace=values.iter().map(|z|magnitude(*z)).fold(f64::INFINITY,f64::min);
    let floor=1e3*trace.max(f64::EPSILON*size);
    let conserved=values.iter().filter(|z|magnitude(**z)<=floor).count();
    let slowest_rate=values.iter().filter(|z|magnitude(**z)>floor&&-z.re>floor).map(|z|-z.re).fold(None,|a:Option<f64>,v|Some(a.map_or(v,|a|a.min(v))));
    let mut sorted=values.clone();sorted.sort_by(|a,b|magnitude(*a).total_cmp(&magnitude(*b)));sorted.truncate(REPORTED);
    Ok(SlowSpectrum{conserved,slowest_rate,floor,slowest:sorted,milliseconds:clock.elapsed().as_secs_f64()*1e3})
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_slow_spectrum_matches_the_dense_centre_generator(){
        // The fixture's ground populations relax by optical pumping through
        // the excited states: one conserved quantity, and the slowest rate
        // against the dense eigenvalues of L0' (first order in
        // lambda / optical rate).
        let f=fixture();let p=prepared(&f);let n=4;let d=16;
        let s=slow_spectrum(&p).unwrap();assert_eq!(s.conserved,1);
        let mut l=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut l);p.fold.targets[0].dense_into(n,0,ONE,&mut l);
        let(values,_)=p.lapack.complex_eigen_right(&super::super::fold::transpose(&l,d),d).unwrap();
        let mut rates:Vec<f64>=values.iter().filter(|z|magnitude(**z)>1e-9).map(|z|-z.re).collect();rates.sort_by(f64::total_cmp);
        let slowest=s.slowest_rate.unwrap();
        assert!((slowest/rates[0]-1.).abs()<0.05,"slowest {slowest:e}, dense {:e}",rates[0]);
    }
}
