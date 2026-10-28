//! The explicit charge set P of a request and its ring Q = ring(P): the
//! lattice the pipeline works on. P's blocks are solved by GMRES, Q is
//! eliminated exactly at every pole (`ring`), everything beyond P u Q is
//! omitted (its outflow is the truncation part of the embedded residual the
//! error estimate propagates).
//!
//! The rule (`explicit_set`, one function):
//! 1. start: {0} and every explicit step s whose first-order mixing
//!    |L_s| / |nu_s| (largest element) is above GATE x tolerance, with -s
//!    (the translation gate: a copy along s carries population weight of
//!    that order);
//! 2. ring closure: while the ring has an internal edge q <- q - t with
//!    rho = ||C_t|| ||D_q^-1|| above RING_COUPLING (the contraction of the
//!    ring's block Jacobi elimination), the ring charge carrying the largest
//!    summed rho joins P with its mirror, at most PROMOTIONS times;
//! 3. cover: every charge whose seed amplitude is at least SEED_AMPLITUDE x
//!    tolerance and that P u Q misses joins P (largest first, with its
//!    mirror);
//! 4. the ring closure again.
//!
//! Seed amplitudes: the slaved (z = 0) amplitudes layer by layer from the
//! probe states on the centre, y_r = -D_r^-1 sum_p C_{r-p} y_p over the
//! layers before, to SEED_DEPTH steps; the probes are x0 and the dressed
//! centre's steady state (unit norm); a charge's amplitude is the largest
//! |y_r| over the probes.
use super::*;
use super::estimate_api::{LatticeSets,ring};
use super::operator::{LatticeOperator,Mask,Parts};
use super::classification::HarmonicStep;

/// Translation gate and explicit start (times the tolerance).
pub const GATE:f64=0.1;
/// Largest rho = ||C_t|| ||D_q^-1|| of an edge inside the ring.
pub const RING_COUPLING:f64=0.1;
const PROMOTIONS:usize=4;
const SEED_DEPTH:usize=3;
/// Seed amplitude (times the tolerance) a charge must reach to be covered.
pub const SEED_AMPLITUDE:f64=0.2;
const NORM_ITERATIONS:usize=30;

#[derive(Clone,Debug)]
pub struct ExplicitSet{
    pub sets:LatticeSets,
    /// The gated steps (one of each pair) and the charges promoted by the
    /// ring closure and the cover.
    pub gated:Vec<Charge>,pub promoted:Vec<Charge>,pub covered:Vec<Charge>,
    /// The largest seed amplitude outside P u Q.
    pub leak:f64,
    /// The ring's internal edges left (target, source, rho).
    pub internal:Vec<(Charge,Charge,f64)>,
    pub milliseconds:f64,
}

/// The largest element of a superoperator's matrix (row (i, j), column
/// (k, l)).
fn largest_element(op:&Superop,n:usize,scratch:&mut Vec<C>)->f64{
    let d=n*n;scratch.clear();scratch.resize(d*d,ZERO);op.dense_into(n,ONE,scratch);
    scratch.iter().map(|z|magnitude(*z)).fold(0.,f64::max)
}
/// First-order mixing |L_s| / |nu_s| of an explicit step (largest element).
pub fn mixing(prepared:&LatticePrepared,step:&HarmonicStep)->f64{
    let n=prepared.generator.n;let mut scratch=Vec::new();
    largest_element(&prepared.generator.components[&step.charge].op,n,&mut scratch)/step.nu.abs()
}
/// The explicit steps passing the translation gate, one of each pair (the
/// larger charge).
pub fn gated_steps(prepared:&LatticePrepared,tolerance:f64)->Vec<Charge>{
    prepared.plan.explicit().into_iter().map(|s|&prepared.plan.steps[s]).filter(|st|st.charge>opposite(&st.charge)&&mixing(prepared,st)>GATE*tolerance)
        .map(|st|st.charge.clone()).collect()
}

fn adjoint_sparse(a:&Sparse)->Sparse{
    let mut entries:Vec<(usize,usize,C)>=a.entries.iter().map(|&(i,j,v)|(j,i,conj(v))).collect();entries.sort_by_key(|e|(e.0,e.1));Sparse{entries}
}
/// ||L||_2 on the Frobenius space by power iteration on L^H L.
fn superop_norm(op:&Superop,n:usize)->f64{
    let adjoint=Superop{left:adjoint_sparse(&op.left),right:adjoint_sparse(&op.right),sandwich:op.sandwich.iter().map(|(p,q)|(adjoint_sparse(p),adjoint_sparse(q))).collect()};
    let d=n*n;let mut x:Vec<C>=(0..d).map(|k|C{re:((k*13+7)%37)as f64/37.-0.5,im:((k*17+3)%41)as f64/41.-0.5}).collect();
    let(mut y,mut scratch)=(vec![ZERO;d],Vec::new());let mut value=0.;
    for _ in 0..NORM_ITERATIONS{
        let size=x.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();if !(size>0.){return 0.;}for z in x.iter_mut(){*z=scaled(*z,1./size);}
        y.fill(ZERO);op.apply_into(n,ONE,&x,&mut y,&mut scratch);x.fill(ZERO);adjoint.apply_into(n,ONE,&y,&mut x,&mut scratch);
        value=x.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
    }
    value.sqrt()
}

/// rho of the ring-internal edges, cached per step and per frequency.
struct RingCondition<'p>{prepared:&'p LatticePrepared,steps:Vec<Charge>,coupling:BTreeMap<Charge,f64>,resolvent:BTreeMap<i64,f64>}
impl RingCondition<'_>{
    /// ||C_t||: the explicit component's norm plus the fold target's bound.
    fn coupling(&mut self,t:&Charge)->f64{
        if let Some(&v)=self.coupling.get(t){return v;}
        let n=self.prepared.generator.n;
        let explicit=self.prepared.plan.explicit().into_iter().any(|s|&self.prepared.plan.steps[s].charge==t);
        let mut v=if explicit{superop_norm(&self.prepared.generator.components[t].op,n)}else{0.};
        v+=self.prepared.plan.targets.iter().filter(|x|&x.charge==t).map(|x|x.size).sum::<f64>();
        self.coupling.insert(t.clone(),v);v
    }
    /// The ring's internal edges (target q, source q - t, rho); the block
    /// resolvent norms of the targets in one batch.
    fn edges(&mut self,explicit:&BTreeSet<Charge>)->Result<Vec<(Charge,Charge,f64)>,String>{
        let set:Vec<Charge>=explicit.iter().cloned().collect();let q_set=ring(self.prepared,&set)?;
        let inside:BTreeSet<&Charge>=q_set.iter().collect();let mut found=Vec::new();
        for q in &q_set{for t in self.steps.clone(){let p=charge_minus(q,&t)?;if inside.contains(&p){found.push((q.clone(),p,t));}}}
        let key=|nu:f64|(nu*1e6).round()as i64;
        let mut missing:Vec<f64>=Vec::new();
        for(q,_,_)in &found{let nu=self.prepared.nu(q)?;if !self.resolvent.contains_key(&key(nu))&&!missing.iter().any(|&m|key(m)==key(nu)){missing.push(nu);}}
        if !missing.is_empty(){for(nu,v)in missing.iter().zip(self.prepared.screening_eigenbasis()?.resolvent_norms(&missing)?){self.resolvent.insert(key(*nu),v);}}
        found.into_iter().map(|(q,p,t)|{let rho=self.coupling(&t)*self.resolvent[&key(self.prepared.nu(&q)?)];Ok((q,p,rho))}).collect()
    }
    /// While an internal edge has rho above RING_COUPLING: the ring charge
    /// with the largest summed rho (ties: the smaller |q|_1) joins with its
    /// mirror. Returns the promoted charges.
    fn close(&mut self,explicit:&mut BTreeSet<Charge>)->Result<Vec<Charge>,String>{
        let mut promoted=Vec::new();
        for _ in 0..PROMOTIONS{
            let strong:Vec<(Charge,Charge,f64)>=self.edges(explicit)?.into_iter().filter(|e|e.2>RING_COUPLING).collect();
            if strong.is_empty(){break;}
            let mut score=BTreeMap::<Charge,f64>::new();
            for(q,p,rho)in &strong{*score.entry(q.clone()).or_default()+=rho;*score.entry(p.clone()).or_default()+=rho;}
            let size=|q:&Charge|q.iter().map(|&k|k.unsigned_abs()as u64).sum::<u64>();
            let best=score.iter().max_by(|a,b|a.1.total_cmp(b.1).then(size(b.0).cmp(&size(a.0)))).map(|x|x.0.clone()).ok_or("Empty ring score")?;
            explicit.insert(opposite(&best));explicit.insert(best.clone());promoted.push(best);
        }
        Ok(promoted)
    }
}

/// Seed amplitudes of every charge within SEED_DEPTH structural steps.
fn seed_amplitudes(prepared:&LatticePrepared,probes:&[Vec<C>])->Result<BTreeMap<Charge,f64>,String>{
    let n=prepared.generator.n;let d=n*n;
    let modes=prepared.modes(SEED_DEPTH)?;let index:BTreeMap<&Charge,usize>=modes.iter().enumerate().map(|(i,q)|(q,i)).collect();
    let mut layer=vec![usize::MAX;modes.len()];
    for k in(0..=SEED_DEPTH).rev(){for q in prepared.modes(k)?{layer[index[&q]]=k;}}
    let eigen=prepared.screening_eigenbasis()?;
    let mut op=LatticeOperator::new(prepared,&modes)?;
    // Layer k+1 from the layers up to k (disjoint: no diagonal block acts).
    let masks:Vec<(Vec<bool>,Vec<bool>)>=(0..SEED_DEPTH).map(|k|(layer.iter().map(|&l|l==k+1).collect(),layer.iter().map(|&l|l<=k).collect())).collect();
    let mut amplitudes=BTreeMap::new();
    for probe in probes{
        let size=probe.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();if !(size>0.){continue;}
        let mut field=vec![ZERO;modes.len()*d];let centre=index[&prepared.generator.zero];
        for(t,v)in field[centre*d..(centre+1)*d].iter_mut().zip(probe){*t=scaled(*v,1./size);}
        for(k,(targets,sources))in masks.iter().enumerate(){
            let mut source=vec![ZERO;field.len()];op.apply_masked(&field,&mut source,Parts::Full,Some(Mask{targets,sources}))?;
            let rows:Vec<usize>=(0..modes.len()).filter(|&q|layer[q]==k+1&&op.canonical[q]).collect();
            let mut blocks:Vec<C>=rows.iter().flat_map(|&q|source[q*d..(q+1)*d].iter().copied()).collect();
            let nu:Vec<f64>=rows.iter().map(|&q|op.nu[q]).collect();
            eigen.solve_blocks(&nu,ZERO,&mut blocks)?;
            for(i,&q)in rows.iter().enumerate(){for(t,v)in field[q*d..(q+1)*d].iter_mut().zip(&blocks[i*d..(i+1)*d]){*t=C{re:-v.re,im:-v.im};}}
            mirror_field(&mut field,n,&op.canonical,&op.mirror);
        }
        for(q,c)in modes.iter().enumerate(){if q==centre{continue;}
            let a=field[q*d..(q+1)*d].iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
            let e=amplitudes.entry(c.clone()).or_insert(0_f64);*e=e.max(a);}
    }
    Ok(amplitudes)
}

/// The explicit set of a request from rho0 (row-major n x n) at the
/// population tolerance.
pub fn explicit_set(prepared:&LatticePrepared,initial:&[C],tolerance:f64)->Result<ExplicitSet,String>{
    let clock=Instant::now();let zero=prepared.generator.zero.clone();
    let mut probes=vec![initial.to_vec()];
    if let Ok(block)=prepared.screening_eigenbasis()?.kernel_block(){probes.push(block);}
    let amplitudes=seed_amplitudes(prepared,&probes)?;
    let mut condition=RingCondition{prepared,steps:prepared.plan.structural_steps(),coupling:BTreeMap::new(),resolvent:BTreeMap::new()};
    // 1. The gated start.
    let gated=gated_steps(prepared,tolerance);
    let mut explicit:BTreeSet<Charge>=BTreeSet::from([zero]);
    for s in &gated{explicit.insert(s.clone());explicit.insert(opposite(s));}
    // 2. Ring closure.
    let mut promoted=condition.close(&mut explicit)?;
    // 3. Cover.
    let need:Vec<(&Charge,f64)>=amplitudes.iter().filter(|(_,&a)|a>=SEED_AMPLITUDE*tolerance).map(|(q,&a)|(q,a)).collect();
    let mut covered=Vec::new();
    loop{
        let set:Vec<Charge>=explicit.iter().cloned().collect();let lattice:BTreeSet<Charge>=set.iter().cloned().chain(ring(prepared,&set)?).collect();
        let Some((c,_))=need.iter().filter(|(q,_)|!lattice.contains(*q)).max_by(|a,b|a.1.total_cmp(&b.1).then(a.0.cmp(b.0)))else{break};
        let c=(*c).clone();explicit.insert(opposite(&c));explicit.insert(c.clone());covered.push(c);
    }
    // 4. Ring closure again.
    promoted.extend(condition.close(&mut explicit)?);
    // Diagnostics: ATOM_DIAG_RINGS=k enlarges P by its ring k times (a
    // reference lattice).
    for _ in 0..super::diagnostics::perturbation("RINGS").map_or(0,|k|k as usize){
        let set:Vec<Charge>=explicit.iter().cloned().collect();explicit.extend(ring(prepared,&set)?);}
    let mut p:Vec<Charge>=explicit.into_iter().collect();
    p.sort_by_key(|q|(q.iter().map(|&k|k.unsigned_abs()as u64).sum::<u64>(),q.clone()));
    let q=ring(prepared,&p)?;
    if super::diagnostics::perturbation("RINGS").is_some(){eprintln!("DBG lattice P {} Q {}",p.len(),q.len());}
    let lattice:BTreeSet<&Charge>=p.iter().chain(&q).collect();
    let leak=amplitudes.iter().filter(|(c,_)|!lattice.contains(c)).map(|(_,&a)|a).fold(0.,f64::max);
    let internal=condition.edges(&p.iter().cloned().collect())?;
    Ok(ExplicitSet{sets:LatticeSets{explicit:p,ring:q},gated,promoted,covered,leak,internal,milliseconds:clock.elapsed().as_secs_f64()*1e3})
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_explicit_set_covers_the_seed_and_closes_its_ring(){
        // The fixture's near pair mixes strongly (0.4 / 3): it passes the gate,
        // P = {0, +-s}; the ring holds the next shell, and every charge with a
        // seed amplitude above the cover threshold is inside P u Q.
        let f=fixture();let p=prepared(&f);let n=4;let mut initial=vec![ZERO;n*n];initial[0]=ONE;
        let set=explicit_set(&p,&initial,5e-3).unwrap();
        let s=p.plan.steps[p.plan.explicit()[0]].charge.clone();
        assert!(set.sets.explicit.contains(&s)&&set.sets.explicit.contains(&opposite(&s))&&set.sets.explicit[0]==p.generator.zero);
        assert!(set.leak<SEED_AMPLITUDE*5e-3,"leak {:e}",set.leak);
        assert!(set.internal.iter().all(|e|e.2<=RING_COUPLING)||set.promoted.len()==2*PROMOTIONS);
        // The superoperator norm against the dense matrix's power iteration.
        let op=&p.generator.components[&s].op;let d=16;let mut m=vec![ZERO;d*d];op.dense_into(n,ONE,&mut m);
        let mut v:Vec<C>=(0..d).map(|k|C{re:((k*5+2)%13)as f64/13.-0.4,im:((k*3+1)%11)as f64/11.-0.6}).collect();let mut sigma=0.;
        for _ in 0..300{let w:Vec<C>=(0..d).map(|r|(0..d).fold(ZERO,|s,c|{let mut s=s;mul_add(&mut s,m[r*d+c],v[c]);s})).collect();
            let u:Vec<C>=(0..d).map(|c|(0..d).fold(ZERO,|s,r|{let mut s=s;mul_add(&mut s,conj(m[r*d+c]),w[r]);s})).collect();
            let size=u.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();sigma=size.sqrt();v=u.iter().map(|z|scaled(*z,1./size)).collect();}
        let norm=superop_norm(op,n);assert!((norm/sigma-1.).abs()<0.02,"{norm:e} against {sigma:e}");
    }
}
