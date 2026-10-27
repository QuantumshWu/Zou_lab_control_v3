//! Harmonic classification: every nonzero Fourier component (charge step) of
//! the generator is an explicit lattice coupling or is folded, decided by its
//! size and frequency only (never by exact zeros), together with the far +
//! far pairs that land on near charges.
//!
//! Near / far: the significant step frequencies are sorted and cut at their
//! largest ratio gap of at least FOLD_GAP_RATIO whose far side is beyond
//! FOLD_DISTANCE ||L0|| (or, when every significant step is that fast, all of
//! them are far). Far steps are folded at second order; a far + far pair with
//! a nonzero near sum t joins the fold target K_t when its group is at least
//! CROSS_FOLD_FLOOR of the diagonal (t = 0) fold, otherwise it is omitted and
//! its bound charged (`ledger`). Near steps below the structural coupling
//! floor do not grow the near ball; they still act exactly on every lattice
//! edge, and their edges leaving the lattice are charged.
use super::*;
use crate::preparation::STRUCTURAL_COUPLING_FLOOR;

/// Smallest ratio between the slowest far and the fastest near significant
/// frequency that makes the far side a folded shell.
pub const FOLD_GAP_RATIO:f64=100.;
/// A far step must be faster than this multiple of ||L0||, so that
/// ||(L0 - i nu)^-1|| <= 1/(|nu| - ||L0||) holds with room to spare.
pub const FOLD_DISTANCE:f64=2.;
/// Relative size (to the diagonal fold) below which a far + far group
/// landing on a nonzero near charge is omitted and charged.
pub const CROSS_FOLD_FLOOR:f64=1e-6;

#[derive(Clone,Debug,PartialEq)]
pub enum Treatment{
    /// A lattice coupling, kept exactly. Structural steps generate the near
    /// ball; the others act on the lattice edges they find.
    Explicit{structural:bool},
    /// Eliminated by the static second-order fold.
    Fold,
}
#[derive(Clone,Debug)]
pub struct HarmonicStep{pub charge:Charge,pub nu:f64,pub size:f64,pub treatment:Treatment}
/// One far + far path q -> q + s -> q + s + back of a fold target, with its
/// size bound ||C_back|| ||C_s|| ||D_s^-1||.
#[derive(Clone,Debug)]
pub struct FoldPair{pub source:usize,pub back:usize,pub bound:f64}
#[derive(Clone,Debug)]
pub struct FoldTarget{pub charge:Charge,pub nu:f64,pub pairs:Vec<FoldPair>,pub size:f64}
#[derive(Clone,Debug)]
pub struct OmittedTarget{pub charge:Charge,pub nu:f64,pub pairs:Vec<FoldPair>,pub size:f64}
pub struct HarmonicPlan{
    pub steps:Vec<HarmonicStep>,
    /// Fastest near and slowest far significant frequency (rad/us).
    pub gap:Option<(f64,f64)>,
    /// Frequency separating near from far charges (rad/us), +inf without a fold.
    pub cut:f64,
    /// Fold targets; the diagonal t = 0 comes first.
    pub targets:Vec<FoldTarget>,
    pub omitted:Vec<OmittedTarget>,
    /// Upper bound of ||L0|| on the Frobenius space (rad/us).
    pub l0_bound:f64,
    pub structural_floor:f64,
}

impl HarmonicPlan{
    pub fn new(generator:&Generator)->Result<Self,String>{
        let n=generator.n;let zero=generator.zero_component();
        // ||L0|| with the Hamiltonian's diagonal centred: NL X + X NR is
        // unchanged by NL - i m, NR + i m.
        let diagonal:Vec<f64>=(0..n).map(|i|zero.op.left.entries.iter().filter(|e|e.0==i&&e.1==i).map(|e|e.2.im).sum()).collect();
        let middle=0.5*(diagonal.iter().copied().fold(f64::NEG_INFINITY,f64::max)+diagonal.iter().copied().fold(f64::INFINITY,f64::min));
        let shift=|s:&Sparse,sign:f64|->Sparse{let mut a=s.dense(n);for i in 0..n{a[i*n+i].im+=sign*middle;}Sparse::from_dense(&a,n)};
        let l0_bound=shift(&zero.op.left,-1.).norm_bound(n)+shift(&zero.op.right,1.).norm_bound(n)
            +zero.op.sandwich.iter().map(|(p,q)|p.norm_bound(n)*q.norm_bound(n)).sum::<f64>();
        if !l0_bound.is_finite(){return Err("Nonfinite L0 norm bound".into());}
        let others:Vec<(&Charge,&Component)>=generator.components.iter().filter(|(q,_)|**q!=generator.zero).collect();
        let largest=others.iter().map(|(_,c)|c.size).fold(0_f64,f64::max);
        let structural_floor=STRUCTURAL_COUPLING_FLOOR*largest;
        let mut frequencies:Vec<f64>=others.iter().filter(|(_,c)|c.size>0.&&c.size>=structural_floor).map(|(_,c)|c.nu.abs()).collect();
        frequencies.sort_by(f64::total_cmp);frequencies.dedup_by(|a,b|(*a-*b).abs()<=1e-12*b.abs());
        let mut best:Option<(f64,usize)>=None;
        for i in 0..frequencies.len(){
            if frequencies[i]<FOLD_DISTANCE*l0_bound{continue;}
            let ratio=if i==0{f64::INFINITY}else{frequencies[i]/frequencies[i-1]};
            if i>0&&ratio<FOLD_GAP_RATIO{continue;}
            if best.is_none_or(|(r,_)|ratio>r){best=Some((ratio,i));}
        }
        let gap=best.map(|(_,i)|(if i==0{0.}else{frequencies[i-1]},frequencies[i]));
        let cut=match gap{None=>f64::INFINITY,Some((0.,far))=>(FOLD_DISTANCE*l0_bound).min(far),Some((near,far))=>(near*far).sqrt()};
        let steps:Vec<HarmonicStep>=others.iter().map(|(q,c)|{
            let far=c.nu.abs()>=cut&&c.nu.abs()>=FOLD_DISTANCE*l0_bound;
            HarmonicStep{charge:(*q).clone(),nu:c.nu,size:c.size,
                treatment:if far{Treatment::Fold}else{Treatment::Explicit{structural:c.size>0.&&c.size>=structural_floor}}}
        }).collect();
        let far:Vec<usize>=(0..steps.len()).filter(|&i|steps[i].treatment==Treatment::Fold).collect();
        let mut groups=BTreeMap::<Charge,(f64,Vec<FoldPair>)>::new();
        for &s in &far{for &b in &far{
            let nu=steps[s].nu+steps[b].nu;if nu.abs()>=cut{continue;}
            let t=charge_plus(&steps[s].charge,&steps[b].charge)?;
            let bound=steps[b].size*steps[s].size/(steps[s].nu.abs()-l0_bound);
            groups.entry(t).or_insert_with(||(nu,Vec::new())).1.push(FoldPair{source:s,back:b,bound});
        }}
        let diagonal=groups.get(&generator.zero).map_or(0.,|(_,p)|p.iter().map(|x|x.bound).sum::<f64>());
        let(mut targets,mut omitted)=(Vec::new(),Vec::new());
        for(charge,(nu,pairs))in groups{
            let size=pairs.iter().map(|x|x.bound).sum::<f64>();
            if charge==generator.zero||(size>0.&&size>=CROSS_FOLD_FLOOR*diagonal){targets.push(FoldTarget{charge,nu,pairs,size});}
            else{omitted.push(OmittedTarget{charge,nu,pairs,size});}
        }
        targets.sort_by_key(|t|t.charge!=generator.zero);
        Ok(Self{steps,gap,cut,targets,omitted,l0_bound,structural_floor})
    }
    pub fn far(&self)->Vec<usize>{(0..self.steps.len()).filter(|&i|self.steps[i].treatment==Treatment::Fold).collect()}
    pub fn explicit(&self)->Vec<usize>{(0..self.steps.len()).filter(|&i|matches!(self.steps[i].treatment,Treatment::Explicit{..})).collect()}
    /// The step with the opposite charge (every Hermitian component has one).
    pub fn opposite(&self,step:usize)->Option<usize>{let o=opposite(&self.steps[step].charge);self.steps.iter().position(|s|s.charge==o)}
    /// The steps that generate the near ball: structural explicit steps and
    /// the nonzero fold targets.
    pub fn structural_steps(&self)->Vec<Charge>{
        let mut out:BTreeSet<Charge>=self.steps.iter().filter(|s|s.treatment==Treatment::Explicit{structural:true}).map(|s|s.charge.clone()).collect();
        out.extend(self.targets.iter().filter(|t|t.charge.iter().any(|&k|k!=0)).map(|t|t.charge.clone()));
        out.into_iter().collect()
    }
    pub fn count(&self,treatment:&Treatment)->usize{self.steps.iter().filter(|s|&s.treatment==treatment).count()}
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_classification_takes_every_step_once_by_size_and_frequency(){
        let f=fixture();let p=prepared(&f);let plan=&p.plan;
        assert_eq!(plan.steps.len(),p.generator.components.len()-1,"every nonzero charge exactly once");
        for s in &plan.steps{assert_eq!(s.treatment==Treatment::Fold,s.nu.abs()>1000.,"{s:?}");}
        let(near,far)=plan.gap.unwrap();assert_eq!(near,3.);assert_eq!(far,2000.);
        // +-2000 and +-2003 fold on t = 0; 2003 - 2000 lands on the near +-3.
        assert_eq!(plan.targets.iter().map(|t|t.charge.clone()).collect::<Vec<_>>(),vec![vec![0,0],vec![-1,0],vec![1,0]]);
        assert_eq!(plan.targets[0].pairs.len(),4);
        assert_eq!(p.modes(1).unwrap().len(),3);assert_eq!(p.modes(2).unwrap().len(),5);
    }
}
