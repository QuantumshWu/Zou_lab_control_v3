//! The closed-form propagation of the fast Ritz modes' sources on the near
//! modes and their population readout.
//!
//! Mode a of class c (eigenvalues mu_j, zeta_j = mu_j - i nu_a, a positive
//! real part clipped to the axis: second-order roundoff that would grow
//! secularly) with the source coefficients beta_jm e^{theta_m t}:
//!   e_j(t) = sum_m G_jm (e^{theta_m t} - e^{zeta_j t}),  G_jm = beta_jm / (theta_m - zeta_j),
//! with the limit beta_jm t e^{zeta_j t} at resonance. The centre's trace
//! mode (zeta = 0) carries no source: the exact error equation conserves the
//! centre trace. The readout of mode a goes to its own populations and, by
//! I + S, to every neighbour a + s on the lattice; read with the target's
//! phase e^{i nu t} (laboratory) or at the centre alone (a horizon's time
//! average), the homogeneous part of every target carries e^{i nu_s t}
//! e^{mu_j t}, so it is summed per class and step before its time GEMM.
//! Only the canonical modes are propagated: a mirrored mode's population
//! errors are the conjugates of its partner's (the Ritz data come in
//! conjugate pairs), so a pair's canonical member counts twice in the real
//! part.
use super::super::*;
use super::lattice::ErrorLattice;
use crate::parallel::Execution;

/// How populations are read: the laboratory density at t (the grid), or
/// the centre alone (the time average at a horizon).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Readout{Laboratory,Centre}

/// The Ritz data of one checkpoint: the Ritz vectors over the basis
/// (column-major k x r), the values and coefficients the outputs use.
pub struct RitzSources<'a>{pub vectors:&'a[C],pub values:&'a[C],pub coefficients:&'a[C]}

/// One mode's contribution: its particular parts per target (target, n x r
/// readout of G), its homogeneous weights per (class slot) and resonant terms.
struct ModePart{particular:Vec<(usize,Vec<C>)>,homogeneous:Vec<(usize,Vec<C>)>,resonant:Vec<(usize,Vec<C>,C)>}

const RESONANT:f64=1e-9;
/// The centre's eigenvalue of least modulus below this (1/us) is its trace
/// mode, zeta = 0 exactly (the block annihilates the trace): a deliberate
/// margin over the eigensystem's error on that value.
const TRACE_MODE:f64=1e-6;


impl ErrorLattice<'_>{
    /// Population errors (one row of n per time) of the SW propagation of
    /// the checkpoint's sources on the near modes and of the dressed-block
    /// propagation on the outer modes (Ritz modes with a zero coefficient
    /// carry none). The Ritz data must be closed under conjugation.
    pub fn populations(&self,ritz:&RitzSources<'_>,times:&[f64],readout:Readout)->Result<Vec<Vec<f64>>,String>{
        let(n,d,k)=(self.n,self.d,self.columns);let r=ritz.values.len();
        if times.is_empty()||r==0{return Ok(vec![vec![0.;n];times.len()]);}
        if ritz.vectors.len()!=k*r||ritz.coefficients.len()!=r{return Err("Estimator: Ritz data dimensions differ from the basis".into());}
        // B a and B theta a (column-major k x r).
        let mut ba=vec![ZERO;k*r];let mut bta=vec![ZERO;k*r];
        for m in 0..r{let(a,ta)=(ritz.coefficients[m],mul(ritz.values[m],ritz.coefficients[m]));
            for i in 0..k{ba[i+k*m]=mul(ritz.vectors[i+k*m],a);bta[i+k*m]=mul(ritz.vectors[i+k*m],ta);}}
        // beta = (z_a b - z_w b theta) a.
        self.propagate(ritz.values,times,readout,|slot,near,weight|{
            let(image,base):(&[C],&[C])=if near{(&self.image_coordinates[slot],&self.basis_coordinates[slot])}else{(&self.outer_image[slot],&self.outer_basis[slot])};
            let(gemm,mut beta)=(self.prepared.lapack.complex_gemm(),vec![ZERO;d*r]);
            gemm.multiply(d,r,k,C{re:weight,im:0.},image,d,&ba,k,ZERO,&mut beta,d)?;
            gemm.multiply(d,r,k,C{re:-weight,im:0.},base,d,&bta,k,ONE,&mut beta,d)?;
            Ok(beta)})
    }
    /// The SW propagation of sources sum_m beta_jm e^{theta_m t} per
    /// canonical near mode (class coordinates) and outer mode
    /// (eigen-coordinates of L0'), beta (column-major d x r) from `source`
    /// for (slot, near, weight), weight 2 for a pair's canonical member.
    fn propagate(&self,theta:&[C],times:&[f64],readout:Readout,source:impl Fn(usize,bool,f64)->Result<Vec<C>,String>+Sync)->Result<Vec<Vec<f64>>,String>{
        let(n,d)=(self.n,self.d);let r=theta.len();let t_count=times.len();
        let slots=self.kernels()?.steps.len()+1;
        let lapack=&self.prepared.lapack;
        // The near modes with the frame's block (key 0), then the outer modes
        // with the dressed blocks of L0' (key `bare`).
        let(class,bare)=(self.class()?,1);
        let outer=self.outer_modes.len();
        let items:Vec<(usize,bool)>=self.near_canonical.iter().map(|&s|(s,true)).chain((0..outer).map(|s|(s,false))).collect();
        let parts=Execution::new()?.map(items,|(slot,near)|->Result<ModePart,String>{
            let a=if near{self.near[slot]}else{self.outer_modes[slot]};
            let gemm=lapack.complex_gemm();
            let(c,values,own):(usize,&[C],&[C])=if near{(0,&class.values,&class.readout)}
                else{(bare,&self.basis()?.lambda,self.basis()?.population_rows())};
            // A mirror pair's canonical member counts twice.
            let mut beta=source(slot,near,if self.mirror[a]==a{1.}else{2.})?;
            let zeta:Vec<C>=values.iter().map(|z|C{re:z.re.min(0.),im:z.im-self.nu[a]}).collect();
            if a==self.centre{
                let j0=(0..d).min_by(|&x,&y|magnitude(zeta[x]).total_cmp(&magnitude(zeta[y]))).unwrap();
                if magnitude(zeta[j0])<TRACE_MODE{for m in 0..r{beta[j0+d*m]=ZERO;}}
            }
            let mut gamma=vec![ZERO;d];let mut resonant=Vec::new();
            for m in 0..r{for j in 0..d{
                let b=beta[j+d*m];if !nonzero(b){continue;}
                let den=C{re:theta[m].re-zeta[j].re,im:theta[m].im-zeta[j].im};
                if magnitude(den)<RESONANT*(1.+magnitude(zeta[j])){resonant.push((j,m,b));beta[j+d*m]=ZERO;continue;}
                let s=den.re*den.re+den.im*den.im;let g=mul(b,C{re:den.re/s,im:-den.im/s});beta[j+d*m]=g;add(&mut gamma[j],g);
            }}
            // Targets: the mode itself (slot 0) and its neighbours a + s (slot s + 1).
            let mut targets:Vec<(usize,usize,&[C])>=vec![(a,0,own)];
            if near{for &s in &class.steps{
                let Some(&t)=self.index.get(&charge_plus(&self.modes[a],&self.kernels()?.steps[s].charge)?)else{continue};
                targets.push((t,s+1,class.step_readout[s].as_deref().unwrap()));
            }}
            if readout==Readout::Centre{targets.retain(|x|x.0==self.centre);}
            let mut part=ModePart{particular:Vec::new(),homogeneous:Vec::new(),resonant:Vec::new()};
            for(t,slot,rows)in targets{
                let mut y=vec![ZERO;n*r];gemm.multiply(n,r,d,ONE,rows,n,&beta,d,ZERO,&mut y,n)?;
                part.particular.push((t,y));
                part.homogeneous.push((c*slots+slot,gamma.clone()));
                for &(j,m,b)in &resonant{part.resonant.push((t,rows[n*j..n*(j+1)].iter().map(|&v|mul(v,b)).collect(),theta[m]));}
            }
            Ok(part)
        });
        // Sum the particular readouts per target and the homogeneous weights per class slot.
        let mut particular=vec![vec![ZERO;n*r];self.modes.len()];let mut used=vec![false;self.modes.len()];
        let mut homogeneous:BTreeMap<usize,Vec<C>>=BTreeMap::new();let mut resonant=Vec::new();
        for part in parts{let part=part?;
            for(t,y)in part.particular{used[t]=true;for(a,b)in particular[t].iter_mut().zip(y){add(a,b);}}
            for(key,g)in part.homogeneous{let e=homogeneous.entry(key).or_insert_with(||vec![ZERO;d]);for(a,b)in e.iter_mut().zip(g){add(a,b);}}
            resonant.extend(part.resonant);
        }
        // Homogeneous part: -e^{i nu_s t} R_{c,s} diag(Gamma) e^{mu t}, per
        // class its values and per slot H = R_{c,s} diag(Gamma) and nu_s.
        let mut by_class:BTreeMap<usize,Vec<(usize,Vec<C>)>>=BTreeMap::new();
        for(key,g)in homogeneous{by_class.entry(key/slots).or_default().push((key%slots,g));}
        let mut classes:Vec<(&[C],Vec<(Vec<C>,f64)>)>=Vec::new();
        for(c,list)in by_class{
            let values:&[C]=if c==bare{&self.basis()?.lambda}else{&class.values};
            let mut terms=Vec::with_capacity(list.len());
            for(slot,g)in list{
                let rows:&[C]=if c==bare{self.basis()?.population_rows()}else if slot==0{&class.readout}else{class.step_readout[slot-1].as_deref().unwrap()};
                let mut h=rows.to_vec();for j in 0..d{for i in 0..n{h[i+n*j]=mul(h[i+n*j],g[j]);}}
                terms.push((h,if slot==0{0.}else{self.kernels()?.steps[slot-1].nu}));
            }
            classes.push((values,terms));
        }
        let targets:Vec<(usize,&Vec<C>)>=particular.iter().enumerate().filter(|(t,_)|used[*t]).collect();
        // The time GEMMs by chunks of the grid over the workers: every
        // output entry is accumulated by one task in the serial order.
        let execution=Execution::new()?;
        let per_time=targets.len()*n*r+classes.iter().map(|(_,t)|t.len()*n*d).sum::<usize>();
        let width=t_count.div_ceil(execution.tasks(t_count,per_time*t_count,crate::parallel::MINIMUM_WORK_PER_TASK));
        let chunks:Vec<(usize,usize)>=(0..t_count).step_by(width).map(|a|(a,(a+width).min(t_count))).collect();
        let parts=execution.map(chunks,|(a,b)|->Result<Vec<C>,String>{
            let(gemm,ts,tc)=(lapack.complex_gemm(),&times[a..b],b-a);
            let mut total=vec![ZERO;n*tc];
            // Particular part: phase_t(t) (Y_t e^{theta t}).
            let ritz_exp:Vec<C>=ts.iter().flat_map(|&t|theta.iter().map(move|&z|cexp(z,t))).collect();
            for &(t,y)in &targets{
                let mut f=vec![ZERO;n*tc];gemm.multiply(n,tc,r,ONE,y,n,&ritz_exp,r,ZERO,&mut f,n)?;
                for(ti,&time)in ts.iter().enumerate(){let ph=self.phase(t,time,readout);for i in 0..n{mul_add(&mut total[i+n*ti],ph,f[i+n*ti]);}}
            }
            for(values,terms)in &classes{
                let block:Vec<C>=ts.iter().flat_map(|&t|values.iter().map(move|&z|cexp(C{re:z.re.min(0.),im:z.im},t))).collect();
                for(h,nu)in terms{
                    let mut f=vec![ZERO;n*tc];gemm.multiply(n,tc,d,ONE,h,n,&block,d,ZERO,&mut f,n)?;
                    for(ti,&time)in ts.iter().enumerate(){let ph=cexp(C{re:0.,im:*nu},time);for i in 0..n{let v=mul(ph,f[i+n*ti]);total[i+n*ti].re-=v.re;total[i+n*ti].im-=v.im;}}
                }
            }
            for(t,v,z)in &resonant{
                for(ti,&time)in ts.iter().enumerate(){let ph=mul(self.phase(*t,time,readout),scaled(cexp(*z,time),time));for i in 0..n{mul_add(&mut total[i+n*ti],ph,v[i]);}}
            }
            Ok(total)});
        let mut total=Vec::with_capacity(n*t_count);for part in parts{total.extend(part?);}
        Ok((0..t_count).map(|ti|(0..n).map(|i|total[i+n*ti].re).collect()).collect())
    }
    fn phase(&self,mode:usize,time:f64,readout:Readout)->C{
        match readout{Readout::Laboratory=>cexp(C{re:0.,im:self.nu[mode]},time),Readout::Centre=>if mode==self.centre{ONE}else{ZERO}}
    }
}
