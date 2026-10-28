//! The Schrieffer-Wolff kernels of the error propagation, in the shared
//! eigenbasis of L0' (`eigenbasis`, complex eigen-coordinates).
//!
//! Every explicit near step s couples mode q - s to q through C_s. The
//! first-order generator S of the lattice's block-diagonalization has the
//! blocks X_s (source q - s, target q), translation invariant and solving
//! (L0' - i nu_q) X_s - X_s (L0' - i nu_{q-s}) = -C_s, elementwise in the
//! eigenbasis:
//!   X~_s[i, j] = -C~_s[i, j] / (lambda_i - lambda_j - i nu_s),  C~_s = E^-1 C_s E.
//! The second-order block of a mode q whose neighbour q + s is on the
//! lattice gains (path q -> q + s -> q)
//!   K_s = 1/2 (C_-s X_s - X_-s C_s).
//! This is the lattice's second-order averaging: the beat-note light shifts
//! and the Raman rates (the recycling term is inside L0', so they include
//! the repopulation by spontaneous emission). The kernels are superoperator
//! (d x d) objects: the separable Floquet frames (`floquet`) leave the
//! recycling out of their kernels and would miss those rates.
//!
//! Support: an entry stays when it is not small at first order,
//! |X~_ij| >= entry, or when its second-order rate over the request's
//! horizon is not negligible, |X~_ij| |C~_ij| T >= horizon_rate (a weak
//! beat step still pumps over milliseconds). Everything left out stays a
//! coupling between modes that the propagation does not transform; the
//! small parameter is the largest kept |X~_ij|; entries above the rule's
//! cap are near resonant and stay couplings as well.
//!
//! Hermitian pairing: C_-s = J C_s J for the flip J (x_q -> x_-q^dagger),
//! whose eigen-coordinates are conjugated and permuted (`partner`), so the
//! kernels of -s are C~_-s[i, j] = conj C~_s[i*, j*] and likewise X~, K~;
//! only one step of each pair is computed.
use super::super::*;
use super::super::eigenbasis::BlockEigenbasis;
use crate::parallel::Execution;

/// Columns of C~_s and K~_s per task: a step's coordinate change is a
/// d x d x d product, spread over the workers even when the steps are
/// fewer.
const KERNEL_PANEL:usize=64;

/// The support rule of the kernels.
#[derive(Clone,Copy,Debug)]
pub struct KernelRule{
    /// First-order floor on |X~_ij|.
    pub entry:f64,
    /// Floor on |X~_ij| |C~_ij| times `horizon` (us), the longest output time.
    pub horizon_rate:f64,pub horizon:f64,
    /// Entries above this are near resonant (the first-order kernel is no
    /// small parameter there): not eliminated, left as couplings.
    pub cap:f64,
}

/// One step's kernel: X~_s as (row, column, value) in eigen-coordinates.
pub struct StepKernel{
    pub charge:Charge,pub nu:f64,
    /// The index of -s.
    pub opposite:usize,
    pub entries:Vec<(usize,usize,C)>,
}

pub struct SwKernels{
    pub steps:Vec<StepKernel>,
    /// K~_s (column-major d x d) of the computed member of each pair, by
    /// step index (None for the mirrored member).
    second_order:Vec<Option<Vec<C>>>,
    partner:Vec<usize>,
}

impl SwKernels{
    pub fn new(prepared:&LatticePrepared,basis:&BlockEigenbasis,rule:KernelRule)->Result<Self,String>{
        let(n,d)=(prepared.generator.n,basis.dimension());
        let plan=&prepared.plan;let explicit=plan.explicit();
        let partner:Vec<usize>=(0..d).map(|j|basis.partner(j)).collect();
        let position=|q:&Charge|explicit.iter().position(|&i|&plan.steps[i].charge==q);
        // Steps in plan order, each with its opposite.
        let mut steps=Vec::with_capacity(explicit.len());
        for &i in &explicit{let st=&plan.steps[i];
            let opposite=position(&opposite(&st.charge)).ok_or("Estimator kernels: an explicit step without its opposite")?;
            steps.push(StepKernel{charge:st.charge.clone(),nu:st.nu,opposite,entries:Vec::new()});}
        let vectors=basis.eigenvector_blocks();let lambda=&basis.lambda;
        let execution=Execution::new()?;
        let computed:Vec<usize>=(0..steps.len()).filter(|&s|steps[s].charge>steps[steps[s].opposite].charge).collect();
        // C~_s (column j = coordinates of C_s e_j) and the kept entries of
        // X~_s, by column panels.
        let tasks:Vec<(usize,usize)>=(0..computed.len()).flat_map(|slot|(0..d).step_by(KERNEL_PANEL).map(move|start|(slot,start))).collect();
        let panels=execution.map(tasks.clone(),|(slot,start)|->Result<(Vec<C>,Vec<(usize,usize,C)>),String>{
            let step=&steps[computed[slot]];let op=&prepared.generator.components[&step.charge].op;let end=(start+KERNEL_PANEL).min(d);
            let mut image=vec![ZERO;d*(end-start)];let mut scratch=Vec::new();
            for j in start..end{op.apply_into(n,ONE,&vectors[j*d..(j+1)*d],&mut image[(j-start)*d..(j-start+1)*d],&mut scratch);}
            let blocks:Vec<&[C]>=image.chunks_exact(d).collect();
            let c=basis.coefficients(&blocks)?;drop(image);
            let mut entries=Vec::new();
            for j in start..end{for i in 0..d{
                let den=C{re:lambda[i].re-lambda[j].re,im:lambda[i].im-lambda[j].im-step.nu};
                let num=c[i+d*(j-start)];let s2=den.re*den.re+den.im*den.im;if s2==0.{continue;}
                let x=C{re:-(num.re*den.re+num.im*den.im)/s2,im:-(num.im*den.re-num.re*den.im)/s2};
                let size=magnitude(x);
                if size<=rule.cap&&(size>=rule.entry||(rule.horizon_rate>0.&&size*magnitude(num)*rule.horizon>=rule.horizon_rate)){entries.push((i,j,x));}
            }}
            Ok((c,entries))
        });
        let mut whole:Vec<(Vec<C>,Vec<(usize,usize,C)>)>=computed.iter().map(|_|(Vec::with_capacity(d*d),Vec::new())).collect();
        for(&(slot,_),panel)in tasks.iter().zip(panels){let(part,entries)=panel?;whole[slot].0.extend(part);whole[slot].1.extend(entries);}
        let mut second_order:Vec<Option<Vec<C>>>=(0..steps.len()).map(|_|None).collect();
        for(&s,(c,entries))in computed.iter().zip(whole){
            let k=second_order_block(&execution,&c,&entries,&partner,d);let o=steps[s].opposite;
            steps[o].entries=entries.iter().map(|&(i,j,x)|(partner[i],partner[j],conj(x))).collect();
            steps[s].entries=entries;second_order[s]=Some(k);
        }
        Ok(Self{steps,second_order,partner})
    }
    /// out += K~_s (column-major d x d).
    pub fn add_second_order(&self,s:usize,out:&mut[C]){
        let d=self.partner.len();
        if let Some(k)=&self.second_order[s]{for(o,&v)in out.iter_mut().zip(k){add(o,v);}return;}
        let k=self.second_order[self.steps[s].opposite].as_ref().expect("one member of every pair is computed");
        let p=&self.partner;
        for j in 0..d{for i in 0..d{add(&mut out[i+d*j],conj(k[p[i]+d*p[j]]));}}
    }
    /// out[a] -= sum over the entries of X~_s: out[i] -= x * input[j] (one
    /// block's coordinates).
    pub fn subtract_kernel(&self,s:usize,input:&[C],out:&mut[C]){
        for &(i,j,x)in &self.steps[s].entries{let v=mul(x,input[j]);out[i].re-=v.re;out[i].im-=v.im;}
    }
    /// The adjoint: out[j] -= conj(x) * input[i].
    pub fn subtract_kernel_adjoint(&self,s:usize,input:&[C],out:&mut[C]){
        for &(i,j,x)in &self.steps[s].entries{let v=mul(conj(x),input[i]);out[j].re-=v.re;out[j].im-=v.im;}
    }
}

/// K~_s = 1/2 (C~_-s X~_s - X~_-s C~_s) from C~_s (column-major) and the
/// entries of X~_s in column order, with C~_-s[l, i] = conj C~_s[l*, i*]
/// and the entries (i*, j*, conj x) of X~_-s: column j of K~_s gains
/// 1/2 x C~_-s[:, i] for each entry (i, j, x) of its column, and X~_-s
/// times column j of C~_s. One task per panel of columns.
fn second_order_block(execution:&Execution,c:&[C],entries:&[(usize,usize,C)],partner:&[usize],d:usize)->Vec<C>{
    debug_assert!(entries.windows(2).all(|w|w[0].1<=w[1].1));
    let mut starts=vec![0;d+1];for &(_,j,_)in entries{starts[j+1]+=1;}for j in 0..d{starts[j+1]+=starts[j];}
    let panels:Vec<usize>=(0..d).step_by(KERNEL_PANEL).collect();
    let parts=execution.map(panels,|start|{let end=(start+KERNEL_PANEL).min(d);let mut k=vec![ZERO;d*(end-start)];
        for(column,out)in(start..end).zip(k.chunks_exact_mut(d)){
            for &(i,_,x)in &entries[starts[column]..starts[column+1]]{
                let(half,source)=(scaled(x,0.5),&c[partner[i]*d..(partner[i]+1)*d]);
                for(t,&p)in out.iter_mut().zip(partner){mul_add(t,half,conj(source[p]));}}
            let source=&c[column*d..(column+1)*d];
            for &(i,j,x)in entries{mul_add(&mut out[partner[i]],scaled(conj(x),-0.5),source[partner[j]]);}
        }
        k});
    parts.concat()
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::super::tests::{fixture,prepared};
    /// Dense d x d helpers (row-major).
    fn product(a:&[C],b:&[C],d:usize)->Vec<C>{let mut c=vec![ZERO;d*d];for i in 0..d{for k in 0..d{let x=a[i*d+k];if !nonzero(x){continue;}for j in 0..d{mul_add(&mut c[i*d+j],x,b[k*d+j]);}}}c}
    #[test]
    fn estimator_kernels_solve_the_lattice_sylvester_equation(){
        // Every entry kept: X_s = E X~_s E^-1 solves L0' X - X L0' - i nu_s X
        // = -C_s in density coordinates, the mirrored step's kernel is the
        // flip of the computed one, and K~_s is 1/2 (C_-s X_s - X_-s C_s).
        let f=fixture();let p=prepared(&f);let basis=p.eigenbasis().unwrap();let(n,d)=(4,16);
        let k=SwKernels::new(&p,basis,KernelRule{entry:0.,horizon_rate:0.,horizon:1.,cap:f64::INFINITY}).unwrap();
        assert!(k.steps.len()>=2&&k.steps.iter().all(|s|s.entries.len()==d*d));
        // E (row-major: column j the eigenvector e_j) and E^-1.
        let blocks=basis.eigenvector_blocks();let mut e=vec![ZERO;d*d];for j in 0..d{for i in 0..d{e[i*d+j]=blocks[j*d+i];}}
        let unit:Vec<Vec<C>>=(0..d).map(|i|{let mut b=vec![ZERO;d];b[i]=ONE;b}).collect();
        let ci=basis.coefficients(&unit.iter().map(|b|&b[..]).collect::<Vec<_>>()).unwrap();
        let mut einv=vec![ZERO;d*d];for j in 0..d{for i in 0..d{einv[i*d+j]=ci[i+d*j];}}
        let dense_kernel=|s:usize|{let mut x=vec![ZERO;d*d];for &(i,j,v)in &k.steps[s].entries{x[i*d+j]=v;}product(&product(&e,&x,d),&einv,d)};
        let l=p.centre_generator();
        for(s,step)in k.steps.iter().enumerate(){
            let x=dense_kernel(s);let mut c=vec![ZERO;d*d];p.generator.components[&step.charge].op.dense_into(n,ONE,&mut c);
            let(lx,xl)=(product(&l,&x,d),product(&x,&l,d));let scale=c.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for t in 0..d*d{let r=C{re:lx[t].re-xl[t].re+step.nu*x[t].im+c[t].re,im:lx[t].im-xl[t].im-step.nu*x[t].re+c[t].im};
                assert!(magnitude(r)<1e-9*scale,"step {:?} entry {t}: {r:?}",step.charge);}
            // K_s against the dense second-order term.
            let o=step.opposite;let xo=dense_kernel(o);let mut co=vec![ZERO;d*d];p.generator.components[&k.steps[o].charge].op.dense_into(n,ONE,&mut co);
            let(a,b)=(product(&co,&x,d),product(&xo,&c,d));
            let mut kt=vec![ZERO;d*d];k.add_second_order(s,&mut kt);
            let mut km=vec![ZERO;d*d];for i in 0..d{for j in 0..d{km[i*d+j]=kt[i+d*j];}}
            let dense=product(&product(&e,&km,d),&einv,d);
            let size=a.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for t in 0..d*d{let want=C{re:0.5*(a[t].re-b[t].re),im:0.5*(a[t].im-b[t].im)};
                assert!(magnitude(C{re:dense[t].re-want.re,im:dense[t].im-want.im})<1e-9*size,"K step {:?} entry {t}",step.charge);}
        }
    }
}
