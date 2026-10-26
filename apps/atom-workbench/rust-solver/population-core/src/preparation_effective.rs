//! The effective generator: the Lindbladian of the sheared coordinates at
//! chosen (slow) charges.
//!
//! # Definition
//!
//! The shear layers change variables as `x = T_last(...T_0(y))`
//! (`PrimitivePreparation::layers`, inner first). Let `S_l` be the union of
//! layer `l`'s charges, an anti-Hermitian Fourier family
//! (`S_{l,-q} = -S_{l,q}^dagger`, phases `exp(i q.theta(t))`), and
//! `U = exp(S_last) ... exp(S_0)`. Then `x = U y U^dagger` maps the
//! Lindbladian `(H, C_c)` onto the Lindbladian of `y`,
//!
//!   `H' = U^dagger (H - i d/dt) U`,   `C'_c = U^dagger C_c U`,
//!
//! whose charge-`p` component has the form of the frame Liouvillian of
//! `slow_dual.rs`:
//!
//!   `L'_p X = -i (H'_p X - X H'_p) - 1/2 (G'_p X + X G'_p) + R'_p(X)`,
//!   `G'_p = sum_c sum_{a-b=p} C'^dagger_{c,b} C'_{c,a}`,
//!   `R'_p(X) = sum_c sum_{a-b=p} C'_{c,a} X C'^dagger_{c,b}`.
//!
//! Unlike the bare frame, `G'_p` and `R'_p` act on every block: the shear
//! mixes ground and excited states, so `R'` also takes ground and optical
//! input (Raman and Rayleigh scattering of a far laser, at second order)
//! and gives optical and excited output.
//!
//! # Expansion
//!
//! Layer by layer, outermost first (each layer acts on the previous one's
//! result, so a near layer sees a far layer's light shift):
//!
//!   `H <- H + D + 1/2 [D, S]`,   `D = [H, S] + nu S` (`nu_q S_q` per charge,
//!   the `-i dS/dt` term; the sheared couplings cancel in `D`),
//!   `C <- C + [C, S] + 1/2 [[C, S], S]`.
//!
//! `H'` keeps every product the layers form: a strong far layer has
//! `|V| |S| ~ 1e2/us`, and its second order times a near layer's `S` is a
//! slow coupling of order 1/us. The jump operators keep total order <= 2 in
//! `S`, and `G'`, `R'` the products of total order <= 2. The same products
//! enter both, so `tr R'_p(X) = tr(G'_p X)` term by term: `L'` conserves the
//! trace exactly, and `H'_{-p} = H'_p^dagger`, `G'_{-p} = G'_p^dagger`.
//!
//! # Remainder
//!
//! Norms are Frobenius norms summed over charges (upper bounds of the
//! spectral norm of the omitted part of any `L'_p`), in rad/us:
//!
//! - Hamiltonian, per layer, orders >= 3:
//!   `sum_{k>=3} (2s)^{k-1} d / k! = d (e^{2s} - 1 - 2s - 2s^2) / (2s)`, with
//!   `s = sum_q |S_q|` and `d` a bound of `|D|` (computed exactly for the bare
//!   Hamiltonian, plus `2 s` times the changes of the outer layers);
//! - jumps: with `t_c = |C_c| (e^{2s} - 1 - 2s - 2s^2)` (`s` summed over all
//!   layers) the omitted orders >= 3 of `C'_c`, and `o_c = 2 |C1| |C2| + |C2|^2`
//!   the omitted products of total order 3 and 4, `G'` and `R'` change by at
//!   most `sum_c 2 |C'_c| (t_c + o_c) + (t_c + o_c)^2`.
//!
//! `frame_difference` bounds how far the pencil's own coordinates are from
//! `U`: a layer is `I + sum_c (J2(S_c) - I)`, not `exp(ad S_l)`. Its orders
//! >= 3 give `d (e^{2s} - 1 - 2s)`, and the class pairs a layer leaves out
//! (whose sums are fast charges, `shear_layers`) give `sum_{c != c'} d_c s_c'`.
//! The recycling shear `Khat` (outside all layers) is not part of `L'`.
use std::collections::{BTreeMap,BTreeSet};
use super::{Charge,Complex64 as C,FourierMatrices,add,conj,plus};
use super::primitives::PrimitivePreparation;
use crate::lapack::Lapack;

type Modes=BTreeSet<Charge>;

/// One transformed jump operator component `C'_{c,a}` of order `order` in S.
pub struct Jump{pub channel:usize,pub charge:Charge,pub order:usize,pub matrix:Vec<C>}

pub struct EffectiveGenerator{
    pub n:usize,pub g:usize,
    /// The requested charges, sorted.
    pub charges:Vec<Charge>,
    /// `H'_p` per requested charge, n x n row-major.
    pub hamiltonian:Vec<Vec<C>>,
    /// `G'_p` per requested charge, n x n row-major.
    pub gamma:Vec<Vec<C>>,
    /// The transformed jump operators (total order <= 2).
    pub jumps:Vec<Jump>,
    /// Per requested charge, the pairs `(a, b)` of `jumps` (same channel,
    /// `charge_a - charge_b = p`, orders summing to <= 2) of `R'_p`.
    recycling:Vec<Vec<(usize,usize)>>,
    /// Bound of the omitted part of every `L'_p` (rad/us); see module docs.
    pub remainder:f64,
    /// Bound of the difference between `L'` and the generator of the
    /// pencil's own coordinates (rad/us); see module docs.
    pub frame_difference:f64,
}

fn norm(a:&[C])->f64{a.iter().fold(0_f64,|s,z|s.hypot(z.re.hypot(z.im)))}
fn family_norm(f:&FourierMatrices)->f64{f.values().map(|a|norm(a)).sum()}
fn minus(a:&[i32],b:&[i32])->Result<Charge,String>{
    a.iter().zip(b).map(|(&x,&y)|x.checked_sub(y).ok_or_else(||"Effective generator charge overflow".to_string())).collect()
}
fn axpy(target:&mut FourierMatrices,q:&Charge,a:&[C],s:C){
    let t=target.entry(q.clone()).or_insert_with(||vec![C::default();a.len()]);
    for(x,&y)in t.iter_mut().zip(a){add(x,C{re:s.re*y.re-s.im*y.im,im:s.re*y.im+s.im*y.re});}
}
/// Row-major n x n products `a b` and `b a` through one GEMM each.
fn products(lap:&Lapack,a:&[C],b:&[C],n:usize)->Result<(Vec<C>,Vec<C>),String>{
    let gemm=lap.complex_gemm();let(one,zero)=(C{re:1.,im:0.},C::default());
    // A row-major buffer read column-major is the transpose: (a b)^T = b^T a^T.
    let mut ab=vec![zero;n*n];gemm.multiply(n,n,n,one,b,n,a,n,zero,&mut ab,n)?;
    let mut ba=vec![zero;n*n];gemm.multiply(n,n,n,one,a,n,b,n,zero,&mut ba,n)?;
    Ok((ab,ba))
}
/// `[x, s]` at the charges `keep` admits (all when `None`).
fn commutator(lap:&Lapack,x:&FourierMatrices,s:&FourierMatrices,n:usize,keep:Option<&Modes>)->Result<FourierMatrices,String>{
    let mut out=FourierMatrices::new();
    for(q,a)in x{for(r,b)in s{let t=plus(q,r)?;if keep.is_some_and(|k|!k.contains(&t)){continue;}
        let(ab,ba)=products(lap,a,b,n)?;let entry=out.entry(t).or_insert_with(||vec![C::default();n*n]);
        for k in 0..n*n{entry[k].re+=ab[k].re-ba[k].re;entry[k].im+=ab[k].im-ba[k].im;}
    }}
    Ok(out)
}
fn tail(x:f64,from:usize)->f64{
    // sum_{k>=from} x^k / k!, computed without cancellation.
    let(mut term,mut sum)=(1_f64,0_f64);for k in 1..from{term*=x/k as f64;}
    let mut k=from;loop{term*=x/k as f64;sum+=term;if term<=1e-17*sum||k>200{break;}k+=1;}sum
}

impl EffectiveGenerator{
    /// `L'_p` at `charges` (see module docs).
    pub fn new(p:&PrimitivePreparation,lap:&Lapack,charges:&[Charge])->Result<Self,String>{
        let source=&p.hilbert.source;let n=source.n;let g=source.ground.len();
        if source.ground!=(0..g).collect::<Vec<_>>(){return Err("Effective generator needs the ground states first".into());}
        let axes=source.phases.axes.len();
        let targets:Modes=charges.iter().cloned().collect();
        if targets.iter().any(|q|q.len()!=axes){return Err("Effective generator charge dimensions differ".into());}
        let layers:Vec<FourierMatrices>=p.layers.iter().map(|classes|{let mut s=FourierMatrices::new();
            for class in classes{for(q,a)in class{s.insert(q.clone(),a.clone());}}s}).collect();
        let zero=vec![0;axes];
        // Supports of the layers, {0} + S + (S + S), and the charges each
        // layer's output is needed at: N(0) = targets, N(l) = N(l-1) - U_{l-1}.
        let support=|s:&FourierMatrices|->Result<Modes,String>{let mut u=Modes::from([zero.clone()]);
            for q in s.keys(){u.insert(q.clone());for r in s.keys(){u.insert(plus(q,r)?);}}Ok(u)};
        let mut needed=vec![targets.clone()];
        for l in 1..layers.len(){let u=support(&layers[l-1])?;let mut next=Modes::new();
            for t in &needed[l-1]{for v in &u{next.insert(minus(t,v)?);}}needed.push(next);}
        let frequency=|q:&[i32]|->Result<f64,String>{let w=source.phases.frequency(q)?;Ok(w.hi+w.lo)};
        let sizes:Vec<f64>=layers.iter().map(family_norm).collect();
        // Hamiltonian, outermost layer first.
        let mut h=source.hamiltonian.clone();
        let(mut remainder,mut frame_difference,mut outer_change)=(0_f64,0_f64,0_f64);
        for l in (0..layers.len()).rev(){let s=&layers[l];let size=sizes[l];
            let out=&needed[l];let mut first_keep=out.clone();
            for t in out{for r in s.keys(){first_keep.insert(minus(t,r)?);}}
            // D = [H, S] + nu S on the charges the layer's output needs.
            let mut d=commutator(lap,&h,s,n,Some(&first_keep))?;
            for(q,a)in s{if first_keep.contains(q){axpy(&mut d,q,a,C{re:frequency(q)?,im:0.});}}
            // The bound of |D| over every charge: exact for the bare H, and
            // 2 s times what the outer layers changed.
            let mut bare=commutator(lap,&source.hamiltonian,s,n,None)?;
            for(q,a)in s{axpy(&mut bare,q,a,C{re:frequency(q)?,im:0.});}
            let d_bound=family_norm(&bare)+2.*size*outer_change;
            if size>0.{remainder+=d_bound*tail(2.*size,3)/(2.*size);frame_difference+=d_bound*tail(2.*size,2);}
            // Class pairs a layer leaves out: sum_{c != c'} d_c s_c'.
            let classes=&p.layers[l];if classes.len()>1{
                let per:Vec<(f64,f64)>=classes.iter().map(|class|->Result<(f64,f64),String>{
                    let mut dc=commutator(lap,&source.hamiltonian,class,n,None)?;for(q,a)in class{axpy(&mut dc,q,a,C{re:frequency(q)?,im:0.});}
                    Ok((family_norm(&dc)+2.*family_norm(class)*outer_change,family_norm(class)))}).collect::<Result<_,_>>()?;
                for(i,(dc,_))in per.iter().enumerate(){for(j,(_,sc))in per.iter().enumerate(){if i!=j{frame_difference+=dc*sc;}}}
            }
            let second=commutator(lap,&d,s,n,Some(out))?;
            let mut next=FourierMatrices::new();
            for(q,a)in &h{if out.contains(q){next.insert(q.clone(),a.clone());}}
            for(q,a)in &d{if out.contains(q){axpy(&mut next,q,a,C{re:1.,im:0.});}}
            for(q,a)in &second{axpy(&mut next,q,a,C{re:0.5,im:0.});}
            h=next;outer_change+=d_bound*(1.+size)+if size>0.{d_bound*tail(2.*size,3)/(2.*size)}else{0.};
        }
        // Jump operators to total order 2, outermost layer first.
        let total_size:f64=sizes.iter().sum();
        let mut jumps=Vec::new();let mut dissipator_bound=0_f64;
        for(channel,family)in source.collapse.iter().enumerate(){
            let mut orders:[FourierMatrices;3]=[family.clone(),FourierMatrices::new(),FourierMatrices::new()];
            for l in (0..layers.len()).rev(){let s=&layers[l];
                let once0=commutator(lap,&orders[0],s,n,None)?;let once1=commutator(lap,&orders[1],s,n,None)?;
                let twice0=commutator(lap,&once0,s,n,None)?;
                for(q,a)in &once0{axpy(&mut orders[1],q,a,C{re:1.,im:0.});}
                for(q,a)in &once1{axpy(&mut orders[2],q,a,C{re:1.,im:0.});}
                for(q,a)in &twice0{axpy(&mut orders[2],q,a,C{re:0.5,im:0.});}
            }
            let m=family_norm(family);let t=m*tail(2.*total_size,3);
            let(c1,c2)=(family_norm(&orders[1]),family_norm(&orders[2]));let o=2.*c1*c2+c2*c2;
            dissipator_bound+=2.*(m+c1+c2)*(t+o)+(t+o)*(t+o);
            for(order,f)in orders.into_iter().enumerate(){for(q,a)in f{jumps.push(Jump{channel,charge:q,order,matrix:a});}}
        }
        remainder+=dissipator_bound;
        // G'_p and the R'_p pairs.
        let mut index:BTreeMap<(usize,Charge),Vec<usize>>=BTreeMap::new();
        for(k,j)in jumps.iter().enumerate(){index.entry((j.channel,j.charge.clone())).or_default().push(k);}
        let targets:Vec<Charge>=targets.into_iter().collect();
        let(mut gamma,mut recycling)=(Vec::with_capacity(targets.len()),Vec::with_capacity(targets.len()));
        for t in &targets{let mut pairs=Vec::new();
            for(a,ja)in jumps.iter().enumerate(){let b_charge=minus(&ja.charge,t)?;
                if let Some(list)=index.get(&(ja.channel,b_charge)){for &b in list{if ja.order+jumps[b].order<=2{pairs.push((a,b));}}}}
            // G'_p = sum_pairs C'^dagger_b C'_a.
            let mut total=vec![C::default();n*n];
            for &(a,b)in &pairs{let b=&jumps[b].matrix;
                let adjoint:Vec<C>=(0..n*n).map(|k|conj(b[(k%n)*n+k/n])).collect();
                let(product,_)=products(lap,&adjoint,&jumps[a].matrix,n)?;
                for(x,&y)in total.iter_mut().zip(&product){add(x,y);}
            }
            gamma.push(total);recycling.push(pairs);
        }
        let hamiltonian=targets.iter().map(|q|h.get(q).cloned().unwrap_or_else(||vec![C::default();n*n])).collect();
        if !remainder.is_finite()||!frame_difference.is_finite(){return Err("Nonfinite effective generator bound".into());}
        Ok(Self{n,g,charges:targets,hamiltonian,gamma,jumps,recycling,remainder,frame_difference})
    }
    pub fn position(&self,charge:&[i32])->Option<usize>{self.charges.iter().position(|q|q==charge)}
    /// The pairs `(C'_a, C'_b)` with `R'_p(X) = sum C'_a X C'_b^dagger`.
    pub fn recycling_pairs(&self,index:usize)->impl Iterator<Item=(&[C],&[C])>{
        self.recycling[index].iter().map(|&(a,b)|(self.jumps[a].matrix.as_slice(),self.jumps[b].matrix.as_slice()))
    }
    /// `R'_p(X)` for an n x n row-major `X`.
    pub fn recycle(&self,index:usize,x:&[C],lap:&Lapack)->Result<Vec<C>,String>{
        let n=self.n;let mut out=vec![C::default();n*n];
        for(a,b)in self.recycling_pairs(index){let(ax,_)=products(lap,a,x,n)?;
            let bh:Vec<C>=(0..n*n).map(|k|conj(b[(k%n)*n+k/n])).collect();let(axb,_)=products(lap,&ax,&bh,n)?;
            for(o,&v)in out.iter_mut().zip(&axb){add(o,v);}}
        Ok(out)
    }
    /// `L'_p X` for an n x n row-major `X`.
    pub fn apply(&self,index:usize,x:&[C],lap:&Lapack)->Result<Vec<C>,String>{
        let n=self.n;let(h,gm)=(&self.hamiltonian[index],&self.gamma[index]);
        let(hx,xh)=products(lap,h,x,n)?;let(gx,xg)=products(lap,gm,x,n)?;
        let mut out=self.recycle(index,x,lap)?;
        for k in 0..n*n{let c=C{re:hx[k].re-xh[k].re,im:hx[k].im-xh[k].im};
            out[k].re+=c.im-0.5*(gx[k].re+xg[k].re);out[k].im+=-c.re-0.5*(gx[k].im+xg[k].im);}
        Ok(out)
    }
}
