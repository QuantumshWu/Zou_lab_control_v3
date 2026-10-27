//! The request frame's generator on Fourier envelopes: one superoperator
//! L_t per charge t, L(t) = sum_t L_t e^{i nu_t t}, built from every
//! Hamiltonian charge and every pair of charges of one collapse channel (the
//! complete non-secular Lindblad form), stored through sparse n x n factors.
use super::*;

/// Sparse n x n matrix: its exact nonzero entries (row, column, value),
/// row-sorted. Zeros are skipped for speed; every stored coefficient acts.
#[derive(Clone,Debug,Default)]
pub struct Sparse{pub entries:Vec<(usize,usize,C)>}
impl Sparse{
    pub fn from_dense(a:&[C],n:usize)->Self{Self{entries:(0..n*n).filter(|&k|nonzero(a[k])).map(|k|(k/n,k%n,a[k])).collect()}}
    pub fn dense(&self,n:usize)->Vec<C>{let mut a=vec![ZERO;n*n];for &(i,j,v)in &self.entries{add(&mut a[i*n+j],v);}a}
    pub fn is_empty(&self)->bool{self.entries.is_empty()}
    /// sqrt(largest row sum * largest column sum) of the moduli: an upper
    /// bound of the spectral norm (Schur test).
    pub fn norm_bound(&self,n:usize)->f64{
        let(mut rows,mut columns)=(vec![0_f64;n],vec![0_f64;n]);
        for &(i,j,v)in &self.entries{let a=magnitude(v);rows[i]+=a;columns[j]+=a;}
        (rows.iter().copied().fold(0.,f64::max)*columns.iter().copied().fold(0.,f64::max)).sqrt()
    }
    /// y += s A x.
    pub fn left_into(&self,n:usize,s:C,x:&[C],y:&mut[C]){
        for &(i,k,a)in &self.entries{let a=mul(s,a);
            for(t,&v)in y[i*n..(i+1)*n].iter_mut().zip(&x[k*n..(k+1)*n]){mul_add(t,a,v);}}
    }
    /// y += s x A.
    pub fn right_into(&self,n:usize,s:C,x:&[C],y:&mut[C]){
        for &(k,j,a)in &self.entries{let a=mul(s,a);for i in 0..n{mul_add(&mut y[i*n+j],x[i*n+k],a);}}
    }
    fn rows(&self)->Vec<usize>{self.entries.iter().map(|e|e.0).collect::<BTreeSet<_>>().into_iter().collect()}
}

/// One Fourier component of the generator acting on n x n blocks:
/// X -> left X + X right + sum_k P_k X Q_k.
#[derive(Clone,Debug,Default)]
pub struct Superop{pub left:Sparse,pub right:Sparse,pub sandwich:Vec<(Sparse,Sparse)>}
impl Superop{
    pub fn is_empty(&self)->bool{self.left.is_empty()&&self.right.is_empty()&&self.sandwich.iter().all(|(p,q)|p.is_empty()||q.is_empty())}
    /// Upper bound of the superoperator norm on the Frobenius space.
    pub fn norm_bound(&self,n:usize)->f64{
        self.left.norm_bound(n)+self.right.norm_bound(n)+self.sandwich.iter().map(|(p,q)|p.norm_bound(n)*q.norm_bound(n)).sum::<f64>()
    }
    /// y += s op(x); `scratch` is resized to n*n.
    pub fn apply_into(&self,n:usize,s:C,x:&[C],y:&mut[C],scratch:&mut Vec<C>){
        self.left.left_into(n,s,x,y);self.right.right_into(n,s,x,y);
        if self.sandwich.is_empty(){return;}
        grow(scratch,n*n);
        for(p,q)in &self.sandwich{
            let rows=p.rows();let t=&mut scratch[..n*n];for &i in &rows{t[i*n..(i+1)*n].fill(ZERO);}
            p.left_into(n,ONE,x,t);
            for &(l,j,b)in &q.entries{let b=mul(s,b);for &i in &rows{mul_add(&mut y[i*n+j],t[i*n+l],b);}}
        }
    }
    /// out += s M, M the row-major d x d matrix of this component.
    pub fn dense_into(&self,n:usize,s:C,out:&mut[C]){
        let d=n*n;
        for &(i,k,a)in &self.left.entries{let a=mul(s,a);for j in 0..n{add(&mut out[(i*n+j)*d+k*n+j],a);}}
        for &(l,j,b)in &self.right.entries{let b=mul(s,b);for i in 0..n{add(&mut out[(i*n+j)*d+i*n+l],b);}}
        for(p,q)in &self.sandwich{for &(i,k,a)in &p.entries{let a=mul(s,a);for &(l,j,b)in &q.entries{add(&mut out[(i*n+j)*d+k*n+l],mul(a,b));}}}
    }
    /// Multiply-adds of one application to one block.
    pub fn work(&self,n:usize)->usize{
        (self.left.entries.len()+self.right.entries.len())*n
            +self.sandwich.iter().map(|(p,q)|p.entries.len()*n+q.entries.len()*p.rows().len()).sum::<usize>()
    }
}

/// One Fourier component L_t of the frame's generator, L(t) = sum_t L_t e^{i nu_t t}.
pub struct Component{pub nu:f64,pub op:Superop,pub size:f64}
/// The generator of the frame on Fourier envelopes: -i[H_t, .] from every
/// Hamiltonian charge t and, from every pair of charges a, b of one collapse
/// channel with a - b = t (the complete non-secular Lindblad form),
/// C_a X C_b^dagger - (C_b^dagger C_a X + X C_b^dagger C_a)/2.
pub struct Generator{pub n:usize,pub zero:Charge,pub components:BTreeMap<Charge,Component>,
    /// Whether each state is a ground state (the sectors of a block).
    pub ground:Vec<bool>}
impl Generator{
    pub fn new(source:&GaugedSource)->Result<Self,String>{
        let n=source.n;let d=n*n;let zero:Charge=vec![0;source.phases.axes.len()];
        let mut left=BTreeMap::<Charge,Vec<C>>::new();let mut right=BTreeMap::<Charge,Vec<C>>::new();
        let mut sandwich=BTreeMap::<Charge,Vec<(Sparse,Sparse)>>::new();
        left.insert(zero.clone(),vec![ZERO;d]);right.insert(zero.clone(),vec![ZERO;d]);
        for(q,h)in &source.hamiltonian{
            let l=left.entry(q.clone()).or_insert_with(||vec![ZERO;d]);for k in 0..d{add(&mut l[k],C{re:h[k].im,im:-h[k].re});}
            let r=right.entry(q.clone()).or_insert_with(||vec![ZERO;d]);for k in 0..d{add(&mut r[k],C{re:-h[k].im,im:h[k].re});}
        }
        for channel in &source.collapse{for(a,ca)in channel{for(b,cb)in channel{
            let t=charge_minus(a,b)?;
            let mut g=vec![ZERO;d];
            for k in 0..n{for i in 0..n{let x=conj(cb[k*n+i]);if !nonzero(x){continue;}for j in 0..n{mul_add(&mut g[i*n+j],x,ca[k*n+j]);}}}
            let l=left.entry(t.clone()).or_insert_with(||vec![ZERO;d]);for k in 0..d{add(&mut l[k],scaled(g[k],-0.5));}
            let r=right.entry(t.clone()).or_insert_with(||vec![ZERO;d]);for k in 0..d{add(&mut r[k],scaled(g[k],-0.5));}
            let dagger:Vec<C>=(0..d).map(|k|conj(cb[(k%n)*n+k/n])).collect();
            sandwich.entry(t).or_default().push((Sparse::from_dense(ca,n),Sparse::from_dense(&dagger,n)));
        }}}
        let charges:BTreeSet<Charge>=left.keys().chain(right.keys()).chain(sandwich.keys()).cloned().collect();
        let mut components=BTreeMap::new();
        for q in charges{
            let op=Superop{left:Sparse::from_dense(left.get(&q).map_or(&vec![ZERO;d],|v|v),n),
                right:Sparse::from_dense(right.get(&q).map_or(&vec![ZERO;d],|v|v),n),sandwich:sandwich.remove(&q).unwrap_or_default()};
            if q!=zero&&op.is_empty(){continue;}
            let f=source.phases.frequency(&q)?;let size=op.norm_bound(n);
            if !size.is_finite(){return Err("Nonfinite lattice generator component".into());}
            components.insert(q,Component{nu:f.hi+f.lo,op,size});
        }
        let ground=(0..n).map(|i|source.ground.contains(&i)).collect();
        Ok(Self{n,zero,components,ground})
    }
    pub fn zero_component(&self)->&Component{&self.components[&self.zero]}
}


#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_generator_components_match_the_dense_lindbladian(){
        // The sum of e^{i nu_t t} L_t over components equals the time-dependent
        // Lindbladian applied to a random block at one time.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let time=0.37;
        let x:Vec<C>=(0..d).map(|k|C{re:((k*7+3)%11)as f64/9.-0.5,im:((k*5+1)%13)as f64/11.-0.5}).collect();
        let mut y=vec![ZERO;d];let mut scratch=Vec::new();
        for c in p.generator.components.values(){let ph=C{re:(c.nu*time).cos(),im:(c.nu*time).sin()};c.op.apply_into(n,ph,&x,&mut y,&mut scratch);}
        let ht:Vec<C>=(0..d).map(|k|C{re:f.h[k]+f.near[k]*(3.*time).cos()+f.far[k]*(2000.*time).cos()+f.far2[k]*(2003.*time).cos(),im:0.}).collect();
        let cm:Vec<C>=f.c.iter().map(|&v|C{re:v,im:0.}).collect();
        let prod=|a:&[C],b:&[C]|->Vec<C>{let mut o=vec![ZERO;d];for i in 0..n{for k in 0..n{for j in 0..n{mul_add(&mut o[i*n+j],a[i*n+k],b[k*n+j]);}}}o};
        let dag=|a:&[C]|->Vec<C>{(0..d).map(|k|conj(a[(k%n)*n+k/n])).collect()};
        let hx=prod(&ht,&x);let xh=prod(&x,&ht);let cdc=prod(&dag(&cm),&cm);let a=prod(&cdc,&x);let b=prod(&x,&cdc);let r=prod(&prod(&cm,&x),&dag(&cm));
        for k in 0..d{let want=C{re:(hx[k].im-xh[k].im)+r[k].re-0.5*(a[k].re+b[k].re),im:-(hx[k].re-xh[k].re)+r[k].im-0.5*(a[k].im+b[k].im)};
            assert!(magnitude(C{re:y[k].re-want.re,im:y[k].im-want.im})<1e-11,"entry {k}");}
        // dense_into agrees with apply_into.
        for c in p.generator.components.values(){let mut m=vec![ZERO;d*d];c.op.dense_into(n,ONE,&mut m);let mut y=vec![ZERO;d];c.op.apply_into(n,ONE,&x,&mut y,&mut scratch);
            for r in 0..d{let mut v=ZERO;for s in 0..d{mul_add(&mut v,m[r*d+s],x[s]);}assert!(magnitude(C{re:v.re-y[r].re,im:v.im-y[r].im})<1e-12);}}
    }
}
