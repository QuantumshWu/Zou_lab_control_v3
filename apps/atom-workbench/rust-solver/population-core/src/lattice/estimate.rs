//! Error equations M delta = f solved in the primal's Krylov space (GCRO
//! continuation, estimates-math.md section 2).
//!
//! The primal leaves M D_0 = Q_0 with D_0 = P^-1 V_m R^-1 and orthonormal
//! Q_0 = V_m+1 Q_H[:, :m] (from Hbar_m = Q_H [R; 0]). A source f is first
//! projected (delta = D c, c = Q^T f); then each augmentation step takes the
//! current residual direction z, adds g = M P^-1 z orthogonalised against Q
//! as a new column (and its direction to D) and updates every open
//! equation: GMRES on the deflated operator (I - Q Q^T) M, whose spectrum no
//! longer holds the slow outliers the primal resolved. The zero-step
//! projection is never accepted: the part of f outside the space leaks back
//! through the non-normal coupling and is amplified by the slow Ritz values,
//! and the primal's own algebraic error lies exactly along the direction
//! the projection cannot reach.
//!
//! Stop (rule R2, calibrated): the first j >= 2 with the last two population
//! changes both at most TAU |delta p_j|. There is no absolute floor: the
//! first steps can sit far below the answer while the leaking part is still
//! outside the space (three lasers + tweezer: 7e-7 after two steps, 4.5e-4
//! actual).
//!
//! Sources:
//! - truncation (`Shell`): the one-step Jacobi source of the next near
//!   shell, f = -B y_1 - r_p with y_1 = -P_s^-1 C x, so it carries the
//!   primal's algebraic error -r_p too;
//! - fold: the third-order fold source (`FoldLibrary::third_order_source`),
//!   whose response is the population effect of the leading fold remainder.
//!
//! The shell correction is the block Gauss-Seidel outer iteration on the
//! shell, y_i = y_i-1 - P_s^-1 (C x_i-1 + D y_i-1), M delta_i = -B y_i - r_p,
//! each inner solve in the same growing space; its contraction
//! q_2 = |p2 - p1| / |p1 - p_k| decides whether it may be used.
use super::*;
use super::krylov::{Arnoldi,System,combine,symmetrize};
use super::operator::LatticeOperator;
use super::preconditioner::{Preconditioner,StructuredBatch,StructuredDressed};
use super::ledger;

/// R2: relative stability of the last two changes.
const TAU:f64=0.1;
/// Augmentation steps before an equation is declared unstable.
const MAXIMUM_STEPS:usize=40;

/// One error equation in the augmented space.
pub struct Equation{
    pub label:&'static str,
    source:Vec<C>,
    /// Coefficients on the columns of Q, and the residual f - Q c.
    c:Vec<f64>,residual:Vec<C>,
    /// Population estimate after each step (the projection first).
    pub history:Vec<Vec<f64>>,
    pub stable:bool,
    /// Augmentation steps at which R2 was (last) met.
    pub stable_at:Option<usize>,
    /// Relative true residual at the stop (when checked).
    pub true_residual:Option<f64>,
}
impl Equation{
    pub fn estimate(&self)->&[f64]{self.history.last().map_or(&[],|p|p.as_slice())}
    /// Augmentation steps since the equation started.
    pub fn steps(&self)->usize{self.history.len().saturating_sub(1)}
}

/// The augmented space: Q (recycled columns implicit, new columns explicit)
/// and D = P^-1 G E over the generators G (v_1..v_m, then the residual
/// directions z_j).
pub struct Augmented{
    arnoldi:Arnoldi,m:usize,
    directions:Vec<Vec<C>>,
    /// Central populations of P^-1 g for every generator.
    diagonal:Vec<Vec<f64>>,
    columns:Vec<Vec<C>>,
    /// Columns of E (one per column of Q), over the generators.
    e:Vec<Vec<f64>>,
    pub steps:usize,
}
impl Augmented{
    pub fn new(arnoldi:Arnoldi)->Self{
        let m=arnoldi.steps();
        let e=(0..m).map(|k|{let mut unit=vec![0.;m];unit[k]=1.;arnoldi.back_substitute(&unit)}).collect();
        let diagonal=arnoldi.diagonal.clone();
        Self{arnoldi,m,directions:vec![],diagonal,columns:vec![],e,steps:0}
    }
    fn generators(&self)->Vec<&[C]>{self.arnoldi.basis[..self.m].iter().chain(&self.directions).map(|v|v.as_slice()).collect()}
    /// c = Q^T w and w - Q c, for the current Q.
    fn project(&self,system:&System<'_,'_>,w:&mut[C])->Result<Vec<f64>,String>{
        let m=self.m;let vectors=&system.vectors;
        let basis:Vec<&[C]>=self.arnoldi.basis.iter().map(|v|v.as_slice()).collect();
        let mut t=vectors.dots(&basis,w);self.arnoldi.rotate(&mut t);
        let mut c:Vec<f64>=t[..m].to_vec();
        let mut back=c.clone();back.push(0.);self.arnoldi.unrotate(&mut back);vectors.subtract(&basis,&back,w)?;
        if !self.columns.is_empty(){let columns:Vec<&[C]>=self.columns.iter().map(|v|v.as_slice()).collect();
            let h=vectors.dots(&columns,w);vectors.subtract(&columns,&h,w)?;c.extend(h);}
        Ok(c)
    }
    fn populations(&self,c:&[f64],n:usize)->Vec<f64>{
        let g=self.diagonal.len();let mut a=vec![0.;g];
        for(column,&ck)in self.e.iter().zip(c){for(t,&v)in a.iter_mut().zip(column){*t+=ck*v;}}
        let mut p=vec![0.;n];for(row,&ak)in self.diagonal.iter().zip(&a){for(t,&v)in p.iter_mut().zip(row){*t+=ak*v;}}
        p
    }
    /// Starts an equation M delta = f: its projection on the current space.
    pub fn equation(&self,system:&System<'_,'_>,label:&'static str,mut source:Vec<C>)->Result<Equation,String>{
        symmetrize(&system.op,&mut source);
        let mut residual=source.clone();let c=self.project(system,&mut residual)?;
        let p=self.populations(&c,system.watch.len());
        Ok(Equation{label,source,c,residual,history:vec![p],stable:false,stable_at:None,true_residual:None})
    }
    /// One augmentation step along the residual of `lead`, updating every
    /// open equation.
    pub fn extend(&mut self,system:&mut System<'_,'_>,equations:&mut[&mut Equation],lead:usize,groups:&[Vec<usize>])->Result<(),String>{
        let length=system.length();let n=system.watch.len();
        let norm=system.vectors.norm(&equations[lead].residual);
        if norm==0.{equations[lead].stable=true;return Ok(());}
        let z:Vec<C>=equations[lead].residual.iter().map(|v|scaled(*v,1./norm)).collect();
        let(mut u,mut g)=(vec![ZERO;length],vec![ZERO;length]);
        system.precondition(&z,&mut u)?;system.apply(&u,&mut g)?;
        let mut h=self.project(system,&mut g)?;
        let h2=self.project(system,&mut g)?;for(a,b)in h.iter_mut().zip(h2){*a+=b;}
        let eta=system.vectors.norm(&g);
        if eta==0.{return Err("Error-equation augmentation broke down".into());}
        for v in g.iter_mut(){*v=scaled(*v,1./eta);}
        // E_new = (e_z - E h) / eta over the generators.
        let generators=self.diagonal.len()+1;
        let mut column=vec![0.;generators];column[generators-1]=1./eta;
        for(ek,&hk)in self.e.iter().zip(&h){for(t,&v)in column.iter_mut().zip(ek){*t-=hk*v/eta;}}
        for ek in &mut self.e{ek.push(0.);}
        self.e.push(column);self.diagonal.push(system.populations(&u));self.directions.push(z);
        for eq in equations.iter_mut(){
            let ck=system.vectors.dot(&g,&eq.residual);
            system.vectors.subtract(&[&g],&[ck],&mut eq.residual)?;eq.c.push(ck);
        }
        self.columns.push(g);self.steps+=1;
        for eq in equations.iter_mut(){
            let p=self.populations(&eq.c,n);eq.history.push(p);
            let k=eq.history.len();
            if k>=3{let size=ledger::metric(&eq.history[k-1],&vec![0.;n],groups);let limit=TAU*size;
                let change=|i:usize|ledger::metric(&eq.history[i],&eq.history[i-1],groups);
                eq.stable=change(k-1)<=limit&&change(k-2)<=limit;
                if !eq.stable{eq.stable_at=None;}else if eq.stable_at.is_none(){eq.stable_at=Some(k-1);}}
        }
        Ok(())
    }
    /// Augments until every equation satisfies R2.
    pub fn solve(&mut self,system:&mut System<'_,'_>,equations:&mut[&mut Equation],groups:&[Vec<usize>])->Result<(),String>{
        let start=self.steps;
        while let Some(lead)=equations.iter().position(|e|!e.stable){
            if self.steps-start>=MAXIMUM_STEPS{return Err(format!("Error equation {} did not stabilise in {MAXIMUM_STEPS} augmentation steps",equations[lead].label));}
            self.extend(system,equations,lead,groups)?;
        }
        Ok(())
    }
    /// delta = D c of an equation (one preconditioner application).
    pub fn solution(&self,system:&mut System<'_,'_>,eq:&Equation)->Result<Vec<C>,String>{
        let g=self.diagonal.len();let mut a=vec![0.;g];
        for(column,&ck)in self.e.iter().zip(&eq.c){for(t,&v)in a.iter_mut().zip(column){*t+=ck*v;}}
        let combination=combine(&self.generators(),&a,system.length());
        let mut delta=vec![ZERO;system.length()];system.precondition(&combination,&mut delta)?;Ok(delta)
    }
    /// The solution with the true-residual guard: when |f - M delta| exceeds
    /// the recursive residual by more than a factor 2, the equation continues
    /// from the true residual.
    pub fn checked_solution(&mut self,system:&mut System<'_,'_>,eq:&mut Equation,groups:&[Vec<usize>])->Result<Vec<C>,String>{
        let length=system.length();let fnorm=system.vectors.norm(&eq.source).max(f64::MIN_POSITIVE);
        for _ in 0..3{
            let delta=self.solution(system,eq)?;let mut w=vec![ZERO;length];system.apply(&delta,&mut w)?;
            let mut r=eq.source.clone();for(a,b)in r.iter_mut().zip(&w){a.re-=b.re;a.im-=b.im;}
            let(actual,recursive)=(system.vectors.norm(&r),system.vectors.norm(&eq.residual));
            eq.true_residual=Some(actual/fnorm);
            if actual<=2.*recursive+1e-12*fnorm{return Ok(delta);}
            eq.residual=r;eq.stable=false;
            self.solve(system,&mut[&mut *eq],groups)?;
        }
        Err(format!("Error equation {}: true residual {:e} does not follow the recursive one",eq.label,eq.true_residual.unwrap_or(f64::NAN)))
    }
}

/// The next near shell: the order-k+1 ball's operator and the dressed
/// preconditioner on its new modes.
pub struct Shell<'a>{
    pub op:LatticeOperator<'a>,
    pre:StructuredBatch<'a>,
    /// Index in the k+1 ball of each order-k mode, and the shell modes.
    ball:Vec<usize>,pub modes:Vec<usize>,
    pub applications:usize,
}
impl<'a> Shell<'a>{
    /// None when the next order adds no mode (no near coupling).
    pub fn new(prepared:&'a LatticePrepared,dressed:&'a StructuredDressed,inner:&LatticeOperator<'_>,order:usize)->Result<Option<Self>,String>{
        let modes=prepared.modes(order+1)?;if modes.len()==inner.modes.len(){return Ok(None);}
        let op=LatticeOperator::new(prepared,&modes)?;
        let ball=inner.modes.iter().map(|q|op.index.get(q).copied().ok_or("Near ball is not nested")).collect::<Result<Vec<_>,_>>()?;
        let mut inside=vec![false;modes.len()];for &b in &ball{inside[b]=true;}
        let shell:Vec<usize>=(0..modes.len()).filter(|&q|!inside[q]).collect();
        let computed:Vec<bool>=(0..modes.len()).map(|q|op.canonical[q]&&!inside[q]).collect();
        let pre=dressed.batch(&op.nu,&computed,&op.mirror,None)?;
        Ok(Some(Self{op,pre,ball,modes:shell,applications:0}))
    }
    fn d(&self)->usize{self.op.n*self.op.n}
    /// A_k+1 [x; y]: x on the order-k ball, y on the shell modes.
    fn apply(&mut self,x:&[C],y:&[C])->Result<Vec<C>,String>{
        let d=self.d();let mut full=vec![ZERO;self.op.dimension()];
        for(k,&b)in self.ball.iter().enumerate(){full[b*d..(b+1)*d].copy_from_slice(&x[k*d..(k+1)*d]);}
        for(k,&s)in self.modes.iter().enumerate(){full[s*d..(s+1)*d].copy_from_slice(&y[k*d..(k+1)*d]);}
        let mut out=vec![ZERO;full.len()];self.op.apply(&full,&mut out)?;self.applications+=1;Ok(out)
    }
    /// |A_k+1 [x; y]| / |[x; y]| (rad/us): the corrected field's residual
    /// on the order-(k+1) lattice.
    pub fn residual(&mut self,x:&[C],y:&[C])->Result<f64,String>{
        let out=self.apply(x,y)?;let norm=|v:&[C]|v.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        Ok(norm(&out)/(norm(x).hypot(norm(y))).max(f64::MIN_POSITIVE))
    }
    fn ball_rows(&self,full:&[C])->Vec<C>{let d=self.d();self.ball.iter().flat_map(|&b|full[b*d..(b+1)*d].iter().copied()).collect()}
    fn shell_rows(&self,full:&[C])->Vec<C>{let d=self.d();self.modes.iter().flat_map(|&s|full[s*d..(s+1)*d].iter().copied()).collect()}
    /// y - P_s^-1 (C x + D y).
    fn relax(&mut self,full:&[C],y:&[C])->Result<Vec<C>,String>{
        let d=self.d();let mut residual=vec![ZERO;full.len()];
        for &s in &self.modes{residual[s*d..(s+1)*d].copy_from_slice(&full[s*d..(s+1)*d]);}
        let mut z=vec![ZERO;full.len()];self.pre.apply(&residual,&mut z)?;
        let z=self.shell_rows(&z);Ok(y.iter().zip(&z).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect())
    }
    /// The one-step Jacobi truncation source of a normalized order-k field x
    /// (v^T x = 1, so r_p = A x): f = -(A x + B y_1), y_1 = -P_s^-1 C x.
    pub fn source(&mut self,x:&[C])->Result<(Vec<C>,Vec<C>),String>{
        let zero=vec![ZERO;self.modes.len()*self.d()];
        let first=self.apply(x,&zero)?;let y=self.relax(&first,&zero)?;
        let full=self.apply(x,&y)?;let f=self.ball_rows(&full).into_iter().map(|v|C{re:-v.re,im:-v.im}).collect();
        Ok((f,y))
    }
    /// One Gauss-Seidel step from (x_i, y_i): the next shell field and the
    /// source of the next correction, f = -(A x + B y_i+1) with x the
    /// uncorrected field.
    pub fn gauss_seidel(&mut self,x:&[C],corrected:&[C],y:&[C])->Result<(Vec<C>,Vec<C>),String>{
        let full=self.apply(corrected,y)?;let next=self.relax(&full,y)?;
        let full=self.apply(x,&next)?;let f=self.ball_rows(&full).into_iter().map(|v|C{re:-v.re,im:-v.im}).collect();
        Ok((f,next))
    }
}
