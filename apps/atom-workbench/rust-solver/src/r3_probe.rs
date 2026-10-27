//! R3 probe (research, validation feature only): the inner shifted solves
//! (A - sigma) w = -v of the finite-time rational Krylov under alternative
//! preconditioners and Krylov recycling. Nothing here is product code.
//!
//! Run (ignored tests): ATOM_R3_REQUEST=<request.json>, optional
//! ATOM_R3_ORDER (2), ATOM_R3_SIGMA (20), ATOM_R3_COLUMNS (40),
//! ATOM_R3_OUT (<file.jsonl>, appended).
use crate::lattice_core_adapter::{groups,prepare,initial_density};
use crate::Model;
use atom_population_core::{Complex64 as C,Lapack};
use atom_population_core::lattice::LatticePrepared;
use atom_population_core::lattice::operator::{LatticeOperator,Parts};
use atom_population_core::lattice::preconditioner::{StructuredDressed,Preconditioner};
use atom_population_core::lattice::krylov::Vectors;
use atom_population_core::lattice::floquet::FloquetFrames;
use serde_json::{json,Value};
use std::time::Instant;

const Z:C=C{re:0.,im:0.};
pub(crate) fn env_f(name:&str,default:f64)->f64{std::env::var(name).ok().and_then(|v|v.parse().ok()).unwrap_or(default)}
pub(crate) fn env_u(name:&str,default:usize)->usize{std::env::var(name).ok().and_then(|v|v.parse().ok()).unwrap_or(default)}
pub(crate) fn emit(value:&Value){
    println!("{value}");
    if let Ok(path)=std::env::var("ATOM_R3_OUT"){use std::io::Write;
        let mut f=std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();writeln!(f,"{value}").unwrap();}
}
pub(crate) struct Case{pub name:String,pub model:Model,pub prepared:LatticePrepared,pub rho0:Vec<C>,pub groups:Vec<Vec<usize>>}
pub(crate) fn load()->Case{
    let path=std::env::var("ATOM_R3_REQUEST").expect("ATOM_R3_REQUEST");
    let request:Value=serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
    let(ground,requested)=groups(&model).unwrap();let(prepared,_)=prepare(&model,&ground).unwrap();
    let rho0=initial_density(&model,&Lapack::load().unwrap()).unwrap();
    let name=std::path::Path::new(&path).file_name().unwrap().to_string_lossy().replace(".request.json","");
    Case{name,model,prepared,rho0,groups:requested.iter().map(|g|g.indices.clone()).collect()}
}

/// A recycled space for the shifted operator M = A - sigma: x-space U, its
/// preconditioned images Uy = P U (y-space) and C = M U orthonormal.
#[derive(Default)]
pub(crate) struct Recycle{pub k:usize,pub u:Vec<Vec<C>>,pub uy:Vec<Vec<C>>,pub c:Vec<Vec<C>>}
#[derive(Default,Clone,Debug)]
pub(crate) struct Stats{pub steps:usize,pub cycles:usize,pub residual:f64,pub apply_ms:f64,pub pre_ms:f64,pub orth_ms:f64,pub update_ms:f64,pub ritz:Vec<(f64,f64)>}

fn refs(v:&[Vec<C>])->Vec<&[C]>{v.iter().map(|x|x.as_slice()).collect()}
/// Column-major real (rows x cols) product a (rows x inner) * b (inner x cols).
fn mm(a:&[f64],b:&[f64],rows:usize,inner:usize,cols:usize)->Vec<f64>{
    let mut out=vec![0.;rows*cols];
    for j in 0..cols{for k in 0..inner{let bk=b[k+inner*j];if bk==0.{continue;}for i in 0..rows{out[i+rows*j]+=a[i+rows*k]*bk;}}}
    out
}
fn transpose(a:&[f64],rows:usize,cols:usize)->Vec<f64>{let mut t=vec![0.;rows*cols];for i in 0..rows{for j in 0..cols{t[j+cols*i]=a[i+rows*j];}}t}

/// Flexible right-preconditioned GMRES in real coordinates on M x = b with
/// optional deflation by a recycled space (GCRO); `pre` may change between
/// steps. Cycles of at most `restart` steps until the true relative residual
/// is below `tol` or `max` steps. Updates `recycle` (GCRO-DR harmonic Ritz,
/// smallest |theta|) from the first cycle when given; `ritz` returns the
/// harmonic Ritz values of the first cycle when asked.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fgmres(vectors:&Vectors,b:&[C],apply:&mut dyn FnMut(&[C],&mut[C])->Result<(),String>,pre:&mut dyn FnMut(&[C],&mut[C])->Result<(),String>,
    tol:f64,restart:usize,max:usize,mut recycle:Option<&mut Recycle>,lapack:&Lapack,want_ritz:bool)->Result<(Vec<C>,Stats),String>{
    let len=b.len();let bnorm=vectors.norm(b);let mut st=Stats::default();
    let mut x=vec![Z;len];let mut r=b.to_vec();
    if let Some(rc)=recycle.as_deref(){if !rc.c.is_empty(){
        let cc=vectors.dots(&refs(&rc.c),&r);
        for(u,&a)in rc.u.iter().zip(&cc){for(t,v)in x.iter_mut().zip(u){t.re+=a*v.re;t.im+=a*v.im;}}
        vectors.subtract(&refs(&rc.c),&cc,&mut r)?;}}
    let mut first=true;
    loop{
        let beta=vectors.norm(&r);
        if beta/bnorm<=tol{st.residual=beta/bnorm;break;}
        let k=recycle.as_deref().map_or(0,|rc|rc.c.len());
        let mut v:Vec<Vec<C>>=vec![r.iter().map(|z|C{re:z.re/beta,im:z.im/beta}).collect()];
        let mut zs:Vec<Vec<C>>=Vec::new();
        let mut hcols:Vec<Vec<f64>>=Vec::new();let mut bcols:Vec<Vec<f64>>=Vec::new();
        let mut rot:Vec<(f64,f64)>=Vec::new();let mut g=vec![beta];
        let mut w=vec![Z;len];let mut z=vec![Z;len];
        for j in 0..restart{
            let t=Instant::now();pre(&v[j],&mut z)?;st.pre_ms+=t.elapsed().as_secs_f64()*1e3;
            let t=Instant::now();apply(&z,&mut w)?;st.apply_ms+=t.elapsed().as_secs_f64()*1e3;st.steps+=1;
            let t=Instant::now();
            let mut bc=vec![0.;k];let mut hc=vec![0.;j+1];
            for _ in 0..2{
                if k>0{let rc=recycle.as_deref().unwrap();let c1=vectors.dots(&refs(&rc.c),&w);vectors.subtract(&refs(&rc.c),&c1,&mut w)?;for(a,b)in bc.iter_mut().zip(&c1){*a+=b;}}
                let h1=vectors.dots(&refs(&v),&w);vectors.subtract(&refs(&v),&h1,&mut w)?;for(a,b)in hc.iter_mut().zip(&h1){*a+=b;}
            }
            let hn=vectors.norm(&w);st.orth_ms+=t.elapsed().as_secs_f64()*1e3;
            let mut col=hc.clone();col.push(hn);hcols.push(col.clone());bcols.push(bc);
            for(i,&(c,s))in rot.iter().enumerate(){let(a,bb)=(col[i],col[i+1]);col[i]=c*a+s*bb;col[i+1]=-s*a+c*bb;}
            let size=col[j].hypot(col[j+1]);let(c,s)=if size>0.{(col[j]/size,col[j+1]/size)}else{(1.,0.)};
            rot.push((c,s));let gj=g[j];g.push(-s*gj);g[j]=c*gj;
            zs.push(z.clone());
            v.push(if hn>0.{w.iter().map(|q|C{re:q.re/hn,im:q.im/hn}).collect()}else{vec![Z;len]});
            if g[j+1].abs()/bnorm<=tol*0.5||hn==0.||st.steps>=max{break;}
        }
        let m=zs.len();
        // y from the rotated least squares: R y = g[..m].
        let mut rr=vec![vec![0.;m];m];
        for(j,col)in hcols.iter().enumerate(){let mut c2=col.clone();for(i,&(c,s))in rot[..j].iter().enumerate(){let(a,bb)=(c2[i],c2[i+1]);c2[i]=c*a+s*bb;c2[i+1]=-s*a+c*bb;}
            let(c,s)=rot[j];let(a,bb)=(c2[j],c2[j+1]);c2[j]=c*a+s*bb;for i in 0..=j{rr[i][j]=c2[i];}}
        let mut y=vec![0.;m];for i in(0..m).rev(){let mut s=g[i];for l in i+1..m{s-=rr[i][l]*y[l];}y[i]=s/rr[i][i];}
        // x += Z y - U (B y).
        for(zj,&a)in zs.iter().zip(&y){for(t,q)in x.iter_mut().zip(zj){t.re+=a*q.re;t.im+=a*q.im;}}
        if k>0{let rc=recycle.as_deref().unwrap();
            for i in 0..k{let s:f64=(0..m).map(|j|bcols[j][i]*y[j]).sum();for(t,q)in x.iter_mut().zip(&rc.u[i]){t.re-=s*q.re;t.im-=s*q.im;}}}
        // True residual.
        let t=Instant::now();apply(&x,&mut w)?;st.apply_ms+=t.elapsed().as_secs_f64()*1e3;
        for((a,&bb),c)in r.iter_mut().zip(b).zip(&w){*a=C{re:bb.re-c.re,im:bb.im-c.im};}
        st.residual=vectors.norm(&r)/bnorm;st.cycles+=1;
        // Harmonic Ritz values / recycle update from the first cycle.
        if first&&(want_ritz||recycle.is_some()){
            let t=Instant::now();
            let kk=k;let n=kk+m;let rows=n+1;
            // G (rows x n): [[D, B],[0, Hbar]].
            let rc_old=recycle.as_deref();
            let d:Vec<f64>=rc_old.map_or(vec![],|rc|rc.uy.iter().map(|u|1./vectors.norm(u)).collect());
            let mut gm=vec![0.;rows*n];
            for i in 0..kk{gm[i+rows*i]=d[i];}
            for j in 0..m{for i in 0..kk{gm[i+rows*(kk+j)]=bcols[j][i];}for(i,&h)in hcols[j].iter().enumerate(){gm[kk+i+rows*(kk+j)]=h;}}
            // W^T Vhat_y (rows x n): [[C^T Uy D, 0],[V^T Uy D, Ibar]].
            let mut wv=vec![0.;rows*n];
            if kk>0{let rc=rc_old.unwrap();
                let cu=vectors.gram(&refs(&rc.c),&refs(&rc.uy))?;let vu=vectors.gram(&refs(&v),&refs(&rc.uy))?;
                for j in 0..kk{for i in 0..kk{wv[i+rows*j]=cu[i*kk+j]*d[j];}for i in 0..=m{wv[kk+i+rows*j]=vu[i*kk+j]*d[j];}}}
            for j in 0..m{wv[kk+j+rows*(kk+j)]=1.;}
            let gt=transpose(&gm,rows,n);
            let a=mm(&gt,&gm,n,rows,n);let bmat=mm(&gt,&wv,n,rows,n);
            // theta: a p = theta bmat p  ->  bmat^-1 a.
            let(lu,piv)=lapack.factor(&bmat,n)?;let q=lapack.solve_matrix_factored(&lu,&piv,&a,n)?;
            let(re,im,vecs)=lapack.eigen(&q,n)?;
            if want_ritz{st.ritz=(0..n).map(|i|(re[i],im[i])).collect();}
            if let Some(rc)=recycle.as_deref_mut(){if rc.k>0{
                let mut order:Vec<usize>=(0..n).collect();order.sort_by(|&i,&j|re[i].hypot(im[i]).total_cmp(&re[j].hypot(im[j])));
                let mut chosen:Vec<usize>=Vec::new();let mut taken=vec![false;n];
                for &i in &order{if chosen.len()>=rc.k{break;}if taken[i]{continue;}
                    if im[i]!=0.{let(a0,a1)=if i+1<n&&im[i]>0.&&im[i+1]==-im[i]{(i,i+1)}else{(i-1,i)};
                        if !taken[a0]{chosen.push(a0);chosen.push(a1);taken[a0]=true;taken[a1]=true;}}
                    else{chosen.push(i);taken[i]=true;}}
                // Real basis Pk (n x kc): dgeev stores a pair as (re, im) columns.
                let kc=chosen.len();let mut pk=vec![0.;n*kc];
                for(c,&i)in chosen.iter().enumerate(){pk[c*n..(c+1)*n].copy_from_slice(&vecs[i*n..(i+1)*n]);}
                // QR of G Pk (rows x kc) by modified Gram-Schmidt.
                let mut gp=mm(&gm,&pk,rows,n,kc);let mut rmat=vec![0.;kc*kc];let mut keep=Vec::new();
                for j in 0..kc{
                    for &i in &keep{let s:f64=(0..rows).map(|t|gp[t+rows*i]*gp[t+rows*j]).sum();rmat[i+kc*j]=s;for t in 0..rows{gp[t+rows*j]-=s*gp[t+rows*i];}}
                    for &i in &keep{let s:f64=(0..rows).map(|t|gp[t+rows*i]*gp[t+rows*j]).sum();rmat[i+kc*j]+=s;for t in 0..rows{gp[t+rows*j]-=s*gp[t+rows*i];}}
                    let nn=(0..rows).map(|t|gp[t+rows*j].powi(2)).sum::<f64>().sqrt();
                    let scale=(0..n).map(|t|pk[t+n*j].powi(2)).sum::<f64>().sqrt();
                    if nn>1e-10*scale.max(1e-300){rmat[j+kc*j]=nn;for t in 0..rows{gp[t+rows*j]/=nn;}keep.push(j);}
                }
                // Keep the independent columns: Pk R^-1 over them.
                let kept=keep.len();
                let mut rk=vec![0.;kept*kept];for(a,&i)in keep.iter().enumerate(){for(bb,&j)in keep.iter().enumerate(){rk[a+kept*bb]=rmat[i+kc*j];}}
                let mut pkk=vec![0.;n*kept];for(a,&j)in keep.iter().enumerate(){pkk[a*n..(a+1)*n].copy_from_slice(&pk[j*n..(j+1)*n]);}
                // Pk R^-1: solve X R = Pk (upper triangular, column by column).
                let mut pr=vec![0.;n*kept];
                for j in 0..kept{for t in 0..n{let mut s=pkk[t+n*j];for i in 0..j{s-=pr[t+n*i]*rk[i+kept*j];}pr[t+n*j]=s/rk[j+kept*j];}}
                let qk:Vec<Vec<f64>>=keep.iter().map(|&j|(0..rows).map(|t|gp[t+rows*j]).collect()).collect();
                let prc:Vec<Vec<f64>>=(0..kept).map(|j|pr[j*n..(j+1)*n].to_vec()).collect();
                // C' = [C, V] Q; U' = [U D, Z] Pk R^-1; Uy' = [Uy D, V_m] Pk R^-1.
                let mut cw:Vec<&[C]>=rc.c.iter().map(|x|x.as_slice()).collect();cw.extend(v.iter().map(|x|x.as_slice()));
                let new_c=vectors.combinations(&cw,&qk)?;
                let scaled_u:Vec<Vec<C>>=rc.u.iter().zip(&d).map(|(u,&s)|u.iter().map(|z|C{re:z.re*s,im:z.im*s}).collect()).collect();
                let scaled_uy:Vec<Vec<C>>=rc.uy.iter().zip(&d).map(|(u,&s)|u.iter().map(|z|C{re:z.re*s,im:z.im*s}).collect()).collect();
                let mut uw:Vec<&[C]>=scaled_u.iter().map(|x|x.as_slice()).collect();uw.extend(zs.iter().map(|x|x.as_slice()));
                let mut yw:Vec<&[C]>=scaled_uy.iter().map(|x|x.as_slice()).collect();yw.extend(v[..m].iter().map(|x|x.as_slice()));
                let new_u=vectors.combinations(&uw,&prc)?;let new_uy=vectors.combinations(&yw,&prc)?;
                rc.c=new_c;rc.u=new_u;rc.uy=new_uy;
            }}
            st.update_ms+=t.elapsed().as_secs_f64()*1e3;
        }
        first=false;
        if st.residual<=tol||st.steps>=max{break;}
        // Restart without the recycle space's projection (r already reflects it).
        if let Some(rc)=recycle.as_deref(){if !rc.c.is_empty(){
            let cc=vectors.dots(&refs(&rc.c),&r);
            for(u,&a)in rc.u.iter().zip(&cc){for(t,q)in x.iter_mut().zip(u){t.re+=a*q.re;t.im+=a*q.im;}}
            vectors.subtract(&refs(&rc.c),&cc,&mut r)?;}}
    }
    Ok((x,st))
}

/// The baseline rational Krylov columns: v_0 = x(0)/|x(0)|, v_{j+1} the
/// orthonormalized (sigma - A)^-1 v_j (separable dressed preconditioner),
/// as finite.rs builds them (without translated copies). Returns the
/// right-hand sides v_j and the baseline stats per column.
pub(crate) fn columns(case:&Case,modes:&[Vec<i32>],sigma:f64,count:usize,tol:f64)->(Vec<Vec<C>>,Vec<Stats>){
    let p=&case.prepared;let lapack=&p.lapack;
    let mut op=LatticeOperator::new(p,modes).unwrap();let vectors=Vectors::new(&op).unwrap();
    let dressed=StructuredDressed::new(p).unwrap();
    let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
    let d=p.generator.n.pow(2);let mut x0=vec![Z;op.dimension()];x0[op.centre*d..(op.centre+1)*d].copy_from_slice(&case.rho0);
    let nrm=vectors.norm(&x0);let mut basis=vec![x0.iter().map(|z|C{re:z.re/nrm,im:z.im/nrm}).collect::<Vec<C>>()];
    let mut stats=Vec::new();
    for j in 0..count{
        let b:Vec<C>=basis[j].iter().map(|z|C{re:-z.re,im:-z.im}).collect();
        let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
        let mut pre=|x:&[C],y:&mut[C]|batch.apply(x,y);
        let(mut w,st)=fgmres(&vectors,&b,&mut apply,&mut pre,tol,60,600,None,lapack,false).unwrap();
        stats.push(st);
        for _ in 0..2{let h=vectors.dots(&refs(&basis),&w);vectors.subtract(&refs(&basis),&h,&mut w).unwrap();}
        let nw=vectors.norm(&w);if !(nw>1e-12){break;}
        basis.push(w.iter().map(|z|C{re:z.re/nw,im:z.im/nw}).collect());
    }
    basis.truncate(stats.len());
    (basis,stats)
}

fn summary(stats:&[Stats])->Value{
    let steps:Vec<usize>=stats.iter().map(|s|s.steps).collect();let total:usize=steps.iter().sum();
    let sum=|f:&dyn Fn(&Stats)->f64|stats.iter().map(f).sum::<f64>();
    json!({"columns":stats.len(),"steps":total,"perColumn":total as f64/stats.len().max(1) as f64,"stepsList":steps,
        "worstResidual":stats.iter().map(|s|s.residual).fold(0.,f64::max),"cycles":stats.iter().map(|s|s.cycles).sum::<usize>(),
        "applyMs":sum(&|s|s.apply_ms),"preMs":sum(&|s|s.pre_ms),"orthMs":sum(&|s|s.orth_ms),"updateMs":sum(&|s|s.update_ms),
        "totalMs":sum(&|s|s.apply_ms+s.pre_ms+s.orth_ms+s.update_ms)})
}

/// Inner solve of (selected parts of A) - sigma to `tol` with the separable
/// block preconditioner: an exact inverse of those parts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parts_solve(op:&mut LatticeOperator<'_>,batch:&mut dyn Preconditioner,vectors:&Vectors,lapack:&Lapack,parts:Parts,sigma:f64,tol:f64,x:&[C],y:&mut[C])->Result<usize,String>{
    let mut apply=|u:&[C],w:&mut[C]|->Result<(),String>{op.apply_parts(u,w,parts)?;for(a,q)in w.iter_mut().zip(u){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
    let mut pre=|u:&[C],w:&mut[C]|batch.apply(u,w);
    let(sol,st)=fgmres(vectors,x,&mut apply,&mut pre,tol,120,2000,None,lapack,false)?;
    y.copy_from_slice(&sol);Ok(st.steps)
}

#[cfg(test)]
mod tests{
    use super::*;
    /// Phase 1: the case's structure, baseline columns, Ritz values of the
    /// preconditioned shifted operator, the bracketing preconditioners
    /// (exact inverse of the diagonal blocks, of the separable lattice, of
    /// the lattice without fold rests) and GCRO-DR recycling.
    #[test]
    #[ignore]
    fn r3_phase1(){
        let case=load();let p=&case.prepared;let lapack=&p.lapack;
        let order=env_u("ATOM_R3_ORDER",2);let sigma=env_f("ATOM_R3_SIGMA",20.);let count=env_u("ATOM_R3_COLUMNS",40);let tol=env_f("ATOM_R3_TOL",1e-6);
        let modes=p.modes(order).unwrap();
        let op=LatticeOperator::new(p,&modes).unwrap();
        let steps:Vec<Value>=p.plan.explicit().into_iter().map(|s|{let st=&p.plan.steps[s];json!({"charge":st.charge,"nu":st.nu,"size":st.size,"mixing":st.size/st.nu.abs()})}).collect();
        let dressed=StructuredDressed::new(p).unwrap();
        let batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
        emit(&json!({"probe":"structure","case":case.name,"order":order,"modes":modes.len(),"canonical":op.canonical_count(),"edges":op.edge_counts,
            "work":op.work_parts(),"preWork":batch.work(),"explicitSteps":steps,"nuMax":op.nu_max(),"precessingRate":dressed.diagnostics.precessing_rate}));
        let only=std::env::var("ATOM_R3_ONLY").unwrap_or_default();
        let run=|name:&str|only.is_empty()||only.split(',').any(|x|x==name);
        let clock=Instant::now();
        let(rhs,base)=columns(&case,&modes,sigma,count,tol);
        emit(&json!({"probe":"baseline","case":case.name,"sigma":sigma,"tol":tol,"ms":clock.elapsed().as_secs_f64()*1e3,"summary":summary(&base)}));
        let mut op=LatticeOperator::new(p,&modes).unwrap();let vectors=Vectors::new(&op).unwrap();
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
        // Harmonic Ritz values of M = (A - sigma) P^-1 from a long Arnoldi on v_5.
        if run("ritz"){
            let b:Vec<C>=rhs[5.min(rhs.len()-1)].iter().map(|z|C{re:-z.re,im:-z.im}).collect();
            let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
            let mut pre=|x:&[C],y:&mut[C]|batch.apply(x,y);
            let(_,st)=fgmres(&vectors,&b,&mut apply,&mut pre,1e-14,100,100,None,lapack,true).unwrap();
            let mut ritz=st.ritz.clone();ritz.sort_by(|a,b|(a.0-1.).hypot(a.1).total_cmp(&(b.0-1.).hypot(b.1)).reverse());
            let outside:Vec<usize>=[0.1f64,0.3,0.5,0.9].iter().map(|&r|ritz.iter().filter(|&&(a,b)|(a-1.).hypot(b)>r).count()).collect();
            emit(&json!({"probe":"ritz","case":case.name,"steps":st.steps,"residual":st.residual,"farthest":ritz.iter().take(30).map(|&(a,b)|[a,b]).collect::<Vec<_>>(),
                "outside":outside,
                "smallest":ritz.iter().map(|&(a,b)|a.hypot(b)).fold(f64::INFINITY,f64::min)}));
        }
        // Bracketing preconditioners, nested (outer flexible GMRES).
        for(name,parts)in[("exact-blocks",Parts::Diagonal),("exact-separable",Parts::Separable),("exact-norest",Parts::NoRest)]{
            if !run(name){continue;}
            let mut inner_op=LatticeOperator::new(p,&modes).unwrap();
            let mut inner_batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
            let inner=std::cell::Cell::new(0);
            let mut pre=|x:&[C],y:&mut[C]|->Result<(),String>{let s=parts_solve(&mut inner_op,&mut inner_batch,&vectors,lapack,parts,sigma,1e-9,x,y)?;inner.set(inner.get()+s);Ok(())};
            let mut stats=Vec::new();let clock=Instant::now();
            for v in rhs.iter().take(env_u("ATOM_R3_NESTED",12)){
                let b:Vec<C>=v.iter().map(|z|C{re:-z.re,im:-z.im}).collect();
                let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
                let(_,st)=fgmres(&vectors,&b,&mut apply,&mut pre,tol,60,600,None,lapack,false).unwrap();stats.push(st);
            }
            emit(&json!({"probe":name,"case":case.name,"ms":clock.elapsed().as_secs_f64()*1e3,"innerSteps":inner.get(),"summary":summary(&stats),
                "baselineSame":base.iter().take(stats.len()).map(|s|s.steps).sum::<usize>()}));
        }
        // GCRO-DR across the columns' solves.
        for k in [10usize,20,40]{
            if !run(&format!("gcrodr{k}")){continue;}
            let mut rc=Recycle{k,..Default::default()};let mut stats=Vec::new();
            for v in &rhs{
                let b:Vec<C>=v.iter().map(|z|C{re:-z.re,im:-z.im}).collect();
                let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
                let mut pre=|x:&[C],y:&mut[C]|batch.apply(x,y);
                let(_,st)=fgmres(&vectors,&b,&mut apply,&mut pre,tol,60,600,Some(&mut rc),lapack,false).unwrap();stats.push(st);
            }
            emit(&json!({"probe":format!("gcrodr{k}"),"case":case.name,"summary":summary(&stats)}));
        }
    }

    /// Floquet frames: the step counts of the shifted solves on the baseline
    /// columns' right-hand sides with Floquet-frame preconditioners
    /// (ATOM_R3_STRONG: comma list of mixing thresholds), and the distance
    /// of each to the exact separable-lattice inverse on one field.
    #[test]
    #[ignore]
    fn r3_floquet(){
        let case=load();let p=&case.prepared;let lapack=&p.lapack;
        let order=env_u("ATOM_R3_ORDER",2);let sigma=env_f("ATOM_R3_SIGMA",20.);let count=env_u("ATOM_R3_COLUMNS",40);let tol=env_f("ATOM_R3_TOL",1e-6);
        let guard=env_f("ATOM_R3_GUARD",0.5);
        let modes=p.modes(order).unwrap();
        let clock=Instant::now();
        let(rhs,base)=columns(&case,&modes,sigma,count,tol);
        emit(&json!({"probe":"baseline","case":case.name,"sigma":sigma,"tol":tol,"ms":clock.elapsed().as_secs_f64()*1e3,"summary":summary(&base)}));
        let mut op=LatticeOperator::new(p,&modes).unwrap();let vectors=Vectors::new(&op).unwrap();
        let dressed=StructuredDressed::new(p).unwrap();
        // The exact separable-lattice inverse on v_3 (inner solve) for the distance.
        let probe_rhs:Vec<C>=rhs[3.min(rhs.len()-1)].clone();
        let mut sep_op=LatticeOperator::new(p,&modes).unwrap();
        let mut sep_batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
        let mut exact=vec![Z;probe_rhs.len()];parts_solve(&mut sep_op,&mut sep_batch,&vectors,lapack,Parts::Separable,sigma,1e-12,&probe_rhs,&mut exact).unwrap();
        let en=vectors.norm(&exact);
        let distance=|y:&[C]|{let diff:Vec<C>=y.iter().zip(&exact).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();vectors.norm(&diff)/en};
        let mut yb=vec![Z;probe_rhs.len()];sep_batch.apply(&probe_rhs,&mut yb).unwrap();
        emit(&json!({"probe":"separable-distance","case":case.name,"blockJacobi":distance(&yb),"preWork":sep_batch.work(),"opWork":op.work_parts()}));
        let strengths:Vec<f64>=std::env::var("ATOM_R3_STRONG").unwrap_or("1e-2".into()).split(',').map(|x|x.parse().unwrap()).collect();
        for strong in strengths{
            let clock=Instant::now();let frames=FloquetFrames::new(p,strong,guard).unwrap();let build=clock.elapsed().as_secs_f64()*1e3;
            let mut batch=frames.batch(&op,sigma).unwrap();
            let mut yf=vec![Z;probe_rhs.len()];batch.apply(&probe_rhs,&mut yf).unwrap();
            let mut stats=Vec::new();
            for v in &rhs{
                let b:Vec<C>=v.iter().map(|z|C{re:-z.re,im:-z.im}).collect();
                let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
                let mut pre=|x:&[C],y:&mut[C]|batch.apply(x,y);
                let(_,st)=fgmres(&vectors,&b,&mut apply,&mut pre,tol,60,600,None,lapack,false).unwrap();stats.push(st);
            }
            emit(&json!({"probe":"floquet","case":case.name,"strong":strong,"guard":guard,"support":frames.support,"buildMs":build,"diagnostics":format!("{:?}",frames.diagnostics),
                "distance":distance(&yf),"work":batch.work,"halfWork":batch.half_work,"summary":summary(&stats)}));
        }
    }

    /// Structure: connected components of NL' (charge 0 left factor, with the
    /// absorbable fold) and of NL' plus the strong steps' left parts; the
    /// conjugate relation between the right and left data.
    #[test]
    #[ignore]
    fn r3_structure(){
        let case=load();let p=&case.prepared;let g=&p.generator;let n=g.n;let d=n*n;
        let zero=g.zero_component();
        let mut nl=zero.op.left.dense(n);let mut nr=zero.op.right.dense(n);
        if let Some(t)=p.fold.targets.first().filter(|t|t.charge==g.zero){for k in 0..d{nl[k].re+=t.left[0][k].re;nl[k].im+=t.left[0][k].im;nr[k].re+=t.right[0][k].re;nr[k].im+=t.right[0][k].im;}}
        let scale=nl.iter().map(|z|z.re.hypot(z.im)).fold(0.,f64::max);
        let comps=|m:&[C],extra:&[Vec<C>]|->Vec<usize>{
            let mut parent:Vec<usize>=(0..n).collect();
            fn find(p:&mut Vec<usize>,i:usize)->usize{let mut r=i;while p[r]!=r{r=p[r];}let mut j=i;while p[j]!=r{let nx=p[j];p[j]=r;j=nx;}r}
            for mm in std::iter::once(m).chain(extra.iter().map(|v|v.as_slice())){
                for i in 0..n{for j in 0..n{if i!=j&&mm[i*n+j].re.hypot(mm[i*n+j].im)>1e-14*scale{let(a,b)=(find(&mut parent,i),find(&mut parent,j));parent[a]=b;}}}}
            let mut sizes=std::collections::BTreeMap::new();for i in 0..n{let r=find(&mut parent,i);*sizes.entry(r).or_insert(0usize)+=1;}
            let mut v:Vec<usize>=sizes.into_values().collect();v.sort_unstable_by(|a,b|b.cmp(a));v};
        let strong:Vec<Vec<C>>=p.plan.explicit().into_iter().map(|s|&p.plan.steps[s]).filter(|st|st.size>=1e-2*st.nu.abs()).map(|st|g.components[&st.charge].op.left.dense(n)).collect();
        let all:Vec<Vec<C>>=p.plan.explicit().into_iter().map(|s|g.components[&p.plan.steps[s].charge].op.left.dense(n)).collect();
        // Conjugate relation: NR'^T = conj(NL') up to a multiple of the identity?
        let mut defect=0_f64;let shift=C{re:nr[0].re-nl[0].re,im:nr[0].im+nl[0].im};
        for i in 0..n{for j in 0..n{let a=nr[j*n+i];let b=nl[i*n+j];let e=if i==j{C{re:a.re-b.re-shift.re,im:a.im+b.im-shift.im}}else{C{re:a.re-b.re,im:a.im+b.im}};defect=defect.max(e.re.hypot(e.im));}}
        let mut step_defect=0_f64;
        for s in p.plan.explicit(){let st=&p.plan.steps[s];let b=g.components[&st.charge].op.right.dense(n);
            let Some(o)=p.plan.opposite(s)else{continue};let a=g.components[&p.plan.steps[o].charge].op.left.dense(n);
            for i in 0..n{for j in 0..n{let x=b[j*n+i];let y=a[i*n+j];step_defect=step_defect.max((x.re-y.re).hypot(x.im+y.im));}}}
        let nnz=nl.iter().filter(|z|z.re.hypot(z.im)>1e-14*scale).count();
        let frames=FloquetFrames::by_kernel(p,0.005,0.5).unwrap();
        let sv:Vec<Vec<f64>>=frames.kernel_singular_values().unwrap().into_iter().map(|v|{let top=v[0];v.into_iter().map(|x|x/top).collect()}).collect();
        let ranks:Vec<[usize;3]>=sv.iter().map(|v|[1e-1,1e-2,1e-3].map(|t|v.iter().filter(|&&x|x>t).count())).collect();
        emit(&json!({"probe":"kernelRank","case":case.name,"support":frames.support,"kernelSizes":frames.diagnostics.kernel_size,"ranksAbove1e-1,1e-2,1e-3":ranks}));
        emit(&json!({"probe":"structure2","case":case.name,"nlNonzeros":nnz,"components":comps(&nl,&[]),"withStrong":comps(&nl,&strong),"withAll":comps(&nl,&all),
            "conjugateDefect":defect/scale,"diagonalShift":[shift.re,shift.im],"stepConjugateDefect":step_defect/scale,
            "explicitSparse":p.plan.explicit().into_iter().map(|s|{let c=&g.components[&p.plan.steps[s].charge].op;[c.left.entries.len(),c.right.entries.len(),c.sandwich.len()]}).collect::<Vec<_>>()}));
    }

    /// Methods on the same right-hand sides (the baseline columns at
    /// ATOM_R3_GEN_SIGMA, default sigma), solved at ATOM_R3_SIGMA:
    /// ATOM_R3_METHODS = list of block | block+wb | floquet:<strong>[:<pairs>],
    /// each optionally +gcro:<k>.
    #[test]
    #[ignore]
    fn r3_solve(){
        let case=load();let p=&case.prepared;let lapack=&p.lapack;
        let order=env_u("ATOM_R3_ORDER",2);let sigma=env_f("ATOM_R3_SIGMA",20.);let gen=env_f("ATOM_R3_GEN_SIGMA",sigma);
        let count=env_u("ATOM_R3_COLUMNS",40);let tol=env_f("ATOM_R3_TOL",1e-6);let guard=env_f("ATOM_R3_GUARD",0.5);
        let modes=p.modes(order).unwrap();
        let(rhs,_)=columns(&case,&modes,gen,count,1e-6);
        let mut op=LatticeOperator::new(p,&modes).unwrap();let vectors=Vectors::new(&op).unwrap();
        let dressed=StructuredDressed::new(p).unwrap();
        let methods=std::env::var("ATOM_R3_METHODS").unwrap_or("block,floquet:1e-2".into());
        for method in methods.split(','){
            let(base,gcro)=match method.split_once("+gcro:"){Some((b,k))=>(b,k.parse::<usize>().unwrap()),None=>(method,0)};
            let(base,cw)=match base.strip_suffix("+cw"){Some(b)=>(b,true),None=>(base,false)};
            let clock=Instant::now();
            let frames:Option<FloquetFrames>=if let Some(rest)=base.strip_prefix("fk:"){Some(FloquetFrames::by_kernel(p,rest.parse().unwrap(),guard).unwrap())}
                else if let Some(rest)=base.strip_prefix("floquet:"){
                    let mut it=rest.split(':');let strong:f64=it.next().unwrap().parse().unwrap();let pairs:usize=it.next().map_or(usize::MAX,|x|x.parse().unwrap());
                    Some(FloquetFrames::with(p,strong,guard,pairs).unwrap())}else{None};
            let mut fb=frames.as_ref().map(|f|f.batch_with(&op,sigma,cw,false).unwrap());
            let mut sb=if frames.is_none(){Some(dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,base=="block+wb").unwrap())}else{None};
            let pre:&mut dyn Preconditioner=if let Some(f)=fb.as_mut(){f}else{sb.as_mut().unwrap()};
            let build=clock.elapsed().as_secs_f64()*1e3;let pre_work=pre.work();
            let mut rc=Recycle{k:gcro,..Default::default()};let mut stats=Vec::new();
            for v in &rhs{
                let b:Vec<C>=v.iter().map(|z|C{re:-z.re,im:-z.im}).collect();
                let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
                let mut prec=|x:&[C],y:&mut[C]|pre.apply(x,y);
                let(_,st)=fgmres(&vectors,&b,&mut apply,&mut prec,tol,80,800,if gcro>0{Some(&mut rc)}else{None},lapack,false).unwrap();stats.push(st);
            }
            let support=frames.as_ref().map(|f|f.support.clone());let kernels=frames.as_ref().map(|f|f.diagnostics.kernel_size.clone());
            emit(&json!({"probe":"solve","case":case.name,"method":method,"support":support,"kernels":kernels,"sigma":sigma,"genSigma":gen,"tol":tol,"modes":modes.len(),"buildMs":build,
                "opWork":op.work_parts(),"preWork":pre_work,"summary":summary(&stats)}));
        }
    }

    /// The steady trace-row solve (production GMRES, population stability
    /// and residual guard) with the dressed preconditioner (Woodbury) and
    /// with Floquet frames + centre Woodbury + trace row (ATOM_R3_KAPPAS).
    #[test]
    #[ignore]
    fn r3_steady(){
        use atom_population_core::lattice::krylov::{self,KrylovOptions,System};
        let case=load();let p=&case.prepared;let order=env_u("ATOM_R3_ORDER",2);let guard=env_f("ATOM_R3_GUARD",0.5);
        let modes=p.modes(order).unwrap();let options=KrylovOptions::default();
        let run=|pre:&mut dyn Preconditioner|{let op=LatticeOperator::new(p,&modes).unwrap();let mut system=System::new(op,pre).unwrap();let b=system.source();
            let clock=Instant::now();let k=krylov::gmres(&mut system,&b,None,&case.groups,&options).unwrap();
            let t:f64=system.watch.iter().map(|&i|k.x[i].re).sum();
            (k.steps,clock.elapsed().as_secs_f64()*1e3,system.watch.iter().map(|&i|k.x[i].re/t).collect::<Vec<f64>>(),k.stop,k.residual,system.timing.clone())};
        let op=LatticeOperator::new(p,&modes).unwrap();
        let dressed=StructuredDressed::new(p).unwrap();
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true).unwrap();let w=batch.work();
        let a=run(&mut batch);
        emit(&json!({"probe":"steady","case":case.name,"method":"dressed+wb","modes":modes.len(),"steps":a.0,"ms":a.1,"stop":a.3,"residual":a.4,"preWork":w,"opWork":op.work_parts(),
            "timing":[a.5.apply_ms,a.5.precondition_ms,a.5.orthogonalize_ms]}));
        for kappa in std::env::var("ATOM_R3_KAPPAS").unwrap_or("0.005".into()).split(','){
            let frames=FloquetFrames::by_kernel(p,kappa.parse().unwrap(),guard).unwrap();
            for cw in [true,false]{
                let mut fb=frames.batch_with(&op,0.,cw,true).unwrap();let w=fb.work;let moved=fb.moved;
                let b=run(&mut fb);
                let diff=a.2.iter().zip(&b.2).map(|(x,y)|(x-y).abs()).fold(0.,f64::max);
                emit(&json!({"probe":"steady","case":case.name,"method":format!("fk:{kappa}{}",if cw{"+cw"}else{""}),"steps":b.0,"ms":b.1,"stop":b.3,"residual":b.4,"preWork":w,"moved":moved,
                    "populationDifference":diff,"timing":[b.5.apply_ms,b.5.precondition_ms,b.5.orthogonalize_ms]}));
            }
        }
    }

    /// End to end through the adapter: ATOM_R3_RUNS = list of
    /// label[:VAR=VALUE;VAR=VALUE] (R3 switches), ATOM_R3_DURATION (us),
    /// ATOM_R3_SAMPLES, ATOM_R3_STEADY=1 for the mean alone. Populations are
    /// compared with the first run's (max abs over rows and states).
    #[test]
    #[ignore]
    fn r3_pipeline(){
        let case=load();
        let duration=env_f("ATOM_R3_DURATION",20.);let samples=env_u("ATOM_R3_SAMPLES",200);let steady=env_u("ATOM_R3_STEADY",0)==1;
        let options:crate::Options=serde_json::from_value(json!({"durationUs":duration,"samples":samples,"steadyOnly":steady,"includeAsymptotic":steady})).unwrap();
        let runs=std::env::var("ATOM_R3_RUNS").unwrap_or("base,floquet:ATOM_R3_PRECOND=floquet".into());
        let mut first:Option<Vec<Vec<f64>>>=None;
        for run in runs.split(','){
            let(label,vars)=run.split_once(':').unwrap_or((run,""));
            for k in ["ATOM_R3_PRECOND","ATOM_R3_KAPPA","ATOM_R3_INNER_RTOL","ATOM_R3_POP_TOL","ATOM_R3_SHELL"]{std::env::remove_var(k);}
            for kv in vars.split(';').filter(|x|!x.is_empty()){let(k,v)=kv.split_once('=').unwrap();std::env::set_var(k,v);}
            let clock=Instant::now();
            let result=crate::lattice_core_adapter::solve(&case.model,&options,&mut |_|{});
            let ms=clock.elapsed().as_secs_f64()*1e3;
            let result=match result{Ok(r)=>r,Err(e)=>{emit(&json!({"probe":"pipeline","case":case.name,"run":label,"error":e}));continue;}};
            let rows:Vec<Vec<f64>>=if steady{vec![serde_json::from_value(result["asymptoticPopulations"].clone()).unwrap_or_default()]}
                else{serde_json::from_value(result["populations"].clone()).unwrap()};
            let diff=first.as_ref().map(|f|f.iter().zip(&rows).map(|(a,b)|a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.,f64::max)).fold(0.,f64::max));
            if first.is_none(){first=Some(rows.clone());}
            let dg=&result["diagnostics"];
            let lattice=&dg["lattice"];
            emit(&json!({"probe":"pipeline","case":case.name,"run":label,"vars":vars,"duration":duration,"steady":steady,"ms":ms,"elapsedMs":dg["elapsedMs"],"difference":diff,
                "finite":if steady{Value::Null}else{json!({"innerSteps":dg["finite"]["innerSteps"],"perColumn":dg["finite"]["innerStepsPerColumn"],"columns":dg["finite"]["columns"],
                    "applications":dg["finite"]["applications"],"preconditions":dg["finite"]["preconditions"],"order":dg["finite"]["order"],"modes":dg["finite"]["modes"],
                    "estimate":result["populationAccuracy"]["estimatedMaximumError"],"basis":dg["finite"]["basisEstimate"],"shell":dg["finite"]["shellEstimate"],"fold":dg["finite"]["foldEstimate"],
                    "longTime":dg["finite"]["longTime"].is_object(),"timing":dg["finiteTimingMs"],"preparation":dg["preparationMs"]})},
                "steadyAttempts":lattice["attempts"].as_array().map(|a|a.iter().map(|x|json!({"order":x["order"],"steps":x["steps"],"augmentation":x["augmentationSteps"],
                    "applications":x["applications"],"preWork":x["preconditionerMacPerApplication"],"opWork":x["operatorMacPerApplication"],"claimed":x["claimed"],"timing":x["timingMs"]})).collect::<Vec<_>>()),
                "asymptoticError":dg["asymptoticAccuracy"]["estimatedMaximumError"]}));
        }
    }

    /// Errors (not residuals) of the inner solves: per method and tolerance,
    /// |w - w*| / |w*| against a 1e-12 solve, its centre-block and
    /// central-population parts, and the preconditioned-residual estimate
    /// |P^-1 r| / |w|, on the baseline columns' right-hand sides.
    #[test]
    #[ignore]
    fn r3_errors(){
        let case=load();let p=&case.prepared;let lapack=&p.lapack;let n=p.generator.n;let d=n*n;
        let order=env_u("ATOM_R3_ORDER",2);let sigma=env_f("ATOM_R3_SIGMA",20.);let count=env_u("ATOM_R3_COLUMNS",12);
        let modes=p.modes(order).unwrap();
        let(rhs,_)=columns(&case,&modes,sigma,count,1e-6);
        let mut op=LatticeOperator::new(p,&modes).unwrap();let vectors=Vectors::new(&op).unwrap();let centre=op.centre;
        let dressed=StructuredDressed::new(p).unwrap();
        let frames=FloquetFrames::by_kernel(p,env_f("ATOM_R3_KAPPA",0.005),0.5).unwrap();
        let mut sb=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
        let mut fb=frames.batch_with(&op,sigma,true,false).unwrap();
        let tols:Vec<f64>=std::env::var("ATOM_R3_TOLS").unwrap_or("1e-6,1e-7,1e-8".into()).split(',').map(|x|x.parse().unwrap()).collect();
        let mut table:Vec<Value>=Vec::new();
        for(j,v)in rhs.iter().enumerate(){
            let b:Vec<C>=v.iter().map(|z|C{re:-z.re,im:-z.im}).collect();
            let mut apply=|x:&[C],y:&mut[C]|->Result<(),String>{op.apply(x,y)?;for(a,q)in y.iter_mut().zip(x){a.re-=sigma*q.re;a.im-=sigma*q.im;}Ok(())};
            let(exact,_)={let mut pre=|x:&[C],y:&mut[C]|fb.apply(x,y);fgmres(&vectors,&b,&mut apply,&mut pre,1e-12,200,2000,None,lapack,false).unwrap()};
            let en=vectors.norm(&exact);
            let part=|e:&[C],centre_only:bool,pops:bool|->f64{let mut f=vec![Z;e.len()];
                for q in 0..e.len()/d{if centre_only&&q!=centre{continue;}for k in 0..d{if pops&&k/n!=k%n{continue;}f[q*d+k]=e[q*d+k];}}vectors.norm(&f)};
            for(name,pre)in[("block",&mut sb as &mut dyn Preconditioner),("floquet",&mut fb as &mut dyn Preconditioner)]{
                for &tol in &tols{
                    let mut prec=|x:&[C],y:&mut[C]|pre.apply(x,y);
                    let(w,st)=fgmres(&vectors,&b,&mut apply,&mut prec,tol,200,2000,None,lapack,false).unwrap();
                    let e:Vec<C>=w.iter().zip(&exact).map(|(a,c)|C{re:a.re-c.re,im:a.im-c.im}).collect();
                    let mut r=vec![Z;w.len()];apply(&w,&mut r).unwrap();for(a,&bb)in r.iter_mut().zip(&b){a.re=bb.re-a.re;a.im=bb.im-a.im;}
                    let mut pr=vec![Z;w.len()];pre.apply(&r,&mut pr).unwrap();
                    table.push(json!({"column":j,"method":name,"tol":tol,"steps":st.steps,"residual":st.residual,"error":vectors.norm(&e)/en,
                        "centre":part(&e,true,false)/en,"centrePops":part(&e,true,true)/en,"allPops":part(&e,false,true)/en,"estimate":vectors.norm(&pr)/en}));
                }
            }
        }
        // Summaries: median and max over columns per (method, tol).
        let mut out=Vec::new();
        for name in["block","floquet"]{for &tol in &tols{
            let rows:Vec<&Value>=table.iter().filter(|r|r["method"]==name&&r["tol"].as_f64()==Some(tol)).collect();
            let stat=|key:&str|{let mut v:Vec<f64>=rows.iter().map(|r|r[key].as_f64().unwrap()).collect();v.sort_by(f64::total_cmp);json!([v[v.len()/2],v[v.len()-1]])};
            out.push(json!({"method":name,"tol":tol,"steps":stat("steps"),"error":stat("error"),"centre":stat("centre"),"centrePops":stat("centrePops"),"allPops":stat("allPops"),"estimate":stat("estimate")}));}}
        emit(&json!({"probe":"errors","case":case.name,"sigma":sigma,"summary":out}));
    }

    /// Complex shifts: plain complex GMRES (all modes, complex inner
    /// product; the operator applied to the two Hermitian-symmetric parts of
    /// a general field) with Floquet/eigen frames at complex sigma
    /// (ATOM_R3_CSIGMAS = re:im list), on the baseline columns.
    #[test]
    #[ignore]
    fn r3_complex(){
        let case=load();let p=&case.prepared;let n=p.generator.n;let d=n*n;
        let order=env_u("ATOM_R3_ORDER",2);let count=env_u("ATOM_R3_COLUMNS",20);let tol=env_f("ATOM_R3_TOL",1e-6);
        let modes=p.modes(order).unwrap();
        let(rhs,_)=columns(&case,&modes,20.,count,1e-6);
        let mut op=LatticeOperator::new(p,&modes).unwrap();let mirror=op.mirror.clone();let len=op.dimension();
        let dagger=|x:&[C]|->Vec<C>{let mut y=vec![Z;x.len()];for q in 0..x.len()/d{let pm=mirror[q];for i in 0..n{for j in 0..n{let v=x[pm*d+j*n+i];y[q*d+i*n+j]=C{re:v.re,im:-v.im};}}}y};
        let dot=|a:&[C],b:&[C]|->C{let mut s=C{re:0.,im:0.};for(x,y)in a.iter().zip(b){s.re+=x.re*y.re+x.im*y.im;s.im+=x.re*y.im-x.im*y.re;}s};
        let nrm=|a:&[C]|->f64{a.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()};
        let sigmas:Vec<C>=std::env::var("ATOM_R3_CSIGMAS").unwrap_or("20:0,20:13,2:13,0.5:13".into()).split(',').map(|t|{let(a,b)=t.split_once(':').unwrap();C{re:a.parse().unwrap(),im:b.parse().unwrap()}}).collect();
        let methods=std::env::var("ATOM_R3_METHODS").unwrap_or("frames,frames+cw,fk:0.005+cw".into());
        for method in methods.split(','){
            let cw=method.ends_with("+cw");let base=method.trim_end_matches("+cw");
            let frames=if base=="frames"{FloquetFrames::by_kernel(p,1e9,0.5).unwrap()}else{FloquetFrames::by_kernel(p,base.strip_prefix("fk:").unwrap().parse().unwrap(),0.5).unwrap()};
            for &sigma in &sigmas{
                let mut pre=frames.batch_complex(&op,sigma,cw).unwrap();let pre_work=pre.work;
                let mut steps=Vec::new();let clock=Instant::now();
                for v in &rhs{
                    let b:Vec<C>=v.iter().map(|z|C{re:-z.re,im:-z.im}).collect();let bn=nrm(&b);
                    let mut apply=|x:&[C]|->Vec<C>{
                        let xd=dagger(x);
                        let a:Vec<C>=x.iter().zip(&xd).map(|(u,w)|C{re:0.5*(u.re+w.re),im:0.5*(u.im+w.im)}).collect();
                        // b = (x - x^dagger) / (2i)
                        let bb:Vec<C>=x.iter().zip(&xd).map(|(u,w)|{let t=C{re:0.5*(u.re-w.re),im:0.5*(u.im-w.im)};C{re:t.im,im:-t.re}}).collect();
                        let(mut ya,mut yb)=(vec![Z;len],vec![Z;len]);op.apply(&a,&mut ya).unwrap();op.apply(&bb,&mut yb).unwrap();
                        ya.iter().zip(&yb).zip(x).map(|((p1,q1),xx)|C{re:p1.re-q1.im-(sigma.re*xx.re-sigma.im*xx.im),im:p1.im+q1.re-(sigma.re*xx.im+sigma.im*xx.re)}).collect()};
                    // Complex GMRES, right preconditioned, CGS2, Givens.
                    let mut vs:Vec<Vec<C>>=vec![b.iter().map(|z|C{re:z.re/bn,im:z.im/bn}).collect()];let mut zs:Vec<Vec<C>>=Vec::new();
                    let mut hcols:Vec<Vec<C>>=Vec::new();let mut rot:Vec<(C,C)>=Vec::new();let mut g=vec![C{re:bn,im:0.}];let mut k=0;
                    loop{
                        let mut z=vec![Z;len];pre.apply(&vs[k],&mut z).unwrap();let mut w=apply(&z);zs.push(z);
                        let mut h=vec![Z;k+1];
                        for _ in 0..2{for(i,vi)in vs.iter().enumerate(){let c=dot(vi,&w);h[i].re+=c.re;h[i].im+=c.im;for(t,x)in w.iter_mut().zip(vi){t.re-=c.re*x.re-c.im*x.im;t.im-=c.re*x.im+c.im*x.re;}}}
                        let hn=nrm(&w);h.push(C{re:hn,im:0.});
                        for(i,&(c,s))in rot.iter().enumerate(){let(a1,a2)=(h[i],h[i+1]);
                            // [c s; -conj(s) c] with c real stored in c.re
                            h[i]=C{re:c.re*a1.re+s.re*a2.re-s.im*a2.im,im:c.re*a1.im+s.re*a2.im+s.im*a2.re};
                            h[i+1]=C{re:c.re*a2.re-(s.re*a1.re+s.im*a1.im),im:c.re*a2.im-(s.re*a1.im-s.im*a1.re)};}
                        let(a1,a2)=(h[k],h[k+1]);let r=(a1.re*a1.re+a1.im*a1.im+a2.re*a2.re+a2.im*a2.im).sqrt();
                        let m1=a1.re.hypot(a1.im);let(c,s)=if m1==0.{(0.,C{re:1.,im:0.})}else{let ph=C{re:a1.re/m1,im:a1.im/m1};
                            // s = ph * conj(a2) / r
                            (m1/r,C{re:(ph.re*a2.re+ph.im*a2.im)/r,im:(ph.im*a2.re-ph.re*a2.im)/r})};
                        rot.push((C{re:c,im:0.},s));h[k]=C{re:c*a1.re+s.re*a2.re-s.im*a2.im,im:c*a1.im+s.re*a2.im+s.im*a2.re};h[k+1]=Z;
                        let gk=g[k];g.push(C{re:-(s.re*gk.re+s.im*gk.im),im:-(s.re*gk.im-s.im*gk.re)});g[k]=C{re:c*gk.re,im:c*gk.im};
                        hcols.push(h);vs.push(w.iter().map(|q|C{re:q.re/hn,im:q.im/hn}).collect());k+=1;
                        if g[k].re.hypot(g[k].im)/bn<=0.5*tol||k>=300{break;}
                    }
                    steps.push(k);
                }
                let total:usize=steps.iter().sum();
                emit(&json!({"probe":"complex","case":case.name,"method":method,"sigma":[sigma.re,sigma.im],"perColumn":total as f64/steps.len() as f64,"steps":steps,
                    "preWorkFull":pre_work,"opWork":op.work_parts(),"ms":clock.elapsed().as_secs_f64()*1e3}));
            }
        }
    }
}
