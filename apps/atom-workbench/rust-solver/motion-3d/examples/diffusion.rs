//! The momentum's diffusion of atoms held at a point or moved along a line at a fixed velocity
//! (engine.rs `Settings::path`), against the second-order expansion's exact moments from the front
//! end's dense generator (ground.rs `local_with`, the paths the tables hold; its gradients by
//! central differences) -- independent of the tables and of the engine's superoperators.
//!
//! Every atom starts at r0 with κ = 0 and its internal state from the scene's initial populations;
//! the kinetic shift is off (the expansion's generator has none). The exact moments: the clusters'
//! densities W0, the first moments W_i (dW_i/dt = ℒW_i + 𝒜_i(W0) − tr 𝒜_i(W0) W0) and
//! dVar(κ_i)/dt = 2 tr 𝒜_i(W_i) + 𝓑_ii(W0) + the emission's k² Σ_x w_xi tr(L_x†L_x W0)
//! (w = 1/5 along the dipole, 2/5 across), by RK4. The engine: its ensemble's Var(κ_i) at the same
//! times. Both as T = (ħ²/m k_B) Var(κ) and as the slope D = ½ dVar/dt over the run's second half.
//! `cargo run --profile iterate -p atom-motion-3d --example diffusion -- <scene> <x0 y0 z0 (um)> [vx vy vz (um/us)] [trajectories] [t_end_us] [interval_us] [dt_us] [axes, e.g. xyz]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::ground::{Local,Path};
use atom_motion_3d::linear::Dense;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales,KT_PER_UK};
use atom_motion_3d::{simulate_from,Complex,Initial,STATES};
use std::time::Instant;

/// The dense generator of every cluster at a point and time, and its gradients along the asked
/// axes: per cluster H_eff, ∂_iH, ∂_iΓ, per class L_x and ∂_iL_x.
#[derive(Clone)]
struct Dense3{effective:Vec<Dense>,force:Vec<[Dense;3]>,decay:Vec<[Dense;3]>,jumps:Vec<Vec<[Dense;3]>>,gradients:Vec<Vec<[[Dense;3];3]>>}

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"PGC3D");let f=|i:usize,d:&str|arg(i,d).parse::<f64>().unwrap();
    let r0=[f(2,"0"),f(3,"0"),f(4,"0")];let v=[f(5,"0"),f(6,"0"),f(7,"0")];
    let count:u64=arg(8,"2048").parse().unwrap();let t_end=f(9,"100");let interval=f(10,"2");let dt=f(11,"0.2");
    let axes:Vec<usize>=arg(12,"xyz").chars().map(|c|match c{'x'=>0,'y'=>1,'z'=>2,_=>panic!("axes are x, y, z")}).collect();
    let request=scene::scene(&name);
    let mut model=Model::new(&request).unwrap();model.doppler=false;let g=&model.ground;
    let tabulated=|p:&Path|model.carriers[p.via.carrier].beams.len()==1;
    let at=|t:f64|->[f64;3]{std::array::from_fn(|a|r0[a]+v[a]*t)};
    println!("{name}: clusters {:?}, from {r0:?} um at {v:?} um/us, axes {axes:?}",g.clusters.iter().map(|c|c.states.len()).collect::<Vec<_>>());
    // The dense generator at (r, t) and its gradients by central differences over h.
    let h=1e-4;
    let generator=|r:[f64;3],t:f64|->Vec<Local>{let e=g.eliminate(&model.hamiltonian_at_point(r)).unwrap();(0..g.clusters.len()).map(|k|g.local_with(&e,k,r,t,&tabulated).unwrap()).collect()};
    let dense=|r:[f64;3],t:f64|->Dense3{
        let here=generator(r,t);
        let mut out=Dense3{effective:here.iter().map(|l|l.effective.clone()).collect(),force:vec![],decay:vec![],jumps:here.iter().map(|l|l.jumps.clone()).collect(),gradients:vec![]};
        let(mut plus,mut minus)=(vec![],vec![]);
        for i in 0..3{if axes.contains(&i){let(mut a,mut b)=(r,r);a[i]+=h;b[i]-=h;plus.push(Some(generator(a,t)));minus.push(Some(generator(b,t)));}else{plus.push(None);minus.push(None);}}
        let zero=|m:&Dense|Dense::zeros(m.rows,m.cols);
        for(k,l)in here.iter().enumerate(){
            let n=l.effective.rows;
            out.force.push(std::array::from_fn(|i|match(&plus[i],&minus[i]){(Some(p),Some(m))=>{
                // F_i = −∂_i H, H = (H_eff + H_eff†)/2.
                let mut d=p[k].effective.clone();d.add_scaled(Complex::new(-1.,0.),&m[k].effective);let d=d.scaled(Complex::new(0.5/h,0.));
                Dense::from_fn(n,n,|a,b|-(d[(a,b)]+d[(b,a)].conj())*0.5)},_=>Dense::zeros(n,n)}));
            out.decay.push(std::array::from_fn(|i|match(&plus[i],&minus[i]){(Some(p),Some(m))=>{
                // ∂_iΓ, Γ = i(H_eff − H_eff†).
                let mut d=p[k].effective.clone();d.add_scaled(Complex::new(-1.,0.),&m[k].effective);let d=d.scaled(Complex::new(0.5/h,0.));
                Dense::from_fn(n,n,|a,b|(d[(a,b)]-d[(b,a)].conj())*Complex::new(0.,1.))},_=>Dense::zeros(n,n)}));
            out.gradients.push(l.jumps.iter().enumerate().map(|(c,j)|std::array::from_fn(|i|std::array::from_fn(|x|match(&plus[i],&minus[i]){(Some(p),Some(m))=>{
                let mut d=p[k].jumps[c][x].clone();d.add_scaled(Complex::new(-1.,0.),&m[k].jumps[c][x]);d.scaled(Complex::new(0.5/h,0.))},_=>zero(&j[x])}))).collect());
        }
        out
    };
    let targets:Vec<Vec<usize>>=g.clusters.iter().map(|c|c.classes.iter().map(|x|x.target).collect()).collect();
    let waves:Vec<Vec<f64>>=g.clusters.iter().map(|c|c.classes.iter().map(|x|x.wave_number).collect()).collect();
    let add=|a:&mut Dense,z:Complex,b:&Dense|a.add_scaled(z,b);
    // ℒY and 𝒜_i(Y) on the clusters' blocks.
    let lindblad=|d:&Dense3,y:&[Dense]|->Vec<Dense>{
        let mut out:Vec<Dense>=y.iter().zip(&d.effective).map(|(y,k)|{let mut z=k.product(y).scaled(Complex::new(0.,-1.));z.add_scaled(Complex::new(0.,1.),&y.product(&k.adjoint()));z}).collect();
        for(k,jumps)in d.jumps.iter().enumerate(){for(c,j)in jumps.iter().enumerate(){for x in 0..3{let t=targets[k][c];let m=j[x].product(&y[k]).product(&j[x].adjoint());add(&mut out[t],Complex::ONE,&m);}}}
        out
    };
    let drive=|d:&Dense3,y:&[Dense],i:usize|->Vec<Dense>{
        let mut out:Vec<Dense>=y.iter().enumerate().map(|(k,y)|{
            let(f,gm)=(&d.force[k][i],&d.decay[k][i]);
            let mut z=f.product(y);z.add_scaled(Complex::ONE,&y.product(f));let mut z=z.scaled(Complex::new(0.5,0.));
            z.add_scaled(Complex::new(0.,0.25),&gm.product(y));z.add_scaled(Complex::new(0.,-0.25),&y.product(gm));z}).collect();
        for(k,jumps)in d.jumps.iter().enumerate(){for(c,j)in jumps.iter().enumerate(){for x in 0..3{
            let(l,dl)=(&j[x],&d.gradients[k][c][i][x]);let t=targets[k][c];
            let a=dl.product(&y[k]).product(&l.adjoint());let b=l.product(&y[k]).product(&dl.adjoint());
            add(&mut out[t],Complex::new(0.,-0.5),&a);add(&mut out[t],Complex::new(0.,0.5),&b);
        }}}
        out
    };
    let trace=|y:&[Dense]|->f64{y.iter().map(|m|(0..m.rows).map(|a|m[(a,a)].re).sum::<f64>()).sum()};
    let white=|d:&Dense3,y:&[Dense],i:usize|->f64{
        let mut s=0.;
        for(k,jumps)in d.jumps.iter().enumerate(){for(c,j)in jumps.iter().enumerate(){for x in 0..3{
            let dl=&d.gradients[k][c][i][x];let a=dl.adjoint().product(dl).product(&y[k]);s+=(0..a.rows).map(|p|a[(p,p)].re).sum::<f64>();
            let b=j[x].adjoint().product(&j[x]).product(&y[k]);let w=if x==i{0.2}else{0.4};
            s+=waves[k][c]*waves[k][c]*w*(0..b.rows).map(|p|b[(p,p)].re).sum::<f64>();
        }}}
        s
    };
    // The exact moments by RK4, the generator at each stage's time and place (once when static).
    let moving=v.iter().any(|&x|x!=0.);
    let fixed=g.clusters.iter().all(|c|c.groups.iter().all(|x|x.entries.iter().all(|e|e.2==0.))&&c.paths.iter().all(|p|p.residual==0.)&&c.classes.iter().all(|k|k.photons.iter().all(|p|p.1==0.)));
    let clock=Instant::now();let frozen=dense(r0,0.);
    let state=|t:f64|->std::borrow::Cow<Dense3>{if moving||!fixed{std::borrow::Cow::Owned(dense(at(t),t))}else{std::borrow::Cow::Borrowed(&frozen)}};
    let w0:Vec<Dense>=g.clusters.iter().map(|c|{let n=c.states.len();Dense::from_fn(n,n,|i,j|if i==j{let s=g.ground_states[c.states[i]];request.initial_density[s*STATES+s]}else{Complex::ZERO})}).collect();
    // (W0, W_i per asked axis, Var_i) as one state.
    type State=(Vec<Dense>,Vec<Vec<Dense>>,Vec<f64>);
    let derivative=|d:&Dense3,s:&State|->State{
        let d0=lindblad(d,&s.0);
        let mut d1=vec![];let mut dv=vec![];
        for(q,&i)in axes.iter().enumerate(){
            let a0=drive(d,&s.0,i);let f=trace(&a0);
            let mut x=lindblad(d,&s.1[q]);for(xk,(ak,wk))in x.iter_mut().zip(a0.iter().zip(&s.0)){xk.add_scaled(Complex::ONE,ak);xk.add_scaled(Complex::new(-f,0.),wk);}
            d1.push(x);dv.push(2.*trace(&drive(d,&s.1[q],i))+white(d,&s.0,i));
        }
        (d0,d1,dv)
    };
    let combine=|s:&State,k:&State,c:f64|->State{
        let m=|a:&[Dense],b:&[Dense]|->Vec<Dense>{a.iter().zip(b).map(|(x,y)|{let mut z=x.clone();z.add_scaled(Complex::new(c,0.),y);z}).collect()};
        (m(&s.0,&k.0),s.1.iter().zip(&k.1).map(|(a,b)|m(a,b)).collect(),s.2.iter().zip(&k.2).map(|(a,b)|a+c*b).collect())
    };
    let zero=|y:&[Dense]|->Vec<Dense>{y.iter().map(|m|Dense::zeros(m.rows,m.cols)).collect()};
    let mut s:State=(w0.clone(),axes.iter().map(|_|zero(&w0)).collect(),vec![0.;axes.len()]);
    let outputs=(t_end/interval).round() as usize+1;let h=interval/((interval/if moving{0.02}else{0.005}).round().max(1.));let steps=(interval/h).round() as usize;
    let mut exact=vec![s.2.clone()];let mut t=0.;
    for _ in 1..outputs{for _ in 0..steps{
        let(a,b,c)=(state(t),state(t+0.5*h),state(t+h));
        let k1=derivative(&a,&s);let k2=derivative(&b,&combine(&s,&k1,0.5*h));let k3=derivative(&b,&combine(&s,&k2,0.5*h));let k4=derivative(&c,&combine(&s,&k3,h));
        for(k,w)in [(&k1,h/6.),(&k2,h/3.),(&k3,h/3.),(&k4,h/6.)]{s=combine(&s,k,w);}
        t+=h;
    }exact.push(s.2.clone());}
    println!("exact moments: {:.1} s; tr W0 at the end {:.6}",clock.elapsed().as_secs_f64(),trace(&s.0));
    // The engine on the same path.
    let initial=Initial::new(&model,&request).unwrap();
    let engine_steps=(interval/dt).round().max(1.) as usize;
    let mut settings=Settings::new(interval/engine_steps as f64,&initial.traps[0],f64::EPSILON,false);
    settings.path=Some(v);settings.loss=false;settings.free=[0,1,2].map(|a|axes.contains(&a));
    let clock=Instant::now();
    let outcome=simulate_from(&model,&settings,&initial,11,0,count,8,engine_steps,outputs,&[],r0);
    let est=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:[1.;3]});
    println!("engine: {count} trajectories, dt {:.3} us, {:.1} s, {} jumps",settings.dt,clock.elapsed().as_secs_f64(),outcome.jumps);
    // The noise's expectation without the open debt: the same trajectories with the covariance
    // kept pending, Var(κ) of their drift and jumps plus the mean covariance owed (the rate's
    // integral), with its standard error over the trajectories.
    let mut quiet=settings.clone();quiet.noise=false;let engine=atom_motion_3d::engine::Engine{model:&model,settings:quiet.clone()};
    let expected:Vec<Vec<(f64,f64)>>={use rayon::prelude::*;
        let per:Vec<Vec<[f64;6]>>=(0..count).into_par_iter().map(|index|{
            let mut t=atom_motion_3d::initial_trajectory(&model,&quiet,&initial,11,index,atom_motion_3d::Stratum{first:index-index%8,size:8});t.r=r0;t.kappa=[0.;3];
            let mut out=vec![[0.;6];outputs];
            for i in 1..outputs{for _ in 0..engine_steps{engine.step(&mut t);}out[i]=[t.kappa[0],t.kappa[1],t.kappa[2],t.pending[0][0],t.pending[1][1],t.pending[2][2]];}
            out}).collect();
        let n=count as f64;
        (0..outputs).map(|i|(0..3).map(|a|{
            let(m,q,p)=per.iter().fold((0.,0.,0.),|(m,q,p),o|(m+o[i][a],q+o[i][a]*o[i][a],p+o[i][3+a]));let(m,q,p)=(m/n,q/n,p/n);
            let x:Vec<f64>=per.iter().map(|o|(o[i][a]-m).powi(2)+o[i][3+a]).collect();let mean=x.iter().sum::<f64>()/n;
            let se=(x.iter().map(|v|(v-mean).powi(2)).sum::<f64>()/(n*(n-1.))).sqrt();
            (q-m*m+p,se)}).collect()).collect()};
    // T of a variance of κ (1/um²): (ħ/m) Var(κ) / (k_B/ħ per μK).
    let temperature=|var:f64|var*model.hbar_over_mass/KT_PER_UK;
    let variance=|temp:f64|temp*KT_PER_UK/model.hbar_over_mass;
    for i in (0..outputs).step_by((outputs/10).max(1)).chain([outputs-1]){
        let row:Vec<String>=axes.iter().enumerate().map(|(q,&a)|{let(e,se,x)=(est[i].temperature[a],est[i].temperature_error[a],temperature(exact[i][q]));
            let(m,ms)=expected[i][a];let(m,ms)=(temperature(m),temperature(ms));
            format!("{}: {:8.2}±{:5.2} expected {:8.2}±{:5.2} exact {:8.2} ({:+5.1}%, {:+5.1}%)",['x','y','z'][a],e,se,m,ms,x,if x>0.{100.*(e/x-1.)}else{0.},if x>0.{100.*(m/x-1.)}else{0.})}).collect();
        println!("t {:6.1}  {}",i as f64*interval,row.join("  "));
    }
    // The slopes over the run's second half: D = ½ dVar/dt (κ²/us), a least-squares line.
    let half=outputs/2;let times:Vec<f64>=(half..outputs).map(|i|i as f64*interval).collect();
    let slope=|y:&[f64]|{let n=y.len() as f64;let(mt,my)=(times.iter().sum::<f64>()/n,y.iter().sum::<f64>()/n);
        let(c,v)=times.iter().zip(y).fold((0.,0.),|(c,v),(t,y)|(c+(t-mt)*(y-my),v+(t-mt)*(t-mt)));c/v};
    for(q,&a)in axes.iter().enumerate(){
        let e:Vec<f64>=(half..outputs).map(|i|variance(est[i].temperature[a])).collect();let x:Vec<f64>=(half..outputs).map(|i|exact[i][q]).collect();
        let(de,dx)=(0.5*slope(&e),0.5*slope(&x));
        println!("{}: D engine {de:.2} exact {dx:.2} κ²/us (ratio {:.3}); Var at the end {:.4e} against {:.4e} (±{:.1e})",['x','y','z'][a],de/dx,
            variance(est[outputs-1].temperature[a]),exact[outputs-1][q],variance(est[outputs-1].temperature_error[a]));
    }
}
