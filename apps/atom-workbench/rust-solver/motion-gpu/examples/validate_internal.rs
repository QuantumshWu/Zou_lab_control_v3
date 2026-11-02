//! The jumps against the exact master equation: the tweezer PGC with the atom held still (a huge
//! mass), so each grid point's internal density evolves by its own local Lindblad equation. The
//! trajectories' populations, averaged with their thermal weights, against the exact solution
//! weighted by the initial |ψ(x)|² = e^{−βU(x)}/Z.
//! `cargo run --profile iterate --example validate_internal`
#[path="scene/mod.rs"] mod scene;
use atom_motion_core::linear::Dense;
use atom_motion_core::{Complex,STATES};
use atom_motion_gpu::engine::{Engine,Internal,Settings};
use atom_motion_gpu::gpu::Gpu;
use atom_motion_gpu::model::{propagator,Grid,Model,BOLTZMANN,HBAR};

fn main(){
    let mut request=scene::turned(scene::load(scene::PGC));
    request.mass_kg=1e3;
    let mut model=Model::new(&request).unwrap();
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:f64|args.get(i).map_or(d,|s|s.parse().unwrap());
    let grid=Grid{nx:16,ny:16,dx:0.05};let dt=arg(1,0.25);let b=arg(2,4096.) as usize;let seed=arg(3,7.) as u64;
    model.tabulate(grid.reach()).unwrap();let g=&model.ground;
    let settings=Settings{grid,dt,doppler_every:4,trajectories:b,seed,cap:None};
    let start=std::time::Instant::now();
    let mut engine=Engine::new(std::rc::Rc::new(Gpu::new().unwrap()),&model,settings).unwrap();
    println!("tables {:.2} s",start.elapsed().as_secs_f64());
    // |2,+2⟩: block 1 (F = 2), its component of ground state "ground-2-2".
    let index=request.states.iter().position(|s|s=="ground-2-2").unwrap();
    let b22=g.ground_states.iter().position(|&i|i==index).unwrap();
    let c22=model.block_states[1].iter().position(|&s|s==b22).unwrap();
    let mut amplitudes=vec![Complex::ZERO;5];amplitudes[c22]=Complex::ONE;
    engine.initialize(&vec![Internal{block:1,amplitudes};b],30.,&model).unwrap();
    let interval=(20./dt).round() as usize;let outputs=10;
    let mut mcwf:Vec<(Vec<f64>,Vec<f64>)>=vec![];
    let collect=|engine:&Engine|->(Vec<f64>,Vec<f64>){
        let(records,components)=engine.read_records().unwrap();
        let s=atom_motion_gpu::stats::sample(&model,&records,&components,&engine.trajectory_weights);(s.populations,s.population_errors)
    };
    engine.observe_initial(false).unwrap();mcwf.push(collect(&engine));
    let run=std::time::Instant::now();
    for _ in 0..outputs{for s in 0..interval{engine.step(s+1==interval,false).unwrap();}mcwf.push(collect(&engine));}
    engine.gpu.stream.synchronize().unwrap();
    println!("{} steps of {b} trajectories on {}x{}: {:.2} s",outputs*interval,grid.nx,grid.ny,run.elapsed().as_secs_f64());
    // The exact local master equations, weighted by the initial e^{−βU}.
    let ng=g.ground_states.len();
    let beta=HBAR/(BOLTZMANN*30e-6)*1e6;
    let u:Vec<f64>=(0..grid.points()).map(|p|{let r=grid.position(p);g.eliminate(&g.hamiltonian_at(r)).unwrap().energies[b22]}).collect();
    let low=u.iter().copied().fold(f64::INFINITY,f64::min);let weights:Vec<f64>=u.iter().map(|x|(-beta*(x-low)).exp()).collect();let z:f64=weights.iter().sum();
    let mut exact=vec![vec![0f64;STATES];outputs+1];
    for p in 0..grid.points(){
        if weights[p]/z<1e-12{continue;}
        let r=grid.position(p);let e=g.eliminate(&g.hamiltonian_at(r)).unwrap();let local=g.local_at(&e,r);
        let point=model.point(r,0.);
        // H_eff on the 8 ground states from the blocks' generators.
        let mut h=Dense::zeros(ng,ng);for k in 0..2{let s=&model.block_states[k];for i in 0..s.len(){for j in 0..s.len(){h[(s[i],s[j])]=point.generator[k][(i,j)];}}}
        // The Lindblad superoperator on vec(ρ) (row-major ρ[i][j] at i ng + j), as −i·(i L) for `propagator`.
        let n2=ng*ng;let mut l=Dense::zeros(n2,n2);
        for i in 0..ng{for j in 0..ng{for k in 0..ng{
            l[(i*ng+j,k*ng+j)]+=Complex::new(0.,-1.)*h[(i,k)];
            l[(i*ng+j,i*ng+k)]+=Complex::new(0.,1.)*h[(j,k)].conj();
        }}}
        for jump in &local.jumps{for i in 0..ng{for j in 0..ng{for k in 0..ng{for m in 0..ng{
            l[(i*ng+j,k*ng+m)]+=jump[(i,k)]*jump[(j,m)].conj();
        }}}}}
        let step=propagator(&l.scaled(Complex::I),interval as f64*dt);
        let mut rho=vec![Complex::ZERO;n2];rho[b22*ng+b22]=Complex::ONE;
        for o in 0..=outputs{
            if o>0{rho=(0..n2).map(|i|(0..n2).fold(Complex::ZERO,|a,k|a+step[(i,k)]*rho[k])).collect();}
            let pops=g.populations(&local,&Dense{rows:ng,cols:ng,entries:rho.clone()});
            for i in 0..STATES{exact[o][i]+=weights[p]/z*pops[i];}
        }
    }
    let mut worst=0f64;let mut chi2=0f64;let mut count=0;
    let mut offenders:Vec<(f64,usize,usize)>=vec![];
    for o in 1..=outputs{let(m,s)=&mcwf[o];for i in 0..STATES{if s[i]>0.{offenders.push((((m[i]-exact[o][i])/s[i]).abs(),o,i));}}}
    offenders.sort_by(|a,b|b.0.total_cmp(&a.0));
    for &(z,o,i)in offenders.iter().take(8){println!("  {} at output {o}: {:.3e} vs {:.3e} (SE {:.1e}, {z:.1} SE)",request.states[i],mcwf[o].0[i],exact[o][i],mcwf[o].1[i]);}
    let excited:Vec<String>=(0..=outputs).map(|o|{let e:f64=g.excited_states.iter().map(|&i|mcwf[o].0[i]).sum();let x:f64=g.excited_states.iter().map(|&i|exact[o][i]).sum();format!("{e:.5}/{x:.5}")}).collect();
    println!("total excited (mcwf/exact): {}",excited.join(" "));
    for o in 0..=outputs{
        let(m,s)=&mcwf[o];
        let line:Vec<String>=(0..STATES).filter(|&i|exact[o][i]>2e-3||m[i]>2e-3).map(|i|format!("{}:{:.4}/{:.4}±{:.4}",request.states[i],m[i],exact[o][i],s[i])).collect();
        println!("t={:5.0} us  {}",o as f64*interval as f64*dt,line.join("  "));
        if o>0{for i in 0..STATES{if s[i]>0.{let z=(m[i]-exact[o][i])/s[i];worst=worst.max(z.abs());chi2+=z*z;count+=1;}}}
    }
    println!("largest |deviation|/SE {worst:.2}, chi2/n {:.2} over {count}",chi2/count as f64);
}
