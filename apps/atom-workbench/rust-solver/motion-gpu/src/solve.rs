//! The 2D solve: batches of trajectories through every output time until the standard errors
//! meet their targets -- as QuTiP's mcsolve with a target tolerance: the populations' and the
//! population groups' errors at most ε absolute, the temperatures' at most
//! `TEMPERATURE_TOLERANCE` relative -- each batch's merged estimates published as a checkpoint.
//! The first batch is the fewest trajectories; each next one is sized from the last one's wall
//! time so that it takes about `BATCH_SECONDS` (a batch's cost per trajectory falls with its size,
//! so the estimate from a smaller batch is an upper bound), and at most to the count the errors
//! so far project the targets to need.
use crate::engine::{Cap,Internal,Settings,REC_EXCITED,REC_GROUND,REC_KX,REC_TAIL,REC_X};
use crate::ensemble::{ladder,Ensemble,Place};
use crate::gpu::Gpu;
use crate::model::{Model,BOLTZMANN,HBAR};
use crate::solution::{Distributions,Map,Running,Solution};
use atom_motion_core::linear::Dense;
use atom_motion_core::{Complex,Request,STATES};
use std::rc::Rc;

/// The temperatures' relative standard error the run aims at.
pub const TEMPERATURE_TOLERANCE:f64=0.1;
/// The step's largest size (us): the splitting's error in T is about 1.5 % at it (against the 1D
/// reference over 2 ms: −5 % at 0.8 us, within the 2 % statistics at 0.4 and 0.2 us).
const LARGEST_STEP:f64=0.4;
/// The most trajectories per batch and per run, and the fewest per batch (the kernels' unit).
const BATCH:usize=512;
const MOST_TRAJECTORIES:usize=8192;
const FEWEST:usize=8;
/// A batch's planned duration (s): checkpoints come about this often while the cost holds (the
/// app ends a solve whose checkpoints stop for two minutes).
const BATCH_SECONDS:f64=30.;
/// How often a running solve says so between its checkpoints (s): well within the app's
/// two-minute deadline, however long a batch runs.
pub const BEAT_SECONDS:f64=10.;
/// The distributions' sample times.
const DISTRIBUTION_SAMPLES:usize=11;
/// The absorbing boundary's rate (1/us): an atom crossing it at the escape speed is absorbed
/// within a fraction of the band (the 1D study's value).
const CAP_RATE:f64=20.;
/// Why a request has no 2D motion to solve.
pub const NO_TRAP:&str="运动平面内没有约束原子的势阱：2D 径向运动计算光镊中单个原子在垂直于光镊的平面内的运动";

/// The sums one output time's estimates are made from, over every batch's held trajectories:
/// Σw, and per quantity Σw v, Σw² v², Σw² v; Σw² and per axis Σw² κ κ²; the lost weight.
#[derive(Clone)]
struct Sums{w:f64,w2:f64,wv:Vec<f64>,w2v2:Vec<f64>,w2v:Vec<f64>,w2k12:[f64;2],lost:f64,total:f64,lost2:Vec<(f64,bool)>}
/// The quantities per trajectory: 32 populations, the population groups (`groups`), then per axis
/// κ, κ², x².
const GROUPS:usize=9;
const POPULATION_GROUPS:usize=STATES+GROUPS;
const Q_KAPPA:usize=POPULATION_GROUPS;
const Q_KAPPA2:usize=Q_KAPPA+2;
const Q_X2:usize=Q_KAPPA2+2;
const QUANTITIES:usize=Q_X2+2;
impl Sums{
    fn new()->Sums{Sums{w:0.,w2:0.,wv:vec![0.;QUANTITIES],w2v2:vec![0.;QUANTITIES],w2v:vec![0.;QUANTITIES],w2k12:[0.;2],lost:0.,total:0.,lost2:vec![]}}
    fn add(&mut self,w:f64,v:&[f64]){
        self.w+=w;self.w2+=w*w;self.total+=w;
        for q in 0..QUANTITIES{self.wv[q]+=w*v[q];self.w2v2[q]+=w*w*v[q]*v[q];self.w2v[q]+=w*w*v[q];}
        for d in 0..2{self.w2k12[d]+=w*w*v[Q_KAPPA+d]*v[Q_KAPPA2+d];}
        self.lost2.push((w,false));
    }
    fn add_lost(&mut self,w:f64){self.lost+=w;self.total+=w;self.lost2.push((w,true));}
    fn mean(&self,q:usize)->f64{self.wv[q]/self.w}
    /// The weighted mean's standard error: Σ w²(v − m)² / (Σw)².
    fn error(&self,q:usize)->f64{let m=self.mean(q);((self.w2v2[q]-2.*m*self.w2v[q]+m*m*self.w2).max(0.)).sqrt()/self.w}
}

/// The population groups the app reports with the states (src/population-selection.ts
/// POPULATION_GROUPS): every hyperfine manifold of the ground state and the D lines, and all the
/// excited states together.
fn groups(states:&[String])->Result<Vec<Vec<usize>>,String>{
    let mut named:std::collections::BTreeMap<String,Vec<usize>>=Default::default();let mut excited=vec![];
    for(i,label)in states.iter().enumerate(){
        let mut f=label.splitn(3,'-');let(m,level)=(f.next().unwrap_or(""),f.next().unwrap_or(""));
        named.entry(format!("{m}:{level}")).or_default().push(i);if m!="ground"{excited.push(i);}
    }
    let mut out:Vec<Vec<usize>>=named.into_values().collect();out.push(excited);
    if out.len()!=GROUPS{return Err("2D 运动需要完整的 Rb87 32 态模型".into());}
    Ok(out)
}
/// A trajectory's quantities from its record.
fn quantities(model:&Model,groups:&[Vec<usize>],r:&[f32])->Vec<f64>{
    let g=&model.ground;let mut v=vec![0.;QUANTITIES];
    for(k,&b)in model.block_states[0].iter().chain(&model.block_states[1]).enumerate(){v[g.ground_states[b]]=r[REC_GROUND+k] as f64;}
    for(k,&state)in g.excited_states.iter().enumerate(){v[state]=r[REC_EXCITED+k] as f64;}
    for(k,members)in groups.iter().enumerate(){v[STATES+k]=members.iter().map(|&i|v[i]).sum();}
    for d in 0..2{v[Q_KAPPA+d]=r[REC_KX+d] as f64;v[Q_KAPPA2+d]=r[REC_KX+2+d] as f64;v[Q_X2+d]=r[REC_X+2+d] as f64;}
    v
}

/// The initial internal states: the ground blocks of the initial density, each diagonalized, as
/// (block, amplitudes, probability). Excited population is outside the eliminated model.
fn initial_states(model:&Model,request:&Request)->Result<Vec<(usize,Vec<Complex>,f64)>,String>{
    let g=&model.ground;
    let excited:f64=g.excited_states.iter().map(|&i|request.initial_density[i*STATES+i].re).sum();
    if excited>1e-9{return Err("2D 运动的初态必须在基态（激发态已被绝热消去）".into());}
    let mut out=vec![];
    for k in 0..2{
        let states:Vec<usize>=model.block_states[k].iter().map(|&b|g.ground_states[b]).collect();
        let rho=Dense::from_fn(states.len(),states.len(),|i,j|request.initial_density[states[i]*STATES+states[j]]);
        let(values,vectors)=hermitian_eigen(&rho);
        for(v,vector)in values.into_iter().zip(vectors){if v>1e-12{out.push((k,vector,v));}}
    }
    let total:f64=out.iter().map(|x|x.2).sum();
    if !(total>0.){return Err("2D 运动的初态没有基态布居".into());}
    for x in &mut out{x.2/=total;}
    Ok(out)
}
/// A small Hermitian matrix's eigenvalues and eigenvectors (complex Jacobi rotations).
fn hermitian_eigen(m:&Dense)->(Vec<f64>,Vec<Vec<Complex>>){
    let n=m.rows;let mut a=m.clone();let mut v=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});
    for _ in 0..100{
        let off:f64=(0..n).flat_map(|i|(0..n).filter(move|&j|j!=i).map(move|j|(i,j))).map(|(i,j)|a[(i,j)].norm_sqr()).sum();
        if off<1e-30{break;}
        for p in 0..n{for q in p+1..n{
            let apq=a[(p,q)];let r=apq.abs();if r<1e-300{continue;}
            let phase=Complex::new(apq.re/r,apq.im/r);
            let theta=0.5*(2.*r).atan2(a[(q,q)].re-a[(p,p)].re);let(s,c)=theta.sin_cos();
            // Columns p, q of a and v: x_p' = c x_p − s phase* x_q... as a unitary rotation G.
            for k in 0..n{let(x,y)=(a[(k,p)],a[(k,q)]);a[(k,p)]=x*c-y*phase.conj()*s;a[(k,q)]=x*phase*s+y*c;}
            for k in 0..n{let(x,y)=(a[(p,k)],a[(q,k)]);a[(p,k)]=x*c-y*phase*s;a[(q,k)]=x*phase.conj()*s+y*c;}
            for k in 0..n{let(x,y)=(v[(k,p)],v[(k,q)]);v[(k,p)]=x*c-y*phase.conj()*s;v[(k,q)]=x*phase*s+y*c;}
        }}
    }
    ((0..n).map(|i|a[(i,i)].re).collect(),(0..n).map(|j|(0..n).map(|i|v[(i,j)]).collect()).collect())
}

/// The radius (um) where state b's potential has risen half the trap's depth.
fn half_depth(model:&Model,b:usize,reach:f64)->f64{
    let g=&model.ground;let energy=|rho:f64|g.eliminate(&g.hamiltonian_at([rho,0.,0.])).map(|e|e.energies[b]).unwrap_or(f64::NAN);
    let waist=g.beams.iter().filter_map(|x|x.waist_um).fold(0.,f64::max);
    let(bottom,top)=(energy(0.),energy(10.*waist.max(reach)));let target=0.5*(bottom+top);
    let(mut lo,mut hi)=(0.,reach);
    if energy(hi)<target{return reach;}
    for _ in 0..60{let mid=0.5*(lo+hi);if energy(mid)<target{lo=mid;}else{hi=mid;}}
    0.5*(lo+hi)
}

/// The display cells a source cell of width `w` around x overlaps, with the fraction of the source
/// cell in each (display cell i: `to[i]` ± half the display step; beyond the display grid dropped).
fn overlaps(to:&[f64],x:f64,w:f64)->impl Iterator<Item=(usize,f64)>+'_{
    let s=to[1]-to[0];let(lo,hi)=(x-w/2.,x+w/2.);
    let first=((lo-to[0])/s+0.5).floor().max(0.) as usize;let last=(((hi-to[0])/s+0.5).ceil().max(0.) as usize).min(to.len());
    (first..last).filter_map(move|i|{let o=(hi.min(to[i]+s/2.)-lo.max(to[i]-s/2.)).max(0.);(o>0.).then_some((i,o/w))})
}
/// Bins `values` (on the uniform points `from`) into the display grid `to`, each value spread
/// over its cell: the probability is kept whatever the two grids' steps (a grid coarser or finer
/// than the display's leaves no empty or doubled bins).
fn bin(to:&[f64],from:&[f64],values:&[f64],out:&mut [f64]){
    let w=(from[1]-from[0]).abs();
    for(&x,&v)in from.iter().zip(values){for(i,f)in overlaps(to,x,w){out[i]+=v*f;}}
}

pub fn solve(request:&Request,checkpoint:&mut dyn FnMut(&Solution)->bool,running:&mut dyn FnMut(&Running))->Result<Solution,String>{
    let gpu=Rc::new(Gpu::new()?);
    let mut model=Model::new(request)?;
    let initial=initial_states(&model,request)?;
    let g=&model.ground;
    let dominant=initial.iter().max_by(|a,b|a.2.total_cmp(&b.2)).map(|(k,v,_)|{
        let c=v.iter().enumerate().max_by(|a,b|a.1.norm_sqr().total_cmp(&b.1.norm_sqr())).unwrap().0;model.block_states[*k][c]}).unwrap();
    let omega=[model.trap_frequency(dominant,0),model.trap_frequency(dominant,1)];
    if !(omega[0]>0.&&omega[1]>0.){return Err(NO_TRAP.into());}
    let _=g;
    let hm=model.hbar_over_mass;
    let steps_ladder=ladder(0.5*(omega[0]+omega[1]),hm);
    let top=*steps_ladder.last().unwrap();
    model.tabulate(top.reach())?;
    let start=half_depth(&model,dominant,top.half_width()).min(0.85*top.half_width());
    // Times: uniform from 0.
    let times=&request.times_us;
    if times.len()<2||times[0]!=0.{return Err("2D 运动需要从 0 开始的等间隔输出时刻".into());}
    let spacing=times[1]-times[0];
    if times.windows(2).any(|w|((w[1]-w[0])-spacing).abs()>1e-9*spacing.max(1.)){return Err("2D 运动需要等间隔的输出时刻".into());}
    let per=((spacing/LARGEST_STEP).ceil() as usize).max(1);let dt=spacing/per as f64;
    let n_times=times.len();
    let samples:Vec<usize>={let s=DISTRIBUTION_SAMPLES.min(n_times);let mut v:Vec<usize>=(0..s).map(|k|((k as f64)*(n_times-1) as f64/((s-1).max(1)) as f64).round() as usize).collect();v.dedup();v};
    let template=Settings{grid:steps_ladder[0],dt,doppler_every:((1./dt).round() as usize).max(1),trajectories:BATCH,seed:0x5EED_2D,cap:Some(Cap{start,rate:CAP_RATE})};
    let(free,_)=gpu.context.memory()?;
    let tolerance=request.tolerance;
    let groups=groups(&request.states)?;
    let mut ensemble=Ensemble::new(gpu.clone(),template,steps_ladder.clone(),BATCH,free/2,tolerance);
    // The initial grid: the smallest whose half width and largest momentum are five thermal widths.
    let temperature=request.temperature_uk;
    let kt=BOLTZMANN*temperature*1e-6/HBAR*1e-6; // k_B T in rad/us
    let x_t=(kt*hm).sqrt()/omega[0].min(omega[1]);let k_t=(kt/hm).sqrt();
    // (Beyond the top grid's reach the thermal state's tail is above half the trap's depth: the top
    // grid takes it and its absorbing boundary the tail.)
    let mut first=steps_ladder.iter().position(|q|q.half_width()>=5.*x_t&&std::f64::consts::PI/q.dx>=5.*k_t).unwrap_or(steps_ladder.len()-1);
    // Display grids: the first grid's positions and velocities.
    let display=steps_ladder[first];
    let positions:Vec<f64>=(0..display.nx).map(|i|display.x(i)).collect();
    let velocities:Vec<f64>={let mut v:Vec<f64>=(0..display.nx).map(|m|hm*display.wave_number(m,display.nx)).collect();v.sort_by(f64::total_cmp);v};
    let mut sums=vec![Sums::new();n_times];
    let mut vbins:Vec<[Vec<f64>;2]>=samples.iter().map(|_|[vec![0.;velocities.len()],vec![0.;velocities.len()]]).collect();
    let mut pbins:Vec<Vec<f64>>=samples.iter().map(|_|vec![0.;positions.len()*positions.len()]).collect();
    let mut solution=Solution::default();
    let mut run=0usize;let mut census=vec![vec![];n_times];
    let mut size=FEWEST;
    // The last batch's wall time per trajectory (the tables aside; batches grow, so the latest is
    // the next ones' best guide), the trajectories the targets are projected to need, and the last
    // message's time.
    let(mut cost,mut need,mut said)=(0f64,None::<usize>,std::time::Instant::now());
    let(steps,duration)=((n_times-1)*per,times[n_times-1]);
    for batch in 0..{
        if run>=MOST_TRAJECTORIES{break;}
        let batch_size=size.min(MOST_TRAJECTORIES-run).min(ensemble.capacity(first));
        let(internal,weights)=stratified(&initial,batch_size);
        let(begun,built)=(std::time::Instant::now(),ensemble.seconds[3]);
        ensemble.reseed(0x5EED_2D^((batch as u64+1)*0x9E37_79B9_7F4A_7C15))?;
        loop{
            ensemble.start(first,&internal,temperature,&model)?;
            ensemble.scale_weights(&weights)?;
            ensemble.observe_initial(samples.contains(&0))?;
            let records=ensemble.records()?;
            // Tails beyond the start grid: start one higher.
            if records.iter().flatten().any(|r|{let s=|t:usize,m:usize|if r[m]>0.{(r[t]/r[m]) as f64}else{0.};
                (0..2).any(|d|s(REC_TAIL+d,REC_X+2+d)>tolerance||s(REC_TAIL+2+d,REC_KX+2+d)>tolerance)})&&first+1<steps_ladder.len(){first+=1;continue;}
            accumulate(&model,&groups,&ensemble,&records,&mut sums[0]);census[0]=ensemble.census();
            if let Some(i)=samples.iter().position(|&s|s==0){collect(&mut ensemble,&positions,&velocities,&mut vbins[i],&mut pbins[i])?;}
            ensemble.migrate(&records,&model)?;
            break;
        }
        for i in 1..n_times{
            let distributions=samples.contains(&i);
            for s in 0..per{
                ensemble.step(s+1==per,distributions&&s+1==per)?;
                // Between checkpoints, a word at least every BEAT_SECONDS: how far the batch is,
                // and the time left once the errors so far project the trajectories needed.
                if s%64==63&&said.elapsed().as_secs_f64()>=BEAT_SECONDS{
                    let done=((i-1)*per+s+1) as f64/steps as f64;
                    let remaining_ms=need.map(|n|{
                        let elapsed=begun.elapsed().as_secs_f64()-(ensemble.seconds[3]-built);
                        let this=if done>0.05{elapsed/done}else{batch_size as f64*cost};
                        ((1.-done)*this+n.saturating_sub(run+batch_size) as f64*cost).max(0.)*1e3
                    });
                    running(&Running{simulated_us:done*duration,duration_us:duration,trajectories:run,remaining_ms});said=std::time::Instant::now();
                }
            }
            let records=ensemble.records()?;
            accumulate(&model,&groups,&ensemble,&records,&mut sums[i]);census[i]=ensemble.census();
            if let Some(k)=samples.iter().position(|&s|s==i){collect(&mut ensemble,&positions,&velocities,&mut vbins[k],&mut pbins[k])?;}
            ensemble.migrate(&records,&model)?;
        }
        // The next batch: this one's size scaled to BATCH_SECONDS (the grids' tables, built once, aside).
        let wall=begun.elapsed().as_secs_f64()-(ensemble.seconds[3]-built);
        size=((batch_size as f64*BATCH_SECONDS/wall.max(1e-3)) as usize/FEWEST*FEWEST).clamp(FEWEST,BATCH);
        run+=batch_size;cost=wall/batch_size as f64;
        solution=assemble(&model,&sums,omega,&samples,&velocities,&vbins,&positions,&pbins,run,ensemble.refused);
        solution.grids=steps_ladder.iter().map(|q|q.nx).collect();solution.census=census.clone();solution.seconds=ensemble.seconds;
        let(population,temperature_error)=(solution.sample_errors.iter().fold(0.,|a:f64,&b|a.max(b)),
            solution.temperature_uk.iter().zip(&solution.temperature_errors_uk).map(|(t,e)|(0..2).map(|d|if t[d]>0.{e[d]/t[d]}else{0.}).fold(0.,f64::max)).fold(0.,f64::max));
        let met=population<=tolerance&&temperature_error<=TEMPERATURE_TOLERANCE;
        solution.reason=if met{None}else{Some(format!("{run} 条轨迹后：布居的标准误差 {population:.1e}（目标 {tolerance}），温度的相对标准误差 {:.1}%（目标 {:.0}%）",
            100.*temperature_error,100.*TEMPERATURE_TOLERANCE))};
        if met{break;}
        // The trajectories the targets need, the errors falling as 1/√n, with two standard errors
        // of the errors' own estimate on top (relative √(1/(2(n−1))), doubled in the squares): the
        // next batch at most the rest.
        let ratio=(population/tolerance).max(temperature_error/TEMPERATURE_TOLERANCE);
        let projected=((run as f64*ratio*ratio*(1.+2.*(2./(run.max(2)-1) as f64).sqrt())).ceil() as usize).min(MOST_TRAJECTORIES);
        size=size.min(projected.saturating_sub(run).div_ceil(FEWEST)*FEWEST).max(FEWEST);need=Some(projected);
        solution.remaining_ms=Some(projected.saturating_sub(run) as f64*cost*1e3);
        if run<MOST_TRAJECTORIES&&!checkpoint(&solution){break;}
        said=std::time::Instant::now();
    }
    solution.remaining_ms=None;
    Ok(solution)
}

/// Adds the held trajectories' quantities (and the lost weight) of one output time.
fn accumulate(model:&Model,groups:&[Vec<usize>],ensemble:&Ensemble,records:&[Option<Vec<f32>>],sums:&mut Sums){
    for(t,r)in records.iter().enumerate(){
        match(r,ensemble.place[t]){(Some(r),_)=>sums.add(ensemble.weights[t],&quantities(model,groups,r)),(None,Place::Lost)=>sums.add_lost(ensemble.weights[t]),_=>{}}
    }
}
/// `n` trajectories' internal states over the initial density's eigenstates (block, amplitudes,
/// probability): each eigenstate's count its share rounded by the largest remainders, its weight
/// probability × n / count.
fn stratified(initial:&[(usize,Vec<Complex>,f64)],n:usize)->(Vec<Internal>,Vec<f64>){
    let exact:Vec<f64>=initial.iter().map(|x|x.2*n as f64).collect();
    let mut counts:Vec<usize>=exact.iter().map(|&e|e.floor() as usize).collect();
    let mut order:Vec<usize>=(0..initial.len()).collect();order.sort_by(|&a,&b|(exact[b]-exact[b].floor()).total_cmp(&(exact[a]-exact[a].floor())));
    let mut left=n-counts.iter().sum::<usize>();for &i in order.iter().cycle(){if left==0{break;}counts[i]+=1;left-=1;}
    let(mut internal,mut weights)=(vec![],vec![]);
    for(i,(k,v,_))in initial.iter().enumerate(){for _ in 0..counts[i]{internal.push(Internal{block:*k,amplitudes:v.clone()});weights.push(exact[i]/counts[i] as f64);}}
    (internal,weights)
}
/// Bins the engines' distributions of the last observation into the display grids, and clears them.
fn collect(ensemble:&mut Ensemble,positions:&[f64],velocities:&[f64],vbins:&mut [Vec<f64>;2],pbins:&mut [f64])->Result<(),String>{
    let n=positions.len();
    for engine in ensemble.engines(){
        let g=engine.grid();let hm=engine.hbar_over_mass();
        let(mx,my,density)=engine.read_distributions()?;
        let v:Vec<f64>=(0..g.nx).map(|m|hm*g.wave_number(m,g.nx)).collect();
        bin(velocities,&v,&mx.iter().map(|&x|x as f64).collect::<Vec<_>>(),&mut vbins[0]);
        bin(velocities,&v,&my.iter().map(|&x|x as f64).collect::<Vec<_>>(),&mut vbins[1]);
        for p in 0..g.points(){
            let(r,d)=(g.position(p),density[p] as f64);if d==0.{continue;}
            let rows:Vec<(usize,f64)>=overlaps(positions,r[1],g.dx).collect();
            for(i,fx)in overlaps(positions,r[0],g.dx){for &(j,fy)in &rows{pbins[j*n+i]+=d*fx*fy;}}
        }
        engine.clear_distributions()?;
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn assemble(model:&Model,sums:&[Sums],omega:[f64;2],samples:&[usize],velocities:&[f64],vbins:&[[Vec<f64>;2]],positions:&[f64],pbins:&[Vec<f64>],run:usize,refused:usize)->Solution{
    let hm=model.hbar_over_mass;let unit=hm*HBAR/BOLTZMANN*1e12;
    let mut s=Solution{trap_frequency_khz:[omega[0]/std::f64::consts::TAU*1e3,omega[1]/std::f64::consts::TAU*1e3],trajectories:run,..Default::default()};
    for x in sums{
        s.populations.push((0..STATES).map(|q|x.mean(q)).collect());
        s.sample_errors.push((0..POPULATION_GROUPS).map(|q|x.error(q)).fold(0.,f64::max));
        let mut t=[0.;2];let mut e=[0.;2];let mut v=[0.;2];let mut nbar=[0.;2];
        for d in 0..2{
            let(k1,k2)=(x.mean(Q_KAPPA+d),x.mean(Q_KAPPA2+d));t[d]=unit*(k2-k1*k1);
            // The delta method: the spread of κ² − 2⟨κ⟩κ.
            let var=x.w2v2[Q_KAPPA2+d]-4.*k1*x.w2k12[d]+4.*k1*k1*x.w2v2[Q_KAPPA+d]-2.*(k2-2.*k1*k1)*(x.w2v[Q_KAPPA2+d]-2.*k1*x.w2v[Q_KAPPA+d])+(k2-2.*k1*k1).powi(2)*x.w2;
            e[d]=unit*var.max(0.).sqrt()/x.w;
            v[d]=hm*k1;nbar[d]=hm*k2/(2.*omega[d])+omega[d]*x.mean(Q_X2+d)/(2.*hm)-0.5;
        }
        s.temperature_uk.push(t);s.temperature_errors_uk.push(e);s.mean_velocity_mps.push(v);s.mean_occupation.push(nbar);
        s.loss_probability.push(if x.total>0.{x.lost/x.total}else{0.});
    }
    // Loss never decreases: a trajectory once lost stays lost.
    for i in 1..s.loss_probability.len(){if s.loss_probability[i]<s.loss_probability[i-1]{s.loss_probability[i]=s.loss_probability[i-1];}}
    let normalize=|v:&[f64]|->Vec<f64>{let t:f64=v.iter().map(|x|x.max(0.)).sum();v.iter().map(|x|if t>0.{x.max(0.)/t}else{0.}).collect()};
    for d in 0..2{s.velocity[d]=Distributions{grid:velocities.to_vec(),samples:samples.to_vec(),probabilities:vbins.iter().map(|b|normalize(&b[d])).collect()};}
    s.position=Map{grid:positions.to_vec(),samples:samples.to_vec(),probabilities:pbins.iter().map(|b|normalize(b)).collect()};
    s.method=format!("量子轨迹（蒙特卡洛波函数，{run} 条，GPU）：激发态按位置绝热消去，含一阶多普勒与反冲；Strang 分裂，局部指数精确，跃迁在步中点，光子方向按偶极辐射图样抽样；每条轨迹在能容纳它的最小网格上运行{}",
        if refused>0{format!("（{refused} 次因显存不足未能升到更大网格）")}else{String::new()});
    s
}

#[cfg(test)]
mod tests{
    use super::bin;
    /// A flat distribution from a grid coarser, finer or as fine as the display's stays flat on
    /// the display inside its span and keeps its total.
    #[test]
    fn binning_keeps_a_flat_distribution_flat_and_its_total(){
        let to:Vec<f64>=(0..64).map(|i|i as f64-32.).collect();
        for step in [std::f64::consts::SQRT_2,std::f64::consts::FRAC_1_SQRT_2,1.]{
            let n=(40./step) as usize;let from:Vec<f64>=(0..n).map(|i|(i as f64-(n/2) as f64)*step).collect();
            let values=vec![step;n];let mut out=vec![0.;to.len()];bin(&to,&from,&values,&mut out);
            let total:f64=out.iter().sum();assert!((total-values.iter().sum::<f64>()).abs()<1e-9,"{step}: {total}");
            for(i,&x)in to.iter().enumerate(){if x.abs()<15.{assert!((out[i]-1.).abs()<1e-9,"{step}: bin {i} holds {}",out[i]);}}
        }
    }
}
