//! Control variates: per trajectory, quantities whose means are known exactly and which follow what
//! the solve estimates, so that each output's estimate -- the trajectories' mean less β times its
//! control's excess over the known mean (stats.rs) -- keeps its expectation with a smaller spread.
//!
//! # T: the harmonic twin
//! The trajectory's initial r₀ and κ₀ carried by the trap's harmonic part alone (the engine's flow,
//! along the trap's principal axes): over each output's window (lib.rs `Windows`) its κ² along a lab
//! axis, less the same with κ₀ = 0, is κ₀ᵀ A κ₀ + 2 κ₀ᵀ B r₀ -- A and B the window's means of products
//! of the axes' cosines and sines. κ₀ is centred and drawn apart from r₀, so the mean is tr(A Σ),
//! Σ = E[κ₀κ₀ᵀ] of the initial states (each state's Maxwell momenta, its trap's axes and Wigner
//! factor), exactly, whatever r₀'s distribution. While the atom still moves in the harmonic trap the
//! control follows its window's κ² closely; the light and the trap's anharmonic part decorrelate it.
//!
//! # Populations: the held twin
//! The same trajectory -- its initial internal state, thresholds and jumps' draws (common random
//! numbers, engine.rs) -- held without motion at a point where the trap beam's intensity is the mean
//! the moving atoms see (the pilot's states): its populations' mean is the eliminated master equation
//! there (`Twin::means`): the engine's generator there, its decay its own jumps' (so that its trace
//! stays one, `Twin::new`), and the engine's jumps. While the moving
//! atom's light differs little from the point's, its jumps follow the twin's; a light pattern on the
//! scale of the motion decorrelates them at once. A twin exists when the generator is static (no
//! residual beats) and every cluster is of ground states (the exact step's clusters with explicit
//! excited states have none yet): its exponentials are then computed once, and the master equation is
//! one exponential of the superoperator per output interval.
use crate::engine::{Held,Settings};
use crate::linear::Dense;
use crate::model::{Model,MOST_STATES};
use crate::stats::KT_PER_UK;
use crate::{Complex,Initial,STATES};

/// The harmonic twin's control: per output and lab axis A and B (on the principal axes of the
/// settings' trap) and its exact mean.
pub struct Harmonic{axes:[[f64;3];3],pub(crate) a:Vec<[[[f64;3];3];3]>,pub(crate) b:Vec<[[[f64;3];3];3]>,pub mean:Vec<[f64;3]>}
impl Harmonic{
    /// The control of a run of `steps` steps of the settings' dt per output, `outputs` outputs, with
    /// the settings' T windows (lib.rs `Windows`: `half` steps per lab axis, cut at t = 0).
    pub fn new(model:&Model,settings:&Settings,initial:&Initial,steps:usize,outputs:usize)->Harmonic{
        let(hm,dt,w)=(model.hbar_over_mass,settings.dt,settings.frequencies);
        let half=crate::half_windows(settings);
        let last=(outputs-1)*steps+half.into_iter().max().unwrap();
        // Running sums over the steps of cos_i cos_j and cos_i sin_j (index 3 i + j), from step 0.
        let mut prefix=vec![[[0.;9];2];last+2];
        for k in 0..=last{
            let t=k as f64*dt;let(s,c):([f64;3],[f64;3])=(std::array::from_fn(|i|(w[i]*t).sin()),std::array::from_fn(|i|(w[i]*t).cos()));
            for i in 0..3{for j in 0..3{prefix[k+1][0][3*i+j]=prefix[k][0][3*i+j]+c[i]*c[j];prefix[k+1][1][3*i+j]=prefix[k][1][3*i+j]+c[i]*s[j];}}
        }
        let sigma=initial_momenta(model,settings,initial);
        let(mut a,mut b,mut mean)=(vec![[[[0.;3];3];3];outputs],vec![[[[0.;3];3];3];outputs],vec![[0.;3];outputs]);
        for i in 0..outputs{
            let at=i*steps;
            for x in 0..3{
                let h=half[x].min(at);let(lo,hi)=(at-h,at+h);let n=(hi-lo+1) as f64;let u:[f64;3]=std::array::from_fn(|p|settings.axes[p][x]);
                for p in 0..3{for q in 0..3{
                    let(cc,cs)=((prefix[hi+1][0][3*p+q]-prefix[lo][0][3*p+q])/n,(prefix[hi+1][1][3*p+q]-prefix[lo][1][3*p+q])/n);
                    a[i][x][p][q]=u[p]*u[q]*cc;b[i][x][p][q]=-u[p]*u[q]*w[q]/hm*cs;mean[i][x]+=a[i][x][p][q]*sigma[p][q];
                }}
            }
        }
        Harmonic{axes:settings.axes,a,b,mean}
    }
    /// The control at output i of a trajectory that started at r₀ with κ₀ (lab axes), per lab axis.
    pub fn value(&self,i:usize,r0:[f64;3],k0:[f64;3])->[f64;3]{
        let on=|v:[f64;3]|->[f64;3]{std::array::from_fn(|p|(0..3).map(|x|self.axes[p][x]*v[x]).sum())};let(k,r)=(on(k0),on(r0));
        std::array::from_fn(|x|{let(a,b)=(&self.a[i][x],&self.b[i][x]);let mut c=0.;
            for p in 0..3{for q in 0..3{c+=k[p]*(a[p][q]*k[q]+2.*b[p][q]*r[q]);}}c})
    }
}

/// E[κ₀κ₀ᵀ] on the settings' principal axes: each initial state's Maxwell momenta along its own trap's
/// axes with the harmonic oscillator's Wigner factor (lib.rs `initial_trajectory`), weighted by its
/// population.
fn initial_momenta(model:&Model,settings:&Settings,initial:&Initial)->[[f64;3];3]{
    let(kt,hm)=(KT_PER_UK*initial.temperature_uk,model.hbar_over_mass);let mut lab=[[0.;3];3];
    for(&(_,_,p),trap)in initial.states.iter().zip(&initial.traps){
        for i in 0..3{let half=trap.frequencies[i]/(2.*kt);let quantum=if half>0.{half/half.tanh()}else{1.};
            for x in 0..3{for y in 0..3{lab[x][y]+=p*kt/hm*quantum*trap.axes[i][x]*trap.axes[i][y];}}}
    }
    std::array::from_fn(|p|std::array::from_fn(|q|(0..3).map(|x|(0..3).map(|y|settings.axes[p][x]*lab[x][y]*settings.axes[q][y]).sum::<f64>()).sum()))
}

/// The held twin: its point, every cluster's generator and jumps there (engine.rs `Held`), and its
/// populations' readout there, a linear map of the cluster's density.
pub struct Twin{pub held:Held,pub(crate) readout:Vec<Vec<[f64;STATES]>>}
impl Twin{
    /// The twin where the trap beam's intensity is `intensity`, on the line through its focus across
    /// it; none when the generator has residual beats.
    pub fn new(model:&Model,intensity:f64)->Option<Twin>{
        if !static_generator(model)||(0..model.clusters()).any(|k|model.mixed(k)){return None;}
        let g=&model.geometry[model.trap_beam];let n=g.direction;
        let pick=if n[0].abs()<0.9{[1.,0.,0.]}else{[0.,1.,0.]};let along=pick[0]*n[0]+pick[1]*n[1]+pick[2]*n[2];
        let across:[f64;3]=std::array::from_fn(|i|pick[i]-along*n[i]);let l=(across[0]*across[0]+across[1]*across[1]+across[2]*across[2]).sqrt();
        let at=|d:f64|->[f64;3]{across.map(|x|x/l*d)};
        let(mut lo,mut hi)=(0.,g.waist?);while g.intensity(at(hi))>intensity{hi*=2.;}
        for _ in 0..60{let mid=0.5*(lo+hi);if g.intensity(at(mid))>intensity{lo=mid;}else{hi=mid;}}
        let point=model.point(at(lo),[0.;3]);
        // Each class's jump operators: column j the photon amplitudes of the basis state j.
        let jumps:Vec<Vec<[Vec<Complex>;3]>>=(0..model.clusters()).map(|k|{
            let n=model.dimension(k);let columns:Vec<Vec<[[Complex;MOST_STATES];3]>>=(0..n).map(|j|{let mut psi=[Complex::ZERO;MOST_STATES];psi[j]=Complex::ONE;
                model.photons(k,&point,&psi,0.,None).amplitude}).collect();
            model.ground.clusters[k].classes.iter().enumerate().map(|(c,class)|{let nt=model.dimension(class.target);
                std::array::from_fn(|x|(0..nt*n).map(|e|columns[e%n][c][x][e/n]).collect())}).collect()}).collect();
        // Each cluster's generator: the Hermitian part of the engine's H_eff there, its decay its own
        // jumps' Σ L†L. The engine's decay comes from the tables of the responses and its jumps from
        // the photons, which take the recoil's (and the Doppler's) first order: the two differ by ~1e-4
        // of the scattering rate, and a generator of the one with the jumps of the other leaks trace.
        // With its jumps' decay the twin's unraveling is exactly the Lindblad equation of its H and L
        // (`means`): its mean exact, its trace one, every correction's trace zero.
        let locals=(0..model.clusters()).map(|k|{
            let n=model.dimension(k);let mut local=model.local(k,&point,0.);let h=local.hamiltonian;
            for p in 0..n{for q in 0..n{
                let mut decay=Complex::ZERO;
                for(class,l)in model.ground.clusters[k].classes.iter().zip(&jumps[k]){let nt=model.dimension(class.target);for x in 0..3{for i in 0..nt{decay+=l[x][i*n+p].conj()*l[x][i*n+q];}}}
                local.hamiltonian[p*n+q]=(h[p*n+q]+h[q*n+p].conj())*0.5-decay*Complex::new(0.,0.5);
            }}
            local}).collect();
        // The readout's images of the density's basis: ψψ† for ψ = e_i, e_i + e_j and e_i + i e_j.
        let readout=(0..model.clusters()).map(|k|{
            let n=model.dimension(k);let pure=|psi:&[Complex;MOST_STATES]|->[f64;STATES]{let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();model.populations(k,&point,psi,0.).map(|x|x*norm)};
            let unit=|i:usize|{let mut v=[Complex::ZERO;MOST_STATES];v[i]=Complex::ONE;v};
            let diagonal:Vec<[f64;STATES]>=(0..n).map(|i|pure(&unit(i))).collect();
            let mut out=vec![[0.;STATES];n*n];
            for i in 0..n{out[i*n+i]=diagonal[i];for j in i+1..n{
                let(mut x,mut y)=(unit(i),unit(i));x[j]=Complex::ONE;y[j]=Complex::I;let(px,py)=(pure(&x),pure(&y));
                // ρ_ij e_ij + ρ_ji e_ji = Re ρ_ij (e_ij + e_ji) + Im ρ_ij (i e_ij − i e_ji): their images.
                out[i*n+j]=std::array::from_fn(|s|px[s]-diagonal[i][s]-diagonal[j][s]);out[j*n+i]=std::array::from_fn(|s|diagonal[i][s]+diagonal[j][s]-py[s]);
            }}
            out}).collect();
        Some(Twin{held:Held{point,locals,jumps},readout})
    }
    /// A twin's populations: those of its state ψ (in cluster k) at its point.
    pub fn populations(&self,k:usize,psi:&[Complex])->[f64;STATES]{
        let n=(self.readout[k].len() as f64).sqrt().round() as usize;let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let rho:Vec<Complex>=(0..n*n).map(|e|psi[e/n]*psi[e%n].conj()*(1./norm)).collect();self.density(k,&rho)
    }
    /// The populations of a cluster's density ρ (n × n, row-major) at the twin's point.
    fn density(&self,k:usize,rho:&[Complex])->[f64;STATES]{
        let m=&self.readout[k];let n=(m.len() as f64).sqrt().round() as usize;let mut out=[0.;STATES];
        for i in 0..n{for s in 0..STATES{out[s]+=rho[i*n+i].re*m[i*n+i][s];}
            for j in i+1..n{let z=rho[i*n+j];for s in 0..STATES{out[s]+=z.re*m[i*n+j][s]+z.im*m[j*n+i][s];}}}
        out
    }
    /// The populations' exact means at `outputs` outputs `interval` apart: the eliminated master
    /// equation at the twin's point -- H_eff and the jumps of the engine's tables there -- from the
    /// initial density's ground populations, by the exponential of its superoperator over an interval.
    pub fn means(&self,model:&Model,initial_density:&[Complex],interval:f64,outputs:usize)->Vec<[f64;STATES]>{
        let g=&model.ground;let sizes:Vec<usize>=(0..model.clusters()).map(|k|model.dimension(k)).collect();
        let offsets:Vec<usize>=sizes.iter().scan(0,|o,&n|{let x=*o;*o+=n*n;Some(x)}).collect();let d:usize=sizes.iter().map(|n|n*n).sum();
        let mut generator=Dense::zeros(d,d);
        for(k,&n)in sizes.iter().enumerate(){
            let h=&self.held.locals[k].hamiltonian;let o=offsets[k];
            // −i(Hρ − ρH†).
            for i in 0..n{for j in 0..n{for m in 0..n{
                generator[(o+i*n+j,o+m*n+j)]+=Complex::new(0.,-1.)*h[i*n+m];
                generator[(o+i*n+j,o+i*n+m)]+=Complex::new(0.,1.)*h[j*n+m].conj();
            }}}
            // Σ L ρ L† into each class's target cluster.
            for(class,c)in g.clusters[k].classes.iter().enumerate(){
                let(nt,ot)=(sizes[c.target],offsets[c.target]);
                for a in 0..3{let l=&self.held.jumps[k][class][a];for i in 0..nt{for j in 0..nt{for p in 0..n{for q in 0..n{
                    generator[(ot+i*nt+j,o+p*n+q)]+=l[i*n+p]*l[j*n+q].conj();
                }}}}}
            }
        }
        let propagator=exponential(&generator,interval);
        let mut rho=vec![Complex::ZERO;d];
        for(k,c)in g.clusters.iter().enumerate(){for(x,&b)in c.states.iter().enumerate(){let s=g.ground_states[b];rho[offsets[k]+x*sizes[k]+x]=initial_density[s*STATES+s];}}
        let read=|rho:&[Complex]|->[f64;STATES]{let mut p=[0.;STATES];for k in 0..sizes.len(){let x=self.density(k,&rho[offsets[k]..offsets[k]+sizes[k]*sizes[k]]);for s in 0..STATES{p[s]+=x[s];}}p};
        let mut out=vec![read(&rho)];
        for _ in 1..outputs{rho=(0..d).map(|i|(0..d).fold(Complex::ZERO,|z,j|z+propagator[(i,j)]*rho[j])).collect();out.push(read(&rho));}
        out
    }
}

/// Whether the generator is static: no residual beat in the kept couplings, the paths, the photons or
/// the readout's coherent sums (the frames make the strongest couplings static, ground.rs).
pub fn static_generator(model:&Model)->bool{
    model.ground.clusters.iter().all(|c|c.groups.iter().all(|x|x.entries.iter().all(|e|e.2==0.))&&c.paths.iter().all(|p|p.residual==0.)
        &&c.classes.iter().all(|k|k.photons.iter().all(|p|p.1==0.))&&c.coherent.iter().all(|g|g.iter().all(|x|x.1==0.)))
}

/// exp(M τ) by scaling and squaring of its Taylor series, summed to double precision.
fn exponential(m:&Dense,tau:f64)->Dense{
    let n=m.rows;let norm=(0..n).map(|j|(0..n).map(|i|m[(i,j)].abs()).sum::<f64>()).fold(0.,f64::max)*tau.abs();
    let squarings=if norm>0.5{(norm/0.5).log2().ceil() as u32}else{0};let scale=tau/2f64.powi(squarings as i32);
    let a=m.scaled(Complex::new(scale,0.));let mut sum=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});let mut term=sum.clone();
    for k in 1..=30{term=term.product(&a).scaled(Complex::new(1./k as f64,0.));sum.add_scaled(Complex::ONE,&term);if term.largest()<=f64::EPSILON*sum.largest(){break;}}
    for _ in 0..squarings{sum=sum.product(&sum);}
    sum
}
