//! The initial ensemble's motion: one initial internal state's bound thermal ensemble in the trap, the
//! lowered isothermal (Michie 1963; King 1966)
//!   f(r, κ) ∝ e^{−E/k_BT} − 1 for E < 0, zero above,
//! E = ħ²κ²/2m + V(r) the mechanical energy with the state's potential in the trap's light relative to its far
//! level (model.rs `Model::trap_level`: the shifts of the trap's beams and of every far focused beam -- what the
//! atom's motion conserves before the near light comes on at t = 0; a cloud at the trap beam's own equilibrium
//! breathes up to 19 times as hot in a trap crossed by a second far beam --, the near light off), T the
//! temperature asked for; every atom is also bound by the loss's own
//! measure, the trap potential with the near light's interference (model.rs `Model::trap_potential`,
//! engine.rs # Loss): its momenta stop where either energy reaches the far level.
//!
//! # Why lowered
//! The bound states of a Gaussian beam fill an unbounded phase space: at r the bound momenta fill a
//! ball of volume ∝ |V|^{3/2}, and along the beam's axis |V| falls off as (z_R/z)² while the beam
//! widens as z -- the volume grows as ln z. A Boltzmann factor cut at the far level, e^{−E/k_BT} θ(−E),
//! weighs each of those states by about one and is not normalizable: in a trap 4.3 k_BT deep every
//! e-fold of the axis' reach adds 0.6 of the core's weight, and the ensemble's kinetic temperature
//! falls without bound with the reach. The lowered isothermal -- the quasi-stationary state of a
//! thermalizing trap that loses its unbound atoms, an absorbing boundary at the escape energy --
//! vanishes linearly at the far level: its tail weighs ∝ |V|^{5/2}, falling off as z^{−3}. It is a
//! function of E alone, so the conservative motion keeps it (Jeans' theorem), with the light off a
//! stationary state; a trap many k_BT deep holds the Boltzmann distribution to e^{−D/k_BT}.
//!
//! # Its temperature
//! T is the distribution's parameter -- the slope of its energy distribution, as release–recapture
//! fits and adiabatic lowering report a trapped atom's. Its kinetic temperature m⟨v²⟩/k_B lies below T
//! where the trap is shallow (`Ensemble::kinetic`): no stationary bound ensemble of a Gaussian beam is
//! hotter than its hottest energy shell, 0.166 D/k_B (at E ≈ −0.27 D), and the lowered isothermal
//! reaches 0.136 D/k_B as T → ∞. The solve reports it at t = 0 and flags it where it departs from T.
//!
//! # The envelope
//! At r the momenta's weight is w(y) = e^y P(d/2, y) − y^{d/2}/Γ(d/2 + 1) in d dimensions (`kept`), y =
//! −V/k_BT, increasing in y. The trap's depth at r is at most c D max_m A_m(r) (model.rs `Model::trap_slope`),
//! A_m the intensity profile of each mode of its light (`Mode`; a beam and its reflection are one), and
//! A_m ≤ 1/(1 + u_m) with u_m = 2ρ²/w₀² + ζ²/z_R² its harmonic coordinate (e^{−t} ≤ 1/(1 + t) on the Gaussian
//! factor): so y ≤ a max_m 1/(1 + u_m), a = c D/k_BT, and the positions' density is dominated by the mixture
//! Σ_m w(a/(1 + u_m)) -- each part a function of its u_m alone, isotropic in the coordinates ξ that whiten it, its
//! mass in proportion to that whitening's volume. A part's radial law, y = a/(1 + u) with density ∝ (a/y − 1)^{d/2−1}
//! w(y)/y², is drawn exactly from pieces ∝ e^y, each bounded by its factors at their worst ends
//! (`Radial`); the direction is uniform. A drawn point is kept with w(y(r))/Σ_m w(a/(1 + u_m)) ≤ 1, the
//! loss's measure taken only where the trap's own lets it pass.
//!
//! # The momenta
//! At r, kinetic energy k (in k_BT) up to k_max -- y, or the loss's depth there if lower -- with
//! density ∝ (e^{y−k} − 1) k^{d/2−1}: by inverting its distribution `kept(d/2, y, k)/kept(d/2, y, k_max)`
//! (Newton's steps kept within a bracket), the direction uniform.
//!
//! # The quadrature
//! Its kinetic temperature and its share of the trap's thermal distribution are integrals over the
//! positions of closed forms in y (`kept` at d/2 and d/2 + 1, and e^y): over the mixture's parts, each with its
//! share of the mixture at the point (one for a beam alone), in its whitened coordinates,
//! Gauss–Legendre in the radial y on panels doubling away from the top (the first in √(a − y)), in
//! cos θ from the beam's axis, and the trapezoidal rule in the azimuth; the region the share is taken
//! against (U_h ≤ D, as the loss's) is an ellipsoid about the focus, so along each direction it ends
//! at a panel's edge.
use crate::model::{Geometry,Model,Trap,MOST_MIXED};
use crate::rng::{normal_quantile,Rng};
use crate::Complex;
use rayon::prelude::*;
use std::f64::consts::{PI,TAU};
use std::sync::OnceLock;

/// Γ(s) at a positive multiple of ½.
fn gamma(s:f64)->f64{
    let(mut g,mut t)=if (2.*s).round() as i64%2==0{(1.,1.)}else{(PI.sqrt(),0.5)};
    while t<s-0.25{g*=t;t+=1.;}
    g
}

/// e^{y−shift} P(s, k) − e^{−shift} k^s/Γ(s + 1) for 0 ≤ k ≤ y: the lowered isothermal's momenta below
/// kinetic energy k at depth y (both in k_BT; s = d/2 for their number, d/2 + 1 for their energy, each
/// over Γ(s)), P the regularized lower incomplete gamma function. To double precision: below s + 1 by
/// P's series, e^k P(s, k) = k^s/Γ(s+1) Σ_{n≥0} k^n/((s+1)…(s+n)), the lowering subtracted from its first
/// term exactly; above by one less its complement's continued fraction (Lentz; Numerical Recipes' gser
/// and gcf).
pub fn kept(s:f64,y:f64,k:f64,shift:f64)->f64{
    if !(k>0.){return 0.;}
    let g=gamma(s+1.);let base=s*k.ln()-shift;
    if k<s+1.{
        let(mut term,mut rest,mut a)=(1.,0.,s);
        loop{a+=1.;term*=k/a;rest+=term;if term<=rest*1e-17{break;}}
        let(high,low)=((base+y-k).exp()/g,base.exp()/g);
        // e^{y−k}(1 + rest) − 1 times the power, its −1 against e^{y−k} without cancellation.
        high*rest+if y-k<1.{low*(y-k).exp_m1()}else{high-low}
    }else{
        let(tiny,mut b)=(1e-300,k+1.-s);let(mut c,mut d)=(1./tiny,1./b);let mut h=d;
        for i in 1..1000{
            let an=-(i as f64)*(i as f64-s);b+=2.;d=an*d+b;if d.abs()<tiny{d=tiny;}c=b+an/c;if c.abs()<tiny{c=tiny;}
            d=1./d;let step=d*c;h*=step;if (step-1.).abs()<=4.*f64::EPSILON{break;}
        }
        let q=(s*k.ln()-k).exp()/gamma(s)*h;
        (y-shift).exp()*(1.-q)-base.exp()/g
    }
}
/// P(s, k), the regularized lower incomplete gamma function.
fn regularized(s:f64,k:f64)->f64{kept(s,k,k,k)+(s*k.ln()-k).exp()/gamma(s+1.)}

/// Gauss–Legendre's nodes and weights on [−1, 1] (Newton's iteration on P_n).
fn legendre(n:usize)->Vec<(f64,f64)>{
    (0..n).map(|i|{
        let mut x=(PI*(i as f64+0.75)/(n as f64+0.5)).cos();
        loop{
            let(mut p0,mut p1)=(1.,x);for k in 2..=n{let p2=((2*k-1) as f64*x*p1-(k-1) as f64*p0)/k as f64;p0=p1;p1=p2;}
            let slope=n as f64*(x*p1-p0)/(x*x-1.);let dx=p1/slope;x-=dx;
            if dx.abs()<=1e-15{return(x,2./((1.-x*x)*slope*slope));}
        }
    }).collect()
}

/// The envelope's radial law in d dimensions (module documentation, # The envelope): y = a/(1 + u) in
/// (0, a), density p(y) ∝ (a/y − 1)^{d/2−1} w(y)/y², w(y) = e^y F(y), F(y) = kept(d/2, y, y, y). Drawn
/// from pieces of (0, a) -- 5 % wide from 10⁻³ (geometrically, where F grows as y^{d/2+1}) and at most
/// ¼ wide, the last to a --, each ∝ e^y and bounded by sup p e^{−y} over it: (a/y − 1)^{d/2−1} at its
/// worst end, F at its upper (F increases), 1/y² at its lower; the first, from 0, by F ≤ y^{d/2+1}/Γ(d/2
/// + 2) (from e^t − 1 ≤ t e^t), a^{d/2−1}/Γ(d/2 + 2) -- (a − y₁)^{−1/2}/Γ(5/2) in one dimension, whose last
/// piece is drawn in t = a − y ∝ t^{−1/2} e^{−t} instead (t = g²/2, g normal), bounded by y^{−3/2} F.
#[derive(Clone,Debug)]
struct Radial{a:f64,d:usize,edges:Vec<f64>,bounds:Vec<f64>,cumulative:Vec<f64>}
impl Radial{
    fn new(a:f64,d:usize)->Radial{
        let s=d as f64/2.;let top=gamma(s+2.);
        let mut edges=vec![0.];let mut y=(1e-3_f64).min(a/8.);
        while y<a{edges.push(y);y+=(0.05*y).min(0.25);}
        edges.push(a);let n=edges.len()-1;
        let f=|y:f64|kept(s,y,y,y);
        let(mut bounds,mut cumulative,mut total)=(Vec::with_capacity(n),Vec::with_capacity(n),0.);
        for j in 0..n{
            let(lo,hi)=(edges[j],edges[j+1]);
            let(bound,mass)=if j==0{
                let b=if d==1{(a-hi).powf(-0.5)/top}else{a.powf(s-1.)/top};(b,b*((hi-a).exp()-(lo-a).exp()))
            }else if d==1&&j==n-1{
                let b=lo.powf(-1.5)*f(a);(b,b*PI.sqrt()*regularized(0.5,a-lo))
            }else{
                let worst=if s>=1.{lo}else{hi};let b=(a/worst-1.).powf(s-1.)*f(hi)/(lo*lo);(b,b*((hi-a).exp()-(lo-a).exp()))
            };
            bounds.push(bound);total+=mass;cumulative.push(total);
        }
        Radial{a,d,edges,bounds,cumulative}
    }
    /// sup p e^{−y}'s factor at y on piece j (its acceptance against `bounds[j]`).
    fn factor(&self,j:usize,y:f64)->f64{
        let s=self.d as f64/2.;let a=self.a;
        if self.d==1&&j==self.edges.len()-2{y.powf(-1.5)*kept(s,y,y,y)}else{(a/y-1.).powf(s-1.)*kept(s,y,y,y)/(y*y)}
    }
    /// A draw, its first proposal from the uniform deviate `first`, the rest from the stream.
    fn sample(&self,first:f64,rng:&mut Rng)->f64{
        let n=self.bounds.len();let total=self.cumulative[n-1];let mut x=first;
        loop{
            let target=x*total;let j=self.cumulative.partition_point(|&c|c<=target).min(n-1);
            let below=if j==0{0.}else{self.cumulative[j-1]};let v=((target-below)/(self.cumulative[j]-below)).clamp(0.,1.);
            let(lo,hi)=(self.edges[j],self.edges[j+1]);
            let y=if self.d==1&&j==n-1{let g=normal_quantile(0.5+0.5*v*regularized(0.5,self.a-lo).min(1.-1e-16));self.a-0.5*g*g}
                else{hi+(v+(1.-v)*(lo-hi).exp()).ln()};
            if y>0.&&y<self.a&&rng.open_uniform()*self.bounds[j]<=self.factor(j,y){return y;}
            x=rng.interior();
        }
    }
}

/// A mode of the trap's light in the envelope's mixture (module documentation, # The envelope): its frame (two
/// axes across the beam, its axis), waist and Rayleigh range.
#[derive(Clone,Debug)]
struct Mode{frame:[[f64;3];3],waist:f64,rayleigh:f64}
impl Mode{
    fn new(g:&Geometry)->Result<Mode,String>{
        let(waist,n)=(g.waist.ok_or("The trap beam has no waist")?,g.direction);
        let rayleigh=g.wave_number*waist*waist/2.;
        let pick=if n[0].abs()<0.9{[1.,0.,0.]}else{[0.,1.,0.]};let along:f64=(0..3).map(|i|pick[i]*n[i]).sum();
        let e1:[f64;3]={let v:[f64;3]=std::array::from_fn(|i|pick[i]-along*n[i]);let l=(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();v.map(|x|x/l)};
        let e2=[n[1]*e1[2]-n[2]*e1[1],n[2]*e1[0]-n[0]*e1[2],n[0]*e1[1]-n[1]*e1[0]];
        Ok(Mode{frame:[e1,e2,n],waist,rayleigh})
    }
    /// Its harmonic coordinate u at r.
    fn coordinate(&self,r:[f64;3])->f64{
        let p=|i:usize|(0..3).map(|c|r[c]*self.frame[i][c]).sum::<f64>();let(x,y,z)=(p(0),p(1),p(2));
        2.*(x*x+y*y)/(self.waist*self.waist)+z*z/(self.rayleigh*self.rayleigh)
    }
    /// The whitened point at u = 1 along (cos θ from the beam's axis, azimuth φ).
    fn along(&self,mu:f64,phi:f64)->[f64;3]{
        let(s,(sp,cp))=((1.-mu*mu).max(0.).sqrt(),phi.sin_cos());let w=self.waist/2f64.sqrt();
        std::array::from_fn(|i|w*s*(cp*self.frame[0][i]+sp*self.frame[1][i])+self.rayleigh*mu*self.frame[2][i])
    }
    /// ξ ↦ r on the free lab axes, the inverse square root of the harmonic coordinate's form u = rᵀBr,
    /// B = (2/w₀²)(1 − nnᵀ) + nnᵀ/z_R², on them (the frozen axes' zero), and its volume.
    fn whitening(&self,free:[bool;3])->([[f64;3];3],f64){
        let n=self.frame[2];let(across,axial)=(2./(self.waist*self.waist),1./(self.rayleigh*self.rayleigh));
        let form:[[f64;3];3]=std::array::from_fn(|i|std::array::from_fn(|j|if free[i]&&free[j]{across*(if i==j{1.}else{0.})+(axial-across)*n[i]*n[j]}else if i==j{1.}else{0.}));
        let(values,vectors)=crate::model::symmetric_eigen(form);
        (std::array::from_fn(|i|std::array::from_fn(|j|if free[i]&&free[j]{(0..3).map(|k|vectors[k][i]*vectors[k][j]/values[k].sqrt()).sum()}else{0.})),values.iter().map(|v|v.sqrt().recip()).product())
    }
}

/// One initial internal state's bound thermal ensemble (module documentation): its cluster and place
/// in it; its depth at the focus in the trap's light D_T (rad/us) and the envelope's top a = c D_T/k_BT
/// (model.rs `Model::trap_slope`); the classical ensemble's kinetic temperature relative to T, per axis
/// (`kinetic`); and its share of the trap's thermal distribution over its region (U_h ≤ D, Maxwell's
/// momenta) that it leaves out -- the unbound atoms and the lowering (`cut`), these two by quadrature
/// (`integrate`). The envelope's modes; the radial laws per number of free axes are made when first drawn from.
#[derive(Clone,Debug)]
pub struct Ensemble{pub cluster:usize,pub state:usize,pub depth:f64,pub top:f64,pub kinetic:f64,pub cut:f64,kt:f64,modes:Vec<Mode>,radial:[OnceLock<Radial>;3]}
impl Ensemble{
    /// The ensemble of state `state` of `cluster` at k_BT `kt` (rad/us), its `kinetic` and `cut` still to come
    /// (`integrate`); none for a state the trap does not hold (no bound state: D_T ≤ 0).
    pub fn new(model:&Model,cluster:usize,state:usize,kt:f64)->Result<Option<Ensemble>,String>{
        let depth=-model.trap_level(cluster,state,[0.;3]);
        if !(depth>0.){return Ok(None);}
        let c=model.trap_slope(cluster,state,depth)?;let top=c*depth/kt;
        let modes=model.trap_modes().into_iter().map(|l|Mode::new(&model.geometry[l])).collect::<Result<Vec<_>,_>>()?;
        Ok(Some(Ensemble{cluster,state,depth,top,kinetic:1.,cut:0.,kt,modes,radial:Default::default()}))
    }
    /// Its kinetic temperature and cut, its trap `trap` and its loss's depth `region` (`Model::depth`, the region
    /// U_h ≤ D its share is taken against).
    pub fn integrate(&mut self,model:&Model,trap:&Trap,region:f64){(self.kinetic,self.cut)=self.integrals(model,trap,region);}
    /// The trap's own depth and the loss's at r, in k_BT (module documentation): y_T, and the momenta's
    /// reach k_max = min(y_T, y_A), zero where either is unbound.
    fn depths(&self,model:&Model,r:[f64;3],psi:&[Complex;MOST_MIXED])->(f64,f64){
        let own=-model.trap_level(self.cluster,self.state,r)/self.kt;if !(own>0.){return(own,0.);}
        let loss=-model.trap_potential(self.cluster,&model.point(r,[0.;3]),psi,0.)/self.kt;
        (own,own.min(loss).max(0.))
    }
    /// The mixture's envelope at r in `s` half dimensions, the part `m` proposed at y there: the sum of the parts, and that part's share of it.
    fn envelope(&self,s:f64,m:usize,y:f64,r:[f64;3])->(f64,f64){
        let a=self.top;let own=kept(s,y,y,a);
        if self.modes.len()==1{return(own,1.);}
        let rest:f64=self.modes.iter().enumerate().filter(|x|x.0!=m).map(|(_,o)|{let v=a/(1.+o.coordinate(r));kept(s,v,v,a)}).sum();
        (own+rest,own/(own+rest))
    }
    /// The classical ensemble's kinetic temperature relative to T and its cut of the trap's thermal
    /// distribution over the region U_h ≤ `region` (module documentation, # The quadrature), in three
    /// dimensions: 32 Gauss nodes in cos θ, 8 azimuths, 16 Gauss nodes per radial panel.
    fn integrals(&self,model:&Model,trap:&Trap,region:f64)->(f64,f64){
        let(a,hm)=(self.top,model.hbar_over_mass);let mut psi=[Complex::ZERO;MOST_MIXED];psi[self.state]=Complex::ONE;
        let(polar,radial)=(legendre(32),legendre(16));let azimuths=8;let every=[true;3];let first=self.modes[0].whitening(every).1;
        // The mixture's parts, each with its volume (relative to the first's) and its share of the mixture at the point.
        let sums=self.modes.iter().enumerate().map(|(m,mode)|{let volume=mode.whitening(every).1/first;
          (0..polar.len()*azimuths).into_par_iter().map(|p|{
            let((mu,wm),phi)=(polar[p/azimuths],TAU*((p%azimuths) as f64+0.5)/azimuths as f64);let unit=mode.along(mu,phi);
            // The region along this direction: U_h = u U_h(unit) ≤ D, at t_R = a u_R/(1 + u_R).
            let harmonic:f64=(0..3).map(|i|{let x:f64=(0..3).map(|c|unit[c]*trap.axes[i][c]).sum();trap.frequencies[i]*trap.frequencies[i]*x*x}).sum::<f64>()/(2.*hm);
            let inside=if harmonic>0.{let u=region/harmonic;a*u/(1.+u)}else{a};
            let mut edges=vec![0.];let mut t=1_f64;while t<a{edges.push(t);t*=2.;}edges.push(a);
            if inside<a{let k=edges.partition_point(|&e|e<inside);if edges[k]!=inside{edges.insert(k,inside);}}
            let mut s=[0.;4];
            for k in 0..edges.len()-1{
                let(lo,hi)=(edges[k],edges[k+1]);
                for &(x,w) in &radial{
                    // The first panel in τ = √t (its t^{1/2} at the top), the rest in t.
                    let(t,dt)=if k==0{let(r0,r1)=(lo.sqrt(),hi.sqrt());let tau=0.5*(r0+r1)+0.5*(r1-r0)*x;(tau*tau,w*0.5*(r1-r0)*2.*tau)}else{(0.5*(lo+hi)+0.5*(hi-lo)*x,w*0.5*(hi-lo))};
                    // u = t/(a − t); the measure ½ u^{1/2} du = ½ (t/(a − t))^{1/2} a/(a − t)² dt.
                    let u=t/(a-t);let r=unit.map(|x|x*u.sqrt());let measure=0.5*u.sqrt()*a/((a-t)*(a-t))*dt*wm*volume*self.envelope(1.5,m,a-t,r).1;
                    let(own,reach)=self.depths(model,r,&psi);
                    let(number,energy)=(kept(1.5,own,reach,a),kept(2.5,own,reach,a));
                    s[0]+=measure*number;s[1]+=measure*energy;
                    if t<inside{s[2]+=measure*number;s[3]+=measure*(own-a).exp();}
                }
            }
            s}).reduce(||[0.;4],|x,y|std::array::from_fn(|i|x[i]+y[i]))
        }).fold([0.;4],|x:[f64;4],y|std::array::from_fn(|i|x[i]+y[i]));
        (sums[1]/sums[0],(1.-sums[2]/sums[3]).max(0.))
    }
    /// The radial law in d dimensions.
    fn radial(&self,d:usize)->&Radial{self.radial[d-1].get_or_init(||Radial::new(self.top,d))}
    /// The Latin deviates a draw takes with the free lab axes `free` (`draw`).
    pub(crate) fn deviates(free:[bool;3])->usize{match free.iter().filter(|&&f|f).count(){0=>0,3=>6,_=>4}}
    /// A classical draw (module documentation): (r, κ) on the free lab axes, the others zero. Its first
    /// proposal and its momenta take the member's Latin deviates (`next`: 2 + 2 × the direction's -- two
    /// in three dimensions, one in fewer --, taken whatever follows, so that its group's stay in step), the
    /// further proposals its own stream's.
    pub(crate) fn draw(&self,model:&Model,free:[bool;3],rng:&mut Rng,next:&mut dyn FnMut(&mut Rng)->f64)->([f64;3],[f64;3]){
        let d=free.iter().filter(|&&f|f).count();if d==0{return([0.;3],[0.;3]);}
        let directions=if d==3{2}else{1};
        let first:Vec<f64>=(0..1+directions).map(|_|next(rng)).collect();let momenta:Vec<f64>=(0..1+directions).map(|_|next(rng)).collect();
        let mut psi=[Complex::ZERO;MOST_MIXED];psi[self.state]=Complex::ONE;
        let(a,s)=(self.top,d as f64/2.);let radial=self.radial(d);
        // The mixture's parts' whitenings and masses, their radial law the same.
        let parts:Vec<([[f64;3];3],f64)>=self.modes.iter().map(|m|m.whitening(free)).collect();let mass:f64=parts.iter().map(|p|p.1).sum();
        let mut deviates=first;
        let(r,own,reach)=loop{
            // The part from the first deviate, the radial law from what is left of it (a beam alone: the deviate itself).
            let(mut x,mut m)=(deviates[0],0);
            for(i,p)in parts.iter().enumerate(){let share=p.1/mass;if x<share||i+1==parts.len(){m=i;x=(x/share).min(1.);break;}x-=share;}
            let y=radial.sample(x,rng);let u=a/y-1.;
            let xi=sphere(free,&deviates[1..]).map(|x|x*u.sqrt());
            let r:[f64;3]=std::array::from_fn(|i|(0..3).map(|j|parts[m].0[i][j]*xi[j]).sum());
            let draw=rng.open_uniform()*self.envelope(s,m,y,r).0;
            let own=-model.trap_level(self.cluster,self.state,r)/self.kt;
            if own>0.{
                debug_assert!(own<=self.modes.iter().map(|o|a/(1.+o.coordinate(r))).fold(y,f64::max)*(1.+1e-12),"the envelope at {r:?}: y {own} above its bound");
                // The trap's own depth first, the loss's only where that lets the point pass.
                if draw<=kept(s,own,own,a){
                    let loss=-model.trap_potential(self.cluster,&model.point(r,[0.;3]),&psi,0.)/self.kt;let reach=own.min(loss).max(0.);
                    if draw<=kept(s,own,reach,a){break(r,own,reach);}
                }
            }
            deviates=(0..1+directions).map(|_|rng.interior()).collect();
        };
        // The kinetic energy k by its distribution, then the direction.
        let k=invert(s,own,reach,momenta[0]);
        let size=(2.*k*self.kt/model.hbar_over_mass).sqrt();
        (r,sphere(free,&momenta[1..]).map(|x|x*size))
    }
}

/// A uniform direction on the free lab axes from uniform deviates: (cos θ, φ) in three dimensions, an
/// angle in two, a sign in one.
fn sphere(free:[bool;3],deviates:&[f64])->[f64;3]{
    let axes:Vec<usize>=(0..3).filter(|&a|free[a]).collect();let mut v=[0.;3];
    match axes.len(){
        3=>{let mu=2.*deviates[0]-1.;let(s,(sp,cp))=((1.-mu*mu).max(0.).sqrt(),(TAU*deviates[1]).sin_cos());v=[s*cp,s*sp,mu];}
        2=>{let(sp,cp)=(TAU*deviates[0]).sin_cos();v[axes[0]]=cp;v[axes[1]]=sp;}
        _=>v[axes[0]]=if deviates[0]<0.5{-1.}else{1.},
    }
    v
}

/// The kinetic energy (in k_BT) whose distribution at depth y, up to k_max, is u: kept(s, y, k)/kept(s, y,
/// k_max) = u, its density ∝ (e^{y−k} − 1) k^{s−1} -- Newton's steps on k^s (smooth at 0) kept within a
/// bracket, bisection where one leaves it.
fn invert(s:f64,y:f64,reach:f64,u:f64)->f64{
    if !(reach>0.){return 0.;}
    let target=u*kept(s,y,reach,y);let(mut lo,mut hi)=(0_f64,reach.powf(s));let mut v=u*hi;
    for _ in 0..200{
        let k=v.powf(1./s);let f=kept(s,y,k,y)-target;
        if f==0.{break;}
        if f<0.{lo=v;}else{hi=v;}
        // d kept/d(k^s) = (e^{y−k} − 1) k^{s−1} e^{−y}/(Γ(s) s k^{s−1}) = (e^{−k} − e^{−y})/Γ(s + 1).
        let slope=((-k).exp()-(-y).exp())/gamma(s+1.);let newton=v-f/slope;
        let next=if slope>0.&&newton>lo&&newton<hi{newton}else{0.5*(lo+hi)};
        let done=(next-v).abs()<=1e-15*v.max(f64::MIN_POSITIVE)||hi-lo<=1e-15*hi;v=next;
        if done{break;}
    }
    v.powf(1./s)
}

#[cfg(test)]
mod tests{
    use super::*;
    /// `kept` against its definition by quadrature, at depths and reaches from 10⁻⁴ to 40 k_BT and both
    /// sides of its two branches, in one to four half-dimensions.
    #[test]
    fn the_kept_momenta_are_their_integral(){
        let nodes=legendre(64);
        for s in [0.5,1.,1.5,2.,2.5]{for y in [1e-4,0.3,1.4,2.6,3.6,7.,40.]{for frac in [0.3,0.99,1.]{
            let k=y*frac;
            // ∫_0^k (e^{y−q} − 1) q^{s−1} dq / Γ(s), in q = k v^2 (smooth at 0), on 8 panels.
            let mut sum=0.;
            for p in 0..8{let(a,b)=(p as f64/8.,(p+1) as f64/8.);for &(x,w) in &nodes{let v=0.5*(a+b)+0.5*(b-a)*x;let q=k*v*v;sum+=w*0.5*(b-a)*2.*k*v*(y-q).exp_m1()*q.powf(s-1.);}}
            let exact=sum/gamma(s);let value=kept(s,y,k,0.);
            assert!((value-exact).abs()<=1e-11*exact.abs().max(1e-300),"s {s} y {y} k {k}: {value:e} against {exact:e}");
        }}}
    }
    /// The radial law's draws against its density (`Radial`) by a binned χ²: 2¹⁶ draws in 32 bins of
    /// equal probability (the density's integral by quadrature), at a shallow and a deep top in each
    /// dimension.
    #[test]
    fn the_radial_law_is_its_density(){
        let nodes=legendre(48);
        for d in [1,2,3]{for a in [0.5,4.3,40.]{
            let r=Radial::new(a,d);let s=d as f64/2.;
            let density=|y:f64|(a/y-1.).powf(s-1.)*kept(s,y,y,a)/(y*y);
            // The density's distribution on a fine grid in y = a(1 − v²)... by panels in √(a − y) and in y.
            let grid=4096;let mut cdf=vec![0.;grid+1];
            for i in 0..grid{let(t0,t1)=((i as f64/grid as f64)*a.sqrt(),((i+1) as f64/grid as f64)*a.sqrt());
                let mut sum=0.;for &(x,w) in &nodes[..]{let tau=0.5*(t0+t1)+0.5*(t1-t0)*x;let y=a-tau*tau;if y>0.{sum+=w*0.5*(t1-t0)*2.*tau*density(y);}}
                cdf[i+1]=cdf[i]+sum;}
            let total=cdf[grid];
            let bins=32;let count=1<<16;let mut hits=vec![0usize;bins];let mut rng=Rng::new(3,crate::rng::Purpose::Initial,d as u64,0);
            for _ in 0..count{let first=rng.interior();let y=r.sample(first,&mut rng);
                // Its distribution's value at y: the τ = √(a − y) grid read backwards.
                let tau=(a-y).max(0.).sqrt()/a.sqrt()*grid as f64;let i=(tau.floor() as usize).min(grid-1);let f=tau-i as f64;
                let below=total-(cdf[i]+f*(cdf[i+1]-cdf[i]));let b=((below/total)*bins as f64).floor() as usize;hits[b.min(bins-1)]+=1;}
            let expected=count as f64/bins as f64;let chi:f64=hits.iter().map(|&h|(h as f64-expected).powi(2)/expected).sum();
            // 31 degrees of freedom: the 0.1 % point is 61.1.
            assert!(chi<=61.1,"d {d} a {a}: χ² {chi:.1} over {hits:?}");
        }}
    }
    /// The momenta's inverse distribution (`invert`) against `kept`, to round-off, across the deviates.
    #[test]
    fn the_kinetic_energy_inverts_its_distribution(){
        for s in [0.5,1.,1.5]{for y in [1e-3,0.7,4.3,30.]{for reach in [y,0.4*y]{for u in [1e-9,0.01,0.3,0.5,0.97,1.-1e-12]{
            let k=invert(s,y,reach,u);let back=kept(s,y,k,y)/kept(s,y,reach,y);
            assert!(k>=0.&&k<=reach*(1.+1e-12)&&(back-u).abs()<=1e-11,"s {s} y {y} reach {reach} u {u}: k {k}, distribution {back}");
        }}}}
    }
}
