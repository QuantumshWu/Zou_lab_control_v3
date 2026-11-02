//! The physics on the grid: `atom_motion_core::ground` evaluated at every grid point of the plane
//! in double precision, as the tables the GPU kernels read in single precision.
//!
//! # The plane
//! The request's motion frame has the plane of 2D motion on its first two axes (the grid's x and
//! y) and the tweezer along the third; positions are r = (x, y, 0) (um). The ground manifold's
//! two hyperfine blocks are the trajectories' blocks: component c of a trajectory in block k is
//! `ground.blocks[k].states[c]`.
//!
//! # The generator with the motion
//! At r the eliminated ground generator (ground.rs) has H_eff(r) = H0g + H_ls − (i/2)Γ and the
//! jumps L_{a,c} = P_to C_a R_c(r) P_from. To first order in the excited state's kinetic shift
//! w_l = ħ k_l·κ/m + ħ k_l²/2m of absorbing from beam l (k_l its wave vector in the plane, ħκ the
//! atom's momentum before the absorption), R_c → R_c + ΔR_c with
//!
//! ```text
//! ΔR_c = Σ_{l∈c} −a_l(r) w_l R2_l = ΔR_c^rec + Σ_d ΔR_c^d κ̂_d,     ΔR_c^d = −Σ_{l∈c} a_l (ħ k_{l,d}/m) R2_l
//! ```
//!
//! the matrices to the left of the momentum operator κ̂_d = −i∂_d (they act after it). The
//! generator's first-order part is then −Herm(Σ_c V_c†ΔR_c) − (i/2) Σ_j (L_j†ΔL_j + ΔL_j†L_j) =
//! ½[P_d κ̂_d + κ̂_d Q_d] (+ the recoil part), with P_d = −V†ΔR^d − i Σ L†ΔL^d and
//! Q_d = −ΔR^d†V − i Σ ΔL^d†L, which is
//!
//! ```text
//! W_d κ̂_d − (i/2) ∂_d Q_d,     W_d = (P_d + Q_d)/2 = ΔH_ls^d − (i/2) ΔΓ^d   (ground.rs `first_order_at`)
//! ```
//!
//! The local part, the recoil increment and −(i/2)∂_dQ_d (central differences), joins H_eff in
//! the local step's exponential; W_d κ̂_d is applied with the spectral derivative (`engine`). The
//! jumps carry their increments, ΔR_c, in the same order.
//!
//! # Along the radius
//! Every focused beam runs along the plane's normal, so the intensities -- and with them H(r), the
//! elimination's K and E_b and every beam's R_l, R2_l -- depend on ρ = |r| alone; the beams'
//! phases a_l(r) carry the rest. The eliminations are made on a uniform radial grid and
//! interpolated (Catmull-Rom, the grid refined until the midpoints agree with the exact
//! elimination to `RADIAL_TOLERANCE`); the GPU combines the same radial tables with a_l(r) at
//! every point (`Radial`).
use atom_motion_core::ground::{Elimination,Ground};
use atom_motion_core::linear::Dense;
use atom_motion_core::{Complex,Request};
use rayon::prelude::*;

pub const HBAR:f64=1.0545718176461565e-34;
pub const BOLTZMANN:f64=1.380649e-23;
/// The step of the central differences of Q_d (um): its truncation (h k)² ≈ 1e-6 relative for
/// the beams' standing waves, its roundoff 1e-12.
const DIFFERENCE_STEP:f64=1e-4;
/// The radial interpolation's largest error relative to the responses' scale, below single
/// precision.
pub const RADIAL_TOLERANCE:f64=1e-8;

/// A square grid of the plane: nx × ny points spaced dx (um), x_i = (i − nx/2) dx, so the focus is
/// the point (nx/2, ny/2). One row (ny = 1) is the line y = 0.
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Grid{pub nx:usize,pub ny:usize,pub dx:f64}
impl Grid{
    pub fn points(&self)->usize{self.nx*self.ny}
    pub fn x(&self,i:usize)->f64{(i as f64-(self.nx/2) as f64)*self.dx}
    pub fn y(&self,j:usize)->f64{if self.ny==1{0.}else{(j as f64-(self.ny/2) as f64)*self.dx}}
    /// The point of index p (row-major: p = j nx + i).
    pub fn position(&self,p:usize)->[f64;3]{[self.x(p%self.nx),self.y(p/self.nx),0.]}
    /// The wave number (rad/um) of frequency m of n points, m in FFT order.
    pub fn wave_number(&self,m:usize,n:usize)->f64{let s=if m<n/2{m as f64}else{m as f64-n as f64};std::f64::consts::TAU*s/(n as f64*self.dx)}
    /// Half the box (um).
    pub fn half_width(&self)->f64{self.nx as f64*self.dx/2.}
    /// The farthest any point (and its difference stencil) lies from the focus (um).
    pub fn reach(&self)->f64{let x=self.half_width();let y=if self.ny==1{0.}else{x};(x*x+y*y).sqrt()+2.*DIFFERENCE_STEP}
}

/// The 2D model of a request: its front end and the plane's constants.
pub struct Model{
    pub ground:Ground,
    /// ħ/m (um²/us).
    pub hbar_over_mass:f64,
    /// Each beam's wave vector in the plane (rad/um).
    pub in_plane:Vec<[f64;2]>,
    /// Each block's size and its states as indices into `ground.ground_states`.
    pub block_states:[Vec<usize>;2],
    pub radial:Radial,
}

/// The eliminations along the radius: at ρ_k = k `step`, k = 0..=K (um).
pub struct Radial{pub step:f64,pub eliminations:Vec<Elimination>}
/// Catmull-Rom weights of the samples k − 1 .. k + 2 at t ∈ [0, 1) past sample k.
pub fn catmull_rom(t:f64)->[f64;4]{let(t2,t3)=(t*t,t*t*t);[(-t3+2.*t2-t)/2.,(3.*t3-5.*t2+2.)/2.,(-3.*t3+4.*t2+t)/2.,(t3-t2)/2.]}
fn mix(parts:&[(&Elimination,f64)])->Elimination{
    let(e0,_)=parts[0];
    let dense=|f:&dyn Fn(&Elimination)->&Dense|{let mut m=Dense::zeros(f(e0).rows,f(e0).cols);for(e,w)in parts{m.add_scaled(Complex::new(*w,0.),f(e));}m};
    Elimination{energies:(0..e0.energies.len()).map(|b|parts.iter().map(|(e,w)|w*e.energies[b]).sum()).collect(),
        hamiltonian:dense(&|e|&e.hamiltonian),coupling:parts.iter().map(|(e,w)|w*e.coupling).sum(),excited_hamiltonian:dense(&|e|&e.excited_hamiltonian),
        responses:(0..e0.responses.len()).map(|l|dense(&|e|&e.responses[l])).collect(),derivatives:(0..e0.derivatives.len()).map(|l|dense(&|e|&e.derivatives[l])).collect()}
}
impl Radial{
    /// The eliminations of `ground` out to `reach` (um), the grid refined from w/64 (w the
    /// smallest waist) until the midpoints agree to RADIAL_TOLERANCE.
    pub fn new(ground:&Ground,reach:f64)->Result<Radial,String>{
        let exact=|rho:f64|ground.eliminate(&ground.hamiltonian_at([rho,0.,0.]));
        let waist=ground.beams.iter().filter_map(|b|b.waist_um).fold(f64::INFINITY,f64::min);
        let mut step=if waist.is_finite(){waist/64.}else{reach.max(1.)};
        loop{
            let count=(reach/step).ceil() as usize+3;
            let eliminations=(0..count).into_par_iter().map(|k|exact(k as f64*step)).collect::<Result<Vec<_>,_>>()?;
            let radial=Radial{step,eliminations};
            let scale=radial.eliminations.iter().flat_map(|e|e.responses.iter().flat_map(|m|m.entries.iter().map(|z|z.abs()))).fold(0.,f64::max).max(f64::MIN_POSITIVE);
            let errors=(0..count-3).into_par_iter().map(|k|->Result<f64,String>{
                let rho=(k as f64+0.5)*step;let(a,b)=(exact(rho)?,radial.at(rho));
                Ok(a.responses.iter().zip(&b.responses).flat_map(|(x,y)|x.entries.iter().zip(&y.entries).map(|(u,v)|(*u-*v).abs())).fold(0.,f64::max))
            }).collect::<Result<Vec<f64>,String>>()?;
            let error=errors.into_iter().fold(0.,f64::max)/scale;
            if error<=RADIAL_TOLERANCE||step<waist/4096.{return Ok(radial);}
            step/=2.;
        }
    }
    /// The elimination at radius ρ, interpolated (even in ρ about the focus).
    pub fn at(&self,rho:f64)->Elimination{
        let x=rho.abs()/self.step;let k=(x.floor() as usize).min(self.eliminations.len()-3);let t=x-k as f64;
        let w=catmull_rom(t);
        let sample=|j:isize|->&Elimination{&self.eliminations[(k as isize+j).unsigned_abs()]};
        mix(&[(sample(-1),w[0]),(sample(0),w[1]),(sample(1),w[2]),(sample(2),w[3])])
    }
    /// The largest radius tabulated with all four samples around it.
    pub fn reach(&self)->f64{(self.eliminations.len()-3) as f64*self.step}
}

/// One grid point's operators, in double precision: per block the local generator (H_eff with
/// the recoil and ordering parts, nk × nk) and W_x, W_y; each ground state's energy E_b (the trap).
pub struct Point{
    pub generator:[Dense;2],
    pub doppler:[[Dense;2];2],
    pub energies:Vec<f64>,
}

fn combine(beams:&[usize],weights:&[Complex],matrices:&[Dense])->Dense{
    let mut out=Dense::zeros(matrices[0].rows,matrices[0].cols);for &l in beams{out.add_scaled(weights[l],&matrices[l]);}out
}
fn block_of(m:&Dense,states:&[usize])->Dense{Dense::from_fn(states.len(),states.len(),|i,j|m[(states[i],states[j])])}

impl Model{
    pub fn new(request:&Request)->Result<Model,String>{
        let ground=Ground::new(request)?;
        if ground.blocks.len()!=2||ground.blocks[0].states.len()!=3||ground.blocks[1].states.len()!=5{
            return Err("2D 运动需要 Rb87 基态的两个超精细块（F=1 与 F=2）".into());}
        let in_plane=ground.beams.iter().map(|b|[b.wave_number_per_um*b.direction[0],b.wave_number_per_um*b.direction[1]]).collect();
        let block_states=[ground.blocks[0].states.clone(),ground.blocks[1].states.clone()];
        if let Some(b)=ground.beams.iter().find(|b|b.waist_um.is_some()&&(b.direction[0]!=0.||b.direction[1]!=0.)){
            return Err(format!("2D 径向运动要求每束聚焦光都垂直于运动平面：「{}」不是",b.name));}
        Ok(Model{hbar_over_mass:HBAR/request.mass_kg*1e6,in_plane,block_states,radial:Radial{step:1.,eliminations:vec![]},ground})
    }
    /// Tabulates the eliminations out to `reach` (um), the farthest point of any grid used.
    pub fn tabulate(&mut self,reach:f64)->Result<(),String>{
        if self.radial.eliminations.is_empty()||self.radial.reach()<reach{self.radial=Radial::new(&self.ground,reach)?;}Ok(())
    }
    /// The elimination at r (interpolated along the radius).
    pub fn elimination(&self,r:[f64;3])->Elimination{
        assert!(!self.radial.eliminations.is_empty(),"Model::tabulate first");self.radial.at((r[0]*r[0]+r[1]*r[1]+r[2]*r[2]).sqrt())
    }
    /// The first-order shifts w_l per unit of κ_d (d = 0, 1) and the recoil shift (d = 2), rad/us.
    fn shift(&self,d:usize,l:usize)->f64{
        let k=self.in_plane[l];
        if d<2{self.hbar_over_mass*k[d]}else{self.hbar_over_mass*(k[0]*k[0]+k[1]*k[1])/2.}
    }
    /// Q_d at r: −Σ_c secular(ΔR_c^d†V_c) − i Σ_j ΔL_j^d†L_j.
    fn ordering(&self,e:&Elimination,r:[f64;3],d:usize)->Dense{
        let g=&self.ground;let a=g.amplitudes_at(r);let ng=g.ground_states.len();
        let weights:Vec<Complex>=a.iter().enumerate().map(|(l,&z)|z*(-self.shift(d,l))).collect();
        let mut q=Dense::zeros(ng,ng);
        for c in &g.carriers{
            let dr=combine(&c.beams,&weights,&e.derivatives);let v=combine(&c.beams,&a,&g.couplings);
            q.add_scaled(-Complex::ONE,&dr.adjoint_product(&v));
        }
        let q=Dense::from_fn(ng,ng,|i,j|if g.block[i]==g.block[j]{q[(i,j)]}else{Complex::ZERO});
        let local=g.local_at(e,r);let first=g.first_order_at(e,r,|l|self.shift(d,l));
        let mut out=q;for(l,dl)in local.jumps.iter().zip(&first.jumps){out.add_scaled(-Complex::I,&dl.adjoint_product(l));}
        out
    }
    /// The operators at r, with the absorbing potential `cap` (1/us) there.
    pub fn point(&self,r:[f64;3],cap:f64)->Point{
        let g=&self.ground;
        let e=self.elimination(r);
        let local=g.local_at(&e,r);
        let mut generator=local.effective_hamiltonian(&e);
        let increment=|e:&Elimination,d:usize|{let f=g.first_order_at(e,r,|l|self.shift(d,l));let mut m=f.light_shift;m.add_scaled(Complex::new(0.,-0.5),&f.scattering);m};
        generator.add_scaled(Complex::ONE,&increment(&e,2));
        // −(i/2) Σ_d ∂_d Q_d, by central differences.
        for d in 0..2{
            let mut plus=r;plus[d]+=DIFFERENCE_STEP;let mut minus=r;minus[d]-=DIFFERENCE_STEP;
            let mut derivative=self.ordering(&self.elimination(plus),plus,d);derivative.add_scaled(-Complex::ONE,&self.ordering(&self.elimination(minus),minus,d));
            generator.add_scaled(Complex::new(0.,-0.5/(2.*DIFFERENCE_STEP)),&derivative);
        }
        for b in 0..generator.rows{generator[(b,b)]+=Complex::new(0.,-0.5*cap);}
        let doppler=[increment(&e,0),increment(&e,1)];
        Point{generator:std::array::from_fn(|k|block_of(&generator,&self.block_states[k])),
            doppler:std::array::from_fn(|k|[block_of(&doppler[0],&self.block_states[k]),block_of(&doppler[1],&self.block_states[k])]),
            energies:e.energies}
    }
    /// Every grid point's operators, in parallel.
    pub fn points(&self,grid:&Grid,cap:&(dyn Fn([f64;3])->f64+Sync))->Vec<Point>{
        (0..grid.points()).into_par_iter().map(|p|{let r=grid.position(p);self.point(r,cap(r))}).collect()
    }
    /// The trap frequency (rad/us) along plane axis d at the focus of ground state b.
    pub fn trap_frequency(&self,b:usize,d:usize)->f64{(self.ground.curvature_along(b,d)*self.hbar_over_mass).max(0.).sqrt()}
}

/// exp(−i H τ) of a small complex matrix: scaling and squaring of the Taylor series, summed to
/// double precision.
pub fn propagator(h:&Dense,tau:f64)->Dense{
    let n=h.rows;let mut a=h.scaled(Complex::new(0.,-tau));
    let norm=a.entries.iter().map(|z|z.abs()).fold(0.,f64::max)*n as f64;
    let squarings=if norm>0.5{(norm/0.5).log2().ceil() as i32}else{0};
    a=a.scaled(Complex::new(0.5f64.powi(squarings),0.));
    let identity=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});
    let mut out=identity.clone();let mut term=identity;
    for k in 1..40{
        term=term.product(&a).scaled(Complex::new(1./k as f64,0.));out.add_scaled(Complex::ONE,&term);
        if term.entries.iter().all(|z|z.abs()<=1e-18){break;}
    }
    for _ in 0..squarings{out=out.product(&out);}
    out
}

