//! The Rb87 motion core: the 32-state atom moving along one lab axis, its internal state and
//! its motion evolved together (1D motion, ui-design.md §6.20). The request is the scene's
//! rb87-full/2 model with the beams kept apart and the spontaneous emission resolved on the
//! axis, as the TypeScript model writes it (src/full-master-model.ts `MotionModel`, the only
//! owner of atomic physics); nothing physical is introduced here.
//!
//! The atom starts in the internal density `initial_density` with a Maxwellian momentum
//! distribution at `temperature_uk` along the axis and a uniform position. Along the axis
//! (position x, momentum p = m v) the Hamiltonian in the model's frame is
//!
//! ```text
//! H = p²/2m + H0 + Σ_beams [ X e^{+i a k x} e^{−i 2π Δ t} + R e^{−i a k x} e^{+i 2π (Δ + 2 ν_frame) t} + h.c. ]
//! ```
//!
//! with each beam's axial projection a, wave number k, co- and counter-rotating couplings X and R,
//! and Δ = ν − ν_frame (`Beam`); ν_frame is the excited states' frame frequency. Spontaneous
//! emission is Σ_c D[C_c] over the three Cartesian components of the lowering dipole (`emission`):
//! the photon of C_c has the angular pattern of a dipole along c, so the atom's recoil along the
//! axis is −ħ k_eg u with u, the cosine of the photon's angle to the axis, distributed as
//! (3/8)(1 + u²) for the two components perpendicular to the axis and (3/4)(1 − u²) for the one
//! along it; k_eg = 2π ν_eg / c, ν_eg the photon frequency of the transition
//! (`Request::frame_frequencies_mhz`).
//!
//! `solve` is the engine. It is not implemented yet: it serves exact free flight -- no beam,
//! and an internal state the model leaves as it is -- and refuses everything else with
//! `NOT_IMPLEMENTED`. Its front end is `ground`: the ground manifold's generator with the excited
//! states eliminated (GMW-o1), on the small dense linear algebra of `linear`.
mod free_flight;
pub mod ground;
pub mod linear;

/// The internal states: the 32 Rb87 states of the model basis.
pub const STATES:usize=32;
/// Why a request beyond free flight is refused, until the engine is implemented.
pub const NOT_IMPLEMENTED:&str="运动引擎尚未实现";

#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct Complex{pub re:f64,pub im:f64}

/// A complex `STATES` × `STATES` matrix, row-major.
#[derive(Clone,Debug,PartialEq)]
pub struct Matrix{pub real:Vec<f64>,pub imag:Vec<f64>}

/// One laser beam. Beams are kept apart: two counter-propagating beams of one frequency are two
/// beams, each with its own spatial phase.
#[derive(Clone,Debug,PartialEq)]
pub struct Beam{
    pub id:String,
    /// k·axis/|k|: +1 or −1, a travelling wave along or against the axis; 0, a beam
    /// perpendicular to the axis, uniform along it.
    pub axial:i8,
    /// |k| = 2πν/c (rad/um).
    pub wave_number_per_um:f64,
    /// Δ = ν − ν_frame (MHz) = Σ_k frequency_index[k] · `Request::frequencies_mhz`[k]: exact
    /// integer coordinates, no beat is rounded.
    pub frequency_index:Vec<i32>,
    /// X, the co-rotating excited ← ground coupling (rad/us) in the model basis: the beam drives
    /// X e^{+i axial |k| x} e^{−i 2π Δ t} + h.c.
    pub coupling:Matrix,
    /// R, the counter-rotating coupling (rad/us): R e^{−i axial |k| x} e^{+i 2π (Δ + 2ν_frame) t} + h.c.
    pub counter_coupling:Matrix,
}

/// A motion solve: the model, the initial state and the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Request{
    /// The model basis: 32 labels `manifold-F-m` (the eigenstates of the static H0).
    pub states:Vec<String>,
    /// H0 in the frame (rad/us) without any beam: the magnetic field and the beams' second-order
    /// shifts through the states outside the model; diagonal in the model basis.
    pub hamiltonian:Matrix,
    /// The frequency each state's frame rotates at (MHz): 0 for the ground states, ν_frame for
    /// the excited ones. A photon emitted e → g has the frequency
    /// (H0_ee − H0_gg)/2π + frame_e − frame_g.
    pub frame_frequencies_mhz:Vec<f64>,
    /// The beams' frequency axes (MHz).
    pub frequencies_mhz:Vec<f64>,
    pub beams:Vec<Beam>,
    /// The lowering dipole's Cartesian components (1/sqrt(us)): two perpendicular to the axis,
    /// then the one along it. Σ_c D[C_c] is the model's spontaneous emission.
    pub emission:[Matrix;3],
    /// The magnetic field (G, lab frame); it is in H0 already.
    pub field_vector_g:[f64;3],
    pub mass_kg:f64,
    /// The initial internal density (row-major, trace one).
    pub initial_density:Vec<Complex>,
    /// The initial Maxwellian's temperature (μK).
    pub temperature_uk:f64,
    /// The output times (us), increasing from 0.
    pub times_us:Vec<f64>,
    /// The absolute population error the outputs are to reach.
    pub tolerance:f64,
}

/// Momentum distributions along the axis.
#[derive(Clone,Debug,PartialEq)]
pub struct Momentum{
    /// The grid: velocities (m/s), strictly increasing.
    pub velocities_mps:Vec<f64>,
    /// The time indices the distributions are given at: increasing, the last time among them.
    pub samples:Vec<usize>,
    /// One distribution per listed time: the probability at each grid velocity, nonnegative and
    /// summing to one.
    pub probabilities:Vec<Vec<f64>>,
}

/// A motion solve's result, at every time of the request.
#[derive(Clone,Debug,PartialEq)]
pub struct Solution{
    /// The internal populations averaged over position: one row of `STATES` probabilities
    /// (nonnegative, summing to one) per time.
    pub populations:Vec<Vec<f64>>,
    /// Per time, the estimated absolute error of every population and every F group.
    pub sample_errors:Vec<f64>,
    /// Per time, T = m·Var(v)/k_B along the axis (μK).
    pub temperature_uk:Vec<f64>,
    /// Per time, the mean velocity along the axis (m/s).
    pub mean_velocity_mps:Vec<f64>,
    pub momentum:Momentum,
    /// The method, one sentence.
    pub method:String,
    /// Why the run stopped with errors above the tolerance (none when it reached it).
    pub reason:Option<String>,
}

/// Solves a request. `checkpoint` is given each intermediate solution as it is made -- complete,
/// at every time of the request, as the final one -- so that a stopped run keeps the last; an
/// engine without such intermediates gives none. The engine is not implemented yet: exact free
/// flight is served, anything else is refused with `NOT_IMPLEMENTED`.
pub fn solve(request:&Request,_checkpoint:&mut dyn FnMut(&Solution))->Result<Solution,String>{
    free_flight::solve(request).ok_or_else(||NOT_IMPLEMENTED.to_string())
}

#[cfg(test)]
mod tests{
    use super::*;
    fn zero()->Matrix{Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]}}
    /// The first 8 states are the ground states; H0 diagonal; one decay channel per excited state.
    fn request(initial:&[(usize,f64)])->Request{
        let states:Vec<String>=(0..STATES).map(|i|if i<8{format!("ground-{}-{}",1+i/3,i%3)}else{format!("D2-{}-{}",i/8,i%8)}).collect();
        let mut hamiltonian=zero();for i in 0..STATES{hamiltonian.real[i*STATES+i]=0.1*i as f64;}
        let mut decay=zero();for e in 8..STATES{decay.real[(e%8)*STATES+e]=6.;}
        let mut density=vec![Complex::default();STATES*STATES];for &(i,p) in initial{density[i*STATES+i].re=p;}
        Request{states,hamiltonian,frame_frequencies_mhz:(0..STATES).map(|i|if i<8{0.}else{3.8e8}).collect(),frequencies_mhz:vec![],beams:vec![],
            emission:[zero(),zero(),decay],field_vector_g:[0.,0.,1.],mass_kg:1.443e-25,initial_density:density,temperature_uk:30.,
            times_us:vec![0.,5.,10.],tolerance:0.005}
    }
    #[test]
    fn free_flight_keeps_the_populations_the_temperature_and_the_maxwellian(){
        let r=request(&[(2,0.25),(5,0.75)]);
        let s=solve(&r,&mut |_|panic!("free flight has no intermediate result")).unwrap();
        assert!(s.populations.iter().all(|row|row[2]==0.25&&row[5]==0.75&&row.iter().sum::<f64>()==1.));
        assert_eq!((s.sample_errors.clone(),s.temperature_uk.clone(),s.mean_velocity_mps.clone()),(vec![0.;3],vec![30.;3],vec![0.;3]));
        let m=&s.momentum;assert_eq!(m.samples,vec![0,2]);
        assert!(m.velocities_mps.windows(2).all(|w|w[1]>w[0]));
        // Each distribution sums to one, its mean is zero and its temperature is the request's.
        let kelvin=r.mass_kg/1.380649e-23*1e6;
        for p in &m.probabilities{
            assert!((p.iter().sum::<f64>()-1.).abs()<1e-15);
            let mean:f64=p.iter().zip(&m.velocities_mps).map(|(p,v)|p*v).sum();
            let variance:f64=p.iter().zip(&m.velocities_mps).map(|(p,v)|p*v*v).sum();
            assert!(mean.abs()<1e-15&&(variance*kelvin/30.-1.).abs()<1e-13,"mean {mean}, T {} uK",variance*kelvin);
        }
    }
    #[test]
    fn anything_but_free_flight_is_not_implemented_yet(){
        // An excited population decays; a beam drives.
        let excited=request(&[(2,0.5),(9,0.5)]);
        assert_eq!(solve(&excited,&mut |_|{}),Err(NOT_IMPLEMENTED.to_string()));
        let mut driven=request(&[(2,1.)]);
        driven.beams.push(Beam{id:"cooling".into(),axial:1,wave_number_per_um:8.05,frequency_index:vec![],coupling:zero(),counter_coupling:zero()});
        assert_eq!(solve(&driven,&mut |_|{}),Err(NOT_IMPLEMENTED.to_string()));
    }
}
