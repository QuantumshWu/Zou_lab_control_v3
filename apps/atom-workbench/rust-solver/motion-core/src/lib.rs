//! The Rb87 motion core: a single atom in an optical tweezer, its internal state and its radial
//! motion evolved together (1D motion, ui-design.md §6.20). The atom moves on a line through the
//! tweezer focus -- the atom's position -- perpendicular to every beam focused there: a lab axis,
//! the motion axis. The request is the scene's rb87-full/2 model with the beams kept apart, their
//! geometry on the line and the spontaneous emission resolved on the axis, as the TypeScript model
//! writes it (src/full-master-model.ts `MotionModel`, the only owner of atomic physics).
//!
//! The atom starts in the internal density `initial_density`, its motion along the line thermal
//! at `temperature_uk`. On the line (position x from the focus, momentum p = m v) the Hamiltonian
//! in the model's frame is
//!
//! ```text
//! H = p²/2m + H0 + Σ_beams [ a(x) X e^{−i 2π Δ t} + a(x)* R e^{+i 2π (Δ + 2 ν_frame) t} + h.c. ]
//! ```
//!
//! with each beam's co- and counter-rotating couplings X and R at the atom, its amplitude a(x)
//! relative to the atom's (`Beam::amplitude`), and Δ = ν − ν_frame; ν_frame is the excited
//! states' frame frequency. Spontaneous emission is Σ_c D[C_c] over the three Cartesian
//! components of the lowering dipole (`emission`): the photon of C_c has the angular pattern of a
//! dipole along c, so the atom's recoil along the axis is −ħ k_eg u with u, the cosine of the
//! photon's angle to the axis, distributed as (3/8)(1 + u²) for the two components perpendicular
//! to the axis and (3/4)(1 − u²) for the one along it; k_eg = 2π ν_eg / c, ν_eg the photon
//! frequency of the transition (`Request::frame_frequencies_mhz`).
//!
//! `solve` runs the engine (`engine`: GMW-o1 in a box, continued by dynamic mode decomposition).
//! Its front end is `ground`: the ground manifold's generator with the excited states eliminated,
//! evaluated at any position on the line, on the small dense linear algebra of `linear`. A
//! request without light is refused with `NO_TRAP`, one whose initial state the light does not
//! confine along the axis with `NOT_CONFINED`.
pub mod engine;
pub mod ground;
pub mod linear;

/// The internal states: the 32 Rb87 states of the model basis.
pub const STATES:usize=32;
/// Why a request without light is refused: nothing confines the atom along the axis.
pub const NO_TRAP:&str="沿运动轴没有约束原子的势阱（没有光束）：1D 运动计算光镊中单个原子的径向运动";
/// Why a request is refused whose light does not confine the initial state along the axis: the
/// potential's curvature at the focus is not positive, or it has no depth.
pub const NOT_CONFINED:&str="初态沿运动轴不受约束（焦点处的势阱曲率或深度不为正）：1D 运动计算光镊中单个原子的径向运动";

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
    /// The laser's name, as the scene shows it.
    pub name:String,
    /// The unit wave vector in the motion frame: its components along the two axes perpendicular to
    /// the motion axis and along the motion axis (the order of `Request::emission`).
    pub direction:[f64;3],
    /// |k| = 2πν/c (rad/um).
    pub wave_number_per_um:f64,
    /// A beam focused at the atom: its waist, the 1/e² intensity radius at the focus (um). None: a
    /// plane wave, of one intensity everywhere.
    pub waist_um:Option<f64>,
    /// The intensity at the atom (mW/cm²), a focused beam's peak: the intensity X and R are given at.
    pub intensity_mw_cm2:f64,
    /// Δ = ν − ν_frame (MHz) = Σ_k frequency_index[k] · `Request::frequencies_mhz`[k]: exact
    /// integer coordinates, no beat is rounded.
    pub frequency_index:Vec<i32>,
    /// X, the co-rotating excited ← ground coupling at the atom (rad/us) in the model basis.
    pub coupling:Matrix,
    /// R, the counter-rotating coupling at the atom (rad/us).
    pub counter_coupling:Matrix,
    /// O, the beam's static shift through the states outside the model and the Rb⁺ core at the atom
    /// (rad/us, Hermitian): its share of `Request::hamiltonian`, which scales with |a(x)|².
    pub outside_shift:Matrix,
}
impl Beam{
    /// d²|a(x)|²/dx² at the focus (1/um²): −4 ρ̂²/w² − 2 n_axis²/z_R² for the Gaussian beam, ρ̂² the
    /// squared share of the line across the beam; 0 for a plane wave.
    pub fn intensity_curvature(&self)->f64{
        let[across_1,across_2,along]=self.direction;
        self.waist_um.map_or(0.,|w|{let rayleigh=self.wave_number_per_um*w*w/2.;-4.*(across_1*across_1+across_2*across_2)/(w*w)-2.*along*along/(rayleigh*rayleigh)})
    }
    /// The wave number along the motion axis, k·axis (rad/um).
    pub fn axial_wave_number(&self)->f64{self.wave_number_per_um*self.direction[2]}
    /// The beam's complex amplitude at x (um) on the motion line, relative to its amplitude at the
    /// atom: a plane wave e^{ikζ}, or the paraxial Gaussian beam focused at the atom,
    /// e^{ikζ} exp(−ρ²/(w² q))/q with q = 1 + iζ/z_R, z_R = k w²/2; ζ = x n_axis is the distance
    /// along the beam from the focus and ρ the distance from the beam's axis.
    pub fn amplitude(&self,x:f64)->Complex{
        let[across_1,across_2,along]=self.direction;let(k,zeta)=(self.wave_number_per_um,x*along);
        let wave=Complex::cis(k*zeta);
        match self.waist_um{
            None=>wave,
            Some(w)=>{
                let q=Complex::new(1.,zeta/(k*w*w/2.));let radial=x*x*(across_1*across_1+across_2*across_2)/(w*w);
                wave*(Complex::new(-radial,0.)/q).exp()/q
            }
        }
    }
}

/// A motion solve: the model, the initial state and the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Request{
    /// The model basis: 32 labels `manifold-F-m` (the eigenstates of the static H0).
    pub states:Vec<String>,
    /// H0 in the frame (rad/us) without any beam: the magnetic field and the beams' second-order
    /// shifts through the states outside the model at the atom; diagonal in the model basis.
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
    /// The initial motion's temperature (μK).
    pub temperature_uk:f64,
    /// The output times (us), increasing from 0.
    pub times_us:Vec<f64>,
    /// The absolute population error the outputs are to reach.
    pub tolerance:f64,
}

/// Distributions along the axis on one grid, at some of the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Distributions{
    /// The grid, strictly increasing.
    pub grid:Vec<f64>,
    /// The time indices the distributions are given at: increasing, the last time among them.
    pub samples:Vec<usize>,
    /// One distribution per listed time: the probability at each grid point, nonnegative and
    /// summing to one.
    pub probabilities:Vec<Vec<f64>>,
}

/// A motion solve's result, at every time of the request. Everything but `loss_probability`
/// describes the atom still held: given that it has not been lost.
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
    /// ω/2π (kHz), the trap frequency at the focus along the axis that `mean_occupation` is relative to.
    pub trap_frequency_khz:f64,
    /// Per time, n̄ = (⟨p²⟩/2m + mω²⟨x²⟩/2)/ħω − 1/2: the mean occupation of the harmonic levels of
    /// that frequency.
    pub mean_occupation:Vec<f64>,
    /// Per time, the probability that the atom has left the trap by then.
    pub loss_probability:Vec<f64>,
    /// Velocity distributions (grid in m/s) and position distributions (grid in um, from the focus).
    pub momentum:Distributions,
    pub position:Distributions,
    /// The method, one sentence.
    pub method:String,
    /// Why the run stopped with errors above the tolerance (none when it reached it).
    pub reason:Option<String>,
}

/// Solves a request. `checkpoint` is given each intermediate solution as it is made -- complete,
/// at every time of the request, as the final one -- so that a stopped run keeps the last, with
/// the estimated wall time to the final one (ms).
pub fn solve(request:&Request,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,String>{
    if request.beams.is_empty(){return Err(NO_TRAP.into());}
    let ground=ground::Ground::new(request)?;
    let lapack=atom_population_core::Lapack::load()?;
    engine::Engine::new(&ground,request,&lapack)?.run(request,checkpoint)
}

#[cfg(test)]
mod tests{
    use super::*;
    fn zero()->Matrix{Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]}}
    fn request()->Request{
        let states:Vec<String>=(0..STATES).map(|i|if i<8{format!("ground-{}-{}",1+i/3,i%3)}else{format!("D2-{}-{}",i/8,i%8)}).collect();
        Request{states,hamiltonian:zero(),frame_frequencies_mhz:(0..STATES).map(|i|if i<8{0.}else{3.8e8}).collect(),frequencies_mhz:vec![],beams:vec![],
            emission:[zero(),zero(),zero()],field_vector_g:[0.;3],mass_kg:1.443e-25,initial_density:vec![Complex::default();STATES*STATES],temperature_uk:30.,
            times_us:vec![0.,5.,10.],tolerance:0.005}
    }
    fn beam(direction:[f64;3],waist_um:Option<f64>)->Beam{
        Beam{id:"beam".into(),name:"光".into(),direction,wave_number_per_um:7.37,waist_um,intensity_mw_cm2:1.,frequency_index:vec![],coupling:zero(),counter_coupling:zero(),outside_shift:zero()}
    }
    #[test]
    fn without_light_or_a_potential_nothing_confines_the_atom(){
        let mut r=request();assert_eq!(solve(&r,&mut |_,_|{}),Err(NO_TRAP.to_string()));
        r.beams.push(beam([0.,1.,0.],Some(1.2)));r.initial_density[0]=Complex{re:1.,im:0.};
        assert_eq!(solve(&r,&mut |_,_|{}),Err(NOT_CONFINED.to_string()));
    }
    /// A plane wave's amplitude on the line is its axial phase; a beam focused at the atom and
    /// perpendicular to the line falls off as exp(−x²/w²) (intensity exp(−2x²/w²)) with no phase;
    /// along its own axis it is the Gaussian beam's on-axis field, with the Gouy phase.
    #[test]
    fn a_beam_s_amplitude_on_the_line_is_a_plane_wave_s_or_a_gaussian_beam_s(){
        let x=0.83;let k=7.37;
        let plane=beam([0.6,0.,0.8],None).amplitude(x);let expected=Complex::cis(k*0.8*x);
        assert!((plane-expected).abs()<=1e-15,"{plane:?}");
        let across=beam([0.,1.,0.],Some(1.2)).amplitude(x);
        assert!((across.re-(-x*x/1.44_f64).exp()).abs()<=1e-15&&across.im==0.,"{across:?}");
        let rayleigh=k*1.44/2.;let along=beam([0.,0.,1.],Some(1.2)).amplitude(x);
        let gouy=(x/rayleigh).atan();let size=1./(1.+(x/rayleigh).powi(2)).sqrt();
        assert!((along-Complex::cis(k*x-gouy)*size).abs()<=1e-15,"{along:?}");
    }
}
