//! The request: a single atom in an optical tweezer, its internal state and its motion in 3D. It
//! is the scene's rb87-full/2 model with the beams kept apart, their geometry on the lab axes and
//! the spontaneous emission resolved on them, as the TypeScript model writes it
//! (src/full-master-model.ts `MotionModel`, the only owner of atomic physics).
//!
//! The atom starts in the internal density `initial_density`, its motion thermal at
//! `temperature_uk`. At the position r (from the trap's focus) with momentum p = m v, the
//! Hamiltonian in the model's frame is
//!
//! ```text
//! H = p²/2m + H0 + Σ_beams [ a(r) X e^{−i 2π Δ t} + a(r)* R e^{+i 2π (Δ + 2 ν_frame) t} + h.c. ]
//! ```
//!
//! with each beam's co- and counter-rotating couplings X and R at the focus, its amplitude a(r)
//! relative to the focus's (`Beam::amplitude_at`), and Δ = ν − ν_frame; ν_frame is the excited
//! states' frame frequency. Spontaneous emission is Σ_c D[C_c] over the three Cartesian components
//! of the lowering dipole (`emission`): the photon of C_c has the angular pattern of a dipole along
//! c, of the wave number k_eg = 2π ν_eg / c of its transition (`Request::frame_frequencies_mhz`).

/// The internal states: the 32 Rb87 states of the model basis.
pub const STATES:usize=32;

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
    /// The unit wave vector on the lab axes x, y, z (the order of `Request::emission`).
    pub direction:[f64;3],
    /// |k| = 2πν/c (rad/um).
    pub wave_number_per_um:f64,
    /// A Gaussian beam focused at the trap's focus: its waist, the 1/e² intensity radius at the
    /// focus (um). None: a plane wave, of one intensity everywhere.
    pub waist_um:Option<f64>,
    /// The intensity at the focus (mW/cm²), a Gaussian beam's peak: the intensity X and R are given at.
    pub intensity_mw_cm2:f64,
    /// Δ = ν − ν_frame (MHz) = Σ_k frequency_index[k] · `Request::frequencies_mhz`[k]: exact
    /// integer coordinates, no beat is rounded.
    pub frequency_index:Vec<i32>,
    /// X, the co-rotating excited ← ground coupling at the focus (rad/us) in the model basis.
    pub coupling:Matrix,
    /// R, the counter-rotating coupling at the focus (rad/us).
    pub counter_coupling:Matrix,
    /// O, the beam's static shift through the states outside the model and the Rb⁺ core at the focus
    /// (rad/us, Hermitian): its share of `Request::hamiltonian`, which scales with |a(r)|².
    pub outside_shift:Matrix,
}
impl Beam{
    /// The beam's complex amplitude at r (um from the focus), relative to its amplitude at the
    /// focus: a plane wave e^{ikζ}, or the paraxial Gaussian beam focused there,
    /// e^{ikζ} exp(−ρ²/(w² q))/q with q = 1 + iζ/z_R, z_R = k w²/2; ζ = r·n is the distance along
    /// the beam and ρ the distance from its axis.
    pub fn amplitude_at(&self,r:[f64;3])->Complex{
        let(k,zeta)=(self.wave_number_per_um,r[0]*self.direction[0]+r[1]*self.direction[1]+r[2]*self.direction[2]);
        let wave=Complex::cis(k*zeta);
        match self.waist_um{
            None=>wave,
            Some(w)=>{
                let q=Complex::new(1.,zeta/(k*w*w/2.));let radial=(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-zeta*zeta).max(0.)/(w*w);
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
    /// shifts through the states outside the model at the focus; diagonal in the model basis.
    pub hamiltonian:Matrix,
    /// The frequency each state's frame rotates at (MHz): 0 for the ground states, ν_frame for
    /// the excited ones. A photon emitted e → g has the frequency
    /// (H0_ee − H0_gg)/2π + frame_e − frame_g.
    pub frame_frequencies_mhz:Vec<f64>,
    /// The beams' frequency axes (MHz).
    pub frequencies_mhz:Vec<f64>,
    pub beams:Vec<Beam>,
    /// The trap beam (an index into `beams`): the tweezer whose focus is the origin.
    pub trap_beam:usize,
    /// The lowering dipole's Cartesian components on the lab axes x, y, z (1/sqrt(us)):
    /// Σ_c D[C_c] is the model's spontaneous emission.
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
}

#[cfg(test)]
mod tests{
    use super::*;
    fn zero()->Matrix{Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]}}
    fn beam(direction:[f64;3],waist_um:Option<f64>)->Beam{
        Beam{id:"beam".into(),name:"光".into(),direction,wave_number_per_um:7.37,waist_um,intensity_mw_cm2:1.,frequency_index:vec![],coupling:zero(),counter_coupling:zero(),outside_shift:zero()}
    }
    /// A plane wave's amplitude is its phase along the beam; a beam focused at the origin falls off
    /// across its axis as exp(−ρ²/w²) (intensity exp(−2ρ²/w²)) with no phase; along its own axis it
    /// is the Gaussian beam's on-axis field, with the Gouy phase.
    #[test]
    fn a_beam_s_amplitude_is_a_plane_wave_s_or_a_gaussian_beam_s(){
        let x=0.83;let k=7.37;let on=|a:usize|{let mut r=[0.;3];r[a]=x;r};
        let plane=beam([0.6,0.,0.8],None).amplitude_at(on(2));let expected=Complex::cis(k*0.8*x);
        assert!((plane-expected).abs()<=1e-15,"{plane:?}");
        let across=beam([0.,1.,0.],Some(1.2)).amplitude_at(on(2));
        assert!((across.re-(-x*x/1.44_f64).exp()).abs()<=1e-15&&across.im==0.,"{across:?}");
        let rayleigh=k*1.44/2.;let along=beam([0.,0.,1.],Some(1.2)).amplitude_at(on(2));
        let gouy=(x/rayleigh).atan();let size=1./(1.+(x/rayleigh).powi(2)).sqrt();
        assert!((along-Complex::cis(k*x-gouy)*size).abs()<=1e-15,"{along:?}");
    }
}
