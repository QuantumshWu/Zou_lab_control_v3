//! The moving atom's ground manifold with the excited states eliminated: the generator of the
//! ground-manifold matrix Wigner method, to first order in the Doppler and recoil shifts (GMW-o1).
//!
//! # Elimination
//! The lattice phase is θ = k_ref x, k_ref the wave number of the first beam along the axis (of the
//! first beam when none is), and a beam's direction is s = `Beam::axial`. The beams of one frequency
//! (equal `Beam::frequency_index`) form a carrier c at δ_c = 2π(ν_c − ν_frame); V_{c,s} sums its
//! couplings of direction s, which drive V_{c,s} e^{isθ} e^{−iδ_c t} + h.c. With the ground energies
//! E_b (H0 is diagonal in the model basis) and the excited states' non-Hermitian Hamiltonian
//! K = H0_ee − (i/2) D, D = Σ_a C_a†C_a|_ee, the excited states and the optical coherences follow the
//! ground states adiabatically, column by column:
//!
//! ```text
//! R_{c,s}[:,b] = (K − E_b − δ_c)⁻¹ V_{c,s}[:,b]        R2_{c,s}[:,b] = (K − E_b − δ_c)⁻² V_{c,s}[:,b]
//! ```
//!
//! and the ground density evolves under the Reiter–Sørensen Lindblad generator, a function of θ:
//!
//! ```text
//! dρ/dt = −i(H_eff ρ − ρ H_eff†) + Σ_j L_j ρ L_j†,     H_eff = H0g + H_ls − (i/2) Γ,
//! H_ls = −Herm Σ_c V_c(θ)† R_c(θ),   Γ = Σ_c R_c(θ)† D R_c(θ) = Σ_j L_j† L_j,   L_{a,c} = C_a R_c(θ),
//! ```
//!
//! R_c(θ) = Σ_s R_{c,s} e^{isθ}, V_c(θ) likewise. The photon of jump (a, c) is polarized along a and
//! kicks the atom by −u κ_c ħk_ref along the axis, κ_c = k_c/k_ref and u distributed by the dipole
//! pattern of a (lib.rs). Readout: the bare ground density is ρ − ½{Y, ρ}, Y = Σ_c R_c†R_c, and an
//! excited state's population is tr(E_i ρ), E_i = Σ_c R_c†|i⟩⟨i|R_c (Σ_i E_i = Y).
//!
//! # First order in the Doppler and recoil shifts
//! A ground momentum q (ħk_ref) shifts carrier c's excited state of direction s by the kinetic energy
//! w_{c,s}(q) = w_r (2 s κ_c q + s² κ_c²), w_r = ħk_ref²/2m; to first order R_{c,s} → R_{c,s} − w_{c,s} R2_{c,s}
//! (`Ground::first_order`), and `Ground::response` gives the shifted response to all orders.
//!
//! # Secular step
//! In the frames of the hyperfine blocks, each rotating at its centroid ν_F, a term absorbing from
//! carrier c and emitting into carrier c′ between blocks F and F′ rotates at the Raman detuning
//! Ω = (ν_F′ − ν_F) − (δ_c − δ_c′): the coherences between F = 1 and F = 2 and the beats of two carriers.
//! These are dropped and the static terms kept, so the generator maps block-diagonal densities
//! (F = 2: 25 elements, F = 1: 9) to block-diagonal ones and its jumps split into L^{F′F} = P_F′ L P_F.
//! A dropped term of coupling |g| leaves a micromotion of relative size ε = |g|/|Ω|; the step is
//! taken when every ε is at most the elimination's own expansion parameter
//! η = ‖H_ls − iΓ/2‖ / min σ_min(K − E_b − δ_c), so it is never the dominant error. Otherwise the beams
//! form a Raman-resonant Λ scheme (gray molasses, EIT) and are refused by name. A norm over θ is
//! bounded by the sum of its harmonics' spectral norms.
//!
//! # Approximations and their criteria
//! - The zeroth-order elimination: η ≪ 1. Its error is the saturation, ≈ 2 p_e locally (the
//!   scattering rate +1 to +2.4 % in 1D PGC).
//! - First order in the Doppler shift: (kv / σ_min)² ≪ 1; `response` is exact.
//! - The secular step: ε ≤ η, enforced.
//! - One lattice wave number: the beams along the axis are on the D2 line, |Δk|/k ≤ 2e-5; κ_c keeps
//!   each carrier's photon momentum.
//! - An emitted photon has its carrier's wave number: ≤ 1e-7 on the D2 line.
//! - The counter-rotating couplings are dropped (optical RWA): Ω/ω ≈ 3e-9.
use crate::linear::{Dense,Lu};
use crate::{Complex,Request,STATES};
use std::f64::consts::TAU;

/// ħ (J s): h/2π, h exact in the SI.
const HBAR:f64=1.0545718176461565e-34;

/// An operator on the ground states as a function of the lattice phase: Σ_m terms[m + 2] e^{imθ},
/// m = −2..=2, the difference of two directions. A jump's L(θ) = Σ_s L[s] e^{isθ} has m = s.
#[derive(Clone,Debug,PartialEq)]
pub struct Harmonics(pub [Dense;5]);
impl Harmonics{
    fn zeros(n:usize)->Harmonics{Harmonics(std::array::from_fn(|_|Dense::zeros(n,n)))}
    pub fn get(&self,m:i8)->&Dense{&self.0[(m+2) as usize]}
    fn add(&mut self,m:i8,z:Complex,term:&Dense){self.0[(m+2) as usize].add_scaled(z,term);}
    /// The operator at θ.
    pub fn at(&self,theta:f64)->Dense{
        let mut out=Dense::zeros(self.0[0].rows,self.0[0].cols);
        for m in -2..=2{out.add_scaled(Complex::cis(m as f64*theta),self.get(m));}
        out
    }
    /// Σ_m ‖term_m‖₂, a bound on ‖op(θ)‖₂ at every θ.
    pub fn bound(&self)->f64{self.0.iter().map(Dense::spectral_norm).sum()}
    fn map(&self,f:impl Fn(&Dense)->Dense)->Harmonics{Harmonics(std::array::from_fn(|k|f(&self.0[k])))}
}

/// A hyperfine block of the ground manifold.
#[derive(Clone,Debug)]
pub struct Block{
    pub f:u8,
    /// Its states, as positions in `Ground::ground_states`.
    pub states:Vec<usize>,
    /// ν_F, the centroid of its energies: the block's frame (rad/us).
    pub frame:f64,
}

/// The beams of one frequency.
#[derive(Clone,Debug)]
pub struct Carrier{
    /// Its beams (indices into `Request::beams`).
    pub beams:Vec<usize>,
    /// δ_c = 2π(ν_c − ν_frame) (rad/us).
    pub detuning:f64,
    /// κ_c = k_c/k_ref.
    pub kappa:f64,
    /// Its travelling waves, one per direction, by increasing s.
    pub waves:Vec<Wave>,
    /// Absorption from one wave and stimulated emission into another, every ordered pair.
    pub pairs:Vec<Pair>,
}
/// The beams of one carrier and direction s.
#[derive(Clone,Debug)]
pub struct Wave{
    pub axial:i8,
    /// V_{c,s}, excited ← ground (rad/us).
    pub coupling:Dense,
    /// R_{c,s} and R2_{c,s}, excited ← ground (dimensionless and us).
    pub response:Dense,
    pub derivative:Dense,
}
/// The secular parts of X = V_{c,s′}† R_{c,s} and P = V_{c,s′}† R2_{c,s} (harmonic m = s − s′):
/// H_ls = −Herm Σ X e^{imθ}, and its first-order shift +Herm Σ w_{c,s} P e^{imθ}.
#[derive(Clone,Debug)]
pub struct Pair{pub emitted:i8,pub absorbed:i8,pub x:Dense,pub p:Dense}

/// A quantum jump: a photon polarized along `axis` scattered from carrier `carrier`, the atom going
/// from block `from` to block `to`.
#[derive(Clone,Debug)]
pub struct Jump{
    /// 0 and 1, the axes perpendicular to the motion axis; 2, the motion axis (`Request::emission`).
    pub axis:usize,
    pub carrier:usize,
    pub from:usize,
    pub to:usize,
    /// κ_c: the photon carries κ_c ħk_ref.
    pub kappa:f64,
    /// L[s] = P_to C_a R_{c,s} P_from and its first-order derivative P_to C_a R2_{c,s} P_from (harmonic s).
    pub operator:Harmonics,
    pub derivative:Harmonics,
}

/// The secular step's report.
#[derive(Clone,Debug)]
pub struct Secular{
    /// η = scale / pole, the elimination's expansion parameter.
    pub eta:f64,
    /// Σ_m ‖H_ls[m] − (i/2) Γ[m]‖₂, the scale of the light-induced ground dynamics (rad/us).
    pub scale:f64,
    /// The nearest excited pole: min σ_min(K − E_b − δ_c) over the coupled (c, b) (rad/us).
    pub pole:f64,
    /// Every rotating term the step drops.
    pub dropped:Vec<Rotating>,
}
impl Secular{
    /// The dropped term of the largest micromotion.
    pub fn largest(&self)->Option<&Rotating>{self.dropped.iter().max_by(|a,b|a.micromotion.total_cmp(&b.micromotion))}
}
/// A rotating term: light absorbed from carrier `carriers[0]` and emitted into `carriers[1]` (by
/// stimulated emission, or as the other photon of a pair of jumps), between blocks `blocks[0]` and
/// `blocks[1]`.
#[derive(Clone,Debug)]
pub struct Rotating{
    pub carriers:[usize;2],
    pub blocks:[usize;2],
    /// Ω = (ν_{blocks[1]} − ν_{blocks[0]}) − (δ_{carriers[0]} − δ_{carriers[1]}), the Raman detuning (rad/us).
    pub frequency:f64,
    /// |g|, a bound on the term's norm (rad/us).
    pub coupling:f64,
    /// ε = |g|/|Ω|.
    pub micromotion:f64,
}

/// The first-order increments of the generator for excited-energy shifts w_{c,s}.
#[derive(Clone,Debug)]
pub struct FirstOrder{pub light_shift:Harmonics,pub scattering:Harmonics,pub jumps:Vec<Harmonics>}

/// The eliminated ground-manifold generator of a request: the operators act on the ground states in
/// the order of `ground_states`, the excited ones in the order of `excited_states`; rates in rad/us,
/// jumps in 1/sqrt(us).
#[derive(Clone,Debug)]
pub struct Ground{
    /// The request's ground and excited states (indices into `Request::states`).
    pub ground_states:Vec<usize>,
    pub excited_states:Vec<usize>,
    /// E_b, the ground states' energies in the frame (rad/us).
    pub energies:Vec<f64>,
    /// The hyperfine blocks, by increasing F, and each ground state's block.
    pub blocks:Vec<Block>,
    pub block:Vec<usize>,
    /// H0g = diag(E_b − ν_F(b)), the ground Hamiltonian in the blocks' frames.
    pub hamiltonian:Dense,
    /// K, the excited states' non-Hermitian Hamiltonian, and C_a (ground ← excited) on the request's axes.
    pub excited_hamiltonian:Dense,
    pub emission:[Dense;3],
    /// k_ref (rad/um) and w_r = ħk_ref²/2m (rad/us).
    pub wave_number:f64,
    pub recoil:f64,
    pub carriers:Vec<Carrier>,
    /// H_ls, Γ and Y: their secular parts.
    pub light_shift:Harmonics,
    pub scattering:Harmonics,
    pub admixture:Harmonics,
    /// E_i per excited state (in the order of `excited_states`): their secular parts.
    pub excited_readout:Vec<Harmonics>,
    pub jumps:Vec<Jump>,
    pub secular:Secular,
}

/// The manifold and F of a state label `manifold-F-m`.
fn level(label:&str)->Result<(&str,u8),String>{
    let mut parts=label.splitn(3,'-');
    match(parts.next(),parts.next().and_then(|f|f.parse().ok()),parts.next()){
        (Some(manifold),Some(f),Some(_))=>Ok((manifold,f)),_=>Err(format!("The state label {label} is not manifold-F-m")),
    }
}
/// The entries of m between the given blocks (each ground state's block in `block`).
fn between(block:&[usize],m:&Dense,to:usize,from:usize)->Dense{
    Dense::from_fn(m.rows,m.cols,|i,j|if block[i]==to&&block[j]==from{m[(i,j)]}else{Complex::ZERO})
}
/// The secular part of a ground operator: its block-diagonal entries.
fn secular_part(block:&[usize],m:&Dense)->Dense{Dense::from_fn(m.rows,m.cols,|i,j|if block[i]==block[j]{m[(i,j)]}else{Complex::ZERO})}
/// The LU factors of K + shift, the denominator of an excited response.
fn denominator(k:&Dense,shift:f64)->Result<Lu,String>{
    let mut a=k.clone();for i in 0..a.rows{a[(i,i)]+=Complex::new(shift,0.);}
    Lu::new(&a).ok_or_else(||"The excited states' resolvent is singular".to_string())
}

impl Ground{
    /// The generator of a request, or why it has none: a Raman-resonant Λ scheme is refused (secular step).
    pub fn new(request:&Request)->Result<Ground,String>{
        let levels=request.states.iter().map(|s|level(s)).collect::<Result<Vec<_>,_>>()?;
        let ground_states:Vec<usize>=(0..STATES).filter(|&i|levels[i].0=="ground").collect();
        let excited_states:Vec<usize>=(0..STATES).filter(|&i|levels[i].0!="ground").collect();
        let(ng,ne)=(ground_states.len(),excited_states.len());
        let h0=Dense::block(&request.hamiltonian,&ground_states,&ground_states);
        if (0..ng).any(|i|(0..ng).any(|j|i!=j&&h0[(i,j)]!=Complex::ZERO)){return Err("H0 must be diagonal on the ground states".into());}
        let energies:Vec<f64>=(0..ng).map(|b|h0[(b,b)].re).collect();
        for beam in &request.beams{
            let outside=(0..STATES*STATES).any(|k|{let(i,j)=(k/STATES,k%STATES);
                !(excited_states.contains(&i)&&ground_states.contains(&j))&&(beam.coupling.real[k]!=0.||beam.coupling.imag[k]!=0.)});
            if outside{return Err(format!("Beam {}'s coupling must be excited ← ground",beam.id));}
        }
        let all:Vec<usize>=(0..STATES).collect();
        let mut decay=Dense::zeros(ne,ne);
        for c in &request.emission{let c=Dense::block(c,&all,&excited_states);decay.add_scaled(Complex::ONE,&c.adjoint_product(&c));}
        let mut excited_hamiltonian=Dense::block(&request.hamiltonian,&excited_states,&excited_states);excited_hamiltonian.add_scaled(Complex::new(0.,-0.5),&decay);
        let emission:[Dense;3]=std::array::from_fn(|a|Dense::block(&request.emission[a],&ground_states,&excited_states));
        // The hyperfine blocks and their frames.
        let mut fs:Vec<u8>=ground_states.iter().map(|&i|levels[i].1).collect();fs.sort();fs.dedup();
        let blocks:Vec<Block>=fs.into_iter().map(|f|{
            let states:Vec<usize>=(0..ng).filter(|&b|levels[ground_states[b]].1==f).collect();
            let frame=states.iter().map(|&b|energies[b]).sum::<f64>()/states.len() as f64;Block{f,states,frame}
        }).collect();
        let block:Vec<usize>=(0..ng).map(|b|blocks.iter().position(|x|x.states.contains(&b)).unwrap()).collect();
        let hamiltonian=Dense::from_fn(ng,ng,|i,j|if i==j{Complex::new(energies[i]-blocks[block[i]].frame,0.)}else{Complex::ZERO});
        // One lattice wave number; the carriers.
        let lead=request.beams.iter().find(|b|b.axial!=0).or(request.beams.first()).ok_or("The ground generator needs a beam")?;
        let wave_number=lead.wave_number_per_um;let recoil=HBAR/request.mass_kg*1e6*(wave_number*wave_number)/2.;
        let mut groups:Vec<Vec<usize>>=vec![];
        for(l,beam)in request.beams.iter().enumerate(){
            match groups.iter_mut().find(|g|request.beams[g[0]].frequency_index==beam.frequency_index){Some(g)=>g.push(l),None=>groups.push(vec![l])}
        }
        // The elimination, column by column; the nearest pole seen by a coupled ground state.
        let mut pole=f64::INFINITY;
        let mut carriers=Vec::with_capacity(groups.len());
        for beams in groups{
            let first=&request.beams[beams[0]];
            let detuning=TAU*first.frequency_index.iter().zip(&request.frequencies_mhz).map(|(&n,&f)|n as f64*f).sum::<f64>();
            let mut waves:Vec<Wave>=(-1..=1).filter_map(|s|{
                let members:Vec<usize>=beams.iter().copied().filter(|&l|request.beams[l].axial==s).collect();
                if members.is_empty(){return None;}
                let mut coupling=Dense::zeros(ne,ng);
                for &l in &members{coupling.add_scaled(Complex::ONE,&Dense::block(&request.beams[l].coupling,&excited_states,&ground_states));}
                Some(Wave{axial:s,response:Dense::zeros(ne,ng),derivative:Dense::zeros(ne,ng),coupling})
            }).collect();
            let mut column=vec![Complex::ZERO;ne];
            for b in 0..ng{
                let lu=denominator(&excited_hamiltonian,-energies[b]-detuning)?;
                for wave in &mut waves{
                    for i in 0..ne{column[i]=wave.coupling[(i,b)];}lu.solve(&mut column);for i in 0..ne{wave.response[(i,b)]=column[i];}
                    lu.solve(&mut column);for i in 0..ne{wave.derivative[(i,b)]=column[i];}
                }
                if waves.iter().any(|w|(0..ne).any(|i|w.coupling[(i,b)]!=Complex::ZERO)){
                    let mut identity=Dense::zeros(ne,ne);for i in 0..ne{identity[(i,i)]=Complex::ONE;}
                    pole=pole.min(1./lu.solve_columns(&identity).spectral_norm());
                }
            }
            let pairs=waves.iter().flat_map(|emitted|waves.iter().map(move|absorbed|(emitted,absorbed))).map(|(emitted,absorbed)|Pair{
                emitted:emitted.axial,absorbed:absorbed.axial,x:secular_part(&block,&emitted.coupling.adjoint_product(&absorbed.response)),
                p:secular_part(&block,&emitted.coupling.adjoint_product(&absorbed.derivative))}).collect();
            carriers.push(Carrier{beams,detuning,kappa:first.wave_number_per_um/wave_number,waves,pairs});
        }
        // The generator's harmonics and the readout.
        let(mut light_shift,mut scattering,mut admixture)=(Harmonics::zeros(ng),Harmonics::zeros(ng),Harmonics::zeros(ng));
        let mut excited_readout=vec![Harmonics::zeros(ng);ne];
        for carrier in &carriers{
            for pair in &carrier.pairs{
                let m=pair.absorbed-pair.emitted;
                light_shift.add(m,Complex::new(-0.5,0.),&pair.x);light_shift.add(-m,Complex::new(-0.5,0.),&pair.x.adjoint());
            }
            for emitted in &carrier.waves{for absorbed in &carrier.waves{
                let m=absorbed.axial-emitted.axial;let(rp,r)=(&emitted.response,&absorbed.response);
                scattering.add(m,Complex::ONE,&rp.adjoint_product(&decay.product(r)));admixture.add(m,Complex::ONE,&rp.adjoint_product(r));
                for(i,readout)in excited_readout.iter_mut().enumerate(){readout.add(m,Complex::ONE,&Dense::from_fn(ng,ng,|j,k|rp[(i,j)].conj()*r[(i,k)]));}
            }}
        }
        let(scattering,admixture)=(scattering.map(|m|secular_part(&block,m)),admixture.map(|m|secular_part(&block,m)));
        let excited_readout:Vec<Harmonics>=excited_readout.iter().map(|e|e.map(|m|secular_part(&block,m))).collect();
        // The jumps, split into blocks.
        let nb=blocks.len();let mut jumps=vec![];
        for(c,carrier)in carriers.iter().enumerate(){for(a,emission)in emission.iter().enumerate(){
            let whole:Vec<(i8,Dense,Dense)>=carrier.waves.iter().map(|w|(w.axial,emission.product(&w.response),emission.product(&w.derivative))).collect();
            for to in 0..nb{for from in 0..nb{
                let(mut operator,mut derivative)=(Harmonics::zeros(ng),Harmonics::zeros(ng));
                for(s,l,d)in &whole{operator.add(*s,Complex::ONE,&between(&block,l,to,from));derivative.add(*s,Complex::ONE,&between(&block,d,to,from));}
                if operator.0.iter().chain(&derivative.0).all(Dense::is_zero){continue;}
                jumps.push(Jump{axis:a,carrier:c,from,to,kappa:carrier.kappa,operator,derivative});
            }}
        }}
        let secular=Secular::new(&carriers,&blocks,&block,&jumps,&light_shift,&scattering,pole);
        let ground=Ground{ground_states,excited_states,energies,blocks,block,hamiltonian,excited_hamiltonian,emission,wave_number,recoil,
            carriers,light_shift,scattering,admixture,excited_readout,jumps,secular};
        match ground.secular.largest(){Some(term)if term.micromotion>ground.secular.eta=>Err(ground.raman_resonance(request,term)),_=>Ok(ground)}
    }

    /// Why a Raman-resonant scheme is refused, naming its beams.
    fn raman_resonance(&self,request:&Request,term:&Rotating)->String{
        let named=|c:usize|self.carriers[c].beams.iter().map(|&l|format!("「{}」",request.beams[l].id)).collect::<Vec<_>>().join("、");
        let[c,d]=term.carriers;let beams=if c==d{named(c)}else{format!("{}与{}",named(c),named(d))};
        let[f,g]=term.blocks.map(|b|self.blocks[b].f);
        let what=if f==g{format!("F={f} 内由两组光的拍频驱动的相干")}else{format!("F={} 与 F={} 之间的相干",f.min(g),f.max(g))};
        format!("所选光束之间存在拉曼共振（{beams}，拉曼失谐 {:.3} MHz），{what}不能忽略，1D 运动暂不支持",term.frequency.abs()/TAU)
    }

    /// H_eff(θ) = H0g + H_ls(θ) − (i/2) Γ(θ).
    pub fn effective_hamiltonian(&self,theta:f64)->Dense{
        let mut h=self.hamiltonian.clone();h.add_scaled(Complex::ONE,&self.light_shift.at(theta));h.add_scaled(Complex::new(0.,-0.5),&self.scattering.at(theta));h
    }

    /// The 32 populations (in the request's order) at lattice phase θ of a block-diagonal ground density
    /// ρ: the bare ground ones, of ρ − ½{Y(θ), ρ}, and the excited ones, tr(E_i(θ) ρ).
    pub fn populations(&self,rho:&Dense,theta:f64)->Vec<f64>{
        let y=self.admixture.at(theta);let(left,right)=(y.product(rho),rho.product(&y));
        let mut out=vec![0.;STATES];
        for(b,&state)in self.ground_states.iter().enumerate(){out[state]=rho[(b,b)].re-0.5*(left[(b,b)].re+right[(b,b)].re);}
        for(i,&state)in self.excited_states.iter().enumerate(){
            let e=self.excited_readout[i].at(theta);out[state]=(0..rho.rows).map(|j|(0..rho.rows).map(|k|(e[(j,k)]*rho[(k,j)]).re).sum::<f64>()).sum();
        }
        out
    }

    /// The first-order increments of H_ls, Γ and the jumps (in the order of `jumps`) for the excited-energy
    /// shifts `shift(c, s)` = w_{c,s} (rad/us): R_{c,s} → R_{c,s} − w_{c,s} R2_{c,s}.
    pub fn first_order(&self,shift:impl Fn(usize,i8)->f64)->FirstOrder{
        let ng=self.ground_states.len();let mut light_shift=Harmonics::zeros(ng);
        for(c,carrier)in self.carriers.iter().enumerate(){for pair in &carrier.pairs{
            let(m,w)=(pair.absorbed-pair.emitted,Complex::new(0.5*shift(c,pair.absorbed),0.));
            light_shift.add(m,w,&pair.p);light_shift.add(-m,w,&pair.p.adjoint());
        }}
        let mut scattering=Harmonics::zeros(ng);
        let jumps=self.jumps.iter().map(|jump|{
            let mut d=Harmonics::zeros(ng);for s in -1..=1{d.add(s,Complex::new(-shift(jump.carrier,s),0.),jump.derivative.get(s));}
            for s in -1..=1{for t in -1..=1{
                let(l,dl)=(jump.operator.get(s),d.get(t));
                scattering.add(t-s,Complex::ONE,&l.adjoint_product(dl));scattering.add(s-t,Complex::ONE,&dl.adjoint_product(l));
            }}
            d
        }).collect();
        FirstOrder{light_shift,scattering,jumps}
    }

    /// R_{c,s} of carrier `carrier`'s wave `wave` to all orders in an excited-energy shift w (rad/us):
    /// column b solved with (K − E_b − δ_c + w)⁻¹. R − w R2 is its first order.
    pub fn response(&self,carrier:usize,wave:usize,shift:f64)->Result<Dense,String>{
        let(carrier,ng)=(&self.carriers[carrier],self.ground_states.len());let coupling=&carrier.waves[wave].coupling;
        let mut out=Dense::zeros(coupling.rows,ng);let mut column=vec![Complex::ZERO;coupling.rows];
        for b in 0..ng{
            let lu=denominator(&self.excited_hamiltonian,shift-self.energies[b]-carrier.detuning)?;
            for i in 0..coupling.rows{column[i]=coupling[(i,b)];}lu.solve(&mut column);for i in 0..coupling.rows{out[(i,b)]=column[i];}
        }
        Ok(out)
    }
}

impl Secular{
    /// Every rotating term of the eliminated generator, with the expansion parameter η: the
    /// stimulated terms V_{c′}†R_c between blocks, and the cross terms L_{a,c} ρ L_{a,c′}† of jumps
    /// from one block into two.
    fn new(carriers:&[Carrier],blocks:&[Block],block:&[usize],jumps:&[Jump],light_shift:&Harmonics,scattering:&Harmonics,pole:f64)->Secular{
        let(nb,ng,nc)=(blocks.len(),block.len(),carriers.len());let mut terms:Vec<([usize;2],[usize;2],f64)>=vec![];
        for(c,absorbing)in carriers.iter().enumerate(){for(d,emitting)in carriers.iter().enumerate(){
            let mut cross=Harmonics::zeros(ng);
            for emitted in &emitting.waves{for absorbed in &absorbing.waves{cross.add(absorbed.axial-emitted.axial,Complex::ONE,&emitted.coupling.adjoint_product(&absorbed.response));}}
            for from in 0..nb{for to in 0..nb{if c!=d||to!=from{terms.push(([c,d],[from,to],cross.map(|m|between(block,m,to,from)).bound()));}}}
        }}
        let norms:Vec<f64>=jumps.iter().map(|j|j.operator.bound()).collect();
        let norm=|a:usize,c:usize,to:usize,from:usize|jumps.iter().position(|j|(j.axis,j.carrier,j.to,j.from)==(a,c,to,from)).map_or(0.,|j|norms[j]);
        for c in 0..nc{for d in 0..nc{for x in 0..nb{for y in 0..nb{if c!=d||x!=y{for from in 0..nb{
            terms.push(([c,d],[y,x],(0..3).map(|a|norm(a,c,x,from)*norm(a,d,y,from)).sum()));
        }}}}}}
        let dropped=terms.into_iter().filter(|term|term.2>0.).map(|([c,d],[from,to],coupling)|{
            let frequency=(blocks[to].frame-blocks[from].frame)-(carriers[c].detuning-carriers[d].detuning);
            Rotating{carriers:[c,d],blocks:[from,to],frequency,coupling,micromotion:coupling/frequency.abs()}
        }).collect();
        let mut generator=light_shift.clone();for m in -2..=2{generator.add(m,Complex::new(0.,-0.5),scattering.get(m));}
        let scale=generator.bound();
        Secular{eta:scale/pole,scale,pole,dropped}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::linear::uniform;
    use crate::{Beam,Matrix};
    use serde_json::Value;

    /// The PGC example's request as the app sends it, and gme.py's numbers for it (tests/fixtures/ground_gme.json).
    fn fixture()->Value{serde_json::from_str(include_str!("../tests/fixtures/ground_gme.json")).unwrap()}
    fn index(v:&Value)->usize{v.as_u64().unwrap() as usize}
    fn float(v:&Value)->f64{v.as_f64().unwrap()}
    /// A rows × cols matrix from its nonzero entries [row, column, re, im].
    fn entries(v:&Value,rows:usize,cols:usize)->Dense{
        let mut m=Dense::zeros(rows,cols);
        for e in v.as_array().unwrap(){m[(index(&e[0]),index(&e[1]))]=Complex::new(float(&e[2]),float(&e[3]));}
        m
    }
    fn matrix(v:&Value)->Matrix{let m=entries(v,STATES,STATES);Matrix{real:m.entries.iter().map(|z|z.re).collect(),imag:m.entries.iter().map(|z|z.im).collect()}}
    fn request(v:&Value)->Request{
        let zero=Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]};
        let beams=v["beams"].as_array().unwrap().iter().map(|b|Beam{id:b["id"].as_str().unwrap().into(),axial:b["axial"].as_i64().unwrap() as i8,
            wave_number_per_um:float(&b["waveNumberPerUm"]),frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),
            coupling:matrix(&b["coupling"]),counter_coupling:zero.clone()}).collect();
        let e=&v["emission"];
        Request{states:v["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&v["hamiltonian"]),
            frame_frequencies_mhz:vec![0.;STATES],frequencies_mhz:v["frequenciesMHz"].as_array().unwrap().iter().map(float).collect(),beams,
            emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],field_vector_g:[0.;3],mass_kg:float(&v["massKg"]),
            initial_density:vec![Complex::ZERO;STATES*STATES],temperature_uk:30.,times_us:vec![],tolerance:5e-3}
    }
    fn row(values:&[f64])->Dense{Dense::from_fn(1,values.len(),|_,j|Complex::new(values[j],0.))}
    fn values(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(float).collect()}
    /// The harmonics of h paired with the reference's {m: entries} (absent: zero).
    fn harmonics<'a>(h:&'a Harmonics,v:&Value)->Vec<(&'a Dense,Dense)>{
        let n=h.get(0).rows;(-2..=2).map(|m|(h.get(m),v.get(m.to_string()).map_or_else(||Dense::zeros(n,n),|e|entries(e,n,n)))).collect()
    }
    /// max |a − b| / max |b| over the pairs.
    fn relative<'a>(pairs:impl IntoIterator<Item=(&'a Dense,Dense)>)->f64{
        let(mut difference,mut scale)=(0_f64,0_f64);
        for(a,b)in pairs{for(x,y)in a.entries.iter().zip(&b.entries){difference=difference.max((*x-*y).abs());scale=scale.max(y.abs());}}
        difference/scale
    }

    /// The PGC example's generator is gme.py's -- the research module GMW-o1 was validated with -- to
    /// round-off, operator by operator (the reference has 12 significant digits).
    #[test]
    fn the_pgc_example_has_the_operators_of_gme(){
        let f=fixture();let(r,g)=(&f["reference"],Ground::new(&request(&f["request"])).unwrap());
        let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
        let pairs=|which:&str|->Vec<(&Dense,Dense)>{r[which].as_object().unwrap().iter().map(|(key,v)|{
            let k:Vec<i64>=key.split(',').map(|x|x.parse().unwrap()).collect();
            let pair=g.carriers[k[0] as usize].pairs.iter().find(|p|(p.emitted as i64,p.absorbed as i64)==(k[1],k[2])).unwrap();
            (if which=="X"{&pair.x}else{&pair.p},entries(v,ng,ng))
        }).collect()};
        let jumps=r["jumps"].as_array().unwrap();assert_eq!(g.jumps.len(),jumps.len());
        let jump=|j:&Value|g.jumps.iter().find(|x|(x.axis,x.carrier,x.to,x.from)==(index(&j["axis"]),index(&j["carrier"]),index(&j["to"]),index(&j["from"]))).unwrap();
        let kv=float(&r["firstOrder"]["kv"]);let first=g.first_order(|_,s|s as f64*kv);
        let exact=&r["response"];let c=index(&exact["carrier"]);
        let wave=g.carriers[c].waves.iter().position(|w|w.axial as i64==exact["axial"].as_i64().unwrap()).unwrap();
        let readout=&r["readout"];let p=g.populations(&entries(&readout["rho"],ng,ng),float(&readout["theta"]));
        let(ground,excited):(Vec<f64>,Vec<f64>)=(g.ground_states.iter().map(|&i|p[i]).collect(),g.excited_states.iter().map(|&i|p[i]).collect());
        let carriers:Vec<f64>=g.carriers.iter().flat_map(|c|[c.detuning,c.kappa]).collect();
        let carriers_gme:Vec<f64>=r["carriers"].as_array().unwrap().iter().flat_map(|c|[float(&c["delta"]),float(&c["kappa"])]).collect();
        let checks=[
            ("k_ref and w_r",relative([(&row(&[g.wave_number,g.recoil]),row(&[float(&r["kref"]),float(&r["wr"])]))])),
            ("carriers: δ and κ",relative([(&row(&carriers),row(&carriers_gme))])),
            ("H_ls",relative(harmonics(&g.light_shift,&r["Hls"]))),("Γ",relative(harmonics(&g.scattering,&r["Gam"]))),("Y",relative(harmonics(&g.admixture,&r["Y"]))),
            ("X",relative(pairs("X"))),("P",relative(pairs("P"))),
            ("jumps L",relative(jumps.iter().flat_map(|j|harmonics(&jump(j).operator,&j["L"])))),
            ("jumps D2",relative(jumps.iter().flat_map(|j|harmonics(&jump(j).derivative,&j["D2"])))),
            ("first order",relative(harmonics(&first.light_shift,&r["firstOrder"]["dHls"]).into_iter().chain(harmonics(&first.scattering,&r["firstOrder"]["dGam"])))),
            ("exact response",relative([(&g.response(c,wave,float(&exact["shift"])).unwrap(),entries(&exact["R"],ne,ng))])),
            ("populations",relative([(&row(&ground),row(&values(&readout["ground"]))),(&row(&excited),row(&values(&readout["excited"])))])),
        ];
        assert!(checks.iter().all(|(_,x)|*x<=1e-10),"{checks:?}");
    }

    /// The PGC example perturbed at random: the ground energies split as by a field, every beam's
    /// coupling scaled by a complex factor, and one more cooling beam across the axis (harmonics ±1).
    fn perturbed(seed:u64)->Request{
        let mut next=uniform(seed);let mut r=request(&fixture()["request"]);
        let scale=|m:&mut Matrix,z:Complex|for(re,im)in m.real.iter_mut().zip(m.imag.iter_mut()){let x=z*Complex::new(*re,*im);(*re,*im)=(x.re,x.im);};
        for i in (0..STATES).filter(|&i|r.states[i].starts_with("ground-")){r.hamiltonian.real[i*STATES+i]+=next();}
        for beam in &mut r.beams{scale(&mut beam.coupling,Complex::new(1.+next(),next()));}
        let mut side=r.beams.iter().find(|b|b.id=="cool-plus").unwrap().clone();(side.id,side.axial)=("cool-side".into(),0);
        scale(&mut side.coupling,Complex::new(next(),next()));r.beams.push(side);r
    }
    /// Σ_m ‖a_m − b_m‖ over the harmonics, against the largest of b.
    fn mismatch(a:&Harmonics,b:&Harmonics)->(f64,f64){
        (-2..=2).map(|m|{let mut d=a.get(m).clone();d.add_scaled(-Complex::ONE,b.get(m));(d.largest(),b.get(m).largest())}).fold((0.,0.),|(x,y),(d,s)|(x.max(d),y.max(s)))
    }

    /// The trace is kept: the jumps, split into blocks, carry the whole scattering, Σ_j L_j†L_j = Γ
    /// harmonic by harmonic.
    #[test]
    fn the_jumps_carry_the_whole_scattering(){
        for seed in 1..=4{
            let g=Ground::new(&perturbed(seed)).unwrap();let mut sum=Harmonics::zeros(g.ground_states.len());
            for j in &g.jumps{for s in -1..=1{for t in -1..=1{sum.add(t-s,Complex::ONE,&j.operator.get(s).adjoint_product(j.operator.get(t)));}}}
            let(error,scale)=mismatch(&sum,&g.scattering);assert!(error<=1e-13*scale,"seed {seed}: {error:e} of {scale:e}");
        }
    }

    /// Γ is the decay of the second-order shift X = V†R: Γ = −i(X − X†) + i[Y, H0g] (harmonic m of X† is
    /// X[−m]†; the commutator, from the column-wise resolvents, vanishes for degenerate ground states).
    /// So the decay inside K and the emission operators agree, and R solves its column equations.
    #[test]
    fn the_scattering_is_the_decay_of_the_second_order_shift(){
        for seed in 1..=4{
            let g=Ground::new(&perturbed(seed)).unwrap();let n=g.ground_states.len();
            let mut x=Harmonics::zeros(n);for c in &g.carriers{for p in &c.pairs{x.add(p.absorbed-p.emitted,Complex::ONE,&p.x);}}
            let mut decay=Harmonics::zeros(n);
            for m in -2..=2{
                decay.add(m,-Complex::I,x.get(m));decay.add(m,Complex::I,&x.get(-m).adjoint());
                let y=g.admixture.get(m);decay.add(m,Complex::I,&y.product(&g.hamiltonian));decay.add(m,-Complex::I,&g.hamiltonian.product(y));
            }
            let(error,scale)=mismatch(&decay,&g.scattering);assert!(error<=1e-12*scale,"seed {seed}: {error:e} of {scale:e}");
        }
    }

    /// The readout keeps the trace: Σ_i E_i = Y, and the 32 populations of a density sum to one.
    #[test]
    fn the_populations_sum_to_one(){
        for seed in 1..=4{
            let g=Ground::new(&perturbed(seed)).unwrap();let n=g.ground_states.len();
            let mut total=Harmonics::zeros(n);for e in &g.excited_readout{for m in -2..=2{total.add(m,Complex::ONE,e.get(m));}}
            let(error,scale)=mismatch(&total,&g.admixture);assert!(error<=1e-13*scale,"seed {seed}: {error:e} of {scale:e}");
            let mut next=uniform(seed);let a=Dense::from_fn(n,n,|i,j|if g.block[i]==g.block[j]{Complex::new(next(),next())}else{Complex::ZERO});
            let mut rho=a.product(&a.adjoint());let trace:f64=(0..n).map(|i|rho[(i,i)].re).sum();rho=rho.scaled(Complex::new(1./trace,0.));
            let sum:f64=g.populations(&rho,TAU*next()).iter().sum();assert!((sum-1.).abs()<=1e-14,"seed {seed}: {sum}");
        }
    }

    /// R − w R2 is the exact response to first order: column b's remainder is w² (K − E_b − δ_c)⁻² R(w)_b,
    /// at most (w/pole)² of R(w)_b. At w = 0 the exact response is R itself.
    #[test]
    fn the_first_order_response_leaves_a_second_order_remainder(){
        let g=Ground::new(&perturbed(1)).unwrap();let pole=g.secular.pole;
        let norm=|m:&Dense,b:usize|(0..m.rows).map(|i|m[(i,b)].norm_sqr()).sum::<f64>().sqrt();
        for(c,carrier)in g.carriers.iter().enumerate(){for(k,wave)in carrier.waves.iter().enumerate(){
            assert_eq!(g.response(c,k,0.).unwrap(),wave.response);
            for w in [pole/8.,-pole/3.]{
                let exact=g.response(c,k,w).unwrap();let mut remainder=exact.clone();
                remainder.add_scaled(-Complex::ONE,&wave.response);remainder.add_scaled(Complex::new(w,0.),&wave.derivative);
                for b in 0..g.ground_states.len(){
                    let bound=(w/pole).powi(2)*norm(&exact,b)+f64::EPSILON*(norm(&wave.response,b)+w.abs()*norm(&wave.derivative,b));
                    assert!(norm(&remainder,b)<=bound,"carrier {c}, wave {k}, w {w}, column {b}: {} > {bound}",norm(&remainder,b));
                }
            }
        }}
    }

    /// The cooler moved to a Raman resonance with the repumper (both on F′ = 2: a Λ scheme) is refused,
    /// naming the beams and the Raman detuning. In the PGC example the cooler is −f0 from the repumper
    /// (frequency axis 0), so f0 = (ν_2 − ν_1)/2π − δ_R puts the Raman detuning at δ_R.
    #[test]
    fn a_raman_resonant_lambda_scheme_is_refused(){
        let pgc=request(&fixture()["request"]);let g=Ground::new(&pgc).unwrap();
        let hyperfine=(g.blocks[1].frame-g.blocks[0].frame)/TAU;
        for detuning in [0.,1.]{
            let mut lambda=pgc.clone();lambda.frequencies_mhz[0]=hyperfine-detuning;
            let error=Ground::new(&lambda).unwrap_err();
            assert!(error.starts_with("所选光束之间存在拉曼共振（")&&error.contains("「cool-plus」、「cool-minus」")&&error.contains("「repump-plus」、「repump-minus」")
                &&error.contains(&format!("拉曼失谐 {detuning:.3} MHz"))&&error.ends_with("F=1 与 F=2 之间的相干不能忽略，1D 运动暂不支持"),"{error}");
        }
    }
}
