//! The atom's ground manifold with the excited states eliminated: the generator of its internal
//! state at any position x on the motion line, to first order in the Doppler and recoil shifts.
//!
//! # Elimination
//! The beams of one frequency (equal `Beam::frequency_index`) form a carrier c at
//! δ_c = 2π(ν_c − ν_frame). Each beam l keeps its own coupling V_l at the atom and its own
//! amplitude a_l(x) on the line (`Beam::amplitude`: its axial wave number, and a focused beam's
//! Gaussian profile), so carrier c drives V_c(x) = Σ_{l∈c} a_l(x) V_l e^{−iδ_c t} + h.c. With the
//! ground energies E_b (H0 is diagonal on them) and the excited states' non-Hermitian
//! Hamiltonian K = H0_ee − (i/2) D, D = Σ_a C_a†C_a|_ee, the excited states and the optical
//! coherences follow the ground states adiabatically, beam by beam and column by column
//! (`Elimination`):
//!
//! ```text
//! R_l[:,b] = (K − E_b − δ_c)⁻¹ V_l[:,b]        R2_l[:,b] = (K − E_b − δ_c)⁻² V_l[:,b]        R_c(x) = Σ_{l∈c} a_l(x) R_l
//! ```
//!
//! and the ground density evolves at x under the Reiter–Sørensen Lindblad generator (`Local`)
//!
//! ```text
//! dρ/dt = −i(H_eff ρ − ρ H_eff†) + Σ_j L_j ρ L_j†,     H_eff = H0g + H_ls − (i/2) Γ,
//! H_ls = −Herm Σ_c V_c(x)† R_c(x),   Γ = Σ_c R_c(x)† D R_c(x) = Σ_j L_j† L_j,   L_{a,c} = C_a R_c(x).
//! ```
//!
//! The photon of jump (a, c) is polarized along a and has carrier c's wave number: it kicks the
//! atom by −u ħk_c along the axis, u distributed by the dipole pattern of a (lib.rs). Readout:
//! the bare ground density is ρ − ½{Y, ρ}, Y = Σ_c R_c†R_c, and an excited state's population is
//! tr(E_i ρ), E_i = Σ_c R_c†|i⟩⟨i|R_c (Σ_i E_i = Y).
//!
//! An `Elimination` belongs to one H0 -- K and E_b -- and the generator at x is evaluated with any
//! of them, so a position-dependent H0 needs only its own eliminations.
//!
//! # First order in the Doppler and recoil shifts
//! Absorbing from beam l raises the atom's momentum by ħk_l along the axis (k_l its axial wave
//! number), which shifts the excited state by the kinetic energy w_l = k_l p/m + ħk_l²/2m; to
//! first order R_l → R_l − w_l R2_l (`Ground::first_order`), and `Ground::response` gives the
//! shifted response to all orders.
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
//! form a Raman-resonant Λ scheme (gray molasses, EIT) and are refused by name. Norms over the
//! line are bounded by the sums of the beams' terms, as |a_l(x)| ≤ 1.
//!
//! # Approximations and their criteria
//! - The zeroth-order elimination: η ≪ 1. Its error is the saturation, ≈ 2 p_e locally (the
//!   scattering rate +1 to +2.4 % in 1D PGC).
//! - First order in the Doppler shift: (kv / σ_min)² ≪ 1; `response` is exact.
//! - The secular step: ε ≤ η, enforced.
//! - An emitted photon has its carrier's wave number: ≤ 1e-7 on the D2 line.
//! - The counter-rotating couplings are dropped (optical RWA).
use crate::linear::{Dense,Lu};
use crate::{Beam,Complex,Matrix,Request,STATES};
use std::f64::consts::TAU;

/// A hyperfine block of the ground manifold.
#[derive(Clone,Debug)]
pub struct Block{
    pub f:u8,
    /// Its states, as positions in `Ground::ground_states`.
    pub states:Vec<usize>,
    /// ν_F, the centroid of its energies in the request's H0: the block's frame (rad/us).
    pub frame:f64,
}
/// The beams of one frequency.
#[derive(Clone,Debug)]
pub struct Carrier{
    /// Its beams (indices into `Ground::beams`).
    pub beams:Vec<usize>,
    /// δ_c = 2π(ν_c − ν_frame) (rad/us).
    pub detuning:f64,
    /// Its photons' wave number |k| (rad/um).
    pub wave_number:f64,
}
/// A quantum jump: a photon polarized along `axis` scattered from carrier `carrier`, the atom going
/// from block `from` to block `to`. Its operator at x is `Local::jumps`.
#[derive(Clone,Debug,PartialEq)]
pub struct Jump{
    /// 0 and 1, the axes perpendicular to the motion axis; 2, the motion axis (`Request::emission`).
    pub axis:usize,
    pub carrier:usize,
    pub from:usize,
    pub to:usize,
}

/// The elimination for one H0: K and E_b, and every beam's excited response.
#[derive(Clone,Debug)]
pub struct Elimination{
    /// E_b, the ground states' energies (rad/us), and H0g = diag(E_b − ν_F(b)) in the blocks' frames.
    pub energies:Vec<f64>,
    pub hamiltonian:Dense,
    /// K, the excited states' non-Hermitian Hamiltonian (rad/us).
    pub excited_hamiltonian:Dense,
    /// Per beam, R_l and R2_l, excited ← ground (dimensionless and us).
    pub responses:Vec<Dense>,
    pub derivatives:Vec<Dense>,
}

/// The generator at one position: the secular parts of H_ls, Γ, Y and E_i (per excited state, in
/// the order of `Ground::excited_states`), and the jumps L_j (in the order of `Ground::jumps`).
#[derive(Clone,Debug)]
pub struct Local{
    pub light_shift:Dense,
    pub scattering:Dense,
    pub admixture:Dense,
    pub excited_readout:Vec<Dense>,
    pub jumps:Vec<Dense>,
}
impl Local{
    /// H_eff = H0g + H_ls − (i/2) Γ, with the elimination's H0g.
    pub fn effective_hamiltonian(&self,elimination:&Elimination)->Dense{
        let mut h=elimination.hamiltonian.clone();h.add_scaled(Complex::ONE,&self.light_shift);h.add_scaled(Complex::new(0.,-0.5),&self.scattering);h
    }
}
/// The first-order increments of the generator at one position for excited-energy shifts w_l.
#[derive(Clone,Debug)]
pub struct FirstOrder{pub light_shift:Dense,pub scattering:Dense,pub jumps:Vec<Dense>}

/// The secular step's report.
#[derive(Clone,Debug)]
pub struct Secular{
    /// η = scale / pole, the elimination's expansion parameter.
    pub eta:f64,
    /// A bound on ‖H_ls − (i/2) Γ‖ over the line, the scale of the light-induced ground dynamics (rad/us).
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

/// The eliminated ground-manifold generator of a request, apart from H0: the operators act on the
/// ground states in the order of `ground_states`, the excited ones in the order of
/// `excited_states`; rates in rad/us, jumps in 1/sqrt(us).
#[derive(Clone,Debug)]
pub struct Ground{
    /// The request's ground and excited states (indices into `Request::states`).
    pub ground_states:Vec<usize>,
    pub excited_states:Vec<usize>,
    /// The hyperfine blocks, by increasing F, and each ground state's block.
    pub blocks:Vec<Block>,
    pub block:Vec<usize>,
    /// D = Σ_a C_a†C_a on the excited states, and C_a (ground ← excited) on the request's axes.
    pub decay:Dense,
    pub emission:[Dense;3],
    /// The request's beams, and each one's V_l (excited ← ground at the atom, rad/us).
    pub beams:Vec<Beam>,
    pub couplings:Vec<Dense>,
    pub carriers:Vec<Carrier>,
    /// Every jump: per carrier, axis and pair of blocks.
    pub jumps:Vec<Jump>,
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
/// Σ_l weight_l m_l over the given beams.
fn combine(beams:&[usize],weights:&[Complex],matrices:&[Dense])->Dense{
    let mut out=Dense::zeros(matrices[0].rows,matrices[0].cols);for &l in beams{out.add_scaled(weights[l],&matrices[l]);}out
}
/// −Herm(m) = −(m + m†)/2.
fn negative_hermitian_part(m:&Dense)->Dense{let mut h=m.scaled(Complex::new(-0.5,0.));h.add_scaled(Complex::new(-0.5,0.),&m.adjoint());h}

impl Ground{
    /// The generator of a request, or why it has none: a Raman-resonant Λ scheme is refused (the
    /// secular step at the request's H0).
    pub fn new(request:&Request)->Result<Ground,String>{
        let levels=request.states.iter().map(|s|level(s)).collect::<Result<Vec<_>,_>>()?;
        let ground_states:Vec<usize>=(0..STATES).filter(|&i|levels[i].0=="ground").collect();
        let excited_states:Vec<usize>=(0..STATES).filter(|&i|levels[i].0!="ground").collect();
        let(ng,ne)=(ground_states.len(),excited_states.len());
        let driven=|k:usize|levels[k/STATES].0!="ground"&&levels[k%STATES].0=="ground";
        for beam in &request.beams{
            if (0..STATES*STATES).any(|k|!driven(k)&&(beam.coupling.real[k]!=0.||beam.coupling.imag[k]!=0.)){return Err(format!("Beam {}'s coupling must be excited ← ground",beam.id));}
        }
        let all:Vec<usize>=(0..STATES).collect();
        let mut decay=Dense::zeros(ne,ne);
        for c in &request.emission{let c=Dense::block(c,&all,&excited_states);decay.add_scaled(Complex::ONE,&c.adjoint_product(&c));}
        let emission:[Dense;3]=std::array::from_fn(|a|Dense::block(&request.emission[a],&ground_states,&excited_states));
        // The hyperfine blocks and their frames, from the request's H0.
        let h0=Dense::block(&request.hamiltonian,&ground_states,&ground_states);
        let mut fs:Vec<u8>=ground_states.iter().map(|&i|levels[i].1).collect();fs.sort();fs.dedup();
        let blocks:Vec<Block>=fs.into_iter().map(|f|{
            let states:Vec<usize>=(0..ng).filter(|&b|levels[ground_states[b]].1==f).collect();
            let frame=states.iter().map(|&b|h0[(b,b)].re).sum::<f64>()/states.len() as f64;Block{f,states,frame}
        }).collect();
        let block:Vec<usize>=(0..ng).map(|b|blocks.iter().position(|x|x.states.contains(&b)).unwrap()).collect();
        // The carriers, and every jump.
        let mut groups:Vec<Vec<usize>>=vec![];
        for(l,beam)in request.beams.iter().enumerate(){
            match groups.iter_mut().find(|g|request.beams[g[0]].frequency_index==beam.frequency_index){Some(g)=>g.push(l),None=>groups.push(vec![l])}
        }
        let carriers:Vec<Carrier>=groups.into_iter().map(|beams|{
            let first=&request.beams[beams[0]];
            let detuning=TAU*first.frequency_index.iter().zip(&request.frequencies_mhz).map(|(&n,&f)|n as f64*f).sum::<f64>();
            Carrier{beams,detuning,wave_number:first.wave_number_per_um}
        }).collect();
        let nb=blocks.len();
        let jumps=(0..carriers.len()).flat_map(|carrier|(0..3).flat_map(move|axis|(0..nb).flat_map(move|to|(0..nb).map(move|from|Jump{axis,carrier,from,to})))).collect();
        let couplings=request.beams.iter().map(|b|Dense::block(&b.coupling,&excited_states,&ground_states)).collect();
        let ground=Ground{ground_states,excited_states,blocks,block,decay,emission,beams:request.beams.clone(),couplings,carriers,jumps};
        let report=ground.secular(&ground.eliminate(&request.hamiltonian)?)?;
        match report.largest(){Some(term)if term.micromotion>report.eta=>Err(ground.raman_resonance(term)),_=>Ok(ground)}
    }

    /// The elimination for an H0 (rad/us, in the frame; diagonal on the ground states): E_b, H0g in
    /// the blocks' frames, K, and every beam's R_l and R2_l, column by column.
    pub fn eliminate(&self,hamiltonian:&Matrix)->Result<Elimination,String>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let h0=Dense::block(hamiltonian,&self.ground_states,&self.ground_states);
        if (0..ng).any(|i|(0..ng).any(|j|i!=j&&h0[(i,j)]!=Complex::ZERO)){return Err("H0 must be diagonal on the ground states".into());}
        let energies:Vec<f64>=(0..ng).map(|b|h0[(b,b)].re).collect();
        let ground_hamiltonian=Dense::from_fn(ng,ng,|i,j|if i==j{Complex::new(energies[i]-self.blocks[self.block[i]].frame,0.)}else{Complex::ZERO});
        let mut excited_hamiltonian=Dense::block(hamiltonian,&self.excited_states,&self.excited_states);excited_hamiltonian.add_scaled(Complex::new(0.,-0.5),&self.decay);
        let(mut responses,mut derivatives)=(vec![Dense::zeros(ne,ng);self.beams.len()],vec![Dense::zeros(ne,ng);self.beams.len()]);
        let mut column=vec![Complex::ZERO;ne];
        for carrier in &self.carriers{for b in 0..ng{
            let lu=denominator(&excited_hamiltonian,-energies[b]-carrier.detuning)?;
            for &l in &carrier.beams{
                for i in 0..ne{column[i]=self.couplings[l][(i,b)];}lu.solve(&mut column);for i in 0..ne{responses[l][(i,b)]=column[i];}
                lu.solve(&mut column);for i in 0..ne{derivatives[l][(i,b)]=column[i];}
            }
        }}
        Ok(Elimination{energies,hamiltonian:ground_hamiltonian,excited_hamiltonian,responses,derivatives})
    }

    /// Every beam's amplitude a_l(x) at x (um) on the line.
    pub fn amplitudes(&self,x:f64)->Vec<Complex>{self.beams.iter().map(|b|b.amplitude(x)).collect()}

    /// The generator at x (um) on the line, with an elimination's K, E_b and responses.
    pub fn local(&self,elimination:&Elimination,x:f64)->Local{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());let a=self.amplitudes(x);
        let(mut light_shift,mut scattering,mut admixture)=(Dense::zeros(ng,ng),Dense::zeros(ng,ng),Dense::zeros(ng,ng));
        let mut excited_readout=vec![Dense::zeros(ng,ng);ne];
        let responses:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&elimination.responses)).collect();
        for(carrier,r)in self.carriers.iter().zip(&responses){
            light_shift.add_scaled(Complex::ONE,&negative_hermitian_part(&combine(&carrier.beams,&a,&self.couplings).adjoint_product(r)));
            scattering.add_scaled(Complex::ONE,&r.adjoint_product(&self.decay.product(r)));admixture.add_scaled(Complex::ONE,&r.adjoint_product(r));
            for(i,readout)in excited_readout.iter_mut().enumerate(){readout.add_scaled(Complex::ONE,&Dense::from_fn(ng,ng,|j,k|r[(i,j)].conj()*r[(i,k)]));}
        }
        let secular=|m:&Dense|secular_part(&self.block,m);
        Local{light_shift:secular(&light_shift),scattering:secular(&scattering),admixture:secular(&admixture),
            excited_readout:excited_readout.iter().map(secular).collect(),jumps:self.split(&responses)}
    }

    /// The jumps of per-carrier responses r_c: P_to C_a r_c P_from.
    fn split(&self,responses:&[Dense])->Vec<Dense>{
        let whole:Vec<[Dense;3]>=responses.iter().map(|r|std::array::from_fn(|a|self.emission[a].product(r))).collect();
        self.jumps.iter().map(|j|between(&self.block,&whole[j.carrier][j.axis],j.to,j.from)).collect()
    }

    /// The first-order increments of H_ls, Γ and the jumps at x for the excited-energy shifts
    /// `shift(l)` = w_l (rad/us) of absorbing from beam l: R_l → R_l − w_l R2_l.
    pub fn first_order(&self,elimination:&Elimination,x:f64,shift:impl Fn(usize)->f64)->FirstOrder{
        let ng=self.ground_states.len();let a=self.amplitudes(x);
        let weights:Vec<Complex>=a.iter().enumerate().map(|(l,&z)|z*(-shift(l))).collect();
        let responses:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&elimination.responses)).collect();
        let increments:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&weights,&elimination.derivatives)).collect();
        let mut light_shift=Dense::zeros(ng,ng);
        for(carrier,dr)in self.carriers.iter().zip(&increments){
            light_shift.add_scaled(Complex::ONE,&negative_hermitian_part(&combine(&carrier.beams,&a,&self.couplings).adjoint_product(dr)));
        }
        let(jumps,increments)=(self.split(&responses),self.split(&increments));
        let mut scattering=Dense::zeros(ng,ng);
        for(l,d)in jumps.iter().zip(&increments){scattering.add_scaled(Complex::ONE,&l.adjoint_product(d));scattering.add_scaled(Complex::ONE,&d.adjoint_product(l));}
        FirstOrder{light_shift:secular_part(&self.block,&light_shift),scattering,jumps:increments}
    }

    /// Beam l's response R_l to all orders in an excited-energy shift w (rad/us): column b solved
    /// with (K − E_b − δ_c + w)⁻¹. R_l − w R2_l is its first order.
    pub fn response(&self,elimination:&Elimination,beam:usize,shift:f64)->Result<Dense,String>{
        let carrier=self.carriers.iter().find(|c|c.beams.contains(&beam)).unwrap();let coupling=&self.couplings[beam];
        let mut out=Dense::zeros(coupling.rows,coupling.cols);let mut column=vec![Complex::ZERO;coupling.rows];
        for b in 0..coupling.cols{
            let lu=denominator(&elimination.excited_hamiltonian,shift-elimination.energies[b]-carrier.detuning)?;
            for i in 0..coupling.rows{column[i]=coupling[(i,b)];}lu.solve(&mut column);for i in 0..coupling.rows{out[(i,b)]=column[i];}
        }
        Ok(out)
    }

    /// The 32 populations (in the request's order) of a block-diagonal ground density ρ at a
    /// position: the bare ground ones, of ρ − ½{Y, ρ}, and the excited ones, tr(E_i ρ).
    pub fn populations(&self,local:&Local,rho:&Dense)->Vec<f64>{
        let y=&local.admixture;let(left,right)=(y.product(rho),rho.product(y));
        let mut out=vec![0.;STATES];
        for(b,&state)in self.ground_states.iter().enumerate(){out[state]=rho[(b,b)].re-0.5*(left[(b,b)].re+right[(b,b)].re);}
        for(e,&state)in local.excited_readout.iter().zip(&self.excited_states){
            out[state]=(0..rho.rows).map(|j|(0..rho.rows).map(|k|(e[(j,k)]*rho[(k,j)]).re).sum::<f64>()).sum();
        }
        out
    }

    /// The secular step for an elimination: every rotating term of the eliminated generator -- the
    /// stimulated terms V_{c′}†R_c between blocks, and the cross terms L_{a,c} ρ L_{a,c′}† of jumps
    /// from one block into two -- with the expansion parameter η.
    pub fn secular(&self,elimination:&Elimination)->Result<Secular,String>{
        let(nb,nc,ne)=(self.blocks.len(),self.carriers.len(),self.excited_states.len());let mut terms:Vec<([usize;2],[usize;2],f64)>=vec![];
        let(r,v,block)=(&elimination.responses,&self.couplings,&self.block[..]);
        for(c,absorbing)in self.carriers.iter().enumerate(){for(d,emitting)in self.carriers.iter().enumerate(){
            let products:Vec<Dense>=emitting.beams.iter().flat_map(|&e|absorbing.beams.iter().map(move|&l|v[e].adjoint_product(&r[l]))).collect();
            for from in 0..nb{for to in 0..nb{if c!=d||to!=from{terms.push(([c,d],[from,to],products.iter().map(|m|between(block,m,to,from).spectral_norm()).sum()));}}}
        }}
        // Per carrier, axis and pair of blocks: the sum of its beams' jump norms.
        let norms:Vec<f64>=self.jumps.iter().map(|j|self.carriers[j.carrier].beams.iter().map(|&l|between(block,&self.emission[j.axis].product(&r[l]),j.to,j.from).spectral_norm()).sum()).collect();
        let norm=|axis:usize,carrier:usize,to:usize,from:usize|norms[self.jumps.iter().position(|j|*j==Jump{axis,carrier,from,to}).unwrap()];
        for c in 0..nc{for d in 0..nc{for x in 0..nb{for y in 0..nb{if c!=d||x!=y{for from in 0..nb{
            terms.push(([c,d],[y,x],(0..3).map(|a|norm(a,c,x,from)*norm(a,d,y,from)).sum()));
        }}}}}}
        let dropped=terms.into_iter().filter(|term|term.2>0.).map(|([c,d],[from,to],coupling)|{
            let frequency=(self.blocks[to].frame-self.blocks[from].frame)-(self.carriers[c].detuning-self.carriers[d].detuning);
            Rotating{carriers:[c,d],blocks:[from,to],frequency,coupling,micromotion:coupling/frequency.abs()}
        }).collect();
        // H_ls − iΓ/2 = Σ_c Σ_{l′,l∈c} a_l′* a_l [−½(V_l′†R_l + R_l′†V_l) − (i/2) R_l′† D R_l], each |a| ≤ 1.
        let mut scale=0.;
        for carrier in &self.carriers{for &e in &carrier.beams{for &l in &carrier.beams{
            let mut term=v[e].adjoint_product(&r[l]).scaled(Complex::new(-0.5,0.));term.add_scaled(Complex::new(-0.5,0.),&r[e].adjoint_product(&v[l]));
            term.add_scaled(Complex::new(0.,-0.5),&r[e].adjoint_product(&self.decay.product(&r[l])));scale+=secular_part(block,&term).spectral_norm();
        }}}
        // The nearest pole any coupled ground state sees.
        let mut pole=f64::INFINITY;let mut identity=Dense::zeros(ne,ne);for i in 0..ne{identity[(i,i)]=Complex::ONE;}
        for carrier in &self.carriers{for b in 0..self.ground_states.len(){
            if carrier.beams.iter().all(|&l|(0..ne).all(|i|v[l][(i,b)]==Complex::ZERO)){continue;}
            let lu=denominator(&elimination.excited_hamiltonian,-elimination.energies[b]-carrier.detuning)?;
            pole=pole.min(1./lu.solve_columns(&identity).spectral_norm());
        }}
        Ok(Secular{eta:scale/pole,scale,pole,dropped})
    }

    /// Why a Raman-resonant scheme is refused, naming its beams.
    fn raman_resonance(&self,term:&Rotating)->String{
        let named=|c:usize|self.carriers[c].beams.iter().map(|&l|format!("「{}」",self.beams[l].name)).collect::<Vec<_>>().join("、");
        let[c,d]=term.carriers;let beams=if c==d{named(c)}else{format!("{}与{}",named(c),named(d))};
        let[f,g]=term.blocks.map(|b|self.blocks[b].f);
        let what=if f==g{format!("F={f} 内由两组光的拍频驱动的相干")}else{format!("F={} 与 F={} 之间的相干",f.min(g),f.max(g))};
        format!("所选光束之间存在拉曼共振（{beams}，拉曼失谐 {:.3} MHz），{what}不能忽略，1D 运动暂不支持",term.frequency.abs()/TAU)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::linear::uniform;
    use serde_json::Value;

    /// The free-space 1D PGC (the PGC example before the tweezer example) as the app sent it, and
    /// gme.py's numbers for it (tests/fixtures/ground_gme.json).
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
    fn floats(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(float).collect()}
    fn request(v:&Value)->Request{
        let zero=Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]};
        let beams=v["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
            direction:[d[0],d[1],d[2]],wave_number_per_um:float(&b["waveNumberPerUm"]),waist_um:None,intensity_mw_cm2:float(&b["intensityMWcm2"]),
            frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),counter_coupling:zero.clone()}}).collect();
        let e=&v["emission"];
        Request{states:v["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&v["hamiltonian"]),
            frame_frequencies_mhz:vec![0.;STATES],frequencies_mhz:floats(&v["frequenciesMHz"]),beams,
            emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],field_vector_g:[0.;3],mass_kg:float(&v["massKg"]),
            initial_density:vec![Complex::ZERO;STATES*STATES],temperature_uk:30.,times_us:vec![],tolerance:5e-3}
    }
    fn row(values:&[f64])->Dense{Dense::from_fn(1,values.len(),|_,j|Complex::new(values[j],0.))}
    /// Σ_m op[m] e^{imθ} of the reference's harmonics {m: entries}.
    fn at(v:&Value,theta:f64,n:usize)->Dense{
        let mut out=Dense::zeros(n,n);for(m,e)in v.as_object().unwrap(){out.add_scaled(Complex::cis(m.parse::<f64>().unwrap()*theta),&entries(e,n,n));}out
    }
    /// max |a − b| / max |b| over the pairs.
    fn relative(pairs:impl IntoIterator<Item=(Dense,Dense)>)->f64{
        let(mut difference,mut scale)=(0_f64,0_f64);
        for(a,b)in pairs{for(x,y)in a.entries.iter().zip(&b.entries){difference=difference.max((*x-*y).abs());scale=scale.max(y.abs());}}
        difference/scale
    }

    /// With plane waves of one shared wave number k (the lattice phase θ = kx), the generator at x is
    /// gme.py's harmonic form at θ -- the research module GMW-o1 was validated with -- to round-off,
    /// operator by operator (the reference has 12 significant digits).
    #[test]
    fn plane_waves_of_one_wave_number_have_the_operators_of_gme(){
        let f=fixture();let r=&f["reference"];let k=float(&r["kref"]);
        let mut pgc=request(&f["request"]);for beam in &mut pgc.beams{beam.wave_number_per_um=k;}
        let g=Ground::new(&pgc).unwrap();let e=g.eliminate(&pgc.hamiltonian).unwrap();
        let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
        // The beam of a reference carrier and direction.
        let beam=|c:usize,s:i64|*g.carriers[c].beams.iter().find(|&&l|g.beams[l].direction[2] as i64==s).unwrap();
        let pairs=|which:&str,responses:&[Dense]|->Vec<(Dense,Dense)>{r[which].as_object().unwrap().iter().map(|(key,v)|{
            let k:Vec<i64>=key.split(',').map(|x|x.parse().unwrap()).collect();let c=k[0] as usize;
            (secular_part(&g.block,&g.couplings[beam(c,k[1])].adjoint_product(&responses[beam(c,k[2])])),entries(v,ng,ng))
        }).collect()};
        let jumps=r["jumps"].as_array().unwrap();
        let jump=|j:&Value|g.jumps.iter().position(|x|*x==Jump{axis:index(&j["axis"]),carrier:index(&j["carrier"]),to:index(&j["to"]),from:index(&j["from"])}).unwrap();
        let detunings=|v:&Value|row(&v.as_array().unwrap().iter().map(|c|float(&c["delta"])).collect::<Vec<_>>());
        let kv=float(&r["firstOrder"]["kv"]);let mut checks=vec![
            ("X",relative(pairs("X",&e.responses))),("P",relative(pairs("P",&e.derivatives))),
            ("carriers: δ",relative([(row(&g.carriers.iter().map(|c|c.detuning).collect::<Vec<_>>()),detunings(&r["carriers"]))])),
        ];
        for theta in [0.,0.7,2.3,-4.1]{
            let(local,first)=(g.local(&e,theta/k),g.first_order(&e,theta/k,|l|g.beams[l].direction[2]*kv));
            // gme's first-order jumps are −s kv D2[s] in harmonic s.
            let derivative=|j:&Value|{let mut d=Dense::zeros(ng,ng);for(s,m)in j["D2"].as_object().unwrap(){let s:f64=s.parse().unwrap();d.add_scaled(Complex::cis(s*theta)*(-s*kv),&entries(m,ng,ng));}d};
            checks.extend([("H_ls",relative([(local.light_shift.clone(),at(&r["Hls"],theta,ng))])),("Γ",relative([(local.scattering.clone(),at(&r["Gam"],theta,ng))])),
                ("Y",relative([(local.admixture.clone(),at(&r["Y"],theta,ng))])),
                ("jumps L",relative(jumps.iter().map(|j|(local.jumps[jump(j)].clone(),at(&j["L"],theta,ng))))),
                ("jumps D2",relative(jumps.iter().map(|j|(first.jumps[jump(j)].clone(),derivative(j))))),
                ("first order",relative([(first.light_shift.clone(),at(&r["firstOrder"]["dHls"],theta,ng)),(first.scattering.clone(),at(&r["firstOrder"]["dGam"],theta,ng))]))]);
        }
        let exact=&r["response"];let shifted=g.response(&e,beam(index(&exact["carrier"]),exact["axial"].as_i64().unwrap()),float(&exact["shift"])).unwrap();
        checks.push(("exact response",relative([(shifted,entries(&exact["R"],ne,ng))])));
        let readout=&r["readout"];let p=g.populations(&g.local(&e,float(&readout["theta"])/k),&entries(&readout["rho"],ng,ng));
        let(ground,excited):(Vec<f64>,Vec<f64>)=(g.ground_states.iter().map(|&i|p[i]).collect(),g.excited_states.iter().map(|&i|p[i]).collect());
        checks.push(("populations",relative([(row(&ground),row(&floats(&readout["ground"]))),(row(&excited),row(&floats(&readout["excited"])))])));
        assert!(checks.iter().all(|(_,x)|*x<=1e-10),"{checks:?}");
    }

    /// The free-space PGC perturbed at random: the ground energies split as by a field, every beam's
    /// coupling scaled by a complex factor, an oblique plane wave added to the cooling carrier and a
    /// beam focused at the atom across the line, at a frequency of its own.
    fn perturbed(seed:u64)->Request{
        let mut next=uniform(seed);let mut r=request(&fixture()["request"]);
        let scale=|m:&mut Matrix,z:Complex|for(re,im)in m.real.iter_mut().zip(m.imag.iter_mut()){let x=z*Complex::new(*re,*im);(*re,*im)=(x.re,x.im);};
        for i in (0..STATES).filter(|&i|r.states[i].starts_with("ground-")){r.hamiltonian.real[i*STATES+i]+=next();}
        for beam in &mut r.beams{scale(&mut beam.coupling,Complex::new(1.+next(),next()));}
        let cooling=r.beams.iter().find(|b|b.id=="cool-plus").unwrap().clone();
        let mut oblique=cooling.clone();(oblique.id,oblique.name,oblique.direction)=("oblique".into(),"斜光".into(),[0.6,0.,0.8]);
        scale(&mut oblique.coupling,Complex::new(next(),next()));
        let mut focused=cooling;(focused.id,focused.name,focused.direction,focused.waist_um)=("focused".into(),"聚焦光".into(),[0.,1.,0.],Some(1.5));
        focused.frequency_index=focused.frequency_index.iter().map(|n|2*n).collect();scale(&mut focused.coupling,Complex::new(3.+next(),next()));
        r.beams.extend([oblique,focused]);r
    }
    /// Positions spread over a few wavelengths and the focused beam's waist (um).
    fn positions(seed:u64)->Vec<f64>{let mut next=uniform(seed);(0..3).map(|_|4.*next()).collect()}
    fn mismatch(a:&Dense,b:&Dense)->(f64,f64){let mut d=a.clone();d.add_scaled(-Complex::ONE,b);(d.largest(),b.largest())}

    /// The trace is kept: at every position the jumps, split into blocks, carry the whole
    /// scattering, Σ_j L_j†L_j = Γ.
    #[test]
    fn the_jumps_carry_the_whole_scattering(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();let e=g.eliminate(&r.hamiltonian).unwrap();
            for x in positions(seed){
                let local=g.local(&e,x);let mut sum=Dense::zeros(g.ground_states.len(),g.ground_states.len());
                for l in &local.jumps{sum.add_scaled(Complex::ONE,&l.adjoint_product(l));}
                let(error,scale)=mismatch(&sum,&local.scattering);assert!(error<=1e-13*scale,"seed {seed}, x {x}: {error:e} of {scale:e}");
            }
        }
    }

    /// Γ is the decay of the second-order shift X = Σ_c V_c†R_c: Γ = −i(X − X†) + i[Y, H0g] (the
    /// commutator, from the column-wise resolvents, vanishes for degenerate ground states). So the
    /// decay inside K and the emission operators agree, and R solves its column equations.
    #[test]
    fn the_scattering_is_the_decay_of_the_second_order_shift(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();let e=g.eliminate(&r.hamiltonian).unwrap();let n=g.ground_states.len();
            for x in positions(seed){
                let(local,a)=(g.local(&e,x),g.amplitudes(x));let mut shift=Dense::zeros(n,n);
                for c in &g.carriers{shift.add_scaled(Complex::ONE,&combine(&c.beams,&a,&g.couplings).adjoint_product(&combine(&c.beams,&a,&e.responses)));}
                let shift=secular_part(&g.block,&shift);let y=&local.admixture;
                let mut decay=shift.scaled(-Complex::I);decay.add_scaled(Complex::I,&shift.adjoint());
                decay.add_scaled(Complex::I,&y.product(&e.hamiltonian));decay.add_scaled(-Complex::I,&e.hamiltonian.product(y));
                let(error,scale)=mismatch(&decay,&local.scattering);assert!(error<=1e-12*scale,"seed {seed}, x {x}: {error:e} of {scale:e}");
            }
        }
    }

    /// The readout keeps the trace: Σ_i E_i = Y, and the 32 populations of a density sum to one.
    #[test]
    fn the_populations_sum_to_one(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();let e=g.eliminate(&r.hamiltonian).unwrap();let n=g.ground_states.len();
            let mut next=uniform(seed);let a=Dense::from_fn(n,n,|i,j|if g.block[i]==g.block[j]{Complex::new(next(),next())}else{Complex::ZERO});
            let mut rho=a.product(&a.adjoint());let trace:f64=(0..n).map(|i|rho[(i,i)].re).sum();rho=rho.scaled(Complex::new(1./trace,0.));
            for x in positions(seed){
                let local=g.local(&e,x);let mut total=Dense::zeros(n,n);for readout in &local.excited_readout{total.add_scaled(Complex::ONE,readout);}
                let(error,scale)=mismatch(&total,&local.admixture);assert!(error<=1e-13*scale,"seed {seed}, x {x}: {error:e} of {scale:e}");
                let sum:f64=g.populations(&local,&rho).iter().sum();assert!((sum-1.).abs()<=1e-14,"seed {seed}, x {x}: {sum}");
            }
        }
    }

    /// R − w R2 is the exact response to first order: column b's remainder is w² (K − E_b − δ_c)⁻² R(w)_b,
    /// at most (w/pole)² of R(w)_b. At w = 0 the exact response is R itself.
    #[test]
    fn the_first_order_response_leaves_a_second_order_remainder(){
        let r=perturbed(1);let g=Ground::new(&r).unwrap();let e=g.eliminate(&r.hamiltonian).unwrap();let pole=g.secular(&e).unwrap().pole;
        let norm=|m:&Dense,b:usize|(0..m.rows).map(|i|m[(i,b)].norm_sqr()).sum::<f64>().sqrt();
        for l in 0..g.beams.len(){
            assert_eq!(g.response(&e,l,0.).unwrap(),e.responses[l]);
            for w in [pole/8.,-pole/3.]{
                let exact=g.response(&e,l,w).unwrap();let mut remainder=exact.clone();
                remainder.add_scaled(-Complex::ONE,&e.responses[l]);remainder.add_scaled(Complex::new(w,0.),&e.derivatives[l]);
                for b in 0..g.ground_states.len(){
                    let bound=(w/pole).powi(2)*norm(&exact,b)+f64::EPSILON*(norm(&e.responses[l],b)+w.abs()*norm(&e.derivatives[l],b));
                    assert!(norm(&remainder,b)<=bound,"beam {l}, w {w}, column {b}: {} > {bound}",norm(&remainder,b));
                }
            }
        }
    }

    /// The cooler moved to a Raman resonance with the repumper (both on F′ = 2: a Λ scheme) is refused,
    /// naming the beams and the Raman detuning. In the PGC the cooler is −f0 from the repumper
    /// (frequency axis 0), so f0 = (ν_2 − ν_1)/2π − δ_R puts the Raman detuning at δ_R.
    #[test]
    fn a_raman_resonant_lambda_scheme_is_refused(){
        let pgc=request(&fixture()["request"]);let g=Ground::new(&pgc).unwrap();
        let hyperfine=(g.blocks[1].frame-g.blocks[0].frame)/TAU;
        for detuning in [0.,0.1]{
            let mut lambda=pgc.clone();lambda.frequencies_mhz[0]=hyperfine-detuning;
            let error=Ground::new(&lambda).unwrap_err();
            assert!(error.starts_with("所选光束之间存在拉曼共振（")&&error.contains("「冷却光 +z」、「冷却光 −z」")&&error.contains("「回泵光 +z」、「回泵光 −z」")
                &&error.contains(&format!("拉曼失谐 {detuning:.3} MHz"))&&error.ends_with("F=1 与 F=2 之间的相干不能忽略，1D 运动暂不支持"),"{error}");
        }
    }
}
