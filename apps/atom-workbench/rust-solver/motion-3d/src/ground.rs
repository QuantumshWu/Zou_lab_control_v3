//! The atom's ground manifold with the excited states eliminated: the generator of its internal
//! state at any point r of the trap, on clusters of states, to first order in the Doppler and
//! recoil shifts.
//!
//! # Far components
//! Every beam drives a co-rotating component, V_l e^{−iδ_l t} with δ_l = 2π(ν_l − ν_frame), and a
//! counter-rotating one, R_l e^{+i2π(ν_l + ν_frame)t}, each with its amplitude a_l(r). They are
//! classified by the rule of the population core (`classify`) against the 3D solve's budget
//! (ledger.rs): a component folds when it is far, |ν| ≥ FOLD_DISTANCE ‖L0‖ with its mixing
//! m = size/(|ν| − ‖L0‖) within m² ≤ budget, beyond the largest frequency gap of at least
//! `fold_gap` = 1/√budget above every component that may not fold. A folded component enters as
//! its second-order (van Vleck) shift F on the ground and the excited states, with the exact
//! bare-energy denominators; the others are near carriers. With each beam's static shift through
//! the states outside the model O_l (in H0 at the focus),
//!
//! ```text
//! H(r) = H0 + Σ_l (|a_l(r)|² − 1) O_l + Σ_l |a_l(r)|² F_l      (`Ground::hamiltonian_at`)
//! ```
//!
//! # Elimination
//! The near components of one frequency (equal `Beam::frequency_index`) form a carrier c at δ_c.
//! Each beam l keeps its own coupling V_l at the focus and its own amplitude a_l(r), so carrier c
//! drives V_c(r) = Σ_{l∈c} a_l(r) V_l e^{−iδ_c t} + h.c. With the ground energies E_b, the diagonal
//! of H(r)'s ground block, and the excited states' non-Hermitian Hamiltonian
//! K = H(r)_ee − (i/2) D, D = Σ_a C_a†C_a|_ee, the excited states and the optical coherences follow
//! the ground states adiabatically, beam by beam and column by column (`Elimination`):
//!
//! ```text
//! R_l[:,b] = (K − E_b − δ_c)⁻¹ V_l[:,b]        R2_l[:,b] = (K − E_b − δ_c)⁻² V_l[:,b]        R_c(r) = Σ_{l∈c} a_l(r) R_l
//! ```
//!
//! # Couplings and clusters
//! In the frame where every ground state b rotates at its own energy, the eliminated generator
//! couples the states pairwise, each coupling rotating at its pair's separation:
//! - absorbing carrier c from b and emitting into c′, landing in a (the channels (c, b) and
//!   (c′, a)): X_ab = [−½(V_c′†R_c + R_c′†V_c) − (i/2) R_c′†D R_c]_ab at
//!   Ω = (E_a + δ_c′) − (E_b + δ_c) -- for a = b and c = c′ the light shift and scattering of b,
//!   between the sublevels of a hyperfine level (c = c′) the light's own couplings at their Zeeman
//!   separation, between F = 1 and F = 2 the Raman terms at the Raman detuning, on one state
//!   (a = b, c ≠ c′) the beat of two carriers;
//! - H's static couplings between ground states, at their separation;
//! - the two photons of a jump from b landing in a₁ and a₂, interfering at the difference of their
//!   frequencies.
//!
//! Every coupling is judged by the rule of the far components (`classify`): its frequency the
//! separation (with H(r)'s Zeeman and state-dependent shifts, the worst over the trap; the light's
//! own shifts and rates bounded in the eliminated generator's ‖L0‖), its size the bound of its
//! superoperator over the trap (every beam at most its peak amplitude). A folded coupling enters at
//! second order:
//! - a pair of channels: its van Vleck shift from the far channel's path (c, b) → (c′, a) → (ĉ, b̂),
//!   (ĉ, b̂) co-resonant with (c, b), W̃ D U with U = −V_c′† G V_c and W̃ = −V_ĉ† G V_c′ at the path's
//!   energy z (the cluster's centroid in H(r) plus δ_c, G = (K − z)⁻¹) and D = 1/(z − E_a − δ_c′);
//! - a static coupling: its van Vleck shift with the exact bare-energy denominators;
//! - a pair of photons: they are distinguishable, and the jump's channels split by frequency.
//!
//! Kept couplings are explicit. The states joined by kept couplings form a cluster (`Cluster`),
//! the trajectory's state space between jumps: near B = 0 a whole hyperfine level, at large fields
//! single sublevels (the rate-equation limit), F = 1 and F = 2 together only through a
//! Raman-resonant pair (a Λ scheme, gray molasses, EIT). A cluster's frame (`Cluster::frames`)
//! makes the strongest kept couplings static; the others keep their residual beat e^{iΩ̃t}, which
//! the time step resolves. The clusters are fixed for a run, judged over the trap. A jump projects
//! onto the cluster it lands in, its channels its classes of photons of one frequency (`Class`):
//!
//! ```text
//! H_eff = Σ_b (E_b − θ_b) |b⟩⟨b| + Σ_kept e^{iΩ̃t} X_ab |a⟩⟨b| + Σ_paths e^{iΩ̃t} W̃ D U,
//! L_{x,φ} = Σ_{(c,b→a)∈φ} e^{iω̃t} [C_x R_c]_ab |a⟩⟨b|,       ω̃ = θ_a − θ_b − δ_c
//! ```
//!
//! The clusters are lists of states, their machinery general over them: an excited state whose
//! couplings must be kept could join one.
//!
//! Readout: the bare ground density is ρ − ½{Y, ρ}, Y = Σ R′†R over the co-resonant channel
//! pairs, and an excited state's population tr(E_i ρ), E_i = Σ R′†|i⟩⟨i|R.
//!
//! # First order in the Doppler and recoil shifts
//! Absorbing from beam l shifts the excited state by the kinetic energy w_l = (ħ/m)(k_l·κ + k_l²/2);
//! to first order R_l → R_l − w_l R2_l, and `Ground::response` gives the shifted response to all
//! orders.
//!
//! # Approximations and their estimates (the solve's ledger)
//! - the far components' fold: Σ m²;
//! - the couplings' fold: Σ m² of the folded pairs of channels and static couplings;
//! - the zeroth-order elimination: η², the excited population along the trajectories (lib.rs);
//! - first order in the Doppler shift: (kv/σ_min)² (engine.rs);
//! - an emitted photon has its carrier's wave number: ≤ 1e-7 on the D2 line.
use crate::ledger::BUDGET;
use crate::linear::{Dense,Lu};
use crate::{Beam,Complex,Request,STATES};
use atom_population_core::lattice::classification::FOLD_DISTANCE;
use std::f64::consts::TAU;

/// A beam's co-rotating component (`counter` false: V e^{−iνt}, ν = δ = 2π(ν_beam − ν_frame)) or its
/// counter-rotating one (R e^{−iνt}, ν = −2π(ν_beam + ν_frame)), each with its Hermitian conjugate;
/// its size 2‖V‖_F bounds its commutator, and `folded` is the rule's verdict.
#[derive(Clone,Debug)]
pub struct Component{pub beam:usize,pub counter:bool,pub nu:f64,pub size:f64,pub folded:bool}
/// The near components of one frequency.
#[derive(Clone,Debug)]
pub struct Carrier{
    /// Its beams (indices into `Ground::beams`).
    pub beams:Vec<usize>,
    /// δ_c = 2π(ν_c − ν_frame) (rad/us).
    pub detuning:f64,
    /// Its photons' wave number |k| (rad/um).
    pub wave_number:f64,
}
/// An absorption channel: carrier `carrier` absorbed from ground state `state` (a position in
/// `Ground::ground_states`).
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub struct Channel{pub carrier:usize,pub state:usize}
/// A jump's photon: carrier `carrier` absorbed from ground state `from`, the atom landing in `to`.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Photon{pub carrier:usize,pub from:usize,pub to:usize}
/// What a coupling of the eliminated generator is.
#[derive(Clone,Debug,PartialEq)]
pub enum Kind{
    /// H's static coupling between ground states `a` and `b`.
    Static{a:usize,b:usize},
    /// The coupling of two channels: absorbing one, emitting into the other's carrier and landing
    /// in its state, and back.
    Raman{first:Channel,second:Channel},
    /// Two photons of a jump from one state into two.
    Photons{first:Photon,second:Photon},
}
/// A coupling, its frequency (the worst separation over the trap, rad/us), its size (the bound of
/// its superoperator) and the rule's verdict.
#[derive(Clone,Debug)]
pub struct Item{pub kind:Kind,pub frequency:f64,pub size:f64,pub folded:bool}

/// The kept couplings of a cluster from one carrier to another: absorbing `absorbed`, emitting
/// into `emitted`, each entry (row a, column b, in the cluster's order) with its residual beat
/// Ω̃ = θ_a − θ_b + δ_emitted − δ_absorbed (rad/us).
#[derive(Clone,Debug)]
pub struct Group{pub absorbed:usize,pub emitted:usize,pub entries:Vec<(usize,usize,f64)>}
/// A folded pair's van Vleck path from the cluster: channel `from` (in the cluster), the far
/// channel `via`, back into `back` (in the cluster, co-resonant with `from`), its residual beat
/// θ_back − θ_from + δ_back − δ_from.
#[derive(Clone,Copy,Debug)]
pub struct Path{pub from:Channel,pub via:Channel,pub back:Channel,pub residual:f64}
/// A jump channel from a cluster: the photons of one frequency landing in cluster `target`, each
/// with its residual ω̃ − ω̃₀ (rad/us), and their wave number.
#[derive(Clone,Debug)]
pub struct Class{pub target:usize,pub photons:Vec<(Photon,f64)>,pub wave_number:f64}
/// A cluster: its states (ground positions) and each one's frame θ (rad/us), its kept couplings
/// (per carrier pair, and H's static ones: row, column), its van Vleck paths (the folded pairs of
/// channels', and the folded static couplings': row, column and the far ground state, between
/// states of one frame), its jumps' classes and its co-resonant channels (each with its residual,
/// the readout's coherent sums).
#[derive(Clone,Debug)]
pub struct Cluster{
    pub states:Vec<usize>,
    pub frames:Vec<f64>,
    pub groups:Vec<Group>,
    pub statics:Vec<(usize,usize)>,
    pub paths:Vec<Path>,
    pub static_paths:Vec<(usize,usize,usize)>,
    pub classes:Vec<Class>,
    pub coherent:Vec<Vec<(Channel,f64)>>,
}
impl Cluster{
    /// A ground state's place in the cluster.
    pub fn position(&self,state:usize)->Option<usize>{self.states.iter().position(|&s|s==state)}
}

/// The elimination for one H(r): the ground block of H (its diagonal the energies E_b), K, and
/// every near beam's excited response.
#[derive(Clone,Debug)]
pub struct Elimination{
    pub energies:Vec<f64>,
    pub ground:Dense,
    /// K, the excited states' non-Hermitian Hamiltonian (rad/us).
    pub excited_hamiltonian:Dense,
    /// Per beam, R_l and R2_l, excited ← ground (dimensionless and us).
    pub responses:Vec<Dense>,
    pub derivatives:Vec<Dense>,
}

/// A cluster's generator at a point and a time (the dense reference): H_eff and the share of it
/// the folded pairs' paths make, every class's jumps per axis (target cluster × cluster), the
/// readout's Y and E_i (per excited state).
#[derive(Clone,Debug)]
pub struct Local{pub effective:Dense,pub shift:Dense,pub jumps:Vec<[Dense;3]>,pub admixture:Dense,pub excited_readout:Vec<Dense>}

/// The eliminated ground-manifold generator of a request: the operators act on the ground states
/// in the order of `ground_states`, the excited ones in the order of `excited_states`; rates in
/// rad/us, jumps in 1/sqrt(us).
#[derive(Clone,Debug)]
pub struct Ground{
    /// The request's ground and excited states (indices into `Request::states`).
    pub ground_states:Vec<usize>,
    pub excited_states:Vec<usize>,
    /// Each ground state's F.
    pub levels:Vec<u8>,
    /// D = Σ_a C_a†C_a on the excited states, and C_a (ground ← excited) on the request's axes.
    pub decay:Dense,
    pub emission:[Dense;3],
    /// The request's beams, and each one's co-rotating V_l (excited ← ground at the focus, rad/us).
    pub beams:Vec<Beam>,
    pub couplings:Vec<Dense>,
    /// H0 at the focus (32 × 32, rad/us); per beam its shift through the states outside the model O_l
    /// and the fold F_l of its folded components (both 32 × 32, at its intensity at the focus).
    pub hamiltonian:Dense,
    pub outside:Vec<Dense>,
    pub folds:Vec<Dense>,
    /// Every beam's two components, classified, and the bare ‖L0‖ they were judged against.
    pub components:Vec<Component>,
    pub l0:f64,
    pub carriers:Vec<Carrier>,
    /// The ground states' couplings, classified, the rule's cut between near and far frequencies
    /// and the eliminated generator's ‖L0‖ they were judged against (rad/us).
    pub items:Vec<Item>,
    pub cut:f64,
    pub generator_scale:f64,
    /// The nearest excited pole any coupled ground state sees over the trap, min σ_min(K − E_b − δ_c),
    /// overall and per carrier.
    pub pole:f64,
    pub poles:Vec<f64>,
    /// The clusters, and each ground state's cluster.
    pub clusters:Vec<Cluster>,
    pub cluster:Vec<usize>,
}

/// The manifold and F of a state label `manifold-F-m`.
fn level(label:&str)->Result<(&str,u8),String>{
    let mut parts=label.splitn(3,'-');
    match(parts.next(),parts.next().and_then(|f|f.parse().ok()),parts.next()){
        (Some(manifold),Some(f),Some(_))=>Ok((manifold,f)),_=>Err(format!("The state label {label} is not manifold-F-m")),
    }
}
/// The LU factors of K + shift, the denominator of an excited response.
pub(crate) fn denominator(k:&Dense,shift:f64)->Result<Lu,String>{
    let mut a=k.clone();for i in 0..a.rows{a[(i,i)]+=Complex::new(shift,0.);}
    Lu::new(&a).ok_or_else(||"The excited states' resolvent is singular".to_string())
}
/// Σ_l weight_l m_l over the given beams.
fn combine(beams:&[usize],weights:&[Complex],matrices:&[Dense])->Dense{
    let mut out=Dense::zeros(matrices[0].rows,matrices[0].cols);for &l in beams{out.add_scaled(weights[l],&matrices[l]);}out
}
/// The rows × cols block of a square matrix.
fn sub(m:&Dense,rows:&[usize],cols:&[usize])->Dense{Dense::from_fn(rows.len(),cols.len(),|i,j|m[(rows[i],cols[j])])}
fn frobenius(m:&Dense)->f64{m.entries.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt()}
/// The second-order (van Vleck) shift of a coupling V e^{−iνt} + h.c. (V excited ← ground) with the
/// exact bare-energy denominators: on the ground states ½ Σ_e V_ea* V_ea′ [1/(E_a + ν − E_e) + 1/(E_a′ + ν − E_e)],
/// on the excited ones ½ Σ_a V_ea V_e′a* [1/(E_e − ν − E_a) + 1/(E_e′ − ν − E_a)].
fn van_vleck(energies:&[f64],v:&Dense,nu:f64,ground:&[usize],excited:&[usize])->Dense{
    let mut out=Dense::zeros(STATES,STATES);
    for(i,&a)in ground.iter().enumerate(){for(j,&b)in ground.iter().enumerate(){let mut z=Complex::ZERO;
        for(e,&x)in excited.iter().enumerate(){z+=v[(e,i)].conj()*v[(e,j)]*(0.5*(1./(energies[a]+nu-energies[x])+1./(energies[b]+nu-energies[x])));}
        out[(a,b)]=z;}}
    for(e,&x)in excited.iter().enumerate(){for(f,&y)in excited.iter().enumerate(){let mut z=Complex::ZERO;
        for(i,&a)in ground.iter().enumerate(){z+=v[(e,i)]*v[(f,i)].conj()*(0.5*(1./(energies[x]-nu-energies[a])+1./(energies[y]-nu-energies[a])));}
        out[(x,y)]=z;}}
    out
}

/// The smallest ratio of a fold's frequency gap, far over near: the first order the fold leaves
/// out, (ν_near/ν_far)², within the budget.
pub fn fold_gap()->f64{1./BUDGET.sqrt()}
/// m = size/(|ν| − l0), a coupling's mixing.
pub fn mixing(nu:f64,size:f64,l0:f64)->f64{size/(nu.abs()-l0).max(f64::MIN_POSITIVE)}
/// The rule of the population core (population-core lattice/classification.rs) against the 3D
/// budget: an item of frequency ν and size g may fold when |ν| ≥ FOLD_DISTANCE l0 with
/// m² = (g/(|ν| − l0))² ≤ budget; the significant frequencies are cut at their largest ratio gap of
/// at least `fold_gap` above every one that may not fold, and the items that may fold beyond the
/// cut fold (an item of size zero folds trivially). The verdicts (true: folded) and the cut.
pub fn classify(items:&[(f64,f64)],l0:f64)->(Vec<bool>,f64){
    let foldable=|nu:f64,size:f64|nu.abs()>=FOLD_DISTANCE*l0&&mixing(nu,size,l0).powi(2)<=BUDGET;
    let mut frequencies:Vec<(f64,bool)>=items.iter().filter(|i|i.1>0.).map(|&(nu,size)|(nu.abs(),foldable(nu,size))).collect();
    frequencies.sort_by(|a,b|a.0.total_cmp(&b.0));
    frequencies.dedup_by(|a,b|{let same=a.0==b.0;if same{b.1&=a.1;}same});
    let first=frequencies.iter().rposition(|f|!f.1).map_or(0,|i|i+1);
    let mut best:Option<(f64,usize)>=None;
    for i in first..frequencies.len(){
        let ratio=if i==0{f64::INFINITY}else{frequencies[i].0/frequencies[i-1].0};
        if i>0&&ratio<fold_gap(){continue;}
        if best.is_none_or(|(r,_)|ratio>r){best=Some((ratio,i));}
    }
    let cut=match best{None=>f64::INFINITY,Some((_,0))=>(FOLD_DISTANCE*l0).min(frequencies[0].0),Some((_,i))=>(frequencies[i-1].0*frequencies[i].0).sqrt()};
    (items.iter().map(|&(nu,size)|size==0.||nu.abs()>=cut&&foldable(nu,size)).collect(),cut)
}

/// A union of sets over 0..n: each element's representative.
struct Union{parent:Vec<usize>}
impl Union{
    fn new(n:usize)->Union{Union{parent:(0..n).collect()}}
    fn find(&mut self,x:usize)->usize{let mut x=x;while self.parent[x]!=x{self.parent[x]=self.parent[self.parent[x]];x=self.parent[x];}x}
    fn join(&mut self,a:usize,b:usize){let(a,b)=(self.find(a),self.find(b));if a!=b{self.parent[a.max(b)]=a.min(b);}}
    /// The sets, each in increasing order, by their smallest element.
    fn sets(&mut self)->Vec<Vec<usize>>{
        let n=self.parent.len();let mut out:Vec<Vec<usize>>=vec![];let mut at=vec![usize::MAX;n];
        for x in 0..n{let r=self.find(x);if at[r]==usize::MAX{at[r]=out.len();out.push(vec![]);}out[at[r]].push(x);}
        out
    }
}

/// The bounds over the trap the couplings are judged by, from the eliminations at intensities
/// spanning every Gaussian beam's range: per pair of channels the largest |X_ab| and |X_ba| (each
/// beam at most its peak), per photon its largest amplitude per axis, per pair of states the
/// largest static coupling, the worst separations, the eliminated generator's scale and the pole.
struct Bounds{
    /// [c][c′][a][b]: the bound of |X_ab| absorbing c, emitting c′.
    raman:Vec<Vec<Vec<Vec<f64>>>>,
    /// [c][b][a][axis]: the bound of |[C_axis R_c]_ab|.
    photons:Vec<Vec<Vec<[f64;3]>>>,
    /// [a][b]: the largest |H_ab|.
    statics:Vec<Vec<f64>>,
    /// [a][b]: the largest E_a − E_b over the intensities (−high[b][a] the smallest).
    high:Vec<Vec<f64>>,
    /// The eliminated generator's light shift and scattering (rad/us): bounds over the trap.
    light:f64,scattering:f64,
    poles:Vec<f64>,
}

impl Ground{
    /// The generator of a request, or why it has none: a counter-rotating component that may not
    /// fold, or a static coupling between ground states that may not.
    pub fn new(request:&Request)->Result<Ground,String>{
        let labels=request.states.iter().map(|s|level(s)).collect::<Result<Vec<_>,_>>()?;
        let ground_states:Vec<usize>=(0..STATES).filter(|&i|labels[i].0=="ground").collect();
        let excited_states:Vec<usize>=(0..STATES).filter(|&i|labels[i].0!="ground").collect();
        let ne=excited_states.len();
        let driven=|k:usize|labels[k/STATES].0!="ground"&&labels[k%STATES].0=="ground";
        for beam in &request.beams{for m in [&beam.coupling,&beam.counter_coupling]{
            if (0..STATES*STATES).any(|k|!driven(k)&&(m.real[k]!=0.||m.imag[k]!=0.)){return Err(format!("Beam {}'s couplings must be excited ← ground",beam.id));}
        }}
        let all:Vec<usize>=(0..STATES).collect();
        let mut decay=Dense::zeros(ne,ne);
        for c in &request.emission{let c=Dense::block(c,&all,&excited_states);decay.add_scaled(Complex::ONE,&c.adjoint_product(&c));}
        let emission:[Dense;3]=std::array::from_fn(|a|Dense::block(&request.emission[a],&ground_states,&excited_states));
        let hamiltonian=Dense::block(&request.hamiltonian,&all,&all);
        if (0..STATES).any(|i|(0..STATES).any(|j|i!=j&&hamiltonian[(i,j)]!=Complex::ZERO)){return Err("H0 must be diagonal in the model basis".into());}
        let energies:Vec<f64>=(0..STATES).map(|i|hamiltonian[(i,i)].re).collect();
        // Every component, classified by the population core's rule. ‖L0‖ with H0's diagonal centred:
        // its spread, the decay and the emission.
        let frame=request.frame_frequencies_mhz[excited_states[0]];
        let spread=energies.iter().copied().fold(f64::NEG_INFINITY,f64::max)-energies.iter().copied().fold(f64::INFINITY,f64::min);
        let l0=spread+frobenius(&decay)+emission.iter().map(|c|frobenius(c).powi(2)).sum::<f64>();
        let couplings:Vec<Dense>=request.beams.iter().map(|b|Dense::block(&b.coupling,&excited_states,&ground_states)).collect();
        let counters:Vec<Dense>=request.beams.iter().map(|b|Dense::block(&b.counter_coupling,&excited_states,&ground_states)).collect();
        let detuning=|beam:&Beam|beam.frequency_index.iter().zip(&request.frequencies_mhz).map(|(&n,&f)|n as f64*f).sum::<f64>();
        let mut components:Vec<Component>=request.beams.iter().enumerate().flat_map(|(l,beam)|{let delta=detuning(beam);
            [Component{beam:l,counter:false,nu:TAU*delta,size:2.*frobenius(&couplings[l]),folded:false},
             Component{beam:l,counter:true,nu:-TAU*(delta+2.*frame),size:2.*frobenius(&counters[l]),folded:false}]}).collect();
        let(verdicts,_)=classify(&components.iter().map(|c|(c.nu,c.size)).collect::<Vec<_>>(),l0);
        for(c,v)in components.iter_mut().zip(verdicts){c.folded=v;}
        if let Some(c)=components.iter().find(|c|c.counter&&!c.folded&&c.size>0.){
            return Err(format!("The counter-rotating coupling of 「{}」 is too strong to fold (population core's rule): 3D motion cannot take it",request.beams[c.beam].name));}
        let mut folds=vec![Dense::zeros(STATES,STATES);request.beams.len()];
        for c in components.iter().filter(|c|c.folded&&c.size>0.){
            let v=if c.counter{&counters[c.beam]}else{&couplings[c.beam]};folds[c.beam].add_scaled(Complex::ONE,&van_vleck(&energies,v,c.nu,&ground_states,&excited_states));
        }
        let outside=request.beams.iter().map(|b|Dense::block(&b.outside_shift,&all,&all)).collect();
        // The near carriers: the near co-rotating components of one frequency.
        let mut groups:Vec<Vec<usize>>=vec![];
        for c in components.iter().filter(|c|!c.counter&&!c.folded&&c.size>0.){
            let beam=&request.beams[c.beam];
            match groups.iter_mut().find(|g|request.beams[g[0]].frequency_index==beam.frequency_index){Some(g)=>g.push(c.beam),None=>groups.push(vec![c.beam])}
        }
        let carriers:Vec<Carrier>=groups.into_iter().map(|beams|{
            let first=&request.beams[beams[0]];Carrier{beams,detuning:TAU*detuning(first),wave_number:first.wave_number_per_um}
        }).collect();
        let levels=ground_states.iter().map(|&i|labels[i].1).collect();
        let mut ground=Ground{ground_states,excited_states,levels,decay,emission,beams:request.beams.clone(),couplings,hamiltonian,outside,folds,components,l0,carriers,
            items:vec![],cut:f64::INFINITY,generator_scale:0.,pole:f64::INFINITY,poles:vec![],clusters:vec![],cluster:vec![]};
        ground.cluster_states()?;
        Ok(ground)
    }

    /// H at a point r (um from the focus): H0 with each beam's outside shift and fold at its
    /// intensity there.
    pub fn hamiltonian_at(&self,r:[f64;3])->Dense{
        let intensities:Vec<f64>=self.beams.iter().map(|b|b.amplitude_at(r).norm_sqr()).collect();self.hamiltonian_with(&intensities)
    }
    /// H with every beam at the given intensity relative to its peak.
    pub fn hamiltonian_with(&self,intensities:&[f64])->Dense{
        let mut h=self.hamiltonian.clone();
        for(l,&s)in intensities.iter().enumerate(){
            if s!=1.{h.add_scaled(Complex::new(s-1.,0.),&self.outside[l]);}
            h.add_scaled(Complex::new(s,0.),&self.folds[l]);
        }
        h
    }

    /// The elimination for H (32 × 32, rad/us, in the frame): H's ground block and its energies, K,
    /// and every near beam's R_l and R2_l, column by column.
    pub fn eliminate(&self,hamiltonian:&Dense)->Result<Elimination,String>{
        let(ng,ne)=(self.ground_states.len(),self.excited_states.len());
        let ground=sub(hamiltonian,&self.ground_states,&self.ground_states);
        let energies:Vec<f64>=(0..ng).map(|b|ground[(b,b)].re).collect();
        let mut excited_hamiltonian=sub(hamiltonian,&self.excited_states,&self.excited_states);excited_hamiltonian.add_scaled(Complex::new(0.,-0.5),&self.decay);
        let(mut responses,mut derivatives)=(vec![Dense::zeros(ne,ng);self.beams.len()],vec![Dense::zeros(ne,ng);self.beams.len()]);
        let mut column=vec![Complex::ZERO;ne];
        for carrier in &self.carriers{for b in 0..ng{
            let lu=denominator(&excited_hamiltonian,-energies[b]-carrier.detuning)?;
            for &l in &carrier.beams{
                for i in 0..ne{column[i]=self.couplings[l][(i,b)];}lu.solve(&mut column);for i in 0..ne{responses[l][(i,b)]=column[i];}
                lu.solve(&mut column);for i in 0..ne{derivatives[l][(i,b)]=column[i];}
            }
        }}
        Ok(Elimination{energies,ground,excited_hamiltonian,responses,derivatives})
    }

    /// Whether carrier c couples ground state b at all (some beam's coupling from b).
    pub fn couples(&self,c:usize,b:usize)->bool{self.carriers[c].beams.iter().any(|&l|(0..self.couplings[l].rows).any(|e|self.couplings[l][(e,b)]!=Complex::ZERO))}

    /// The bounds over the trap (`Bounds`), from the eliminations at intensities {0, ¼, ½, ¾, 1} of
    /// every Gaussian beam (the plane waves at their own): H depends on the point through them, the
    /// near beams' amplitudes through |a_l| ≤ 1.
    fn bounds(&self)->Result<Bounds,String>{
        let(ng,nc)=(self.ground_states.len(),self.carriers.len());
        let gaussian:Vec<usize>=(0..self.beams.len()).filter(|&l|self.beams[l].waist_um.is_some()).collect();
        let levels=[0.,0.25,0.5,0.75,1.];let points=levels.len().pow(gaussian.len() as u32);
        let mut out=Bounds{raman:vec![vec![vec![vec![0.;ng];ng];nc];nc],photons:vec![vec![vec![[0.;3];ng];ng];nc],statics:vec![vec![0.;ng];ng],
            high:vec![vec![f64::NEG_INFINITY;ng];ng],light:0.,scattering:0.,poles:vec![f64::INFINITY;nc]};
        let mut identity=Dense::zeros(self.excited_states.len(),self.excited_states.len());for i in 0..identity.rows{identity[(i,i)]=Complex::ONE;}
        for point in 0..points{
            let mut intensities=vec![1.;self.beams.len()];
            for(f,&l)in gaussian.iter().enumerate(){intensities[l]=levels[(point/levels.len().pow(f as u32))%levels.len()];}
            let e=self.eliminate(&self.hamiltonian_with(&intensities))?;
            for a in 0..ng{for b in 0..ng{
                out.high[a][b]=out.high[a][b].max(e.energies[a]-e.energies[b]);
                if a!=b{out.statics[a][b]=out.statics[a][b].max(e.ground[(a,b)].abs());}
            }}
            // Per pair of beams the three products; per carrier pair their sums of moduli.
            let mut raman=vec![vec![vec![vec![0.;ng];ng];nc];nc];let(mut light,mut scattering)=(vec![0.;ng],vec![0.;ng]);
            for(c,carrier)in self.carriers.iter().enumerate(){for(d,emitted)in self.carriers.iter().enumerate(){
                for &l in &carrier.beams{for &k in &emitted.beams{
                    let(v,w)=(self.couplings[k].adjoint_product(&e.responses[l]),e.responses[k].adjoint_product(&self.couplings[l]));
                    let q=e.responses[k].adjoint_product(&self.decay.product(&e.responses[l]));
                    for a in 0..ng{for b in 0..ng{
                        raman[c][d][a][b]+=0.5*(v[(a,b)].abs()+w[(a,b)].abs()+q[(a,b)].abs());
                        if c==d{light[a]+=((v[(a,b)]+w[(a,b)])*0.5).abs();scattering[a]+=q[(a,b)].abs();}
                    }}
                }}
            }}
            for c in 0..nc{for d in 0..nc{for a in 0..ng{for b in 0..ng{out.raman[c][d][a][b]=out.raman[c][d][a][b].max(raman[c][d][a][b]);}}}}
            out.light=out.light.max(light.iter().copied().fold(0.,f64::max));out.scattering=out.scattering.max(scattering.iter().copied().fold(0.,f64::max));
            for(c,carrier)in self.carriers.iter().enumerate(){
                let jumps:Vec<[Dense;3]>=carrier.beams.iter().map(|&l|std::array::from_fn(|x|self.emission[x].product(&e.responses[l]))).collect();
                for b in 0..ng{for a in 0..ng{for x in 0..3{
                    let n:f64=jumps.iter().map(|j|j[x][(a,b)].abs()).sum();out.photons[c][b][a][x]=out.photons[c][b][a][x].max(n);
                }}}
                for b in 0..ng{
                    if !self.couples(c,b){continue;}
                    let lu=denominator(&e.excited_hamiltonian,-e.energies[b]-carrier.detuning)?;
                    out.poles[c]=out.poles[c].min(1./lu.solve_columns(&identity).spectral_norm());
                }
            }
        }
        Ok(out)
    }

    /// The couplings of the ground states classified over the trap (`classify`), and the clusters
    /// with their frames, kept couplings, van Vleck paths, jump classes and co-resonant channels.
    fn cluster_states(&mut self)->Result<(),String>{
        let(ng,nc)=(self.ground_states.len(),self.carriers.len());
        let b=self.bounds()?;self.pole=b.poles.iter().copied().fold(f64::INFINITY,f64::min);self.poles=b.poles.clone();
        // ‖L0‖ of the eliminated generator without the states' energies: its Hermitian part on both
        // sides, its decay and its jumps.
        let l0=2.*b.light+2.*b.scattering;self.generator_scale=l0;
        // The worst separation of a pair over the trap with a constant beat added: the separation's
        // range [low, high] of E_a − E_b, shifted.
        let separation=|a:usize,c:usize,beat:f64|->f64{
            let(lo,hi)=(beat-b.high[c][a],beat+b.high[a][c]);if lo<=0.&&hi>=0.{0.}else{lo.abs().min(hi.abs())}
        };
        let mut items:Vec<Item>=vec![];
        for a in 0..ng{for c in a+1..ng{if b.statics[a][c]>0.{
            items.push(Item{kind:Kind::Static{a,b:c},frequency:separation(a,c,0.),size:2.*b.statics[a][c].max(b.statics[c][a]),folded:false});}}}
        let channels:Vec<Channel>=(0..nc).flat_map(|c|(0..ng).map(move|s|Channel{carrier:c,state:s})).filter(|ch|self.couples(ch.carrier,ch.state)).collect();
        for(i,&x)in channels.iter().enumerate(){for &y in &channels[i+1..]{
            // Absorbing x, emitting into y's carrier and landing in y's state: Ω = (E_y + δ_y) − (E_x + δ_x).
            let beat=self.carriers[y.carrier].detuning-self.carriers[x.carrier].detuning;
            let size=2.*b.raman[x.carrier][y.carrier][y.state][x.state].max(b.raman[y.carrier][x.carrier][x.state][y.state]);
            if size>0.{items.push(Item{kind:Kind::Raman{first:x,second:y},frequency:separation(y.state,x.state,beat),size,folded:false});}
        }}
        for from in 0..ng{
            let photons:Vec<Photon>=(0..nc).flat_map(|c|(0..ng).map(move|to|Photon{carrier:c,from,to})).filter(|p|b.photons[p.carrier][from][p.to].iter().any(|&n|n>0.)).collect();
            for(i,&p)in photons.iter().enumerate(){for &q in &photons[i+1..]{
                if p.to==q.to{continue;}
                // Their frequencies (δ_p + E_from − E_p.to) − (δ_q + E_from − E_q.to).
                let beat=self.carriers[p.carrier].detuning-self.carriers[q.carrier].detuning;
                let size:f64=(0..3).map(|x|b.photons[p.carrier][from][p.to][x]*b.photons[q.carrier][from][q.to][x]).sum();
                if size>0.{items.push(Item{kind:Kind::Photons{first:p,second:q},frequency:separation(q.to,p.to,beat),size,folded:false});}
            }}
        }
        let(verdicts,cut)=classify(&items.iter().map(|i|(i.frequency,i.size)).collect::<Vec<_>>(),l0);
        for(item,v)in items.iter_mut().zip(verdicts){item.folded=v;}
        self.cut=cut;
        if let Some(item)=items.iter().find(|i|!i.folded&&matches!(i.kind,Kind::Static{..})&&{let Kind::Static{a,b}=i.kind else{unreachable!()};self.levels[a]!=self.levels[b]}){
            return Err(format!("光束在 F=1 与 F=2 之间的静态耦合不能忽略（混合 {:.1e}），3D 运动暂不支持",mixing(item.frequency,item.size,l0)));}
        // The clusters: the components of the graph of kept couplings.
        let mut union=Union::new(ng);
        for item in items.iter().filter(|i|!i.folded){match item.kind{
            Kind::Static{a,b}=>union.join(a,b),
            Kind::Raman{first,second}=>union.join(first.state,second.state),
            Kind::Photons{first,second}=>union.join(first.to,second.to),
        }}
        let sets=union.sets();
        let mut cluster=vec![0;ng];for(k,s)in sets.iter().enumerate(){for &x in s{cluster[x]=k;}}
        // Each cluster's frames: from its first state along a spanning tree of its kept couplings,
        // the strongest first, each nulling its coupling's beat.
        let focus=self.eliminate(&self.hamiltonian_with(&vec![1.;self.beams.len()]))?;
        let delta=|c:usize|self.carriers[c].detuning;
        let mut frames=vec![f64::NAN;ng];
        let mut kept:Vec<&Item>=items.iter().filter(|i|!i.folded).collect();kept.sort_by(|x,y|y.size.total_cmp(&x.size));
        // An edge (a, b) with θ_a − θ_b = offset.
        let edge=|item:&Item|->(usize,usize,f64){match item.kind{
            Kind::Static{a,b}=>(a,b,0.),
            Kind::Raman{first,second}=>(second.state,first.state,delta(first.carrier)-delta(second.carrier)),
            Kind::Photons{first,second}=>(first.to,second.to,delta(first.carrier)-delta(second.carrier)),
        }};
        for set in &sets{
            frames[set[0]]=focus.energies[set[0]];
            while let Some((a,b,offset))=kept.iter().map(|i|edge(i)).find(|&(a,b,_)|frames[a].is_nan()!=frames[b].is_nan()&&cluster[a]==cluster[set[0]]){
                if frames[a].is_nan(){frames[a]=frames[b]+offset;}else{frames[b]=frames[a]-offset;}
            }
        }
        let theta=|s:usize|frames[s];
        let folded_raman=|x:Channel,y:Channel|->Option<bool>{items.iter().find_map(|i|match i.kind{
            Kind::Raman{first,second}if(first==x&&second==y)||(first==y&&second==x)=>Some(i.folded),_=>None})};
        // Two channels are co-resonant: the same, or a kept pair.
        let coresonant=|x:Channel,y:Channel|x==y||folded_raman(x,y)==Some(false);
        let mut clusters:Vec<Cluster>=sets.iter().map(|states|Cluster{states:states.clone(),frames:states.iter().map(|&s|theta(s)).collect(),
            groups:vec![],statics:vec![],paths:vec![],static_paths:vec![],classes:vec![],coherent:vec![]}).collect();
        for(k,cl)in clusters.iter_mut().enumerate(){
            let inside:Vec<Channel>=channels.iter().copied().filter(|ch|cluster[ch.state]==k).collect();
            // The kept couplings per carrier pair: absorbing x = (c, b), emitting y = (c′, a).
            for c in 0..nc{for d in 0..nc{
                let mut entries=vec![];
                for(j,&s)in cl.states.iter().enumerate(){for(i,&t)in cl.states.iter().enumerate(){
                    let(x,y)=(Channel{carrier:c,state:s},Channel{carrier:d,state:t});
                    if !(self.couples(c,s)&&self.couples(d,t)){continue;}
                    if x==y||folded_raman(x,y)==Some(false){entries.push((i,j,theta(t)-theta(s)+delta(d)-delta(c)));}
                }}
                if !entries.is_empty(){cl.groups.push(Group{absorbed:c,emitted:d,entries});}
            }}
            let folded_static=|a:usize,b:usize|items.iter().any(|i|i.folded&&i.kind==Kind::Static{a:a.min(b),b:a.max(b)});
            for item in items.iter().filter(|i|!i.folded){if let Kind::Static{a,b}=item.kind{
                if cluster[a]==k{let(i,j)=(cl.position(a).unwrap(),cl.position(b).unwrap());cl.statics.push((i,j));cl.statics.push((j,i));}
            }}
            for i in 0..cl.states.len(){for j in 0..cl.states.len(){if theta(cl.states[i])==theta(cl.states[j]){
                for f in 0..ng{if folded_static(cl.states[i],f)&&folded_static(f,cl.states[j]){cl.static_paths.push((i,j,f));}}
            }}}
            // The folded pairs' paths: from a channel of the cluster through a far one, back into a
            // co-resonant channel of the cluster.
            for &x in &inside{for &y in &channels{
                if folded_raman(x,y)!=Some(true){continue;}
                for &z in &inside{if coresonant(x,z){
                    cl.paths.push(Path{from:x,via:y,back:z,residual:theta(z.state)-theta(x.state)+delta(z.carrier)-delta(x.carrier)});
                }}
            }}
            // The co-resonant channels (the readout's coherent sums), each with its residual.
            let mut union=Union::new(inside.len());
            for i in 0..inside.len(){for j in i+1..inside.len(){if coresonant(inside[i],inside[j]){union.join(i,j);}}}
            cl.coherent=union.sets().into_iter().map(|set|{let x0=inside[set[0]];
                set.iter().map(|&i|{let x=inside[i];(x,theta(x.state)+delta(x.carrier)-theta(x0.state)-delta(x0.carrier))}).collect()}).collect();
        }
        // The jump classes: the photons from a cluster, joined when they land in one cluster with
        // coherent amplitudes -- in one state from co-resonant channels, in two states as a kept pair
        // of photons from one state (or from two states, at a residual beat below the cut).
        for k in 0..clusters.len(){
            let photons:Vec<Photon>=(0..nc).flat_map(|c|clusters[k].states.iter().flat_map(move|&from|(0..ng).map(move|to|Photon{carrier:c,from,to})))
                .filter(|p|b.photons[p.carrier][p.from][p.to].iter().any(|&n|n>0.)).collect();
            let rate=|p:&Photon|theta(p.to)-theta(p.from)-delta(p.carrier);
            let mut union=Union::new(photons.len());
            for i in 0..photons.len(){for j in i+1..photons.len(){
                let(p,q)=(photons[i],photons[j]);
                if cluster[p.to]!=cluster[q.to]{continue;}
                let joined=if p.to==q.to{coresonant(Channel{carrier:p.carrier,state:p.from},Channel{carrier:q.carrier,state:q.from})}
                    else if p.from==q.from{items.iter().any(|it|!it.folded&&matches!(it.kind,Kind::Photons{first,second}if(first==p&&second==q)||(first==q&&second==p)))}
                    else{(rate(&p)-rate(&q)).abs()<cut};
                if joined{union.join(i,j);}
            }}
            clusters[k].classes=union.sets().into_iter().map(|set|{let p0=photons[set[0]];
                Class{target:cluster[p0.to],photons:set.iter().map(|&i|(photons[i],rate(&photons[i])-rate(&p0))).collect(),wave_number:self.carriers[p0.carrier].wave_number}}).collect();
        }
        self.items=items;self.clusters=clusters;self.cluster=cluster;
        Ok(())
    }

    /// Every beam's amplitude at a point r (um from the focus).
    pub fn amplitudes_at(&self,r:[f64;3])->Vec<Complex>{self.beams.iter().map(|b|b.amplitude_at(r)).collect()}

    /// The path energy of a cluster's channels of carrier c at an elimination: the centroid of its
    /// states' energies plus δ_c.
    pub fn path_energy(&self,e:&Elimination,cluster:usize,c:usize)->f64{
        let s=&self.clusters[cluster].states;s.iter().map(|&b|e.energies[b]).sum::<f64>()/s.len() as f64+self.carriers[c].detuning
    }

    /// A cluster's generator at a point r and a time t (the dense reference of model.rs, with the
    /// elimination of H(r)): H_eff, the jumps of every class, the readout.
    pub fn local_at(&self,e:&Elimination,k:usize,r:[f64;3],t:f64)->Result<Local,String>{self.local_with(e,k,r,t,&|_|true)}
    /// The same with the van Vleck paths `path` keeps (model.rs tabulates those through a carrier of
    /// one beam).
    pub fn local_with(&self,e:&Elimination,k:usize,r:[f64;3],t:f64,path:&dyn Fn(&Path)->bool)->Result<Local,String>{
        let cl=&self.clusters[k];let(n,ne)=(cl.states.len(),self.excited_states.len());let a=self.amplitudes_at(r);
        let v:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&self.couplings)).collect();
        let rc:Vec<Dense>=self.carriers.iter().map(|c|combine(&c.beams,&a,&e.responses)).collect();
        let mut h=Dense::zeros(n,n);
        for(i,&s)in cl.states.iter().enumerate(){h[(i,i)]=Complex::new(e.energies[s]-cl.frames[i],0.);}
        for &(i,j)in &cl.statics{h[(i,j)]+=Complex::cis((cl.frames[i]-cl.frames[j])*t)*e.ground[(cl.states[i],cl.states[j])];}
        // The folded static couplings' van Vleck shift, with the exact bare-energy denominators.
        for &(i,j,f)in &cl.static_paths{let(s,u)=(cl.states[i],cl.states[j]);
            h[(i,j)]+=e.ground[(s,f)]*e.ground[(f,u)]*(0.5*(1./(e.energies[s]-e.energies[f])+1./(e.energies[u]-e.energies[f])));}
        for g in &cl.groups{
            let(c,d)=(g.absorbed,g.emitted);
            let mut x=v[d].adjoint_product(&rc[c]).scaled(Complex::new(-0.5,0.));x.add_scaled(Complex::new(-0.5,0.),&rc[d].adjoint_product(&v[c]));
            x.add_scaled(Complex::new(0.,-0.5),&rc[d].adjoint_product(&self.decay.product(&rc[c])));
            for &(i,j,w)in &g.entries{h[(i,j)]+=Complex::cis(w*t)*x[(cl.states[i],cl.states[j])];}
        }
        let mut shift=Dense::zeros(n,n);
        for p in cl.paths.iter().filter(|p|path(p)){
            let z=self.path_energy(e,k,p.from.carrier);let lu=denominator(&e.excited_hamiltonian,-z)?;
            let(c,d,f)=(p.from.carrier,p.via.carrier,p.back.carrier);
            // U = −[V_d† G V_c]_{via, from}, W̃ = −[V_f† G V_d]_{back, via}.
            let mut gc:Vec<Complex>=(0..ne).map(|i|v[c][(i,p.from.state)]).collect();lu.solve(&mut gc);
            let mut gd:Vec<Complex>=(0..ne).map(|i|v[d][(i,p.via.state)]).collect();lu.solve(&mut gd);
            let u=-(0..ne).fold(Complex::ZERO,|s,i|s+v[d][(i,p.via.state)].conj()*gc[i]);
            let w=-(0..ne).fold(Complex::ZERO,|s,i|s+v[f][(i,p.back.state)].conj()*gd[i]);
            let(i,j)=(cl.position(p.back.state).unwrap(),cl.position(p.from.state).unwrap());
            shift[(i,j)]+=Complex::cis(p.residual*t)*w*u*(1./(z-e.energies[p.via.state]-self.carriers[d].detuning));
        }
        h.add_scaled(Complex::ONE,&shift);
        let jumps=cl.classes.iter().map(|class|{let target=&self.clusters[class.target];
            std::array::from_fn(|x|{let mut m=Dense::zeros(target.states.len(),n);
                for &(p,w)in &class.photons{let j=self.emission[x].product(&rc[p.carrier]);
                    m[(target.position(p.to).unwrap(),cl.position(p.from).unwrap())]+=Complex::cis(w*t)*j[(p.to,p.from)];}
                m})}).collect();
        let mut admixture=Dense::zeros(n,n);let mut excited_readout=vec![Dense::zeros(n,n);ne];
        for group in &cl.coherent{for &(x,wx)in group{for &(y,wy)in group{
            // The excited amplitude of channel x, e^{−i ε̃_x t} R_c x; Y and E_i their coherent sums.
            let(i,j)=(cl.position(y.state).unwrap(),cl.position(x.state).unwrap());let phase=Complex::cis((wy-wx)*t);
            for q in 0..ne{let z=rc[y.carrier][(q,y.state)].conj()*rc[x.carrier][(q,x.state)]*phase;admixture[(i,j)]+=z;excited_readout[q][(i,j)]+=z;}
        }}}
        Ok(Local{effective:h,shift,jumps,admixture,excited_readout})
    }

    /// The 32 populations (in the request's order) of a cluster's density ρ (n × n) at a point: the
    /// bare ground ones, of ρ − ½{Y, ρ}, and the excited ones, tr(E_i ρ).
    pub fn populations(&self,k:usize,local:&Local,rho:&Dense)->Vec<f64>{
        let y=&local.admixture;let(left,right)=(y.product(rho),rho.product(y));
        let mut out=vec![0.;STATES];
        for(i,&b)in self.clusters[k].states.iter().enumerate(){out[self.ground_states[b]]=rho[(i,i)].re-0.5*(left[(i,i)].re+right[(i,i)].re);}
        for(e,&state)in local.excited_readout.iter().zip(&self.excited_states){
            out[state]=(0..rho.rows).map(|j|(0..rho.rows).map(|k|(e[(j,k)]*rho[(k,j)]).re).sum::<f64>()).sum();
        }
        out
    }

    /// Beam l's response R_l to all orders in an excited-energy shift w (rad/us): column b solved
    /// with (K − E_b − δ_c + w)⁻¹. R_l − w R2_l is its first order.
    pub fn response(&self,elimination:&Elimination,beam:usize,shift:f64)->Result<Dense,String>{
        let carrier=self.carriers.iter().find(|c|c.beams.contains(&beam)).unwrap();let coupling=&self.couplings[beam];
        let mut out=Dense::zeros(coupling.rows,coupling.cols);let mut column=vec![Complex::ZERO;coupling.rows];
        for b in 0..coupling.cols{
            if (0..coupling.rows).all(|i|coupling[(i,b)]==Complex::ZERO){continue;}
            let lu=denominator(&elimination.excited_hamiltonian,shift-elimination.energies[b]-carrier.detuning)?;
            for i in 0..coupling.rows{column[i]=coupling[(i,b)];}lu.solve(&mut column);for i in 0..coupling.rows{out[(i,b)]=column[i];}
        }
        Ok(out)
    }

    /// The far components' fold, Σ m² of the folded ones, and the couplings', Σ m² of the folded
    /// pairs of channels and static couplings (the jumps' photons split by frequency are exact).
    pub fn fold_estimates(&self)->(f64,f64){
        let far=self.components.iter().filter(|c|c.folded&&c.size>0.).map(|c|mixing(c.nu,c.size,self.l0).powi(2)).sum();
        let near=self.items.iter().filter(|i|i.folded&&!matches!(i.kind,Kind::Photons{..})).map(|i|mixing(i.frequency,i.size,self.generator_scale).powi(2)).sum();
        (far,near)
    }
}

#[cfg(test)]
pub(crate) mod tests{
    use super::*;
    use crate::linear::uniform;
    use crate::Matrix;
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
    /// A request from a fixture: the matrices its beams lack (the counter-rotating coupling, the outside shift) are zero.
    pub(crate) fn request(v:&Value)->Request{
        let or_zero=|m:&Value|if m.is_null(){Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]}}else{matrix(m)};
        let beams=v["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
            direction:[d[0],d[1],d[2]],wave_number_per_um:float(&b["waveNumberPerUm"]),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:float(&b["intensityMWcm2"]),
            frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
            counter_coupling:or_zero(&b["counterCoupling"]),outside_shift:or_zero(&b["outsideShift"])}}).collect::<Vec<Beam>>();
        // The trap beam: the focused one (none in the free-space fixture).
        let trap_beam=beams.iter().position(|b|b.waist_um.is_some()).unwrap_or(0);
        let e=&v["emission"];
        let frames=if v["frameFrequenciesMHz"].is_null(){vec![0.;STATES]}else{floats(&v["frameFrequenciesMHz"])};
        Request{states:v["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&v["hamiltonian"]),
            frame_frequencies_mhz:frames,frequencies_mhz:floats(&v["frequenciesMHz"]),beams,
            emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],field_vector_g:[0.;3],mass_kg:float(&v["massKg"]),
            trap_beam,initial_density:vec![Complex::ZERO;STATES*STATES],temperature_uk:30.,times_us:vec![]}
    }
    fn mismatch(a:&Dense,b:&Dense)->(f64,f64){let mut d=a.clone();d.add_scaled(-Complex::ONE,b);(d.largest(),b.largest())}

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
        r.beams.extend([oblique,focused]);r.trap_beam=r.beams.len()-1;r
    }
    /// Points over a few wavelengths and the focused beam's waist (um).
    fn points(seed:u64)->Vec<[f64;3]>{let mut next=uniform(seed);(0..3).map(|_|[next(),2.*next(),4.*next()]).collect()}

    /// The trace is kept: at every point and time each cluster's jumps carry its whole decay,
    /// Σ_φ L_φ†L_φ = i(H_eff − H_eff†).
    #[test]
    fn the_jumps_carry_the_whole_decay(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();
            for(p,&x)in points(seed).iter().enumerate(){
                let e=g.eliminate(&g.hamiltonian_at(x)).unwrap();
                for k in 0..g.clusters.len(){
                    let local=g.local_at(&e,k,x,0.37*p as f64).unwrap();let n=g.clusters[k].states.len();
                    let mut sum=Dense::zeros(n,n);for j in &local.jumps{for m in j{sum.add_scaled(Complex::ONE,&m.adjoint_product(m));}}
                    // The folded pairs' paths change the decay at second order, outside the jumps (the
                    // budget's); the rest is the jumps' exactly.
                    let mut h=local.effective.clone();h.add_scaled(-Complex::ONE,&local.shift);
                    let decay=Dense::from_fn(n,n,|i,j|(h[(i,j)]-h[(j,i)].conj())*Complex::new(0.,1.));
                    let(error,scale)=mismatch(&sum,&decay);
                    assert!(error<=1e-12*scale,"seed {seed}, cluster {k}: {error:e} of {scale:e}");
                }
            }
        }
    }

    /// The readout keeps the trace: Σ_i E_i = Y, and the 32 populations of a density sum to one.
    #[test]
    fn the_populations_sum_to_one(){
        for seed in 1..=4{
            let r=perturbed(seed);let g=Ground::new(&r).unwrap();let mut next=uniform(seed);
            for x in points(seed){
                let e=g.eliminate(&g.hamiltonian_at(x)).unwrap();
                for k in 0..g.clusters.len(){
                    let n=g.clusters[k].states.len();let local=g.local_at(&e,k,x,1.3).unwrap();
                    let a=Dense::from_fn(n,n,|_,_|Complex::new(next(),next()));let mut rho=a.product(&a.adjoint());
                    let trace:f64=(0..n).map(|i|rho[(i,i)].re).sum();rho=rho.scaled(Complex::new(1./trace,0.));
                    let mut total=Dense::zeros(n,n);for readout in &local.excited_readout{total.add_scaled(Complex::ONE,readout);}
                    let(error,scale)=mismatch(&total,&local.admixture);assert!(error<=1e-13*scale.max(1e-300),"seed {seed}: {error:e} of {scale:e}");
                    let sum:f64=g.populations(k,&local,&rho).iter().sum();assert!((sum-1.).abs()<=1e-13,"seed {seed}: {sum}");
                }
            }
        }
    }

    /// R − w R2 is the exact response to first order: column b's remainder is w² (K − E_b − δ_c)⁻² R(w)_b,
    /// at most (w/pole)² of R(w)_b. At w = 0 the exact response is R itself.
    #[test]
    fn the_first_order_response_leaves_a_second_order_remainder(){
        let r=perturbed(1);let g=Ground::new(&r).unwrap();let e=g.eliminate(&g.hamiltonian_at([0.;3])).unwrap();let pole=g.pole;
        let norm=|m:&Dense,b:usize|(0..m.rows).map(|i|m[(i,b)].norm_sqr()).sum::<f64>().sqrt();
        for l in g.carriers.iter().flat_map(|c|c.beams.iter().copied()){
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

    /// The app example's tweezer alone (tests/fixtures/tweezer_focus.json): its co- and counter-rotating
    /// components both fold, so it is no carrier, and at the focus its fold and outside shift give the
    /// pipeline study's numbers: |2,+2⟩ 21.733 MHz deep, the D2 stretched resonance 24.04 MHz up, the
    /// radial trap frequency 100.55 kHz.
    #[test]
    fn the_tweezer_folds_into_the_study_s_trap(){
        let f:Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_focus.json")).unwrap();
        let r=request(&f["request"]);let g=Ground::new(&r).unwrap();
        assert!(g.components.iter().all(|c|c.folded)&&g.carriers.is_empty(),"{:?}",g.components);
        let(focus,far)=(g.hamiltonian_at([0.;3]),g.hamiltonian_at([0.,0.,1e6]));
        let state=|label:&str|r.states.iter().position(|s|s==label).unwrap();
        let shift=|i:usize|(focus[(i,i)].re-far[(i,i)].re)/TAU;
        let(depth,resonance)=(shift(state("ground-2-2")),shift(state("D2-3-3"))-shift(state("ground-2-2")));
        // The radial curvature of |2,+2⟩'s energy at the focus, by central differences across the tweezer.
        let h=1e-3;let energy=|x:f64|g.hamiltonian_at([x,0.,0.])[(state("ground-2-2"),state("ground-2-2"))].re;
        let curvature=(energy(h)+energy(-h)-2.*energy(0.))/(h*h);
        let trap=(curvature*1.0545718176461565e-34/r.mass_kg*1e6).sqrt()/TAU*1e3;
        assert!((depth+21.733).abs()<=1e-3&&(resonance-24.04).abs()<=1e-2&&(trap-100.55).abs()<=1e-2,"depth {depth} MHz, resonance {resonance} MHz, trap {trap} kHz");
    }

    /// Without a field the sublevels of a hyperfine level are degenerate and the light couples them
    /// -- the free-space PGC's lin⊥lin pairs along z by Δm = 0, ±2, so each level splits into its
    /// even and odd m (the jumps' photons keep the parity too) -- and the cooler and the repumper far
    /// from Raman resonance leave F = 1 and F = 2 apart: clusters of 1, 2, 2 and 3 states. The cooler
    /// moved to a Raman resonance with the repumper (both on F′ = 2: a Λ scheme) joins F = 1 and F = 2
    /// within each parity: two clusters of four, each one's frame making its Raman pairs' beat
    /// static, F = 2 offset from F = 1 by the Raman detuning. In the PGC the cooler is −f0 from the
    /// repumper (frequency axis 0), so f0 = (ν_2 − ν_1)/2π − δ_R puts the Raman detuning at δ_R.
    #[test]
    fn a_raman_resonant_lambda_scheme_joins_the_hyperfine_levels(){
        let pgc=request(&fixture()["request"]);let g=Ground::new(&pgc).unwrap();
        let sizes=|g:&Ground|{let mut s:Vec<usize>=g.clusters.iter().map(|c|c.states.len()).collect();s.sort();s};
        assert_eq!(sizes(&g),vec![1,2,2,3]);
        assert!(g.clusters.iter().all(|c|c.states.iter().all(|&b|g.levels[b]==g.levels[c.states[0]])),"F = 1 and F = 2 apart");
        let e=g.eliminate(&g.hamiltonian_at([0.;3])).unwrap();
        let level=|f:u8|(0..8).filter(|&b|g.levels[b]==f).map(|b|e.energies[b]).sum::<f64>()/if f==1{3.}else{5.};
        let hyperfine=(level(2)-level(1))/TAU;
        for detuning in [0.,0.1]{
            let mut lambda=pgc.clone();lambda.frequencies_mhz[0]=hyperfine-detuning;
            let g=Ground::new(&lambda).unwrap();assert_eq!(sizes(&g),vec![4,4],"Raman detuning {detuning} MHz");
            for cl in &g.clusters{
                let offset=|f:u8|{let s:Vec<f64>=cl.states.iter().zip(&cl.frames).filter(|x|g.levels[*x.0]==f).map(|(&b,&theta)|e.energies[b]-theta).collect();s.iter().sum::<f64>()/s.len() as f64};
                assert!(((offset(2)-offset(1))/TAU-detuning).abs()<=1e-6,"{} MHz against {detuning} MHz",(offset(2)-offset(1))/TAU);
            }
        }
    }
}
