// node expect.mjs [ids...] -- the expected values of the references (contract §6) at the output times of their scenes:
// expect/<id>.json, each a list of series (quantity, the output indices, the values, their standard uncertainty u and
// the acceptance target), the conditions the formula needs with their values, and the derivation in brief. Everything
// is computed from the scene's own request (refs/scenes/<id>.json: the model the solver receives) and the app's level
// chart (all-order Floquet light shifts, independent of the solver); a formula's literature constant is cited where used.
//
// Units: time µs, length µm, energies and rates rad/µs (an energy E as E/ħ), temperature µK, wave number rad/µm.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {levelShifts,laserMHz} from './lib/app.mjs';
import {KT_PER_UK,RAD_PER_MHZ,beamMeasure,branchRate,coupling,decayRates,gaussHermite,gaussLegendre,hbarOverMass,hermitianEigen,index,integrate,inverse3,kineticRatio,preparedTrap,rayleigh,recoilRate,recoilUK,roundoff,runningTrap,solve3,uKOfMHz,wignerTemperature} from './lib/phys.mjs';
import {ramanNumbers,standingPhase} from './lib/defs.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const read=f=>JSON.parse(fs.readFileSync(path.join(here,f),'utf8'));
const write=(id,e)=>{const f=path.join(here,'expect',`${id}.json`);fs.mkdirSync(path.dirname(f),{recursive:true});fs.writeFileSync(f,JSON.stringify(e,null,1));console.log(id,e.series?`${e.series.length} series`:`relation of ${e.members.map(m=>m.id).join(', ')}`);};
const grid=line=>{const {durationUs:T,samples:n}=line.options;return Array.from({length:n},(_,i)=>T*i/(n-1));};
const all=line=>grid(line).map((_,i)=>i);
const beamOf=(line,id)=>line.model.motion.beams.find(b=>b.id===id);
/** The user's targets (SPEC §0): populations and loss 0.05 absolute, T 0.10 relative. */
const POPULATION={absolute:0.05},TEMPERATURE={relative:0.10};
/** A quantity the user sets no target for -- a trap frequency, a mean position or velocity, one axis's T (the solver
 * holds each axis's T to nothing: its target is on the total, solve.rs "# T's target is on the total temperature") --:
 * judged against this reference's own u and the result's statistics alone (check.mjs, rule 0). */
const OWN=null;
/** The emission pattern of a σ± dipole about the quantization axis: ⟨cos²θ⟩ = ∫(1 + c²)c² / ∫(1 + c²) = 2/5 (c = cos θ
 * over [−1, 1]); a π dipole's ∝ (1 − c²): 1/5. */
const XI_SIGMA=2/5;
const expectation=(id,reference,title,line,fields)=>({id,reference,title,request:`scenes/${id}.json`,durationUs:line.options.durationUs,samples:line.options.samples,kind:'pass-fail',...fields});

// ---------------------------------------------------------------------------------------------------------------------
// The trap and the initial ensemble (R1, R2, R3, R9).

/** The prepared ensemble's exact second moments (SPEC §2.5, §1.7 "T(0) 为初始系综的精确矩"): the lowered isothermal at T
 * in the prepared trap (lib/phys.mjs preparedTrap: the solver's potential of the reference state, its depth and its
 * frequencies; classical kinetic temperature T_kin = T·I(5/2)/I(3/2), lib/phys.mjs kineticRatio; for a single Gaussian
 * beam of depth y₀ = U₀/k_BT exactly, by the beam's measure) times, per lab axis a of frequency ω_a, the thermal
 * oscillator's Wigner factor: T_a(0) = (T_kin/T)·T*_a, T*_a = (ħω_a/2k_B) coth(ħω_a/2k_BT). Their uncertainty u0: the
 * frequency's (the logarithmic slope of T⋆ in ω is at most 1); the ratio's quadrature (its change at two thirds of the nodes) and its sums'
 * rounding (lib/phys.mjs `roundoff` of the nodes' count); and, for a trap of several beams, the lowering's whole effect
 * |1 − T_kin/T|, which the one beam's measure does not state for it. */
function initialTemperatures(trap,T){
  const y=trap.depth/(KT_PER_UK*T),nodes=24,ratio=kineticRatio(beamMeasure(y,nodes)),open=Math.abs(kineticRatio(beamMeasure(y,2*nodes/3))/ratio-1)+roundoff(4*nodes**3)+(trap.beams.length>1?Math.abs(1-ratio):0);
  const T0=trap.omegas.map(w=>ratio*wignerTemperature(w,T));
  return {ratio,T0,u0:T0.map((t,a)=>t*(trap.u[a]/trap.omegas[a]+open))};
}
/** ω (rad/µs) → trapFrequencyKHz. */
const kHz=w=>w/(2*Math.PI)*1e3;
/** A trap's frequencies along the lab axes (rad/µs) with their uncertainty, as series. */
const trapSeries=(omegas,u)=>['x','y','z'].map((a,i)=>({quantity:'trapFrequency',axis:a,value:kHz(omegas[i]),u:kHz(u[i]),target:OWN}));
/** T's series of a reference that gives every axis (per output Ts[j][a] with us[j][a]): the total (T_x + T_y + T_z)/3,
 * the quantity the user's target is on -- its u the mean of the axes' (what they share adds fully) --, and each axis
 * against its own u. */
const temperatureSeries=(line,Ts,us)=>[{quantity:'T',total:true,regime:'any',indices:all(line),values:Ts.map(t=>(t[0]+t[1]+t[2])/3),u:us.map(u=>(u[0]+u[1]+u[2])/3),target:TEMPERATURE},
  ...['x','y','z'].map((a,i)=>({quantity:'T',axis:a,regime:'any',indices:all(line),values:Ts.map(t=>t[i]),u:us.map(u=>u[i]),target:OWN}))];

// ---------------------------------------------------------------------------------------------------------------------
// R1: conservative motion.

/** R1. With no near light the motion is conservative: H = p²/2m + V_prep(r), V_prep the reference state's shift by the
 * trap light alone (SPEC §1.3): the model's second-order form over the beam pairs of the trap's frequency, their
 * interference in it (lib/phys.mjs trapForm). (i) The trap frequencies are those of V_prep's Hessian at its minimum
 * along the lab axes, ω_a² = (ħ/m)∂_a²V (lib/phys.mjs preparedTrap: the model's own potential from the request, its
 * uncertainty that evaluation's rounding and the centre's). At the focus a pair (l′, l) gives the Hessian
 * Re Q_l′l [G_l′ + G_l − ΔκΔκᵀ], G_l = −(2/w₀²)(1 − nnᵀ) − nnᵀ/z_R² and Δκ = κ_l′n_l′ − κ_l n_l, κ = k − 1/z_R (the Gouy
 * phase): a Gaussian beam of depth U₀ along n has 2U₀/z_R² along n and 4U₀/w₀² across; two counter-propagating beams
 * in phase, of peak depth U_p, 2U_p[(k − 1/z_R)² + 1/z_R²] along and 4U_p/w₀² across; two crossed beams of orthogonal
 * linear polarizations add their Hessians, and their pair term is the reference state's vector shift, Im Q ∝ ⟨J_z⟩ --
 * which the bias field leaves nonzero in |F, 0⟩ --: 2 Im Q_l′l sin(Δκ·r) has no curvature at the focus but the slope
 * 2 Im Q Δκ, so the minimum lies off the focus by −H⁻¹ of it, where the fringe's third derivative adds
 * (2 Im Q)²(ΔκᵀH⁻¹Δκ) ΔκΔκᵀ to the Hessian H (Δω/ω = 1.4e-5 on x and y for R1_CROSSED). The level chart's all-order
 * shift of the same light is not this model: it differs by the fold's fourth order (the shift times Σ|X/δ|² over every
 * coupled level -- D1 and D2, both rotations --, and the cross terms with the states outside the model), the solver's
 * model error, which its ledger books (远失谐光的折叠) and the conditions list at the focus; the static couplings the
 * trap light makes between the ground states (ledger: 基态…的远光静态耦合) shift the level in fourth order too and are
 * in neither. (ii) The lowered isothermal is a function of the energy alone, so the flow keeps it (Jeans):
 * T_a(t) = T_a(0) for all t; the Wigner scaling (the ratio T* to T per axis) is exactly stationary in the harmonic part
 * and breaks it only through the anharmonic part, relatively ~k_BT/U, so u_a = T_a(0)·|T*_a/T − 1|·k_BT/U after t = 0
 * (at t = 0 the moment's own evaluation, `initialTemperatures`). (iii) No
 * atom is unbound (E < 0) and energy is conserved: L ≡ 0. (iv) The far light's scattering Γ_sc = Σ Γ_e|X_eg|²/δ²
 * (incoherent sum, an upper bound of the coherent one) bounds any population change by Γ_sc·t. */
async function r1(id){
  const line=read(`scenes/${id}.json`),scene=read(`scenes/${id}.scene.json`),m=line.model.motion,T=m.temperatureUK,init=scene.dynamics.initialStateId,times=grid(line);
  const trap=preparedTrap(line,init),U=trap.depth,omegas=trap.omegas,{ratio,T0,u0}=initialTemperatures(trap,T);
  const chart=(await levelShifts(scene,trap.beams.map(b=>b.id)))[init];
  const series=[];
  series.push(...trapSeries(omegas,trap.u));
  const uT=T0.map((T0a,i)=>T0a*Math.abs(wignerTemperature(omegas[i],T)/T-1)*(KT_PER_UK*T/U));
  series.push(...temperatureSeries(line,times.map(()=>T0),times.map((_,j)=>j?uT.map((u,i)=>Math.hypot(u,u0[i])):u0)));
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  const gamma=decayRates(line),E=line.model.motion.hamiltonian.real.filter((_,q)=>q%33===0),g0=index(line,init);
  let scatter=0;
  for(let e=8;e<32;e++){let amp=0;for(const b of m.beams){const x=coupling(line,b.id,e,g0),d=frameDetuning(line,b)-(E[e]-E[g0]);amp+=Math.hypot(x.re,x.im)/Math.abs(d);}scatter+=gamma[e]*amp*amp;}
  for(const s of line.model.states.filter(s=>s.startsWith('ground-')))
    series.push({quantity:'population',state:s,indices:all(line),values:times.map(()=>s===init?1:0),u:times.map(t=>scatter*t),target:POPULATION});
  write(id,expectation(id,'R1',`R1 保守阱（${trap.beams.map(b=>b.name).join(' + ')}，深 ${uKOfMHz(U/RAD_PER_MHZ).toFixed(1)} µK）`,line,{
    derivation:'Conservative motion in the far trap: ω_a² = (ħ/m)∂_a²V at the minimum of the model’s own potential, V = Σ Re[conj(a_l′) a_l Q_l′l] over the trap light’s beam pairs (second order, from the request; Gaussian beam 2U₀/z_R² along, 4U₀/w₀² across; standing wave 2U_p[(k−1/z_R)²+1/z_R²] with the Gouy phase; crossed orthogonal beams: the pair’s vector-shift fringe moves the minimum off the focus); the lowered isothermal times the Wigner factors is stationary up to its anharmonic part, so T_a(t) = T_a(0) = (T_kin/T)·T*_a; L ≡ 0; populations change only by far scattering.',
    conditions:[{what:'trap depth over k_BT (lowered isothermal deep)',value:U/(KT_PER_UK*T),limit:'≫ 1'},
      {what:'the model’s second-order depth at the focus over the level chart’s all-order shift, less 1 (the far fold’s fourth order: the solver’s model error, booked in its ledger)',value:trap.focus/(-chart.shiftMHz*RAD_PER_MHZ)-1,limit:'≪ 1'},
      {what:'far scattering per µs bounding population changes',value:scatter,limit:'Γ_sc·t ≪ 0.05'}],
    series,info:{depthUK:uKOfMHz(U/RAD_PER_MHZ),trapKHz:omegas.map(kHz),centreUm:trap.centre,chartDepthUK:uKOfMHz(-chart.shiftMHz),chartBoundUK:uKOfMHz(chart.boundMHz),kineticRatio:ratio,T0,wignerFactors:omegas.map(w=>wignerTemperature(w,T)/T)}}));
}
/** A beam's Δ_l (rad/µs) from the frame (phys.mjs frameDetuning by beam object). */
function frameDetuning(line,b){const f=line.model.frequencyCoordinates.frequenciesMHz;return RAD_PER_MHZ*b.frequencyIndex.reduce((s,n,k)=>s+n*f[k],0);}

// ---------------------------------------------------------------------------------------------------------------------
// The two-level atom |2,2⟩ ↔ |3′,3′⟩ (R2, R3, R9).

/** The closed σ+ cycle's data from the request and the level chart: Γ (1/µs) of |3′,3′⟩, Ω = 2|X| of each σ+ beam on
 * it, the detuning of each beam from the line at the trap's bottom δ_b (rad/µs: the beam's frequency less the
 * Floquet-shifted transition), the depth U₀ of |2,2⟩ and the differential shift's ratio κ = δ_LS(0)/U₀: both shifts ∝
 * |a(r)|², so at potential energy V above the bottom the line sits at δ_b + κV from the beam (κV a rate). */
async function cycle(id,beams){
  const line=read(`scenes/${id}.json`),scene=read(`scenes/${id}.scene.json`),g=index(line,'ground-2-2'),e=index(line,'D2-3-3');
  const sh=await levelShifts(scene,['tweezer']),U0=-sh['ground-2-2'].shiftMHz*RAD_PER_MHZ,dLS=(sh['D2-3-3'].shiftMHz-sh['ground-2-2'].shiftMHz)*RAD_PER_MHZ;
  const nu=sh['D2-3-3'].absoluteMHz-sh['ground-2-2'].absoluteMHz,gamma=decayRates(line)[e];
  const out={line,scene,gamma,U0,kappa:dLS/U0,dLS,beams:{}};
  for(const id2 of beams){const x=coupling(line,id2,e,g);out.beams[id2]={omega:2*Math.hypot(x.re,x.im),delta:RAD_PER_MHZ*((await laserMHz(scene,id2))-nu),k:beamOf(line,id2).waveNumberPerUm};}
  return out;
}
/** The steady-state photon rate of a two-level atom: R = (Γ/2) s/(1 + s + (2δ/Γ)²), s = 2Ω²/Γ² (exact for the
 * optical Bloch equations' steady state). */
const rate=(gamma,omega,delta)=>{const s=2*omega*omega/(gamma*gamma);return gamma/2*s/(1+s+(2*delta/gamma)**2);};
/** The running trap's factor (lib/phys.mjs `runningTrap`: ∂²V_run = ∂²V_prep × factor) by the formula of one excited
 * level at a time, on the level chart's all-order levels (independent of the request's second order): each beam's
 * shift |X^l_eg|² δ_le/(δ_le² + Γ_e²/4) with δ_le = δ_le,b + κ_e V at the potential energy V above the bottom,
 * κ_e = (shift_e − shift_g)/U₀, so factor = 1 + Σ κ_e |X|²(Γ_e²/4 − δ²)/(δ² + Γ_e²/4)². It is the running trap's where
 * the light reaches one excited level of its m alone (the closed cycle of R2, R3, R10: the two agree to 10⁻⁷ of the
 * frequency and better, the far fold's fourth order and the photon's recoil apart); where the trap light mixes the
 * levels the light reaches (R4's lin⊥lin pairs on F′ = 1, 2, 3 of one m) neither the levels nor the couplings are linear
 * in the trap's intensity, and this formula errs in the same order as it computes (3·10⁻⁵ of the frequency on
 * R4_LIN_1). Listed in the conditions as the difference to `runningTrap`; no series reads it. */
async function levelFactor(line,scene,g,beams){
  const states=line.model.states,gi=index(line,g),gamma=decayRates(line),sh=await levelShifts(scene,['tweezer']),U0=-sh[g].shiftMHz;
  let slope=0;
  for(const id of beams){const nu=await laserMHz(scene,id);
    for(let e=8;e<32;e++){const x=coupling(line,id,e,gi),X2=x.re*x.re+x.im*x.im;if(!X2)continue;
      const delta=RAD_PER_MHZ*(nu-(sh[states[e]].absoluteMHz-sh[g].absoluteMHz)),kappa=(sh[states[e]].shiftMHz-sh[g].shiftMHz)/U0,G2=gamma[e]**2/4;
      slope+=kappa*X2*(G2-delta*delta)/((delta*delta+G2)**2);}}
  return 1+slope;
}
/** The conditions every reference with a running trap lists of it. */
const trapConditions=async(line,scene,g,run)=>{const near=line.model.motion.beams.filter(b=>b.id!==line.model.motion.trapBeamId).map(b=>b.id);
  return [{what:'running trap: ∂²V_run over ∂²V_prep, less 1',value:run.factor-1},{what:'excited population the other lights make in second order (the running trap\'s next order is 3 times it, of the line above)',value:run.population,limit:'≪ 1'},
    {what:'the one-level formula on the level chart, its factor less the running trap\'s (zero but for the trap light\'s mixing of the excited levels the light reaches, the far fold\'s fourth order and the recoil)',value:(await levelFactor(line,scene,g,near))-run.factor}];};
/** Mandel's Q of the two-level atom's photon count, Q = 2R ∫₀^∞ [g²(τ) − 1] dτ, g²(τ) = ρ_ee(τ | ground)/ρ_ee^ss (the
 * quantum regression theorem): with the Bloch equations ẋ = Mx + b, ∫(x(τ) − x_ss) dτ = −M⁻¹(x(0) − x_ss). The
 * variance of the number of absorbed photons is (1 + Q) times their mean, the absorption part of the momentum
 * diffusion along the beam. */
function mandelQ(gamma,omega,delta){
  // Bloch vector (u, v, w) with ρ_ee = (1 + w)/2: u̇ = −Γ/2 u + δ v; v̇ = −δ u − Γ/2 v − Ω w; ẇ = Ω v − Γ(w + 1).
  const M=[[-gamma/2,delta,0],[-delta,-gamma/2,-omega],[0,omega,-gamma]],b=[0,0,-gamma];
  const xs=solve3(M,b.map(x=>-x)),x0=[0,0,-1],dx=x0.map((x,i)=>x-xs[i]),integral=solve3(M,dx).map(x=>-x);
  const ree=(1+xs[2])/2,R=gamma*ree,area=integral[2]/2;  // ∫(ρ_ee(τ|g) − ρ_ee^ss) dτ
  return 2*R*area/ree;
}

/** R2. The closed cycle |2,2⟩ ↔ |3′,3′⟩ driven far off resonance (δ_b = −100Γ, s = 200) by a σ+ plane wave along the
 * tweezer's axis z ∥ B, in a deep σ+ tweezer (both stretched states its eigenstates). Photon rate: R(δ) above at the
 * local detuning δ_b + κV, averaged over the cloud to second order: ⟨R⟩ = R(δ_b + κ⟨V⟩) + ½R''κ²Var V, ⟨V⟩ = Σ k_BT_a/2,
 * Var V = Σ (k_BT_a)²/2 (harmonic). Heating (each photon: absorption ħk along z, emission along the σ dipole pattern;
 * Var p_z grows by (ħk)²(1 + Q + ξ) per photon, Var p_x,y by (ħk)²(1 − ξ)/2): per axis energy E_a grows at
 * E_rec R(1 + Q + ξ) on z and E_rec R(1 − ξ)/2 on x, y, and k_BT_a = E_a/C_a with the separable quartic's heat capacity
 * C_a = 1 + (3/2)ε_a k_BT_a/U₀: the Gaussian beam's potential to fourth order is U₀(s + u − s²/2 − 2su − u²), s =
 * 2ρ²/w₀², u = z²/z_R² (the −2su: the beam widens along z), so ε = 1 axially and 1/2 radially, and to first order in
 * k_BT/U₀ an axis' potential energy is k_BT/2 + (3/4)ε(k_BT)²/U₀; the cross terms (−2U₀su, −U₀s_xs_y), not in C_a, are
 * of the same order and bound the rest by (k_BT_other/U₀)·ΔT_a. Radiation pressure: the mean force ħk⟨R⟩ moves the
 * axial equilibrium by z₀ = ħk⟨R⟩/(mω_dyn²), ω_dyn² = ω_z²(1 − κΩ²/4δ_b²): the probe's own light shift Ω²/4δ(r),
 * δ(r) = δ_b + κV, softens the spring the atoms feel: the running trap, whose frequencies are the reported
 * trapFrequencyKHz (SPEC §1.3; the prepared trap, the tweezer alone, gives only the initial ensemble's Wigner factors).
 * Switched on at t = 0 the cloud oscillates about it and dephases; each atom's orbit centre is F/(mω(E)²) with its own
 * anharmonic frequency ω(E) = ω(1 − (3/4)E_z/U₀ − (1/2)E_ρ/U₀) (action-angle, first order), so the time average of
 * ⟨z⟩ is z₀(1 + 2(3/4 + 1)k_BT/U₀) = z₀(1 + 3.5k_BT/U₀); the thermal-equilibrium response F·Var z/k_BT gives
 * z₀(1 + 5k_BT/U₀) -- their difference is the uncertainty. Observed as the mean of the position distributions' means
 * over their samples after t = 0 (the oscillation, aliased by the sampling, averages out over the run's ~20 samples:
 * its remainder bounded by the oscillation's amplitude over √(samples)). Populations: ρ_ee = ⟨R⟩/Γ in |3′,3′⟩.
 * The trap's force differs between the two states (the line moves by κV): its fluctuation, the force ħ∇δ P_e, adds
 * the diffusion 2D_aa = ħ²(∂_aδ)²·ee on every axis (`correlations`, R10; ee = R/(δ² + Γ²/4) at low saturation), and
 * with ⟨(∂_aδ)²⟩ = κ²mω_a²k_BT_a/ħ² the heating k_B dT_a/dt = κ²ω_a²·ee·k_BT_a/2: 10⁻⁵ of T here, at δ = −100Γ. */
async function r2(){
  const id='R2_RECOIL',c=await cycle(id,['probe']),{line,gamma,U0,kappa}=c,p=c.beams.probe,m=line.model.motion,mass=m.massKg,T=m.temperatureUK;
  const prep=preparedTrap(line,'ground-2-2'),omegas=prep.omegas,[radial]=omegas;
  const {ratio,T0,u0}=initialTemperatures(prep,T),Erec=recoilUK(p.k,mass),Q=mandelQ(gamma,p.omega,p.delta);
  const R=delta=>rate(gamma,p.omega,delta),h=1e-3*Math.abs(p.delta),R2=(R(p.delta+h)-2*R(p.delta)+R(p.delta-h))/(h*h);
  const meanR=Ts=>{const kT=Ts.map(t=>KT_PER_UK*t),V=kT.reduce((a,b)=>a+b,0)/2,varV=kT.reduce((a,b)=>a+b*b/2,0);return R(p.delta+kappa*V)+R2*kappa*kappa*varV/2;};
  const rates=[(1-XI_SIGMA)/2,(1-XI_SIGMA)/2,1+Q+XI_SIGMA],eps=[0.5,0.5,1];
  const stateForce=Ts=>{const ee=correlations(gamma,p.omega,p.delta+kappa*KT_PER_UK*Ts.reduce((a,b)=>a+b,0)/2).ee;return omegas.map(w=>kappa*kappa*w*w*ee/2);};
  const derivative=Ts=>{const r=meanR(Ts),st=stateForce(Ts);return Ts.map((t,a)=>(Erec*rates[a]*r+st[a]*t)/(1+1.5*eps[a]*KT_PER_UK*t/U0));};
  const times=grid(line),Ts=[T0.slice()],Rs=[meanR(T0)];let state=T0.slice();
  for(let i=1;i<times.length;i++){const n=20,dt=(times[i]-times[i-1])/n;for(let j=0;j<n;j++){const k1=derivative(state),k2=derivative(state.map((x,a)=>x+dt/2*k1[a])),k3=derivative(state.map((x,a)=>x+dt/2*k2[a])),k4=derivative(state.map((x,a)=>x+dt*k3[a]));state=state.map((x,a)=>x+dt/6*(k1[a]+2*k2[a]+2*k3[a]+k4[a]));}Ts.push(state.slice());Rs.push(meanR(state));}
  // The radiation-pressure offset, the spring softened by the probe's light shift and by the cloud's anharmonicity.
  const rbar=Rs.reduce((a,b)=>a+b,0)/Rs.length,r=hbarOverMass(mass),run=runningTrap(line,'ground-2-2'),dyn=run.omegas[2];
  const z0=r*p.k*rbar/(dyn*dyn),kTbar=KT_PER_UK*Ts.reduce((a,s)=>a+(s[0]+s[1]+s[2])/3,0)/Ts.length/U0;
  // The oscillation's remainder in the average over the 20 evenly spaced distribution samples after t = 0 (contract §2.2:
  // at most 21 times with 0 and the end): at most max |mean cos ωt_i| over the anharmonic frequency window
  // [ω_dyn(1 − 5k_BT/U₀), ω_dyn], times z₀ (no dephasing credited).
  const tS=Array.from({length:20},(_,i)=>times.at(-1)*(i+1)/20);let rest=0;
  for(let w=dyn*(1-5*kTbar);w<=dyn;w+=dyn*1e-5)rest=Math.max(rest,Math.abs(tS.reduce((s,t)=>s+Math.cos(w*t),0))/tS.length);
  const zbar=z0*(1+3.5*kTbar),zu=Math.hypot(z0*1.5*kTbar,z0*Math.abs(Rs.at(-1)-Rs[0])/(2*rbar),z0*rest);
  const series=[];
  // The reported trap is the running one (SPEC §1.3; lib/phys.mjs runningTrap): the prepared one softened by the
  // probe's second-order shift at the local detuning (ω² × (1 − κΩ²/4δ²) for |δ| ≫ Γ). The dressed two-level shift
  // (−δ + sgn δ·√(δ² + Ω²))/2 softens by (3/4)(Ω/δ)² of that less: the next order, in its u.
  series.push(...trapSeries(run.omegas,run.u));
  // u(T): the cross-axis quartic terms (k_BT_other/U₀)·ΔT_a; the centre-of-mass energy ½mω²z₀² that dephasing turns
  // into heat (z, after t = 0); the Wigner scaling's non-stationarity (R1). Mandel's Q is in the rate.
  const push=0.5*mass*(dyn*1e6)**2*(z0*1e-6)**2/1.380649e-23*1e6;
  series.push(...temperatureSeries(line,Ts,Ts.map((s,j)=>[0,1,2].map(i=>{const other=Math.max(...s.filter((_,b)=>b!==i)),dT=s[i]-T0[i];
    return Math.hypot(u0[i],KT_PER_UK*other/U0*dT,i===2&&j>0?push:0,j?T0[i]*Math.abs(wignerTemperature(omegas[i],T)/T-1)*KT_PER_UK*T/U0:0);}))));
  // The offset: no user's target is on a position (it is judged against zu and the distributions' own spread).
  series.push({quantity:'meanPositionAverage',axis:'z',value:zbar,u:zu,target:OWN});
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  const ree=Rs.map(x=>x/gamma);
  series.push({quantity:'population',state:'ground-2-2',indices:all(line),values:ree.map((x,i)=>i?1-x:1),u:ree.map((x,i)=>i?0.1*x:0),target:POPULATION});
  series.push({quantity:'population',state:'D2-3-3',indices:all(line),values:ree.map((x,i)=>i?x:0),u:ree.map((x,i)=>i?0.1*x:0),target:POPULATION});
  const s=2*p.omega**2/gamma**2;
  write(id,expectation(id,'R2','R2 二能级辐射压与反冲加热',line,{
    derivation:'Two-level photon rate R = (Γ/2)s/(1+s+(2δ/Γ)²) at the local detuning (cloud average to 2nd order in κV); per photon Var p grows by (ħk)²(1+Q+ξ) along the beam and (ħk)²(1−ξ)/2 across (ξ = 2/5, σ dipole): k_B dT_z/dt = (7/5)E_rec R, k_B dT_x,y/dt = (3/10)E_rec R, with the separable quartic heat capacity, and the trap force’s fluctuation between the two states, κ²ω_a²·ee·k_BT_a/2 (R10); radiation pressure offset z₀ = ħkR/(mω²) (the spring softened by the probe’s light shift), time-averaged over the dephasing oscillation with the anharmonic orbit-centre shift.',
    conditions:[{what:'|δ_b|/Γ',value:Math.abs(p.delta)/gamma,limit:'≫ 1'},{what:'s',value:s},{what:'κ·⟨V⟩/|δ_b| (cloud spread of the detuning)',value:kappa*KT_PER_UK*1.5*T/Math.abs(p.delta),limit:'≪ 1'},
      {what:'k_BT/U₀ at the end (quartic corrections)',value:KT_PER_UK*Math.max(...Ts.at(-1))/U0,limit:'≪ 1'},{what:'Mandel Q',value:Q},{what:'k_BT ≥ 4ħω_radial at t = 0',value:KT_PER_UK*T/radial,limit:'≥ 4'},
      {what:'state-force diffusion: its relative heating over the run, stiffest axis',value:Math.max(...stateForce(T0))*times.at(-1),limit:'≪ 1'},
      {what:'one beam\'s Doppler damping time m/β (s), β = ħk²|dR/dδ|',value:1/(hbarOverMass(mass)*p.k**2*rbar*8*Math.abs(p.delta)/(gamma*gamma)/(1+s+(2*p.delta/gamma)**2))*1e-6,limit:'≫ the run'},
      ...await trapConditions(line,c.scene,'ground-2-2',run)],
    series,info:{depthUK:uKOfMHz(U0/RAD_PER_MHZ),trapKHz:omegas.map(kHz),kineticRatio:ratio,T0,s,deltaOverGamma:p.delta/gamma,photonRatePerUs:rbar,recoilUK:Erec,mandelQ:Q,z0Um:z0,meanZUm:zbar,runningTrapFactor:run.factor,kappa}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R3: 1D Doppler cooling in a trap that shifts the line.

/** R3. The closed cycle driven by a σ+σ+ pair along the tweezer's axis z ∥ B (s = 0.02 each, x = 2δ/Γ): the low-
 * saturation Doppler theory of two beams (averaged over the standing wave, which the axial motion crosses many times:
 * amplitude ≫ λ/2), friction β = −4ħk² s x/(1 + x²)² and photon rate R = Γs/(1 + x²), momentum diffusion along z
 * (ħk)²(1 + Q + ξ)R/2 and across (ħk)²(1 − ξ)R/4 (σ dipole, ξ = 2/5): free of the trap, k_BT = ħΓ(1 + ξ)(1 + x²)/(16|δ|/Γ)
 * (contract §6 R3; 0.35ħΓ at δ = −Γ/2). No tweezer wavelength in the app's model (785–1600 nm, π or σ±) makes
 * |2,2⟩ ↔ |3′,3′⟩ magic: the tweezer moves the line by κV (V the potential energy above the bottom, κ = δ_LS(0)/U₀,
 * exactly linear since both shifts ∝ |a|²), and at the Doppler temperature κV spans ~Γ/2 whatever the depth -- so
 * (contract: "按局域 δ(r) 对热分布平均") the friction and diffusion are averaged over the atom's motion. The trap
 * frequencies (kHz) ≫ the cooling rate (1/ms), so each atom's per-axis energies E_a are slow and obey phase-averaged
 * Itô equations (harmonic separable phases: V_a = E_a sin²φ_a, kinetic E_a cos²φ_a, φ uniform):
 *   dE_z = ⟨−2(β/m)E_z cos²φ_z + D_z/m⟩dt + √(⟨4E_z cos²φ_z D_z/m⟩dt) dW,  dE_x,y = ⟨D_x/m⟩dt + √(⟨4E_x cos²φ_x D_x/m⟩dt) dW,
 * every rate at the local detuning δ_b + κΣ_a V_a; ⟨·⟩ by Gauss–Chebyshev in each phase. A Monte Carlo of these SDEs
 * (fixed seed) from the prepared ensemble gives ⟨E_a⟩(t), and T_a = E_a/C_a with the separable quartic heat capacity
 * (R2). Uncertainty: the Monte Carlo's standard error; the O(s) saturation (2s/(1 + x²) of the cooled T and of the
 * heating); the quartic cross terms (k_BT_other/U₀)|ΔT_a| and the harmonic phase relation's anharmonic error
 * (k_BT_a/U₀)|ΔT_a|/2. Mandel's Q is in the diffusion. The run's T on x and y only grows (no cooling across).
 * The trap's force differs between the two states (the line moves by κV): its fluctuation, the force ħ∇δ P_e, adds
 * the diffusion 2D_aa = ħ²(∂_aδ)²·ee on every axis (`correlations`, R10; at this low saturation ee = R/(δ² + Γ²/4), R
 * the photon rate): with ħ²(∂_aδ)² = 2κ²mω_a²V_a the energy of axis a drifts by κ²ω_a²·ee·V_a more, V_a = E_a sin²φ_a,
 * and its noise takes the same D; about 2 % of T_x,y by the end of the run, nothing on the soft axis z. */
async function r3(){
  const id='R3_DOPPLER',c=await cycle(id,['cool-plus','cool-minus']),{line,gamma,U0,kappa}=c,b=c.beams['cool-plus'],m=line.model.motion,mass=m.massKg,T=m.temperatureUK;
  const prep=preparedTrap(line,'ground-2-2'),omegas=prep.omegas,axial=omegas[2],eps=[0.5,0.5,1];
  const {ratio,T0,u0}=initialTemperatures(prep,T),Erec=recoilUK(b.k,mass),wrec=hbarOverMass(mass)*b.k*b.k/2,s=2*b.omega**2/gamma**2;
  const Q=mandelQ(gamma,Math.SQRT2*b.omega,b.delta),times=grid(line);
  const capacity=(t,a)=>1+1.5*eps[a]*KT_PER_UK*t/U0,energy=(t,a)=>t*(1+0.75*eps[a]*KT_PER_UK*t/U0),temperature=(E,a)=>{const c2=0.75*eps[a]*KT_PER_UK/U0;return c2>0?(-1+Math.sqrt(1+4*c2*E))/(2*c2):E;};
  // Gauss–Chebyshev nodes of sin²φ (arcsine law on [0, 1]), equal weights.
  const n=5,nodes=Array.from({length:n},(_,j)=>(1-Math.cos(Math.PI*(2*j+1)/(2*n)))/2);
  const coefficients=(E)=>{
    let Az=0,Bz=0,Ax=0,Bx=0,Ay=0,By=0;
    for(const sx of nodes)for(const sy of nodes)for(const sz of nodes){
      const V=E[0]*sx+E[1]*sy+E[2]*sz,x=2*(b.delta+kappa*KT_PER_UK*V)/gamma,R=gamma*s/(1+x*x),beta=-8*wrec*s*x/((1+x*x)**2);
      const state=kappa*kappa*4*R/(gamma*gamma*(1+x*x)),recoil=Erec*(1-XI_SIGMA)/2*R;
      const Dz=Erec*(1+Q+XI_SIGMA)*R+state*omegas[2]**2*E[2]*sz,Dx=recoil+state*omegas[0]**2*E[0]*sx,Dy=recoil+state*omegas[1]**2*E[1]*sy;
      Az+=-2*beta*E[2]*(1-sz)+Dz;Bz+=4*E[2]*(1-sz)*Dz;Ax+=Dx;Bx+=4*E[0]*(1-sx)*Dx;Ay+=Dy;By+=4*E[1]*(1-sy)*Dy;}
    const w=1/n**3;return [[Ax*w,Bx*w],[Ay*w,By*w],[Az*w,Bz*w]];};
  let seed=0x2545F491;const next=()=>{seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  const normal=()=>{const u=1-next(),v=next();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
  const N=8000,dt=4,atoms=Array.from({length:N},()=>[0,1,2].map(a=>-energy(T0[a],a)*Math.log(1-next())));
  const sums=times.map(()=>[[0,0],[0,0],[0,0]]),record=i=>{for(const E of atoms)for(let a=0;a<3;a++){sums[i][a][0]+=E[a];sums[i][a][1]+=E[a]*E[a];}};
  record(0);let t=0;
  for(let i=1;i<times.length;i++){
    while(t<times[i]-1e-9){const h=Math.min(dt,times[i]-t);
      for(const E of atoms){const co=coefficients(E);for(let a=0;a<3;a++){E[a]+=co[a][0]*h+Math.sqrt(Math.max(0,co[a][1])*h)*normal();if(E[a]<0)E[a]=-E[a];}}
      t+=h;}
    record(i);}
  // At t = 0 the result reports the prepared ensemble's exact moment (SPEC §1.7): T0, not the sample's mean.
  const Ts=sums.map((row,j)=>row.map(([s1,s2],a)=>{if(!j)return [T0[a],0];const mean=s1/N,se=Math.sqrt(Math.max(0,s2/N-mean*mean)/N);return [temperature(mean,a),se/capacity(temperature(mean,a),a)];}));
  const xb=2*b.delta/gamma,meanX=2*(b.delta+kappa*KT_PER_UK*1.5*Ts.at(-1).reduce((s2,q)=>s2+q[0],0)/3)/gamma;
  const doppler=X=>(1+XI_SIGMA)*(1+X*X)/(8*Math.abs(X))*gamma/KT_PER_UK;  // ħΓ(1+ξ)(1+x²)/(16|δ|/Γ) in µK with |δ|/Γ = |x|/2
  // The reported trap (SPEC §1.3; lib/phys.mjs runningTrap): the prepared one softened or stiffened by the cooling
  // pair's phase-averaged shift.
  const run=runningTrap(line,'ground-2-2'),series=trapSeries(run.omegas,run.u);
  series.push(...temperatureSeries(line,Ts.map(row=>row.map(q=>q[0])),Ts.map((row,j)=>[0,1,2].map(i=>{const other=Math.max(...row.filter((_,k)=>k!==i).map(q=>q[0])),dT=Math.abs(row[i][0]-T0[i]);
    return Math.hypot(u0[i],row[i][1],j?2*s/(1+meanX*meanX)*(i===2?row[i][0]:dT):0,KT_PER_UK*other/U0*dT,KT_PER_UK*row[i][0]/U0*dT/2);}))));
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  // The excited population of the pair, averaged over the wave: R/Γ = s/(1 + x²), at the end's mean detuning; u: its
  // change over the run (from the bottom's detuning, where the cold cloud starts) and the saturation's next order.
  const excited=X=>s/(1+X*X);
  series.push({quantity:'population',state:'ground-2-2',indices:all(line),values:times.map((_,i)=>i?1-excited(meanX):1),u:times.map((_,i)=>i?Math.abs(excited(meanX)-excited(xb))+2*s*excited(meanX):0),target:POPULATION});
  write(id,expectation(id,'R3','R3 一维多普勒冷却（σ⁺σ⁺ 驻波，非魔术光镊：按局域失谐平均）',line,{
    derivation:'Low-saturation two-beam Doppler friction and diffusion at the local detuning δ_b + κV (the tweezer moves the cycling line with the potential energy; no magic wavelength exists in the model), with the trap force\'s fluctuation between the two states (2D_aa = 2κ²mω_a²V_a·ee, R10), averaged over each atom\'s fast trap motion: phase-averaged Itô equations for the per-axis energies, solved by Monte Carlo from the prepared ensemble; free-space limit k_BT = ħΓ(1+ξ)(1+x²)/(16|δ|/Γ).',
    conditions:[{what:'s per beam',value:s,limit:'≤ 0.02'},{what:'x = 2δ/Γ at the bottom',value:xb},{what:'x at the end\'s mean potential energy',value:meanX},{what:'κ = δ_LS(0)/U₀',value:kappa},
      {what:'k v_rms/Γ at the Doppler T',value:b.k*Math.sqrt(hbarOverMass(mass)*KT_PER_UK*doppler(meanX))/gamma,limit:'≪ 1'},{what:'depth over k_BT at the end',value:U0/(KT_PER_UK*Math.max(...Ts.at(-1).map(q=>q[0]))),limit:'≥ 20'},
      {what:'axial ω over the cooling rate β/m',value:axial/(8*wrec*s*Math.abs(xb)/((1+xb*xb)**2)),limit:'≫ 1'},{what:'Mandel Q',value:Q},
      {what:'state-force diffusion: its relative heating of a stiff axis over the run, at the end\'s mean detuning',value:kappa*kappa*omegas[0]**2*4*s/(gamma*(1+meanX*meanX)**2)/2*times.at(-1),limit:'≪ 1'},
      ...await trapConditions(line,c.scene,'ground-2-2',run)],
    series,info:{depthUK:uKOfMHz(U0/RAD_PER_MHZ),trapKHz:omegas.map(kHz),kineticRatio:ratio,T0,s,kappa,freeDopplerUK:doppler(meanX),freeDopplerAtBottomUK:doppler(xb),runningTrapFactor:run.factor,monteCarlo:{N,dt,phaseNodes:n}}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R5: optical pumping, rate equations.

/** R5. Low-saturation rate equations on the 8 ground states (contract §6 R5): beam l excites g → e at
 * r = Γ_e|X^l_eg|²/(δ² + Γ_e²/4) (the two-level ρ_ee at s ≪ 1 times Γ_e; δ from the level chart at the trap's bottom),
 * e decays to g′ with the branching Γ_{e→g′}/Γ_e of the request's own emission operators (polarization-summed CG²):
 * W_{g→g′} = Σ_{l,e} r b_{e→g′}, dp/dt = Wᵀp − (W·1)∘p, excited ρ_e = Σ r p_g/Γ_e (quasi-static). |2,2⟩ is the only
 * state no σ+ beam excites: the dark state. The pump on D1 and the repump on D2 share no excited state, so no
 * two-photon (CPT) coherence forms; the off-resonant excitations of other F′ are in the sum. Uncertainty: the O(s)
 * saturation and the cloud's spread of the detuning (the tweezer's differential shift over k_BT, the Doppler shift)
 * scale the rates by 1 ± ε_R, ε_R = s_max + (2δ_spread/Γ)² + (2k v_rms/Γ)², propagated as the change of p(t) for rates
 * ×(1 + ε_R); the tweezer's own scattering bounds the rest. */
async function r5(){
  const id='R5_PUMP',line=read(`scenes/${id}.json`),scene=read(`scenes/${id}.scene.json`),states=line.model.states,gamma=decayRates(line);
  const sh=await levelShifts(scene,['tweezer']),times=grid(line),beams=['pump','repump'],G=states.map((s,i)=>i).filter(i=>states[i].startsWith('ground-'));
  const nu={};for(const b of beams)nu[b]=await laserMHz(scene,b);
  const r=[];let smax=0,spread=0;
  const T=line.model.motion.temperatureUK,kT=KT_PER_UK*T,U0=-sh['ground-2-2'].shiftMHz*RAD_PER_MHZ;
  for(const b of beams)for(const g of G)for(let e=8;e<32;e++){
    const x=coupling(line,b,e,g),X2=x.re*x.re+x.im*x.im;if(!X2)continue;
    const delta=RAD_PER_MHZ*(nu[b]-(sh[states[e]].absoluteMHz-sh[states[g]].absoluteMHz)),rr=gamma[e]*X2/(delta*delta+gamma[e]**2/4);
    r.push({g,e,rate:rr});
    if(Math.abs(delta)<gamma[e]){smax=Math.max(smax,8*X2/gamma[e]**2);
      const kappa=((sh[states[e]].shiftMHz-sh[states[g]].shiftMHz)*RAD_PER_MHZ)/U0;spread=Math.max(spread,(2*kappa*1.5*kT/gamma[e])**2);}
  }
  const k=Math.max(...beams.map(b=>beamOf(line,b).waveNumberPerUm)),vrms=Math.sqrt(hbarOverMass(line.model.motion.massKg)*kT),doppler=(2*k*vrms/gamma[16])**2;
  const epsR=smax+spread+doppler;
  const p0=G.map(g=>line.model.initialReal[g*33]);
  const W=G.map(()=>G.map(()=>0)),Re=Array(32).fill(0).map(()=>G.map(()=>0));
  for(const {g,e,rate} of r){for(const g2 of G)W[G.indexOf(g)][G.indexOf(g2)]+=rate*branchRate(line,e,g2)/gamma[e];Re[e][G.indexOf(g)]+=rate/gamma[e];}
  const solve=scale=>{
    const out=[p0.slice()];let p=p0.slice();
    const f=q=>G.map((_,j)=>scale*(G.reduce((s,_,i)=>s+W[i][j]*q[i],0)-W[j].reduce((s,w)=>s+w,0)*q[j]));
    for(let i=1;i<times.length;i++){const n=50,dt=(times[i]-times[i-1])/n;for(let s=0;s<n;s++){const k1=f(p),k2=f(p.map((x,j)=>x+dt/2*k1[j])),k3=f(p.map((x,j)=>x+dt/2*k2[j])),k4=f(p.map((x,j)=>x+dt*k3[j]));p=p.map((x,j)=>x+dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j]));}out.push(p.slice());}
    return out.map((q,i)=>{const ex=Array(32).fill(0);if(i)for(let e=8;e<32;e++)ex[e]=scale*Re[e].reduce((s,x,j)=>s+x*q[j],0);const tot=ex.reduce((a,b)=>a+b,0);
      return states.map((_,s)=>s<8?q[G.indexOf(s)]*(1-tot):ex[s]);});
  };
  const base=solve(1),varied=solve(1+epsR);
  let twScatter=0;{const g0=index(line,'ground-2-2'),E=line.model.motion.hamiltonian.real.filter((_,q)=>q%33===0),tw=beamOf(line,'tweezer');
    for(let e=8;e<32;e++){const x=coupling(line,'tweezer',e,g0),d=frameDetuning(line,tw)-(E[e]-E[g0]);twScatter+=gamma[e]*(x.re*x.re+x.im*x.im)/(d*d);}}
  const series=[];
  states.forEach((s,i)=>{if(Math.max(...base.map(q=>q[i]))<1e-4)return;
    series.push({quantity:'population',state:s,indices:all(line),values:base.map(q=>q[i]),u:base.map((q,t)=>Math.abs(varied[t][i]-q[i])+twScatter*times[t]),target:POPULATION});});
  for(const [label,F] of [['F=1',1],['F=2',2]]){const ids=states.filter(s=>s.startsWith(`ground-${F}-`)),ix=ids.map(s=>states.indexOf(s));
    series.push({quantity:'group',label,states:ids,indices:all(line),values:base.map(q=>ix.reduce((a,j)=>a+q[j],0)),u:base.map((q,t)=>Math.abs(ix.reduce((a,j)=>a+varied[t][j]-q[j],0))+twScatter*times[t]),target:POPULATION});}
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  write(id,expectation(id,'R5','R5 光泵到拉伸暗态（速率方程）',line,{
    derivation:'Low-saturation rate equations: excitation r = Γ|X|²/(δ²+Γ²/4) per beam and transition (level-chart detunings at the trap bottom), decay branching from the emission operators; |2,2⟩ is the unique σ+ dark state; pump (D1) and repump (D2) share no excited state, so no CPT. The exact static evolution of the same model is R0_R5_PUMP.',
    conditions:[{what:'largest resonant saturation s',value:smax,limit:'≲ 0.02–0.04 (O(s) error)'},{what:'detuning spread over the cloud (2κ·1.5k_BT/Γ)²',value:spread},{what:'Doppler (2kv/Γ)²',value:doppler},{what:'tweezer scattering per µs',value:twScatter}],
    series,info:{rateScale:epsR}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R6: the Raman Λ.

/** R6. The Λ |1,0⟩ ↔ |2,0⟩ in second order (lib/defs.mjs ramanNumbers): Ω_R = 2Σ_e X^A_{e1}X^B*_{e2}/Δ_e, the
 * two-photon detuning δ₂ (compensated to ~0 at v = 0), the scattering Γ̄ = mean over the two states of
 * Σ_{l,e} Γ_e|X^l_eg|²/δ². An atom that scatters loses the coherence, so P₂ = e^{−Γ̄t}P₂^coh + (1 − e^{−Γ̄t})/2 with
 * u = (1 − e^{−Γ̄t})/2. With Δk = (k_A n_A − k_B n_B)·z -- 2k for counter-propagating beams, the two wave numbers'
 * difference (10⁻⁴ rad/µm) for co-propagating ones, one formula for both: no value of Δk is told from another, and
 * at Δk → 0 it is P₂ = (Ω_R²/Ω_eff²) sin²(Ω_eff t/2), Ω_eff = √(Ω_R² + δ₂²) --
 * the transition |1, p⟩ → |2, p + ħΔk⟩ has δ₂(t) = −Δk v_z(t) − ħΔk²/2m;
 * along each orbit of the harmonic axial trap, v_z(t) = v₀ cos ωt − ωz₀ sin ωt, the two-level amplitudes are integrated
 * (RK4) and averaged over the prepared ensemble's (z₀, v₀) (Gaussian, velocity variance k_BT(0)/m and position variance k_BT* over mω²; Gauss–Hermite), and
 * the mean momentum ħΔk·P̄₂ moves the centre of mass: ⟨v_z⟩(t) = (ħΔk/m)∫₀^t cos ω(t − s) dP̄₂(s). */
async function r6(id){
  const line=read(`scenes/${id}.json`),scene=read(`scenes/${id}.scene.json`),m=line.model.motion,mass=m.massKg,T=m.temperatureUK,times=grid(line),states=line.model.states;
  const {omegaR,twoPhotonMHz}=await ramanNumbers(line,scene),d2=twoPhotonMHz*RAD_PER_MHZ,gamma=decayRates(line);
  const E=m.hamiltonian.real.filter((_,q)=>q%33===0),g1=index(line,'ground-1-0'),g2=index(line,'ground-2-0');
  let scatter=0;for(const g of [g1,g2])for(const b of ['ramanA','ramanB'])for(let e=8;e<32;e++){const x=coupling(line,b,e,g),d=frameDetuning(line,beamOf(line,b))-(E[e]-E[g]);scatter+=gamma[e]*(x.re*x.re+x.im*x.im)/(d*d+gamma[e]**2/4)/2;}
  const A=beamOf(line,'ramanA'),B=beamOf(line,'ramanB'),dk=A.waveNumberPerUm*A.direction[2]-B.waveNumberPerUm*B.direction[2],r=hbarOverMass(mass),recoil=r*dk*dk/2;
  const prep=preparedTrap(line,'ground-1-0'),axial=prep.omegas[2],{T0}=initialTemperatures(prep,T);
  const sv=Math.sqrt(r*KT_PER_UK*T0[2]),sz=Math.sqrt(r*KT_PER_UK*wignerTemperature(axial,T))/axial;
  // i ċ = Hc, H = ½[[−δ, Ω_R],[Ω_R, δ]] in the two-photon rotating frame (P₂ is even in δ), δ(t) given; P₂ on the grid.
  const evolve=(delta)=>{
    let a=[1,0,0,0];const P=[0];const f=(t,y)=>{const d=delta(t),h=omegaR/2,[r1,i1,r2,i2]=y;return [-d/2*i1+h*i2,d/2*r1-h*r2,h*i1+d/2*i2,-h*r1-d/2*r2];};
    for(let i=1;i<times.length;i++){const n=8,dt=(times[i]-times[i-1])/n;let t=times[i-1];for(let s=0;s<n;s++){const k1=f(t,a),k2=f(t+dt/2,a.map((x,j)=>x+dt/2*k1[j])),k3=f(t+dt/2,a.map((x,j)=>x+dt/2*k2[j])),k4=f(t+dt,a.map((x,j)=>x+dt*k3[j]));a=a.map((x,j)=>x+dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j]));t+=dt;}P.push(a[2]**2+a[3]**2);}
    return P;};
  const nodes=gaussHermite(24),P2=times.map(()=>0);
  for(const [x,wx] of nodes)for(const [y,wy] of nodes){const z0=sz*x,v0=sv*y,P=evolve(t=>d2-dk*(v0*Math.cos(axial*t)-axial*z0*Math.sin(axial*t))-recoil);P.forEach((p,i)=>P2[i]+=wx*wy*p);}
  const f=times.map(t=>1-Math.exp(-scatter*t));
  const p2=P2.map((p,i)=>(1-f[i])*p+f[i]/2),p1=P2.map((p,i)=>(1-f[i])*(1-p)+f[i]/2);
  const series=[{quantity:'population',state:'ground-2-0',indices:all(line),values:p2,u:f.map(x=>x/2),target:POPULATION},
    {quantity:'population',state:'ground-1-0',indices:all(line),values:p1,u:f.map(x=>x/2),target:POPULATION}];
  // The centre of mass (m/s): no user's target is on a velocity (judged against u and the thermal spread's statistics).
  const vel=times.map((t,i)=>{let s=0;for(let j=1;j<=i;j++)s+=Math.cos(axial*(t-(times[j]+times[j-1])/2))*(P2[j]*(1-f[j])-P2[j-1]*(1-f[j-1]));return r*dk*s;});
  series.push({quantity:'meanVelocity',axis:'z',indices:all(line),values:vel,u:vel.map((_,i)=>Math.abs(r*dk)*f[i]),target:OWN});
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  write(id,expectation(id,'R6',`R6 拉曼 Λ（Δk = ${(dk/A.waveNumberPerUm).toFixed(2)} k）`,line,{
    derivation:'Two-photon Rabi oscillation with the Doppler detuning −Δk·v(t) along harmonic axial orbits and the recoil shift ħΔk²/2m, averaged over the prepared ensemble; ⟨v_z⟩ = (ħΔk/m)∫cos ω(t−s) dP̄₂; scattering destroys the coherence (u). One formula for every Δk: co-propagating beams are its Δk → 0, P₂ = (Ω_R²/Ω_eff²)sin²(Ω_eff t/2) (the exact static evolution of that one is R0_R6_RAMAN_CO).',
    conditions:[{what:'Ω_R/2π (MHz)',value:omegaR/(2*Math.PI)},{what:'residual two-photon detuning (MHz)',value:twoPhotonMHz},{what:'scattering per µs (mean of the two states)',value:scatter},
      {what:'Δk (rad/µm)',value:dk},{what:'axial ω·t at the end (orbit integration needed when not ≪ 1)',value:axial*times.at(-1)},{what:'Δk·v_rms/Ω_R',value:Math.abs(dk)*sv/omegaR}],
    series,info:{omegaR,twoPhotonMHz,scatter,recoilShift:recoil,axialKHz:kHz(axial),T0z:T0[2]}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R9 (b): the push out of the trap.

/** R9 (b). A σ+ beam along the tweezer's axis z pushes the atom out of the trap's region: the contract's 1D equation
 *   m z̈ = ħk R(δ − kż) − ∂_zU_g + ρ_ee ħ∂_zδ,
 * the radiation pressure at the two-level atom's steady rate R = Γρ_ee, ρ_ee = (s/2)/(1 + s + (2δ/Γ)²), the ground
 * state's trap force on the axis, U_g = −U₀g, g = 1/(1 + z²/z_R²), and the force the excited atom feels more (ħ∇δ P_e
 * in the mean); δ = δ_free − δ_LS(z) − kż with δ_free the detuning from the free line and δ_LS(z) = δ_LS(0)g (both
 * shifts ∝ |a|²). The photon kicks add the momentum diffusion d Var p/dt = (ħk)²(1 + Q + ξ)R (absorption with
 * Mandel's Q, σ emission): a Langevin equation, solved by a Monte Carlo from the prepared axial distribution
 * (Gaussian: T*_z and T(0)_z of the deep trap), N paths with a fixed seed.
 *
 * The region's edge is not at one z: the atom is outside where the tweezer's intensity g e^{−2ρ²g/w₀²} is below 1/20,
 * that is where ρ² > ρ_c²(z) = (w₀²/2g) ln(20g) -- on the axis at z = √19 z_R, earlier off it by kρ²/(2√19·…) ≈ 1 µm
 * per µm² --, and the emitted photons spread the atom across the axis as they push it along. So each path carries
 * the second moments of one radial axis (the two are alike), B = ⟨x²⟩, C = ⟨xv⟩, A = ⟨v²⟩:
 *   dB/dt = 2C,  dC/dt = A − ω_ρ²g²(1 − κρ_ee)B,  dA/dt = −2ω_ρ²g²(1 − κρ_ee)C + (ħk/m)²(1 − ξ)R/2,
 * the radial spring of the beam at z (4U₀g²/w₀², less the excited atom's share) and the emission's recoil -- exact for
 * ρ ≪ w(z), adiabatic near the focus and ballistic once the spring has gone --, from the prepared radial ensemble. With
 * x, y Gaussian of variance B, ρ² is exponential of mean 2B: the path is outside with the probability e^{−ρ_c²/2B},
 * and L(t) is the paths' mean of its largest value up to t (the kinetic energy there is far above the trap: E_s > 0).
 * The same B gives the cloud's average of what the axis' force leaves out: off the axis the trap's axial force is
 * (1 − sg)e^{−sg} of the axis' and the line's shift e^{−sg} of it, s = 2ρ²/w₀², ⟨sg⟩ = 4Bg/w₀²: the force
 * (1 − 2⟨sg⟩) and the shift (1 − ⟨sg⟩) of the axis', in the equation.
 *
 * The scene makes this its physics: the push is far off resonance (δ = +40Γ at the bottom, +44Γ outside) and
 * saturates only to ρ_ee ≤ 0.05, so that the trap keeps its radial force (it scales as 1 − κρ_ee, κ = δ_LS(0)/U₀),
 * the rate changes little where the atom sits off the axis or moves (the line's shift across the trap and the Doppler
 * shift at the exit are a few Γ at most, against 40Γ), and the plane wave's own light shift is uniform; the radiation
 * pressure is about five times the trap's largest axial force, (3√3/8)U₀/z_R at z_R/√3.
 *
 * u: the paths' standard error; the radial moments' next order -- they hold for ρ ≪ w(z), and the beam's radial
 * expansion goes on in sg, whose cloud average at the edge is ⟨sg⟩ = 4Bg/w₀²: 2⟨sg⟩ of what the moments move L by
 * (the run again without them) --; and the internal state's lag, the rate relaxing at Γ (the run again, common
 * random numbers). */
async function r9(){
  const id='R9_PUSH',c=await cycle(id,['push']),{line,gamma,U0,dLS,kappa}=c,p=c.beams.push,m=line.model.motion,mass=m.massKg,T=m.temperatureUK,r=hbarOverMass(mass);
  const tw=beamOf(line,'tweezer'),zR=rayleigh(tw.waveNumberPerUm,tw.waistUm),w2=tw.waistUm**2,prep=preparedTrap(line,'ground-2-2'),[radial,,axial]=prep.omegas,{T0}=initialTemperatures(prep,T);
  const deltaFree=p.delta+dLS,Q=mandelQ(gamma,p.omega,p.delta),edge=Math.sqrt(19)*zR,times=grid(line),s=2*p.omega**2/gamma**2;
  const excited=delta=>s/2/(1+s+(2*delta/gamma)**2);
  const sz=Math.sqrt(r*KT_PER_UK*wignerTemperature(axial,T))/axial,sk=Math.sqrt(KT_PER_UK*T0[2]/r);
  // A seeded generator (mulberry32; each run restarts it: common random numbers between the runs) and normal deviates
  // (Box–Muller).
  let seed=0;const next=()=>{seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  const normal=()=>{const u=1-next(),v=next();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
  /** The paths: per path its probability of being outside at every output (the largest so far), and its exit on the
   * axis (time, velocity µm/µs, photons, B there). `across` false leaves the radial moments out of the edge and of
   * the force (the axis' equation alone); `lag` lets the rate relax at Γ. */
  const run=({N,across=true,lag=false})=>{
    const outside=times.map(()=>0),square=times.map(()=>0),exits=[],dt=0.02,every=Math.round((times[1]-times[0])/dt);seed=0x9E3779B9;
    for(let n=0;n<N;n++){
      let z=sz*normal(),kap=sk*normal(),R=0,photons=0,B=r*KT_PER_UK*wignerTemperature(radial,T)/(radial*radial),C=0,A=r*KT_PER_UK*T0[0],most=0,exit=null;
      for(let i=1;i<times.length;i++){
        for(let j=0;j<every&&!exit;j++){
          const g=1/(1+z*z/(zR*zR)),off=across?4*B*g/w2:0,delta=deltaFree-dLS*g*(1-off)-p.k*r*kap,target=gamma*excited(delta);
          R=lag?R+(target-R)*Math.min(1,dt*gamma):target;
          const force=p.k*R-(1-2*off)*2*U0*z*g*g/(zR*zR)+R/gamma*2*dLS*z*g*g/(zR*zR);
          // The radial moments through dt: the oscillator's own rotation at the local spring (exact for a constant one,
          // whatever its size: sin(ωdt)/ω → dt as the spring goes), then the emission's kicks.
          const spring=radial*radial*g*g*(1-kappa*R/gamma),w=Math.sqrt(spring),co=Math.cos(w*dt),si=w*dt>1e-8?Math.sin(w*dt)/w:dt;
          [A,B,C]=[spring*spring*si*si*B-2*spring*si*co*C+co*co*A+dt*(r*p.k)**2*(1-XI_SIGMA)/2*R,co*co*B+2*co*si*C+si*si*A,-spring*si*co*B+(co*co-spring*si*si)*C+co*si*A];
          kap+=force*dt+Math.sqrt(p.k*p.k*(1+Q+XI_SIGMA)*R*dt)*normal();z+=r*kap*dt;photons+=R*dt;
          if(z>edge)exit={t:times[i-1]+(j+1)*dt,v:r*kap,photons,B};
          else if(across)most=Math.max(most,Math.exp(-w2/(2*g)*Math.log(20*g)/(2*B)));
        }
        const P=exit?1:most;outside[i]+=P;square[i]+=P*P;
      }
      exits.push(exit);
    }
    return {L:outside.map(x=>x/N),se:outside.map((x,i)=>Math.sqrt(Math.max(0,square[i]/N-(x/N)**2)/N)),exits};};
  const N=4000,full=run({N}),axis=run({N,across:false}),lagged=run({N,lag:true}),left=axis.exits.filter(x=>x),mean=f=>left.reduce((a,x)=>a+f(x),0)/left.length;
  const next2=2*4*mean(x=>x.B)/(20*w2);
  const series=[{quantity:'loss',indices:all(line),values:full.L,u:full.L.map((x,i)=>Math.hypot(full.se[i],next2*(x-axis.L[i]),lagged.L[i]-x)),target:POPULATION},
    {quantity:'population',state:'ground-2-2',indices:[0],values:[1],u:[0],target:POPULATION}];
  const crossing=(L,q)=>{const i=L.findIndex(x=>x>=q);return i<1?NaN:times[i-1]+(times[i]-times[i-1])*(q-L[i-1])/(L[i]-L[i-1]);},Erec=recoilUK(p.k,mass),photons=mean(x=>x.photons),spread=mean(x=>x.B);
  write(id,expectation(id,'R9','R9 推出：离开 1/20 区域的时刻给出 L(t) 的台阶',line,{
    derivation:'Langevin equation along the axis: radiation pressure at the two-level steady rate with the Doppler and position-dependent detuning, the ground trap force and the excited state’s extra force, momentum diffusion (ħk)²(1+Q+ξ)R; each path carries the second moments of the radial motion (the beam’s radial spring at z, the emission’s recoil), which place the 1/20 contour for the atom off the axis: outside with probability exp(−ρ_c²(z)/2⟨x²⟩), ρ_c² = (w₀²/2g)ln(20g). Monte Carlo from the prepared ensemble. The push is 40Γ off resonance and excites at most 5 %: the trap keeps its radial force, and the rate does not depend on where the atom sits.',
    conditions:[{what:'s = 2Ω²/Γ²',value:s},{what:'δ at the bottom / Γ',value:p.delta/gamma},{what:'δ outside / Γ',value:deltaFree/gamma},
      {what:'ρ_ee at the bottom',value:excited(p.delta),limit:'≤ 0.05'},{what:'ρ_ee outside, at rest',value:excited(deltaFree),limit:'≤ 0.05'},
      {what:'radial trap force kept, 1 − κρ_ee, least',value:1-kappa*Math.max(excited(p.delta),excited(deltaFree)),limit:'≥ 0.89'},
      {what:'radiation pressure at the bottom over the trap’s largest axial force (3√3/8)U₀/z_R',value:p.k*gamma*excited(p.delta)/(3*Math.sqrt(3)/8*U0/zR),limit:'≈ 5'},
      {what:'photons scattered until the axis’ edge (mean)',value:photons},{what:'velocity at the axis’ edge (m/s, mean)',value:mean(x=>x.v)},
      {what:'Doppler shift there over Γ',value:p.k*mean(x=>x.v)/gamma},{what:'Doppler shift there over |δ| outside',value:p.k*mean(x=>x.v)/Math.abs(deltaFree),limit:'≪ 1'},
      {what:'time at which L = 0.05, 0.5, 0.95 (µs)',value:[0.05,0.5,0.95].map(q=>crossing(full.L,q))},{what:'the same for the axis’ equation alone (µs)',value:[0.05,0.5,0.95].map(q=>crossing(axis.L,q))},
      {what:'radial rms per axis at the axis’ edge (µm), and ⟨sg⟩ = 2⟨ρ²⟩/w(z)² there',value:[Math.sqrt(spread),4*spread/(20*w2)],limit:'≪ 1'},
      {what:'radial recoil heating over the push (µK per axis), of the depth',value:[(1-XI_SIGMA)/2*Erec*photons,KT_PER_UK*(1-XI_SIGMA)/2*Erec*photons/U0],limit:'≪ 1'},
      {what:'line shift across the trap over |δ| at the bottom',value:dLS/Math.abs(p.delta),limit:'≪ 1'},{what:'atoms that do not reach the axis’ edge within the run',value:1-left.length/N},
      {what:'region edge on the axis (µm)',value:edge}],
    series,info:{mandelQ:Q,zR,edge,sz,sk,N,kappa,depthUK:U0/KT_PER_UK,T0,axisAloneLoss:axis.L}}));
}

/** R9 (c). Heating-driven escape with no mean force: a red and a blue σ+ beam of one intensity counter-propagate along
 * the tweezer's axis, 100Γ below and above the line. Their radiation pressures, their light shifts and their Doppler
 * forces cancel -- each to the order κU₀/ħ|δ|, the line's shift across the trap over the detuning -- and what is left
 * is the photons' momentum diffusion, the same everywhere: D_zz = (1 + Q + ξ)(ħk)²R/2 along the beams and
 * (1 − ξ)(ħk)²R/4 across, R the two beams' rate (two frequencies: their excitations add). A classical particle
 * diffusing in momentum in the Gaussian beam's potential V = U₀(1 − |a|²): its energy ε above the bottom drifts by
 * A = Σ_a D_aa/m = E_rec(2 + Q)R, exactly, and spreads by B = Σ_a 2(D_aa/m)⟨p_a²/m⟩_ε, which depends on how the
 * kinetic energy is shared among the axes. Two closures bound it: the ergodic one, ⟨p_a²/m⟩ = (2/3)⟨KE⟩ on every
 * axis, B = (4/3)⟨KE⟩A; and the axial one, all the kinetic energy along the beams, B = 4⟨KE⟩(1 + Q + ξ)E_rec R --
 * the motion in a Gaussian beam is nearly separable, 7/10 of the heating goes to z, and the radial action is
 * adiabatically invariant (ω_ρ/ω_z = πw₀√2/λ), so the truth lies between. ⟨·⟩_ε are microcanonical shell averages
 * over the exact beam (weight (ε − V)^{1/2}, kinetic (ε − V)^{3/2}; lib/phys.mjs' measure). The Fokker–Planck
 * equation ∂P/∂t = −∂(AP) + ½∂²(BP) on [0, U₀] (reflecting at the bottom, absorbing at the far level, where the atom
 * is unbound) from the prepared lowered isothermal P ∝ g(ε)(e^{−ε/k_BT} − e^{−U₀/k_BT}) gives the escape; an unbound
 * atom is counted lost when it leaves the 1/20 region: within µs across the beam, up to τ_max along it (the run from
 * the centre to √19 z_R at E = 0) -- L(t) is averaged over a delay uniform in [0, τ_max].
 * The expectation is the mean of the two closures. u: half their difference; half the delay's spread; the forces
 * that do not cancel -- off the centre the red beam is nearer the line and the blue farther by κV, so that the pair
 * pushes with ħkR·2κV/ħ|δ|, a tilt kRz_R·2κ/|δ| of the depth over a Rayleigh range: that fraction of t·dL/dt --; and
 * the boundary itself: an atom above the depth is lost only at the region's edge, up to τ_max later, and until then
 * its energy goes on spreading by √(Bτ_max/2), some of it back below the depth, which the absorbing boundary does not
 * allow: L is late by up to the time the drift takes over that layer, √(Bτ_max/2)/A.
 *
 * Characterization, not a judge: that last term is a third to a half of a millisecond here, 0.1 to 0.2 of L where L
 * rises -- several times the loss's target --, and no closure of the energy equation removes it (the loss is a
 * passage in space, not in energy). The equation is kept so that the solver's loss stands beside an estimate. */
async function r9heat(){
  const id='R9_HEAT',c=await cycle(id,['red','blue']),{line,gamma,U0,kappa}=c,beams=Object.values(c.beams),m=line.model.motion,mass=m.massKg,T=m.temperatureUK;
  const tw=beamOf(line,'tweezer'),zR=rayleigh(tw.waveNumberPerUm,tw.waistUm),k=beams[0].k,Erec=recoilUK(k,mass),Q=beams.map(b=>mandelQ(gamma,b.omega,b.delta)),times=grid(line);
  const U=U0/KT_PER_UK;  // depth in µK
  // Shell integrals over the beam: ∫d³r F(y), y = |a|², = (πw₀²z_R/2)∫ sec⁴θ G(cos²θ) dθ, G(Y) = ∫₀^Y F(y) dy/y.
  const shell=(eps,f)=>{const ymin=1-eps/U;const G=Y=>{if(Y<=ymin)return 0;let s=0;const a=Math.max(ymin,0),L=Math.log(Y/Math.max(a,1e-12));
      for(const [v,w] of gaussLegendre(24,0,Math.min(L,40)))s+=w*f(Y*Math.exp(-v));return s;};
    return integrate(th=>{const cs=Math.cos(th);return G(cs*cs)/(cs**4);},-Math.PI/2*0.9999,Math.PI/2*0.9999,{panels:16,n:16});};
  // Per beam the rate at the potential energy V (µK); the drift's and the axial spread's sums over the beams.
  const R=(b,V)=>rate(gamma,b.omega,b.delta+kappa*V*KT_PER_UK),sum=(V,f)=>beams.reduce((t,b,i)=>t+f(Q[i])*R(b,V),0);
  const n=160,de=U/n,eps=Array.from({length:n},(_,i)=>(i+0.5)*de),root=(e,y)=>Math.sqrt(Math.max(0,e-U*(1-y)));
  const stats=eps.map(e=>{const g=shell(e,y=>root(e,y)),KE=shell(e,y=>root(e,y)**3)/g;
    return {g,KE,drift:Erec*shell(e,y=>root(e,y)*sum(U*(1-y),q=>2+q))/g,axial:Erec*shell(e,y=>root(e,y)*sum(U*(1-y),q=>1+q+XI_SIGMA))/g};});
  const A=stats.map(s=>s.drift),closures={ergodic:stats.map(s=>4/3*s.KE*s.drift),axial:stats.map(s=>4*s.KE*s.axial)};
  const start=(()=>{const P=stats.map((s,i)=>s.g*(Math.exp(-eps[i]/T)-Math.exp(-U/T))),norm=P.reduce((a,b)=>a+b,0)*de;return P.map(x=>x/norm);})();
  // Explicit flux-form steps: J_{i+1/2} = (A P)_{mid} − ½ ∂(B P)/∂ε; reflecting at 0, absorbing at U.
  const escape=B=>{let P=start.slice(),t=0;const dtMax=0.4*de*de/Math.max(...B),L=[0];
    for(let i=1;i<times.length;i++){while(t<times[i]-1e-9){const h=Math.min(dtMax,times[i]-t),J=new Array(n+1).fill(0);
        for(let j=1;j<n;j++)J[j]=0.5*(A[j-1]*P[j-1]+A[j]*P[j])-0.5*(B[j]*P[j]-B[j-1]*P[j-1])/de;
        J[n]=A[n-1]*P[n-1]+0.5*B[n-1]*P[n-1]/(de/2);  // P = 0 at ε = U: the outgoing flux
        P=P.map((x,j)=>x-h*(J[j+1]-J[j])/de);t+=h;}
      L.push(1-P.reduce((a,b)=>a+b,0)*de);}
    return L;};
  // The axial run from the centre to √19 z_R at E = 0: ∫dz/v, v = √(2U₀/(m(1 + z²/z_R²))) (µm/µs = m/s).
  const r=hbarOverMass(mass),edge=Math.sqrt(19)*zR,tauMax=integrate(z=>1/Math.sqrt(2*U0*r/(1+z*z/(zR*zR))),0,edge,{panels:8,n:16});
  const at=(L,t)=>{if(t<=0)return 0;const x=t/(times[1]-times[0]),j=Math.floor(x);return j>=times.length-1?L.at(-1):L[j]+(L[j+1]-L[j])*(x-j);};
  const delayed=L=>times.map(t=>{let s=0;const K=40;for(let q=0;q<K;q++)s+=at(L,t-tauMax*(q+0.5)/K);return s/K;});
  const raw=Object.fromEntries(Object.entries(closures).map(([name,B])=>[name,escape(B)])),ergodic=delayed(raw.ergodic),axial=delayed(raw.axial),value=ergodic.map((x,i)=>(x+axial[i])/2);
  const bottom=sum(0,()=>1),tilt=k*bottom*zR*2*kappa/Math.abs(beams[0].delta),mid=L=>times.map(t=>(at(L,t)-at(L,t-tauMax))/2);
  const other=mid(raw.axial),spread=mid(raw.ergodic).map((x,i)=>(x+other[i])/2),pushed=times.map((t,i)=>Math.abs((at(value,t*(1+tilt))-at(value,t*(1-tilt)))/2));
  const late=Math.sqrt(closures.ergodic.at(-1)*tauMax/2)/A.at(-1),returned=times.map(t=>at(value,t)-at(value,t-late));
  const series=[{quantity:'loss',indices:all(line),values:value,u:value.map((x,i)=>Math.hypot((ergodic[i]-axial[i])/2,spread[i],pushed[i],returned[i])),target:POPULATION}];
  const crossing=(L,q)=>{const i=L.findIndex(x=>x>=q);return i<1?NaN:times[i-1]+(times[i]-times[i-1])*(q-L[i-1])/(L[i]-L[i-1]);};
  write(id,{...expectation(id,'R9','R9 加热逃逸：红蓝对向光的动量扩散越过阱深（能量 Fokker–Planck，两种闭合；只作表征）',line,{
    derivation:'A red and a blue σ+ beam of one intensity counter-propagating along the tweezer’s axis, ∓100Γ: radiation pressure, light shift and Doppler force cancel; uniform momentum diffusion remains. Energy Fokker–Planck with the exact drift A = E_rec(2+Q)R and the spread B between two closures (ergodic momenta, B = (4/3)⟨KE⟩A; all kinetic energy along the beams, B = 4⟨KE⟩(1+Q+ξ)E_rec R), microcanonical shells of the Gaussian beam, absorbing at the depth, from the lowered isothermal; L delayed by the run to the 1/20 region (0 to τ_max). Value: the closures’ mean; u: half their difference, half the delay’s spread, the uncancelled forces’ tilt, and the return below the depth before the edge is reached (a delay up to √(Bτ_max/2)/A): too wide to judge a loss to 0.05 -- reported, not judged.',
    conditions:[{what:'depth (µK)',value:U},{what:'initial depth over k_BT',value:U/T},{what:'δ of the two beams at the bottom / Γ',value:beams.map(b=>b.delta/gamma)},{what:'s of each beam',value:beams.map(b=>2*b.omega**2/gamma**2)},
      {what:'excited population (both beams)',value:bottom/gamma,limit:'≪ 1'},{what:'heating E_rec(2 + Q)R at the bottom (µK/µs)',value:A[0]},{what:'time for the drift to cross the depth, U/A (µs)',value:U/A[0]},
      {what:'rate imbalance of the two beams at the bottom, (R_red − R_blue)/R',value:(R(beams[0],0)-R(beams[1],0))/bottom,limit:'≪ 1'},
      {what:'tilt of the uncancelled push over a Rayleigh range, kRz_R·2κ/|δ| (of the depth)',value:tilt,limit:'≪ 1'},{what:'E_rec/U₀ (diffusion approximation)',value:Erec/U,limit:'≪ 1'},
      {what:'axial run to the region edge at E = 0, τ_max (µs)',value:tauMax},{what:'time at which L = 0.05, 0.5, 0.95: ergodic closure (µs)',value:[0.05,0.5,0.95].map(q=>crossing(ergodic,q))},
      {what:'the same: axial closure (µs)',value:[0.05,0.5,0.95].map(q=>crossing(axial,q))},{what:'largest difference of the two closures in L',value:Math.max(...ergodic.map((x,i)=>Math.abs(x-axial[i])))},
      {what:'energy spread during the run to the edge, √(Bτ_max/2) (µK), and the drift’s time over it (µs)',value:[Math.sqrt(closures.ergodic.at(-1)*tauMax/2),late]},{what:'largest u of L',value:Math.max(...series[0].u),limit:'≤ 0.05 to judge'}],
    series,info:{zR,edge,tauMax,mandelQ:Q,kappa,closures:{ergodic,axial}}}),kind:'characterization'});
}

// ---------------------------------------------------------------------------------------------------------------------
// R10: the momentum diffusion where beams interfere.

/** The two-level atom at rest in a field of Rabi frequency Ω (the coupling is ħΩ/2: Dalibard and Cohen-Tannoudji, J. Opt.
 * Soc. Am. B 2 (1985) 1707, Eq. 2.3) detuned by δ = ω_L − ω₀ (their Eq. 2.2): the steady state's ρ_ee and the force
 * operators' correlation integrals by the quantum regression theorem on the Bloch equations ẋ = Mx + b (`mandelQ`'s).
 * For operators A, B, ∫₀^∞⟨δA(τ)δB(0)⟩dτ is A's Bloch component of −M⁻¹(x_B − ⟨B⟩x_ss), x_B the Bloch components of
 * Bρ_ss: of σ_xρ, (1, −iw, iv) with trace u; of P_eρ (P_e = |e⟩⟨e| = (1 + σ_z)/2), ((u − iv)/2, (v + iu)/2, (1 + w)/2)
 * with trace (1 + w)/2. M is real, so the integrals' real parts are those of the real right-hand sides. Returns ρ_ee
 * and the two-sided symmetric integrals xx = ∫⟨δσ_xδσ_x⟩, ee = ∫⟨δP_eδP_e⟩, xe = ∫⟨δσ_xδP_e⟩ (1/rate). The force
 * F = −∇H = ħ∇δ P_e − (ħ∇Ω/2)σ_x then has the momentum diffusion (their Eq. 4.1: D = ∫₀^∞⟨δF(t)·δF(t + τ)⟩dτ,
 * d⟨p²⟩/dt = 2D)
 *   2D_ab = ħ²[∂_aδ∂_bδ·ee + ∂_aΩ∂_bΩ·xx/4 − (∂_aδ∂_bΩ + ∂_aΩ∂_bδ)·xe/2].
 * Its limits: s → 0: xx = Γ/L, ee = Γρ_ee/L, xe = 2Γρ_eeδ/(ΩL), L = δ² + Γ²/4 -- with ρ_ee = Ω²/4L these are
 * 2D_ab = ħ²Γρ_ee Re[conj(∂_a ln f) ∂_b ln f] of the scattering amplitude f = Ω/(δ + iΓ/2): the local rule; on
 * resonance xx = 4/Γ at every Ω; Ω ≫ Γ: xx = (4/Γ)(Ω²/(Ω² + 2δ²))³, their Eq. 4.6. */
function correlations(gamma,omega,delta){
  const A=inverse3([[-gamma/2,delta,0],[-delta,-gamma/2,-omega],[0,omega,-gamma]]).map(row=>row.map(x=>-x));
  const act=q=>A.map(row=>row[0]*q[0]+row[1]*q[1]+row[2]*q[2]);
  const [u,v,w]=act([0,0,-gamma]),t=(1+w)/2,x=act([1-u*u,-u*v,-u*w]),e=act([u*(0.5-t),v*(0.5-t),t*(1-w)]);
  return {ree:t,xx:2*x[0],ee:e[2],xe:e[0]+x[2]/2};
}

/** R10. The closed cycle |2,2⟩ ↔ |3′,3′⟩ in a σ+σ+ standing wave along z ∥ B, Ω(z) = Ω₁cos(kz + θ), resonant at the
 * bottom of a tight tweezer (along x, π: both states its eigenstates) that holds the atom within kσ_z ≪ 1 of the
 * wave's phase θ at its centre (lib/defs.mjs `standing`).
 *
 * The diffusion. An atom at rest at r feels the force operator F = ħ∇δ P_e − (ħ∇Ω/2)σ_x; its momentum diffusion is the
 * force's correlation integral (`correlations`: quantum regression on the optical Bloch equations, exact at any
 * saturation) plus the emitted photons' recoil ħ²k²⟨n_an_b⟩Γρ_ee (σ dipole: ⟨n_z²⟩ = ξ = 2/5, 3/10 across). On
 * resonance and for s₀ = 2Ω₁²/Γ² → 0 this is, along the wave,
 *   2D_zz(z) = ħ²k²R₀[ξcos²(kz + θ) + sin²(kz + θ)],  R₀ = Ω₁²/Γ the photon rate at the antinode:
 * the dipole force's fluctuations, ħ²|∇Ω|²/Γ -- on resonance at every intensity (Gordon and Ashkin, Phys. Rev. A 21
 * (1980) 1606; at Ω ≫ Γ Dalibard and Cohen-Tannoudji's Eq. 4.6) --, do not vanish at the node, where nothing is
 * scattered; an incoherent sum over the two beams, (ξ + 1)ħ²k² per photon, would give (ξ + 1)cos² instead.
 * Equivalently 2D_ab = ħ²Γρ_ee Re[conj(∂_a ln f)∂_b ln f] + recoil with the scattering amplitude f(r) ∝ Ω(r)/(δ(r) +
 * iΓ/2): the second Kramers–Moyal coefficient of the jump f ρ f† with its decay, and for a real f exactly d⟨p_a²⟩/dt of
 * the low-saturation master equation, whatever the atom's state of motion (classical or not). The tweezer moves the
 * line with the potential energy V above the bottom, δ(r) = κV(r)/ħ (κ = δ_LS(0)/U₀: both shifts ∝ |a|²), so
 * ∇δ = κ∇V/ħ: the part ħ²(∇δ)²·ee = 4R(κ∇V)²/Γ² is the fluctuation of the trap's force between the two states, on
 * every axis.
 *
 * The moments. Per axis the kinetic A = ⟨p²⟩/mk_B, the potential B = mω²⟨r²⟩/k_B and C = ω⟨rp + pr⟩/2k_B obey, in the
 * harmonic trap,
 *   dA/dt = −2ωC(1 − κρ̃) + 2⟨D⟩/mk_B + 2⟨φ⟩A,  dB/dt = 2ωC,  dC/dt = ω[A − B(1 − κρ̃)] + ⟨φ⟩C:
 * the kicks give the momentum's square 2D; the mean force is the trap's less the share κρ_ee the excited atom does not
 * feel, ρ̃ = ⟨ρ_ee r²⟩/⟨r²⟩ (exact in dC/dt, and in dA/dt for jointly Gaussian r and p); φ is the force of first order
 * in the velocity (the dipole's lag at low saturation: with c_e = c⁰ − iċ⁰/(δ + iΓ/2), c⁰ = (Ω/2)/(δ + iΓ/2), the
 * power of F = −ħ∇Ω Re c_e + ħ∇δ|c_e|²),
 *   φ_a = (ħ/m)[(∂_aΩ)²δΓ/2L² − Ω∂_aΩ∂_aδ((3δ²Γ/2 − Γ³/8)/2L³ + Γ/4L²) + Ω²(∂_aδ)²δΓ/2L³],  L = δ² + Γ²/4,
 * the Doppler heating of the atoms the tweezer leaves to the blue (δ ≥ 0 off the centre). The averages are over the
 * cloud's positions, Gaussian of variances k_BB_a/mω_a² (Gauss–Hermite), every coefficient at the local Ω(z) and δ(r).
 * Slow heating gives A = B and k_B dT/dt = ⟨D⟩/m + ⟨φ⟩k_BT (half of the kicks' energy becomes potential); the
 * equations hold as well where the heating is as fast as the trap. The result's T is A over its window (one trap
 * period, no longer than the output interval; at t = 0 the instant).
 *
 * The round tweezer. Its two radial axes y, z are degenerate, and the beam's quartic term turns each atom's radial
 * orbit at Ω_p = ω²L_x/8U₀ ≤ ωE_ρ/8U₀ (L_x, the angular momentum about the tweezer's axis, is conserved; E_ρ the radial
 * energy), which carries energy between y and z and leaves their sum: the equations are solved without that exchange
 * and with it complete (the y and z moments equal throughout), the radial mean (T_y + T_z)/2 judged at the mean of
 * the two with half their difference in u, T_y and T_z alone reported (without exchange; u reaching the other limit).
 * The wave's light also modulates the y spring at twice the z frequency, by κ times ρ_ee(z(t))'s part at 2ω: a
 * parametric drive of the degenerate y, of gain γ = ωκ|ρ_ee″|z_A²/8 per unit time on the energy for an atom of z
 * amplitude z_A, the sign by its orbit's phase: to second order the cloud's T_y grows by T_y(ωκ|ρ_ee″|σ_z²t)²/32, less
 * where the quartic term detunes the two axes: in u, whole (half of it in the radial mean's).
 *
 * Uncertainty (per axis, added in quadrature): the cloud's shape (the averages close on a Gaussian: the whole quartic
 * term 2(kσ_z)⁴ of ⟨cos 2kz⟩ = e^{−2k²σ²}cos 2θ); the beam's quartic terms (3ε_a/2·k_BT_a + k_BT_other)|ΔT_a|/U₀ (R2's
 * heat capacity and cross terms); the friction's next order in the saturation, s₀ of its part; the motion's next
 * order, (2kv_rms/Γ)²|ΔT_a|; the far scattering of the tweezer -- 2E_rec per photon, and the atoms its Raman part
 * takes out of the cycle, their share of |ΔT_a| --; the excited atom's share of the trap force beyond its linear part,
 * κρ̃·2(kσ_z)²T_a(0); the parametric drive (above); the Wigner scaling's non-stationarity (R1); the quadrature (its
 * change by two nodes an axis).
 *
 * The point particle's kicks near a node (conditions, not in u): the rule's kick there is ±ħk tan(kz + θ) → ħ/x at the
 * distance x from the node, at the rate R₀k²x²: of the dipole part of the heating the kicks above an energy E carry
 * the share 2x_c p(node)/⟨sin²⟩, kx_c = atan√(E_rec/E), p the cloud's density at the node -- ∝ E^{−1/2}: no fourth
 * moment --, and the kicks above the depth eject atoms at the rate R₀(2/3)(kx_c)³p(node)/k: the loss's u. */
async function r10(id){
  const c=await cycle(id,['wave-plus','wave-minus']),{line,scene,gamma,kappa}=c,m=line.model.motion,mass=m.massKg,T=m.temperatureUK,times=grid(line);
  const wave=standingPhase(line),k=wave.k,theta=wave.thetaDeg*Math.PI/180,omega1=wave.omega1,deltaB=c.beams['wave-plus'].delta;
  if(Math.abs(wave.imbalance)>1e-9||c.beams['wave-minus'].delta!==deltaB)throw new Error(`${id}: the two beams differ: not a standing wave`);
  const tw=beamOf(line,m.trapBeamId),prep=preparedTrap(line,'ground-2-2'),omegas=prep.omegas,U0=prep.depth,{ratio,T0,u0}=initialTemperatures(prep,T);
  if(Math.abs(tw.direction[0])!==1)throw new Error(`${id}: the tweezer is not along x`);
  const r=hbarOverMass(mass),Erec=recoilUK(k,mass),R0=omega1*omega1/gamma,s0=2*omega1*omega1/(gamma*gamma),eps=[1,0.5,0.5],emission=[(1-XI_SIGMA)/2,(1-XI_SIGMA)/2,XI_SIGMA];
  // The nearest nodes' distances in phase, the cloud's density there, and how near a node a kick's energy exceeds E.
  const toNode=Math.PI/2-theta-Math.PI*Math.round((Math.PI/2-theta)/Math.PI),density=(ks,at)=>Math.exp(-at*at/(2*ks*ks))/(Math.sqrt(2*Math.PI)*ks);
  const atNode=ks=>density(ks,toNode)+(Math.abs(toNode)>1e-9?density(ks,Math.PI-Math.abs(toNode)):0),reach=E=>Math.atan(Math.sqrt(r*k*k/2/E));
  /** The cloud's averages where its positions have the potential temperatures B: per axis ⟨D_aa⟩/mk_B and its
   * state-force part (µK/µs), the friction's rate φ_a (1/µs) and ρ̃_a; ρ_ee, ⟨(2δ/Γ)²⟩, ⟨sin²(kz + θ)⟩, kσ_z. */
  const averages=(B,nodes)=>{
    const gh=nodes.map(gaussHermite),sig=B.map((t,a)=>Math.sqrt(r*KT_PER_UK*t)/omegas[a]),heat=[0,0,0],state=[0,0,0],phi=[0,0,0],soft=[0,0,0];let ree=0,spread=0,sin2=0;
    for(const [gx,wx] of gh[0])for(const [gy,wy] of gh[1])for(const [gz,wz] of gh[2]){
      const g=[gx,gy,gz],w=wx*wy*wz,V=0.5*KT_PER_UK*(B[0]*gx*gx+B[1]*gy*gy+B[2]*gz*gz),delta=deltaB+kappa*V,L=delta*delta+gamma*gamma/4;
      const phase=k*sig[2]*gz+theta,Om=omega1*Math.cos(phase),dOm=-k*omega1*Math.sin(phase),q=correlations(gamma,Om,delta);
      ree+=w*q.ree;spread+=w*(2*delta/gamma)**2;sin2+=w*Math.sin(phase)**2;
      for(let a=0;a<3;a++){
        const dDelta=kappa*omegas[a]*omegas[a]*sig[a]*g[a]/r,dO=a===2?dOm:0,st=dDelta*dDelta*q.ee;
        state[a]+=w*st;heat[a]+=w*(k*k*emission[a]*gamma*q.ree+st+dO*dO*q.xx/4-dDelta*dO*q.xe);soft[a]+=w*q.ree*g[a]*g[a];
        phi[a]+=w*(dO*dO*delta*gamma/(2*L*L)-Om*dO*dDelta*((1.5*delta*delta*gamma-gamma**3/8)/(2*L**3)+gamma/(4*L*L))+Om*Om*dDelta*dDelta*delta*gamma/(2*L**3));
      }
    }
    const scale=r/(2*KT_PER_UK);
    return {heat:heat.map(x=>scale*x),state:state.map(x=>scale*x),phi:phi.map(x=>r*x),soft,ree,spread,sin2,ksigma:k*sig[2]};
  };
  // The quadrature: the detuning alone varies along x and y; its error is the heating's change by two more nodes an axis.
  const NODES=[6,6,12],first=averages(T0,NODES),finer=averages(T0,NODES.map(n=>n+2)),quadrature=Math.max(...first.heat.map((x,a)=>Math.abs(finer.heat[a]/x-1)));
  // ρ_ee″ at the centre (1/µm²: the parametric drive's strength), by a central difference over 10⁻³ rad of the wave.
  const curvature=(()=>{const h=1e-3/k,at=z=>correlations(gamma,omega1*Math.cos(k*z+theta),deltaB).ree;return (at(h)-2*at(0)+at(-h))/(h*h);})();
  const run=runningTrap(line,'ground-2-2'),step=times[1]-times[0];
  const windows=run.omegas.map(w=>Math.min(2*Math.PI/w,step));
  /** The moments' equations from the prepared ensemble, the y–z exchange absent or complete, on a grid that resolves
   * the trap; per output T_a (A over its window), B_a, and the accumulated friction's and state force's parts of each
   * axis' energy, the photons, ∫2(kσ_z)⁴dt and the point model's ejections. */
  const solve=exchange=>{
    const mix=v=>exchange?[v[0],(v[1]+v[2])/2,(v[1]+v[2])/2]:v;
    const f=y=>{const A=mix(y.slice(0,3)),B=mix(y.slice(3,6)),C=mix(y.slice(6,9)),q=averages(B,NODES),keep=q.soft.map(x=>1-kappa*x);
      return [...mix(A.map((x,a)=>-2*omegas[a]*C[a]*keep[a]+2*q.heat[a]+2*q.phi[a]*x)),...mix(C.map((x,a)=>2*omegas[a]*x)),...mix(C.map((x,a)=>omegas[a]*(A[a]-B[a]*keep[a])+q.phi[a]*x)),
        ...A.map((x,a)=>q.phi[a]*x),...q.state,gamma*q.ree,2*q.ksigma**4,R0*2/3*reach(U0)**3*atNode(q.ksigma)];};
    const sub=2*Math.ceil(step*Math.max(...omegas)),h=step/sub,fine=[[...T0,...T0,0,0,0,0,0,0,0,0,0,0,0,0]];
    for(let n=0;n<(times.length-1)*sub+sub/2;n++){const y=fine[n],k1=f(y),k2=f(y.map((x,j)=>x+h/2*k1[j])),k3=f(y.map((x,j)=>x+h/2*k2[j])),k4=f(y.map((x,j)=>x+h*k3[j]));
      fine.push(y.map((x,j)=>x+h/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j])));}
    // ∫₀^t A_a by the trapezoid on the fine grid, A linear within a cell.
    const cumulative=[0,1,2].map(a=>{const S=[0];for(let n=1;n<fine.length;n++)S.push(S[n-1]+h*(fine[n-1][a]+fine[n][a])/2);return S;});
    const integral=(a,t)=>{const n=Math.min(Math.floor(t/h),fine.length-2),d=t-n*h,A0=fine[n][a],A1=fine[n+1][a];return cumulative[a][n]+d*(A0+(A0+(A1-A0)*d/h))/2;};
    return times.map((t,i)=>{const y=fine[i*sub];
      return {T:[0,1,2].map(a=>i?(integral(a,t+windows[a]/2)-integral(a,t-windows[a]/2))/windows[a]:y[a]),B:y.slice(3,6),friction:y.slice(9,12),state:y.slice(12,15),photons:y[15],shape:y[16],ejected:y[17]};});
  };
  const apart=solve(false),mixed=solve(true),last=apart.at(-1),end=averages(last.B,NODES),tEnd=times.at(-1);
  // The tweezer's own scattering in second order: to the ground state g′ at Σ_a|Σ_e C_a[g′,e]X_eg/Δ_e|² (the excited
  // states' paths add coherently) -- back to |2,2⟩ (Rayleigh) or away from the cycle (Raman: that atom scatters and
  // heats no more as the others do) --, each photon 2E_rec of its own.
  const E=m.hamiltonian.real.filter((_,q)=>q%33===0),g0=index(line,'ground-2-2'),at=(M,i,j)=>({re:M.real[i*32+j],im:M.imag[i*32+j]});let scatter=0,raman=0;
  for(let g=0;g<8;g++)for(const C of m.emission){let re=0,im=0;
    for(let e=8;e<32;e++){const x=coupling(line,tw.id,e,g0),d=frameDetuning(line,tw)-(E[e]-E[g0]),q=at(C,g,e);re+=(q.re*x.re-q.im*x.im)/d;im+=(q.re*x.im+q.im*x.re)/d;}
    scatter+=re*re+im*im;if(g!==g0)raman+=re*re+im*im;}
  const far=2*recoilUK(tw.waveNumberPerUm,mass)*scatter;
  // The reported trap is the running one (R2's).
  const series=trapSeries(run.omegas,run.u);
  // An axis' u at output j of a solution (the doc's list), and the series.
  const shape=[(1-XI_SIGMA)/4,(1-XI_SIGMA)/4,(1-XI_SIGMA)/2].map(x=>x*Math.abs(Math.cos(2*theta))*Erec*R0);
  const sigma2=y=>r*KT_PER_UK*y.B[2]/(omegas[2]*omegas[2]),parametric=(y,j)=>y.T[1]*(omegas[1]*kappa*Math.abs(curvature)*sigma2(y)*times[j])**2/32;
  const uncertainty=(sol,j,a)=>{const y=sol[j],dT=Math.abs(y.T[a]-T0[a]),other=Math.max(...[0,1,2].filter(b=>b!==a).map(b=>y.T[b])),v=Math.sqrt(r*KT_PER_UK*y.T[2]),ks2=k*k*sigma2(y);
    return j?Math.hypot(shape[a]*y.shape,KT_PER_UK*(1.5*eps[a]*y.T[a]+other)/U0*dT,s0*Math.abs(y.friction[a]),(2*k*v/gamma)**2*dT,far*times[j],raman*times[j]*dT,kappa*first.soft[a]*2*ks2*T0[a],
      a===1?parametric(y,j):0,T0[a]*Math.abs(wignerTemperature(omegas[a],T)/T-1)*KT_PER_UK*T/U0,quadrature*dT,u0[a]):u0[a];};
  const radial=sol=>sol.map(y=>(y.T[1]+y.T[2])/2),ra=radial(apart),rm=radial(mixed),uRadial=ra.map((x,j)=>Math.hypot((uncertainty(apart,j,1)+uncertainty(apart,j,2))/2,(x-rm[j])/2));
  // The total (T_x + T_y + T_z)/3 = (T_x + 2·radial mean)/3, the quantity the user's target is on; T_x and the radial
  // mean each against its own u.
  const Tx=apart.map((y,j)=>(y.T[0]+mixed[j].T[0])/2),uTx=apart.map((y,j)=>Math.hypot(uncertainty(apart,j,0),(y.T[0]-mixed[j].T[0])/2)),Tr=ra.map((x,j)=>(x+rm[j])/2);
  series.push({quantity:'T',total:true,regime:'any',indices:all(line),values:Tx.map((x,j)=>(x+2*Tr[j])/3),u:uTx.map((u,j)=>(u+2*uRadial[j])/3),target:TEMPERATURE});
  series.push({quantity:'T',axis:'x',regime:'any',indices:all(line),values:Tx,u:uTx,target:OWN});
  series.push({quantity:'T',axes:['y','z'],label:'radial mean',regime:'any',indices:all(line),values:Tr,u:uRadial,target:OWN});
  ['y','z'].forEach((a,n)=>series.push({quantity:'T',axis:a,regime:'any',reported:true,indices:all(line),values:apart.map(y=>y.T[1+n]),u:apart.map((y,j)=>Math.hypot(uncertainty(apart,j,1+n),y.T[1+n]-rm[j])),target:OWN}));
  series.push({quantity:'Tincrease',axes:['y','z'],label:'radial mean',reported:true,indices:all(line).slice(1),values:ra.slice(1).map((x,j)=>(x+rm[j+1])/2-ra[0]),u:uRadial.slice(1),target:OWN});
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:apart.map(y=>y.ejected),target:POPULATION});
  const ree=apart.map((y,j)=>j?averages(y.B,NODES).ree:0),uree=apart.map((y,j)=>j?ree[j]*(y.shape/times[j]+s0)+raman*times[j]:0);
  series.push({quantity:'population',state:'ground-2-2',indices:all(line),values:ree.map(x=>1-x),u:uree,target:POPULATION});
  series.push({quantity:'population',state:'D2-3-3',indices:all(line),values:ree,u:uree,target:POPULATION});
  const hw=omegas.map(w=>w/KT_PER_UK),share=E2=>2*reach(E2)*atNode(end.ksigma)/end.sin2,factor=read('scenes/manifest.json').scenes[id].held?.mass??1;
  write(id,expectation(id,'R10',`R10 驻波中的动量扩散（θ = ${wave.thetaDeg.toFixed(0)}°，s₀ = ${+s0.toPrecision(2)}${factor>1?`，质量 ×${factor}`:''}）`,line,{
    derivation:'Two-level atom in a resonant standing wave Ω₁cos(kz+θ), held within kσ_z ≪ 1 of the phase θ by a tight tweezer. Momentum diffusion of the atom at rest = correlation integral of the force ħ∇δ P_e − (ħ∇Ω/2)σ_x (quantum regression on the Bloch equations, exact in the saturation) + emission recoil; at low saturation on resonance 2D_zz = ħ²k²R₀[ξcos²(kz+θ) + sin²(kz+θ)], R₀ = Ω₁²/Γ (Gordon–Ashkin: the dipole force’s fluctuations ħ²|∇Ω|²/Γ remain at the node), and the trap’s state-dependent force adds 4R(κ∇V)²/Γ² on every axis. Harmonic moment equations for ⟨p²⟩, ⟨r²⟩, ⟨rp⟩ per axis with the kicks 2D, the first-order friction and the excited atom’s share of the trap force, averaged over the Gaussian cloud at the local Ω(z) and δ = κV. Judged: the total T at the user\'s target; T_x and the radial mean (T_y+T_z)/2 each against its own u (the round tweezer exchanges energy between y and z).',
    conditions:[{what:'s₀ = 2Ω₁²/Γ² at the antinode',value:s0},{what:'θ, the wave’s phase at the trap’s centre (deg)',value:wave.thetaDeg},
      {what:'kσ_z at t = 0',value:first.ksigma,limit:'≪ π/2'},{what:'kσ_z at the end (no exchange)',value:end.ksigma,limit:'≪ π/2'},
      {what:'least k_BT_a/ħω_a at t = 0 (the result reports its validated regime per output: `classical`)',value:Math.min(...T0.map((t,a)=>t/hw[a]))},
      {what:'⟨(2δ/Γ)²⟩ over the cloud at t = 0 (the tweezer’s line shift κV)',value:first.spread,limit:'≪ 1'},{what:'⟨(2δ/Γ)²⟩ at the end',value:end.spread,limit:'≪ 1'},
      {what:'ρ_ee over the cloud at t = 0',value:first.ree,limit:'≪ 1'},{what:'photons scattered per atom over the run',value:last.photons},
      {what:'heating of z per trap period over T_z at t = 0 (slow: ≪ 1)',value:first.heat[2]*2*Math.PI/omegas[2]/T0[2]},
      {what:'friction’s share of T_z’s increase',value:last.friction[2]/(last.T[2]-T0[2])},{what:'state-force diffusion’s share of T_y’s increase',value:last.state[1]/(last.T[1]-T0[1])},
      {what:'k_BT_z/U₀ at the end (quartic terms)',value:KT_PER_UK*last.T[2]/U0,limit:'≪ 1'},
      {what:'y–z exchange angle ω k_B(T_y+T_z)t/8U₀ at the end (rad)',value:omegas[2]*KT_PER_UK*(last.T[1]+last.T[2])*tEnd/(8*U0)},
      {what:'parametric drive of y: its part of T_y at the end over T_y',value:parametric(last,times.length-1)/last.T[1],limit:'≪ 1'},
      {what:'tweezer photons over the run',value:scatter*tEnd,limit:'≪ the wave’s'},{what:'atoms the tweezer’s Raman scattering takes out of the cycle over the run',value:raman*tEnd,limit:'≪ 1'},
      {what:'κ = δ_LS(0)/U₀',value:kappa},{what:'point model: share of the dipole heating in kicks above the depth (ejected)',value:share(U0)},
      {what:'point model: share of the dipole heating in kicks above 10 k_BT_z',value:share(10*KT_PER_UK*last.T[2])},
      ...await trapConditions(line,scene,'ground-2-2',run)],
    series,info:{depthUK:U0/KT_PER_UK,kappa,trapKHz:omegas.map(kHz),runningTrapFactor:run.factor,kineticRatio:ratio,T0,recoilUK:Erec,antinodeRatePerUs:R0,massFactor:factor,windowsUs:windows,
      zerothOrderRatesUKperUs:{z:Erec*R0*((1+XI_SIGMA)/2-(1-XI_SIGMA)/2*Math.cos(2*theta)*Math.exp(-2*first.ksigma**2)),xy:Erec*R0*(1-XI_SIGMA)/4*(1+Math.cos(2*theta)*Math.exp(-2*first.ksigma**2)),zIncoherent:Erec*R0*(1+XI_SIGMA)/2*(1+Math.cos(2*theta)*Math.exp(-2*first.ksigma**2))},
      ratesAtStartUKperUs:{heat:first.heat,stateForce:first.state,friction:first.phi.map((x,a)=>x*T0[a])},endNoExchange:last.T,endFullExchange:mixed.at(-1).T,quadratureChange:quadrature}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R4: Sisyphus cooling -- the scaling law (judged), and the old 1D quantum results TW1/TW2 (reported).

/** R4, reported only (another engine's old numeric result is never a judge: check.mjs `reported`, kind
 * "characterization"). The 1D GMW quantum results (LONGREF_TW*.json, contract §3.5: reused, not recomputed) along the
 * radial axis z through the focus: T_z(t) and the F = 1 population at the scene's output times (every 10 µs, their
 * rows), compared at the outputs the result calls classical (its validated regime: below it the semiclassical method
 * is known to run 10–55 % cold). The files are kept beside the references (refs/data). Their own
 * uncertainty: the step (dt 0.2 vs 0.1 µs: ≤ 1.2 % in T, the companions) and grid; the 3D scene's other axes are
 * assumed separable (their coupling through the tweezer's light shift ~1 % at 30 µK). */
async function r4(id){
  const line=read(`scenes/${id}.json`),name=id.slice(3),ref=read(`data/LONGREF_${name}.json`);
  const times=grid(line),rows=new Map(ref.rows.map(r=>[Math.round(r.t_us),r])),hw=ref.physics.omega_r_over_2pi_kHz*1e-3*2*Math.PI/KT_PER_UK;
  // The reference's numerical error in T at each time: its companions (the step: dt 0.2 vs 0.1 µs, the Strang error of
  // dt 0.2 being 4/3 of that difference; the momentum grid), interpolated in t; without companions (TW1) the step bias
  // its report states, 1.2 %.
  const comps=(ref.companions??[]).filter(c=>c.rows?.length).map(c=>({scale:/^dt/.test(c.label)?4/3:1,rows:c.rows}));
  const at=(rs,t)=>{let k=rs.findIndex(r=>r.t_us>=t);if(k<0)k=rs.length-1;if(k===0)return Math.abs(rs[0].rel);const r0=rs[k-1],r1=rs[k],f=(t-r0.t_us)/(r1.t_us-r0.t_us);return Math.abs(r0.rel+(r1.rel-r0.rel)*Math.min(1,Math.max(0,f)));};
  const relNum=t=>comps.length?Math.hypot(...comps.map(c=>c.scale*at(c.rows,t))):0.012;
  // The 3D cloud's x, y extent moves the cooling line by κ·(k_BT_x + k_BT_y)/2·2 ≈ κk_BT relative to the 1D reference's
  // axis (both shifts ∝ |a|²): the Sisyphus T ∝ Ω²/|δ| moves relatively by κk_BT/|δ_eff| -- the separability's size.
  const sh0=await levelShifts(read(`scenes/${id}.scene.json`),['tweezer']),kap=(sh0['D2-3-3'].shiftMHz-sh0['ground-2-2'].shiftMHz)/(-sh0['ground-2-2'].shiftMHz);
  const cool=line.model.motion.beams.find(b=>/^cool/.test(b.id)).id,dEff=RAD_PER_MHZ*Math.abs((await laserMHz(read(`scenes/${id}.scene.json`),cool))-(sh0['D2-3-3'].absoluteMHz-sh0['ground-2-2'].absoluteMHz));
  const relT=t=>{const T=rows.get(Math.round(t)).T_uK;return Math.hypot(relNum(t),kap*KT_PER_UK*T/dEff);};
  const idx=all(line).filter(i=>rows.has(Math.round(times[i])));
  // The reported trap (SPEC §1.3): the prepared one of |2,2⟩ (the tweezer along x, π: the model's own, lib/phys.mjs
  // preparedTrap, with its uncertainty) with the lin⊥lin pairs' phase-averaged shift.
  const scene=read(`scenes/${id}.scene.json`),run=runningTrap(line,'ground-2-2'),trap=run.omegas;
  const series=[...trapSeries(run.omegas,run.u),{quantity:'T',axis:'z',reported:true,regime:'classical',indices:idx,values:idx.map(i=>rows.get(Math.round(times[i])).T_uK),u:idx.map(i=>relT(times[i])*rows.get(Math.round(times[i])).T_uK),target:OWN},
    {quantity:'group',label:'F=1',reported:true,states:line.model.states.filter(s=>s.startsWith('ground-1-')),indices:idx,values:idx.map(i=>rows.get(Math.round(times[i])).pF1),u:idx.map(()=>0.003),target:POPULATION},
    {quantity:'loss',reported:true,indices:idx,values:idx.map(i=>Math.max(0,rows.get(Math.round(times[i])).loss)),u:idx.map(()=>0.002),target:POPULATION}];
  write(id,expectation(id,'R4',`R4 西西弗斯 ${name}：1D GMW 量子结果（只报告）`,line,{kind:'characterization',
    derivation:'Reported, never judged: the old 1D quantum (GMW) result of the same scene along the radial axis z, T_z and p(F=1), compared where the result calls its output classical. A 1D result is no reference of the 3D run\'s T_z in this tweezer: its two tight axes exchange energy (diffusion.status.md §9); R4D_TW2 compares like with like.',
    conditions:[{what:'ħω_r/k_B (µK) of the reference',value:hw},{what:'reference numerical error in T at the end (relative)',value:relNum(times.at(-1))},{what:'separability κk_BT/|δ_eff| at t = 0',value:kap*KT_PER_UK*30/dEff},...await trapConditions(line,scene,'ground-2-2',run)],
    series,info:{reference:`LONGREF_${name}.json`,note:ref.note,wall_s:ref.wall_s,trapKHz:trap.map(kHz),referenceRadialKHz:ref.physics.omega_r_over_2pi_kHz}}));
}

/** The lin⊥lin pair's light on F = 2 along its axis, at low saturation: the field X(z) = X₊e^{ikz} + X₋e^{−ikz} of the
 * two beams `ids` (their couplings at the focus), the light-shift operator on the ground states of F = 2,
 * W(z) = Σ_e X_e†X_e δ_e/(δ_e² + Γ_e²/4) (δ_e the light's detuning from each excited level as the trap light shifts
 * it: `detuning(e)`), and its lowest eigenstate c(z) -- the adiabatic potential the cold atom sits in --, on `points`
 * equal steps of one period π/k of the polarization. Of that state: the excited population Σ_e |X_e c|²/(δ_e² + Γ_e²/4),
 * the photon rate (Γ_e times it) and the rate of the jumps that leave it, Σ_a (‖B_a c‖² − |⟨c|B_a|c⟩|²),
 * B_a = Σ_e C_a X_e/(δ_e + iΓ_e/2) (an emission that returns the atom to c is no pumping). Returns the potential's
 * modulation and its deepest shift (rad/µs), the oscillation frequency in its well Ω_osc = √((ħ/m)∂²W_low) at the
 * minimum, the largest excited population, and the period's means of the photon and the pumping rates. */
function lightWells(line,ids,detuning,points=200){
  const m=line.model.motion,gamma=decayRates(line),k=beamOf(line,ids[0]).waveNumberPerUm,n=line.model.dimension,labelled=prefix=>line.model.states.map((s,i)=>[s,i]).filter(([s])=>s.startsWith(prefix)).map(([,i])=>i);
  const F2=labelled('ground-2-'),grounds=labelled('ground-'),excited=gamma.map((g,i)=>[g,i]).filter(([g])=>g>0).map(([,i])=>i).filter(e=>F2.some(g=>ids.some(id=>{const x=coupling(line,id,e,g);return x.re||x.im;})));
  const at=z=>{
    const c=Math.cos(k*z),s=Math.sin(k*z),xs=excited.map(e=>F2.map(g=>{const p=coupling(line,ids[0],e,g),q=coupling(line,ids[1],e,g);return {re:(p.re+q.re)*c-(p.im-q.im)*s,im:(p.im+q.im)*c+(p.re-q.re)*s};}));
    const A=F2.map(()=>F2.map(()=>0)),B=F2.map(()=>F2.map(()=>0));
    excited.forEach((e,ei)=>{const d=detuning(e),w=d/(d*d+gamma[e]**2/4);for(let a=0;a<F2.length;a++)for(let b=0;b<F2.length;b++){const x=xs[ei][a],y=xs[ei][b];A[a][b]+=w*(x.re*y.re+x.im*y.im);B[a][b]+=w*(x.re*y.im-x.im*y.re);}});
    const low=hermitianEigen(A,B)[0],v=low.vector;
    // X_e c per excited level, then the populations and the jumps.
    const amp=excited.map((e,ei)=>{let re=0,im=0;for(let a=0;a<F2.length;a++){re+=xs[ei][a].re*v[a].re-xs[ei][a].im*v[a].im;im+=xs[ei][a].re*v[a].im+xs[ei][a].im*v[a].re;}return {re,im};});
    let population=0,photons=0,pump=0;
    excited.forEach((e,ei)=>{const d=detuning(e),p=(amp[ei].re**2+amp[ei].im**2)/(d*d+gamma[e]**2/4);population+=p;photons+=gamma[e]*p;});
    for(const C of m.emission){let total=0,back={re:0,im:0};
      for(const g of grounds){let re=0,im=0;
        excited.forEach((e,ei)=>{const cr=C.real[g*n+e],ci=C.imag[g*n+e];if(!cr&&!ci)return;const d=detuning(e),den=d*d+gamma[e]**2/4,ar=(amp[ei].re*d+amp[ei].im*gamma[e]/2)/den,ai=(amp[ei].im*d-amp[ei].re*gamma[e]/2)/den;re+=cr*ar-ci*ai;im+=cr*ai+ci*ar;});
        total+=re*re+im*im;const j=F2.indexOf(g);if(j>=0){back.re+=v[j].re*re+v[j].im*im;back.im+=v[j].re*im-v[j].im*re;}}
      pump+=total-(back.re**2+back.im**2);}
    return {low:low.value,population,photons,pump};
  };
  const h=Math.PI/(k*points),pts=Array.from({length:points},(_,i)=>at(i*h)),lows=pts.map(p=>p.low),i0=lows.indexOf(Math.min(...lows)),mean=f=>pts.reduce((t,p)=>t+f(p),0)/points;
  const curvature=(lows[(i0+1)%points]-2*lows[i0]+lows[(i0+points-1)%points])/(h*h);
  return {k,modulation:Math.max(...lows)-lows[i0],deepest:-lows[i0],oscillation:Math.sqrt(hbarOverMass(m.massKg)*curvature),saturation:Math.max(...pts.map(p=>p.population)),photons:mean(p=>p.photons),pump:mean(p=>p.pump)};
}

/** R4, the scaling law (contract §6 R4): for |δ| ≫ Γ and low saturation the steady state of lin⊥lin cooling is
 * k_BT = c·ħΩ²/|δ| + O(E_rec) (Dalibard and Cohen-Tannoudji, J. Opt. Soc. Am. B 6 (1989) 2023; for F = 2 → F′ = 3 the
 * number c is known from numerics only), so at one detuning T is linear in the intensity, and three intensities
 * I, 2I, 4I satisfy T(4I) − 3T(2I) + 2T(I) = 0 whatever c and the offset are. The relation is between three scenes
 * (lib/defs.mjs `sisyphus`): its expectation names them (`members`), the combination's coefficients and each member's
 * window, the outputs the steady state is read over (the run's last third); check.mjs forms D = Σ c_k T̄_k from the
 * three results. Each member is judged alone by its reported trap only.
 *
 * What it tests, and what it does not. It tests that the steady T_z is an affine function of the intensity over
 * I … 4I, rising with it: D = 0 within z√(SE_D² + u²), the reference's own allowance and the results' statistics, and
 * T̄(4I) > T̄(I) beyond its standard error (without a rise the second difference vanishes for any physics: not judged;
 * a fall contradicts c > 0: fails). D is no temperature: the user's 10 % of T is no allowance of it, and none is
 * granted. It does not test the slope c, the offset or T itself: every T = a + bI with b > 0 passes. In particular it
 * is blind to a deficit of T that keeps T affine in I -- and the point-particle model has one in these scenes: in 1D
 * its T_z runs 18–27 % below the same model's quantum answer (motion3d_status/rewrite/diffusion.status.md §4.3, §5:
 * the Ehrenfest mean force drops the traceless part of the first order; reported by the reference R4D_TW2, accepted
 * by the owner, no part of the error budget). So R4 is no absolute test of the Sisyphus temperature, and none exists
 * in this set: T ∝ I^α gives D/T(4I) = 1 − 3·2^{−α} + 2·4^{−α} (−0.12 at α = ½, +0.19 at α = 3/2), and check.mjs prints
 * the exponents the allowance cannot tell from 1.
 *
 * What the law needs, as numbers of the scenes (the conditions) or of the results (the requirements, which check.mjs
 * reads before it judges): the detuning from the cycle's line in the trap, (Γ/2δ)² ≪ 1; the largest excited
 * population of the atom in the lowest light-shift potential, p ≪ 1, at 4I; that potential's modulation U well above
 * the recoil at I (below some hundred E_rec the law bends toward its minimum: no closed form for this atom, so a
 * condition and not a term of u) and the atom oscillating in its wells many times between two pumpings, Ω_osc τ_p ≫ 1
 * (the regime in which T depends on U alone); the tweezer no more than a slow background of the molasses -- its
 * frequency against that oscillation, (ω_z/Ω_osc)², and its line shift over the cloud against the detuning,
 * κk_BT/ħ|δ| --; the cloud over many periods of the polarization (the pattern's first harmonic averages to
 * e^{−2k²σ_z²}, σ_z² = k_BT/mω_z²); every output of the window one the result calls classical (its validated regime,
 * which the solver reports and decides nothing by); and the steady state reached (T̄ over the window's two halves equal
 * within twice their standard errors). u, relative to T(4I): p(4I) + (Γ/2δ)² + (ω_z/Ω_osc(I))² from the scenes, and
 * what check.mjs adds from the results, κk_BT(4I)/ħ|δ| + e^{−2k²σ_z²(I)}. */
async function r4linear(){
  const ids=['R4_LIN_1','R4_LIN_2','R4_LIN_4'],facts=[];
  for(const id of ids){
    const line=read(`scenes/${id}.json`),scene=read(`scenes/${id}.scene.json`),m=line.model.motion,states=line.model.states,gamma=decayRates(line)[index(line,'D2-3-3')],cool=m.beams.filter(b=>/^cool/.test(b.id)).map(b=>b.id);
    const sh=await levelShifts(scene,['tweezer']),nu=await laserMHz(scene,cool[0]),detuning=e=>RAD_PER_MHZ*(nu-(sh[states[e]].absoluteMHz-sh['ground-2-2'].absoluteMHz));
    const delta=detuning(index(line,'D2-3-3')),kappa=(sh['D2-3-3'].shiftMHz-sh['ground-2-2'].shiftMHz)/(-sh['ground-2-2'].shiftMHz),wells=lightWells(line,cool,detuning);
    const run=runningTrap(line,'ground-2-2'),trap=run.omegas;
    const {durationUs,samples}=line.options,recoil=recoilRate(wells.k,m.massKg),intensity=beamOf(line,cool[0]).intensityMWcm2;
    facts.push({id,delta,gamma,kappa,wells,trap,recoil,durationUs,samples,intensity,mass:m.massKg});
    write(id,expectation(id,'R4',`R4 西西弗斯标度律：每束 ${intensity} mW/cm²，阱内 δ = ${(delta/gamma).toFixed(1)}Γ`,line,{
      derivation:'A member of R4_LINEAR (the scaling law at three intensities): alone it is judged by its reported trap only, the prepared one of |2,2⟩ with the lin⊥lin pairs’ phase-averaged shift (lib/phys.mjs runningTrap: the trap light mixes the excited levels these pairs reach, so the resolvent is kept whole).',
      conditions:[{what:'δ from the cycle’s line in the trap / Γ',value:delta/gamma},{what:'largest excited population in the lowest light-shift potential',value:wells.saturation,limit:'≪ 1'},
        {what:'the lowest light-shift potential’s modulation over E_rec',value:wells.modulation/recoil,limit:'≫ 100'},{what:'Ω_osc τ_p: oscillations in the light’s wells between two pumpings',value:wells.oscillation/wells.pump,limit:'≫ 1'},
        {what:'photons scattered over the run',value:wells.photons*durationUs},{what:'pumping times over the run',value:wells.pump*durationUs},...await trapConditions(line,scene,'ground-2-2',run)],
      series:trapSeries(run.omegas,run.u),
      info:{trapKHz:trap.map(kHz),modulationUK:wells.modulation/KT_PER_UK,deepestShiftUK:wells.deepest/KT_PER_UK,oscillationKHz:kHz(wells.oscillation),photonRatePerUs:wells.photons,pumpRatePerUs:wells.pump}}));
  }
  const low=facts[0],high=facts.at(-1),fixed=high.wells.saturation+(low.gamma/(2*low.delta))**2+(low.trap[2]/low.wells.oscillation)**2;
  write('R4_LINEAR',{id:'R4_LINEAR',reference:'R4',title:'R4 西西弗斯标度律：T(4I) − 3T(2I) + 2T(I) = 0（lin⊥lin 稳态 T_z 对光强线性；不检验斜率与 T 本身）',kind:'relation',
    members:facts.map(f=>({id:f.id,request:`scenes/${f.id}.json`,durationUs:f.durationUs,samples:f.samples,window:[Math.ceil(2*(f.samples-1)/3),f.samples-1]})),
    quantity:'T',axis:'z',coefficients:[2,-3,1],scale:facts.length-1,value:0,u:{relative:fixed},target:OWN,
    derivation:'k_BT = c·ħΩ²/|δ| + O(E_rec) at |δ| ≫ Γ and low saturation: at one detuning T is linear in the intensity, so T(4I) − 3T(2I) + 2T(I) = 0 for the steady states of three scenes that differ in the intensity alone. Tested: that the steady T_z rises with I and is an affine function of it, the second difference zero within z·√(SE² + u²) (no user\'s target: D is no temperature). Not tested: the slope c, the offset, T itself -- every T = a + bI with b > 0 passes, so does a deficit of T that keeps it affine in I.',
    conditions:[{what:'δ from the cycle’s line in the trap / Γ',value:low.delta/low.gamma},{what:'(Γ/2δ)²',value:(low.gamma/(2*low.delta))**2,limit:'≪ 1'},
      {what:'largest excited population in the lowest light-shift potential, at I, 2I, 4I',value:facts.map(f=>f.wells.saturation),limit:'≪ 1'},
      {what:'the lowest light-shift potential’s modulation over E_rec, at I, 2I, 4I',value:facts.map(f=>f.wells.modulation/f.recoil),limit:'≫ 100'},
      {what:'that modulation U (µK), at I, 2I, 4I',value:facts.map(f=>f.wells.modulation/KT_PER_UK)},
      {what:'Ω_osc τ_p, at I, 2I, 4I: oscillations in the light’s wells between two pumpings',value:facts.map(f=>f.wells.oscillation/f.wells.pump),limit:'≫ 1'},
      {what:'pumping times over the run, at I, 2I, 4I',value:facts.map(f=>f.wells.pump*f.durationUs)},
      {what:'photons scattered over the run, at I, 2I, 4I',value:facts.map(f=>f.wells.photons*f.durationUs)},
      {what:'(ω_z/Ω_osc)² at I: the tweezer against the oscillation in the light’s wells',value:(low.trap[2]/low.wells.oscillation)**2,limit:'≪ 1'},
      {what:'ħω_z/k_B (µK)',value:low.trap[2]/KT_PER_UK},
      {what:'κ = δ_LS(0)/U₀ (the line’s shift over the cloud is κk_BT)',value:low.kappa},{what:'u from the scenes (relative to T(4I))',value:fixed}],
    requirements:['classical: every output of the window is one the result calls classical, in every member','steady: T̄ over the window’s two halves equal within twice their standard errors, in every member','a rise: T̄(4I) − T̄(I) beyond twice its standard error (a fall beyond it fails)'],
    info:{k:low.wells.k,kappa:low.kappa,deltaRadPerUs:low.delta,trapZ:low.trap[2],hbarOverMass:hbarOverMass(low.mass),modulationUK:facts.map(f=>f.wells.modulation/KT_PER_UK),intensities:facts.map(f=>f.intensity)}});
}

// ---------------------------------------------------------------------------------------------------------------------
// R8: momentum diffusion of the held atom (characterization).

/** R8 (characterization only, contract §6). The exact local momentum diffusion of R5051's atom held at its trap's centre
 * (the quantum regression theorem on the master equation with every D2 state explicit, the force operator −∇H, the
 * spontaneous photons' recoil: refs/data/qrt_R5051_n.jsonl, d⟨κ_a²⟩/dt = 2D_a, D in 1/(µm² µs)), with the force
 * correlation cut off by e^{−γτ}: γ → 0 holds the slow intra-state fluctuations a point particle's Ehrenfest force
 * leaves out (SPEC §8), which a moving atom's motion cuts off at about its trap frequency. The held atom (mass ×100,
 * scenes.mjs R8_HELD) does not cut them off; its T grows by ΔT_a = (ħ/Mm)·2D_a t/k_B, here at γ = 1/µs (the
 * cutoff kick.md compared the engines at) with u reaching the γ = 0.3/µs value -- reported, not judged; the γ = 0
 * values are in the info.
 * Held: the trap is a site of the retro-reflected tweezer's standing wave, 0.19 µm from the focus (the QRT's own
 * centre and its frequencies along the lab axes, ω_a² = (ħ/m)∂_a²V there, for ⁸⁷Rb: 1/√M of them here). With the
 * site's Hessian ω_⊥²(1 − nnᵀ) + ω_∥²nnᵀ about the tweezer's axis n (the two numbers fitted to the three lab values,
 * the fit's residual listed) the initial ensemble at k_BT ≪ ħω_⊥ is the site's ground state: spreads √(ħ/2Mmω) along
 * and across, to be far below the probe's wavelength (kσ ≪ 1) and the standing wave's period (2k_tσ_∥ ≪ 1); its
 * temperature along a lab axis is Σ_j (e_j·a)² ħω_j/2k_B, below ħω_a/2k_B of that axis' own frequency (the mean of the
 * roots against the root of the mean). */
async function r8(){
  const id='R8_R5051',line=read(`scenes/${id}.json`),times=grid(line),m=line.model.motion,r=hbarOverMass(m.massKg),{R8_HELD}=await import('./scenes.mjs');
  const qrt=fs.readFileSync(path.join(here,'data','qrt_R5051_n.jsonl'),'utf8').split('\n').filter(x=>x.trim()).map(x=>JSON.parse(x));
  const centre=qrt.slice(1).find(q=>q.offset.every(v=>v===0)),D=g=>centre.cutoffs.find(c=>c.gamma===g).D;
  // The site: ω_a along the lab axes, and ω_⊥², ω_∥² by least squares on ω_a² = ω_⊥² + (ω_∥² − ω_⊥²) n_a².
  const lab=qrt[0].trap.lab.map(w=>w/Math.sqrt(R8_HELD.mass)),tw=beamOf(line,m.trapBeamId),norm=Math.hypot(...tw.direction),n2=tw.direction.map(x=>(x/norm)**2);
  const my=lab.reduce((t,w)=>t+w*w,0)/3,slope=lab.reduce((t,w,a)=>t+(w*w-my)*(n2[a]-1/3),0)/n2.reduce((t,x)=>t+(x-1/3)**2,0),across=Math.sqrt(my-slope/3),along=Math.sqrt(my-slope/3+slope);
  const residual=Math.max(...lab.map((w,a)=>Math.abs(across**2+slope*n2[a]-w*w)/(w*w))),T=m.temperatureUK,spread=w=>Math.sqrt(r*KT_PER_UK*wignerTemperature(w,T))/w,probe=Math.max(...m.beams.filter(b=>b.id!==tw.id&&String(b.frequencyIndex)!==String(tw.frequencyIndex)).map(b=>b.waveNumberPerUm));
  const series=['x','y','z'].map((a,i)=>({quantity:'Tincrease',axis:a,indices:all(line),values:times.map(t=>r*2*D(1)[i]*t/KT_PER_UK),u:times.map(t=>r*2*Math.abs(D(0.3)[i]-D(1)[i])*t/KT_PER_UK),target:OWN}));
  write(id,{...expectation(id,'R8','R8 静止原子的动量扩散（R5051，QRT；只作表征）',line,{
    derivation:'Held atom (mass ×100): Var κ_a grows as 2D_a t with the QRT diffusion of the exact master equation at the trap centre; D at the correlation cutoff γ = 1/µs, u to γ = 0.3/µs. The γ → 0 limit (slow intra-state fluctuations) is beyond the semiclassical point particle (SPEC §8).',
    conditions:[{what:'photon rate at the centre (1/µs)',value:centre.photonRate},{what:'excited population',value:centre.excited},
      {what:'held atom (mass factor, initial T in µK)',value:R8_HELD},{what:'the site’s centre from the focus (µm)',value:Math.hypot(...qrt[0].trap.centre)},
      {what:'trap frequencies along x, y, z (kHz)',value:lab.map(kHz)},{what:'across and along the standing wave (kHz); the fit’s largest relative residual in ω_a²',value:[kHz(across),kHz(along),residual]},
      {what:'ħω/k_BT across and along',value:[across,along].map(w=>w/(KT_PER_UK*T)),limit:'≫ 1: the site’s ground state'},
      {what:'Wigner factor less 1, across and along: coth(ħω/2k_BT) − 1',value:[across,along].map(w=>1/Math.tanh(w/(2*KT_PER_UK*T))-1),limit:'≪ 1'},
      {what:'position spread across and along (µm)',value:[spread(across),spread(along)]},{what:'the classical thermal part of it, √(k_BT/Mm)/ω (µm)',value:[across,along].map(w=>Math.sqrt(r*KT_PER_UK*T)/w)},
      {what:'kσ: the probe’s wave number times the larger spread',value:probe*spread(across),limit:'≪ 1: held'},{what:'2k_tσ_∥: the standing wave’s phase over the spread along it',value:2*tw.waveNumberPerUm*spread(along),limit:'≪ 1: held'},
      {what:'initial T along x, y, z (µK): Σ_j (e_j·a)² T*_j',value:n2.map(c=>c*wignerTemperature(along,T)+(1-c)*wignerTemperature(across,T))},
      {what:'expected increase over the run along x, y, z (µK)',value:[0,1,2].map(i=>r*2*D(1)[i]*times.at(-1)/KT_PER_UK)}],
    series,info:{source:'refs/data/qrt_R5051_n.jsonl',cutoffs:centre.cutoffs.map(c=>({gamma:c.gamma,D:c.D,Dforce:c.Dforce})),Drecoil:centre.Drecoil}}),kind:'characterization'});
}

const RUN={R1_TWEEZER:()=>r1('R1_TWEEZER'),R1_RETRO:()=>r1('R1_RETRO'),R1_SIGMA_P:()=>r1('R1_SIGMA_P'),R1_SIGMA_M:()=>r1('R1_SIGMA_M'),R1_CROSSED:()=>r1('R1_CROSSED'),
  R2_RECOIL:r2,R3_DOPPLER:r3,R5_PUMP:r5,R6_RAMAN_CO:()=>r6('R6_RAMAN_CO'),R6_RAMAN_COUNTER:()=>r6('R6_RAMAN_COUNTER'),R9_PUSH:r9,R4_TW1:()=>r4('R4_TW1'),R4_TW2:()=>r4('R4_TW2'),R4_LINEAR:r4linear,R8_R5051:r8,R9_HEAT:r9heat,
  ...Object.fromEntries(['ANTINODE','QUARTER','NODE'].flatMap(at=>[`R10_${at}`,`R10_HELD_${at}`]).map(id=>[id,()=>r10(id)]))};
const ids=process.argv.slice(2);
for(const [id,f] of Object.entries(RUN))if(!ids.length||ids.includes(id))await f();
