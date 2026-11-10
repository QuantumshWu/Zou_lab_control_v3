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
import {KT_PER_UK,RAD_PER_MHZ,beamMeasure,branchRate,coupling,decayRates,gaussLegendre,hbarOverMass,index,integrate,kineticRatio,preparedTrap,rayleigh,recoilUK,solve3,uKOfMHz,wignerTemperature} from './lib/phys.mjs';
import {ramanNumbers} from './lib/defs.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const read=f=>JSON.parse(fs.readFileSync(path.join(here,f),'utf8'));
const write=(id,e)=>{const f=path.join(here,'expect',`${id}.json`);fs.mkdirSync(path.dirname(f),{recursive:true});fs.writeFileSync(f,JSON.stringify(e,null,1));console.log(id,e.series.length,'series');};
const grid=line=>{const {durationUs:T,samples:n}=line.options;return Array.from({length:n},(_,i)=>T*i/(n-1));};
const all=line=>grid(line).map((_,i)=>i);
const beamOf=(line,id)=>line.model.motion.beams.find(b=>b.id===id);
/** The user's targets (SPEC §0): populations and loss 0.05 absolute, T 0.10 relative. */
const POPULATION={absolute:0.05},TEMPERATURE={relative:0.10};
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
 * oscillator's Wigner factor: T_a(0) = (T_kin/T)·T*_a, T*_a = (ħω_a/2k_B) coth(ħω_a/2k_BT). */
function initialTemperatures(trap,T){
  const ratio=kineticRatio(beamMeasure(trap.depth/(KT_PER_UK*T)));
  return {ratio,T0:trap.omegas.map(w=>ratio*wignerTemperature(w,T))};
}
/** ω (rad/µs) → trapFrequencyKHz. */
const kHz=w=>w/(2*Math.PI)*1e3;

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
 * and breaks it only through the anharmonic part, relatively ~k_BT/U, so u_a = T_a(0)·|T*_a/T − 1|·k_BT/U. (iii) No
 * atom is unbound (E < 0) and energy is conserved: L ≡ 0. (iv) The far light's scattering Γ_sc = Σ Γ_e|X_eg|²/δ²
 * (incoherent sum, an upper bound of the coherent one) bounds any population change by Γ_sc·t. */
async function r1(id){
  const line=read(`scenes/${id}.json`),scene=read(`scenes/${id}.scene.json`),m=line.model.motion,T=m.temperatureUK,init=scene.dynamics.initialStateId,times=grid(line);
  const trap=preparedTrap(line,init),U=trap.depth,omegas=trap.omegas,{ratio,T0}=initialTemperatures(trap,T);
  const chart=(await levelShifts(scene,trap.beams.map(b=>b.id)))[init];
  const series=[];
  ['x','y','z'].forEach((a,i)=>series.push({quantity:'trapFrequency',axis:a,value:kHz(omegas[i]),u:kHz(trap.u[i]),target:{relative:0}}));
  ['x','y','z'].forEach((a,i)=>{const T0a=T0[i],u=T0a*Math.abs(wignerTemperature(omegas[i],T)/T-1)*(KT_PER_UK*T/U);
    series.push({quantity:'T',axis:a,regime:'any',indices:all(line),values:times.map(()=>T0a),u:times.map(()=>u),target:TEMPERATURE});});
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
/** The running trap (SPEC §1.3: the reported trap frequency is the running trap's) of a scene whose near beams drive
 * weakly (|X|²/(δ² + Γ²/4) ≪ 1, whatever their detuning): the trap light's potential plus the other lights'
 * phase-averaged smooth shift of the reference state g, U = Σ_{l,e} |X^l_eg|² δ_le/(δ_le² + Γ_e²/4) (the second-order
 * shift with the excited state's width -- the order of the solver's elimination --, beams of one frequency summed
 * without their fringes), each δ_le = δ_le,b + κ_e V at the potential energy V above the bottom
 * (κ_e = (shift_e − shift_g)/U₀, both ∝ |a|²): so ∂²V_run = ∂²V_prep (1 + dU/dV),
 * dU/dV = Σ κ_e |X|²(Γ_e²/4 − δ²)/(δ² + Γ_e²/4)². Returns that factor and the next order's size (the factor's
 * departure times the largest |X|²/(δ² + Γ²/4), as the reference's uncertainty). */
async function runningFactor(line,scene,g,beams){
  const states=line.model.states,gi=index(line,g),gamma=decayRates(line),sh=await levelShifts(scene,['tweezer']),U0=-sh[g].shiftMHz;
  let slope=0,smax=0;
  for(const id of beams){const nu=await laserMHz(scene,id);
    for(let e=8;e<32;e++){const x=coupling(line,id,e,gi),X2=x.re*x.re+x.im*x.im;if(!X2)continue;
      const delta=RAD_PER_MHZ*(nu-(sh[states[e]].absoluteMHz-sh[g].absoluteMHz)),kappa=(sh[states[e]].shiftMHz-sh[g].shiftMHz)/U0,G2=gamma[e]**2/4;
      slope+=kappa*X2*(G2-delta*delta)/((delta*delta+G2)**2);smax=Math.max(smax,X2/(delta*delta+G2));}}
  return {factor:1+slope,u:Math.abs(slope)*smax};
}
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
 * its remainder bounded by the oscillation's amplitude over √(samples)). Populations: ρ_ee = ⟨R⟩/Γ in |3′,3′⟩. */
async function r2(){
  const id='R2_RECOIL',c=await cycle(id,['probe']),{line,gamma,U0,kappa}=c,p=c.beams.probe,m=line.model.motion,mass=m.massKg,T=m.temperatureUK;
  const prep=preparedTrap(line,'ground-2-2'),omegas=prep.omegas,[radial,,axial]=omegas;
  const {ratio,T0}=initialTemperatures(prep,T),Erec=recoilUK(p.k,mass),Q=mandelQ(gamma,p.omega,p.delta);
  const R=delta=>rate(gamma,p.omega,delta),h=1e-3*Math.abs(p.delta),R2=(R(p.delta+h)-2*R(p.delta)+R(p.delta-h))/(h*h);
  const meanR=Ts=>{const kT=Ts.map(t=>KT_PER_UK*t),V=kT.reduce((a,b)=>a+b,0)/2,varV=kT.reduce((a,b)=>a+b*b/2,0);return R(p.delta+kappa*V)+R2*kappa*kappa*varV/2;};
  const rates=[(1-XI_SIGMA)/2,(1-XI_SIGMA)/2,1+Q+XI_SIGMA],eps=[0.5,0.5,1];
  const derivative=Ts=>{const r=meanR(Ts);return Ts.map((t,a)=>Erec*rates[a]*r/(1+1.5*eps[a]*KT_PER_UK*t/U0));};
  const times=grid(line),Ts=[T0.slice()],Rs=[meanR(T0)];let state=T0.slice();
  for(let i=1;i<times.length;i++){const n=20,dt=(times[i]-times[i-1])/n;for(let j=0;j<n;j++){const k1=derivative(state),k2=derivative(state.map((x,a)=>x+dt/2*k1[a])),k3=derivative(state.map((x,a)=>x+dt/2*k2[a])),k4=derivative(state.map((x,a)=>x+dt*k3[a]));state=state.map((x,a)=>x+dt/6*(k1[a]+2*k2[a]+2*k3[a]+k4[a]));}Ts.push(state.slice());Rs.push(meanR(state));}
  // The radiation-pressure offset, the spring softened by the probe's light shift and by the cloud's anharmonicity.
  const rbar=Rs.reduce((a,b)=>a+b,0)/Rs.length,r=hbarOverMass(mass),run=await runningFactor(line,c.scene,'ground-2-2',['probe']),dyn=axial*Math.sqrt(run.factor);
  const z0=r*p.k*rbar/(dyn*dyn),kTbar=KT_PER_UK*Ts.reduce((a,s)=>a+(s[0]+s[1]+s[2])/3,0)/Ts.length/U0;
  // The oscillation's remainder in the average over the 20 evenly spaced distribution samples after t = 0 (contract §2.2:
  // at most 21 times with 0 and the end): at most max |mean cos ωt_i| over the anharmonic frequency window
  // [ω_dyn(1 − 5k_BT/U₀), ω_dyn], times z₀ (no dephasing credited).
  const tS=Array.from({length:20},(_,i)=>times.at(-1)*(i+1)/20);let rest=0;
  for(let w=dyn*(1-5*kTbar);w<=dyn;w+=dyn*1e-5)rest=Math.max(rest,Math.abs(tS.reduce((s,t)=>s+Math.cos(w*t),0))/tS.length);
  const zbar=z0*(1+3.5*kTbar),zu=Math.hypot(z0*1.5*kTbar,z0*Math.abs(Rs.at(-1)-Rs[0])/(2*rbar),z0*rest);
  const series=[];
  // The reported trap is the running one (SPEC §1.3): the prepared one (the model's own, lib/phys.mjs preparedTrap, with
  // its uncertainty) softened by the probe's second-order shift at the local detuning (runningFactor: ω² × (1 − κΩ²/4δ²)
  // for |δ| ≫ Γ). The dressed two-level shift (−δ + sgn δ·√(δ² + Ω²))/2 softens by (3/4)(Ω/δ)² of that less: the fold's
  // fourth order, the solver's model error (its ledger: the probe's drive), not this reference's.
  ['x','y','z'].forEach((a,i)=>series.push({quantity:'trapFrequency',axis:a,value:kHz(omegas[i]*Math.sqrt(run.factor)),u:Math.hypot(kHz(prep.u[i])*Math.sqrt(run.factor),kHz(omegas[i])*run.u/2),target:{relative:0}}));
  // u(T): the cross-axis quartic terms (k_BT_other/U₀)·ΔT_a; the centre-of-mass energy ½mω²z₀² that dephasing turns
  // into heat (z, after t = 0); the Wigner scaling's non-stationarity (R1). Mandel's Q is in the rate.
  const push=0.5*mass*(dyn*1e6)**2*(z0*1e-6)**2/1.380649e-23*1e6;
  ['x','y','z'].forEach((a,i)=>{
    const u=Ts.map((s,j)=>{const other=Math.max(...s.filter((_,b)=>b!==i)),dT=s[i]-T0[i];return Math.hypot(KT_PER_UK*other/U0*dT,i===2&&j>0?push:0,T0[i]*Math.abs(wignerTemperature(omegas[i],T)/T-1)*KT_PER_UK*T/U0);});
    series.push({quantity:'T',axis:a,regime:'any',indices:all(line),values:Ts.map(s=>s[i]),u,target:TEMPERATURE});});
  // The offset's target: the force's, the T target 10 % (the heating that R2 judges is ∝ the same rate R).
  series.push({quantity:'meanPositionAverage',axis:'z',value:zbar,u:zu,target:{absolute:TEMPERATURE.relative*zbar}});
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  const ree=Rs.map(x=>x/gamma);
  series.push({quantity:'population',state:'ground-2-2',indices:all(line),values:ree.map((x,i)=>i?1-x:1),u:ree.map((x,i)=>i?0.1*x:0),target:POPULATION});
  series.push({quantity:'population',state:'D2-3-3',indices:all(line),values:ree.map((x,i)=>i?x:0),u:ree.map((x,i)=>i?0.1*x:0),target:POPULATION});
  const s=2*p.omega**2/gamma**2;
  write(id,expectation(id,'R2','R2 二能级辐射压与反冲加热',line,{
    derivation:'Two-level photon rate R = (Γ/2)s/(1+s+(2δ/Γ)²) at the local detuning (cloud average to 2nd order in κV); per photon Var p grows by (ħk)²(1+Q+ξ) along the beam and (ħk)²(1−ξ)/2 across (ξ = 2/5, σ dipole): k_B dT_z/dt = (7/5)E_rec R, k_B dT_x,y/dt = (3/10)E_rec R, with the separable quartic heat capacity; radiation pressure offset z₀ = ħkR/(mω²) (the spring softened by the probe’s light shift), time-averaged over the dephasing oscillation with the anharmonic orbit-centre shift.',
    conditions:[{what:'|δ_b|/Γ',value:Math.abs(p.delta)/gamma,limit:'≫ 1'},{what:'s',value:s},{what:'κ·⟨V⟩/|δ_b| (cloud spread of the detuning)',value:kappa*KT_PER_UK*1.5*T/Math.abs(p.delta),limit:'≪ 1'},
      {what:'k_BT/U₀ at the end (quartic corrections)',value:KT_PER_UK*Math.max(...Ts.at(-1))/U0,limit:'≪ 1'},{what:'Mandel Q',value:Q},{what:'k_BT ≥ 4ħω_radial at t = 0',value:KT_PER_UK*T/radial,limit:'≥ 4'},
      {what:'one beam\'s Doppler damping time m/β (s), β = ħk²|dR/dδ|',value:1/(hbarOverMass(mass)*p.k**2*rbar*8*Math.abs(p.delta)/(gamma*gamma)/(1+s+(2*p.delta/gamma)**2))*1e-6,limit:'≫ the run'}],
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
 * (k_BT_a/U₀)|ΔT_a|/2. Mandel's Q is in the diffusion. The run's T on x and y only grows (no cooling across). */
async function r3(){
  const id='R3_DOPPLER',c=await cycle(id,['cool-plus','cool-minus']),{line,gamma,U0,kappa}=c,b=c.beams['cool-plus'],m=line.model.motion,mass=m.massKg,T=m.temperatureUK;
  const prep=preparedTrap(line,'ground-2-2'),omegas=prep.omegas,axial=omegas[2],eps=[0.5,0.5,1];
  const {ratio,T0}=initialTemperatures(prep,T),Erec=recoilUK(b.k,mass),wrec=hbarOverMass(mass)*b.k*b.k/2,s=2*b.omega**2/gamma**2;
  const Q=mandelQ(gamma,Math.SQRT2*b.omega,b.delta),times=grid(line);
  const capacity=(t,a)=>1+1.5*eps[a]*KT_PER_UK*t/U0,energy=(t,a)=>t*(1+0.75*eps[a]*KT_PER_UK*t/U0),temperature=(E,a)=>{const c2=0.75*eps[a]*KT_PER_UK/U0;return c2>0?(-1+Math.sqrt(1+4*c2*E))/(2*c2):E;};
  // Gauss–Chebyshev nodes of sin²φ (arcsine law on [0, 1]), equal weights.
  const n=5,nodes=Array.from({length:n},(_,j)=>(1-Math.cos(Math.PI*(2*j+1)/(2*n)))/2);
  const coefficients=(E)=>{
    let Az=0,Bz=0,Ax=0,Bx=0,Ay=0,By=0;
    for(const sx of nodes)for(const sy of nodes)for(const sz of nodes){
      const V=E[0]*sx+E[1]*sy+E[2]*sz,x=2*(b.delta+kappa*KT_PER_UK*V)/gamma,R=gamma*s/(1+x*x),beta=-8*wrec*s*x/((1+x*x)**2);
      const Dz=Erec*(1+Q+XI_SIGMA)*R,Dx=Erec*(1-XI_SIGMA)/2*R;
      Az+=-2*beta*E[2]*(1-sz)+Dz;Bz+=4*E[2]*(1-sz)*Dz;Ax+=Dx;Bx+=4*E[0]*(1-sx)*Dx;Ay+=Dx;By+=4*E[1]*(1-sy)*Dx;}
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
  // The reported trap (SPEC §1.3): the prepared one (the model's own, lib/phys.mjs preparedTrap, with its uncertainty)
  // softened or stiffened by the cooling pair's phase-averaged shift.
  const run=await runningFactor(line,c.scene,'ground-2-2',['cool-plus','cool-minus']);
  const series=['x','y','z'].map((a,i)=>({quantity:'trapFrequency',axis:a,value:kHz(omegas[i]*Math.sqrt(run.factor)),u:Math.hypot(kHz(prep.u[i])*Math.sqrt(run.factor),kHz(omegas[i])*run.u/2),target:{relative:0}}));
  ['x','y','z'].forEach((a,i)=>{
    const u=Ts.map((row,j)=>{const other=Math.max(...row.filter((_,k)=>k!==i).map(q=>q[0])),dT=Math.abs(row[i][0]-T0[i]);
      return Math.hypot(row[i][1],j?2*s/(1+meanX*meanX)*(i===2?row[i][0]:dT):0,KT_PER_UK*other/U0*dT,KT_PER_UK*row[i][0]/U0*dT/2);});
    series.push({quantity:'T',axis:a,regime:'any',indices:all(line),values:Ts.map(row=>row[i][0]),u,target:TEMPERATURE});});
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  series.push({quantity:'population',state:'ground-2-2',indices:all(line),values:times.map((_,i)=>i?1-gamma*s/(1+meanX*meanX)/gamma/2:1),u:times.map((_,i)=>i?s:0),target:POPULATION});
  write(id,expectation(id,'R3','R3 一维多普勒冷却（σ⁺σ⁺ 驻波，非魔术光镊：按局域失谐平均）',line,{
    derivation:'Low-saturation two-beam Doppler friction and diffusion at the local detuning δ_b + κV (the tweezer moves the cycling line with the potential energy; no magic wavelength exists in the model), averaged over each atom\'s fast trap motion: phase-averaged Itô equations for the per-axis energies, solved by Monte Carlo from the prepared ensemble; free-space limit k_BT = ħΓ(1+ξ)(1+x²)/(16|δ|/Γ).',
    conditions:[{what:'s per beam',value:s,limit:'≤ 0.02'},{what:'x = 2δ/Γ at the bottom',value:xb},{what:'x at the end\'s mean potential energy',value:meanX},{what:'κ = δ_LS(0)/U₀',value:kappa},
      {what:'k v_rms/Γ at the Doppler T',value:b.k*Math.sqrt(hbarOverMass(mass)*KT_PER_UK*doppler(meanX))/gamma,limit:'≪ 1'},{what:'depth over k_BT at the end',value:U0/(KT_PER_UK*Math.max(...Ts.at(-1).map(q=>q[0]))),limit:'≥ 20'},
      {what:'axial ω over the cooling rate β/m',value:axial/(8*wrec*s*Math.abs(xb)/((1+xb*xb)**2)),limit:'≫ 1'},{what:'Mandel Q',value:Q}],
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
 * Σ_{l,e} Γ_e|X^l_eg|²/δ². Co-propagating (Δk ≈ 0): P₂ = (Ω_R²/Ω_eff²) sin²(Ω_eff t/2), Ω_eff = √(Ω_R² + δ₂²); an atom
 * that scatters loses the coherence, so P₂ = e^{−Γ̄t}P₂^coh + (1 − e^{−Γ̄t})/2 with u = (1 − e^{−Γ̄t})/2.
 * Counter-propagating (Δk = k_A + k_B along z): the transition |1, p⟩ → |2, p + ħΔk⟩ has δ₂(t) = −Δk v_z(t) − ħΔk²/2m;
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
  // Gauss–Hermite (probabilists') nodes for a standard normal, by the Golub–Welsch eigenproblem of the Jacobi matrix.
  const hermite=n=>{const a=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>Math.abs(i-j)===1?Math.sqrt(Math.max(i,j)):0));
    // Jacobi eigenvalue iteration (symmetric).
    const v=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
    for(let it=0;it<200;it++){let off=0;for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){if(Math.abs(a[p][q])<1e-15)continue;off+=a[p][q]**2;const th=0.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(th),s=Math.sin(th);
      for(let k=0;k<n;k++){const akp=a[k][p],akq=a[k][q];a[k][p]=c*akp-s*akq;a[k][q]=s*akp+c*akq;}for(let k=0;k<n;k++){const apk=a[p][k],aqk=a[q][k];a[p][k]=c*apk-s*aqk;a[q][k]=s*apk+c*aqk;}
      for(let k=0;k<n;k++){const vkp=v[k][p],vkq=v[k][q];v[k][p]=c*vkp-s*vkq;v[k][q]=s*vkp+c*vkq;}}if(off<1e-28)break;}
    return Array.from({length:n},(_,i)=>[a[i][i],v[0][i]**2]);};
  const counter=Math.abs(dk)>1;
  // i ċ = Hc, H = ½[[−δ, Ω_R],[Ω_R, δ]] in the two-photon rotating frame (P₂ is even in δ), δ(t) given; P₂ on the grid.
  const evolve=(delta)=>{
    let a=[1,0,0,0];const P=[0];const f=(t,y)=>{const d=delta(t),h=omegaR/2,[r1,i1,r2,i2]=y;return [-d/2*i1+h*i2,d/2*r1-h*r2,h*i1+d/2*i2,-h*r1-d/2*r2];};
    for(let i=1;i<times.length;i++){const n=8,dt=(times[i]-times[i-1])/n;let t=times[i-1];for(let s=0;s<n;s++){const k1=f(t,a),k2=f(t+dt/2,a.map((x,j)=>x+dt/2*k1[j])),k3=f(t+dt/2,a.map((x,j)=>x+dt/2*k2[j])),k4=f(t+dt,a.map((x,j)=>x+dt*k3[j]));a=a.map((x,j)=>x+dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j]));t+=dt;}P.push(a[2]**2+a[3]**2);}
    return P;};
  let P2;
  if(!counter){P2=evolve(()=>d2);}
  else{const nodes=hermite(24);P2=times.map(()=>0);
    for(const [x,wx] of nodes)for(const [y,wy] of nodes){const z0=sz*x,v0=sv*y,P=evolve(t=>d2-dk*(v0*Math.cos(axial*t)-axial*z0*Math.sin(axial*t))-recoil);P.forEach((p,i)=>P2[i]+=wx*wy*p);}}
  const f=times.map(t=>1-Math.exp(-scatter*t));
  const p2=P2.map((p,i)=>(1-f[i])*p+f[i]/2),p1=P2.map((p,i)=>(1-f[i])*(1-p)+f[i]/2);
  const series=[{quantity:'population',state:'ground-2-0',indices:all(line),values:p2,u:f.map(x=>x/2),target:POPULATION},
    {quantity:'population',state:'ground-1-0',indices:all(line),values:p1,u:f.map(x=>x/2),target:POPULATION}];
  if(counter){
    const vel=times.map((t,i)=>{let s=0;for(let j=1;j<=i;j++)s+=Math.cos(axial*(t-(times[j]+times[j-1])/2))*(P2[j]*(1-f[j])-P2[j-1]*(1-f[j-1]));return r*dk*s;});
    const unit=r*dk;
    series.push({quantity:'meanVelocity',axis:'z',indices:all(line),values:vel,u:vel.map((_,i)=>Math.abs(unit)*f[i]),target:{absolute:Math.abs(unit)*POPULATION.absolute}});
  }
  series.push({quantity:'loss',indices:all(line),values:times.map(()=>0),u:times.map(()=>0),target:POPULATION});
  write(id,expectation(id,'R6',`R6 拉曼 Λ（${counter?'对向，Δk = 2k':'同向'}）`,line,{
    derivation:counter?'Two-photon Rabi oscillation with the Doppler detuning −Δk·v(t) along harmonic axial orbits and the recoil shift ħΔk²/2m, averaged over the prepared ensemble; ⟨v_z⟩ = (ħΔk/m)∫cos ω(t−s) dP̄₂; scattering destroys the coherence (u).':'Two-photon Rabi oscillation P₂ = (Ω_R²/Ω_eff²)sin²(Ω_eff t/2) (second-order Λ, light shifts compensated); scattering destroys the coherence (u). The exact static evolution is R0_R6_RAMAN_CO.',
    conditions:[{what:'Ω_R/2π (MHz)',value:omegaR/(2*Math.PI)},{what:'residual two-photon detuning (MHz)',value:twoPhotonMHz},{what:'scattering per µs (mean of the two states)',value:scatter},
      {what:'Δk (rad/µm)',value:dk},{what:'axial ω·t at the end (orbit integration needed when not ≪ 1)',value:axial*times.at(-1)},{what:'Δk·v_rms/Ω_R',value:Math.abs(dk)*sv/omegaR}],
    series,info:{omegaR,twoPhotonMHz,scatter,recoilShift:recoil,axialKHz:kHz(axial),T0z:T0[2]}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R9 (b): the push out of the trap.

/** R9 (b). A σ+ beam along the tweezer's axis z, power-broadened (s = 100) over the differential shift, pushes the
 * atom out. On the axis, κ̇ = F/ħ with F/ħ = k R(δ) − ∂_zU_g/ħ − ρ_ee ∂_zδ_LS (radiation pressure at the exact
 * steady-state rate, the ground state's trap force and the excited state's extra force while excited), δ = δ_free −
 * δ_LS(z) − k v (δ_free from the free line; δ_LS(z) = δ_LS(0)/(1 + z²/z_R²) since both shifts ∝ |a|²), ż = (ħ/m)κ.
 * The photon kicks add the momentum diffusion d Var κ/dt = k²(1 + Q + ξ)R (absorption with Mandel's Q, σ emission).
 * An atom is lost when it leaves the 1/20 region, z > √19 z_R on the axis (its kinetic energy there is far above the
 * trap: E_s > 0 holds). L(t) = P(τ ≤ t) by a Monte Carlo of this 1D Langevin equation from the prepared axial
 * distribution (Gaussian: T*_z and T(0)_z of the deep trap), N paths with a fixed seed; u = the binomial error plus
 * |dL/dt|·u_τ, u_τ the exit time's change when the neglected parts are added (the radial offset's 2ρ²/w² ≈ 1 % of the
 * trap force, the internal state's lag 1/Γ_eff of the rate's changes): estimated by rerunning with the trap force
 * ×(1 ± 2ρ²/w²) and the rate delayed by 1/Γ. */
async function r9(){
  const id='R9_PUSH',c=await cycle(id,['push']),{line,gamma,U0,dLS}=c,p=c.beams.push,m=line.model.motion,mass=m.massKg,T=m.temperatureUK,r=hbarOverMass(mass);
  const tw=beamOf(line,'tweezer'),zR=rayleigh(tw.waveNumberPerUm,tw.waistUm),prep=preparedTrap(line,'ground-2-2'),[radial,,axial]=prep.omegas,{T0}=initialTemperatures(prep,T);
  const deltaFree=p.delta+dLS,Q=mandelQ(gamma,p.omega,deltaFree-dLS),edge=Math.sqrt(19)*zR,times=grid(line);
  const sz=Math.sqrt(r*KT_PER_UK*wignerTemperature(axial,T))/axial,sk=Math.sqrt(KT_PER_UK*T0[2]/r);
  // A seeded generator (mulberry32; each run restarts it: common random numbers between the runs) and normal deviates
  // (Box–Muller).
  let seed=0;const next=()=>{seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  const normal=()=>{const u=1-next(),v=next();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
  const run=({N,trapScale=1,lag=0})=>{
    const tau=[];const dt=0.01;seed=0x9E3779B9;
    for(let n=0;n<N;n++){let z=sz*normal(),kap=sk*normal(),t=0,R=0,ree=0;
      while(t<times.at(-1)+dt){
        const g=1/(1+z*z/(zR*zR)),delta=deltaFree-dLS*g-p.k*r*kap,s=2*p.omega**2/gamma**2,target=gamma/2*s/(1+s+(2*delta/gamma)**2);
        R=lag?R+(target-R)*Math.min(1,dt*gamma*Math.sqrt(1+s)/1):target;ree=R/gamma;
        const force=p.k*R-trapScale*2*U0*z*g*g/(zR*zR)+ree*2*dLS*z*g*g/(zR*zR);
        kap+=force*dt+Math.sqrt(p.k*p.k*(1+Q+XI_SIGMA)*R*dt)*normal();z+=r*kap*dt;t+=dt;
        if(z>edge){tau.push(t);break;}}
      if(z<=edge)tau.push(Infinity);}
    return times.map(t=>tau.filter(x=>x<=t).length/N);};
  const N=4000,L=run({N}),Lt=run({N,trapScale:1+2*(sz*axial/radial/tw.waistUm)**2}),Ll=run({N,lag:1});
  const series=[{quantity:'loss',indices:all(line),values:L,u:L.map((x,i)=>Math.hypot(Math.sqrt(x*(1-x)/N),Math.abs(Lt[i]-x),Math.abs(Ll[i]-x))),target:POPULATION},
    {quantity:'population',state:'ground-2-2',indices:[0],values:[1],u:[0],target:POPULATION}];
  write(id,expectation(id,'R9','R9 推出：离开 1/20 区域的时刻给出 L(t) 的台阶',line,{
    derivation:'1D Langevin equation along the axis: radiation pressure at the exact steady-state two-level rate with the Doppler and position-dependent detuning, the ground trap force and the excited state\'s extra force, momentum diffusion k²(1+Q+ξ)R; exit at z = √19 z_R (1/20 of the focus intensity, energy far above the trap); Monte Carlo from the prepared axial distribution.',
    conditions:[{what:'s',value:2*p.omega**2/gamma**2},{what:'δ at the bottom / Γ',value:p.delta/gamma},{what:'δ outside / Γ',value:deltaFree/gamma},{what:'max trap force over radiation pressure at the bottom',value:(0.65*U0/zR)/(p.k*rate(gamma,p.omega,p.delta)),limit:'≪ 1'},{what:'region edge z (µm)',value:edge}],
    series,info:{mandelQ:Q,zR,edge,sz,sk,N}}));
}

/** R9 (c). Heating-driven escape (contract §6 R9c): R2's far-detuned σ+ heating in a shallow σ+ tweezer. The energy
 * ε above the trap's bottom diffuses: per unit time the photon kicks add A(ε) = Σ_a D_a/m = E_rec(2 + Q)⟨R⟩_ε exactly
 * (zero-mean kicks; R at the local detuning δ_b + κV, averaged over the shell), with variance rate B(ε) =
 * Σ_a 2(D_a/m)⟨p_a²/m⟩_ε = (4/3)⟨KE⟩_ε A(ε) when the momenta are shared ergodically (⟨p_a²/m⟩ = (2/3)⟨KE⟩); the
 * microcanonical shell averages use the exact Gaussian-beam potential V = U₀(1 − |a|²): weight (ε − V)^{1/2}, kinetic
 * (ε − V)^{3/2} (the beam measure, lib/phys.mjs). The Fokker–Planck equation ∂P/∂t = −∂(AP) + ½∂²(BP) on [0, U₀]
 * (B → 0 at the bottom; absorbing at ε = U₀, the far level, where the atom is unbound) from the prepared lowered
 * isothermal P ∝ g(ε)(e^{−ε/k_BT} − e^{−U₀/k_BT}) gives the escape; an unbound atom is counted lost when it leaves the
 * 1/20 region: within µs radially (ρ = w₀√(ln 20/2) ≈ 1.5 µm), up to τ_max (the axial run from the centre to
 * √19 z_R at E = 0) along the axis -- L(t) is averaged over a delay uniform in [0, τ_max], u holds half that spread
 * and the ergodic closure's error: the motion in a Gaussian beam is nearly separable and the heating puts 7/10 of the
 * energy on z, so the escape is neither ergodic nor one-dimensional (contract: 10–20 %): 20 % of L. */
async function r9heat(){
  const id='R9_HEAT',c=await cycle(id,['probe']),{line,gamma,U0,kappa}=c,p=c.beams.probe,m=line.model.motion,mass=m.massKg,T=m.temperatureUK;
  const tw=beamOf(line,'tweezer'),zR=rayleigh(tw.waveNumberPerUm,tw.waistUm),Erec=recoilUK(p.k,mass),Q=mandelQ(gamma,p.omega,p.delta),times=grid(line);
  const U=U0/KT_PER_UK;  // depth in µK
  // Shell integrals over the beam: ∫d³r F(y), y = |a|², = (πw₀²z_R/2)∫ sec⁴θ G(cos²θ) dθ, G(Y) = ∫₀^Y F(y) dy/y.
  const shell=(eps,f)=>{const ymin=1-eps/U;const G=Y=>{if(Y<=ymin)return 0;let s=0;const a=Math.max(ymin,0),L=Math.log(Y/Math.max(a,1e-12));
      for(const [v,w] of gaussLegendre(24,0,Math.min(L,40)))s+=w*f(Y*Math.exp(-v));return s;};
    return integrate(th=>{const cs=Math.cos(th);return G(cs*cs)/(cs**4);},-Math.PI/2*0.9999,Math.PI/2*0.9999,{panels:16,n:16});};
  const R=V=>rate(gamma,p.omega,p.delta+kappa*V*KT_PER_UK);
  const n=160,de=U/n,eps=Array.from({length:n},(_,i)=>(i+0.5)*de);
  const stats=eps.map(e=>{const g=shell(e,y=>Math.sqrt(Math.max(0,e-U*(1-y)))),k=shell(e,y=>Math.max(0,e-U*(1-y))**1.5),r=shell(e,y=>Math.sqrt(Math.max(0,e-U*(1-y)))*R(U*(1-y)));
    return {g,KE:k/g,R:r/g};});
  const A=stats.map(s=>Erec*(2+Q)*s.R),B=stats.map((s,i)=>4/3*s.KE*A[i]);
  let P=stats.map((s,i)=>s.g*(Math.exp(-eps[i]/T)-Math.exp(-U/T)));const norm=P.reduce((a,b)=>a+b,0)*de;P=P.map(x=>x/norm);
  // Explicit flux-form steps: J_{i+1/2} = (A P)_{mid} − ½ ∂(B P)/∂ε; reflecting at 0, absorbing at U.
  const dtMax=0.4*de*de/Math.max(...B),Lfp=[0];let t=0;
  for(let i=1;i<times.length;i++){while(t<times[i]-1e-9){const h=Math.min(dtMax,times[i]-t),J=new Array(n+1).fill(0);
      for(let j=1;j<n;j++)J[j]=0.5*(A[j-1]*P[j-1]+A[j]*P[j])-0.5*(B[j]*P[j]-B[j-1]*P[j-1])/de;
      J[n]=A[n-1]*P[n-1]+0.5*B[n-1]*P[n-1]/(de/2);  // P = 0 at ε = U: the outgoing flux
      P=P.map((x,j)=>x-h*(J[j+1]-J[j])/de);t+=h;}
    Lfp.push(1-P.reduce((a,b)=>a+b,0)*de);}
  // The axial run from the centre to √19 z_R at E = 0: ∫dz/v, v = √(2U₀/(m(1 + z²/z_R²))) (µm/µs = m/s).
  const r=hbarOverMass(mass),edge=Math.sqrt(19)*zR,tauMax=integrate(z=>1/Math.sqrt(2*U0*r/(1+z*z/(zR*zR))),0,edge,{panels:8,n:16});
  const Lat=t=>{if(t<=0)return 0;const k=t/(times[1]-times[0]),j=Math.floor(k);return j>=times.length-1?Lfp.at(-1):Lfp[j]+(Lfp[j+1]-Lfp[j])*(k-j);};
  const Ldel=times.map(t=>{let s=0;const K=40;for(let q=0;q<K;q++)s+=Lat(t-tauMax*(q+0.5)/K);return s/K;});
  const series=[{quantity:'loss',indices:all(line),values:Ldel,u:times.map((t,i)=>Math.hypot(0.5*Math.abs(Lat(t)-Lat(t-tauMax)),0.2*Ldel[i])),target:POPULATION}];
  write(id,expectation(id,'R9','R9 加热逃逸：反冲加热越过阱深（能量 Fokker–Planck）',line,{
    derivation:'Energy Fokker–Planck with A = Σ D_a/m (exact) and B = (4/3)⟨KE⟩A (ergodic momenta) from the Gaussian beam\'s microcanonical shells, absorbing at the depth, from the lowered isothermal; L delayed by the run to the 1/20 region (0 to τ_max); u: half the delay spread and 20 % of L for the ergodic closure (contract 10–20 %).',
    conditions:[{what:'depth (µK)',value:U},{what:'initial depth over k_BT',value:U/T},{what:'heating 2E_rec R at the bottom (µK/µs)',value:A[0]},{what:'E_rec/U₀ (diffusion approximation)',value:Erec/U,limit:'≪ 1'},{what:'axial run to the region edge at E = 0 (µs)',value:tauMax}],
    series,info:{zR,edge,tauMax,mandelQ:Q,Lfp}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R4: Sisyphus against the 1D quantum references TW1/TW2.

/** R4. The 1D GMW quantum references (LONGREF_TW*.json, contract §3.5: reused, not recomputed) along the radial axis z
 * through the focus: T_z(t) and the F = 1 population at the scene's output times (every 10 µs, the references' rows).
 * Judged only where k_BT ≥ 4ħω (the method's validated regime; below it the semiclassical method is known to run 10–55 %
 * cold). The reference's own uncertainty: its step (dt 0.2 vs 0.1 µs: ≤ 1.2 % in T, its companions) and grid; the
 * 3D scene's other axes are assumed separable (their coupling through the tweezer's light shift ~1 % at 30 µK). */
async function r4(id){
  const line=read(`scenes/${id}.json`),name=id.slice(3),ref=JSON.parse(fs.readFileSync(`${(await import('./scenes.mjs')).TW_REFERENCES}/LONGREF_${name}.json`,'utf8'));
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
  const scene=read(`scenes/${id}.scene.json`),prep=preparedTrap(line,'ground-2-2'),run=await runningFactor(line,scene,'ground-2-2',line.model.motion.beams.filter(b=>b.id!=='tweezer').map(b=>b.id));
  const trap=prep.omegas.map(w=>w*Math.sqrt(run.factor));
  const series=[...['x','y','z'].map((a2,i)=>({quantity:'trapFrequency',axis:a2,value:kHz(trap[i]),u:Math.hypot(kHz(prep.u[i])*Math.sqrt(run.factor),kHz(trap[i])*run.u/2),target:{relative:0}})),{quantity:'T',axis:'z',regime:'classical',indices:idx,values:idx.map(i=>rows.get(Math.round(times[i])).T_uK),u:idx.map(i=>relT(times[i])*rows.get(Math.round(times[i])).T_uK),
      judged:idx.map(i=>rows.get(Math.round(times[i])).T_uK>=4*hw),target:TEMPERATURE},
    {quantity:'group',label:'F=1',states:line.model.states.filter(s=>s.startsWith('ground-1-')),indices:idx,values:idx.map(i=>rows.get(Math.round(times[i])).pF1),u:idx.map(()=>0.003),target:POPULATION},
    {quantity:'loss',indices:idx,values:idx.map(i=>Math.max(0,rows.get(Math.round(times[i])).loss)),u:idx.map(()=>0.002),target:POPULATION}];
  write(id,expectation(id,'R4',`R4 西西弗斯 ${name}：1D GMW 量子参照`,line,{
    derivation:'The 1D quantum (GMW) reference of the same scene along the radial axis z: T_z and p(F=1); T judged where k_BT ≥ 4ħω.',
    conditions:[{what:'ħω_r/k_B (µK) of the reference',value:hw},{what:'reference numerical error in T at the end (relative)',value:relNum(times.at(-1))},{what:'separability κk_BT/|δ_eff| at t = 0',value:kap*KT_PER_UK*30/dEff},{what:'running trap factor (ω² over the tweezer’s)',value:run.factor}],
    series,info:{reference:`LONGREF_${name}.json`,note:ref.note,wall_s:ref.wall_s,trapKHz:trap.map(kHz),referenceRadialKHz:ref.physics.omega_r_over_2pi_kHz}}));
}

// ---------------------------------------------------------------------------------------------------------------------
// R8: momentum diffusion of the held atom (characterization).

/** R8 (characterization only, contract §6). The exact local momentum diffusion of R5051's atom held at its trap's centre
 * (the quantum regression theorem on the master equation with every D2 state explicit, the force operator −∇H, the
 * spontaneous photons' recoil: kick/runs/qrt_R5051_n.jsonl, d⟨κ_a²⟩/dt = 2D_a, D in 1/(µm² µs)), with the force
 * correlation cut off by e^{−γτ}: γ → 0 holds the slow intra-state fluctuations a point particle's Ehrenfest force
 * leaves out (SPEC §8), which a moving atom's motion cuts off at about its trap frequency. The held atom (mass ×100,
 * scenes.mjs R8_MASS) does not cut them off; its T grows by ΔT_a = (ħ/Mm)·2D_a t/k_B, here at γ = 1/µs (the
 * cutoff kick.md compared the engines at) with u reaching the γ = 0.3/µs value -- reported, not judged; the γ = 0
 * values are in the info. */
async function r8(){
  const id='R8_R5051',line=read(`scenes/${id}.json`),times=grid(line),r=hbarOverMass(line.model.motion.massKg);
  const qrt=fs.readFileSync('C:/Users/eadri/AppData/Local/Temp/claude/c0815e347c54866c/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad/kick/runs/qrt_R5051_n.jsonl','utf8').split('\n').filter(x=>x.trim()).map(x=>JSON.parse(x));
  const centre=qrt.slice(1).find(q=>q.offset.every(v=>v===0)),D=g=>centre.cutoffs.find(c=>c.gamma===g).D;
  const series=['x','y','z'].map((a,i)=>({quantity:'Tincrease',axis:a,indices:all(line),values:times.map(t=>r*2*D(1)[i]*t/KT_PER_UK),u:times.map(t=>r*2*Math.abs(D(0.3)[i]-D(1)[i])*t/KT_PER_UK),target:TEMPERATURE}));
  write(id,{...expectation(id,'R8','R8 静止原子的动量扩散（R5051，QRT；只作表征）',line,{
    derivation:'Held atom (mass ×100): Var κ_a grows as 2D_a t with the QRT diffusion of the exact master equation at the trap centre; D at the correlation cutoff γ = 1/µs, u to γ = 0.3/µs. The γ → 0 limit (slow intra-state fluctuations) is beyond the semiclassical point particle (SPEC §8).',
    conditions:[{what:'photon rate at the centre (1/µs)',value:centre.photonRate},{what:'excited population',value:centre.excited}],
    series,info:{source:'kick/runs/qrt_R5051_n.jsonl (NS scratchpad)',cutoffs:centre.cutoffs.map(c=>({gamma:c.gamma,D:c.D,Dforce:c.Dforce})),Drecoil:centre.Drecoil}}),kind:'characterization'});
}

const RUN={R1_TWEEZER:()=>r1('R1_TWEEZER'),R1_RETRO:()=>r1('R1_RETRO'),R1_SIGMA_P:()=>r1('R1_SIGMA_P'),R1_SIGMA_M:()=>r1('R1_SIGMA_M'),R1_CROSSED:()=>r1('R1_CROSSED'),
  R2_RECOIL:r2,R3_DOPPLER:r3,R5_PUMP:r5,R6_RAMAN_CO:()=>r6('R6_RAMAN_CO'),R6_RAMAN_COUNTER:()=>r6('R6_RAMAN_COUNTER'),R9_PUSH:r9,R4_TW1:()=>r4('R4_TW1'),R4_TW2:()=>r4('R4_TW2'),R8_R5051:r8,R9_HEAT:r9heat};
const ids=process.argv.slice(2);
for(const [id,f] of Object.entries(RUN))if(!ids.length||ids.includes(id))await f();
