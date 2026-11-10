// The reference scenes (contract §6, R0–R9, and R10): each an ordinary app scene -- what an operator could set up -- whose
// physics has a clear, analytic or nearly analytic answer. `build()` returns the app scene (before the app's importer
// reads it back); a scene whose parameters depend on the atom (a detuning from a shifted line, a two-photon resonance)
// takes them from the app's own level chart or model, never from the solver. A scene's `mass` multiplies the request's
// mass (R10's held atom: a classical point in the same light); R0 and R8 edit the motion block in scenes.mjs: a mass
// ×10⁶ and a temperature of 1e-3 µK hold the atom at the focus (the static limit).
import {laserMHz,levelShifts,motionRequest,transitionMHz,zeroFieldMHz} from './app.mjs';
import {base,beam,reflection,tweezer} from './scene.mjs';
import {RAD_PER_MHZ,coupling,decayRates,index} from './phys.mjs';

/** Γ/2π of the D lines (MHz), as the app's atomic data gives them (physics.ts RB87.gammaHz). */
const GAMMA_D2=38.117e6/(2*Math.PI*1e6),GAMMA_D1=36.129e6/(2*Math.PI*1e6);
/** I_sat of the D2 cycling transition |2,±2⟩ ↔ |3′,±3′⟩ (mW/cm²), cε₀Γ²ħ²/(4|d|²) with the app's d and Γ: the s of a
 * σ± beam on it is I/I_sat. Steck, "Rubidium 87 D Line Data" (rev. 2.3.3), Table 7: 1.66933(35) mW/cm². */
const ISAT_CYCLING=1.66933;
/** The tweezer of the default scene (types.ts defaultTweezer): 852 nm, 4 mW, waist 1.2 µm. */
const DEFAULT_TWEEZER={nm:852,powerMW:4,waistUm:1.2};

/** R10's saturation parameter at the standing wave's antinode, s₀ = 2Ω₁²/Γ² (each beam s₀/4): the photon rate there is
 * Γs₀/2 = 0.38/µs, the heating at the node E_rec Γ s₀/2 ≈ 0.07 µK/µs. */
export const R10_S0=0.02;
/** R10's two settings. The app's scene: 87Rb at 2 µK, in the tweezer's radial ground state (ħω/k_B = 8.2 µK: the atom's
 * extent is the zero-point one, kσ_z = 0.15), 120 µs -- 43 photons at the antinode, 8 µK of heating along z at the
 * node; k_BT is below 4ħω, where the solver reports T without holding it to its target (it runs 96 trajectories). And
 * the held atom: the request's mass ×100 (the R8 device) makes it a classical point in the same light and the same
 * springs -- the extent σ² = k_BT/mω² does not depend on the mass, ħω ∝ m^{−1/2} does: at 5 µK k_BT = 6.1ħω on the
 * stiff axes and kσ_z = 0.17 --, the recoil energy and with it every heating rate 100 times smaller: 7.2 ms (2700
 * photons at the antinode, 5 µK of heating along z at the node). */
const R10_RB={temperatureUK:2,durationUs:120,samples:49};
export const R10_HELD={mass:100,temperatureUK:5,durationUs:7200,samples:49};
/** The detuning (MHz) a beam on `line` F → F′ needs to sit `deltaMHz` from the transition g → e of `scene` with the
 * lasers `shown` shifting it (the app's level chart): its offset from the zero-field transition. */
async function offsetFor(scene,line,Fg,Fe,groundId,excitedId,shown,deltaMHz){
  return (await transitionMHz(scene,groundId,excitedId,shown))-(await zeroFieldMHz(line,Fg,Fe))+deltaMHz;
}

export const SCENES={
  // R1 -- conservative motion: the trap's geometry, the far light's shift, the symplectic step, the initial ensemble.
  R1_TWEEZER:{ref:'R1',title:'R1 保守阱：默认光镊单独（852 nm，π 沿 B）',async build(){
    const s=await base({name:'R1 保守阱：光镊单独',initialStateId:'ground-2-0',temperatureUK:30,durationUs:1000,samples:101});
    await tweezer(s,{...DEFAULT_TWEEZER,along:'+x'});return s;}},
  R1_RETRO:{ref:'R1',title:'R1 保守阱：光镊加反射光（R = 1，φ = 0，同偏振：驻波）',async build(){
    const s=await base({name:'R1 保守阱：反射光驻波',initialStateId:'ground-2-0',temperatureUK:30,durationUs:1000,samples:101});
    await tweezer(s,{...DEFAULT_TWEEZER,along:'+x'});await reflection(s,'tweezer',{reflectivity:1,phaseDeg:0,quarterWave:false});return s;}},
  R1_SIGMA_P:{ref:'R1',title:'R1 保守阱：σ⁺ 光镊沿 B，原子在 |2,+2⟩（矢量光移）',async build(){
    const s=await base({name:'R1 保守阱：σ+ 光镊 |2,+2>',initialStateId:'ground-2-2',temperatureUK:30,durationUs:1000,samples:101});
    await tweezer(s,{...DEFAULT_TWEEZER,along:'+z',ellipticityDeg:45});return s;}},
  R1_SIGMA_M:{ref:'R1',title:'R1 保守阱：σ⁺ 光镊沿 B，原子在 |2,−2⟩（矢量光移）',async build(){
    const s=await base({name:'R1 保守阱：σ+ 光镊 |2,-2>',initialStateId:'ground-2--2',temperatureUK:30,durationUs:1000,samples:101});
    await tweezer(s,{...DEFAULT_TWEEZER,along:'+z',ellipticityDeg:45});return s;}},
  R1_CROSSED:{ref:'R1',title:'R1 保守阱：两束正交偏振的同频交叉远光（秩 2 的场）',async build(){
    const s=await base({name:'R1 保守阱：交叉远光',initialStateId:'ground-2-0',temperatureUK:30,durationUs:1000,samples:101});
    await tweezer(s,{...DEFAULT_TWEEZER,along:'+x',polarizationAngleDeg:90});
    await beam(s,{id:'trap2',name:'第二束远光',wavelengthNm:852,line:'D2',intensityMWcm2:2*DEFAULT_TWEEZER.powerMW/(Math.PI*DEFAULT_TWEEZER.waistUm**2)*1e8,waistUm:DEFAULT_TWEEZER.waistUm,along:'+y',polarizationAngleDeg:0});
    return s;}},
  // R2 -- the two-level atom |2,2⟩ ↔ |3′,3′⟩: photon rate, radiation pressure, recoil heating 7/5, 3/10, 3/10.
  R2_RECOIL:{ref:'R2',title:'R2 二能级辐射压与反冲加热（σ⁺ 光沿 B，δ = −100Γ，s = 200）',async build(){
    const s=await base({name:'R2 二能级反冲加热',initialStateId:'ground-2-2',temperatureUK:30,durationUs:1000,samples:101});
    await tweezer(s,{nm:852,powerMW:15.8,waistUm:1.2,along:'+z',ellipticityDeg:45});
    const offset=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],-100*GAMMA_D2);
    await beam(s,{id:'probe',name:'σ+ 探测光',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:200*ISAT_CYCLING,along:'+z',ellipticityDeg:45});
    return s;}},
  // R3 -- 1D Doppler cooling on the closed cycle with a σ+σ+ standing wave along the tweezer's axis (B).
  R3_DOPPLER:{ref:'R3',title:'R3 一维多普勒冷却（σ⁺σ⁺ 沿 ±z，s = 0.02，阱底 δ = −1.5Γ）',async build(){
    const s=await base({name:'R3 一维多普勒冷却',initialStateId:'ground-2-2',temperatureUK:30,durationUs:4000,samples:201});
    await tweezer(s,{nm:852,powerMW:32,waistUm:2,along:'+z',ellipticityDeg:-45});
    // The beams sit 1.5Γ red of the line at the trap's bottom: the tweezer's differential shift moves the line by κV with
    // the potential energy V (no magic wavelength exists in the model), and the transverse axes, never cooled, heat by
    // ~0.3ħΓ/k_B·(1 + x²)/(8|x|) per cooling time -- the cloud's detuning drifts toward the blue by Δx ≈ 0.2 per cooling
    // time and the cooled axis' own rise toward T_D adds as much: started at x = −3 the cloud stays red (x ≈ −2.7 → −2.2)
    // over 4 ms, while from x = −2 it runs away to the blue within 6 ms (the averaged Monte Carlo, expect.mjs R3). The run
    // is a transient: T_z climbs from 30 µK toward T_D with the friction holding it ~40 µK below free heating.
    const offset=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],-1.5*GAMMA_D2);
    await beam(s,{id:'cool-plus',name:'冷却光 +z',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:0.02*ISAT_CYCLING,along:'+z',ellipticityDeg:45});
    await beam(s,{id:'cool-minus',name:'冷却光 −z',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:0.02*ISAT_CYCLING,along:'-z',ellipticityDeg:-45});
    return s;}},
  // R5 -- optical pumping into the stretched dark state |2,2⟩: D1 σ+ on F=2 → F′=2, the repump D2 σ+ on F=1 → F′=2
  // (another line: no common excited state, so no coherent population trapping between F=1 and F=2).
  R5_PUMP:{ref:'R5',title:'R5 光泵到拉伸暗态（D1 σ⁺ F=2→F′=2，D2 σ⁺ 回泵 F=1→F′=2，s ≈ 0.02）',async build(){
    const s=await base({name:'R5 光泵到暗态',initialStateId:'ground-2-0',fieldG:1,temperatureUK:10,durationUs:200,samples:201});
    s.dynamics.initialState={kind:'unpolarized',manifold:'ground'};
    await tweezer(s,{nm:852,powerMW:1,waistUm:1.2,along:'+x'});
    const pump=await offsetFor(s,'D1',2,2,'ground-2-1','D1-2-2',['tweezer'],0),rep=await offsetFor(s,'D2',1,2,'ground-1-1','D2-2-2',['tweezer'],0);
    await beam(s,{id:'pump',name:'D1 σ+ 光泵',line:'D1',Fg:2,Fe:2,offsetMHz:pump,intensityMWcm2:0.06,along:'+z',ellipticityDeg:45});
    await beam(s,{id:'repump',name:'D2 σ+ 回泵',line:'D2',Fg:1,Fe:2,offsetMHz:rep,intensityMWcm2:0.03,along:'+z',ellipticityDeg:45});
    return s;}},
  // R6 -- the Λ system |1,0⟩ ↔ |2,0⟩ through D1 σ+σ+ Raman beams 20 GHz red of F′=2, two-photon resonance compensated.
  R6_RAMAN_CO:{ref:'R6',title:'R6 拉曼 Λ：同向 σ⁺σ⁺（Δ = −20 GHz，Ω_R ≈ 2π·0.25 MHz）',build:()=>raman('+z')},
  R6_RAMAN_COUNTER:{ref:'R6',title:'R6 拉曼 Λ：对向 σ⁺σ⁺（Δk = 2k，多普勒与 2ħk 反冲）',build:()=>raman('-z')},
  // R9 -- loss by a push out of the trap's region: a σ+ tweezer along z (7 mW, waist 2 µm: 0.48 mK, z_R = 14.7 µm)
  // and a σ+ beam along its axis 40Γ blue of the line at the bottom, saturating to ρ_ee = 0.05 there (0.043 outside):
  // its radiation pressure is five times the trap's largest axial force, the trap keeps 0.89 of its radial force
  // (1 − κρ_ee), and the line's shift across the trap (3.7Γ) and the Doppler shift at the exit (0.2Γ) change the rate
  // by a sixth and by a hundredth. Blue, because F′ = 3 is the highest level of the line: no other transition of the
  // ground states comes near the light (red by 40Γ it sits 4Γ from F = 2 → F′ = 2). About 200 photons carry the atom
  // out of the 1/20 region (64 µm on the axis) in 0.12 ms.
  R9_PUSH:{ref:'R9',title:'R9 推出：σ⁺ 辐射压为阱的最大轴向力的 5 倍，ρ_ee ≤ 0.05（δ = +40Γ，s = 711）',async build(){
    const s=await base({name:'R9 推出',initialStateId:'ground-2-2',temperatureUK:20,durationUs:200,samples:201});
    await tweezer(s,{nm:852,powerMW:7,waistUm:2,along:'+z',ellipticityDeg:45});
    const offset=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],40*GAMMA_D2);
    await beam(s,{id:'push',name:'σ+ 推出光',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:711*ISAT_CYCLING,along:'+z',ellipticityDeg:45});
    return s;}},
  // R9 (c) -- heating-driven escape without a mean force: a shallow σ+ tweezer (~200 µK) and two σ+ beams of one
  // intensity along its axis, one 100Γ red along +z and one 100Γ blue along −z: their radiation pressures, light
  // shifts and Doppler forces cancel, and the recoil heating (2E_rec per photon, ~70 µK/ms) carries the atoms over
  // the depth within the run. (One beam alone tilts the axial potential by kRz_R/U₀ = 0.32 and lowers its barrier
  // to a quarter: the atoms are then pushed out, not heated out.)
  R9_HEAT:{ref:'R9',title:'R9 加热逃逸：红蓝对向 σ⁺ 光（平均力为零），浅光镊中的反冲加热（能量 Fokker–Planck）',async build(){
    const s=await base({name:'R9 加热逃逸',initialStateId:'ground-2-2',temperatureUK:20,durationUs:4000,samples:201});
    await tweezer(s,{nm:852,powerMW:1.05,waistUm:1.2,along:'+z',ellipticityDeg:45});
    const line=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],0);
    await beam(s,{id:'red',name:'σ+ 加热光（红，+z）',line:'D2',Fg:2,Fe:3,offsetMHz:line-100*GAMMA_D2,intensityMWcm2:210*ISAT_CYCLING,along:'+z',ellipticityDeg:45});
    await beam(s,{id:'blue',name:'σ+ 加热光（蓝，−z）',line:'D2',Fg:2,Fe:3,offsetMHz:line+100*GAMMA_D2,intensityMWcm2:210*ISAT_CYCLING,along:'-z',ellipticityDeg:-45});
    return s;}},
  // R10 -- the absorption kick where beams interfere: the momentum diffusion of the closed cycle in a weak resonant
  // σ+σ+ standing wave along z ∥ B, the atom held within kσ ≈ 0.15 of a chosen phase of the wave by a tight tweezer:
  // the app's scene (87Rb, in the trap's ground state), and the same with the request's mass ×100 (a classical point).
  ...Object.fromEntries([['ANTINODE',0,'阱心在波腹（θ = 0°）'],['QUARTER',45,'阱心在波腹与波节正中（θ = 45°）'],['NODE',90,'阱心在波节（θ = 90°，中心处不散射）']].flatMap(([at,theta,where])=>[
    [`R10_${at}`,{ref:'R10',title:`R10 驻波中的动量扩散：${where}，⁸⁷Rb，2 µK`,build:()=>standing(theta,R10_RB)}],
    [`R10_HELD_${at}`,{ref:'R10',title:`R10 驻波中的动量扩散：${where}，质量 ×${R10_HELD.mass}，5 µK`,mass:R10_HELD.mass,build:()=>standing(theta,R10_HELD)}]])),
};

/** R10: an 852 nm tweezer along +x, π on z ∥ B (10 mW, waist 1 µm: 3 mK deep, 170 kHz along y and z, so that the cold
 * atom's extent along z is its zero-point one, kσ_z ≈ 0.15; |2,2⟩ and |3′,3′⟩ stay its eigenstates), and a σ+σ+ pair
 * along ±z on the closed cycle |2,2⟩ ↔ |3′,3′⟩, resonant at the trap's bottom (the level chart's line with the tweezer
 * shifting it), of one intensity: the standing wave Ω(z) = Ω₁ cos(kz + θ). θ is the wave's phase at the trap's centre
 * (0: antinode, 90°: node): the app gives each beam its initial phase at the atom (初相位 φ₀), +θ on the +z beam and
 * −θ on the −z beam, less the relative phase the two beams' couplings of the cycle have at φ₀ = 0 (their polarization
 * vectors' convention), read from the request the app builds for it. */
async function standing(thetaDeg,{temperatureUK,durationUs,samples}){
  const {finish}=await import('./app.mjs');
  const make=async back=>{
    const s=await base({name:`R10 驻波中的动量扩散（θ = ${thetaDeg}°）`,initialStateId:'ground-2-2',temperatureUK,durationUs,samples});
    await tweezer(s,{nm:852,powerMW:10,waistUm:1,along:'+x'});
    const offset=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],0);
    await beam(s,{id:'wave-plus',name:'驻波 +z',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:R10_S0/4*ISAT_CYCLING,along:'+z',ellipticityDeg:45,phaseDeg:thetaDeg});
    await beam(s,{id:'wave-minus',name:'驻波 −z',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:R10_S0/4*ISAT_CYCLING,along:'-z',ellipticityDeg:-45,phaseDeg:back});
    return s;
  };
  const wrap=d=>d-360*Math.round(d/360);
  let back=wrap(-thetaDeg);
  for(let pass=0;pass<2;pass++){
    const line=await motionRequest('R10',await finish(await make(back)),{device:'cpu'});
    back=wrap(back+2*(standingPhase(line).thetaDeg-thetaDeg));
  }
  return make(back);
}
/** The standing wave of R10 from the request: the two beams' couplings of the cycle at the focus, X₊ and X₋ (rad/µs),
 * make the amplitude X₊e^{ikz} + X₋e^{−ikz} = e^{i(a₊+a₋)/2}[(|X₊| + |X₋|) cos(kz + θ) + i(|X₊| − |X₋|) sin(kz + θ)],
 * θ = (a₊ − a₋)/2: its phase at the focus (degrees), the Rabi frequency at its antinode Ω₁ = 2(|X₊| + |X₋|), the
 * imbalance (|X₊| − |X₋|)/(|X₊| + |X₋|) (0: no travelling part), and the beams' wave number. */
export function standingPhase(line){
  const e=index(line,'D2-3-3'),g=index(line,'ground-2-2'),p=coupling(line,'wave-plus',e,g),m=coupling(line,'wave-minus',e,g);
  const ap=Math.hypot(p.re,p.im),am=Math.hypot(m.re,m.im),k=line.model.motion.beams.find(b=>b.id==='wave-plus').waveNumberPerUm;
  return {thetaDeg:(Math.atan2(p.im,p.re)-Math.atan2(m.im,m.re))/2*180/Math.PI,omega1:2*(ap+am),imbalance:(ap-am)/(ap+am),k};
}

/** R6: beam A on F=1 → F′=2 and beam B on F=2 → F′=2 of D1, both 20 GHz red, σ+ (A along +z; B along `along`), of one
 * intensity; B's frequency moved by the two-photon light shift so that |1,0⟩ → |2,0⟩ is resonant (from the request's
 * own couplings: the second-order shifts of the two states by both beams, every D1/D2 excited state). The intensity
 * sets Ω_R = 2π × 0.25 MHz (two passes: the coupling is linear in the field, Ω_R in the intensity). */
async function raman(along){
  const {finish}=await import('./app.mjs');
  const make=async(I,shiftMHz)=>{
    const s=await base({name:`R6 拉曼 Λ（${along==='+z'?'同向':'对向'}）`,initialStateId:'ground-1-0',temperatureUK:30,durationUs:20,samples:401});
    await tweezer(s,{nm:852,powerMW:18,waistUm:3,along:'+z',polarizationAngleDeg:0});
    await beam(s,{id:'ramanA',name:'拉曼光 A（F=1）',line:'D1',Fg:1,Fe:2,offsetMHz:-20000,intensityMWcm2:I,along:'+z',ellipticityDeg:45});
    await beam(s,{id:'ramanB',name:'拉曼光 B（F=2）',line:'D1',Fg:2,Fe:2,offsetMHz:-20000+shiftMHz,intensityMWcm2:I,along,ellipticityDeg:along==='+z'?45:-45});
    return s;
  };
  let I=1000,shift=0;
  for(let pass=0;pass<3;pass++){
    const s=await finish(await make(I,shift)),line=await motionRequest('R6',s,{device:'cpu'}),r=await ramanNumbers(line,s);
    I*=2*Math.PI*0.25/r.omegaR;shift+=r.twoPhotonMHz;
  }
  return make(I,shift);
}

/** The Λ |1,0⟩ ↔ |2,0⟩ of R6 in second order in the Raman beams (Ω/2Δ ≈ 2e-3): Ω_R = 2 Σ_e X^A_{e,1} conj(X^B_{e,2})/Δ_e
 * (rad/µs; every D1 and D2 excited state, Δ_e the mean of the two single-photon detunings), the Raman beams' light
 * shifts ΔE_g = Σ_{l,e} |X^l_{eg}|²/δ^l_{eg}, and the two-photon detuning (MHz). The detunings are between the levels
 * dressed by the tweezer -- the level chart's all-order energies with the tweezer shown: its 12 MHz shift of the ground
 * and its shift of 5P move Δ_e by ~10⁻³ of 20 GHz, Ω_R by as much, and the two-photon resonance by the tweezer's
 * differential hyperfine shift (kHz; its spread over the cloud, ~5 % of that, is ≪ Ω_R). */
export async function ramanNumbers(line,scene){
  const g1=index(line,'ground-1-0'),g2=index(line,'ground-2-0'),states=line.model.states,N=32;
  const levels=await levelShifts(scene,['tweezer']),nu={ramanA:await laserMHz(scene,'ramanA'),ramanB:await laserMHz(scene,'ramanB')};
  const delta=(id,e,g)=>RAD_PER_MHZ*(nu[id]-(levels[states[e]].absoluteMHz-levels[states[g]].absoluteMHz)),abs2=z=>z.re*z.re+z.im*z.im;
  let omega={re:0,im:0};const shift=[0,0];
  for(let e=8;e<N;e++){
    const a1=coupling(line,'ramanA',e,g1),b2=coupling(line,'ramanB',e,g2),D=(delta('ramanA',e,g1)+delta('ramanB',e,g2))/2;
    omega.re+=2*(a1.re*b2.re+a1.im*b2.im)/D;omega.im+=2*(a1.im*b2.re-a1.re*b2.im)/D;
    for(const [k,g] of [[0,g1],[1,g2]])for(const id of ['ramanA','ramanB'])shift[k]+=abs2(coupling(line,id,e,g))/delta(id,e,g);
  }
  // Two-photon detuning of A − B against |2,0⟩ − |1,0⟩, both dressed and shifted (MHz).
  const two=(nu.ramanA-nu.ramanB)-((levels['ground-2-0'].absoluteMHz+shift[1]/RAD_PER_MHZ)-(levels['ground-1-0'].absoluteMHz+shift[0]/RAD_PER_MHZ));
  return {omegaR:Math.hypot(omega.re,omega.im),twoPhotonMHz:two,shiftsMHz:shift.map(x=>x/RAD_PER_MHZ)};
}
