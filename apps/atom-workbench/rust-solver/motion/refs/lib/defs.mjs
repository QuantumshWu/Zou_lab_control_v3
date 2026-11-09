// The reference scenes (contract §6, R0–R9): each an ordinary app scene -- what an operator could set up -- whose
// physics has a clear, analytic or nearly analytic answer. `build()` returns the app scene (before the app's importer
// reads it back); a scene whose parameters depend on the atom (a detuning from a shifted line, a two-photon resonance)
// takes them from the app's own level chart or model, never from the solver. `motion` (R0, R8 only) edits the motion
// block of the request: a mass ×10⁶ and a temperature of 1e-3 µK hold the atom at the focus (the static limit).
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
    await tweezer(s,{nm:852,powerMW:7.9,waistUm:1.2,along:'+z',ellipticityDeg:45});
    const offset=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],-100*GAMMA_D2);
    await beam(s,{id:'probe',name:'σ+ 探测光',line:'D2',Fg:2,Fe:3,offsetMHz:offset,intensityMWcm2:200*ISAT_CYCLING,along:'+z',ellipticityDeg:45});
    return s;}},
  // R3 -- 1D Doppler cooling on the closed cycle with a σ+σ+ standing wave along the tweezer's axis (B).
  R3_DOPPLER:{ref:'R3',title:'R3 一维多普勒冷却（σ⁺σ⁺ 沿 ±z，s = 0.02，δ ≈ −Γ/2）',async build(){
    const s=await base({name:'R3 一维多普勒冷却',initialStateId:'ground-2-2',temperatureUK:30,durationUs:4000,samples:201});
    await tweezer(s,{nm:852,powerMW:32,waistUm:2,along:'+z',ellipticityDeg:-45});
    // The beams sit −Γ/2 from the line at the cloud's mean potential energy (3/2)k_B·60 µK above the bottom: the
    // differential shift κV moves the line by κ = δ_LS(0)/U₀ per unit potential energy (expect.mjs R3).
    const sh=await levelShifts(s,['tweezer']);
    const U0=-sh['ground-2-2'].shiftMHz,kappa=(sh['D2-3-3'].shiftMHz-sh['ground-2-2'].shiftMHz)/U0,meanV=1.5*60*1.380649e-23/6.62607015e-34*1e-12;
    const offset=await offsetFor(s,'D2',2,3,'ground-2-2','D2-3-3',['tweezer'],-GAMMA_D2/2-kappa*meanV);
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
  // R9 -- loss by a push out of the trap's region.
  R9_PUSH:{ref:'R9',title:'R9 推出：σ⁺ 辐射压远大于阱力（s = 100，沿光镊轴）',async build(){
    const s=await base({name:'R9 推出',initialStateId:'ground-2-2',temperatureUK:30,durationUs:40,samples:401});
    await tweezer(s,{...DEFAULT_TWEEZER,along:'+z',ellipticityDeg:45});
    const bottom=await transitionMHz(s,'ground-2-2','D2-3-3',['tweezer']),free=await transitionMHz(s,'ground-2-2','D2-3-3',[]);
    await beam(s,{id:'push',name:'σ+ 推出光',line:'D2',Fg:2,Fe:3,offsetMHz:(bottom+free)/2-(await zeroFieldMHz('D2',2,3)),intensityMWcm2:100*ISAT_CYCLING,along:'+z',ellipticityDeg:45});
    return s;}},
};

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
    const s=await finish(await make(I,shift)),line=await motionRequest('R6',s,{device:'cpu'}),r=ramanNumbers(line,s);
    I*=2*Math.PI*0.25/r.omegaR;shift+=r.twoPhotonMHz;
  }
  return make(I,shift);
}

/** The Λ |1,0⟩ ↔ |2,0⟩ of R6 in second order (Ω/2Δ ≪ 1): Ω_R = 2 Σ_e X^A_{e,1} conj(X^B_{e,2})/Δ_e (rad/µs, Δ_e the
 * mean of the two single-photon detunings, every D1 and D2 excited state), and the two-photon detuning (MHz) from the
 * frequencies and the light shifts ΔE_g = Σ_{l,e} |X^l_{eg}|²/δ^l_{eg} of the two states by every beam at the focus (the
 * tweezer's differential hyperfine shift included; its variation over the cloud, ~5 % of a few kHz, is ≪ Ω_R). */
export function ramanNumbers(line,scene){
  const g1=index(line,'ground-1-0'),g2=index(line,'ground-2-0'),m=line.model.motion,N=32;
  const E=m.hamiltonian.real.filter((_,k)=>k%(N+1)===0),nu=line.model.frequencyCoordinates.frequenciesMHz;
  const freq=id=>RAD_PER_MHZ*m.beams.find(b=>b.id===id).frequencyIndex.reduce((s,n,k)=>s+n*nu[k],0);
  const dA=freq('ramanA'),dB=freq('ramanB'),abs2=z=>z.re*z.re+z.im*z.im;
  let omega={re:0,im:0};const shift=[0,0];
  for(let e=8;e<N;e++){
    const a1=coupling(line,'ramanA',e,g1),b2=coupling(line,'ramanB',e,g2);
    const deltaA1=dA-(E[e]-E[g1]),deltaB2=dB-(E[e]-E[g2]),D=(deltaA1+deltaB2)/2;
    omega.re+=2*(a1.re*b2.re+a1.im*b2.im)/D;omega.im+=2*(a1.im*b2.re-a1.re*b2.im)/D;
    for(const [k,g] of [[0,g1],[1,g2]])for(const b of m.beams)shift[k]+=abs2(coupling(line,b.id,e,g))/(freq(b.id)-(E[e]-E[g]));
  }
  // Two-photon detuning of A − B against |2,0⟩ − |1,0⟩ with both shifted (rad/µs → MHz).
  const two=((dA-dB)-((E[g2]+shift[1])-(E[g1]+shift[0])))/RAD_PER_MHZ;
  return {omegaR:Math.hypot(omega.re,omega.im),twoPhotonMHz:two,shifts:shift};
}
