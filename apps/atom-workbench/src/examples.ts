// 示例…: ready-made scenes. Loading one replaces the whole scene; the caller asks first when the
// current scene holds edits. Three scenes of one Rb87 atom moving in 3D in an 852 nm tweezer on the
// D2 line, each starting in |F=2, m=2⟩ with its motion thermal at EXAMPLE_TEMPERATURE_UK along every
// axis, solved over EXAMPLE_DURATION_US. The app starts with STARTUP_EXAMPLE (scene-file.ts startupScene).
// A retro-reflected beam is the app's reflection of its source (reflection.ts), never a laser of its own.
import {defaultScene,defaultTweezer,newLaser} from './types.js';
import type {AxisName,Direction,FrequencyExpr,Laser,Scene} from './types.js';
import {RB87,resolveSceneFrequencies,setPolarizationPreset} from './physics.js';
import {normalizeLaserIntensity} from './laser-intensity.js';
import {addReflection} from './reflection.js';
import {oppositeDirection} from './direction.js';

/** Scene 1: the tweezer's power (mW) and waist (μm); the cooling beams' detuning from F=2 → F′=3 (in units of
 * the D2 Γ, as in free space) and each source beam's intensity (mW/cm², its reflection the same); the repump's
 * intensity (mW/cm²), on resonance with F=1 → F′=2. */
export const PGC3D={tweezerMW:3.5,tweezerWaistUm:1,coolingDetuningGamma:-6,coolingMWcm2:3,repumpMWcm2:1};
/** Scenes 2 and 3, the lab's: the tweezer's power (mW) for its 1.5 mK depth (the default tweezer -- 852 nm, waist
 * 1.2 μm -- makes 828 μK at 4 mW, the depth linear in the power); the probe's power (mW), waist (μm), offset from
 * the zero-field F=2 → F′=3 line (MHz, red) and, in scene 2, its tilt from +y toward the tweezer's axis (°). */
export const LAB={tweezerMW:7.25,probeMW:1,probeWaistUm:1000,probeOffsetMHz:-20,probeTiltDeg:4};
/** Scene 3: the field (G) along +z, the 1D PGC axis. */
export const PGC1D={fieldG:3};
/** The ground F=2 Larmor frequency per gauss, MHz/G per Δm = 1: g_F μ_B/h with g_F = (g_J + 3 g_I)/4 ≈ 1/2
 * (F=2, I=3/2, J=1/2), the slope of the app's own ground Zeeman Hamiltonian (physics.ts) at low field. */
export const F2_LARMOR_MHZ_PER_G=RB87.muBOverH*(RB87.groundGJ+3*RB87.gI)/4;

export const EXAMPLES:readonly {key:string;label:string}[]=[
  {key:'pgc3d',label:'光镊 · 3D PGC'},
  {key:'probe',label:'光镊 + 探测光 + 回泵光 · 零场'},
  {key:'pgc1d',label:`光镊 + 探测光 + 1D PGC · ${PGC1D.fieldG} G`},
];
/** The scene the app starts with: the 3D PGC, the first example (its startup solve the lightest of the three). */
export const STARTUP_EXAMPLE='pgc3d';
/** Every example's initial motion temperature (μK), duration (μs, written in ms) and samples. */
export const EXAMPLE_TEMPERATURE_UK=100,EXAMPLE_DURATION_US=1e3,EXAMPLE_SAMPLES=1001;

const axis=(name:AxisName):Direction=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
/** An offset from the zero-field D2 transition F → F′. */
const detuned=(offsetMHz:number,Fg:1|2,Fe:number):FrequencyExpr=>({base:{kind:'transition',line:'D2',Fg,Fe},offsetMHz});

/** Scene 1, 3D PGC. Free space would do for the molasses, but motion holds the atom in a designated tweezer
 * (tweezer.ts tweezerRefusal: 运动需要光镊), so the atom sits in an 852 nm tweezer along z, polarized along x (π to
 * the quantization axis: no field, its reference axis along x). Three σ+σ− D2 cooling pairs below F=2 → F′=3: a
 * circular beam along each of +x, +y and +z, retro-reflected by a mirror through a λ/4 plate passed twice, which
 * keeps the handedness about the beam's own k (the MOT's retro-reflection); the z pair, along the quadrupole axis,
 * of the other helicity than the x and y pairs. A pair's two circular beams make a polarization turning along the
 * pair at constant intensity, so the pairs need no choice of linear axes. One repump beam along +x on F=1 → F′=2.
 * Every beam's phase at the focus is its source's plus its mirror's (zero): the six beams' fixed interference
 * there, a frozen snapshot of an experiment's drifting phases, sets how well each axis cools. Eight lasers, the
 * scene's most. */
function pgc3d():Scene{
  const scene=defaultScene(),plane=scene.lasers[0]!;
  Object.assign(scene.fields[0]!,{magnitudeG:0,direction:axis('+x')});
  const gamma=RB87.D2.gammaHz/(2*Math.PI*1e6);
  const beam=(id:string,name:string,frequency:FrequencyExpr,intensityMWcm2:number,along:AxisName,ellipticityDeg:number):Laser=>
    ({...structuredClone(plane),id,name,frequency,intensityMWcm2,direction:axis(along),polarizationAngleDeg:0,ellipticityDeg,phaseDeg:0});
  const cooling=(name:'x'|'y'|'z',helicity:1|-1)=>beam(`cool-${name}`,`冷却光 +${name}`,detuned(PGC3D.coolingDetuningGamma*gamma,2,3),PGC3D.coolingMWcm2,`+${name}`,45*helicity);
  const sources=[cooling('x',1),cooling('y',1),cooling('z',-1)];
  scene.lasers=[{...defaultTweezer(),direction:axis('+z'),powerMW:PGC3D.tweezerMW,waistUm:PGC3D.tweezerWaistUm},
    ...sources,beam('repump','回泵光 +x',detuned(0,1,2),PGC3D.repumpMWcm2,'+x',0)];
  resolveSceneFrequencies(scene);
  for(const source of sources)addReflection(scene,source).reflection!.quarterWave=true;
  return scene;
}

/** Scenes 2 and 3's lab, as the operator makes it: the probe (a laser added with ＋, types.ts newLaser) at LAB's
 * power, waist and offset; the default tweezer (along +x, linear along z) at LAB.tweezerMW; a repump added with ＋
 * (50 μW in a 1 mm waist, along +x, linear along z) on F=1 → F′=2; the default bias field (+z) at `fieldG`. */
function lab(fieldG:number):{scene:Scene;probe:Laser}{
  const scene=defaultScene(),probe=newLaser('probe','探测光','D2'),repump=newLaser('repump','回泵光','D2');
  scene.fields[0]!.magnitudeG=fieldG;
  Object.assign(probe,{frequency:detuned(LAB.probeOffsetMHz,2,3),powerMW:LAB.probeMW,waistUm:LAB.probeWaistUm,powerUnit:'mW'});
  repump.frequency=detuned(0,1,2);
  scene.lasers=[probe,{...defaultTweezer(),powerMW:LAB.tweezerMW},repump];
  return {scene,probe};
}

/** Scene 2, the lab's probe at zero field: the probe π -- linear along the field's reference axis z, the
 * quantization axis at zero field (ψ = 90° in its frame) -- across the tweezer along +y, tilted LAB.probeTiltDeg
 * toward its axis (the lab's geometry), retro-reflected through the mirror's λ/4 plate at its default 45°: the
 * reflection comes back linear across z (σ+ and σ− half each), with the probe a lin⊥lin pair. */
function probeAtZeroField():Scene{
  const {scene,probe}=lab(0);
  Object.assign(probe,{direction:{reference:{kind:'axis',axis:'+y'},thetaDeg:LAB.probeTiltDeg,phiDeg:0},polarizationAngleDeg:90,ellipticityDeg:0});
  resolveSceneFrequencies(scene);
  addReflection(scene,probe).reflection!.quarterWave=true;
  return scene;
}

/** Scene 3, a 1D PGC in a field: the probe σ+ along the field B (+z, PGC1D.fieldG), and a PGC beam of the probe's
 * power and waist σ− along −z (circular of the probe's handedness about its own k). The tweezer, linear along z, is
 * π to B.
 * Why this frequency: the σ+σ− pair's sum is linear light whose axis turns about B along z. In a frame turning
 * about B at the F=2 Larmor frequency ω_L = g_F μ_B B/ħ (0.70 MHz/G per Δm = 1, 2.10 MHz at 3 G) the ground Zeeman
 * term vanishes, and the σ± light read there shifts by ∓ω_L. With ω_probe − ω_PGC = 2ω_L both beams land on one
 * frequency: a static σ+σ− molasses at zero field, cooling toward v = 0 (at equal frequencies the field moves that
 * point to kv = ω_L, 1.6 m/s at 3 G). In the lab frame: the two-photon transition |2,m⟩ → |2,m+2⟩ -- a σ+ photon
 * absorbed from the probe, one emitted into the σ− PGC beam -- is resonant for an atom at rest,
 * ω_probe − ω_PGC = (E(m+2) − E(m))/ħ = 2ω_L. So the PGC beam is 2 × 0.70 MHz/G × 3 G ≈ 4.2 MHz red of the probe,
 * −24.2 MHz from the zero-field F=2 → F′=3 line. Left out of the picture: the excited state's g_F′ = 2/3, a residual
 * 0.7 MHz per m′ in the turning frame against the 22 MHz detuning, and F=1 (g_F = −1/2), which the repump empties.
 * The tweezer's light, π, is unchanged by the turning frame. */
function pgc1dInField():Scene{
  const {scene,probe}=lab(PGC1D.fieldG);
  setPolarizationPreset(probe,scene.fields,'sigma-plus');
  const raman=2*F2_LARMOR_MHZ_PER_G*PGC1D.fieldG;
  // The opposite direction's frame keeps the probe's x and reverses its y (direction.ts): χ = +45° there is σ− about B.
  const pgc:Laser={...structuredClone(probe),id:'pgc',name:'PGC 光',direction:oppositeDirection(probe.direction),ellipticityDeg:45,
    frequency:detuned(LAB.probeOffsetMHz-raman,2,3)};
  scene.lasers.splice(1,0,pgc);
  return scene;
}

const SCENES:Record<string,()=>Scene>={pgc3d,probe:probeAtZeroField,pgc1d:pgc1dInField};
/** The example `key` (EXAMPLES): every one on D2, opened on the dynamics page. */
export function exampleScene(key:string):Scene{
  const build=SCENES[key],label=EXAMPLES.find(example=>example.key===key)?.label;
  if(!build||label===undefined)throw new Error(`没有示例「${key}」。`);
  const scene=build();scene.name=label;
  Object.assign(scene.dynamics,{initialStateId:'ground-2-2',durationUs:EXAMPLE_DURATION_US,durationUnit:'ms',samples:EXAMPLE_SAMPLES});
  scene.dynamics.motion.temperatureUK=EXAMPLE_TEMPERATURE_UK;
  resolveSceneFrequencies(scene);scene.lasers.forEach(normalizeLaserIntensity);
  return scene;
}
