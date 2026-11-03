export type Vec3 = [number, number, number];
export interface AxisAngles { thetaDeg:number; phiDeg:number; thetaResidualDeg?:number; phiResidualDeg?:number }
export type SpectralLine = 'D1' | 'D2';
export type DetuningUnit = 'Hz' | 'kHz' | 'MHz' | 'GHz' | 'THz';
export interface ZeroFieldTransition { fromManifold?:'ground'|'D1'; line: SpectralLine; Fg: 1 | 2; Fe: number }
/** What a defined frequency is relative to (ui-design.md §1.2): nothing (an absolute
 * frequency), a zero-field transition, a reference frequency or another laser (an offset lock). */
export type FrequencyBaseSpec={kind:'absolute'}|({kind:'transition'}&ZeroFieldTransition)|{kind:'reference';id:string}|{kind:'laser';id:string};
/** The one form of a defined frequency: a base and the offset from it (MHz, canonical), or
 * a vacuum wavelength exactly as written. */
export type FrequencyExpr={base:FrequencyBaseSpec;offsetMHz:number}|{wavelengthNm:number};
/** The unit a definition's value is written in (a wavelength unit only for a wavelength). */
export type FrequencyDisplayUnit=DetuningUnit|'nm'|'μm';
/** A frequency with no intensity, polarization or direction ("冷却光锁频点"): it can be
 * selected (the chart shows detunings from it) and be a base of other frequencies. */
export interface ReferenceFrequency {
  id: string; name: string; frequency: FrequencyExpr; frequencyUnit: FrequencyDisplayUnit;
  /** Derived: the resolved frequency (resolveSceneFrequencies). */
  frequencyAnchor?: LaserFrequencyAnchor;
}
export interface LaserFrequencyAnchor {
  /** A fixed fine-structure centroid plus an offset defines the physical beam.
   * Changing the displayed transition reference must not re-round this value. */
  line: SpectralLine; offsetMHz: number;
  /** The expression (and followed carrier) it was resolved from: an anchor whose fingerprint no
   * longer matches is resolved again. */
  viewKey: string;
}
export interface LaserIntensityAnchor {
  /** Canonical local intensity; switching P/w and I only changes its view. */
  intensityMWcm2:number; viewKey:string;
}
/** A coordinate axis, as a scene saves it. */
export type AxisName='+x'|'-x'|'+y'|'-y'|'+z'|'-z';
/** What a direction is measured from (direction.ts): angles (with the exact residuals a saved
 * fixed axis may carry), a coordinate axis, or a link to a magnetic field's reference axis. */
export type ReferenceAxis=({kind:'angles'}&AxisAngles)|{kind:'axis';axis:AxisName}|{kind:'field';id:string};
/** A field's or a laser's direction: its reference axis and the offset (θ′, φ′) from it, read in
 * the frame the minimal rotation from z to the reference axis carries. */
export interface Direction { reference: ReferenceAxis; thetaDeg: number; phiDeg: number }
export interface MagneticField {
  id: string; name: string; enabled: boolean;
  magnitudeG: number; direction: Direction;
}
export interface Laser {
  /** `line` is derived from the frequency (the nearer D line of its carrier). */
  id: string; name: string; enabled: boolean; line: SpectralLine;
  /** The laser's frequency (ui-design.md §1.2) and the unit its value is written in. */
  frequency: FrequencyExpr; frequencyUnit: FrequencyDisplayUnit;
  /** Spectroscopy/diagram origin only. Omitted follows the actual initial state;
   * this does not restrict drive couplings or alter the density initial state. */
  referenceStateId?: string;
  /** Display only ("显示光移"): the levels chart's background includes this laser's
   * static light shift (spectroscopy.ts). It never changes the dynamics, where every
   * enabled laser of nonzero intensity is an explicit drive of the master equation,
   * and it is no part of a request or of the dynamics cache key. */
  showLightShift: boolean;
  /** With a reference level as referenceStateId: whether its photon is absorbed or emitted. */
  referenceProcess?: 'absorption' | 'emission';
  /** Derived (resolveSceneFrequencies): the resolved carrier, kept bit for bit while its
   * expression stands; absent, it is resolved from the expression. */
  frequencyAnchor?: LaserFrequencyAnchor;
  /** How the local intensity is given (laser-intensity.ts): directly, by power and waist, or by
   * power and the numerical aperture the beam fills (its waist then follows from the wavelength). */
  intensityMode: IntensityMode;
  /** The local intensity: the input in intensity mode, else the value the other inputs give. */
  intensityMWcm2: number;
  /** Derived: the exact local intensity for the inputs it was taken at; absent, it is computed. */
  intensityAnchor?: LaserIntensityAnchor;
  /** Gaussian beam power and 1/e² intensity radius at the atom's axial plane: inputs of the
   * power-based modes, kept across mode switches once given. Power mode uses 2 P / (pi w²). */
  powerMW?: number; waistUm?: number;
  /** NA mode: the numerical aperture the beam fills, 0 < NA ≤ 1 (the atoms are in vacuum). */
  numericalAperture?: number;
  /** The units power, waist and local intensity are written or read in (laser-intensity.ts):
   * an input's number keeps its digits when its unit changes; a computed one converts. */
  powerUnit: PowerUnit; waistUnit: WaistUnit; intensityUnit: IntensityUnit;
  /** Propagation direction k; the polarization angle ψ is measured from the x of its frame. */
  direction: Direction; polarizationAngleDeg: number;
  /** Jones vector cos(chi) u + i sin(chi) v in the documented transverse basis. */
  ellipticityDeg: number;
  /** Optical phase at the atom; relevant when coherent beams share a carrier. */
  phaseDeg: number;
  /** Present: this laser is the retro-reflection of another (reflection.ts). Its mirror is its own;
   * every physical field above is derived from the source wherever the scene is resolved
   * (resolveSceneFrequencies), so it always follows the source. */
  reflection?: LaserReflection;
}
/** A mirror far away, perpendicular to the beam of laser `of`, that sends it straight back. */
export interface LaserReflection {
  of: string;
  /** The reflected power over the incident, 0 to 1. */
  reflectivity: number;
  /** The reflected beam's optical phase at the atom relative to the source's: the mirror's distance. */
  phaseDeg: number;
  /** A double pass through a quarter-wave plate (the MOT retro-reflection): the handedness about the
   * beam's own k stays, so σ⁺ comes back σ⁺. */
  quarterWave: boolean;
}
/** The most lasers a scene holds; a reflection is one of them. */
export const MAXIMUM_LASERS=8;
/** The one scene schema (scene-file.ts validates files strictly against it). */
export const SCENE_VERSION=3;
export interface Scene {
  version: typeof SCENE_VERSION; name: string; fields: MagneticField[]; lasers: Laser[];
  /** The laser the operator designates as the tweezer (设为光镊, tweezer.ts): the trap motion holds
   * the atom in. One at most; absent, none is. */
  tweezerId?: string;
  marks: Record<string,string>;
  references: ReferenceLevel[];
  referenceFrequencies: ReferenceFrequency[];
  /** Display preferences, when any were chosen: never physics, never part of a request or a cache key. */
  display?: {readouts?: Record<string,{unit?:string;base?:string}>};
  dynamics: DynamicsSettings;
}
export interface ReferenceLevel {
  id: string; name: string;
  /** Annotation energy = anchor physical state's energy + offsetMHz. Not a dynamical state. */
  anchorId: string; offsetMHz: number;
  /** The unit the operator writes the offset in (its digits stay when it changes). */
  offsetUnit: DetuningUnit;
  /** Follow the instantaneous anchor, or hold its zero-field energy fixed. */
  anchorMode: 'follow' | 'zero-field';
  /** A linked photon (ui-design.md §1.6): E = E(anchor) ± hν(source) + h·offset, following
   * the laser or reference frequency when it changes. */
  photon?: {source:{kind:'laser'|'reference';id:string};direction:'absorb'|'emit'};
}
/** Initial density of the dynamics, diagonal in the master-model basis (the
 * eigenbasis of the static H0). State ids are those 32 eigenstates.
 * - state: one eigenstate;
 * - unpolarized: equal weights over the 8 ground states, or over F=1 / F=2;
 * - thermal: Boltzmann weights over the 8 ground eigenstates of the static H0
 *   at temperatureK > 0 (a large temperature tends to the uniform mixture);
 * - weights: nonnegative weights (normalized to one) over any of the 32 states. */
export type InitialStateSpec =
  | {kind:'state';id:string}
  | {kind:'unpolarized';manifold:'ground'|'F1'|'F2'}
  | {kind:'thermal';temperatureK:number}
  | {kind:'weights';weights:Record<string,number>};
export interface DynamicsSettings {
  initialStateId: string; monitoredStateIds: string[];
  /** When present it defines the initial density; otherwise the initial
   * state is {kind:'state', id: initialStateId}. Scene loading folds a
   * 'state' spec into initialStateId, so this holds only mixed kinds. */
  initialState?: InitialStateSpec;
  /** The time T of the long-time population p̄(T), in us: 1e5 (100 ms) or 1e6 (1 s). */
  meanHorizonUs: number;
  monitorMode:'auto'|'manual';
  monitoredGroupIds:string[];
  durationUs: number; samples: number;
  /** The unit the operator writes the duration in: its digits stay when it changes, so a
   * new unit is a new duration. The physical duration and request identity are in μs. */
  durationUnit:DurationUnit;
  /** A thermal initial state's temperature unit (as durationUnit); present exactly when it is thermal. */
  temperatureUnit?:TemperatureUnit;
  /** Motion (ui-design.md §6.20), on by default: the trajectory is the joint evolution of the
   * internal state and the atom's motion in the tweezer; off, the internal state's alone (the atom
   * at rest at the focus). Kept while off, as an inactive mode's inputs are. */
  motion:MotionSettings;
}
/** A lab coordinate axis: motion's results come per axis, x, y and z. */
export type MotionAxis='x'|'y'|'z';
/** Where a motion solve runs: an NVIDIA GPU, or the CPU. The engine falls back to the CPU, saying
 * why, when the GPU cannot run it. */
export type MotionDevice='gpu'|'cpu';
/** A single atom in the tweezer (Scene.tweezerId) moving in 3D (semiclassical). It starts in the
 * internal initial state, its motion thermal at temperatureUK along each axis. The trajectory is the
 * joint evolution of the internal state and this motion, solved on `device`. */
export interface MotionSettings {
  enabled:boolean;
  /** The initial temperature (μK, canonical) and the unit the operator writes it in (as durationUnit). */
  temperatureUK:number; temperatureUnit:TemperatureUnit;
  device:MotionDevice;
}
export const DEFAULT_MOTION_TEMPERATURE_UK=30;
/** The motion settings 运动 starts with. */
export function defaultMotion():MotionSettings{return {enabled:true,temperatureUK:DEFAULT_MOTION_TEMPERATURE_UK,temperatureUnit:'μK',device:'gpu'};}
export type DurationUnit='ns'|'μs'|'ms'|'s';
export type TemperatureUnit='K'|'mK'|'μK';
export type IntensityMode='intensity'|'power'|'na';
export type PowerUnit='W'|'mW'|'μW'|'nW';
export type WaistUnit='mm'|'μm'|'nm';
export type IntensityUnit='kW/cm²'|'W/cm²'|'mW/cm²'|'μW/cm²';
/** The one resolution of the dynamics initial state (see InitialStateSpec). */
export function resolveInitialState(settings:Pick<DynamicsSettings,'initialState'|'initialStateId'>):InitialStateSpec {
  return settings.initialState??{kind:'state',id:settings.initialStateId};
}
/** Canonical (key-ordered) form for request identity. */
export function canonicalInitialState(spec:InitialStateSpec):InitialStateSpec {
  if(spec.kind==='weights')return {kind:'weights',weights:Object.fromEntries(Object.entries(spec.weights).sort(([a],[b])=>a<b?-1:a>b?1:0))};
  return spec.kind==='state'?{kind:'state',id:spec.id}:spec.kind==='thermal'?{kind:'thermal',temperatureK:spec.temperatureK}:{kind:'unpolarized',manifold:spec.manifold};
}
/** p̄(T): the population at t = T with only the fast oscillations around T (beats, optical
 * micromotion) averaged out -- the zero-frequency Fourier component of the state at T, not an
 * average over 0–T. T is 100 ms or 1 s; there is no t → ∞ mean. */
export const MEAN_HORIZONS_US:readonly number[]=[1e5,1e6];
export const DEFAULT_MEAN_HORIZON_US=1e6;
/** The longest trajectory (10 s, in us) and its most samples: what the solver core solves. */
export const MAXIMUM_DURATION_US=1e7;
export const MAXIMUM_SAMPLES=2000;
export const DEFAULT_DYNAMICS_SAMPLES=1000;
export const DEFAULT_TWEEZER_WAVELENGTH_NM=852;
/** The population accuracy of every solve: an absolute error of 0.01 in probability
 * units (1%), sent with each request. */
export const POPULATION_TOLERANCE=.01;
/** The default trajectory: motion on, 100 μs -- the probe's optical pumping and several radial
 * periods of the default tweezer (about 13 μs each); its motion solves in about 0.4 s on the CPU. */
export const DEFAULT_DURATION_US=100;
export function defaultDynamics(): DynamicsSettings {
  return {initialStateId:'ground-2-0',monitoredStateIds:['ground-2-0','ground-1--1','ground-1-0','ground-1-1'],
    durationUs:DEFAULT_DURATION_US,durationUnit:'μs',samples:DEFAULT_DYNAMICS_SAMPLES,meanHorizonUs:DEFAULT_MEAN_HORIZON_US,monitorMode:'auto',monitoredGroupIds:[],motion:defaultMotion()};
}
export interface Level {
  id: string; manifold: 'ground' | 'D1' | 'D2'; F: number; m: number;
  /** MHz relative to the centre of gravity of the fine-structure manifold. */
  energyMHz: number; shiftMHz: number;
  /** Relative to the zero-external-field state; null means no static spectrum is justified. */
  totalShiftMHz?:number|null;
  starkShiftMHz?:number|null;
  /** Numerical error bounds of the total shift, MHz: the Floquet partitioning (layers,
   * energy self-consistency) and the left-out couplings through states outside the
   * model. Not the atomic data's uncertainty. Null where totalShiftMHz is. */
  shiftBound?:{floquetMHz:number;outsideMHz:number}|null;
  /** Column vector in the solver's 32-state magnetic-field eigenbasis. */
  backgroundState?:{real:number[];imag:number[]};
  mixed?:boolean;
}
export interface SpectroscopyMetadata {
  status:'bare'|'dispersive'|'unresolved';
  backgroundLaserIds:string[];
  excludedLaserId?:string;
  modelScope:string;
  nearResonantLaserIds:string[];
  requiresDensityProjection:boolean;
  warnings?:string[];
  maximumMixingRatio?:number;
  /** The static background's partitioning: Floquet layers, clusters, cluster solves and
   * the largest bound of each kind over the 32 levels (MHz). */
  numerics?:{floquetLayers:number;clusters:number;solves:number;floquetBoundMHz:number;outsideBoundMHz:number};
  sources?:string[];
}
export interface ComplexAmplitude {re:number;im:number}
/** A laser's polarization on the quantization axis (the total magnetic field; at zero field the
 * first field's reference axis): the weights and complex amplitudes of q = −1, 0, +1, and its
 * lab geometry -- k, and the Jones axes u, v. */
export interface Polarization {
  minus: number; pi: number; plus: number;
  amplitudes: [ComplexAmplitude,ComplexAmplitude,ComplexAmplitude];
  k: Vec3; u: Vec3; v: Vec3; angleToBDeg: number;
}
export interface Transition {
  id: string; laserId: string; line: SpectralLine; groundId: string; excitedId: string;
  Fg: number; mg: number; Fe: number; me: number; q: -1 | 0 | 1;
  /** Transition frequency relative to the beam's selected zero-field reference. */
  offsetMHz: number; detuningMHz: number;
  /** Relative electric-dipole strength, cycling D2 sigma transition = 1, before polarization. */
  strength: number; weight: number; rabiMHz: number;
  /** A dressed-state dipole can contain several spherical components. */
  mixedPolarization?:boolean;
  /** Display labels are not exact m eigenvalues in this background. */
  mLabelApproximate?:boolean;
}
export interface Analysis {
  /** The total field, and the quantization axis: along it, or at zero field the first field's reference axis. */
  fieldVector: Vec3; fieldMagnitudeG: number; fieldAxis:AxisAngles;
  ground: Level[]; D1: Level[]; D2: Level[];
  polarizations: Record<string, Polarization>;
  transitions: Transition[];
  clockMHz: number; clockShiftKHz: number; clockSensitivityHzPerMG: number;
  computeMs: number;
  spectroscopy?:SpectroscopyMetadata;
}

export function defaultScene(): Scene {
  return {
    version:SCENE_VERSION,name:`探测光 + ${DEFAULT_TWEEZER_WAVELENGTH_NM} nm 光镊`,tweezerId:'tweezer',
    marks:{},references:[],referenceFrequencies:[],dynamics:defaultDynamics(),
    fields: [{id:'bias',name:'偏置磁场',enabled:true,magnitudeG:3,direction:{reference:{kind:'axis',axis:'+z'},thetaDeg:0,phiDeg:0}}],
    // The probe runs along the bias field: its reference axis is linked to the field's.
    lasers: [{id:'probe',name:'探测光',enabled:true,showLightShift:false,line:'D2',
      frequency:{base:{kind:'transition',line:'D2',Fg:2,Fe:2},offsetMHz:0},frequencyUnit:'MHz',
      intensityMode:'intensity',intensityMWcm2:1,powerUnit:'mW',waistUnit:'μm',intensityUnit:'mW/cm²',
      direction:{reference:{kind:'field',id:'bias'},thetaDeg:0,phiDeg:0},polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0}, defaultTweezer()],
  };
}

/** A new laser (＋): a cooling or probe beam given by power and waist -- a Gaussian beam focused at
 * the atom, its waist on the millimetre scale of such beams: 50 μW in a 1 mm waist,
 * I = 2P/(πw₀²) ≈ 3.2 mW/cm² at the atom, about twice the saturation intensity of the D2 cycling
 * transition. On the `line`'s F=2 → F′=3 (F″=2 on D1), along +x. */
export const NEW_LASER_POWER_MW=.05,NEW_LASER_WAIST_UM=1000;
export function newLaser(id:string,name:string,line:SpectralLine): Laser {
  return {id,name,enabled:true,showLightShift:false,line,frequency:{base:{kind:'transition',line,Fg:2,Fe:line==='D1'?2:3},offsetMHz:0},frequencyUnit:'MHz',
    intensityMode:'power',powerMW:NEW_LASER_POWER_MW,waistUm:NEW_LASER_WAIST_UM,intensityMWcm2:2*NEW_LASER_POWER_MW/(Math.PI*NEW_LASER_WAIST_UM**2)*1e8,
    powerUnit:'μW',waistUnit:'mm',intensityUnit:'mW/cm²',direction:{reference:{kind:'axis',axis:'+x'},thetaDeg:0,phiDeg:0},polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0};
}

/** A tweezer is an ordinary enabled laser, with its physical optical frequency.
 * No separate trap approximation, hidden beam exclusion, or effective shift. */
export function defaultTweezer(): Laser {
  return {id:'tweezer',name:'光镊',enabled:true,showLightShift:true,line:'D2',frequency:{wavelengthNm:DEFAULT_TWEEZER_WAVELENGTH_NM},frequencyUnit:'nm',
    intensityMode:'power',powerMW:4,waistUm:1.2,intensityMWcm2:2*4/(Math.PI*1.2**2)*1e8,powerUnit:'mW',waistUnit:'μm',intensityUnit:'kW/cm²',
    direction:{reference:{kind:'axis',axis:'+x'},thetaDeg:0,phiDeg:0},polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0};
}
