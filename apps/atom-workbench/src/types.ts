export type Vec3 = [number, number, number];
export interface AxisAngles { thetaDeg:number; phiDeg:number; thetaResidualDeg?:number; phiResidualDeg?:number }
export type SpectralLine = 'D1' | 'D2';
export type DetuningUnit = 'Hz' | 'kHz' | 'MHz' | 'GHz' | 'THz';
export interface ZeroFieldTransition { fromManifold?:'ground'|'D1'; line: SpectralLine; Fg: 1 | 2; Fe: number }
export interface LaserFrequencyAnchor {
  /** A fixed fine-structure centroid plus an offset defines the physical beam.
   * Changing the displayed transition reference must not re-round this value. */
  line: SpectralLine; offsetMHz: number;
  /** Exact representation fingerprint. Direct edits to legacy frequency fields
   * invalidate this anchor rather than creating competing sources of truth. */
  viewKey: string;
}
export interface LaserIntensityAnchor {
  /** Canonical local intensity; switching P/w and I only changes its view. */
  intensityMWcm2:number; viewKey:string;
}
export interface LaserDirectionAnchor extends AxisAngles {
  /** Only exact geometry shortcuts create this fixed lab direction. Explicit
   * theta/phi entry invalidates it; later B edits never rotate the laser. */
  viewKey:string;
}
export interface MagneticField {
  id: string; name: string; enabled: boolean;
  magnitudeG: number; thetaDeg: number; phiDeg: number;
}
export interface Laser {
  id: string; name: string; enabled: boolean; line: SpectralLine;
  /** Spectroscopy/diagram origin only. Omitted follows the actual initial state;
   * this does not restrict drive couplings or alter the density initial state. */
  referenceStateId?: string;
  /** Display only: 'far' shows this laser's second-order light shift in the
   * levels chart. It does not change the dynamics, where every enabled laser
   * of nonzero intensity is an explicit drive of the master equation. */
  regime?: 'near' | 'far';
  referenceProcess?: 'absorption' | 'emission';
  /** Detuning from referenceTransition, in MHz, independent of the display unit.
   * Authoritative in transition mode; wavelength mode derives it from lambda.
   * Legacy scenes without a reference use F=2 → F'=3 (D2) or F'=2 (D1). */
  detuningMHz: number; intensityMWcm2: number;
  referenceTransition?: ZeroFieldTransition;
  frequencyMode?: 'transition' | 'wavelength';
  /** Vacuum wavelength. Authoritative only in wavelength mode. */
  wavelengthNm?: number;
  detuningUnit?: DetuningUnit;
  frequencyAnchor?: LaserFrequencyAnchor;
  intensityMode?: 'intensity' | 'power';
  intensityAnchor?: LaserIntensityAnchor;
  /** Gaussian beam power and 1/e² intensity radius at the atom's axial plane.
   * Power mode uses the on-axis intensity 2 P / (pi w²). */
  powerMW?: number; waistUm?: number;
  thetaDeg: number; phiDeg: number; polarizationAngleDeg: number;
  directionAnchor?:LaserDirectionAnchor;
  /** Jones vector cos(chi) u + i sin(chi) v in the documented transverse basis. */
  ellipticityDeg: number;
  /** Optical phase at the atom; relevant when coherent beams share a carrier. */
  phaseDeg?: number;
}
export interface Scene {
  version: 1; name: string; fields: MagneticField[]; lasers: Laser[];
  initialF: 1 | 2; initialM: number;
  /** Optional fields allow first-demo scene files to remain readable. */
  marks?: Record<string,string>;
  references?: ReferenceLevel[];
  dynamics?: DynamicsSettings;
}
export interface ReferenceLevel {
  id: string; name: string;
  /** Annotation energy = anchor physical state's energy + offsetMHz. Not a dynamical state. */
  anchorId: string; offsetMHz: number;
  /** Follow the instantaneous anchor, or hold its zero-field energy fixed. */
  anchorMode?: 'follow' | 'zero-field';
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
  laserId: string; initialStateId: string; monitoredStateIds: string[];
  /** When present it defines the initial density; otherwise the initial
   * state is {kind:'state', id: initialStateId}. Scene loading folds a
   * 'state' spec into initialStateId, so this holds only mixed kinds. */
  initialState?: InitialStateSpec;
  monitorMode?:'auto'|'manual';
  monitoredGroupIds?:string[];
  durationUs: number; samples: number; autoRun: boolean;
  /** Absolute population error target in probability units, not percent. */
  populationTolerance: number;
  /** Display unit only; physical duration and request identity remain in μs. */
  durationUnit?:DurationUnit;
  /** @deprecated Legacy scene-file metadata only. No UI control, numerical
   * mode, cache key or native request uses this field. */
  quality?: 'preview' | 'verified';
}
export type DurationUnit='ns'|'μs'|'ms'|'s';
/** The one resolution of the dynamics initial state (see InitialStateSpec). */
export function resolveInitialState(settings:Partial<DynamicsSettings>|undefined,scene:Scene):InitialStateSpec {
  if(settings?.initialState)return settings.initialState;
  if(scene.dynamics?.initialState&&settings?.initialStateId===undefined)return scene.dynamics.initialState;
  return {kind:'state',id:settings?.initialStateId??scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`};
}
/** Canonical (key-ordered) form for request identity. */
export function canonicalInitialState(spec:InitialStateSpec):InitialStateSpec {
  if(spec.kind==='weights')return {kind:'weights',weights:Object.fromEntries(Object.entries(spec.weights).sort(([a],[b])=>a<b?-1:a>b?1:0))};
  return spec.kind==='state'?{kind:'state',id:spec.id}:spec.kind==='thermal'?{kind:'thermal',temperatureK:spec.temperatureK}:{kind:'unpolarized',manifold:spec.manifold};
}
export const DEFAULT_DYNAMICS_SAMPLES=1000;
export const DEFAULT_TWEEZER_WAVELENGTH_NM=852;
export const DEFAULT_POPULATION_TOLERANCE=.005;
export const MAX_POPULATION_TOLERANCE=.02;
/** Missing legacy settings use the default; supplied invalid values fail. */
export function resolvePopulationTolerance(value:unknown):number {
  if(value===undefined)return DEFAULT_POPULATION_TOLERANCE;
  if(typeof value!=='number'||!Number.isFinite(value)||value<=0||value>MAX_POPULATION_TOLERANCE)throw new Error('populationTolerance 必须满足 0 < ε ≤ 0.02');
  return value;
}
export function defaultDynamics(): DynamicsSettings {
  return {laserId:'probe',initialStateId:'ground-2-0',monitoredStateIds:['ground-2-0','ground-1--1','ground-1-0','ground-1-1'],
    durationUs:20,samples:DEFAULT_DYNAMICS_SAMPLES,populationTolerance:DEFAULT_POPULATION_TOLERANCE,autoRun:true,monitorMode:'auto',monitoredGroupIds:[]};
}
export interface Level {
  id: string; manifold: 'ground' | 'D1' | 'D2'; F: number; m: number;
  /** MHz relative to the centre of gravity of the fine-structure manifold. */
  energyMHz: number; shiftMHz: number;
  /** Relative to the zero-external-field state; null means no static spectrum is justified. */
  totalShiftMHz?:number|null;
  starkShiftMHz?:number|null;
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
  sources?:string[];
}
export interface Polarization {
  minus: number; pi: number; plus: number;
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
    version: 1, name: `探测光 + ${DEFAULT_TWEEZER_WAVELENGTH_NM} nm 光镊`, initialF: 2, initialM: 0,
    marks:{},references:[],dynamics:defaultDynamics(),
    fields: [{id:'bias',name:'偏置磁场',enabled:true,magnitudeG:3,thetaDeg:0,phiDeg:0}],
    lasers: [{id:'probe',name:'探测光',enabled:true,regime:'near',line:'D2',detuningMHz:0,intensityMWcm2:1,
      referenceTransition:{line:'D2',Fg:2,Fe:2},frequencyMode:'transition',detuningUnit:'MHz',
      thetaDeg:0,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0}, defaultTweezer()],
  };
}

/** A tweezer is an ordinary enabled laser, with its physical optical frequency.
 * No separate trap approximation, hidden beam exclusion, or effective shift. */
export function defaultTweezer(): Laser {
  return {id:'tweezer',name:'光镊',enabled:true,regime:'far',line:'D2',frequencyMode:'wavelength',wavelengthNm:DEFAULT_TWEEZER_WAVELENGTH_NM,
    referenceTransition:{line:'D2',Fg:2,Fe:3},detuningMHz:0,detuningUnit:'THz',
    intensityMode:'power',powerMW:4,waistUm:1.2,intensityMWcm2:2*4/(Math.PI*1.2**2)*1e8,
    thetaDeg:90,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0};
}
