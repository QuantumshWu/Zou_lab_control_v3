import { accurateSum, atomicSystem, complexPolarization, composeMagneticField, laserCarrierOffsetMHz, laserFrequencyDifferenceMHz, laserIntensityMWcm2, RB87, sinCosDegrees, transitionReferenceMHz } from './physics.js';
import type { DynamicsSettings, Level, Scene, SpectralLine, Vec3 } from './types.js';
import { ATOMIC_MODEL_SCHEMA } from './solver-protocol.js';
import {missingManifoldLightShift} from './spectroscopy.js';

/** Serialized 32-state Lindblad model consumed by the Rust solver.
 * All matrices are complex row-major arrays. H has units rad / us; each
 * collapse matrix has units 1 / sqrt(us). No rate equation or adiabatic
 * elimination of the represented 32 states is used. Additional far-detuned
 * electronic states contribute the documented second-order DC light shifts.
 */
export interface MasterMatrix { real: number[]; imag: number[] }
/** H(t) = H0 + sum [cos(omega*t) Hcos + sin(omega*t) Hsin].
 * Both coefficient matrices are Hermitian; omega is in rad/us. */
export interface MasterDrive { omega: number; cos: MasterMatrix; sin: MasterMatrix; laserIds: string[] }
export interface DecayChannel {
  excitedId: string; groundId: string; ratePerUs: number; branchingFraction: number;
}
/** Signed lowering dipoles in q = -1, 0, +1 order. Entries have units
 * 1/sqrt(us), normalized at referenceAngularFrequency (rad/us), not SI dipole
 * units. The native bath assigns the emitted-frequency factor coherently,
 * before squaring amplitudes that lead to the same emitted photon. */
export interface RadiationModel {
  referenceAngularFrequency: number; carrierAngularFrequency: number;
  dipoles: [MasterMatrix, MasterMatrix, MasterMatrix];
}
export interface MasterModel {
  modelSchema: typeof ATOMIC_MODEL_SCHEMA; radiation: RadiationModel;
  dimension: 32; states: Level[];
  /** Free-atom resolved-D-line operators for bare-decay checks only. Atomic
   * protocol v2 replaces them with the bath built from radiation.dipoles. */
  hReal: number[]; hImag: number[]; collapse: MasterMatrix[]; initialIndex: number;
  fieldVector: Vec3; fieldMagnitudeG: number; laserCount: number;
  carriersMHz: Record<SpectralLine, number>; decayChannels: DecayChannel[];
  timeDependent: boolean; drives: MasterDrive[]; rotatingCarrierOffsetMHz: number; assumptions: string[];
}

const N = 32, N2 = N * N, TAU = 2 * Math.PI;
const HBAR = 1.0545718176461565e-34, EPS0 = 8.8541878128e-12, C = 299792458;
const LINES: SpectralLine[] = ['D1', 'D2'];
const STARTS: Record<SpectralLine, number> = { D1: 8, D2: 16 };
const zeros = () => Array<number>(N2).fill(0);
type AtomicSystem = ReturnType<typeof atomicSystem>;
interface FieldModel {
  systems: Record<SpectralLine, AtomicSystem>;
  states: Level[]; collapse: MasterMatrix[]; decayChannels: DecayChannel[];
  dipoles: RadiationModel['dipoles'];
}
// Cache both D lines together: physics.atomicSystem itself keeps only one line.
// This is independent of optical power, detuning, polarization and initial state.
const fieldCache = new Map<number, FieldModel>();

function finite(value: number, description: string): number {
  if (!Number.isFinite(value)) throw new Error(`${description} must be finite.`);
  return value;
}

export function fieldModel(magnitude: number): FieldModel {
  const cached = fieldCache.get(magnitude);
  if (cached) return cached;
  const systems = { D1: atomicSystem('D1', magnitude), D2: atomicSystem('D2', magnitude) };
  const publicLevel = ({ id, manifold, F, m, energyMHz, shiftMHz }: Level): Level =>
    ({ id, manifold, F, m, energyMHz, shiftMHz });
  const states = [...systems.D1.ground, ...systems.D1.excited, ...systems.D2.excited].map(publicLevel);
  const collapse: MasterMatrix[] = [], decayChannels: DecayChannel[] = [];
  const dipoles = [-1, 0, 1].map((): MasterMatrix => ({ real: zeros(), imag: zeros() })) as RadiationModel['dipoles'];
  for (const line of LINES) {
    const system = systems[line], data = RB87[line], start = STARTS[line];
    const lineOperators = [-1, 0, 1].map((): MasterMatrix => ({ real: zeros(), imag: zeros() }));
    for (let e = 0; e < system.excited.length; e++) {
      const excited = system.excited[e], channels: Omit<DecayChannel, 'branchingFraction'>[] = [];
      let totalRate = 0;
      for (let g = 0; g < system.ground.length; g++) {
        const ground = system.ground[g], q = excited.m - ground.m;
        if (q !== -1 && q !== 0 && q !== 1) continue;
        // At B=0 this is an exact symmetry rule, not a numerical cutoff.
        if (magnitude === 0 && Math.abs(excited.F - ground.F) > 1) continue;
        // atomicSystem uses the D2 stretched dipole as its common unit.
        // Undo the D1 / D2 radial ratio here: Gamma already contains it.
        const amplitude = system.dipoles[e * 8 + g] * RB87.D2.dipole / data.dipole;
        const frequencyHz = data.frequencyHz + 1e6 * (excited.energyMHz - ground.energyMHz);
        if (!(frequencyHz > 0)) throw new Error('Magnetic field exceeds the positive optical-frequency domain of this D-line model.');
        const frequencyFactor = (frequencyHz / data.frequencyHz) ** 1.5;
        const value = Math.sqrt(data.gammaHz * 1e-6) * amplitude * frequencyFactor;
        // One operator per spherical component, NOT one jump per transition:
        // C_q rho C_q^dagger must preserve radiative coherence transfer.
        lineOperators[q + 1].real[g * N + start + e] = value;
        // Remove the bare transition's omega^(3/2) weighting. Both fine
        // manifolds share the same signed q operator at the D2 reference
        // frequency, so indistinguishable Raman paths can interfere.
        dipoles[q + 1]!.real[g * N + start + e] += value / (frequencyHz / RB87.D2.frequencyHz) ** 1.5;
        const ratePerUs = value * value;
        totalRate += ratePerUs;
        if (ratePerUs > 0) channels.push({ excitedId: excited.id, groundId: ground.id, ratePerUs });
      }
      for (const channel of channels) decayChannels.push({ ...channel, branchingFraction: channel.ratePerUs / totalRate });
    }
    collapse.push(...lineOperators);
  }
  const result = { systems, states, collapse, decayChannels, dipoles };
  if (fieldCache.size >= 16) fieldCache.delete(fieldCache.keys().next().value!);
  fieldCache.set(magnitude, result);
  return result;
}

/** Build the full optical Bloch model in the field-dressed basis.
 * The labels F denote zero-field adiabatic labels at nonzero field. The optical
 * RWA and an isotropic free-space Markov vacuum are explicit assumptions.
 * A common three-dipole bath is resolved by emitted photon frequency in Rust;
 * the free-atom collapse metadata are not the driven-atom bath.
 * Every beam couples both fine-structure manifolds. Reference-transition
 * selection only parametrizes the actual optical frequency. A common optical
 * rotating frame leaves all unequal-carrier beats explicitly time dependent.
 */
export function buildMasterModel(scene: Scene, settings?: Partial<DynamicsSettings>): MasterModel {
  const {fieldVector,fieldAxis,fieldMagnitudeG}=composeMagneticField(scene.fields);
  finite(fieldMagnitudeG, 'Total magnetic field');
  const model = fieldModel(fieldMagnitudeG), hReal = zeros(), hImag = zeros();
  const lasers = scene.lasers.filter(laser => {
    if (!laser.enabled) return false;
    const intensity = finite(laserIntensityMWcm2(laser), 'Laser intensity');
    if (intensity < 0) throw new Error('Laser intensity cannot be negative.');
    return intensity > 0;
  });
  // Offsets from the same fine-structure centroid avoid subtracting two
  // ~384 THz numbers when a near-resonant detuning is specified in Hz.
  const offsets = lasers.map(laser => finite(laserCarrierOffsetMHz(laser, 'D2'), 'Laser frequency'));
  const rotatingCarrierOffsetMHz = offsets[0] ?? transitionReferenceMHz('D2');
  const centroidDifferenceMHz = (RB87.D1.frequencyHz - RB87.D2.frequencyHz) * 1e-6;
  const carriersMHz: Record<SpectralLine, number> = {
    D1: rotatingCarrierOffsetMHz - centroidDifferenceMHz - transitionReferenceMHz('D1'),
    D2: rotatingCarrierOffsetMHz - transitionReferenceMHz('D2'),
  };
  const drives: MasterDrive[] = [];
  // Beam amplitudes interfere before squaring. Collect every represented term
  // per matrix entry: a weak drive between two cancelling strong beams must
  // survive regardless of scene order, for DC and both harmonic quadratures.
  const opticalTerms=new Map<number[],Map<number,number[]>>();
  const addOptical=(target:number[],index:number,value:number)=>{
    if(value===0)return;
    let entries=opticalTerms.get(target);if(!entries){entries=new Map();opticalTerms.set(target,entries);}
    let terms=entries.get(index);if(!terms){terms=[];entries.set(index,terms);}terms.push(value);
  };
  const groundF2EnergyMHz = 0.75 * RB87.groundA;
  model.states.forEach((state, index) => {
    const rotationMHz = state.manifold === 'ground' ? 0 : rotatingCarrierOffsetMHz - (state.manifold === 'D1' ? centroidDifferenceMHz : 0);
    hReal[index * N + index] = TAU * (state.energyMHz - groundF2EnergyMHz - rotationMHz);
  });
  for (const [laserIndex, laser] of lasers.entries()) {
    if (laser.line !== 'D1' && laser.line !== 'D2') throw new Error(`Unknown D line: ${laser.line}`);
    for (const [label, value] of Object.entries({ theta: laser.thetaDeg, phi: laser.phiDeg,
      polarizationAngle: laser.polarizationAngleDeg, ellipticity: laser.ellipticityDeg, phase: laser.phaseDeg ?? 0 })) finite(value, `Laser ${label}`);
    const polarization = complexPolarization(laser, fieldVector,fieldAxis);
    const phase = sinCosDegrees(laser.phaseDeg ?? 0), cosPhase = phase.cos, sinPhase = phase.sin;
    const electric = Math.sqrt(2 * laserIntensityMWcm2(laser) * 10 / (EPS0 * C));
    // D2 radial unit agrees with atomicSystem.dipoles, including for D1.
    const factor = -0.5 * RB87.D2.dipole * electric / HBAR * 1e-6;
    const beatMHz = laserFrequencyDifferenceMHz(laser, lasers[0]);
    let drive: MasterDrive | undefined;
    if (beatMHz !== 0) {
      const omega = TAU * Math.abs(beatMHz);
      drive = drives.find(item => item.omega === omega);
      if (!drive) {
        drive = { omega, cos: { real: zeros(), imag: zeros() }, sin: { real: zeros(), imag: zeros() }, laserIds: [] };
        drives.push(drive);
      }
      drive.laserIds.push(laser.id);
    }
    for (const line of LINES) {
    const system = model.systems[line], start = STARTS[line];
    for (let e = 0; e < system.excited.length; e++) for (let g = 0; g < 8; g++) {
      const excited = system.excited[e], ground = system.ground[g], q = excited.m - ground.m;
      if (q !== -1 && q !== 0 && q !== 1) continue;
      if (fieldMagnitudeG === 0 && Math.abs(excited.F - ground.F) > 1) continue;
      const p = polarization[q + 1], amplitude = factor * system.dipoles[e * 8 + g];
      const real = amplitude * (p.re * cosPhase - p.im * sinPhase);
      const imag = amplitude * (p.re * sinPhase + p.im * cosPhase);
      const eg = (start + e) * N + g, ge = g * N + start + e;
      if (drive) {
        addOptical(drive.cos.real,eg,real); addOptical(drive.cos.imag,eg,imag);
        addOptical(drive.cos.real,ge,real); addOptical(drive.cos.imag,ge,-imag);
        // Lower optical block: A exp(-i delta*t). Upper block is its adjoint.
        const sign = Math.sign(beatMHz);
        addOptical(drive.sin.real,eg,sign*imag); addOptical(drive.sin.imag,eg,-sign*real);
        addOptical(drive.sin.real,ge,sign*imag); addOptical(drive.sin.imag,ge,sign*real);
      } else {
        addOptical(hReal,eg,real); addOptical(hImag,eg,imag);
        addOptical(hReal,ge,real); addOptical(hImag,ge,-imag);
      }
    }
    }
  }
  for(const [target,entries] of opticalTerms)for(const [index,terms] of entries)target[index]=accurateSum([target[index]!,...terms]);
  // The spectroscopy graph and the driven model share the same missing-state
  // correction. D-line co-rotating couplings remain explicit above and must
  // not be counted a second time as an added AC shift.
  const externalShift=missingManifoldLightShift(scene);
  for(let index=0;index<N*N;index++){
    hReal[index]=accurateSum([hReal[index]!,TAU*externalShift.real[index]!]);
    hImag[index]=accurateSum([hImag[index]!,TAU*externalShift.imag[index]!]);
    finite(hReal[index]!,'Hamiltonian real entry');finite(hImag[index]!,'Hamiltonian imaginary entry');
  }
  // Exact coherent cancellations remove a beat, without a magnitude cutoff.
  const activeDrives = drives.filter(item => [...item.cos.real, ...item.cos.imag, ...item.sin.real, ...item.sin.imag].some(value => value !== 0));
  const initialId = settings?.initialStateId ?? scene.dynamics?.initialStateId ?? `ground-${scene.initialF}-${scene.initialM}`;
  const initialIndex = model.states.findIndex(state => state.id === initialId);
  if (initialIndex < 0) throw new Error(`Unknown initial state: ${initialId}`);
  return {
    modelSchema: ATOMIC_MODEL_SCHEMA,
    radiation: {
      referenceAngularFrequency: TAU * RB87.D2.frequencyHz * 1e-6,
      carrierAngularFrequency: TAU * (RB87.D2.frequencyHz * 1e-6 + rotatingCarrierOffsetMHz),
      dipoles: model.dipoles.map(operator => ({ real: operator.real.slice(), imag: operator.imag.slice() })) as RadiationModel['dipoles'],
    },
    dimension: 32, states: model.states.map(state => ({ ...state })), hReal, hImag,
    collapse: model.collapse.map(operator => ({ real: operator.real.slice(), imag: operator.imag.slice() })),
    initialIndex, fieldVector, fieldMagnitudeG, laserCount: lasers.length, carriersMHz,
    decayChannels: model.decayChannels.map(channel => ({ ...channel })),
    timeDependent: activeDrives.length > 0, drives: activeDrives, rotatingCarrierOffsetMHz,
    assumptions: [
      '32 internal states: 5S1/2 (8), 5P1/2 (8), 5P3/2 (16); all hyperfine/Zeeman coherences retained.',
      'Electric-dipole optical RWA and isotropic free-space Markov vacuum; the common three-dipole bath resolves emitted photon frequencies coherently.',
      'Signed spherical dipoles are transformed with field-mixed eigenvectors. Reference-normalized coefficients have units 1/sqrt(us); the native bath applies the emitted-frequency cubed correction.',
      'decayChannels and collapse contain bare free-atom decay metadata only; driven-atom radiation is constructed from radiation.dipoles under rb87-dipole/2.',
      'Every laser couples both D1 and D2. Unequal optical carriers retain their explicit beat-frequency Hamiltonians; no far-detuned beam is discarded.',
      ...externalShift.assumptions,
      'Atomic centre-of-mass motion, explicit population of higher electronic manifolds and technical laser noise are outside this 32-state internal model.',
      'Long-time populations are labelled as stationary values or time averages according to the solved dynamics; nonunique limits retain the specified initial state.',
    ],
  };
}
