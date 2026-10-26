import type {Scene,DynamicsSettings,Laser} from './types.js';
import {spectroscopyAnalysis} from './spectroscopy.js';
import {classifyLasers} from './laser-regime.js';
import {atomicSystem,complexPolarization,laserFrequencyDifferenceMHz,laserIntensityMWcm2,laserCarrierOffsetMHz,captureLaserFrequencyAnchor,RB87,sinCosDegrees,accurateSum} from './physics.js';
import {FULL_MODEL_SCHEMA} from './solver-protocol.js';

/** The rb87-full/2 model sent to the Rust worker: 32 states, complex row-major
 * matrices, H in rad/us, collapse amplitudes in 1/sqrt(us), and
 * H(t) = H0 + sum [cos(omega t) Hcos + sin(omega t) Hsin] with omega in rad/us. */
interface MasterMatrix {real:number[];imag:number[]}
interface MasterDrive {omega:number;cos:MasterMatrix;sin:MasterMatrix;laserIds:string[]}
const N=32,TAU=2*Math.PI,zeros=()=>Array(1024).fill(0),matrix=():MasterMatrix=>({real:zeros(),imag:zeros()});

type AtomicSystem=ReturnType<typeof atomicSystem>;
interface BareFieldModel {systems:Record<'D1'|'D2',AtomicSystem>;collapse:MasterMatrix[]}
// Both D lines together (physics.atomicSystem caches one line). Independent of
// optical power, detuning, polarization and initial state.
const bareCache=new Map<number,BareFieldModel>();
/** Field-dressed D-line systems and their free-atom emission operators: one
 * operator per line and spherical component q, NOT one jump per transition, so
 * C_q rho C_q^dagger keeps radiative coherence transfer. */
function bareFieldModel(magnitude:number):BareFieldModel {
 const cached=bareCache.get(magnitude);if(cached)return cached;
 const systems={D1:atomicSystem('D1',magnitude),D2:atomicSystem('D2',magnitude)},collapse:MasterMatrix[]=[];
 for(const line of ['D1','D2'] as const){
  const system=systems[line],data=RB87[line],start=line==='D1'?8:16,operators=[matrix(),matrix(),matrix()];
  for(let e=0;e<system.excited.length;e++)for(let g=0;g<system.ground.length;g++){
   const excited=system.excited[e]!,ground=system.ground[g]!,q=excited.m-ground.m;
   if(q!==-1&&q!==0&&q!==1)continue;
   // At B=0 this is an exact symmetry rule, not a numerical cutoff.
   if(magnitude===0&&Math.abs(excited.F-ground.F)>1)continue;
   // atomicSystem uses the D2 stretched dipole as its common unit. Undo the
   // D1/D2 radial ratio here: Gamma already contains it.
   const amplitude=system.dipoles[e*8+g]!*RB87.D2.dipole/data.dipole;
   const frequencyHz=data.frequencyHz+1e6*(excited.energyMHz-ground.energyMHz);
   if(!(frequencyHz>0))throw new Error('Magnetic field exceeds the positive optical-frequency domain of this D-line model.');
   operators[q+1]!.real[g*N+start+e]=Math.sqrt(data.gammaHz*1e-6)*amplitude*(frequencyHz/data.frequencyHz)**1.5;
  }
  collapse.push(...operators);
 }
 const result={systems,collapse};
 if(bareCache.size>=16)bareCache.delete(bareCache.keys().next().value!);
 bareCache.set(magnitude,result);
 return result;
}
/** Laser phases, built from their carrier provenance BEFORE multiplying by
 * TAU. A minimum-distance spanning tree keeps a small true beat between two
 * nearby carriers as its own small axis, even when both lie far from the
 * optical reference. Equal carriers share a phase coordinate; no tolerance,
 * rational-frequency search, or independently rounded drive omega is used. */
function laserPhaseCoordinates(lasers:readonly Laser[]):{frequencies:number[];frequenciesMHz:number[];laserIndices:number[][]}{
 const frequencies:number[]=[],frequenciesMHz:number[]=[],laserIndices:number[][]=Array.from({length:lasers.length},()=>[]);
 if(lasers.length===0)return {frequencies,frequenciesMHz,laserIndices};
 const carriers=lasers.map(captureLaserFrequencyAnchor);
 const connected=new Set([0]);
 while(connected.size<lasers.length){
  let next:{child:number;parent:number;difference:number}|undefined;
  for(let child=0;child<lasers.length;child++)if(!connected.has(child))for(const parent of connected){
   const a=carriers[child]!,b=carriers[parent]!;
   const difference=accurateSum([a.offsetMHz,-b.offsetMHz,a.line===b.line?0:(RB87[a.line].frequencyHz-RB87[b.line].frequencyHz)/1e6]);
   if(!Number.isFinite(difference))throw new Error('激光相位坐标包含非有限拍频');
   if(!next||Math.abs(difference)<Math.abs(next.difference))next={child,parent,difference};
  }
  if(!next)throw new Error('无法构造激光载频的相位坐标');
  if(next.difference!==0){
   const frequency=TAU*Math.abs(next.difference);
   if(!Number.isFinite(frequency)||frequency===0)throw new Error('非零激光拍频超出可表示的频率范围');
   frequenciesMHz.push(Math.abs(next.difference));frequencies.push(frequency);for(const index of laserIndices)index.push(0);
   laserIndices[next.child]=laserIndices[next.parent]!.slice();
   laserIndices[next.child]![frequencies.length-1]=Math.sign(next.difference);
  }else laserIndices[next.child]=laserIndices[next.parent]!.slice();
  connected.add(next.child);
 }
 return {frequencies,frequenciesMHz,laserIndices};
}
/** U† A U, with columns U being the DC background eigenstates in the bare-B basis. */
function transformOperator(a:MasterMatrix,u:MasterMatrix):MasterMatrix {
 const tmp=matrix(),out=matrix();
 for(let i=0;i<N;i++)for(let k=0;k<N;k++){const ar=a.real[i*N+k]!,ai=a.imag[i*N+k]!;if(ar===0&&ai===0)continue;for(let j=0;j<N;j++){const br=u.real[k*N+j]!,bi=u.imag[k*N+j]!;tmp.real[i*N+j]+=ar*br-ai*bi;tmp.imag[i*N+j]+=ar*bi+ai*br;}}
 for(let i=0;i<N;i++)for(let k=0;k<N;k++){const ar=u.real[k*N+i]!,ai=-u.imag[k*N+i]!;if(ar===0&&ai===0)continue;for(let j=0;j<N;j++){const br=tmp.real[k*N+j]!,bi=tmp.imag[k*N+j]!;out.real[i*N+j]+=ar*br-ai*bi;out.imag[i*N+j]+=ar*bi+ai*br;}}return out;
}
export function buildFullMasterModel(scene:Scene,settings?:Partial<DynamicsSettings>,frame:'common'|'resonant'='resonant'){
 const classification=classifyLasers(scene),invalid=classification.find(c=>!c.valid);if(invalid)throw new Error(invalid.message);
 const a=spectroscopyAnalysis(scene,undefined,false),states=[...a.ground,...a.D1,...a.D2];
 if(a.spectroscopy?.status==='unresolved')throw new Error(a.spectroscopy.warnings?.[0]??'远失谐背景不满足色散条件');
 const bare=bareFieldModel(a.fieldMagnitudeG),u=matrix();states.forEach((s,j)=>{for(let i=0;i<N;i++){u.real[i*N+j]=s.backgroundState?.real[i]??(i===j?1:0);u.imag[i*N+j]=s.backgroundState?.imag[i]??0;}});
 const near=scene.lasers.filter(l=>l.enabled&&(l.regime??'near')==='near'&&laserIntensityMWcm2(l)>0);
 // One optical reference; a single carrier keeps every coupling static.
 // With multiple carriers, optional ground-coordinate phases improve conditioning.
 // ALL cross-F terms remain explicit harmonics; these phases remove no physics.
 const reference=near[0];
 const {frequencies,frequenciesMHz,laserIndices}=laserPhaseCoordinates(near);
 const offsets={ground:0,D1:reference?laserCarrierOffsetMHz(reference,'D1'):(RB87.D2.frequencyHz-RB87.D1.frequencyHz)/1e6,D2:reference?laserCarrierOffsetMHz(reference,'D2'):0};
 const opticalFrameMHz=RB87.D2.frequencyHz/1e6+offsets.D2;
 const groundFrames=new Map<number,number>([[1,0],[2,0]]),assigned=new Set<number>();
 const groundPhaseIndices=new Map<number,number[]>([[1,frequencies.map(()=>0)],[2,frequencies.map(()=>0)]]);
 if(frame==='resonant'&&reference)for(const [laserIndex,l] of near.entries()){const F=classification.find(c=>c.id===l.id)!.groundF;if(!assigned.has(F)){groundFrames.set(F,-laserFrequencyDifferenceMHz(l,reference));groundPhaseIndices.set(F,laserIndices[laserIndex]!.map(value=>-value));assigned.add(F);}}
 const h=matrix(),groundReference=states.find(s=>s.id==='ground-2-0')!.energyMHz;
 states.forEach((s,i)=>h.real[i*N+i]=TAU*(s.energyMHz-groundReference-(s.manifold==='ground'?groundFrames.get(s.F)!:offsets[s.manifold])));
 const drives:MasterDrive[]=[],driveIndices:number[][]=[],phaseDrives=new Map<string,MasterDrive>(),terms=new Map<number[],Map<number,number[]>>(),put=(m:number[],i:number,v:number)=>{let entries=terms.get(m);if(!entries){entries=new Map();terms.set(m,entries);}let values=entries.get(i);if(!values){values=[];entries.set(i,values);}values.push(v);};
 for(const [laserIndex,l] of near.entries()){const pol=complexPolarization(l,a.fieldVector,a.fieldAxis),phase=sinCosDegrees(l.phaseDeg??0),E=Math.sqrt(2*laserIntensityMWcm2(l)*10/(8.8541878128e-12*299792458)),absorption=matrix();
  for(const line of ['D1','D2'] as const){const sys=bare.systems[line],start=line==='D1'?8:16;for(let e=0;e<sys.excited.length;e++)for(let g=0;g<8;g++){const q=sys.excited[e]!.m-sys.ground[g]!.m;if(q< -1||q>1)continue;const p=pol[q+1]!,v=-.5*sys.dipoles[e*8+g]!*RB87.D2.dipole*E/1.0545718176461565e-34*1e-6;absorption.real[(start+e)*N+g]=v*(p.re*phase.cos-p.im*phase.sin);absorption.imag[(start+e)*N+g]=v*(p.re*phase.sin+p.im*phase.cos);}}
  const A=transformOperator(absorption,u);
  for(const F of [1,2]){const phaseIndex=laserIndices[laserIndex]!.map((value,i)=>value+groundPhaseIndices.get(F)![i]!);
  const dynamic=phaseIndex.some(value=>value!==0),beat=accurateSum(phaseIndex.map((value,i)=>value*frequencies[i]!));let drive:MasterDrive|undefined;
  if(dynamic){
   if(!Number.isFinite(beat)||beat===0)throw new Error('非零激光相位组合无法表示为非零拍频；未将其合并为静态项');
   const index=phaseIndex.map(value=>value===0?0:value*Math.sign(beat)),key=index.join(',');drive=phaseDrives.get(key);
   if(!drive){drive={omega:accurateSum(index.map((value,i)=>value*frequencies[i]!)),cos:matrix(),sin:matrix(),laserIds:[]};drives.push(drive);driveIndices.push(index);phaseDrives.set(key,drive);}drive.laserIds.push(l.id);
  }
  for(let e=8;e<N;e++)for(let g=0;g<8;g++){if(states[g]!.F!==F)continue;const k=e*N+g,re=A.real[k]!,im=A.imag[k]!;if(re===0&&im===0)continue;
   if(!drive){put(h.real,k,re);put(h.imag,k,im);put(h.real,g*N+e,re);put(h.imag,g*N+e,-im);}else{const sign=Math.sign(beat);put(drive.cos.real,k,re);put(drive.cos.imag,k,im);put(drive.cos.real,g*N+e,re);put(drive.cos.imag,g*N+e,-im);put(drive.sin.real,k,sign*im);put(drive.sin.imag,k,-sign*re);put(drive.sin.real,g*N+e,sign*im);put(drive.sin.imag,g*N+e,sign*re);}
  }
 }}
 for(const [m,entries]of terms)for(const [i,v]of entries)m[i]=accurateSum([m[i]!,...v]);
 // Spontaneous emission retains all branches in the background basis. Resolve
 // different ground-hyperfine emission frequencies so the dissipator is static
 // in the chosen manifold frame; no population-rate equation is introduced.
 const collapse:MasterMatrix[]=[];
 for(const c of bare.collapse){const transformed=transformOperator(c,u);for(const F of [1,2]){const channel=matrix();for(let g=0;g<8;g++)if(states[g]!.F===F)for(let e=8;e<N;e++){channel.real[g*N+e]=transformed.real[g*N+e]!;channel.imag[g*N+e]=transformed.imag[g*N+e]!;}if(channel.real.some(v=>v!==0)||channel.imag.some(v=>v!==0))collapse.push(channel);}}
 const initialId=settings?.initialStateId??scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`,initialIndex=states.findIndex(s=>s.id===initialId);if(initialIndex<0)throw new Error('动力学初态必须是真实原子态');
 return {modelSchema:FULL_MODEL_SCHEMA,dimension:32,states,hReal:h.real,hImag:h.imag,collapse,initialIndex,drives,
 // MHz coordinates are authoritative carrier differences. Angular values
 // are redundant representations; multiplication by TAU must not decide
 // whether an integer relation exists between the original carrier phases.
 frequencyCoordinates:{frequenciesMHz,frequencies,driveIndices,frameIndices:states.map(s=>s.manifold==='ground'?groundPhaseIndices.get(s.F)!.slice():frequencies.map(()=>0)),frameSectors:states.map(s=>s.manifold==='ground'?0:1)},
 fieldVector:a.fieldVector,fieldMagnitudeG:a.fieldMagnitudeG,laserCount:near.length,frameFrequenciesMHz:states.map(s=>s.manifold==='ground'?groundFrames.get(s.F)!:opticalFrameMHz),
 assumptions:['远失谐光仅贡献背景 Stark Hamiltonian；不包含远失谐光散射。','统一光学旋转系；每束近共振光显式保留两个基态 F 与 D1、D2 全部允许电偶极耦合。','仅去除光学反旋转项，不按失谐或流形删除驱动矩阵元。','不同光频的真实拍频显式进入 Lindblad 方程。','衰变算符变换到相同的背景本征基；不同基态超精细发射频率采用 secular 分辨。','初态及返回布居定义在远失谐背景本征基；参考能级不进入动力学。']};
}
