import type {AxisAngles,Scene,DynamicsSettings,Laser,Level,InitialStateSpec,Vec3} from './types.js';
import {resolveInitialState} from './types.js';
import {staticModelBackground} from './spectroscopy.js';
import {atomicSystem,complexPolarization,laserIntensityMWcm2,laserCarrierOffsetMHz,laserFrequencyMHz,captureLaserFrequencyAnchor,RB87,sinCosDegrees,accurateSum} from './physics.js';
import {FULL_MODEL_SCHEMA} from './solver-protocol.js';

/** The rb87-full/2 model sent to the Rust worker: 32 states, complex row-major
 * matrices, H in rad/us, collapse amplitudes in 1/sqrt(us), and
 * H(t) = H0 + sum [cos(omega t) Hcos + sin(omega t) Hsin] with omega in rad/us.
 *
 * Physics: every enabled laser of nonzero intensity drives every allowed D1
 * and D2 dipole, co- AND counter-rotating, whatever its detuning. The frame
 * is one optical rotating frame (excited states at the reference carrier,
 * ground states unrotated), in which spontaneous emission is static. Only
 * states outside the 32-state model enter as a static second-order shift. */
interface MasterMatrix {real:number[];imag:number[]}
interface MasterDrive {omega:number;cos:MasterMatrix;sin:MasterMatrix;laserIds:string[]}
const N=32,TAU=2*Math.PI,HBAR=1.0545718176461565e-34,H_PLANCK=6.62607015e-34,EPS=8.8541878128e-12,C=299792458,zeros=()=>Array(N*N).fill(0),matrix=():MasterMatrix=>({real:zeros(),imag:zeros()});
/** h * 1 MHz / k_B, in kelvin. */
const MHZ_KELVIN=6.62607015e-34*1e6/1.380649e-23;

type AtomicSystem=ReturnType<typeof atomicSystem>;
interface BareFieldModel {systems:Record<'D1'|'D2',AtomicSystem>;collapse:MasterMatrix[]}
// Both D lines together (physics.atomicSystem caches one line). Independent of
// optical power, detuning, polarization and initial state.
const bareCache=new Map<number,BareFieldModel>();
/** Field-dressed D-line systems and their free-atom emission operators: one
 * operator per spherical component q of the emitted photon, NOT one jump per
 * transition, line or ground manifold. Each C_q contains both D lines and both
 * ground-F channels with amplitude d_eg sqrt(K omega_eg^3) (the Born-Markov
 * form, no secular approximation), so C_q rho C_q^dagger keeps radiative
 * coherence transfer, and far-detuned Raman scattering keeps the D1/D2
 * interference that suppresses spin flips. */
function bareFieldModel(magnitude:number):BareFieldModel {
 const cached=bareCache.get(magnitude);if(cached)return cached;
 const systems={D1:atomicSystem('D1',magnitude),D2:atomicSystem('D2',magnitude)},collapse=[matrix(),matrix(),matrix()];
 for(const line of ['D1','D2'] as const){
  const system=systems[line],data=RB87[line],start=line==='D1'?8:16;
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
   collapse[q+1]!.real[g*N+start+e]=Math.sqrt(data.gammaHz*1e-6)*amplitude*(frequencyHz/data.frequencyHz)**1.5;
  }
 }
 const result={systems,collapse};
 if(bareCache.size>=16)bareCache.delete(bareCache.keys().next().value!);
 bareCache.set(magnitude,result);
 return result;
}
/** Laser phases, built from their carrier provenance BEFORE multiplying by
 * TAU. A minimum-distance spanning tree rooted at the reference keeps a small
 * true beat between two nearby carriers as its own small axis, even when both
 * lie far from the reference. Equal carriers share a phase coordinate; no
 * tolerance, rational-frequency search, or independently rounded drive omega
 * is used. laserIndices[l] . frequenciesMHz = nu_l - nu_reference. */
function laserPhaseCoordinates(lasers:readonly Laser[]):{frequenciesMHz:number[];laserIndices:number[][]}{
 const frequenciesMHz:number[]=[],laserIndices:number[][]=Array.from({length:lasers.length},()=>[]);
 if(lasers.length===0)return {frequenciesMHz,laserIndices};
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
   if(!Number.isFinite(TAU*Math.abs(next.difference)))throw new Error('非零激光拍频超出可表示的频率范围');
   frequenciesMHz.push(Math.abs(next.difference));for(const index of laserIndices)index.push(0);
   laserIndices[next.child]=laserIndices[next.parent]!.slice();
   laserIndices[next.child]![frequenciesMHz.length-1]=Math.sign(next.difference);
  }else laserIndices[next.child]=laserIndices[next.parent]!.slice();
  connected.add(next.child);
 }
 return {frequenciesMHz,laserIndices};
}
/** U† A U, with columns U being the model eigenstates in the bare-B basis. */
function transformOperator(a:MasterMatrix,u:MasterMatrix):MasterMatrix {
 const tmp=matrix(),out=matrix();
 for(let i=0;i<N;i++)for(let k=0;k<N;k++){const ar=a.real[i*N+k]!,ai=a.imag[i*N+k]!;if(ar===0&&ai===0)continue;for(let j=0;j<N;j++){const br=u.real[k*N+j]!,bi=u.imag[k*N+j]!;tmp.real[i*N+j]+=ar*br-ai*bi;tmp.imag[i*N+j]+=ar*bi+ai*br;}}
 for(let i=0;i<N;i++)for(let k=0;k<N;k++){const ar=u.real[k*N+i]!,ai=-u.imag[k*N+i]!;if(ar===0&&ai===0)continue;for(let j=0;j<N;j++){const br=tmp.real[k*N+j]!,bi=tmp.imag[k*N+j]!;out.real[i*N+j]+=ar*br-ai*bi;out.imag[i*N+j]+=ar*bi+ai*br;}}return out;
}
const fieldAmplitude=(l:Laser)=>Math.sqrt(2*laserIntensityMWcm2(l)*10/(EPS*C));
/** Largest Rabi/(|detuning| + linewidth) over a laser's allowed D-line
 * couplings. The optical reference is the laser with the largest value: the
 * frame is an exact change of coordinates, so this only decides which
 * coupling is static (best conditioned), by magnitude. */
function mixing(l:Laser,bare:BareFieldModel,fieldVector:Vec3,fieldAxis:AxisAngles):number {
 const pol=complexPolarization(l,fieldVector,fieldAxis),E=fieldAmplitude(l);let best=0;
 for(const line of ['D1','D2'] as const){const sys=bare.systems[line],width=RB87[line].gammaHz/(TAU*1e6),carrier=laserCarrierOffsetMHz(l,line);
  for(let e=0;e<sys.excited.length;e++)for(let g=0;g<8;g++){const q=sys.excited[e]!.m-sys.ground[g]!.m;if(q< -1||q>1)continue;
   const rabi=Math.abs(sys.dipoles[e*8+g]!)*RB87.D2.dipole*E/H_PLANCK/1e6*Math.hypot(pol[q+1]!.re,pol[q+1]!.im);
   best=Math.max(best,rabi/(Math.abs(carrier-(sys.excited[e]!.energyMHz-sys.ground[g]!.energyMHz))+width));
  }}
 return best;
}
/** Diagonal initial populations in the model basis (see InitialStateSpec). */
function initialPopulations(spec:InitialStateSpec,states:Level[]):number[] {
 let weights:number[];
 if(spec.kind==='state'){const index=states.findIndex(s=>s.id===spec.id);if(index<0)throw new Error('动力学初态必须是真实原子态');weights=states.map((_,i)=>+(i===index));}
 else if(spec.kind==='unpolarized')weights=states.map(s=>+(s.manifold==='ground'&&(spec.manifold==='ground'||s.F===(spec.manifold==='F1'?1:2))));
 else if(spec.kind==='thermal'){
  const T=spec.temperatureK;if(!Number.isFinite(T)||!(T>0))throw new Error('热平衡初态需要有限的正温度');
  const lowest=Math.min(...states.filter(s=>s.manifold==='ground').map(s=>s.energyMHz));
  weights=states.map(s=>s.manifold==='ground'?Math.exp(-MHZ_KELVIN*(s.energyMHz-lowest)/T):0);
 }else{
  const ids=new Set(states.map(s=>s.id));
  if(Object.entries(spec.weights).some(([id,w])=>!ids.has(id)||!Number.isFinite(w)||w<0))throw new Error('初态权重必须是真实原子态上的非负有限数');
  weights=states.map(s=>spec.weights[s.id]??0);
 }
 const total=accurateSum(weights);if(!(total>0)||!Number.isFinite(total))throw new Error('初态权重之和必须为正');
 return weights.map(w=>w/total);
}
function initialText(spec:InitialStateSpec):string {
 if(spec.kind==='state')return `初态：模型基中的单一本征态 ${spec.id}。`;
 if(spec.kind==='unpolarized')return spec.manifold==='ground'?'初态：8 个基态等权混合（非极化）。':`初态：F=${spec.manifold==='F1'?1:2} 的基态等权混合（非极化）。`;
 if(spec.kind==='thermal')return `初态：静态 H0 的 8 个基态本征能量上的 Boltzmann 分布，T = ${spec.temperatureK} K。`;
 return '初态：按给定权重（归一化）在模型基中的对角混合。';
}

export function buildFullMasterModel(scene:Scene,settings?:Partial<DynamicsSettings>){
 // One basis for H0, C, the initial density and the returned populations:
 // eigenstates of the static H0 (B field + second-order shifts from states
 // outside the model and the Rb+ core). No D-line dressing: those couplings
 // are explicit below, so nothing is counted twice.
 const background=staticModelBackground(scene),a=background.analysis,states=background.levels;
 const bare=bareFieldModel(a.fieldMagnitudeG),u=matrix();states.forEach((s,j)=>{for(let i=0;i<N;i++){u.real[i*N+j]=s.backgroundState?.real[i]??(i===j?1:0);u.imag[i*N+j]=s.backgroundState?.imag[i]??0;}});
 const enabled=scene.lasers.filter(l=>l.enabled&&laserIntensityMWcm2(l)>0);
 let referenceIndex=0,best=-1;enabled.forEach((l,i)=>{const value=mixing(l,bare,a.fieldVector,a.fieldAxis);if(value>best){best=value;referenceIndex=i;}});
 const reference=enabled[referenceIndex],lasers=reference?[reference,...enabled.filter((_,i)=>i!==referenceIndex)]:[];
 // Frequency axes: the tree of laser beats, then the reference optical
 // carrier itself. In the frame rotating at nu_ref a laser's co-rotating
 // coupling has charge -(nu_l - nu_ref) and its counter-rotating coupling
 // charge +(nu_l + nu_ref) = 2 nu_ref + (nu_l - nu_ref): both are integer
 // combinations of the declared axes, with no rounding of a 768 THz phase.
 const tree=laserPhaseCoordinates(lasers),carrierMHz=reference?laserFrequencyMHz(reference):undefined;
 const frequenciesMHz=carrierMHz===undefined?tree.frequenciesMHz:[...tree.frequenciesMHz,carrierMHz],frequencies=frequenciesMHz.map(f=>TAU*f),axes=frequenciesMHz.length;
 const beatIndex=(l:number)=>carrierMHz===undefined?tree.laserIndices[l]!:[...tree.laserIndices[l]!,0];
 const offsets={ground:0,D1:reference?laserCarrierOffsetMHz(reference,'D1'):(RB87.D2.frequencyHz-RB87.D1.frequencyHz)/1e6,D2:reference?laserCarrierOffsetMHz(reference,'D2'):0};
 const opticalFrameMHz=RB87.D2.frequencyHz/1e6+offsets.D2;
 const h=matrix(),groundReference=states.find(s=>s.id==='ground-2-0')!.energyMHz;
 states.forEach((s,i)=>h.real[i*N+i]=TAU*(s.energyMHz-groundReference-(s.manifold==='ground'?0:offsets[s.manifold])));
 const drives:MasterDrive[]=[],driveIndices:number[][]=[],driveByKey=new Map<string,MasterDrive>(),terms=new Map<number[],Map<number,number[]>>();
 const put=(m:number[],i:number,v:number)=>{let entries=terms.get(m);if(!entries){entries=new Map();terms.set(m,entries);}let values=entries.get(i);if(!values){values=[];entries.set(i,values);}values.push(v);};
 /** Adds X e^{+i q.omega t} + h.c. with X_ij = value. */
 const harmonic=(index:number[],i:number,j:number,re:number,im:number,laserId:string)=>{
  if(re===0&&im===0)return;
  let q=index;
  if(q.some(value=>value!==0)){
   const value=accurateSum(q.map((n,k)=>n*frequenciesMHz[k]!));
   if(!Number.isFinite(value)||value===0)throw new Error('非零激光相位组合无法表示为非零频率；未将其合并为静态项');
   if(value<0){q=q.map(n=>n===0?0:-n);[i,j]=[j,i];im=-im;}
  }else{put(h.real,i*N+j,re);put(h.imag,i*N+j,im);put(h.real,j*N+i,re);put(h.imag,j*N+i,-im);return;}
  const key=q.join(',');let drive=driveByKey.get(key);
  if(!drive){drive={omega:accurateSum(q.map((n,k)=>n*frequencies[k]!)),cos:matrix(),sin:matrix(),laserIds:[]};drives.push(drive);driveIndices.push(q);driveByKey.set(key,drive);}
  if(!drive.laserIds.includes(laserId))drive.laserIds.push(laserId);
  // cos: X + X^dagger; sin: i (X - X^dagger).
  put(drive.cos.real,i*N+j,re);put(drive.cos.imag,i*N+j,im);put(drive.cos.real,j*N+i,re);put(drive.cos.imag,j*N+i,-im);
  put(drive.sin.real,i*N+j,-im);put(drive.sin.imag,i*N+j,re);put(drive.sin.real,j*N+i,-im);put(drive.sin.imag,j*N+i,-re);
 };
 for(const [laserIndex,l] of lasers.entries()){
  const pol=complexPolarization(l,a.fieldVector,a.fieldAxis),phase=sinCosDegrees(l.phaseDeg??0),E=fieldAmplitude(l);
  // E(t) = (E/2)[eps e^{i phi} e^{-i w t} + c.c.]. The e^{+i w t} part drives
  // Delta m = q through (eps^*)_q = (-1)^q conj(eps_{-q}).
  const co=pol.map(p=>({re:p.re*phase.cos-p.im*phase.sin,im:p.re*phase.sin+p.im*phase.cos})),counter=[0,1,2].map(k=>{const r=co[2-k]!;return k===1?{re:r.re,im:-r.im}:{re:-r.re,im:r.im};});
  const absorption=(field:{re:number;im:number}[])=>{const out=matrix();
   for(const line of ['D1','D2'] as const){const sys=bare.systems[line],start=line==='D1'?8:16;for(let e=0;e<sys.excited.length;e++)for(let g=0;g<8;g++){const q=sys.excited[e]!.m-sys.ground[g]!.m;if(q< -1||q>1)continue;const p=field[q+1]!,v=-.5*sys.dipoles[e*8+g]!*RB87.D2.dipole*E/HBAR*1e-6;out.real[(start+e)*N+g]=v*p.re;out.imag[(start+e)*N+g]=v*p.im;}}
   return transformOperator(out,u);};
  const A=absorption(co),R=absorption(counter),beat=beatIndex(laserIndex),coIndex=beat.map(n=>n===0?0:-n),counterIndex=beat.map((n,k)=>k===axes-1?n+2:n);
  for(let e=8;e<N;e++)for(let g=0;g<8;g++){const k=e*N+g;harmonic(coIndex,e,g,A.real[k]!,A.imag[k]!,l.id);harmonic(counterIndex,e,g,R.real[k]!,R.imag[k]!,l.id);}
 }
 for(const [m,entries]of terms)for(const [i,v]of entries)m[i]=accurateSum([m[i]!,...v]);
 // Coherent beams of one carrier can cancel exactly (e.g. opposite phases).
 // Such a harmonic is identically zero and is not sent.
 const live=drives.map(d=>[d.cos,d.sin].some(x=>x.real.some(v=>v!==0)||x.imag.some(v=>v!==0)));
 const sentDrives=drives.filter((_,j)=>live[j]),sentIndices=driveIndices.filter((_,j)=>live[j]);
 // Spontaneous emission in the model basis. In this frame every excited
 // state rotates at the same carrier and no ground state rotates, so each
 // operator, cross-F terms included, is static.
 const collapse=bare.collapse.map(c=>transformOperator(c,u)).filter(c=>c.real.some(v=>v!==0)||c.imag.some(v=>v!==0));
 const initialState=resolveInitialState(settings,scene),populations=initialPopulations(initialState,states);
 const initialReal=zeros();populations.forEach((p,i)=>initialReal[i*N+i]=p);
 const initialIndex=populations.indexOf(Math.max(...populations));
 return {modelSchema:FULL_MODEL_SCHEMA,dimension:32,states,hReal:h.real,hImag:h.imag,collapse,
 // Full initial density, diagonal in the model basis; initialIndex is its
 // largest population (the solver's coordinate anchor).
 initialIndex,initialReal,initialImag:zeros(),initialPopulations:populations,initialState,drives:sentDrives,
 // MHz coordinates are authoritative carrier differences and the reference
 // carrier. Angular values are redundant representations; multiplication by
 // TAU must not decide whether an integer relation exists between them.
 frequencyCoordinates:{frequenciesMHz,frequencies,driveIndices:sentIndices,frameIndices:states.map(()=>Array(axes).fill(0)),frameSectors:states.map(s=>s.manifold==='ground'?0:1)},
 fieldVector:a.fieldVector,fieldMagnitudeG:a.fieldMagnitudeG,laserCount:lasers.length,referenceLaserId:reference?.id,carrierMHz,
 frameFrequenciesMHz:states.map(s=>s.manifold==='ground'?0:opticalFrameMHz),staticMixingRatio:background.maximumMixingRatio,
 assumptions:[
  `每束开启且强度非零的激光（${lasers.length} 束）都是 32 态主方程中的显式光学驱动：保留 D1、D2 全部允许电偶极耦合，旋转项与反旋转项都保留（不作 RWA），不按近/远失谐区分。`,
  `统一光学旋转系：激发态以参考光${reference?`「${reference.name}」`:''}的载频旋转，基态不旋转；激光间拍频和反旋转项（频率 ω_L+ω_ref，参考载频是独立频率轴）都以精确整数相位坐标进入。`,
  '自发辐射：每个光子球分量 q 一个塌缩算符，同时包含 D1、D2 两条线和两个基态 F 通道（Born–Markov 形式，无 secular 近似，按跃迁频率保留 ω³ 因子）；远失谐 Raman 散射因此保留 D1/D2 干涉。',
  '32 态以外的能级（5S 的 6–9P；5P 的 6–10S、4–9D）和 Rb+ 芯极化（Arora & Sahoo 2012：α_c = 9.1 au，5S 另加 α_vc = −0.26 au）只作为二阶静态（van Vleck）光移进入 H0，并检查耦合/失谐比 ≤ 0.01、失谐 ≥ 10 GHz；不含这些能级的跨载频项和散射、更高价态尾项与连续谱。',
  '基矢：H0 静态部分（磁场 + 上述 32 态外静态光移）的本征基，初态与返回布居都在此基中；态标签 F、mF 指与之重叠最大的磁场本征态。D 线光学缀饰不进入基矢，而由动力学显式给出。',
  initialText(initialState),
  '返回瞬时布居（含反旋转项引起的光频微运动，幅度约 (Ω/ω)²）；长期结果是时间平均。',
 ]};
}
