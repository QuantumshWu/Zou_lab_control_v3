import type {Analysis,Laser,Level,Scene,SpectralLine} from './types.js';
import {RB87,accurateSum,analyze,atomicSystem,laserCarrierOffsetMHz,laserFrequencyDifferenceMHz,laserFrequencyMHz,laserIntensityMWcm2,sinCosDegrees,transitionReferenceMHz,wigner3j as rawWigner3j} from './physics.js';
import type {Eigenstate} from './physics.js';
import {CORE_POLARIZABILITY_AU,SPECTROSCOPY_SOURCES,VIRTUAL_MANIFOLDS} from './spectroscopy-data.js';
import {cmatrix,solveStaticBackground} from './floquet-background.js';
import {withUnit} from './format.js';
import type {BackgroundProblem,BackgroundSolution,FarCarrier,Manifold,OutsideState} from './floquet-background.js';
/** A laser's spherical amplitudes on the analysis' quantization axis (physics.ts polarization). */
const polarizationOf=(analysis:Analysis,laser:Laser)=>{const p=analysis.polarizations[laser.id];if(!p)throw new Error(`「${laser.name}」不在这次分析中。`);return p.amplitudes;};

const angularValues=new Map<string,number>();
function wigner3j(a:number,b:number,c:number,d:number,e:number,f:number):number{const key=[a,b,c,d,e,f].join(':');const old=angularValues.get(key);if(old!==undefined)return old;const value=rawWigner3j(a,b,c,d,e,f);angularValues.set(key,value);return value;}
const H=6.62607015e-34,EPS=8.8541878128e-12,C=299792458,EA0=1.602176634e-19*5.29177210903e-11,EV_MHZ=1.602176634e-19/H/1e6,N=32;
/** Atomic unit of polarizability, e^2 a0^2 / Eh, in C m^2 / V (CODATA 2018). */
const POLARIZABILITY_AU=1.64877727436e-41;
type Z={re:number;im:number};
type Matrix={real:number[];imag:number[]};
type Group={beams:Laser[];field:Z[]};
const mul=(a:Z,b:Z):Z=>({re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re});
const conj=(a:Z):Z=>({re:a.re,im:-a.im});
const norm=(a:Z)=>Math.hypot(a.re,a.im);
/** The levels chart's background: the static light shift of the lasers whose
 * "显示光移" switch is on. Display only: the dynamics never uses it (every laser
 * is an explicit drive there, full-master-model.ts). */
const ASSUMPTIONS=[
  '能级图光移仅用于显示：打开“显示光移”的激光取静态光移，即这些光的 Floquet 准能量：D 线旋转与反旋转项、32 态以外能级（5S 的 6–9P；5P 的 6–10S、4–9D）和 Rb+ 芯极化，用能量相关的 Löwdin（Brillouin–Wigner）分块全阶求解，保留偏振、磁子态混合和多束光之间的交叉项。',
  '动力学不使用这一显示背景：所有开启的激光都是主方程中的显式驱动。',
  '芯极化取 Arora & Sahoo 2012 的静态值（5S 9.1−0.26 au，5P 9.1 au）；未包含更高价态尾项和连续谱，高态超精细/同位素修正未分辨。',
  '32 态以外能级只与各自的精细流形耦合（数据只有矩元大小），不含经它们的 5P1/2–5P3/2 耦合；静态光移不含散射。',
  '图中布居投影到背景本征态（准能态的零电荷分量，对称正交化）；它不是含光学激发态混入及微运动的 Floquet 态布居。',
];
/** The largest mixing |Ω/2Δ| for which a laser's static light shift represents its effect. */
export const STATIC_MIXING_LIMIT=.01;

type AtomicSystem=ReturnType<typeof atomicSystem>;
const atomicCache=new Map<number,{D1:AtomicSystem;D2:AtomicSystem;states:Eigenstate[]}>();
function cachedAtom(fieldG:number){
  let item=atomicCache.get(fieldG);if(!item){const D1=atomicSystem('D1',fieldG),D2=atomicSystem('D2',fieldG);item={D1,D2,states:[...D1.ground,...D1.excited,...D2.excited]};atomicCache.set(fieldG,item);if(atomicCache.size>8)atomicCache.delete(atomicCache.keys().next().value!);}return item;
}
function groups(scene:Scene,analysis:Analysis,excluded?:string):Group[]{
  const result:Group[]=[];
  for(const beam of scene.lasers)if(beam.enabled&&beam.id!==excluded&&laserIntensityMWcm2(beam)!==0){
    let group=result.find(item=>laserFrequencyDifferenceMHz(item.beams[0]!,beam)===0);
    if(!group){group={beams:[],field:[]};result.push(group);}group.beams.push(beam);
  }
  for(const group of result){
    const fields=group.beams.map(beam=>{const p=polarizationOf(analysis,beam),phase=sinCosDegrees(beam.phaseDeg??0),scale=Math.sqrt(2*laserIntensityMWcm2(beam)*10/(EPS*C));return p.map(q=>({re:scale*(q.re*phase.cos-q.im*phase.sin),im:scale*(q.re*phase.sin+q.im*phase.cos)}));});
    group.field=[0,1,2].map(q=>({re:accurateSum(fields.map(f=>f[q]!.re)),im:accurateSum(fields.map(f=>f[q]!.im))}));
  }
  return result.filter(group=>group.field.some(q=>q.re!==0||q.im!==0));
}
/** The spherical components of E* (the counter-rotating part's polarization). */
function reversedField(field:Z[]):Z[]{return [0,1,2].map(q=>({re:(q===1?1:-1)*field[2-q]!.re,im:(q===1?-1:1)*field[2-q]!.im}));}
const MANIFOLDS:{name:Manifold;start:number;size:number;J:number}[]=[{name:'ground',start:0,size:8,J:.5},{name:'D1',start:8,size:8,J:RB87.D1.J},{name:'D2',start:16,size:16,J:RB87.D2.J}];
const outsideCache=new Map<number,Record<Manifold,OutsideState[]>>();
/** The states outside the model in the uncoupled |m_I, m_J⟩ basis with their Zeeman
 * energies, and their dipoles to each in-model state of their manifold. Angular
 * elements depend on B only; every element is kept, including exact zeros. */
function outsideStates(states:Eigenstate[],fieldG:number):Record<Manifold,OutsideState[]>{
  let rows=outsideCache.get(fieldG);if(rows)return rows;rows={ground:[],D1:[],D2:[]};
  for(const {name,start,size,J} of MANIFOLDS)for(const virtual of VIRTUAL_MANIFOLDS[name]){
    const gJ=1+(virtual.J*(virtual.J+1)+.75-virtual.L*(virtual.L+1))/(2*virtual.J*(virtual.J+1));
    for(let mi=-1.5;mi<=1.5;mi++)for(let mj=-virtual.J;mj<=virtual.J;mj++){
      const dipoles=new Float64Array(size*3);
      for(let k=0;k<size;k++)for(let q=-1;q<=1;q++){const s=states[start+k]!;dipoles[k*3+q+1]=accurateSum(s.basis.map((b,j)=>b.mi===mi&&mj-b.mj===q?s.vector[j]!*((-1)**Math.round(virtual.J-mj))*wigner3j(virtual.J,1,J,-mj,q,b.mj)*virtual.reducedEA0*EA0:0));}
      rows[name].push({name:virtual.name,energyMHz:virtual.energyEV*EV_MHZ+RB87.muBOverH*fieldG*(gJ*mj+RB87.gI*mi),dipoles});
    }
  }
  outsideCache.set(fieldG,rows);if(outsideCache.size>4)outsideCache.delete(outsideCache.keys().next().value!);return rows;
}
const scaled=(field:Z[])=>field.map(q=>({re:q.re/(2*H*1e6),im:q.im/(2*H*1e6)}));
/** The far carriers as a Floquet problem (floquet-background.ts): bare HFS + Zeeman
 * energies, the Rb+ core, every D1/D2 dipole's rotating and counter-rotating
 * coupling and the states outside the model. */
function backgroundProblem(analysis:Analysis,beams:Group[]):BackgroundProblem{
  const atom=cachedAtom(analysis.fieldMagnitudeG),states=atom.states,energies=Float64Array.from(states,s=>s.energyMHz),core=new Float64Array(N);
  const carriers:FarCarrier[]=beams.map(group=>{
    const rotatingField=scaled(group.field),counterField=scaled(reversedField(group.field)),rotating=cmatrix(24,8),counter=cmatrix(24,8);
    for(const line of ['D1','D2'] as const){const system=atom[line],offset=line==='D1'?0:8;
      for(let e=0;e<system.excited.length;e++)for(let g=0;g<8;g++){const q=system.excited[e]!.m-system.ground[g]!.m;if(q<-1||q>1)continue;const d=system.dipoles[e*8+g]!*RB87.D2.dipole,k=(offset+e)*8+g;
        rotating.re[k]=d*rotatingField[q+1]!.re;rotating.im[k]=d*rotatingField[q+1]!.im;counter.re[k]=d*counterField[q+1]!.re;counter.im[k]=d*counterField[q+1]!.im;}}
    // Rb+ core: the scalar shift −α|E|²/4 of every state of a manifold, per carrier.
    const squared=accurateSum(group.field.flatMap(q=>[q.re*q.re,q.im*q.im]));
    states.forEach((state,i)=>core[i]+=-CORE_POLARIZABILITY_AU[state.manifold]*POLARIZABILITY_AU*squared/(4*H*1e6));
    return {name:group.beams.map(b=>b.name).join(' + '),offsetMHz:laserCarrierOffsetMHz(group.beams[0]!,'D2'),rotatingField,counterField,rotating,counter};
  });
  return {energies,core,carriers,outside:outsideStates(states,analysis.fieldMagnitudeG)};
}
/** Whether a static light shift represents the carriers' effect: for every intermediate
 * state of each carrier, sqrt(Σ |V/Δ|²) over the levels of one manifold it couples to
 * (bare energies), and whether every such detuning keeps 100 MHz (D lines) or 10 GHz
 * (states outside the model). A refusal names the first carrier that fails. */
function staticMixing(problem:BackgroundProblem):{ratio:number;withinGap:boolean;failure?:string}{
  const cD2=RB87.D2.frequencyHz/1e6,dD1=(RB87.D1.frequencyHz-RB87.D2.frequencyHz)/1e6,e=problem.energies;
  // Energies relative to the 5P3/2 centroid; a ground state sits at −ν(D2).
  const relative=Float64Array.from(e,(x,i)=>i<8?x:i<16?dD1+x:x);
  let ratio=0,withinGap=true,failure:string|undefined;
  const add=(sum:number,re:number,im:number,detuning:number,minimumGap:number,label:()=>string)=>{
    if(re===0&&im===0)return sum;const gap=Math.abs(detuning);
    if(!(gap>=minimumGap)){withinGap=false;failure??=`${label()}：虚跃迁失谐小于 ${minimumGap} MHz，超出静态光移模型。`;}
    return sum+(gap===0?Infinity:(re*re+im*im)/(gap*gap));};
  for(const carrier of problem.carriers){
    const name=carrier.name,offset=carrier.offsetMHz;
    for(const [counter,m] of [[false,carrier.rotating],[true,carrier.counter]] as const){
      // E_g − (E_e − ω) = E_g − E_e + offset (rotating); E_g − (E_e + ω) = E_g − E_e − 2ν(D2) − offset.
      const detuning=(x:number,g:number)=>counter?e[g]!-relative[8+x]!-2*cD2-offset:e[g]!-relative[8+x]!+offset;
      for(let x=0;x<24;x++){let sum=0;for(let g=0;g<8;g++)sum=add(sum,m.re[x*8+g]!,m.im[x*8+g]!,detuning(x,g),100,()=>`D${x<8?1:2} / ${name}`);ratio=Math.max(ratio,Math.sqrt(sum));}
      for(const [from,to] of [[0,8],[8,24]])for(let g=0;g<8;g++){let sum=0;for(let x=from;x<to;x++)sum=add(sum,m.re[x*8+g]!,m.im[x*8+g]!,detuning(x,g),100,()=>`D${from?2:1} / ${name}`);ratio=Math.max(ratio,Math.sqrt(sum));}
    }
    for(const {name:manifold,start,size} of MANIFOLDS)for(const state of problem.outside[manifold])for(const [sign,field] of [[1,carrier.rotatingField],[-1,carrier.counterField]] as const){
      // E_i ± ω − E_v, with E_i absolute.
      const omega=sign*(cD2+offset);let sum=0;
      for(let k=0;k<size;k++){let re=0,im=0;for(let q=0;q<3;q++){const d=state.dipoles[k*3+q]!;re+=d*field[q]!.re;im+=d*field[q]!.im;}
        sum=add(sum,re,im,(start+k<8?e[start+k]!:cD2+relative[start+k]!)+omega-state.energyMHz,10000,()=>`${state.name} 高态 / ${name}`);}
      ratio=Math.max(ratio,Math.sqrt(sum));}
    if(failure===undefined&&!(ratio<=STATIC_MIXING_LIMIT))failure=`${name}：静态光移的混合比 |Ω/2Δ| = ${(100*ratio).toPrecision(3)}%，超过 ${100*STATIC_MIXING_LIMIT}% 的适用范围。`;
  }
  return {ratio,withinGap,failure};
}
/** The lasers whose light shift the levels chart shows ("显示光移"). */
const shownLasers=(scene:Scene):Laser[]=>scene.lasers.filter(l=>l.showLightShift===true);
/** The shown lasers of a scene as the Floquet problem the levels chart solves. */
export function staticBackgroundProblem(scene:Scene,excludedLaserId?:string):BackgroundProblem{
  const analysis=analyze(scene);return backgroundProblem(analysis,groups({...scene,lasers:shownLasers(scene)},analysis,excludedLaserId));
}
const solutionCache=new Map<string,BackgroundSolution>();
/** The solved background, cached on its physical inputs only: selecting another state
 * or reference cannot alter it. */
function solvedBackground(analysis:Analysis,beams:Group[],problem:BackgroundProblem):BackgroundSolution{
  const key=JSON.stringify([analysis.fieldMagnitudeG,beams.map(g=>[laserCarrierOffsetMHz(g.beams[0]!,'D2'),g.field])]);
  let solution=solutionCache.get(key);if(solution)return solution;
  solution=solveStaticBackground(problem);solutionCache.set(key,solution);if(solutionCache.size>12)solutionCache.delete(solutionCache.keys().next().value!);return solution;
}

function identityState(index:number):Level['backgroundState']{return {real:Array.from({length:N},(_,i)=>i===index?1:0),imag:Array(N).fill(0)};}
const allLevels=(analysis:Analysis)=>[...analysis.ground,...analysis.D1,...analysis.D2];
type DipoleEntry={fieldG:number;ground:Matrix;excited:Matrix;dipole:Matrix};
type ProjectionPlan={ids:string[];vectors:Matrix[];support:number[][];key:string};
type BasisContext={dipoles:Map<string,DipoleEntry>;projection?:ProjectionPlan};
const analysisContexts=new WeakMap<Analysis,BasisContext>();
const contextFor=(analysis:Analysis):BasisContext=>{let value=analysisContexts.get(analysis);if(!value){value={dipoles:new Map()};analysisContexts.set(analysis,value);}return value;};

// The master model's static H0 (full-master-model.ts): the magnetic eigenstates plus, for
// every enabled laser, the second-order (van Vleck) shift through the states outside the
// 32-state model and the Rb+ core. The D-line couplings are explicit drives of the master
// equation and are not part of it, so nothing is counted twice.
const zero=(n:number):Matrix=>({real:Array(n*n).fill(0),imag:Array(n*n).fill(0)});
const entry=(m:Matrix,k:number):Z=>({re:m.real[k]!,im:m.imag[k]!});
const amplitude=(d:number[],field:Z[]):Z=>({re:accurateSum(d.map((v,q)=>v*field[q]!.re))/(2*H*1e6),im:accurateSum(d.map((v,q)=>v*field[q]!.im))/(2*H*1e6)});
const centroid=(manifold:Level['manifold']):number=>manifold==='ground'?0:RB87[manifold].frequencyHz/1e6;
/** Adds one virtual state's symmetric second-order term and returns its mixing ratio. Every
 * represented amplitude is kept; a coupling within minimumGap of resonance or a mixing above
 * 1% is refused. */
function addVirtual(matrix:Matrix,indices:number[],values:Z[],denominators:number[],minimumGap:number,label:string):number{
  const magnitudes=values.map(norm),active=magnitudes.flatMap((value,i)=>value!==0?[i]:[]);let ratio2=0;
  for(const a of active){
    const gap=Math.abs(denominators[a]!);if(!(gap>=minimumGap))throw new Error(`${label}：虚跃迁失谐小于 ${minimumGap} MHz，超出静态色散光移模型。`);
    ratio2+=(magnitudes[a]!/gap)**2;
  }
  const ratio=Math.sqrt(ratio2);if(!(ratio<=.01))throw new Error(`${label}：光耦合/失谐比 ${ratio.toPrecision(3)} 超过二阶色散模型范围。`);
  for(let ai=0;ai<active.length;ai++)for(let bi=ai;bi<active.length;bi++){
    const a=active[ai]!,b=active[bi]!;
    const ab=mul(conj(values[a]!),values[b]!),factor=.5*(1/denominators[a]!+1/denominators[b]!),i=indices[a]!,j=indices[b]!,k=i*N+j;
    matrix.real[k]!+=factor*ab.re;matrix.imag[k]!+=factor*ab.im;
    if(i!==j){matrix.real[j*N+i]=matrix.real[k]!;matrix.imag[j*N+i]=-matrix.imag[k]!;}
  }
  return ratio;
}
type ModelCorrection={matrix:Matrix;maximumMixingRatio:number};
const modelCorrectionCache=new Map<string,ModelCorrection>();
/** The static correction in the magnetic eigenbasis (MHz): every carrier's co- and
 * counter-rotating coupling to each state outside the model, then its core shift; with the
 * largest mixing ratio of those couplings. */
function modelCorrection(analysis:Analysis,beams:Group[]):ModelCorrection{
  const key=JSON.stringify([analysis.fieldMagnitudeG,beams.map(g=>[laserCarrierOffsetMHz(g.beams[0]!,'D1'),laserCarrierOffsetMHz(g.beams[0]!,'D2'),g.field])]);
  const cached=modelCorrectionCache.get(key);if(cached)return cached;
  const states=cachedAtom(analysis.fieldMagnitudeG).states,outside=outsideStates(states,analysis.fieldMagnitudeG),matrix=zero(N);let maximumMixingRatio=0;
  for(const group of beams){
    const reverse=reversedField(group.field),frequency=laserFrequencyMHz(group.beams[0]!),names=group.beams.map(b=>b.name).join(' + ');
    for(const {name:manifold,start,size} of MANIFOLDS){const indices=Array.from({length:size},(_,k)=>start+k);
      for(const state of outside[manifold])for(const counter of [false,true]){
        const values=indices.map((_,k)=>amplitude([state.dipoles[k*3]!,state.dipoles[k*3+1]!,state.dipoles[k*3+2]!],counter?reverse:group.field));
        const delta=indices.map(i=>centroid(states[i]!.manifold)+states[i]!.energyMHz-state.energyMHz+(counter?-frequency:frequency));
        maximumMixingRatio=Math.max(maximumMixingRatio,addVirtual(matrix,indices,values,delta,10000,`${state.name} 高态 / ${names}`));
      }}
    // Rb+ core: the scalar shift −α|E|²/4 of every state of a manifold, per carrier.
    const squared=accurateSum(group.field.flatMap(q=>[q.re*q.re,q.im*q.im]));
    states.forEach((state,i)=>{matrix.real[i*N+i]!+=-CORE_POLARIZABILITY_AU[state.manifold]*POLARIZABILITY_AU*squared/(4*H*1e6);});
  }
  const result={matrix,maximumMixingRatio};modelCorrectionCache.set(key,result);if(modelCorrectionCache.size>12)modelCorrectionCache.delete(modelCorrectionCache.keys().next().value!);return result;
}
/** Cyclic complex-Hermitian Jacobi. No amplitude pruning; a rotation is skipped only when
 * its off-diagonal entry is exactly zero. The residual is checked against the input. */
function diagonalizeHermitian(input:Matrix,n:number):{values:number[];vectors:Matrix}{
  const a={real:input.real.slice(),imag:input.imag.slice()},v=zero(n);for(let i=0;i<n;i++)v.real[i*n+i]=1;
  const scale=Math.max(1,...input.real.map(Math.abs),...input.imag.map(Math.abs));
  for(let sweep=0;sweep<60;sweep++){
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
      const apq=entry(a,p*n+q),r=norm(apq);if(r===0)continue;
      const phase={re:apq.re/r,im:apq.im/r},tau=(a.real[q*n+q]!-a.real[p*n+p]!)/(2*r),t=tau===0?1:Math.sign(tau)/(Math.abs(tau)+Math.hypot(1,tau)),c=1/Math.hypot(1,t),s=t*c;
      if(s===0)continue;
      for(let k=0;k<n;k++){
        if(k!==p&&k!==q){const kp=entry(a,k*n+p),kq=entry(a,k*n+q),cq=mul(conj(phase),kq),cp=mul(phase,kp),np={re:c*kp.re-s*cq.re,im:c*kp.im-s*cq.im},nq={re:s*cp.re+c*kq.re,im:s*cp.im+c*kq.im};
          a.real[k*n+p]=a.real[p*n+k]=np.re;a.imag[k*n+p]=np.im;a.imag[p*n+k]=-np.im;a.real[k*n+q]=a.real[q*n+k]=nq.re;a.imag[k*n+q]=nq.im;a.imag[q*n+k]=-nq.im;}
        const vp=entry(v,k*n+p),vq=entry(v,k*n+q),cq=mul(conj(phase),vq),cp=mul(phase,vp);v.real[k*n+p]=c*vp.re-s*cq.re;v.imag[k*n+p]=c*vp.im-s*cq.im;v.real[k*n+q]=s*cp.re+c*vq.re;v.imag[k*n+q]=s*cp.im+c*vq.im;
      }
      a.real[p*n+p]!-=t*r;a.real[q*n+q]!+=t*r;a.real[p*n+q]=a.real[q*n+p]=a.imag[p*n+q]=a.imag[q*n+p]=0;
    }
    let off=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)off=Math.max(off,norm(entry(a,i*n+j)));if(off<=scale*Number.EPSILON*2)break;
  }
  const values=Array.from({length:n},(_,i)=>a.real[i*n+i]!);let residual=0;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){const products=Array.from({length:n},(_,k)=>mul(entry(input,i*n+k),entry(v,k*n+j)));residual=Math.max(residual,Math.hypot(accurateSum(products.map(x=>x.re))-values[j]!*v.real[i*n+j]!,accurateSum(products.map(x=>x.im))-values[j]!*v.imag[i*n+j]!));}
  if(!(residual<=scale*1e-11))throw new Error(`背景 Hamiltonian 对角化未收敛 (${residual} MHz)。`);
  return {values,vectors:v};
}
/** Minimum-cost assignment of eigenvectors to state labels, without duplicates at crossings. */
function labels(v:Matrix,n:number):number[]{
  const u=Array(n+1).fill(0),w=Array(n+1).fill(0),p=Array(n+1).fill(0),way=Array(n+1).fill(0);
  for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=Array(n+1).fill(Infinity),used=Array(n+1).fill(false);
    do{used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=n;j++)if(!used[j]){const k=(i0-1)*n+j-1,cur=-(v.real[k]!**2+v.imag[k]!**2)-u[i0]-w[j];if(cur<min[j]){min[j]=cur;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}for(let j=0;j<=n;j++){if(used[j]){u[p[j]]+=delta;w[j]-=delta;}else min[j]-=delta;}j0=j1;}while(p[j0]!==0);
    do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);
  }
  const result=Array(n).fill(0);for(let j=1;j<=n;j++)result[p[j]-1]=j-1;return result;
}
/** The static part of the master-model H0 and its eigenbasis (the model basis): each
 * fine-structure manifold's bare energies plus the static correction, diagonalized, each
 * eigenvector labelled by the magnetic eigenstate it overlaps most; and the largest mixing
 * ratio of the correction's couplings. */
export function staticModelBackground(scene:Scene):{analysis:Analysis;levels:Level[];maximumMixingRatio:number}{
  const analysis=analyze(scene),{matrix:correction,maximumMixingRatio}=modelCorrection(analysis,groups(scene,analysis)),zeroStates=cachedAtom(0).states,levels:Level[]=[];
  for(const [offset,count,key] of [[0,8,'ground'],[8,8,'D1'],[16,16,'D2']] as const){
    const original=analysis[key],h=zero(count);for(let i=0;i<count;i++)for(let j=0;j<count;j++){h.real[i*count+j]=correction.real[(offset+i)*N+offset+j]!+(i===j?original[i]!.energyMHz:0);h.imag[i*count+j]=correction.imag[(offset+i)*N+offset+j]!;}
    const eig=diagonalizeHermitian(h,count),assignment=labels(eig.vectors,count);
    levels.push(...original.map((state,i)=>{const col=assignment[i]!,vector={real:Array(N).fill(0),imag:Array(N).fill(0)};for(let k=0;k<count;k++){vector.real[offset+k]=eig.vectors.real[k*count+col]!;vector.imag[offset+k]=eig.vectors.imag[k*count+col]!;}const mixed=vector.real.some((x,k)=>k!==offset+i&&(x!==0||vector.imag[k]!==0));return {...state,energyMHz:eig.values[col]!,shiftMHz:eig.values[col]!-zeroStates[offset+i]!.energyMHz,totalShiftMHz:eig.values[col]!-zeroStates[offset+i]!.energyMHz,starkShiftMHz:eig.values[col]!-state.energyMHz,backgroundState:vector,mixed};}));
  }
  return {analysis,levels,maximumMixingRatio};
}

/** A laser's largest mixing amplitude |Ω/2Δ|, alone at its own carrier -- the ratio above
 * which the levels chart does not draw a static light shift (STATIC_MIXING_LIMIT, 1%) --
 * and whether each of its virtual transitions keeps the minimum detuning. The hint beside
 * the "显示光移" switch; it changes nothing. */
export function displayMixing(scene:Scene,laserId:string):{ratio:number;withinGap:boolean}{
  const laser=scene.lasers.find(l=>l.id===laserId);if(!laser||laserIntensityMWcm2(laser)===0)return {ratio:0,withinGap:true};
  const analysis=analyze(scene),m=staticMixing(backgroundProblem(analysis,groups({...scene,lasers:[{...laser,enabled:true}]},analysis)));
  return {ratio:m.ratio,withinGap:m.withinGap};
}
export function spectroscopyAnalysis(scene:Scene,excludedLaserId?:string,includeTransitions=true):Analysis{
  const start=performance.now(),analysis=analyze(scene),beams=groups({...scene,lasers:shownLasers(scene)},analysis,excludedLaserId);
  let solution:BackgroundSolution|undefined,failure='',maximumMixingRatio:number|undefined;
  if(beams.length){
    const problem=backgroundProblem(analysis,beams),validity=staticMixing(problem);maximumMixingRatio=validity.ratio;
    if(validity.failure)failure=validity.failure;
    else try{solution=solvedBackground(analysis,beams,problem);}catch(error){failure=error instanceof Error?error.message:String(error);}
  }
  const status=failure?'unresolved':beams.length?'dispersive':'bare',numerics=solution&&{floquetLayers:solution.layers,clusters:solution.clusters,solves:solution.solves,floquetBoundMHz:Math.max(...solution.levels.map(l=>l.floquetBoundMHz)),outsideBoundMHz:Math.max(...solution.levels.map(l=>l.outsideBoundMHz))};
  // The scope's first line: how accurate the numbers on the chart are.
  const accuracy=numerics?[`静态光移的数值误差：32 个能级均 ≤ ${withUnit((numerics.floquetBoundMHz+numerics.outsideBoundMHz)*1e6,'Hz',{digits:2})}（Floquet 取 ${numerics.floquetLayers} 层；各能级的界见悬停提示）。这只是数值误差，不含原子数据（矩阵元、能级、芯极化）的不确定度；后者这里不计算，通常大得多。`]:[];
  analysis.spectroscopy={status,backgroundLaserIds:beams.flatMap(g=>g.beams.map(b=>b.id)),excludedLaserId,modelScope:failure?'裸态位置/布居；静态总光移未定义。':beams.length?'磁场本征基上打开“显示光移”的激光的静态背景谱（仅显示）：Floquet 准能量，全阶分块求解；含偏振混合、D 线旋转与反旋转、列明高态和芯极化。':'HFS + Zeeman 裸态谱。',nearResonantLaserIds:failure&&beams.length===1?beams.flatMap(g=>g.beams.map(b=>b.id)):[],requiresDensityProjection:false,warnings:failure?[failure,...ASSUMPTIONS]:beams.length?[...accuracy,...ASSUMPTIONS]:[],maximumMixingRatio,sources:SPECTROSCOPY_SOURCES.slice(),...(numerics?{numerics}:{})};
  const zeroStates=cachedAtom(0).states;
  for(const [offset,key] of [[0,'ground'],[8,'D1'],[16,'D2']] as const)analysis[key]=analysis[key].map((state,i)=>{
    if(!solution)return {...state,totalShiftMHz:failure?null:state.shiftMHz,starkShiftMHz:failure?null:0,shiftBound:failure?null:{floquetMHz:0,outsideMHz:0},backgroundState:identityState(offset+i),mixed:false};
    const level=solution.levels[offset+i]!,real=Array.from(level.vector.re),imag=Array.from(level.vector.im),mixed=real.some((x,k)=>k!==offset+i&&(x!==0||imag[k]!==0));
    return {...state,energyMHz:level.energyMHz,shiftMHz:level.energyMHz-zeroStates[offset+i]!.energyMHz,totalShiftMHz:level.energyMHz-zeroStates[offset+i]!.energyMHz,starkShiftMHz:level.energyMHz-state.energyMHz,shiftBound:{floquetMHz:level.floquetBoundMHz,outsideMHz:level.outsideBoundMHz},backgroundState:{real,imag},mixed};
  });
  analysis.spectroscopy.requiresDensityProjection=status==='dispersive'&&allLevels(analysis).some(s=>s.mixed);
  analysis.transitions=[];
  if(includeTransitions&&status!=='unresolved')for(const laser of scene.lasers)if(laser.enabled){
    const source=allLevels(analysis).find(s=>s.id===(laser.referenceStateId??scene.dynamics.initialStateId));if(!source)continue;
    for(const target of allLevels(analysis)){if((source.manifold==='ground')===(target.manifold==='ground'))continue;const g=source.manifold==='ground'?source:target,e=source.manifold==='ground'?target:source,coupling=spectroscopyCoupling(analysis,laser,g.id,e.id);if(!coupling?.allowed)continue;
      const field=Math.sqrt(2*laserIntensityMWcm2(laser)*10/(EPS*C)),fullRabi=field*coupling.amplitudeCM/H/1e6;
      analysis.transitions.push({id:`${laser.id}:${g.id}:${e.id}`,laserId:laser.id,line:e.manifold as SpectralLine,groundId:g.id,excitedId:e.id,Fg:g.F,mg:g.m,Fe:e.F,me:e.m,q:coupling.q as -1|0|1,offsetMHz:coupling.offsetMHz,detuningMHz:coupling.detuningMHz,strength:coupling.strength,weight:fullRabi===0?0:(coupling.rabiMHz/fullRabi)**2,rabiMHz:coupling.rabiMHz,mixedPolarization:coupling.mixedPolarization,mLabelApproximate:coupling.mLabelApproximate});
    }
  }
  const c1=analysis.ground.find(s=>s.F===1&&s.m===0),c2=analysis.ground.find(s=>s.F===2&&s.m===0);if(c1&&c2&&status!=='unresolved'){analysis.clockMHz=c2.energyMHz-c1.energyMHz;analysis.clockShiftKHz=(analysis.clockMHz-2*RB87.groundA)*1e3;}
  analysis.computeMs=performance.now()-start;return analysis;
}

export function spectroscopyCoupling(analysis:Analysis,laser:Laser,firstId:string,secondId:string):{allowed:boolean;q:number;amplitudeCM:number;dipoleCM:number;strength:number;rabiMHz:number;offsetMHz:number;detuningMHz:number;line:SpectralLine|null;mixedPolarization:boolean;mLabelApproximate:boolean;dipoleComponentsCM:Matrix}|null{
  if(analysis.spectroscopy?.status==='unresolved')return null;
  const states=allLevels(analysis),g=states.find(s=>(s.id===firstId||s.id===secondId)&&s.manifold==='ground'),e=states.find(s=>(s.id===firstId||s.id===secondId)&&s.manifold!=='ground');
  if(!g||!e)return {allowed:false,q:0,amplitudeCM:0,dipoleCM:0,strength:0,rabiMHz:0,offsetMHz:NaN,detuningMHz:NaN,line:null,mixedPolarization:false,mLabelApproximate:false,dipoleComponentsCM:{real:[0,0,0],imag:[0,0,0]}};
  const line=e.manifold as SpectralLine,ug=g.backgroundState??identityState(states.indexOf(g))!,ue=e.backgroundState??identityState(states.indexOf(e))!,cache=contextFor(analysis).dipoles,key=`${g.id}:${e.id}`;
  let item=cache.get(key);
  if(!item||item.fieldG!==analysis.fieldMagnitudeG||!sameVector(item.ground,ug)||!sameVector(item.excited,ue)){
    // A background excited state may hold both fine-structure manifolds: sum both D lines.
    const d={real:[0,0,0],imag:[0,0,0]},terms:Z[][]=[[],[],[]];
    for(const branch of ['D1','D2'] as const){const system=cachedAtom(analysis.fieldMagnitudeG)[branch],offset=branch==='D1'?8:16;
      for(let ei=0;ei<system.excited.length;ei++)for(let gi=0;gi<8;gi++){const q=system.excited[ei]!.m-system.ground[gi]!.m;if(q<-1||q>1)continue;const amplitude=system.dipoles[ei*8+gi]!*RB87.D2.dipole,c=mul({re:ue.real[offset+ei]!,im:-ue.imag[offset+ei]!},{re:ug.real[gi]!,im:ug.imag[gi]!});terms[q+1]!.push({re:amplitude*c.re,im:amplitude*c.im});}}
    for(let q=0;q<3;q++){d.real[q]=accurateSum(terms[q]!.map(t=>t.re));d.imag[q]=accurateSum(terms[q]!.map(t=>t.im));}
    item={fieldG:analysis.fieldMagnitudeG,ground:{real:ug.real.slice(),imag:ug.imag.slice()},excited:{real:ue.real.slice(),imag:ue.imag.slice()},dipole:d};cache.set(key,item);
  }
  const d=item.dipole;
  const components=[0,1,2].map(q=>Math.hypot(d.real[q]!,d.imag[q]!)),amplitudeCM=Math.hypot(...components),p=polarizationOf(analysis,laser),coupled=p.map((x,k)=>mul(x,{re:d.real[k]!,im:d.imag[k]!})),rabiMHz=Math.hypot(accurateSum(coupled.map(x=>x.re)),accurateSum(coupled.map(x=>x.im)))*Math.sqrt(2*laserIntensityMWcm2(laser)*10/(EPS*C))/H/1e6;
  const driven= coupled.map(z=>Math.hypot(z.re,z.im)),displayComponents=driven.some(v=>v!==0)?driven:components,q=displayComponents.indexOf(Math.max(...displayComponents))-1;
  // This tolerance is for the label only; no Hamiltonian/dipole entry is dropped.
  const mixedM=(state:Level,v:Matrix)=>states.reduce((sum,bare,i)=>sum+(bare.m!==state.m?v.real[i]!**2+v.imag[i]!**2:0),0)>1e-8;
  const mLabelApproximate=mixedM(g,ug)||mixedM(e,ue);
  const offsetMHz=(RB87[line].frequencyHz-RB87[laser.line].frequencyHz)/1e6+e.energyMHz-g.energyMHz-transitionReferenceMHz(laser.line);
  return {allowed:amplitudeCM!==0,q,amplitudeCM,dipoleCM:amplitudeCM,strength:(amplitudeCM/RB87.D2.dipole)**2,rabiMHz,offsetMHz,detuningMHz:laserCarrierOffsetMHz(laser,line)-(e.energyMHz-g.energyMHz),line,mixedPolarization:displayComponents.filter(x=>x!==0).length>1,mLabelApproximate,dipoleComponentsCM:{real:d.real.slice(),imag:d.imag.slice()}};
}

function sameVector(first:Matrix,second:Matrix):boolean{return first.real.length===second.real.length&&first.imag.length===second.imag.length&&first.real.every((value,i)=>value===second.real[i]&&first.imag[i]===second.imag[i]);}
function projectionPlan(analysis:Analysis,levels:Level[]):ProjectionPlan{
  const context=contextFor(analysis),old=context.projection;
  if(old&&levels.length===old.ids.length&&levels.every((level,i)=>level.id===old.ids[i]&&sameVector(old.vectors[i]!,level.backgroundState??identityState(i)!)))return old;
  const vectors=levels.map((level,i)=>{const v=level.backgroundState??identityState(i)!;return {real:v.real.slice(),imag:v.imag.slice()};}),ids=levels.map(level=>level.id),support=vectors.map(v=>v.real.flatMap((value,i)=>value!==0||v.imag[i]!==0?[i]:[]));
  const result={ids,vectors,support,key:JSON.stringify([ids,vectors])};context.projection=result;return result;
}
const populationProjectionCache=new Map<string,Record<string,number>>();
export function projectSpectroscopyPopulations(analysis:Analysis,result:{states:string[];populations:number[];density?:Matrix;basis?:Level[]}):Record<string,number>|null{
  let levels=allLevels(analysis);
  if(result.basis?.some(s=>s.backgroundState)){
    const source=result.basis;
    levels=levels.map((level,a)=>{const v=level.backgroundState??identityState(a)!,real=Array(N).fill(0),imag=Array(N).fill(0);
      for(let j=0;j<N;j++){const u=source[j]!.backgroundState??identityState(j)!;const rs:number[]=[],is:number[]=[];for(let i=0;i<N;i++){rs.push(u.real[i]!*v.real[i]!+u.imag[i]!*v.imag[i]!);is.push(u.real[i]!*v.imag[i]!-u.imag[i]!*v.real[i]!);}real[j]=accurateSum(rs);imag[j]=accurateSum(is);}
      return {...level,backgroundState:{real,imag}};
    });
  }
  const out:Record<string,number>={},n=result.states.length,index=levels.map(s=>result.states.indexOf(s.id));
  if(index.some(i=>i<0))return null;
  if(!result.density){if(analysis.spectroscopy?.requiresDensityProjection)return null;for(let i=0;i<levels.length;i++)out[levels[i]!.id]=result.populations[index[i]!]!;return out;}
  const rho=result.density;if(rho.real.length!==n*n||rho.imag.length!==n*n)return null;
  // Content-based identity accepts new result wrappers and detects in-place rho
  // edits. Neither object identity nor populations alone can key a full-rho
  // projection, and changing the background basis must invalidate it as well.
  const plan=projectionPlan(analysis,levels),key=plan.key+'|'+JSON.stringify([result.states,rho.real,rho.imag]),saved=populationProjectionCache.get(key);if(saved)return {...saved};
  for(let a=0;a<levels.length;a++){
    const v=plan.vectors[a]!,support=plan.support[a]!,terms:number[]=[];
    // Skip only exact-zero vector entries. The nonzero i,j order and both
    // complex multiplications match v†rho v above; every coherence remains.
    for(const i of support)for(const j of support){const vr=v.real[i]!,vi=-v.imag[i]!,k=index[i]!*n+index[j]!,rr=rho.real[k]!,ri=rho.imag[k]!,leftRe=vr*rr-vi*ri,leftIm=vr*ri+vi*rr;terms.push(leftRe*v.real[j]!-leftIm*v.imag[j]!);}
    out[levels[a]!.id]=accurateSum(terms);
  }
  populationProjectionCache.set(key,{...out});if(populationProjectionCache.size>8)populationProjectionCache.delete(populationProjectionCache.keys().next().value!);
  return out;
}
