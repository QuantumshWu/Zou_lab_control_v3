import type {Analysis,Laser,Level,Scene,SpectralLine} from './types.js';
import {RB87,accurateSum,analyze,atomicSystem,complexPolarization,laserCarrierOffsetMHz,laserFrequencyDifferenceMHz,laserFrequencyMHz,laserIntensityMWcm2,sinCosDegrees,transitionReferenceMHz,wigner3j as rawWigner3j} from './physics.js';
import type {Eigenstate} from './physics.js';
import {CORE_POLARIZABILITY_AU,SPECTROSCOPY_SOURCES,VIRTUAL_MANIFOLDS} from './spectroscopy-data.js';

const angularValues=new Map<string,number>();
function wigner3j(a:number,b:number,c:number,d:number,e:number,f:number):number{const key=[a,b,c,d,e,f].join(':');const old=angularValues.get(key);if(old!==undefined)return old;const value=rawWigner3j(a,b,c,d,e,f);angularValues.set(key,value);return value;}
const H=6.62607015e-34,EPS=8.8541878128e-12,C=299792458,EA0=1.602176634e-19*5.29177210903e-11,EV_MHZ=1.602176634e-19/H/1e6,N=32;
/** Atomic unit of polarizability, e^2 a0^2 / Eh, in C m^2 / V (CODATA 2018). */
const POLARIZABILITY_AU=1.64877727436e-41;
type Z={re:number;im:number};
type Matrix={real:number[];imag:number[]};
type Group={beams:Laser[];frequency:number;field:Z[]};
const zero=(n:number):Matrix=>({real:Array(n*n).fill(0),imag:Array(n*n).fill(0)});
const mul=(a:Z,b:Z):Z=>({re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re});
const conj=(a:Z):Z=>({re:a.re,im:-a.im});
const z=(m:Matrix,k:number):Z=>({re:m.real[k]!,im:m.imag[k]!});
const norm=(a:Z)=>Math.hypot(a.re,a.im);
/** Levels-chart background only: the dynamics never uses it (every laser is
 * an explicit drive there, see full-master-model.ts). */
const ASSUMPTIONS=[
  '能级图光移仅用于显示：选为显示光移的激光取 D 线旋转与反旋转二阶光移、32 态以外能级（5S 的 6–9P；5P 的 6–10S、4–9D）的二阶光移和 Rb+ 芯极化，保留偏振和磁子态混合。',
  '动力学不使用这一显示背景：所有开启的激光都是主方程中的显式驱动。',
  '芯极化取 Arora & Sahoo 2012 的静态值（5S 9.1−0.26 au，5P 9.1 au）；未包含更高价态尾项和连续谱，高态超精细/同位素修正未分辨，数据精度不等于求解器数值精度。',
  '分开处理 5S、5P1/2、5P3/2 的二阶光移；不包含高态诱导的两条精细流形间耦合。',
  '图中布居投影到这些流形内部 DC 本征基；它不是含光学激发态混入及微运动的 Floquet 态布居。',
];

type AtomicSystem=ReturnType<typeof atomicSystem>;
const atomicCache=new Map<number,{D1:AtomicSystem;D2:AtomicSystem;states:Eigenstate[]}>();
function cachedAtom(fieldG:number){
  let item=atomicCache.get(fieldG);if(!item){const D1=atomicSystem('D1',fieldG),D2=atomicSystem('D2',fieldG);item={D1,D2,states:[...D1.ground,...D1.excited,...D2.excited]};atomicCache.set(fieldG,item);if(atomicCache.size>8)atomicCache.delete(atomicCache.keys().next().value!);}return item;
}
function basis(analysis:Analysis):Eigenstate[]{return cachedAtom(analysis.fieldMagnitudeG).states;}
function groups(scene:Scene,analysis:Analysis,excluded?:string):Group[]{
  const result:Group[]=[];
  for(const beam of scene.lasers)if(beam.enabled&&beam.id!==excluded&&laserIntensityMWcm2(beam)!==0){
    let group=result.find(item=>laserFrequencyDifferenceMHz(item.beams[0]!,beam)===0);
    if(!group){group={beams:[],frequency:laserFrequencyMHz(beam),field:[]};result.push(group);}group.beams.push(beam);
  }
  for(const group of result){
    const fields=group.beams.map(beam=>{const p=complexPolarization(beam,analysis.fieldVector,analysis.fieldAxis),phase=sinCosDegrees(beam.phaseDeg??0),scale=Math.sqrt(2*laserIntensityMWcm2(beam)*10/(EPS*C));return p.map(q=>({re:scale*(q.re*phase.cos-q.im*phase.sin),im:scale*(q.re*phase.sin+q.im*phase.cos)}));});
    group.field=[0,1,2].map(q=>({re:accurateSum(fields.map(f=>f[q]!.re)),im:accurateSum(fields.map(f=>f[q]!.im))}));
  }
  return result.filter(group=>group.field.some(q=>q.re!==0||q.im!==0));
}
function reversedField(field:Z[]):Z[]{return [0,1,2].map(q=>({re:(q===1?1:-1)*field[2-q]!.re,im:(q===1?-1:1)*field[2-q]!.im}));}
function amplitude(d:number[],field:Z[]):Z{return {re:accurateSum(d.map((v,q)=>v*field[q]!.re))/(2*H*1e6),im:accurateSum(d.map((v,q)=>v*field[q]!.im))/(2*H*1e6)};}
function centroid(manifold:Level['manifold']):number{return manifold==='ground'?0:RB87[manifold].frequencyHz/1e6;}

interface Correction {matrix:Matrix;maximumMixingRatio:number}
/** The display checks measured instead of enforced: did every gap keep its minimum? */
type Measure={withinGap:boolean};
const correctionCache=new Map<string,Correction>();
type VirtualRow={indices:number[];dipoles:number[][];energyMHz:number;name:string};
const virtualRowsCache=new Map<number,VirtualRow[]>();
/** Angular matrix elements depend on B, not intensity, frequency or Jones
 * vector. Cache every row, including exact zeros; there is no pruning cutoff. */
function virtualRows(states:Eigenstate[],fieldG:number):VirtualRow[]{
  let rows=virtualRowsCache.get(fieldG);if(rows)return rows;rows=[];
  for(const manifold of ['ground','D1','D2'] as const){
    const indices=states.flatMap((s,i)=>s.manifold===manifold?[i]:[]),J=manifold==='ground'?.5:RB87[manifold].J;
    for(const virtual of VIRTUAL_MANIFOLDS[manifold]){
      const gJ=1+(virtual.J*(virtual.J+1)+.75-virtual.L*(virtual.L+1))/(2*virtual.J*(virtual.J+1));
      for(let mi=-1.5;mi<=1.5;mi++)for(let mj=-virtual.J;mj<=virtual.J;mj++){
        const dipoles=indices.map(i=>[-1,0,1].map(q=>accurateSum(states[i]!.basis.map((b,k)=>b.mi===mi&&mj-b.mj===q?states[i]!.vector[k]!*((-1)**Math.round(virtual.J-mj))*wigner3j(virtual.J,1,J,-mj,q,b.mj)*virtual.reducedEA0*EA0:0))));
        const energyMHz=virtual.energyEV*EV_MHZ+RB87.muBOverH*fieldG*(gJ*mj+RB87.gI*mi);rows.push({indices,dipoles,energyMHz,name:virtual.name});
      }
    }
  }
  virtualRowsCache.set(fieldG,rows);if(virtualRowsCache.size>4)virtualRowsCache.delete(virtualRowsCache.keys().next().value!);return rows;
}
/** Symmetric second-order (van Vleck) Hamiltonian. Each virtual row retains all
 * represented amplitudes. The safety guard controls the expansion, not a
 * dipole/level cutoff. Resonant nonzero couplings are an explicit failure. */
function addVirtual(matrix:Matrix,indices:number[],values:Z[],denominators:number[],minimumGap:number,label:string,measure?:Measure):number{
  const magnitudes=values.map(norm),active=magnitudes.flatMap((value,i)=>value!==0?[i]:[]);let ratio2=0;
  for(const a of active){
    const gap=Math.abs(denominators[a]!);if(!(gap>=minimumGap)){if(!measure)throw new Error(`${label}：虚跃迁失谐小于 ${minimumGap} MHz，超出静态色散光移模型。`);measure.withinGap=false;if(gap===0){ratio2=Infinity;continue;}}
    ratio2+=(magnitudes[a]!/gap)**2;
  }
  const ratio=Math.sqrt(ratio2);
  // Measured only: the same ratio, nothing added and nothing refused.
  if(measure)return ratio;
  if(!(ratio<=.01))throw new Error(`${label}：光耦合/失谐比 ${ratio.toPrecision(3)} 超过二阶色散模型范围。`);
  for(let ai=0;ai<active.length;ai++)for(let bi=ai;bi<active.length;bi++){
    const a=active[ai]!,b=active[bi]!;
    const ab=mul(conj(values[a]!),values[b]!),factor=.5*(1/denominators[a]!+1/denominators[b]!),i=indices[a]!,j=indices[b]!,k=i*N+j;
    matrix.real[k]!+=factor*ab.re;matrix.imag[k]!+=factor*ab.im;
    if(i!==j){matrix.real[j*N+i]=matrix.real[k]!;matrix.imag[j*N+i]=-matrix.imag[k]!;}
  }
  return ratio;
}

/** The second-order DC correction in the magnetic eigenbasis. `dLines` adds
 * the D-line rotating and counter-rotating terms (levels-chart display only);
 * states outside the 32-state model and the Rb+ core are always included. The
 * master model calls this with dLines=false: its D-line couplings are explicit. */
function correction(states:Eigenstate[],analysis:Analysis,beams:Group[],dLines:boolean,measure?:Measure):Correction{
  // Cache physical inputs only: selecting another displayed state/reference
  // cannot alter the correction or retrigger the virtual-state summation.
  const key=JSON.stringify([analysis.fieldMagnitudeG,dLines,beams.map(g=>[laserCarrierOffsetMHz(g.beams[0]!,'D1'),laserCarrierOffsetMHz(g.beams[0]!,'D2'),g.field])]);
  const cached=measure?undefined:correctionCache.get(key);if(cached)return cached;
  const matrix=zero(N);let maximumMixingRatio=0;
  for(const group of beams){
    const reverse=reversedField(group.field);
    if(dLines)for(const line of ['D1','D2'] as const){
      const system=cachedAtom(analysis.fieldMagnitudeG)[line],offset=line==='D1'?8:16,excited=system.excited;
      // Excited<-ground absorption and its conjugate stimulated-emission row.
      for(const counter of [false,true]){
        const field=counter?reverse:group.field,up:Z[][]=excited.map((e,ei)=>system.ground.map((g,gi)=>{const q=e.m-g.m,d=[0,0,0];if(q===-1||q===0||q===1)d[q+1]=system.dipoles[ei*8+gi]!*RB87.D2.dipole;return amplitude(d,field);}));
        for(let e=0;e<excited.length;e++){
          const delta=system.ground.map(g=>counter?g.energyMHz-excited[e]!.energyMHz-centroid(line)-group.frequency:laserCarrierOffsetMHz(group.beams[0]!,line)+g.energyMHz-excited[e]!.energyMHz);
          maximumMixingRatio=Math.max(maximumMixingRatio,addVirtual(matrix,[0,1,2,3,4,5,6,7],up[e]!,delta,100,`${line} / ${group.beams.map(b=>b.name).join(' + ')}`,measure));
        }
        for(let g=0;g<8;g++){
          const delta=excited.map(e=>counter?centroid(line)+e.energyMHz-system.ground[g]!.energyMHz+group.frequency:e.energyMHz-system.ground[g]!.energyMHz-laserCarrierOffsetMHz(group.beams[0]!,line));
          maximumMixingRatio=Math.max(maximumMixingRatio,addVirtual(matrix,excited.map((_,i)=>offset+i),up.map(row=>conj(row[g]!)),delta,100,`${line} / ${group.beams.map(b=>b.name).join(' + ')}`,measure));
        }
      }
    }
    for(const row of virtualRows(states,analysis.fieldMagnitudeG)){
      for(const counter of [false,true]){
        const values=row.dipoles.map(d=>amplitude(d,counter?reverse:group.field));
        const delta=row.indices.map(i=>centroid(states[i]!.manifold)+states[i]!.energyMHz-row.energyMHz+(counter?-group.frequency:group.frequency));
        maximumMixingRatio=Math.max(maximumMixingRatio,addVirtual(matrix,row.indices,values,delta,10000,`${row.name} 高态 / ${group.beams.map(b=>b.name).join(' + ')}`,measure));
      }
    }
    // Rb+ core: the scalar shift -alpha |E|^2 / 4 of every state of a manifold
    // (E is the coherent field amplitude of this carrier group).
    const squared=accurateSum(group.field.flatMap(q=>[q.re*q.re,q.im*q.im]));
    states.forEach((state,i)=>{matrix.real[i*N+i]!+=-CORE_POLARIZABILITY_AU[state.manifold]*POLARIZABILITY_AU*squared/(4*H*1e6);});
  }
  const result={matrix,maximumMixingRatio};if(measure)return result;correctionCache.set(key,result);if(correctionCache.size>12)correctionCache.delete(correctionCache.keys().next().value!);return result;
}


/** Cyclic complex-Hermitian Jacobi. No amplitude pruning; a rotation is skipped
 * only when its represented off-diagonal entry is exactly zero. The residual
 * is checked against the input after convergence. Matrices have size <=16. */
export function diagonalizeSpectroscopyHermitian(input:Matrix,n:number):{values:number[];vectors:Matrix;residual:number}{
  const a={real:input.real.slice(),imag:input.imag.slice()},v=zero(n);for(let i=0;i<n;i++)v.real[i*n+i]=1;
  const scale=Math.max(1,...input.real.map(Math.abs),...input.imag.map(Math.abs));
  for(let sweep=0;sweep<60;sweep++){
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
      const apq=z(a,p*n+q),r=norm(apq);if(r===0)continue;
      const phase={re:apq.re/r,im:apq.im/r},tau=(a.real[q*n+q]!-a.real[p*n+p]!)/(2*r),t=tau===0?1:Math.sign(tau)/(Math.abs(tau)+Math.hypot(1,tau)),c=1/Math.hypot(1,t),s=t*c;
      if(s===0)continue;
      for(let k=0;k<n;k++){
        if(k!==p&&k!==q){const kp=z(a,k*n+p),kq=z(a,k*n+q),cq=mul(conj(phase),kq),cp=mul(phase,kp),np={re:c*kp.re-s*cq.re,im:c*kp.im-s*cq.im},nq={re:s*cp.re+c*kq.re,im:s*cp.im+c*kq.im};
          a.real[k*n+p]=a.real[p*n+k]=np.re;a.imag[k*n+p]=np.im;a.imag[p*n+k]=-np.im;a.real[k*n+q]=a.real[q*n+k]=nq.re;a.imag[k*n+q]=nq.im;a.imag[q*n+k]=-nq.im;}
        const vp=z(v,k*n+p),vq=z(v,k*n+q),cq=mul(conj(phase),vq),cp=mul(phase,vp);v.real[k*n+p]=c*vp.re-s*cq.re;v.imag[k*n+p]=c*vp.im-s*cq.im;v.real[k*n+q]=s*cp.re+c*vq.re;v.imag[k*n+q]=s*cp.im+c*vq.im;
      }
      a.real[p*n+p]!-=t*r;a.real[q*n+q]!+=t*r;a.real[p*n+q]=a.real[q*n+p]=a.imag[p*n+q]=a.imag[q*n+p]=0;
    }
    let off=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)off=Math.max(off,norm(z(a,i*n+j)));if(off<=scale*Number.EPSILON*2)break;
  }
  const values=Array.from({length:n},(_,i)=>a.real[i*n+i]!);let residual=0;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){const products=Array.from({length:n},(_,k)=>mul(z(input,i*n+k),z(v,k*n+j)));residual=Math.max(residual,Math.hypot(accurateSum(products.map(x=>x.re))-values[j]!*v.real[i*n+j]!,accurateSum(products.map(x=>x.im))-values[j]!*v.imag[i*n+j]!));}
  if(!(residual<=scale*1e-11))throw new Error(`背景 Hamiltonian 对角化未收敛 (${residual} MHz)。`);
  return {values,vectors:v,residual};
}
/** Minimum-cost assignment, avoiding duplicate labels at level crossings. */
function labels(v:Matrix,n:number):number[]{
  const u=Array(n+1).fill(0),w=Array(n+1).fill(0),p=Array(n+1).fill(0),way=Array(n+1).fill(0);
  for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=Array(n+1).fill(Infinity),used=Array(n+1).fill(false);
    do{used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=n;j++)if(!used[j]){const k=(i0-1)*n+j-1,cur=-(v.real[k]!**2+v.imag[k]!**2)-u[i0]-w[j];if(cur<min[j]){min[j]=cur;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}for(let j=0;j<=n;j++){if(used[j]){u[p[j]]+=delta;w[j]-=delta;}else min[j]-=delta;}j0=j1;}while(p[j0]!==0);
    do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);
  }
  const result=Array(n).fill(0);for(let j=1;j<=n;j++)result[p[j]-1]=j-1;return result;
}
function identityState(index:number):Level['backgroundState']{return {real:Array.from({length:N},(_,i)=>i===index?1:0),imag:Array(N).fill(0)};}
const allLevels=(analysis:Analysis)=>[...analysis.ground,...analysis.D1,...analysis.D2];
type DipoleEntry={fieldG:number;ground:Matrix;excited:Matrix;dipole:Matrix};
type ProjectionPlan={ids:string[];vectors:Matrix[];support:number[][];key:string};
type BasisContext={dipoles:Map<string,DipoleEntry>;projection?:ProjectionPlan};
type Background={ground:Level[];D1:Level[];D2:Level[];context:BasisContext};
const backgroundCache=new WeakMap<Correction,Background>();
const analysisContexts=new WeakMap<Analysis,BasisContext>();
const contextFor=(analysis:Analysis):BasisContext=>{let value=analysisContexts.get(analysis);if(!value){value={dipoles:new Map()};analysisContexts.set(analysis,value);}return value;};
function copyLevel(level:Level):Level{return {...level,backgroundState:level.backgroundState?{real:level.backgroundState.real.slice(),imag:level.backgroundState.imag.slice()}:undefined};}
function background(analysis:Analysis,result:Correction):Background{
  const cached=backgroundCache.get(result);if(cached)return cached;
  const zeroStates=cachedAtom(0).states,out:Background={ground:[],D1:[],D2:[],context:{dipoles:new Map()}};
  for(const [offset,count,key] of [[0,8,'ground'],[8,8,'D1'],[16,16,'D2']] as const){
    const original=analysis[key],h=zero(count);for(let i=0;i<count;i++)for(let j=0;j<count;j++){h.real[i*count+j]=result.matrix.real[(offset+i)*N+offset+j]!+(i===j?original[i]!.energyMHz:0);h.imag[i*count+j]=result.matrix.imag[(offset+i)*N+offset+j]!;}
    const eig=diagonalizeSpectroscopyHermitian(h,count),assignment=labels(eig.vectors,count);
    out[key]=original.map((state,i)=>{const col=assignment[i]!,vector={real:Array(N).fill(0),imag:Array(N).fill(0)};for(let k=0;k<count;k++){vector.real[offset+k]=eig.vectors.real[k*count+col]!;vector.imag[offset+k]=eig.vectors.imag[k*count+col]!;}const mixed=vector.real.some((x,k)=>k!==offset+i&&(x!==0||vector.imag[k]!==0));return {...state,energyMHz:eig.values[col]!,shiftMHz:eig.values[col]!-zeroStates[offset+i]!.energyMHz,totalShiftMHz:eig.values[col]!-zeroStates[offset+i]!.energyMHz,starkShiftMHz:eig.values[col]!-state.energyMHz,backgroundState:vector,mixed};});
  }
  backgroundCache.set(result,out);return out;
}

/** The static part of the master-model H0 and its eigenbasis: magnetic
 * eigenstates plus, for every enabled laser, the second-order shift through
 * states OUTSIDE the 32-state model and the Rb+ core, with the same validity
 * check as the display. The D-line couplings are explicit drives of the
 * master equation and are not part of it (no double counting). */
export function staticModelBackground(scene:Scene):{analysis:Analysis;levels:Level[];maximumMixingRatio:number;laserIds:string[]}{
  const analysis=analyze(scene),beams=groups(scene,analysis),result=correction(basis(analysis),analysis,beams,false),saved=background(analysis,result);
  return {analysis,levels:[...saved.ground,...saved.D1,...saved.D2].map(copyLevel),maximumMixingRatio:result.maximumMixingRatio,laserIds:beams.flatMap(g=>g.beams.map(b=>b.id))};
}

/** A laser's largest mixing amplitude |Ω/2Δ| in the levels chart's display model -- the
 * ratio that model refuses above 1% when the laser's light shift is shown -- and whether
 * each of its virtual transitions keeps that model's minimum detuning. A hint for the
 * operator; it changes nothing. */
export function displayMixing(scene:Scene,laserId:string):{ratio:number;withinGap:boolean}{
  const laser=scene.lasers.find(l=>l.id===laserId);if(!laser||laserIntensityMWcm2(laser)===0)return {ratio:0,withinGap:true};
  const analysis=analyze(scene),beams=groups({...scene,lasers:[{...laser,enabled:true}]},analysis),measure:Measure={withinGap:true};
  return {ratio:correction(basis(analysis),analysis,beams,true,measure).maximumMixingRatio,withinGap:measure.withinGap};
}
export function spectroscopyAnalysis(scene:Scene,excludedLaserId?:string,includeTransitions=true):Analysis{
  const start=performance.now(),analysis=analyze(scene),states=basis(analysis),beams=groups({...scene,lasers:scene.lasers.filter(l=>l.regime==='far')},analysis,excludedLaserId);
  let result:Correction|undefined,failure='';
  // Independent far-detuned fields contribute DC shifts under the declared model.
  if(!failure)try{result=correction(states,analysis,beams,true);}catch(error){failure=error instanceof Error?error.message:String(error);}
  const status=failure?'unresolved':beams.length?'dispersive':'bare';
  analysis.spectroscopy={status,backgroundLaserIds:beams.flatMap(g=>g.beams.map(b=>b.id)),excludedLaserId,modelScope:failure?'裸态位置/布居；静态总光移未定义。':beams.length?'磁场本征基上的 DC 二阶背景谱（仅显示）；含偏振混合、D 线旋转与反旋转、列明高态和芯极化，非全原子/Floquet 精确谱。':'HFS + Zeeman 裸态谱。',nearResonantLaserIds:failure&&beams.length===1?beams.flatMap(g=>g.beams.map(b=>b.id)):[],requiresDensityProjection:false,warnings:failure?[failure,...ASSUMPTIONS]:beams.length?ASSUMPTIONS.slice():[],maximumMixingRatio:result?.maximumMixingRatio,sources:SPECTROSCOPY_SOURCES.slice()};
  if(status==='dispersive'){
    const saved=background(analysis,result!);for(const key of ['ground','D1','D2'] as const)analysis[key]=saved[key].map(copyLevel);analysisContexts.set(analysis,saved.context);
  }else for(const [offset,key] of [[0,'ground'],[8,'D1'],[16,'D2']] as const)analysis[key]=analysis[key].map((state,i)=>({...state,totalShiftMHz:failure?null:state.shiftMHz,starkShiftMHz:failure?null:0,backgroundState:identityState(offset+i),mixed:false}));
  analysis.spectroscopy.requiresDensityProjection=status==='dispersive'&&allLevels(analysis).some(s=>s.mixed);
  analysis.transitions=[];
  if(includeTransitions&&status!=='unresolved')for(const laser of scene.lasers)if(laser.enabled){
    const source=allLevels(analysis).find(s=>s.id===(laser.referenceStateId??scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`));if(!source)continue;
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
    const system=cachedAtom(analysis.fieldMagnitudeG)[line],offset=line==='D1'?8:16,d={real:[0,0,0],imag:[0,0,0]};
    for(let q=-1;q<=1;q++){const terms:Z[]=[];for(let ei=0;ei<system.excited.length;ei++)for(let gi=0;gi<8;gi++)if(system.excited[ei]!.m-system.ground[gi]!.m===q){const amplitude=system.dipoles[ei*8+gi]!*RB87.D2.dipole,c=mul({re:ue.real[offset+ei]!,im:-ue.imag[offset+ei]!},{re:ug.real[gi]!,im:ug.imag[gi]!});terms.push({re:amplitude*c.re,im:amplitude*c.im});}d.real[q+1]=accurateSum(terms.map(t=>t.re));d.imag[q+1]=accurateSum(terms.map(t=>t.im));}
    item={fieldG:analysis.fieldMagnitudeG,ground:{real:ug.real.slice(),imag:ug.imag.slice()},excited:{real:ue.real.slice(),imag:ue.imag.slice()},dipole:d};cache.set(key,item);
  }
  const d=item.dipole;
  const components=[0,1,2].map(q=>Math.hypot(d.real[q]!,d.imag[q]!)),amplitudeCM=Math.hypot(...components),p=complexPolarization(laser,analysis.fieldVector,analysis.fieldAxis),coupled=p.map((x,k)=>mul(x,{re:d.real[k]!,im:d.imag[k]!})),rabiMHz=Math.hypot(accurateSum(coupled.map(x=>x.re)),accurateSum(coupled.map(x=>x.im)))*Math.sqrt(2*laserIntensityMWcm2(laser)*10/(EPS*C))/H/1e6;
  const driven= coupled.map(z=>Math.hypot(z.re,z.im)),displayComponents=driven.some(v=>v!==0)?driven:components,q=displayComponents.indexOf(Math.max(...displayComponents))-1;
  // This tolerance is for the label only; no Hamiltonian/dipole entry is dropped.
  const mixedM=(state:Level,v:Matrix)=>states.reduce((sum,bare,i)=>sum+(bare.m!==state.m?v.real[i]!**2+v.imag[i]!**2:0),0)>1e-8;
  const mLabelApproximate=mixedM(g,ug)||mixedM(e,ue);
  const ref=laser.referenceTransition,offsetMHz=(RB87[line].frequencyHz-RB87[laser.line].frequencyHz)/1e6+e.energyMHz-g.energyMHz-transitionReferenceMHz(laser.line,ref?.Fg,ref?.Fe,ref?.fromManifold);
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
