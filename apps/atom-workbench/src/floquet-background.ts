/** The far lasers' static background: the quasi-energies of their Floquet problem, to
 * all orders, by energy-dependent Löwdin (Brillouin–Wigner) partitioning.
 *
 * States: the 32 magnetic eigenstates and, for each of their three fine-structure
 * manifolds, the states outside the model (spectroscopy-data.ts), which couple only to
 * that manifold. Far carrier s has frequency ω_s; a Floquet copy has a charge n ∈ Z^K
 * and energy E + n·ω. Absorption couples an excited state at charge n to a lower state
 * at n + e_s (the rotating part) and at n − e_s (the counter-rotating part). Ground and
 * excited copies alternate with the parity of Σn, so the 8 ground and the 24 excited
 * states at charge 0 are two independent problems (sectors).
 *
 * P is a sector's states at charge 0; Q is every other in-model copy with |n|₁ ≤ L
 * (layers 1..L, L = 2 first) and the outside copies they reach. Outside states have no
 * couplings of their own, so they are eliminated exactly: between in-model copies n, n'
 * they add Σ(E) = Σ_v u_n^† u_n' / (E − E_v), kept for |n|₁ + |n'|₁ ≤ 2(L − 1). A cluster
 * of nearby levels at energy E_c gets the dressed columns X = [I; Φ(E_c)],
 * Φ = (E_c − H_QQ)⁻¹ H_QP, by block Gauss–Seidel over the layers; the linearized
 * Brillouin–Wigner equation is then the Rayleigh–Ritz problem M c = θ S c, M = X†HX,
 * S = X†X (the outside states' weight −Σ' included). E_c is moved to the cluster's levels
 * (self-consistency), clusters are split and L raised until each level's bound meets
 * TARGET_MHZ.
 *
 * Bounds are a-posteriori (clusterStep): the Kato–Temple bound of each Ritz pair on the
 * kept space, plus what reaches the left-out copies (second order) and the outside pairs
 * not kept (first order). They are numerical bounds only: the atomic data's own
 * uncertainty is not part of them. */
import {RB87,accurateSum} from './physics.js';

export type Manifold='ground'|'D1'|'D2';
export interface Complex {re:number;im:number}
/** Row-major complex matrix. */
export interface CMatrix {rows:number;cols:number;re:Float64Array;im:Float64Array}
export const cmatrix=(rows:number,cols:number):CMatrix=>({rows,cols,re:new Float64Array(rows*cols),im:new Float64Array(rows*cols)});
/** A state outside the model: its name, absolute energy and real dipole components
 * (q = −1, 0, +1, C·m) to each state of its manifold, row-major n_m × 3. */
export interface OutsideState {name:string;energyMHz:number;dipoles:Float64Array}
export interface FarCarrier {
  name:string;
  /** ω − ω(5P3/2 centroid), MHz. */
  offsetMHz:number;
  /** Spherical components q = −1, 0, +1 of the field's rotating (e^{−iωt}) and
   * counter-rotating (e^{+iωt}) parts divided by 2h: MHz per C·m. */
  rotatingField:Complex[];counterField:Complex[];
  /** ⟨e|V|g⟩ of the two parts, MHz, 24 excited × 8 ground (state order 8–31 × 0–7). */
  rotating:CMatrix;counter:CMatrix;
}
export interface BackgroundProblem {
  /** HFS + Zeeman energies of the 32 states relative to their manifold's centroid, MHz. */
  energies:Float64Array;
  /** Each state's static scalar shift (Rb+ core), MHz. */
  core:Float64Array;
  carriers:FarCarrier[];
  outside:Record<Manifold,OutsideState[]>;
}
/** One level of the background: label index 0–31, energy relative to its manifold's
 * centroid, its normalized state at charge 0 in the bare magnetic basis, and bounds. */
export interface BackgroundLevel {index:number;energyMHz:number;vector:{re:Float64Array;im:Float64Array};floquetBoundMHz:number;outsideBoundMHz:number}
export interface BackgroundSolution {levels:BackgroundLevel[];layers:number;clusters:number;solves:number}

/** Each level's bound is driven below this (0.05 Hz). */
export const TARGET_MHZ=5e-8;
const MAX_LAYERS=4,FIRST_LAYERS=2,CLUSTER_GAP_MHZ=500,MAX_SWEEPS=200;
const C_D2=RB87.D2.frequencyHz/1e6;
/** Centroid minus the 5P3/2 centroid. */
const CENTROID:Record<Manifold,number>={ground:-C_D2,D1:(RB87.D1.frequencyHz-RB87.D2.frequencyHz)/1e6,D2:0};
const MANIFOLD_OF=(i:number):Manifold=>i<8?'ground':i<16?'D1':'D2';
const BLOCKS={ground:[{manifold:'ground' as Manifold,local:0,global:0,size:8}],excited:[{manifold:'D1' as Manifold,local:0,global:8,size:8},{manifold:'D2' as Manifold,local:8,global:16,size:16}]};

// ---------------------------------------------------------------- dense complex kernels
/** out += op(A) B, op = A or A^†. */
function multiply(out:CMatrix,a:CMatrix,b:CMatrix,adjoint=false):void{
  const n=b.cols,ore=out.re,oim=out.im,are=a.re,aim=a.im,bre=b.re,bim=b.im;
  if(!adjoint){const rows=a.rows,inner=a.cols;
    for(let i=0;i<rows;i++)for(let k=0;k<inner;k++){const xr=are[i*inner+k]!,xi=aim[i*inner+k]!;if(xr===0&&xi===0)continue;const o=i*n,bk=k*n;for(let j=0;j<n;j++){const yr=bre[bk+j]!,yi=bim[bk+j]!;ore[o+j]+=xr*yr-xi*yi;oim[o+j]+=xr*yi+xi*yr;}}
  }else{const rows=a.cols,inner=a.rows;
    for(let k=0;k<inner;k++)for(let i=0;i<rows;i++){const xr=are[k*rows+i]!,xi=-aim[k*rows+i]!;if(xr===0&&xi===0)continue;const o=i*n,bk=k*n;for(let j=0;j<n;j++){const yr=bre[bk+j]!,yi=bim[bk+j]!;ore[o+j]+=xr*yr-xi*yi;oim[o+j]+=xr*yi+xi*yr;}}
  }
}
/** A^† B, accumulated into out. */
function gram(out:CMatrix,a:CMatrix,b:CMatrix):void{multiply(out,a,b,true);}
/** out += op(A), op = A or A^†. */
function addTo(out:CMatrix,a:CMatrix,adjoint=false):void{
  if(!adjoint){for(let k=0;k<a.re.length;k++){out.re[k]+=a.re[k]!;out.im[k]+=a.im[k]!;}return;}
  const rows=a.rows,cols=a.cols;for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){out.re[j*rows+i]+=a.re[i*cols+j]!;out.im[j*rows+i]-=a.im[i*cols+j]!;}
}
/** LU with partial pivoting, in place; returns the pivots. */
function factor(a:CMatrix):Int32Array{
  const n=a.rows,re=a.re,im=a.im,pivot=new Int32Array(n);
  for(let k=0;k<n;k++){
    let p=k,best=-1;for(let i=k;i<n;i++){const v=re[i*n+k]!**2+im[i*n+k]!**2;if(v>best){best=v;p=i;}}
    pivot[k]=p;if(p!==k)for(let j=0;j<n;j++){let t=re[k*n+j]!;re[k*n+j]=re[p*n+j]!;re[p*n+j]=t;t=im[k*n+j]!;im[k*n+j]=im[p*n+j]!;im[p*n+j]=t;}
    const dr=re[k*n+k]!,di=im[k*n+k]!,d=dr*dr+di*di;if(!(d>0))throw new Error('Floquet 副本的预解式奇异。');
    for(let i=k+1;i<n;i++){const xr=re[i*n+k]!,xi=im[i*n+k]!;if(xr===0&&xi===0)continue;const lr=(xr*dr+xi*di)/d,li=(xi*dr-xr*di)/d;re[i*n+k]=lr;im[i*n+k]=li;
      for(let j=k+1;j<n;j++){const ur=re[k*n+j]!,ui=im[k*n+j]!;re[i*n+j]-=lr*ur-li*ui;im[i*n+j]-=lr*ui+li*ur;}}
  }
  return pivot;
}
/** Solve LU x = b for every column of b, in place. */
function solve(lu:CMatrix,pivot:Int32Array,b:CMatrix):void{
  const n=lu.rows,m=b.cols,re=lu.re,im=lu.im,br=b.re,bi=b.im;
  for(let k=0;k<n;k++){const p=pivot[k]!;if(p!==k)for(let j=0;j<m;j++){let t=br[k*m+j]!;br[k*m+j]=br[p*m+j]!;br[p*m+j]=t;t=bi[k*m+j]!;bi[k*m+j]=bi[p*m+j]!;bi[p*m+j]=t;}}
  for(let i=0;i<n;i++)for(let k=0;k<i;k++){const lr=re[i*n+k]!,li=im[i*n+k]!;if(lr===0&&li===0)continue;for(let j=0;j<m;j++){const xr=br[k*m+j]!,xi=bi[k*m+j]!;br[i*m+j]-=lr*xr-li*xi;bi[i*m+j]-=lr*xi+li*xr;}}
  for(let i=n-1;i>=0;i--){for(let k=i+1;k<n;k++){const ur=re[i*n+k]!,ui=im[i*n+k]!;if(ur===0&&ui===0)continue;for(let j=0;j<m;j++){const xr=br[k*m+j]!,xi=bi[k*m+j]!;br[i*m+j]-=ur*xr-ui*xi;bi[i*m+j]-=ur*xi+ui*xr;}}
    const dr=re[i*n+i]!,di=im[i*n+i]!,d=dr*dr+di*di;for(let j=0;j<m;j++){const xr=br[i*m+j]!,xi=bi[i*m+j]!;br[i*m+j]=(xr*dr+xi*di)/d;bi[i*m+j]=(xi*dr-xr*di)/d;}}
}
/** Cyclic complex-Hermitian Jacobi (n ≤ 32): eigenvalues and unitary eigenvectors (columns). */
function hermitianEigen(input:CMatrix):{values:Float64Array;vectors:CMatrix}{
  const n=input.rows,are=Float64Array.from(input.re),aim=Float64Array.from(input.im),v=cmatrix(n,n);for(let i=0;i<n;i++)v.re[i*n+i]=1;
  for(let i=0;i<n;i++){aim[i*n+i]=0;for(let j=i+1;j<n;j++){const r=(are[i*n+j]!+are[j*n+i]!)/2,m=(aim[i*n+j]!-aim[j*n+i]!)/2;are[i*n+j]=are[j*n+i]=r;aim[i*n+j]=m;aim[j*n+i]=-m;}}
  let scale=0;for(let k=0;k<n*n;k++)scale=Math.max(scale,Math.abs(are[k]!),Math.abs(aim[k]!));
  for(let sweep=0;sweep<80;sweep++){
    let off=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)off=Math.max(off,Math.hypot(are[i*n+j]!,aim[i*n+j]!));
    if(off<=scale*Number.EPSILON)break;
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
      const xr=are[p*n+q]!,xi=aim[p*n+q]!,r=Math.hypot(xr,xi);if(r===0)continue;
      const er=xr/r,ei=xi/r,tau=(are[q*n+q]!-are[p*n+p]!)/(2*r),t=Math.sign(tau||1)/(Math.abs(tau)+Math.hypot(1,tau)),c=1/Math.hypot(1,t),s=t*c;
      if(s===0)continue;
      for(let k=0;k<n;k++){
        if(k!==p&&k!==q){const kpr=are[k*n+p]!,kpi=aim[k*n+p]!,kqr=are[k*n+q]!,kqi=aim[k*n+q]!;
          // a_kp' = c a_kp − s conj(e) a_kq ; a_kq' = s e a_kp + c a_kq
          const npr=c*kpr-s*(er*kqr+ei*kqi),npi=c*kpi-s*(er*kqi-ei*kqr),nqr=s*(er*kpr-ei*kpi)+c*kqr,nqi=s*(er*kpi+ei*kpr)+c*kqi;
          are[k*n+p]=are[p*n+k]=npr;aim[k*n+p]=npi;aim[p*n+k]=-npi;are[k*n+q]=are[q*n+k]=nqr;aim[k*n+q]=nqi;aim[q*n+k]=-nqi;}
        const vpr=v.re[k*n+p]!,vpi=v.im[k*n+p]!,vqr=v.re[k*n+q]!,vqi=v.im[k*n+q]!;
        v.re[k*n+p]=c*vpr-s*(er*vqr+ei*vqi);v.im[k*n+p]=c*vpi-s*(er*vqi-ei*vqr);v.re[k*n+q]=s*(er*vpr-ei*vpi)+c*vqr;v.im[k*n+q]=s*(er*vpi+ei*vpr)+c*vqi;
      }
      are[p*n+p]-=t*r;are[q*n+q]+=t*r;are[p*n+q]=are[q*n+p]=aim[p*n+q]=aim[q*n+p]=0;
    }
  }
  return {values:Float64Array.from({length:n},(_,i)=>are[i*n+i]!),vectors:v};
}
/** Cholesky S = L L^† of a Hermitian positive-definite matrix (lower L). */
function cholesky(s:CMatrix):CMatrix{
  const n=s.rows,l=cmatrix(n,n);
  for(let j=0;j<n;j++){
    let d=s.re[j*n+j]!;for(let k=0;k<j;k++)d-=l.re[j*n+k]!**2+l.im[j*n+k]!**2;
    if(!(d>0))throw new Error('Floquet 缀饰基的度规不正定。');const root=Math.sqrt(d);l.re[j*n+j]=root;
    for(let i=j+1;i<n;i++){let xr=s.re[i*n+j]!,xi=s.im[i*n+j]!;for(let k=0;k<j;k++){const ar=l.re[i*n+k]!,ai=l.im[i*n+k]!,br=l.re[j*n+k]!,bi=-l.im[j*n+k]!;xr-=ar*br-ai*bi;xi-=ar*bi+ai*br;}l.re[i*n+j]=xr/root;l.im[i*n+j]=xi/root;}
  }
  return l;
}
/** The generalized Hermitian problem M c = θ S c: θ ascending, columns c with c^† S c = 1. */
function generalizedEigen(m:CMatrix,s:CMatrix):{values:Float64Array;vectors:CMatrix}{
  const n=m.rows,l=cholesky(s),a=cmatrix(n,n),t=cmatrix(n,n);
  // t = L⁻¹ M ; a = t L^{-†} = (L⁻¹ t^†)^†
  const forward=(src:CMatrix,dst:CMatrix,adjointSource:boolean)=>{for(let j=0;j<n;j++)for(let i=0;i<n;i++){let xr=adjointSource?src.re[j*n+i]!:src.re[i*n+j]!,xi=adjointSource?-src.im[j*n+i]!:src.im[i*n+j]!;for(let k=0;k<i;k++){const lr=l.re[i*n+k]!,li=l.im[i*n+k]!,yr=dst.re[k*n+j]!,yi=dst.im[k*n+j]!;xr-=lr*yr-li*yi;xi-=lr*yi+li*yr;}const d=l.re[i*n+i]!;dst.re[i*n+j]=xr/d;dst.im[i*n+j]=xi/d;}};
  forward(m,t,false);forward(t,a,true);
  const eig=hermitianEigen(a),order=Array.from({length:n},(_,i)=>i).sort((p,q)=>eig.values[p]!-eig.values[q]!),c=cmatrix(n,n),values=new Float64Array(n);
  // c = L^{-†} z: back substitution with L^†.
  for(const [col,src] of order.entries()){values[col]=eig.values[src]!;
    for(let i=n-1;i>=0;i--){let xr=eig.vectors.re[i*n+src]!,xi=eig.vectors.im[i*n+src]!;for(let k=i+1;k<n;k++){const lr=l.re[k*n+i]!,li=-l.im[k*n+i]!,yr=c.re[k*n+col]!,yi=c.im[k*n+col]!;xr-=lr*yr-li*yi;xi-=lr*yi+li*yr;}const d=l.re[i*n+i]!;c.re[i*n+col]=xr/d;c.im[i*n+col]=xi/d;}}
  return {values,vectors:c};
}

// ---------------------------------------------------------------- the Floquet structure
type Sector='ground'|'excited';
/** A Floquet copy: charge, layer |n|₁, kind, frame energies of its states (HFS + Zeeman +
 * core + n·ω), and the margin by which its couplings can displace its levels. */
interface Copy {charge:Int32Array;layer:number;excited:boolean;dim:number;diag:Float64Array;margin:number}
interface Edge {excited:number;ground:number;coupling:CMatrix}
/** An outside state: its energy and nonzero dipoles to its manifold per q (index, value). */
/** An outside state: energy; the products d_iq d_jr of its dipoles as flat entries
 * (k·256 + i·n + j, value) grouped by k = 3q + r (group k is entries[start[k]..start[k+1]]);
 * the norm of its couplings through each slot, and through all. */
interface OutsideRow {energyMHz:number;entry:Int32Array;value:Float64Array;start:Int32Array;slots:Float64Array;norm:number}
/** The states of one outside fine-structure level (its Zeeman sublevels): energy range,
 * Σ of their squared coupling norms per slot, and the largest norm of one of them. */
interface OutsideLevel {low:number;high:number;slots:Float64Array;norm:number}
/** A manifold's outside states, grouped into levels, and for each pair of spherical
 * components (q, r) the entries (i, j) where some state has both d_iq and d_jr nonzero. */
interface OutsideManifold {rows:OutsideRow[];levels:OutsideLevel[];support:Int32Array[]}
interface Prepared {problem:BackgroundProblem;K:number;nu:number;mu:number;outside:Record<Manifold,OutsideManifold>;/** HFS + Zeeman + core of each state. */levels:Float64Array}
const chargeKey=(n:Int32Array)=>n.join(',');
const layerOf=(n:Int32Array)=>n.reduce((sum,x)=>sum+Math.abs(x),0);
const frobenius=(m:CMatrix)=>{let s=0;for(let k=0;k<m.re.length;k++)s+=m.re[k]!**2+m.im[k]!**2;return Math.sqrt(s);};
/** Slot 2s is carrier s's rotating part, 2s + 1 its counter-rotating part: its field over 2h. */
const slotField=(problem:BackgroundProblem,slot:number)=>{const c=problem.carriers[slot>>1]!;return slot&1?c.counterField:c.rotatingField;};

function prepare(problem:BackgroundProblem):Prepared{
  const K=problem.carriers.length,outside={} as Record<Manifold,OutsideManifold>;
  for(const manifold of ['ground','D1','D2'] as const){
    const n=manifold==='D2'?16:8,seen=Array.from({length:9},()=>new Uint8Array(n*n));
    const rows=problem.outside[manifold].map(state=>{const slots=new Float64Array(2*K);
      const index:Int32Array[]=[],value:Float64Array[]=[];
      for(let q=0;q<3;q++){const at:number[]=[],val:number[]=[];for(let i=0;i<n;i++){const d=state.dipoles[i*3+q]!;if(d!==0){at.push(i);val.push(d);}}index.push(Int32Array.from(at));value.push(Float64Array.from(val));}
      const entry:number[]=[],product:number[]=[],start=new Int32Array(10);
      for(let q=0;q<3;q++)for(let r=0;r<3;r++){start[q*3+r]=entry.length;index[q]!.forEach((i,x)=>index[r]!.forEach((j,y)=>{seen[q*3+r]![i*n+j]=1;entry.push((q*3+r)*256+i*n+j);product.push(value[q]![x]!*value[r]![y]!);}));}
      start[9]=entry.length;
      for(let slot=0;slot<2*K;slot++){const f=slotField(problem,slot);let sum=0;for(let i=0;i<n;i++){let x=0,y=0;for(let q=0;q<3;q++){const d=state.dipoles[i*3+q]!;x+=d*f[q]!.re;y+=d*f[q]!.im;}sum+=x*x+y*y;}slots[slot]=Math.sqrt(sum);}
      return {energyMHz:state.energyMHz,entry:Int32Array.from(entry),value:Float64Array.from(product),start,slots,norm:Math.hypot(...slots)};
    });
    const levels=new Map<string,OutsideLevel>();
    problem.outside[manifold].forEach((state,i)=>{const row=rows[i]!;let level=levels.get(state.name);if(!level){level={low:Infinity,high:-Infinity,slots:new Float64Array(2*K),norm:0};levels.set(state.name,level);}
      level.low=Math.min(level.low,row.energyMHz);level.high=Math.max(level.high,row.energyMHz);level.norm=Math.max(level.norm,row.norm);for(let k=0;k<2*K;k++)level.slots[k]+=row.slots[k]!**2;});
    outside[manifold]={rows,levels:[...levels.values()],support:seen.map(mask=>Int32Array.from(Array.from(mask).flatMap((x,k)=>x?[k]:[])))};
  }
  // Weyl margin of the in-model Q-like levels: a norm bound of every D-line coupling.
  const nu=problem.carriers.reduce((sum,c)=>sum+frobenius(c.rotating)+frobenius(c.counter),0);
  // The largest coupling norm of a state outside the model.
  const mu=Math.max(0,...(['ground','D1','D2'] as const).flatMap(m=>outside[m].rows.map(r=>r.norm)));
  const levels=Float64Array.from(problem.energies,(x,i)=>accurateSum([x,problem.core[i]!]));
  return {problem,K,nu,mu,outside,levels};
}
/** Frame energy of manifold m at charge n: the ground sector's frame is the ground
 * centroid, the excited sector's the 5P3/2 centroid. */
function frameOffset(p:Prepared,sector:Sector,manifold:Manifold,n:Int32Array):number{
  const N=n.reduce((a,b)=>a+b,0),terms=[(N+(sector==='ground'?1:0))*C_D2,CENTROID[manifold]];
  for(let s=0;s<p.K;s++)if(n[s]!==0)terms.push(n[s]!*p.problem.carriers[s]!.offsetMHz);
  return accurateSum(terms);
}
/** The absolute local energy E + f − n·ω of a copy at charge n for frame energy E. */
function localEnergy(p:Prepared,sector:Sector,E:number,n:Int32Array):number{
  const N=n.reduce((a,b)=>a+b,0),terms=[E,sector==='ground'?0:C_D2,-N*C_D2];
  for(let s=0;s<p.K;s++)if(n[s]!==0)terms.push(-n[s]!*p.problem.carriers[s]!.offsetMHz);
  return accurateSum(terms);
}
/** The frame energies of a copy's states: HFS + Zeeman + core, plus the copy's offset. */
function frameDiag(p:Prepared,sector:Sector,charge:Int32Array,excited:boolean,offset=frameOffset):Float64Array{
  const diag=new Float64Array(excited?24:8);
  for(const b of excited?BLOCKS.excited:BLOCKS.ground){const shift=offset(p,sector,b.manifold,charge);for(let i=0;i<b.size;i++)diag[b.local+i]=p.levels[b.global+i]!+shift;}
  return diag;
}
/** frameOffset in plain arithmetic, for the margins only. */
function roughOffset(p:Prepared,sector:Sector,manifold:Manifold,n:Int32Array):number{
  let N=0,sum=CENTROID[manifold];for(let s=0;s<p.K;s++){N+=n[s]!;sum+=n[s]!*p.problem.carriers[s]!.offsetMHz;}return sum+(N+(sector==='ground'?1:0))*C_D2;
}
/** A copy's levels move by at most ν = ν_D + μ (Weyl: all its D-line and outside couplings);
 * when every coupled neighbour copy lies g ≫ ν away, by at most ν²/(g − 2ν) (second order). */
function makeCopy(p:Prepared,sector:Sector,charge:Int32Array):Copy{
  const N=charge.reduce((a,b)=>a+b,0),excited=(sector==='excited')===(N%2===0),diag=frameDiag(p,sector,charge,excited);
  let g=Infinity;
  for(let s=0;s<p.K;s++)for(const sign of [1,-1]){charge[s]+=sign;const other=frameDiag(p,sector,charge,!excited,roughOffset);charge[s]-=sign;for(const x of diag)for(const y of other)g=Math.min(g,Math.abs(x-y));}
  const nu=p.nu+p.mu;return {charge,layer:layerOf(charge),excited,dim:diag.length,diag,margin:g>2*nu?nu*nu/(g-2*nu):nu};
}
/** Every charge with |n|₁ = layer. */
function shell(K:number,layer:number):Int32Array[]{
  const out:Int32Array[]=[],cur=new Int32Array(K);
  const visit=(s:number,left:number)=>{if(s===K){if(left===0)out.push(cur.slice());return;}for(let v=-left;v<=left;v++){cur[s]=v;visit(s+1,left-Math.abs(v));}cur[s]=0;};
  visit(0,layer);return out;
}
/** Outside copy c and the kept in-model copies it reaches: copy n through slot (s,
 * rotating) when c = n − e_s, (s, counter-rotating) when c = n + e_s. */
interface OutsideCopy {charge:Int32Array;attached:{copy:number;slot:number}[];
  /** Pairs (a ≤ b, indices into attached) kept: |n_a|₁ + |n_b|₁ ≤ 2(L − 1); the others are bounded. */
  pairs:[number,number][];omitted:[number,number][]}
/** `next`: the in-model copies of layers L + 1 and L + 2, which the kept copies reach. */
interface Structure {sector:Sector;layers:number;copies:Copy[];edges:Edge[][];next:Copy[];outside:OutsideCopy[];
  /** D-line couplings from the kept layer L into layer L + 1: kept copy, next copy, norm. */
  leaks:{kept:number;next:number;norm:number}[];
  /** For each outside copy (same order as `outside`): the left-out copies it reaches, by slot. */
  outsideLeaks:{next:number;slot:number}[][]}
function structure(p:Prepared,sector:Sector,layers:number):Structure{
  const copies:Copy[]=[],index=new Map<string,number>();
  for(let layer=0;layer<=layers;layer++)for(const n of shell(p.K,layer)){index.set(chargeKey(n),copies.length);copies.push(makeCopy(p,sector,n));}
  const edges:Edge[][]=copies.map(()=>[]);
  copies.forEach((copy,e)=>{if(!copy.excited)return;
    for(let s=0;s<p.K;s++)for(const [sign,coupling] of [[1,p.problem.carriers[s]!.rotating],[-1,p.problem.carriers[s]!.counter]] as const){
      const n=copy.charge.slice();n[s]+=sign;const g=index.get(chargeKey(n));if(g===undefined)continue;
      const edge={excited:e,ground:g,coupling};edges[e]!.push(edge);edges[g]!.push(edge);}
  });
  // Outside copies reached from the kept in-model copies. Σ is kept between pairs with
  // |n|₁ + |n'|₁ ≤ 2(L − 1) (same kind, so the sum is even); the rest is bounded.
  const groups=new Map<string,OutsideCopy>();
  copies.forEach((copy,i)=>{for(let slot=0;slot<2*p.K;slot++){const c=copy.charge.slice();c[slot>>1]+=slot&1?1:-1;const key=chargeKey(c);let g=groups.get(key);if(!g){g={charge:c,attached:[],pairs:[],omitted:[]};groups.set(key,g);}g.attached.push({copy:i,slot});}});
  const outside=[...groups.values()];
  for(const g of outside)g.attached.forEach((a,i)=>g.attached.forEach((b,j)=>{if(i<=j)(copies[a.copy]!.layer+copies[b.copy]!.layer<=2*(layers-1)?g.pairs:g.omitted).push([i,j]);}));
  const next=[...shell(p.K,layers+1),...shell(p.K,layers+2)].map(n=>makeCopy(p,sector,n)),nextIndex=new Map(next.map((c,i)=>[chargeKey(c.charge),i]));
  const leaks:{kept:number;next:number;norm:number}[]=[];
  copies.forEach((copy,i)=>{if(copy.layer!==layers)return;
    for(let s=0;s<p.K;s++)for(const sign of [1,-1]){const n=copy.charge.slice();n[s]+=sign;const j=nextIndex.get(chargeKey(n));if(j===undefined)continue;
      const c=p.problem.carriers[s]!,norm=frobenius(copy.excited===(sign===1)?c.rotating:c.counter);leaks.push({kept:i,next:j,norm});}});
  const outsideLeaks=outside.map(g=>{const out:{next:number;slot:number}[]=[];
    for(let slot=0;slot<2*p.K;slot++){const n=g.charge.slice();n[slot>>1]-=slot&1?1:-1;const j=nextIndex.get(chargeKey(n));if(j!==undefined)out.push({next:j,slot});}return out;});
  return {sector,layers,copies,edges,next,outside,leaks,outsideLeaks};
}

// ---------------------------------------------------------------- outside states
/** Σ(E) = Σ_v u_n^† u_n' / (w − E_v) between in-model copies n ≤ n' (w = the outside
 * copy's local energy; the n' > n block is its adjoint) and its weight −Σ'(E). */
interface OutsideBlock {from:number;to:number;sigma:CMatrix;metric:CMatrix}
/** For every outside copy and slot s: S_s = Σ_v |u_vs|² / |w − E_v|, so that the block of slots
 * s, s' has ‖Σ‖ ≤ (S_s S_s')^½ (Cauchy–Schwarz over the outside states). */
interface OutsideScale {copy:OutsideCopy;S:Float64Array}
/** Σ blocks, S per outside copy, and the distance from E to the nearest outside level less
 * the margin its couplings can move it by (second order: μ²/(g − 2μ), g its distance to the
 * in-model levels it couples to). */
function outsideBlocks(p:Prepared,st:Structure,E:number):{blocks:OutsideBlock[];scales:OutsideScale[];minimum:number}{
  const blocks=new Map<string,OutsideBlock>(),scales:OutsideScale[]=[];let minimum=Infinity;
  // Y_qr(w)_ij = Σ_v d_iq d_jr / (w − E_v) (and its square for the weight), then for each
  // pair Σ = Σ_qr conj(F_q) F'_r Y_qr: the field enters only through nine coefficients.
  const Y=new Float64Array(9*256),Y2=new Float64Array(9*256);
  for(const g of st.outside){
    const pairs:{block:OutsideBlock;kappa:Float64Array}[]=[],needed=new Uint8Array(9);
    for(const [i,j] of g.pairs){const a=g.attached[i]!,b=g.attached[j]!;
      const key=`${a.copy},${b.copy}`;let block=blocks.get(key);if(!block){const da=st.copies[a.copy]!.dim,db=st.copies[b.copy]!.dim;block={from:a.copy,to:b.copy,sigma:cmatrix(da,db),metric:cmatrix(da,db)};blocks.set(key,block);}
      const fa=slotField(p.problem,a.slot),fb=slotField(p.problem,b.slot),kappa=new Float64Array(18);
      for(let q=0;q<3;q++)for(let r=0;r<3;r++){const k=q*3+r,re=fa[q]!.re*fb[r]!.re+fa[q]!.im*fb[r]!.im,im=fa[q]!.re*fb[r]!.im-fa[q]!.im*fb[r]!.re;kappa[2*k]=re;kappa[2*k+1]=im;if(re!==0||im!==0)needed[k]=1;}
      pairs.push({block,kappa});}
    const w=localEnergy(p,st.sector,E,g.charge);
    // Offsets E − D of the in-model levels this copy couples to, for the margins (built when needed).
    let partners:Float64Array|undefined;
    const S=new Float64Array(2*p.K);
    for(const part of st.copies[g.attached[0]!.copy]!.excited?BLOCKS.excited:BLOCKS.ground){
      const {rows,levels,support}=p.outside[part.manifold],n=part.size;
      // S and the margins level by level: the nearest of its sublevels bounds them all.
      for(const level of levels){const d=w<level.low?w-level.low:w>level.high?w-level.high:0,mu=level.norm,near=Math.abs(d);
        for(let k=0;k<S.length;k++)S[k]+=level.slots[k]!/near;
        if(near-mu<minimum){
          partners??=Float64Array.from(g.attached.flatMap(a=>Array.from(st.copies[a.copy]!.diag,x=>E-x))).sort();
          let lo=0,hi=partners.length;while(lo<hi){const mid=(lo+hi)>>1;if(partners[mid]!<d)lo=mid+1;else hi=mid;}
          const gap=Math.min(lo<partners.length?Math.abs(partners[lo]!-d):Infinity,lo>0?Math.abs(partners[lo-1]!-d):Infinity)-(level.high-level.low);
          minimum=Math.min(minimum,near-(gap>2*mu?mu*mu/(gap-2*mu):mu));}}
      if(!pairs.length)continue;
      for(let k=0;k<9;k++)if(needed[k])for(const ij of support[k]!){Y[k*256+ij]=0;Y2[k*256+ij]=0;}
      for(const row of rows){const inv=1/(w-row.energyMHz),inv2=inv*inv,entry=row.entry,value=row.value;
        for(let k=0;k<9;k++){if(!needed[k])continue;for(let e=row.start[k]!,end=row.start[k+1]!;e<end;e++){const t=value[e]!,at=entry[e]!;Y[at]+=t*inv;Y2[at]+=t*inv2;}}}
      for(const {block,kappa} of pairs){const cols=block.sigma.cols,sr=block.sigma.re,si=block.sigma.im,mr=block.metric.re,mi=block.metric.im;
        for(let k=0;k<9;k++){const kr=kappa[2*k]!,ki=kappa[2*k+1]!;if(kr===0&&ki===0)continue;
          for(const ij of support[k]!){const i=(ij/n)|0,j=ij-i*n,y=Y[k*256+ij]!,y2=Y2[k*256+ij]!,o=(part.local+i)*cols+part.local+j;sr[o]+=kr*y;si[o]+=ki*y;mr[o]+=kr*y2;mi[o]+=ki*y2;}}}
    }
    scales.push({copy:g,S});
  }
  return {blocks:[...blocks.values()],scales,minimum};
}

// ---------------------------------------------------------------- the dressed space at E
/** The dressed columns X = [I; Φ(E)] and, with H − E applied (outside self-energies
 * included), the shifted Rayleigh–Ritz pair M = X†(H − E)X, S = X†X + X†S_v X, where
 * S_v = −Σ'(E) is the outside states' weight. */
interface Dressed {phi:CMatrix[];hx:CMatrix[];m:CMatrix;s:CMatrix;outsideMetric:CMatrix;scales:OutsideScale[];poleGap:number}
function dressed(p:Prepared,st:Structure,E:number):Dressed{
  const copies=st.copies,nP=copies[0]!.dim,{blocks,scales,minimum}=outsideBlocks(p,st,E);
  // Each block acts from `to` into `from`, and as its adjoint from `from` into `to`.
  const into=copies.map(()=>[] as {block:OutsideBlock;to:number;adjoint:boolean}[]);
  for(const b of blocks){into[b.from]!.push({block:b,to:b.to,adjoint:false});if(b.from!==b.to)into[b.to]!.push({block:b,to:b.from,adjoint:true});}
  // Resolvents: dense LU where the outside self-energy fills the copy, diagonal elsewhere.
  const lu=copies.map((copy,q)=>{if(q===0)return undefined;const self=into[q]!.find(b=>b.to===q)?.block;if(!self)return undefined;
    const a=cmatrix(copy.dim,copy.dim);for(let i=0;i<copy.dim;i++)a.re[i*copy.dim+i]=E-copy.diag[i]!;for(let k=0;k<a.re.length;k++){a.re[k]-=self.sigma.re[k]!;a.im[k]-=self.sigma.im[k]!;}
    return {a,pivot:factor(a)};});
  const phi=copies.map((copy,q)=>{const x=cmatrix(copy.dim,nP);if(q===0)for(let i=0;i<nP;i++)x.re[i*nP+i]=1;return x;}),spare=copies.map(copy=>cmatrix(copy.dim,nP));
  // Products with the charge-0 columns (the identity) are additions.
  const product=(out:CMatrix,a:CMatrix,q:number,adjoint=false)=>{if(q!==0){multiply(out,a,phi[q]!,adjoint);return;}addTo(out,a,adjoint);};
  const couple=(q:number,out:CMatrix,self:boolean)=>{
    for(const e of st.edges[q]!){if(e.excited===q)product(out,e.coupling,e.ground);else product(out,e.coupling,e.excited,true);}
    for(const e of into[q]!)if(self||e.to!==q)product(out,e.block.sigma,e.to,e.adjoint);};
  // Block Gauss–Seidel over the layers. Its remaining error is in the residuals below.
  for(let sweeps=0,last=Infinity;copies.length>1;sweeps++){
    let change=0,size=0;
    for(let q=1;q<copies.length;q++){const copy=copies[q]!,next=spare[q]!;next.re.fill(0);next.im.fill(0);couple(q,next,false);
      const f=lu[q];if(f)solve(f.a,f.pivot,next);else for(let i=0;i<copy.dim;i++){const g=1/(E-copy.diag[i]!);for(let j=0;j<nP;j++){next.re[i*nP+j]*=g;next.im[i*nP+j]*=g;}}
      const old=phi[q]!;for(let k=0;k<next.re.length;k++){change=Math.max(change,Math.abs(next.re[k]!-old.re[k]!),Math.abs(next.im[k]!-old.im[k]!));size=Math.max(size,Math.abs(next.re[k]!),Math.abs(next.im[k]!));}
      spare[q]=old;phi[q]=next;}
    // The remaining error enters the residuals (and the bound); a contraction above ½ is a
    // near resonance.
    if(change<=size*1e-9)break;
    if(sweeps>=MAX_SWEEPS||sweeps>=3&&change>.5*last)throw new Error('远失谐光的 Floquet 分块不收敛：某个 Floquet 副本与能级近共振（例如两束远失谐光的拍频接近能级间隔）。静态光移未定义。');
    last=change;
  }
  const hx=copies.map((copy,q)=>{const out=spare[q]!;out.re.fill(0);out.im.fill(0);couple(q,out,true);const x=phi[q]!;for(let i=0;i<copy.dim;i++){const d=copy.diag[i]!-E;for(let j=0;j<nP;j++){out.re[i*nP+j]+=d*x.re[i*nP+j]!;out.im[i*nP+j]+=d*x.im[i*nP+j]!;}}return out;});
  const m=cmatrix(nP,nP),s=cmatrix(nP,nP),outsideMetric=cmatrix(nP,nP);
  addTo(m,hx[0]!);for(let i=0;i<nP;i++)s.re[i*nP+i]+=1;
  for(let q=1;q<copies.length;q++){gram(m,phi[q]!,hx[q]!);gram(s,phi[q]!,phi[q]!);}
  // Σ over n ≤ n' of X_n† M X_n' and its adjoint: twice the off-diagonal blocks, then Hermitian part.
  for(const b of blocks){const t=cmatrix(b.metric.rows,nP);product(t,b.metric,b.to);if(b.from!==b.to)for(let k=0;k<t.re.length;k++){t.re[k]*=2;t.im[k]*=2;}if(b.from===0)addTo(outsideMetric,t);else gram(outsideMetric,phi[b.from]!,t);}
  for(let k=0;k<nP*nP;k++){s.re[k]+=outsideMetric.re[k]!;s.im[k]+=outsideMetric.im[k]!;}
  for(const x of [m,s,outsideMetric])for(let i=0;i<nP;i++){x.im[i*nP+i]=0;for(let j=i+1;j<nP;j++){const r=(x.re[i*nP+j]!+x.re[j*nP+i]!)/2,v=(x.im[i*nP+j]!-x.im[j*nP+i]!)/2;x.re[i*nP+j]=x.re[j*nP+i]=r;x.im[i*nP+j]=v;x.im[j*nP+i]=-v;}}
  return {phi,hx,m,s,outsideMetric,scales,poleGap:minimum};
}

// ---------------------------------------------------------------- clusters
/** A level from a cluster's Ritz problem: frame energy θ = E + δ (both kept: E minus a
 * centroid is exact, so the level's offset from its centroid keeps the precision of δ),
 * charge-0 coefficients (re, im) and bounds. */
interface Level {energy:number;E:number;delta:number;span:number;c:[Float64Array,Float64Array];floquet:number;outside:number}
/** A floor for floating-point rounding: ulps of the level's offset and of the span of the
 * charge-0 energies its Ritz problem holds (7 THz in the excited sector: D1 and D2). */
const rounding=(offset:number,span:number)=>Number.EPSILON*(16*Math.abs(offset)+4*span);
const quadratic=(a:CMatrix,re:Float64Array,im:Float64Array)=>{const n=a.rows;let sum=0;for(let i=0;i<n;i++){let xr=0,xi=0;for(let j=0;j<n;j++){const ar=a.re[i*n+j]!,ai=a.im[i*n+j]!;xr+=ar*re[j]!-ai*im[j]!;xi+=ar*im[j]!+ai*re[j]!;}sum+=re[i]!*xr+im[i]!*xi;}return sum;};
/** Distance from θ to the levels of these copies, less each copy's margin. */
const gapTo=(theta:number,copies:Copy[],extra=0)=>{let g=Infinity;for(const c of copies)for(const x of c.diag)g=Math.min(g,Math.abs(theta-x)-c.margin-extra);return g;};
/** One linearized Brillouin–Wigner step for the cluster of Ritz values lo..hi−1 (the
 * sector's levels in ascending order) at energy E, with each level's bounds.
 *
 * The residual r_j = (H − θ_j)X c_j is orthogonal to the dressed space X, so it couples
 * only to Q-like states. Copy by copy (the complement's resolvent taken block-diagonal,
 * each copy's levels moved by at most its margin): |δθ_j| ≤ Σ_q |r_jq| Σ_{l∈cluster} |r_lq| / η_q,
 * the outside states' part with the gap to their nearest level. The left-out copies add
 * (what reaches them)² / gap, and the outside pairs not kept add their first order. Twice
 * the sum is reported. */
function clusterStep(p:Prepared,st:Structure,lo:number,hi:number,E:number):{levels:Level[];accepted:boolean;deeper:boolean}{
  const d=dressed(p,st,E),nP=st.copies[0]!.dim,eig=generalizedEigen(d.m,d.s),span=Math.max(...Array.from(st.copies[0]!.diag,x=>Math.abs(x-E)));
  const copies=st.copies.length,members=hi-lo,residual=new Float64Array(members*(copies+1));
  const found=Array.from({length:members},(_,k)=>{const j=lo+k;
    const re=new Float64Array(nP),im=new Float64Array(nP);for(let i=0;i<nP;i++){re[i]=eig.vectors.re[i*nP+j]!;im[i]=eig.vectors.im[i*nP+j]!;}
    // r = (H − E)Xc − δ Xc with δ = θ − E per copy; the outside part is δ times its amplitude.
    const delta=eig.values[j]!,theta=E+delta,amplitude=new Float64Array(copies);
    st.copies.forEach((copy,q)=>{const x=d.phi[q]!,h=d.hx[q]!;let norm=0,r2=0;for(let a=0;a<copy.dim;a++){let hr=0,hi=0,xr=0,xi=0;for(let b=0;b<nP;b++){const k=a*nP+b;hr+=h.re[k]!*re[b]!-h.im[k]!*im[b]!;hi+=h.re[k]!*im[b]!+h.im[k]!*re[b]!;xr+=x.re[k]!*re[b]!-x.im[k]!*im[b]!;xi+=x.re[k]!*im[b]!+x.im[k]!*re[b]!;}
      r2+=(hr-delta*xr)**2+(hi-delta*xi)**2;norm+=xr*xr+xi*xi;}amplitude[q]=Math.sqrt(norm);residual[k*(copies+1)+q]=Math.sqrt(r2);});
    residual[k*(copies+1)+copies]=Math.abs(delta)*Math.sqrt(Math.max(0,quadratic(d.outsideMetric,re,im)));
    // What reaches each left-out copy: D-line couplings from layer L and outside paths; and
    // the outside pairs not kept, at first order.
    const leakIn=new Float64Array(st.next.length),leakOut=new Float64Array(st.next.length);let pairs=0;
    for(const l of st.leaks)leakIn[l.next]+=l.norm*amplitude[l.kept]!;
    d.scales.forEach(({copy:g,S},n)=>{
      let source=0;for(const a of g.attached)source+=amplitude[a.copy]!*Math.sqrt(S[a.slot]!);
      for(const l of st.outsideLeaks[n]!)leakOut[l.next]+=Math.sqrt(S[l.slot]!)*source;
      for(const [i,j] of g.omitted){const a=g.attached[i]!,b=g.attached[j]!;pairs+=(i===j?1:2)*Math.sqrt(S[a.slot]!*S[b.slot]!)*amplitude[a.copy]!*amplitude[b.copy]!;}});
    let inModel=0,outsideTail=0;
    st.next.forEach((copy,n)=>{if(leakIn[n]===0&&leakOut[n]===0)return;const gap=gapTo(theta,[copy]);
      if(!(gap>0))throw new Error('远失谐光的 Floquet 分块不满足能隙条件：能级离某个 Floquet 副本太近（多光子共振），无法界定误差。静态光移未定义。');
      inModel+=leakIn[n]!**2/gap;outsideTail+=(leakOut[n]!**2+2*leakIn[n]!*leakOut[n]!)/gap;});
    return {level:{energy:theta,E,delta,span,c:[re,im] as [Float64Array,Float64Array],floquet:0,outside:0},theta,inModel,outside:outsideTail+pairs};
  });
  let deeper=false,worst=0;
  found.forEach((f,k)=>{
    let partition=0;
    for(let q=1;q<=copies;q++){
      const eta=q<copies?gapTo(f.theta,[st.copies[q]!]):d.poleGap-Math.abs(f.level.delta);let others=0;
      for(let l=0;l<members;l++)others+=residual[l*(copies+1)+q]!;const own=residual[k*(copies+1)+q]!;if(own===0)continue;
      if(!(eta>4*others))throw new Error('远失谐光的 Floquet 分块不满足能隙条件：能级离某个 Floquet 副本或 32 态以外能级太近，无法界定误差。静态光移未定义。');
      partition+=own*others/eta;}
    // The charge-0 part of r couples to the Q-like states only through their dressing.
    let own=residual[k*(copies+1)]!,others=0;for(let l=0;l<members;l++)others+=residual[l*(copies+1)]!;
    if(own>0)partition+=own*others/Math.max(gapTo(f.theta,st.copies.slice(1)),Number.MIN_VALUE);
    const total=2*(partition+f.inModel+f.outside);worst=Math.max(worst,total);
    f.level.floquet=2*(partition+f.inModel);f.level.outside=2*f.outside;
    if(f.inModel+f.outside>partition&&total>TARGET_MHZ)deeper=true;
  });
  return {levels:found.map(f=>f.level),accepted:worst<=TARGET_MHZ,deeper};
}

/** A sector's levels: clusters are windows of its ascending levels, split where the bare
 * spectrum has a gap above CLUSTER_GAP_MHZ; each is iterated to its own energy (the
 * Brillouin–Wigner self-consistency) and split further until its bound meets TARGET_MHZ. */
function solveSector(p:Prepared,sector:Sector,layers:number):{levels:Level[];deeper:boolean;clusters:number;solves:number}{
  const st=structure(p,sector,layers),P=st.copies[0]!,bare=Array.from(P.diag).sort((a,b)=>a-b);
  const pending:{lo:number;hi:number;E:number;rounds:number}[]=[];let lo=0;
  const mean=(values:number[])=>values.reduce((a,b)=>a+b,0)/values.length;
  for(let k=1;k<=bare.length;k++)if(k===bare.length||bare[k]!-bare[k-1]!>CLUSTER_GAP_MHZ){pending.push({lo,hi:k,E:mean(bare.slice(lo,k)),rounds:0});lo=k;}
  const done:Level[]=[];let clusters=0,solves=0;
  while(pending.length){
    const cluster=pending.shift()!,result=clusterStep(p,st,cluster.lo,cluster.hi,cluster.E);solves++;
    if(result.deeper&&layers<MAX_LAYERS)return {levels:[],deeper:true,clusters,solves};
    const energies=result.levels.map(l=>l.energy),centre=mean(energies),spread=energies[energies.length-1]!-energies[0]!;
    const finish=()=>{done.push(...result.levels);clusters++;};
    if(result.accepted||cluster.rounds>=8||cluster.hi-cluster.lo===1&&Math.abs(cluster.E-centre)<=TARGET_MHZ){finish();continue;}
    // Self-consistency first: move E to the cluster's levels; once there, split at the largest gap.
    if(Math.abs(cluster.E-centre)>.1*spread+TARGET_MHZ){pending.push({...cluster,E:centre,rounds:cluster.rounds+1});continue;}
    if(cluster.hi-cluster.lo===1){finish();continue;}
    let cut=1;for(let k=2;k<energies.length;k++)if(energies[k]!-energies[k-1]!>energies[cut]!-energies[cut-1]!)cut=k;
    pending.push({lo:cluster.lo,hi:cluster.lo+cut,E:mean(energies.slice(0,cut)),rounds:cluster.rounds+1},{lo:cluster.lo+cut,hi:cluster.hi,E:mean(energies.slice(cut)),rounds:cluster.rounds+1});
  }
  return {levels:done,deeper:false,clusters,solves};
}

/** Labels: the assignment of levels to bare states with the largest total overlap. */
function assignment(weights:number[][]):number[]{
  const n=weights.length,u=Array(n+1).fill(0),v=Array(n+1).fill(0),p=Array(n+1).fill(0),way=Array(n+1).fill(0);
  for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=Array(n+1).fill(Infinity),used=Array(n+1).fill(false);
    do{used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=n;j++)if(!used[j]){const cur=-weights[i0-1]![j-1]!-u[i0]-v[j];if(cur<min[j]){min[j]=cur;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}for(let j=0;j<=n;j++){if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else min[j]-=delta;}j0=j1;}while(p[j0]!==0);
    do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);
  }
  const result=Array(n).fill(0);for(let j=1;j<=n;j++)result[p[j]-1]=j-1;return result;
}

/** The background spectrum: every in-model level, its charge-0 state and its bounds. */
export function solveStaticBackground(problem:BackgroundProblem):BackgroundSolution{
  const p=prepare(problem),levels:BackgroundLevel[]=[];let layersUsed=FIRST_LAYERS,clusters=0,solves=0;
  for(const sector of ['ground','excited'] as const){
    let layers=FIRST_LAYERS,result=solveSector(p,sector,layers);solves+=result.solves;
    while(result.deeper){layers++;result=solveSector(p,sector,layers);solves+=result.solves;}
    layersUsed=Math.max(layersUsed,layers);clusters+=result.clusters;
    const start=sector==='ground'?0:8,nP=sector==='ground'?8:24,found=result.levels;
    if(found.length!==nP)throw new Error('远失谐背景的能级数与模型不符。');
    // Charge-0 parts, orthonormalized symmetrically (Löwdin): the dressed states are
    // orthonormal in the full space, their charge-0 parts only nearly so.
    const v=cmatrix(nP,nP);found.forEach((l,j)=>{for(let i=0;i<nP;i++){v.re[i*nP+j]=l.c[0]![i]!;v.im[i*nP+j]=l.c[1]![i]!;}});
    const overlap=cmatrix(nP,nP);gram(overlap,v,v);const eig=hermitianEigen(overlap);
    if(!(Math.min(...eig.values)>.5))throw new Error('远失谐背景的能级态线性相关。');
    const root=cmatrix(nP,nP);for(let i=0;i<nP;i++)for(let j=0;j<nP;j++){let re=0,im=0;for(let k=0;k<nP;k++){const f=1/Math.sqrt(eig.values[k]!),ar=eig.vectors.re[i*nP+k]!,ai=eig.vectors.im[i*nP+k]!,br=eig.vectors.re[j*nP+k]!,bi=-eig.vectors.im[j*nP+k]!;re+=f*(ar*br-ai*bi);im+=f*(ar*bi+ai*br);}root.re[i*nP+j]=re;root.im[i*nP+j]=im;}
    const u=cmatrix(nP,nP);multiply(u,v,root);
    const labels=assignment(Array.from({length:nP},(_,i)=>Array.from({length:nP},(_,j)=>u.re[i*nP+j]!**2+u.im[i*nP+j]!**2)));
    labels.forEach((col,i)=>{const level=found[col]!,re=new Float64Array(32),im=new Float64Array(32);for(let k=0;k<nP;k++){re[start+k]=u.re[k*nP+col]!;im[start+k]=u.im[k*nP+col]!;}
      const offset=accurateSum([level.E,sector==='ground'?0:-CENTROID[MANIFOLD_OF(start+i)],level.delta]);
      levels.push({index:start+i,energyMHz:offset,vector:{re,im},floquetBoundMHz:level.floquet+rounding(offset,level.span),outsideBoundMHz:level.outside});});
  }
  return {levels:levels.sort((a,b)=>a.index-b.index),layers:layersUsed,clusters,solves};
}
