// An independent reference for the static background (src/floquet-background.ts): the
// truncated Floquet matrix built directly over every charge |n|₁ ≤ N, its eigenvalues
// near the charge-0 levels found by shift-invert subspace iteration from the charge-0
// unit vectors and Rayleigh–Ritz. No partitioning, no clusters of the solver's, no
// layer-by-layer elimination. The states outside the model couple only to in-model
// states, so they are eliminated exactly: Σ(E) between every pair of in-model copies in
// the window (all Raman pairs). Its dependence on E is followed per level: each level is
// solved again centred on itself, with a Hellmann–Feynman step for the last (tiny) offset.
const D2_MHZ=384.2304844685e12/1e6,D1_MINUS_D2=(377.107463380e12-384.2304844685e12)/1e6;
const CENTROID=[...Array(8).fill(-D2_MHZ),...Array(8).fill(D1_MINUS_D2),...Array(16).fill(0)];

/** Neumaier-compensated sum. */
function fsum(values){let s=0,c=0;for(const v of values){const t=s+v;c+=Math.abs(s)>=Math.abs(v)?(s-t)+v:(v-t)+s;s=t;}return s+c;}
/** Every charge with |n|₁ ≤ N, charge 0 first. */
function charges(K,N){const out=[],cur=Array(K).fill(0);const visit=(s,left)=>{if(s===K){out.push(cur.slice());return;}for(let v=-left;v<=left;v++){cur[s]=v;visit(s+1,left-Math.abs(v));}cur[s]=0;};visit(0,N);return out.sort((a,b)=>a.reduce((x,y)=>x+Math.abs(y),0)-b.reduce((x,y)=>x+Math.abs(y),0));}
const key=n=>n.join(',');

/** The sector's in-model copies (charge, state indices, frame energies). */
function sectorCopies(problem,sector,N){
  const K=problem.carriers.length,list=[];let offset=0;
  for(const n of charges(K,N)){const total=n.reduce((a,b)=>a+b,0),excited=(sector==='excited')===(total%2===0),states=excited?[...Array(24).keys()].map(i=>8+i):[...Array(8).keys()];
    // E_abs − f = centroid + ε + core + n·ω − f, ω_s = ν(D2) + offset_s, f = 0 or ν(D2).
    const diag=states.map(i=>fsum([CENTROID[i],problem.energies[i],problem.core[i],(total+(sector==='ground'?1:0))*D2_MHZ,...n.map((x,s)=>x*problem.carriers[s].offsetMHz)]));
    list.push({charge:n,excited,states,diag,offset});offset+=states.length;}
  return {copies:list,index:new Map(list.map((c,i)=>[key(c.charge),i])),dim:offset};
}
/** Dense complex H (in-model couplings) of the sector, without the outside states. */
function inModel(problem,sector,N){
  const sc=sectorCopies(problem,sector,N),D=sc.dim,re=new Float64Array(D*D),im=new Float64Array(D*D);
  for(const c of sc.copies)c.diag.forEach((x,k)=>{re[(c.offset+k)*D+c.offset+k]=x;});
  for(const c of sc.copies){if(!c.excited)continue;
    problem.carriers.forEach((carrier,s)=>{for(const [sign,m] of [[1,carrier.rotating],[-1,carrier.counter]]){const n=c.charge.slice();n[s]+=sign;const g=sc.index.get(key(n));if(g===undefined)continue;const G=sc.copies[g];
      for(let e=0;e<24;e++)for(let x=0;x<8;x++){const vr=m.re[e*8+x],vi=m.im[e*8+x];if(vr===0&&vi===0)continue;const i=c.offset+e,j=G.offset+x;re[i*D+j]+=vr;im[i*D+j]+=vi;re[j*D+i]+=vr;im[j*D+i]-=vi;}}});}
  return {...sc,re,im};
}
/** Σ(E) and −Σ'(E) of every outside copy reaching the window, between all in-model copies. */
function outside(problem,sc,sector,E){
  const D=sc.dim,K=problem.carriers.length,sig={re:new Float64Array(D*D),im:new Float64Array(D*D)},met={re:new Float64Array(D*D),im:new Float64Array(D*D)};
  const groups=new Map();
  sc.copies.forEach((c,ci)=>{for(let s=0;s<K;s++)for(const counter of [false,true]){const v=c.charge.slice();v[s]+=counter?1:-1;const k=key(v);if(!groups.has(k))groups.set(k,{charge:v,attached:[]});groups.get(k).attached.push({copy:ci,field:counter?problem.carriers[s].counterField:problem.carriers[s].rotatingField});}});
  const blocks=[{name:'ground',start:0,size:8,local:0},{name:'D1',start:8,size:8,local:0},{name:'D2',start:16,size:16,local:8}];
  for(const g of groups.values()){
    const total=g.charge.reduce((a,b)=>a+b,0),w=fsum([E,sector==='ground'?0:D2_MHZ,-total*D2_MHZ,...g.charge.map((x,s)=>-x*problem.carriers[s].offsetMHz)]);
    const excited=sc.copies[g.attached[0].copy].excited;
    for(const b of blocks){if((b.name==='ground')===excited)continue;
      for(const row of problem.outside[b.name]){const inv=1/(w-row.energyMHz),inv2=inv*inv,u=[];
        for(const a of g.attached){const c=sc.copies[a.copy];for(let k=0;k<b.size;k++){let x=0,y=0;for(let q=0;q<3;q++){const d=row.dipoles[k*3+q];x+=d*a.field[q].re;y+=d*a.field[q].im;}if(x!==0||y!==0)u.push([c.offset+(excited?b.start-8:0)+k,x,y]);}}
        for(const [i,ar,ai] of u)for(const [j,br,bi] of u){const r=ar*br+ai*bi,m=ar*bi-ai*br;sig.re[i*D+j]+=r*inv;sig.im[i*D+j]+=m*inv;met.re[i*D+j]+=r*inv2;met.im[i*D+j]+=m*inv2;}}
    }
  }
  return {sig,met};
}
function luFactor(re,im,D){const piv=new Int32Array(D);
  for(let k=0;k<D;k++){let p=k,best=-1;for(let i=k;i<D;i++){const v=re[i*D+k]**2+im[i*D+k]**2;if(v>best){best=v;p=i;}}piv[k]=p;
    if(p!==k)for(let j=0;j<D;j++){let t=re[k*D+j];re[k*D+j]=re[p*D+j];re[p*D+j]=t;t=im[k*D+j];im[k*D+j]=im[p*D+j];im[p*D+j]=t;}
    const dr=re[k*D+k],di=im[k*D+k],d=dr*dr+di*di;
    for(let i=k+1;i<D;i++){const xr=re[i*D+k],xi=im[i*D+k];if(xr===0&&xi===0)continue;const lr=(xr*dr+xi*di)/d,li=(xi*dr-xr*di)/d;re[i*D+k]=lr;im[i*D+k]=li;
      for(let j=k+1;j<D;j++){const ur=re[k*D+j],ui=im[k*D+j];if(ur===0&&ui===0)continue;re[i*D+j]-=lr*ur-li*ui;im[i*D+j]-=lr*ui+li*ur;}}}
  return piv;}
function luSolve(re,im,piv,D,br,bi){
  for(let k=0;k<D;k++){const p=piv[k];if(p!==k){let t=br[k];br[k]=br[p];br[p]=t;t=bi[k];bi[k]=bi[p];bi[p]=t;}}
  for(let i=0;i<D;i++){let xr=br[i],xi=bi[i];for(let k=0;k<i;k++){const lr=re[i*D+k],li=im[i*D+k];if(lr===0&&li===0)continue;xr-=lr*br[k]-li*bi[k];xi-=lr*bi[k]+li*br[k];}br[i]=xr;bi[i]=xi;}
  for(let i=D-1;i>=0;i--){let xr=br[i],xi=bi[i];for(let k=i+1;k<D;k++){const ur=re[i*D+k],ui=im[i*D+k];if(ur===0&&ui===0)continue;xr-=ur*br[k]-ui*bi[k];xi-=ur*bi[k]+ui*br[k];}const dr=re[i*D+i],di=im[i*D+i],d=dr*dr+di*di;br[i]=(xr*dr+xi*di)/d;bi[i]=(xi*dr-xr*di)/d;}
}
/** Cyclic Jacobi for a small complex Hermitian matrix (row-major re, im), in place. */
export function jacobi(are,aim,n){const vr=new Float64Array(n*n),vi=new Float64Array(n*n);for(let i=0;i<n;i++)vr[i*n+i]=1;
  for(let sweep=0;sweep<100;sweep++){let off=0,scale=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++){const v=Math.hypot(are[i*n+j],aim[i*n+j]);if(i!==j)off=Math.max(off,v);scale=Math.max(scale,v);}if(off<=scale*1e-17)break;
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){const xr=are[p*n+q],xi=aim[p*n+q],r=Math.hypot(xr,xi);if(r===0)continue;const er=xr/r,ei=xi/r,tau=(are[q*n+q]-are[p*n+p])/(2*r),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.hypot(1,tau)),c=1/Math.hypot(1,t),s=t*c;
      const rot=(Ar,Ai,k)=>{const pr=Ar[k*n+p],pi=Ai[k*n+p],qr=Ar[k*n+q],qi=Ai[k*n+q];Ar[k*n+p]=c*pr-s*(er*qr+ei*qi);Ai[k*n+p]=c*pi-s*(er*qi-ei*qr);Ar[k*n+q]=s*(er*pr-ei*pi)+c*qr;Ai[k*n+q]=s*(er*pi+ei*pr)+c*qi;};
      for(let k=0;k<n;k++){rot(vr,vi,k);if(k!==p&&k!==q){rot(are,aim,k);are[p*n+k]=are[k*n+p];aim[p*n+k]=-aim[k*n+p];are[q*n+k]=are[k*n+q];aim[q*n+k]=-aim[k*n+q];}}
      are[p*n+p]-=t*r;are[q*n+q]+=t*r;are[p*n+q]=are[q*n+p]=aim[p*n+q]=aim[q*n+p]=0;}}
  return {values:Array.from({length:n},(_,i)=>are[i*n+i]),vr,vi};}
/** Orthonormal columns (modified Gram–Schmidt, twice). */
function orthonormalize(B,D,k){for(let pass=0;pass<2;pass++)for(let j=0;j<k;j++){for(let l=0;l<j;l++){let pr=0,pi=0;for(let i=0;i<D;i++){pr+=B.re[l][i]*B.re[j][i]+B.im[l][i]*B.im[j][i];pi+=B.re[l][i]*B.im[j][i]-B.im[l][i]*B.re[j][i];}for(let i=0;i<D;i++){B.re[j][i]-=pr*B.re[l][i]-pi*B.im[l][i];B.im[j][i]-=pr*B.im[l][i]+pi*B.re[l][i];}}let n=0;for(let i=0;i<D;i++)n+=B.re[j][i]**2+B.im[j][i]**2;n=Math.sqrt(n);for(let i=0;i<D;i++){B.re[j][i]/=n;B.im[j][i]/=n;}}}

/** The sector's charge-0-like eigenvalues of H(E) at E = shift, by shift-invert subspace
 * iteration from the charge-0 unit vectors until ranks lo..hi−1 converge; each with the
 * Hellmann–Feynman step to θ = λ(θ): θ − E = δ/(1 + y†S_v y), λ' = −y†S_v y. */
function levelsAt(problem,sector,N,shift,lo,hi){
  const H=inModel(problem,sector,N),D=H.dim,{sig,met}=outside(problem,H,sector,shift),P=sector==='ground'?8:24;
  const are=new Float64Array(D*D),aim=new Float64Array(D*D);for(let x=0;x<D*D;x++){are[x]=H.re[x]+sig.re[x];aim[x]=H.im[x]+sig.im[x];}for(let i=0;i<D;i++)are[i*D+i]-=shift;
  const lre=are.slice(),lim=aim.slice(),piv=luFactor(lre,lim,D);
  const B={re:Array.from({length:P},(_,j)=>{const v=new Float64Array(D);v[j]=1;return v;}),im:Array.from({length:P},()=>new Float64Array(D))};
  let ritz;
  for(let it=0;it<400;it++){for(let j=0;j<P;j++)luSolve(lre,lim,piv,D,B.re[j],B.im[j]);orthonormalize(B,D,P);
    // Rayleigh–Ritz with H(E) − E.
    const AB={re:B.re.map(()=>new Float64Array(D)),im:B.re.map(()=>new Float64Array(D))};
    for(let j=0;j<P;j++)for(let i=0;i<D;i++){let xr=0,xi=0;for(let l=0;l<D;l++){const ar=are[i*D+l],ai=aim[i*D+l];if(ar===0&&ai===0)continue;xr+=ar*B.re[j][l]-ai*B.im[j][l];xi+=ar*B.im[j][l]+ai*B.re[j][l];}AB.re[j][i]=xr;AB.im[j][i]=xi;}
    const mre=new Float64Array(P*P),mim=new Float64Array(P*P);for(let a=0;a<P;a++)for(let b=0;b<P;b++){let xr=0,xi=0;for(let i=0;i<D;i++){xr+=B.re[a][i]*AB.re[b][i]+B.im[a][i]*AB.im[b][i];xi+=B.re[a][i]*AB.im[b][i]-B.im[a][i]*AB.re[b][i];}mre[a*P+b]=xr;mim[a*P+b]=xi;}
    for(let a=0;a<P;a++){mim[a*P+a]=0;for(let b=a+1;b<P;b++){const r=(mre[a*P+b]+mre[b*P+a])/2,m=(mim[a*P+b]-mim[b*P+a])/2;mre[a*P+b]=mre[b*P+a]=r;mim[a*P+b]=m;mim[b*P+a]=-m;}}
    const eig=jacobi(mre,mim,P),order=eig.values.map((v,i)=>i).sort((a,b)=>eig.values[a]-eig.values[b]).slice(lo,hi);
    const next=order.map(col=>{const yr=new Float64Array(D),yi=new Float64Array(D);for(let a=0;a<P;a++){const cr=eig.vr[a*P+col],ci=eig.vi[a*P+col];for(let i=0;i<D;i++){yr[i]+=cr*B.re[a][i]-ci*B.im[a][i];yi[i]+=cr*B.im[a][i]+ci*B.re[a][i];}}
      let sv=0;for(let i=0;i<D;i++)for(let j=0;j<D;j++){const mr=met.re[i*D+j],mi=met.im[i*D+j];if(mr===0&&mi===0)continue;sv+=yr[i]*(mr*yr[j]-mi*yi[j])+yi[i]*(mr*yi[j]+mi*yr[j]);}
      return {shift,delta:eig.values[col]/(1+sv)};});
    if(ritz&&next.every((x,i)=>Math.abs(x.delta-ritz[i].delta)<=1e-13*Math.max(1,Math.abs(x.delta))))return next;
    ritz=next;}
  return ritz;
}
/** All 32 levels (MHz relative to their manifold centroid, ascending within each
 * manifold). Clusters: windows of the ascending levels where the bare spectrum has
 * gaps above 1 GHz, solved centred on the cluster twice; then every level once more
 * centred on itself, so the Hellmann–Feynman remainder is second order in a tiny step. */
export function exactStaticLevels(problem,N){
  const out={ground:[],D1:[],D2:[]};
  for(const sector of ['ground','excited']){
    const P=sector==='ground'?8:24,start=sector==='ground'?0:8,bare=Array.from({length:P},(_,i)=>fsum([CENTROID[start+i],sector==='ground'?D2_MHZ:0,problem.energies[start+i],problem.core[start+i]])).sort((a,b)=>a-b);
    let lo=0;
    for(let hi=1;hi<=P;hi++){if(hi<P&&bare[hi]-bare[hi-1]<=1000)continue;
      const theta=l=>l.shift+l.delta;let centre=bare.slice(lo,hi).reduce((a,b)=>a+b,0)/(hi-lo),levels;
      for(let pass=0;pass<2;pass++){levels=levelsAt(problem,sector,N,centre,lo,hi);centre=levels.reduce((a,b)=>a+theta(b),0)/levels.length;}
      levels=levels.map((l,k)=>levelsAt(problem,sector,N,theta(l),lo+k,lo+k+1)[0]);
      // Shift minus the D1 centroid is exact (both near −7.1 THz), so δ keeps its precision.
      for(const l of levels){const manifold=sector==='ground'?'ground':theta(l)<D1_MINUS_D2/2?'D1':'D2';out[manifold].push(fsum([l.shift,manifold==='D1'?-D1_MINUS_D2:0,l.delta]));}
      lo=hi;}
  }
  for(const k of Object.keys(out))out[k].sort((a,b)=>a-b);
  return out;
}
