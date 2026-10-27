import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {transitionReferenceMHz} from '../dist/js/atom-workbench/src/physics.js';
import {spectroscopyAnalysis,staticBackgroundProblem} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {TARGET_MHZ,solveStaticBackground} from '../dist/js/atom-workbench/src/floquet-background.js';
import {levelsChart} from '../dist/js/atom-workbench/src/charts.js';
import {STATIC_SHIFT_SCENES,BEYOND_LIMIT_SCENES} from './static-shift-scenes.mjs';
import {exactStaticLevels,jacobi} from './static-shift-reference.mjs';

const fixture=JSON.parse(readFileSync(new URL('./fixtures/static-shift-second-order.json',import.meta.url),'utf8'));
const D2=384.2304844685e12/1e6,D1=377.107463380e12/1e6;
const MANIFOLDS=[['ground',0,8],['D1',8,8],['D2',16,16]];
/** Sorted energies per manifold of a 32-level list (MHz relative to the centroid). */
const sorted=levels=>Object.fromEntries(MANIFOLDS.map(([m,start,size])=>[m,levels.slice(start,start+size).sort((a,b)=>a-b)]));

/** The partitioning stopped at first order: the symmetric second-order (van Vleck)
 * Hamiltonian of the same Floquet problem, each fine-structure manifold diagonalized on
 * its own: the second-order display background. */
function secondOrderLevels(problem){
  const absolute=i=>i<8?problem.energies[i]:(i<16?D1:D2)+problem.energies[i],re=new Float64Array(1024),im=new Float64Array(1024);
  const add=(i,j,ar,ai,br,bi,f)=>{re[i*32+j]+=f*(ar*br+ai*bi);im[i*32+j]+=f*(ar*bi-ai*br);};// conj(a) b f
  for(let i=0;i<32;i++)re[i*33]=problem.energies[i]+problem.core[i];
  for(const carrier of problem.carriers){const omega=D2+carrier.offsetMHz;
    for(const [m,sign] of [[carrier.rotating,1],[carrier.counter,-1]]){const a=(e,g)=>[m.re[e*8+g],m.im[e*8+g]];
      for(let e=0;e<24;e++){const d=g=>absolute(g)-absolute(8+e)+sign*omega;for(let x=0;x<8;x++)for(let y=0;y<8;y++)add(x,y,...a(e,x),...a(e,y),(1/d(x)+1/d(y))/2);}
      for(let g=0;g<8;g++){const d=e=>absolute(8+e)-absolute(g)-sign*omega;for(const [from,to] of [[0,8],[8,24]])for(let x=from;x<to;x++)for(let y=from;y<to;y++){const [xr,xi]=a(x,g),[yr,yi]=a(y,g);add(8+x,8+y,yr,yi,xr,xi,(1/d(x)+1/d(y))/2);}}
    }
    for(const [,start,size] of MANIFOLDS)for(const state of problem.outside[MANIFOLDS.find(m=>m[1]===start)[0]])for(const [field,sign] of [[carrier.rotatingField,1],[carrier.counterField,-1]]){
      const u=Array.from({length:size},(_,k)=>[0,1,2].reduce(([r,i],q)=>[r+state.dipoles[k*3+q]*field[q].re,i+state.dipoles[k*3+q]*field[q].im],[0,0])),d=k=>absolute(start+k)+sign*omega-state.energyMHz;
      for(let x=0;x<size;x++)for(let y=0;y<size;y++)add(start+x,start+y,...u[x],...u[y],(1/d(x)+1/d(y))/2);}
  }
  const out=[];for(const [,start,size] of MANIFOLDS){const br=new Float64Array(size*size),bi=new Float64Array(size*size);for(let i=0;i<size;i++)for(let j=0;j<size;j++){br[i*size+j]=re[(start+i)*32+start+j];bi[i*size+j]=im[(start+i)*32+start+j];}out.push(...jacobi(br,bi,size).values);}
  return out;
}

test('stopped at first order, the partitioning reproduces the second-order (van Vleck) levels to rounding',()=>{
  for(const [name,build] of Object.entries(STATIC_SHIFT_SCENES)){
    const mine=sorted(secondOrderLevels(staticBackgroundProblem(build()))),second=sorted(Object.values(fixture.levels[name]));
    for(const [m] of MANIFOLDS)mine[m].forEach((x,i)=>assert.ok(Math.abs(x-second[m][i])<=1e-9,`${name} ${m} ${i}: ${x} vs ${second[m][i]}`));
  }
});

test('all orders: the levels move from second order by the expected small amounts, each with its bound',()=>{
  for(const [name,build] of Object.entries(STATIC_SHIFT_SCENES)){
    const a=spectroscopyAnalysis(build(),undefined,false),levels=[...a.ground,...a.D1,...a.D2];assert.equal(a.spectroscopy.status,'dispersive',name);
    const mixing=a.spectroscopy.maximumMixingRatio,second=fixture.levels[name];
    for(const level of levels){
      // Fourth order: |Δθ| ≲ (mixing)² × (largest shift) plus the D1–D2 coupling through the
      // ground copies (second order in a second-order coupling, over the 7 THz fine structure).
      const shift=Math.max(...levels.map(l=>Math.abs(l.starkShiftMHz)));
      assert.ok(Math.abs(level.energyMHz-second[level.id])<=10*mixing**2*shift+shift**2/7e6,`${name} ${level.id}`);
      assert.ok(level.shiftBound.floquetMHz+level.shiftBound.outsideMHz<=TARGET_MHZ+1e-8,`${name} ${level.id} bound`);
    }
  }
});

test('the all-order levels agree with an exact Floquet diagonalization within their bounds, also beyond 1% mixing',()=>{
  const scenes={'pi-3G':STATIC_SHIFT_SCENES['pi-3G'],'pi-0G-high':STATIC_SHIFT_SCENES['pi-0G-high'],'near-4D5/2':STATIC_SHIFT_SCENES['near-4D5/2'],'pi-3G-2%':BEYOND_LIMIT_SCENES['pi-3G-2%']};
  for(const [name,build] of Object.entries(scenes)){
    const problem=staticBackgroundProblem(build()),ours=solveStaticBackground(problem).levels,exact=exactStaticLevels(problem,4);
    for(const [m,start,size] of MANIFOLDS){const mine=ours.slice(start,start+size).sort((a,b)=>a.energyMHz-b.energyMHz);
      mine.forEach((l,i)=>{const bound=l.floquetBoundMHz+l.outsideBoundMHz;assert.ok(Math.abs(l.energyMHz-exact[m][i])<=bound,`${name} ${m} ${i}: ${l.energyMHz-exact[m][i]} > ${bound}`);});}
  }
});

test('the π tweezer line centre moves from 23.274469 MHz (second order) by about −13 Hz',()=>{
  const a=spectroscopyAnalysis(defaultScene(),'probe'),g=a.ground.find(l=>l.id==='ground-2-2'),e=a.D2.find(l=>l.id==='D2-3-3');
  const centre=e.energyMHz-g.energyMHz-transitionReferenceMHz('D2',2,3),bound=[g,e].reduce((s,l)=>s+l.shiftBound.floquetMHz+l.shiftBound.outsideMHz,0);
  assert.ok(Math.abs(centre-23.274468620337302+13.23e-6)<1e-7,`${centre}`);assert.ok(bound<1e-7);
});

test('the levels chart says each total shift has a numerical bound, not a data uncertainty',()=>{
  const s=defaultScene(),svg=levelsChart(s,spectroscopyAnalysis(s),undefined,{line:'D2',labelled:[],initialId:s.dynamics.initialStateId});
  const label=svg.match(/data-state="ground-2-2"[^>]*aria-label="([^"]*)"/)[1];
  assert.match(label,/总能移 .*MHz/);assert.match(label,/数值误差 ≤ /);
});
