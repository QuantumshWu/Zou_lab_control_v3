// The static background's validation table (docs/population-core-delivery.md, "静态光移"):
// per scene, the largest change from the release's second order, the largest difference
// from the exact truncated Floquet diagonalization (tests/static-shift-reference.mjs), the
// largest bound and the largest |difference| / bound; then the update time for 1–5 static
// lasers. Run after `npm run build`: node scripts/static-shift-report.mjs
import {readFileSync} from 'node:fs';
import {spectroscopyAnalysis,staticBackgroundProblem} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {solveStaticBackground} from '../dist/js/atom-workbench/src/floquet-background.js';
import {STATIC_SHIFT_SCENES,BEYOND_LIMIT_SCENES} from '../tests/static-shift-scenes.mjs';
import {exactStaticLevels} from '../tests/static-shift-reference.mjs';

const fixture=JSON.parse(readFileSync(new URL('../tests/fixtures/static-shift-second-order.json',import.meta.url),'utf8'));
const MANIFOLDS=[['ground',0,8],['D1',8,8],['D2',16,16]],hz=x=>(x*1e6).toExponential(2);
const far=(id,nm,I)=>({id,name:id,enabled:true,regime:'far',line:'D2',frequencyMode:'wavelength',wavelengthNm:nm,referenceTransition:{line:'D2',Fg:2,Fe:3},detuningMHz:0,detuningUnit:'THz',intensityMode:'intensity',intensityMWcm2:I,thetaDeg:60,phiDeg:20,polarizationAngleDeg:10,ellipticityDeg:15,phaseDeg:0});
const EXTRA=[far('1064',1064,2e7),far('810',810,2e6),far('532',532,5e6),far('880',880,3e6)];
const withLasers=K=>{const s=STATIC_SHIFT_SCENES['pi-3G']();s.lasers.push(...EXTRA.slice(0,K-1).map(l=>({...l})));return s;};
const scenes=[...Object.entries(STATIC_SHIFT_SCENES).map(([n,b])=>[n,b,4]),...Object.entries(BEYOND_LIMIT_SCENES).map(([n,b])=>[n,b,4]),['3 lasers',()=>withLasers(3),3]];

console.log('scene | mixing | max |all − 2nd| Hz | max |all − exact| Hz | max bound Hz | max |diff|/bound | layers');
for(const [name,build,N] of scenes){
  const scene=build(),problem=staticBackgroundProblem(scene),ours=solveStaticBackground(problem),exact=exactStaticLevels(problem,N);
  const release=fixture.levels[name],mixing=spectroscopyAnalysis(scene,undefined,false).spectroscopy.maximumMixingRatio;
  let change=0,diff=0,bound=0,ratio=0;
  for(const [m,start,size] of MANIFOLDS){const mine=ours.levels.slice(start,start+size).sort((a,b)=>a.energyMHz-b.energyMHz);
    mine.forEach((l,i)=>{const b=l.floquetBoundMHz+l.outsideBoundMHz,d=Math.abs(l.energyMHz-exact[m][i]);diff=Math.max(diff,d);bound=Math.max(bound,b);ratio=Math.max(ratio,d/b);});}
  if(release)ours.levels.forEach((l,i)=>{const id=Object.keys(release)[i];change=Math.max(change,Math.abs(l.energyMHz-release[id]));});
  console.log(`${name} | ${(100*mixing).toPrecision(2)}% | ${release?hz(change):'—'} | ${hz(diff)} (N = ${N}) | ${hz(bound)} | ${ratio.toFixed(3)} | ${ours.layers}`);
}
console.log('\nstatic lasers | ms per levels-chart update (median of 15)');
for(let K=1;K<=5;K++){const times=[];
  for(let k=0;k<15;k++){const s=withLasers(K),t=s.lasers[0];Object.assign(t,{intensityMode:'intensity',intensityMWcm2:t.intensityMWcm2*(1+1e-6*k),intensityAnchor:undefined,powerMW:undefined});
    const start=performance.now();spectroscopyAnalysis(s,undefined,false);times.push(performance.now()-start);}
  console.log(`${K} | ${times.sort((a,b)=>a-b)[7].toFixed(1)}`);}
