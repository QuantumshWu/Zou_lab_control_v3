import test from 'node:test';import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';

// The coordinates of the outside form (spectroscopy.ts `outsideFieldForm`): |E_q|², then per pair p < q Re(E_p* E_q), Im(E_p* E_q).
const coordinates=e=>{const c=e.map(z=>z.re*z.re+z.im*z.im);
 for(let p=0;p<3;p++)for(let q=p+1;q<3;q++)c.push(e[p].re*e[q].re+e[p].im*e[q].im,e[p].re*e[q].im-e[p].im*e[q].re);return c;};
const at=(form,e)=>{const c=coordinates(e);return {real:form[0].real.map((_,k)=>form.reduce((s,m,j)=>s+c[j]*m.real[k],0)),imag:form[0].imag.map((_,k)=>form.reduce((s,m,j)=>s+c[j]*m.imag[k],0))};};
const close=(a,b,what)=>{const scale=Math.max(...b.real.map(Math.abs),...b.imag.map(Math.abs));
 const error=Math.max(...a.real.map((x,k)=>Math.abs(x-b.real[k])),...a.imag.map((x,k)=>Math.abs(x-b.imag[k])));assert.ok(error<=1e-10*scale,`${what}: ${error} of ${scale}`);};
/** The default scene's tweezer with three more beams of its frequency: other directions, polarizations and initial phases. */
function crossed(){
 const scene=defaultScene(),tweezer=scene.lasers.find(l=>l.id==='tweezer');Object.assign(tweezer,{ellipticityDeg:13,polarizationAngleDeg:21,phaseDeg:-37});
 for(const [i,axis] of ['+y','-z','+z'].entries())scene.lasers.push({...structuredClone(tweezer),id:`cross${i}`,name:`交叉光 ${i}`,
  direction:{reference:{kind:'axis',axis},thetaDeg:0,phiDeg:0},ellipticityDeg:29-17*i,polarizationAngleDeg:7+40*i,phaseDeg:53+71*i});
 return scene;
}

test('each beam carries its field and each frequency one outside form, which gives every beam its own outside shift',()=>{
 const model=buildFullMasterModel(crossed(),undefined,{axis:'3d',temperatureUK:20}),motion=model.motion;
 const key=index=>index.join(',');
 assert.equal(new Set(motion.outsideForms.map(f=>key(f.frequencyIndex))).size,motion.outsideForms.length);
 assert.deepEqual(new Set(motion.outsideForms.map(f=>key(f.frequencyIndex))),new Set(motion.beams.map(b=>key(b.frequencyIndex))));
 for(const beam of motion.beams){
  const form=motion.outsideForms.find(f=>key(f.frequencyIndex)===key(beam.frequencyIndex));assert.equal(form.matrices.length,9);
  // The field's intensity is the beam's, I = (ε0 c/2) |E|².
  const squared=beam.electricField.reduce((s,z)=>s+z.re*z.re+z.im*z.im,0);
  assert.ok(Math.abs(squared/(2*beam.intensityMWcm2*10/(8.8541878128e-12*299792458))-1)<1e-12,beam.id);
  close(at(form.matrices,beam.electricField),beam.outsideShift,beam.id);
 }
});

test('the outside form is a Hermitian form in the field: quadratic, and its cross terms the interference of two beams',()=>{
 const motion=buildFullMasterModel(crossed(),undefined,{axis:'3d',temperatureUK:20}).motion;
 const beams=motion.beams.filter(b=>b.waistUm!==undefined),form=motion.outsideForms.find(f=>f.frequencyIndex.join()===beams[0].frequencyIndex.join()).matrices;
 const [e1,e2]=[beams[0].electricField,beams[1].electricField],sum=e1.map((z,q)=>({re:z.re+e2[q].re,im:z.im+e2[q].im}));
 const twice=at(form,e1.map(z=>({re:2*z.re,im:2*z.im}))),once=at(form,e1);close(twice,{real:once.real.map(x=>4*x),imag:once.imag.map(x=>4*x)},'quadratic');
 for(const m of [at(form,sum),at(form,e1)]){const scale=Math.max(...m.real.map(Math.abs),...m.imag.map(Math.abs));for(let i=0;i<32;i++)for(let j=0;j<32;j++){
  assert.ok(Math.abs(m.real[i*32+j]-m.real[j*32+i])<=1e-12*scale&&Math.abs(m.imag[i*32+j]+m.imag[j*32+i])<=1e-12*scale,'Hermitian');}}
 // Crossed polarizations interfere: the summed field's shift is not the two beams' own shifts added.
 const [o1,o2,o]=[beams[0].outsideShift,beams[1].outsideShift,at(form,sum)],k=o.real.reduce((best,x,i)=>Math.abs(x-o1.real[i]-o2.real[i])>Math.abs(o.real[best]-o1.real[best]-o2.real[best])?i:best,0);
 assert.ok(Math.abs(o.real[k]-o1.real[k]-o2.real[k])>1e-6*Math.max(...o.real.map(Math.abs)),'no interference');
});
