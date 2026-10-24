import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {arrowGeometry} from '../dist/js/atom-workbench/src/space-arrow.js';

const points=mesh=>mesh.flatMap(face=>face.vertices);
test('a rotating arrow remains volumetric and continuous when it points along the camera',()=>{
  const meshAt=angle=>arrowGeometry([0,0,0],[Math.cos(angle),Math.sin(angle),0],.012,[0,0,1]);
  const before=points(meshAt(Math.PI/2-1e-6)),after=points(meshAt(Math.PI/2+1e-6)),endOn=points(meshAt(Math.PI/2));
  assert.equal(before.length,after.length);
  assert.equal(before.length,endOn.length);
  assert.ok(endOn.some(p=>Math.abs(p[0])>.01));
  assert.ok(endOn.some(p=>Math.abs(p[2])>.01));
  assert.ok(endOn.some(p=>p[1]>.8&&p[1]<.95),'head base has its actual depth behind the tip');
  for(let i=0;i<before.length;i++)assert.ok(Math.hypot(...before[i].map((v,j)=>v-after[i][j]))<2.1e-6);
});

test('linear-polarization arrows shrink through zero without a fixed-size head or endpoint symbol',()=>{
  assert.deepEqual(arrowGeometry([0,0,0],[0,0,0],.012),[]);
  for(const length of [1e-3,1e-6,1e-9])for(const sign of [-1,1]){
    const mesh=arrowGeometry([0,0,0],[length*sign,0,0],.012,[0,0,1]);
    assert.ok(mesh.length>0);
    for(const p of points(mesh))assert.ok(Math.hypot(...p)<=length*1.001);
    for(const f of mesh)assert.ok(Math.abs(Math.hypot(...f.normal)-1)<1e-12);
  }
});

test('monitor selection and hover change paint without changing shared-checkbox box metrics',async()=>{
  const css=(await readFile(new URL('../style.css',import.meta.url),'utf8')).replace(/\/\*[\s\S]*?\*\//g,'');
  const rules=[...css.matchAll(/([^{}]+)\{([^{}]+)\}/g)].filter(m=>m[1].includes('.monitor-option'));
  const base=rules.find(m=>m[1].trim()==='.monitor-option.check');
  assert.ok(base);assert.match(base[2],/border:\s*0\s*;/);
  const geometry=/^(?:border(?:-(?:width|style|top|bottom|left|right))?|padding(?:-.+)?|margin(?:-.+)?|(?:min-|max-)?(?:width|height)|font(?:-.+)?|gap|transform|scale)$/;
  for(const rule of rules.filter(m=>m[1].includes(':'))){
    for(const declaration of rule[2].split(';')){
      const property=declaration.split(':')[0].trim();
      assert.ok(!geometry.test(property),`${rule[1].trim()} changes ${property}`);
    }
  }
  assert.ok(rules.some(m=>m[1].includes('[aria-checked="true"]')&&m[2].includes('box-shadow: inset')));
  assert.ok(rules.every(m=>!m[1].includes('input:checked')),'uses the actual shared checkbox state');
});
