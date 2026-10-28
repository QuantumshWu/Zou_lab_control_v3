import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze,composeMagneticField,direction,polarization} from '../dist/js/atom-workbench/src/physics.js';
import {directionAxis,directionVector,linkableFields,referenceAxis,releaseLinks,shortcutDirection} from '../dist/js/atom-workbench/src/direction.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {along,field,spherical} from './directions.mjs';

const close=(a,b,tol=1e-14)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<=tol,`${a} != ${b}`));
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const roundtrip=s=>validateScene(JSON.parse(JSON.stringify(s)));
const DEG=Math.PI/180;
/** Rodrigues: v turned by `angle` about the unit `axis`. */
function turn(v,axis,angle){const c=Math.cos(angle),s=Math.sin(angle),k=dot(axis,v),x=[axis[1]*v[2]-axis[2]*v[1],axis[2]*v[0]-axis[0]*v[2],axis[0]*v[1]-axis[1]*v[0]];return v.map((vi,i)=>vi*c+x[i]*s+axis[i]*k*(1-c));}
/** The minimal rotation from z to the axis at (θ, φ), applied to v. */
const minimal=(theta,phi,v)=>turn(v,[-Math.sin(phi*DEG),Math.cos(phi*DEG),0],theta*DEG);
const laserWith=(dir,extra={})=>({...defaultScene().lasers[0],direction:dir,...extra});

test('on the axis z the offset is the usual spherical (θ, φ); elsewhere it is read in the minimally rotated frame',()=>{
 for(const [theta,phi] of [[0,0],[27,-35],[90,73],[163,180]])close(directionVector([],spherical(theta,phi)),direction(theta,phi));
 for(const [nt,np,t,p] of [[30,40,20,10],[90,90,45,0],[120,-60,70,135],[180,0,30,-90]]){
  const expected=minimal(nt,np,direction(t,p)),dir={reference:{kind:'angles',thetaDeg:nt,phiDeg:np},thetaDeg:t,phiDeg:p};
  close(directionVector([],dir),expected);
 }
 // φ′ = 0 is the x the minimal rotation carries; ψ is measured from it.
 const pol=polarization(laserWith({reference:{kind:'angles',thetaDeg:50,phiDeg:70},thetaDeg:0,phiDeg:0}),[field('b',1,0,0)]);
 close(pol.u,minimal(50,70,[1,0,0]));close(pol.v,minimal(50,70,[0,1,0]));close(pol.k,direction(50,70));
 // A coordinate axis is its angles exactly.
 assert.deepEqual(directionAxis([],{reference:{kind:'axis',axis:'-y'},thetaDeg:0,phiDeg:0}),{thetaDeg:90,phiDeg:-90});
});

test('a link connects reference axes: a field\'s own offset leaves linked objects, its reference axis carries them',()=>{
 const a={...field('a',1),direction:{reference:{kind:'angles',thetaDeg:30,phiDeg:40},thetaDeg:20,phiDeg:10}};
 const b={...field('b',1),direction:{reference:{kind:'field',id:'a'},thetaDeg:5,phiDeg:0}},fields=[a,b];
 const light=laserWith({reference:{kind:'field',id:'b'},thetaDeg:0,phiDeg:0});
 const k=()=>polarization(light,fields).k;
 close(k(),direction(30,40),1e-15);
 a.direction.thetaDeg=70;close(k(),direction(30,40),1e-15);
 a.direction.reference={kind:'axis',axis:'+x'};close(k(),[1,0,0],1e-15);
 // The light keeps its own offset from the linked axis.
 light.direction.thetaDeg=90;light.direction.phiDeg=90;close(k(),minimal(90,0,[0,1,0]),1e-15);
 assert.deepEqual(linkableFields(fields,'a').map(f=>f.id),[],'b links to a: a cannot link to b');
 assert.deepEqual(linkableFields(fields,'b').map(f=>f.id),['a']);
 assert.deepEqual(linkableFields(fields).map(f=>f.id),['a','b'],'a light may link to any field');
 a.direction.reference={kind:'field',id:'b'};assert.throws(()=>referenceAxis(fields,light.direction.reference),/成环/);
});

test('a cycle or a missing linked field is refused on import; deleting a linked field keeps every direction',()=>{
 const s=defaultScene();s.fields.push({...field('second',1),direction:{reference:{kind:'field',id:'bias'},thetaDeg:30,phiDeg:0}});
 s.fields[0].direction={reference:{kind:'field',id:'second'},thetaDeg:0,phiDeg:0};assert.throws(()=>roundtrip(s),/参考轴无效/);
 s.fields[0].direction={reference:{kind:'field',id:'nowhere'},thetaDeg:0,phiDeg:0};assert.throws(()=>roundtrip(s),/参考轴无效/);
 s.fields[0].direction=along(35,-20);s.lasers[0].direction={reference:{kind:'field',id:'bias'},thetaDeg:12,phiDeg:7};
 const loaded=roundtrip(s),objects=[...loaded.fields,...loaded.lasers],before=objects.map(o=>directionVector(loaded.fields,o.direction));
 releaseLinks(loaded.fields,objects,'bias');loaded.fields=loaded.fields.filter(f=>f.id!=='bias');
 assert.ok(objects.every(o=>o.direction.reference.kind!=='field'||o.direction.reference.id!=='bias'));
 objects.filter(o=>o.id!=='bias').forEach(o=>assert.deepEqual(directionVector(loaded.fields,o.direction),before[objects.indexOf(o)]));
});

test('沿 +B, 沿 −B and 垂直 B link to the field and place the light against its actual direction exactly',()=>{
 for(const offset of [{thetaDeg:0,phiDeg:0},{thetaDeg:25,phiDeg:-40},{thetaDeg:120,phiDeg:75}]){
  const fields=[{...field('bias',2),direction:{reference:{kind:'angles',thetaDeg:33,phiDeg:-12},...offset}}];
  for(const [mode,chi,psi,component] of [['parallel',45,0,'plus'],['anti',45,0,'minus'],['perpendicular',0,0,'pi']]){
   const dir=shortcutDirection(fields[0],mode),p=polarization(laserWith(dir,{ellipticityDeg:chi,polarizationAngleDeg:psi}),fields);
   assert.deepEqual(dir.reference,{kind:'field',id:'bias'});
   for(const name of ['minus','pi','plus'])if(name!==component)assert.ok(p[name]<1e-30,`${mode} ${JSON.stringify(offset)}: ${name} ${p[name]}`);
   assert.ok(Math.abs(p[component]-1)<1e-15);
  }
  // Along B the frames cancel exactly: no opposite helicity, not even at rounding.
  const along=polarization(laserWith(shortcutDirection(fields[0],'parallel'),{ellipticityDeg:45}),fields);assert.equal(along.minus,0);assert.equal(along.pi,0);
 }
});

test('σ, π are the actual polarization on the total field; at zero field on the first field\'s reference axis',()=>{
 const light=laserWith(spherical(67,-23),{polarizationAngleDeg:19,ellipticityDeg:21}),c=Math.cos(21*DEG),d=Math.sin(21*DEG);
 const projected=(p,b)=>(c*dot(p.u,b))**2+(d*dot(p.v,b))**2;
 // Two fields at an angle: their vector sum is the quantization axis.
 const fields=[field('a',1.3,20,10),field('b',0.7,110,-60)],sum=composeMagneticField(fields).fieldVector,unit=sum.map(v=>v/Math.hypot(...sum));
 let p=polarization(light,fields);assert.ok(Math.abs(p.pi-projected(p,unit))<1e-13);assert.ok(Math.abs(p.angleToBDeg-Math.acos(dot(p.k,unit))/DEG)<1e-9);
 // Cancelling fields: the first field's reference axis, not its offset.
 const zero=[{...field('first',1),direction:{reference:{kind:'angles',thetaDeg:40,phiDeg:30},thetaDeg:25,phiDeg:0}},{...field('second',1),direction:{reference:{kind:'angles',thetaDeg:40,phiDeg:30},thetaDeg:25,phiDeg:0}}];
 zero[1].magnitudeG=-1;const composed=composeMagneticField(zero);assert.equal(composed.fieldMagnitudeG,0);assert.deepEqual(composed.fieldAxis,{thetaDeg:40,phiDeg:30});
 p=polarization(light,zero);assert.ok(Math.abs(p.pi-projected(p,direction(40,30)))<1e-13);
 // No field at all: +z.
 p=polarization(light,[]);assert.ok(Math.abs(p.pi-projected(p,[0,0,1]))<1e-13);
 const s=defaultScene();s.fields[0].magnitudeG=0;s.fields[0].direction={reference:{kind:'axis',axis:'+x'},thetaDeg:30,phiDeg:0};
 assert.deepEqual(analyze(s).fieldAxis,{thetaDeg:90,phiDeg:0});
});

