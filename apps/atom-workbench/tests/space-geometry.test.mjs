import assert from 'node:assert/strict';
import test from 'node:test';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {polarization} from '../dist/js/atom-workbench/src/physics.js';
import {electricFieldVector} from '../dist/js/atom-workbench/src/space3d.js';
import {fieldAlong,spherical} from './directions.mjs';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);

test('displayed polarization stays perpendicular to physical k at every phase and direction',()=>{
  for(const theta of [0,27,90,163,180])for(const phi of [-180,-35,0,73,180])for(const chi of [-45,-19,0,32,45]){
    const laser={...defaultScene().lasers[0],direction:spherical(theta,phi),polarizationAngleDeg:31,ellipticityDeg:chi,phaseDeg:43};
    const pol=polarization(laser,[fieldAlong([.7,-.3,.2])]);
    for(const phase of [0,.37,Math.PI/2,Math.PI,5.2])close(dot(electricFieldVector(laser,pol,phase),pol.k),0);
  }
});

test('polarization ellipse semiaxes and optical phase match the Jones convention',()=>{
  const laser={...defaultScene().lasers[0],direction:spherical(72,-29),polarizationAngleDeg:37,ellipticityDeg:23,phaseDeg:41},pol=polarization(laser,[fieldAlong([0,0,3])]),p=41*Math.PI/180;
  const major=electricFieldVector(laser,pol,p),minor=electricFieldVector(laser,pol,p+Math.PI/2);
  close(Math.hypot(...major),Math.cos(23*Math.PI/180));close(Math.hypot(...minor),Math.sin(23*Math.PI/180));close(dot(major,minor),0);
  close(dot(major,pol.v),0);close(dot(minor,pol.u),0);
});

test('positive and negative circular polarization have the same handedness as solver sigma weights',()=>{
  for(const chi of [-45,45]){
    const laser={...defaultScene().lasers[0],direction:spherical(0,0),ellipticityDeg:chi,phaseDeg:0},pol=polarization(laser,[fieldAlong([0,0,3])]);
    const first=electricFieldVector(laser,pol,0),quarter=electricFieldVector(laser,pol,Math.PI/2);
    close(Math.hypot(...first),Math.SQRT1_2);close(Math.hypot(...quarter),Math.SQRT1_2);
    assert.equal(Math.sign(dot(cross(first,quarter),pol.k)),Math.sign(chi));
    close(chi>0?pol.plus:pol.minus,1);
  }
});
