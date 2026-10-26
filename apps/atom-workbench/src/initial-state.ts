// The one initial state (ui-design.md §3): a description, InitialStateSpec, edited in
// the scene panel only. A single state is stored as dynamics.initialStateId alone (as
// scene files store it); a mixture as dynamics.initialState, with initialStateId the
// single state that stands for it where one is needed (a laser's diagram origin).
import type {InitialStateSpec,Level,Scene} from './types.js';
import {canonicalInitialState,resolveInitialState} from './types.js';
import {initialPopulations} from './full-master-model.js';
import {formatNumber,idLabel,withUnit} from './format.js';

export type InitialKind=InitialStateSpec['kind'];
export type UnpolarizedManifold=Extract<InitialStateSpec,{kind:'unpolarized'}>['manifold'];
export type TemperatureUnit='K'|'mK'|'μK';
export const INITIAL_KINDS:[InitialKind,string][]=[['state','单态'],['unpolarized','未极化'],['thermal','热分布'],['weights','自定义']];
export const UNPOLARIZED:[UnpolarizedManifold,string][]=[['ground','全部基态'],['F1','F=1'],['F2','F=2']];
export const TEMPERATURE_K:Readonly<Record<TemperatureUnit,number>>={K:1,mK:1e-3,'μK':1e-6};
export const DEFAULT_TEMPERATURE_K=1e-3;

export function initialSpec(scene:Scene):InitialStateSpec{return resolveInitialState(scene.dynamics,scene);}

/** The populations a description puts on these states (normalized), by id. */
export function initialWeights(spec:InitialStateSpec,levels:readonly Level[]):Record<string,number>{
  const p=initialPopulations(spec,[...levels]);return Object.fromEntries(levels.map((l,i)=>[l.id,p[i]!]).filter(([,w])=>(w as number)>0));
}
/** The single state standing for a description: the state itself, else its most populated
 * state (the first of equals, in the states' order). */
export function initialAnchor(spec:InitialStateSpec,levels:readonly Level[]):string{
  if(spec.kind==='state')return spec.id;
  let best='',most=-1;for(const [id,w] of Object.entries(initialWeights(spec,levels)))if(w>most+1e-12){best=id;most=w;}
  return best||levels[0]!.id;
}
/** Writes the description, and nothing else that could contradict it. */
export function setInitialSpec(scene:Scene,spec:InitialStateSpec,levels:readonly Level[]):void{
  const d=scene.dynamics!;
  if(spec.kind==='state'){d.initialStateId=spec.id;delete d.initialState;}
  else{initialPopulations(spec,[...levels]);d.initialState=canonicalInitialState(spec);d.initialStateId=initialAnchor(spec,levels);}
  const anchor=levels.find(l=>l.id===d.initialStateId);if(anchor?.manifold==='ground'){scene.initialF=anchor.F as 1|2;scene.initialM=anchor.m;}
}
/** What choosing another kind starts from: the current state where it has one. */
export function startingSpec(kind:InitialKind,current:InitialStateSpec,levels:readonly Level[]):InitialStateSpec{
  if(kind===current.kind)return current;
  if(kind==='state')return {kind:'state',id:initialAnchor(current,levels)};
  if(kind==='unpolarized')return {kind:'unpolarized',manifold:'ground'};
  if(kind==='thermal')return {kind:'thermal',temperatureK:DEFAULT_TEMPERATURE_K};
  // A mixture's populations, as weights one can read and edit: six digits, none below 1e-9.
  return {kind:'weights',weights:Object.fromEntries(Object.entries(initialWeights(current,levels)).filter(([,w])=>w>=1e-9).map(([id,w])=>[id,Number(w.toPrecision(6))]))};
}
export function naturalTemperatureUnit(kelvin:number):TemperatureUnit{return kelvin>=1?'K':kelvin>=1e-3?'mK':'μK';}
/** One line saying what the initial state is. */
export function initialSummary(spec:InitialStateSpec):string{
  if(spec.kind==='state')return idLabel(spec.id);
  if(spec.kind==='unpolarized')return `未极化 · ${UNPOLARIZED.find(([m])=>m===spec.manifold)![1]}`;
  if(spec.kind==='thermal'){const unit=naturalTemperatureUnit(spec.temperatureK);return `热分布 · T = ${withUnit(spec.temperatureK/TEMPERATURE_K[unit],unit,{digits:4})}`;}
  const entries=Object.entries(spec.weights).filter(([,w])=>w>0),total=entries.reduce((s,[,w])=>s+w,0);
  return entries.length===1?idLabel(entries[0]![0]):`自定义 · ${entries.length} 个态，最大 ${formatNumber(Math.max(...entries.map(([,w])=>w))/total*100,{digits:3})}%`;
}
