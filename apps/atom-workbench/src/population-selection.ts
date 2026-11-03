import type {DynamicsResult} from './solver-types.js';
import type {DynamicsSettings} from './types.js';
import {hyperfineLabel} from './format.js';
export const POPULATION_GROUPS=[
 ...(['ground','D1','D2'] as const).flatMap(manifold=>(manifold==='D2'?[0,1,2,3]:[1,2]).map(F=>({
  id:`F:${manifold}:${F}`,label:hyperfineLabel(manifold,F),
  members:Array.from({length:2*F+1},(_,i)=>`${manifold}-${F}-${i-F}`),
 }))),
 {id:'F:excited',label:'激发态总布居',members:(['D1','D2'] as const).flatMap(manifold=>(manifold==='D1'?[1,2]:[0,1,2,3]).flatMap(F=>Array.from({length:2*F+1},(_,i)=>`${manifold}-${F}-${i-F}`)))},
];
/** 丢失: a motion result's probability that the atom has left the trap, a curve beside the states'
 * (each the probability of that state with the atom still held): together they add up to one. */
export const LOSS_CURVE={id:'loss',label:'丢失'} as const;
/** The curves a scene may show besides single states: the F groups, and with motion 丢失. */
export const DERIVED_CURVE_IDS:readonly string[]=[...POPULATION_GROUPS.map(g=>g.id),LOSS_CURVE.id];
/** 初末各 3 态: the three most populated curves at the first and at the last sample -- states, and
 * for a motion result 丢失 among them. Display selection only. Do not fill top-three slots with
 * zero readouts or eigensystem roundoff; the screen displays probabilities to 0.01%. */
export function recommendedCurves(result:DynamicsResult|undefined,initialId:string):string[]{
 if(!result||result.timesUs.length<2)return [initialId];
 const loss=result.motion?.lossProbability;
 const rank=(index:number)=>[...result.states.map((s,order)=>({id:s.id,order,p:result.populations[s.id]?.[index]??0})),...(loss?[{id:LOSS_CURVE.id,order:result.states.length,p:loss[index]!}]:[])]
  .filter(s=>Number.isFinite(s.p)&&Math.round(s.p*10000)>0).sort((a,b)=>b.p-a.p||a.order-b.order).slice(0,3).map(s=>s.id);
 return [...new Set([...rank(0),...rank(result.timesUs.length-1)])];
}
export function populationDisplayIds(settings:DynamicsSettings):string[]{return [...new Set([...settings.monitoredStateIds,...(settings.monitoredGroupIds??[])])];}
/** The result with the derived curves asked for: each F group the sum of its states, and 丢失 a
 * motion result's loss. The solver's data stay as they are. */
export function withPopulationGroups(result:DynamicsResult,groups:readonly string[]):DynamicsResult{
 const populations={...result.populations},mean=result.mean.populations?{...result.mean.populations}:null;
 if(groups.includes(LOSS_CURVE.id)&&result.motion)populations[LOSS_CURVE.id]=result.motion.lossProbability;
 for(const id of groups){const group=POPULATION_GROUPS.find(g=>g.id===id);if(!group)continue;
  populations[id]=result.timesUs.map((_,i)=>group.members.reduce((sum,state)=>sum+(result.populations[state]?.[i]??0),0));
  if(mean)mean[id]=group.members.reduce((sum,state)=>sum+(result.mean.populations?.[state]??0),0);
 }
 return {...result,populations,mean:{...result.mean,populations:mean}};
}
