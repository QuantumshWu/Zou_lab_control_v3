import type {DynamicsResult} from './solver-types.js';
import type {DynamicsSettings} from './types.js';
export const POPULATION_GROUPS=[
 ...(['ground','D1','D2'] as const).flatMap(manifold=>(manifold==='D2'?[0,1,2,3]:[1,2]).map(F=>({
  id:`F:${manifold}:${F}`,label:`${manifold==='ground'?'5S₁/₂':manifold==='D1'?'5P₁/₂':'5P₃/₂'} · ${manifold==='ground'?'F':manifold==='D1'?'F″':'F′'}=${F}`,
  members:Array.from({length:2*F+1},(_,i)=>`${manifold}-${F}-${i-F}`),
 }))),
 {id:'F:excited',label:'激发态总布居',members:(['D1','D2'] as const).flatMap(manifold=>(manifold==='D1'?[1,2]:[0,1,2,3]).flatMap(F=>Array.from({length:2*F+1},(_,i)=>`${manifold}-${F}-${i-F}`)))},
];
/** Display selection only. Do not fill top-three slots with zero readouts or
 * eigensystem roundoff; the screen displays probabilities to 0.01%. */
export function recommendedPopulationStates(result:DynamicsResult|undefined,initialId:string):string[]{
 if(!result||result.timesUs.length<2)return [initialId];
 const rank=(index:number)=>result.states.map((s,order)=>({id:s.id,order,p:result.populations[s.id]?.[index]??0}))
  .filter(s=>Number.isFinite(s.p)&&Math.round(s.p*10000)>0).sort((a,b)=>b.p-a.p||a.order-b.order).slice(0,3).map(s=>s.id);
 return [...new Set([...rank(0),...rank(result.timesUs.length-1)])];
}
export function populationDisplayIds(settings:DynamicsSettings):string[]{return [...new Set([...settings.monitoredStateIds,...(settings.monitoredGroupIds??[])])];}
export function withPopulationGroups(result:DynamicsResult,groups:readonly string[]):DynamicsResult{
 const populations={...result.populations},steady=result.steady.populations?{...result.steady.populations}:null;
 for(const id of groups){const group=POPULATION_GROUPS.find(g=>g.id===id);if(!group)continue;
  populations[id]=result.timesUs.map((_,i)=>group.members.reduce((sum,state)=>sum+(result.populations[state]?.[i]??0),0));
  if(steady)steady[id]=group.members.reduce((sum,state)=>sum+(result.steady.populations?.[state]??0),0);
 }
 return {...result,populations,steady:{...result.steady,populations:steady}};
}
