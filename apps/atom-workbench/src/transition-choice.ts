// The two-state readout's labels 1 and 2 (ui-design.md §1.8). Nothing is labelled until the
// operator labels it: a click on a state gives it the first free label, 1 before 2; a click on a
// labelled state removes its label; with both set a click on a third state replaces 2 and keeps
// 1. Labels never move between states by themselves.
export type StateLabels={first?:string;second?:string};
export type LabelSlot='first'|'second';

/** A click on a state in the levels chart. */
export function clickState(labels:StateLabels,id:string):StateLabels{
  if(labels.first===id)return withLabel(labels,'first',undefined);
  if(labels.second===id)return withLabel(labels,'second',undefined);
  return withLabel(labels,labels.first===undefined?'first':'second',id);
}
/** One label set (or removed, `id` undefined) on its own; a state holds one label at most, so the
 * other label leaves a state that takes this one. */
export function withLabel(labels:StateLabels,slot:LabelSlot,id:string|undefined):StateLabels{
  const other:LabelSlot=slot==='first'?'second':'first',next:StateLabels={...labels,[slot]:id};
  if(id!==undefined&&next[other]===id)delete next[other];
  if(id===undefined)delete next[slot];
  return next;
}
/** ⇄: the operator's own exchange of 1 and 2. */
export function swapLabels(labels:StateLabels):StateLabels{return {...(labels.second!==undefined?{first:labels.second}:{}),...(labels.first!==undefined?{second:labels.first}:{})};}
/** Labels whose states no longer exist are removed. */
export function keepLabels(labels:StateLabels,exists:(id:string)=>boolean):StateLabels{
  return {...(labels.first!==undefined&&exists(labels.first)?{first:labels.first}:{}),...(labels.second!==undefined&&exists(labels.second)?{second:labels.second}:{})};
}
/** The labelled states in label order, for the chart's badges. */
export function labelled(labels:StateLabels):{id:string;label:1|2}[]{
  return [...(labels.first!==undefined?[{id:labels.first,label:1 as const}]:[]),...(labels.second!==undefined?[{id:labels.second,label:2 as const}]:[])];
}
/** Saved labels, read back; anything else is no labels. */
export function readLabels(input:unknown):StateLabels{
  if(!input||typeof input!=='object'||Array.isArray(input))return {};
  const o=input as Record<string,unknown>,id=(v:unknown)=>typeof v==='string'&&v?v:undefined;
  return withLabel(id(o.first)!==undefined?{first:id(o.first)}:{},'second',id(o.second));
}
