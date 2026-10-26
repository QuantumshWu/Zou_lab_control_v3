import type {TreeGroup} from '../../desktop/ui/src/fluent/tree_combo.js';
import type {Level} from './types.js';
import {zeroFieldTransitionOptions} from './laser-frequency.js';
import {TERMS as terms,fSymbol,signedInteger,stateLabel} from './format.js';
const fLabel=(m:string)=>fSymbol(m as Level['manifold']);
export function stateTree(states:readonly Level[],references:readonly {id:string;name:string}[]=[]):TreeGroup[]{
 const groups:TreeGroup[]=(['ground','D1','D2']as const).map(manifold=>({heading:terms[manifold],leaves:[],children:[...new Set(states.filter(s=>s.manifold===manifold).map(s=>s.F))].sort((a,b)=>a-b).map(F=>({heading:`${fLabel(manifold)}=${F}`,leaves:states.filter(s=>s.manifold===manifold&&s.F===F).sort((a,b)=>a.m-b.m).map(s=>({key:s.id,label:`m=${signedInteger(s.m)}`,text:stateLabel(s)}))}))}));
 if(references.length)groups.push({heading:'参考能级',leaves:references.map(r=>({key:r.id,label:r.name,text:r.name+' · 参考能级'}))});return groups;
}
export function transitionTree():TreeGroup[]{
 const refs=zeroFieldTransitionOptions();const groups:TreeGroup[]=[];
 for(const r of refs){const from=r.fromManifold??'ground',heading=`${terms[from]} → ${terms[r.line]}`;let group=groups.find(g=>g.heading===heading);if(!group){group={heading,leaves:[],children:[]};groups.push(group);}const branch=`${fLabel(from)}=${r.Fg}`;let child=group.children!.find(g=>g.heading===branch);if(!child){child={heading:branch,leaves:[]};group.children!.push(child);}child.leaves.push({key:`${from}-${r.line}-${r.Fg}-${r.Fe}`,label:`${fLabel(r.line)}=${r.Fe}`,text:`${terms[from]} ${fLabel(from)}=${r.Fg} → ${terms[r.line]} ${fLabel(r.line)}=${r.Fe}`});}
 return groups;
}
