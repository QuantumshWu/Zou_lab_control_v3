import type {TreeGroup} from '../../desktop/ui/src/fluent/tree_combo.js';
import type {Level} from './types.js';
import {zeroFieldTransitionOptions} from './laser-frequency.js';
import {transitionKey} from './frequency-value.js';
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

/** The bases a frequency can be read or defined relative to (ui-design.md §1.3–1.4):
 * 绝对 (with its words, and what the closed box says for it), the zero-field transitions,
 * and other lasers when offered. */
export function frequencyBaseTree(options:{absolute:string;absoluteText?:string;lasers?:readonly {id:string;name:string}[]}):TreeGroup[]{
 const groups:TreeGroup[]=[{heading:'绝对',leaves:[{key:'absolute',label:options.absolute,text:options.absoluteText??`绝对（${options.absolute}）`}]}];
 for(const group of transitionTree())groups.push({heading:group.heading,leaves:[],children:group.children!.map(child=>({heading:child.heading,leaves:child.leaves.map(leaf=>{const [from,line,Fg,Fe]=leaf.key.split('-');return {...leaf,key:transitionKey({...(from==='D1'?{fromManifold:'D1' as const}:{}),line:line as 'D1'|'D2',Fg:Number(Fg) as 1|2,Fe:Number(Fe)})};})}))});
 if(options.lasers?.length)groups.push({heading:'激光',leaves:options.lasers.map(l=>({key:`laser:${l.id}`,label:l.name,text:`激光 ${l.name}`}))});
 return groups;
}
