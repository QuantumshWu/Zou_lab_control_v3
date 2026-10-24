import test from 'node:test';
import assert from 'node:assert/strict';
import {stateTree,transitionTree} from '../dist/js/atom-workbench/src/choice-trees.js';
import {flattenTree} from '../dist/js/desktop/ui/src/fluent/tree_model.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {zeroFieldTransitionOptions} from '../dist/js/atom-workbench/src/laser-frequency.js';
const a=analyze(defaultScene()),states=[...a.ground,...a.D1,...a.D2];
test('state choices retain all 32 identities under J and F, with symmetric m order',()=>{
 const tree=stateTree(states),flat=flattenTree(tree),leaves=flat.flatMap(g=>g.leaves);
 assert.equal(tree.length,3);assert.deepEqual(new Set(leaves.map(l=>l.key)),new Set(states.map(s=>s.id)));
 assert.ok(flat.filter(g=>g.leaves.length).every(g=>g.depth===1&&g.parent!==null));
 assert.deepEqual(tree[0].children[0].leaves.map(l=>l.key),['ground-1--1','ground-1-0','ground-1-1']);
 assert.ok(tree[1].children[0].heading.includes('F″'));assert.ok(tree[2].children[0].heading.includes('F′'));
});
test('repeated F labels have distinct branch identities and references remain separate',()=>{
 const tree=stateTree(states,[{id:'reference-1',name:'中间态'}]),flat=flattenTree(tree);
 assert.equal(new Set(flat.map(g=>g.path)).size,flat.length);
 assert.deepEqual(tree.at(-1).leaves,[{key:'reference-1',label:'中间态',text:'中间态 · 参考能级'}]);
});
test('transition tree preserves every energy reference, including dipole-forbidden references',()=>{
 const leaves=flattenTree(transitionTree()).flatMap(g=>g.leaves),references=zeroFieldTransitionOptions();
 assert.deepEqual(new Set(leaves.map(l=>l.key)),new Set(references.map(r=>`${r.fromManifold??'ground'}-${r.line}-${r.Fg}-${r.Fe}`)));
 assert.ok(leaves.some(l=>l.key==='ground-D2-1-3'));assert.ok(leaves.some(l=>l.key.startsWith('D1-D2-')));
});
test('legacy two-level v3 trees retain ordering and leaf text',()=>{
 const groups=[{heading:'Devices',leaves:[{key:'a',label:'A',text:'Device A'}]},{heading:'Signals',leaves:[{key:'b',label:'B',text:'Signal B'}]}];
 const flat=flattenTree(groups);assert.deepEqual(flat.map(g=>g.leaves),groups.map(g=>g.leaves));assert.ok(flat.every(g=>g.depth===0&&g.parent===null));
});
