import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {dirname,join,resolve,relative,isAbsolute,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {collectNumericSourceInputs,verifyNumericSourceManifest,NUMERIC_INPUT_POLICY,NUMERIC_MANIFEST_SCHEMA} from '../scripts/numeric-source-manifest.mjs';

const app=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const parent=join(app,'.cache','provenance-tests');
async function put(root,path,text){const file=join(root,path);await mkdir(dirname(file),{recursive:true});await writeFile(file,text);}
async function fixture(){
  await mkdir(parent,{recursive:true});const root=await mkdtemp(join(parent,'numeric-'));
  const policy=JSON.parse(await readFile(join(app,NUMERIC_INPUT_POLICY),'utf8'));
  for(const file of policy.files)if(file.required)await put(root,file.path,'fixture '+file.path+'\n');
  await put(root,NUMERIC_INPUT_POLICY,JSON.stringify(policy));
  await put(root,'rust-solver/src/main.rs','fn main() {}\n');
  await put(root,'rust-solver/src/nested/operator.rs','pub fn operator() {}\n');
  await put(root,'rust-solver/src/nested/ignored.txt','not Rust source\n');
  return root;
}
async function cleanup(root){
  const target=resolve(root),rel=relative(resolve(parent),target);
  assert.ok(rel&&rel!=='..'&&!rel.startsWith('..'+sep)&&!isAbsolute(rel),'Only the created fixture may be removed');
  await rm(target,{recursive:true,force:true});
}
const manifest=root=>({schema:NUMERIC_MANIFEST_SCHEMA,pathBase:'atom-workbench',inputPolicy:NUMERIC_INPUT_POLICY,files:collectNumericSourceInputs(root),libraries:[]});

test('recursive Rust/core sources and manifests share an exact relative file set',async()=>{
  const root=await fixture();
  try{
    const before=manifest(root);assert.ok(before.files.some(x=>x.path==='rust-solver/src/nested/operator.rs'));
    assert.ok(before.files.some(x=>x.path==='desktop-native/solver/Cargo.toml'));
    assert.ok(!before.files.some(x=>x.path.endsWith('ignored.txt')));
    verifyNumericSourceManifest(root,before);
    await put(root,'rust-solver/population-core/Cargo.toml','[package]\nname="population-core"\n');
    await put(root,'rust-solver/population-core/src/deep/solver.rs','pub fn solve() {}\n');
    assert.throws(()=>verifyNumericSourceManifest(root,before),/file set changed/);
    const current=manifest(root);verifyNumericSourceManifest(root,current);
    assert.ok(current.files.some(x=>x.path==='rust-solver/population-core/src/deep/solver.rs'));
    assert.ok(current.files.every(x=>!x.path.includes('\\')&&!isAbsolute(x.path)));
    await put(root,'rust-solver/population-core/src/deep/solver.rs','pub fn solve() { /* changed */ }\n');
    assert.throws(()=>verifyNumericSourceManifest(root,current),/stale.*population-core/);
  }finally{await cleanup(root);}
});

test('old manifests, incomplete sets, duplicate paths and traversal are rejected',async()=>{
  const root=await fixture();
  try{
    const good=manifest(root);
    assert.throws(()=>verifyNumericSourceManifest(root,{...good,schema:'atom-static-numerics/1'}),/recursive canonical/);
    assert.throws(()=>verifyNumericSourceManifest(root,{...good,files:good.files.slice(1)}),/file set changed/);
    assert.throws(()=>verifyNumericSourceManifest(root,{...good,files:[...good.files,good.files[0]]}),/duplicate/);
    assert.throws(()=>verifyNumericSourceManifest(root,{...good,files:[{path:'../outside.rs',sha256:'0'.repeat(64)}]}),/relative|escaped/);
    const policy=JSON.parse(await readFile(join(root,NUMERIC_INPUT_POLICY),'utf8'));policy.trees.push({path:'../outside',extensions:['rs'],required:false});
    await put(root,NUMERIC_INPUT_POLICY,JSON.stringify(policy));
    assert.throws(()=>collectNumericSourceInputs(root),/relative|escaped/);
  }finally{await cleanup(root);}
});
