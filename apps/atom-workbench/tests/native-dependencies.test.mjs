import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,writeFile,rm} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {appRoot,within,prepareNativeDependencies} from '../scripts/prepare-native-deps.mjs';

test('native cache rejects traversal and sibling roots',()=>{
  const root=resolve(appRoot,'.cache/native');
  assert.equal(within(root,join(root,'src/source.c')),true);
  assert.equal(within(root,resolve(root,'../elsewhere')),false);
  assert.equal(within(root,root+'-sibling'),false);
});

async function isolated(run){
  const parent=resolve(appRoot,'.cache/native-prep-tests');await mkdir(parent,{recursive:true});
  const folder=await mkdtemp(join(parent,'case-'));
  try{await run(folder);}finally{
    assert.equal(within(parent,folder),true);
    await rm(folder,{recursive:true,force:true});
  }
}

test('offline preparation fails explicitly instead of using a research artifact',()=>isolated(async cacheRoot=>{
  await assert.rejects(prepareNativeDependencies({cacheRoot,offline:true}),/not cached \(offline\): openblas/);
}));

test('a corrupted cached archive is rejected before extraction or compilation',()=>isolated(async cacheRoot=>{
  const folder=join(cacheRoot,'downloads');await mkdir(folder,{recursive:true});
  await writeFile(join(folder,'OpenBLAS-0.3.33.tar.gz'),'not the pinned archive');
  await assert.rejects(prepareNativeDependencies({cacheRoot,offline:true}),/checksum mismatch/);
}));
