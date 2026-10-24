import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {deflateRawSync} from 'node:zlib';
import {isGeneratedAtomExe} from '../scripts/memory-exe-format.mjs';
const sha=data=>createHash('sha256').update(data).digest();
function fixture(name='Atom Workbench'){
  const build=Buffer.from(JSON.stringify({name,distribution:'single-exe-memory',numericBackend:'same-executable-static'}));
  const payload=deflateRawSync(build),index=Buffer.from(JSON.stringify({schema:'atom-memory-exe/1',files:[{path:'build.json',bytes:build.length,sha256:sha(build).toString('hex')}]}));
  const host=Buffer.alloc(128);host.write('MZ');const footer=Buffer.alloc(96);
  footer.writeBigUInt64LE(BigInt(payload.length));footer.writeBigUInt64LE(BigInt(index.length),8);sha(payload).copy(footer,16);sha(index).copy(footer,48);footer.write('ATOM-MEMORY-V001',80);
  return Buffer.concat([host,payload,index,footer]);
}
test('only an intact Atom Workbench generated container can be replaced',()=>{
  assert.equal(isGeneratedAtomExe(fixture()),true);
  assert.equal(isGeneratedAtomExe(Buffer.from('ordinary user executable')),false);
  assert.equal(isGeneratedAtomExe(fixture('Different application')),false);
  const broken=fixture();broken[130]^=1;assert.equal(isGeneratedAtomExe(broken),false);
  const badLength=fixture();badLength.writeBigUInt64LE(2n**62n,badLength.length-96);assert.equal(isGeneratedAtomExe(badLength),false);
});
