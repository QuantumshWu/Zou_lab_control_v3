// Standard one-command release: current v3 source -> fresh UI + bridge ->
// verified static native host -> one memory-resident resource EXE.
import {readFile,mkdir,stat} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {verifyUiSourceManifest} from './ui-source.mjs';
import {prepareNativeDependencies} from './prepare-native-deps.mjs';
import {powershellExecutable} from './build-tools.mjs';
import {collectNumericSourceInputs} from './numeric-source-manifest.mjs';
const app=resolve(dirname(fileURLToPath(import.meta.url)),'..'),arg=name=>{const i=process.argv.indexOf(name);return i<0?undefined:process.argv[i+1];};
const stamp=new Date().toISOString().replace(/[-:.TZ]/g,''),output=resolve(arg('--out')??join(app,'../../atom_workbench.exe'));
const evidence=resolve(arg('--evidence')??join(app,'artifacts/single-exe-release-'+stamp));await mkdir(evidence,{recursive:true});
const run=(program,args)=>execFileSync(program,args,{cwd:app,stdio:'inherit',windowsHide:true});
const node=script=>run(process.execPath,[join(app,'scripts',script)]);
const powershell=powershellExecutable(),psArgs=['-NoProfile','-ExecutionPolicy','Bypass'];
// Validate the same canonical numeric input policy used by build.rs/packer
// before paying for any build. The packer verifies the compiled hashes later.
collectNumericSourceInputs(app);
// This is the sole fresh frontend build. The lower-level packer verifies,
// but never asks the operator to manually rebuild a second UI snapshot.
node('build.mjs');node('build-desktop-native-ui.mjs');verifyUiSourceManifest(join(app,'dist'));
const native=await prepareNativeDependencies();
const hash=async file=>createHash('sha256').update(await readFile(file)).digest('hex');
let prepared=false,validOpenblas=false;
try{const proof=JSON.parse((await readFile(join(native.libDir,'build-provenance.json'),'utf8')).replace(/^\uFEFF/,''));validOpenblas=proof.schema==='atom-native-static-build/1'&&proof.dependencyLockSha256===native.lockSha256&&proof.openblasSha256===await hash(join(native.libDir,'openblas.lib'));prepared=validOpenblas&&proof.superluSha256===await hash(join(native.libDir,'atom_superlu.lib'))
 &&proof.superluSourceManifestSha256===await hash(join(native.superluSource,'source-manifest.json'))&&proof.hooksSha256===await hash(join(native.superluSource,'atom_hooks_static.h'));}catch{}
if(!prepared||process.argv.includes('--rebuild-numerics')){
 run(powershell,[...psArgs,'-File','scripts/build-static-numerics.ps1','-Action',validOpenblas&&!process.argv.includes('--rebuild-numerics')?'superlu':'build']);
}
const uiSource=JSON.parse(await readFile(join(app,'dist/ui-source-manifest.json'),'utf8'));
execFileSync(powershell,[...psArgs,'-File','scripts/build-desktop-native.ps1','-EmbeddedSolver','-RestoreDependencies'],{cwd:app,stdio:'inherit',windowsHide:true,env:{...process.env,ATOM_V3_SOURCE_ROOT:uiSource.source.repositoryRoot,ATOM_V3_ICON_SOURCE:join(uiSource.source.repositoryRoot,'crates/zlc-desktop/assets/zlc.ico')}});
// Re-read actual shared sources after all builds. Concurrent upstream edits
// invalidate the release instead of quietly shipping an old widget bundle.
verifyUiSourceManifest(join(app,'dist'));
run(process.execPath,['scripts/package-memory-exe.mjs','--out',output,'--replace-generated','--report',join(evidence,'build.json')]);
console.log(JSON.stringify({output,evidence,distribution:'single EXE; static solver; in-memory resources'}));
