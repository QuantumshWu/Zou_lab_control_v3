// True single EXE: native static solver/BLAS/SuperLU plus in-memory UI assets.
// No application resource is expanded to disk by the resulting executable.
import {readFile,writeFile,readdir,mkdir,stat,rename} from 'node:fs/promises';
import {resolve,join,dirname,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {deflateRawSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {verifyUiSourceManifest} from './ui-source.mjs';
import {DESKTOP_SOLVER_DEFAULTS} from './desktop-solver-settings.mjs';
import {prepareNativeDependencies} from './prepare-native-deps.mjs';
import {isGeneratedAtomExe} from './memory-exe-format.mjs';
const app=resolve(dirname(fileURLToPath(import.meta.url)),'..'),arg=name=>{const i=process.argv.indexOf(name);return i<0?undefined:process.argv[i+1];};
if(!arg('--out'))throw Error('Use --out <new standalone EXE>.');
const output=resolve(arg('--out')),dist=resolve(arg('--dist')??join(app,'dist'));
const launcher=resolve(arg('--launcher')??join(process.env.ATOM_NATIVE_TARGET_DIR??join(app,'desktop-native/target'),'x86_64-pc-windows-msvc/release/atom-workbench-native.exe'));
verifyUiSourceManifest(dist);
const native=await prepareNativeDependencies();
const source=JSON.parse(await readFile(join(dist,'ui-source-manifest.json'),'utf8'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const backend=JSON.parse(execFileSync(launcher,['--backend-info'],{encoding:'utf8',windowsHide:true,env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));
if(!backend.staticSolver||backend.numericSource?.schema!=='atom-static-numerics/1')throw Error('The launcher does not contain the statically linked solver.');
for(const file of backend.numericSource.files)if(sha(await readFile(join(app,'rust-solver/src',file.file)))!==file.sha256)throw Error('Compiled solver source is stale: '+file.file);
for(const file of backend.numericSource.libraries)if(sha(await readFile(join(native.libDir,file.file)))!==file.sha256)throw Error('Compiled static library is stale: '+file.file);
const files=new Map();
for(const file of source.outputs)files.set('app/dist/'+file.path,await readFile(join(dist,file.path)));
for(const name of ['desktop-bridge.js','desktop-model-worker.js','zlc.ico'])if(!files.has('app/dist/'+name))throw Error('Fresh UI build omitted '+name);
const index=files.get('app/dist/index.html').toString(),nativeIndex=index.replace('</head>','<script src="./desktop-bridge.js"></script>\n</head>');
if(nativeIndex===index)throw Error('Missing HTML head in fresh UI output.');
files.set('app/dist/index.html',Buffer.from(nativeIndex));source.packaging={kind:'native-ipc',originalIndexSha256:sha(index),insertedScript:'desktop-bridge.js'};
const indexOutput=source.outputs.find(file=>file.path==='index.html');indexOutput.sha256=sha(nativeIndex);indexOutput.bytes=Buffer.byteLength(nativeIndex);
files.set('app/dist/ui-source-manifest.json',Buffer.from(JSON.stringify(source,null,2)+'\n'));
const icon=await readFile(join(source.source.repositoryRoot,'crates/zlc-desktop/assets/zlc.ico'));
if(!files.get('app/dist/zlc.ico').equals(icon))throw Error('Application icon differs from current v3.');
if(backend.v3IconSha256!==sha(icon))throw Error('Compiled native icon is stale; rebuild against the current v3 icon.');
const add=async(name,path)=>files.set(name,await readFile(path));
async function addTree(name,path){for(const file of await readdir(path,{withFileTypes:true})){if(file.isDirectory())await addTree(name+'/'+file.name,join(path,file.name));else if(file.isFile())await add(name+'/'+file.name,join(path,file.name));}}
await add('licenses/OpenBLAS.txt',native.licenses.openblas);
await add('licenses/LAPACK.txt',native.licenses.lapack);
for(const [name,file]of [['SUPERLU-LICENSE.txt',native.licenses.superlu],['COLAMD-BSD-LICENSE.txt',native.licenses.colamd],['SuiteSparse-config-README.txt',native.licenses.suiteSparseConfig]])await add('licenses/'+name,file);
await add('licenses/KaTeX.txt',join(app,'node_modules/katex/LICENSE'));
await add('licenses/Microsoft-WebView2.html',native.licenses.webview2);
const common=execFileSync('git',['rev-parse','--path-format=absolute','--git-common-dir'],{cwd:app,encoding:'utf8'}).trim(),tools=process.env.ZLC_V3_TOOLS??join(dirname(common),'.tools'),rust=join(tools,'rust-1.99.1-x86_64-pc-windows-gnu/bin');
const metadata=JSON.parse(execFileSync(join(rust,'cargo.exe'),['metadata','--manifest-path',join(app,'desktop-native/Cargo.toml'),'--features','embedded-solver','--format-version','1','--filter-platform','x86_64-pc-windows-msvc','--locked','--offline'],{env:{...process.env,CARGO_HOME:join(tools,'cargo-home'),RUSTC:join(rust,'rustc.exe')},encoding:'utf8',maxBuffer:16*1024*1024}));
const used=new Set(metadata.resolve.nodes.map(node=>node.id)),notices=[];
for(const pkg of metadata.packages)if(pkg.source?.startsWith('registry+')&&used.has(pkg.id)){
 const folder=dirname(pkg.manifest_path),name=pkg.name+'-'+pkg.version;notices.push(`${name}: ${pkg.license??'see notices'}${pkg.repository?' — '+pkg.repository:''}`);
 for(const file of await readdir(folder,{withFileTypes:true}))if(/^(license|licence|copying|notice)([._-]|$)/i.test(file.name)){if(file.isFile())await add(`licenses/rust/${name}/${file.name}`,join(folder,file.name));else if(file.isDirectory())await addTree(`licenses/rust/${name}/${file.name}`,join(folder,file.name));}
 if(pkg.license_file)await add(`licenses/rust/${name}/${basename(pkg.license_file)}`,join(folder,pkg.license_file));
}
files.set('licenses/Rust-dependencies.txt',Buffer.from(notices.sort().join('\n')+'\n'));
files.set('numeric-source-manifest.json',Buffer.from(JSON.stringify(backend.numericSource,null,2)+'\n'));
const build={name:'Atom Workbench',transport:'native-ipc',node:null,webview:'installed-runtime-required',distribution:'single-exe-memory',numericBackend:'same-executable-static',solverDefaults:DESKTOP_SOLVER_DEFAULTS,v3IconSha256:sha(icon),numericSourceSha256:sha(JSON.stringify(backend.numericSource))};
files.set('build.json',Buffer.from(JSON.stringify(build,null,2)+'\n'));
const entries=[...files].map(([path,data])=>({path,bytes:data.length,sha256:sha(data)}));
if(entries.some(file=>/\.(dll|exe)$/i.test(file.path)))throw Error('Program binaries must be statically linked, not resource sidecars.');
const payload=deflateRawSync(Buffer.concat([...files.values()]),{level:9}),indexBytes=Buffer.from(JSON.stringify({schema:'atom-memory-exe/1',files:entries}));
const footer=Buffer.alloc(96);footer.writeBigUInt64LE(BigInt(payload.length),0);footer.writeBigUInt64LE(BigInt(indexBytes.length),8);Buffer.from(sha(payload),'hex').copy(footer,16);Buffer.from(sha(indexBytes),'hex').copy(footer,48);footer.write('ATOM-MEMORY-V001',80,'ascii');
const host=await readFile(launcher);if(host.subarray(-16).equals(Buffer.from('ATOM-MEMORY-V001')))throw Error('Launcher already contains in-memory assets.');
try{
 await stat(output);
 if(!process.argv.includes('--replace-generated'))throw Error('Refusing to overwrite an existing delivery: '+output);
 const previous=await readFile(output);
 if(!isGeneratedAtomExe(previous))throw Error('Existing output is not a verified generated Atom Workbench EXE: '+output);
}catch(error){if(error.code!=='ENOENT')throw error;}
await mkdir(dirname(output),{recursive:true});const temporary=output+'.assembling-'+process.pid,complete=Buffer.concat([host,payload,indexBytes,footer]);await writeFile(temporary,complete,{flag:'wx'});
try{await rename(temporary,output);}catch(error){throw Error('Cannot atomically install the generated EXE; close a running copy and retry. Previous output is preserved. New file: '+temporary+'; '+error.message);}
const report={output,bytes:complete.length,sha256:sha(complete),hostBytes:host.length,compressedResourcesBytes:payload.length,expandedResourceBytes:entries.reduce((sum,file)=>sum+file.bytes,0),resourceFiles:entries.length,resourceDirectoriesCreated:false,backend,numericResultCache:false,systemDependency:'Microsoft WebView2 Evergreen Runtime',v3IconSha256:sha(icon)};
if(arg('--report'))await writeFile(resolve(arg('--report')),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,backend:{staticSolver:true,numericSourceSha256:build.numericSourceSha256}},null,2));
