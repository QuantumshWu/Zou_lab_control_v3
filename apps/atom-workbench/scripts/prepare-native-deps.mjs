// Reproducible build-only dependencies. No research artifact is a build input.
import fs from 'node:fs/promises';
import {createHash,randomUUID} from 'node:crypto';
import {dirname,resolve,join,relative,isAbsolute,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';

export const appRoot=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const digest=data=>createHash('sha256').update(data).digest('hex');
const readJson=async file=>JSON.parse((await fs.readFile(file,'utf8')).replace(/^\uFEFF/,''));
export const within=(root,file)=>{const p=relative(root,file);return p===''||(!p.startsWith('..\\')&&!p.startsWith('../')&&p!=='..'&&!isAbsolute(p));};
const safe=(root,p)=>{const file=resolve(root,p);if(!within(root,file))throw Error('Dependency path escapes its generated cache: '+p);return file;};
const exists=async file=>fs.stat(file).then(()=>true,()=>false);
const hashFile=async file=>digest(await fs.readFile(file));
async function writeJson(file,value){await fs.mkdir(dirname(file),{recursive:true});await fs.writeFile(file,JSON.stringify(value,null,2)+'\n');}

async function archiveFor(pkg,downloads,seeds,offline){
  if(!/^[a-zA-Z0-9._-]+$/.test(pkg.file)||!/^https:\/\//.test(pkg.url)||! /^[a-f0-9]{64}$/.test(pkg.sha256))throw Error('Malformed pinned dependency: '+pkg.id);
  const destination=safe(downloads,pkg.file);
  if(await exists(destination)){
    if(await hashFile(destination)!==pkg.sha256)throw Error('Cached dependency checksum mismatch; remove only this generated file and retry: '+destination);
    return destination;
  }
  const seedNames=[pkg.file,basename(new URL(pkg.url).pathname)];
  for(const folder of seeds)for(const name of seedNames){
    const candidate=join(folder,name);
    if(await exists(candidate)&&await hashFile(candidate)===pkg.sha256){await fs.copyFile(candidate,destination);return destination;}
  }
  if(offline)throw Error('Pinned dependency is not cached (offline): '+pkg.id);
  const temporary=destination+'.downloading-'+randomUUID();
  try{
    const response=await fetch(pkg.url,{signal:AbortSignal.timeout(120000)});
    if(!response.ok)throw Error(`Dependency download ${response.status}: ${pkg.url}`);
    const bytes=Buffer.from(await response.arrayBuffer());
    if(digest(bytes)!==pkg.sha256)throw Error('Downloaded dependency checksum mismatch: '+pkg.id);
    await fs.writeFile(temporary,bytes,{flag:'wx'});await fs.rename(temporary,destination);
  }catch(error){await fs.unlink(temporary).catch(()=>{});throw error;}
  return destination;
}

async function unpack(pkg,archive,cache){
  if(pkg.format==='file')return archive;
  if(!['zip','tar.gz'].includes(pkg.format)||!['src','tools'].includes(pkg.category))throw Error('Unsupported pinned archive: '+pkg.id);
  const destination=safe(cache,`${pkg.category}/${pkg.id}-${pkg.version}`);
  const marker=join(destination,'.prepared.json');
  const expectedRoot=safe(destination,pkg.archiveRoot??'');
  if(await exists(marker)){
    const proof=await readJson(marker);
    if(proof.archiveSha256!==pkg.sha256)throw Error('Cached extracted dependency changed version: '+destination);
    for(const entry of proof.requiredFiles??[]){
      const file=safe(expectedRoot,entry.path);
      if(!await exists(file)||await hashFile(file)!==entry.sha256)throw Error('Cached extracted dependency changed: '+file);
    }
    if((proof.requiredFiles?.length??0)!==pkg.required.length)throw Error('Incomplete extracted dependency proof: '+destination);
    return expectedRoot;
  }
  if(await exists(destination))throw Error('Incomplete generated dependency directory; remove that directory and retry: '+destination);
  const temporary=safe(cache,`.unpack-${pkg.id}-${randomUUID()}`);await fs.mkdir(temporary,{recursive:true});
  try{
    const names=execFileSync('tar',['-tf',archive],{encoding:'utf8',maxBuffer:16*1024*1024,windowsHide:true}).split(/\r?\n/).filter(Boolean);
    for(const name of names){
      const normalized=name.replaceAll('\\','/');
      if(normalized.startsWith('/')||/^[a-zA-Z]:/.test(normalized)||normalized.split('/').includes('..'))throw Error('Unsafe path in pinned dependency: '+name);
    }
    execFileSync('tar',['-xf',archive,'-C',temporary],{stdio:'pipe',windowsHide:true});
    const extracted=safe(temporary,pkg.archiveRoot??'');
    const requiredFiles=[];
    for(const name of pkg.required){const file=safe(extracted,name);if(!(await fs.stat(file)).isFile())throw Error('Missing dependency file: '+name);requiredFiles.push({path:name,sha256:await hashFile(file)});}
    await writeJson(join(temporary,'.prepared.json'),{schema:'atom-extracted-dependency/1',id:pkg.id,version:pkg.version,archiveSha256:pkg.sha256,requiredFiles});
    await fs.mkdir(dirname(destination),{recursive:true});await fs.rename(temporary,destination);
  }catch(error){
    // This is the new, uniquely named extraction directory created above.
    // The resolved path is checked inside this application's generated cache.
    if(!within(cache,temporary))throw Error('Refusing unsafe temporary cleanup');
    await fs.rm(temporary,{recursive:true,force:true});throw error;
  }
  return expectedRoot;
}

export async function prepareNativeDependencies({cacheRoot=join(appRoot,'.cache/native'),seedDirectories=[],offline=false}={}){
  const cache=resolve(cacheRoot),lockFile=join(appRoot,'native/dependencies.lock.json'),lockBytes=await fs.readFile(lockFile),lock=JSON.parse(lockBytes);
  if(lock.schema!=='atom-native-dependencies/1')throw Error('Unsupported native dependency lock');
  for(const file of lock.localFiles){const input=safe(appRoot,file.path);if(await hashFile(input)!==file.sha256)throw Error('Native build input changed without updating its lock: '+file.path);}
  const downloads=join(cache,'downloads');await fs.mkdir(downloads,{recursive:true});
  const packages={};
  for(const pkg of lock.packages){
    if(!/^[a-zA-Z0-9._-]+$/.test(pkg.id)||!/^[a-zA-Z0-9._-]+$/.test(pkg.version))throw Error('Unsafe dependency identity');
    const archive=await archiveFor(pkg,downloads,seedDirectories.map(p=>resolve(p)),offline);
    packages[pkg.id]=await unpack(pkg,archive,cache);
  }
  const superluSource=join(cache,'superlu-source');await fs.mkdir(superluSource,{recursive:true});
  for(const file of lock.superluInputs){
    if(basename(file.name)!==file.name)throw Error('Invalid curated SuperLU file name');
    const input=file.local?safe(appRoot,file.local):file.path?safe(packages[file.package],file.path):packages[file.package];
    const bytes=await fs.readFile(input);if(digest(bytes)!==file.sha256)throw Error('Curated SuperLU source checksum differs: '+file.name);
    const output=join(superluSource,file.name);if(!await exists(output)||await hashFile(output)!==file.sha256)await fs.writeFile(output,bytes);
  }
  for(const name of ['source-manifest.json','atom_hooks_static.h'])await fs.copyFile(join(appRoot,'native/superlu',name),join(superluSource,name));
  const lockSha256=digest(lockBytes);
  const paths={schema:'atom-native-paths/1',appRoot,cacheRoot:cache,lockSha256,packages,
    buildRoot:join(cache,'build'),libDir:join(cache,'lib'),superluSource,openblasSource:packages.openblas,
    tools:{zig:join(packages.zig,'zig.exe'),cmake:join(packages.cmake,'bin/cmake.exe'),ninja:join(packages.ninja,'ninja.exe')},
    licenses:{openblas:join(packages.openblas,'LICENSE'),lapack:join(packages.openblas,'lapack-netlib/LICENSE'),superlu:join(packages.superlu,'License.txt'),
      colamd:packages['suitesparse-colamd-bsd-license-txt'],suiteSparseConfig:packages['suitesparse-suitesparse-config-readme-txt'],webview2:join(appRoot,'native/licenses/Microsoft-WebView2.html')}};
  for(const directory of [paths.buildRoot,paths.libDir])await fs.mkdir(directory,{recursive:true});
  await writeJson(join(cache,'paths.json'),paths);return paths;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const arg=name=>{const i=process.argv.indexOf(name);return i<0?undefined:process.argv[i+1];};
  const seedDirectories=process.argv.flatMap((value,i)=>value==='--seed-dir'?[process.argv[i+1]]:[]);
  const paths=await prepareNativeDependencies({cacheRoot:arg('--cache')??join(appRoot,'.cache/native'),seedDirectories,offline:process.argv.includes('--offline')});
  console.log(JSON.stringify({paths:join(paths.cacheRoot,'paths.json'),lockSha256:paths.lockSha256,packages:Object.keys(paths.packages).length}));
}
