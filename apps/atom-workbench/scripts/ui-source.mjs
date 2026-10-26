// One source of v3 widgets for type checking, JS bundles and CSS. The worktree
// never vendors a second widget implementation or silently uses old assets.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
export const workbenchRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export const worktreeUiRoot=path.resolve(workbenchRoot,'../desktop/ui');
const slash=value=>value.replaceAll('\\','/');
const sha=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const git=(cwd,args)=>{const result=spawnSync('git',['-C',cwd,...args],{encoding:'utf8',windowsHide:true});return result.status===0?result.stdout.trim():null;};
export const within=(root,file)=>{const relative=path.relative(root,file);return relative===''||(!relative.startsWith('..'+path.sep)&&relative!=='..'&&!path.isAbsolute(relative));};
export function resolveUiSource(){
  const common=git(workbenchRoot,['rev-parse','--path-format=absolute','--git-common-dir']);
  if(!process.env.ZLC_V3_UI_ROOT&&!common)throw Error('Cannot locate primary v3 UI; set ZLC_V3_UI_ROOT');
  const uiRoot=fs.realpathSync(path.resolve(process.env.ZLC_V3_UI_ROOT??path.join(path.dirname(common),'apps/desktop/ui')));
  for(const name of ['src/fluent/spin.ts','src/fluent/tree_combo.ts','src/scale.ts','src/title_bar.ts','tokens.css','fluent.css','title_bar.css'])if(!fs.existsSync(path.join(uiRoot,name)))throw Error(`v3 UI source missing ${name}; set ZLC_V3_UI_ROOT to its complete apps/desktop/ui directory`);
  const repositoryRoot=git(uiRoot,['rev-parse','--show-toplevel']);
  const iconPath=path.join(repositoryRoot??path.resolve(workbenchRoot,'../..'),'crates/zlc-desktop/assets/zlc.ico');
  if(!fs.existsSync(iconPath))throw Error(`v3 icon is missing: ${iconPath}`);
  return {uiRoot,repositoryRoot,commit:repositoryRoot?git(repositoryRoot,['rev-parse','HEAD']):null,iconPath};
}
function filesUnder(root){
  const result=[];const visit=folder=>{for(const entry of fs.readdirSync(folder,{withFileTypes:true})){const file=path.join(folder,entry.name);if(entry.isDirectory())visit(file);else if(entry.isFile())result.push(file);}};
  if(fs.existsSync(root))visit(root);return result;
}
const identify=(root,files)=>files.sort().map(file=>({path:slash(path.relative(root,file)),sha256:sha(file)}));
export function captureUiInputs(source=resolveUiSource()){
  const shared=identify(source.uiRoot,[...filesUnder(path.join(source.uiRoot,'src')),path.join(source.uiRoot,'tokens.css'),path.join(source.uiRoot,'fluent.css'),path.join(source.uiRoot,'title_bar.css')]);
  const local=identify(workbenchRoot,[...filesUnder(path.join(workbenchRoot,'src')),...filesUnder(path.join(workbenchRoot,'desktop-native/ui')),
    ...['index.html','style.css','tsconfig.json','scripts/build.mjs','scripts/ui-source.mjs','scripts/build-desktop-native-ui.mjs','scripts/native-service-core.mjs','scripts/native-protocol.mjs'].map(p=>path.join(workbenchRoot,p))]);
  return {source:{...source,files:shared,iconSha256:sha(source.iconPath)},workbench:{root:workbenchRoot,files:local}};
}
export function assertUiInputs(snapshot){
  const current=resolveUiSource();if(path.normalize(current.uiRoot)!==path.normalize(snapshot.source.uiRoot)||current.commit!==snapshot.source.commit)throw Error('v3 UI source/commit changed; rebuild the frontend');
  for(const group of [snapshot.source,snapshot.workbench])for(const entry of group.files){const root=group.uiRoot??group.root,file=path.resolve(root,entry.path);if(!within(root,file)||!fs.existsSync(file)||sha(file)!==entry.sha256)throw Error(`UI source changed; fresh build required: ${file}`);}
  const after=captureUiInputs(current);if(JSON.stringify(after)!==JSON.stringify(snapshot))throw Error('UI source file set changed during the build; retry the fresh build');
}
export function redirectUiImport(specifier,importer,source){
  if(!specifier.startsWith('.'))return null;
  const candidate=path.resolve(path.dirname(importer),specifier);if(!within(worktreeUiRoot,candidate))return null;
  const target=path.join(source.uiRoot,path.relative(worktreeUiRoot,candidate));
  return target.endsWith('.js')?target.slice(0,-3)+'.ts':target;
}
export function uiResolver(source){return {name:'current-v3-ui',setup(build){build.onResolve({filter:/^\./},args=>{const target=redirectUiImport(args.path,args.importer,source);return target?{path:target}:undefined;});}};}
export function typecheckAndEmit(ts,configFile,source,output){
  const config=ts.readConfigFile(configFile,ts.sys.readFile);if(config.error)throw Error(ts.flattenDiagnosticMessageText(config.error.messageText,'\n'));
  const parsed=ts.parseJsonConfigFileContent(config.config,ts.sys,path.dirname(configFile));
  // A virtual output mapping preserves the public dist/js layout even though
  // the primary shared source and worktree live in different directories.
  const options={...parsed.options,rootDir:path.parse(workbenchRoot).root,outDir:output};
  const host=ts.createCompilerHost(options);
  host.resolveModuleNames=(names,importer)=>names.map(name=>{
    const target=redirectUiImport(name,importer,source);
    // Preserve the source's .js specifier semantics; TypeScript itself maps
    // it to .ts. Resolving an invented .ts specifier marks the import as a
    // TypeScript-extension import and contradicts the original syntax node.
    return ts.resolveModuleName(target?.replace(/\.ts$/,'.js')??name,importer,options,host).resolvedModule;
  });
  host.writeFile=(_name,content,_bom,_error,sources)=>{
    const file=sources?.[0]?.fileName;if(!file)throw Error('Unexpected TypeScript output without source');
    const relative=within(source.uiRoot,file)?path.join('desktop/ui',path.relative(source.uiRoot,file)):
      within(path.join(workbenchRoot,'src'),file)?path.join('atom-workbench',path.relative(workbenchRoot,file)):null;
    if(!relative)throw Error(`Unexpected TypeScript source outside Workbench/shared v3 UI: ${file}`);
    const destination=path.join(output,relative.replace(/\.tsx?$/,'.js'));fs.mkdirSync(path.dirname(destination),{recursive:true});fs.writeFileSync(destination,content);
  };
  const program=ts.createProgram(parsed.fileNames,options,host),diagnostics=ts.getPreEmitDiagnostics(program);
  if(diagnostics.length){console.error(ts.formatDiagnosticsWithColorAndContext(diagnostics,{getCanonicalFileName:x=>x,getCurrentDirectory:()=>workbenchRoot,getNewLine:()=> '\n'}));throw Error('TypeScript validation against current v3 UI failed');}
  if(program.emit().emitSkipped)throw Error('TypeScript emission failed');
}
function outputRecord(dist,relative){const file=path.join(dist,relative);return {path:slash(relative),sha256:sha(file),bytes:fs.statSync(file).size};}
export function writeUiSourceManifest(dist,snapshot){
  assertUiInputs(snapshot);
  const outputs=['index.html','app.bundle.js','style.css','zlc.ico',...filesUnder(path.join(dist,'vendor')).map(file=>path.relative(dist,file))].map(name=>outputRecord(dist,name));
  const manifest={schema:'atom-workbench-v3-ui/1',createdAt:new Date().toISOString(),buildEntry:'scripts/build.mjs',...snapshot,outputs};
  fs.writeFileSync(path.join(dist,'ui-source-manifest.json'),JSON.stringify(manifest,null,2));return manifest;
}
export function verifyUiSourceManifest(dist,{allowPackaging=true}={}){
  const file=path.join(dist,'ui-source-manifest.json');if(!fs.existsSync(file))throw Error('Current v3 UI provenance is missing; run a fresh frontend build');
  const manifest=JSON.parse(fs.readFileSync(file,'utf8'));if(manifest.schema!=='atom-workbench-v3-ui/1')throw Error('Unsupported UI provenance schema');
  assertUiInputs({source:manifest.source,workbench:manifest.workbench});
  if(manifest.packaging&&!allowPackaging)throw Error('Expected an unmodified frontend build, received packaged HTML');
  if(manifest.packaging){
    const p=manifest.packaging;
    const original=fs.readFileSync(path.join(workbenchRoot,'index.html'),'utf8').replace('./js/atom-workbench/src/app.js','./app.bundle.js');
    const originalSha=createHash('sha256').update(original).digest('hex');
    if(p.kind!=='native-ipc'||p.insertedScript!=='desktop-bridge.js'||p.originalIndexSha256!==originalSha||
       fs.readFileSync(path.join(dist,'index.html'),'utf8')!==original.replace('</head>','<script src="./desktop-bridge.js"></script>\n</head>'))throw Error('Packaged UI has an unverified HTML transformation');
  }
  for(const entry of manifest.outputs){const target=path.resolve(dist,entry.path);if(!within(dist,target)||!fs.existsSync(target)||sha(target)!==entry.sha256||fs.statSync(target).size!==entry.bytes)throw Error(`UI output changed or missing: ${entry.path}`);}
  return manifest;
}
export function appendNativeUiOutputs(dist){
  const manifest=verifyUiSourceManifest(dist,{allowPackaging:false});
  manifest.outputs=manifest.outputs.filter(x=>!['desktop-bridge.js','desktop-model-worker.js'].includes(x.path));
  manifest.outputs.push(...['desktop-bridge.js','desktop-model-worker.js'].map(name=>outputRecord(dist,name)));
  manifest.nativeUiBuiltAt=new Date().toISOString();fs.writeFileSync(path.join(dist,'ui-source-manifest.json'),JSON.stringify(manifest,null,2));return manifest;
}
