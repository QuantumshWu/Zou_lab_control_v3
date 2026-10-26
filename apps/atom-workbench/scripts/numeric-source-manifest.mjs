// Build script and packer share this declarative input policy. The binary's
// manifest cannot substitute an incomplete or outside-root file whitelist.
import {readFileSync,readdirSync,lstatSync,realpathSync,existsSync} from 'node:fs';
import {resolve,relative,join,extname,isAbsolute,sep} from 'node:path';
import {createHash} from 'node:crypto';
export const NUMERIC_INPUT_POLICY='native/numeric-source-inputs.json';
export const NUMERIC_MANIFEST_SCHEMA='atom-static-numerics/2';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const within=(root,path)=>{const r=relative(root,path);return r===''||(!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep));};
function inputPath(app,path){
  if(typeof path!=='string'||!path||path.includes('\\')||path.includes(':')||path.split('/').some(p=>!p||p==='.'||p==='..'))throw Error('Invalid app-relative numeric input path');
  const result=resolve(app,path);if(!within(app,result))throw Error('Numeric source path escaped the application');return result;
}
function checked(app,path){
  if(lstatSync(path).isSymbolicLink())throw Error('Numeric source symlinks are not provenance inputs: '+path);
  const actual=realpathSync(path);if(!within(app,actual))throw Error('Numeric source resolves outside the application');return actual;
}
export function collectNumericSourceInputs(appRoot){
  const app=realpathSync(appRoot),policy=JSON.parse(readFileSync(join(app,NUMERIC_INPUT_POLICY),'utf8'));
  if(policy.schema!=='atom-numeric-source-inputs/1'||!Array.isArray(policy.trees)||!Array.isArray(policy.files))throw Error('Unsupported numeric source input policy');
  const paths=new Set();
  function add(file){const actual=checked(app,file);if(!lstatSync(actual).isFile())throw Error('Numeric source input is not a file');paths.add(relative(app,actual).split(sep).join('/'));}
  function walk(directory,extensions){
    directory=checked(app,directory);if(!lstatSync(directory).isDirectory())throw Error('Numeric source tree is not a directory');
    for(const entry of readdirSync(directory,{withFileTypes:true})){
      const path=join(directory,entry.name);if(entry.isSymbolicLink())throw Error('Numeric source tree contains a symlink: '+path);
      if(entry.isDirectory())walk(path,extensions);else if(entry.isFile()&&extensions.includes(extname(entry.name).slice(1).toLowerCase()))add(path);
    }
  }
  for(const tree of policy.trees){
    const path=inputPath(app,tree.path);if(!existsSync(path)){if(tree.required!==false)throw Error('Required numeric source tree missing: '+tree.path);continue;}
    if(!Array.isArray(tree.extensions)||tree.extensions.some(e=>typeof e!=='string'))throw Error('Invalid numeric source extensions');
    walk(path,tree.extensions.map(e=>e.toLowerCase()));
  }
  for(const item of policy.files){const path=inputPath(app,item.path);if(!existsSync(path)){if(item.required!==false)throw Error('Required numeric input missing: '+item.path);continue;}add(path);}
  // Rust String/BTreeMap order is UTF8 byte order, not locale collation.
  return [...paths].sort((a,b)=>Buffer.compare(Buffer.from(a),Buffer.from(b)))
    .map(path=>({path,sha256:sha(readFileSync(inputPath(app,path)))}));
}
export function verifyNumericSourceManifest(appRoot,manifest){
  if(manifest?.schema!==NUMERIC_MANIFEST_SCHEMA||manifest.pathBase!=='atom-workbench'||manifest.inputPolicy!==NUMERIC_INPUT_POLICY||!Array.isArray(manifest.files)){
    throw Error('The launcher lacks recursive canonical numeric provenance; rebuild the static host');
  }
  const expected=collectNumericSourceInputs(appRoot),seen=new Set();
  for(const entry of manifest.files){
    inputPath(realpathSync(appRoot),entry.path);
    if(seen.has(entry.path)||typeof entry.sha256!=='string'||!(/^[a-f0-9]{64}$/).test(entry.sha256))throw Error('Invalid/duplicate compiled numeric source record');
    seen.add(entry.path);
  }
  if(expected.length!==manifest.files.length||expected.some((file,i)=>file.path!==manifest.files[i].path))throw Error('Numeric source file set changed; rebuild the static host');
  for(let i=0;i<expected.length;i++)if(expected[i].sha256!==manifest.files[i].sha256)throw Error('Compiled numeric source is stale: '+expected[i].path);
  return {files:expected.length,schema:NUMERIC_MANIFEST_SCHEMA,pathBase:'atom-workbench'};
}
