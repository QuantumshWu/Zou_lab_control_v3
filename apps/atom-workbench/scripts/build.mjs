import {existsSync, mkdirSync, copyFileSync, readFileSync, writeFileSync, cpSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, resolve, join} from 'node:path';
import {createRequire} from 'node:module';
import {resolveUiSource,captureUiInputs,typecheckAndEmit,uiResolver,writeUiSourceManifest} from './ui-source.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const {build} = require('esbuild');
const uiSource=resolveUiSource(),inputSnapshot=captureUiInputs(uiSource);
let compiler;
try { compiler = require.resolve('typescript/lib/tsc.js'); }
catch {
  const candidates = [process.env.ZLC_TYPESCRIPT, join(root, '../desktop/ui/node_modules/typescript/lib/tsc.js')];
  compiler = candidates.find(path => path && existsSync(path));
}
if (!compiler) throw new Error('TypeScript is missing. Run npm ci in apps/atom-workbench, or set ZLC_TYPESCRIPT to an installed tsc.js.');
const ts=require(join(dirname(compiler),'typescript.js'));
typecheckAndEmit(ts,join(root,'tsconfig.json'),uiSource,join(root,'dist/js'));
mkdirSync(join(root, 'dist/vendor'), {recursive:true});
for (const file of ['index.html','style.css']) copyFileSync(join(root,file), join(root,'dist',file));
copyFileSync(uiSource.iconPath,join(root,'dist/zlc.ico'));
for (const file of ['tokens.css','fluent.css']) copyFileSync(join(uiSource.uiRoot,file), join(root,'dist/vendor',file));
const katexDir=join(root,'node_modules/katex/dist');
mkdirSync(join(root,'dist/vendor/katex'),{recursive:true});
copyFileSync(join(katexDir,'katex.min.css'),join(root,'dist/vendor/katex/katex.min.css'));
cpSync(join(katexDir,'fonts'),join(root,'dist/vendor/katex/fonts'),{recursive:true});
// The desktop packer embeds this same current-v3 UI bundle in memory.
await build({entryPoints:[join(root,'src/app.ts')],bundle:true,plugins:[uiResolver(uiSource)],format:'iife',target:'es2022',outfile:join(root,'dist/app.bundle.js'),legalComments:'inline'});
await build({entryPoints:[join(uiSource.uiRoot,'src/scale.ts')],bundle:true,format:'iife',target:'es2022',outfile:join(root,'dist/vendor/scale.js')});
const index=readFileSync(join(root,'index.html'),'utf8').replace('./js/atom-workbench/src/app.js','./app.bundle.js');
writeFileSync(join(root,'dist/index.html'),index);
writeUiSourceManifest(join(root,'dist'),inputSnapshot);
console.log('Built atom workbench: dist/index.html');
console.log(`Shared v3 UI: ${uiSource.uiRoot} (${uiSource.commit??'explicit source directory'})`);
