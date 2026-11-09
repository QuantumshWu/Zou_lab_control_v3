// The app's own model builder, headless: the TypeScript of this worktree compiled by the repo's tool (scripts/ui-source.mjs
// `typecheckAndEmit`, the same type check and emit as scripts/build.mjs, into apps/atom-workbench/dist/js) and the one
// mapping from a scene to the native request (scripts/native-protocol.mjs `prepareNativeSolve`). Nothing here models an
// atom: every request and every level shift comes from the app's code (full-master-model.ts, spectroscopy.ts).
//
// TypeScript: the workbench's own node_modules when present, else ZLC_TYPESCRIPT (a typescript/lib/tsc.js), as
// scripts/build.mjs looks for it. The shared v3 UI the type check needs: ZLC_V3_UI_ROOT, else this worktree's
// apps/desktop/ui (the same commit as the workbench).
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
/** apps/atom-workbench of this worktree. */
export const APP=path.resolve(here,'../../../..');
const DIST=path.join(APP,'dist/js/atom-workbench/src');

function typescript(){
  const require=createRequire(path.join(APP,'package.json'));
  try{return require(require.resolve('typescript'));}catch{}
  const tsc=process.env.ZLC_TYPESCRIPT;
  if(!tsc||!fs.existsSync(tsc))throw new Error('TypeScript is missing: npm ci in apps/atom-workbench, or set ZLC_TYPESCRIPT to typescript/lib/tsc.js');
  return require(path.join(path.dirname(tsc),'typescript.js'));
}
function newest(folder){
  let t=0;for(const e of fs.readdirSync(folder,{withFileTypes:true})){const f=path.join(folder,e.name);t=Math.max(t,e.isDirectory()?newest(f):fs.statSync(f).mtimeMs);}return t;
}
/** Compiles the workbench's TypeScript into dist/js when a source is newer than the last emit (type checked: an error
 * stops it, as the app's build does). */
export async function build({force=false}={}){
  const stamp=path.join(DIST,'full-master-model.js');
  if(!force&&fs.existsSync(stamp)&&fs.statSync(stamp).mtimeMs>=newest(path.join(APP,'src')))return false;
  process.env.ZLC_V3_UI_ROOT??=path.resolve(APP,'../desktop/ui');
  const {resolveUiSource,typecheckAndEmit}=await import(pathToFileURL(path.join(APP,'scripts/ui-source.mjs')).href);
  typecheckAndEmit(typescript(),path.join(APP,'tsconfig.json'),resolveUiSource(),path.join(APP,'dist/js'));
  return true;
}

let modules;
/** The app's modules this tool uses (compiled first if needed). */
export async function app(){
  if(modules)return modules;
  await build();
  const load=name=>import(pathToFileURL(path.join(DIST,name)).href);
  const [types,examples,sceneFile,physics,intensity,spectroscopy,motion,reflection,protocol]=await Promise.all([
    load('types.js'),load('examples.js'),load('scene-file.js'),load('physics.js'),load('laser-intensity.js'),load('spectroscopy.js'),load('motion.js'),load('reflection.js'),
    import(pathToFileURL(path.join(APP,'scripts/native-protocol.mjs')).href)]);
  modules={types,examples,sceneFile,physics,intensity,spectroscopy,motion,reflection,protocol};
  absolute=physics.absoluteEnergyMHz;
  return modules;
}

/** A scene as the app keeps it: frequencies and intensities resolved, reflections derived from their sources, then read
 * back through the app's importer (scene-file.ts validateScene: strict, as a saved file is opened). */
export async function finish(scene){
  const {physics,intensity,sceneFile}=await app();
  physics.resolveSceneFrequencies(scene);scene.lasers.forEach(intensity.normalizeLaserIntensity);
  return sceneFile.validateScene(JSON.parse(JSON.stringify(scene)));
}

/** The native request of a scene's motion solve, as the app sends it (scripts/native-protocol.mjs prepareNativeSolve:
 * a trajectory alone, the model with its motion block), with its id. `device` is the app's default. */
export async function motionRequest(id,scene,{device='gpu'}={}){
  const {motion,protocol}=await app();
  const asked=motion.motionRequest(scene.dynamics);if(!asked)throw new Error(`${id}: motion is off in the scene`);
  const {payload}=protocol.prepareNativeSolve({scene,options:{...scene.dynamics,includeMean:false,meanOnly:false,motion:asked,device}});
  return {id,...payload};
}
/** The same scene's static request (no motion block): the population core's trajectory of the atom at rest at the focus
 * (contract §1.2: the app's request with motion off). */
export async function staticRequest(id,scene){
  const {protocol}=await app();
  const {motion,...dynamics}=scene.dynamics;
  const {payload}=protocol.prepareNativeSolve({scene,options:{...dynamics,includeMean:false,meanOnly:false}});
  return {id,...payload};
}

/** The levels chart's static spectrum of `scene` with only the lasers `shown` (ids) showing their light shift: the app's
 * full-order Floquet quasienergies (spectroscopy.ts spectroscopyAnalysis), per level id its light shift (MHz, the
 * Floquet energy less the bare level at the scene's field) and the numerical bound of it (MHz). Independent of the
 * motion solver. */
export async function levelShifts(scene,shown){
  const {spectroscopy,physics,intensity}=await app();
  const s=JSON.parse(JSON.stringify(scene));physics.resolveSceneFrequencies(s);s.lasers.forEach(intensity.normalizeLaserIntensity);for(const l of s.lasers)l.showLightShift=shown.includes(l.id);
  const a=spectroscopy.spectroscopyAnalysis(s,undefined,false);
  if(a.spectroscopy?.status!==(shown.length?'dispersive':'bare'))throw new Error(`levels chart: ${a.spectroscopy?.status} (${a.spectroscopy?.warnings?.[0]??''})`);
  // `mixing`: the largest |Ω/2Δ| of the shown light (the static shift's validity, STATIC_MIXING_LIMIT): a second-order
  // shift differs from this all-order one by a relative (Ω/2Δ)².
  const out={mixing:a.spectroscopy.maximumMixingRatio??0};
  for(const l of [...a.ground,...a.D1,...a.D2])out[l.id]={shiftMHz:l.starkShiftMHz,energyMHz:l.energyMHz,absoluteMHz:spectroscopyAbsolute(l),boundMHz:(l.shiftBound?.floquetMHz??0)+(l.shiftBound?.outsideMHz??0),mixed:!!l.mixed};
  return out;
}
let absolute;
const spectroscopyAbsolute=level=>absolute(level);
/** The transition frequency e − g (MHz) of `scene` with the lasers `shown` shifting the levels (levels chart, full-order
 * Floquet), and a laser's frequency (MHz): the detuning of a laser from a transition the app's own way. */
export async function transitionMHz(scene,groundId,excitedId,shown){const s=await levelShifts(scene,shown);return s[excitedId].absoluteMHz-s[groundId].absoluteMHz;}
export async function laserMHz(scene,id){const {physics}=await app();return physics.laserFrequencyMHz(scene.lasers.find(l=>l.id===id));}
/** The zero-field transition F → F′ of a line (MHz). */
export async function zeroFieldMHz(line,Fg,Fe){const {physics}=await app();return physics.zeroFieldTransitionFrequencyMHz(line,Fg,Fe);}
/** The app's polarization of each enabled laser on the quantization axis (physics.ts polarization): weights of σ−, π,
 * σ+, and the lab vectors k, u, v. */
export async function polarizations(scene){const {physics}=await app();const a=physics.analyze(scene);return a.polarizations;}
