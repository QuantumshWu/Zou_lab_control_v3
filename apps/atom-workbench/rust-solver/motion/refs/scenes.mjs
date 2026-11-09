// node scenes.mjs -- writes every reference request, built by the app's own model builder (lib/app.mjs):
// - scenes/<id>.scene.json and scenes/<id>.json: the R1–R9 scenes (lib/defs.mjs), R4's TW1/TW2 (from the quantum
//   references' own specification) and R8 (R5051 held at the focus): the app scene, and the native request the app
//   sends for it (a motion trajectory, device "cpu");
// - out/R0/<id>.json and out/R0/<id>.static.json: the static limit (contract §6 R0) of the R0 set -- the scene's motion
//   request with the mass ×10⁶ and 1e-3 µK, and the same scene without motion for the population core (r0.mjs);
// Every request is checked against what request::Request needs (lib/contract.mjs). The app commit is recorded in
// scenes/manifest.json; regenerate after the app's model builder changes.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {APP,finish,motionRequest,staticRequest} from './lib/app.mjs';
import {checkRequest} from './lib/contract.mjs';
import {SCENES} from './lib/defs.mjs';
import {base,beam,tweezer} from './lib/scene.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const out=(...p)=>{const f=path.join(here,...p);fs.mkdirSync(path.dirname(f),{recursive:true});return f;};
const write=(file,value)=>fs.writeFileSync(file,JSON.stringify(value));
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));

/** The static limit's motion block (contract §6 R0): a mass ×10⁶ and 1e-3 µK hold the atom at the focus -- its
 * thermal spread √(k_BT/κ) (κ the trap's stiffness, independent of m) is ~10⁻³ of the waist, its Doppler shift and
 * recoil 10⁻³ and 10⁻⁶ of the usual: m → ∞ decouples the motion, and the jump ensemble's mean is the Lindblad equation
 * at the focus. */
export const HELD={mass:1e6,temperatureUK:1e-3};
function held(line,mass=HELD.mass){const m=structuredClone(line);m.model.motion.massKg*=mass;m.model.motion.temperatureUK=HELD.temperatureUK;return m;}
/** R8's mass factor: the momentum diffusion D adds (ħ²/Mm)·2Dt to k_BT while the held atom's zero-point floor is
 * ħω/2 ∝ 1/√M, so their ratio falls as 1/√M -- at ×10⁶ the diffusion is ~10⁻³ of the floor (invisible). ×100 keeps the
 * atom within its zero-point spread √(ħ/2Mmω) ≈ λ/150 of the trap centre (no motional decorrelation: held) while the
 * diffusion of R5051 over 100 µs is ~15 % of the floor at the QRT's γ = 1/µs cutoff. */
export const R8_MASS=100;

/** R4: the 1D quantum references' scenes (TW1, TW2; contract §3.5, the 1D GMW engine's LONGREF_TW*.json) rebuilt from
 * their own specification: the app's 1D PGC example (a lin⊥lin D2 cooling pair along ±z, E ∥ x on +z and ∥ y on −z,
 * and a repump pair alike) with the 852 nm tweezer (3.5 mW, waist 1 µm) along +x, π on the quantization axis z (B = 0),
 * |2,+2⟩ at 30 µK. Three ms for TW1 (heats), one for TW2 (cools: its T leaves the validated regime after ~500 µs). */
export const TW_REFERENCES='C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri--claude-spares-spare-cmNOsa/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad/pgc1d/tweezer/refs';
async function twScene(name,durationUs){
  const spec=read(`${TW_REFERENCES}/LONGREF_${name}.json`).spec;
  const s=await base({name:`R4 西西弗斯 ${name}（1D 量子参照的场景）`,fieldG:spec.fieldG,fieldAxis:spec.fieldAxis,initialStateId:spec.initialStateId,temperatureUK:30,durationUs,samples:durationUs/10+1});
  await tweezer(s,{nm:852,powerMW:3.5,waistUm:1,along:'+x',polarizationAngleDeg:0});
  for(const b of spec.beams){
    const along=b.axis,x=b.electric[0]===1;
    // A +z beam's polarization angle 0 is x; a −z beam's 90° is y (lib/scene.mjs; checked against the app's u).
    await beam(s,{id:b.id,name:b.id,line:b.line,Fg:b.Fg,Fe:b.Fe,offsetMHz:b.offsetMHz,intensityMWcm2:b.intensityMWcm2,along,polarizationAngleDeg:along==='+z'?(x?0:90):(x?180:90)});
  }
  return s;
}

/** The R0 set: physically different scenes, each with a duration that reaches its static limit's steady state at a
 * modest cost (the solver's work at m ×10⁶ is its jumps). */
export const R0_SET={DEFAULT:[100,201],THREE:[1000,201],PGC3D:[200,201],GMD1C:[200,201],F10_28MHZ:[10,101],R5059:[500,101],R5068:[1000,101],RETRO:[100,101],R5_PUMP:[200,201],R6_RAMAN_CO:[20,401]};

async function main(){
  const commit=execFileSync('git',['-C',APP,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  const manifest={app:commit,device:'cpu',scenes:{},r0:{}};
  const sources={};
  const emit=async(id,scene,title,ref)=>{
    const s=await finish(scene),line=await motionRequest(id,s,{device:'cpu'});
    const c=checkRequest(line);write(out('scenes',`${id}.scene.json`),s);write(out('scenes',`${id}.json`),line);
    manifest.scenes[id]={ref,title,...c};sources[id]=s;console.log(id,JSON.stringify(c));
  };
  for(const [id,d] of Object.entries(SCENES))await emit(id,await d.build(),d.title,d.ref);
  await emit('R4_TW1',await twScene('TW1',3000),'R4 西西弗斯 TW1：app 的 1D PGC 例子加光镊（加热）','R4');
  await emit('R4_TW2',await twScene('TW2',1000),'R4 西西弗斯 TW2：典型阱内 PGC（冷却）','R4');
  // R8: R5051 (a retro-reflected 880 nm tweezer, a D2 probe near F′=3 and a weak F=1 → F′=0 beam) held at its trap's centre.
  {const src=read(path.join(here,'regress','R5051.scene.json'));src.dynamics.samples=101;
   const s=await finish(src),line=held(await motionRequest('R8_R5051',s,{device:'cpu'}),R8_MASS);const c=checkRequest(line);
   write(out('scenes','R8_R5051.scene.json'),s);write(out('scenes','R8_R5051.json'),line);
   manifest.scenes.R8_R5051={ref:'R8',title:'R8 静止原子的动量扩散（R5051，质量 ×100）',...c,held:{mass:R8_MASS,temperatureUK:HELD.temperatureUK}};console.log('R8_R5051',JSON.stringify(c));}
  for(const [id,[durationUs,samples]] of Object.entries(R0_SET)){
    const src=sources[id]??read(path.join(here,'regress',`${id}.scene.json`));
    const s=await finish({...structuredClone(src),dynamics:{...src.dynamics,durationUs,samples}});
    const line=held(await motionRequest(`R0_${id}`,s,{device:'cpu'})),still=await staticRequest(`R0_${id}`,s);
    checkRequest(line);write(out('out','R0',`${id}.json`),line);write(out('out','R0',`${id}.static.json`),still);
    manifest.r0[id]={durationUs,samples,held:HELD};console.log('R0',id,durationUs,samples);
  }
  write(out('scenes','manifest.json'),manifest);
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
