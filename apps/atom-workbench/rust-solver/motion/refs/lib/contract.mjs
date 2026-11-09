// What a native motion request must carry for the new crate's `request::Request` (motion/src/request.rs; contract §1.3,
// §1.4): every field the native adapter maps into it, with the shapes and the invariants `Request::validate` relies on.
// Until the adapter's JSON parsing exists this is the check that a scene file is complete; the Rust test
// tests/refs_scenes.rs then parses every scene with the crate itself.
//
// Request field          <- native JSON
// states                 <- model.states (32 labels manifold-F-m)
// hamiltonian            <- model.motion.hamiltonian (32 x 32, diagonal)
// frame_frequencies_mhz  <- model.motion.frameFrequenciesMHz (32)
// frequencies_mhz        <- model.frequencyCoordinates.frequenciesMHz
// beams[]                <- model.motion.beams[]: name, direction, waveNumberPerUm, waistUm?, frequencyIndex, coupling,
//                           counterCoupling, electricField (required)
// outside_forms[]        <- model.motion.outsideForms[] (one per distinct frequencyIndex, 9 matrices each)
// trap_beam              <- the index of model.motion.trapBeamId among the beams (a Gaussian beam: waistUm)
// emission               <- model.motion.emission (3 matrices, ground <- excited)
// mass_kg, temperature_uk<- model.motion.massKg, temperatureUK
// initial_density        <- model.initialReal / initialImag (32 x 32, diagonal, trace 1)
// times_us               <- options.durationUs, options.samples (0 to the duration, evenly spaced)
const N=32,GROUND=8;
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function matrix(m,what){
  if(!m||!Array.isArray(m.real)||!Array.isArray(m.imag)||m.real.length!==N*N||m.imag.length!==N*N||![...m.real,...m.imag].every(finite))throw new Error(`${what}: not a 32 x 32 matrix of finite numbers`);
  return m;
}
const entries=(m,keep)=>{const out=[];for(let i=0;i<N;i++)for(let j=0;j<N;j++){const k=i*N+j;if((m.real[k]!==0||m.imag[k]!==0)&&!keep(i,j))out.push([i,j]);}return out;};
const label=/^(ground|D1|D2)-(\d)-(-?\d)$/;

/** Checks a native request line; returns a short summary or throws why it lacks what the Request needs. */
export function checkRequest(line){
  const what=line.id??'request';
  if(line.modelSchema!=='rb87-full/2')throw new Error(`${what}: modelSchema`);
  const o=line.options??{},model=line.model??{},m=model.motion;
  if(!finite(o.durationUs)||!(o.durationUs>0)||o.durationUs>1e7)throw new Error(`${what}: options.durationUs`);
  if(!Number.isInteger(o.samples)||o.samples<2||o.samples>2000)throw new Error(`${what}: options.samples`);
  if(o.includeMean!==false||o.meanOnly!==false)throw new Error(`${what}: a motion request is a trajectory alone`);
  if(o.device!==undefined&&o.device!=='gpu'&&o.device!=='cpu')throw new Error(`${what}: options.device`);
  if(model.dimension!==N||!Array.isArray(model.states)||model.states.length!==N||!model.states.every(s=>label.test(s)))throw new Error(`${what}: model.states`);
  const ground=model.states.map(s=>s.startsWith('ground-'));
  if(ground.filter(Boolean).length!==GROUND)throw new Error(`${what}: 8 ground states`);
  const rho=matrix({real:model.initialReal,imag:model.initialImag},`${what}: initial density`);
  if(entries(rho,(i,j)=>i===j).length)throw new Error(`${what}: initial density not diagonal`);
  const p=model.states.map((_,i)=>rho.real[i*N+i]);
  if(p.some(x=>x<0)||Math.abs(p.reduce((a,b)=>a+b,0)-1)>1e-12||!p.some((x,i)=>ground[i]&&x>0))throw new Error(`${what}: initial populations`);
  const axes=model.frequencyCoordinates?.frequenciesMHz;
  if(!Array.isArray(axes)||!axes.every(finite))throw new Error(`${what}: frequencyCoordinates.frequenciesMHz`);
  if(!m)throw new Error(`${what}: no motion block`);
  if(m.axis!=='3d')throw new Error(`${what}: motion.axis`);
  if(!(finite(m.temperatureUK)&&m.temperatureUK>0)||!(finite(m.massKg)&&m.massKg>0))throw new Error(`${what}: motion temperature or mass`);
  if(!Array.isArray(m.fieldVectorG)||m.fieldVectorG.length!==3||!m.fieldVectorG.every(finite))throw new Error(`${what}: motion.fieldVectorG`);
  const h=matrix(m.hamiltonian,`${what}: motion.hamiltonian`);
  if(entries(h,(i,j)=>i===j).length)throw new Error(`${what}: motion.hamiltonian not diagonal`);
  if(!Array.isArray(m.frameFrequenciesMHz)||m.frameFrequenciesMHz.length!==N||!m.frameFrequenciesMHz.every(finite))throw new Error(`${what}: frameFrequenciesMHz`);
  if(!Array.isArray(m.beams)||!m.beams.length)throw new Error(`${what}: no beams`);
  const keys=new Set();
  m.beams.forEach((b,k)=>{
    const at=`${what}: beam ${b.id??k}`;
    if(typeof b.id!=='string'||typeof b.name!=='string')throw new Error(`${at}: id/name`);
    if(!Array.isArray(b.direction)||b.direction.length!==3||!b.direction.every(finite)||Math.abs(Math.hypot(...b.direction)-1)>1e-12)throw new Error(`${at}: direction`);
    if(!(finite(b.waveNumberPerUm)&&b.waveNumberPerUm>0))throw new Error(`${at}: waveNumberPerUm`);
    if(b.waistUm!==undefined&&!(finite(b.waistUm)&&b.waistUm>0))throw new Error(`${at}: waistUm`);
    if(!(finite(b.intensityMWcm2)&&b.intensityMWcm2>0))throw new Error(`${at}: intensityMWcm2`);
    if(!Array.isArray(b.frequencyIndex)||b.frequencyIndex.length!==axes.length||!b.frequencyIndex.every(Number.isInteger))throw new Error(`${at}: frequencyIndex`);
    for(const key of ['coupling','counterCoupling']){const c=matrix(b[key],`${at}: ${key}`);if(entries(c,(i,j)=>!ground[i]&&ground[j]).length)throw new Error(`${at}: ${key} not excited <- ground`);}
    matrix(b.outsideShift,`${at}: outsideShift`);
    if(!Array.isArray(b.electricField)||b.electricField.length!==3||!b.electricField.every(z=>finite(z?.re)&&finite(z?.im))||b.electricField.every(z=>z.re===0&&z.im===0))throw new Error(`${at}: electricField`);
    keys.add(b.frequencyIndex.join());
  });
  if(!Array.isArray(m.outsideForms))throw new Error(`${what}: outsideForms`);
  const forms=new Set(m.outsideForms.map(f=>f.frequencyIndex?.join()));
  if(forms.size!==m.outsideForms.length||forms.size!==keys.size||[...keys].some(k=>!forms.has(k)))throw new Error(`${what}: one outside form per distinct beam frequency`);
  m.outsideForms.forEach((f,k)=>{if(!Array.isArray(f.matrices)||f.matrices.length!==9)throw new Error(`${what}: outside form ${k}: 9 matrices`);f.matrices.forEach((x,i)=>matrix(x,`${what}: outside form ${k} matrix ${i}`));});
  const trap=m.beams.findIndex(b=>b.id===m.trapBeamId);
  if(trap<0)throw new Error(`${what}: trapBeamId names no beam`);
  if(m.beams[trap].waistUm===undefined)throw new Error(`${what}: the trap beam has no waist`);
  if(!Array.isArray(m.emission)||m.emission.length!==3)throw new Error(`${what}: emission`);
  m.emission.forEach((c,a)=>{matrix(c,`${what}: emission ${a}`);if(entries(c,(i,j)=>ground[i]&&!ground[j]).length)throw new Error(`${what}: emission ${a} not ground <- excited`);});
  return {beams:m.beams.length,frequencies:keys.size,trapBeam:trap,axes:axes.length,durationUs:o.durationUs,samples:o.samples};
}
