// The v3 unit rule in Atom Workbench (ui-design.md §0.4): a number the operator wrote keeps its
// digits when its unit changes -- the quantity changes; a computed number is converted -- the
// quantity stays. The inspector writes a unit change as the box's digits in the new unit
// (app.ts bindQuantity), so each input is checked here as `write(typedDigits(read().value), unit)`.
import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {laserFrequencyMHz} from '../dist/js/atom-workbench/src/physics.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {typedDigits} from '../dist/js/atom-workbench/src/unit-input.js';
import {expressionBase,frequencyValue,setFrequencyBase,setFrequencyValue} from '../dist/js/atom-workbench/src/laser-frequency.js';
import {MHZ_PER_MK,inUnit} from '../dist/js/atom-workbench/src/frequency-value.js';
import {offsetReading,setReferenceOffset} from '../dist/js/atom-workbench/src/reference-energy.js';
import {initialSpec,initialSummary,setInitialSpec,setTemperature,temperatureReading} from '../dist/js/atom-workbench/src/initial-state.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {INTENSITY_MWCM2,WAIST_UM,intensityReading,laserIntensityMWcm2,laserWaistUm,laserWavelengthUm,powerReading,setLaserIntensityMode,setLaserNumericalAperture,
  waistReading,writeLaserIntensity,writeLaserPower,writeLaserWaist} from '../dist/js/atom-workbench/src/laser-intensity.js';

const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
const key=scene=>dynamicsCacheKey(scene,scene.dynamics,false);
/** The inspector's unit change on an input: the digits in the box, written in the new unit. */
const pick=(read,write,unit)=>write(typedDigits(read().value),unit);
const levels=scene=>{const a=analyze(scene);return [...a.ground,...a.D1,...a.D2];};

test('the digits a unit change keeps are the operator\'s, without the float dust of the stored unit',()=>{
  // 7.7 μK kept in K and read back in μK, 15.7 ns kept in μs and read back in ns.
  assert.equal(7.7*1e-6/1e-6,7.699999999999999,'the dust this removes is real');assert.equal(15.7*1e-3/1e-3,15.699999999999998);
  assert.equal(typedDigits(7.7*1e-6/1e-6),7.7);assert.equal(typedDigits(15.7*1e-3/1e-3),15.7);assert.equal(typedDigits(-15.5*1e-6/1e-6),-15.5);assert.equal(typedDigits(0),0);
  // A number with all its digits is its own digits.
  for(const value of [.12345678912345678,Math.PI,2.2619467105846511e-6,-1e-300])assert.equal(typedDigits(value),value);
});

test('power, waist and local intensity are inputs: a unit keeps the digits and changes the beam',()=>{
  const s=roundtrip(defaultScene()),l=s.lasers[1];assert.equal(l.intensityMode,'power');
  writeLaserPower(l,4,'mW');writeLaserWaist(l,1.2,'μm');const i0=laserIntensityMWcm2(l),k0=key(s);
  pick(()=>powerReading(l),(v,u)=>writeLaserPower(l,v,u),'W');
  assert.deepEqual(powerReading(l),{value:4,unit:'W'});assert.equal(l.powerMW,4000);assert.equal(laserIntensityMWcm2(l),2*4000/(Math.PI*1.2**2)*1e8);assert.notEqual(key(s),k0,'a new power is solved anew');
  pick(()=>waistReading(l),(v,u)=>writeLaserWaist(l,v,u),'mm');
  assert.deepEqual(waistReading(l),{value:1.2,unit:'mm'});assert.equal(l.waistUm,1200);
  pick(()=>powerReading(l),(v,u)=>writeLaserPower(l,v,u),'mW');pick(()=>waistReading(l),(v,u)=>writeLaserWaist(l,v,u),'μm');
  assert.equal(laserIntensityMWcm2(l),i0);assert.equal(key(s),k0,'the same digits in the first units: the first beam');
  // Intensity mode: the local intensity is the input.
  setLaserIntensityMode(l,'intensity');writeLaserIntensity(l,250,'mW/cm²');
  pick(()=>intensityReading(l),(v,u)=>writeLaserIntensity(l,v,u),'W/cm²');assert.deepEqual(intensityReading(l),{value:250,unit:'W/cm²'});assert.equal(laserIntensityMWcm2(l),2.5e5);
  // The unit is saved with the number.
  const back=roundtrip(s).lasers[1];assert.deepEqual(intensityReading(back),{value:250,unit:'W/cm²'});assert.equal(back.powerUnit,'mW');assert.equal(back.waistUnit,'μm');
  assert.throws(()=>writeLaserPower(l,1,'kW'),/单位/);
});

test('a computed intensity or waist converts: the same quantity in another unit',()=>{
  const s=roundtrip(defaultScene()),l=s.lasers[1],intensity=laserIntensityMWcm2(l);
  for(const unit of ['kW/cm²','W/cm²','mW/cm²','μW/cm²']){l.intensityUnit=unit;const r=intensityReading(l);assert.equal(r.unit,unit);assert.ok(Math.abs(r.value*INTENSITY_MWCM2[unit]/intensity-1)<1e-15);assert.equal(laserIntensityMWcm2(l),intensity);}
  setLaserIntensityMode(l,'na');const waist=laserWaistUm(l);
  for(const unit of ['mm','μm','nm']){l.waistUnit=unit;const r=waistReading(l);assert.equal(r.unit,unit);assert.ok(Math.abs(r.value*WAIST_UM[unit]/waist-1)<1e-15);assert.equal(laserWaistUm(l),waist);}
});

test('a detuning or offset frequency is an input: the base stays, the digits stay, the frequency moves',()=>{
  const s=roundtrip(defaultScene()),probe=s.lasers[0],base=expressionBase(probe.frequency);
  setFrequencyValue(s,probe,10,'MHz');const before=laserFrequencyMHz(probe);
  pick(()=>frequencyValue(probe),(v,u)=>setFrequencyValue(s,probe,v,u),'GHz');
  assert.deepEqual(frequencyValue(probe),{value:10,unit:'GHz'});assert.deepEqual(expressionBase(probe.frequency),base);assert.ok(Math.abs(laserFrequencyMHz(probe)-before-9990)<1e-6);
  // Relative to another laser: the offset lock keeps its base.
  const repump={...structuredClone(probe),id:'repump',name:'回泵光'};s.lasers.push(repump);setFrequencyBase(s,repump,{kind:'laser',id:'probe'});setFrequencyValue(s,repump,6.568,'GHz');
  pick(()=>frequencyValue(repump),(v,u)=>setFrequencyValue(s,repump,v,u),'MHz');
  assert.deepEqual(repump.frequency.base,{kind:'laser',id:'probe'});assert.ok(Math.abs(laserFrequencyMHz(repump)-laserFrequencyMHz(probe)-6.568)<1e-6);
  assert.equal(roundtrip(s).lasers[2].frequencyUnit,'MHz');
});

test('an absolute frequency read in a wavelength unit becomes that wavelength, and back',()=>{
  const s=roundtrip(defaultScene()),probe=s.lasers[0];setFrequencyBase(s,probe,{kind:'absolute'});setFrequencyValue(s,probe,384.23,'THz');
  pick(()=>frequencyValue(probe),(v,u)=>setFrequencyValue(s,probe,v,u),'nm');
  assert.deepEqual(probe.frequency,{wavelengthNm:384.23});assert.equal(probe.frequencyUnit,'nm');
  pick(()=>frequencyValue(probe),(v,u)=>setFrequencyValue(s,probe,v,u),'μm');assert.deepEqual(probe.frequency,{wavelengthNm:384230});
  pick(()=>frequencyValue(probe),(v,u)=>setFrequencyValue(s,probe,v,u),'THz');
  assert.deepEqual(probe.frequency,{base:{kind:'absolute'},offsetMHz:384.23e6});assert.equal(laserFrequencyMHz(probe),384.23e6);
  // Only an absolute frequency is a wavelength.
  setFrequencyBase(s,probe,{kind:'transition',line:'D2',Fg:2,Fe:3});assert.throws(()=>setFrequencyValue(s,probe,780,'nm'),/绝对/);
});

test('a unit whose frequency the physics refuses changes nothing and says why',()=>{
  const s=roundtrip(defaultScene()),probe=s.lasers[0];setFrequencyValue(s,probe,-500,'MHz');const before=JSON.stringify(s.lasers);
  // Δ = −500 THz from the 384 THz D2 line is no optical frequency: the field shows it invalid, the scene keeps −500 MHz.
  assert.throws(()=>pick(()=>frequencyValue(probe),(v,u)=>setFrequencyValue(s,probe,v,u),'THz'),/光频必须是有限正数/);
  assert.equal(JSON.stringify(s.lasers),before);assert.deepEqual(frequencyValue(probe),{value:-500,unit:'MHz'});
  setFrequencyValue(s,probe,6000,'MHz');assert.throws(()=>pick(()=>frequencyValue(probe),(v,u)=>setFrequencyValue(s,probe,v,u),'THz'),/5000 THz/);
});

test('a reference level\'s offset δ is an input with its unit saved; a refused unit changes nothing',()=>{
  const s=roundtrip(defaultScene());s.references=[{id:'reference-1',name:'虚拟态',anchorId:'ground-2-0',anchorMode:'follow',offsetMHz:0}];const r=s.references[0];
  setReferenceOffset(r,12,'MHz');pick(()=>offsetReading(r),(v,u)=>setReferenceOffset(r,v,u),'kHz');
  assert.deepEqual(offsetReading(r),{value:12,unit:'kHz'});assert.equal(r.offsetMHz,.012);
  assert.deepEqual(offsetReading(roundtrip(s).references[0]),{value:12,unit:'kHz'});
  setReferenceOffset(r,5e6,'MHz');assert.throws(()=>pick(()=>offsetReading(r),(v,u)=>setReferenceOffset(r,v,u),'THz'),/偏移/);assert.equal(r.offsetMHz,5e6);assert.equal(r.offsetUnit,'MHz');
  // A scene without the unit reads the offset in its natural unit.
  delete r.offsetUnit;assert.deepEqual(offsetReading(r),{value:5,unit:'THz'});
});

test('a thermal temperature is an input with its unit saved; another initial state drops the unit',()=>{
  const s=roundtrip(defaultScene()),all=levels(s);setTemperature(s,1,'mK',all);
  assert.deepEqual(temperatureReading(initialSpec(s).temperatureK,s.dynamics.temperatureUnit),{value:1,unit:'mK'});
  setTemperature(s,typedDigits(temperatureReading(initialSpec(s).temperatureK,s.dynamics.temperatureUnit).value),'K',all);
  assert.deepEqual(initialSpec(s),{kind:'thermal',temperatureK:1});assert.equal(s.dynamics.temperatureUnit,'K');assert.equal(initialSummary(initialSpec(s),'K'),'热分布 · T = 1 K');
  setTemperature(s,500,'μK',all);const back=roundtrip(s),read=temperatureReading(initialSpec(back).temperatureK,back.dynamics.temperatureUnit);assert.equal(back.dynamics.temperatureUnit,'μK');assert.equal(read.unit,'μK');assert.equal(typedDigits(read.value),500);
  assert.equal(key(back),key(s),'the same initial state and request');
  for(const value of [0,-1,NaN])assert.throws(()=>setTemperature(s,value,'K',all),/有限正数/);
  setInitialSpec(s,{kind:'state',id:'ground-2-0'},all);assert.equal('temperatureUnit' in s.dynamics,false);
});

test('ν₁₂ reads as a temperature ΔE/k_B in mK, converted with the exact SI k_B/h',()=>{
  assert.equal(MHZ_PER_MK,1.380649e-23/6.62607015e-34/1e9);assert.ok(Math.abs(MHZ_PER_MK-20.836619)<1e-6);
  assert.equal(inUnit(20.836619123632,'mK').toFixed(9),'1.000000000');
  const clock=6834.682610904;assert.ok(Math.abs(inUnit(clock,'mK')*MHZ_PER_MK-clock)<1e-9);assert.equal(inUnit(clock,'GHz'),clock/1e3);
  // The unit is the readout's display preference, saved with the scene.
  const s=roundtrip(defaultScene());s.display={readouts:{'pair-frequency':{unit:'mK'}}};assert.deepEqual(roundtrip(s).display,s.display);
});

test('NA mode: the waist is λ/(π·NA) at the laser\'s own wavelength and I = 2P/(π·w₀²)',()=>{
  const s=roundtrip(defaultScene()),l=s.lasers[1];setLaserIntensityMode(l,'na');setLaserNumericalAperture(l,.5);writeLaserPower(l,10,'mW');
  const lambda=299792458/laserFrequencyMHz(l);assert.ok(Math.abs(lambda-.852)<1e-12,'852 nm');assert.equal(laserWavelengthUm(l),lambda);
  const waist=lambda/(Math.PI*.5);assert.equal(laserWaistUm(l),waist);assert.equal(laserIntensityMWcm2(l),2*10/(Math.PI*waist**2)*1e8);
  // The power keeps its unit rule; the NA is dimensionless.
  pick(()=>powerReading(l),(v,u)=>writeLaserPower(l,v,u),'W');assert.equal(l.powerMW,1e4);assert.equal(l.intensityMode,'na');assert.equal(laserIntensityMWcm2(l),2*1e4/(Math.PI*waist**2)*1e8);
});

test('NA mode: a change of frequency moves the waist and the intensity',()=>{
  const s=roundtrip(defaultScene()),l=s.lasers[1];setLaserIntensityMode(l,'na');const w0=laserWaistUm(l),i0=laserIntensityMWcm2(l),k0=key(s);
  setFrequencyValue(s,l,1064,'nm');
  assert.ok(Math.abs(laserWaistUm(l)/w0-1064/852)<1e-12);assert.ok(Math.abs(laserIntensityMWcm2(l)/i0-(852/1064)**2)<1e-12);assert.notEqual(key(s),k0);
  setFrequencyValue(s,l,852,'nm');assert.equal(laserIntensityMWcm2(l),i0,'the same wavelength gives back the same intensity');
});

test('switching to and from NA mode keeps the local intensity exactly',()=>{
  let s=roundtrip(defaultScene()),l=s.lasers[1];writeLaserIntensity(l,.12345678912345678,'mW/cm²');const intensity=laserIntensityMWcm2(l);
  for(const [from,to] of [['intensity','na'],['na','intensity'],['intensity','power'],['power','na'],['na','power'],['power','intensity']]){
    assert.equal(l.intensityMode,from);setLaserIntensityMode(l,to);assert.equal(laserIntensityMWcm2(l),intensity,`${from} → ${to}`);s=roundtrip(s);l=s.lasers[1];assert.equal(laserIntensityMWcm2(l),intensity,`${from} → ${to}, saved`);
  }
  // NA mode starts from the laser's NA, or 0.5; power mode from its waist.
  setLaserIntensityMode(l,'na');assert.equal(l.numericalAperture,.5);setLaserNumericalAperture(l,.7);setLaserIntensityMode(l,'power');setLaserIntensityMode(l,'na');assert.equal(l.numericalAperture,.7);
});

test('an NA-mode scene survives the scene file, also when its laser follows another one',()=>{
  const s=roundtrip(defaultScene()),[probe,l]=s.lasers;setLaserIntensityMode(l,'na');setLaserNumericalAperture(l,.42);writeLaserPower(l,3,'μW');l.waistUnit='nm';l.intensityUnit='W/cm²';
  setFrequencyBase(s,l,{kind:'laser',id:'probe'});
  const back=roundtrip(s),b=back.lasers[1];
  for(const field of ['intensityMode','numericalAperture','powerMW','powerUnit','waistUnit','intensityUnit'])assert.equal(b[field],l[field],field);
  assert.equal(laserIntensityMWcm2(b),laserIntensityMWcm2(l));assert.equal(laserWaistUm(b),laserWaistUm(l));assert.equal(key(back),key(s));
  // It follows the probe: the probe's frequency moves its waist.
  const w=laserWaistUm(b);setFrequencyValue(back,back.lasers[0],-1,'THz');assert.notEqual(laserWaistUm(b),w);assert.ok(probe);
});

test('an NA outside 0 < NA ≤ 1 is refused with its reason, in the inspector and in a file',()=>{
  const s=roundtrip(defaultScene()),l=s.lasers[1];setLaserIntensityMode(l,'na');const before=JSON.stringify(l);
  for(const value of [0,-.1,1.2,NaN,Infinity])assert.throws(()=>setLaserNumericalAperture(l,value),/0 < NA ≤ 1/);
  assert.equal(JSON.stringify(l),before,'a refused NA changes nothing');setLaserNumericalAperture(l,1);assert.equal(l.numericalAperture,1);
  const file=JSON.parse(JSON.stringify(s));file.lasers[1].numericalAperture=1.5;assert.throws(()=>validateScene(file),/0 < NA ≤ 1/);
  delete file.lasers[1].numericalAperture;assert.throws(()=>validateScene(file),/数值孔径/);
  file.lasers[1].intensityMode='objective';assert.throws(()=>validateScene(file),/强度模式/);
});

test('scenes saved before NA mode load with the same intensities bit for bit',()=>{
  // A power-mode beam saved with its exact intensity anchor (1 ULP off the formula) keeps it.
  const file=JSON.parse(JSON.stringify(roundtrip(defaultScene()))),t=file.lasers[1];delete t.powerUnit;delete t.waistUnit;delete t.intensityUnit;
  const exact=2*t.powerMW/(Math.PI*t.waistUm**2)*1e8,saved=exact*(1+Number.EPSILON);t.intensityAnchor={intensityMWcm2:saved,viewKey:JSON.stringify(['power',t.powerMW,t.waistUm])};t.intensityMWcm2=saved;
  const loaded=validateScene(file);assert.equal(laserIntensityMWcm2(loaded.lasers[1]),saved);assert.equal(loaded.lasers[1].numericalAperture,undefined);
  assert.equal(loaded.dynamics.temperatureUnit,undefined);
});
