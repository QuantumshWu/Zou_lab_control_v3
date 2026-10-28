// The frequency readout control (ui-design.md §1.4): a v3 Readout, a v3 unit
// picker and a "相对" base. A readout is a display: another unit or base converts
// the reading (the quantity stays); both are this readout position's display
// preference, never physics.
import {Readout} from '../../desktop/ui/src/fluent/readout.js';
import {TreeCombo,type TreeGroup} from '../../desktop/ui/src/fluent/tree_combo.js';
import {element} from '../../desktop/ui/src/fluent/dom.js';
import type {Scene} from './types.js';
import {FREQUENCY_UNITS,TEMPERATURE_EQUIVALENT_UNITS,WAVELENGTH_UNITS,baseName,baseValue,difference,inUnit,isTemperatureEquivalentUnit,isWavelengthUnit,parseBase,readoutText,
  type FrequencyBase,type FrequencyUnit,type FrequencyValue} from './frequency-value.js';
import {naturalDetuningUnit} from './laser-frequency.js';

export type ReadoutPreference={unit?:FrequencyUnit;base?:string};
/** Where readout preferences are kept: the scene's display preferences (`store`), saved with it. */
export class ReadoutPreferences{
  constructor(private store:()=>Record<string,ReadoutPreference>,private saved:()=>void=()=>{}){}
  get(key:string):ReadoutPreference{return {...this.store()[key]};}
  set(key:string,pref:ReadoutPreference):void{const prefs=this.store();if(pref.unit===undefined&&pref.base===undefined)delete prefs[key];else prefs[key]=pref;this.saved();}
}

/** The units a reading is offered in: frequencies; wavelengths for an absolute positive frequency;
 * the temperature ΔE/k_B for an energy difference whose owner asks for it. */
export function unitTree(absolute:boolean,temperature=false):TreeGroup[]{
  const groups:TreeGroup[]=[{heading:'频率',leaves:FREQUENCY_UNITS.map(u=>({key:u,label:u,text:u}))}];
  if(absolute)groups.push({heading:'波长',leaves:WAVELENGTH_UNITS.map(u=>({key:u,label:u,text:u}))});
  if(temperature)groups.push({heading:'温度',leaves:TEMPERATURE_EQUIVALENT_UNITS.map(u=>({key:u,label:u,text:u}))});
  return groups;
}

const leafKeys=(groups:readonly TreeGroup[]):string[]=>groups.flatMap(g=>[...g.leaves.map(l=>l.key),...leafKeys(g.children??[])]);

export type ReadoutOptions={
  /** The bases offered, keyed as frequency-value.ts baseKey; none: the reading is absolute, with no base to choose. */
  bases?:()=>TreeGroup[];
  /** What the reading is relative to until the operator chooses a base (the pair's zero-field ν₁₂): its name and frequency. */
  automatic?:()=>{name:string;value:FrequencyValue|undefined};
  /** The fixed default base of a readout without an `automatic` one. */
  defaultBase?:string;
  /** false: no unit picker; the reading carries its natural unit in its words. */
  unitPicker?:false;
  /** The reading is an energy difference that may also be read as the temperature ΔE/k_B (mK). */
  temperature?:true;
  changed:()=>void;
};

/** "Δν  +2.099  [MHz ▾]  相对 [零场 ν₁₂ ▾]"; without bases "ν₁₂  6.834 682 610 9  [GHz ▾]". */
export class FrequencyReadout{
  readonly readout=new Readout('');
  readonly unit:TreeCombo;
  readonly base:TreeCombo|undefined;
  readonly baseLabel=element('span','frequency-base-label','相对');
  private last:{value:FrequencyValue|undefined;scene:Pick<Scene,'lasers'|'referenceFrequencies'>}|undefined;
  private automaticWords:string|null=null;
  constructor(readonly key:string,private prefs:ReadoutPreferences,private options:ReadoutOptions){
    this.readout.root.classList.add('frequency-reading');
    this.unit=new TreeCombo(unit=>{this.prefs.set(key,{...this.prefs.get(key),unit:unit as FrequencyUnit});this.refresh();this.options.changed();});
    this.unit.root.classList.add('unit-picker');this.unit.root.setAttribute('aria-label','读数单位');
    this.base=options.bases?new TreeCombo(base=>{const pref=this.prefs.get(key);this.prefs.set(key,{...pref,base:base||undefined});this.refresh();this.options.changed();}):undefined;
    this.base?.root.classList.add('frequency-base');this.base?.root.setAttribute('aria-label','读数基准');
  }
  /** The operator's base and unit, when they chose one; a base this readout no longer offers
   * (a deleted laser, the laser being read) is no choice. */
  chosenBase():string|undefined{const base=this.prefs.get(this.key).base;return base!==undefined&&leafKeys(this.options.bases?.()??[]).includes(base)?base:undefined;}
  chosenUnit():FrequencyUnit|undefined{return this.prefs.get(this.key).unit;}
  /** The base the reading is relative to and its frequency: the operator's, else the readout's
   * default (a base whose frequency is gone reads absolutely); undefined while the automatic base stands. */
  fixedBase(scene:Pick<Scene,'lasers'|'referenceFrequencies'>):{base:FrequencyBase;value:FrequencyValue}|undefined{
    const key=this.chosenBase();if(key===undefined&&this.options.automatic)return undefined;
    const base=parseBase(key??this.options.defaultBase??'absolute'),value=base&&baseValue(base,scene);
    return base&&value?{base,value}:{base:{kind:'absolute'},value:baseValue({kind:'absolute'},scene)!};
  }
  /** The unit in effect: the operator's when the reading offers it, else the reading's natural one. */
  private unitFor(reading:number,absolute:boolean):FrequencyUnit{
    const chosen=this.options.unitPicker===false?undefined:this.prefs.get(this.key).unit;
    const offered=chosen&&(isWavelengthUnit(chosen)?absolute&&reading>0:isTemperatureEquivalentUnit(chosen)?this.options.temperature===true:true);
    if(chosen&&offered)return chosen;
    return Number.isFinite(reading)?naturalDetuningUnit(reading):'MHz';
  }
  show(value:FrequencyValue|undefined,scene:Pick<Scene,'lasers'|'referenceFrequencies'>):void{this.last={value,scene};this.refresh();}
  private refresh():void{
    if(!this.last)return;
    const {value,scene}=this.last,fixed=this.fixedBase(scene),automatic=this.options.automatic?.();
    const reference=fixed?fixed.value:automatic?.value,name=fixed?baseName(fixed.base,scene):automatic?.name,absolute=fixed?.base.kind==='absolute';
    const reading=value&&reference?difference(value,reference):NaN,unit=this.unitFor(reading,absolute);
    // The reading alone when its unit is the picker beside it.
    const shown=inUnit(reading,unit),words=value?`${readoutText(shown,!absolute)}${this.options.unitPicker===false?` ${unit}`:''}`:'—';
    this.readout.text=words;
    // The whole number, as computed, for the one who needs every digit.
    this.readout.root.title=value&&Number.isFinite(shown)?`${shown} ${unit}${isTemperatureEquivalentUnit(unit)?'（ΔE/k_B）':absolute?'':`（相对 ${name}）`}`:'';
    this.unit.setTree(unitTree(absolute&&reading>0,this.options.temperature),unit);
    if(!this.base)return;
    const automaticText=automatic?automatic.name:null;
    const bases=this.options.bases!();
    this.base.setTree(bases,this.chosenBase()??(automaticText?'':this.options.defaultBase??'absolute'),automaticText);
    // A WebView2 base-select keeps showing the old words of a picked option that is renamed:
    // the automatic choice is picked again so the closed box says what it now resolves to.
    if(automaticText!==this.automaticWords&&this.base.value===''){const leaf=bases[0]?.leaves[0]?.key;if(leaf){this.base.value=leaf;this.base.value='';}}
    this.automaticWords=automaticText;
  }
}
