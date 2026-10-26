// The frequency readout control (ui-design.md §1.4): a v3 Readout, a v3 unit
// picker and a "相对" base. Unit and base only change how the frequency is read;
// they are this readout position's display preference, never physics.
import {Readout} from '../../desktop/ui/src/fluent/readout.js';
import {TreeCombo,type TreeGroup} from '../../desktop/ui/src/fluent/tree_combo.js';
import {element} from '../../desktop/ui/src/fluent/dom.js';
import type {Scene} from './types.js';
import {FREQUENCY_UNITS,WAVELENGTH_UNITS,baseName,baseValue,difference,inUnit,isWavelengthUnit,parseBase,readingHeading,readoutText,
  type FrequencyBase,type FrequencyUnit,type FrequencyValue} from './frequency-value.js';
import {naturalDetuningUnit} from './laser-frequency.js';

export type ReadoutPreference={unit?:FrequencyUnit;base?:string};
/** Where readout preferences are kept (this machine until scenes carry display preferences). */
export class ReadoutPreferences{
  private prefs:Record<string,ReadoutPreference>={};
  constructor(private storageKey:string){try{const saved=JSON.parse(localStorage.getItem(storageKey)??'{}');if(saved&&typeof saved==='object')this.prefs=saved;}catch{}}
  get(key:string):ReadoutPreference{return {...this.prefs[key]};}
  set(key:string,pref:ReadoutPreference):void{this.prefs[key]=pref;try{localStorage.setItem(this.storageKey,JSON.stringify(this.prefs));}catch{}}
}

export function unitTree(absolute:boolean):TreeGroup[]{
  const groups:TreeGroup[]=[{heading:'频率',leaves:FREQUENCY_UNITS.map(u=>({key:u,label:u,text:u}))}];
  if(absolute)groups.push({heading:'波长',leaves:WAVELENGTH_UNITS.map(u=>({key:u,label:u,text:u}))});
  return groups;
}

const leafKeys=(groups:readonly TreeGroup[]):string[]=>groups.flatMap(g=>[...g.leaves.map(l=>l.key),...leafKeys(g.children??[])]);

export type ReadoutOptions={
  /** The bases offered, keyed as frequency-value.ts baseKey. */
  bases:()=>TreeGroup[];
  /** The base when the operator has chosen none ("自动"); undefined: absolute. */
  automatic?:()=>string|undefined;
  /** The fixed default base of a readout without "自动". */
  defaultBase?:string;
  changed:()=>void;
};

/** "ν₁₂  +6.834 682 610 9 GHz  [GHz ▾]  相对 [绝对 ▾]" */
export class FrequencyReadout{
  readonly readout=new Readout('');
  readonly unit:TreeCombo;
  readonly base:TreeCombo;
  readonly baseLabel=element('span','frequency-base-label','相对');
  private last:{value:FrequencyValue|undefined;scene:Pick<Scene,'lasers'>}|undefined;
  private automaticWords:string|null=null;
  constructor(readonly key:string,private prefs:ReadoutPreferences,private options:ReadoutOptions){
    this.readout.root.classList.add('frequency-reading');
    this.unit=new TreeCombo(unit=>{this.prefs.set(key,{...this.prefs.get(key),unit:unit as FrequencyUnit});this.refresh();this.options.changed();});
    this.unit.root.classList.add('unit-picker');this.unit.root.setAttribute('aria-label','读数单位');
    this.base=new TreeCombo(base=>{const pref=this.prefs.get(key);this.prefs.set(key,{...pref,base:base||undefined});this.refresh();this.options.changed();});
    this.base.root.classList.add('frequency-base');this.base.root.setAttribute('aria-label','读数基准');
  }
  /** The operator's base and unit, when they chose one; a base this readout no longer offers
   * (a deleted laser, the laser being read) is no choice. */
  chosenBase():string|undefined{const base=this.prefs.get(this.key).base;return base!==undefined&&leafKeys(this.options.bases()).includes(base)?base:undefined;}
  chosenUnit():FrequencyUnit|undefined{return this.prefs.get(this.key).unit;}
  /** The base in effect: the operator's, else "自动"'s, else the readout's default, else absolute. */
  baseKey():string{return this.chosenBase()??this.options.automatic?.()??this.options.defaultBase??'absolute';}
  resolvedBase(scene:Pick<Scene,'lasers'>):FrequencyBase{const base=parseBase(this.baseKey());return base&&baseValue(base,scene)?base:{kind:'absolute'};}
  /** The unit in effect: the operator's when the base allows it, else the reading's natural one. */
  unitFor(reading:number,absolute:boolean):FrequencyUnit{
    const chosen=this.prefs.get(this.key).unit;
    if(chosen&&(!isWavelengthUnit(chosen)||(absolute&&reading>0)))return chosen;
    return Number.isFinite(reading)?naturalDetuningUnit(reading):'MHz';
  }
  /** ν − ν(base) in MHz, or NaN. */
  relative(value:FrequencyValue|undefined,scene:Pick<Scene,'lasers'>):number{
    const base=baseValue(this.resolvedBase(scene),scene);return value&&base?difference(value,base):NaN;
  }
  show(value:FrequencyValue|undefined,scene:Pick<Scene,'lasers'>):void{this.last={value,scene};this.refresh();}
  /** The column heading a table reading this base and unit says: "ν − ν(探测光) / MHz". */
  heading(scene:Pick<Scene,'lasers'>,unit:FrequencyUnit):string{return readingHeading(this.resolvedBase(scene),scene,unit);}
  private refresh():void{
    if(!this.last)return;
    const {value,scene}=this.last,base=this.resolvedBase(scene),absolute=base.kind==='absolute';
    const reading=this.relative(value,scene),unit=this.unitFor(reading,absolute);
    const shown=inUnit(reading,unit),words=value?`${readoutText(shown,!absolute)} ${unit}`:'—';
    this.readout.text=words;
    // The whole number, as computed, for the one who needs every digit.
    this.readout.root.title=value&&Number.isFinite(shown)?`${shown} ${unit}${absolute?'':`（相对 ${baseName(base,scene)}）`}`:'';
    this.unit.setTree(unitTree(absolute&&reading>0),unit);
    const automatic=this.options.automatic?`自动（${baseName(parseBase(this.options.automatic()??'absolute')??{kind:'absolute'},scene)}）`:null;
    const bases=this.options.bases();
    this.base.setTree(bases,this.chosenBase()??(automatic?'':this.baseKey()),automatic);
    // A WebView2 base-select keeps showing the old words of a picked option that is renamed:
    // "自动（…）" is picked again so the closed box says what it now resolves to.
    if(automatic!==this.automaticWords&&this.base.value===''){const leaf=bases[0]?.leaves[0]?.key;if(leaf){this.base.value=leaf;this.base.value='';}}
    this.automaticWords=automatic;
  }
}
