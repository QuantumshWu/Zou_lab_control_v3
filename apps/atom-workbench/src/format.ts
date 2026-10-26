// One spelling for every number and every state the workbench shows.
//
// Numbers: a significant-digit rule (or fixed decimals where a column needs
// them), the true minus sign U+2212, and SI digit grouping -- a narrow no-break
// space every three digits of an integer part of five or more digits, none in
// a four-digit one (8083.4 ms, 7 123 689.760 MHz).  Very large and very small
// magnitudes are written as a power of ten (6.7 × 10⁻¹⁶).
import type {Level} from './types.js';

export const MINUS='−';
const GROUP=' ';
const SUPERSCRIPT:Record<string,string>={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻','+':''};
const SUBSCRIPT='₀₁₂₃₄₅₆₇₈₉';

export type NumberStyle={
  /** Significant digits (default 6); trailing zeros are dropped. */
  digits?:number;
  /** Fixed decimals instead of significant digits (a table column). */
  places?:number;
  /** Write a plus sign on positive numbers. */
  signed?:boolean;
};

function grouped(integer:string):string{
  if(integer.length<5)return integer;
  let out='';
  for(let i=0;i<integer.length;i++){if(i>0&&(integer.length-i)%3===0)out+=GROUP;out+=integer[i];}
  return out;
}

/** 10⁻¹⁶ */
export function power(exponent:number):string{return `10${String(exponent).replace(/[-+\d]/g,c=>SUPERSCRIPT[c]!)}`;}

export function formatNumber(value:number,style:NumberStyle={}):string{
  if(Number.isNaN(value))return '—';
  if(!Number.isFinite(value))return value>0?'∞':`${MINUS}∞`;
  const magnitude=Math.abs(value);
  let body:string,zero=magnitude===0;
  if(style.places!==undefined){
    const fixed=magnitude.toFixed(style.places),[integer,fraction]=fixed.split('.');
    zero=Number(fixed)===0;
    body=grouped(integer!)+(fraction?`.${fraction}`:'');
  }else{
    const digits=Math.max(1,Math.min(15,style.digits??6));
    if(magnitude!==0&&(magnitude>=1e9||magnitude<1e-4)){
      const [mantissa,exponent]=magnitude.toExponential(digits-1).split('e');
      const trimmed=mantissa!.includes('.')?mantissa!.replace(/\.?0+$/,''):mantissa!;
      body=`${trimmed} × ${power(Number(exponent))}`;
    }else{
      const plain=String(Number(magnitude.toPrecision(digits)));
      const [integer,fraction]=(plain.includes('e')?magnitude.toFixed(Math.max(0,digits-1)):plain).split('.');
      body=grouped(integer!)+(fraction?`.${fraction}`:'');
    }
  }
  const sign=zero?'':value<0?MINUS:style.signed?'+':'';
  return sign+body;
}

/** A number with its unit, a space apart ("741.6 ms"); a dimensionless one alone. */
export function withUnit(value:number,unit:string,style:NumberStyle={}):string{
  const text=formatNumber(value,style);
  return unit?`${text} ${unit}`:text;
}

/** A computation's wall time: ms below a second, else s, four digits. */
export function formatElapsed(ms:number):string{
  if(!Number.isFinite(ms))return '—';
  return ms<1000?withUnit(ms,'ms',{digits:4}):withUnit(ms/1000,'s',{digits:4});
}

/** Typed numbers keep the ASCII minus; shown ones the true minus. */
export function signedInteger(n:number):string{return n>0?`+${n}`:n<0?`${MINUS}${-n}`:'0';}

/** λ₁: an object's symbol and its 1-based place in its list. */
export function subscript(n:number):string{return String(n).replace(/\d/g,d=>SUBSCRIPT[Number(d)]!);}

export const TERMS={ground:'5S₁/₂',D1:'5P₁/₂',D2:'5P₃/₂'} as const;
/** F, F″ (5P₁/₂) and F′ (5P₃/₂): the hyperfine quantum number of each manifold. */
export function fSymbol(manifold:Level['manifold']):string{return manifold==='ground'?'F':manifold==='D1'?'F″':'F′';}

/** The one state label: "5S₁/₂ · F=2 · m=−1". */
export function stateLabel(level:Pick<Level,'manifold'|'F'|'m'>):string{
  return `${TERMS[level.manifold]} · ${fSymbol(level.manifold)}=${level.F} · m=${signedInteger(level.m)}`;
}
/** A hyperfine level: "5S₁/₂ · F=2". */
export function hyperfineLabel(manifold:Level['manifold'],F:number):string{return `${TERMS[manifold]} · ${fSymbol(manifold)}=${F}`;}

/** The state an id names ("ground-2--1"), when it is one of the 32. */
export function parseStateId(id:string):Pick<Level,'manifold'|'F'|'m'>|undefined{
  const match=/^(ground|D1|D2)-(\d)-(-?\d)$/.exec(id);
  return match?{manifold:match[1] as Level['manifold'],F:Number(match[2]),m:Number(match[3])}:undefined;
}
/** A label for any id the workbench shows: a state, else what `fallback` says (a reference's name). */
export function idLabel(id:string,fallback?:(id:string)=>string|undefined):string{
  const state=parseStateId(id);
  return state?stateLabel(state):fallback?.(id)??id;
}
/** A message that names states by their ids ("ground-2-2 → D2-3-3") in the one state label. */
export function labelStateIds(text:string):string{return text.replace(/\b(?:ground|D1|D2)-\d--?\d\b/g,id=>idLabel(id));}
