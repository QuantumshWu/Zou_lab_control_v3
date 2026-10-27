import { renderToString } from 'katex';
import type { Level } from './types.js';

const rendered=new Map<string,string>();
export function math(source:string,display=false):string {
  const key=`${display}:${source}`,cached=rendered.get(key);if(cached)return cached;
  const html=renderToString(source,{displayMode:display,throwOnError:false,strict:'error',output:'html',trust:false});
  if(rendered.size<512)rendered.set(key,html);return html;
}
export function term(manifold:Level['manifold']):string {
  return `<span class="atomic-term">${manifold==='ground'?'5S':'5P'}<sub>${manifold==='D2'?'3/2':'1/2'}</sub></span>`;
}
export function shortKet(F:number,m:number):string {return `|${F}, ${m}⟩`;}
