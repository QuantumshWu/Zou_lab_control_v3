import {inkBox} from '../../desktop/ui/src/fluent/ink.js';
export function readoutBaseline(line:number,side:'above'|'below',ink:{top:number;bottom:number},gap=8):number{return side==='above'?line-gap-ink.bottom:line+gap-ink.top;}
/** Match the actual ink-to-energy-line gap, not unlike alphabetic baselines. */
export function alignLevelReadouts(host:HTMLElement):void{
 for(const node of host.querySelectorAll<SVGTextElement>('text[data-line-y]')){
  const line=Number(node.dataset.lineY),side=node.dataset.side as 'above'|'below';
  const family=node.ownerSVGElement?.getAttribute('font-family')??'Segoe UI, sans-serif';
  const font=`${node.getAttribute('font-weight')??400} ${node.getAttribute('font-size')??13}px ${family}`,ink=inkBox(node.textContent??'',font);
  const cx=Number(node.dataset.centerX);node.setAttribute('x',String(cx-(ink.left+ink.right)/2));node.setAttribute('y',String(readoutBaseline(line,side,ink)));node.setAttribute('text-anchor','start');node.removeAttribute('dominant-baseline');
 }
}
