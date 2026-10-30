import {populationAppearance} from './charts.js';
import {populationValue} from './population-chart.js';

/** Only population presentation is mutable here. Energy coordinates, arrows,
 * detunings and focusable state nodes retain their identity and attributes. */
export function patchLevelPopulations(host:HTMLElement,options:{populations?:Record<string,number>;mean?:Record<string,number>;label:string;meanLabel:string;initialId:string;
  /** After a value outside the population tolerance: its estimated error. */
  note?:string;meanNote?:string}):void {
  for(const node of host.querySelectorAll<SVGGElement>('g.level-node:not(.reference-node)')){
    const id=node.dataset.state!,p=options.populations?.[id],mean=options.mean?.[id],appearance=id===options.initialId?{fill:'#c8c6c4',ink:'#323130'}:p===undefined?{fill:'var(--surface, #fff)',ink:'#323130'}:populationAppearance(p);
    (node.querySelector('.state-population-background')??node.querySelector('rect'))?.setAttribute('fill',appearance.fill);
    node.querySelector('.state-energy')?.setAttribute('stroke',appearance.ink);
    for(const text of node.querySelectorAll('.adiabatic-label,.mean-population-label'))text.setAttribute('fill',appearance.ink);
    node.querySelector('.level-value-label')?.setAttribute('fill','#323130');
    if(id===options.initialId)(node.querySelector('.state-initial-dot')??node.querySelector('circle[r="2.7"]'))?.setAttribute('fill',p===undefined?'#2f6fb0':appearance.ink);
    // The number in its slot, "%" after it (combined-level-chart.ts readout).
    const value=node.querySelector('.mean-population-label');if(value){value.textContent=mean===undefined?'—':populationValue(mean).replace(/%$/,'');const unit=value.nextElementSibling;if(unit?.classList.contains('level-readout-unit'))unit.textContent=mean===undefined?'':'%';}
    const title=node.querySelector('title'),base=node.dataset.baseLabel??(title?.textContent??'').replace(/; (?:长期平均布居|布居|p\([^;]+?\)|p̄\([^;]+?\)|长期布居极限|长期布居) =?[^;]*/g,'');
    node.dataset.baseLabel=base;
    // A ground state also reads p̄(T); when the colouring is p̄(T) itself it is said once.
    const same=p!==undefined&&options.label===options.meanLabel;
    const label=base+(p===undefined?'':`; ${options.label} = ${populationValue(p)}${options.note??''}`)+(id.startsWith('ground-')&&!same?`; ${options.meanLabel} = ${mean===undefined?'未知':populationValue(mean)}${mean===undefined?'':options.meanNote??''}`:'');
    if(title)title.textContent=label;node.setAttribute('aria-label',label);
  }
}
