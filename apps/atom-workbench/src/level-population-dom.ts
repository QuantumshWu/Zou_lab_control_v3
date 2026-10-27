import {populationAppearance} from './charts.js';
import {populationValue} from './population-chart.js';

/** Only population presentation is mutable here. Energy coordinates, arrows,
 * detunings and focusable state nodes retain their identity and attributes. */
export function patchLevelPopulations(host:HTMLElement,options:{populations?:Record<string,number>;steady?:Record<string,number>;label:string;steadyLabel:string;initialId:string}):void {
  for(const node of host.querySelectorAll<SVGGElement>('g.level-node:not(.reference-node)')){
    const id=node.dataset.state!,p=options.populations?.[id],steady=options.steady?.[id],appearance=id===options.initialId?{fill:'#c8c6c4',ink:'#323130'}:p===undefined?{fill:'var(--surface, #fff)',ink:'#323130'}:populationAppearance(p);
    (node.querySelector('.state-population-background')??node.querySelector('rect'))?.setAttribute('fill',appearance.fill);
    node.querySelector('.state-energy')?.setAttribute('stroke',appearance.ink);
    for(const text of node.querySelectorAll('.adiabatic-label,.steady-population-label'))text.setAttribute('fill',appearance.ink);
    node.querySelector('.level-value-label')?.setAttribute('fill','#323130');
    if(id===options.initialId)(node.querySelector('.state-initial-dot')??node.querySelector('circle[r="2.7"]'))?.setAttribute('fill',p===undefined?'#2f6fb0':appearance.ink);
    // The number in its slot, "%" after it (combined-level-chart.ts readout).
    const value=node.querySelector('.steady-population-label');if(value){value.textContent=steady===undefined?'—':populationValue(steady).replace(/%$/,'');const unit=value.nextElementSibling;if(unit?.classList.contains('level-readout-unit'))unit.textContent=steady===undefined?'':'%';}
    const title=node.querySelector('title'),base=node.dataset.baseLabel??(title?.textContent??'').replace(/; (?:稳态布居|长期平均布居|布居|p\([^;]+?\)|p̄\([^;]+?\)|长期布居极限|长期布居) =?[^;]*/g,'');
    node.dataset.baseLabel=base;
    const label=base+(p===undefined?'':`; ${options.label} = ${populationValue(p)}`)+(id.startsWith('ground-')?`; ${options.steadyLabel} ${steady===undefined?'未知':populationValue(steady)}`:'');
    if(title)title.textContent=label;node.setAttribute('aria-label',label);
  }
}
