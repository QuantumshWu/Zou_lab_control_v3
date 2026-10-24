import {timeDisplayUnit} from './duration.js';
import type {DurationUnit} from './types.js';
import type {SteadyResult} from './solver-types.js';

export interface PopulationPlot {timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];steady?:SteadyResult;}
export interface PopulationChartOptions {scale?:'linear'|'log';viewportWidth?:number;initialId?:string;timeUnit?:DurationUnit;showTotalExcited?:boolean;}
// v3 scientific series palette: crates/zlc-plot/src/style.rs, Palette::default.
const palette=['#517FA6','#B17D4B','#568C79','#99719B','#B36F72','#558F9C','#7E7CA5','#92924F'];
const esc=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const clamp=(n:number,a:number,b:number)=>Math.min(b,Math.max(a,n));
const SVG='http://www.w3.org/2000/svg';
function stateIndex(id:string):number {
  const m=/^(ground|D1|D2)-(\d)-(-?\d)$/.exec(id);if(!m)return [...id].reduce((n,c)=>(n*31+c.charCodeAt(0))>>>0,0);
  const F=Number(m[2]),z=Number(m[3]);return m[1]==='ground'?(F===1?z+1:z+5):m[1]==='D1'?(F===1?z+9:z+13):16+F*F+z+F;
}
export const populationColor=(id:string):string=>palette[(stateIndex(id)+3+(id.startsWith('D1-')?1:id.startsWith('D2-')?2:0))%palette.length]!;
export function populationValue(p:number):string {
  if(!Number.isFinite(p))return '—';
  const rounded=Math.round(p*10000)/100;
  return `${Object.is(rounded,-0)?'0.00':rounded.toFixed(2)}%`;
}
function digits(n:number):string{return String(Number(n.toPrecision(6)));}
function powerLabel(n:number):string{return `10${String(n).replace(/[\d-]/g,c=>({'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'}[c]!))}`;}
export function populationSamplingInfo(result:PopulationPlot):string {
  if(result.timesUs.length<2)return '没有时间演化采样。';
  const intervals=result.timesUs.slice(1).map((t,i)=>t-result.timesUs[i]!),min=Math.min(...intervals),max=Math.max(...intervals);
  if(!Number.isFinite(min)||min<=0)return '时间采样无效。';
  const unit=timeDisplayUnit(max);
  const uniform=max-min<=Math.max(1e-14,max*1e-8),spacing=uniform?`Δt = ${digits(max*unit.factor)} ${unit.name}`:`Δt = ${digits(min*unit.factor)}–${digits(max*unit.factor)} ${unit.name}（非均匀）`;
  return `${result.timesUs.length} 个实际求解采样；${spacing}。连线仅连接样点，不推断未采样的运动。`;
}

/** Pure axis geometry. Missing values are never turned into physical zeros. */
export function populationLayout(result:PopulationPlot,ids:string[],options:PopulationChartOptions={}) {
  const left=64,right=778,top=32,bottom=300;
  let error='';const times=result.timesUs;
  if(times.length<2)error='没有时间演化采样。请计算所选时长的轨迹。';
  else if(times.some((t,i)=>!Number.isFinite(t)||t<0||(i>0&&t<=times[i-1]!)))error='时间采样无效，未绘制轨迹。';
  else if([...ids.map(id=>result.populations[id]),result.totalExcited].some(values=>!values||values.length!==times.length||values.some(v=>!Number.isFinite(v))))error='求解结果缺少完整采样数据，未绘制轨迹。';
  const minTime=times[0]??0,maxTime=times.at(-1)??1,span=maxTime-minTime;
  const unit=timeDisplayUnit(maxTime,options.timeUnit);
  const peaks=ids.map(id=>Math.max(0,...(result.populations[id]??[]).filter(Number.isFinite))).filter(p=>p>0);
  const minPeak=peaks.length?Math.min(...peaks):1;
  const floorExponent=Math.max(-30,Math.min(-4,Math.floor(Math.log10(minPeak))-2)),floor=10**floorExponent;
  const x=(t:number)=>left+(t-minTime)/Math.max(1e-30,span)*(right-left);
  const y=(p:number)=>options.scale==='log'?bottom-(Math.log10(Math.max(floor,p))-floorExponent)/(-floorExponent)*(bottom-top):bottom-p*(bottom-top);
  return {left,right,top,bottom,minTime,maxTime,unit,floor,floorExponent,x,y,error};
}

export function populationChart(result:PopulationPlot,monitoredIds:string[],labels:Record<string,string>,options:PopulationChartOptions={}):string {
  const g=populationLayout(result,monitoredIds,options),{left,right,top,bottom,x,y}=g;
  if(g.error)return `<div class="empty-state population-data-empty" role="status">${esc(g.error)}</div>`;
  const scale=options.scale??'linear';
  const text=(xx:number,yy:number,label:string,attrs='')=>`<text x="${xx}" y="${yy}" ${attrs}>${esc(label)}</text>`;
  const line=(x1:number,y1:number,x2:number,y2:number,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
  let svg='<defs><clipPath id="population-clip"><rect x="64" y="30" width="714" height="272"/></clipPath></defs>';
  const ticks=scale==='log'?Array.from({length:-g.floorExponent+1},(_,i)=>g.floorExponent+i).filter((e,i,all)=>e===0||i===0||e%Math.ceil(-g.floorExponent/6)===0).map(e=>({p:10**e,label:`${e===g.floorExponent?'≤ ':''}${e===0?'1':powerLabel(e)}`})):Array.from({length:5},(_,i)=>({p:i/4,label:digits(i/4)}));
  for(const tick of ticks)svg+=line(left,y(tick.p),right,y(tick.p),'stroke="#e1dfdd" stroke-width="1"')+text(left-12,y(tick.p)+4,tick.label,'text-anchor="end" font-size="12" fill="#6a6866"');
  for(let i=0;i<=4;i++){const t=g.minTime+(g.maxTime-g.minTime)*i/4;svg+=text(x(t),bottom+24,digits(t*g.unit.factor),'text-anchor="middle" font-size="12" fill="#6a6866"');}
  svg+=text(left,19,scale==='log'?'布居 p · 对数':'布居 p','font-size="12" fill="#6a6866"')+text(right,bottom+47,`t / ${g.unit.name}`,'text-anchor="end" font-size="12" fill="#6a6866"');
  const series=(id:string,values:number[],color:string,aggregate=false)=>{
    const points=result.timesUs.map((t,i)=>`${x(t).toFixed(4)},${y(values[i]!).toFixed(4)}`).join(' '),initial=id===options.initialId;
    const title=`${aggregate?'所有激发态总布居':labels[id]??id}${initial?' · 初态':''}`;
    let markup=`<polyline data-series-id="${esc(id)}" points="${points}" fill="none" stroke="${color}" stroke-width="${aggregate?1.4:initial?2.5:2}"><title>${esc(title)}</title></polyline>`;
    if(scale==='log')for(const i of [0,values.length-1])if(values[i]!<=g.floor)markup+=`<circle cx="${x(result.timesUs[i]!)}" cy="${bottom}" r="2.6" fill="white" stroke="${color}"><title>${esc(`${title}: ${populationValue(values[i]!)}（对数轴下界以下）`)}</title></circle>`;
    return markup;
  };
  svg+='<g clip-path="url(#population-clip)">';
  if(options.showTotalExcited!==false)svg+=series('__excited__',result.totalExcited,'#808080',true);
  for(const id of monitoredIds)svg+=series(id,result.populations[id]!,populationColor(id));
  svg+='</g><g class="population-cursor" pointer-events="none" style="display:none"></g>';
  return `<svg class="chart-svg population-svg" viewBox="0 0 800 355" xmlns="${SVG}" role="img" tabindex="0" data-population-scale="${scale}" aria-label="态布居随时间的演化。移动指针读取采样值，方向键逐点查看，Home 和 End 到达首末采样。" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" style="fill:#323130;outline:none"><title>${esc(populationSamplingInfo(result))}</title>${svg}</svg>`;
}

/** A common time cursor reads actual solver samples, including values below
 * the displayed log floor. It never interpolates missing data or invents p∞. */
export function bindPopulationChart(host:HTMLElement,result:PopulationPlot,ids:string[],labels:Record<string,string>,options:PopulationChartOptions={}):()=>void {
  const found=host.querySelector<SVGSVGElement>('svg.population-svg');if(!found)return ()=>{};const svg:SVGSVGElement=found;
  const g=populationLayout(result,ids,options),cursor=svg.querySelector<SVGGElement>('.population-cursor')!;
  const tooltip=document.createElement('div');tooltip.className='population-readout';tooltip.setAttribute('role','region');tooltip.setAttribute('aria-label','所选时刻的态布居');tooltip.tabIndex=0;tooltip.hidden=true;
  tooltip.style.cssText='position:absolute;z-index:5;pointer-events:auto;background:var(--surface,#fff);border:1px solid var(--divider,#e1dfdd);border-radius:4px;padding:9px 11px;box-sizing:border-box;overflow-y:auto;overscroll-behavior:contain;box-shadow:0 4px 16px #32313014;font:12px var(--font-family,Segoe UI);max-width:calc(100% - 16px);min-width:240px';host.append(tooltip);
  const abort=new AbortController();let index=result.timesUs.length-1;
  const element=(name:string,attrs:Record<string,string>)=>{const el=document.createElementNS(SVG,name);for(const [key,value]of Object.entries(attrs))el.setAttribute(key,value);return el;};
  function show(next:number):void {
    index=clamp(next,0,result.timesUs.length-1);const t=result.timesUs[index]!,xx=g.x(t);
    cursor.replaceChildren(element('line',{x1:String(xx),x2:String(xx),y1:String(g.top),y2:String(g.bottom),stroke:'#a19f9d','stroke-width':'1'}));
    for(const id of ids){const p=result.populations[id]![index]!;cursor.append(element('circle',{cx:String(xx),cy:String(g.y(p)),r:'3',fill:options.scale==='log'&&p<=g.floor?'white':populationColor(id),stroke:'white','stroke-width':'1.5'}));}
    cursor.style.display='';
    const steady=result.steady?.converged&&result.steady.populations?result.steady:undefined,mean=steady?.status==='periodic-mean'||steady?.status==='initial-dependent-mean';
    const head=steady?`<div style="display:grid;grid-template-columns:minmax(100px,1fr) 80px 80px;gap:10px;color:#6a6866;font-size:11px;margin:0 0 5px"><span></span><span style="text-align:right">p(t)</span><span style="text-align:right">${mean?'〈p〉∞':'p∞'}</span></div>`:'';
    const asymptotic=(id:string)=>{if(id!=='__excited__')return steady?.populations?.[id]??NaN;const excited=Object.entries(steady?.populations??{}).filter(([state])=>/^(D1|D2)-/.test(state));return excited.length?excited.reduce((sum,[,p])=>sum+p,0):NaN;};
    const row=(id:string,name:string,p:number,color:string)=>`<div data-population-readout="${esc(id)}" style="display:grid;grid-template-columns:minmax(100px,1fr) 80px${steady?' 80px':''};gap:10px;align-items:center;margin:4px 0"><span title="${esc(name)}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px;color:${color}">${esc(name)}${id===options.initialId?' · ψ₀':''}</span><span style="text-align:right;font-variant-numeric:tabular-nums">${esc(populationValue(p))}</span>${steady?`<span style="text-align:right;color:#6a6866;font-variant-numeric:tabular-nums">${esc(populationValue(asymptotic(id)))}</span>`:''}</div>`;
    tooltip.innerHTML=`<div style="font-weight:600;margin-bottom:7px">t = ${digits(t*g.unit.factor)} ${g.unit.name}</div>${head}${ids.map(id=>row(id,labels[id]??id,result.populations[id]![index]!,populationColor(id))).join('')}${options.showTotalExcited===false?'':row('__excited__','激发态合计',result.totalExcited[index]!,'#808080')}${mean?'<div style="color:#6a6866;font-size:11px;margin-top:6px">〈p〉∞ 为长期时间平均，不是瞬时布居。</div>':''}`;
    tooltip.hidden=false;tooltip.dataset.sampleIndex=String(index);tooltip.style.maxHeight=`${Math.max(100,host.clientHeight-35)}px`;
    const bounds=svg.getBoundingClientRect(),parent=host.getBoundingClientRect(),px=(xx/800)*bounds.width+bounds.left-parent.left,tw=tooltip.offsetWidth;
    tooltip.style.left=`${clamp(px<host.clientWidth/2?px+18:px-tw-18,8,Math.max(8,host.clientWidth-tw-8))}px`;tooltip.style.top='27px';
  }
  function hide():void {cursor.style.display='none';tooltip.hidden=true;}
  svg.addEventListener('pointermove',event=>{
    const transform=svg.getScreenCTM();if(!transform)return;const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());if(point.x<g.left||point.x>g.right||point.y<g.top||point.y>g.bottom){hide();return;}
    const t=g.minTime+(point.x-g.left)/(g.right-g.left)*(g.maxTime-g.minTime);let lo=0,hi=result.timesUs.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(result.timesUs[mid]!<t)lo=mid;else hi=mid;}show(Math.abs(result.timesUs[lo]!-t)<=Math.abs(result.timesUs[hi]!-t)?lo:hi);
  },{signal:abort.signal});
  const enters=(container:Element,target:EventTarget|null)=>target instanceof Node&&container.contains(target);
  svg.addEventListener('pointerleave',event=>{if(!enters(tooltip,event.relatedTarget))hide();},{signal:abort.signal});
  svg.addEventListener('blur',event=>{if(!enters(tooltip,event.relatedTarget))hide();},{signal:abort.signal});
  tooltip.addEventListener('pointerleave',event=>{if(!enters(svg,event.relatedTarget)&&!tooltip.contains(document.activeElement))hide();},{signal:abort.signal});
  tooltip.addEventListener('focusout',event=>{if(!enters(svg,event.relatedTarget)&&!enters(tooltip,event.relatedTarget))hide();},{signal:abort.signal});
  tooltip.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();hide();svg.focus({preventScroll:true});}},{signal:abort.signal});
  svg.addEventListener('keydown',event=>{let next=index;switch(event.key){case'ArrowLeft':next--;break;case'ArrowRight':next++;break;case'Home':next=0;break;case'End':next=result.timesUs.length-1;break;case'Escape':hide();return;default:return;}event.preventDefault();show(next);},{signal:abort.signal});
  return ()=>{abort.abort();tooltip.remove();};
}
