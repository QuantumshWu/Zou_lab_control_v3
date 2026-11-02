// 1D motion's charts (ui-design.md §6.20): the temperature and the mean velocity along the axis
// against time, sharing one time axis, and the momentum distribution -- as a velocity density --
// at the last time and, when asked, at another listed time. Each reads actual samples only.
import type {DurationUnit} from './types.js';
import type {MotionResult} from './solver-types.js';
import {RB87} from './physics.js';
import {timeDisplayUnit} from './duration.js';
import {formatNumber} from './format.js';

/** k_B (J/K), exact in the SI. */
const BOLTZMANN=1.380649e-23;
const SVG='http://www.w3.org/2000/svg';
const esc=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const clamp=(n:number,a:number,b:number)=>Math.min(b,Math.max(a,n));
/** The v3 series palette's first two (population-chart.ts): the last time, then the compared one. */
export const MOTION_INKS=['#517FA6','#B17D4B'] as const;
const GRID='#e1dfdd',INK='#6a6866',LEFT=64,RIGHT=778;

/** Velocity units, largest first. */
const VELOCITY_UNITS:readonly (readonly [string,number])[]=[['m/s',1],['cm/s',1e-2],['mm/s',1e-3],['μm/s',1e-6]];
/** The largest velocity unit in which `magnitude` (m/s) reads at least 1. */
export function velocityUnit(magnitude:number):{name:string;factor:number}{
  const [name,size]=VELOCITY_UNITS.find(([,size])=>magnitude>=size)??VELOCITY_UNITS.at(-1)!;return {name,factor:1/size};
}
/** The thermal width of the velocity along one axis, sqrt(k_B T/m) (m/s), at T (μK). */
export function thermalVelocity(temperatureUK:number):number{return Math.sqrt(BOLTZMANN*temperatureUK*1e-6/RB87.massKg);}
/** The step of 1, 2 or 5 × 10ⁿ that cuts `span` into at most `intervals` parts. */
function niceStep(span:number,intervals:number):number{
  const raw=span/intervals,decade=10**Math.floor(Math.log10(raw));return [1,2,5,10].map(f=>f*decade).find(s=>s>=raw)!;
}
/** Ticks 0, s, 2s … up to the first that reaches `top` (> 0), about `intervals` of them. */
function ticks(top:number,intervals=4):number[]{const step=niceStep(top,intervals);return Array.from({length:Math.ceil(top/step-1e-9)+1},(_,i)=>i*step);}
/** The multiples of a step of 1, 2 or 5 × 10ⁿ within [low, high], about `intervals` of them. */
function ticksWithin(low:number,high:number,intervals:number):number[]{
  const step=niceStep(high-low,intervals),out:number[]=[];for(let k=Math.ceil(low/step-1e-9);k*step<=high*(1+1e-9);k++)out.push(k*step);return out;
}
const digits=(n:number)=>formatNumber(n,{digits:4});

/** A grid point's share of the velocity axis: half the way to each neighbour (the whole way at an end). */
function cellWidths(v:readonly number[]):number[]{return v.map((_,i)=>((v[Math.min(i+1,v.length-1)]!-v[Math.max(i-1,0)]!)/(i===0||i===v.length-1?1:2)));}
/** A listed distribution as a density on the velocity axis (per m/s). */
export function momentumDensity(result:MotionResult,listed:number):number[]{
  const widths=cellWidths(result.momentum.velocitiesMps);return result.momentum.probabilities[listed]!.map((p,i)=>p/widths[i]!);
}

export interface MotionSeriesLayout {top:{top:number;bottom:number};bottom:{top:number;bottom:number};unit:{name:DurationUnit;factor:number};
  velocity:{name:string;factor:number};temperatureTop:number;velocityTop:number;x:(t:number)=>number;yT:(T:number)=>number;yV:(v:number)=>number}
/** The two panels: T from 0 to the top of its ticks; ⟨v⟩ symmetric about 0, as wide as the larger
 * of its own reach and the thermal width -- a drift is read against the spread it moves in. */
export function motionSeriesLayout(timesUs:readonly number[],motion:MotionResult,timeUnit?:DurationUnit):MotionSeriesLayout{
  const top={top:32,bottom:176},bottom={top:212,bottom:300},span=timesUs.at(-1)!-timesUs[0]!;
  const temperatureTop=ticks(Math.max(...motion.temperatureUK)).at(-1)!;
  const reach=Math.max(...motion.meanVelocityMps.map(Math.abs),...motion.temperatureUK.map(thermalVelocity)),velocity=velocityUnit(reach);
  const velocityTop=ticks(reach*velocity.factor,2).at(-1)!/velocity.factor;
  return {top,bottom,unit:timeDisplayUnit(timesUs.at(-1)!,timeUnit),velocity,temperatureTop,velocityTop,
    x:t=>LEFT+(t-timesUs[0]!)/span*(RIGHT-LEFT),yT:T=>top.bottom-T/temperatureTop*(top.bottom-top.top),
    yV:v=>(bottom.top+bottom.bottom)/2-v/velocityTop*(bottom.bottom-bottom.top)/2};
}
const line=(x1:number,y1:number,x2:number,y2:number,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const text=(x:number,y:number,label:string,attrs='')=>`<text x="${x}" y="${y}" font-size="12" fill="${INK}" ${attrs}>${esc(label)}</text>`;
const polyline=(xs:readonly number[],ys:readonly number[],ink:string,attrs='')=>`<polyline points="${xs.map((x,i)=>`${x.toFixed(4)},${ys[i]!.toFixed(4)}`).join(' ')}" fill="none" stroke="${ink}" stroke-width="2" ${attrs}/>`;
const svg=(cls:string,label:string,body:string)=>`<svg class="chart-svg ${cls}" viewBox="0 0 800 355" xmlns="${SVG}" role="img" tabindex="0" aria-label="${esc(label)}" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" style="fill:#323130;outline:none">${body}</svg>`;

/** T(t) above ⟨v⟩(t), on the result's samples. */
export function motionSeriesChart(timesUs:readonly number[],motion:MotionResult,timeUnit?:DurationUnit):string{
  const g=motionSeriesLayout(timesUs,motion,timeUnit),xs=timesUs.map(g.x);
  let body='';
  for(const T of ticks(g.temperatureTop))body+=line(LEFT,g.yT(T),RIGHT,g.yT(T),`stroke="${GRID}"`)+text(LEFT-12,g.yT(T)+4,digits(T),'text-anchor="end"');
  const top=g.velocityTop*g.velocity.factor;
  for(const v of ticks(top,2).flatMap(v=>v?[-v,v]:[0]))body+=line(LEFT,g.yV(v/g.velocity.factor),RIGHT,g.yV(v/g.velocity.factor),`stroke="${GRID}"`)+text(LEFT-12,g.yV(v/g.velocity.factor)+4,digits(v),'text-anchor="end"');
  for(let i=0;i<=4;i++){const t=timesUs[0]!+(timesUs.at(-1)!-timesUs[0]!)*i/4;body+=text(g.x(t),g.bottom.bottom+24,digits(t*g.unit.factor),'text-anchor="middle"');}
  body+=text(LEFT,19,'温度 T / μK')+text(LEFT,g.bottom.top-8,`平均速度 ⟨v⟩ / ${g.velocity.name}`)+text(RIGHT,g.bottom.bottom+47,`t / ${g.unit.name}`,'text-anchor="end"');
  body+=polyline(xs,motion.temperatureUK.map(g.yT),MOTION_INKS[0],'data-series-id="temperature"')+polyline(xs,motion.meanVelocityMps.map(g.yV),MOTION_INKS[1],'data-series-id="mean-velocity"');
  return svg('motion-series-svg','沿运动轴的温度与平均速度随时间的变化。移动指针读取采样值，方向键逐点查看。',body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}

/** The momentum chart's axes: the grid's whole span; densities from 0 to the top of their ticks. */
function momentumLayout(motion:MotionResult,listed:readonly number[]){
  const v=motion.momentum.velocitiesMps,unit=velocityUnit(Math.max(Math.abs(v[0]!),Math.abs(v.at(-1)!)));
  const curves=listed.map(i=>momentumDensity(motion,i)),top=ticks(Math.max(...curves.flat())/unit.factor).at(-1)!;
  return {v,unit,curves,top,x:(w:number)=>LEFT+(w-v[0]!)/(v.at(-1)!-v[0]!)*(RIGHT-LEFT),y:(density:number)=>300-density/unit.factor/top*(300-32)};
}
/** The momentum distributions drawn: the last listed time's and, when asked, another listed one's. */
export function momentumChart(motion:MotionResult,listed:readonly number[]):string{
  const {v,unit,curves,top,x,y}=momentumLayout(motion,listed);
  let body='';
  for(const d of ticks(top))body+=line(LEFT,y(d*unit.factor),RIGHT,y(d*unit.factor),`stroke="${GRID}"`)+text(LEFT-12,y(d*unit.factor)+4,digits(d),'text-anchor="end"');
  for(const w of ticksWithin(v[0]!*unit.factor,v.at(-1)!*unit.factor,8))body+=text(x(w/unit.factor),324,digits(w),'text-anchor="middle"');
  body+=text(LEFT,19,`概率密度 / (${unit.name})⁻¹`)+text(RIGHT,347,`v / ${unit.name}`,'text-anchor="end"');
  // The compared time behind, dashed; the last time on top.
  curves.slice().reverse().forEach((density,k)=>{const index=curves.length-1-k;body+=polyline(v.map(x),density.map(y),MOTION_INKS[index]!,`data-series-id="momentum-${index}"${index?' stroke-dasharray="6 4"':''}`);});
  return svg('momentum-svg','沿运动轴的速度分布。移动指针读取各速度的概率密度。',body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}

/** A readout card beside a cursor, on its wider side (as the population chart's). */
function readoutCard(host:HTMLElement,label:string):HTMLDivElement{
  const card=document.createElement('div');card.className='population-readout card bordered motion-readout';card.setAttribute('role','region');card.setAttribute('aria-label',label);card.hidden=true;host.append(card);return card;
}
function placeCard(host:HTMLElement,svg:SVGSVGElement,card:HTMLElement,xx:number):void{
  card.hidden=false;
  const bounds=svg.getBoundingClientRect(),parent=host.getBoundingClientRect(),px=(xx/800)*bounds.width+bounds.left-parent.left,right=px<host.clientWidth/2;
  card.style.maxWidth=`${Math.max(0,right?host.clientWidth-px-26:px-26)}px`;
  const width=card.offsetWidth;card.style.left=`${clamp(right?px+18:px-width-18,8,Math.max(8,host.clientWidth-width-8))}px`;card.style.top='27px';
}
/** A cursor that follows the pointer (and the arrow keys) over `count` points at plot x `at(i)`. */
function bindCursor(host:HTMLElement,svg:SVGSVGElement,count:number,at:(i:number)=>number,show:(i:number,cursor:SVGGElement)=>void,hide:()=>void):()=>void{
  const cursor=svg.querySelector<SVGGElement>('.motion-cursor')!,abort=new AbortController();let index=count-1;
  const go=(next:number)=>{index=clamp(next,0,count-1);cursor.replaceChildren();cursor.style.display='';show(index,cursor);};
  const off=()=>{cursor.style.display='none';hide();};
  svg.addEventListener('pointermove',event=>{
    const transform=svg.getScreenCTM();if(!transform)return;const p=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
    if(p.x<LEFT||p.x>RIGHT||p.y<32||p.y>300){off();return;}
    let best=0;for(let i=1;i<count;i++)if(Math.abs(at(i)-p.x)<Math.abs(at(best)-p.x))best=i;go(best);
  },{signal:abort.signal});
  svg.addEventListener('pointerleave',off,{signal:abort.signal});svg.addEventListener('blur',off,{signal:abort.signal});
  svg.addEventListener('keydown',event=>{const next={ArrowLeft:index-1,ArrowRight:index+1,Home:0,End:count-1}[event.key];if(event.key==='Escape'){off();return;}if(next===undefined)return;event.preventDefault();go(next);},{signal:abort.signal});
  return ()=>abort.abort();
}
const mark=(name:string,attrs:Record<string,string>)=>{const node=document.createElementNS(SVG,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node;};

/** The time cursor of the T and ⟨v⟩ chart: the sample's T and ⟨v⟩. */
export function bindMotionSeries(host:HTMLElement,timesUs:readonly number[],motion:MotionResult,timeUnit?:DurationUnit):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.motion-series-svg');if(!svg)return ()=>{};
  const g=motionSeriesLayout(timesUs,motion,timeUnit),card=readoutCard(host,'所选时刻的温度与平均速度');
  const unbind=bindCursor(host,svg,timesUs.length,i=>g.x(timesUs[i]!),(i,cursor)=>{
    const xx=g.x(timesUs[i]!),T=motion.temperatureUK[i]!,v=motion.meanVelocityMps[i]!;
    cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:String(g.top.top),y2:String(g.bottom.bottom),stroke:'#a19f9d','stroke-width':'1'}),
      mark('circle',{cx:String(xx),cy:String(g.yT(T)),r:'3',fill:MOTION_INKS[0],stroke:'white','stroke-width':'1.5'}),mark('circle',{cx:String(xx),cy:String(g.yV(v)),r:'3',fill:MOTION_INKS[1],stroke:'white','stroke-width':'1.5'}));
    card.innerHTML=`<div class="readout-time">t = ${esc(digits(timesUs[i]!*g.unit.factor))} ${esc(g.unit.name)}</div>`+
      `<div class="readout-row two"><span class="readout-name" style="color:${MOTION_INKS[0]}">温度 T</span><span>${esc(formatNumber(T,{digits:4}))} μK</span></div>`+
      `<div class="readout-row two"><span class="readout-name" style="color:${MOTION_INKS[1]}">⟨v⟩</span><span>${esc(formatNumber(v*g.velocity.factor,{digits:4}))} ${esc(g.velocity.name)}</span></div>`;
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
/** The velocity cursor of the momentum chart: each drawn distribution's density there. */
export function bindMomentum(host:HTMLElement,motion:MotionResult,listed:readonly number[],names:readonly string[]):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.momentum-svg');if(!svg)return ()=>{};
  const {v,unit,curves,x,y}=momentumLayout(motion,listed),card=readoutCard(host,'所选速度的概率密度');
  const unbind=bindCursor(host,svg,v.length,i=>x(v[i]!),(i,cursor)=>{
    const xx=x(v[i]!);cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:'32',y2:'300',stroke:'#a19f9d','stroke-width':'1'}));
    curves.forEach((density,k)=>cursor.append(mark('circle',{cx:String(xx),cy:String(y(density[i]!)),r:'3',fill:MOTION_INKS[k]!,stroke:'white','stroke-width':'1.5'})));
    card.innerHTML=`<div class="readout-time">v = ${esc(digits(v[i]!*unit.factor))} ${esc(unit.name)}</div>`+curves.map((density,k)=>
      `<div class="readout-row two"><span class="readout-name" style="color:${MOTION_INKS[k]}">${esc(names[k]??'')}</span><span>${esc(formatNumber(density[i]!/unit.factor,{digits:4}))} (${esc(unit.name)})⁻¹</span></div>`).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
