// Motion's charts (ui-design.md §6.20): two panels against time sharing one time axis -- the
// temperature above n̄, a line per lab axis and the total temperature beside them -- and a
// distribution along one axis -- of the velocity or of the position, as a density -- at the last
// time and, when asked, at another listed time. Each reads actual samples only. The samples outside
// the method's validated regime (or its approximation ledger) lie on a light grey band: a reference
// only. A standard error is read in the cursor's readout, never drawn.
import type {DurationUnit,MotionAxis} from './types.js';
import type {Distributions,MotionResult} from './solver-types.js';
import {MOTION_AXES} from './motion.js';
import {timeDisplayUnit} from './duration.js';
import {formatNumber} from './format.js';
import {referenceBands,sampleMark} from './population-chart.js';

const SVG='http://www.w3.org/2000/svg';
const esc=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const clamp=(n:number,a:number,b:number)=>Math.min(b,Math.max(a,n));
/** The v3 series palette's first two (population-chart.ts): a distribution's last time, then the
 * compared time. */
export const MOTION_INKS=['#517FA6','#B17D4B'] as const;
/** The palette's first three, one per lab axis x, y, z. */
export const AXIS_INKS:Readonly<Record<MotionAxis,string>>={x:'#517FA6',y:'#B17D4B',z:'#568C79'};
/** The total temperature's line: the text's dark ink and wider than an axis's, so it reads apart
 * from the three. */
export const TOTAL_INK='#323130',TOTAL_WIDTH=3;
const GRID='#e1dfdd',INK='#6a6866',LEFT=64,RIGHT=778;

/** A display unit: its name and the factor from the base unit. */
interface Unit {name:string;factor:number}
/** Velocity units (in m/s) and length units (in μm), largest first. */
const VELOCITY_UNITS:readonly (readonly [string,number])[]=[['m/s',1],['cm/s',1e-2],['mm/s',1e-3],['μm/s',1e-6]];
const LENGTH_UNITS:readonly (readonly [string,number])[]=[['μm',1],['nm',1e-3]];
/** The largest of `units` in which `magnitude` (base units) reads at least 1. */
function unitFor(units:readonly (readonly [string,number])[],magnitude:number):Unit{
  const [name,size]=units.find(([,size])=>magnitude>=size)??units.at(-1)!;return {name,factor:1/size};
}
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

/** A line of a panel: its name, its values (base units), its ink and width; with its standard error
 * per sample (`error`, base units; null where it has none), read in the cursor's readout as value ± error. */
export interface Line {name:string;values:readonly number[];ink:string;width?:number;error?:readonly (number|null)[]}
/** One panel: its name, its axis title, its lines and display unit; its axis runs from 0 to the top
 * of its ticks. */
export interface Series {id:string;name:string;title:string;lines:Line[];unit:Unit}
const along=(axis:MotionAxis)=>`（${axis} 轴）`;
/** What a T is an average over (rust-solver/motion estimate.rs `half_window`, SPEC §1.7): the window about the output's
 * time t, one trap period of the axis, cut at t = 0 and kept centred there -- from 0 to 2t while t is under half a
 * period; t = 0 itself is that instant. */
export const TEMPERATURE_WINDOW='各轴一个阱周期内的平均；t 不到半个周期时取 0 到 2t';
/** The two panels of a motion result's time chart: T on each axis with the total temperature, above
 * n̄ on each axis (relative to the trap frequency the engine used along that axis). */
export function motionSeries(motion:MotionResult):[Series,Series]{
  const temperature:Line[]=[...MOTION_AXES.map((axis,k)=>({name:`T${along(axis)}`,values:motion.temperatureUK[k]!,ink:AXIS_INKS[axis],error:motion.temperatureErrorUK[k]!})),
    {name:'总温度',values:motion.temperatureTotalUK,ink:TOTAL_INK,width:TOTAL_WIDTH,error:motion.temperatureTotalErrorUK}];
  return [{id:'temperature',name:'温度 T',title:`温度 T / μK（${TEMPERATURE_WINDOW}）`,lines:temperature,unit:{name:'μK',factor:1}},
    {id:'occupation',name:'n̄',title:`平均振动量子数 n̄ = m⟨v²⟩/ħω − ½（各轴阱底处的阱频 ω/2π：${MOTION_AXES.map((axis,k)=>`${axis} ${digits(motion.trapFrequencyKHz[k]!)}`).join('、')} kHz）`,
      lines:MOTION_AXES.map((axis,k)=>({name:`n̄${along(axis)}`,values:motion.meanOccupation[k]!,ink:AXIS_INKS[axis]})),unit:{name:'',factor:1}}];
}
/** A panel: its series, its vertical span and ticks (display units), and where a value goes. */
interface Panel {series:Series;top:number;bottom:number;ticks:number[];y:(value:number)=>number}
function panel(series:Series,top:number,bottom:number):Panel{
  const factor=series.unit.factor,highest=ticks(Math.max(0,...series.lines.flatMap(l=>l.values))*factor||1).at(-1)!;
  return {series,top,bottom,ticks:ticks(highest),y:v=>bottom-v*factor/highest*(bottom-top)};
}
interface SeriesLayout {panels:[Panel,Panel];unit:{name:DurationUnit;factor:number};x:(t:number)=>number}
function seriesLayout(timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit):SeriesLayout{
  const span=timesUs.at(-1)!-timesUs[0]!;
  return {panels:[panel(series[0],32,176),panel(series[1],212,300)],unit:timeDisplayUnit(timesUs.at(-1)!,timeUnit),x:t=>LEFT+(t-timesUs[0]!)/span*(RIGHT-LEFT)};
}
const line=(x1:number,y1:number,x2:number,y2:number,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const text=(x:number,y:number,label:string,attrs='')=>`<text x="${x}" y="${y}" font-size="12" fill="${INK}" ${attrs}>${esc(label)}</text>`;
const polyline=(xs:readonly number[],ys:readonly number[],ink:string,attrs='',width=2)=>`<polyline points="${xs.map((x,i)=>`${x.toFixed(4)},${ys[i]!.toFixed(4)}`).join(' ')}" fill="none" stroke="${ink}" stroke-width="${width}" ${attrs}/>`;
const svg=(cls:string,label:string,body:string)=>`<svg class="chart-svg ${cls}" viewBox="0 0 800 355" xmlns="${SVG}" role="img" tabindex="0" aria-label="${esc(label)}" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" style="fill:#323130;outline:none">${body}</svg>`;

/** What a series chart marks on its samples: those outside the validated regime (a reference only). */
export interface SeriesMarks {reference?:readonly boolean[]}
/** The panels on the result's samples, a line per series line, the samples outside the validated
 * regime on their band behind each panel. */
export function seriesChart(timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit,marks:SeriesMarks={}):string{
  const g=seriesLayout(timesUs,series,timeUnit),xs=timesUs.map(g.x);
  let body=referenceBands(xs,marks.reference,g.panels.map(p=>[p.top,p.bottom] as const));
  for(const p of g.panels)for(const v of p.ticks){const y=p.y(v/p.series.unit.factor);body+=line(LEFT,y,RIGHT,y,`stroke="${GRID}"`)+text(LEFT-12,y+4,digits(v),'text-anchor="end"');}
  for(let i=0;i<=4;i++){const t=timesUs[0]!+(timesUs.at(-1)!-timesUs[0]!)*i/4;body+=text(g.x(t),g.panels[1].bottom+24,digits(t*g.unit.factor),'text-anchor="middle"');}
  body+=text(LEFT,19,g.panels[0].series.title)+text(LEFT,g.panels[1].top-8,g.panels[1].series.title)+text(RIGHT,g.panels[1].bottom+47,`t / ${g.unit.name}`,'text-anchor="end"');
  g.panels.forEach(p=>p.series.lines.forEach((l,k)=>{body+=polyline(xs,l.values.map(p.y),l.ink,`data-series-id="${p.series.id}-${k}"`,l.width);}));
  return svg('motion-series-svg',`${series.map(s=>s.name).join('与')}随时间的变化。移动指针读取采样值，方向键逐点查看。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}

/** The distribution kinds: the velocity (grid in m/s) and the position from the focus (grid in μm). */
export type DistributionKind='momentum'|'position';
const DISTRIBUTIONS:Readonly<Record<DistributionKind,{units:readonly (readonly [string,number])[];words:string}>>={
  momentum:{units:VELOCITY_UNITS,words:'速度'},position:{units:LENGTH_UNITS,words:'位置'},
};
/** A result's distributions of a kind along its axis number `axis` (an index into MOTION_AXES). */
export function distributions(motion:MotionResult,kind:DistributionKind,axis:number):Distributions{return (kind==='momentum'?motion.momentum:motion.position)[axis]!;}
/** A grid point's share of the axis: half the way to each neighbour (the whole way at an end). */
function cellWidths(v:readonly number[]):number[]{return v.map((_,i)=>((v[Math.min(i+1,v.length-1)]!-v[Math.max(i-1,0)]!)/(i===0||i===v.length-1?1:2)));}
/** A listed distribution as a density on its grid (per base unit). */
export function density(d:Distributions,listed:number):number[]{const widths=cellWidths(d.grid);return d.probabilities[listed]!.map((p,i)=>p/widths[i]!);}
/** The distribution chart's axes: the grid's whole span; densities from 0 to the top of their ticks. */
function distributionLayout(motion:MotionResult,kind:DistributionKind,listed:readonly number[],axis:number){
  const d=distributions(motion,kind,axis),grid=d.grid,unit=unitFor(DISTRIBUTIONS[kind].units,Math.max(Math.abs(grid[0]!),Math.abs(grid.at(-1)!)));
  const curves=listed.map(i=>density(d,i)),top=ticks(Math.max(...curves.flat())/unit.factor).at(-1)!;
  return {grid,unit,curves,top,x:(w:number)=>LEFT+(w-grid[0]!)/(grid.at(-1)!-grid[0]!)*(RIGHT-LEFT),y:(value:number)=>300-value/unit.factor/top*(300-32)};
}
/** The variable a distribution chart reads: the velocity along the axis, v（x 轴）, or the position, x. */
function axisSymbol(kind:DistributionKind,axis:number):string{const a=MOTION_AXES[axis]!;return kind==='momentum'?`v${along(a)}`:a;}
/** The distributions drawn: the last listed time's and, when asked, another listed one's. */
export function distributionChart(motion:MotionResult,kind:DistributionKind,listed:readonly number[],axis:number):string{
  const {grid,unit,curves,top,x,y}=distributionLayout(motion,kind,listed,axis),{words}=DISTRIBUTIONS[kind];
  let body='';
  for(const d of ticks(top))body+=line(LEFT,y(d*unit.factor),RIGHT,y(d*unit.factor),`stroke="${GRID}"`)+text(LEFT-12,y(d*unit.factor)+4,digits(d),'text-anchor="end"');
  for(const w of ticksWithin(grid[0]!*unit.factor,grid.at(-1)!*unit.factor,8))body+=text(x(w/unit.factor),324,digits(w),'text-anchor="middle"');
  body+=text(LEFT,19,`概率密度 / (${unit.name})⁻¹`)+text(RIGHT,347,`${axisSymbol(kind,axis)} / ${unit.name}`,'text-anchor="end"');
  // The compared time behind, dashed; the last time on top.
  curves.slice().reverse().forEach((values,k)=>{const index=curves.length-1-k;body+=polyline(grid.map(x),values.map(y),MOTION_INKS[index]!,`data-series-id="${kind}-${index}"${index?' stroke-dasharray="6 4"':''}`);});
  return svg('distribution-svg',`沿 ${MOTION_AXES[axis]} 轴的${words}分布。移动指针读取各${words}的概率密度。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}
/** √⟨x²⟩ of a listed distribution (its grid's unit): for positions, the root-mean-square distance from the focus. */
export function rootMeanSquare(d:Distributions,listed:number):number{return Math.sqrt(d.probabilities[listed]!.reduce((sum,p,i)=>sum+p*d.grid[i]!**2,0));}
/** A length (μm) in the unit that fits it. */
export function lengthText(um:number):string{const unit=unitFor(LENGTH_UNITS,Math.abs(um));return `${formatNumber(um*unit.factor,{digits:4})} ${unit.name}`;}

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

/** A readout row's value: the line's value at the sample, with its standard error when it has one
 * (T ± 误差), in the panel's unit; a line of errors with none at the sample says so (a T carried over a
 * sample with fewer than two atoms held). */
export function lineReading(l:Line,i:number,unit:Unit):string{
  const value=formatNumber(l.values[i]!*unit.factor,{digits:4}),error=l.error?.[i],name=unit.name?` ${unit.name}`:'';
  return error==null?`${value}${name}${l.error?'（误差未知）':''}`:`${value} ± ${formatNumber(error*unit.factor,{digits:2})}${name}`;
}
/** The time cursor of a series chart: every line of both panels at the sample, one outside the
 * validated regime marked. */
export function bindSeries(host:HTMLElement,timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit,marks:SeriesMarks={}):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.motion-series-svg');if(!svg)return ()=>{};
  const g=seriesLayout(timesUs,series,timeUnit),card=readoutCard(host,`所选时刻的${series.map(s=>s.name).join('与')}`);
  const unbind=bindCursor(host,svg,timesUs.length,i=>g.x(timesUs[i]!),(i,cursor)=>{
    const xx=g.x(timesUs[i]!);
    cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:String(g.panels[0].top),y2:String(g.panels[1].bottom),stroke:'#a19f9d','stroke-width':'1'}),
      ...g.panels.flatMap(p=>p.series.lines.map(l=>mark('circle',{cx:String(xx),cy:String(p.y(l.values[i]!)),r:'3',fill:l.ink,stroke:'white','stroke-width':'1.5'}))));
    card.innerHTML=`<div class="readout-time">t = ${esc(digits(timesUs[i]!*g.unit.factor))} ${esc(g.unit.name)}${sampleMark(!!marks.reference?.[i])}</div>`+g.panels.flatMap(p=>p.series.lines.map(l=>
      `<div class="readout-row two"><span class="readout-name" style="color:${l.ink}">${esc(l.name)}</span><span>${esc(lineReading(l,i,p.series.unit))}</span></div>`)).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
/** The grid cursor of a distribution chart: each drawn distribution's density there. */
export function bindDistribution(host:HTMLElement,motion:MotionResult,kind:DistributionKind,listed:readonly number[],names:readonly string[],axis:number):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.distribution-svg');if(!svg)return ()=>{};
  const {grid,unit,curves,x,y}=distributionLayout(motion,kind,listed,axis),{words}=DISTRIBUTIONS[kind],symbol=axisSymbol(kind,axis),card=readoutCard(host,`所选${words}的概率密度`);
  const unbind=bindCursor(host,svg,grid.length,i=>x(grid[i]!),(i,cursor)=>{
    const xx=x(grid[i]!);cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:'32',y2:'300',stroke:'#a19f9d','stroke-width':'1'}));
    curves.forEach((values,k)=>cursor.append(mark('circle',{cx:String(xx),cy:String(y(values[i]!)),r:'3',fill:MOTION_INKS[k]!,stroke:'white','stroke-width':'1.5'})));
    card.innerHTML=`<div class="readout-time">${esc(symbol)} = ${esc(digits(grid[i]!*unit.factor))} ${esc(unit.name)}</div>`+curves.map((values,k)=>
      `<div class="readout-row two"><span class="readout-name" style="color:${MOTION_INKS[k]}">${esc(names[k]??'')}</span><span>${esc(formatNumber(values[i]!/unit.factor,{digits:4}))} (${esc(unit.name)})⁻¹</span></div>`).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
