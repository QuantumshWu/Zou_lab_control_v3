// Motion's charts (ui-design.md §6.20–§6.22): two panels against time sharing one time axis -- the
// temperature and the mean velocity, or n̄ and the loss probability -- each with a line per axis the
// atom moves along (one in 1D, the plane's two in 2D, x, y and z in 3D), and a distribution along
// one axis -- of the velocity or of the position, as a density -- at the last time and, when asked,
// at another listed time; a 2D result's positions as maps of the plane. Each reads actual samples
// only. A 1D result's continuation (the samples after its last simulated time) lies on its band; a
// 3D result's samples outside the method's validated regime lie on a light grey band: a reference
// only.
import type {DurationUnit,MotionAxis} from './types.js';
import type {Distributions,MotionResult,PositionMaps} from './solver-types.js';
import {RB87} from './physics.js';
import {timeDisplayUnit} from './duration.js';
import {formatNumber} from './format.js';
import {continuationBand,referenceBands,sampleMark} from './population-chart.js';

/** k_B (J/K), exact in the SI. */
const BOLTZMANN=1.380649e-23;
const SVG='http://www.w3.org/2000/svg';
const esc=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const clamp=(n:number,a:number,b:number)=>Math.min(b,Math.max(a,n));
/** The v3 series palette's first two (population-chart.ts): the top panel's series and the last
 * time, then the bottom panel's and the compared time. */
export const MOTION_INKS=['#517FA6','#B17D4B'] as const;
/** The palette's first three, one per lab axis x, y, z of a 3D result's lines. */
export const AXIS_INKS:Readonly<Record<MotionAxis,string>>={x:'#517FA6',y:'#B17D4B',z:'#568C79'};
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

/** A line of a panel: its name, its values (base units) and its ink; a trajectory method's line
 * its standard error per sample (`band`, base units), drawn as a band of ±1 around it. */
export interface Line {name:string;values:readonly number[];ink:string;band?:readonly number[]}
/** One panel: its name, its axis title, its lines and display unit; its axis runs from 0 to the top
 * of its ticks, or symmetrically about 0 out to `reach` (base units). */
export interface Series {id:string;name:string;title:string;lines:Line[];unit:Unit;reach?:number}
export type SeriesKind='temperature'|'trap';
/** The lines of per-axis values: one in the panel's ink for 1D, one per axis in its ink for 2D and
 * 3D; with `errors`, each with its standard errors. */
function axisLines(motion:MotionResult,values:readonly (readonly number[])[],name:string,ink:string,errors?:readonly (readonly number[])[]):Line[]{
  const band=(k:number)=>errors?{band:errors[k]!}:{};
  return motion.axes.length===1?[{name,values:values[0]!,ink,...band(0)}]:motion.axes.map((axis,k)=>({name:`${name}${subscript(axis)}`,values:values[k]!,ink:AXIS_INKS[axis],...band(k)}));
}
const subscript=(axis:MotionAxis)=>`（${axis} 轴）`;
/** The two panels of a motion result's time chart: T above ⟨v⟩ -- the drift read against the
 * thermal width it moves in, its axis as wide as the larger of the two -- or n̄ (relative to the
 * trap frequency the engine used along each axis) above the loss probability. */
export function motionSeries(motion:MotionResult,kind:SeriesKind):[Series,Series]{
  if(kind==='trap')return [
    {id:'occupation',name:'n̄',title:motion.axes.length===1?`平均振动量子数 n̄（ω/2π = ${digits(motion.trapFrequencyKHz[0]!)} kHz）`:`平均振动量子数 n̄（相对各轴焦点处的阱频 ω/2π：${motion.axes.map((axis,k)=>`${axis} ${digits(motion.trapFrequencyKHz[k]!)}`).join('、')} kHz）`,lines:axisLines(motion,motion.meanOccupation,'n̄',MOTION_INKS[0]),unit:{name:'',factor:1}},
    {id:'loss',name:'损失概率',title:'累计损失概率 / %',lines:[{name:'损失概率',values:motion.lossProbability,ink:motion.axes.length===1?MOTION_INKS[1]:INK}],unit:{name:'%',factor:100}}];
  const reach=Math.max(...motion.meanVelocityMps.flat().map(Math.abs),...motion.temperatureUK.flat().map(thermalVelocity)),velocity=unitFor(VELOCITY_UNITS,reach);
  // A 3D result's T and ⟨v⟩ are each axis's means over one trap period around the sample.
  const period=motion.axis==='3d'?'（各轴一个阱周期内的平均）':'';
  return [{id:'temperature',name:'温度 T',title:`温度 T / μK${period}`,lines:axisLines(motion,motion.temperatureUK,'T',MOTION_INKS[0],motion.temperatureErrorUK),unit:{name:'μK',factor:1}},
    {id:'mean-velocity',name:'⟨v⟩',title:`平均速度 ⟨v⟩ / ${velocity.name}${period}`,lines:axisLines(motion,motion.meanVelocityMps,'⟨v⟩',MOTION_INKS[1]),unit:velocity,reach}];
}
/** A panel: its series, its vertical span and ticks (display units), and where a value goes. */
interface Panel {series:Series;top:number;bottom:number;ticks:number[];y:(value:number)=>number}
function panel(series:Series,top:number,bottom:number):Panel{
  const factor=series.unit.factor;
  if(series.reach!==undefined){
    const reach=ticks(series.reach*factor,2).at(-1)!;
    return {series,top,bottom,ticks:ticks(reach,2).flatMap(v=>v?[-v,v]:[0]),y:v=>(top+bottom)/2-v*factor/reach*(bottom-top)/2};
  }
  const highest=ticks(Math.max(0,...series.lines.flatMap(l=>l.values.map((v,i)=>v+(l.band?.[i]??0))))*factor||1).at(-1)!;
  return {series,top,bottom,ticks:ticks(highest),y:v=>bottom-v*factor/highest*(bottom-top)};
}
interface SeriesLayout {panels:[Panel,Panel];unit:{name:DurationUnit;factor:number};x:(t:number)=>number}
function seriesLayout(timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit):SeriesLayout{
  const span=timesUs.at(-1)!-timesUs[0]!;
  return {panels:[panel(series[0],32,176),panel(series[1],212,300)],unit:timeDisplayUnit(timesUs.at(-1)!,timeUnit),x:t=>LEFT+(t-timesUs[0]!)/span*(RIGHT-LEFT)};
}
const line=(x1:number,y1:number,x2:number,y2:number,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const text=(x:number,y:number,label:string,attrs='')=>`<text x="${x}" y="${y}" font-size="12" fill="${INK}" ${attrs}>${esc(label)}</text>`;
const polyline=(xs:readonly number[],ys:readonly number[],ink:string,attrs='')=>`<polyline points="${xs.map((x,i)=>`${x.toFixed(4)},${ys[i]!.toFixed(4)}`).join(' ')}" fill="none" stroke="${ink}" stroke-width="2" ${attrs}/>`;
const svg=(cls:string,label:string,body:string)=>`<svg class="chart-svg ${cls}" viewBox="0 0 800 355" xmlns="${SVG}" role="img" tabindex="0" aria-label="${esc(label)}" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" style="fill:#323130;outline:none">${body}</svg>`;

/** What a series chart marks on its samples: a 1D result's continuation (the samples after its last
 * simulated time) and a 3D result's samples outside its validated regime (a reference only). */
export interface SeriesMarks {continuedFromUs?:number;reference?:readonly boolean[]}
/** The panels on the result's samples, a line per series line; the continuation's samples on their
 * band across both panels, those outside the validated regime on theirs behind each. */
export function seriesChart(timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit,marks:SeriesMarks={}):string{
  const g=seriesLayout(timesUs,series,timeUnit),xs=timesUs.map(g.x),from=marks.continuedFromUs;
  let body=referenceBands(xs,marks.reference,g.panels.map(p=>[p.top,p.bottom] as const));
  if(from!==undefined)body+=continuationBand(g.x(from),RIGHT,g.panels[0].top,g.panels[1].bottom,`${formatNumber(from*g.unit.factor,{digits:3})} ${g.unit.name}`);
  for(const p of g.panels)for(const v of p.ticks){const y=p.y(v/p.series.unit.factor);body+=line(LEFT,y,RIGHT,y,`stroke="${GRID}"`)+text(LEFT-12,y+4,digits(v),'text-anchor="end"');}
  for(let i=0;i<=4;i++){const t=timesUs[0]!+(timesUs.at(-1)!-timesUs[0]!)*i/4;body+=text(g.x(t),g.panels[1].bottom+24,digits(t*g.unit.factor),'text-anchor="middle"');}
  body+=text(LEFT,19,g.panels[0].series.title)+text(LEFT,g.panels[1].top-8,g.panels[1].series.title)+text(RIGHT,g.panels[1].bottom+47,`t / ${g.unit.name}`,'text-anchor="end"');
  // A line's standard errors as a band of ±1 behind the lines (a quantity at least 0 kept so).
  g.panels.forEach(p=>p.series.lines.forEach(l=>{if(!l.band)return;const low=p.series.reach===undefined?0:-Infinity;
    const upper=l.values.map((v,i)=>`${xs[i]!.toFixed(4)},${p.y(v+l.band![i]!).toFixed(4)}`),lower=l.values.map((v,i)=>`${xs[i]!.toFixed(4)},${p.y(Math.max(low,v-l.band![i]!)).toFixed(4)}`).reverse();
    body+=`<polygon points="${[...upper,...lower].join(' ')}" fill="${l.ink}" fill-opacity="0.16" stroke="none"/>`;}));
  g.panels.forEach(p=>p.series.lines.forEach((l,k)=>{body+=polyline(xs,l.values.map(p.y),l.ink,`data-series-id="${p.series.id}${p.series.lines.length>1?`-${k}`:''}"`);}));
  return svg('motion-series-svg',`${series.map(s=>s.name).join('与')}随时间的变化。移动指针读取采样值，方向键逐点查看。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}

/** The distribution kinds: the velocity (grid in m/s) and the position from the focus (grid in μm). */
export type DistributionKind='momentum'|'position';
const DISTRIBUTIONS:Readonly<Record<DistributionKind,{units:readonly (readonly [string,number])[];symbol:string;words:string}>>={
  momentum:{units:VELOCITY_UNITS,symbol:'v',words:'速度'},position:{units:LENGTH_UNITS,symbol:'x',words:'位置'},
};
/** A result's distributions of a kind along its axis number `axis` (an index into `motion.axes`). */
export function distributions(motion:MotionResult,kind:DistributionKind,axis=0):Distributions{return (kind==='momentum'?motion.momentum:motion.position)[axis]!;}
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
/** The distributions drawn: the last listed time's and, when asked, another listed one's. */
export function distributionChart(motion:MotionResult,kind:DistributionKind,listed:readonly number[],axis=0):string{
  const {grid,unit,curves,top,x,y}=distributionLayout(motion,kind,listed,axis),{words}=DISTRIBUTIONS[kind],symbol=axisSymbol(motion,kind,axis);
  let body='';
  for(const d of ticks(top))body+=line(LEFT,y(d*unit.factor),RIGHT,y(d*unit.factor),`stroke="${GRID}"`)+text(LEFT-12,y(d*unit.factor)+4,digits(d),'text-anchor="end"');
  for(const w of ticksWithin(grid[0]!*unit.factor,grid.at(-1)!*unit.factor,8))body+=text(x(w/unit.factor),324,digits(w),'text-anchor="middle"');
  body+=text(LEFT,19,`概率密度 / (${unit.name})⁻¹`)+text(RIGHT,347,`${symbol} / ${unit.name}`,'text-anchor="end"');
  // The compared time behind, dashed; the last time on top.
  curves.slice().reverse().forEach((values,k)=>{const index=curves.length-1-k;body+=polyline(grid.map(x),values.map(y),MOTION_INKS[index]!,`data-series-id="${kind}-${index}"${index?' stroke-dasharray="6 4"':''}`);});
  return svg('distribution-svg',`沿${motion.axes[axis]} 轴的${words}分布。移动指针读取各${words}的概率密度。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}
/** The variable a distribution chart reads: v or x, with its axis in 3D (vₓ, x …). */
function axisSymbol(motion:MotionResult,kind:DistributionKind,axis:number):string{
  const a=motion.axes[axis]!;return motion.axes.length===1?DISTRIBUTIONS[kind].symbol:kind==='momentum'?`v${subscript(a)}`:a;
}
/** √⟨x²⟩ of a listed distribution (its grid's unit): for positions, the root-mean-square distance from the focus. */
export function rootMeanSquare(d:Distributions,listed:number):number{return Math.sqrt(d.probabilities[listed]!.reduce((sum,p,i)=>sum+p*d.grid[i]!**2,0));}
/** The root-mean-square distance from the focus √⟨r²⟩ (μm) of a listed map. */
export function mapRootMeanSquare(maps:PositionMaps,listed:number):number{
  const n=maps.grid.length;return Math.sqrt(maps.probabilities[listed]!.reduce((sum,p,k)=>sum+p*(maps.grid[k%n]!**2+maps.grid[Math.floor(k/n)]!**2),0));
}
/** A length (μm) in the unit that fits it. */
export function lengthText(um:number):string{const unit=unitFor(LENGTH_UNITS,Math.abs(um));return `${formatNumber(um*unit.factor,{digits:4})} ${unit.name}`;}

/** A plane's position maps are drawn side by side in time order on one density scale (μm⁻²): each
 * square on the grid's span, its first axis across and its second up, the colour bar beside them. */
const MAP=268,MAP_GAP=96,BAR=12;
/** The maps' sequential ramp, one hue light to dark: the levels' population ramp (--population-0,
 * --population-1 in style.css) continued to the series blue and beyond. */
const RAMP:readonly (readonly [number,number,number])[]=[[247,249,251],[134,179,216],[81,127,166],[38,66,94]];
/** The ramp's colour at s ∈ [0, 1], linear between its stops (as an SVG gradient through them). */
function shade(s:number):[number,number,number]{
  const x=clamp(s,0,1)*(RAMP.length-1),k=Math.min(Math.floor(x),RAMP.length-2),f=x-k,a=RAMP[k]!,b=RAMP[k+1]!;
  return [0,1,2].map(c=>Math.round(a[c]!+(b[c]!-a[c]!)*f)) as [number,number,number];
}
/** The maps' layout: the grid's step and cell edges (μm), each map's left edge, the colour bar's,
 * and where a position goes across (from a map's left edge) and up. */
function mapLayout(maps:PositionMaps,count:number){
  const grid=maps.grid,n=grid.length,step=(grid[n-1]!-grid[0]!)/(n-1),low=grid[0]!-step/2,high=grid[n-1]!+step/2;
  const width=count*MAP+(count-1)*MAP_GAP+24+BAR+56,first=Math.max(LEFT,(800-width)/2);
  return {n,step,low,high,top:32,bottom:32+MAP,lefts:Array.from({length:count},(_,k)=>first+k*(MAP+MAP_GAP)),bar:first+count*MAP+(count-1)*MAP_GAP+24,
    x:(left:number,w:number)=>left+(w-low)/(high-low)*MAP,y:(w:number)=>32+MAP-(w-low)/(high-low)*MAP};
}
/** A listed map as densities (μm⁻²), row-major as the map. */
function mapDensity(maps:PositionMaps,listed:number,step:number):number[]{return maps.probabilities[listed]!.map(p=>p/(step*step));}
/** A map as a PNG of one pixel per grid point coloured by its density over `top`, its rows down
 * the second axis from the highest position. */
function mapImage(values:readonly number[],n:number,top:number):string{
  const canvas=document.createElement('canvas');canvas.width=n;canvas.height=n;
  const context=canvas.getContext('2d')!,image=context.createImageData(n,n);
  for(let j=0;j<n;j++)for(let i=0;i<n;i++){const [r,g,b]=shade(values[j*n+i]!/top),k=((n-1-j)*n+i)*4;image.data[k]=r;image.data[k+1]=g;image.data[k+2]=b;image.data[k+3]=255;}
  context.putImageData(image,0,0);return canvas.toDataURL('image/png');
}
/** The maps at the listed times (in time order), each with its title above it. */
export function mapChart(maps:PositionMaps,axes:readonly MotionAxis[],listed:readonly number[],titles:readonly string[]):string{
  const g=mapLayout(maps,listed.length),[a,b]=axes,values=listed.map(at=>mapDensity(maps,at,g.step));
  const top=ticks(Math.max(...values.flat())).at(-1)!;
  let body=`<defs><linearGradient id="map-ramp" x1="0" y1="1" x2="0" y2="0">${RAMP.map((c,k)=>`<stop offset="${k/(RAMP.length-1)}" stop-color="rgb(${c.join(',')})"/>`).join('')}</linearGradient></defs>`;
  g.lefts.forEach((left,k)=>{
    body+=`<image x="${left}" y="${g.top}" width="${MAP}" height="${MAP}" preserveAspectRatio="none" style="image-rendering:pixelated" href="${mapImage(values[k]!,g.n,top)}"/>`;
    body+=`<rect x="${left}" y="${g.top}" width="${MAP}" height="${MAP}" fill="none" stroke="${GRID}"/>`;
    for(const w of ticksWithin(g.low,g.high,6)){
      const x=g.x(left,w),y=g.y(w);
      body+=line(x,g.bottom,x,g.bottom+4,`stroke="${INK}"`)+line(left-4,y,left,y,`stroke="${INK}"`)+text(x,g.bottom+20,digits(w),'text-anchor="middle"')+text(left-8,y+4,digits(w),'text-anchor="end"');
    }
    body+=text(left+MAP/2,19,titles[k]??'','text-anchor="middle"')+text(left+MAP,g.bottom+47,`${a} / μm`,'text-anchor="end"')+text(left-10,19,`${b} / μm`,'text-anchor="end"');
  });
  body+=`<rect x="${g.bar}" y="${g.top}" width="${BAR}" height="${MAP}" fill="url(#map-ramp)" stroke="${GRID}"/>`;
  for(const d of ticks(top)){const y=g.bottom-d/top*MAP;body+=line(g.bar+BAR,y,g.bar+BAR+3,y,`stroke="${INK}"`)+text(g.bar+BAR+6,y+4,digits(d));}
  body+=text(g.bar+BAR+58,19,'概率密度 / μm⁻²','text-anchor="end"');
  return svg('map-svg',`${a}${b} 平面内的位置分布（概率密度）。移动指针读取各位置的概率密度，方向键逐格查看。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
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

/** The time cursor of a series chart: every line of both panels at the sample, a continuation's
 * sample or one outside the validated regime marked. */
export function bindSeries(host:HTMLElement,timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit,marks:SeriesMarks={}):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.motion-series-svg');if(!svg)return ()=>{};
  const g=seriesLayout(timesUs,series,timeUnit),card=readoutCard(host,`所选时刻的${series.map(s=>s.name).join('与')}`);
  const unbind=bindCursor(host,svg,timesUs.length,i=>g.x(timesUs[i]!),(i,cursor)=>{
    const xx=g.x(timesUs[i]!),from=marks.continuedFromUs,marked=sampleMark(from!==undefined&&timesUs[i]!>from,!!marks.reference?.[i]);
    cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:String(g.panels[0].top),y2:String(g.panels[1].bottom),stroke:'#a19f9d','stroke-width':'1'}),
      ...g.panels.flatMap(p=>p.series.lines.map(l=>mark('circle',{cx:String(xx),cy:String(p.y(l.values[i]!)),r:'3',fill:l.ink,stroke:'white','stroke-width':'1.5'}))));
    card.innerHTML=`<div class="readout-time">t = ${esc(digits(timesUs[i]!*g.unit.factor))} ${esc(g.unit.name)}${marked}</div>`+g.panels.flatMap(p=>p.series.lines.map(l=>
      `<div class="readout-row two"><span class="readout-name" style="color:${l.ink}">${esc(l.name)}</span><span>${esc(formatNumber(l.values[i]!*p.series.unit.factor,{digits:4}))}${l.band?` ± ${esc(formatNumber(l.band[i]!*p.series.unit.factor,{digits:2}))}`:''}${p.series.unit.name?` ${esc(p.series.unit.name)}`:''}</span></div>`)).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
/** The grid cursor of a distribution chart: each drawn distribution's density there. */
export function bindDistribution(host:HTMLElement,motion:MotionResult,kind:DistributionKind,listed:readonly number[],names:readonly string[],axis=0):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.distribution-svg');if(!svg)return ()=>{};
  const {grid,unit,curves,x,y}=distributionLayout(motion,kind,listed,axis),{words}=DISTRIBUTIONS[kind],symbol=axisSymbol(motion,kind,axis),card=readoutCard(host,`所选${words}的概率密度`);
  const unbind=bindCursor(host,svg,grid.length,i=>x(grid[i]!),(i,cursor)=>{
    const xx=x(grid[i]!);cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:'32',y2:'300',stroke:'#a19f9d','stroke-width':'1'}));
    curves.forEach((values,k)=>cursor.append(mark('circle',{cx:String(xx),cy:String(y(values[i]!)),r:'3',fill:MOTION_INKS[k]!,stroke:'white','stroke-width':'1.5'})));
    card.innerHTML=`<div class="readout-time">${symbol} = ${esc(digits(grid[i]!*unit.factor))} ${esc(unit.name)}</div>`+curves.map((values,k)=>
      `<div class="readout-row two"><span class="readout-name" style="color:${MOTION_INKS[k]}">${esc(names[k]??'')}</span><span>${esc(formatNumber(values[i]!/unit.factor,{digits:4}))} (${esc(unit.name)})⁻¹</span></div>`).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
/** The cell cursor of the maps: the cell under the pointer (the arrow keys move it) outlined on
 * every map, its position and each map's density there, the maps named by `labels` (markup). */
export function bindMap(host:HTMLElement,maps:PositionMaps,axes:readonly MotionAxis[],listed:readonly number[],labels:readonly string[]):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.map-svg');if(!svg)return ()=>{};
  const g=mapLayout(maps,listed.length),values=listed.map(at=>mapDensity(maps,at,g.step)),[a,b]=axes,size=MAP/g.n;
  const cursor=svg.querySelector<SVGGElement>('.motion-cursor')!,card=readoutCard(host,'所选位置的概率密度'),abort=new AbortController();
  // The cell and the map it is read on (the card stands beside that map; the keys read the last).
  let cell={i:g.n>>1,j:g.n>>1,map:listed.length-1};
  const show=(next:{i:number;j:number;map:number})=>{
    cell={i:clamp(next.i,0,g.n-1),j:clamp(next.j,0,g.n-1),map:next.map};cursor.replaceChildren();cursor.style.display='';
    for(const left of g.lefts)cursor.append(mark('rect',{x:String(left+cell.i*size),y:String(g.top+(g.n-1-cell.j)*size),width:String(size),height:String(size),fill:'none',stroke:'#323130','stroke-width':'1.5'}));
    card.innerHTML=`<div class="readout-time">${esc(`${a} = ${digits(maps.grid[cell.i]!)} μm · ${b} = ${digits(maps.grid[cell.j]!)} μm`)}</div>`+values.map((v,k)=>
      `<div class="readout-row two"><span class="readout-name">${labels[k]??''}</span><span>${esc(`${formatNumber(v[cell.j*g.n+cell.i]!,{digits:4})} μm⁻²`)}</span></div>`).join('');
    placeCard(host,svg,card,g.lefts[cell.map]!+(cell.i+.5)*size);
  };
  const off=()=>{cursor.style.display='none';card.hidden=true;};
  svg.addEventListener('pointermove',event=>{
    const transform=svg.getScreenCTM();if(!transform)return;const p=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
    const map=g.lefts.findIndex(left=>p.x>=left&&p.x<left+MAP);if(map<0||p.y<g.top||p.y>=g.bottom){off();return;}
    show({i:Math.floor((p.x-g.lefts[map]!)/size),j:g.n-1-Math.floor((p.y-g.top)/size),map});
  },{signal:abort.signal});
  svg.addEventListener('pointerleave',off,{signal:abort.signal});svg.addEventListener('blur',off,{signal:abort.signal});
  svg.addEventListener('keydown',event=>{
    if(event.key==='Escape'){off();return;}
    const step=({ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]} as Record<string,[number,number]>)[event.key];if(!step)return;
    event.preventDefault();show({i:cell.i+step[0],j:cell.j+step[1],map:cell.map});
  },{signal:abort.signal});
  return ()=>{abort.abort();card.remove();};
}
