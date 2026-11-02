// Radial motion's charts (ui-design.md §6.20, §6.21). Against time, two panels sharing one time
// axis -- the temperature and the mean velocity, or n̄ and the loss probability -- with a curve per
// axis of the motion: one along a line (1D), one per axis of a plane (2D, coloured by axis, the
// temperature's standard error a band). At the last time and, when asked, another listed time: the
// distribution of the velocity along each axis, or of the position along the line, as a density;
// in a plane the position's map, as a density. Each reads actual samples only.
import type {DurationUnit} from './types.js';
import type {Distributions,MotionResult,PlaneMotionResult,PositionMaps} from './solver-types.js';
import {isPlaneMotion} from './solver-types.js';
import {RB87} from './physics.js';
import {timeDisplayUnit} from './duration.js';
import {formatNumber} from './format.js';

/** k_B (J/K), exact in the SI. */
const BOLTZMANN=1.380649e-23;
const SVG='http://www.w3.org/2000/svg';
const esc=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const clamp=(n:number,a:number,b:number)=>Math.min(b,Math.max(a,n));
/** The v3 series palette's first three (population-chart.ts). Along a line: the top panel's series
 * and the last time, then the bottom panel's and the compared time. In a plane: its first axis, its
 * second, then the loss beside their n̄. */
export const MOTION_INKS=['#517FA6','#B17D4B','#568C79'] as const;
const GRID='#e1dfdd',INK='#6a6866',CURSOR='#a19f9d',LEFT=64,RIGHT=778;

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

/** A curve's name: its symbol and, in a plane, the axis it is along (Tₓ). */
export interface Name {symbol:string;axis?:string}
/** A name in a readout or a legend: the axis as a subscript. */
export function nameMarkup(name:Name):string{return esc(name.symbol)+(name.axis?`<sub>${esc(name.axis)}</sub>`:'');}
/** A name in words, for an accessible label. */
function nameWords(name:Name):string{return name.axis?`沿 ${name.axis} 轴的 ${name.symbol}`:name.symbol;}

/** A panel's curve: its name, ink and values (base units) and, for a temperature in a plane, the
 * standard error of each value (drawn as a band). */
export interface Curve {name:Name;ink:string;values:readonly number[];errors?:readonly number[]}
/** One panel: its id, axis title, curves and display unit; its axis runs from 0 to the top of its
 * ticks, or symmetrically about 0 out to `reach` (base units). */
export interface Series {id:string;title:string;curves:readonly Curve[];unit:Unit;reach?:number}
export type SeriesKind='temperature'|'trap';
/** The two panels of a motion result's time chart: T above ⟨v⟩ -- the drift read against the
 * thermal width it moves in, its axis as wide as the larger of the two -- or n̄ (relative to the
 * trap frequency the engine used) above the loss probability; in a plane, a curve per axis. */
export function motionSeries(motion:MotionResult,kind:SeriesKind):[Series,Series]{
  if(isPlaneMotion(motion)){
    const axes=motion.axes,curves=(symbol:string,values:readonly (readonly number[])[],errors?:readonly (readonly number[])[]):Curve[]=>
      axes.map((axis,d)=>({name:{symbol,axis},ink:MOTION_INKS[d]!,values:values[d]!,...(errors?{errors:errors[d]!}:{})}));
    if(kind==='trap')return [
      {id:'occupation',title:`平均振动量子数 n̄（ω/2π：${axes.map((axis,d)=>`${axis} 轴 ${digits(motion.trapFrequencyKHz[d])} kHz`).join('，')}）`,curves:curves('n̄',motion.meanOccupation),unit:{name:'',factor:1}},
      {id:'loss',title:'累计损失概率 / %',curves:[{name:{symbol:'损失概率'},ink:MOTION_INKS[2],values:motion.lossProbability}],unit:{name:'%',factor:100}}];
    const reach=Math.max(...motion.meanVelocityMps.flat().map(Math.abs),...motion.temperatureUK.flat().map(thermalVelocity)),velocity=unitFor(VELOCITY_UNITS,reach);
    return [{id:'temperature',title:'温度 T / μK',curves:curves('T',motion.temperatureUK,motion.temperatureErrorUK),unit:{name:'μK',factor:1}},
      {id:'mean-velocity',title:`平均速度 ⟨v⟩ / ${velocity.name}`,curves:curves('⟨v⟩',motion.meanVelocityMps),unit:velocity,reach}];
  }
  if(kind==='trap')return [
    {id:'occupation',title:`平均振动量子数 n̄（ω/2π = ${digits(motion.trapFrequencyKHz)} kHz）`,curves:[{name:{symbol:'n̄'},ink:MOTION_INKS[0],values:motion.meanOccupation}],unit:{name:'',factor:1}},
    {id:'loss',title:'累计损失概率 / %',curves:[{name:{symbol:'损失概率'},ink:MOTION_INKS[1],values:motion.lossProbability}],unit:{name:'%',factor:100}}];
  const reach=Math.max(...motion.meanVelocityMps.map(Math.abs),...motion.temperatureUK.map(thermalVelocity)),velocity=unitFor(VELOCITY_UNITS,reach);
  return [{id:'temperature',title:'温度 T / μK',curves:[{name:{symbol:'温度 T'},ink:MOTION_INKS[0],values:motion.temperatureUK}],unit:{name:'μK',factor:1}},
    {id:'mean-velocity',title:`平均速度 ⟨v⟩ / ${velocity.name}`,curves:[{name:{symbol:'⟨v⟩'},ink:MOTION_INKS[1],values:motion.meanVelocityMps}],unit:velocity,reach}];
}
/** A panel: its series, its vertical span and ticks (display units), and where a value goes. */
interface Panel {series:Series;top:number;bottom:number;ticks:number[];y:(value:number)=>number}
function panel(series:Series,top:number,bottom:number):Panel{
  const factor=series.unit.factor;
  if(series.reach!==undefined){
    const reach=ticks(series.reach*factor,2).at(-1)!;
    return {series,top,bottom,ticks:ticks(reach,2).flatMap(v=>v?[-v,v]:[0]),y:v=>(top+bottom)/2-v*factor/reach*(bottom-top)/2};
  }
  const peak=Math.max(0,...series.curves.flatMap(c=>c.values.map((v,i)=>v+(c.errors?.[i]??0))));
  const highest=ticks(peak*factor||1).at(-1)!;
  return {series,top,bottom,ticks:ticks(highest),y:v=>bottom-v*factor/highest*(bottom-top)};
}
interface SeriesLayout {panels:[Panel,Panel];unit:{name:DurationUnit;factor:number};x:(t:number)=>number}
function seriesLayout(timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit):SeriesLayout{
  const span=timesUs.at(-1)!-timesUs[0]!;
  return {panels:[panel(series[0],32,176),panel(series[1],212,300)],unit:timeDisplayUnit(timesUs.at(-1)!,timeUnit),x:t=>LEFT+(t-timesUs[0]!)/span*(RIGHT-LEFT)};
}
const line=(x1:number,y1:number,x2:number,y2:number,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const text=(x:number,y:number,label:string,attrs='')=>`<text x="${x}" y="${y}" font-size="12" fill="${INK}" ${attrs}>${esc(label)}</text>`;
const points=(xs:readonly number[],ys:readonly number[])=>xs.map((x,i)=>`${x.toFixed(4)},${ys[i]!.toFixed(4)}`).join(' ');
const polyline=(xs:readonly number[],ys:readonly number[],ink:string,attrs='')=>`<polyline points="${points(xs,ys)}" fill="none" stroke="${ink}" stroke-width="2" ${attrs}/>`;
/** A band between an upper and a lower edge, in the curve's ink. */
const band=(xs:readonly number[],upper:readonly number[],lower:readonly number[],ink:string)=>
  `<polygon points="${points([...xs,...xs.slice().reverse()],[...upper,...lower.slice().reverse()])}" fill="${ink}" fill-opacity="0.18" stroke="none"/>`;
const svg=(cls:string,label:string,body:string)=>`<svg class="chart-svg ${cls}" viewBox="0 0 800 355" xmlns="${SVG}" role="img" tabindex="0" aria-label="${esc(label)}" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" style="fill:#323130;outline:none">${body}</svg>`;

/** The two panels on the result's samples: each curve, behind them the bands of the standard errors. */
export function seriesChart(timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit):string{
  const g=seriesLayout(timesUs,series,timeUnit),xs=timesUs.map(g.x);
  let body='';
  for(const p of g.panels)for(const v of p.ticks){const y=p.y(v/p.series.unit.factor);body+=line(LEFT,y,RIGHT,y,`stroke="${GRID}"`)+text(LEFT-12,y+4,digits(v),'text-anchor="end"');}
  for(let i=0;i<=4;i++){const t=timesUs[0]!+(timesUs.at(-1)!-timesUs[0]!)*i/4;body+=text(g.x(t),g.panels[1].bottom+24,digits(t*g.unit.factor),'text-anchor="middle"');}
  body+=text(LEFT,19,g.panels[0].series.title)+text(LEFT,g.panels[1].top-8,g.panels[1].series.title)+text(RIGHT,g.panels[1].bottom+47,`t / ${g.unit.name}`,'text-anchor="end"');
  // A temperature is never below 0: its band stops there.
  for(const p of g.panels)for(const c of p.series.curves)if(c.errors)body+=band(xs,c.values.map((v,i)=>p.y(v+c.errors![i]!)),c.values.map((v,i)=>p.y(Math.max(0,v-c.errors![i]!))),c.ink);
  for(const p of g.panels)for(const c of p.series.curves)body+=polyline(xs,c.values.map(p.y),c.ink,`data-series-id="${p.series.id}${c.name.axis?`-${c.name.axis}`:''}"`);
  const names=series.flatMap(s=>s.curves.map(c=>nameWords(c.name)));
  return svg('motion-series-svg',`${names.join('、')}随时间的变化。移动指针读取采样值，方向键逐点查看。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
}

/** The distribution kinds: the velocity (grid in m/s) and the position from the focus (grid in μm). */
export type DistributionKind='momentum'|'position';
const DISTRIBUTIONS:Readonly<Record<DistributionKind,{units:readonly (readonly [string,number])[];symbol:string;words:string}>>={
  momentum:{units:VELOCITY_UNITS,symbol:'v',words:'速度'},position:{units:LENGTH_UNITS,symbol:'x',words:'位置'},
};
/** The sample indices a kind's distributions are listed at. */
export function listedSamples(motion:MotionResult,kind:DistributionKind):readonly number[]{
  return isPlaneMotion(motion)?(kind==='momentum'?motion.momentum[0]:motion.position).samples:motion[kind].samples;
}
/** A grid point's share of the axis: half the way to each neighbour (the whole way at an end). */
function cellWidths(v:readonly number[]):number[]{return v.map((_,i)=>((v[Math.min(i+1,v.length-1)]!-v[Math.max(i-1,0)]!)/(i===0||i===v.length-1?1:2)));}
/** A listed distribution as a density on its grid (per base unit). */
export function density(d:Distributions,listed:number):number[]{const widths=cellWidths(d.grid);return d.probabilities[listed]!.map((p,i)=>p/widths[i]!);}
/** A drawn distribution along an axis: its name, the listed time it is at (an index into the
 * distributions' samples), its ink and dash -- the compared time's dashed -- and its density on the
 * grid (per base unit). */
export interface DistributionCurve {name:Name;listed:number;ink:string;dashed:boolean;density:number[]}
/** The distributions along an axis drawn at the listed times -- the last first, then the compared:
 * along a line one per time, its ink the time's; the velocity in a plane one per axis and time, its
 * ink the axis' (a plane's positions are maps: mapChart). Their one grid: a plane's two velocity
 * marginals are on one. */
export function distributionCurves(motion:MotionResult,kind:DistributionKind,listed:readonly number[]):{grid:readonly number[];curves:DistributionCurve[]}{
  const symbol=DISTRIBUTIONS[kind].symbol;
  if(isPlaneMotion(motion)){
    if(kind!=='momentum')throw new Error('A plane\'s positions are maps');
    return {grid:motion.momentum[0].grid,curves:listed.flatMap((at,k)=>motion.axes.map((axis,d)=>
      ({name:{symbol,axis},listed:at,ink:MOTION_INKS[d]!,dashed:k>0,density:density(motion.momentum[d]!,at)})))};
  }
  const d=motion[kind];
  return {grid:d.grid,curves:listed.map((at,k)=>({name:{symbol},listed:at,ink:MOTION_INKS[k]!,dashed:k>0,density:density(d,at)}))};
}
/** The distribution chart's axes: the grid's whole span; densities from 0 to the top of their ticks. */
function distributionLayout(grid:readonly number[],curves:readonly DistributionCurve[],kind:DistributionKind){
  const unit=unitFor(DISTRIBUTIONS[kind].units,Math.max(Math.abs(grid[0]!),Math.abs(grid.at(-1)!)));
  const top=ticks(Math.max(...curves.flatMap(c=>c.density))/unit.factor).at(-1)!;
  return {unit,top,x:(w:number)=>LEFT+(w-grid[0]!)/(grid.at(-1)!-grid[0]!)*(RIGHT-LEFT),y:(value:number)=>300-value/unit.factor/top*(300-32)};
}
/** The distributions drawn, the compared time's (dashed) behind the last time's. */
export function distributionChart(grid:readonly number[],curves:readonly DistributionCurve[],kind:DistributionKind):string{
  const {unit,top,x,y}=distributionLayout(grid,curves,kind),{symbol,words}=DISTRIBUTIONS[kind];
  let body='';
  for(const d of ticks(top))body+=line(LEFT,y(d*unit.factor),RIGHT,y(d*unit.factor),`stroke="${GRID}"`)+text(LEFT-12,y(d*unit.factor)+4,digits(d),'text-anchor="end"');
  for(const w of ticksWithin(grid[0]!*unit.factor,grid.at(-1)!*unit.factor,8))body+=text(x(w/unit.factor),324,digits(w),'text-anchor="middle"');
  body+=text(LEFT,19,`概率密度 / (${unit.name})⁻¹`)+text(RIGHT,347,`${symbol} / ${unit.name}`,'text-anchor="end"');
  for(const dashed of [true,false])curves.forEach((c,k)=>{if(c.dashed===dashed)body+=polyline(grid.map(x),c.density.map(y),c.ink,`data-series-id="${kind}-${k}"${dashed?' stroke-dasharray="6 4"':''}`);});
  const along=curves.some(c=>c.name.axis)?'沿各轴':'沿运动轴';
  return svg('distribution-svg',`${along}的${words}分布。移动指针读取各${words}的概率密度。`,body+'<g class="motion-cursor" pointer-events="none" style="display:none"></g>');
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
export function mapChart(motion:PlaneMotionResult,listed:readonly number[],titles:readonly string[]):string{
  const maps=motion.position,g=mapLayout(maps,listed.length),[a,b]=motion.axes,values=listed.map(at=>mapDensity(maps,at,g.step));
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
/** The pointer's position in the chart's coordinates, when the chart has them. */
function chartPoint(svg:SVGSVGElement,event:PointerEvent):DOMPoint|undefined{
  const transform=svg.getScreenCTM();return transform?new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse()):undefined;
}
/** A cursor that follows the pointer (and the arrow keys) over `count` points at plot x `at(i)`. */
function bindCursor(host:HTMLElement,svg:SVGSVGElement,count:number,at:(i:number)=>number,show:(i:number,cursor:SVGGElement)=>void,hide:()=>void):()=>void{
  const cursor=svg.querySelector<SVGGElement>('.motion-cursor')!,abort=new AbortController();let index=count-1;
  const go=(next:number)=>{index=clamp(next,0,count-1);cursor.replaceChildren();cursor.style.display='';show(index,cursor);};
  const off=()=>{cursor.style.display='none';hide();};
  svg.addEventListener('pointermove',event=>{
    const p=chartPoint(svg,event);if(!p)return;
    if(p.x<LEFT||p.x>RIGHT||p.y<32||p.y>300){off();return;}
    let best=0;for(let i=1;i<count;i++)if(Math.abs(at(i)-p.x)<Math.abs(at(best)-p.x))best=i;go(best);
  },{signal:abort.signal});
  svg.addEventListener('pointerleave',off,{signal:abort.signal});svg.addEventListener('blur',off,{signal:abort.signal});
  svg.addEventListener('keydown',event=>{const next={ArrowLeft:index-1,ArrowRight:index+1,Home:0,End:count-1}[event.key];if(event.key==='Escape'){off();return;}if(next===undefined)return;event.preventDefault();go(next);},{signal:abort.signal});
  return ()=>abort.abort();
}
const mark=(name:string,attrs:Record<string,string>)=>{const node=document.createElementNS(SVG,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node;};
/** A readout row: a name (markup) in its ink and a value (markup). */
const row=(ink:string,name:string,value:string)=>`<div class="readout-row two"><span class="readout-name" style="color:${ink}">${name}</span><span>${value}</span></div>`;

/** The time cursor of a series chart: every curve at the sample, a temperature with its standard error. */
export function bindSeries(host:HTMLElement,timesUs:readonly number[],series:readonly [Series,Series],timeUnit?:DurationUnit):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.motion-series-svg');if(!svg)return ()=>{};
  const g=seriesLayout(timesUs,series,timeUnit),card=readoutCard(host,`所选时刻的${series.flatMap(s=>s.curves.map(c=>nameWords(c.name))).join('与')}`);
  const unbind=bindCursor(host,svg,timesUs.length,i=>g.x(timesUs[i]!),(i,cursor)=>{
    const xx=g.x(timesUs[i]!);
    cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:String(g.panels[0].top),y2:String(g.panels[1].bottom),stroke:CURSOR,'stroke-width':'1'}),
      ...g.panels.flatMap(p=>p.series.curves.map(c=>mark('circle',{cx:String(xx),cy:String(p.y(c.values[i]!)),r:'3',fill:c.ink,stroke:'white','stroke-width':'1.5'}))));
    card.innerHTML=`<div class="readout-time">t = ${esc(digits(timesUs[i]!*g.unit.factor))} ${esc(g.unit.name)}</div>`+g.panels.flatMap(p=>p.series.curves.map(c=>{
      const unit=p.series.unit,error=c.errors?` ± ${formatNumber(c.errors[i]!*unit.factor,{digits:2})}`:'';
      return row(c.ink,nameMarkup(c.name),esc(`${formatNumber(c.values[i]!*unit.factor,{digits:4})}${error}${unit.name?` ${unit.name}`:''}`));
    })).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
/** The grid cursor of a distribution chart: each drawn distribution's density there, named by `labels` (markup). */
export function bindDistribution(host:HTMLElement,grid:readonly number[],curves:readonly DistributionCurve[],kind:DistributionKind,labels:readonly string[]):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.distribution-svg');if(!svg)return ()=>{};
  const {unit,x,y}=distributionLayout(grid,curves,kind),{symbol,words}=DISTRIBUTIONS[kind],card=readoutCard(host,`所选${words}的概率密度`);
  const unbind=bindCursor(host,svg,grid.length,i=>x(grid[i]!),(i,cursor)=>{
    const xx=x(grid[i]!);cursor.append(mark('line',{x1:String(xx),x2:String(xx),y1:'32',y2:'300',stroke:CURSOR,'stroke-width':'1'}));
    curves.forEach(c=>cursor.append(mark('circle',{cx:String(xx),cy:String(y(c.density[i]!)),r:'3',fill:c.ink,stroke:'white','stroke-width':'1.5'})));
    card.innerHTML=`<div class="readout-time">${symbol} = ${esc(digits(grid[i]!*unit.factor))} ${esc(unit.name)}</div>`+curves.map((c,k)=>
      row(c.ink,labels[k]??'',esc(`${formatNumber(c.density[i]!/unit.factor,{digits:4})} (${unit.name})⁻¹`))).join('');
    placeCard(host,svg,card,xx);
  },()=>{card.hidden=true;});
  return ()=>{unbind();card.remove();};
}
/** The cell cursor of the maps: the cell under the pointer (the arrow keys move it) outlined on
 * every map, its position and each map's density there, the maps named by `labels` (markup). */
export function bindMap(host:HTMLElement,motion:PlaneMotionResult,listed:readonly number[],labels:readonly string[]):()=>void{
  const svg=host.querySelector<SVGSVGElement>('svg.map-svg');if(!svg)return ()=>{};
  const maps=motion.position,g=mapLayout(maps,listed.length),values=listed.map(at=>mapDensity(maps,at,g.step)),[a,b]=motion.axes,size=MAP/g.n;
  const cursor=svg.querySelector<SVGGElement>('.motion-cursor')!,card=readoutCard(host,'所选位置的概率密度'),abort=new AbortController();
  // The cell and the map it is read on (the card stands beside that map; the keys read the last).
  let cell={i:g.n>>1,j:g.n>>1,map:listed.length-1};
  const show=(next:{i:number;j:number;map:number})=>{
    cell={i:clamp(next.i,0,g.n-1),j:clamp(next.j,0,g.n-1),map:next.map};cursor.replaceChildren();cursor.style.display='';
    for(const left of g.lefts)cursor.append(mark('rect',{x:String(left+cell.i*size),y:String(g.top+(g.n-1-cell.j)*size),width:String(size),height:String(size),fill:'none',stroke:'#323130','stroke-width':'1.5'}));
    card.innerHTML=`<div class="readout-time">${esc(`${a} = ${digits(maps.grid[cell.i]!)} μm · ${b} = ${digits(maps.grid[cell.j]!)} μm`)}</div>`+values.map((v,k)=>
      row('inherit',labels[k]??'',esc(`${formatNumber(v[cell.j*g.n+cell.i]!,{digits:4})} μm⁻²`))).join('');
    placeCard(host,svg,card,g.lefts[cell.map]!+(cell.i+.5)*size);
  };
  const off=()=>{cursor.style.display='none';card.hidden=true;};
  svg.addEventListener('pointermove',event=>{
    const p=chartPoint(svg,event);if(!p)return;
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
