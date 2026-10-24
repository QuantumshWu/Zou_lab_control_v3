import type { Level, SpectralLine } from './types.js';
import { absoluteEnergyMHz } from './physics.js';

/** The two displayed manifolds have separate *linear* energy axes. The large
 * optical interval is omitted; there is no per-F Zeeman exaggeration. */
export function levelLayout(ground:Level[], excited:Level[], line:SpectralLine) {
  const range=(levels:Level[])=>({min:Math.min(...levels.map(absoluteEnergyMHz)),max:Math.max(...levels.map(absoluteEnergyMHz))});
  const g=range(ground),e=range(excited);
  const closestSameM=(levels:Level[])=>{
    let gap=Infinity;
    for(const a of levels)for(const b of levels)if(a.id!==b.id&&a.m===b.m)gap=Math.min(gap,Math.abs(absoluteEnergyMHz(a)-absoluteEnergyMHz(b)));
    return gap;
  };
  // A stronger field may expand the spectrum. Grow its single linear axis
  // enough that different F states in the same mF column remain selectable.
  const axis=(levels:Level[],span:number,minimum:number,separation:number,maximum:number)=>{
    const required=separation*span/closestSameM(levels),schematic=!Number.isFinite(required)||required>maximum;
    return {height:schematic?maximum:Math.max(minimum,required),schematic};
  };
  const excitedAxis=axis(excited,e.max-e.min,line==='D2'?208:124,35,340),groundAxis=axis(ground,g.max-g.min,78,58,220);
  const top=76,excitedBottom=top+excitedAxis.height;
  const groundTop=excitedBottom+148,groundBottom=groundTop+groundAxis.height;
  const map=(energy:number,r:{min:number;max:number},a:number,b:number)=>b-(energy-r.min)/Math.max(1e-9,r.max-r.min)*(b-a);
  const linearY=(level:Level)=>level.manifold==='ground'?map(absoluteEnergyMHz(level),g,groundTop,groundBottom):map(absoluteEnergyMHz(level),e,top,excitedBottom);
  const expanded=new Map<string,number>();
  const expand=(levels:Level[],a:number,b:number,separation:number)=>{
    for(const m of new Set(levels.map(level=>level.m))){
      const column=levels.filter(level=>level.m===m),positions=packLabels(column.map(linearY),a,b,separation);
      column.forEach((level,i)=>expanded.set(level.id,positions[i]!));
    }
  };
  if(excitedAxis.schematic)expand(excited,top,excitedBottom,35);
  if(groundAxis.schematic)expand(ground,groundTop,groundBottom,58);
  const y=(level:Level)=>expanded.get(level.id)??linearY(level);
  return {
    top,excitedBottom,groundTop,groundBottom,bandBottom:groundBottom+26,
    excitedSchematic:excitedAxis.schematic,groundSchematic:groundAxis.schematic,
    schematic:excitedAxis.schematic||groundAxis.schematic,
    x:(m:number)=>419+m*84,y,groundRange:g,excitedRange:e,
    photon:(energy:number,target:'excited'|'ground'='excited')=> {
      // Values outside the local optical viewport are explicitly truncated in
      // the axis break or at the upper edge, never painted as a nearby state.
      if(target==='ground'){
        const yy=map(energy,g,groundTop,groundBottom);
        if(yy>=groundTop-16&&yy<=groundBottom+16)return {y:yy,placement:'inside' as const,edgeMHz:0};
        if(energy<g.min)return {y:groundBottom+24,placement:'below' as const,edgeMHz:energy-g.min};
        if(energy>g.max)return {y:groundTop-32,placement:'above' as const,edgeMHz:energy-g.max};
        return {y:yy,placement:'inside' as const,edgeMHz:0};
      }
      // The visible viewport includes the space occupied by the end-state
      // cards. Continue its linear axis through that padding; this is display
      // geometry, not a frequency tolerance or a changed atomic energy.
      const yy=map(energy,e,top,excitedBottom);
      if(yy>=top-16&&yy<=excitedBottom+16)return {y:yy,placement:'inside' as const,edgeMHz:0};
      if(energy<e.min)return {y:excitedBottom+48,placement:'below' as const,edgeMHz:energy-e.min};
      if(energy>e.max)return {y:42,placement:'above' as const,edgeMHz:energy-e.max};
      return {y:yy,placement:'inside' as const,edgeMHz:0};
    },
  };
}

/** Label positions may move, but the laser's physical endpoint never does. */
export function packLabels(positions:number[],top:number,bottom:number,gap=35):number[] {
  const order=positions.map((y,i)=>({y,i})).sort((a,b)=>a.y-b.y);
  const actualGap=Math.min(gap,(bottom-top)/Math.max(1,order.length-1));
  const packed:number[]=[];
  order.forEach((item,i)=>packed.push(Math.max(top,item.y,i?packed[i-1]!+actualGap:top)));
  for(let i=packed.length-1;i>=0;i--)packed[i]=Math.min(packed[i]!,i===packed.length-1?bottom:packed[i+1]!-actualGap);
  const result:number[]=[];order.forEach((item,i)=>result[item.i]=packed[i]!);return result;
}
