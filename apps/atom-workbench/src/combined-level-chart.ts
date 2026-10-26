import type {Analysis,Laser,Level,Scene} from './types.js';
import type {LevelChartOptions} from './charts.js';
import {escapeHtml,populationAppearance} from './charts.js';
import {absoluteEnergyMHz,laserFrequencyMHz} from './physics.js';
import {referenceEnergyMHz} from './reference-energy.js';
import {laserEmits,laserReferenceStateId,levelReadoutMHz,referenceReadoutMHz,transitionEndpoints} from './transition-view.js';
import {packLabels} from './level-layout.js';
import {populationValue} from './population-chart.js';
import {fSymbol,idLabel,stateLabel,withUnit} from './format.js';
import {IDENTITY_SLOTS,laserSlot} from './identity.js';
const term=(m:string)=>m==='ground'?'5S₁/₂':m==='D1'?'5P₁/₂':'5P₃/₂';
const fmt=(v:number|undefined|null)=>{if(v==null||!Number.isFinite(v))return '—';const a=Math.abs(v),s=a>=1e6?1e6:a>=1000?1000:a>=1||a===0?1:a>=.001?.001:1e-6,u=s===1e6?'THz':s===1000?'GHz':s===1?'MHz':s===.001?'kHz':'Hz';return `${v>0?'+':v<0?'−':''}${Number((a/s).toPrecision(s===1000?5:4))} ${u}`;};
const short=(s:string,width=116)=>{let out='',size=0;for(const c of s){const next=/[\u2e80-\u9fff]/.test(c)?13:7;if(size+next>width-10)return out+'…';out+=c;size+=next;}return out;};
const pct=(p:number|undefined)=>p===undefined?'—':populationValue(p);
// Compact Fluent marker: ordinary numerals, the shared selected face and text
// colors. Its dedicated gutter never recolors or overlays a population readout.
const selectionNumber=(x:number,y:number,index:number,both=false)=>{const label=both?'1·2':String(index+1);return `<g class="state-selection-number" data-badge-number="${label}" data-badge-x="${x-51}" data-badge-y="${y-14}" aria-label="所选态 ${label}"><circle cx="${x-43}" cy="${y}" r="7" fill="var(--orange-dark, #A65F00)"/><text x="${x-43}" y="${y+3.5}" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">${label}</text></g>`;};
export function combinedLevelsChart(scene:Scene,a:Analysis,laser:Laser|undefined,o:LevelChartOptions):string{
 const all=[...a.ground,...a.D1,...a.D2],byId=new Map(all.map(s=>[s.id,s])),refs=scene.references??[],W=990,x=(m:number)=>480+m*90;
 const refE=new Map(refs.map(r=>[r.id,referenceEnergyMHz(r,all)!]));
 const energyManifold=(energy:number)=>energy<1e8?'ground':Math.abs(energy-absoluteEnergyMHz(a.D1[0]!))<Math.abs(energy-absoluteEnergyMHz(a.D2[0]!))?'D1':'D2';
 const count=(name:string)=>refs.filter(r=>energyManifold(refE.get(r.id)!)===name).length;
 const d2Span=Math.max(156,(count('D2')-1)*60),d1Base=88+d2Span+106,d1Span=Math.max(58,(count('D1')-1)*60),groundBase=d1Base+d1Span+140,groundSpan=Math.max(70,(count('ground')-1)*60);
 let H=Math.max(685,groundBase+groundSpan+67);
 const groups=[{name:'D2',levels:a.D2,base:88,Fs:[3,2,1,0],step:52},{name:'D1',levels:a.D1,base:d1Base,Fs:[2,1],step:58},{name:'ground',levels:a.ground,base:groundBase,Fs:[2,1],step:70}];
 const pos=new Map<string,number>();for(const group of groups)for(const F of group.Fs){const levels=group.levels.filter(s=>s.F===F),mean=levels.reduce((v,s)=>v+s.energyMHz,0)/levels.length,extent=Math.max(1,...levels.map(s=>Math.abs(s.energyMHz-mean)));for(const s of levels)pos.set(s.id,group.base+group.Fs.indexOf(F)*group.step-8*(s.energyMHz-mean)/extent);}
 const mapEnergy=(energy:number)=>{const branch=energy<1e8?groups[2]!:Math.abs(energy-absoluteEnergyMHz(a.D1[0]!))<Math.abs(energy-absoluteEnergyMHz(a.D2[0]!))?groups[1]!:groups[0]!;const sorted=branch.levels.slice().sort((p,q)=>absoluteEnergyMHz(p)-absoluteEnergyMHz(q));let low=sorted[0]!,high=sorted.at(-1)!;for(let i=1;i<sorted.length;i++)if(absoluteEnergyMHz(sorted[i]!)>=energy){low=sorted[i-1]!;high=sorted[i]!;break;}const d=absoluteEnergyMHz(high)-absoluteEnergyMHz(low),value=d?pos.get(low.id)!+(pos.get(high.id)!-pos.get(low.id)!)*(energy-absoluteEnergyMHz(low))/d:pos.get(low.id)!;return Math.max(branch.base-25,Math.min(branch.base+(branch.Fs.length-1)*branch.step+25,value));};
 const refY=new Map<string,number>();
 for(const group of groups){const members=refs.filter(r=>energyManifold(refE.get(r.id)!)===group.name),span=group.name==='D2'?d2Span:group.name==='D1'?d1Span:groundSpan;
  const positions=packLabels(members.map(r=>mapEnergy(refE.get(r.id)!)),group.base-13,group.base+span+13,60);members.forEach((r,i)=>refY.set(r.id,positions[i]!));}
 // Include each actual shifted state's complete card and a visible lower
 // margin, instead of ending the mF identity bands at the zero-field layout.
 const bandBottom=Math.max(...pos.values(),...refY.values())+26+12;
 H=Math.max(H,bandBottom+24);

 const energy=(id:string)=>{const s=byId.get(id);return s?absoluteEnergyMHz(s):refE.get(id);},point=(id:string)=>{const s=byId.get(id);return s?{x:x(s.m),y:pos.get(id)!}:{x:940,y:refY.get(id)??H/2};};
 const sourceId=laserReferenceStateId(scene,laser),source=byId.get(sourceId),sourceEnergy=energy(sourceId),emission=laser?laserEmits(scene,laser,all):false,virtual=laser&&sourceEnergy!==undefined?sourceEnergy+(emission?-1:1)*laserFrequencyMHz(laser):undefined;
 const photonSide={sourceEnergyMHz:sourceEnergy,sourceManifold:source?.manifold??(sourceEnergy===undefined?undefined:energyManifold(sourceEnergy)),emission,photonEnergyMHz:virtual};
 // With a laser, how far each state lies above its anchor (levelReadoutMHz); a reference-level origin reads by the same rule.
 const delta=(s:Level)=>{if(!laser)return s.totalShiftMHz;if(source)return levelReadoutMHz(source,s,laser);return referenceReadoutMHz(photonSide,absoluteEnergyMHz(s),s.manifold);};
 const t=(xx:number,yy:number,text:string,attrs='')=>`<text x="${xx}" y="${yy}" ${attrs}>${escapeHtml(text)}</text>`,ln=(x1:number,y1:number,x2:number,y2:number,attrs='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
 let svg=`<defs>${Array.from({length:IDENTITY_SLOTS},(_,i)=>`<marker id="combined-photon-${i}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0L5 2.5L0 5" style="fill:var(--identity-${i})"/></marker>`).join('')}</defs>`;
 const identity=(id:string)=>`var(--identity-${laserSlot(scene,id)})`;
 const bands=['#f4eddc','#dce9dd','#f0dccd','#eaddec'];
 for(let m=-3;m<=3;m++)svg+=`<g class="mf-band" data-mf="${m}"><rect x="${x(m)-43}" y="48" width="86" height="${bandBottom-48}" fill="${bands[Math.abs(m)]}" opacity=".85"/>${t(x(m),33,m>0?'+'+m:String(m),'text-anchor="middle" font-size="13"')}</g>`;
 svg+=t(145,33,'m','font-size="12"')+t(810,33,laser?'Δ':'总能移','font-size="12" style="fill:var(--muted-ink)"');
 const prefix=svg;svg='';
 for(const group of groups){const dim=group.name!=='ground'&&group.name!==o.line?.toString();svg+=`<g class="manifold" data-manifold="${group.name}" opacity="${dim?.38:1}">`+t(16,group.base-28,term(group.name),'font-size="17" font-weight="600"');for(const F of group.Fs){const yy=group.base+group.Fs.indexOf(F)*group.step;svg+=t(30,yy+4,`${fSymbol(group.name as Level['manifold'])}=${F}`,'font-size="12"')+ln(85,yy,163,yy,'stroke="#aaa" stroke-width=".7" stroke-dasharray="2 3"');}
 for(let i=0;i<group.Fs.length-1;i++){const upper=group.levels.filter(s=>s.F===group.Fs[i]),lower=group.levels.filter(s=>s.F===group.Fs[i+1]),base=(items:Level[])=>items.reduce((v,s)=>v+s.energyMHz-(s.totalShiftMHz??s.shiftMHz),0)/items.length,gap=base(upper)-base(lower),top=group.base+i*group.step,bottom=top+group.step;svg+=`<g class="hyperfine-reference"><title>零外场超精细间隔</title>`+ln(155,top+8,155,bottom-8,'stroke="#aaa" stroke-width=".7" stroke-dasharray="2 3"')+t(145,(top+bottom)/2+3,fmt(gap).replace(/^\+/,''),'text-anchor="end" font-size="12" fill="#777"')+'</g>';}
 if(group.name==='ground')svg+=t(150,group.base-21,o.steadyPopulationLabel==='mean'?'长期平均':'长期布居','class="steady-population-symbol" text-anchor="end" font-size="12" fill="#605e5c"');
 for(const s of group.levels){const yy=pos.get(s.id)!,xx=x(s.m),initial=s.id===o.initialId,p=o.populations?.[s.id],style=p===undefined?{fill:'var(--surface, #fff)',ink:'#323130'}:populationAppearance(p),read=delta(s),selected=o.selectedIds.includes(s.id),ground=s.manifold==='ground',label=`${stateLabel(s)}${s.mixed?'（背景混合态，F/m 为裸态主成分标签）':''}; ${laser?'Δ':'总能移'} ${read==null?'—':withUnit(read,'MHz',{digits:7})}; Stark ${s.starkShiftMHz==null?'—':withUnit(s.starkShiftMHz,'MHz',{digits:7})}`;
 svg+=`<g class="level-node${selected?' selected':''}" data-state="${s.id}" data-base-label="${escapeHtml(label)}" role="button" tabindex="0" aria-pressed="${selected}" aria-label="${escapeHtml(label)}"><title>${escapeHtml(label)}</title><rect class="state-population-background" x="${xx-38}" y="${yy-(ground?26:9)}" width="76" rx="6" height="${ground?52:35}" fill="${initial?'#c8c6c4':style.fill}" stroke="none"/>`;
 svg+=ln(xx-29,yy,xx+29,yy,`class="state-energy" stroke="${initial?'#323130':style.ink}" stroke-width="2"`);
 if(selected)svg+=selectionNumber(xx,yy,o.selectedIds.indexOf(s.id),o.selectedIds.filter(id=>id===s.id).length>1);
 if(ground)svg+=t(xx,yy-15,pct(o.steadyPopulations?.[s.id]),`class="steady-population-label" data-line-y="${yy}" data-center-x="${xx}" data-side="above" data-state-id="${s.id}" dominant-baseline="central" text-anchor="middle" font-size="13" fill="${initial?'#323130':style.ink}"`);
 svg+=t(xx,yy+15,fmt(read),`class="level-value-label" data-line-y="${yy}" data-center-x="${xx}" data-side="below" data-state-id="${s.id}" data-quantity="${laser?'detuning':'total-shift'}" data-value-mhz="${read??''}" dominant-baseline="central" text-anchor="middle" font-size="13" fill="#323130"`);
 if(scene.marks?.[s.id])svg+=t(xx,yy-(ground?31:14),scene.marks[s.id]!, 'text-anchor="middle" font-size="10"');svg+='</g>';}
 svg+='</g>';}
 const stateLayer=svg;svg=prefix;
 // Coupling paths refer to the selected laser only; all manifolds stay visible.
 if(laser&&source)for(const tr of a.transitions.filter(tr=>tr.laserId===laser.id&&tr.rabiMHz>0)){const ends=transitionEndpoints(tr,source.id);if(!ends)continue;const from=point(ends.fromId),to=point(ends.toId),relative=tr.rabiMHz/Math.max(...a.transitions.filter(t=>t.laserId===laser.id).map(t=>t.rabiMHz),1e-30);svg+=ln(from.x,from.y,to.x,to.y,`class="transition-link" data-transition="${escapeHtml(tr.id)}" style="stroke:${identity(laser.id)}" stroke-width="1.2" opacity="${.1+.6*relative}" pointer-events="stroke"`);}
 const beams=scene.lasers.filter(b=>b.enabled),beamInfo=beams.map(b=>{const id=laserReferenceStateId(scene,b),e=energy(id),s=byId.get(id),down=laserEmits(scene,b,all);return {b,id,e,down,end:e===undefined?undefined:e+(down?-1:1)*laserFrequencyMHz(b)};}),ys=packLabels(beamInfo.map(b=>b.end===undefined?80:mapEnergy(b.end)),70,H-40,42);
 for(const [i,b] of beamInfo.entries()){if(b.end===undefined)continue;const start=point(b.id),yy=mapEnergy(b.end),active=b.b.id===laser?.id;svg+=`<g class="laser-energy-marker${active?' active':''}" data-laser-id="${escapeHtml(b.b.id)}" data-initial-id="${escapeHtml(b.id)}" data-process="${b.down?'stimulated-emission':'absorption'}" data-photon-mhz="${b.end}" opacity="${laser&&!active?.18:1}" role="button" tabindex="0" aria-pressed="${active}" aria-label="选择激光 ${escapeHtml(b.b.name)}"><title>${escapeHtml(`${b.b.name} · 起点 ${idLabel(b.id,id=>refs.find(r=>r.id===id)?.name)} · ν=${fmt(laserFrequencyMHz(b.b))}`)}</title>`+ln(start.x,start.y,810,yy,`style="stroke:${identity(b.b.id)}" stroke-width="1.6" marker-end="url(#combined-photon-${laserSlot(scene,b.b.id)})"`)+ln(813,yy,825,ys[i]!,`style="stroke:${identity(b.b.id)}" stroke-width=".7" opacity=".6"`)+t(830,ys[i]!+4,short(b.b.name,65),`font-size="12" style="fill:${identity(b.b.id)}"`)+'</g>';}
 svg+=stateLayer;
 if(refs.length)svg+=t(940,33,'参考能级','text-anchor="middle" font-size="12" fill="#605e5c"');
 for(const r of refs){const yy=refY.get(r.id)!,e=refE.get(r.id)!,value=laser?(r.id===sourceId?0:referenceReadoutMHz(photonSide,e,energyManifold(e))):e-referenceEnergyMHz({...r,anchorMode:'zero-field'},all)!,selected=o.selectedIds.includes(r.id);svg+=`<g class="level-node reference-node${selected?' selected':''}" data-state="${escapeHtml(r.id)}" tabindex="0" role="button" aria-pressed="${selected}" aria-label="${escapeHtml(r.name)}，仅能量参考"><title>${escapeHtml(r.name)} · E/h=${fmt(e)}</title>`+`<rect class="state-population-background" x="902" y="${yy-26}" width="76" height="52" rx="6" fill="var(--surface, #fff)" stroke="none"/>`+t(940,yy-15,short(r.name,72),`data-line-y="${yy}" data-center-x="940" data-side="above" dominant-baseline="central" text-anchor="middle" font-size="13"`)+(selected?selectionNumber(940,yy,o.selectedIds.indexOf(r.id),o.selectedIds.filter(id=>id===r.id).length>1):'')+ln(911,yy,969,yy,'class="state-energy" stroke="#323130" stroke-width="2"')+t(940,yy+15,fmt(value),`dominant-baseline="central" class="level-value-label reference-detuning-label" data-line-y="${yy}" data-center-x="940" data-side="below" data-state-id="${escapeHtml(r.id)}" data-value-mhz="${value??''}" text-anchor="middle" font-size="13" fill="#323130"`)+'</g>';}
 return `<svg class="chart-svg levels-svg" role="group" aria-label="Rb87 全部32态与参考能级" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" style="font-variant-numeric:tabular-nums">${svg}</svg>`;
}
