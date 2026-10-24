import {combinedLevelsChart} from './combined-level-chart.js';
import type { Analysis, Laser, Level, Scene, SpectralLine } from './types.js';
import { absoluteEnergyMHz, laserFrequencyMHz, laserIntensityMWcm2, RB87, transitionReferenceMHz } from './physics.js';
import { levelLayout, packLabels } from './level-layout.js';
import { referenceEnergyMHz } from './reference-energy.js';
import { incidentTransitions, laserReferenceStateId, levelReadoutMHz, photonEndpoint, targetDetuningMHz, transitionEndpoints } from './transition-view.js';

export const escapeHtml = (value: unknown): string => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export function number(value:number,places=3):string {
  if(!Number.isFinite(value))return value===Infinity?'∞':'—';
  return value.toLocaleString('en-US',{maximumFractionDigits:places,minimumFractionDigits:places});
}
const colors={blue:'#0067b8',ink:'#323130',muted:'#605e5c',grid:'#dedddc',field:'#287c58',laser:'#9a4f20'};
export const populationColors=['#0067b8','#b64b26','#287c58','#8a47a5','#ac7b00','#007e91','#b53868','#545454'];
const signed=(n:number)=>n>0?`+${n}`:String(n);
const text=(x:number,y:number,t:string,extra='')=>`<text x="${x}" y="${y}" ${extra}>${t}</text>`;
const line=(x1:number,y1:number,x2:number,y2:number,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
const frame=(content:string,viewBox='0 0 800 450',label='Scientific visualization',css='')=>`<svg class="chart-svg ${css}" role="group" aria-label="${escapeHtml(label)}" viewBox="${viewBox}" xmlns="http://www.w3.org/2000/svg" style="font-family:inherit;font-size:16px;fill:${colors.ink}">${content}</svg>`;
const termSvg=(manifold:Level['manifold'])=>manifold==='ground'?'5S<tspan baseline-shift="sub" font-size="13">1/2</tspan>':manifold==='D1'?'5P<tspan baseline-shift="sub" font-size="13">1/2</tspan>':'5P<tspan baseline-shift="sub" font-size="13">3/2</tspan>';

/** Fixed physical probability scale, shared by chart rendering and live patches.
 * Its lightest/darkest colors keep the same dark text readable at every p. */
export function populationAppearance(population:number):{fill:string;ink:string}{
  const t=Math.max(0,Math.min(1,population));
  const rgb=[Math.round(247-113*t),Math.round(249-70*t),Math.round(251-35*t)];
  return {fill:`rgb(${rgb.join(',')})`,ink:'#323130'};
}

export interface LevelChartOptions {
  line:SpectralLine; selectedIds:string[]; initialId:string;
  populations?:Record<string,number>; populationLabel?:string;
  /** Only a verified long-time result; never reuse a finite-time population. */
  steadyPopulations?:Record<string,number>|null;
  steadyPopulationLabel?:'mean'|'limit';
  /** Actual SVG viewport width, preserving readable text in narrow panels. */
  viewportWidth?:number;
}
export const levelsChart=combinedLevelsChart;
export function legacyLevelsChart(scene:Scene,analysis:Analysis,laser:Laser|undefined,options:LevelChartOptions):string {
  const excited=analysis[options.line],levels=[...analysis.ground,...excited],all=[...analysis.ground,...analysis.D1,...analysis.D2];
  const layout=levelLayout(analysis.ground,excited,options.line),{x,y}=layout;
  const physicalInitial=all.find(l=>l.id===options.initialId);
  if(!physicalInitial)return frame(text(24,40,'请选择有效的物理初态'),'0 0 900 100','初态不存在','levels-svg');
  const initial=all.find(l=>l.id===(laser?.referenceStateId??options.initialId));
  if(!initial)return frame(text(24,40,'请选择有效的激光参考态'),'0 0 900 100','参考态不存在','levels-svg');
  const emission=initial.manifold!=='ground',unresolved=analysis.spectroscopy?.status==='unresolved';
  const byId=new Map(levels.map(l=>[l.id,l]));
  const transitions=unresolved?[]:incidentTransitions(analysis,initial.id,laser?.id,options.line).filter(t=>t.rabiMHz>0);
  const maxCoupling=Math.max(0,...transitions.map(t=>t.weight*t.strength))||1;
  const fontScale=1.08*Math.max(.82,Math.min(1.55,900/(options.viewportWidth??900)));
  const compact=(n:number,places=2)=>`${n>0?'+':n<0?'−':''}${number(Math.abs(n),places)}`;
  const frequencyText=(n:number)=>{const unit=Math.abs(n)>=1e6?{scale:1e6,name:'THz'}:Math.abs(n)>=1000?{scale:1000,name:'GHz'}:{scale:1,name:'MHz'};return `${compact(n/unit.scale,Math.abs(n/unit.scale)>=100?1:2)} ${unit.name}`;};
  const cardFrequency=(n:number)=>{
    if(!Number.isFinite(n))return '—';
    const a=Math.abs(n),unit=a>=1e6?{scale:1e6,name:'THz'}:a>=1000?{scale:1000,name:'GHz'}:a>=1||a===0?{scale:1,name:'MHz'}:a>=.001?{scale:.001,name:'kHz'}:{scale:.000001,name:'Hz'};
    const v=n/unit.scale,body=Math.abs(v)!==0&&Math.abs(v)<.01?Math.abs(v).toExponential(1):Number(Math.abs(v).toPrecision(unit.name==='GHz'?4:3)).toString();
    return `${v>0?'+':v<0?'−':''}${body} ${unit.name}`;
  };
  const compactPopulation=(population:number)=>{
    const percent=population*100;
    if(!Number.isFinite(percent))return '—';
    return `${(Math.round(percent*100)/100).toFixed(2)}%`;
  };
  const precise=(n:number)=>n!==0&&Math.abs(n)<1e-4?n.toExponential(3):number(n,6);
  const short=(label:string,n=16)=>label.length>n?`${label.slice(0,n-1)}…`:label;
  const fitText=(label:string,width:number,size:number)=>{
    const measure=(s:string)=>[...s].reduce((sum,c)=>sum+(/[\u2e80-\u9fff\uac00-\ud7af]/u.test(c)?size:size*.54)*fontScale,0);
    if(measure(label)<=width)return label;
    let result='';for(const c of label){if(measure(result+c+'…')>width)break;result+=c;}return result+'…';
  };
  const interleaved=(group:Level[])=>{
    const fs=[...new Set(group.map(l=>l.F))],ranges=fs.map(F=>{const es=group.filter(l=>l.F===F).map(l=>l.energyMHz);return {min:Math.min(...es),max:Math.max(...es)};});
    return ranges.some((a,i)=>ranges.some((b,j)=>j>i&&a.min<b.max&&b.min<a.max));
  };
  const mixedExcited=interleaved(excited),mixedGround=interleaved(analysis.ground);
  const bandColors=['#f3e7ed','#f4e9df','#f4efd9','#e6f0e5','#e2eff1','#e7edf7','#efe8f4'];
  const bandAccents=['#b57b96','#b98b67','#b5a66c','#82a180','#7f9fa5','#8599b6','#9b87ad'];
  let svg='<defs><marker id="level-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#2f6fb0"/></marker><marker id="photon-arrow" viewBox="0 0 9 8" refX="8" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L9 4L0 8" fill="#8a4b1f"/></marker></defs>';
  svg+=text(18,27,termSvg(options.line),'font-size="16" font-weight="600"');
  svg+=text(126,27,'m<tspan baseline-shift="sub" font-size="10">F</tspan>','font-size="12" text-anchor="end" fill="#6a6866"');
  // Every mF has exactly one uninterrupted column identity across both manifolds.
  for(let m=-3;m<=3;m++){
    const xx=x(m),index=m+3;
    svg+=`<g class="mf-band" data-mf="${m}"><rect x="${xx-40}" y="40" width="80" height="${layout.bandBottom-40}" rx="3" fill="${bandColors[index]}"/><rect x="${xx-40}" y="38" width="80" height="3" rx="1.5" fill="${bandAccents[index]}"/>`;
    svg+=text(xx,27,signed(m),'text-anchor="middle" font-size="13" font-weight="600" fill="#4b4947"')+'</g>';
  }
  svg+=text(750,27,laser?(unresolved?'Δ（裸态）':'Δ'):'总位移','font-size="12" fill="#8a4b1f"');
  if(laser)svg+=text(750,44,escapeHtml(fitText(laser.name,131,11)),'font-size="11" fill="#8a4b1f"');
  const Fs=[...new Set(excited.map(l=>l.F))].sort((a,b)=>b-a);
  const axisTicks=(group:Level[])=>{
    const min=Math.min(...group.map(l=>l.energyMHz)),max=Math.max(...group.map(l=>l.energyMHz)),rough=(max-min)/4,power=10**Math.floor(Math.log10(rough)),step=[1,2,5,10].map(n=>n*power).find(n=>n>=rough)!;
    let result='';for(let e=Math.ceil(min/step)*step;e<=max;e+=step){const yy=y({...group[0]!,energyMHz:e});result+=text(112,yy+4,String(Math.round(e)),'text-anchor="end" font-size="11" fill="#6a6866"')+line(119,yy,126,yy,'stroke="#a19f9d" stroke-width="1"');}return result;
  };
  if(mixedExcited&&!layout.excitedSchematic){svg+=axisTicks(excited)+text(18,49,'E/h · MHz','font-size="10" fill="#6a6866"');}
  else for(const F of Fs){const group=excited.filter(l=>l.F===F),yy=group.reduce((sum,l)=>sum+y(l),0)/group.length;svg+=text(29,yy+4,`F′ = ${F}`,'font-size="13" font-weight="600"');}
  svg+=text(18,layout.groundTop-30,termSvg('ground'),'font-size="16" font-weight="600"');
  svg+=text(112,layout.groundTop-12,`${options.steadyPopulationLabel==='mean'?'p̄':'p'}<tspan baseline-shift="sub" font-size="7">∞</tspan>`,'class="steady-population-symbol" text-anchor="end" font-size="10" fill="#777573"');
  if(mixedGround&&!layout.groundSchematic)svg+=axisTicks(analysis.ground);
  else for(const F of [2,1]){const group=analysis.ground.filter(l=>l.F===F),yy=group.reduce((sum,l)=>sum+y(l),0)/group.length;svg+=text(29,yy+4,`F = ${F}`,'font-size="13" font-weight="600"');}
  const referenceIntervals=(group:Level[],mixed:boolean)=>{
    if(mixed)return '';
    const rows=[...new Set(group.map(l=>l.F))].map(F=>{
      const states=group.filter(l=>l.F===F);
      return {F,y:states.reduce((sum,l)=>sum+y(l),0)/states.length};
    }).sort((a,b)=>a.y-b.y);
    let result='';
    for(let i=0;i<rows.length-1;i++){
      const upper=rows[i]!,lower=rows[i+1]!,middle=(upper.y+lower.y)/2;
      const gap=group[0]!.manifold==='ground'?2*RB87.groundA:transitionReferenceMHz(options.line,2,upper.F)-transitionReferenceMHz(options.line,2,lower.F);
      result+=`<g class="hyperfine-reference" data-upper-f="${upper.F}" data-lower-f="${lower.F}"><title>零外场超精细参考间隔；有限场各 mF 态的间隔分别计算。</title>`+
        line(112,upper.y+5,112,lower.y-5,'stroke="#b9b7b5" stroke-width=".8" stroke-dasharray="2 3"')+
        line(108,upper.y+5,118,upper.y+5,'stroke="#b9b7b5" stroke-width=".8"')+
        line(108,lower.y-5,118,lower.y-5,'stroke="#b9b7b5" stroke-width=".8"')+
        text(104,middle+4,escapeHtml(cardFrequency(Math.abs(gap)).replace(/^\+/,'')),'text-anchor="end" font-size="9.5" fill="#777573"')+'</g>';
    }
    return result;
  };
  svg+=text(18,51,'零场间隔','font-size="9" fill="#8a8886"')+referenceIntervals(excited,mixedExcited)+referenceIntervals(analysis.ground,mixedGround);
  const breakY=layout.excitedBottom+47;
  svg+=`<g class="energy-axis-break"><title>光学间隔省略；${layout.schematic?'近简并处为避免遮挡已作纵向示意展开，精确频率见读数。':'每个精细结构流形内共用线性能量尺度。'}</title><path d="M 103 ${breakY-4} l 8 5 m -8 0 l 8 5" fill="none" stroke="#a19f9d" stroke-width="1.5"/></g>`;
  for(const t of transitions){
    const g=byId.get(t.groundId),e=byId.get(t.excitedId);if(!g||!e)continue;
    const rel=t.strength*t.weight/maxCoupling,endpoints=transitionEndpoints(t,initial.id)!,from=byId.get(endpoints.fromId)!,to=byId.get(endpoints.toId)!;
    const direction=endpoints.emission?'受激辐射':'吸收';
    const quantumNumbers='mixedPolarization' in t&&t.mixedPolarization===true?'混合偏振耦合；q 与 Δm 不唯一':`q=${t.q}（光场），Δm=${endpoints.deltaM}`;
    svg+=`<g class="transition-link" data-transition="${escapeHtml(t.id)}" data-from-state="${from.id}" data-to-state="${to.id}" data-process="${endpoints.emission?'stimulated-emission':'absorption'}"><title>${escapeHtml(`${direction} · ${from.id} → ${to.id}; ${quantumNumbers}; |d·ε|²/dcyc² = ${precise(t.strength*t.weight)}; Ω/2π = ${precise(t.rabiMHz)} MHz; 相对最大 ${precise(rel)}`)}</title>`+line(x(from.m),y(from)+(emission?2:-2),x(to.m),y(to)+(emission?-2:2),`stroke="#2f6fb0" stroke-width="1.5" opacity="${.08+.69*Math.sqrt(rel)}" marker-end="url(#level-arrow)"`)+`</g>`;
  }
  const beams=scene.lasers.filter(l=>l.enabled);
  const beamData=beams.flatMap((beam,i)=>{
    const source=all.find(state=>state.id===laserReferenceStateId(scene,beam));
    if(!source)return [];
    const endpoint=photonEndpoint(source,beam),energy=endpoint.energyMHz,point=layout.photon(energy,endpoint.target);
    const beamTargets=source.manifold==='ground'?excited:analysis.ground;
    const coupledIds=new Set(analysis.transitions.filter(t=>t.laserId===beam.id&&t.rabiMHz>0).map(t=>transitionEndpoints(t,source.id)?.toId));
    const coupled=beamTargets.filter(target=>coupledIds.has(target.id)),candidates=coupled.length?coupled:beamTargets;
    const nearest=candidates.reduce((best,l)=>Math.abs(absoluteEnergyMHz(l)-energy)<Math.abs(absoluteEnergyMHz(best)-energy)?l:best);
    const visible=byId.has(source.id),startX=visible?x(source.m):720,startY=visible?y(source):layout.photon(absoluteEnergyMHz(source)).y;
    return [{beam,source,emission:endpoint.emission,energy,point,nearest,coupled:coupled.length>0,energized:laserIntensityMWcm2(beam)>0,x:726+i*2,startX,startY,visible}];
  });
  const labelTop=Math.max(70,44+30*fontScale),labelBottom=layout.groundBottom-22,denseLabels=(labelBottom-labelTop)/Math.max(1,beams.length-1)<36*fontScale;
  const labelYs=packLabels(beamData.map(b=>b.point.y),labelTop,labelBottom,(denseLabels?22:40)*fontScale);
  for(const [i,b]of beamData.entries()){
    const active=b.beam.id===laser?.id,delta=targetDetuningMHz(b.source,b.nearest,b.beam)!,yy=b.point.y,labelY=labelYs[i]!,{startX,startY}=b;
    const outside=b.point.placement!=='inside',name=fitText(b.beam.name,132,12),dimmed=!!laser&&!active;
    const tip=`${b.beam.name}: ${b.emission?'受激辐射 · Einit − hν':'吸收 · Einit + hν'}; 显示参考态 ${b.source.id}（不改变动力学初态）; ${unresolved?'静态背景未定义，箭头以裸能级作几何参考；':''}${b.energized?'':'I=0，仅显示频率标记；'}ν = ${number(laserFrequencyMHz(b.beam)/1e6,9)} THz; ${outside?'当前能级窗口之外，箭头在断轴处截断；':''}${b.coupled?'最近耦合态':'无初态直接耦合；最近能量态'} ${b.nearest.id}${unresolved?'':`; Δ = νL − νeg = ${precise(delta)} MHz`}${b.emission?'；更高流形的吸收不属于当前 32 态模型':''}`;
    svg+=`<g class="laser-energy-marker${active?' active':''}" data-laser-id="${escapeHtml(b.beam.id)}" data-initial-id="${b.source.id}" data-process="${b.emission?'stimulated-emission':'absorption'}" data-photon-mhz="${b.energy}" data-placement="${b.point.placement}" data-nearest-target="${b.nearest.id}" data-has-coupling="${b.coupled}" opacity="${dimmed?.18:1}" tabindex="0" role="button" aria-label="${escapeHtml(`选择激光 ${b.beam.name}`)}" aria-pressed="${active}"><title>${escapeHtml(tip)}</title>`;
    // A photon is a single ray from the actual initial state. Never a routed L.
    svg+=line(startX,startY,b.x,yy,'stroke="transparent" stroke-width="12" pointer-events="stroke"');
    svg+=line(startX,startY,b.x,yy,`class="photon-ray" stroke="#8a4b1f" stroke-opacity="${b.energized?1:.4}" stroke-width="${active?2:1.2}" marker-end="url(#photon-arrow)"`);
    if(outside){
      const dx=b.x-startX,dy=yy-startY,len=Math.hypot(dx,dy),ux=dx/Math.max(1,len),uy=dy/Math.max(1,len),cx=b.x-ux*15,cy=yy-uy*15;
      svg+=line(cx-uy*4,cy+ux*4,cx+uy*4,cy-ux*4,'stroke="white" stroke-width="5"');
      svg+=line(cx-uy*4-ux*2,cy+ux*4-uy*2,cx+uy*4-ux*2,cy-ux*4-uy*2,'stroke="#8a4b1f" stroke-width="1.2"');
      svg+=line(cx-uy*4+ux*2,cy+ux*4+uy*2,cx+uy*4+ux*2,cy-ux*4+uy*2,'stroke="#8a4b1f" stroke-width="1.2"');
    }
    svg+=line(b.x+3,yy,743,labelY-4*fontScale,'stroke="#b99174" stroke-width=".8"');
    svg+=text(750,labelY+(denseLabels?4:-4)*fontScale,escapeHtml(name),`class="laser-name-label" font-size="12" font-weight="${active?600:400}" fill="${active&&b.energized?'#8a4b1f':'#6a6866'}"`);
    if(!denseLabels)svg+=text(750,labelY+14*fontScale,outside||unresolved?`${number(299792458000/laserFrequencyMHz(b.beam),2)} nm`:b.coupled?`Δ ${escapeHtml(frequencyText(delta))}`:`${number(299792458000/laserFrequencyMHz(b.beam),2)} nm`,'class="laser-frequency-label" font-size="10" fill="#6a6866"');
    if(!b.visible)svg+=text(startX-4,startY+15,`ref · ${b.source.manifold}`,'text-anchor="end" font-size="11" fill="#8a4b1f"');
    svg+='</g>';
  }
  for(const l of levels){
    const isInitial=l.id===physicalInitial.id,isReference=!!laser&&l.id===initial.id,selected=options.selectedIds.includes(l.id),xx=x(l.m),yy=y(l),mark=scene.marks?.[l.id],population=options.populations?.[l.id];
    // A missing static light-shift solution does not erase the well-defined
    // detuning from the displayed bare HFS/Zeeman reference spectrum.
    const hasPopulation=population!==undefined,delta=laser?levelReadoutMHz(initial,l,laser):undefined;
    const readout=laser?delta:l.totalShiftMHz,label=readout==null?'—':cardFrequency(readout);
    const steady=options.steadyPopulations?.[l.id],ground=l.manifold==='ground';
    const mixed=l.manifold==='ground'?mixedGround:mixedExcited;
    const stateLabel=l.mixed?'裸态主成分标签 ':mixed?'零场绝热标签 ':'';
    const sameSide=initial.manifold==='ground'?ground:l.manifold===initial.manifold;
    const definition=(unresolved?'裸态参考，未含未确定的背景光移；':'')+(sameSide?'Δ = (Eref − E)/h':emission?'受激辐射 · Δ = νL − (Eref − E)/h':'吸收 · Δ = νL − (E − Eref)/h');
    const baseTip=`${l.manifold} ${stateLabel}F=${l.F}, mF=${l.m}; ${l.totalShiftMHz!=null?`总位移（相对零外场） ${precise(l.totalShiftMHz)} MHz`:'总位移待定：当前光场未得到可靠的静态能级'}${delta!==undefined?`; ${laser?.name}: ${definition} = ${precise(delta)} MHz`:''}${mark?`; ${mark}`:''}`;
    const tip=baseTip+`${hasPopulation?`; ${options.populationLabel??'布居'} = ${precise(population*100)}%`:''}${ground?`; ${options.steadyPopulationLabel==='mean'?'长期平均布居':'长期布居极限'} ${steady===undefined?'未知':`${precise(steady*100)}%`}`:''}`;
    const appearance=hasPopulation?populationAppearance(population):{fill:'#fff',ink:'#323130'};
    svg+=`<g class="level-node${selected?' selected':''}" data-state="${l.id}" data-base-label="${escapeHtml(baseTip)}" tabindex="0" role="button" aria-pressed="${selected}" aria-label="${escapeHtml(tip)}"><title>${escapeHtml(tip)}</title>`;
    svg+=`<rect class="state-population-background" x="${xx-39}" y="${yy-(ground?25:8)}" width="78" height="${ground?47:29}" rx="3" fill="${appearance.fill}" stroke="${selected?'#004578':isReference?'#b77a50':isInitial?'#77aadd':'none'}" stroke-width="${selected?2:1.4}"/>`;
    if(mixed)svg+=text(xx-33,yy+2,`${l.manifold==='ground'?'F':'F′'}${l.F}`,`class="adiabatic-label" font-size="9" fill="${appearance.ink}"`);
    svg+=line(xx+(mixed?-7:-30),yy,xx+30,yy,`class="state-energy" stroke="${appearance.ink}" stroke-width="2.4"`);
    if(ground)svg+=text(xx,yy-11,steady===undefined?'—':escapeHtml(compactPopulation(steady)),`class="steady-population-label" data-state-id="${l.id}" text-anchor="middle" font-size="10" fill="${appearance.ink}"`);
    // Keep the detuning on a white strip so its warm semantic colour remains
    // legible even when the population background reaches its darkest blue.
    if(laser)svg+=`<rect x="${xx-37}" y="${yy+3}" width="74" height="17" rx="2" fill="#fff" fill-opacity=".95"/>`;
    const labelMarkup=escapeHtml(label).replace(/ (Hz|kHz|MHz|GHz|THz)$/,' <tspan font-size="8.7">$1</tspan>');
    svg+=text(xx,yy+15,labelMarkup,`class="state-readout level-value-label" data-state-id="${l.id}" data-quantity="${laser?'detuning':'total-shift'}" data-value-mhz="${readout??''}" text-anchor="middle" font-size="10.3" font-weight="${laser?600:400}" fill="#323130"`);
    if(isInitial)svg+=`<circle class="state-initial-dot" cx="${xx-33}" cy="${yy-4}" r="2.7" fill="${hasPopulation?appearance.ink:'#2f6fb0'}"/>`;
    if(mark)svg+=text(xx,yy-(ground?29:12),escapeHtml(fitText(mark,78,10)),'class="state-mark" text-anchor="middle" font-size="10" font-weight="600" fill="#004578" paint-order="stroke" stroke="white" stroke-width="3" stroke-linejoin="round"');
    if(selected)svg+=`<circle cx="${xx+33}" cy="${yy-5}" r="7" fill="#004578"/>`+text(xx+33,yy-2,String(options.selectedIds.indexOf(l.id)+1),'text-anchor="middle" font-size="9" font-weight="700" fill="white"');
    svg+='</g>';
  }
  let bottom=layout.bandBottom+21;
  if(layout.schematic){svg+=text(18,bottom,'近简并能级已展开 · 纵向示意','font-size="10" fill="#6a6866"');bottom+=20;}
  if(mixedExcited||mixedGround||levels.some(l=>l.mixed)){svg+=text(18,bottom,levels.some(l=>l.mixed)?'混合态 F、mF：裸态主成分标签':'F、mF：零场绝热标签','font-size="10" fill="#6a6866"');bottom+=20;}
  // Annotation energies remain selectable, and retain their exact physical
  // anchor. They are references, not extra Lindblad basis states.
  for(const ref of scene.references??[]){
    const selected=options.selectedIds.includes(ref.id),energy=referenceEnergyMHz(ref,all);
    const tip=`能量参考（非动力学态，${ref.anchorMode==='zero-field'?'固定零场':'跟随当前磁场'}）· ${ref.name}: ${ref.anchorId} ${frequencyText(ref.offsetMHz)}${energy===undefined?'':`; E/h = ${number(energy,6)} MHz`}`;
    svg+=`<g data-state="${escapeHtml(ref.id)}" class="level-node reference-node" tabindex="0" role="button" aria-label="${escapeHtml(tip)}" aria-pressed="${selected}"><title>${escapeHtml(tip)}</title><rect x="18" y="${bottom-13}" width="658" height="25" rx="3" fill="${selected?'#e4eff9':'#f3f3f3'}"/>`+text(28,bottom+3,escapeHtml(short(ref.name,24)),'font-size="12"')+line(230,bottom,430,bottom,'stroke="#8a4b1f" stroke-width="1.5"')+text(665,bottom+3,escapeHtml(frequencyText(ref.offsetMHz)),'text-anchor="end" font-size="12" fill="#6a6866"')+'</g>';bottom+=30;
  }
  svg=svg.replace(/font-size="([\d.]+)"/g,(_,size:string)=>`font-size="${(Number(size)*fontScale).toFixed(2)}"`);
  return frame(svg,`0 0 900 ${bottom+5}`,'Rb87 能级；纵向颜色标识 mF，态的蓝色深浅表示固定 0–100% 布居。激光实线从初态出发，窗口外频率在断轴处明确截断。','levels-svg');
}


export { populationChart, populationColor,  populationValue, populationLayout, populationSamplingInfo, bindPopulationChart } from './population-chart.js';
export type { PopulationPlot, PopulationChartOptions } from './population-chart.js';
