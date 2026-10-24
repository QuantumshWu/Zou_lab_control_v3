import type { Analysis, Laser, Polarization, Scene, Vec3 } from './types.js';
import { Command } from '../../desktop/ui/src/fluent/button.js';
import {Segmented,checkBox,setChecked} from '../../desktop/ui/src/fluent/toggles.js';
import {depthOpacity,splitSegment,rayDepth} from './space-depth.js';
import {SpaceDepthRenderer} from './space-depth-renderer.js';
import {arrowGeometry,shadeArrow} from './space-arrow.js';

/** A retained, camera-controlled view. update() never resets the camera. */
export interface SpaceView {
  update(scene: Scene, analysis: Analysis, selectedLaserId?: string): void;
  destroy(): void;
}

type Point = { x: number; y: number; z: number; scale: number };
type Segment = { a: Vec3; b: Vec3; color: string; width: number; alpha?: number };
type Face = { vertices: Vec3[]; color: string; alpha: number };
type Label = { p: Vec3; text: string; color: string; alpha?:number; strong?: boolean; rank: number; laserId?: string };
type HitRay={a:Point;b:Point;laserId:string};
type HitLabel={x:number;y:number;w:number;h:number;laserId:string};
const TAU = 2 * Math.PI, DEG = Math.PI / 180;
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, v: number): Vec3 => [a[0] * v, a[1] * v, a[2] * v];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clamp = (n: number, a: number, b: number): number => Math.min(b, Math.max(a, n));
const ZERO: Vec3 = [0, 0, 0];
const FONT = '"Segoe UI", "Microsoft YaHei UI", sans-serif';
const ACCENT='#2f6fb0',LASER='#8a4b1f',MUTED='#8a8886',GREEN='#287c58';

/** The same real Jones field used by physics.ts, with a normalized optical phase.
 * Positive chi traverses u toward v for epsilon exp(-i omega t). */
export function electricFieldVector(laser:Laser,polarization:Polarization,phase:number):Vec3 {
  const chi=laser.ellipticityDeg*DEG,t=phase-(laser.phaseDeg??0)*DEG;
  return add(mul(polarization.u,Math.cos(chi)*Math.cos(t)),mul(polarization.v,Math.sin(chi)*Math.sin(t)));
}

/**
 * True perspective geometry with per-pixel depth-tested 3D primitives, deliberately local
 * and dependency-free so the same view works in the offline single-file demo.
 * World +z is up. Each polarization ellipse uses the solver's u,v transverse frame;
 * its centre is displaced along -k for visibility, never rotated to face the camera.
 * E(t) = Re[(cos(chi) u + i sin(chi) v) exp(i phi - i omega t)].
 * Beam width and vector lengths are schematic, not intensity / magnetic-field scales.
 */
export function renderSpace3D(container: HTMLElement, scene: Scene, analysis: Analysis, selectedLaserId?: string): SpaceView {
  let currentScene = scene, currentAnalysis = analysis, currentSelection = selectedLaserId;
  let width = 760, height = 480, dpr = 1, yaw = -1.03, pitch = .50, distance = 7.6;
  let animation = 0, destroyed = false, animateField = false, phase = 0, previousTime = 0;
  let showPlane=true;
  let hitRays:HitRay[]=[],hitLabels:HitLabel[]=[],hoverLaserId:string|undefined;
  let pointerOrigin:{x:number;y:number}|undefined,dragged=false;
  const abort = new AbortController(), eventOptions = { signal: abort.signal };
  const pointers = new Map<number, {x: number; y: number}>();

  const host = document.createElement('div');
  host.className = 'atom-space3d';
  host.style.cssText = 'position:relative;min-width:0;min-height:430px;height:clamp(430px,58vh,660px);overflow:hidden;background:var(--surface,#fff);border-radius:3px;isolation:isolate';
  const depthCanvas=document.createElement('canvas');depthCanvas.className='space-depth-canvas';
  depthCanvas.setAttribute('aria-hidden','true');depthCanvas.style.cssText='position:absolute;inset:0;width:100%;height:100%;pointer-events:none';
  const depthRenderer=new SpaceDepthRenderer(depthCanvas);
  const canvas = document.createElement('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', '可旋转的三维实验空间：全部启用激光、偏振椭圆、总磁场和原子。方向键旋转，加减键缩放，Home 键复位。');
  canvas.style.cssText = 'display:block;width:100%;height:100%;position:absolute;inset:0;touch-action:none;cursor:grab;outline:none';
  const context = canvas.getContext('2d');
  if (!context) throw new Error('浏览器无法创建三维示意画布。');
  const ctx = context;

  const topbar=document.createElement('div');
  topbar.className='space-toolbar';
  const beamPicker=document.createElement('div');beamPicker.className='space-beam-picker';beamPicker.setAttribute('role','toolbar');beamPicker.setAttribute('aria-label','选择空间图中的激光');
  beamPicker.style.cssText='display:flex;flex-wrap:wrap;align-items:center;gap:var(--row-gap);min-width:0;pointer-events:auto';
  const toolbar = document.createElement('div');
  toolbar.setAttribute('role', 'toolbar');
  toolbar.setAttribute('aria-label', '三维视角');
  toolbar.style.cssText = 'display:flex;gap:2px;align-items:center;flex:none;pointer-events:auto';
  const buttons: HTMLButtonElement[] = [];
  function button(text: string, label: string, action: () => void): HTMLButtonElement {
    const el = new Command(text,action,{role:'quiet',glyph:text.length===1,square:text.length===1,title:label}).root;
    el.setAttribute('aria-label', label);
    buttons.push(el); toolbar.append(el); return el;
  }
  button('−', '缩小', () => zoom(1.14));
  button('＋', '放大', () => zoom(.88));
  button('复位', '复位三维视角', reset);
  const playButton = button('▶ E', '播放偏振电场方向（归一化光学周期）', () => {
    animateField = !animateField;
    playButton.textContent = animateField ? 'Ⅱ E' : '▶ E';
    playButton.classList.toggle('active',animateField);
    playButton.setAttribute('aria-label',animateField?'暂停偏振电场动画':'播放偏振电场方向（归一化光学周期）');
    playButton.setAttribute('aria-pressed', String(animateField));
    previousTime = 0; queueDraw();
  });
  playButton.setAttribute('aria-pressed', 'false');
  const viewbar=document.createElement('div');viewbar.className='space-viewbar';
  const views=new Segmented(value=>{if(value==='xy'){yaw=-Math.PI/2;pitch=Math.PI/2;}else if(value==='xz'){yaw=-Math.PI/2;pitch=0;}else if(value==='yz'){yaw=0;pitch=0;}else{yaw=-1.03;pitch=.50;}views.setCurrent(value);queueDraw();});
  views.setChoices([['free','3D'],['xy','XY'],['xz','XZ'],['yz','YZ']]);views.setCurrent('free');views.root.setAttribute('aria-label','观察方向');
  const plane=checkBox('xy 参考面',on=>{showPlane=on;queueDraw();});setChecked(plane,true);viewbar.append(views.root,plane);
  const hint = document.createElement('div');hint.textContent='拖动旋转';
  hint.style.cssText = 'position:absolute;left:16px;bottom:14px;color:var(--muted-ink,#6a6866);font:11px ' + FONT + ';pointer-events:none';
  topbar.append(beamPicker,toolbar,viewbar);host.append(depthCanvas,canvas,topbar,hint);
  container.replaceChildren(host);

  function reset(): void { yaw = -1.03; pitch = .50; distance = 7.6; phase = 0; views.setCurrent('free');queueDraw(); }
  function zoom(factor: number): void { distance = clamp(distance * factor, 4.5, 13); queueDraw(); }
  function queueDraw(): void { if (!animation && !destroyed) animation = requestAnimationFrame(draw); }
  function frame() {
    const forward: Vec3 = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
    const right: Vec3 = [-Math.sin(yaw), Math.cos(yaw), 0];
    const up: Vec3 = [-Math.sin(pitch) * Math.cos(yaw), -Math.sin(pitch) * Math.sin(yaw), Math.cos(pitch)];
    const focal = Math.min(width, height * 1.48) * 1.06;
    return { forward, right, up, focal };
  }
  function selectLaser(id:string):void {container.dispatchEvent(new CustomEvent('atom-select-laser',{bubbles:true,detail:{id}}));}
  function updateStatus(): void {
    const lasers = currentScene.lasers.filter(l => l.enabled);
    const focused=document.activeElement instanceof HTMLElement?document.activeElement.dataset.spaceLaser:undefined;
    beamPicker.replaceChildren(...lasers.map(laser=>{
      const active=laser.id===currentSelection,el=new Command(laser.name,()=>selectLaser(laser.id),{role:'quiet'}).root;
      el.dataset.spaceLaser=laser.id;el.setAttribute('aria-pressed',String(active));el.setAttribute('aria-label',`选择激光 ${laser.name}`);el.classList.toggle('active',active);
      el.classList.toggle('dimmed',Boolean(currentSelection)&&!active);
      el.style.cssText='max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';return el;
    }));
    if(focused)beamPicker.querySelector<HTMLButtonElement>(`[data-space-laser="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
    hint.textContent=currentAnalysis.fieldMagnitudeG===0?'B = 0 · 参考轴 +z':'拖动旋转 · 滚轮缩放';
  }
  function draw(time: number): void {
    animation = 0;
    if (destroyed || !host.isConnected) return;
    if (animateField && previousTime) phase = (phase + Math.min(time - previousTime, 60) * .0018) % TAU;
    previousTime = time;
    canvas.dataset.camera=JSON.stringify({yaw,pitch,distance});canvas.dataset.selectedLaser=currentSelection??'';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    depthRenderer.begin(width,height,dpr,distance);
    const camera = frame();
    const project = (p: Vec3): Point => {
      const z = dot(camera.forward, p), scale = camera.focal / (distance - z);
      return { x: width * .50 + dot(camera.right, p) * scale, y: height * .50 - dot(camera.up, p) * scale, z, scale };
    };
    hitRays=[];hitLabels=[];
    const labels: Label[] = [];
    const segment = ({a, b, color, width: strokeWidth, alpha = 1}: Segment): void => {
      for(const [first,last]of splitSegment(a,b)){
      const start = project(first), end = project(last),depth=(start.z+end.z)/2;
      const lineWidth=strokeWidth*(.85+.2*depthOpacity(depth)),opacity=alpha*depthOpacity(depth);
      depthRenderer.line(start,end,lineWidth,color,opacity);
      }
    };
    const face = ({vertices, color, alpha}: Face): void => {
      for(let i=1;i<vertices.length-1;i++){
      const points = [vertices[0]!,vertices[i]!,vertices[i+1]!].map(project);
      depthRenderer.face(points,color,alpha*depthOpacity(points.reduce((s,p)=>s+p.z,0)/points.length));
      }
    };
    const curve = (points: Vec3[], color: string, strokeWidth: number, alpha = 1): void => {
      for(let i=1;i<points.length;i++)segment({a:points[i-1]!,b:points[i]!,color,width:strokeWidth,alpha});
    };
    const light=add(mul(camera.forward,.8),mul(camera.up,.6));
    const arrow = (from: Vec3, to: Vec3, color: string, strokeWidth: number, alpha = 1, transverse?:Vec3): void => {
      if(alpha<=0)return;
      const middle=mul(add(from,to),.5),radius=strokeWidth/(2*project(middle).scale);
      for(const facet of arrowGeometry(from,to,radius,transverse)){
        const points=facet.vertices.map(project),depth=points.reduce((sum,p)=>sum+p.z,0)/points.length;
        depthRenderer.face(points,shadeArrow(color,facet.normal,light),alpha*depthOpacity(depth),true);
      }
    };

    if(showPlane){
      const steps=[-1.6,-.8,0,.8,1.6];
      for(let i=0;i<4;i++)for(let j=0;j<4;j++)face({vertices:[[steps[i]!,steps[j]!,0],[steps[i+1]!,steps[j]!,0],[steps[i+1]!,steps[j+1]!,0],[steps[i]!,steps[j+1]!,0]],color:'#b7c5d1',alpha:.09});
      for(const v of steps){segment({a:[v,-1.6,0],b:[v,1.6,0],color:'#aab7c2',width:.65,alpha:.30});segment({a:[-1.6,v,0],b:[1.6,v,0],color:'#aab7c2',width:.65,alpha:.30});}
    }
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]).forEach(axis => {
      segment({a: mul(axis, -1.65), b: mul(axis,1.65), color: '#c8c6c4', width: .7, alpha: .55});
    });

    const enabled = currentScene.lasers.filter(l => l.enabled);
    enabled.forEach((laser, index) => {
      const pol = currentAnalysis.polarizations[laser.id];
      if (!pol) return;
      const selected = laser.id === currentSelection,hover=laser.id===hoverLaserId;
      const dimmed=Boolean(currentSelection)&&!selected,color=selected||hover||!currentSelection?LASER:MUTED,alpha=selected?1:hover?.9:dimmed?.35:.9;
      const {k, u, v} = pol, start = mul(k, -2.05), end = mul(k, 1.95);
      const chi = laser.ellipticityDeg * DEG, center = mul(k, -1.18);
      const radius = .68;
      // A translucent, narrow cylinder makes a beam readable from every viewpoint.
      const side = (t: number): Vec3 => add(mul(u, Math.cos(t) * (selected?.085:.055)), mul(v, Math.sin(t) * (selected?.085:.055)));
      for (let n = 0; n < 8; n++) {
        const a = side(n * TAU / 8), b = side((n + 1) * TAU / 8);
        for(const [lo,hi]of splitSegment(start,end,.34))face({vertices: [add(lo, a), add(hi, a), add(hi, b), add(lo, b)], color, alpha: selected ? .052 : dimmed?.009:.025});
      }
      segment({a: start, b: mul(k, -.08), color, width: selected ? 2.1 : 1.35, alpha});
      arrow(mul(k, .1), end, color, selected ? 2.1 : 1.35, alpha,u);
      hitRays.push({a:project(start),b:project(end),laserId:laser.id});
      // Short k-direction arrows on the incoming half remove ambiguity about propagation.
      arrow(mul(k, -1.9), mul(k, -1.6), color, selected ? 1.8 : 1.1, alpha,u);
      if (selected) {
        if(showPlane&&Math.abs(end[2])>.08){const foot:Vec3=[end[0],end[1],0];segment({a:end,b:foot,color,width:.8,alpha:.45});segment({a:add(foot,[-.06,0,0]),b:add(foot,[.06,0,0]),color,width:1,alpha:.6});segment({a:add(foot,[0,-.06,0]),b:add(foot,[0,.06,0]),color,width:1,alpha:.6});}
        const disc: Vec3[] = Array.from({length: 60}, (_, j) => add(center, add(mul(u, radius * Math.cos(j * TAU / 60)), mul(v, radius * Math.sin(j * TAU / 60)))));
        face({vertices: disc, color:ACCENT, alpha: .065});
        curve([...disc, disc[0]], ACCENT, .85, .38);
        segment({a: add(center, mul(u, -radius)), b: add(center, mul(u, radius)), color:ACCENT, width: .8, alpha: .22});
        segment({a: add(center, mul(v, -radius)), b: add(center, mul(v, radius)), color:ACCENT, width: .8, alpha: .22});
        const fieldAt = (t: number): Vec3 => add(center,mul(electricFieldVector(laser,pol,t),radius));
        const ellipse=Math.abs(chi)<1e-12?[add(center,mul(u,-radius)),add(center,mul(u,radius))]:Array.from({length:97},(_,j)=>fieldAt(j*TAU/96));
        curve(ellipse, ACCENT, 2.2,1);
        // With epsilon exp(i phi - i omega t), E advances u→v for positive chi.
        arrow(center,fieldAt(phase),ACCENT,2.3,1,k);
        arrow(fieldAt(phase + .65),fieldAt(phase + .87),ACCENT,1.7,Math.min(1,Math.abs(chi)/.08),k);
        labels.push({p:add(center,mul(u,radius*1.17)),text:'E',color:ACCENT,strong:true,rank:2,laserId:laser.id});
        labels.push({p:mul(k,2.13),text:'k',color,strong:true,rank:1,laserId:laser.id});
      }else{
        labels.push({p:mul(k,-2.16-index*.045),text:laser.name,color:hover?LASER:'#6a6866',alpha:dimmed&&!hover?.7:1,rank:3,laserId:laser.id});
      }
    });

    const fieldMagnitude = currentAnalysis.fieldMagnitudeG;
    if (fieldMagnitude > 0) {
      const end = currentAnalysis.fieldVector.map(value=>value/fieldMagnitude*1.42) as Vec3;
      // Draw at actual depth; a front vector may cover a rear one.
      const start=mul(end,.06);
      arrow(start, end, GREEN, 2.8);
      labels.push({p: mul(end, 1.11), text: `B  ${fieldMagnitude>=.001?fieldMagnitude.toFixed(3):fieldMagnitude.toExponential(2)} G`, color: GREEN, strong: true, rank: 0});
    }
    const origin = project(ZERO);
    depthRenderer.circle(origin,9,'#ffffff');depthRenderer.circle(origin,5.5,'#323130');
    depthRenderer.flush();
    // Text is screen-facing; geometric anchors retain their true 3D coordinates.
    // Resolve label collisions locally rather than hiding one of two collinear beams.
    const occupied: {x: number; y: number; w: number; h: number}[] = [
      {x: 0, y: 0, w:width,h:Math.max(52,topbar.offsetHeight+22)},
    ];
    labels.sort((a, b) => a.rank - b.rank).forEach(label => {
      const p = project(label.p), size = label.rank<=2?13:12;
      ctx.font = `${label.strong ? 600 : 500} ${size}px ${FONT}`;
      const maxLabelWidth = Math.max(80, width * .34);
      let text = label.text;
      while (ctx.measureText(text).width > maxLabelWidth && text.length > 4) text = text.slice(0, -2) + '…';
      const w = ctx.measureText(text).width + 14, h = 24;
      const baseX = clamp(p.x + (p.x > width / 2 ? 8 : -w - 8), 12, width - w - 12);
      const baseY = clamp(p.y - h / 2, 62, height - 50);
      let x = baseX, y = baseY;
      const overlaps = (xx: number, yy: number) => occupied.some(o => xx < o.x + o.w + 4 && xx + w + 4 > o.x && yy < o.y + o.h + 4 && yy + h + 4 > o.y);
      for (let attempt = 0; overlaps(x, y) && attempt < 24; attempt++) {
        const move = Math.ceil((attempt + 1) / 2) * 27;
        y = clamp(baseY + (attempt % 2 ? -move : move), 58, height - 54);
        if (attempt >= 12) x = clamp(baseX + (baseX < width / 2 ? 1 : -1) * w, 12, width - w - 12);
      }
      occupied.push({x, y, w, h});
      if(label.laserId)hitLabels.push({x,y,w,h,laserId:label.laserId});
      ctx.globalAlpha=(label.alpha??1)*depthOpacity(p.z);
      ctx.fillStyle = label.color; ctx.textBaseline = 'middle'; ctx.fillText(text, x + 7, y + h / 2 + .5);
      ctx.globalAlpha=1;
    });
    // A fixed corner triad is independent of scene scale and makes orbit orientation explicit.
    const triadX = width - 49, triadY = height - 45;
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]).map((v, i) => ({v, i, z: dot(camera.forward, v)})).sort((a, b) => a.z - b.z).forEach(({v, i}) => {
      const x = dot(camera.right, v) * 26, y = -dot(camera.up, v) * 26;
      ctx.strokeStyle = ['#865B59', '#58795D', '#4E6D92'][i]; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(triadX, triadY); ctx.lineTo(triadX + x, triadY + y); ctx.stroke();
      ctx.font = '11px ' + FONT; ctx.textAlign = 'center'; ctx.fillText(['x', 'y', 'z'][i], triadX + x * 1.3, triadY + y * 1.3);
    });
    ctx.textAlign = 'left';
    if (animateField) queueDraw();
  }

  function resize(): void {
    const bounds = host.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    width = bounds.width; height = bounds.height; dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    queueDraw();
  }
  function pinchDistance(): number {
    const list = [...pointers.values()];
    return list.length < 2 ? 0 : Math.hypot(list[1].x - list[0].x, list[1].y - list[0].y);
  }
  function beamCandidates(x:number,y:number):string[]{
    const labels=hitLabels.filter(l=>x>=l.x&&x<=l.x+l.w&&y>=l.y&&y<=l.y+l.h).map(l=>l.laserId);
    const rays=hitRays.map(ray=>{const dx=ray.b.x-ray.a.x,dy=ray.b.y-ray.a.y,t=clamp(((x-ray.a.x)*dx+(y-ray.a.y)*dy)/Math.max(1,dx*dx+dy*dy),0,1);return {id:ray.laserId,d:Math.hypot(x-ray.a.x-t*dx,y-ray.a.y-t*dy),z:rayDepth(ray.a,ray.b,t)};}).filter(r=>r.d<9).sort((a,b)=>Math.abs(a.d-b.d)<2?b.z-a.z:a.d-b.d);
    return [...new Set([...labels,...rays.map(r=>r.id)])];
  }
  canvas.addEventListener('pointerdown', event => {
    pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
    if(pointers.size===1){pointerOrigin={x:event.clientX,y:event.clientY};dragged=false;}else dragged=true;
    canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing'; canvas.focus({preventScroll: true});
  }, eventOptions);
  canvas.addEventListener('pointermove', event => {
    const last = pointers.get(event.pointerId);
    if(!last){
      const rect=canvas.getBoundingClientRect(),hit=beamCandidates(event.clientX-rect.left,event.clientY-rect.top)[0];
      if(hoverLaserId!==hit){hoverLaserId=hit;queueDraw();}
      canvas.style.cursor=hit?'pointer':'grab';canvas.removeAttribute('title');return;
    }
    if(pointerOrigin&&Math.hypot(event.clientX-pointerOrigin.x,event.clientY-pointerOrigin.y)>4)dragged=true;
    const oldDistance = pinchDistance();
    pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
    if (pointers.size > 1) { const nextDistance = pinchDistance(); if (oldDistance > 0 && nextDistance > 0) zoom(oldDistance / nextDistance); }
    else { views.setCurrent('free');yaw -= (event.clientX - last.x) * .008; pitch = clamp(pitch + (event.clientY - last.y) * .007, -1.5, 1.5); queueDraw(); }
  }, eventOptions);
  const stopPointer = (event: PointerEvent) => {
    if(event.type==='pointerup'&&pointers.size===1&&!dragged){
      const rect=canvas.getBoundingClientRect(),candidates=beamCandidates(event.clientX-rect.left,event.clientY-rect.top);
      if(candidates.length)selectLaser(candidates[0]!);
      else container.dispatchEvent(new CustomEvent('atom-clear-selection',{bubbles:true}));
    }
    pointers.delete(event.pointerId); if (!pointers.size) canvas.style.cursor = 'grab';
  };
  canvas.addEventListener('pointerup', stopPointer, eventOptions);
  canvas.addEventListener('pointercancel', stopPointer, eventOptions);
  canvas.addEventListener('lostpointercapture', stopPointer, eventOptions);
  canvas.addEventListener('pointerleave',()=>{if(!pointers.size&&hoverLaserId){hoverLaserId=undefined;queueDraw();}},eventOptions);
  canvas.addEventListener('wheel', event => { event.preventDefault(); zoom(Math.exp(clamp(event.deltaY, -100, 100) * .0015)); }, {passive: false, signal: abort.signal});
  canvas.addEventListener('dblclick', reset, eventOptions);
  canvas.addEventListener('keydown', event => {
    switch (event.key) {
      case 'ArrowLeft': yaw -= .14; break;
      case 'ArrowRight': yaw += .14; break;
      case 'ArrowUp': pitch = clamp(pitch + .12, -1.5, 1.5); break;
      case 'ArrowDown': pitch = clamp(pitch - .12, -1.5, 1.5); break;
      case '+': case '=': zoom(.9); break;
      case '-': zoom(1.1); break;
      case 'Home': reset(); break;
      default: return;
    }
    if(event.key.startsWith('Arrow'))views.setCurrent('free');event.preventDefault(); queueDraw();
  }, eventOptions);
  const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(host);
  window.addEventListener('resize', resize, eventOptions);
  updateStatus(); resize(); queueDraw();
  return {
    update(nextScene, nextAnalysis, nextSelection) {
      currentScene = nextScene; currentAnalysis = nextAnalysis; currentSelection = nextSelection;
      updateStatus(); queueDraw();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true; abort.abort(); resizeObserver.disconnect(); cancelAnimationFrame(animation); depthRenderer.destroy();host.remove();
    },
  };
}
