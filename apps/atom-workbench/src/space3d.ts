import type { Analysis, Laser, Polarization, Scene, Vec3 } from './types.js';
import {depthOpacity,splitSegment,rayDepth} from './space-depth.js';
import {SpaceDepthRenderer} from './space-depth-renderer.js';
import {arrowGeometry,shadeArrow} from './space-arrow.js';
import {direction} from './physics.js';
import {identityToken,resolveColour} from './identity.js';
import {withUnit} from './format.js';

/** What the window has selected: the 3D view shows it, never a selection of its own. */
export type SpaceSelection={kind:'laser'|'field'|'none';id:string};
/** 3D: the free camera; x, y, z: along a lab axis; B: along the quantization axis; k: along the selected beam. */
export type SpaceViewName='free'|'x'|'y'|'z'|'B'|'k';

/** A retained, camera-controlled view. update() never resets the camera. */
export interface SpaceView {
  update(scene: Scene, analysis: Analysis, selection: SpaceSelection): void;
  setView(name: SpaceViewName): void;
  setAnimation(on: boolean): void;
  setPlane(on: boolean): void;
  zoomBy(factor: number): void;
  reset(): void;
  destroy(): void;
}
export type SpaceOptions={
  /** The camera left a named view (a drag, the keyboard) or took one. */
  onViewChange?(name: SpaceViewName): void;
};

type Point = { x: number; y: number; z: number; scale: number };
type Segment = { a: Vec3; b: Vec3; color: string; width: number; alpha?: number };
type Face = { vertices: Vec3[]; color: string; alpha: number };
type Target={kind:'laser'|'field';id:string};
type Label = { p: Vec3; text: string; color: string; alpha?:number; strong?: boolean; rank: number; target?: Target };
type HitRay={a:Point;b:Point;target:Target};
type HitLabel={x:number;y:number;w:number;h:number;target:Target};
const TAU = 2 * Math.PI, DEG = Math.PI / 180;
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, v: number): Vec3 => [a[0] * v, a[1] * v, a[2] * v];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clamp = (n: number, a: number, b: number): number => Math.min(b, Math.max(a, n));
const ZERO: Vec3 = [0, 0, 0];
const FONT = '"Segoe UI", "Microsoft YaHei UI", sans-serif';
const HINT_KEY='atom-space-hint-seen';
const FREE={yaw:-1.03,pitch:.5};

/** The same real Jones field used by physics.ts, with a normalized optical phase.
 * Positive chi traverses u toward v for epsilon exp(-i omega t). */
export function electricFieldVector(laser:Laser,polarization:Polarization,phase:number):Vec3 {
  const chi=laser.ellipticityDeg*DEG,t=phase-(laser.phaseDeg??0)*DEG;
  return add(mul(polarization.u,Math.cos(chi)*Math.cos(t)),mul(polarization.v,Math.sin(chi)*Math.sin(t)));
}

/** The camera on the +d side of the atom, looking back along -d (d toward the
 * viewer); near a pole x stays to the right. */
export function cameraAlong(d:Vec3):{yaw:number;pitch:number}{
  const length=Math.hypot(...d);if(!(length>0))return {...FREE};
  const u=mul(d,1/length),pitch=Math.asin(clamp(u[2],-1,1));
  return {pitch:clamp(pitch,-Math.PI/2,Math.PI/2),yaw:Math.hypot(u[0],u[1])<1e-9?-Math.PI/2:Math.atan2(u[1],u[0])};
}

/**
 * True perspective geometry with per-pixel depth-tested 3D primitives, deliberately local
 * and dependency-free so the same view works in the offline single-file demo.
 * World +z is up. The selected beam's polarization ellipse is drawn at the atom in its
 * transverse u,v plane, never rotated to face the camera; an arrow on it gives the sense
 * of rotation. E(t) = Re[(cos(chi) u + i sin(chi) v) exp(i phi - i omega t)].
 * Beam width and vector lengths are schematic, not intensity / magnetic-field scales.
 */
export function renderSpace3D(container: HTMLElement, scene: Scene, analysis: Analysis, selection: SpaceSelection, options: SpaceOptions = {}): SpaceView {
  let currentScene = scene, currentAnalysis = analysis, currentSelection = selection;
  let width = 760, height = 480, dpr = 1, yaw = FREE.yaw, pitch = FREE.pitch, distance = 7.6, viewName:SpaceViewName='free';
  let animation = 0, destroyed = false, animateField = false, phase = 0, previousTime = 0;
  let showPlane=true;
  let hitRays:HitRay[]=[],hitLabels:HitLabel[]=[],hover:Target|undefined;
  let pointerOrigin:{x:number;y:number}|undefined,dragged=false;
  const abort = new AbortController(), eventOptions = { signal: abort.signal };
  const pointers = new Map<number, {x: number; y: number}>();

  const host = document.createElement('div');
  host.className = 'atom-space3d';
  const depthCanvas=document.createElement('canvas');depthCanvas.className='space-depth-canvas';depthCanvas.setAttribute('aria-hidden','true');
  const depthRenderer=new SpaceDepthRenderer(depthCanvas);
  const canvas = document.createElement('canvas');canvas.className='space-canvas';
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', '可旋转的三维实验空间：全部启用的激光与磁场、选中光束的偏振和原子。方向键旋转，加减键缩放，Home 键复位。');
  const context = canvas.getContext('2d');
  if (!context) throw new Error('浏览器无法创建三维示意画布。');
  const ctx = context;
  // Shown until the camera is first moved, then never again.
  const hint = document.createElement('div');hint.className='space-hint';hint.textContent='拖动旋转 · 滚轮缩放';
  let hintSeen=false;try{hintSeen=localStorage.getItem(HINT_KEY)==='1';}catch{}
  hint.hidden=hintSeen;
  const used=()=>{if(hintSeen)return;hintSeen=true;hint.hidden=true;try{localStorage.setItem(HINT_KEY,'1');}catch{}};
  host.append(depthCanvas,canvas,hint);
  container.replaceChildren(host);

  const colours=()=>({
    field:resolveColour(host,'--field-ink'),muted:resolveColour(host,'--muted-ink'),text:resolveColour(host,'--text'),surface:resolveColour(host,'--surface'),
    axis:resolveColour(host,'--neutral-fill'),plane:resolveColour(host,'--selection-edge'),
    triad:[resolveColour(host,'--axis-x'),resolveColour(host,'--axis-y'),resolveColour(host,'--axis-z')],
  });
  const selectedLaser=()=>currentSelection.kind==='laser'?currentScene.lasers.find(l=>l.id===currentSelection.id&&l.enabled):undefined;
  const directionFor=(name:SpaceViewName):Vec3|undefined=>{
    if(name==='B')return currentAnalysis.fieldMagnitudeG>0?currentAnalysis.fieldVector:[0,0,1];
    if(name==='k'){const laser=selectedLaser(),pol=laser?currentAnalysis.polarizations[laser.id]:undefined;return pol?.k;}
    return undefined;
  };
  function applyView(name:SpaceViewName):void{
    viewName=name;
    if(name==='free'){yaw=FREE.yaw;pitch=FREE.pitch;}
    // The lab-axis views keep their plane in its natural orientation (y right of x, z up).
    else if(name==='z'){yaw=-Math.PI/2;pitch=Math.PI/2;}
    else if(name==='y'){yaw=-Math.PI/2;pitch=0;}
    else if(name==='x'){yaw=0;pitch=0;}
    else{const d=directionFor(name);if(d)({yaw,pitch}=cameraAlong(d));else viewName='free';}
    queueDraw();
  }
  function leaveView():void{if(viewName!=='free'){viewName='free';options.onViewChange?.('free');}}
  function reset(): void { distance = 7.6; phase = 0; applyView('free'); options.onViewChange?.('free'); }
  function zoom(factor: number): void { distance = clamp(distance * factor, 4.5, 13); queueDraw(); }
  function queueDraw(): void { if (!animation && !destroyed) animation = requestAnimationFrame(draw); }
  function frame() {
    const forward: Vec3 = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
    const right: Vec3 = [-Math.sin(yaw), Math.cos(yaw), 0];
    const up: Vec3 = [-Math.sin(pitch) * Math.cos(yaw), -Math.sin(pitch) * Math.sin(yaw), Math.cos(pitch)];
    const focal = Math.min(width, height * 1.48) * 1.06;
    return { forward, right, up, focal };
  }
  const same=(a:Target|undefined,b:Target|SpaceSelection|undefined)=>Boolean(a&&b&&a.kind===b.kind&&a.id===b.id);
  function choose(target:Target):void{
    container.dispatchEvent(new CustomEvent(target.kind==='laser'?'atom-select-laser':'atom-select-field',{bubbles:true,detail:{id:target.id}}));
  }
  function draw(time: number): void {
    animation = 0;
    if (destroyed || !host.isConnected) return;
    if (animateField && previousTime) phase = (phase + Math.min(time - previousTime, 60) * .0018) % TAU;
    previousTime = time;
    const ink=colours();
    canvas.dataset.camera=JSON.stringify({yaw,pitch,distance,view:viewName});canvas.dataset.selected=currentSelection.kind==='none'?'':`${currentSelection.kind}:${currentSelection.id}`;
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
        depthRenderer.line(start,end,strokeWidth*(.85+.2*depthOpacity(depth)),color,alpha*depthOpacity(depth));
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
      for(let i=0;i<4;i++)for(let j=0;j<4;j++)face({vertices:[[steps[i]!,steps[j]!,0],[steps[i+1]!,steps[j]!,0],[steps[i+1]!,steps[j+1]!,0],[steps[i]!,steps[j+1]!,0]],color:ink.plane,alpha:.09});
      for(const v of steps){segment({a:[v,-1.6,0],b:[v,1.6,0],color:ink.plane,width:.65,alpha:.45});segment({a:[-1.6,v,0],b:[1.6,v,0],color:ink.plane,width:.65,alpha:.45});}
    }
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]).forEach(axis => segment({a: mul(axis, -1.65), b: mul(axis,1.65), color: ink.axis, width: .7, alpha: .55}));

    const anySelected=currentSelection.kind!=='none';
    currentScene.lasers.forEach((laser,slot) => {
      const pol = laser.enabled?currentAnalysis.polarizations[laser.id]:undefined;
      if (!pol) return;
      const target:Target={kind:'laser',id:laser.id},selected=same(target,currentSelection),hovered=same(target,hover);
      const color=resolveColour(host,identityToken(slot)),alpha=selected?1:hovered?.9:anySelected?.32:.75,lineWidth=selected?2.2:1.1;
      const {k, u, v} = pol, start = mul(k, -2.05), end = mul(k, 1.95);
      // A translucent, narrow cylinder makes a beam readable from every viewpoint.
      const side = (t: number): Vec3 => add(mul(u, Math.cos(t) * (selected?.085:.05)), mul(v, Math.sin(t) * (selected?.085:.05)));
      for (let n = 0; n < 8; n++) {
        const a = side(n * TAU / 8), b = side((n + 1) * TAU / 8);
        for(const [lo,hi]of splitSegment(start,end,.34))face({vertices: [add(lo, a), add(hi, a), add(hi, b), add(lo, b)], color, alpha: selected ? .052 : anySelected?.008:.02});
      }
      segment({a: start, b: mul(k, -.08), color, width: lineWidth, alpha});
      arrow(mul(k, .1), end, color, lineWidth, alpha,u);
      // Short k-direction arrows on the incoming half remove ambiguity about propagation.
      arrow(mul(k, -1.9), mul(k, -1.6), color, selected ? 1.8 : 1, alpha,u);
      hitRays.push({a:project(start),b:project(end),target});
      if(selected||hovered)labels.push({p:mul(k,2.13),text:laser.name,color,strong:selected,rank:selected?1:3,target});
      if (selected) {
        if(showPlane&&Math.abs(end[2])>.08){const foot:Vec3=[end[0],end[1],0];segment({a:end,b:foot,color,width:.8,alpha:.45});segment({a:add(foot,[-.06,0,0]),b:add(foot,[.06,0,0]),color,width:1,alpha:.6});segment({a:add(foot,[0,-.06,0]),b:add(foot,[0,.06,0]),color,width:1,alpha:.6});}
        // The polarization at the atom: its plane, the ellipse E traces, E now, and the sense of rotation.
        const chi = laser.ellipticityDeg * DEG, radius = .68, center=ZERO;
        const disc: Vec3[] = Array.from({length: 60}, (_, j) => add(center, add(mul(u, radius * Math.cos(j * TAU / 60)), mul(v, radius * Math.sin(j * TAU / 60)))));
        face({vertices: disc, color, alpha: .06});
        curve([...disc, disc[0]!], color, .8, .35);
        segment({a: mul(u, -radius), b: mul(u, radius), color, width: .8, alpha: .22});
        segment({a: mul(v, -radius), b: mul(v, radius), color, width: .8, alpha: .22});
        const fieldAt = (t: number): Vec3 => mul(electricFieldVector(laser,pol,t),radius);
        const ellipse=Math.abs(chi)<1e-12?[mul(u,-radius),mul(u,radius)]:Array.from({length:97},(_,j)=>fieldAt(j*TAU/96));
        curve(ellipse, color, 2.2,1);
        arrow(center,fieldAt(phase),color,2.3,1,k);
        // With epsilon exp(i phi - i omega t), E advances u→v for positive chi.
        arrow(fieldAt(phase + .65),fieldAt(phase + .87),color,1.7,Math.min(1,Math.abs(chi)/.08),k);
        labels.push({p:mul(u,radius*1.17),text:'E',color,strong:true,rank:2,target});
      }
    });

    // Each field its own arrow; the resultant (the quantization axis) bold.
    const fields=currentScene.fields.filter(f=>f.enabled&&f.magnitudeG>0),total=currentAnalysis.fieldMagnitudeG;
    const longest=Math.max(total,...fields.map(f=>f.magnitudeG));
    if(fields.length>1)for(const field of fields){
      const target:Target={kind:'field',id:field.id},selected=same(target,currentSelection),hovered=same(target,hover);
      const end=mul(direction(field.thetaDeg,field.phiDeg),1.42*field.magnitudeG/longest),start=mul(end,.06);
      arrow(start,end,ink.field,selected?2.2:1.2,selected?1:hovered?.9:.55);
      hitRays.push({a:project(start),b:project(end),target});
      if(selected||hovered)labels.push({p:mul(end,1.1),text:`${field.name}  ${withUnit(field.magnitudeG,'G',{digits:4})}`,color:ink.field,strong:selected,rank:selected?1:3,target});
    }
    if (total > 0) {
      const end = mul(currentAnalysis.fieldVector,1.42/longest),start=mul(end,.06);
      const only=fields.length===1?{kind:'field' as const,id:fields[0]!.id}:undefined;
      arrow(start, end, ink.field, 2.9, only&&anySelected&&!same(only,currentSelection)&&!same(only,hover)?.75:1);
      if(only)hitRays.push({a:project(start),b:project(end),target:only});
      labels.push({p: mul(end, 1.11), text: `${fields.length>1?'合成 B':'B'}  ${withUnit(total,'G',{digits:4})}`, color: ink.field, strong: true, rank: 0, target:only});
    }
    const origin = project(ZERO);
    depthRenderer.circle(origin,9,ink.surface);depthRenderer.circle(origin,5.5,ink.text);
    depthRenderer.flush();
    // Text is screen-facing; geometric anchors retain their true 3D coordinates.
    // Resolve label collisions locally rather than hiding one of two collinear beams.
    const occupied: {x: number; y: number; w: number; h: number}[] = [];
    labels.sort((a, b) => a.rank - b.rank).forEach(label => {
      const p = project(label.p), size = label.rank<=2?13:12;
      ctx.font = `${label.strong ? 600 : 500} ${size}px ${FONT}`;
      const maxLabelWidth = Math.max(80, width * .34);
      let text = label.text;
      while (ctx.measureText(text).width > maxLabelWidth && text.length > 4) text = text.slice(0, -2) + '…';
      const w = ctx.measureText(text).width + 14, h = 24;
      const baseX = clamp(p.x + (p.x > width / 2 ? 8 : -w - 8), 12, width - w - 12);
      const baseY = clamp(p.y - h / 2, 12, height - 50);
      let x = baseX, y = baseY;
      const overlaps = (xx: number, yy: number) => occupied.some(o => xx < o.x + o.w + 4 && xx + w + 4 > o.x && yy < o.y + o.h + 4 && yy + h + 4 > o.y);
      for (let attempt = 0; overlaps(x, y) && attempt < 24; attempt++) {
        const move = Math.ceil((attempt + 1) / 2) * 27;
        y = clamp(baseY + (attempt % 2 ? -move : move), 12, height - 54);
        if (attempt >= 12) x = clamp(baseX + (baseX < width / 2 ? 1 : -1) * w, 12, width - w - 12);
      }
      occupied.push({x, y, w, h});
      if(label.target)hitLabels.push({x,y,w,h,target:label.target});
      ctx.globalAlpha=(label.alpha??1)*depthOpacity(p.z);
      ctx.fillStyle = label.color; ctx.textBaseline = 'middle'; ctx.fillText(text, x + 7, y + h / 2 + .5);
      ctx.globalAlpha=1;
    });
    // A fixed corner triad is independent of scene scale and makes orbit orientation explicit.
    const triadX = width - 49, triadY = height - 45;
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]).map((v, i) => ({v, i, z: dot(camera.forward, v)})).sort((a, b) => a.z - b.z).forEach(({v, i}) => {
      const x = dot(camera.right, v) * 26, y = -dot(camera.up, v) * 26;
      ctx.strokeStyle = ink.triad[i]!; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(triadX, triadY); ctx.lineTo(triadX + x, triadY + y); ctx.stroke();
      ctx.font = '11px ' + FONT; ctx.textAlign = 'center'; ctx.fillText(['x', 'y', 'z'][i]!, triadX + x * 1.3, triadY + y * 1.3);
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
    return list.length < 2 ? 0 : Math.hypot(list[1]!.x - list[0]!.x, list[1]!.y - list[0]!.y);
  }
  /** What is under the pointer: labels first, then the nearest ray (the front one of two equally near). */
  function candidates(x:number,y:number):Target[]{
    const labels=hitLabels.filter(l=>x>=l.x&&x<=l.x+l.w&&y>=l.y&&y<=l.y+l.h).map(l=>l.target);
    const rays=hitRays.map(ray=>{const dx=ray.b.x-ray.a.x,dy=ray.b.y-ray.a.y,t=clamp(((x-ray.a.x)*dx+(y-ray.a.y)*dy)/Math.max(1,dx*dx+dy*dy),0,1);return {target:ray.target,d:Math.hypot(x-ray.a.x-t*dx,y-ray.a.y-t*dy),z:rayDepth(ray.a,ray.b,t)};}).filter(r=>r.d<9).sort((a,b)=>Math.abs(a.d-b.d)<2?b.z-a.z:a.d-b.d);
    const seen=new Set<string>();
    return [...labels,...rays.map(r=>r.target)].filter(t=>{const key=`${t.kind}:${t.id}`;if(seen.has(key))return false;seen.add(key);return true;});
  }
  canvas.addEventListener('pointerdown', event => {
    pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
    if(pointers.size===1){pointerOrigin={x:event.clientX,y:event.clientY};dragged=false;}else dragged=true;
    canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing'; canvas.focus({preventScroll: true});
  }, eventOptions);
  canvas.addEventListener('pointermove', event => {
    const last = pointers.get(event.pointerId);
    if(!last){
      const rect=canvas.getBoundingClientRect(),hit=candidates(event.clientX-rect.left,event.clientY-rect.top)[0];
      if(!same(hit,hover)&&(hit||hover)){hover=hit;queueDraw();}
      canvas.style.cursor=hit?'pointer':'grab';return;
    }
    if(pointerOrigin&&Math.hypot(event.clientX-pointerOrigin.x,event.clientY-pointerOrigin.y)>4)dragged=true;
    const oldDistance = pinchDistance();
    pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
    if (pointers.size > 1) { const nextDistance = pinchDistance(); if (oldDistance > 0 && nextDistance > 0) zoom(oldDistance / nextDistance); used(); }
    else if(dragged){ leaveView();used();yaw -= (event.clientX - last.x) * .008; pitch = clamp(pitch + (event.clientY - last.y) * .007, -1.5, 1.5); queueDraw(); }
  }, eventOptions);
  const stopPointer = (event: PointerEvent) => {
    // A click on a beam or a field selects it; a click on empty space changes nothing.
    if(event.type==='pointerup'&&pointers.size===1&&!dragged){
      const rect=canvas.getBoundingClientRect(),hit=candidates(event.clientX-rect.left,event.clientY-rect.top)[0];
      if(hit)choose(hit);
    }
    pointers.delete(event.pointerId); if (!pointers.size) canvas.style.cursor = 'grab';
  };
  canvas.addEventListener('pointerup', stopPointer, eventOptions);
  canvas.addEventListener('pointercancel', stopPointer, eventOptions);
  canvas.addEventListener('lostpointercapture', stopPointer, eventOptions);
  canvas.addEventListener('pointerleave',()=>{if(!pointers.size&&hover){hover=undefined;queueDraw();}},eventOptions);
  canvas.addEventListener('wheel', event => { event.preventDefault(); used(); zoom(Math.exp(clamp(event.deltaY, -100, 100) * .0015)); }, {passive: false, signal: abort.signal});
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
    if(event.key.startsWith('Arrow'))leaveView();event.preventDefault(); queueDraw();
  }, eventOptions);
  const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(host);
  window.addEventListener('resize', resize, eventOptions);
  resize(); queueDraw();
  return {
    update(nextScene, nextAnalysis, nextSelection) {
      currentScene = nextScene; currentAnalysis = nextAnalysis; currentSelection = nextSelection;
      // 沿 B / 沿 k follow the axis they name; 沿 k without a selected beam is not a view.
      if(viewName==='B'||viewName==='k'){const d=directionFor(viewName);if(d)({yaw,pitch}=cameraAlong(d));else{viewName='free';options.onViewChange?.('free');}}
      queueDraw();
    },
    setView(name){applyView(name);if(viewName!==name)options.onViewChange?.(viewName);},
    setAnimation(on){animateField=on;previousTime=0;queueDraw();},
    setPlane(on){showPlane=on;queueDraw();},
    zoomBy(factor){used();zoom(factor);},
    reset,
    destroy() {
      if (destroyed) return;
      destroyed = true; abort.abort(); resizeObserver.disconnect(); cancelAnimationFrame(animation); depthRenderer.destroy();host.remove();
    },
  };
}
