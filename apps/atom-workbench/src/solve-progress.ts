// A running motion solve's course (the user, 11-21: "显示目前到计算的流程里面的哪一步了，就像一个进度条或者是
// check point条一样"): a row of checkpoints under the status strip, while the solve runs. It shows progress only:
// nothing of a checkpoint is drawn but its numbers, and the result is drawn once, at the end (or on 停止, the last
// checkpoint).
//
// What the course knows is what the engine sends (rust-solver/motion/src/solve.rs, # Checkpoints): a complete result
// every 60 s while trajectories run (the scout, the pilot, a batch), and one at every batch of the product -- none
// while the run is made (the scene, the three-point selection, the fold), since nothing runs trajectories there.
// So the course has an opening -- from the request to its first checkpoint: the model made, the states selected, the
// scout and the pilot begun, the first checkpoint at the latest when the product begins --, the checkpoints in the
// order they came, each with its largest estimated population error, and the run going on to its result. Beside it a
// gauge says how far the last checkpoint's error is from the target, and on what it rests: the trajectories and the
// time step.
import type {DynamicsResult} from './solver-types.js';
import {SEMICLASSICAL_POPULATION_TARGET} from './solver-types.js';
import {errorPercent} from './accuracy.js';
import {formatNumber,withUnit} from './format.js';
import {Keyed,element,reconcileChildren,setText,setTitle} from '../../desktop/ui/src/fluent/dom.js';

/** One checkpoint as the course keeps it: when it came (ms after the request), the trajectories and the time step (μs)
 * it rests on, and its largest estimated population error (absolute) over the outputs. */
export interface CourseMark {atMs:number;trajectories:number;stepUs:number;error:number}
/** A running motion solve's course: when it was requested (performance.now()) and its checkpoints in the order they came. */
export interface Course {startedAt:number;marks:CourseMark[]}

/** A motion checkpoint's mark (none for a result with no motion or no population accuracy: a mean's). */
export function courseMark(result:DynamicsResult,atMs:number):CourseMark|undefined{
  const motion=result.motion,accuracy=result.populationAccuracy;
  return motion&&accuracy?{atMs,trajectories:motion.trajectories,stepUs:motion.timeStepUs,error:accuracy.estimatedMaximumError}:undefined;
}

/** The checkpoints the course shows by themselves: the first and the last ones, the rest between them as one. */
export const SHOWN_CHECKPOINTS=4;
export type CourseState='done'|'current'|'pending';
/** A node of the course: its words, the line under them and its tooltip. */
export interface CourseNode {key:string;label:string;detail:string;state:CourseState;title:string}
/** The course as shown: its nodes in order, and beside them the gauge -- the target over the last checkpoint's error,
 * 1 at the target or within it -- with its words and the words on what it rests; no gauge before a checkpoint. */
export interface CourseView {nodes:CourseNode[];gauge?:{fraction:number;words:string;title:string};words:string}

/** A wall time as the course counts it: "42 s", "3 min 5 s", "1 h 12 min" (whole seconds, down). */
export function clockWords(ms:number):string{
  const s=Math.max(0,Math.floor(ms/1000));
  if(s<60)return `${s} s`;
  if(s<3600)return `${Math.floor(s/60)} min ${s%60} s`;
  return `${Math.floor(s/3600)} h ${Math.floor(s%3600/60)} min`;
}
/** When the course's clock next changes, from `ms` after the request: at the next whole second. */
export function clockChangesInMs(ms:number):number{return 1000-Math.max(0,ms)%1000;}

/** The course at `nowMs` (performance.now()). */
export function courseView(course:Course,nowMs:number):CourseView{
  const marks=course.marks,elapsed=nowMs-course.startedAt,target=SEMICLASSICAL_POPULATION_TARGET,last=marks.at(-1);
  const opening:CourseNode={key:'opening',label:'开头',state:last?'done':'current',detail:last?clockWords(marks[0]!.atMs):`已用 ${clockWords(elapsed)}`,
    title:'从开始到第一个检查点：引擎先建模型、选态，再跑侦察与先导（检查时间步长）；第一个检查点最晚在正式批次开始时到来，在此之前没有中间结果。'};
  const checkpoint=(mark:CourseMark,i:number):CourseNode=>({key:`checkpoint-${i}`,label:`检查点 ${i+1}`,detail:errorPercent(mark.error),state:'done',
    title:`检查点 ${i+1}（开始后 ${clockWords(mark.atMs)}）：${mark.trajectories} 条轨迹，时间步长 ${stepWords(mark.stepUs)}；布居的估计误差最大 ${errorPercent(mark.error)}，目标 ${errorPercent(target)}`});
  // The first checkpoint and the last ones by themselves, those between them as one.
  const hidden=marks.length-SHOWN_CHECKPOINTS;
  const shown=hidden>0?[checkpoint(marks[0]!,0),{key:'hidden',label:'…',detail:`${hidden+1} 个`,state:'done' as const,title:`检查点 2–${hidden+2}`},
    ...marks.slice(-(SHOWN_CHECKPOINTS-2)).map((mark,k)=>checkpoint(mark,marks.length-(SHOWN_CHECKPOINTS-2)+k))]:marks.map(checkpoint);
  const running:CourseNode[]=last?[{key:'running',label:'计算中',state:'current',detail:`已用 ${clockWords(elapsed)}`,title:'引擎在加轨迹（侦察、先导或正式批次）：下一个检查点或最终结果到来之前不画中间结果。'}]:[];
  const result:CourseNode={key:'result',label:'结果',state:'pending',detail:'',title:'布居与温度都到目标时引擎停下，最终结果一次画出；按“停止”时画最后一个检查点。'};
  if(!last)return {nodes:[opening,result],words:'尚无检查点'};
  return {nodes:[opening,...shown,...running,result],
    gauge:{fraction:Math.min(1,target/last.error),words:`布居误差 ${errorPercent(last.error)}，目标 ${errorPercent(target)}`,
      title:last.error>target?`最后一个检查点的布居估计误差是目标的 ${formatNumber(last.error/target,{digits:2})} 倍`:'最后一个检查点的布居估计误差已在目标内；引擎在布居与温度都到目标时停下'},
    words:`${last.trajectories} 条轨迹 · 时间步长 ${stepWords(last.stepUs)}`};
}
/** The time step as 详情 says it. */
function stepWords(stepUs:number):string{return withUnit(stepUs,'μs',{digits:3});}

/** One node's view: a dot, its words and the line under them. */
class NodeView {
  readonly root=element('li','course-node');
  private readonly dot=element('span','course-dot');
  private readonly label=element('span','course-label');
  private readonly detail=element('span','course-detail');
  constructor(){this.dot.setAttribute('aria-hidden','true');this.root.append(this.dot,this.label,this.detail);}
  show(node:CourseNode):void{
    if(this.root.dataset.state!==node.state)this.root.dataset.state=node.state;
    setText(this.label,node.label);setText(this.detail,node.detail);setTitle(this.root,node.title);
    if(node.state==='current')this.root.setAttribute('aria-current','step');else this.root.removeAttribute('aria-current');
  }
}

/** The course's row (`#solve-progress`): its nodes joined by links -- a link done up to the node in progress --, the
 * gauge and its words. Hidden with no course. Updated in place: a node keeps its tooltip while the clock runs. */
export class CourseStrip {
  readonly root=element('div','solve-progress');
  private readonly list=element('ol','course');
  private readonly gauge=element('div','course-gauge');
  private readonly fill=element('span','course-fill');
  private readonly gaugeWords=element('span','course-gauge-words');
  private readonly words=element('span','course-words');
  private readonly nodes=new Keyed<NodeView>(()=>new NodeView());
  private readonly links=new Keyed<{root:HTMLElement}>(()=>{const root=element('li','course-link');root.setAttribute('aria-hidden','true');return {root};});
  constructor(){
    this.root.hidden=true;this.list.setAttribute('aria-label','计算流程');
    this.gauge.setAttribute('role','meter');this.gauge.setAttribute('aria-valuemin','0');this.gauge.setAttribute('aria-valuemax','1');this.gauge.append(this.fill);
    this.root.append(this.list,this.gauge,this.gaugeWords,this.words);
  }
  show(view:CourseView|undefined):void{
    this.root.hidden=!view;if(!view)return;
    const nodes=this.nodes.sync(view.nodes.map(n=>n.key)),links=this.links.sync(view.nodes.slice(1).map(n=>n.key));
    view.nodes.forEach((n,i)=>nodes[i]!.show(n));
    // A link is done when the node it leads to is reached or in progress.
    links.forEach((link,i)=>link.root.classList.toggle('done',view.nodes[i+1]!.state!=='pending'));
    reconcileChildren(this.list,view.nodes.flatMap((_,i)=>i?[links[i-1]!.root,nodes[i]!.root]:[nodes[i]!.root]));
    const gauge=view.gauge;this.gauge.hidden=!gauge;this.gaugeWords.hidden=!gauge;
    if(gauge){
      this.fill.style.width=`${100*gauge.fraction}%`;this.gauge.setAttribute('aria-valuenow',String(gauge.fraction));this.gauge.setAttribute('aria-valuetext',gauge.words);
      setText(this.gaugeWords,gauge.words);setTitle(this.gauge,gauge.title);setTitle(this.gaugeWords,gauge.title);
    }
    setText(this.words,view.words);
  }
}
