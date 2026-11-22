// A running motion solve's course (the user, 11-21: "显示目前到计算的流程里面的哪一步了，就像一个进度条或者是
// check point条一样"): a row under the status strip, while the solve runs. It shows progress only: nothing of a
// checkpoint is drawn but its numbers, and the result is drawn once, at the end (or on 停止, the last checkpoint).
//
// What the course knows is what the engine sends (rust-solver/motion/src/solve.rs, # Checkpoints, # Steps):
// - a stage as each step of its pipeline begins -- the model, the scout, the pilot, the selection, every batch of the
//   product -- and as the step's check moves to another rung (solver-types.ts `SolveStage`), in the order it runs
//   them: a new selection after the pilot, and on a chosen model other than the all-ground one the model, the scout
//   and the pilot again before the product; a selection kept for the scene first, at once;
// - a checkpoint, a complete result so far, every 60 s while trajectories run and at every batch.
// So the course is the steps as they came, each done or in progress -- a done step with the time it took, the step in
// progress with the time it has taken so far --, the steps still to come and the result; beside them what the step in
// progress is at (the rungs of its time step, a batch's trajectories), the checkpoints so far, and a gauge of how far
// the run stands from its targets: a batch's own (its looks' largest total over target, the stop's number), before the
// product the last checkpoint's largest population error against the target.
import type {DynamicsResult,SolveStage,SolveStep} from './solver-types.js';
import {SEMICLASSICAL_POPULATION_TARGET,SOLVE_STEPS} from './solver-types.js';
import {errorPercent} from './accuracy.js';
import {formatNumber,withUnit} from './format.js';
import {Keyed,element,reconcileChildren,setText,setTitle} from '../../desktop/ui/src/fluent/dom.js';

/** One checkpoint as the course keeps it: when it came (ms after the request), the trajectories and the time step (μs)
 * it rests on, and its largest estimated population error (absolute) over the outputs. */
export interface CourseMark {atMs:number;trajectories:number;stepUs:number;error:number}
/** A running motion solve's course: when it was requested (performance.now()), its stages and its checkpoints, each
 * when it came, in the order they came. */
export interface Course {startedAt:number;stages:{atMs:number;stage:SolveStage}[];marks:CourseMark[]}

/** A motion checkpoint's mark (none for a result with no motion or no population accuracy: a mean's). */
export function courseMark(result:DynamicsResult,atMs:number):CourseMark|undefined{
  const motion=result.motion,accuracy=result.populationAccuracy;
  return motion&&accuracy?{atMs,trajectories:motion.trajectories,stepUs:motion.timeStepUs,error:accuracy.estimatedMaximumError}:undefined;
}
/** A checkpoint into the course: the engine's last result sent again, when nothing better is in (solve.rs
 * `Messages::again`), is no new checkpoint. */
export function addMark(course:Course,mark:CourseMark):void{
  const last=course.marks.at(-1);
  if(!last||last.trajectories!==mark.trajectories||last.stepUs!==mark.stepUs||last.error!==mark.error)course.marks.push(mark);
}

export type CourseState='done'|'current'|'pending';
/** A node of the course: its words, the line under them and its tooltip. */
export interface CourseNode {key:string;label:string;detail:string;state:CourseState;title:string}
/** The course as shown: its nodes in order; beside them the gauge -- 1 at the targets or within them -- with its words,
 * none before a checkpoint or a batch; and the words on what the step in progress is at. */
export interface CourseView {nodes:CourseNode[];gauge?:{fraction:number;words:string;title:string};words:string}

/** Each step's name and what it does. */
const STEPS:Record<SolveStep,{label:string;does:string}>={
  model:{label:'建模型',does:'按选定的能级折叠其余能级（还没有选定时先用全基态模型），再建初态的准备阱与初始系综'},
  scout:{label:'侦察',does:'先导各组的第一条轨迹，在第一级时间步长和它的两倍（或一半）上跑，检查步长，按代价选先导用的一级'},
  pilot:{label:'先导',does:'先导各组的全部轨迹，在两层步长上跑，检查时间步长，并为选态提供原子走过的路径'},
  select:{label:'选态',does:'沿先导原子走过的路径，把各候选模型与 32 态参照比较，在误差达标的候选里选总代价最小的；同一场景只改时长或采样点时沿用之前的选择'},
  product:{label:'正式批次',does:'每批至多把轨迹数翻倍，直到布居与总温度在各分析时刻都到目标'},
};

/** A wall time as the course counts it: "42 s", "3 min 5 s", "1 h 12 min" (whole seconds, down). */
export function clockWords(ms:number):string{
  const s=Math.max(0,Math.floor(ms/1000));
  if(s<60)return `${s} s`;
  if(s<3600)return `${Math.floor(s/60)} min ${s%60} s`;
  return `${Math.floor(s/3600)} h ${Math.floor(s%3600/60)} min`;
}
/** When the course's clocks next change, from `ms` after the request: at the next whole second of the step in
 * progress, which began `sinceMs` after the request. */
export function clockChangesInMs(ms:number,sinceMs=0):number{return 1000-Math.max(0,ms-sinceMs)%1000;}
/** A time step as 详情 says it. */
function stepWords(stepUs:number):string{return withUnit(stepUs,'μs',{digits:3});}
/** The rungs a step ran at, in order: "3.33 → 1.67 μs". */
function rungWords(stages:readonly SolveStage[]):string{
  const rungs=stages.map(s=>s.stepUs).filter((t,i,all):t is number=>t!==undefined&&t!==all[i-1]);
  return rungs.length?`${rungs.map(t=>formatNumber(t,{digits:3})).join(' → ')} μs`:'';
}

/** A step of the course as it came: a run of stages of one step, from its first stage's arrival -- the first step's
 * from the request -- to the next step's; the step again after another (a second round's model, scout and pilot) is
 * another. Keyed by its step, the second time and on with its count. */
interface Reached {key:string;step:SolveStep;fromMs:number;stages:SolveStage[]}
function reached(course:Course):Reached[]{
  const steps:Reached[]=[];
  for(const {atMs,stage} of course.stages){
    const last=steps.at(-1);if(last?.step===stage.step){last.stages.push(stage);continue;}
    const before=steps.filter(s=>s.step===stage.step).length;
    steps.push({key:before?`${stage.step}-${before+1}`:stage.step,step:stage.step,fromMs:steps.length?atMs:0,stages:[stage]});
  }
  return steps;
}
/** The steps as they came, the last in progress; before any stage is in, the first step in their order (the model)
 * from the request on. */
function reachedSoFar(course:Course):Reached[]{
  const steps=reached(course),first=SOLVE_STEPS[0];
  return steps.length?steps:[{key:first,step:first,fromMs:0,stages:[]}];
}

/** The step in progress and when it began (ms after the request): its own run of stages, not the step's first time;
 * the model from the request on, before any stage is in. */
export function currentStep(course:Course):{step:SolveStep;sinceMs:number}{
  const now=reachedSoFar(course).at(-1)!;
  return {step:now.step,sinceMs:now.fromMs};
}

/** The course at `nowMs` (performance.now()): the steps as they came, the last in progress; then, in their order
 * (`SOLVE_STEPS`), those after the one in progress that have not come -- the product last until it comes --; then the
 * result. */
export function courseView(course:Course,nowMs:number):CourseView{
  const elapsed=nowMs-course.startedAt,steps=reachedSoFar(course),current=steps.at(-1)!,now=current.step;
  const nodes:CourseNode[]=steps.map((s,i):CourseNode=>{
    const words=STEPS[s.step],rungs=rungWords(s.stages),done=i<steps.length-1;
    const took=clockWords(done?steps[i+1]!.fromMs-s.fromMs:elapsed-s.fromMs);
    const when=`（开始后 ${clockWords(s.fromMs)} 开始，${done?'用时':'已用'} ${took}${rungs?`；时间步长 ${rungs}`:''}）`;
    return {key:s.key,label:words.label,detail:done?took:`已用 ${took}`,state:done?'done':'current',title:`${words.label}：${words.does}${when}`};
  });
  for(const step of SOLVE_STEPS.slice(SOLVE_STEPS.indexOf(now)+1))if(!steps.some(s=>s.step===step))
    nodes.push({key:step,label:STEPS[step].label,detail:'',state:'pending',title:`${STEPS[step].label}：${STEPS[step].does}`});
  nodes.push({key:'result',label:'结果',state:'pending',detail:'',title:'结果：布居与温度都到目标时引擎停下，最终结果一次画出；按“停止”时画最后一个检查点。'});
  // What the step in progress is at, and the checkpoints so far.
  const stages=current.stages,batch=now==='product'?stages.at(-1)?.batch:undefined,marks=course.marks,last=marks.at(-1);
  const doing=batch?`第 ${batch.number} 批：${batch.trajectories} → ${batch.want} 条轨迹 · 时间步长 ${stepWords(stages.at(-1)!.stepUs!)}`
    :now==='scout'||now==='pilot'?`${STEPS[now].label} · 时间步长 ${rungWords(stages)}`:'';
  const words=[doing,marks.length?`检查点 ${marks.length}`:'尚无检查点'].filter(Boolean).join(' · ');
  // How far the run stands from its targets: a batch's looks, else the last checkpoint's populations.
  const target=SEMICLASSICAL_POPULATION_TARGET;
  const gauge=batch?.worst!==undefined?{fraction:Math.min(1,1/batch.worst),words:`误差最大的一项是目标的 ${formatNumber(batch.worst,{digits:2})} 倍`,
      title:'这一批开始时，各分析时刻的布居与总温度里，总误差（模型、步长与统计）除以目标的最大值；各项都不超过 1 时引擎停下'}
    :last?{fraction:Math.min(1,target/last.error),words:`布居误差 ${errorPercent(last.error)}，目标 ${errorPercent(target)}`,
      title:`检查点 ${marks.length}（开始后 ${clockWords(last.atMs)}）：${last.trajectories} 条轨迹，时间步长 ${stepWords(last.stepUs)}；各输出时刻布居的估计误差最大 ${errorPercent(last.error)}，目标 ${errorPercent(target)}`}
    :undefined;
  return {nodes,...(gauge?{gauge}:{}),words};
}

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
