import { SpinBox } from '../../desktop/ui/src/fluent/spin.js';
import { form, row as fieldRow, sectionTitle, shareLabelColumn } from '../../desktop/ui/src/fluent/form.js';
import { prompt } from '../../desktop/ui/src/fluent/dialog.js';
import { command } from '../../desktop/ui/src/fluent/button.js';
import { reconcileChildren } from '../../desktop/ui/src/fluent/dom.js';
import { comboFace } from '../../desktop/ui/src/fluent/combo.js';
import {TreeCombo,type TreeGroup} from '../../desktop/ui/src/fluent/tree_combo.js';
import {comparisonPair,chooseComparisonFrom,chooseComparisonTo,swapComparison,clickComparison,type ComparisonChoice} from './transition-choice.js';
import {referenceFromLaser} from './reference-draft.js';
import {stateTree,transitionTree} from './choice-trees.js';
import {alignLevelReadouts} from './readout-alignment.js';
import {drawBadges} from '../../desktop/ui/src/pulse/widgets.js';
import { Segmented, setChecked, switchButton, checkBox } from '../../desktop/ui/src/fluent/toggles.js';
import { DETUNING_UNIT_MHZ, detuningInUnit, laserDetuningMHz, laserReferenceTransition, normalizeLaserFrequency, setLaserDetuningUnit, setLaserDetuningValue, setLaserFrequencyMode, setLaserTransitionReference, setLaserWavelengthNm, zeroFieldTransitionOptions } from './laser-frequency.js';
import { absoluteEnergyMHz, analyze, dipoleBetweenStates, laserFrequencyMHz, laserIntensityMWcm2, laserDirectionAxis, setLaserDirectionAxis, relativeDirectionAxis, setPolarizationPreset, composeMagneticField, type PolarizationPreset } from './physics.js';
import {normalizeLaserIntensity,setLaserIntensityMode,setLaserLocalIntensity,setLaserPowerMW,setLaserWaistUm} from './laser-intensity.js';
import {dynamicsCacheKey} from './dynamics-cache-key.js';
import {CurrentDynamicsResults} from './current-dynamics-results.js';
import { referenceEnergyMHz } from './reference-energy.js';
import {spectroscopyAnalysis,spectroscopyCoupling,projectSpectroscopyPopulations} from './spectroscopy.js';
import { defaultDynamics, defaultScene, DEFAULT_TWEEZER_WAVELENGTH_NM } from './types.js';
import type { Analysis, AxisAngles, DynamicsSettings, DetuningUnit, Laser, Level, MagneticField, Scene, SpectralLine, Transition } from './types.js';
import type { DynamicsResult } from './solver-types.js';
import { DynamicsClient, type SolverFailureRecord } from './solver-client.js';
import { validateScene } from './scene-file.js';
import { incidentTransitions, laserReferenceStateId, photonEndpoint, transitionEndpoints } from './transition-view.js';
import { escapeHtml as esc, levelsChart, number, populationChart, bindPopulationChart, populationColor, populationValue, populationSamplingInfo } from './charts.js';
import { math, shortKet, term } from './math.js';
import { renderSpace3D } from './space3d.js';
import {classifyLasers} from './laser-regime.js';
import {POPULATION_GROUPS,recommendedPopulationStates,populationDisplayIds,withPopulationGroups} from './population-selection.js';
import {durationUnit,durationValue,setDurationUnit,setDurationValue,durationText,DURATION_UNITS} from './duration.js';
import type {DurationUnit} from './types.js';
import {patchLevelPopulations} from './level-population-dom.js';
import {installWindowChrome} from './window-chrome.js';

installWindowChrome();

type View='levels'|'dynamics'|'space';
type Selection={kind:'field'|'laser'|'none';id:string};
const STORAGE='rb87-workbench-demo-v4',root=document.querySelector<HTMLDivElement>('#app')!;
let scene=defaultScene(),view:View='levels',spectral:SpectralLine='D2',selection:Selection={kind:'laser',id:'probe'};
let populationMode:'steady'|'time'='steady',populationTimeIndex=-1,populationScale:'linear'|'log'='linear';let unbindPopulation:(()=>void)|undefined;
let spaceRenderer:ReturnType<typeof renderSpace3D>|undefined;const spaceMount=document.createElement('div');spaceMount.className='retained-space-view';
let referenceUnit:DetuningUnit='MHz',referenceEditingId:string|null=null,referenceSpin:SpinBox|undefined,referenceEditorNode:HTMLElement|undefined,referenceNumber:{readMHz:()=>number;setMHz:(value:number)=>void}|undefined;
let saveState:'saved'|'pending'|'failed'='saved',saveFailure='',loadFailure='',unreadableScene:string|null=null;
let comparisonSource:string|undefined;let selectedStates:string[]=[],dirty=false,saveTimer=0,toastTimer=0,renderCount=0;
let latestAnalysis:Analysis, solverError='',solverBusy=false,solverQueued=false,solverRevision=0,requestedKey='',solverTimer=0,solverWallMs=0,controlsKey='',renderedPopulationKey='';
let requestedPhysicalKey='',requestedSteadyOnly=false;
let failureRecords:SolverFailureRecord[]=[];
try{const saved=JSON.parse(localStorage.getItem(`${STORAGE}-solver-failures`)??'[]');if(Array.isArray(saved))failureRecords=saved.slice(-5);}catch{}
function rememberSolverFailure(record:SolverFailureRecord):void{failureRecords=[...failureRecords.slice(-4),record];try{localStorage.setItem(`${STORAGE}-solver-failures`,JSON.stringify(failureRecords));}catch{}}
const solver=new DynamicsClient({debounceMs:0,onFailure:rememberSolverFailure}),analysisCache=new Map<string,Analysis>();
const treeOwners=new WeakMap<HTMLSelectElement,TreeCombo>();
let initialTree:TreeCombo|undefined;
const directionSegments: Array<{control:Segmented;kind:'axes'|'relative'}>=[];
const monitorModeSegments=new Map<HTMLElement,Segmented>();
const projectedPopulationCache=new WeakMap<Analysis,WeakMap<object,Record<string,number>|null>>();
let spectrumBuildCount=0,populationPaintCount=0;
type SolvedRecord={result:DynamicsResult;wallMs:number};const displayedResults=new CurrentDynamicsResults<SolvedRecord>();
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;


try{const saved=localStorage.getItem(STORAGE);if(saved){unreadableScene=saved;scene=validateScene(JSON.parse(saved));unreadableScene=null;}}catch(error){loadFailure=error instanceof Error?error.message:String(error);scene.dynamics!.autoRun=false;}
if(!scene.lasers.some(l=>l.id===selection.id))selection=scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:{kind:'field',id:scene.fields[0]?.id??''};
scene.marks??={};scene.references??=[];scene.dynamics??=defaultDynamics();
try{comparisonSource=localStorage.getItem(`${STORAGE}-comparison-source`)||undefined;}catch{}
try{const saved=JSON.parse(localStorage.getItem(`${STORAGE}-selected-states`)??'[]');if(Array.isArray(saved))selectedStates=[...new Set(saved.filter((id:unknown):id is string=>typeof id==='string'))].slice(0,2);}catch{/* Ignore malformed presentation state. */}

root.innerHTML=`<div class="workbench app-shell">
<header class="app-header topbar"><div class="brand"><div class="atom-mark"><img src="./zlc.ico" width="20" height="20" alt="" draggable="false"></div><div class="brand-copy"><h1>Atom Workbench</h1></div><button class="quiet scene-name" id="rename-scene" aria-label="重命名场景" title="重命名场景"></button></div><div class="header-actions"><button class="button button-subtle" id="model-button">模型说明</button><button class="button button-subtle" id="import-button">导入</button><button class="button" id="export-button">导出场景</button></div></header>
<div class="workspace"><aside class="scene-panel"><div class="panel-heading"><h2>实验场景</h2><button id="toggle-scene" class="quiet scene-collapse" aria-label="展开或收起场景" title="展开或收起场景"><span class="scene-chevron" aria-hidden="true"></span></button></div><div class="section-label">磁场 <span id="field-count"></span></div><div id="field-list" class="scene-list"></div><button class="add-object" id="add-field" aria-label="添加磁场">＋ 添加磁场</button><div class="section-label">激光 <span id="laser-count"></span></div><div id="laser-list" class="scene-list"></div><button class="add-object" id="add-laser" aria-label="添加激光">＋ 添加激光</button><button class="quiet scene-initial-shortcut" id="initial-shortcut" aria-label="选择初始态" title="选择初始态">ψ₀</button><div class="state-selector"><div class="section-label">初始态</div><div id="initial-state-tree"></div></div></aside>
<main class="main-panel"><div class="main-toolbar"><div class="view-tabs" role="tablist" aria-label="分析视图"><button data-view="levels" role="tab">能级与跃迁</button><button data-view="dynamics" role="tab">Population</button><button data-view="space" role="tab">方向与偏振</button></div><button class="quiet inspector-shortcut" id="toggle-inspector">参数</button><select class="preset-select" id="preset" aria-label="加载示例"><option value="">加载示例</option><option value="three">光镊 + Probe + Repump</option><option value="open">单束光 · F=2 → F′=2</option><option value="cycling">D2 · Cycling</option><option value="f3-linear">F=2 → F′=3 · 线偏振</option><option value="rabi">D2 · Rabi 演化</option><option value="dark">D1 · F2 → F1 抽运</option><option value="both">D1 + D2 · 同时驱动</option><option value="tilted">倾斜磁场</option></select></div><section class="summary-strip" id="summary" aria-label="当前场景和求解状态"></section><section class="visual-card science-card"><div class="card-heading"><div><span id="plot-eyebrow" class="eyebrow">ENERGY LEVELS</span><h2 id="plot-title">裸能级</h2></div><div class="plot-actions"><div class="line-tabs" id="line-tabs" role="tablist" aria-label="精细结构支路"><button class="button" data-line="D2" aria-label="D2 · 5P3/2">D2 · ${term('D2')}</button><button class="button" data-line="D1" aria-label="D1 · 5P1/2">D1 · ${term('D1')}</button></div><span class="plot-badge" id="plot-badge"></span><button class="quiet icon-button" id="clear-laser-selection" aria-label="取消激光选择" title="取消激光选择 · Escape">×</button><button class="quiet chart-info" id="chart-info" aria-label="能级图说明" title="图中能级为裸 HFS + Zeeman；激光参与完整动力学。纵向颜色区按 |mF| 对称；蓝色深浅对应固定 0–100% 布居。Δ = νL − νeg。基态初态显示吸收上箭头，激发态初态显示受激辐射下箭头。极弱耦合保留可见线条，强度数值见悬停。每个精细流形内按能量线性排列，光学间隔省略。">ⓘ</button></div></div><div id="population-toolbar" class="population-toolbar" hidden></div><div id="dynamics-controls" class="dynamics-controls" hidden></div><div class="chart-surface" id="main-chart"></div><div id="selection-readout" class="state-selection-panel"></div><div class="chart-caption" id="chart-caption"></div></section><section class="details-card transition-panel" id="details"></section></main>
<aside class="inspector-panel inspector"><div class="inspector-heading"><span class="eyebrow">PARAMETERS</span><h2 id="inspector-title">探测光</h2><p id="inspector-subtitle">选中对象的参数</p><button class="quiet icon-button inspector-dismiss" id="close-inspector" aria-label="关闭参数">×</button></div><div id="inspector-controls"></div></aside></div><footer class="status-bar"><span><span class="live-dot"></span> <span id="status-text">计算就绪</span></span><span id="model-status">Rb87 · hyperfine + Zeeman</span><button class="text-button" id="reset-button">重置场景</button></footer></div>
<dialog class="model-dialog" id="model-dialog"><div class="dialog-header"><div><span class="eyebrow">MODEL & CONVENTIONS</span><h2>模型、单位与适用范围</h2></div><button class="icon-button" id="close-model" aria-label="关闭模型说明">×</button></div><div class="dialog-content" tabindex="0" aria-label="模型说明内容"><h3>32 态密度矩阵</h3><p>基态 8 个、D1 激发态 8 个、D2 激发态 16 个。动力学由本机 Rust 求解器计算 Lindblad master equation，保留密度矩阵的布居与相干项。激光相干驱动和自发辐射均参与演化。用户指定远失谐光或近共振光。远失谐光仅产生背景光移；近共振光对两个基态 F 及 D1、D2 的全部允许电偶极跃迁进入密度矩阵驱动；不同频率的拍频项显式保留。光镊不再作为显式光学 driving。模型限于 32 个内部态，Zeeman 混合在各精细结构流形内计算。显式驱动和辐射采用电偶极光学 RWA 与各向同性自由空间 Markov 真空；自发衰变算符随背景本征基变换；区分不同基态超精细发射频率，同一通道保留相干。远失谐背景 Hamiltonian 包含 D 线反旋转项及列明高电子态的二阶 DC 光移；近共振光只采用光学 RWA，不截断跨 F 驱动，也不重复加微扰 Stark 项。高态未加入传播的 Hilbert 空间，其布居和额外散射通道未被显式求解；还未包含原子运动、技术噪声和驱动引起的辐射 Lamb shift。数值收敛误差不等于这些物理近似的误差。</p><p>${math('\\dot\\rho=-i[H/\\hbar,\\rho]+\\sum_j\\mathcal D[L_j]\\rho')}</p><details class="physical-scope-detail"><summary>850 nm 远失谐光的适用范围</summary><p>此前的独立弱场 Kramers–Heisenberg 对照中，850 nm 反旋转辐射项使总散射率比光学 RWA 结果高约 7.84%。这个数值针对该弱场辐射对照，并不是任意场景的误差上界。当前新增的二阶 DC 光移修正不会补上缺失的辐射通道；更小的数值残差也不会消除该散射模型偏差。</p></details><h3>磁场与 decay</h3><p>超精细和 Zeeman Hamiltonian 对角化给出当前磁场下本征态。驱动和辐射偶极算子在同一个本征基中变换；辐射分支因本征态混合而改变。通向相同发射光子的 D1、D2 路径保留相干叠加。具体数值残差和求解状态在“计算详情”内查看。</p><h3>布居着色</h3><p>“稳态”仅使用通过收敛检查的长期布居极限。多频驱动若通过收敛检查，则显示“长期时间平均”；单频周期驱动时，它对应周期平均。若存在多个长期平均解，则保留给定初态的投影，详情中标明初态依赖。颜色使用固定的 0–100% 布居色标，不随其他态的布居或选中状态变化。未获得稳态时不绘制虚构的无限时结果。基态上方始终显示已验证的长期布居。背景基发生混合时，布居由完整长期密度矩阵的 v†ρv 得到，保留相干项；缺少完整密度矩阵时显示未知。“指定时刻”的着色只在现有瞬态数据足以支持所示基底时使用；Population 视图保留原始磁场本征态的 p(t)。非零稳态布居本身不等同于严格量子暗态。</p><h3>图形约定</h3><p>选中激光时，光谱图把其余启用外场作为背景，排除所选光束的自作用；取消选择时显示全部外场的总位移。可定义的远失谐背景采用静态 Stark 光谱模型，并在同一背景基中计算能差、矩元和布居。近共振、强混合或不可静态约化的多频背景没有这里使用的单一静态光移，图中保留裸态几何并将相应数值标为未定义；不会把裸 Zeeman 能量冒充总光移。该光谱图模型不替代完整 Rust 密度矩阵演化，所有启用光束始终参与动力学。每个精细流形内按能量线性排列，光学频率间隔省略。每束光可单独指定图示参考态，缺省跟随动力学初态；基态参考画吸收上箭头，激发态参考画受激辐射下箭头。改变参考态或选择对象不改变 ρ₀。参考能级只参与能差计算。</p><p>参数框的 Δ = νL − νref，以所选零场跃迁为参考；图和矩元读数的 ΔL = νL − νeg，νeg 为当前背景下 g↔e 的正频率。频率单位切换保持实际频率不变。偏振主轴 ψ 相对光束球坐标的 eθ 方向，横向另一轴为 eφ；χ = +45° 在 k 与 B 同向时对应 σ⁺。零磁场以 +z 为显示参考轴。σ 分量采用光场约定；受激辐射的原子 Δm 与对应吸收相反。</p><p>数据：<a href="https://steck.us/alkalidata/rubidium87numbers.pdf" target="_blank" rel="noopener noreferrer">Steck · Rubidium 87 Data</a>；独立校验：<a href="https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html" target="_blank" rel="noopener noreferrer">QuTiP</a>。</p></div><div class="dialog-footer"><button class="button button-primary" id="understood">返回工作台</button></div></dialog><input type="file" id="file-input" accept="application/json,.json" hidden><div class="toast" id="toast" role="status" hidden></div>`;

root.querySelectorAll<HTMLButtonElement>('button.icon-button').forEach(old=>{const control=command(old.textContent??'×',()=>{},{role:'quiet',glyph:true,square:true,title:old.title}).root;control.id=old.id;for(const name of old.classList)if(name!=='icon-button'&&name!=='quiet')control.classList.add(name);const label=old.getAttribute('aria-label');if(label)control.setAttribute('aria-label',label);old.replaceWith(control);});
const manifoldFocus=new Segmented(value=>{spectral=value as SpectralLine;manifoldFocus.setCurrent(spectral);renderResults(false);});
manifoldFocus.setChoices([['D2','D2 · 5P₃/₂'],['D1','D1 · 5P₁/₂']]);manifoldFocus.setCurrent(spectral);manifoldFocus.root.id='line-tabs';manifoldFocus.root.classList.add('line-tabs');manifoldFocus.root.setAttribute('aria-label','能级关注流形');
manifoldFocus.root.querySelectorAll<HTMLButtonElement>('button').forEach((button,i)=>{button.dataset.line=i===0?'D2':'D1';button.setAttribute('aria-label',i===0?'D2 · 5P3/2':'D1 · 5P1/2');});$('line-tabs').replaceWith(manifoldFocus.root);

function toast(message:string):void {const el=$('toast');el.textContent=message;el.hidden=false;clearTimeout(toastTimer);toastTimer=window.setTimeout(()=>el.hidden=true,4200);}
function setText(id:string,value:string):void {$(id).textContent=value;}
function populationPercent(p:number|undefined):string{return p===undefined?'—':populationValue(p);}
function initialId():string{return scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`;}
function allLevels(a=latestAnalysis):Level[]{return [...a.ground,...a.D1,...a.D2];}
function getLevel(id:string,a=latestAnalysis):Level|undefined{return allLevels(a).find(l=>l.id===id);}
function stateText(l:Level):string{return `${l.manifold==='ground'?'5S1/2':l.manifold==='D1'?'5P1/2':'5P3/2'} · ${l.manifold==='D1'?'F″':l.manifold==='D2'?'F′':'F'}=${l.F}, m=${l.m}`;}
function measuredNumber(value:number,places=4):string{if(value!==0&&Number.isFinite(value)&&Math.abs(value)<10**(-places))return value.toExponential(2);return number(value,places);}
function frequencyText(valueMHz:number):string{if(!Number.isFinite(valueMHz))return '—';const magnitude=Math.abs(valueMHz),scale=magnitude>=1e6?1e6:magnitude>=1e3?1e3:magnitude>=1?1:magnitude>=.001?.001:.000001,unit=scale===1e6?'THz':scale===1e3?'GHz':scale===1?'MHz':scale===.001?'kHz':'Hz';return `${number(valueMHz/scale,scale===1e6?12:scale===1e3?9:6)} ${unit}`;}
function stateHTML(id:string):string {const l=getLevel(id);return l?`<span class="state-math" role="img" aria-label="${esc(stateText(l))}">${term(l.manifold)} ${shortKet(l.F,l.m)}</span>`:esc(scene.references?.find(r=>r.id===id)?.name??id);}
function absolute(id:string):number|undefined {const l=getLevel(id);if(l)return absoluteEnergyMHz(l);const r=scene.references?.find(r=>r.id===id);return r?referenceEnergyMHz(r,allLevels()):undefined;}
function activeLaser(_line?:SpectralLine):Laser|undefined {return selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):undefined;}
function referenceStateId(laser=activeLaser()):string{return laserReferenceStateId(scene,laser);}
function dynamics():DynamicsSettings{return scene.dynamics??(scene.dynamics=defaultDynamics());}
function saveSceneNow():void{clearTimeout(saveTimer);try{if(unreadableScene!==null){localStorage.setItem(`${STORAGE}-unreadable`,unreadableScene);unreadableScene=null;}localStorage.setItem(STORAGE,JSON.stringify(scene));localStorage.setItem(`${STORAGE}-selected-states`,JSON.stringify(selectedStates));localStorage.setItem(`${STORAGE}-comparison-source`,comparisonSource??'');saveState='saved';saveFailure='';}catch(error){saveState='failed';saveFailure=error instanceof Error?error.message:String(error);}refreshStatus();}
function persist():void{saveState='pending';clearTimeout(saveTimer);saveTimer=window.setTimeout(saveSceneNow,200);}
function refreshStatus():void{setText('status-text',saveState==='failed'?'本地保存失败，请导出场景':loadFailure?'原本地场景无法载入 · 可下载原文件':currentSolverError()?'布居计算失败':currentSolverBusy()?(solverQueued?'布居待计算':'布居计算中'):saveState==='pending'?'正在保存…':'场景已保存');$('status-text').title=saveFailure||loadFailure;root.querySelector('.live-dot')?.classList.toggle('save-failed',saveState==='failed'||Boolean(loadFailure));}
function update():void{persist();invalidateObsoletePopulation();if(dirty)return;dirty=true;requestAnimationFrame(()=>{dirty=false;renderResults();});}
function select(next:Selection,toggle=false):void{const focusedObject=document.activeElement?.closest<HTMLElement>('[data-object-id]'),focusId=focusedObject?.dataset.objectId;selection=toggle&&next.kind===selection.kind&&next.id===selection.id?{kind:'none',id:''}:next;if(innerWidth<=920&&selection.kind!=='none')root.classList.add('inspector-open');if(selection.kind==='none')root.classList.remove('inspector-open');root.classList.remove('scene-expanded');syncSceneLayout();renderObjects();renderInspector();renderResults(false);if(focusId)root.querySelector<HTMLButtonElement>(`[data-object-id="${CSS.escape(focusId)}"]`)?.focus({preventScroll:true});}
function clearSourceSelection():void{if(selection.kind!=='none')select({kind:'none',id:''});}
function comparisonChoice():ComparisonChoice{return {selected:selectedStates,from:comparisonSource};}
function applyComparison(choice:ComparisonChoice):void{selectedStates=choice.selected;comparisonSource=choice.from;persist();renderResults(false);}
function markedComparison(l:Laser|undefined):string[]{const p=comparisonPair(comparisonChoice(),referenceStateId(l));return p.to?[p.from,p.to]:comparisonSource?[comparisonSource]:[];}
function selectState(id:string):void{applyComparison(clickComparison(comparisonChoice(),id));}

function item(kind:'field'|'laser',id:string,name:string,sub:string,enabled:boolean,onToggle:(v:boolean)=>void,onRemove?:()=>void):HTMLElement{
  const div=document.createElement('div');div.className=`scene-object${selection.kind===kind&&selection.id===id?' selected':''}${enabled?'':' disabled'}`;
  const button=document.createElement('button');button.type='button';button.className='object-select';button.dataset.objectId=id;button.dataset.objectKind=kind;button.setAttribute('aria-label',`编辑${name}`);button.setAttribute('aria-pressed',String(selection.kind===kind&&selection.id===id));const symbol=document.createElement('span');symbol.className=`object-symbol ${kind}`;symbol.textContent=kind==='field'?'B':'λ';const copy=document.createElement('span');copy.className='object-copy';const title=document.createElement('strong');title.textContent=name;const subtitle=document.createElement('small');subtitle.textContent=sub;copy.append(title,subtitle);button.append(symbol,copy);button.addEventListener('pointerenter',()=>{button.title=title.scrollWidth>title.clientWidth||subtitle.scrollWidth>subtitle.clientWidth?`${name} · ${sub}`:'';});button.onclick=()=>select({kind,id},kind==='laser');const toggle=switchButton(next=>{onToggle(next);setChecked(toggle,next);div.classList.toggle('disabled',!next);update();});setChecked(toggle,enabled);toggle.classList.add('object-toggle');toggle.setAttribute('aria-label',`启用${name}`);div.append(button,toggle);
  if(onRemove){const remove=command('×',onRemove,{role:'destructive',glyph:true,square:true,title:`删除${name}`}).root;remove.classList.add('remove-object');remove.setAttribute('aria-label',`删除${name}`);div.append(remove);}return div;
}
function removeObject(kind:'field'|'laser',id:string):void{
  const original=kind==='field'?scene.fields:scene.lasers,index=original.findIndex(o=>o.id===id),selected=selection.kind===kind&&selection.id===id;
  if(kind==='field')scene.fields=scene.fields.filter(o=>o.id!==id);else scene.lasers=scene.lasers.filter(o=>o.id!==id);
  if(selected){const remaining=kind==='field'?scene.fields:scene.lasers,next=remaining[Math.min(index,remaining.length-1)];selection=next?{kind,id:next.id}:scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:{kind:'field',id:scene.fields[0]?.id??''};}
  controlsKey='';renderObjects();renderInspector();update();if(selected)requestAnimationFrame(()=>root.querySelector<HTMLButtonElement>(`[data-object-id="${selection.id}"]`)?.focus({preventScroll:true}));
}
function renderObjects():void{setText('rename-scene',scene.name);$('rename-scene').title=`${scene.name} · 重命名场景`;
  $('field-list').replaceChildren(...scene.fields.map(f=>item('field',f.id,f.name,`${measuredNumber(f.magnitudeG,2)} G · θ ${number(f.thetaDeg,0)}°`,f.enabled,v=>f.enabled=v,()=>removeObject('field',f.id))));
  $('laser-list').replaceChildren(...scene.lasers.map(l=>item('laser',l.id,l.name,l.frequencyMode==='wavelength'?`${Number((l.wavelengthNm??DEFAULT_TWEEZER_WAVELENGTH_NM).toPrecision(5))} nm`:`Δ ${Number(detuningInUnit(l).toPrecision(4))} ${l.detuningUnit??'MHz'}`,l.enabled,v=>l.enabled=v,()=>removeObject('laser',l.id))));
  setText('field-count',String(scene.fields.length));setText('laser-count',String(scene.lasers.length));$('add-field').toggleAttribute('disabled',scene.fields.length>=8);$('add-laser').toggleAttribute('disabled',scene.lasers.length>=8);
}
type NumericControl={input:HTMLInputElement;setValue:(value:number)=>void;setUnit:(unit:string)=>void};
const numericControls=new Map<string,NumericControl>();
function group(title:string,parent=$('inspector-controls')):HTMLElement{const section=document.createElement('section');section.className='control-section';const g=form();g.append(sectionTitle(title));section.append(g);parent.append(section);return g;}
function note(parent:HTMLElement,value:string):void{parent.title=value;}
function numeric(parent:HTMLElement,label:string,key:string,value:number,min:number,max:number,step:number,unit:string,onChange:(value:number)=>void,_slider?:{min?:number;max?:number;window?:number},unitOptions?:{choices:string[];change:(unit:string)=>number;enabled?:(unit:string)=>boolean}):NumericControl{
  const holder=document.createElement('div');holder.className='number-field';const input=document.createElement('input');input.type='text';input.inputMode='decimal';input.id=`number-${key}`;input.value=String(value);input.spellcheck=false;input.setAttribute('aria-label',label);holder.append(input);
  const unitEl=unitOptions?document.createElement('select'):document.createElement('span');unitEl.className=unitOptions?'unit-picker':'unit';
  if(unitEl instanceof HTMLSelectElement){unitEl.setAttribute('aria-label',`${label}单位`);unitEl.append(comboFace(),...unitOptions!.choices.map(u=>new Option(u,u)));unitEl.value=unit;}else unitEl.textContent=unit;
  const refreshUnits=()=>{if(unitEl instanceof HTMLSelectElement&&unitOptions?.enabled){let disabled=false;for(const option of unitEl.options){option.disabled=!unitOptions.enabled(option.value);option.title=option.disabled?`当前数值超出 ±5000 ${option.value}`:'';disabled ||= option.disabled;}unitEl.title=disabled?'灰色单位无法在 ±5000 内表示当前失谐':'';}};refreshUnits();
  const labelEl=document.createElement('label');labelEl.textContent=label;labelEl.htmlFor=input.id;const wrapper=fieldRow(labelEl,holder,unitEl);wrapper.classList.add('numeric-field');parent.append(wrapper);let current=value;
  const commit=(next:number,source:string)=>{if(!Number.isFinite(next)||next<min||next>max){input.setAttribute('aria-invalid','true');return;}try{onChange(next);}catch(error){input.setAttribute('aria-invalid','true');input.title=error instanceof Error?error.message:String(error);return;}current=next;input.removeAttribute('aria-invalid');input.title='';if(source!=='number')input.value=String(next);spin.refresh();refreshUnits();update();renderObjects();};
  const spin=new SpinBox(holder,input,{step:()=>step,bounds:()=>[min,max],commit:n=>commit(n,'spin'),decimal:!Number.isInteger(step)});spin.up.title=`增加 ${step} ${unit}`;spin.down.title=`减少 ${step} ${unit}`;spin.refresh();
  input.oninput=()=>{const raw=input.value.trim();if(raw!==''&&Number.isFinite(Number(raw)))commit(Number(raw),'number');};input.onchange=()=>{if(!input.value.trim()||!Number.isFinite(Number(input.value))||Number(input.value)<min||Number(input.value)>max){input.value=String(current);input.removeAttribute('aria-invalid');spin.refresh();}};
  const control={input,setValue:(n:number)=>{current=Number(n.toPrecision(15));input.value=String(current);spin.refresh();},setUnit:(u:string)=>{if(unitEl instanceof HTMLSelectElement)unitEl.value=u;else unitEl.textContent=u;spin.up.title=`增加 ${step} ${u}`;spin.down.title=`减少 ${step} ${u}`;refreshUnits();}};
  if(unitEl instanceof HTMLSelectElement)unitEl.onchange=()=>{try{control.setValue(unitOptions!.change(unitEl.value));control.setUnit(unitEl.value);update();renderObjects();}catch(error){unitEl.value=unit;toast(error instanceof Error?error.message:String(error));}};
  numericControls.set(key,control);return control;
}
function selectControl(parent:HTMLElement,label:string,value:string,options:[string,string][],changed:(value:string)=>void):HTMLSelectElement{
  if(label==='零场参考'||label==='参考态'||label==='初态'){
    const groups=label==='零场参考'?transitionTree():stateTree(allLevels(),label==='参考态'?scene.references:[]);
    const tree=new TreeCombo(changed);tree.setTree(groups,value,label==='参考态'?'跟随动力学初态':null);tree.root.setAttribute('aria-label',label);treeOwners.set(tree.root,tree);const row=fieldRow(label,tree.root);row.classList.add('choice-field');parent.append(row);return tree.root;
  }
  const el=document.createElement('select');el.setAttribute('aria-label',label);el.append(comboFace(),...options.map(([v,t])=>new Option(t,v)));el.value=value;el.onchange=()=>changed(el.value);const r=fieldRow(label,el);r.classList.add('choice-field');parent.append(r);return el;
}
function setChoiceValue(root:HTMLSelectElement,value:string):void{const tree=treeOwners.get(root);if(tree)tree.value=value;else root.value=value;}
function disposeTrees(host:HTMLElement):void{host.querySelectorAll<HTMLSelectElement>('select.tree-combo').forEach(root=>treeOwners.get(root)?.dispose());}
function segmentedControl(parent:HTMLElement,label:string,value:string,options:[string,string][],changed:(value:string)=>void):Segmented{
  const control=new Segmented(next=>{try{changed(String(next));control.setCurrent(next);}catch(error){toast(error instanceof Error?error.message:String(error));}});control.setChoices(options);control.setCurrent(value);control.root.setAttribute('aria-label',label);parent.append(fieldRow(label,control.root));return control;
}
function sectionChoice(parent:HTMLElement,label:string,value:string,options:[string,string][],changed:(value:string)=>void):HTMLElement{
  const control=new Segmented(next=>{try{changed(String(next));control.setCurrent(next);}catch(error){toast(error instanceof Error?error.message:String(error));}});control.setChoices(options);control.setCurrent(value);control.root.setAttribute('aria-label',label);control.root.classList.add('section-choice');parent.querySelector('.section-title')!.append(control.root);return control.root;
}
/** Presets are commands, not modes. Rendering never selects or re-applies one. */
function polarizationButtons(laser:Laser,choices:readonly (readonly [PolarizationPreset,string])[],label:string):HTMLElement{
  const actions=document.createElement('div');actions.className='polarization-presets preset-actions full-form-row';actions.role='group';actions.setAttribute('aria-label',label);
  for(const [preset,text]of choices){
    const action=command(text,()=>{
      setPolarizationPreset(laser,scene.fields,preset);
      for(const [key,value]of [['laser-theta',laser.thetaDeg],['laser-phi',laser.phiDeg],['polarization',laser.polarizationAngleDeg],['ellipticity',laser.ellipticityDeg]] as const)numericControls.get(key)?.setValue(value);
      renderObjects();update();
    });
    action.root.dataset.polarizationAction=preset;actions.append(action.root);
  }
  return actions;
}
function directionButtons(parent:HTMLElement,apply:(theta:number,phi:number,exact?:AxisAngles)=>void,relative=false):void{
  const axes:[string,string][]=[['90,0','+x'],['90,180','−x'],['90,90','+y'],['90,-90','−y'],['0,0','+z'],['180,0','−z']];
  const control=new Segmented(key=>{const [theta,phi]=String(key).split(',').map(Number);apply(theta!,phi!);});control.setChoices(axes);control.root.classList.add('direction-presets','full-form-row');control.root.setAttribute('aria-label','坐标轴方向');parent.append(control.root);directionSegments.push({control,kind:'axes'});
  if(relative){const control=new Segmented(key=>{const axis=relativeDirectionAxis(composeMagneticField(scene.fields).fieldAxis,String(key) as 'parallel'|'anti'|'perpendicular');apply(axis.thetaDeg,axis.phiDeg,axis);});control.setChoices([['parallel','∥ B'],['anti','−B'],['perpendicular','⊥ B']]);control.root.classList.add('direction-relative','full-form-row');control.root.setAttribute('aria-label','相对磁场方向');parent.append(control.root);directionSegments.push({control,kind:'relative'});}
}
function renderInspector():void{
  renameObject.root.hidden=!(selection.kind==='laser'?scene.lasers.some(l=>l.id===selection.id):scene.fields.some(f=>f.id===selection.id));const host=$('inspector-controls');const scroll=$('inspector-title').closest<HTMLElement>('.inspector-panel')!.scrollTop;disposeTrees(host);host.replaceChildren();numericControls.clear();directionSegments.length=0;setText('inspector-title','参数');setText('inspector-subtitle','');
  if(selection.kind==='none'){setText('inspector-title','总场');const empty=document.createElement('p');empty.className='selection-empty inspector-empty';empty.textContent='选择光束或磁场查看参数';host.append(empty);return;}
  if(selection.kind==='field'){
    const field=scene.fields.find(f=>f.id===selection.id);if(!field)return;setText('inspector-title',field.name);const g=group('磁场');numeric(g,'大小','magnitude',field.magnitudeG,0,20,.01,'G',n=>field.magnitudeG=n);numeric(g,'极角 θ','field-theta',field.thetaDeg,0,180,1,'°',n=>field.thetaDeg=n);numeric(g,'方位 φ','field-phi',field.phiDeg,-180,180,1,'°',n=>field.phiDeg=n);directionButtons(g,(theta,phi)=>{field.thetaDeg=theta;field.phiDeg=phi;numericControls.get('field-theta')?.setValue(theta);numericControls.get('field-phi')?.setValue(phi);renderObjects();update();});
  }else{
    const l=scene.lasers.find(l=>l.id===selection.id);if(!l)return;normalizeLaserFrequency(l);normalizeLaserIntensity(l);setText('inspector-title',l.name);
    const mode=group('激光类型');segmentedControl(mode,'处理方式',l.regime??'near',[['near','近共振'],['far','远失谐']],value=>{l.regime=value as 'near'|'far';update();});
    const validation=document.createElement('p');validation.id='laser-regime-status';validation.className='laser-regime-status';mode.append(validation);
    const f=group('频率');sectionChoice(f,'频率指定方式',l.frequencyMode??'transition',[['transition','跃迁失谐'],['wavelength','真空波长']],v=>{setLaserFrequencyMode(l,v as 'transition'|'wavelength');renderInspector();renderObjects();update();});
    const ref=laserReferenceTransition(l),refs=zeroFieldTransitionOptions();const referencePicker=selectControl(f,'零场参考',`${ref.fromManifold??'ground'}-${ref.line}-${ref.Fg}-${ref.Fe}`,refs.map(r=>[`${r.fromManifold??'ground'}-${r.line}-${r.Fg}-${r.Fe}`,r.label]),v=>{const next=refs.find(r=>`${r.fromManifold??'ground'}-${r.line}-${r.Fg}-${r.Fe}`===v)!;setLaserTransitionReference(l,next);numericControls.get('detuning')?.setValue(detuningInUnit(l));numericControls.get('detuning')?.setUnit(l.detuningUnit??'MHz');renderObjects();update();});referencePicker.title='改变失谐参考保持光频率不变；输入 Δ=0 可设为该零场跃迁共振。';
    const originPicker=selectControl(f,'参考态',l.referenceStateId??'',[['','跟随动力学初态'],...allLevels().map(state=>[state.id,stateText(state)] as [string,string]),...(scene.references??[]).map(ref=>[ref.id,ref.name+' · 参考能级'] as [string,string])],id=>{if(id==='')delete l.referenceStateId;else l.referenceStateId=id;persist();renderInspector();renderResults(false);});originPicker.id='laser-reference-state';originPicker.setAttribute('aria-label','参考出发态');originPicker.title='只改变这束光的箭头起点和矩元参考；不改变动力学初态或任何驱动。';
    if(l.frequencyMode==='wavelength')numeric(f,'波长 λ','wavelength',l.wavelengthNm??DEFAULT_TWEEZER_WAVELENGTH_NM,Number.MIN_VALUE,Number.MAX_VALUE,.01,'nm',n=>setLaserWavelengthNm(l,n));
    else numeric(f,'失谐 Δ','detuning',detuningInUnit(l),-5000,5000,.01,l.detuningUnit??'MHz',n=>setLaserDetuningValue(l,n),undefined,{choices:['Hz','kHz','MHz','GHz','THz'],enabled:u=>Math.abs(laserDetuningMHz(l))<=5000*DETUNING_UNIT_MHZ[u as DetuningUnit],change:u=>{const actual=setLaserDetuningUnit(l,u as NonNullable<Laser['detuningUnit']>);queueMicrotask(()=>numericControls.get('detuning')?.setUnit(actual));return detuningInUnit(l);}});
    const freq=document.createElement('span');freq.className='field-readout';freq.id='laser-absolute-frequency';f.append(fieldRow('频率 ν',freq));
    if(scene.references?.some(r=>r.id===l.referenceStateId))selectControl(f,'光子方向',l.referenceProcess??'absorption',[['absorption','吸收 ↑'],['emission','受激辐射 ↓']],v=>{l.referenceProcess=v as 'absorption'|'emission';update();});
    const strength=group('强度');sectionChoice(strength,'强度指定方式',l.intensityMode??'intensity',[['intensity','局域强度'],['power','功率与束腰']],v=>{setLaserIntensityMode(l,v as 'intensity'|'power');renderInspector();update();});
    if(l.intensityMode==='power'){numeric(strength,'功率 P','power',l.powerMW??4,0,Number.MAX_VALUE,.01,'mW',n=>setLaserPowerMW(l,n));numeric(strength,'束腰 w₀','waist',l.waistUm??1.2,Number.MIN_VALUE,Number.MAX_VALUE,.01,'μm',n=>setLaserWaistUm(l,n));}else numeric(strength,'强度 I','intensity',l.intensityMWcm2,0,Number.MAX_VALUE,.01,'mW/cm²',n=>setLaserLocalIntensity(l,n));
    const d=group('传播方向');numeric(d,'极角 θ','laser-theta',l.thetaDeg,0,180,1,'°',n=>setLaserDirectionAxis(l,{thetaDeg:n,phiDeg:l.phiDeg}));numeric(d,'方位 φ','laser-phi',l.phiDeg,-180,180,1,'°',n=>setLaserDirectionAxis(l,{thetaDeg:l.thetaDeg,phiDeg:n}));directionButtons(d,(theta,phi,exact)=>{setLaserDirectionAxis(l,exact??{thetaDeg:theta,phiDeg:phi},Boolean(exact));numericControls.get('laser-theta')?.setValue(theta);numericControls.get('laser-phi')?.setValue(phi);renderObjects();update();},true);
    const p=group('光束偏振');numeric(p,'主轴 ψ','polarization',l.polarizationAngleDeg,-90,90,1,'°',n=>l.polarizationAngleDeg=n);numeric(p,'椭圆率 χ','ellipticity',l.ellipticityDeg,-45,45,1,'°',n=>l.ellipticityDeg=n);
    const shapeActions=polarizationButtons(l,[['circular-minus','圆−'],['linear','线偏振'],['circular-plus','圆+']],'一次性设置光束偏振形状');shapeActions.title='只设置一次椭圆率，保持当前传播方向和主轴。';p.append(shapeActions);
    const atomicActions=polarizationButtons(l,[['sigma-minus','σ−'],['pi','π'],['sigma-plus','σ+']],'按当前磁场一次性设置纯偏振');atomicActions.title='按当前 B 设置一次（B=0 时按 z）：π 令 k ⟂ B，σ 令 k ∥ B；之后各参数独立。';p.append(fieldRow('纯偏振',atomicActions));
    const pol=document.createElement('div');pol.id='inspector-polarization';pol.className='polarization-bars full-form-row';p.append(pol);
    const more=document.createElement('details');more.className='advanced-controls';more.innerHTML='<summary>相位</summary>';host.append(more);const phase=form();more.append(phase);numeric(phase,'初相位 φ','laser-phase',l.phaseDeg??0,-180,180,1,'°',n=>l.phaseDeg=n);
  }
  requestAnimationFrame(()=>{shareLabelColumn(host,64);$('inspector-title').closest<HTMLElement>('.inspector-panel')!.scrollTop=scroll;});
}
function renderInitial():void{
  const current=getLevel(initialId())??getLevel(`ground-${scene.initialF}-${scene.initialM}`)!;if(!current)return;
  initialTree??=new TreeCombo(id=>setInitialState(id));initialTree.root.id='initial-state-picker';initialTree.root.setAttribute('aria-label','初始态');initialTree.setTree(stateTree(allLevels()),current.id);initialTree.title=stateText(current);$('initial-state-tree').replaceChildren(initialTree.root);
  $('initial-shortcut').title=`初态：${stateText(current)}`;
}
function setInitialState(id:string):void{const state=getLevel(id);if(!state)return;dynamics().initialStateId=id;if(state.manifold==='ground'){scene.initialF=state.F as 1|2;scene.initialM=state.m;}else spectral=state.manifold;renderInitial();update();}

function polarizationBars(a:Analysis,l:Laser|undefined):string{const p=l?a.polarizations[l.id]:undefined;if(!p)return '<p class="helper-text">启用激光后显示球基分量。</p>';return `<span class="polarization-axis-label">${a.fieldMagnitudeG===0?'相对 z（B = 0）':'相对 B'}</span>`+ [[math('\\sigma^-'),p.minus],[math('\\pi'),p.pi],[math('\\sigma^+'),p.plus]].map(([label,v])=>`<div class="pol-row" title="偏振强度权重 ${esc(populationValue(Number(v)))}"><span>${label}</span><div class="bar-track"><span class="bar-fill" style="width:${Math.max(0,Math.min(100,Number(v)*100))}%"></span></div><strong>${esc(populationPercent(Number(v)))}</strong></div>`).join('');}

function paintStateBadges():void{
  if(view==='levels')alignLevelReadouts($('main-chart'));
  document.querySelectorAll<HTMLElement>('.selection-index[data-badge-number]').forEach(host=>drawBadges(host,[[host.dataset.badgeNumber!,'scan']]));
  $('main-chart').querySelectorAll<SVGGElement>('.state-selection-number').forEach(group=>{const holder=document.createElement('span');drawBadges(holder,[[group.dataset.badgeNumber!,'scan']]);const svg=holder.firstElementChild!;svg.setAttribute('x',group.dataset.badgeX!);svg.setAttribute('y',group.dataset.badgeY!);group.replaceChildren(svg);});
}
function selectionReadout(a:Analysis,l:Laser|undefined):string{
  const origin=referenceStateId(l),pair=comparisonPair(comparisonChoice(),origin),from=pair.from,to=pair.to;
  const pickers=`<div class="selection-top"><div class="selection-pair"><div class="selection-endpoint"><label><span class="selection-index" data-badge-number="1">1</span>起点</label><div id="comparison-from"></div></div><div id="comparison-swap"></div><div class="selection-endpoint"><label><span class="selection-index" data-badge-number="2">2</span>终点</label><div id="comparison-to"></div></div></div><button class="text-button" id="clear-selection">清除</button></div>`;
  if(!to)return pickers+'<div class="selection-empty">点选能级或选择终点。</div>';
  const ea=absolute(from),eb=absolute(to),defined=a.spectroscopy?.status!=='unresolved',gap=!defined||ea===undefined||eb===undefined?NaN:eb-ea;
  const first=getLevel(from),second=getLevel(to);let coupling='';
  if(first&&second){const referenceLaser=l??scene.lasers[0],transition=defined&&referenceLaser?spectroscopyCoupling(a,referenceLaser,first.id,second.id):undefined;coupling=!defined?'<span>背景矩元未定义</span>':transition?(transition.allowed?`<span title="与所示背景能级使用同一基底的电偶极矩元">|d| <b>${measuredNumber(Math.abs(transition.amplitudeCM)/1e-29,4)} × 10⁻²⁹ C·m</b></span>${l?`<span>${l.enabled?'Ω/2π':'Ω/2π（停用预览）'} <b>${measuredNumber(transition.rabiMHz,4)} MHz</b></span><span title="${esc(l.name)} · ΔL = νL − |Ee − Eg|/h；背景包含其余启用光束，排除所选激光。">Δ<sub>L</sub> <b>${frequencyText(transition.detuningMHz)}</b></span>`:''}`:'<span>E1 禁戒</span>'):'';}
  else coupling='<span>参考能级</span>';
  if(first&&second&&defined&&!scene.lasers.length){const bare=dipoleBetweenStates(first.id,second.id,a.fieldMagnitudeG);coupling=bare.allowed?`<span title="无光背景的磁场本征态偶极矩元">|d| <b>${measuredNumber(Math.abs(bare.amplitudeCM)/1e-29,4)} × 10⁻²⁹ C·m</b></span>`:'<span>E1 禁戒</span>';}
  const pop=populationSnapshot().populations?.[to];
  return `${pickers}<div class="compact-readout"><span title="终点能量 − 起点能量">ΔE/h <b>${defined?frequencyText(gap):'未定义'}</b></span>${coupling}<span data-current-population ${pop===undefined?'hidden':''}>p <b>${pop===undefined?'—':populationValue(pop)}</b></span></div><div class="selection-actions"><button class="button button-subtle" id="use-as-initial" ${second?'':'disabled'}>末态设为初态</button><button class="button button-subtle" id="monitor-selected" ${second?'':'disabled'}>观察布居</button><div class="mark-editor"><input id="mark-text" maxlength="32" aria-label="能级标记" placeholder="末态标记" value="${esc(scene.marks?.[to]??'')}" ${second?'':'disabled'}><button class="button" id="save-mark" ${second?'':'disabled'}>标记</button></div></div>`;
}
/** Solver responses update scientific readouts without unmounting an operator's editor. */
function renderSelection(a:Analysis,l:Laser|undefined):void {
  const host=$('selection-readout'),key=JSON.stringify([selectedStates,comparisonSource,referenceStateId(l),l?.id,scene.references?.map(r=>[r.id,r.name])]),next=document.createElement('div');next.innerHTML=selectionReadout(a,l);
  if(host.dataset.selectionKey!==key){disposeTrees(host);host.innerHTML=next.innerHTML;host.dataset.selectionKey=key;bindSelection();return;}
  const readout=host.querySelector('.compact-readout'),updated=next.querySelector('.compact-readout');if(readout&&updated&&readout.innerHTML!==updated.innerHTML)readout.innerHTML=updated.innerHTML;
}

function renderDetails(markup:string):void {
  const host=$('details'),next=document.createElement('div');next.innerHTML=markup;const nodes:HTMLElement[]=[];
  for(const incoming of [...next.children]as HTMLElement[]){const existing=([...host.children].find(el=>el.className===incoming.className)as HTMLElement|undefined)??(incoming.classList.contains('reference-editor')?referenceEditorNode:undefined);
    if(existing){
      if(existing.classList.contains('reference-editor')){if(!(existing as HTMLDetailsElement).open&&!referenceEditingId&&existing.dataset.dirty!=='true'){const anchor=existing.querySelector<HTMLSelectElement>('#reference-anchor'),nextAnchor=incoming.querySelector<HTMLSelectElement>('#reference-anchor');if(anchor&&nextAnchor)setChoiceValue(anchor,nextAnchor.value);}const list=existing.querySelector<HTMLElement>('.reference-list'),newList=incoming.querySelector<HTMLElement>('.reference-list');if(list&&newList&&list.dataset.markup!==newList.innerHTML){list.innerHTML=newList.innerHTML;list.dataset.markup=newList.innerHTML;}const summary=existing.querySelector('summary'),newSummary=incoming.querySelector('summary');if(summary&&newSummary)summary.innerHTML=newSummary.innerHTML;}
      else if(existing.innerHTML!==incoming.innerHTML)existing.innerHTML=incoming.innerHTML;
      nodes.push(existing);
    }else nodes.push(incoming);
    if(incoming.classList.contains('reference-editor'))referenceEditorNode=existing??incoming;
  }
  reconcileChildren(host,nodes);
}
function bindSelection():void{
  const laser=activeLaser(),origin=referenceStateId(laser),pair=comparisonPair(comparisonChoice(),origin),groups=stateTree(allLevels(),scene.references);
  const from=new TreeCombo(id=>applyComparison(chooseComparisonFrom(comparisonChoice(),id)));from.setTree(groups,comparisonSource??(selectedStates.length>1?selectedStates[0]!:'') ,`${laser?'跟随光束参考态':'跟随动力学初态'} · ${getLevel(origin)?stateText(getLevel(origin)!):scene.references?.find(r=>r.id===origin)?.name??origin}`);from.root.setAttribute('aria-label','跃迁起点');treeOwners.set(from.root,from);$('comparison-from').append(from.root);
  const to=new TreeCombo(id=>applyComparison(chooseComparisonTo(comparisonChoice(),id)));to.setTree(groups,pair.to??'','选择终点');to.root.setAttribute('aria-label','跃迁终点');treeOwners.set(to.root,to);$('comparison-to').append(to.root);
  const swap=command('⇄',()=>applyComparison(swapComparison(comparisonChoice(),origin)),{role:'quiet',glyph:true,square:true,title:'交换起点和终点'});swap.disabled=!pair.to;$('comparison-swap').append(swap.root);
  if($('clear-selection'))$('clear-selection').onclick=()=>applyComparison({selected:[]});
  if($('use-as-initial'))$('use-as-initial').onclick=()=>{const l=getLevel(selectedStates.at(-1)!);if(l)setInitialState(l.id);};
  if($('save-mark'))$('save-mark').onclick=()=>{const id=selectedStates.at(-1);if(id){const v=$<HTMLInputElement>('mark-text').value.trim();if(v)scene.marks![id]=v;else delete scene.marks![id];update();}};
  if($('monitor-selected'))$('monitor-selected').onclick=()=>{const ids=[...new Set(markedComparison(activeLaser()))].filter(id=>getLevel(id));const chosen=getLevel(ids.at(-1)??'');const matching=chosen?.manifold==='D1'||chosen?.manifold==='D2'?scene.lasers.find(l=>l.enabled&&l.line===chosen.manifold):activeLaser();if(matching)dynamics().laserId=matching.id;const monitored=[...new Set([...dynamics().monitoredStateIds,...ids])];dynamics().monitoredStateIds=monitored;dynamics().monitorMode='manual';view='dynamics';controlsKey='';update();};
}
function transitionRows(transitions:Transition[]):string{return transitions.map(t=>{
  const endpoints=transitionEndpoints(t,referenceStateId(scene.lasers.find(l=>l.id===t.laserId)))!,mixed=t.mixedPolarization===true;
  return `<tr data-transition="${esc(t.id)}" data-from-state="${endpoints.fromId}" data-to-state="${endpoints.toId}" tabindex="0"><td title="${endpoints.emission?'受激辐射':'吸收'}">${stateHTML(endpoints.fromId)} → ${stateHTML(endpoints.toId)}</td><td title="${mixed?'混合背景态：多个球分量相干耦合':t.mLabelApproximate?`光场 q=${t.q}；mF 是混合本征态的主成分标签，标签之差不是严格的 ΔmF`:`光场 q=${t.q}；原子 Δm=${endpoints.deltaM}`}" aria-label="${mixed?'混合分量':t.q===-1?'sigma minus':t.q===0?'pi':'sigma plus'}">${mixed?'混合':math(t.q===-1?'\\sigma^-':t.q===0?'\\pi':'\\sigma^+')}</td><td>${measuredNumber(t.offsetMHz,3)}</td><td>${measuredNumber(t.detuningMHz,3)}</td><td>${measuredNumber(t.rabiMHz,4)}</td><td><div class="strength-cell"><span style="width:${Math.min(100,t.strength*t.weight*100)}%"></span><small>${measuredNumber(t.strength*t.weight,4)}</small></div></td></tr>`;
}).join('');}
function referenceAnchorDefault():string{return [...selectedStates].reverse().find(id=>getLevel(id))??(comparisonSource&&getLevel(comparisonSource)?comparisonSource:initialId());}
function referenceEditor():string{
  const anchor=referenceAnchorDefault();return `<details class="reference-editor"><summary>参考能级 <span>${scene.references?.length??0}</span></summary><div class="reference-form-caption" id="reference-form-caption">新建参考能级</div><div class="reference-form form"><label class="field-label" for="reference-name">名称</label><div class="field-control"><input id="reference-name" maxlength="32" value="虚拟中间态" aria-label="参考能级名称"></div><label class="field-label" for="reference-anchor">锚点</label><div class="field-control"><select id="reference-anchor" aria-label="参考能级锚点">${allLevels().map(l=>`<option value="${l.id}" ${l.id===anchor?'selected':''}>${esc(stateText(l))}</option>`).join('')}</select></div><label class="field-label" for="reference-anchor-mode">能量定义</label><div class="field-control"><div id="reference-anchor-mode" data-value="follow"></div></div><label class="field-label" for="reference-offset">频率偏移</label><div class="field-control"><div class="number-field" id="reference-number"><input id="reference-offset" type="text" inputmode="decimal" value="${100/DETUNING_UNIT_MHZ[referenceUnit]}" aria-label="参考能级频率偏移"></div><select id="reference-unit" class="unit-picker" aria-label="参考能级频率单位">${Object.keys(DETUNING_UNIT_MHZ).map(unit=>`<option value="${unit}" ${unit===referenceUnit?'selected':''}>${unit}</option>`).join('')}</select><button class="quiet reference-laser-copy" id="reference-from-laser" aria-label="用当前激光建立参考能级">取激光终点</button></div><div class="reference-definition full-form-row" id="reference-definition"></div><div class="form-actions"><button class="button" id="add-reference">添加参考能级</button><button class="quiet" id="cancel-reference-edit" hidden>取消编辑</button></div></div><div class="reference-list">${scene.references!.map(r=>`<div class="reference-row"><button class="state-chip" data-state="${r.id}">${esc(r.name)}</button><span title="${r.anchorMode==='zero-field'?'使用零磁场锚点，随当前磁场保持固定':'跟随当前磁场下的锚点能量'}">${stateHTML(r.anchorId)} ${r.anchorMode==='zero-field'?'· B=0':''} ${r.offsetMHz<0?'−':'+'} ${frequencyText(Math.abs(r.offsetMHz))}</span><button class="quiet" data-edit-reference="${r.id}" aria-label="编辑参考能级 ${esc(r.name)}">编辑</button><button class="icon-button" data-remove-reference="${r.id}" aria-label="删除参考能级 ${esc(r.name)}">×</button></div>`).join('')}</div></details>`;
}
function bindReferences():void{
  const editor=$('details').querySelector<HTMLElement>('.reference-editor');if(!editor)return;
  const input=$<HTMLInputElement>('reference-offset'),unitPicker=$<HTMLSelectElement>('reference-unit'),modeRoot=$('reference-anchor-mode');let modeControl:Segmented|undefined;const mode={get value(){return modeRoot.dataset.value??'follow';},set value(value:string){modeRoot.dataset.value=value;modeControl?.setCurrent(value);}};
  const definition=()=>{$('reference-definition').innerHTML=math(mode.value==='zero-field'?'E_{ref}=E_{anchor}(B=0)+h\\delta':'E_{ref}(B)=E_{anchor}(B)+h\\delta');};
  if(!editor.dataset.bound){editor.dataset.bound='true';let unit=unitPicker.value as DetuningUnit,canonical=Number(input.value)*DETUNING_UNIT_MHZ[unit],shown=input.value;const readMHz=()=>{const text=spin.text;if(text!==shown){canonical=Number(text)*DETUNING_UNIT_MHZ[unit];shown=text;}return canonical;};const setMHz=(value:number)=>{canonical=value;input.value=String(value/DETUNING_UNIT_MHZ[unit]);shown=input.value;spin.refresh();};const spin=referenceSpin=new SpinBox($('reference-number'),input,{step:()=>.01,bounds:()=>[-1e12/DETUNING_UNIT_MHZ[unit],1e12/DETUNING_UNIT_MHZ[unit]],commit:n=>setMHz(n*DETUNING_UNIT_MHZ[unit]),decimal:true});referenceNumber={readMHz,setMHz};spin.refresh();unitPicker.onchange=()=>{const next=unitPicker.value as DetuningUnit,value=readMHz();if(!input.value.trim()||!Number.isFinite(value)){unitPicker.value=unit;return;}unit=next;referenceUnit=next;setMHz(value);};editor.addEventListener('input',event=>{editor.dataset.dirty='true';(event.target as HTMLElement).removeAttribute('aria-invalid');});}
  modeControl=new Segmented(value=>{mode.value=String(value);editor.dataset.dirty='true';definition();});modeControl.setChoices([['follow','随锚点'],['zero-field','固定零场']]);modeControl.setCurrent(mode.value);modeControl.root.setAttribute('aria-label','参考能级能量定义');modeRoot.replaceChildren(modeControl.root);
  const anchorPicker=$<HTMLSelectElement>('reference-anchor');if(!treeOwners.has(anchorPicker)){const tree=new TreeCombo(()=>{editor.dataset.dirty='true';});tree.setTree(stateTree(allLevels()),anchorPicker.value);tree.root.id=anchorPicker.id;tree.root.setAttribute('aria-label','参考能级锚点');treeOwners.set(tree.root,tree);anchorPicker.replaceWith(tree.root);}
  requestAnimationFrame(()=>shareLabelColumn(editor,64));
  const resetDraft=()=>{referenceEditingId=null;$<HTMLInputElement>('reference-name').value='虚拟中间态';setChoiceValue($<HTMLSelectElement>('reference-anchor'),referenceAnchorDefault());mode.value='follow';referenceNumber!.setMHz(100);delete editor.dataset.dirty;$('reference-form-caption').textContent='新建参考能级';$('add-reference').textContent='添加参考能级';$('cancel-reference-edit').hidden=true;definition();};
  definition();const beam=activeLaser(),photon=beam?referenceFromLaser(scene,beam,allLevels()):undefined;$<HTMLButtonElement>('reference-from-laser').disabled=!photon;$('reference-from-laser').title=photon?.emission?'参考出发态 − hν':'参考出发态 + hν';$('reference-from-laser').onclick=()=>{if(!photon)return;setChoiceValue($<HTMLSelectElement>('reference-anchor'),photon.anchorId);mode.value=photon.anchorMode;editor.dataset.dirty='true';referenceNumber!.setMHz(photon.offsetMHz);definition();};
  $('add-reference').onclick=()=>{if(!referenceEditingId&&scene.references!.length>=12){toast('最多 12 个参考能级');return;}const name=$<HTMLInputElement>('reference-name').value.trim(),raw=input.value.trim(),offset=referenceNumber!.readMHz();if(!name||!raw||!Number.isFinite(offset)||Math.abs(offset)>1e12){$<HTMLInputElement>('reference-name').toggleAttribute('aria-invalid',!name);input.toggleAttribute('aria-invalid',!raw||!Number.isFinite(offset)||Math.abs(offset)>1e12);toast('请填写名称与有限的频率偏移（最大 ±1,000,000 THz）');return;}const id=referenceEditingId??`reference-${Date.now()}`,reference={id,name,anchorId:$<HTMLSelectElement>('reference-anchor').value,anchorMode:mode.value as 'follow'|'zero-field',offsetMHz:offset},index=scene.references!.findIndex(r=>r.id===id);if(index>=0)scene.references![index]=reference;else scene.references!.push(reference);resetDraft();comparisonSource=undefined;selectedStates=[id];persist();renderInspector();renderResults(false);};
  $('cancel-reference-edit').onclick=resetDraft;
  editor.querySelectorAll<HTMLButtonElement>('[data-edit-reference]').forEach(button=>button.onclick=()=>{const ref=scene.references!.find(r=>r.id===button.dataset.editReference)!;referenceEditingId=ref.id;(editor as HTMLDetailsElement).open=true;delete editor.dataset.dirty;$('reference-form-caption').textContent='编辑参考能级';$<HTMLInputElement>('reference-name').value=ref.name;setChoiceValue($<HTMLSelectElement>('reference-anchor'),ref.anchorId);mode.value=ref.anchorMode??'follow';referenceNumber!.setMHz(ref.offsetMHz);$('add-reference').textContent='保存参考能级';$('cancel-reference-edit').hidden=false;definition();$<HTMLInputElement>('reference-name').focus();});
  editor.querySelectorAll<HTMLButtonElement>('[data-remove-reference]').forEach(button=>{const action=()=>{const id=button.dataset.removeReference;for(const beam of scene.lasers)if(beam.referenceStateId===id)delete beam.referenceStateId;scene.references=scene.references!.filter(r=>r.id!==id);selectedStates=selectedStates.filter(id=>id!==button.dataset.removeReference);if(referenceEditingId===id)resetDraft();update();};const remove=command('×',action,{role:'destructive',glyph:true,square:true,title:button.getAttribute('aria-label')??'删除参考能级'}).root;remove.dataset.removeReference=button.dataset.removeReference;remove.setAttribute('aria-label',button.getAttribute('aria-label')??'删除参考能级');button.replaceWith(remove);});
}

function dynamicsKey(steadyOnly=view==='levels'&&populationMode==='steady'):string{return dynamicsCacheKey(scene,dynamics(),steadyOnly);}

function cachedAnalysis():Analysis {
  const key=JSON.stringify({kind:view==='levels'?'spectrum':'bare',fields:scene.fields,lasers:scene.lasers,initial:initialId(),excluded:view==='levels'?activeLaser()?.id:undefined});
  const cached=analysisCache.get(key);if(cached)return cached;
  const start=performance.now(),analysis=view==='levels'?spectroscopyAnalysis(scene,activeLaser()?.id):analyze(scene);
  analysisCache.set(key,analysis);while(analysisCache.size>4)analysisCache.delete(analysisCache.keys().next().value!);
  root.dataset.spectrumBuildCount=String(++spectrumBuildCount);root.dataset.spectrumComputeMs=String(performance.now()-start);return analysis;
}
function invalidateObsoletePopulation():void {
  // Retain the displayed calculation only; revisiting earlier parameters must solve again.
  const physicalKey=dynamicsKey(false);displayedResults.activate(physicalKey);
  const key=dynamicsKey();if(requestedPhysicalKey&&requestedPhysicalKey!==physicalKey)cancelObsoleteSolve();
  if(renderedPopulationKey&&renderedPopulationKey!==key){
    if(view==='levels')patchLevelPopulations($('main-chart'),{label:'布居待计算',mean:false,selectedLaser:!!activeLaser(),initialId:initialId()});
    const oldReadout=$('selection-readout').querySelector<HTMLElement>('[data-current-population]');if(oldReadout){oldReadout.hidden=true;const value=oldReadout.querySelector('b');if(value)value.textContent='—';}
    root.dataset.populationFresh='false';if(latestAnalysis&&view==='levels')populationToolbar();
  }
}
function refreshPopulationOnly():void {
  if(view==='levels'){
    const snapshot=populationSnapshot(),steady=steadyPopulationsInView();populationToolbar();
    patchLevelPopulations($('main-chart'),{populations:snapshot.populations,steady,label:snapshot.label,mean:isDisplayedMean(),selectedLaser:!!activeLaser(),initialId:initialId()});
    alignLevelReadouts($('main-chart'));
    // The readout editor and its focusable state chips remain mounted.
    const p=$('selection-readout').querySelector<HTMLElement>('[data-current-population]'),value=selectedStates.length?snapshot.populations?.[selectedStates.at(-1)!]:undefined;
    if(p){p.hidden=value===undefined;const b=p.querySelector('b');if(b)b.textContent=value===undefined?'—':populationValue(value);}
    const details=$('details').querySelector<HTMLDetailsElement>('.solver-details');if(details){const next=document.createElement('div');next.innerHTML=solverDetails();const summary=details.querySelector<HTMLElement>('summary');reconcileChildren(details,[...next.firstElementChild!.children].map(child=>child.tagName==='SUMMARY'&&summary?summary:child as HTMLElement));}
  }else if(view==='dynamics')renderDynamics(latestAnalysis,false);
  renderedPopulationKey=dynamicsKey();root.dataset.populationFresh=String(!!currentResult());root.dataset.populationPaintCount=String(++populationPaintCount);refreshStatus();
}

function requestSolve(force=false):void{
  const invalid=classifyLasers(scene).find(c=>!c.valid);
  if(invalid){cancelObsoleteSolve();solverError=invalid.message;requestedKey=dynamicsKey();refreshPopulationOnly();return;}
  const key=dynamicsKey(),physicalKey=dynamicsKey(false),steadyOnly=view==='levels'&&populationMode==='steady';
  displayedResults.activate(physicalKey);
  const completed=displayedResults.get(physicalKey,steadyOnly?'mean':'trajectory');
  if(!force&&completed)return;
  if(!force&&key===requestedKey)return;
  // A current full solve already supplies its own mean. Merely opening the
  // level view must not cancel/restart it or submit a duplicate mean request.
  if(!force&&solverBusy&&requestedPhysicalKey===physicalKey&&(!requestedSteadyOnly||steadyOnly))return;
  if(force)displayedResults.clear();
  const epoch=displayedResults.activate(physicalKey);
  solver.cancel();requestedKey=key;requestedPhysicalKey=physicalKey;requestedSteadyOnly=steadyOnly;
  const revision=++solverRevision;clearTimeout(solverTimer);solverBusy=true;solverQueued=!force;solverError='';
  const submittedScene=structuredClone(scene),submittedSettings=structuredClone(dynamics()),submittedAt=performance.now();
  const stillCurrent=()=>revision===solverRevision&&physicalKey===dynamicsKey(false);
  const submittedOptions={steadyOnly,onProgress:(result:DynamicsResult)=>{
    if(!stillCurrent())return;
    const record={result,wallMs:performance.now()-submittedAt};
    displayedResults.set(physicalKey,epoch,'mean',record);displayedResults.set(physicalKey,epoch,'progress',record);
    refreshPopulationOnly();
  }};
  solverTimer=window.setTimeout(async()=>{
    if(!stillCurrent())return;solverQueued=false;refreshPopulationOnly();const begin=performance.now();
    try{
      const result=await solver.solve(submittedScene,submittedSettings,submittedOptions);if(!stillCurrent())return;
      solverWallMs=performance.now()-begin;const record={result,wallMs:solverWallMs};
      displayedResults.set(physicalKey,epoch,steadyOnly?'mean':'trajectory',record);
      if(result.steady.converged)displayedResults.set(physicalKey,epoch,'mean',record);
      solverBusy=false;refreshPopulationOnly();
    }catch(e){if(!stillCurrent())return;solverBusy=false;solverError=e instanceof Error?e.message:String(e);refreshPopulationOnly();}
  },force?0:16);
  refreshPopulationOnly();
}
function cancelObsoleteSolve():void{solver.cancel();solverRevision++;clearTimeout(solverTimer);requestedKey='';requestedPhysicalKey='';requestedSteadyOnly=false;solverBusy=false;solverQueued=false;solverError='';}
function renderDynamicsControls(a:Analysis):void{
  const d=dynamics(),enabled=scene.lasers.filter(l=>l.enabled&&laserIntensityMWcm2(l)>0);d.durationUnit??=durationUnit(d);if(!enabled.some(l=>l.id===d.laserId))d.laserId=enabled[0]?.id??'';
  const states=allLevels(a);d.monitoredGroupIds??=[];
  if(!states.some(l=>l.id===d.initialStateId))d.initialStateId=initialId();d.monitoredStateIds=d.monitoredStateIds.filter(id=>states.some(l=>l.id===id));
  const key=JSON.stringify(states.map(l=>l.id)),parent=$('dynamics-controls');
  if(key===controlsKey){
    const initial=parent.querySelector<HTMLSelectElement>('[aria-label="初态"]');if(initial)setChoiceValue(initial,d.initialStateId);
    monitorModeSegments.get(parent)?.setCurrent(d.monitorMode??'auto');
    parent.querySelectorAll<HTMLButtonElement>('[data-monitor-group]').forEach(input=>setChecked(input,d.monitoredGroupIds!.includes(input.dataset.monitorGroup!)));
    const beamWords=parent.querySelector('.active-beams strong');if(beamWords)beamWords.textContent=enabled.length?enabled.map(l=>l.name).join(' + '):'无光束';
    parent.querySelectorAll<HTMLButtonElement>('[data-monitor]').forEach(input=>setChecked(input,d.monitoredStateIds.includes(input.dataset.monitor!)));const count=parent.querySelector('.monitor-controls summary span');if(count)count.textContent=`${populationDisplayIds(d).length} 条`;
    $<HTMLButtonElement>('monitor-all').disabled=d.monitoredStateIds.length===states.length;$<HTMLButtonElement>('monitor-none').disabled=populationDisplayIds(d).length===0;
    return;
  }
  controlsKey=key;const monitorsWereOpen=parent.querySelector<HTMLDetailsElement>('.monitor-controls')?.open??false;disposeTrees(parent);parent.replaceChildren();
  const config=form('dynamics-config');parent.append(config);
  const beamNames=document.createElement('strong');beamNames.textContent=enabled.length?enabled.map(l=>l.name).join(' + '):'无光束';const beams=fieldRow('驱动',beamNames);beams.classList.add('active-beams');config.append(beams);
  selectControl(config,'初态',d.initialStateId,states.map(l=>[l.id,stateText(l)]),id=>setInitialState(id));
  monitorModeSegments.set(parent,segmentedControl(config,'显示态',d.monitorMode??'auto',[['auto','初末各 3 态'],['manual','手动选择']],value=>{d.monitorMode=value as 'auto'|'manual';if(value==='auto')d.monitoredGroupIds=[];else parent.querySelector<HTMLDetailsElement>('.monitor-controls')!.open=true;update();}));
  const duration=form('duration-control');parent.append(duration);numeric(duration,'时长','duration',durationValue(d),Number.MIN_VALUE,Number.MAX_VALUE,.02,durationUnit(d),n=>setDurationValue(d,n),{max:8},{choices:DURATION_UNITS,change:unit=>setDurationUnit(d,unit as DurationUnit)});numeric(duration,'data_points','data-points',d.samples,2,2001,1,'',n=>{if(!Number.isInteger(n))throw new Error('数据点数必须为整数');d.samples=n;},{max:4});
  const actions=document.createElement('div');actions.className='dynamics-actions';actions.innerHTML='<button id="run-dynamics" class="button button-primary">计算</button>';parent.append(actions);$('run-dynamics').onclick=()=>{requestSolve(true);};
  const monitors=document.createElement('details');monitors.className='monitor-controls';monitors.open=monitorsWereOpen;monitors.innerHTML=`<summary>观察曲线 <span>${populationDisplayIds(d).length} 条</span></summary><div class="monitor-actions"><button class="button quiet" id="monitor-all" ${d.monitoredStateIds.length===states.length?'disabled':''}>全部 m_F</button><button class="button quiet" id="monitor-none" ${populationDisplayIds(d).length===0?'disabled':''}>清空</button></div><div class="monitor-tree"></div>`;parent.append(monitors);
  $('monitor-all').onclick=()=>{d.monitorMode='manual';d.monitoredStateIds=states.map(state=>state.id);update();};$('monitor-none').onclick=()=>{d.monitorMode='manual';d.monitoredStateIds=[];d.monitoredGroupIds=[];update();};
  const list=monitors.querySelector<HTMLElement>('.monitor-tree')!;
  const groupBox=(group:typeof POPULATION_GROUPS[number],label=group.label)=>{const box=checkBox(label,on=>{d.monitorMode='manual';d.monitoredGroupIds=on?[...d.monitoredGroupIds!,group.id]:d.monitoredGroupIds!.filter(id=>id!==group.id);update();});box.dataset.monitorGroup=group.id;box.classList.add('monitor-option');box.title=group.label+' · 所有 m 态布居之和';setChecked(box,d.monitoredGroupIds!.includes(group.id));return box;};
  list.append(groupBox(POPULATION_GROUPS.find(g=>g.id==='F:excited')!));
  for(const branch of stateTree(states)){
    const section=document.createElement('details');section.className='monitor-manifold';section.open=branch.heading==='5S₁/₂';const summary=document.createElement('summary');summary.textContent=branch.heading;section.append(summary);
    for(const F of branch.children??[]){const row=document.createElement('div');row.className='monitor-f-row';const ids=F.leaves.map(l=>l.key),group=POPULATION_GROUPS.find(g=>g.members.length===ids.length&&g.members.every(id=>ids.includes(id)))!;row.append(groupBox(group,F.heading+' 总布居'));const choices=document.createElement('div');choices.className='monitor-list';
      for(const leaf of F.leaves){const box=checkBox(leaf.label,on=>{d.monitorMode='manual';d.monitoredStateIds=on?[...d.monitoredStateIds,leaf.key]:d.monitoredStateIds.filter(id=>id!==leaf.key);update();});box.dataset.monitor=leaf.key;box.classList.add('monitor-option');box.title=leaf.text;box.setAttribute('aria-label','观察 '+leaf.text);setChecked(box,d.monitoredStateIds.includes(leaf.key));choices.append(box);}row.append(choices);section.append(row);
    }list.append(section);
  }

  requestAnimationFrame(()=>shareLabelColumn(parent,64));
}
function currentRecord():SolvedRecord|undefined{
  const key=dynamicsKey(false);
  if(view==='levels'&&populationMode==='steady')return displayedResults.get(key,'mean');
  return displayedResults.get(key,'trajectory')??displayedResults.get(key,'progress');
}
function currentResult():DynamicsResult|undefined{return currentRecord()?.result;}
function currentSolverError():string{return requestedKey===dynamicsKey()?solverError:'';}
function currentSolverBusy():boolean{return solverBusy&&requestedPhysicalKey===dynamicsKey(false)&&(!requestedSteadyOnly||(view==='levels'&&populationMode==='steady'));}
function isMeanStatus(status:string|undefined):boolean{return status==='periodic-mean'||status==='initial-dependent-mean';}
function isDisplayedMean(result=currentResult(),a=latestAnalysis):boolean{return isMeanStatus(result?.steady?.status)||(view==='levels'&&a.spectroscopy?.requiresDensityProjection===true&&result?.steady?.status==='population-stationary');}
function steadyPopulationsInView(a=latestAnalysis):Record<string,number>|undefined {
  const result=currentResult(),steady=result?.steady;if(!result||!steady?.converged||!steady.populations)return undefined;
  if(view!=='levels')return steady.populations;
  let entries=projectedPopulationCache.get(a);if(!entries){entries=new WeakMap();projectedPopulationCache.set(a,entries);}if(entries.has(steady))return entries.get(steady)??undefined;
  const projected=projectSpectroscopyPopulations(a,{states:result.states.map(state=>state.id),populations:result.states.map(state=>steady.populations![state.id]??0),density:steady.density,basis:result.states});entries.set(steady,projected);return projected??undefined;
}
function populationSnapshot():{populations?:Record<string,number>;label:string;timeUs?:number}{
  const result=currentResult();
  if(!result)return {label:currentSolverBusy()?'计算中':'待计算'};
  if(populationMode==='steady'){const populations=steadyPopulationsInView();return result.steady?.converged&&result.steady.populations?{populations,label:populations?(isDisplayedMean(result)?'长期平均布居':'稳态布居'):'当前背景布居未定义'}:{label:'长期布居未解析'};}
  const index=populationTimeIndex<0?result.timesUs.length-1:Math.min(populationTimeIndex,result.timesUs.length-1),timeUs=result.timesUs[index]??0;
  if(view==='levels'&&latestAnalysis.spectroscopy?.status==='dispersive'&&latestAnalysis.spectroscopy.requiresDensityProjection)return {label:'该基底的 p(t) 需要完整瞬态密度矩阵',timeUs};
  return {populations:Object.fromEntries(Object.entries(result.populations).map(([id,values])=>[id,values[index]??0])),label:`p(${number(timeUs,3)} μs)`,timeUs};
}
function solverStatus():string{
  const result=currentResult(),error=currentSolverError();
  if(error)return `<span class="solver-state error" title="${esc(error)}"><i></i>${result?.steady.converged?'时间演化失败 · 长期布居已就绪':'布居失败'}</span>`;
  if(currentSolverBusy())return `<span class="solver-state ${solverQueued?'pending':'busy'}"><i></i>${result?.steady.converged?'长期布居已就绪 · 演化计算中':solverQueued?'布居待计算':'布居计算中'}</span>`;
  if(!result)return '<span class="solver-state pending"><i></i>布居待计算</span>';
  if(view==='dynamics'&&result.timesUs.length>1)return `<span class="solver-state ready"><i></i>${result.steady?.converged?'轨迹已计算':'轨迹已计算 · 长期布居未解析'}</span>`;
  return `<span class="solver-state ${result.steady?.converged?'ready':'error'}"><i></i>${result.steady?.converged?(isDisplayedMean(result)?'长期平均已收敛':'稳态已收敛'):'长期布居未解析'}</span>`;
}
function solverDetails():string{
  const result=currentResult(),error=currentSolverError();
  return `<details class="solver-details"><summary>计算详情</summary>${error?`<p class="solver-error">${esc(error)}</p>`:''}${result?`<div class="solver-readouts"><span>32 态 · ${esc(result.metrics.kernel)}<b>${number(result.metrics.totalMs,1)} ms</b></span><span>含通信<b>${number(currentRecord()?.wallMs??solverWallMs,1)} ms</b></span><span>概率守恒误差<b>${result.validation.maxTraceError.toExponential(2)}</b></span><span>最小布居<b>${result.validation.minPopulation.toExponential(2)}</b></span><span>稳态残差<b>${result.steady?.residual==null?'—':result.steady.residual.toExponential(2)}</b></span><span>状态<b>${esc(result.steady?.status??'not-converged')}</b></span></div><p class="helper-text">${esc(result.method)}</p><p class="helper-text">${esc(result.steady?.method??'')}</p>${result.steady?.reason?`<p class="helper-text">${esc(result.steady.reason)}</p>`:''}${result.warnings.map(w=>`<p class="helper-text">${esc(w)}</p>`).join('')}`:'<p class="helper-text">没有当前参数的求解结果。</p>'}${failureRecords.length?'<button class="button" data-export-solver-failures>导出失败记录</button>':''}</details>`;
}
function populationToolbar():void{
  const target=$('population-toolbar'),snap=populationSnapshot(),result=currentResult(),p=snap.populations;
  const f1=p?latestAnalysis.ground.filter(l=>l.F===1).reduce((v,l)=>v+(p[l.id]??0),0):undefined,f2=p?latestAnalysis.ground.filter(l=>l.F===2).reduce((v,l)=>v+(p[l.id]??0),0):undefined;
  const controlKey=`${populationMode}:${result?.timesUs.length??0}:${result?.steady?.status??''}:${isDisplayedMean(result)}`;
  if(target.dataset.controlKey!==controlKey){
    target.dataset.controlKey=controlKey;
    const incoming=document.createElement('div');incoming.innerHTML=`<select id="population-mode" aria-label="能级布居着色"><option value="steady" ${populationMode==='steady'?'selected':''}>${isDisplayedMean(result)?'长期平均布居':'稳态布居'}</option><option value="time" ${populationMode==='time'?'selected':''}>指定时刻 p(t)</option></select>${populationMode==='time'&&result?`<input id="population-time" type="range" aria-label="布居时间" min="0" max="${result.timesUs.length-1}" value="${populationTimeIndex<0?result.timesUs.length-1:Math.min(populationTimeIndex,result.timesUs.length-1)}"><span id="population-time-value"></span>`:''}<span class="population-scale" title="固定色标：0% → 100%，不按当前最大布居重新缩放"></span><div class="population-data" id="population-data"></div>`;
    const controls=[...incoming.children].map(child=>{
      const element=child as HTMLElement,selector=element.id?`#${element.id}`:element.classList.contains('population-scale')?'.population-scale':undefined,existing=selector?target.querySelector<HTMLElement>(selector):null;
      if(!existing)return element;
      if(existing instanceof HTMLSelectElement){const option=element.querySelector('option[value="steady"]');if(option)existing.options[0]!.text=option.textContent??'';existing.value=populationMode;}
      if(existing instanceof HTMLInputElement&&element instanceof HTMLInputElement){existing.min=element.min;existing.max=element.max;existing.value=element.value;}
      return existing;
    });reconcileChildren(target,controls);
    $<HTMLSelectElement>('population-mode').onchange=e=>{populationMode=(e.target as HTMLSelectElement).value as 'steady'|'time';update();};
    if($('population-time'))$<HTMLInputElement>('population-time').oninput=e=>{populationTimeIndex=Number((e.target as HTMLInputElement).value);update();};
  }
  if($('population-time-value'))setText('population-time-value',durationText(snap.timeUs??0,durationUnit(dynamics())));
  const pe=p?Object.entries(p).reduce((sum,[id,value])=>sum+(id.startsWith('ground-')?0:value),0):undefined;
  const incomingData=document.createElement('div');incomingData.innerHTML=`<span class="population-total" title="${f1===undefined?esc(snap.label):`${populationValue(f1)}`}">F = 1 <b>${populationPercent(f1)}</b></span><span class="population-total" title="${f2===undefined?esc(snap.label):`${populationValue(f2)}`}">F = 2 <b>${populationPercent(f2)}</b></span><span class="population-total" title="${pe===undefined?esc(snap.label):`${populationValue(pe)}`}">激发 <b>${populationPercent(pe)}</b></span>${solverStatus()}${currentSolverError()?'<button class="text-button solver-error-link" data-show-solver-details>查看原因</button>':''}<button class="button" data-run-dynamics ${currentSolverBusy()&&!solverQueued?'disabled':''}>计算布居</button>`;
  const dataHost=$('population-data');reconcileChildren(dataHost,[...incomingData.children].map(child=>{
    const element=child as HTMLElement,key=element.hasAttribute('data-run-dynamics')?'[data-run-dynamics]':element.hasAttribute('data-show-solver-details')?'[data-show-solver-details]':undefined,existing=key?dataHost.querySelector<HTMLButtonElement>(key):null;
    if(existing){existing.disabled=element.hasAttribute('disabled');existing.textContent=element.textContent;return existing;}return element;
  }));
}
function renderDynamics(a:Analysis,allowSolve=true):void{
  const d=dynamics(),raw=currentResult();if((d.monitorMode??'auto')==='auto')d.monitoredStateIds=recommendedPopulationStates(raw,d.initialStateId);
  renderDynamicsControls(a);const ids=populationDisplayIds(d),result=raw?withPopulationGroups(raw,d.monitoredGroupIds??[]):undefined;
  $('summary').innerHTML=`<span>32 态 Lindblad</span>${solverStatus()}${result?`<span>${number(result.metrics.totalMs,1)} ms</span>`:''}`;
  setText('plot-title','Population');setText('plot-badge','');
  const labels=Object.fromEntries([...allLevels(a).map(l=>[l.id,stateText(l)]),...POPULATION_GROUPS.map(g=>[g.id,g.label])]);
  unbindPopulation?.();unbindPopulation=undefined;
  const plotOptions={showTotalExcited:false,timeUnit:durationUnit(d),scale:populationScale,viewportWidth:Math.max(580,$('main-chart').clientWidth-18),initialId:d.initialStateId};
  $('main-chart').innerHTML=result&&result.timesUs.length>1?populationChart(result,ids,labels,plotOptions):result?.steady.converged?`<div class="empty-state"><div><p>长期布居</p><div class="solver-readouts">${ids.map(id=>`<span>${esc(labels[id]??id)}<b>${populationValue(result.steady.populations?.[id]??0)}</b></span>`).join('')}</div><p>${currentSolverError()?'时间演化未完成，详情见下方。':'时间演化计算中…'}</p></div></div>`:`<div class="empty-state">${currentSolverError()?esc(currentSolverError()):currentSolverBusy()?'正在求解当前场景…':'点击计算或开启自动求解。'}</div>`;
  if(result&&result.timesUs.length>1)unbindPopulation=bindPopulationChart($('main-chart'),result,ids,labels,plotOptions);
  $('chart-caption').innerHTML=ids.map(id=>`<span class="population-legend"><svg width="18" height="5" aria-hidden="true"><line x1="0" y1="2.5" x2="18" y2="2.5" stroke="${populationColor(id)}" stroke-width="2"/></svg>${id===d.initialStateId?'<small class="initial-symbol">ψ₀</small>':''}${id.startsWith('F:')?esc(labels[id]):stateHTML(id)}</span>`).join('');

  renderDetails(solverDetails());
}

function refreshDirectionStates(a:Analysis):void {
  const current=selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):scene.fields.find(f=>f.id===selection.id);if(!current)return;
  const currentAxis:AxisAngles=selection.kind==='laser'?laserDirectionAxis(current as Laser):current;
  const sameAxis=(axis:AxisAngles)=>currentAxis.thetaDeg===axis.thetaDeg&&(currentAxis.thetaResidualDeg??0)===(axis.thetaResidualDeg??0)&&(((axis.thetaDeg===0||axis.thetaDeg===180)&&!(axis.thetaResidualDeg??0))||((currentAxis.phiDeg===axis.phiDeg||Math.abs(currentAxis.phiDeg-axis.phiDeg)===360)&&(currentAxis.phiResidualDeg??0)===(axis.phiResidualDeg??0)));
  const sameDirection=(theta:number,phi:number)=>sameAxis({thetaDeg:theta,phiDeg:phi});
  const axis=a.fieldAxis;
  for(const {control,kind}of directionSegments){if(kind==='axes'){const found=['90,0','90,180','90,90','90,-90','0,0','180,0'].find(value=>{const [theta,phi]=value.split(',').map(Number);return sameDirection(theta!,phi!);});control.setCurrent(found??null);}else{const modes=['parallel','anti','perpendicular']as const;for(const mode of modes)control.setWithheld(mode,a.fieldMagnitudeG===0);control.setCurrent(a.fieldMagnitudeG>0?modes.find(mode=>sameAxis(relativeDirectionAxis(axis,mode)))??null:null);}}

}

function renderResults(allowSolve=true):void{
  const focused=document.activeElement?.closest<HTMLElement>('[data-state],[data-laser-id],[data-transition]'),focusHost=focused&&($('main-chart').contains(focused)?$('main-chart'):$('details').contains(focused)?$('details'):undefined),focusAttribute=focused&&['data-state','data-laser-id','data-transition'].find(attr=>focused.hasAttribute(attr)),focusValue=focusAttribute?focused?.getAttribute(focusAttribute):undefined;
  invalidateObsoletePopulation();
  let a:Analysis;try{a=cachedAnalysis();latestAnalysis=a;selectedStates=selectedStates.filter(id=>allLevels(a).some(state=>state.id===id)||scene.references?.some(ref=>ref.id===id));if(comparisonSource&&!allLevels(a).some(s=>s.id===comparisonSource)&&!scene.references?.some(r=>r.id===comparisonSource))comparisonSource=undefined;}catch(error){setText('status-text','能级参数无效');console.error(error);return;}
  refreshDirectionStates(a);
  const regimeErrors=classifyLasers(scene).filter(c=>!c.valid);
  const regimeStatus=document.getElementById('laser-regime-status');if(regimeStatus){const selectedCheck=classifyLasers(scene).find(c=>c.id===activeLaser()?.id);regimeStatus.textContent=selectedCheck?.message??'';regimeStatus.hidden=!selectedCheck?.message;}
  const laser=activeLaser(view==='levels'?spectral:undefined),pol=laser?a.polarizations[laser.id]:undefined;$('clear-laser-selection').hidden=!laser;
  document.querySelectorAll<HTMLButtonElement>('.view-tabs button[data-view]').forEach(b=>{const active=b.dataset.view===view;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));});manifoldFocus.setCurrent(spectral);
  $('line-tabs').hidden=view!=='levels';$('chart-info').hidden=view==='space';$('chart-info').setAttribute('aria-label',view==='dynamics'?'布居图说明':'能级图说明');$('chart-info').title=view==='dynamics'?'采样说明':'能级图说明';$('selection-readout').hidden=view!=='levels';$('dynamics-controls').hidden=view!=='dynamics';$('population-toolbar').hidden=view!=='levels';
  if(view!=='space'&&spaceMount.isConnected)spaceMount.remove();
  if(view!=='dynamics'&&unbindPopulation){unbindPopulation();unbindPopulation=undefined;}populationScaleControl.root.hidden=view!=='dynamics';
  root.dataset.view=view;
  if($('inspector-polarization'))$('inspector-polarization').innerHTML=polarizationBars(a,selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):laser);
  if($('laser-absolute-frequency')){const l=scene.lasers.find(l=>l.id===selection.id);if(l)setText('laser-absolute-frequency',`${number(laserFrequencyMHz(l)/1e6,12)} THz`);}
  if(view==='levels'){
    populationToolbar();
    const initial=referenceStateId(laser),diagramAnalysis=a,transitions=(a.spectroscopy?.status==='unresolved'?[]:[...incidentTransitions(a,initial,laser?.id,'D1'),...incidentTransitions(a,initial,laser?.id,'D2')]).sort((x,y)=>y.strength*y.weight-x.strength*x.weight),snapshot=populationSnapshot();
    const clockGap=a.spectroscopy?.status==='unresolved'?NaN:(a.ground.find(state=>state.id==='ground-2-0')!.energyMHz-a.ground.find(state=>state.id==='ground-1-0')!.energyMHz);
    $('summary').innerHTML=`<span>B <b>${measuredNumber(a.fieldMagnitudeG,3)} G</b></span><span>Clock <b>${number(clockGap/1000,6)} GHz</b></span><span class="subtle">32 态 · Lindblad</span>`;
    setText('plot-title',a.spectroscopy?.status==='unresolved'?'裸态能级':laser?'背景能级':'总场能级');setText('plot-badge',a.spectroscopy?.status==='unresolved'?(laser?'裸态失谐 · 背景光移未确定':'静态光移未定义'):populationMode==='time'&&a.spectroscopy?.requiresDensityProjection?'当前基底 p(t) 未提供':laser&&!laser.enabled?'已停用':'');if(regimeErrors.length){setText('plot-badge','激光类型与失谐不匹配');$('plot-badge').title=regimeErrors.map(c=>c.message).join('\n');}const scope=[...regimeErrors.map(c=>c.message),a.spectroscopy?.modelScope,...(a.spectroscopy?.warnings??[])].filter(Boolean).join('\n');$('plot-badge').title=scope;$('chart-info').title+='\n'+scope;
    renderSelection(a,laser);
    $('main-chart').innerHTML=levelsChart(scene,diagramAnalysis,laser,{line:spectral,selectedIds:markedComparison(laser),initialId:dynamics().initialStateId,populations:snapshot.populations,steadyPopulations:steadyPopulationsInView(a),steadyPopulationLabel:isDisplayedMean(currentResult(),a)?'mean':'limit',populationLabel:snapshot.label,viewportWidth:Math.max(620,$('main-chart').clientWidth-18)});
    $('chart-caption').innerHTML='';
    renderDetails(`<details class="transition-disclosure"><summary>跃迁矩元 <span>${laser?esc(laser.name):spectral}</span></summary><div class="table-scroll"><table class="transition-table"><thead><tr><th>参考态 → 末态</th><th>分量</th><th>(ν − νref) / MHz</th><th>Δ / MHz</th><th>Ω/2π / MHz</th><th title="相对 D2 stretched cycling 跃迁的偶极强度，并乘当前偏振权重">耦合强度</th></tr></thead><tbody>${transitionRows(transitions)}</tbody></table></div></details>${referenceEditor()}${solverDetails()}`);bindReferences();
  }else if(view==='dynamics')renderDynamics(a,allowSolve);
  else if(view==='space'){
    $('summary').innerHTML=`<span>B <b>${measuredNumber(a.fieldMagnitudeG,3)} G</b></span><span>∠(k, ${a.fieldMagnitudeG>0?'B':'z'}) <b>${pol?number(pol.angleToBDeg,1):'—'}°</b></span>`;
    setText('plot-title','方向与偏振');setText('plot-badge',laser?.name??'');
    if(!spaceMount.isConnected)$('main-chart').replaceChildren(spaceMount);if(!spaceRenderer)spaceRenderer=renderSpace3D(spaceMount,scene,a,laser?.id??'');else spaceRenderer.update(scene,a,laser?.id??'');
    $('chart-caption').innerHTML='';
    $('details').innerHTML='';
  }
  paintStateBadges();renderCount++;renderedPopulationKey=dynamicsKey();root.dataset.populationFresh=String(!!currentResult());refreshStatus();setText('model-status',view==='levels'?'能级实时 · Rust 布居':'Rb87 · Rust master equation');root.dataset.renderCount=String(renderCount);root.dataset.energyRenderedAt=String(performance.now());
  if(focusHost&&focusAttribute&&focusValue)focusHost.querySelector<HTMLElement>(`[${focusAttribute}="${CSS.escape(focusValue)}"]`)?.focus({preventScroll:true});
  if(allowSolve&&!loadFailure&&view!=='space'&&!regimeErrors.length)requestSolve();
}

function replaceScene(next:Scene,message:string):void{loadFailure='';solver.cancel();solverRevision++;clearTimeout(solverTimer);requestedKey='';requestedPhysicalKey='';requestedSteadyOnly=false;solverError='';solverBusy=false;controlsKey='';displayedResults.clear();referenceEditingId=null;referenceEditorNode=undefined;referenceSpin=undefined;referenceNumber=undefined;disposeTrees($('details'));$('details').replaceChildren();$('selection-readout').dataset.selectionKey='';scene=next;selectedStates=selectedStates.filter(id=>!id.startsWith('reference-')||scene.references?.some(r=>r.id===id));populationTimeIndex=-1;selection=scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:{kind:'field',id:scene.fields[0]?.id??''};renderInitial();renderObjects();renderInspector();update();toast(message);}

document.querySelectorAll<HTMLButtonElement>('.view-tabs button[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view as View;update();});
$('clear-laser-selection').onclick=clearSourceSelection;
root.addEventListener('atom-select-laser',event=>{const id=(event as CustomEvent<{id:string}>).detail?.id;if(scene.lasers.some(l=>l.id===id))select({kind:'laser',id},true);});root.addEventListener('atom-clear-selection',clearSourceSelection);
root.addEventListener('click',event=>{const target=event.target as Element;if(target.closest('[data-export-solver-failures]')){const url=URL.createObjectURL(new Blob([JSON.stringify(failureRecords,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='rb87-solver-failures.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);return;}if(root.classList.contains('scene-expanded')&&!target.closest('.scene-panel')){root.classList.remove('scene-expanded');syncSceneLayout();}if(target.closest('[data-run-dynamics]')){requestSolve(true);return;}if(target.closest('[data-show-solver-details]')){const detail=$('details').querySelector<HTMLDetailsElement>('.solver-details');if(detail){detail.open=true;detail.scrollIntoView({block:'nearest',behavior:'smooth'});}return;}const beam=target.closest<HTMLElement>('[data-laser-id]');if(beam){select({kind:'laser',id:beam.dataset.laserId!},true);return;}const state=target.closest<HTMLElement>('[data-state]');if(state){selectState(state.dataset.state!);return;}const tr=target.closest<HTMLElement>('[data-transition]');if(tr){const t=latestAnalysis.transitions.find(t=>t.id===tr.dataset.transition);if(t){const endpoints=transitionEndpoints(t,referenceStateId(scene.lasers.find(l=>l.id===t.laserId)));if(endpoints){comparisonSource=undefined;selectedStates=[endpoints.fromId,endpoints.toId];persist();renderResults(false);}}return;}if(target.closest('#main-chart')&&!target.closest('canvas,button,input,select'))clearSourceSelection();});root.addEventListener('keydown',event=>{if(event.key==='Escape'){clearSourceSelection();if(root.classList.contains('inspector-open')){root.classList.remove('inspector-open');$('toggle-inspector').focus();event.preventDefault();return;}if(root.classList.contains('scene-expanded')){root.classList.remove('scene-expanded');syncSceneLayout();$('toggle-scene').focus();event.preventDefault();return;}}if(event.key!=='Enter'&&event.key!==' ')return;const target=event.target as Element,el=target.closest<HTMLElement>('g[data-state],g[data-laser-id],tr[data-transition]');if(el){event.preventDefault();el.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
const populationScaleControl=new Segmented(value=>{populationScale=value as 'linear'|'log';populationScaleControl.setCurrent(populationScale);update();});populationScaleControl.setChoices([['linear','线性'],['log','对数']]);populationScaleControl.setCurrent(populationScale);populationScaleControl.root.classList.add('population-scale-control');populationScaleControl.root.setAttribute('aria-label','布居坐标');$('line-tabs').after(populationScaleControl.root);
const renameObject=command('✎',async()=>{const object=selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):scene.fields.find(f=>f.id===selection.id);if(!object)return;const chosen=await prompt('重命名','名称',object.name);if(chosen===null)return;const name=chosen.trim();if(!name||name.length>64){toast('名称应包含 1 至 64 个字符');return;}object.name=name;setText('inspector-title',name);renderObjects();update();},{role:'quiet',glyph:true,square:true});renameObject.root.setAttribute('aria-label','重命名对象');renameObject.root.title='重命名';$('inspector-title').after(renameObject.root);
function nextObjectName(prefix:string,objects:{name:string}[]):string{const used=new Set(objects.map(o=>o.name));let n=1;while(used.has(`${prefix} ${n}`))n++;return `${prefix} ${n}`;}
function syncSceneLayout():void{if(innerWidth>920)root.classList.remove('inspector-open');if(innerWidth>1180)root.classList.remove('scene-expanded');root.classList.toggle('scene-compact',innerWidth<=1180?!root.classList.contains('scene-expanded'):root.classList.contains('scene-collapsed'));$('toggle-scene').setAttribute('aria-expanded',String(!root.classList.contains('scene-compact')));$('toggle-inspector').setAttribute('aria-expanded',String(root.classList.contains('inspector-open')));}
function toggleScene(force?:boolean):void{if(innerWidth<=1180){root.classList.toggle('scene-expanded',force??!root.classList.contains('scene-expanded'));}else root.classList.toggle('scene-collapsed',force===undefined?!root.classList.contains('scene-collapsed'):!force);$('toggle-scene').setAttribute('aria-expanded',String(innerWidth<=1180?root.classList.contains('scene-expanded'):!root.classList.contains('scene-collapsed')));syncSceneLayout();}
window.addEventListener('pagehide',()=>{if(saveState==='pending')saveSceneNow();cancelObsoleteSolve();});
window.addEventListener('pageshow',event=>{if(event.persisted)renderResults();});
window.addEventListener('resize',syncSceneLayout);syncSceneLayout();$('close-inspector').onclick=()=>{root.classList.remove('inspector-open');$('toggle-inspector').focus();};
$('toggle-scene').onclick=()=>toggleScene();$('initial-shortcut').onclick=()=>toggleScene(true);$('toggle-inspector').onclick=()=>root.classList.toggle('inspector-open');$('chart-info').onclick=()=>dialog.showModal();
$('add-field').onclick=()=>{if(scene.fields.length>=8)return;const id=`field-${Date.now()}`;scene.fields.push({id,name:nextObjectName('磁场',scene.fields),enabled:true,magnitudeG:1,thetaDeg:90,phiDeg:0});select({kind:'field',id});update();};$('add-laser').onclick=()=>{if(scene.lasers.length>=8)return;const id=`laser-${Date.now()}`;scene.lasers.push({id,name:nextObjectName('激光',scene.lasers),enabled:true,regime:'near',line:spectral,detuningMHz:0,intensityMWcm2:.1,thetaDeg:90,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0});controlsKey='';select({kind:'laser',id});update();};
$<HTMLSelectElement>('preset').onchange=event=>{const sel=event.target as HTMLSelectElement,p=sel.value;if(!p)return;const next=defaultScene();delete next.lasers[0]!.referenceTransition;next.lasers[0]!.frequencyMode='transition';next.lasers[0]!.detuningUnit='MHz';next.lasers.filter(l=>l.id==='tweezer').forEach(l=>l.enabled=false);next.fields[0]!.thetaDeg=0;next.fields[0]!.magnitudeG=3;next.dynamics!.durationUs=1;next.lasers[0]!.detuningMHz=4.2;
  if(p==='three'){next.name='光镊 + Probe + Repump';next.lasers[1]!.enabled=true;Object.assign(next.lasers[0]!,{regime:'near',name:'Probe',referenceTransition:{line:'D2',Fg:2,Fe:2},detuningMHz:0});delete next.lasers[0]!.frequencyAnchor;normalizeLaserFrequency(next.lasers[0]!);const repump={...next.lasers[0]!,id:'repump',name:'Repump',referenceTransition:{line:'D2' as const,Fg:1 as const,Fe:2},detuningMHz:0};delete repump.frequencyAnchor;normalizeLaserFrequency(repump);next.lasers.push(repump);next.dynamics={...defaultDynamics(),durationUs:20};view='levels';spectral='D2';}
  if(p==='open'){next.name='单束光 · F=2 → F′=2';next.fields[0]!.magnitudeG=3;next.fields[0]!.thetaDeg=0;next.fields[0]!.phiDeg=0;next.initialM=0;next.lasers[0]!.line='D2';next.lasers[0]!.detuningMHz=-266.65;next.lasers[0]!.intensityMWcm2=1;next.lasers[0]!.thetaDeg=0;next.lasers[0]!.phiDeg=0;next.lasers[0]!.ellipticityDeg=0;next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-1--1','ground-1-0','ground-1-1','ground-2-0','D2-2-1'],durationUs:20};view='levels';spectral='D2';}
  if(p==='cycling'){next.name='D2 · Cycling';next.initialM=2;next.lasers[0]!.thetaDeg=0;next.lasers[0]!.phiDeg=0;next.lasers[0]!.ellipticityDeg=45;next.lasers[0]!.intensityMWcm2=.35;next.dynamics={...defaultDynamics(),initialStateId:'ground-2-2',monitoredStateIds:['ground-2-2','D2-3-3','ground-1-0'],durationUs:5};view='levels';spectral='D2';}
  if(p==='f3-linear'){next.name='F=2 → F′=3 · 线偏振';next.initialM=0;next.lasers[0]!.detuningMHz=0;next.lasers[0]!.intensityMWcm2=1;next.lasers[0]!.ellipticityDeg=0;next.dynamics={...defaultDynamics(),durationUs:20,initialStateId:'ground-2-0'};view='levels';spectral='D2';}
  if(p==='rabi'){next.name='D2 · Rabi 演化';next.fields[0]!.magnitudeG=0;next.lasers[0]!.detuningMHz=0;next.lasers[0]!.intensityMWcm2=6.7;next.dynamics!.durationUs=.25;view='dynamics';spectral='D2';}
  if(p==='dark'){next.name='D1 · F2 → F1 抽运';next.lasers[0]!.line='D1';next.lasers[0]!.detuningMHz=0;next.lasers[0]!.intensityMWcm2=2;next.initialM=0;next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-1--1','ground-1-0','ground-1-1','ground-2-0','D1-2-1'],durationUs:3};view='dynamics';spectral='D1';}
  if(p==='both'){next.name='D1 + D2 · 同时驱动';next.fields[0]!.magnitudeG=1.5;next.fields[0]!.thetaDeg=32;next.initialM=0;next.lasers[0]!.name='D2 驱动';next.lasers[0]!.detuningMHz=0;next.lasers[0]!.intensityMWcm2=2.5;next.lasers[0]!.ellipticityDeg=18;next.lasers.push({id:'d1-drive',name:'D1 驱动',enabled:true,line:'D1',detuningMHz:0,intensityMWcm2:2.5,thetaDeg:90,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0});next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-2-0','D1-2-1','D1-2-0','D2-3-1','D2-3-0'],durationUs:.5};view='dynamics';spectral='D2';}
  if(p==='tilted'){next.name='倾斜磁场';next.fields[0]!.thetaDeg=67;next.fields[0]!.phiDeg=35;next.lasers[0]!.ellipticityDeg=28;next.initialM=0;view='space';}
  next.lasers.forEach(l=>normalizeLaserFrequency(l));replaceScene(next,'已加载示例');sel.value='';};
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!document.querySelector('dialog[open]')){clearSourceSelection();}});
const dialog=$<HTMLDialogElement>('model-dialog');$('model-button').onclick=()=>dialog.showModal();$('close-model').onclick=$('understood').onclick=()=>dialog.close();dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
$('rename-scene').onclick=async()=>{const chosen=await prompt('重命名场景','名称',scene.name);if(chosen===null)return;const name=chosen.trim();if(!name||name.length>100){toast('场景名称应包含 1 至 100 个字符');return;}scene.name=name;renderObjects();persist();};
$('export-button').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(scene,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`${scene.name.replace(/[\\/:*?"<>|]/g,'_').slice(0,100)||'rb87-scene'}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出场景、标记与动力学设置');};$('import-button').onclick=()=>$<HTMLInputElement>('file-input').click();$<HTMLInputElement>('file-input').onchange=async e=>{const input=e.target as HTMLInputElement,file=input.files?.[0];if(!file)return;try{if(file.size>100_000)throw new Error('场景文件应小于 100 KB');replaceScene(validateScene(JSON.parse(await file.text())),'已导入场景');}catch(error){toast(`无法导入：${error instanceof Error?error.message:'场景无效'}`);}finally{input.value='';}};$('reset-button').onclick=()=>replaceScene(defaultScene(),'已恢复默认场景');
let chartWidth=0,resizeFrame=0;new ResizeObserver(entries=>{const width=Math.round(entries[0]?.contentRect.width??0);if(width!==chartWidth){chartWidth=width;cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(latestAnalysis&&(view==='levels'||view==='dynamics'))renderResults(false);});}}).observe($('main-chart'));
latestAnalysis=analyze(scene);renderInitial();renderObjects();renderInspector();renderResults();
if(loadFailure){
  const original=unreadableScene,button=document.createElement('button');button.id='recover-scene';button.className='button';button.textContent='下载原本地场景';button.title=loadFailure;button.hidden=original===null;
  button.onclick=()=>{const url=URL.createObjectURL(new Blob([original!],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='rb87-unreadable-scene.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('import-button').before(button);toast(`原本地场景未载入：${loadFailure}。已显示默认场景，原数据保留。`);refreshStatus();
}
