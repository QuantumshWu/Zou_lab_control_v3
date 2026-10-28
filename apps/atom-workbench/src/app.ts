// Atom Workbench: one v3 window -- the shared title bar, one header card, one
// status strip, the scene list, a tab widget of the three analysis pages and
// 详情, and the inspector of the selected field or laser.
import {Field,alignUnitPickers,form,row as fieldRow,sectionTitle,shareLabelColumn,type FieldSpec} from '../../desktop/ui/src/fluent/form.js';
import {Dialog,confirm} from '../../desktop/ui/src/fluent/dialog.js';
import {command,type Command} from '../../desktop/ui/src/fluent/button.js';
import {element,reconcileChildren,type Value} from '../../desktop/ui/src/fluent/dom.js';
import {comboFace} from '../../desktop/ui/src/fluent/combo.js';
import {TreeCombo,type TreeGroup} from '../../desktop/ui/src/fluent/tree_combo.js';
import {StatusStrip,setState,statusDot,type StatusWord} from '../../desktop/ui/src/fluent/status.js';
import {TabWidget} from '../../desktop/ui/src/fluent/tabs.js';
import {Menu} from '../../desktop/ui/src/fluent/menu.js';
import {setMiddle} from '../../desktop/ui/src/fluent/elide.js';
import {drawBadges} from '../../desktop/ui/src/pulse/widgets.js';
import {Segmented,setChecked,switchButton,checkBox} from '../../desktop/ui/src/fluent/toggles.js';
import {clickState,keepLabels,labelled,readLabels,swapLabels,withLabel,type LabelSlot,type StateLabels} from './transition-choice.js';
import {offsetReading,photonFrequencyMHz,setReferenceOffset} from './reference-energy.js';
import {typedDigits,type Reading} from './unit-input.js';
import {frequencyBaseTree,stateTree} from './choice-trees.js';
import {alignLevelReadouts} from './readout-alignment.js';
import {expressionBase,followers,frequencyValue,frequencyValueLabel,migrateSceneFrequencies,naturalDetuningUnit,objectAnchor,objectKey,objectOf,releaseFollowers,setFrequencyBase,setFrequencyValue,type DefinedFrequency} from './laser-frequency.js';
import {FREQUENCY_UNITS,WAVELENGTH_UNITS,anchorValue,baseKey,difference,gap,inUnit,isDetuningUnit,laserValue,parseBase,photon,readingHeading,readoutText,stateValue,zeroFieldStateValue,type FrequencyValue} from './frequency-value.js';
import {FrequencyReadout,ReadoutPreferences,type ReadoutPreference} from './frequency-controls.js';
import {analyze,dipoleBetweenStates,resolveSceneFrequencies,laserIntensityMWcm2,setPolarizationPreset,type PolarizationPreset} from './physics.js';
import {AXIS_NAMES,axisWords,directionAxis,linkableFields,linkedTo,referenceAxis,releaseLinks,sameDirection,shortcutDirection,shortcutField,type DirectionShortcut} from './direction.js';
import {DEFAULT_NUMERICAL_APERTURE,INTENSITY_UNITS,POWER_UNITS,WAIST_UNITS,intensityReading,intensityUnit,normalizeLaserIntensity,powerReading,setLaserIntensityMode,setLaserNumericalAperture,waistReading,waistUnit,writeLaserIntensity,writeLaserPower,writeLaserWaist} from './laser-intensity.js';
import {dynamicsCacheKey} from './dynamics-cache-key.js';
import {CurrentDynamicsResults} from './current-dynamics-results.js';
import {STATIC_MIXING_LIMIT,displayMixing,spectroscopyAnalysis,spectroscopyCoupling,projectSpectroscopyPopulations} from './spectroscopy.js';
import {defaultDynamics,defaultScene,POPULATION_TOLERANCE,resolveMeanHorizon} from './types.js';
import type {Analysis,AxisName,Direction,DynamicsSettings,DetuningUnit,FrequencyDisplayUnit,InitialStateSpec,IntensityMode,IntensityUnit,MagneticField,PowerUnit,ReferenceAxis,ReferenceFrequency,ReferenceLevel,Laser,Level,Scene,SpectralLine,TemperatureUnit,Transition,WaistUnit} from './types.js';
import type {DynamicsResult} from './solver-types.js';
import {DynamicsClient,type SolverFailureRecord} from './solver-client.js';
import {validateScene} from './scene-file.js';
import {incidentTransitions,laserEmits,laserReferenceStateId,setLaserReferenceState,transitionEndpoints} from './transition-view.js';
import {escapeHtml as esc,levelsChart,number,populationChart,bindPopulationChart,populationColor,populationValue,populationSamplingInfo} from './charts.js';
import {math} from './math.js';
import {renderSpace3D,type SpaceSelection,type SpaceViewName} from './space3d.js';
import {laserColour} from './identity.js';
import {POPULATION_GROUPS,recommendedPopulationStates,populationDisplayIds,withPopulationGroups} from './population-selection.js';
import {durationReading,durationUnit,setDuration,durationText,timeDisplayUnit,DURATION_UNITS} from './duration.js';
import type {DurationUnit} from './types.js';
import {patchLevelPopulations} from './level-population-dom.js';
import {idMappingError} from './population-basis.js';
import {installWindowChrome} from './window-chrome.js';
import {formatElapsed,formatNumber,hyperfineLabel,idLabel,labelStateIds,signedInteger,stateLabel,subscript,withUnit} from './format.js';
import {solverText,steadyStatusText} from './solver-words.js';
import {errorPercent,outsideRuns,partialSentence,sampleErrors} from './accuracy.js';
import {definitionNotes,modelNotes} from './model-notes.js';
import {InlineName} from './inline-name.js';
import {EXAMPLES,exampleScene} from './examples.js';
import {DEFAULT_TEMPERATURE_K,INITIAL_KINDS,TEMPERATURE_UNITS,UNPOLARIZED,initialSpec,initialSummary,setInitialSpec,setTemperature,startingSpec,temperatureReading,type InitialKind,type UnpolarizedManifold} from './initial-state.js';

installWindowChrome();

type View='levels'|'dynamics'|'space'|'details';
/** The page whose populations the solver serves: 详情 reads the last of the two shown. */
type ResultView='levels'|'dynamics';
type Selection={kind:'field'|'laser'|'reference'|'level'|'none';id:string};
type ObjectKind=Exclude<Selection['kind'],'none'>;
const APP_TITLE='Rb87 实验工作台';
const STORAGE='rb87-workbench-demo-v4',root=document.querySelector<HTMLDivElement>('#app')!;
let scene=migrateSceneFrequencies(defaultScene()),view:View='levels',resultView:ResultView='levels',spectral:SpectralLine='D2',selection:Selection={kind:'laser',id:'probe'};
let populationMode:'steady'|'time'='steady',populationTimeIndex=-1,populationScale:'linear'|'log'='linear';let unbindPopulation:(()=>void)|undefined;
let spaceRenderer:ReturnType<typeof renderSpace3D>|undefined;const spaceMount=document.createElement('div');spaceMount.className='retained-space-view';
let saveState:'saved'|'pending'|'failed'='saved',saveFailure='',loadFailure='',unreadableScene:string|null=null;
let stateLabels:StateLabels={},dirty=false,saveTimer=0,renderCount=0;
let latestAnalysis:Analysis,analysisError='',solverError='',solverBusy=false,solverQueued=false,solverRevision=0,requestedKey='',solverTimer=0,solverWallMs=0,controlsKey='',renderedPopulationKey='';
let requestedPhysicalKey='',requestedSteadyOnly=false,stoppedKey='';
let requestedHorizon:number|null|undefined;
let notice:{text:string;word:StatusWord;until:number}|undefined;
let failureRecords:SolverFailureRecord[]=[];
try{const saved=JSON.parse(localStorage.getItem(`${STORAGE}-solver-failures`)??'[]');if(Array.isArray(saved))failureRecords=saved.slice(-5);}catch{}
function rememberSolverFailure(record:SolverFailureRecord):void{failureRecords=[...failureRecords.slice(-4),record];try{localStorage.setItem(`${STORAGE}-solver-failures`,JSON.stringify(failureRecords));}catch{}}
const solver=new DynamicsClient({debounceMs:0,onFailure:rememberSolverFailure}),analysisCache=new Map<string,Analysis>();
const treeOwners=new WeakMap<HTMLSelectElement,TreeCombo>();
let initialTree:TreeCombo|undefined;
let initialBody='';
const directionCommands:{command:Command;mode:DirectionShortcut}[]=[];
const projectedPopulationCache=new WeakMap<Analysis,WeakMap<object,Record<string,number>|null>>();
let spectrumBuildCount=0,populationPaintCount=0;
/** A solve's result and the mean horizon it was asked for. */
type SolvedRecord={result:DynamicsResult;wallMs:number;horizon:number|null};const displayedResults=new CurrentDynamicsResults<SolvedRecord>();
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const message=(error:unknown)=>error instanceof Error?error.message:String(error);

try{const saved=localStorage.getItem(STORAGE);if(saved){unreadableScene=saved;scene=validateScene(JSON.parse(saved));unreadableScene=null;}}catch(error){loadFailure=message(error);}
if(!scene.lasers.some(l=>l.id===selection.id))selection=scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:{kind:'field',id:scene.fields[0]?.id??''};
scene.marks??={};scene.references??=[];scene.dynamics??=defaultDynamics();
try{stateLabels=readLabels(JSON.parse(localStorage.getItem(`${STORAGE}-state-labels`)??'{}'));}catch{/* Ignore malformed presentation state. */}

// ---------------------------------------------------------------- the window
root.innerHTML=`<div class="workbench">
<div class="workspace">
<aside class="card scene-panel" aria-label="实验场景">
 <div class="title-row panel-heading"><span class="heading">实验场景</span><span class="spacer"></span><span id="toggle-scene-slot"></span></div>
 <div class="title-row object-heading"><span class="heading">磁场</span><span class="object-count" id="field-count"></span><span class="spacer"></span><span id="add-field-slot"></span></div>
 <div id="field-list" class="scene-list"></div>
 <div class="title-row object-heading"><span class="heading">激光</span><span class="object-count" id="laser-count"></span><span class="spacer"></span><span id="add-laser-slot"></span></div>
 <div id="laser-list" class="scene-list"></div>
 <div class="title-row object-heading"><span class="heading">参考</span><span class="object-count" id="reference-count"></span><span class="spacer"></span><span id="add-reference-slot"></span></div>
 <div id="reference-list" class="scene-list"></div>
 <span id="initial-shortcut-slot"></span>
 <div class="state-selector"><div class="title-row object-heading"><span class="heading">初态</span></div><div id="initial-state-controls" class="initial-controls"></div></div>
</aside>
<aside class="card inspector-panel" aria-label="参数">
 <div class="title-row inspector-heading"><span class="heading" id="inspector-title"></span><span id="inspector-name-slot"></span><span class="spacer"></span><span id="inspector-commands"></span></div>
 <div id="inspector-controls"></div>
</aside>
</div></div><input type="file" id="file-input" accept="application/json,.json" hidden>`;

const workbench=root.querySelector<HTMLElement>('.workbench')!,workspace=root.querySelector<HTMLElement>('.workspace')!;

// The header card: the scene's identity (its name edited where it stands), its muted summary and
// the file commands (v3 TaskConsole header).
const header=element('header','card header');
const sceneDot=statusDot('idle'),sceneName=new InlineName(name=>renameScene(name),'场景名称','heading document-name');sceneName.root.id='scene-name';
const identity=element('div','identity');identity.append(sceneDot,sceneName.root);
const summaryLine=element('span','summary');summaryLine.id='summary';
const examplesCommand=command('示例…',()=>showExamples());examplesCommand.root.id='examples-button';
const importCommand=command('导入…',()=>$<HTMLInputElement>('file-input').click());importCommand.root.id='import-button';
const exportCommand=command('导出…',()=>exportScene());exportCommand.root.id='export-button';
const inspectorToggle=command('参数',()=>toggleInspector(),{role:'quiet'});inspectorToggle.root.classList.add('inspector-toggle');inspectorToggle.root.setAttribute('aria-controls','inspector-controls');
header.append(identity,summaryLine,inspectorToggle.root,examplesCommand.root,importCommand.root,exportCommand.root);

// The status strip: save and solver state, Stop while a solve runs (v3 FluentStatusStrip).
const strip=new StatusStrip({action:['停止',()=>stopSolve()]});strip.root.id='status';strip.text.id='status-text';
const detailsCommand=command('详情',()=>showView('details'),{role:'quiet'});detailsCommand.hidden=true;
const recoverCommand=command('下载原本地场景',()=>downloadUnreadableScene());recoverCommand.hidden=true;
strip.trailing.prepend(detailsCommand.root,recoverCommand.root);
workbench.prepend(header,strip.root);

// The three analysis pages and 详情 (v3 FluentTabWidget); each page's own controls on its title
// row. The analysis pages hold data only; what it means is on 详情.
const tabs=new TabWidget();tabs.root.classList.add('main-tabs');
const page=(key:View,markup:string)=>{const node=element('div',`atom-page ${key}-page`);node.innerHTML=markup;return node;};
const levelsPage=page('levels',`<div class="title-row page-title"><span class="shift-tag" id="shift-tag"></span><span class="shift-warning" id="shift-warning"></span><span class="spacer"></span><span id="levels-commands"></span></div>
<div class="population-toolbar" id="population-toolbar"></div><div class="chart-surface" id="levels-chart"></div>
<div id="selection-readout" class="state-selection-panel"></div><section class="details-card" id="levels-details"></section>`);
const dynamicsPage=page('dynamics',`<div class="title-row page-title"><span class="summary" id="dynamics-beams"></span><span id="dynamics-commands"></span></div>
<div id="dynamics-controls" class="dynamics-controls"></div><div class="chart-surface" id="population-chart"></div><div class="chart-caption" id="population-caption"></div>`);
const spacePage=page('space',`<div class="title-row page-title space-title"><span id="space-commands"></span></div><div class="chart-surface space-surface" id="space-chart"></div><div class="space-legend" id="space-legend"></div>`);
const detailsPage=page('details',`<div class="details-notes model-notes"><div id="details-current"></div><div id="details-static"></div></div>`);
tabs.addPermanent('levels','能级与跃迁',levelsPage);tabs.addPermanent('dynamics','布居演化',dynamicsPage);tabs.addPermanent('space','方向与偏振',spacePage);tabs.addPermanent('details','详情',detailsPage);
// TabWidget sizes a label by a canvas measure of its words; for CJK the canvas falls back to a
// narrower face than the page does, so the labels keep their own width and nothing is cut.
(['levels','dynamics','space','details'] as const).forEach((key,i)=>{const tab=tabs.strip.children[i] as HTMLElement|undefined;if(!tab)return;tab.dataset.view=key;tab.querySelector<HTMLElement>('.tab-label')?.style.removeProperty('width');});
tabs.setCurrent(view);
tabs.onChange=key=>{if(view!==key)showView(key as View);};
function showView(next:View):void{view=next;if(next==='levels'||next==='dynamics')resultView=next;update();}
workspace.querySelector('.scene-panel')!.after(tabs.root);

// Per-page commands.
const manifoldFocus=new Segmented(value=>{spectral=value as SpectralLine;manifoldFocus.setCurrent(spectral);renderResults(false);});
manifoldFocus.setChoices([['D2','D2 · 5P₃/₂'],['D1','D1 · 5P₁/₂']]);manifoldFocus.setCurrent(spectral);manifoldFocus.root.id='line-tabs';manifoldFocus.root.setAttribute('aria-label','能级关注流形');
$('levels-commands').replaceWith(manifoldFocus.root);
const populationScaleControl=new Segmented(value=>{populationScale=value as 'linear'|'log';populationScaleControl.setCurrent(populationScale);update();});
populationScaleControl.setChoices([['linear','线性'],['log','对数']]);populationScaleControl.setCurrent(populationScale);populationScaleControl.root.setAttribute('aria-label','布居坐标');
$('dynamics-commands').replaceWith(populationScaleControl.root);
// 方向与偏振: where the camera looks, the field animation, the reference plane and the zoom.
let spaceView:SpaceViewName='free';
const spaceViews=new Segmented(value=>{spaceView=value as SpaceViewName;spaceViews.setCurrent(spaceView);spaceRenderer?.setView(spaceView);});
spaceViews.setChoices([['free','3D'],['x','沿 x'],['y','沿 y'],['z','沿 z'],['B','沿 B'],['k','沿 k']]);spaceViews.setCurrent(spaceView);spaceViews.root.setAttribute('aria-label','观察方向');
const spaceAnimation=switchButton(on=>{setChecked(spaceAnimation,on);spaceRenderer?.setAnimation(on);},'电场动画');spaceAnimation.classList.add('row-switch');
const spacePlane=checkBox('xy 参考面',on=>spaceRenderer?.setPlane(on));setChecked(spacePlane,true);
const zoomOut=command('−',()=>spaceRenderer?.zoomBy(1.14),{role:'quiet',glyph:true,title:'缩小'}),zoomIn=command('＋',()=>spaceRenderer?.zoomBy(.88),{role:'quiet',glyph:true,title:'放大'}),resetView=command('⟲',()=>spaceRenderer?.reset(),{role:'quiet',glyph:true,title:'复位视角'});
zoomOut.root.setAttribute('aria-label','缩小');zoomIn.root.setAttribute('aria-label','放大');resetView.root.setAttribute('aria-label','复位视角');
$('space-commands').replaceWith(spaceViews.root,spaceAnimation,spacePlane,element('span','spacer'),zoomOut.root,zoomIn.root,resetView.root);

// Scene panel commands.
const toggleScene=command('',()=>toggleScenePanel(),{role:'quiet',square:true});toggleScene.root.id='toggle-scene';toggleScene.root.classList.add('scene-collapse');
toggleScene.root.append(element('span','scene-chevron'));$('toggle-scene-slot').replaceWith(toggleScene.root);
const addField=command('＋',()=>addObject('field'),{role:'quiet',glyph:true,title:'添加磁场'});addField.root.id='add-field';addField.root.setAttribute('aria-label','添加磁场');$('add-field-slot').replaceWith(addField.root);
const addLaser=command('＋',()=>addObject('laser'),{role:'quiet',glyph:true,title:'添加激光'});addLaser.root.id='add-laser';addLaser.root.setAttribute('aria-label','添加激光');$('add-laser-slot').replaceWith(addLaser.root);
const addReference=command('＋',()=>new Menu([{label:'参考频率',action:()=>addObject('reference')},{label:'参考能级',action:()=>addObject('level')}]).openBelow(addReference.root),{role:'quiet',glyph:true,title:'添加参考频率或参考能级'});addReference.root.id='add-reference';addReference.root.setAttribute('aria-label','添加参考');$('add-reference-slot').replaceWith(addReference.root);
const initialShortcut=command('ψ₀',()=>toggleScenePanel(true),{role:'quiet',square:true});initialShortcut.root.classList.add('scene-initial-shortcut');initialShortcut.root.setAttribute('aria-label','选择初态');$('initial-shortcut-slot').replaceWith(initialShortcut.root);
// The inspector's heading: the selected object's name, edited where it stands, and Close.
const inspectorName=new InlineName(name=>renameSelected(name),'对象名称','heading');$('inspector-name-slot').replaceWith(inspectorName.root);
const closeInspector=command('×',()=>{root.classList.remove('inspector-open');syncLayout();inspectorToggle.root.focus();},{role:'quiet',glyph:true,title:'关闭参数'});closeInspector.root.classList.add('inspector-dismiss');closeInspector.root.setAttribute('aria-label','关闭参数');
$('inspector-commands').replaceWith(closeInspector.root);


// ---------------------------------------------------------------- helpers
function setText(id:string,value:string):void{const node=$(id);if(node.textContent!==value)node.textContent=value;}
function populationPercent(p:number|undefined):string{return p===undefined?'—':populationValue(p);}
function initialId():string{return scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`;}
/** The initial state's card or curve, when it is one state. */
function singleInitialId():string{const spec=initialSpec(scene);return spec.kind==='state'?spec.id:'';}
function allLevels(a=latestAnalysis):Level[]{return [...a.ground,...a.D1,...a.D2];}
function getLevel(id:string,a=latestAnalysis):Level|undefined{return allLevels(a).find(l=>l.id===id);}
function stateText(l:Level):string{return stateLabel(l);}
function measuredNumber(value:number,places=4):string{if(value!==0&&Number.isFinite(value)&&Math.abs(value)<10**(-places))return formatNumber(value,{digits:3});return number(value,places);}
function frequencyText(valueMHz:number):string{if(!Number.isFinite(valueMHz))return '—';const unit=naturalDetuningUnit(valueMHz);return `${readoutText(inUnit(valueMHz,unit),false)} ${unit}`;}
function stateCell(id:string):string{return `<span class="state-label">${esc(idText(id))}</span>`;}
function idText(id:string):string{const l=getLevel(id);return l?stateText(l):scene.references?.find(r=>r.id===id)?.name??id;}
function activeLaser(_line?:SpectralLine):Laser|undefined{
  if(selection.kind==='reference'){const r=scene.referenceFrequencies?.find(r=>r.id===selection.id);return r?referenceBeam(r):undefined;}
  return selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):undefined;
}
/** A reference frequency as the levels chart reads a beam: its frequency, no intensity, no geometry. */
function referenceBeam(r:ReferenceFrequency):Laser{
  return {id:r.id,name:r.name,enabled:true,line:r.frequencyAnchor?.line??'D2',frequency:r.frequency,frequencyAnchor:r.frequencyAnchor,intensityMWcm2:0,direction:{reference:{kind:'axis',axis:'+z'},thetaDeg:0,phiDeg:0},polarizationAngleDeg:0,ellipticityDeg:0};
}
/** The selected beam itself, not a reference frequency read as one. */
function selectedBeam(l:Laser|undefined):boolean{return !!l&&scene.lasers.includes(l);}
function referenceStateId(laser=activeLaser()):string{return laserReferenceStateId(scene,laser);}
function dynamics():DynamicsSettings{return scene.dynamics??(scene.dynamics=defaultDynamics());}
function normalizeScene(next:Scene):void{try{migrateSceneFrequencies(next);}catch{/* The status reports an invalid frequency. */}for(const laser of next.lasers){try{normalizeLaserIntensity(laser);}catch{/* The inspector reports an invalid laser. */}}}

// Edits since the scene was loaded: derived anchors and automatic curve choices are not edits.
let baseline:string|null=null;
try{baseline=localStorage.getItem(`${STORAGE}-baseline`);}catch{}
function editKey(value:Scene=scene):string{
  // Readout preferences, derived anchors and the automatic curve choice are not edits; a unit
  // picked for the duration or the temperature is one through the quantity it changes.
  const d=value.dynamics,settings=d?{initialStateId:d.initialStateId,initialState:d.initialState,durationUs:d.durationUs,samples:d.samples,
    curves:d.monitorMode==='manual'?[d.monitoredStateIds,d.monitoredGroupIds??[]]:'auto'}:null;
  return JSON.stringify({...value,display:undefined,dynamics:settings},(key,item)=>key.endsWith('Anchor')?undefined:item);
}
function markBaseline():void{baseline=editKey();try{localStorage.setItem(`${STORAGE}-baseline`,baseline);}catch{}}
function sceneEdited():boolean{return baseline===null||baseline!==editKey();}

function saveSceneNow():void{clearTimeout(saveTimer);try{if(unreadableScene!==null){localStorage.setItem(`${STORAGE}-unreadable`,unreadableScene);unreadableScene=null;}localStorage.setItem(STORAGE,JSON.stringify(scene));localStorage.setItem(`${STORAGE}-state-labels`,JSON.stringify(stateLabels));saveState='saved';saveFailure='';}catch(error){saveState='failed';saveFailure=message(error);}refreshStatus();}
function persist():void{saveState='pending';clearTimeout(saveTimer);saveTimer=window.setTimeout(saveSceneNow,200);}
function update():void{try{resolveSceneFrequencies(scene);}catch{/* analyze() reports it */}persist();invalidateObsoletePopulation();if(dirty)return;dirty=true;requestAnimationFrame(()=>{dirty=false;renderResults();});}
function note(text:string,word:StatusWord='task'):void{notice={text,word,until:performance.now()+6000};refreshStatus();window.setTimeout(refreshStatus,6100);}

// ---------------------------------------------------------------- status
/** The solver's line: words, their status, and the solve's wall time, written last so a time of
 * another length moves nothing before it. Computing is automatic; 停止 ends a running solve. */
function solverStatus():[string,StatusWord,string?]{
  const result=currentResult(),error=currentSolverError(),checkpoint=displayedCheckpoint();
  if(error)return [result?.steady.converged?`时间演化失败（长期布居已就绪）：${labelStateIds(solverText(error))}`:`布居计算失败：${labelStateIds(solverText(error))}`,'error'];
  // A streamed checkpoint stands on the page while the solve goes on.
  if(currentSolverBusy())return [checkpoint&&result?.populationAccuracy?`计算中：当前估计误差最大 ${errorPercent(result.populationAccuracy.estimatedMaximumError)}`:result?.steady.converged?'长期布居已就绪 · 时间演化计算中…':solverQueued?'布居待计算':'布居计算中…','task'];
  const stopped=stoppedKey!==''&&stoppedKey===solveKey();
  if(view==='space')return ['方向与偏振实时显示','idle'];
  if(!result)return stopped?['计算已停止 · 修改参数后重新计算','warning']:['布居待计算','idle'];
  const elapsed=formatElapsed(currentRecord()?.wallMs??solverWallMs),partial=trajectorySentence(result),mean=meanSentence(result);
  const trajectory=resultPage()==='dynamics'||populationMode==='time';
  // 停止 keeps the last checkpoint, marked as any partial result is.
  if(stopped&&checkpoint&&trajectory)return [`计算已停止 · 显示最后的中间结果${partial?`：${partial}`:'（各时刻 ≤ 0.5%）'}`,'warning'];
  if(resultPage()==='dynamics'&&result.timesUs.length>1)return [['轨迹已计算',partial,result.steady?.converged?mean:'长期布居未解析'].filter(Boolean).join(' · '),result.steady?.converged&&!partial&&!mean?'idle':'warning',elapsed];
  if(resultPage()==='dynamics')return stopped?[`计算已停止 · ${steadyStatusText(result.steady?.status)}已就绪，时间演化未计算`,'warning']:[result.steady?.converged?`${steadyStatusText(result.steady.status)}${mean?` · ${mean}`:'已收敛'} · 时间演化待计算`:'长期布居未解析',mean?'warning':'idle'];
  if(!result.steady?.converged)return ['长期布居未解析','warning'];
  // The levels chart: p̄(T), or p(t) of the trajectory at the chosen time.
  const words=[`${steadyStatusText(result.steady.status)}${mean?` · ${mean}`:'已收敛'}`,...(populationMode==='time'&&partial?[partial]:[])];
  return [words.join(' · '),mean||(populationMode==='time'&&partial)?'warning':'idle',elapsed];
}
/** Where the trajectory misses the population tolerance, in one sentence (empty: nowhere, or no trajectory). */
function trajectorySentence(result:DynamicsResult):string{const errors=sampleErrors(result);return result.timesUs.length>1&&errors?partialSentence(result.timesUs,errors,durationUnit(dynamics())):'';}
/** A computed p̄(T) outside the tolerance: its estimated error (empty when within). */
function meanSentence(result:DynamicsResult):string{const s=result.steady;return s.converged&&!s.withinTolerance?`${meanSymbol(result)} 估计误差 ${s.estimatedError===null?'未知':errorPercent(s.estimatedError)}（超过 ${errorPercent(POPULATION_TOLERANCE)}）`:'';}
/** The trajectory shown is a streamed checkpoint, not a final result. */
function displayedCheckpoint():boolean{const key=trajectoryKey();return !displayedResults.get(key,'trajectory')&&(displayedResults.get(key,'progress')?.result.timesUs.length??0)>1;}
function refreshStatus():void{
  if(notice&&performance.now()>notice.until)notice=undefined;
  let [text,word,elapsed]=solverStatus();
  if(analysisError)[text,word,elapsed]=[`能级参数无效：${analysisError}`,'error',undefined];
  if(notice)[text,word,elapsed]=[notice.text,notice.word,undefined];
  if(saveState==='failed')[text,word,elapsed]=[`本地保存失败：${saveFailure}；请导出场景`,'error',undefined];
  else if(loadFailure&&!notice)[text,word,elapsed]=[`原本地场景无法载入：${loadFailure}。已显示默认场景，原数据已保留`,'warning',undefined];
  if(word==='idle')text+=saveState==='pending'?' · 正在保存…':' · 场景已保存';
  if(elapsed)text+=` · ${elapsed}`;
  strip.show(text,word);strip.setActionVisible(currentSolverBusy());
  detailsCommand.hidden=!currentSolverError();recoverCommand.hidden=!loadFailure;
  const sceneWord:StatusWord=saveState==='failed'?'error':loadFailure?'warning':'idle';setState(sceneDot,sceneWord);
  sceneDot.title=saveState==='failed'?'本地保存失败':loadFailure?'原本地场景未载入':'场景已保存在本机';
}

// ---------------------------------------------------------------- selection
function select(next:Selection):void{
  if(selection.kind===next.kind&&selection.id===next.id)return;
  const focusedObject=document.activeElement?.closest<HTMLElement>('[data-object-id]'),focusId=focusedObject?.dataset.objectId;
  selection=next;
  if(layoutMode()==='narrow'&&selection.kind!=='none')root.classList.add('inspector-open');
  if(selection.kind==='none')root.classList.remove('inspector-open');
  root.classList.remove('scene-open');syncLayout();renderObjects();renderInspector();renderResults(false);
  if(focusId)root.querySelector<HTMLButtonElement>(`[data-object-id="${CSS.escape(focusId)}"]`)?.focus({preventScroll:true});
}
/** A click on an object -- in the scene list, on its arrow or name in a chart -- selects it; a
 * click on empty space clears the selection (the levels chart then shows the total shift). */
function pick(next:Selection):void{select(next);}
function clearSelection():void{select({kind:'none',id:''});}
function applyLabels(next:StateLabels):void{stateLabels=next;persist();renderResults(false);}
function selectState(id:string):void{applyLabels(clickState(stateLabels,id));}

// ---------------------------------------------------------------- scene list
function item(kind:ObjectKind,id:string,index:number,name:string,sub:string,enabled:boolean,onToggle?:(v:boolean)=>void):HTMLElement{
  const div=element('div',`scene-object${selection.kind===kind&&selection.id===id?' selected':''}${enabled?'':' disabled'}`);
  const button=element('button','object-select');button.type='button';button.dataset.objectId=id;button.dataset.objectKind=kind;
  button.setAttribute('aria-label',`编辑${name}`);button.setAttribute('aria-pressed',String(selection.kind===kind&&selection.id===id));
  const symbol=element('span',`object-symbol ${kind}`);symbol.append(kind==='field'?'B':kind==='laser'?'λ':kind==='reference'?'ν':'E',element('span','object-index',subscript(index)));if(kind==='laser'||kind==='reference')symbol.style.color=laserColour(scene,id);
  const copy=element('span','object-copy'),title=element('span','object-name'),subtitle=element('span','object-detail');
  setMiddle(title,name,'cut');setMiddle(subtitle,sub,'cut');copy.append(title,subtitle);button.append(symbol,copy);
  // In the collapsed rail only the symbol shows: the button says what it stands for.
  button.addEventListener('pointerenter',()=>{button.title=copy.offsetParent===null?`${name} · ${sub}`:'';});
  button.onclick=()=>pick({kind,id});
  const toggle=onToggle?switchButton(next=>{onToggle(next);setChecked(toggle!,next);div.classList.toggle('disabled',!next);update();}):undefined;
  if(toggle){setChecked(toggle,enabled);toggle.classList.add('object-toggle');toggle.setAttribute('aria-label',`启用${name}`);}
  const remove=command('×',()=>void removeObject(kind,id),{role:'destructive',glyph:true,square:true,title:`删除${name}`}).root;
  remove.classList.add('remove-object');remove.setAttribute('aria-label',`删除${name}`);
  div.append(button,...(toggle?[toggle]:[element('span','object-toggle-gap')]),remove);return div;
}
async function removeObject(kind:ObjectKind,id:string):Promise<void>{
  if(kind==='reference'||kind==='level'){await removeReference(kind,id);return;}
  const object=(kind==='field'?scene.fields:scene.lasers).find(o=>o.id===id);if(!object)return;
  const noun=kind==='field'?'磁场':'激光',following=kind==='laser'?[...scene.lasers,...(scene.referenceFrequencies??[])].filter(o=>o.frequency&&'base' in o.frequency&&o.frequency.base.kind==='laser'&&o.frequency.base.id===id).map(o=>o.name):[];
  const linked=kind==='field'?linkedTo<{name:string;direction:Direction}>([...scene.fields,...scene.lasers],id).map(o=>o.name):[];
  const released=following.length?`跟随它的${following.join('、')}将改为等效的零场或绝对定义，频率不变。`:linked.length?`链接它参考轴的${linked.join('、')}将改为等效的角度参考轴，方向不变。`:'';
  if(!await confirm(`删除${noun}`,`删除${noun}「${object.name}」？${released}它的全部参数将一并删除，无法撤销。`,{confirmText:'删除',cancelText:'取消',kind:'warning'}))return;
  if(kind==='laser')try{releaseFollowers(scene,object as Laser);unlinkPhotons('laser',id);}catch(error){note(message(error),'error');return;}
  if(kind==='field')try{releaseLinks(scene.fields,[...scene.fields,...scene.lasers],id);}catch(error){note(message(error),'error');return;}
  const original=kind==='field'?scene.fields:scene.lasers,index=original.findIndex(o=>o.id===id),selected=selection.kind===kind&&selection.id===id;
  if(kind==='field')scene.fields=scene.fields.filter(o=>o.id!==id);else scene.lasers=scene.lasers.filter(o=>o.id!==id);
  if(selected){const remaining=kind==='field'?scene.fields:scene.lasers,next=remaining[Math.min(index,remaining.length-1)];selection=next?{kind,id:next.id}:scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:scene.fields[0]?{kind:'field',id:scene.fields[0].id}:{kind:'none',id:''};}
  controlsKey='';renderObjects();renderInspector();update();
  if(selected)requestAnimationFrame(()=>root.querySelector<HTMLButtonElement>(`[data-object-id="${CSS.escape(selection.id)}"]`)?.focus({preventScroll:true}));
}
/** How a frequency is defined, in a list line: 852 nm · 384.23 THz · Δ −12 MHz · 跟随 探测光 +6.568 GHz. */
function frequencyDetail(object:DefinedFrequency):string{
  const f=object.frequency;if(!f)return '';
  const {value,unit}=frequencyValue(object);if('wavelengthNm' in f||f.base.kind==='absolute')return withUnit(value,unit,{digits:6});
  const offset=withUnit(value,unit,{digits:4,signed:true});if(f.base.kind==='transition')return `Δ ${offset}`;
  return `跟随 ${objectOf(scene,`${f.base.kind}:${f.base.id}`)?.name??f.base.id} ${offset}`;
}
function laserDetail(l:Laser):string{return frequencyDetail(l);}
/** A reference level in a list line: 5S₁/₂ · F=2 · m=0 + 探测光 −12 MHz. */
function levelDetail(r:ReferenceLevel):string{
  const anchor=getLevel(r.anchorId),source=r.photon?objectOf(scene,`${r.photon.source.kind}:${r.photon.source.id}`)?.name:undefined;
  return `${anchor?stateText(anchor):r.anchorId}${source?` ${r.photon!.direction==='emit'?'−':'+'} ${source}`:''}${r.offsetMHz?` ${frequencyText(r.offsetMHz).replace(/^(?=[\d≈])/,'+')}`:''}`;
}
/** A photon link that goes keeps the level where it was: its frequency joins the offset. */
function unlinkPhotons(kind:'laser'|'reference',id:string):void{
  for(const r of scene.references??[])if(r.photon?.source.kind===kind&&r.photon.source.id===id){const nu=photonFrequencyMHz(r,scene)??0;r.offsetMHz+=r.photon.direction==='emit'?-nu:nu;delete r.photon;}
}
async function removeReference(kind:'reference'|'level',id:string):Promise<void>{
  const object=kind==='reference'?scene.referenceFrequencies?.find(r=>r.id===id):scene.references?.find(r=>r.id===id);if(!object)return;
  const noun=kind==='reference'?'参考频率':'参考能级',following=kind==='reference'?[...scene.lasers,...(scene.referenceFrequencies??[])].filter(o=>o.frequency&&'base' in o.frequency&&o.frequency.base.kind==='reference'&&o.frequency.base.id===id).map(o=>o.name):[];
  const linked=kind==='reference'?(scene.references??[]).filter(r=>r.photon?.source.kind==='reference'&&r.photon.source.id===id).map(r=>r.name):[];
  const consequences=[following.length?`跟随它的${following.join('、')}将改为等效的零场或绝对定义，频率不变`:'',linked.length?`链接它的${linked.join('、')}将改为固定偏移，能量不变`:''].filter(Boolean).join('；');
  if(!await confirm(`删除${noun}`,`删除${noun}「${object.name}」？${consequences?consequences+'。':''}无法撤销。`,{confirmText:'删除',cancelText:'取消',kind:'warning'}))return;
  if(kind==='reference'){
    try{releaseFollowers(scene,object as ReferenceFrequency);unlinkPhotons('reference',id);}catch(error){note(message(error),'error');return;}
    scene.referenceFrequencies=scene.referenceFrequencies!.filter(r=>r.id!==id);
  }else{
    for(const beam of scene.lasers)if(beam.referenceStateId===id)setLaserReferenceState(scene,beam,undefined);
    scene.references=scene.references!.filter(r=>r.id!==id);stateLabels=keepLabels(stateLabels,state=>state!==id);
  }
  if(selection.kind===kind&&selection.id===id)selection={kind:'none',id:''};
  controlsKey='';renderObjects();renderInspector();update();
}
function renderObjects():void{
  if(!sceneName.editing)sceneName.value=scene.name;
  const title=`${scene.name} — ${APP_TITLE}`;if(document.title!==title)document.title=title;
  $('field-list').replaceChildren(...scene.fields.map((f,i)=>item('field',f.id,i+1,f.name,`${withUnit(f.magnitudeG,'G',{digits:4})} · ${directionDetail(f.direction)}`,f.enabled,v=>f.enabled=v)));
  $('laser-list').replaceChildren(...scene.lasers.map((l,i)=>item('laser',l.id,i+1,l.name,laserDetail(l),l.enabled,v=>l.enabled=v)));
  const references=[...(scene.referenceFrequencies??[]).map((r,i)=>item('reference',r.id,i+1,r.name,frequencyDetail(r),true)),...(scene.references??[]).map((r,i)=>item('level',r.id,i+1,r.name,levelDetail(r),true))];
  $('reference-list').replaceChildren(...references);
  setText('field-count',String(scene.fields.length));setText('laser-count',String(scene.lasers.length));setText('reference-count',String(references.length));
  addField.disabled=scene.fields.length>=8;addLaser.disabled=scene.lasers.length>=8;
}
/** A direction in a list line: its reference axis, and its offset when there is one: +z · 链接 偏置磁场 · θ′ 10°. */
function directionDetail(d:Direction):string{
  const r=d.reference,reference=r.kind==='axis'?axisWords(r.axis):r.kind==='field'?`链接 ${scene.fields.find(f=>f.id===r.id)?.name??r.id}`:`θ ${formatNumber(r.thetaDeg,{digits:4})}° φ ${formatNumber(r.phiDeg,{digits:4})}°`;
  return d.thetaDeg?`${reference} · θ′ ${formatNumber(d.thetaDeg,{digits:4})}°`:reference;
}
function nextObjectName(prefix:string,objects:{name:string}[]):string{const used=new Set(objects.map(o=>o.name));let n=1;while(used.has(`${prefix} ${n}`))n++;return `${prefix} ${n}`;}
function addObject(kind:ObjectKind):void{
  if(kind==='reference'){scene.referenceFrequencies??=[];if(scene.referenceFrequencies.length>=8){note('最多 8 个参考频率','warning');return;}const id=`frequency-${Date.now()}`;scene.referenceFrequencies.push({id,name:nextObjectName('参考频率',scene.referenceFrequencies),frequency:{base:{kind:'transition',line:'D2',Fg:2,Fe:3},offsetMHz:0},frequencyUnit:'MHz'});resolveSceneFrequencies(scene);select({kind:'reference',id});update();return;}
  if(kind==='level'){
    // From the selected beam's origin, one of its photons up (or down): the old 取激光终点, now a link.
    scene.references??=[];if(scene.references.length>=12){note('最多 12 个参考能级','warning');return;}const id=`reference-${Date.now()}`,beam=selectedBeam(activeLaser())?activeLaser():undefined,origin=beam?referenceStateId(beam):initialId();
    scene.references.push({id,name:nextObjectName('参考能级',scene.references),anchorId:getLevel(origin)?origin:initialId(),anchorMode:'follow',offsetMHz:0,...(beam?{photon:{source:{kind:'laser' as const,id:beam.id},direction:laserEmits(scene,beam,allLevels())?'emit' as const:'absorb' as const}}:{})});
    select({kind:'level',id});update();return;
  }
  if(kind==='field'){if(scene.fields.length>=8)return;const id=`field-${Date.now()}`;scene.fields.push({id,name:nextObjectName('磁场',scene.fields),enabled:true,magnitudeG:1,direction:{reference:{kind:'axis',axis:'+x'},thetaDeg:0,phiDeg:0}});select({kind:'field',id});update();}
  else{if(scene.lasers.length>=8)return;const id=`laser-${Date.now()}`;scene.lasers.push({id,name:nextObjectName('激光',scene.lasers),enabled:true,showLightShift:false,line:spectral,frequency:{base:{kind:'transition',line:spectral,Fg:2,Fe:spectral==='D1'?2:3},offsetMHz:0},frequencyUnit:'MHz',intensityMWcm2:.1,direction:{reference:{kind:'axis',axis:'+x'},thetaDeg:0,phiDeg:0},polarizationAngleDeg:0,ellipticityDeg:0});controlsKey='';select({kind:'laser',id});update();}

  update();
}

// ---------------------------------------------------------------- inspector
type BoundField={field:Field;refresh():void};
const inspectorFields=new Map<string,BoundField>(),dynamicsFields=new Map<string,BoundField>(),fieldProblems=new Map<string,string>();
let fieldRequest=0;
type BindOptions={unit?:(unit:string)=>void;after?:()=>void;
  /** Whose row it is (an object's key): its problem and kept draft are its own. */
  scope?:string;
  /** Escape drops the row's kept draft with its problem. */
  cancel?:()=>void;
  /** The row shows a kept draft its constraints refuse: marked invalid until the next edit. */
  invalid?:()=>boolean;
  /** A computed value where an input stands in another mode: the row unavailable, its unit picker live. */
  computed?:true};
/** A v3 Field on the scene: `apply` writes the operator's value (or throws its
 * problem, shown under the row); the row is then re-projected from the scene. */
function bindField(registry:Map<string,BoundField>,parent:HTMLElement,prefix:string,spec:()=>FieldSpec,apply:(value:Value)=>void,options:BindOptions={}):Field{
  const key=`${prefix}:${options.scope??''}:${spec().name}`,current=():FieldSpec=>({...spec(),problem:fieldProblems.get(key)});
  const after=options.after??(()=>{renderObjects();update();});
  const show=(response?:number,failed=false):void=>{
    field.update(current(),response,failed);
    if(options.computed)field.cell.querySelector<HTMLSelectElement>('select.unit-picker')?.removeAttribute('disabled');
    if(options.invalid?.()&&field.input instanceof HTMLInputElement)field.input.classList.add('invalid');
  };
  const field:Field=new Field(prefix,current(),{
    commit:value=>{
      const id=++fieldRequest;
      try{apply(value);fieldProblems.delete(key);}
      catch(error){fieldProblems.set(key,message(error));queueMicrotask(()=>show(id,true));return id;}
      queueMicrotask(()=>show(id));after();return id;
    },
    unit:options.unit?unit=>{
      const id=++fieldRequest;let failed=false;
      try{options.unit!(unit);fieldProblems.delete(key);}catch(error){failed=true;fieldProblems.set(key,message(error));}
      queueMicrotask(()=>show(id,failed));after();return id;
    }:undefined,
    cancel:()=>{options.cancel?.();fieldProblems.delete(key);show();},
  },false);
  parent.append(field.root);registry.set(key,{field,refresh:()=>show()});show();return field;
}
function refreshFields(registry:Map<string,BoundField>):void{for(const {refresh} of registry.values())refresh();}
const numberSpec=(name:string,label:string,value:number,unit:string,extra:Partial<FieldSpec>={}):FieldSpec=>({name,label,kind:'number',value,unit,...extra});
const unitChoices=(units:readonly string[],group:string):NonNullable<FieldSpec['units']>=>units.map(u=>({value:u,label:u,group}));
/** A number the operator writes in a unit they pick (ui-design.md §0.4): read in its unit, written
 * with one. Its digits are the operator's: another unit writes the same digits in it -- a real
 * change, solved like any edit. A unit whose quantity the constraints refuse is kept as a draft,
 * as Device Control keeps one: those digits in that unit, shown invalid with the reason while the
 * scene stays as it was, until the next edit; Escape drops it. */
type Quantity={read():Reading;write(value:number,unit:string):void;units():NonNullable<FieldSpec['units']>};
const unitDrafts=new Map<string,Reading>();
function bindQuantity(registry:Map<string,BoundField>,parent:HTMLElement,prefix:string,scope:string,spec:()=>Omit<FieldSpec,'kind'|'value'|'unit'|'units'>,quantity:Quantity,after?:()=>void):Field{
  const key=`${prefix}:${scope}:${spec().name}`,shown=():Reading=>unitDrafts.get(key)??quantity.read();
  const field=bindField(registry,parent,prefix,()=>({...spec(),kind:'number',...shown(),units:quantity.units()}),
    v=>{quantity.write(Number(v),shown().unit);unitDrafts.delete(key);},
    {scope,after,cancel:()=>unitDrafts.delete(key),invalid:()=>unitDrafts.has(key),unit:unit=>{
      // The digits in the box -- the operator's, a refused entry's included -- without float dust.
      const digits=typedDigits(Number(field.value));
      try{quantity.write(digits,unit);unitDrafts.delete(key);}catch(error){unitDrafts.set(key,{value:digits,unit});throw error;}
    }});
  return field;
}
/** A computed quantity in an input's place (强度 I from P and w₀, the output waist from NA): its
 * box disabled (the tooltip says what it follows from), its unit picker live. A display converts:
 * another unit reads the same quantity. */
function bindDerived(registry:Map<string,BoundField>,parent:HTMLElement,prefix:string,scope:string,spec:{name:string;label:string;description:string;from:string},reading:()=>Reading,units:NonNullable<FieldSpec['units']>,pick:(unit:string)=>void):Field{
  const {from,...words}=spec;
  return bindField(registry,parent,prefix,()=>{const r=reading();return {...words,kind:'number',spin:true,inexact:true,allow_none:true,placeholder:'—',value:Number.isFinite(r.value)?r.value:null,unit:r.unit,units,unavailable_reason:from};},
    ()=>{throw new Error(from);},{scope,computed:true,unit:pick,after:()=>persist()});
}
/** A reading that may not be computable now (a frequency not yet resolved): NaN in its unit. */
const attempt=(reading:()=>Reading,unit:()=>string):Reading=>{try{return reading();}catch{return {value:NaN,unit:unit()};}};

function group(title:string,parent=$('inspector-controls')):HTMLElement{const section=element('section','control-section');const g=form();g.append(sectionTitle(title));section.append(g);parent.append(section);return g;}
function selectControl(parent:HTMLElement,label:string,value:string,options:[string,string][],changed:(value:string)=>void):HTMLSelectElement{
  if(label==='参考态'){
    const tree=new TreeCombo(changed);tree.setTree(stateTree(allLevels(),scene.references),value,'跟随动力学初态');tree.root.setAttribute('aria-label',label);treeOwners.set(tree.root,tree);const r=fieldRow(label,tree.root);r.classList.add('choice-field');parent.append(r);return tree.root;
  }
  const el=element('select');el.setAttribute('aria-label',label);el.append(comboFace(el),...options.map(([v,t])=>new Option(t,v)));el.value=value;el.onchange=()=>changed(el.value);const r=fieldRow(label,el);r.classList.add('choice-field');parent.append(r);return el;
}
function disposeTrees(host:HTMLElement):void{host.querySelectorAll<HTMLSelectElement>('select.tree-combo').forEach(node=>treeOwners.get(node)?.dispose());}
/** Presets are commands, not modes. Rendering never selects or re-applies one. */
function polarizationButtons(laser:Laser,choices:readonly (readonly [PolarizationPreset,string])[],label:string):HTMLElement{
  const actions=element('div','preset-actions');actions.role='group';actions.setAttribute('aria-label',label);
  for(const [preset,text] of choices){
    const action=command(text,()=>{setPolarizationPreset(laser,scene.fields,preset);refreshFields(inspectorFields);renderObjects();renderInspector();update();});
    action.root.dataset.polarizationAction=preset;actions.append(action.root);
  }
  return actions;
}
/** 方向 (direction.ts): the reference axis -- a coordinate axis, a field's reference axis or
 * angles -- and the offset (θ′, φ′) from it, then the actual direction it gives. A light's 沿 B
 * shortcuts link it to a field's reference axis. */
function directionSection(g:HTMLElement,object:MagneticField|Laser,kind:'field'|'laser'):void{
  const fields=inspectorFields,after=()=>{renderObjects();renderInspector();update();};
  const set=(direction:Direction)=>{object.direction=direction;};
  const keyOf=(r:ReferenceAxis)=>r.kind==='axis'?`axis:${r.axis}`:r.kind==='field'?`field:${r.id}`:'angles';
  const targets=()=>linkableFields(scene.fields,kind==='field'?object.id:undefined);
  const tree=():TreeGroup[]=>[{heading:'坐标轴',leaves:AXIS_NAMES.map(a=>({key:`axis:${a}`,label:axisWords(a),text:axisWords(a)}))},
    ...(targets().length?[{heading:'磁场的参考轴',leaves:targets().map(f=>({key:`field:${f.id}`,label:f.name,text:`链接 ${f.name}`}))}]:[]),
    {heading:'角度',leaves:[{key:'angles',label:'θ, φ',text:'角度 θ, φ'}]}];
  bindField(fields,g,'inspector',()=>({name:'reference',label:'参考轴',kind:'choice',tree:tree(),value:keyOf(object.direction.reference),description:'链接只连接参考轴：磁场自身的偏移不移动链接它的对象'}),v=>{
    const key=String(v);let reference:ReferenceAxis;
    // To angles: the axis it is now, so nothing moves until an angle is edited.
    if(key==='angles')reference={kind:'angles',...referenceAxis(scene.fields,object.direction.reference)};
    else if(key.startsWith('axis:')&&AXIS_NAMES.includes(key.slice(5) as AxisName))reference={kind:'axis',axis:key.slice(5) as AxisName};
    else if(key.startsWith('field:')&&targets().some(f=>f.id===key.slice(6)))reference={kind:'field',id:key.slice(6)};
    else throw new Error('不能用作这个方向的参考轴');
    set({...object.direction,reference});},{after});
  if(object.direction.reference.kind==='angles'){
    const angles=()=>object.direction.reference as Extract<ReferenceAxis,{kind:'angles'}>;
    bindField(fields,g,'inspector',()=>numberSpec('reference-theta','参考 θ',angles().thetaDeg,'°',{minimum:0,maximum:180,step:1}),v=>set({...object.direction,reference:{kind:'angles',thetaDeg:Number(v),phiDeg:angles().phiDeg}}));
    bindField(fields,g,'inspector',()=>numberSpec('reference-phi','参考 φ',angles().phiDeg,'°',{minimum:-180,maximum:180,step:1}),v=>set({...object.direction,reference:{kind:'angles',thetaDeg:angles().thetaDeg,phiDeg:Number(v)}}));
  }
  bindField(fields,g,'inspector',()=>numberSpec('offset-theta','偏离 θ′',object.direction.thetaDeg,'°',{minimum:0,maximum:180,step:1,description:'相对参考轴的极角'}),v=>set({...object.direction,thetaDeg:Number(v)}));
  bindField(fields,g,'inspector',()=>numberSpec('offset-phi','偏离 φ′',object.direction.phiDeg,'°',{minimum:-180,maximum:180,step:1,description:'相对参考轴的方位角；0 为最小转动带到参考轴的 x 方向'}),v=>set({...object.direction,phiDeg:Number(v)}));
  if(kind==='laser'){
    const actions=element('div','preset-actions full-form-row');actions.role='group';actions.setAttribute('aria-label','相对磁场方向');
    for(const [mode,text] of [['parallel','沿 +B'],['anti','沿 −B'],['perpendicular','垂直 B']] as const){
      const action=command(text,()=>{set(shortcutDirection(shortcutField(scene.fields,object.direction),mode));after();});
      action.root.dataset.direction=mode;actions.append(action.root);directionCommands.push({command:action,mode});
    }
    g.append(actions);
  }
  g.append(fieldRow('实际方向',actualDirection));refreshDirection(object);
}
/** The direction a field or laser points in: θ and φ in fixed slots. */
const actualDirection=element('span','direction-reading');
function refreshDirection(object:MagneticField|Laser):void{
  let markup='—';
  try{const axis=directionAxis(scene.fields,object.direction),angle=(name:string,deg:number)=>`<span class="direction-angle"><span class="slot-label">${name}</span><span class="slot angle">${formatNumber(deg,{places:2})}°</span></span>`;markup=angle('θ',axis.thetaDeg)+angle('φ',axis.phiDeg);}catch{/* The reference row says why. */}
  if(actualDirection.innerHTML!==markup)actualDirection.innerHTML=markup;
}
function renderInspector():void{
  const host=$('inspector-controls');const panel=host.closest<HTMLElement>('.inspector-panel')!,scroll=panel.scrollTop;
  disposeTrees(host);host.replaceChildren();inspectorFields.clear();directionCommands.length=0;
  const object=selectedObject();
  setText('inspector-title',object?'':'未选中');$('inspector-title').hidden=!!object;inspectorName.root.hidden=!object;if(object&&!inspectorName.editing)inspectorName.value=object.name;
  if(selection.kind==='none'){host.append(element('p','note wrap inspector-empty','点选磁场、激光或参考以编辑。'));return;}
  const fields=inspectorFields;
  if(selection.kind==='reference'){
    const r=scene.referenceFrequencies?.find(r=>r.id===selection.id);if(!r)return;
    frequencySection(group('频率'),r,referenceReadout);
  }else if(selection.kind==='level'){
    const r=scene.references?.find(r=>r.id===selection.id);if(!r)return;levelSection(group('定义'),r);
  }else if(selection.kind==='field'){
    const field=scene.fields.find(f=>f.id===selection.id);if(!field)return;
    const g=group('大小与方向');
    bindField(fields,g,'inspector',()=>numberSpec('magnitude','大小',field.magnitudeG,'G',{minimum:0,maximum:20,spin:true}),v=>{field.magnitudeG=Number(v);});
    directionSection(g,field,'field');
  }else{
    const l=scene.lasers.find(l=>l.id===selection.id);if(!l)return;normalizeLaserIntensity(l);
    // ui-design.md §4: every laser drives the dynamics; this only chooses the levels chart's background.
    const chart=group('能级图');bindField(fields,chart,'inspector',()=>({name:'show-light-shift',label:'显示光移',kind:'boolean',value:l.showLightShift===true,description:'开：未选中对象时，能级图的背景包含这束光的静态光移（全阶，仅显示，不影响动力学）'}),v=>{l.showLightShift=v===true;});
    chart.append(mixingWarning);refreshMixing(l);
    const f=group('频率');
    frequencySection(f,l,laserReadout);
    const originPicker=selectControl(f,'参考态',l.referenceStateId??'',[['','跟随动力学初态'],...allLevels().map(state=>[state.id,stateText(state)] as [string,string]),...(scene.references??[]).map(ref=>[ref.id,ref.name+' · 参考能级'] as [string,string])],id=>{setLaserReferenceState(scene,l,id||undefined);persist();renderInspector();renderResults(false);});originPicker.id='laser-reference-state';originPicker.setAttribute('aria-label','参考出发态');originPicker.title='只改变这束光的箭头起点和矩元参考；不改变动力学初态或任何驱动。';
    if(scene.references?.some(r=>r.id===l.referenceStateId))selectControl(f,'光子方向',l.referenceProcess??'absorption',[['absorption','吸收 ↑'],['emission','受激辐射 ↓']],v=>{l.referenceProcess=v as 'absorption'|'emission';update();});
    // 强度 (laser-intensity.ts): the local intensity itself, or power with a waist or with the NA the
    // beam fills. The mode's inputs are written in units the operator picks; what follows from
    // them -- the intensity, the NA mode's waist -- stands disabled in the intensity's place.
    const strength=group('强度'),scope=`laser:${l.id}`,mode=l.intensityMode??'intensity';
    bindField(fields,strength,'inspector',()=>({name:'intensity-mode',label:'指定方式',kind:'choice',choices:[['intensity','局域强度'],['power','功率与束腰'],['na','功率与数值孔径']],value:l.intensityMode??'intensity'}),v=>{setLaserIntensityMode(l,v as IntensityMode);},{scope,after:()=>{renderInspector();update();}});
    if(mode!=='intensity')bindQuantity(fields,strength,'inspector',scope,()=>({name:'power',label:'功率 P',minimum:0,spin:true,inexact:true}),
      {read:()=>powerReading(l),write:(v,u)=>writeLaserPower(l,v,u as PowerUnit),units:()=>unitChoices(POWER_UNITS,'功率')});
    if(mode==='power')bindQuantity(fields,strength,'inspector',scope,()=>({name:'waist',label:'束腰 w₀',minimum:0,spin:true,inexact:true}),
      {read:()=>waistReading(l),write:(v,u)=>writeLaserWaist(l,v,u as WaistUnit),units:()=>unitChoices(WAIST_UNITS,'长度')});
    if(mode==='na'){
      bindField(fields,strength,'inspector',()=>numberSpec('numerical-aperture','数值孔径 NA',l.numericalAperture??DEFAULT_NUMERICAL_APERTURE,'',{spin:true,description:'光束充满的物镜数值孔径；原子在真空中，0 < NA ≤ 1'}),
        v=>setLaserNumericalAperture(l,Number(v)),{scope});
      bindDerived(fields,strength,'inspector',scope,{name:'output-waist',label:'输出束腰 w₀',description:'w₀ = λ/(π·NA)，λ = c/ν 为这束光的真空波长（傍轴高斯光束）',from:'由数值孔径和这束光的波长算出：w₀ = λ/(π·NA)'},
        ()=>attempt(()=>waistReading(l),()=>waistUnit(l)),unitChoices(WAIST_UNITS,'长度'),u=>{l.waistUnit=u as WaistUnit;});
    }
    if(mode==='intensity')bindQuantity(fields,strength,'inspector',scope,()=>({name:'intensity',label:'强度 I',minimum:0,spin:true,inexact:true}),
      {read:()=>intensityReading(l),write:(v,u)=>writeLaserIntensity(l,v,u as IntensityUnit),units:()=>unitChoices(INTENSITY_UNITS,'强度')});
    else bindDerived(fields,strength,'inspector',scope,{name:'intensity',label:'强度 I',description:'原子处的局域强度：高斯光束轴上的 I = 2P/(πw₀²)',from:'由功率和束腰算出：I = 2P/(πw₀²)'},
      ()=>attempt(()=>intensityReading(l),()=>intensityUnit(l)),unitChoices(INTENSITY_UNITS,'强度'),u=>{l.intensityUnit=u as IntensityUnit;});
    directionSection(group('传播方向'),l,'laser');
    const p=group('光束偏振');
    bindField(fields,p,'inspector',()=>numberSpec('polarization','主轴 ψ',l.polarizationAngleDeg,'°',{minimum:-90,maximum:90,step:1}),v=>{l.polarizationAngleDeg=Number(v);});
    bindField(fields,p,'inspector',()=>numberSpec('ellipticity','椭圆率 χ',l.ellipticityDeg,'°',{minimum:-45,maximum:45,step:1}),v=>{l.ellipticityDeg=Number(v);});
    const shape=polarizationButtons(l,[['circular-minus','圆−'],['linear','线偏振'],['circular-plus','圆+']],'一次性设置光束偏振形状');shape.classList.add('full-form-row');p.append(shape);
    const atomic=polarizationButtons(l,[['sigma-minus','σ−'],['pi','π'],['sigma-plus','σ+']],'链接到磁场参考轴并设置纯偏振');atomic.title='链接到磁场的参考轴：σ 令 k 沿 B，π 令 k 垂直 B、E 沿 B';p.append(fieldRow('纯偏振',atomic));
    const pol=element('div','polarization-bars full-form-row');pol.id='polarization-bars';p.append(pol);
    const more=element('details','advanced-controls');more.innerHTML='<summary>相位</summary>';host.append(more);const phase=form();more.append(phase);
    bindField(fields,phase,'inspector',()=>numberSpec('laser-phase','初相位 φ₀',l.phaseDeg??0,'°',{minimum:-180,maximum:180,step:1}),v=>{l.phaseDeg=Number(v);});
  }
  // One label column; within a section one unit width, so its number boxes share a right edge.
  requestAnimationFrame(()=>{shareLabelColumn(host);host.querySelectorAll<HTMLElement>('.control-section').forEach(alignUnitPickers);panel.scrollTop=scroll;});
}
/** 频率 (ui-design.md §1.3–1.4): the definition -- "相对" a base and the value against it -- then
 * the absolute frequency it defines. */
const readoutPrefs=new ReadoutPreferences(()=>((scene.display??={}).readouts??={}) as Record<string,ReadoutPreference>,()=>persist());
const laserReadout=new FrequencyReadout('laser-frequency',readoutPrefs,{unitPicker:false,changed:()=>{}});
const mixingWarning=element('p','note wrap mixing-warning','离共振太近：能级图不画它的光移');
/** A laser's |Ω/2Δ| and whether it is too close to resonance for a static light shift to
 * represent its effect: the levels chart draws one only while |Ω/2Δ| ≤ 1% and every virtual
 * transition keeps its minimum detuning (spectroscopy.ts displayMixing). Recomputed only
 * when a field or a laser changed: the inspector and the levels tag both ask. */
let mixingKey='',mixings=new Map<string,{ratio:number;tooClose:boolean}|undefined>();
function mixingOf(l:Laser):{ratio:number;tooClose:boolean}|undefined{
  const key=JSON.stringify([scene.fields,scene.lasers]);if(key!==mixingKey){mixingKey=key;mixings=new Map();}
  if(mixings.has(l.id))return mixings.get(l.id);
  let found:{ratio:number;tooClose:boolean}|undefined;
  try{const m=displayMixing(scene,l.id);found={ratio:m.ratio,tooClose:!(m.ratio<=STATIC_MIXING_LIMIT)||!m.withinGap};}catch{found=undefined;}
  mixings.set(l.id,found);return found;
}
/** The one warning beside 显示光移: shown while the switch is on and the light is too close. */
function refreshMixing(l:Laser):void{
  const m=mixingOf(l);
  mixingWarning.hidden=!(l.showLightShift===true&&m?.tooClose);
  mixingWarning.title=m&&Number.isFinite(m.ratio)?`|Ω/2Δ| 最大 ${populationValue(m.ratio)}，静态光移要求不超过 1%`:'';
}
const referenceReadout=new FrequencyReadout('reference-frequency',readoutPrefs,{unitPicker:false,changed:()=>{}});
/** 参考能级 (§1.6): E = E(锚点) ± hν(光子) + hδ -- a linked photon follows its laser or reference frequency. */
function levelSection(f:HTMLElement,r:ReferenceLevel):void{
  const fields=inspectorFields,after=()=>{renderObjects();renderInspector();update();};
  bindField(fields,f,'inspector',()=>({name:'level-anchor',label:'锚点',kind:'choice',tree:stateTree(allLevels()),value:r.anchorId}),v=>{if(!getLevel(String(v)))throw new Error('锚点必须是真实原子态');r.anchorId=String(v);},{after});
  bindField(fields,f,'inspector',()=>({name:'level-mode',label:'锚点能量',kind:'choice',segmented:true,choices:[['follow','随锚点'],['zero-field','固定零场']],value:r.anchorMode??'follow',description:'随锚点：跟随当前磁场和背景；固定零场：取锚点的零场能量'}),v=>{r.anchorMode=v as 'follow'|'zero-field';},{after});
  const sources=[...(scene.lasers.length?[{heading:'激光',leaves:scene.lasers.map(l=>({key:`laser:${l.id}`,label:l.name,text:l.name}))}]:[]),...(scene.referenceFrequencies?.length?[{heading:'参考频率',leaves:scene.referenceFrequencies.map(o=>({key:`reference:${o.id}`,label:o.name,text:o.name}))}]:[])];
  bindField(fields,f,'inspector',()=>({name:'level-photon',label:'光子',kind:'choice',tree:sources,allow_none:true,placeholder:'无',value:r.photon?`${r.photon.source.kind}:${r.photon.source.id}`:null,description:'链接后，参考能级随这束光或参考频率移动'}),
    v=>{if(v===null||v===''){if(r.photon){const nu=photonFrequencyMHz(r,scene)??0;r.offsetMHz+=r.photon.direction==='emit'?-nu:nu;delete r.photon;}return;}
      const [kind,...rest]=String(v).split(':'),id=rest.join(':');if((kind!=='laser'&&kind!=='reference')||!objectOf(scene,String(v)))throw new Error('未知的光子来源');r.photon={source:{kind,id},direction:r.photon?.direction??'absorb'};},{after});
  if(r.photon)bindField(fields,f,'inspector',()=>({name:'level-direction',label:'方向',kind:'choice',segmented:true,choices:[['absorb','吸收 +hν'],['emit','受激辐射 −hν']],value:r.photon?.direction??'absorb'}),v=>{if(r.photon)r.photon.direction=v as 'absorb'|'emit';},{after});
  bindQuantity(fields,f,'inspector',`level:${r.id}`,()=>({name:'level-offset',label:'偏移 δ',spin:true,inexact:true}),
    {read:()=>offsetReading(r),write:(v,u)=>setReferenceOffset(r,v,u as DetuningUnit),units:()=>unitChoices(FREQUENCY_UNITS,'频率')},after);
  const source=r.photon?objectOf(scene,`${r.photon.source.kind}:${r.photon.source.id}`)?.name:undefined;
  f.append(element('p','note wrap form-note',`E = E(锚点)${source?` ${r.photon!.direction==='emit'?'−':'+'} hν(${source})`:''} + hδ${r.anchorMode==='zero-field'?'，锚点取零场能量':''}`));
}
function frequencySection(f:HTMLElement,object:DefinedFrequency,readout:FrequencyReadout):void{
  const fields=inspectorFields,key=objectKey(scene,object),following=new Set([key,...followers(scene,key)]);
  // No base that is itself or follows it: no cycle can be chosen.
  const offered=<T extends {id:string}>(kind:'laser'|'reference',list:readonly T[])=>list.filter(o=>!following.has(`${kind}:${o.id}`));
  bindField(fields,f,'inspector',()=>({name:'frequency-base',label:'相对',kind:'choice',
    tree:frequencyBaseTree({absolute:'光频或真空波长',absoluteText:'绝对',follow:true,lasers:offered('laser',scene.lasers),references:offered('reference',scene.referenceFrequencies??[])}),
    value:baseKey(expressionBase(object.frequency!)),description:'换基准不改变光频；跟随另一对象时随它移动'}),
    v=>{const base=parseBase(String(v));if(!base)throw new Error('未知的频率基准');setFrequencyBase(scene,object,base);},{after:()=>{renderInspector();renderObjects();update();}});
  bindQuantity(fields,f,'inspector',key,()=>({name:'frequency-value',label:frequencyValueLabel(object),spin:true,inexact:true}),
    {read:()=>frequencyValue(object),write:(v,u)=>setFrequencyValue(scene,object,v,u as FrequencyDisplayUnit),
      units:()=>[...unitChoices(FREQUENCY_UNITS,'频率'),...(expressionBase(object.frequency!).kind==='absolute'?unitChoices(WAVELENGTH_UNITS,'波长'):[])]});
  // The absolute frequency the definition gives, in its natural unit; an absolute definition is it already.
  if(expressionBase(object.frequency!).kind!=='absolute')f.append(fieldRow('频率 ν',readout.readout.root));
  readout.show(anchorValue(objectAnchor(scene,object)),scene);
  const names=followers(scene,key).map(k=>objectOf(scene,k)?.name).filter((name):name is string=>!!name);
  if(names.length)f.append(element('p','note wrap form-note',`被 ${names.join('、')} 跟随`));
}
/** 初态 (ui-design.md §3): one description, edited here only -- a kind, then what that kind needs. */
// Four kinds do not fit the scene panel side by side: a combo box.
const initialKind={root:element('select','initial-kind'),setCurrent(kind:string){if(this.root.value!==kind)this.root.value=kind;}};
initialKind.root.append(comboFace(initialKind.root),...INITIAL_KINDS.map(([kind,words])=>new Option(words,kind)));initialKind.root.setAttribute('aria-label','初态类型');
initialKind.root.onchange=()=>{const kind=initialKind.root.value as InitialKind,levels=allLevels();applyInitial(startingSpec(kind,initialSpec(scene),levels));if(kind==='weights')void editInitialWeights();};
const unpolarizedChoice=new Segmented(manifold=>applyInitial({kind:'unpolarized',manifold:manifold as UnpolarizedManifold}));
unpolarizedChoice.setChoices(UNPOLARIZED);unpolarizedChoice.root.setAttribute('aria-label','未极化的流形');
const initialFields=new Map<string,BoundField>();
const weightsSummary=element('span','initial-summary'),editWeights=command('编辑权重…',()=>void editInitialWeights(),{role:'quiet'});
function renderInitial():void{
  const levels=allLevels(),spec=initialSpec(scene),host=$('initial-state-controls');
  initialKind.setCurrent(spec.kind);
  if(initialBody!==spec.kind||!host.contains(initialKind.root)){
    initialBody=spec.kind;disposeTrees(host);initialFields.clear();host.replaceChildren(initialKind.root);
    if(spec.kind==='state'){
      initialTree??=new TreeCombo(id=>setInitialState(id));initialTree.root.id='initial-state-picker';initialTree.root.setAttribute('aria-label','初态');host.append(initialTree.root);
    }else if(spec.kind==='unpolarized')host.append(unpolarizedChoice.root);
    else if(spec.kind==='thermal'){
      const rows=form('initial-form');host.append(rows);
      bindQuantity(initialFields,rows,'initial','initial',()=>({name:'temperature',label:'温度 T',spin:true,inexact:true,description:'静态哈密顿量 H0 的 8 个基态本征态上的 Boltzmann 分布'}),
        {read:()=>{const now=initialSpec(scene);return temperatureReading(now.kind==='thermal'?now.temperatureK:DEFAULT_TEMPERATURE_K,dynamics().temperatureUnit);},
          write:(v,u)=>setTemperature(scene,v,u as TemperatureUnit,allLevels()),units:()=>unitChoices(TEMPERATURE_UNITS,'温度')},()=>{renderInitial();update();});
    }else host.append(weightsSummary,editWeights.root);
  }
  if(spec.kind==='state')initialTree!.setTree(stateTree(levels),spec.id);
  else if(spec.kind==='unpolarized')unpolarizedChoice.setCurrent(spec.manifold);
  else if(spec.kind==='thermal')refreshFields(initialFields);
  else weightsSummary.textContent=initialSummary(spec);
  initialShortcut.title=`初态：${initialSummary(spec,dynamics().temperatureUnit)}`;
}
function applyInitial(spec:InitialStateSpec):void{
  try{setInitialSpec(scene,spec,allLevels());}catch(error){note(message(error),'error');return;}
  renderInitial();update();
}
function setInitialState(id:string):void{const state=getLevel(id);if(!state)return;if(state.manifold!=='ground')spectral=state.manifold;applyInitial({kind:'state',id});}
/** 自定义: a weight for any of the 32 states, in an F × m grid; shown normalized. */
async function editInitialWeights():Promise<void>{
  const levels=allLevels(),start=initialSpec(scene),weights=start.kind==='weights'?start.weights:(startingSpec('weights',start,levels) as {weights:Record<string,number>}).weights;
  const dialog=new Dialog('自定义初态'),grid=element('div','weights-grid'),inputs=new Map<string,HTMLInputElement>(),fractions=new Map<string,HTMLElement>();
  grid.append(element('span','weights-corner','m'),...[-3,-2,-1,0,1,2,3].map(m=>element('span','weights-head',signedInteger(m))));
  for(const [manifold,F] of [['ground',1],['ground',2],['D1',1],['D1',2],['D2',0],['D2',1],['D2',2],['D2',3]] as const){
    grid.append(element('span','weights-row',hyperfineLabel(manifold,F)));
    for(let m=-3;m<=3;m++){
      const cell=element('span','weights-cell');grid.append(cell);if(Math.abs(m)>F)continue;
      const id=`${manifold}-${F}-${m}`,input=element('input');input.type='text';input.inputMode='decimal';input.spellcheck=false;input.setAttribute('aria-label',`${idLabel(id)} 的权重`);
      input.value=weights[id]?String(Number(weights[id]!.toPrecision(12))):'';
      const share=element('span','weights-fraction');cell.append(input,share);inputs.set(id,input);fractions.set(id,share);
    }
  }
  const total=element('p','note wrap weights-total');
  dialog.content.append(element('p','note wrap','填写非负权重，按总和归一化；空白为 0。'),grid,total);
  dialog.addAction('取消','cancel');const ok=dialog.addAction('确定','ok','primary');
  const read=():Record<string,number>|undefined=>{let bad=false;const out:Record<string,number>={};
    for(const [id,input] of inputs){const text=input.value.trim(),value=text===''?0:Number(text),valid=Number.isFinite(value)&&value>=0;input.toggleAttribute('aria-invalid',!valid);if(!valid){bad=true;continue;}if(value>0)out[id]=value;}
    return bad?undefined:out;};
  const refresh=():void=>{const w=read(),sum=w?Object.values(w).reduce((s,v)=>s+v,0):0;
    for(const [id,share] of fractions)share.textContent=w&&sum>0&&w[id]?`${formatNumber(w[id]!/sum*100,{digits:3})}%`:'';
    ok.disabled=!w||!(sum>0);total.textContent=!w?'权重必须是非负数。':sum>0?`${Object.keys(w).length} 个态`:'至少一个态的权重为正。';};
  grid.addEventListener('input',refresh);refresh();
  const answer=dialog.open();inputs.values().next().value?.focus();
  if(await answer!=='ok')return;
  const chosen=read();if(chosen&&Object.keys(chosen).length)applyInitial({kind:'weights',weights:chosen});
}

/** σ⁻, π, σ⁺ of the actual polarization on the quantization axis: the total field, at zero
 * field the first field's reference axis (physics.ts composeMagneticField). */
function quantizationWords(a:Analysis):string{return a.fieldMagnitudeG>0?'总磁场 B':scene.fields[0]?`${scene.fields[0].name}的参考轴（B = 0）`:'+z（无磁场）';}
function polarizationBars(a:Analysis,l:Laser|undefined):string{const p=l?a.polarizations[l.id]:undefined;if(!p)return '';return `<span class="polarization-axis-label">相对${esc(quantizationWords(a))}</span>`+[[math('\\sigma^-'),p.minus],[math('\\pi'),p.pi],[math('\\sigma^+'),p.plus]].map(([label,v])=>`<div class="pol-row"><span>${label}</span><div class="bar-track"><span class="bar-fill" style="width:${Math.max(0,Math.min(100,Number(v)*100))}%"></span></div><strong>${esc(populationPercent(Number(v)))}</strong></div>`).join('');}


function paintStateBadges():void{
  if(view==='levels')alignLevelReadouts($('levels-chart'));
  document.querySelectorAll<HTMLElement>('.selection-index[data-badge-number]').forEach(host=>drawBadges(host,[[host.dataset.badgeNumber!,'scan']]));
  $('levels-chart').querySelectorAll<SVGGElement>('.state-selection-number').forEach(group=>{const holder=document.createElement('span');drawBadges(holder,[[group.dataset.badgeNumber!,'scan']]);const svg=holder.firstElementChild!;svg.setAttribute('x',group.dataset.badgeX!);svg.setAttribute('y',group.dataset.badgeY!);group.replaceChildren(svg);});
}
/** A transition's photon frequency ν₁₂ = |E₂ − E₁|/h (the pickers say absorption or emission);
 * `zeroField`: the same two states with no field and no light, the pair's default reference (ui-design.md §1.8). */
function photonValue(fromId:string,toId:string,zeroField=false):FrequencyValue|undefined{
  const read=zeroField?zeroFieldStateValue:stateValue,a=read(fromId,allLevels(),scene),b=read(toId,allLevels(),scene);return a&&b?photon(gap(a,b)):undefined;
}
/** The two-state readout (ui-design.md §1.8): ν₁₂ itself, then Δν = ν₁₂ − ν(参考) against the
 * frequency chosen beside it; by default the same two states at zero field, so Δν is what the
 * field and the lasers shown (显示光移) move this transition by. */
let pairValue:FrequencyValue|undefined,pairZeroField:FrequencyValue|undefined;
const pairFrequency=new FrequencyReadout('pair-frequency',readoutPrefs,{temperature:true,changed:()=>{}});
const pairReadout=new FrequencyReadout('pair',readoutPrefs,{
  bases:()=>frequencyBaseTree({absolute:'光频',absoluteText:'绝对',lasers:scene.lasers,references:scene.referenceFrequencies}),
  automatic:()=>({name:'零场 ν₁₂',value:pairZeroField}),
  changed:()=>renderResults(false),
});
const pairRow=element('div','pair-frequency'),pairPart=(...nodes:HTMLElement[])=>{const part=element('span','pair-part');part.append(...nodes);return part;};
const pairLabel=element('span','pair-label','ν₁₂'),shiftLabel=element('span','pair-label','Δν');
pairLabel.title='跃迁频率 |E₂ − E₁|/h';shiftLabel.title='ν₁₂ − ν(参考)：参考缺省为这两个态在零场（无磁场、无光移）下的 ν₁₂';
pairRow.append(pairPart(pairLabel,pairFrequency.readout.root,pairFrequency.unit.root),pairPart(shiftLabel,pairReadout.readout.root,pairReadout.unit.root,pairReadout.baseLabel,pairReadout.base!.root));
function showPair():void{pairFrequency.show(pairValue,scene);pairReadout.show(pairValue,scene);}
/** The state the actions act on: the one labelled 2, else the one labelled 1. */
function labelTarget():string|undefined{return stateLabels.second??stateLabels.first;}
/** The two-state readout: the states labelled 1 and 2 (each badge clears its label), and with
 * both ν₁₂, Δν, the coupling and the population of 2. */
function selectionReadout(a:Analysis,l:Laser|undefined):string{
  const from=stateLabels.first,to=stateLabels.second,target=labelTarget(),physical=!!target&&!!getLevel(target);
  const badge=(slot:LabelSlot,n:1|2)=>`<span class="selection-index${stateLabels[slot]?' set':''}" data-badge-number="${n}"${stateLabels[slot]?` data-clear-label="${slot}" role="button" tabindex="0" title="取消标记 ${n}"`:''}>${n}</span>`;
  const pickers=`<div class="selection-top"><div class="selection-pair"><div class="selection-endpoint">${badge('first',1)}<div id="label-first"></div></div><div id="label-swap"></div><div class="selection-endpoint">${badge('second',2)}<div id="label-second"></div></div></div><button class="quiet" id="clear-selection"${from||to?'':' disabled'}>清除</button></div>`;
  const actions=`<div class="selection-actions"><button class="quiet" id="use-as-initial"${physical?'':' disabled'}>设为初态</button><button class="quiet" id="monitor-selected"${[from,to].some(id=>id&&getLevel(id))?'':' disabled'}>观察布居</button><div class="mark-editor"><input id="mark-text" maxlength="32" aria-label="能级标记" placeholder="标记" value="${esc(target&&physical?scene.marks?.[target]??'':'')}"${physical?'':' disabled'}><button id="save-mark"${physical?'':' disabled'}>标记</button></div></div>`;
  pairValue=pairZeroField=undefined;
  if(!from||!to)return pickers+actions;
  const defined=a.spectroscopy?.status!=='unresolved';
  pairValue=defined?photonValue(from,to):undefined;pairZeroField=photonValue(from,to,true);
  const first=getLevel(from),second=getLevel(to);let coupling='';
  // Each reading in a slot as wide as its longest value, so none moves another.
  const slot=(label:string,value:string,width:string,title='')=>`<span${title?` title="${title}"`:''}>${label} <b class="slot ${width}">${value}</b></span>`;
  if(first&&second){const referenceLaser=selectedBeam(l)?l:scene.lasers[0],transition=defined&&referenceLaser?spectroscopyCoupling(a,referenceLaser,first.id,second.id):undefined;coupling=!defined?'<span>背景矩元未定义</span>':transition?(transition.allowed?slot('|d|',`${measuredNumber(Math.abs(transition.amplitudeCM)/1e-29,4)} × 10⁻²⁹ C·m`,'dipole','与所示背景能级使用同一基底的电偶极矩元')+(l&&selectedBeam(l)?slot(l.enabled?'Ω/2π':'Ω/2π（停用预览）',`${measuredNumber(transition.rabiMHz,4)} MHz`,'rabi'):''):'<span>E1 禁戒</span>'):'';}
  else coupling='<span>参考能级</span>';
  if(first&&second&&defined&&!scene.lasers.length){const bare=dipoleBetweenStates(first.id,second.id,a.fieldMagnitudeG);coupling=bare.allowed?slot('|d|',`${measuredNumber(Math.abs(bare.amplitudeCM)/1e-29,4)} × 10⁻²⁹ C·m`,'dipole','无光背景的磁场本征态偶极矩元'):'<span>E1 禁戒</span>';}
  const pop=populationSnapshot().populations?.[to];
  return `${pickers}<div id="pair-frequency-slot"></div><div class="compact-readout">${coupling}<span data-current-population ${pop===undefined?'hidden':''}>p₂ <b class="slot percent">${pop===undefined?'—':populationValue(pop)}</b></span></div>${actions}`;
}
/** Solver responses update scientific readouts without unmounting an operator's editor. */
function renderSelection(a:Analysis,l:Laser|undefined):void{
  const host=$('selection-readout'),key=JSON.stringify([stateLabels,scene.marks,scene.references?.map(r=>[r.id,r.name])]),next=element('div');next.innerHTML=selectionReadout(a,l);
  if(host.dataset.selectionKey!==key){disposeTrees(host);host.innerHTML=next.innerHTML;host.dataset.selectionKey=key;bindSelection();$('pair-frequency-slot')?.replaceWith(pairRow);showPair();return;}
  if(!pairRow.isConnected)$('pair-frequency-slot')?.replaceWith(pairRow);
  showPair();
  const readout=host.querySelector('.compact-readout'),updated=next.querySelector('.compact-readout');if(readout&&updated&&readout.innerHTML!==updated.innerHTML)readout.innerHTML=updated.innerHTML;
}
function renderDetails(hostId:string,markup:string):void{
  const host=$(hostId),next=element('div');next.innerHTML=markup;const nodes:HTMLElement[]=[];
  for(const incoming of [...next.children] as HTMLElement[]){
    const existing=[...host.children].find(el=>el.className===incoming.className) as HTMLElement|undefined;
    if(existing){if(existing.innerHTML!==incoming.innerHTML)existing.innerHTML=incoming.innerHTML;nodes.push(existing);}else nodes.push(incoming);
  }
  reconcileChildren(host,nodes);
}
function bindSelection():void{
  const groups=stateTree(allLevels(),scene.references);
  const picker=(slot:LabelSlot,host:string,words:string)=>{const tree=new TreeCombo(id=>applyLabels(withLabel(stateLabels,slot,id||undefined)));tree.setTree(groups,stateLabels[slot]??'','未标记');tree.root.setAttribute('aria-label',words);treeOwners.set(tree.root,tree);$(host).append(tree.root);};
  picker('first','label-first','标记 1 的态');picker('second','label-second','标记 2 的态');
  const swap=command('⇄',()=>applyLabels(swapLabels(stateLabels)),{role:'quiet',glyph:true,square:true,title:'交换 1 和 2'});swap.disabled=!stateLabels.first||!stateLabels.second;$('label-swap').append(swap.root);
  if($('clear-selection'))$('clear-selection').onclick=()=>applyLabels({});
  if($('use-as-initial'))$('use-as-initial').onclick=()=>{const l=getLevel(labelTarget()??'');if(l)setInitialState(l.id);};
  if($('save-mark'))$('save-mark').onclick=()=>{const id=labelTarget();if(id&&getLevel(id)){const v=$<HTMLInputElement>('mark-text').value.trim();if(v)scene.marks![id]=v;else delete scene.marks![id];update();}};
  if($('monitor-selected'))$('monitor-selected').onclick=()=>{const ids=labelled(stateLabels).map(l=>l.id).filter(id=>getLevel(id));const chosen=getLevel(ids.at(-1)??'');const matching=chosen?.manifold==='D1'||chosen?.manifold==='D2'?scene.lasers.find(l=>l.enabled&&l.line===chosen.manifold):activeLaser();if(matching)dynamics().laserId=matching.id;dynamics().monitoredStateIds=[...new Set([...dynamics().monitoredStateIds,...ids])];dynamics().monitorMode='manual';controlsKey='';showView('dynamics');};
}
/** The table reads the two-state readout's base and unit (ui-design.md §1.8): the chosen base for
 * every row, or under the zero-field default each row against its own two states at zero field. Only components with Ω ≠ 0 are listed. */
function transitionTable(transitions:Transition[]):{heading:string;rows:string}{
  const listed=transitions.filter(t=>t.rabiMHz>0),ends=(t:Transition)=>transitionEndpoints(t,referenceStateId(scene.lasers.find(l=>l.id===t.laserId)))!;
  const fixed=pairReadout.fixedBase(scene),relative=fixed?.base.kind!=='absolute';
  const pref=pairReadout.chosenUnit(),unit:DetuningUnit=pref&&isDetuningUnit(pref)?pref:relative?'MHz':'THz';
  return {heading:fixed?readingHeading(fixed.base,scene,unit):`ν − ν(零场) / ${unit}`,rows:listed.map(t=>{
  const endpoints=ends(t),mixed=t.mixedPolarization===true,ν=photonValue(endpoints.fromId,endpoints.toId),origin=fixed?.value??photonValue(endpoints.fromId,endpoints.toId,true);
  return `<tr data-transition="${esc(t.id)}" data-from-state="${endpoints.fromId}" data-to-state="${endpoints.toId}" tabindex="0"><td title="${endpoints.emission?'受激辐射':'吸收'}">${stateCell(endpoints.fromId)} → ${stateCell(endpoints.toId)}</td><td class="component" title="${mixed?'混合背景态：多个球分量相干耦合':t.mLabelApproximate?`光场 q=${t.q}；mF 是混合本征态的主成分标签，标签之差不是严格的 ΔmF`:`光场 q=${t.q}；原子 Δm=${endpoints.deltaM}`}" aria-label="${mixed?'混合分量':t.q===-1?'sigma minus':t.q===0?'pi':'sigma plus'}">${mixed?'混合':math(t.q===-1?'\\sigma^-':t.q===0?'\\pi':'\\sigma^+')}</td><td class="num frequency">${ν&&origin?formatNumber(inUnit(difference(ν,origin),unit),{digits:relative?9:12,signed:relative}):'—'}</td><td class="num rabi">${measuredNumber(t.rabiMHz,4)}</td><td><div class="strength-cell"><span style="width:${Math.min(100,t.strength*t.weight*100)}%"></span><small>${measuredNumber(t.strength*t.weight,4)}</small></div></td></tr>`;
}).join('')};}

// ---------------------------------------------------------------- solver
/** The page whose results are shown: 详情 reads the last analysis page. */
function resultPage():View{return view==='details'?resultView:view;}
function dynamicsKey(steadyOnly=resultPage()==='levels'&&populationMode==='steady'):string{return dynamicsCacheKey(scene,dynamics(),steadyOnly);}
/** What a trajectory depends on: everything but the mean's horizon (the displayed results' key). */
function trajectoryKey():string{return dynamicsCacheKey(scene,dynamics(),false,false);}
/** A solve for these inputs asks only for the mean: in the levels' long-time mode, or when a
 * trajectory is here whose mean was solved at another horizon. Its key is the request's. */
function meanOnly():boolean{const t=displayedResults.get(trajectoryKey(),'trajectory');return resultPage()==='levels'&&populationMode==='steady'||(!!t&&t.horizon!==resolveMeanHorizon(dynamics()));}
function solveKey():string{return dynamicsKey(meanOnly());}
function meanRecord(key=trajectoryKey()):SolvedRecord|undefined{const record=displayedResults.get(key,'mean');return record&&record.horizon===resolveMeanHorizon(dynamics())?record:undefined;}
function cachedAnalysis():Analysis{
  const spectrum=resultPage()==='levels',key=JSON.stringify({kind:spectrum?'spectrum':'bare',fields:scene.fields,lasers:scene.lasers,initial:initialId(),excluded:spectrum?activeLaser()?.id:undefined});
  const cached=analysisCache.get(key);if(cached)return cached;
  const start=performance.now(),analysis=spectrum?spectroscopyAnalysis(scene,activeLaser()?.id):analyze(scene);

  analysisCache.set(key,analysis);while(analysisCache.size>4)analysisCache.delete(analysisCache.keys().next().value!);
  root.dataset.spectrumBuildCount=String(++spectrumBuildCount);root.dataset.spectrumComputeMs=String(performance.now()-start);return analysis;
}
function invalidateObsoletePopulation():void{
  // Retain the displayed calculation only; revisiting earlier parameters must solve again.
  const physicalKey=trajectoryKey();displayedResults.activate(physicalKey);
  // A stop holds for the inputs it stopped; other inputs, even the stopped ones again later, solve.
  if(stoppedKey&&stoppedKey!==solveKey())stoppedKey='';
  // A request for other inputs or another horizon is obsolete; opening another view is not.
  const key=dynamicsKey();if(requestedPhysicalKey&&(requestedPhysicalKey!==physicalKey||requestedHorizon!==resolveMeanHorizon(dynamics())))cancelObsoleteSolve();
  if(renderedPopulationKey&&renderedPopulationKey!==key){
    if(view==='levels')patchLevelPopulations($('levels-chart'),{label:'布居待计算',steadyLabel:meanSymbol(undefined),initialId:singleInitialId()});
    const oldReadout=$('selection-readout').querySelector<HTMLElement>('[data-current-population]');if(oldReadout){oldReadout.hidden=true;const value=oldReadout.querySelector('b');if(value)value.textContent='—';}
    root.dataset.populationFresh='false';if(latestAnalysis&&view==='levels')populationToolbar();
  }
}
/** The levels chart's populations and their hover words; a value outside the tolerance says its error. */
function paintLevelPopulations(snapshot:ReturnType<typeof populationSnapshot>):void{
  const result=currentResult(),outside=(error:number|null|undefined,over:boolean|undefined)=>over?`（估计误差 ${error==null?'未知':errorPercent(error)}，超过 ${errorPercent(POPULATION_TOLERANCE)}）`:'';
  patchLevelPopulations($('levels-chart'),{populations:snapshot.populations,steady:steadyPopulationsInView(),label:snapshot.label,steadyLabel:meanSymbol(),initialId:singleInitialId(),
    note:outside(snapshot.error,snapshot.outside),steadyNote:result?.steady.converged?outside(result.steady.estimatedError,!result.steady.withinTolerance):''});
}
function refreshPopulationOnly():void{
  if(analysisError){refreshStatus();return;}
  if(view==='levels'){
    const snapshot=populationSnapshot();populationToolbar();paintLevelPopulations(snapshot);
    alignLevelReadouts($('levels-chart'));
    // The readout editor and its focusable state chips remain mounted.
    const p=$('selection-readout').querySelector<HTMLElement>('[data-current-population]'),value=stateLabels.second?snapshot.populations?.[stateLabels.second]:undefined;
    if(p){p.hidden=value===undefined;const b=p.querySelector('b');if(b)b.textContent=value===undefined?'—':populationValue(value);}
  }else if(view==='dynamics')renderDynamics(latestAnalysis);
  else if(view==='details')renderDetailsPage();
  renderedPopulationKey=dynamicsKey();root.dataset.populationFresh=String(!!currentResult());root.dataset.populationPaintCount=String(++populationPaintCount);refreshStatus();
}
/** Every change solves automatically what the page shows: the mean in the levels' long-time
 * mode, else the trajectory and its mean. Nothing is solved twice or after 停止 for those inputs. */
function requestSolve():void{
  // A trajectory already here for these inputs needs only its mean at the new horizon.
  const physicalKey=trajectoryKey(),trajectory=displayedResults.get(physicalKey,'trajectory'),steadyOnly=meanOnly();
  const key=dynamicsKey(steadyOnly),horizon=resolveMeanHorizon(dynamics());
  if(key===stoppedKey)return;
  displayedResults.activate(physicalKey);
  const completed=steadyOnly?meanRecord(physicalKey):trajectory;
  if(completed||key===requestedKey)return;
  // A current full solve already supplies its own mean. Merely opening the
  // level view must not cancel/restart it or submit a duplicate mean request.
  if(solverBusy&&requestedPhysicalKey===physicalKey&&requestedHorizon===horizon&&(!requestedSteadyOnly||steadyOnly))return;
  const epoch=displayedResults.activate(physicalKey);
  solver.cancel();requestedKey=key;requestedPhysicalKey=physicalKey;requestedSteadyOnly=steadyOnly;requestedHorizon=horizon;
  const revision=++solverRevision;clearTimeout(solverTimer);solverBusy=true;solverQueued=true;solverError='';
  const submittedScene=structuredClone(scene),submittedSettings=structuredClone(dynamics()),submittedAt=performance.now();
  const stillCurrent=()=>revision===solverRevision&&physicalKey===trajectoryKey()&&horizon===resolveMeanHorizon(dynamics());
  const submittedOptions={steadyOnly,onProgress:(result:DynamicsResult)=>{
    if(!stillCurrent())return;
    // A computed mean is the long-time result so far; a checkpoint also carries the trajectory
    // evaluated so far, shown until the next one or the final result replaces it.
    const record={result,wallMs:performance.now()-submittedAt,horizon};
    if(result.steady.converged)displayedResults.set(physicalKey,epoch,'mean',record);
    displayedResults.set(physicalKey,epoch,'progress',record);
    refreshPopulationOnly();
  }};
  solverTimer=window.setTimeout(async()=>{
    if(!stillCurrent())return;solverQueued=false;refreshPopulationOnly();const begin=performance.now();
    try{
      const result=await solver.solve(submittedScene,submittedSettings,submittedOptions);if(!stillCurrent())return;
      solverWallMs=performance.now()-begin;const record={result,wallMs:solverWallMs,horizon};
      displayedResults.set(physicalKey,epoch,steadyOnly?'mean':'trajectory',record);
      if(result.steady.converged)displayedResults.set(physicalKey,epoch,'mean',record);
      solverBusy=false;refreshPopulationOnly();
    }catch(e){if(!stillCurrent())return;solverBusy=false;solverError=message(e);refreshPopulationOnly();}
  },16);
  refreshPopulationOnly();
}
function cancelObsoleteSolve():void{solver.cancel();solverRevision++;clearTimeout(solverTimer);requestedKey='';requestedPhysicalKey='';requestedSteadyOnly=false;solverBusy=false;solverQueued=false;solverError='';}
/** 停止: the running solve ends here; a mean it already delivered stays. Other inputs solve again. */
function stopSolve():void{
  if(!currentSolverBusy())return;
  const key=solveKey();cancelObsoleteSolve();stoppedKey=key;requestedKey=key;
  if(view==='dynamics')renderDynamics(latestAnalysis);else refreshPopulationOnly();
  refreshStatus();
}
function currentRecord():SolvedRecord|undefined{
  const key=trajectoryKey();
  if(resultPage()==='levels'&&populationMode==='steady')return meanRecord(key);
  const trajectory=displayedResults.get(key,'trajectory')??displayedResults.get(key,'progress');
  if(!trajectory||trajectory.horizon===resolveMeanHorizon(dynamics()))return trajectory;
  // The trajectory stays; its long-time part is the mean at the current horizon, or pending.
  const mean=meanRecord(key);return withMean(trajectory,mean);
}
const mergedRecords=new WeakMap<SolvedRecord,{mean:SolvedRecord|undefined;record:SolvedRecord}>();
function withMean(trajectory:SolvedRecord,mean:SolvedRecord|undefined):SolvedRecord{
  const saved=mergedRecords.get(trajectory);if(saved&&saved.mean===mean)return saved.record;
  const steady=mean?.result.steady??{converged:false,status:'not-requested' as const,populations:null,estimatedError:null,withinTolerance:false,residual:null,initialDependent:false};
  const record={...trajectory,result:{...trajectory.result,steady},horizon:resolveMeanHorizon(dynamics())};mergedRecords.set(trajectory,{mean,record});return record;
}
function currentResult():DynamicsResult|undefined{return currentRecord()?.result;}
function currentSolverError():string{return requestedKey===dynamicsKey(requestedSteadyOnly)?solverError:'';}
function currentSolverBusy():boolean{return solverBusy&&requestedPhysicalKey===trajectoryKey()&&requestedHorizon===resolveMeanHorizon(dynamics())&&(!requestedSteadyOnly||(resultPage()==='levels'&&populationMode==='steady')||!!displayedResults.get(trajectoryKey(),'trajectory'));}
function steadyPopulationsInView(a=latestAnalysis):Record<string,number>|undefined{
  const result=currentResult(),steady=result?.steady;if(!result||!steady?.converged||!steady.populations)return undefined;
  if(resultPage()!=='levels')return steady.populations;
  let entries=projectedPopulationCache.get(a);if(!entries){entries=new WeakMap();projectedPopulationCache.set(a,entries);}if(entries.has(steady))return entries.get(steady)??undefined;
  const projected=projectSpectroscopyPopulations(a,{states:result.states.map(state=>state.id),populations:result.states.map(state=>steady.populations![state.id]??0),density:steady.density,basis:result.states});entries.set(steady,projected);return projected??undefined;
}
/** The populations the levels chart is coloured by, what they are, and their estimated error:
 * p̄(T)'s, or the chosen sample's. */
function populationSnapshot():{populations?:Record<string,number>;label:string;timeUs?:number;error?:number|null;outside?:boolean}{
  const result=currentResult();
  if(!result)return {label:currentSolverBusy()?'计算中':'待计算'};
  if(populationMode==='steady'){const populations=steadyPopulationsInView(),s=result.steady;return s?.converged&&s.populations?{populations,label:populations?meanSymbol(result):'当前背景布居未定义',error:s.estimatedError,outside:!s.withinTolerance}:{label:'长期布居未解析'};}
  const index=populationTimeIndex<0?result.timesUs.length-1:Math.min(populationTimeIndex,result.timesUs.length-1),timeUs=result.timesUs[index]??0;
  const error=sampleErrors(result)?.[index],accuracy=error===undefined?{}:{error,outside:error>POPULATION_TOLERANCE};
  // p(t) is in the model basis; on the chart's states it reads by id only while the bases'
  // difference stays within the population accuracy (population-basis.ts).
  const basisError=resultPage()==='levels'?timeBasisError(result):0;
  if(!(basisError<=POPULATION_TOLERANCE))return {label:'p(t) 未着色：模型基与能级图的基相差过大',timeUs};
  return {populations:Object.fromEntries(Object.entries(result.populations).map(([id,values])=>[id,values[index]??0])),label:`p(${formatNumber(timeUs,{digits:4})} μs)${basisError>1e-6?` · 基差 ≤ ${formatNumber(basisError,{digits:2})}`:''}`,timeUs,...accuracy};
}
/** How much carrying the result's p(t) over to the chart's states by id can change them. */
const basisErrors=new WeakMap<object,{analysis:Analysis;error:number}>();
function timeBasisError(result:DynamicsResult,a=latestAnalysis):number{
  const saved=basisErrors.get(result);if(saved?.analysis===a)return saved.error;
  const error=idMappingError(result.states,allLevels(a));basisErrors.set(result,{analysis:a,error});return error;
}
/** Under 最慢弛豫时间: a relaxation slower than the chosen horizon has not reached t → ∞ by then. */
function unreachedLine(result:DynamicsResult):string{
  const horizon=resolveMeanHorizon(dynamics()),slowest=result.steady?.slowestRelaxationUs;if(slowest==null||horizon===null||!(slowest>horizon))return '';
  const unit=timeDisplayUnit(slowest);return `最慢弛豫约 ${withUnit(slowest*unit.factor,unit.name,{digits:2})}，长于 ${horizonText(horizon)}：尚未到达 t→∞ 的稳态`;
}
/** 计算详情 on 详情: the solve's numbers, its methods and assumptions. */
function solverDetails():string{
  const result=currentResult(),error=currentSolverError();
  const readout=(label:string,value:string)=>`<span class="field-label">${label}</span><span class="readout-edit">${esc(value)}</span>`;
  const small=(value:number)=>formatNumber(value,{digits:2});
  const accuracy=result?.populationAccuracy;
  const rows=result?[
    readout('求解耗时',formatElapsed(result.metrics.totalMs)),readout('含通信',formatElapsed(currentRecord()?.wallMs??solverWallMs)),
    ...(accuracy?[readout('布居误差估计',accuracy.status==='estimated-partial'?`最大 ${small(accuracy.estimatedMaximumError)}，部分时刻超过目标 ${small(accuracy.requestedTolerance)}`:`≤ ${small(accuracy.estimatedMaximumError)}（目标 ${small(accuracy.requestedTolerance)}）`)]:[]),
    ...(accuracy?.status==='estimated-partial'?[`<p class="horizon-note">${esc([trajectorySentence(result),accuracy.reason?solverText(accuracy.reason):''].filter(Boolean).join('。'))}</p>`]:[]),
    ...(result.steady.converged&&result.steady.estimatedError!==null?[readout('长期布居误差估计',result.steady.withinTolerance?`≤ ${small(result.steady.estimatedError)}（目标 ${small(POPULATION_TOLERANCE)}）`:`${small(result.steady.estimatedError)}，超过目标 ${small(POPULATION_TOLERANCE)}`)]:[]),
    readout('概率守恒误差',small(result.validation.maxTraceError)),readout('最小布居',small(result.validation.minPopulation)),
    readout('长期解',`${steadyStatusText(result.steady?.status)}${result.steady?.residual==null?'':` · 残差 ${small(result.steady.residual)}`}`),
    // The limit is reached after a few of these; more than one conserved quantity makes it initial-dependent.
    ...(result.steady?.slowestRelaxationUs!=null?[readout('最慢弛豫时间',(()=>{const unit=timeDisplayUnit(result.steady.slowestRelaxationUs!);return withUnit(result.steady.slowestRelaxationUs!*unit.factor,unit.name,{digits:3});})()),...(unreachedLine(result)?[`<p class="horizon-note">${esc(unreachedLine(result))}</p>`]:[])]:[]),
    ...(result.steady?.conservedQuantities!=null?[readout('守恒量个数',`${result.steady.conservedQuantities}（含迹）`)]:[]),
  ].join(''):'';
  const notes=result?[...(result.method?[`时间演化：${solverText(result.method)}`]:[]),...(result.steady.method?[`长期解：${solverText(result.steady.method)}`]:[]),...(result.steady.reason?[solverText(result.steady.reason)]:[]),...result.assumptions.map(labelStateIds)]:[];
  return `<section class="solver-details"><h4>计算详情</h4>${error?`<p class="solver-error">${esc(solverText(error))}</p>`:''}${result?`<div class="form solver-readouts">${rows}</div>${notes.map(text=>`<p>${esc(text)}</p>`).join('')}`:'<p>没有当前参数的求解结果。</p>'}${failureRecords.length?'<button data-export-solver-failures>导出失败记录</button>':''}</section>`;
}
/** 详情's first section: what is computed now -- the quantization axis, the levels chart's scope,
 * the sampling and the solve; then the definitions and the model, written once. */
function renderDetailsPage():void{
  const a=latestAnalysis,result=currentResult(),scope=resultPage()==='levels'?levelScope():[];
  const axis=a?.fieldAxis,angle=(deg:number)=>`${formatNumber(deg,{places:2})}°`;
  const field=!a?'':a.fieldMagnitudeG>0?`量子化轴：总磁场 |B| = ${withUnit(a.fieldMagnitudeG,'G',{digits:6})}，方向 θ ${angle(axis!.thetaDeg)}、φ ${angle(axis!.phiDeg)}。`
    :scene.fields[0]?`总磁场为零：量子化轴取第一个磁场「${scene.fields[0].name}」的参考轴，θ ${angle(axis!.thetaDeg)}、φ ${angle(axis!.phiDeg)}。`:'没有磁场：量子化轴取 +z。';
  const sampling=result&&result.timesUs.length>1?[populationSamplingInfo(result),'布居演化在模型基中：静态 H0 的本征态，标签为重叠最大的磁场本征态。']:[];
  const markup=`<section class="details-current"><h3>当前计算</h3>${[field,...scope,...sampling].map(line=>`<p>${esc(line)}</p>`).join('')}${solverDetails()}</section>`;
  const host=$('details-current');if(host.dataset.markup!==markup){host.dataset.markup=markup;host.innerHTML=markup;}
  const notes=$('details-static');if(!notes.childElementCount)notes.innerHTML=definitionNotes()+modelNotes();
}

// The levels page's population row: what the cards are coloured by, the scale and the F totals.
const populationMode$=element('select');populationMode$.id='population-mode';populationMode$.setAttribute('aria-label','能级布居着色');
const populationModeFace=comboFace(populationMode$);populationMode$.append(populationModeFace,new Option('长期布居 p̄','steady'),new Option('指定时刻 p(t)','time'));
/** p̄(T)'s time T: ∞ · 1 s · 100 ms (in the toolbar beside the mode). p̄(T) is the population at t = T
 * with only the fast oscillations around T averaged out, not an average over 0–T. Changing T re-solves the mean only. */
const HORIZONS:[string,number|null][]=[['inf',null],['1e6',1e6],['1e5',1e5]];
const meanHorizon=new Segmented(value=>{dynamics().meanHorizonUs=HORIZONS.find(([key])=>key===value)![1];meanHorizon.setCurrent(value);update();});
meanHorizon.setChoices(HORIZONS.map(([key,us])=>[key,horizonText(us)]));meanHorizon.root.setAttribute('aria-label','长期布居的时刻 T');meanHorizon.root.title='p̄(T)：时刻 T 的布居，只平均掉快振荡，不是 0 到 T 的平均';meanHorizon.root.classList.add('mean-horizon');
function horizonText(us:number|null|undefined):string{return us==null?'∞':us===1e6?'1 s':us===1e5?'100 ms':durationText(us);}
/** The name of the long-time population a result holds: p̄ at the horizon it was solved at (the
 * solver's echo; none echoed reads t → ∞). While no converged mean is here (pending, refused or
 * unresolved), the horizon being asked for, never a stale ∞. */
function meanSymbol(result=currentResult()):string{const steady=result?.steady;return `p̄(${horizonText(steady?.converged?steady.horizonUs:resolveMeanHorizon(dynamics()))})`;}
populationMode$.onchange=()=>{populationMode=populationMode$.value as 'steady'|'time';update();};
const populationTime=element('input');populationTime.type='range';populationTime.id='population-time';populationTime.setAttribute('aria-label','布居时间');
populationTime.oninput=()=>{populationTimeIndex=Number(populationTime.value);update();};
const populationTimeValue=element('span','population-time-value');populationTimeValue.id='population-time-value';
const populationLegend=element('span','population-scale');populationLegend.setAttribute('aria-label','固定布居色标 0 至 100%');
populationLegend.append(element('span','scale-end','0'),element('span','scale-bar'),element('span','scale-end','100%'));
const populationTotals=element('span','population-data');populationTotals.id='population-data';
/** Shown in place of the totals when p(t) cannot colour the chart. */
const populationNote=element('span','population-note');
/** The coloured populations' estimated error when it exceeds the tolerance: they are shown, marked. */
const populationAccuracy=element('span','population-accuracy');populationAccuracy.hidden=true;
$('population-toolbar').append(populationMode$,meanHorizon.root,populationTime,populationTimeValue,populationAccuracy,populationLegend,populationTotals,populationNote);
function populationToolbar():void{
  const snap=populationSnapshot(),result=currentResult(),p=snap.populations;
  if(populationMode$.value!==populationMode)populationMode$.value=populationMode;
  meanHorizon.setCurrent(HORIZONS.find(([,us])=>us===resolveMeanHorizon(dynamics()))![0]);
  const timed=populationMode==='time'&&!!result;populationTime.hidden=!timed;populationTimeValue.hidden=!timed;
  if(timed){const max=String(result.timesUs.length-1),value=String(populationTimeIndex<0?result.timesUs.length-1:Math.min(populationTimeIndex,result.timesUs.length-1));if(populationTime.max!==max)populationTime.max=max;if(populationTime.value!==value)populationTime.value=value;populationTimeValue.textContent=durationText(snap.timeUs??0,durationUnit(dynamics()));}
  const sum=(filter:(id:string)=>boolean)=>p?Object.entries(p).reduce((total,[id,value])=>total+(filter(id)?value:0),0):undefined;
  const f1=sum(id=>/^ground-1-/.test(id)),f2=sum(id=>/^ground-2-/.test(id)),pe=sum(id=>!id.startsWith('ground-'));
  // Each total in a slot as wide as 100.00%, so a total of another length moves nothing.
  const markup=`<span class="population-total">F=1 <b class="slot percent">${populationPercent(f1)}</b></span><span class="population-total">F=2 <b class="slot percent">${populationPercent(f2)}</b></span><span class="population-total">激发态 <b class="slot percent">${populationPercent(pe)}</b></span>`;
  if(populationTotals.innerHTML!==markup)populationTotals.innerHTML=markup;
  const uncoloured=!!result&&!p&&populationMode==='time';populationNote.textContent=uncoloured?snap.label:'';populationNote.hidden=!uncoloured;
  const marked=!!p&&snap.outside===true,words=marked?`⚠ 估计误差 ${snap.error==null?'未知':errorPercent(snap.error)}`:'';
  if(populationAccuracy.textContent!==words)populationAccuracy.textContent=words;populationAccuracy.hidden=!marked;
  populationAccuracy.title=marked?`${snap.label} 的估计误差超过 ${errorPercent(POPULATION_TOLERANCE)}；能级仍按算出的布居着色`:'';
}

// ---------------------------------------------------------------- 布居演化
function renderDynamicsControls(a:Analysis):void{
  const d=dynamics(),enabled=scene.lasers.filter(l=>l.enabled&&laserIntensityMWcm2(l)>0);d.durationUnit??=durationUnit(d);if(!enabled.some(l=>l.id===d.laserId))d.laserId=enabled[0]?.id??'';
  const states=allLevels(a);d.monitoredGroupIds??=[];
  if(!states.some(l=>l.id===d.initialStateId))d.initialStateId=initialId();d.monitoredStateIds=d.monitoredStateIds.filter(id=>states.some(l=>l.id===id));
  setText('dynamics-beams',enabled.length?`驱动：${enabled.map(l=>l.name).join(' + ')}`:'驱动：无光束');
  const key=JSON.stringify(states.map(l=>l.id)),parent=$('dynamics-controls');
  if(key===controlsKey){
    refreshFields(dynamicsFields);
    parent.querySelectorAll<HTMLButtonElement>('[data-monitor-group]').forEach(input=>setChecked(input,d.monitoredGroupIds!.includes(input.dataset.monitorGroup!)));
    parent.querySelectorAll<HTMLButtonElement>('[data-monitor]').forEach(input=>setChecked(input,d.monitoredStateIds.includes(input.dataset.monitor!)));
    const count=parent.querySelector('.monitor-count');if(count)count.textContent=`${populationDisplayIds(d).length} 条`;
    monitorAll.disabled=d.monitoredStateIds.length===states.length;monitorNone.disabled=populationDisplayIds(d).length===0;
    return;
  }
  controlsKey=key;const monitorsWereOpen=parent.querySelector<HTMLDetailsElement>('.monitor-controls')?.open??false;disposeTrees(parent);parent.replaceChildren();dynamicsFields.clear();
  const time=form('dynamics-config'),curves=form('dynamics-config');parent.append(time,curves);
  const after=()=>update();
  bindQuantity(dynamicsFields,time,'dynamics','dynamics',()=>({name:'duration',label:'时长',spin:true,inexact:true}),
    {read:()=>durationReading(d),write:(v,u)=>setDuration(d,v,u as DurationUnit),units:()=>unitChoices(DURATION_UNITS,'时间')},after);
  bindField(dynamicsFields,time,'dynamics',()=>({name:'samples',label:'采样点数',kind:'integer',value:d.samples,minimum:2,maximum:2001,step:1}),v=>{const n=Number(v);if(!Number.isInteger(n))throw new Error('采样点数必须为整数');d.samples=n;},{after});
  bindField(dynamicsFields,curves,'dynamics',()=>({name:'monitor-mode',label:'显示曲线',kind:'choice',segmented:true,choices:[['auto','初末各 3 态'],['manual','手动选择']],value:d.monitorMode??'auto'}),v=>{d.monitorMode=v as 'auto'|'manual';if(v==='auto')d.monitoredGroupIds=[];else parent.querySelector<HTMLDetailsElement>('.monitor-controls')!.open=true;},{after});
  const monitors=element('details','monitor-controls');monitors.open=monitorsWereOpen;
  const summary=element('summary');summary.append(element('span','monitor-title','观察曲线'),element('span','monitor-count',`${populationDisplayIds(d).length} 条`),element('span','spacer'),monitorAll.root,monitorNone.root);
  monitors.append(summary);const list=element('div','monitor-tree');monitors.append(list);parent.append(monitors);
  monitorAll.disabled=d.monitoredStateIds.length===states.length;monitorNone.disabled=populationDisplayIds(d).length===0;
  const groupBox=(group:typeof POPULATION_GROUPS[number],label=group.label)=>{const box=checkBox(label,on=>{d.monitorMode='manual';d.monitoredGroupIds=on?[...d.monitoredGroupIds!,group.id]:d.monitoredGroupIds!.filter(id=>id!==group.id);update();});box.dataset.monitorGroup=group.id;box.classList.add('monitor-option');setChecked(box,d.monitoredGroupIds!.includes(group.id));return box;};
  list.append(groupBox(POPULATION_GROUPS.find(g=>g.id==='F:excited')!));
  for(const branch of stateTree(states)){
    const section=element('details','monitor-manifold');section.open=branch.heading==='5S₁/₂';section.append(element('summary','',branch.heading));
    for(const F of branch.children??[]){const rowNode=element('div','monitor-f-row');const ids=F.leaves.map(l=>l.key),group=POPULATION_GROUPS.find(g=>g.members.length===ids.length&&g.members.every(id=>ids.includes(id)))!;rowNode.append(groupBox(group,`${F.heading} 总布居`));const choices=element('div','monitor-list');
      for(const leaf of F.leaves){const box=checkBox(leaf.label,on=>{d.monitorMode='manual';d.monitoredStateIds=on?[...d.monitoredStateIds,leaf.key]:d.monitoredStateIds.filter(id=>id!==leaf.key);update();});box.dataset.monitor=leaf.key;box.classList.add('monitor-option');box.setAttribute('aria-label','观察 '+leaf.text);setChecked(box,d.monitoredStateIds.includes(leaf.key));choices.append(box);}
      rowNode.append(choices);section.append(rowNode);
    }list.append(section);
  }
  requestAnimationFrame(()=>{shareLabelColumn(time);shareLabelColumn(curves);alignUnitPickers(time);});
}
const monitorAll=command('全部磁子能级',()=>{const d=dynamics();d.monitorMode='manual';d.monitoredStateIds=allLevels().map(state=>state.id);update();},{role:'quiet'});monitorAll.root.id='monitor-all';
const monitorNone=command('清空',()=>{const d=dynamics();d.monitorMode='manual';d.monitoredStateIds=[];d.monitoredGroupIds=[];update();},{role:'quiet'});monitorNone.root.id='monitor-none';
function renderDynamics(a:Analysis):void{
  const d=dynamics(),raw=currentResult();if((d.monitorMode??'auto')==='auto')d.monitoredStateIds=recommendedPopulationStates(raw,d.initialStateId);
  renderDynamicsControls(a);const ids=populationDisplayIds(d),result=raw?withPopulationGroups(raw,d.monitoredGroupIds??[]):undefined;
  const labels=Object.fromEntries([...allLevels(a).map(l=>[l.id,stateText(l)]),...POPULATION_GROUPS.map(g=>[g.id,g.label])]);
  // A newer checkpoint (or the final result) redraws the same axes; an open readout stays at its sample.
  const host=$('population-chart'),open=host.querySelector<HTMLElement>('.population-readout:not([hidden])')?.dataset.sampleIndex;
  unbindPopulation?.();unbindPopulation=undefined;
  const plotOptions={showTotalExcited:false,timeUnit:durationUnit(d),scale:populationScale,viewportWidth:host.clientWidth,initialId:singleInitialId(),meanLabel:meanSymbol(raw),...(open===undefined?{}:{cursor:Number(open)})};
  const stopped=stoppedKey&&stoppedKey===solveKey();
  host.innerHTML=result&&result.timesUs.length>1?populationChart(result,ids,labels,plotOptions):result?.steady.converged?`<div class="empty-state"><div><p>长期布居 ${esc(meanSymbol(raw))}</p><div class="form solver-readouts">${ids.map(id=>`<span class="field-label">${esc(labels[id]??id)}</span><span class="readout-edit">${populationValue(result.steady.populations?.[id]??0)}</span>`).join('')}</div><p class="note">${currentSolverError()?'时间演化未完成':stopped?'时间演化已停止':'时间演化计算中…'}</p></div></div>`:`<div class="empty-state">${currentSolverError()?esc(solverText(currentSolverError())):stopped?'计算已停止':'正在求解当前场景…'}</div>`;
  if(result&&result.timesUs.length>1)unbindPopulation=bindPopulationChart(host,result,ids,labels,plotOptions);
  const errors=raw&&raw.timesUs.length>1?sampleErrors(raw):undefined,banded=!!errors&&outsideRuns(errors).length>0;
  const legend=(banded?`<span class="population-legend accuracy-legend"><span class="accuracy-swatch" aria-hidden="true"></span>估计误差 &gt; ${errorPercent(POPULATION_TOLERANCE)}</span>`:'')+ids.map(id=>`<span class="population-legend"><svg width="18" height="5" aria-hidden="true"><line x1="0" y1="2.5" x2="18" y2="2.5" stroke="${populationColor(id)}" stroke-width="2"/></svg>${esc(labels[id]??id)}${id===singleInitialId()?'<span class="initial-tag">初态</span>':''}</span>`).join('');
  if($('population-caption').innerHTML!==legend)$('population-caption').innerHTML=legend;
}

/** The 沿 B shortcut the selected light's direction is now wears the active face. */
function refreshDirectionStates():void{
  const current=selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):selection.kind==='field'?scene.fields.find(f=>f.id===selection.id):undefined;if(!current)return;
  if(actualDirection.isConnected)refreshDirection(current);
  const target=shortcutField(scene.fields,current.direction);
  for(const {command:action,mode} of directionCommands){
    action.disabled=!target;action.title=target?`链接到「${target.name}」的参考轴`:'没有磁场';
    action.setActive(!!target&&sameDirection(current.direction,shortcutDirection(target,mode)));
  }
}

/** The levels chart's tag: what the shown shift includes -- B and the lasers shown -- and, when a
 * shown laser is too close to resonance for a static shift, that warning alone. */
function shiftTag(a:Analysis):{tag:string;warning:string}{
  const background=a.spectroscopy?.backgroundLaserIds??[],names=background.map(id=>scene.lasers.find(l=>l.id===id)?.name??id);
  const tag=[...(a.fieldMagnitudeG>0?['B']:[]),...names].join(' + ')||'零场';
  if(a.spectroscopy?.status!=='unresolved')return {tag,warning:''};
  const close=background.map(id=>scene.lasers.find(l=>l.id===id)).filter((l):l is Laser=>!!l&&mixingOf(l)?.tooClose===true).map(l=>l.name);
  return {tag,warning:`光移未定义${close.length?`：${close.join('、')}离共振太近`:''}`};
}
/** What the levels chart is showing now: its model scope and warnings (详情). */
function levelScope():string[]{
  const a=latestAnalysis;if(!a)return [];
  return [a.spectroscopy?.modelScope,...(a.spectroscopy?.warnings??[])].filter((line):line is string=>Boolean(line)).flatMap(line=>line.split('\n'));
}
/** 方向与偏振's legend: |B| and each field, every beam in its colour; the selected beam's ∠(k, B)
 * and σ⁻/π/σ⁺ (as in the inspector), each number in a slot as wide as its longest value. */
function renderSpaceLegend(a:Analysis):void{
  const fields=scene.fields.filter(f=>f.enabled&&f.magnitudeG>0),zero=a.fieldMagnitudeG===0;
  const swatch=(colour:string,thick=false)=>`<span class="legend-swatch${thick?' thick':''}" style="background:${colour}"></span>`;
  const isSelected=(kind:'field'|'laser',id:string)=>selection.kind===kind&&selection.id===id;
  const items=[`<span class="legend-item total">${swatch('var(--field-ink)',true)}<span>${fields.length>1?'合成 B':'B'} <b class="slot field">${zero?'0 G':withUnit(a.fieldMagnitudeG,'G',{digits:4})}</b></span></span>`];
  if(fields.length>1)for(const f of fields)items.push(`<button class="quiet legend-item${isSelected('field',f.id)?' selected':''}" data-field-id="${esc(f.id)}">${swatch('var(--field-ink)')}<span>${esc(f.name)} <b class="slot field">${withUnit(f.magnitudeG,'G',{digits:4})}</b></span></button>`);
  for(const l of scene.lasers.filter(l=>l.enabled)){
    const p=a.polarizations[l.id],chosen=isSelected('laser',l.id),detail=chosen&&p?` · ∠(k, B) <b class="slot angle">${formatNumber(p.angleToBDeg,{places:1})}°</b> · σ⁻ <b class="slot percent">${populationPercent(p.minus)}</b> · π <b class="slot percent">${populationPercent(p.pi)}</b> · σ⁺ <b class="slot percent">${populationPercent(p.plus)}</b>`:'';

    items.push(`<button class="quiet legend-item${chosen?' selected':''}" data-laser-id="${esc(l.id)}">${swatch(laserColour(scene,l.id))}<span>${esc(l.name)}${detail}</span></button>`);
  }
  const markup=items.join('');if($('space-legend').innerHTML!==markup)$('space-legend').innerHTML=markup;
}
function showAnalysisError(text:string):void{
  const markup=`<div class="empty-state error" role="alert">能级参数无效：${esc(text)}</div>`;
  if(view==='levels'){$('levels-chart').innerHTML=markup;$('selection-readout').hidden=true;$('population-toolbar').hidden=true;$('levels-details').replaceChildren();}
  else if(view==='dynamics'){unbindPopulation?.();unbindPopulation=undefined;$('population-chart').innerHTML=markup;$('population-caption').replaceChildren();}
  else if(view==='space'){spaceMount.remove();$('space-chart').innerHTML=markup;}
}

function renderResults(allowSolve=true):void{
  const focused=document.activeElement?.closest<HTMLElement>('[data-state],[data-laser-id],[data-transition]'),levelsChart$=$('levels-chart'),levelsDetails=$('levels-details');
  const focusHost=focused&&(levelsChart$.contains(focused)?levelsChart$:levelsDetails.contains(focused)?levelsDetails:undefined),focusAttribute=focused&&['data-state','data-laser-id','data-transition'].find(attr=>focused.hasAttribute(attr)),focusValue=focusAttribute?focused?.getAttribute(focusAttribute):undefined;
  if(tabs.currentKey!==view)tabs.setCurrent(view);
  invalidateObsoletePopulation();
  let a:Analysis;
  try{a=cachedAnalysis();latestAnalysis=a;analysisError='';stateLabels=keepLabels(stateLabels,id=>allLevels(a).some(state=>state.id===id)||!!scene.references?.some(ref=>ref.id===id));}
  catch(error){analysisError=message(error);console.error(error);showAnalysisError(analysisError);setText('summary','—');refreshStatus();return;}
  refreshDirectionStates();refreshFields(inspectorFields);
  const laser=activeLaser(view==='levels'?spectral:undefined),pol=laser?a.polarizations[laser.id]:undefined;
  manifoldFocus.setCurrent(spectral);root.dataset.view=view;
  if(view!=='space'&&spaceMount.isConnected)spaceMount.remove();
  if(view!=='dynamics'&&unbindPopulation){unbindPopulation();unbindPopulation=undefined;}
  if($('polarization-bars'))$('polarization-bars').innerHTML=polarizationBars(a,selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):laser);
  if(selection.kind==='laser'){const l=scene.lasers.find(l=>l.id===selection.id);if(l&&laserReadout.readout.root.isConnected){try{laserReadout.show(laserValue(l),scene);}catch{laserReadout.show(undefined,scene);}}if(l&&mixingWarning.isConnected)refreshMixing(l);}
  if(selection.kind==='reference'){const r=scene.referenceFrequencies?.find(r=>r.id===selection.id);if(r&&referenceReadout.readout.root.isConnected){try{referenceReadout.show(anchorValue(objectAnchor(scene,r)),scene);}catch{referenceReadout.show(undefined,scene);}}}
  const field=withUnit(a.fieldMagnitudeG,'G',{digits:4});
  // The header's summary: the field's magnitude last, so a value of another length moves nothing.
  const summary=(...parts:string[])=>setText('summary',parts.join(' · ')),fieldPart=`B ${field}`;
  if(view==='levels'){
    $('selection-readout').hidden=false;$('population-toolbar').hidden=false;
    populationToolbar();
    const initial=referenceStateId(laser),transitions=(a.spectroscopy?.status==='unresolved'?[]:[...incidentTransitions(a,initial,laser?.id,'D1'),...incidentTransitions(a,initial,laser?.id,'D2')]).sort((x,y)=>y.strength*y.weight-x.strength*x.weight),snapshot=populationSnapshot();
    const clockGap=a.spectroscopy?.status==='unresolved'?NaN:(a.ground.find(state=>state.id==='ground-2-0')!.energyMHz-a.ground.find(state=>state.id==='ground-1-0')!.energyMHz);
    summary(`钟跃迁 ${Number.isFinite(clockGap)?withUnit(clockGap/1000,'GHz',{places:6}):'未定义'}`,fieldPart);
    const {tag,warning}=shiftTag(a);setText('shift-tag',tag);$('shift-tag').title='能级图背景中的外场';setText('shift-warning',warning);$('shift-warning').hidden=!warning;
    renderSelection(a,laser);
    levelsChart$.innerHTML=levelsChart(scene,a,laser,{line:spectral,labelled:labelled(stateLabels),initialId:singleInitialId(),populations:snapshot.populations,steadyPopulations:steadyPopulationsInView(a),populationLabel:snapshot.label,viewportWidth:levelsChart$.clientWidth});
    // The cards' hover words say their populations (and an estimated error outside the tolerance) from the start.
    paintLevelPopulations(snapshot);
    const table=transitionTable(transitions);
    renderDetails('levels-details',`<details class="transition-disclosure"><summary>跃迁矩元 <span>${laser?esc(laser.name):spectral}</span></summary><div class="table-scroll"><table class="transition-table"><thead><tr><th>参考态 → 末态</th><th>分量</th><th class="num">${esc(table.heading)}</th><th class="num">Ω/2π / MHz</th><th title="相对 D2 stretched cycling 跃迁的偶极强度，并乘当前偏振权重">耦合强度</th></tr></thead><tbody>${table.rows}</tbody></table></div></details>`);
  }else if(view==='dynamics'){summary(fieldPart);renderDynamics(a);}
  else if(view==='details'){summary(fieldPart);renderDetailsPage();}
  else{
    summary(fieldPart);
    spaceViews.setWithheld('k',!laser||!pol);
    const spaceSelection:SpaceSelection=selection.kind==='field'||selection.kind==='laser'?{kind:selection.kind,id:selection.id}:{kind:'none',id:''};
    const host=$('space-chart');if(!spaceMount.isConnected)host.replaceChildren(spaceMount);if(!spaceRenderer)spaceRenderer=renderSpace3D(spaceMount,scene,a,spaceSelection,{onViewChange:name=>{spaceView=name;spaceViews.setCurrent(name);}});else spaceRenderer.update(scene,a,spaceSelection);
    renderSpaceLegend(a);
  }
  paintStateBadges();renderCount++;renderedPopulationKey=dynamicsKey();root.dataset.populationFresh=String(!!currentResult());refreshStatus();root.dataset.renderCount=String(renderCount);root.dataset.energyRenderedAt=String(performance.now());
  if(focusHost&&focusAttribute&&focusValue)focusHost.querySelector<HTMLElement>(`[${focusAttribute}="${CSS.escape(focusValue)}"]`)?.focus({preventScroll:true});
  if(allowSolve&&view!=='space')requestSolve();
}

// ---------------------------------------------------------------- the scene as a document
function replaceScene(next:Scene,text:string):void{
  loadFailure='';unitDrafts.clear();fieldProblems.clear();solver.cancel();solverRevision++;clearTimeout(solverTimer);requestedKey='';requestedPhysicalKey='';requestedSteadyOnly=false;stoppedKey='';solverError='';solverBusy=false;controlsKey='';displayedResults.clear();
  disposeTrees($('levels-details'));$('levels-details').replaceChildren();$('selection-readout').dataset.selectionKey='';
  normalizeScene(next);scene=next;stateLabels=keepLabels(stateLabels,id=>!id.startsWith('reference-')||!!scene.references?.some(r=>r.id===id));populationTimeIndex=-1;
  selection=scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:scene.fields[0]?{kind:'field',id:scene.fields[0].id}:{kind:'none',id:''};
  renderInitial();renderObjects();renderInspector();markBaseline();update();note(text);
}
/** Replacing a scene that holds edits asks first; nothing else keeps them. */
async function confirmReplace(action:string):Promise<boolean>{
  if(!sceneEdited())return true;
  return confirm('替换当前场景',`当前场景「${scene.name}」有修改。${action}会替换整个场景，且无法撤销；需要保留时请先导出。`,{confirmText:'替换',cancelText:'取消',kind:'warning'});
}
function showExamples():void{
  new Menu([...EXAMPLES.map(example=>({label:example.label,action:()=>void loadExample(example.key)})),'separator',{label:'默认场景',action:()=>void resetScene()}]).openBelow(examplesCommand.root);
}
async function loadExample(key:string):Promise<void>{
  if(!await confirmReplace('加载示例'))return;
  const example=exampleScene(key);view=example.view;if(view==='levels'||view==='dynamics')resultView=view;spectral=example.line;replaceScene(example.scene,`已加载示例「${example.scene.name}」`);
}
async function resetScene():Promise<void>{if(await confirmReplace('恢复默认场景'))replaceScene(defaultScene(),'已恢复默认场景');}
function download(text:string,name:string):void{const url=URL.createObjectURL(new Blob([text],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportScene():void{const name=`${scene.name.replace(/[\\/:*?"<>|]/g,'_').slice(0,100)||'rb87-scene'}.json`;download(JSON.stringify(scene,null,2),name);markBaseline();note(`已导出场景、标记与动力学设置：${name}`);}
function downloadUnreadableScene():void{let original:string|null=null;try{original=localStorage.getItem(`${STORAGE}-unreadable`)??unreadableScene;}catch{original=unreadableScene;}if(original)download(original,'rb87-unreadable-scene.json');}
$<HTMLInputElement>('file-input').onchange=async event=>{
  const input=event.target as HTMLInputElement,file=input.files?.[0];if(!file)return;
  try{if(file.size>100_000)throw new Error('场景文件应小于 100 KB');const next=validateScene(JSON.parse(await file.text()));if(await confirmReplace('导入场景'))replaceScene(next,`已导入场景「${next.name}」`);}
  catch(error){note(`无法导入：${error instanceof Error?error.message:'场景无效'}`,'error');}
  finally{input.value='';}
};
/** A name edited in place (inline-name.ts): the scene's, 1 to 100 characters. */
function renameScene(name:string):string|void{
  if(!name||name.length>100){const problem='场景名称应包含 1 至 100 个字符';note(problem,'warning');return problem;}
  scene.name=name;renderObjects();persist();refreshStatus();
}
function selectedObject():{name:string}|undefined{const lists={field:scene.fields,laser:scene.lasers,reference:scene.referenceFrequencies??[],level:scene.references??[]};return selection.kind==='none'?undefined:(lists[selection.kind] as {id:string;name:string}[]).find(o=>o.id===selection.id);}
/** The selected object's name, 1 to 64 characters. */
function renameSelected(name:string):string|void{
  const object=selectedObject();if(!object)return;
  if(!name||name.length>64){const problem='名称应包含 1 至 64 个字符';note(problem,'warning');return problem;}
  object.name=name;renderObjects();update();
}

// ---------------------------------------------------------------- layout: CSS decides, the page asks it
function layoutMode():'wide'|'rail'|'narrow'{const mode=getComputedStyle(document.documentElement).getPropertyValue('--atom-layout').trim();return mode==='rail'||mode==='narrow'?mode:'wide';}
function syncLayout():void{
  const mode=layoutMode();
  if(mode!=='narrow')root.classList.remove('inspector-open');
  if(mode==='wide')root.classList.remove('scene-open');
  const compact=mode==='wide'?root.classList.contains('scene-collapsed'):!root.classList.contains('scene-open');
  root.classList.toggle('scene-compact',compact);
  toggleScene.root.setAttribute('aria-expanded',String(!compact));toggleScene.title=compact?'展开场景':'收起场景';toggleScene.root.setAttribute('aria-label',compact?'展开场景':'收起场景');
  inspectorToggle.setActive(root.classList.contains('inspector-open'));inspectorToggle.root.setAttribute('aria-expanded',String(root.classList.contains('inspector-open')));
}
function toggleScenePanel(open?:boolean):void{
  if(layoutMode()==='wide')root.classList.toggle('scene-collapsed',open===undefined?!root.classList.contains('scene-collapsed'):!open);
  else root.classList.toggle('scene-open',open??!root.classList.contains('scene-open'));
  syncLayout();
}
function toggleInspector():void{root.classList.toggle('inspector-open');syncLayout();}

// ---------------------------------------------------------------- events
root.addEventListener('atom-select-laser',event=>{const id=(event as CustomEvent<{id:string}>).detail?.id;if(scene.lasers.some(l=>l.id===id))pick({kind:'laser',id});});root.addEventListener('atom-select-field',event=>{const id=(event as CustomEvent<{id:string}>).detail?.id;if(scene.fields.some(f=>f.id===id))pick({kind:'field',id});});
root.addEventListener('atom-select-none',()=>clearSelection());
/** Empty space: the levels chart's background, and the scene list where no object or control is. */
function emptySpace(target:Element):boolean{
  if(target.closest('#levels-chart'))return !target.closest('[data-state],[data-laser-id],[data-transition]');
  return !!target.closest('.scene-panel')&&!target.closest('.scene-object,button,select,input,label,a,.state-selector,[role=button]');
}
root.addEventListener('click',event=>{
  const target=event.target as Element;
  if(target.closest('[data-export-solver-failures]')){download(JSON.stringify(failureRecords,null,2),'rb87-solver-failures.json');return;}
  if(root.classList.contains('scene-open')&&!target.closest('.scene-panel')){root.classList.remove('scene-open');syncLayout();}
  const badge=target.closest<HTMLElement>('[data-clear-label]');if(badge){applyLabels(withLabel(stateLabels,badge.dataset.clearLabel as LabelSlot,undefined));return;}
  const beam=target.closest<HTMLElement>('[data-laser-id]');if(beam){pick({kind:'laser',id:beam.dataset.laserId!});return;}
  const fieldItem=target.closest<HTMLElement>('[data-field-id]');if(fieldItem){pick({kind:'field',id:fieldItem.dataset.fieldId!});return;}
  const state=target.closest<HTMLElement>('[data-state]');if(state){selectState(state.dataset.state!);return;}
  const tr=target.closest<HTMLElement>('[data-transition]');if(tr){const t=latestAnalysis.transitions.find(t=>t.id===tr.dataset.transition);if(t){const endpoints=transitionEndpoints(t,referenceStateId(scene.lasers.find(l=>l.id===t.laserId)));if(endpoints)applyLabels({first:endpoints.fromId,second:endpoints.toId});}return;}
  if(emptySpace(target))clearSelection();
});
root.addEventListener('keydown',event=>{
  // Escape closes only the topmost overlay; an edit's own Escape (v3 Field) never reaches here.
  if(event.key==='Escape'&&!event.defaultPrevented&&!document.querySelector('dialog[open]')){
    if(root.classList.contains('inspector-open')){root.classList.remove('inspector-open');syncLayout();inspectorToggle.root.focus();event.preventDefault();return;}
    if(root.classList.contains('scene-open')){root.classList.remove('scene-open');syncLayout();toggleScene.root.focus();event.preventDefault();return;}
  }
  if(event.key!=='Enter'&&event.key!==' ')return;
  const el=(event.target as Element).closest<HTMLElement>('g[data-state],g[data-laser-id],tr[data-transition],[data-clear-label]');if(el){event.preventDefault();el.dispatchEvent(new MouseEvent('click',{bubbles:true}));}

});
window.addEventListener('pagehide',()=>{if(saveState==='pending')saveSceneNow();cancelObsoleteSolve();});
window.addEventListener('pageshow',event=>{if(event.persisted)renderResults();});
window.addEventListener('resize',syncLayout);
let chartWidth=0,resizeFrame=0;new ResizeObserver(entries=>{const width=Math.round(entries[0]?.contentRect.width??0);if(width!==chartWidth){chartWidth=width;cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(latestAnalysis&&(view==='levels'||view==='dynamics'))renderResults(false);});}}).observe(tabs.stack);

// ---------------------------------------------------------------- start
normalizeScene(scene);if(baseline===null&&!unreadableScene&&!loadFailure){let stored:string|null=null;try{stored=localStorage.getItem(STORAGE);}catch{}if(stored===null)markBaseline();}
syncLayout();latestAnalysis=analyze(scene);renderInitial();renderObjects();renderInspector();renderResults();
if(loadFailure)refreshStatus();
