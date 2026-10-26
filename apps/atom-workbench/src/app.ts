// Atom Workbench: one v3 window -- the shared title bar, one header card, one
// status strip, the scene list, a tab widget of the three analysis pages and
// the inspector of the selected field or laser.
import {SpinBox} from '../../desktop/ui/src/fluent/spin.js';
import {Field,form,row as fieldRow,sectionTitle,shareLabelColumn,type FieldSpec} from '../../desktop/ui/src/fluent/form.js';
import {Dialog,confirm} from '../../desktop/ui/src/fluent/dialog.js';
import {command,type Command} from '../../desktop/ui/src/fluent/button.js';
import {element,reconcileChildren,type Value} from '../../desktop/ui/src/fluent/dom.js';
import {comboFace} from '../../desktop/ui/src/fluent/combo.js';
import {TreeCombo} from '../../desktop/ui/src/fluent/tree_combo.js';
import {StatusStrip,setState,statusDot,type StatusWord} from '../../desktop/ui/src/fluent/status.js';
import {TabWidget} from '../../desktop/ui/src/fluent/tabs.js';
import {Menu} from '../../desktop/ui/src/fluent/menu.js';
import {setMiddle} from '../../desktop/ui/src/fluent/elide.js';
import {drawBadges} from '../../desktop/ui/src/pulse/widgets.js';
import {Segmented,setChecked,switchButton,checkBox} from '../../desktop/ui/src/fluent/toggles.js';
import {comparisonPair,chooseComparisonFrom,chooseComparisonTo,swapComparison,clickComparison,type ComparisonChoice} from './transition-choice.js';
import {referenceFromLaser} from './reference-draft.js';
import {frequencyBaseTree,stateTree} from './choice-trees.js';
import {alignLevelReadouts} from './readout-alignment.js';
import {DETUNING_UNIT_MHZ,detuningInUnit,laserReferenceTransition,naturalDetuningUnit,normalizeLaserFrequency,setLaserDetuningUnit,setLaserDetuningValue,setLaserFrequencyBase,setLaserWavelengthNm,zeroFieldTransitionOptions} from './laser-frequency.js';
import {FREQUENCY_UNITS,WAVELENGTH_UNITS,UNIT_NM,baseValue,difference,gap,inUnit,isWavelengthUnit,laserValue,levelValue,nearestTransition,parseBase,photon,readingHeading,readoutText,referenceLevelValue,transitionKey,type FrequencyValue,type WavelengthUnit} from './frequency-value.js';
import {FrequencyReadout,ReadoutPreferences} from './frequency-controls.js';
import {analyze,dipoleBetweenStates,laserIntensityMWcm2,laserDirectionAxis,setLaserDirectionAxis,relativeDirectionAxis,setPolarizationPreset,composeMagneticField,type PolarizationPreset} from './physics.js';
import {normalizeLaserIntensity,setLaserIntensityMode,setLaserLocalIntensity,setLaserPowerMW,setLaserWaistUm} from './laser-intensity.js';
import {dynamicsCacheKey} from './dynamics-cache-key.js';
import {CurrentDynamicsResults} from './current-dynamics-results.js';
import {spectroscopyAnalysis,spectroscopyCoupling,projectSpectroscopyPopulations} from './spectroscopy.js';
import {defaultDynamics,defaultScene,DEFAULT_TWEEZER_WAVELENGTH_NM,MAX_POPULATION_TOLERANCE,resolvePopulationTolerance} from './types.js';
import type {Analysis,AxisAngles,DynamicsSettings,DetuningUnit,Laser,Level,Scene,SpectralLine,Transition} from './types.js';
import {isTimeAverage,type DynamicsResult} from './solver-types.js';
import {DynamicsClient,type SolverFailureRecord} from './solver-client.js';
import {validateScene} from './scene-file.js';
import {incidentTransitions,laserEmits,laserReferenceStateId,setLaserReferenceState,transitionEndpoints} from './transition-view.js';
import {escapeHtml as esc,levelsChart,number,populationChart,bindPopulationChart,populationColor,populationValue,populationSamplingInfo} from './charts.js';
import {math} from './math.js';
import {renderSpace3D,type SpaceSelection,type SpaceViewName} from './space3d.js';
import {laserColour} from './identity.js';
import {POPULATION_GROUPS,recommendedPopulationStates,populationDisplayIds,withPopulationGroups} from './population-selection.js';
import {durationUnit,durationValue,setDurationUnit,setDurationValue,durationText,timeDisplayUnit,DURATION_UNITS} from './duration.js';
import type {DurationUnit} from './types.js';
import {patchLevelPopulations} from './level-population-dom.js';
import {installWindowChrome} from './window-chrome.js';
import {formatElapsed,formatNumber,labelStateIds,stateLabel,subscript,withUnit} from './format.js';
import {solverText,steadyStatusText} from './solver-words.js';
import {modelNotes,type ModelSection} from './model-notes.js';
import {EXAMPLES,exampleScene} from './examples.js';

installWindowChrome();

type View='levels'|'dynamics'|'space';
type Selection={kind:'field'|'laser'|'none';id:string};
const APP_TITLE='Rb87 实验工作台';
const STORAGE='rb87-workbench-demo-v4',root=document.querySelector<HTMLDivElement>('#app')!;
let scene=defaultScene(),view:View='levels',spectral:SpectralLine='D2',selection:Selection={kind:'laser',id:'probe'};
let populationMode:'steady'|'time'='steady',populationTimeIndex=-1,populationScale:'linear'|'log'='linear';let unbindPopulation:(()=>void)|undefined;
let spaceRenderer:ReturnType<typeof renderSpace3D>|undefined;const spaceMount=document.createElement('div');spaceMount.className='retained-space-view';
let referenceUnit:DetuningUnit='MHz',referenceEditingId:string|null=null,referenceEditorNode:HTMLElement|undefined,referenceNumber:{readMHz:()=>number;setMHz:(value:number)=>void}|undefined;
let saveState:'saved'|'pending'|'failed'='saved',saveFailure='',loadFailure='',unreadableScene:string|null=null;
let comparisonSource:string|undefined;let selectedStates:string[]=[],dirty=false,saveTimer=0,renderCount=0;
let latestAnalysis:Analysis,analysisError='',solverError='',solverBusy=false,solverQueued=false,solverRevision=0,requestedKey='',solverTimer=0,solverWallMs=0,controlsKey='',renderedPopulationKey='';
let requestedPhysicalKey='',requestedSteadyOnly=false,stoppedKey='';
let notice:{text:string;word:StatusWord;until:number}|undefined;
let failureRecords:SolverFailureRecord[]=[];
try{const saved=JSON.parse(localStorage.getItem(`${STORAGE}-solver-failures`)??'[]');if(Array.isArray(saved))failureRecords=saved.slice(-5);}catch{}
function rememberSolverFailure(record:SolverFailureRecord):void{failureRecords=[...failureRecords.slice(-4),record];try{localStorage.setItem(`${STORAGE}-solver-failures`,JSON.stringify(failureRecords));}catch{}}
const solver=new DynamicsClient({debounceMs:0,onFailure:rememberSolverFailure}),analysisCache=new Map<string,Analysis>();
const treeOwners=new WeakMap<HTMLSelectElement,TreeCombo>();
let initialTree:TreeCombo|undefined;
const directionCommands:{command:Command;kind:'axes'|'relative';key:string}[]=[];
const projectedPopulationCache=new WeakMap<Analysis,WeakMap<object,Record<string,number>|null>>();
let spectrumBuildCount=0,populationPaintCount=0;
type SolvedRecord={result:DynamicsResult;wallMs:number};const displayedResults=new CurrentDynamicsResults<SolvedRecord>();
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const message=(error:unknown)=>error instanceof Error?error.message:String(error);

try{const saved=localStorage.getItem(STORAGE);if(saved){unreadableScene=saved;scene=validateScene(JSON.parse(saved));unreadableScene=null;}}catch(error){loadFailure=message(error);scene.dynamics!.autoRun=false;}
if(!scene.lasers.some(l=>l.id===selection.id))selection=scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:{kind:'field',id:scene.fields[0]?.id??''};
scene.marks??={};scene.references??=[];scene.dynamics??=defaultDynamics();
try{comparisonSource=localStorage.getItem(`${STORAGE}-comparison-source`)||undefined;}catch{}
try{const saved=JSON.parse(localStorage.getItem(`${STORAGE}-selected-states`)??'[]');if(Array.isArray(saved))selectedStates=[...new Set(saved.filter((id:unknown):id is string=>typeof id==='string'))].slice(0,2);}catch{/* Ignore malformed presentation state. */}

// ---------------------------------------------------------------- the window
root.innerHTML=`<div class="workbench">
<div class="workspace">
<aside class="card scene-panel" aria-label="实验场景">
 <div class="title-row panel-heading"><span class="heading">实验场景</span><span class="spacer"></span><span id="toggle-scene-slot"></span></div>
 <div class="title-row object-heading"><span class="heading">磁场</span><span class="object-count" id="field-count"></span><span class="spacer"></span><span id="add-field-slot"></span></div>
 <div id="field-list" class="scene-list"></div>
 <div class="title-row object-heading"><span class="heading">激光</span><span class="object-count" id="laser-count"></span><span class="spacer"></span><span id="add-laser-slot"></span></div>
 <div id="laser-list" class="scene-list"></div>
 <span id="initial-shortcut-slot"></span>
 <div class="state-selector"><div class="title-row object-heading"><span class="heading">初态</span></div><div id="initial-state-tree"></div></div>
</aside>
<aside class="card inspector-panel" aria-label="参数">
 <div class="title-row inspector-heading"><span class="heading" id="inspector-title"></span><span class="spacer"></span><span id="inspector-commands"></span></div>
 <div id="inspector-controls"></div>
</aside>
</div></div><input type="file" id="file-input" accept="application/json,.json" hidden>`;

const workbench=root.querySelector<HTMLElement>('.workbench')!,workspace=root.querySelector<HTMLElement>('.workspace')!;

// The header card: the scene's identity, its muted summary and the file commands (v3 TaskConsole header).
const header=element('header','card header');
const sceneDot=statusDot('idle'),sceneName=element('span','heading document-name');sceneName.id='scene-name';
const renameScene=command('✎',()=>void renameSceneDialog(),{role:'quiet',glyph:true,title:'重命名场景'});renameScene.root.setAttribute('aria-label','重命名场景');
const identity=element('div','identity');identity.append(sceneDot,sceneName,renameScene.root);
const summaryLine=element('span','summary');summaryLine.id='summary';
const examplesCommand=command('示例…',()=>showExamples());examplesCommand.root.id='examples-button';
const importCommand=command('导入…',()=>$<HTMLInputElement>('file-input').click());importCommand.root.id='import-button';
const exportCommand=command('导出…',()=>exportScene());exportCommand.root.id='export-button';
const modelCommand=command('模型说明',()=>openModel('states'));modelCommand.root.id='model-button';
const inspectorToggle=command('参数',()=>toggleInspector(),{role:'quiet'});inspectorToggle.root.classList.add('inspector-toggle');inspectorToggle.root.setAttribute('aria-controls','inspector-controls');
header.append(identity,summaryLine,inspectorToggle.root,examplesCommand.root,importCommand.root,exportCommand.root,modelCommand.root);

// The status strip: save and solver state, Stop while a solve runs (v3 FluentStatusStrip).
const strip=new StatusStrip({action:['停止',()=>stopSolve()]});strip.root.id='status';strip.text.id='status-text';
const detailsCommand=command('详情',()=>showSolverDetails(),{role:'quiet'});detailsCommand.hidden=true;
const recoverCommand=command('下载原本地场景',()=>downloadUnreadableScene());recoverCommand.hidden=true;
strip.trailing.prepend(detailsCommand.root,recoverCommand.root);
workbench.prepend(header,strip.root);

// The three analysis pages (v3 FluentTabWidget); each page's own controls on its title row.
const tabs=new TabWidget();tabs.root.classList.add('main-tabs');
const page=(key:View,markup:string)=>{const node=element('div',`atom-page ${key}-page`);node.innerHTML=markup;return node;};
const levelsPage=page('levels',`<div class="title-row page-title"><span class="heading" id="levels-heading"></span><span id="beam-chip-slot"></span><span class="summary" id="levels-badge"></span><span id="levels-commands"></span></div>
<p class="note wrap levels-legend" id="levels-legend"></p><div class="population-toolbar" id="population-toolbar"></div><div class="chart-surface" id="levels-chart"></div>
<div id="selection-readout" class="state-selection-panel"></div><section class="details-card" id="levels-details"></section>`);
const dynamicsPage=page('dynamics',`<div class="title-row page-title"><span class="summary" id="dynamics-beams"></span><span id="dynamics-commands"></span></div>
<div id="dynamics-controls" class="dynamics-controls"></div><div class="chart-surface" id="population-chart"></div><div class="chart-caption" id="population-caption"></div>
<section class="details-card" id="dynamics-details"></section>`);
const spacePage=page('space',`<div class="title-row page-title space-title"><span id="space-commands"></span></div><div class="chart-surface space-surface" id="space-chart"></div><div class="space-legend" id="space-legend"></div>`);
tabs.addPermanent('levels','能级与跃迁',levelsPage);tabs.addPermanent('dynamics','布居演化',dynamicsPage);tabs.addPermanent('space','方向与偏振',spacePage);
// TabWidget sizes a label by a canvas measure of its words; for CJK the canvas falls back to a
// narrower face than the page does, so the labels keep their own width and nothing is cut.
(['levels','dynamics','space'] as const).forEach((key,i)=>{const tab=tabs.strip.children[i] as HTMLElement|undefined;if(!tab)return;tab.dataset.view=key;tab.querySelector<HTMLElement>('.tab-label')?.style.removeProperty('width');});
tabs.setCurrent(view);
tabs.onChange=key=>{if(view!==key){view=key as View;update();}};
workspace.querySelector('.scene-panel')!.after(tabs.root);

// Per-page commands.
const manifoldFocus=new Segmented(value=>{spectral=value as SpectralLine;manifoldFocus.setCurrent(spectral);renderResults(false);});
manifoldFocus.setChoices([['D2','D2 · 5P₃/₂'],['D1','D1 · 5P₁/₂']]);manifoldFocus.setCurrent(spectral);manifoldFocus.root.id='line-tabs';manifoldFocus.root.setAttribute('aria-label','能级关注流形');
const levelsInfo=command('ⓘ',()=>openModel('figure',levelScope()),{role:'quiet',glyph:true,title:'模型说明'});levelsInfo.root.setAttribute('aria-label','能级图模型说明');
$('levels-commands').replaceWith(manifoldFocus.root,levelsInfo.root);
const populationScaleControl=new Segmented(value=>{populationScale=value as 'linear'|'log';populationScaleControl.setCurrent(populationScale);update();});
populationScaleControl.setChoices([['linear','线性'],['log','对数']]);populationScaleControl.setCurrent(populationScale);populationScaleControl.root.setAttribute('aria-label','布居坐标');
const runDynamics=command('计算',()=>requestSolve(true));runDynamics.root.id='run-dynamics';
const dynamicsInfo=command('ⓘ',()=>openModel('population',samplingScope()),{role:'quiet',glyph:true,title:'模型说明'});dynamicsInfo.root.setAttribute('aria-label','布居演化模型说明');
$('dynamics-commands').replaceWith(populationScaleControl.root,runDynamics.root,dynamicsInfo.root);
// 方向与偏振: where the camera looks, the field animation, the reference plane and the zoom.
let spaceView:SpaceViewName='free';
const spaceViews=new Segmented(value=>{spaceView=value as SpaceViewName;spaceViews.setCurrent(spaceView);spaceRenderer?.setView(spaceView);});
spaceViews.setChoices([['free','3D'],['x','沿 x'],['y','沿 y'],['z','沿 z'],['B','沿 B'],['k','沿 k']]);spaceViews.setCurrent(spaceView);spaceViews.root.setAttribute('aria-label','观察方向');
const spaceAnimation=switchButton(on=>{setChecked(spaceAnimation,on);spaceRenderer?.setAnimation(on);},'电场动画');spaceAnimation.classList.add('row-switch');
const spacePlane=checkBox('xy 参考面',on=>spaceRenderer?.setPlane(on));setChecked(spacePlane,true);
const zoomOut=command('−',()=>spaceRenderer?.zoomBy(1.14),{role:'quiet',glyph:true,title:'缩小'}),zoomIn=command('＋',()=>spaceRenderer?.zoomBy(.88),{role:'quiet',glyph:true,title:'放大'}),resetView=command('⟲',()=>spaceRenderer?.reset(),{role:'quiet',glyph:true,title:'复位视角'});
zoomOut.root.setAttribute('aria-label','缩小');zoomIn.root.setAttribute('aria-label','放大');resetView.root.setAttribute('aria-label','复位视角');
$('space-commands').replaceWith(spaceViews.root,spaceAnimation,spacePlane,element('span','spacer'),zoomOut.root,zoomIn.root,resetView.root);
// The selected laser of the level diagram, and the one way to clear it.
const beamChip=element('span','beam-chip'),beamChipName=element('span','beam-chip-name');
const clearBeam=command('×',()=>clearSourceSelection(),{role:'quiet',glyph:true,title:'取消选择光束'});clearBeam.root.setAttribute('aria-label','取消选择光束');
beamChip.append(beamChipName,clearBeam.root);$('beam-chip-slot').replaceWith(beamChip);

// Scene panel commands.
const toggleScene=command('',()=>toggleScenePanel(),{role:'quiet',square:true});toggleScene.root.id='toggle-scene';toggleScene.root.classList.add('scene-collapse');
toggleScene.root.append(element('span','scene-chevron'));$('toggle-scene-slot').replaceWith(toggleScene.root);
const addField=command('＋',()=>addObject('field'),{role:'quiet',glyph:true,title:'添加磁场'});addField.root.id='add-field';addField.root.setAttribute('aria-label','添加磁场');$('add-field-slot').replaceWith(addField.root);
const addLaser=command('＋',()=>addObject('laser'),{role:'quiet',glyph:true,title:'添加激光'});addLaser.root.id='add-laser';addLaser.root.setAttribute('aria-label','添加激光');$('add-laser-slot').replaceWith(addLaser.root);
const initialShortcut=command('ψ₀',()=>toggleScenePanel(true),{role:'quiet',square:true});initialShortcut.root.classList.add('scene-initial-shortcut');initialShortcut.root.setAttribute('aria-label','选择初态');$('initial-shortcut-slot').replaceWith(initialShortcut.root);
// Inspector commands.
const renameObject=command('✎',()=>void renameSelected(),{role:'quiet',glyph:true,title:'重命名'});renameObject.root.setAttribute('aria-label','重命名对象');
const closeInspector=command('×',()=>{root.classList.remove('inspector-open');syncLayout();inspectorToggle.root.focus();},{role:'quiet',glyph:true,title:'关闭参数'});closeInspector.root.classList.add('inspector-dismiss');closeInspector.root.setAttribute('aria-label','关闭参数');
$('inspector-commands').replaceWith(renameObject.root,closeInspector.root);

// ---------------------------------------------------------------- helpers
function setText(id:string,value:string):void{const node=$(id);if(node.textContent!==value)node.textContent=value;}
function populationPercent(p:number|undefined):string{return p===undefined?'—':populationValue(p);}
function initialId():string{return scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`;}
function allLevels(a=latestAnalysis):Level[]{return [...a.ground,...a.D1,...a.D2];}
function getLevel(id:string,a=latestAnalysis):Level|undefined{return allLevels(a).find(l=>l.id===id);}
function stateText(l:Level):string{return stateLabel(l);}
function measuredNumber(value:number,places=4):string{if(value!==0&&Number.isFinite(value)&&Math.abs(value)<10**(-places))return formatNumber(value,{digits:3});return number(value,places);}
function frequencyText(valueMHz:number):string{if(!Number.isFinite(valueMHz))return '—';const unit=naturalDetuningUnit(valueMHz);return `${readoutText(inUnit(valueMHz,unit),false)} ${unit}`;}
function stateCell(id:string):string{return `<span class="state-label">${esc(idText(id))}</span>`;}
function idText(id:string):string{const l=getLevel(id);return l?stateText(l):scene.references?.find(r=>r.id===id)?.name??id;}
function activeLaser(_line?:SpectralLine):Laser|undefined{return selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):undefined;}
function referenceStateId(laser=activeLaser()):string{return laserReferenceStateId(scene,laser);}
function dynamics():DynamicsSettings{const d=scene.dynamics??(scene.dynamics=defaultDynamics());d.populationTolerance=resolvePopulationTolerance(d.populationTolerance);return d;}
function normalizeScene(next:Scene):void{for(const laser of next.lasers){try{normalizeLaserFrequency(laser);normalizeLaserIntensity(laser);}catch{/* The inspector reports an invalid laser. */}}}

// Edits since the scene was loaded: derived anchors and automatic curve choices are not edits.
let baseline:string|null=null;
try{baseline=localStorage.getItem(`${STORAGE}-baseline`);}catch{}
function editKey(value:Scene=scene):string{
  // The display unit, derived anchors and the automatic curve choice are not edits.
  const d=value.dynamics,settings=d?{initialStateId:d.initialStateId,durationUs:d.durationUs,samples:d.samples,populationTolerance:d.populationTolerance,
    curves:d.monitorMode==='manual'?[d.monitoredStateIds,d.monitoredGroupIds??[]]:'auto'}:null;
  return JSON.stringify({...value,dynamics:settings},(key,item)=>key.endsWith('Anchor')?undefined:item);
}
function markBaseline():void{baseline=editKey();try{localStorage.setItem(`${STORAGE}-baseline`,baseline);}catch{}}
function sceneEdited():boolean{return baseline===null||baseline!==editKey();}

function saveSceneNow():void{clearTimeout(saveTimer);try{if(unreadableScene!==null){localStorage.setItem(`${STORAGE}-unreadable`,unreadableScene);unreadableScene=null;}localStorage.setItem(STORAGE,JSON.stringify(scene));localStorage.setItem(`${STORAGE}-selected-states`,JSON.stringify(selectedStates));localStorage.setItem(`${STORAGE}-comparison-source`,comparisonSource??'');saveState='saved';saveFailure='';}catch(error){saveState='failed';saveFailure=message(error);}refreshStatus();}
function persist():void{saveState='pending';clearTimeout(saveTimer);saveTimer=window.setTimeout(saveSceneNow,200);}
function update():void{persist();invalidateObsoletePopulation();if(dirty)return;dirty=true;requestAnimationFrame(()=>{dirty=false;renderResults();});}
function note(text:string,word:StatusWord='task'):void{notice={text,word,until:performance.now()+6000};refreshStatus();window.setTimeout(refreshStatus,6100);}

// ---------------------------------------------------------------- status
function solverStatus():[string,StatusWord]{
  const result=currentResult(),error=currentSolverError();
  if(error)return [result?.steady.converged?`时间演化失败（长期布居已就绪）：${labelStateIds(solverText(error))}`:`布居计算失败：${labelStateIds(solverText(error))}`,'error'];
  if(currentSolverBusy())return [result?.steady.converged?'长期布居已就绪 · 时间演化计算中…':solverQueued?'布居待计算':'布居计算中…','task'];
  const stopped=stoppedKey!==''&&stoppedKey===dynamicsKey();
  if(view==='space')return ['方向与偏振实时显示','idle'];
  if(!result)return stopped?['计算已停止；按“计算”重新开始','warning']:['布居待计算','idle'];
  const elapsed=formatElapsed(currentRecord()?.wallMs??solverWallMs);
  if(view==='dynamics'&&result.timesUs.length>1)return [result.steady?.converged?`轨迹已计算 · ${elapsed}`:`轨迹已计算 · 长期布居未解析 · ${elapsed}`,result.steady?.converged?'idle':'warning'];
  if(view==='dynamics')return stopped?[`计算已停止：${steadyStatusText(result.steady?.status)}已就绪，时间演化未计算；按“计算”重新开始`,'warning']:[result.steady?.converged?`${steadyStatusText(result.steady.status)}已收敛 · 时间演化待计算`:'长期布居未解析','idle'];
  return result.steady?.converged?[`${steadyStatusText(result.steady.status)}已收敛 · ${elapsed}`,'idle']:['长期布居未解析','warning'];
}
function refreshStatus():void{
  if(notice&&performance.now()>notice.until)notice=undefined;
  let [text,word]=solverStatus();
  if(analysisError)[text,word]=[`能级参数无效：${analysisError}`,'error'];
  if(notice)[text,word]=[notice.text,notice.word];
  if(saveState==='failed')[text,word]=[`本地保存失败：${saveFailure}；请导出场景`,'error'];
  else if(loadFailure&&!notice)[text,word]=[`原本地场景无法载入：${loadFailure}。已显示默认场景，原数据已保留`,'warning'];
  if(word==='idle')text+=saveState==='pending'?' · 正在保存…':' · 场景已保存';
  strip.show(text,word);strip.setActionVisible(currentSolverBusy());
  detailsCommand.hidden=!currentSolverError();recoverCommand.hidden=!loadFailure;
  const sceneWord:StatusWord=saveState==='failed'?'error':loadFailure?'warning':'idle';setState(sceneDot,sceneWord);
  sceneDot.title=saveState==='failed'?'本地保存失败':loadFailure?'原本地场景未载入':'场景已保存在本机';
}

// ---------------------------------------------------------------- selection
function select(next:Selection):void{
  const focusedObject=document.activeElement?.closest<HTMLElement>('[data-object-id]'),focusId=focusedObject?.dataset.objectId;
  selection=next;
  if(layoutMode()==='narrow'&&selection.kind!=='none')root.classList.add('inspector-open');
  if(selection.kind==='none')root.classList.remove('inspector-open');
  root.classList.remove('scene-open');syncLayout();renderObjects();renderInspector();renderResults(false);
  if(focusId)root.querySelector<HTMLButtonElement>(`[data-object-id="${CSS.escape(focusId)}"]`)?.focus({preventScroll:true});
}
function clearSourceSelection():void{if(selection.kind!=='none')select({kind:'none',id:''});}
function comparisonChoice():ComparisonChoice{return {selected:selectedStates,from:comparisonSource};}
function applyComparison(choice:ComparisonChoice):void{selectedStates=choice.selected;comparisonSource=choice.from;persist();renderResults(false);}
function markedComparison(l:Laser|undefined):string[]{const p=comparisonPair(comparisonChoice(),referenceStateId(l));return p.to?[p.from,p.to]:comparisonSource?[comparisonSource]:[];}
function selectState(id:string):void{applyComparison(clickComparison(comparisonChoice(),id));}

// ---------------------------------------------------------------- scene list
function item(kind:'field'|'laser',id:string,index:number,name:string,sub:string,enabled:boolean,onToggle:(v:boolean)=>void):HTMLElement{
  const div=element('div',`scene-object${selection.kind===kind&&selection.id===id?' selected':''}${enabled?'':' disabled'}`);
  const button=element('button','object-select');button.type='button';button.dataset.objectId=id;button.dataset.objectKind=kind;
  button.setAttribute('aria-label',`编辑${name}`);button.setAttribute('aria-pressed',String(selection.kind===kind&&selection.id===id));
  const symbol=element('span',`object-symbol ${kind}`);symbol.append(kind==='field'?'B':'λ',element('span','object-index',subscript(index)));if(kind==='laser')symbol.style.color=laserColour(scene,id);
  const copy=element('span','object-copy'),title=element('span','object-name'),subtitle=element('span','object-detail');
  setMiddle(title,name,'cut');setMiddle(subtitle,sub,'cut');copy.append(title,subtitle);button.append(symbol,copy);
  // In the collapsed rail only the symbol shows: the button says what it stands for.
  button.addEventListener('pointerenter',()=>{button.title=copy.offsetParent===null?`${name} · ${sub}`:'';});
  button.onclick=()=>select({kind,id});
  const toggle=switchButton(next=>{onToggle(next);setChecked(toggle,next);div.classList.toggle('disabled',!next);update();});
  setChecked(toggle,enabled);toggle.classList.add('object-toggle');toggle.setAttribute('aria-label',`启用${name}`);
  const remove=command('×',()=>void removeObject(kind,id),{role:'destructive',glyph:true,square:true,title:`删除${name}`}).root;
  remove.classList.add('remove-object');remove.setAttribute('aria-label',`删除${name}`);
  div.append(button,toggle,remove);return div;
}
async function removeObject(kind:'field'|'laser',id:string):Promise<void>{
  const object=(kind==='field'?scene.fields:scene.lasers).find(o=>o.id===id);if(!object)return;
  const noun=kind==='field'?'磁场':'激光';
  if(!await confirm(`删除${noun}`,`删除${noun}「${object.name}」？它的全部参数将一并删除，无法撤销。`,{confirmText:'删除',cancelText:'取消',kind:'warning'}))return;
  const original=kind==='field'?scene.fields:scene.lasers,index=original.findIndex(o=>o.id===id),selected=selection.kind===kind&&selection.id===id;
  if(kind==='field')scene.fields=scene.fields.filter(o=>o.id!==id);else scene.lasers=scene.lasers.filter(o=>o.id!==id);
  if(selected){const remaining=kind==='field'?scene.fields:scene.lasers,next=remaining[Math.min(index,remaining.length-1)];selection=next?{kind,id:next.id}:scene.lasers[0]?{kind:'laser',id:scene.lasers[0].id}:scene.fields[0]?{kind:'field',id:scene.fields[0].id}:{kind:'none',id:''};}
  controlsKey='';renderObjects();renderInspector();update();
  if(selected)requestAnimationFrame(()=>root.querySelector<HTMLButtonElement>(`[data-object-id="${CSS.escape(selection.id)}"]`)?.focus({preventScroll:true}));
}
function laserDetail(l:Laser):string{
  return l.frequencyMode==='wavelength'?withUnit(l.wavelengthNm??DEFAULT_TWEEZER_WAVELENGTH_NM,'nm',{digits:5}):`Δ ${withUnit(detuningInUnit(l),l.detuningUnit??'MHz',{digits:4})}`;
}
function renderObjects():void{
  setMiddle(sceneName,scene.name,'cut');
  const title=`${scene.name} — ${APP_TITLE}`;if(document.title!==title)document.title=title;
  $('field-list').replaceChildren(...scene.fields.map((f,i)=>item('field',f.id,i+1,f.name,`${withUnit(f.magnitudeG,'G',{digits:4})} · θ ${formatNumber(f.thetaDeg,{digits:4})}°`,f.enabled,v=>f.enabled=v)));
  $('laser-list').replaceChildren(...scene.lasers.map((l,i)=>item('laser',l.id,i+1,l.name,laserDetail(l),l.enabled,v=>l.enabled=v)));
  setText('field-count',String(scene.fields.length));setText('laser-count',String(scene.lasers.length));
  addField.disabled=scene.fields.length>=8;addLaser.disabled=scene.lasers.length>=8;
}
function nextObjectName(prefix:string,objects:{name:string}[]):string{const used=new Set(objects.map(o=>o.name));let n=1;while(used.has(`${prefix} ${n}`))n++;return `${prefix} ${n}`;}
function addObject(kind:'field'|'laser'):void{
  if(kind==='field'){if(scene.fields.length>=8)return;const id=`field-${Date.now()}`;scene.fields.push({id,name:nextObjectName('磁场',scene.fields),enabled:true,magnitudeG:1,thetaDeg:90,phiDeg:0});select({kind:'field',id});}
  else{if(scene.lasers.length>=8)return;const id=`laser-${Date.now()}`;scene.lasers.push({id,name:nextObjectName('激光',scene.lasers),enabled:true,regime:'near',line:spectral,detuningMHz:0,intensityMWcm2:.1,thetaDeg:90,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0});controlsKey='';select({kind:'laser',id});}
  update();
}

// ---------------------------------------------------------------- inspector
type BoundField={field:Field;spec:()=>FieldSpec};
const inspectorFields=new Map<string,BoundField>(),dynamicsFields=new Map<string,BoundField>(),fieldProblems=new Map<string,string>();
let fieldRequest=0;
/** A v3 Field on the scene: `apply` writes the operator's value (or throws its
 * problem, shown under the row); the row is then re-projected from the scene. */
function bindField(registry:Map<string,BoundField>,parent:HTMLElement,prefix:string,spec:()=>FieldSpec,apply:(value:Value)=>void,options:{unit?:(unit:string)=>void;after?:()=>void}={}):Field{
  const key=`${prefix}:${spec().name}`,current=():FieldSpec=>({...spec(),problem:fieldProblems.get(key)});
  const after=options.after??(()=>{renderObjects();update();});
  const field:Field=new Field(prefix,current(),{
    commit:value=>{
      const id=++fieldRequest;
      try{apply(value);fieldProblems.delete(key);}
      catch(error){fieldProblems.set(key,message(error));queueMicrotask(()=>field.update(current(),id,true));return id;}
      queueMicrotask(()=>field.update(current(),id));after();return id;
    },
    unit:options.unit?unit=>{try{options.unit!(unit);fieldProblems.delete(key);}catch(error){fieldProblems.set(key,message(error));}field.update(current());after();}:undefined,
  },false);
  parent.append(field.root);registry.set(key,{field,spec:current});return field;
}
function refreshFields(registry:Map<string,BoundField>):void{for(const {field,spec} of registry.values())field.update(spec());}
const numberSpec=(name:string,label:string,value:number,unit:string,extra:Partial<FieldSpec>={}):FieldSpec=>({name,label,kind:'number',value,unit,...extra});

function group(title:string,parent=$('inspector-controls')):HTMLElement{const section=element('section','control-section');const g=form();g.append(sectionTitle(title));section.append(g);parent.append(section);return g;}
function selectControl(parent:HTMLElement,label:string,value:string,options:[string,string][],changed:(value:string)=>void):HTMLSelectElement{
  if(label==='参考态'){
    const tree=new TreeCombo(changed);tree.setTree(stateTree(allLevels(),scene.references),value,'跟随动力学初态');tree.root.setAttribute('aria-label',label);treeOwners.set(tree.root,tree);const r=fieldRow(label,tree.root);r.classList.add('choice-field');parent.append(r);return tree.root;
  }
  const el=element('select');el.setAttribute('aria-label',label);el.append(comboFace(),...options.map(([v,t])=>new Option(t,v)));el.value=value;el.onchange=()=>changed(el.value);const r=fieldRow(label,el);r.classList.add('choice-field');parent.append(r);return el;
}
function setChoiceValue(control:HTMLSelectElement,value:string):void{const tree=treeOwners.get(control);if(tree)tree.value=value;else control.value=value;}
function disposeTrees(host:HTMLElement):void{host.querySelectorAll<HTMLSelectElement>('select.tree-combo').forEach(node=>treeOwners.get(node)?.dispose());}
/** Presets are commands, not modes. Rendering never selects or re-applies one. */
function polarizationButtons(laser:Laser,choices:readonly (readonly [PolarizationPreset,string])[],label:string):HTMLElement{
  const actions=element('div','preset-actions');actions.role='group';actions.setAttribute('aria-label',label);
  for(const [preset,text] of choices){
    const action=command(text,()=>{setPolarizationPreset(laser,scene.fields,preset);refreshFields(inspectorFields);renderObjects();update();});
    action.root.dataset.polarizationAction=preset;actions.append(action.root);
  }
  return actions;
}
/** Direction presets: commands that set k (or B) once; the one the current direction equals wears the active face. */
function directionButtons(parent:HTMLElement,apply:(theta:number,phi:number,exact?:AxisAngles)=>void,relative=false):void{
  const axes:[string,string][]=[['90,0','+x'],['90,180','−x'],['90,90','+y'],['90,-90','−y'],['0,0','+z'],['180,0','−z']];
  const make=(label:string,choices:[string,string][],kind:'axes'|'relative',act:(key:string)=>void)=>{
    const actions=element('div','preset-actions full-form-row');actions.role='group';actions.setAttribute('aria-label',label);
    for(const [key,text] of choices){const action=command(text,()=>act(key));action.root.dataset.direction=key;actions.append(action.root);directionCommands.push({command:action,kind,key});}
    parent.append(actions);
  };
  make('坐标轴方向',axes,'axes',key=>{const [theta,phi]=key.split(',').map(Number);apply(theta!,phi!);});
  if(relative)make('相对磁场方向',[['parallel','沿 +B'],['anti','沿 −B'],['perpendicular','垂直 B']],'relative',key=>{const axis=relativeDirectionAxis(composeMagneticField(scene.fields).fieldAxis,key as 'parallel'|'anti'|'perpendicular');apply(axis.thetaDeg,axis.phiDeg,axis);});
}
function renderInspector():void{
  const host=$('inspector-controls');const panel=host.closest<HTMLElement>('.inspector-panel')!,scroll=panel.scrollTop;
  disposeTrees(host);host.replaceChildren();inspectorFields.clear();directionCommands.length=0;
  renameObject.hidden=!(selection.kind==='laser'?scene.lasers.some(l=>l.id===selection.id):scene.fields.some(f=>f.id===selection.id));
  if(selection.kind==='none'){setText('inspector-title','总场');host.append(element('p','note wrap inspector-empty','选择磁场或激光以编辑其参数。'));return;}
  const fields=inspectorFields;
  if(selection.kind==='field'){
    const field=scene.fields.find(f=>f.id===selection.id);if(!field)return;setText('inspector-title',field.name);
    const g=group('大小与方向');
    bindField(fields,g,'inspector',()=>numberSpec('magnitude','大小',field.magnitudeG,'G',{minimum:0,maximum:20,spin:true}),v=>{field.magnitudeG=Number(v);});
    bindField(fields,g,'inspector',()=>numberSpec('field-theta','极角 θ',field.thetaDeg,'°',{minimum:0,maximum:180,step:1}),v=>{field.thetaDeg=Number(v);});
    bindField(fields,g,'inspector',()=>numberSpec('field-phi','方位 φ',field.phiDeg,'°',{minimum:-180,maximum:180,step:1}),v=>{field.phiDeg=Number(v);});
    directionButtons(g,(theta,phi)=>{field.thetaDeg=theta;field.phiDeg=phi;refreshFields(fields);renderObjects();update();});
  }else{
    const l=scene.lasers.find(l=>l.id===selection.id);if(!l)return;normalizeLaserFrequency(l);normalizeLaserIntensity(l);setText('inspector-title',l.name);
    // ui-design.md §4: every laser drives the dynamics; this only chooses the levels chart's background.
    const chart=group('能级图');bindField(fields,chart,'inspector',()=>({name:'show-light-shift',label:'显示光移',kind:'boolean',value:l.regime==='far',description:'开：未选中对象时，能级图的背景包含这束光的二阶光移（仅显示，不影响动力学）'}),v=>{l.regime=v===true?'far':'near';});
    const f=group('频率');
    frequencySection(f,l);
    const originPicker=selectControl(f,'参考态',l.referenceStateId??'',[['','跟随动力学初态'],...allLevels().map(state=>[state.id,stateText(state)] as [string,string]),...(scene.references??[]).map(ref=>[ref.id,ref.name+' · 参考能级'] as [string,string])],id=>{setLaserReferenceState(scene,l,id||undefined);persist();renderInspector();renderResults(false);});originPicker.id='laser-reference-state';originPicker.setAttribute('aria-label','参考出发态');originPicker.title='只改变这束光的箭头起点和矩元参考；不改变动力学初态或任何驱动。';
    if(scene.references?.some(r=>r.id===l.referenceStateId))selectControl(f,'光子方向',l.referenceProcess??'absorption',[['absorption','吸收 ↑'],['emission','受激辐射 ↓']],v=>{l.referenceProcess=v as 'absorption'|'emission';update();});
    const strength=group('强度');
    bindField(fields,strength,'inspector',()=>({name:'intensity-mode',label:'指定方式',kind:'choice',segmented:true,choices:[['intensity','局域强度'],['power','功率与束腰']],value:l.intensityMode??'intensity'}),v=>{setLaserIntensityMode(l,v as 'intensity'|'power');},{after:()=>{renderInspector();update();}});
    if(l.intensityMode==='power'){
      bindField(fields,strength,'inspector',()=>numberSpec('power','功率 P',l.powerMW??4,'mW',{minimum:0,spin:true}),v=>setLaserPowerMW(l,Number(v)));
      bindField(fields,strength,'inspector',()=>numberSpec('waist','束腰 w₀',l.waistUm??1.2,'μm',{spin:true}),v=>setLaserWaistUm(l,Number(v)));
    }else bindField(fields,strength,'inspector',()=>numberSpec('intensity','强度 I',l.intensityMWcm2,'mW/cm²',{minimum:0,spin:true}),v=>setLaserLocalIntensity(l,Number(v)));
    const d=group('传播方向');
    bindField(fields,d,'inspector',()=>numberSpec('laser-theta','极角 θ',l.thetaDeg,'°',{minimum:0,maximum:180,step:1}),v=>setLaserDirectionAxis(l,{thetaDeg:Number(v),phiDeg:l.phiDeg}));
    bindField(fields,d,'inspector',()=>numberSpec('laser-phi','方位 φ',l.phiDeg,'°',{minimum:-180,maximum:180,step:1}),v=>setLaserDirectionAxis(l,{thetaDeg:l.thetaDeg,phiDeg:Number(v)}));
    directionButtons(d,(theta,phi,exact)=>{setLaserDirectionAxis(l,exact??{thetaDeg:theta,phiDeg:phi},Boolean(exact));refreshFields(fields);renderObjects();update();},true);
    const p=group('光束偏振');
    bindField(fields,p,'inspector',()=>numberSpec('polarization','主轴 ψ',l.polarizationAngleDeg,'°',{minimum:-90,maximum:90,step:1}),v=>{l.polarizationAngleDeg=Number(v);});
    bindField(fields,p,'inspector',()=>numberSpec('ellipticity','椭圆率 χ',l.ellipticityDeg,'°',{minimum:-45,maximum:45,step:1}),v=>{l.ellipticityDeg=Number(v);});
    const shape=polarizationButtons(l,[['circular-minus','圆−'],['linear','线偏振'],['circular-plus','圆+']],'一次性设置光束偏振形状');shape.classList.add('full-form-row');p.append(shape);
    const atomic=polarizationButtons(l,[['sigma-minus','σ−'],['pi','π'],['sigma-plus','σ+']],'按当前磁场一次性设置纯偏振');atomic.title='按当前 B 设置一次（B=0 时按 z）：π 令 k ⟂ B，σ 令 k ∥ B；之后各参数独立。';p.append(fieldRow('纯偏振',atomic));
    const pol=element('div','polarization-bars full-form-row');pol.id='inspector-polarization';p.append(pol);
    const more=element('details','advanced-controls');more.innerHTML='<summary>相位</summary>';host.append(more);const phase=form();more.append(phase);
    bindField(fields,phase,'inspector',()=>numberSpec('laser-phase','初相位 φ₀',l.phaseDeg??0,'°',{minimum:-180,maximum:180,step:1}),v=>{l.phaseDeg=Number(v);});
  }
  requestAnimationFrame(()=>{shareLabelColumn(host);panel.scrollTop=scroll;});
}
/** 频率 (ui-design.md §1.3–1.4): the definition -- "相对" a base and the value against it, a
 * wavelength when the base is absolute -- then the frequency's readout, which reads it in any
 * unit against any base without changing it. */
const wavelengthUnits=new Map<string,WavelengthUnit>();
const readoutPrefs=new ReadoutPreferences(`${STORAGE}-readouts`);
const laserReadout=new FrequencyReadout('laser-frequency',readoutPrefs,{
  bases:()=>frequencyBaseTree({absolute:'光频',absoluteText:'绝对',lasers:scene.lasers.filter(other=>other.id!==selection.id)}),defaultBase:'absolute',changed:()=>{},
});
function frequencySection(f:HTMLElement,l:Laser):void{
  const fields=inspectorFields,absolute=l.frequencyMode==='wavelength';
  bindField(fields,f,'inspector',()=>({name:'frequency-base',label:'相对',kind:'choice',tree:frequencyBaseTree({absolute:'真空波长'}),
    value:l.frequencyMode==='wavelength'?'absolute':transitionKey(laserReferenceTransition(l)),description:'换基准不改变光频'}),
    v=>{const base=parseBase(String(v));if(!base||base.kind==='laser')throw new Error('未知的频率基准');setLaserFrequencyBase(l,base);},{after:()=>{renderInspector();renderObjects();update();}});
  if(absolute){
    const unit=()=>wavelengthUnits.get(l.id)??'nm';
    bindField(fields,f,'inspector',()=>({...numberSpec('wavelength','波长 λ',(l.wavelengthNm??DEFAULT_TWEEZER_WAVELENGTH_NM)/UNIT_NM[unit()],unit(),{spin:true,inexact:true}),units:WAVELENGTH_UNITS.map(u=>({value:u,label:u,group:'波长'}))}),
      v=>setLaserWavelengthNm(l,Number(v)*UNIT_NM[unit()]),{unit:u=>{if(isWavelengthUnit(u))wavelengthUnits.set(l.id,u);}});
  }else bindField(fields,f,'inspector',()=>({...numberSpec('detuning','失谐 Δ',detuningInUnit(l),l.detuningUnit??'MHz',{spin:true,inexact:true}),units:FREQUENCY_UNITS.map(u=>({value:u,label:u,group:'频率'}))}),
    v=>setLaserDetuningValue(l,Number(v)),{unit:u=>{setLaserDetuningUnit(l,u as DetuningUnit);}});
  // The reading has the row to itself (twelve digits need it); how it is read sits under it.
  f.append(fieldRow('频率 ν',laserReadout.readout.root),fieldRow('读数相对',laserReadout.base.root,laserReadout.unit.root));
  laserReadout.show(laserValue(l),scene);
}
function renderInitial():void{
  const current=getLevel(initialId())??getLevel(`ground-${scene.initialF}-${scene.initialM}`)!;if(!current)return;
  initialTree??=new TreeCombo(id=>setInitialState(id));initialTree.root.id='initial-state-picker';initialTree.root.setAttribute('aria-label','初态');initialTree.setTree(stateTree(allLevels()),current.id);
  if(!initialTree.root.isConnected)$('initial-state-tree').replaceChildren(initialTree.root);
  initialShortcut.title=`初态：${stateText(current)}`;
}
function setInitialState(id:string):void{const state=getLevel(id);if(!state)return;dynamics().initialStateId=id;if(state.manifold==='ground'){scene.initialF=state.F as 1|2;scene.initialM=state.m;}else spectral=state.manifold;renderInitial();update();}

function polarizationBars(a:Analysis,l:Laser|undefined):string{const p=l?a.polarizations[l.id]:undefined;if(!p)return '<p class="note">启用激光后显示球基分量。</p>';return `<span class="polarization-axis-label">${a.fieldMagnitudeG===0?'相对 z（B = 0）':'相对 B'}</span>`+[[math('\\sigma^-'),p.minus],[math('\\pi'),p.pi],[math('\\sigma^+'),p.plus]].map(([label,v])=>`<div class="pol-row"><span>${label}</span><div class="bar-track"><span class="bar-fill" style="width:${Math.max(0,Math.min(100,Number(v)*100))}%"></span></div><strong>${esc(populationPercent(Number(v)))}</strong></div>`).join('');}

function paintStateBadges():void{
  if(view==='levels')alignLevelReadouts($('levels-chart'));
  document.querySelectorAll<HTMLElement>('.selection-index[data-badge-number]').forEach(host=>drawBadges(host,[[host.dataset.badgeNumber!,'scan']]));
  $('levels-chart').querySelectorAll<SVGGElement>('.state-selection-number').forEach(group=>{const holder=document.createElement('span');drawBadges(holder,[[group.dataset.badgeNumber!,'scan']]);const svg=holder.firstElementChild!;svg.setAttribute('x',group.dataset.badgeX!);svg.setAttribute('y',group.dataset.badgeY!);group.replaceChildren(svg);});
}
/** A state's energy as a frequency, E/h; undefined where the background spectrum is. */
function stateValue(id:string):FrequencyValue|undefined{const l=getLevel(id);if(l)return levelValue(l);const r=scene.references?.find(r=>r.id===id);return r?referenceLevelValue(r,allLevels()):undefined;}
/** A transition's photon frequency ν₁₂ = |E₂ − E₁|/h (the pickers say absorption or emission). */
function photonValue(fromId:string,toId:string):FrequencyValue|undefined{const a=stateValue(fromId),b=stateValue(toId);return a&&b?photon(gap(a,b)):undefined;}
/** "自动" (ui-design.md §1.4): a transition inside one manifold reads absolutely; an optical one
 * relative to the selected laser (its detuning), else to the nearest zero-field transition. */
function automaticBase(v:FrequencyValue|undefined):string|undefined{
  if(!v||v.centroidMHz===0)return 'absolute';
  const laser=activeLaser();if(laser)return `laser:${laser.id}`;
  const t=nearestTransition(v,zeroFieldTransitionOptions());return t?transitionKey(t):undefined;
}
let pairValue:FrequencyValue|undefined;
const pairReadout=new FrequencyReadout('pair',readoutPrefs,{
  bases:()=>frequencyBaseTree({absolute:'光频',absoluteText:'绝对',lasers:scene.lasers}),
  automatic:()=>automaticBase(pairValue),
  changed:()=>renderResults(false),
});
const pairRow=element('div','pair-frequency'),pairLabel=element('span','pair-label','ν₁₂');pairLabel.title='跃迁频率 |E(终点) − E(起点)|/h';pairRow.append(pairLabel,pairReadout.readout.root,pairReadout.unit.root,pairReadout.baseLabel,pairReadout.base.root);
function selectionReadout(a:Analysis,l:Laser|undefined):string{
  const origin=referenceStateId(l),pair=comparisonPair(comparisonChoice(),origin),from=pair.from,to=pair.to;
  const pickers=`<div class="selection-top"><div class="selection-pair"><div class="selection-endpoint"><label><span class="selection-index" data-badge-number="1">1</span>起点</label><div id="comparison-from"></div></div><div id="comparison-swap"></div><div class="selection-endpoint"><label><span class="selection-index" data-badge-number="2">2</span>终点</label><div id="comparison-to"></div></div></div><button class="quiet" id="clear-selection">清除</button></div>`;
  pairValue=undefined;
  if(!to)return pickers+'<div class="selection-empty">点选能级或选择终点。</div>';
  const defined=a.spectroscopy?.status!=='unresolved';
  pairValue=defined?photonValue(from,to):undefined;
  const first=getLevel(from),second=getLevel(to);let coupling='';
  if(first&&second){const referenceLaser=l??scene.lasers[0],transition=defined&&referenceLaser?spectroscopyCoupling(a,referenceLaser,first.id,second.id):undefined;coupling=!defined?'<span>背景矩元未定义</span>':transition?(transition.allowed?`<span title="与所示背景能级使用同一基底的电偶极矩元">|d| <b>${measuredNumber(Math.abs(transition.amplitudeCM)/1e-29,4)} × 10⁻²⁹ C·m</b></span>${l?`<span>${l.enabled?'Ω/2π':'Ω/2π（停用预览）'} <b>${measuredNumber(transition.rabiMHz,4)} MHz</b></span>`:''}`:'<span>E1 禁戒</span>'):'';}
  else coupling='<span>参考能级</span>';
  if(first&&second&&defined&&!scene.lasers.length){const bare=dipoleBetweenStates(first.id,second.id,a.fieldMagnitudeG);coupling=bare.allowed?`<span title="无光背景的磁场本征态偶极矩元">|d| <b>${measuredNumber(Math.abs(bare.amplitudeCM)/1e-29,4)} × 10⁻²⁹ C·m</b></span>`:'<span>E1 禁戒</span>';}
  const pop=populationSnapshot().populations?.[to];
  return `${pickers}<div id="pair-frequency-slot"></div><div class="compact-readout">${coupling}<span data-current-population ${pop===undefined?'hidden':''}>p <b>${pop===undefined?'—':populationValue(pop)}</b></span></div><div class="selection-actions"><button class="quiet" id="use-as-initial" ${second?'':'disabled'}>末态设为初态</button><button class="quiet" id="monitor-selected" ${second?'':'disabled'}>观察布居</button><div class="mark-editor"><input id="mark-text" maxlength="32" aria-label="能级标记" placeholder="末态标记" value="${esc(scene.marks?.[to]??'')}" ${second?'':'disabled'}><button id="save-mark" ${second?'':'disabled'}>标记</button></div></div>`;
}
/** Solver responses update scientific readouts without unmounting an operator's editor. */
function renderSelection(a:Analysis,l:Laser|undefined):void{
  const host=$('selection-readout'),key=JSON.stringify([selectedStates,comparisonSource,referenceStateId(l),l?.id,scene.references?.map(r=>[r.id,r.name])]),next=element('div');next.innerHTML=selectionReadout(a,l);
  if(host.dataset.selectionKey!==key){disposeTrees(host);host.innerHTML=next.innerHTML;host.dataset.selectionKey=key;bindSelection();$('pair-frequency-slot')?.replaceWith(pairRow);pairReadout.show(pairValue,scene);return;}
  if(!pairRow.isConnected)$('pair-frequency-slot')?.replaceWith(pairRow);
  pairReadout.show(pairValue,scene);
  const readout=host.querySelector('.compact-readout'),updated=next.querySelector('.compact-readout');if(readout&&updated&&readout.innerHTML!==updated.innerHTML)readout.innerHTML=updated.innerHTML;
}
function renderDetails(hostId:string,markup:string):void{
  const host=$(hostId),next=element('div');next.innerHTML=markup;const nodes:HTMLElement[]=[];
  for(const incoming of [...next.children] as HTMLElement[]){const existing=([...host.children].find(el=>el.className===incoming.className) as HTMLElement|undefined)??(incoming.classList.contains('reference-editor')?referenceEditorNode:undefined);
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
  const from=new TreeCombo(id=>applyComparison(chooseComparisonFrom(comparisonChoice(),id)));from.setTree(groups,comparisonSource??(selectedStates.length>1?selectedStates[0]!:''),`${laser?'跟随光束参考态':'跟随动力学初态'} · ${idText(origin)}`);from.root.setAttribute('aria-label','跃迁起点');treeOwners.set(from.root,from);$('comparison-from').append(from.root);
  const to=new TreeCombo(id=>applyComparison(chooseComparisonTo(comparisonChoice(),id)));to.setTree(groups,pair.to??'','选择终点');to.root.setAttribute('aria-label','跃迁终点');treeOwners.set(to.root,to);$('comparison-to').append(to.root);
  const swap=command('⇄',()=>applyComparison(swapComparison(comparisonChoice(),origin)),{role:'quiet',glyph:true,square:true,title:'交换起点和终点'});swap.disabled=!pair.to;$('comparison-swap').append(swap.root);
  if($('clear-selection'))$('clear-selection').onclick=()=>applyComparison({selected:[]});
  if($('use-as-initial'))$('use-as-initial').onclick=()=>{const l=getLevel(selectedStates.at(-1)!);if(l)setInitialState(l.id);};
  if($('save-mark'))$('save-mark').onclick=()=>{const id=selectedStates.at(-1);if(id){const v=$<HTMLInputElement>('mark-text').value.trim();if(v)scene.marks![id]=v;else delete scene.marks![id];update();}};
  if($('monitor-selected'))$('monitor-selected').onclick=()=>{const ids=[...new Set(markedComparison(activeLaser()))].filter(id=>getLevel(id));const chosen=getLevel(ids.at(-1)??'');const matching=chosen?.manifold==='D1'||chosen?.manifold==='D2'?scene.lasers.find(l=>l.enabled&&l.line===chosen.manifold):activeLaser();if(matching)dynamics().laserId=matching.id;dynamics().monitoredStateIds=[...new Set([...dynamics().monitoredStateIds,...ids])];dynamics().monitorMode='manual';view='dynamics';controlsKey='';update();};
}
/** The table reads the two-state readout's base and unit (ui-design.md §1.8); its "自动" is the
 * same rule applied to the table's strongest row. Only components with Ω ≠ 0 are listed. */
function transitionTable(transitions:Transition[]):{heading:string;rows:string}{
  const listed=transitions.filter(t=>t.rabiMHz>0),ends=(t:Transition)=>transitionEndpoints(t,referenceStateId(scene.lasers.find(l=>l.id===t.laserId)))!;
  const strongest=listed[0]&&ends(listed[0]),key=pairReadout.chosenBase()??automaticBase(strongest&&photonValue(strongest.fromId,strongest.toId));
  const chosen=parseBase(key??'absolute'),base=chosen&&baseValue(chosen,scene)?chosen:{kind:'absolute' as const},origin=baseValue(base,scene)!,relative=base.kind!=='absolute';
  const pref=pairReadout.chosenUnit(),unit:DetuningUnit=pref&&!isWavelengthUnit(pref)?pref:relative?'MHz':'THz';
  return {heading:readingHeading(base,scene,unit),rows:listed.map(t=>{
  const endpoints=ends(t),mixed=t.mixedPolarization===true,ν=photonValue(endpoints.fromId,endpoints.toId);
  return `<tr data-transition="${esc(t.id)}" data-from-state="${endpoints.fromId}" data-to-state="${endpoints.toId}" tabindex="0"><td title="${endpoints.emission?'受激辐射':'吸收'}">${stateCell(endpoints.fromId)} → ${stateCell(endpoints.toId)}</td><td title="${mixed?'混合背景态：多个球分量相干耦合':t.mLabelApproximate?`光场 q=${t.q}；mF 是混合本征态的主成分标签，标签之差不是严格的 ΔmF`:`光场 q=${t.q}；原子 Δm=${endpoints.deltaM}`}" aria-label="${mixed?'混合分量':t.q===-1?'sigma minus':t.q===0?'pi':'sigma plus'}">${mixed?'混合':math(t.q===-1?'\\sigma^-':t.q===0?'\\pi':'\\sigma^+')}</td><td>${ν?formatNumber(inUnit(difference(ν,origin),unit),{digits:relative?9:12,signed:relative}):'—'}</td><td>${measuredNumber(t.rabiMHz,4)}</td><td><div class="strength-cell"><span style="width:${Math.min(100,t.strength*t.weight*100)}%"></span><small>${measuredNumber(t.strength*t.weight,4)}</small></div></td></tr>`;
}).join('')};}
function referenceAnchorDefault():string{return [...selectedStates].reverse().find(id=>getLevel(id))??(comparisonSource&&getLevel(comparisonSource)?comparisonSource:initialId());}
function referenceEditor():string{
  const anchor=referenceAnchorDefault();return `<details class="reference-editor"><summary>参考能级 <span>${scene.references?.length??0}</span></summary><div class="reference-form-caption" id="reference-form-caption">新建参考能级</div><div class="reference-form form"><label class="field-label" for="reference-name">名称</label><div class="field-control"><input id="reference-name" maxlength="32" value="虚拟中间态" aria-label="参考能级名称"></div><label class="field-label" for="reference-anchor">锚点</label><div class="field-control"><select id="reference-anchor" aria-label="参考能级锚点">${allLevels().map(l=>`<option value="${l.id}" ${l.id===anchor?'selected':''}>${esc(stateText(l))}</option>`).join('')}</select></div><label class="field-label" for="reference-anchor-mode">能量定义</label><div class="field-control"><div id="reference-anchor-mode" data-value="follow"></div></div><label class="field-label" for="reference-offset">频率偏移</label><div class="field-control"><div class="number-field" id="reference-number"><input id="reference-offset" type="text" inputmode="decimal" value="${100/DETUNING_UNIT_MHZ[referenceUnit]}" aria-label="参考能级频率偏移"></div><select id="reference-unit" class="unit-picker" aria-label="参考能级频率单位">${Object.keys(DETUNING_UNIT_MHZ).map(unit=>`<option value="${unit}" ${unit===referenceUnit?'selected':''}>${unit}</option>`).join('')}</select><button class="quiet reference-laser-copy" id="reference-from-laser" aria-label="用当前激光建立参考能级">取激光终点</button></div><div class="reference-definition full-form-row" id="reference-definition"></div><div class="form-actions"><button id="add-reference">添加参考能级</button><button class="quiet" id="cancel-reference-edit" hidden>取消编辑</button></div></div><div class="reference-list">${scene.references!.map(r=>`<div class="reference-row"><button class="state-chip" data-state="${r.id}">${esc(r.name)}</button><span title="${r.anchorMode==='zero-field'?'使用零磁场锚点，随当前磁场保持固定':'跟随当前磁场下的锚点能量'}">${stateCell(r.anchorId)} ${r.anchorMode==='zero-field'?'· B=0':''} ${r.offsetMHz<0?'−':'+'} ${frequencyText(Math.abs(r.offsetMHz))}</span><button class="quiet" data-edit-reference="${r.id}" aria-label="编辑参考能级 ${esc(r.name)}">编辑</button><button class="icon-button" data-remove-reference="${r.id}" aria-label="删除参考能级 ${esc(r.name)}">×</button></div>`).join('')}</div></details>`;
}
function bindReferences():void{
  const editor=$('levels-details').querySelector<HTMLElement>('.reference-editor');if(!editor)return;
  const input=$<HTMLInputElement>('reference-offset'),unitPicker=$<HTMLSelectElement>('reference-unit'),modeRoot=$('reference-anchor-mode');let modeControl:Segmented|undefined;const mode={get value(){return modeRoot.dataset.value??'follow';},set value(value:string){modeRoot.dataset.value=value;modeControl?.setCurrent(value);}};
  const definition=()=>{$('reference-definition').innerHTML=math(mode.value==='zero-field'?'E_{ref}=E_{anchor}(B=0)+h\\delta':'E_{ref}(B)=E_{anchor}(B)+h\\delta');};
  if(!editor.dataset.bound){editor.dataset.bound='true';let unit=unitPicker.value as DetuningUnit,canonical=Number(input.value)*DETUNING_UNIT_MHZ[unit],shown=input.value;const readMHz=()=>{const text=spin.text;if(text!==shown){canonical=Number(text)*DETUNING_UNIT_MHZ[unit];shown=text;}return canonical;};const setMHz=(value:number)=>{canonical=value;input.value=String(value/DETUNING_UNIT_MHZ[unit]);shown=input.value;spin.refresh();};const spin=new SpinBox($('reference-number'),input,{step:()=>.01,bounds:()=>[-1e12/DETUNING_UNIT_MHZ[unit],1e12/DETUNING_UNIT_MHZ[unit]],commit:n=>setMHz(n*DETUNING_UNIT_MHZ[unit]),decimal:true});referenceNumber={readMHz,setMHz};spin.refresh();unitPicker.onchange=()=>{const next=unitPicker.value as DetuningUnit,value=readMHz();if(!input.value.trim()||!Number.isFinite(value)){unitPicker.value=unit;return;}unit=next;referenceUnit=next;setMHz(value);};editor.addEventListener('input',event=>{editor.dataset.dirty='true';(event.target as HTMLElement).removeAttribute('aria-invalid');});}
  modeControl=new Segmented(value=>{mode.value=String(value);editor.dataset.dirty='true';definition();});modeControl.setChoices([['follow','随锚点'],['zero-field','固定零场']]);modeControl.setCurrent(mode.value);modeControl.root.setAttribute('aria-label','参考能级能量定义');modeRoot.replaceChildren(modeControl.root);
  const anchorPicker=$<HTMLSelectElement>('reference-anchor');if(!treeOwners.has(anchorPicker)){const tree=new TreeCombo(()=>{editor.dataset.dirty='true';});tree.setTree(stateTree(allLevels()),anchorPicker.value);tree.root.id=anchorPicker.id;tree.root.setAttribute('aria-label','参考能级锚点');treeOwners.set(tree.root,tree);anchorPicker.replaceWith(tree.root);}
  requestAnimationFrame(()=>shareLabelColumn(editor));
  const resetDraft=()=>{referenceEditingId=null;$<HTMLInputElement>('reference-name').value='虚拟中间态';setChoiceValue($<HTMLSelectElement>('reference-anchor'),referenceAnchorDefault());mode.value='follow';referenceNumber!.setMHz(100);delete editor.dataset.dirty;$('reference-form-caption').textContent='新建参考能级';$('add-reference').textContent='添加参考能级';$('cancel-reference-edit').hidden=true;definition();};
  definition();const beam=activeLaser(),photon=beam?referenceFromLaser(scene,beam,allLevels()):undefined;$<HTMLButtonElement>('reference-from-laser').disabled=!photon;$('reference-from-laser').title=photon?.emission?'参考出发态 − hν':'参考出发态 + hν';$('reference-from-laser').onclick=()=>{if(!photon)return;setChoiceValue($<HTMLSelectElement>('reference-anchor'),photon.anchorId);mode.value=photon.anchorMode;editor.dataset.dirty='true';referenceNumber!.setMHz(photon.offsetMHz);definition();};
  $('add-reference').onclick=()=>{if(!referenceEditingId&&scene.references!.length>=12){note('最多 12 个参考能级','warning');return;}const name=$<HTMLInputElement>('reference-name').value.trim(),raw=input.value.trim(),offset=referenceNumber!.readMHz();if(!name||!raw||!Number.isFinite(offset)||Math.abs(offset)>1e12){$<HTMLInputElement>('reference-name').toggleAttribute('aria-invalid',!name);input.toggleAttribute('aria-invalid',!raw||!Number.isFinite(offset)||Math.abs(offset)>1e12);note('请填写名称与有限的频率偏移（最大 ±1 000 000 THz）','warning');return;}const id=referenceEditingId??`reference-${Date.now()}`,reference={id,name,anchorId:$<HTMLSelectElement>('reference-anchor').value,anchorMode:mode.value as 'follow'|'zero-field',offsetMHz:offset},index=scene.references!.findIndex(r=>r.id===id);if(index>=0)scene.references![index]=reference;else scene.references!.push(reference);resetDraft();comparisonSource=undefined;selectedStates=[id];persist();renderInspector();renderResults(false);};
  $('cancel-reference-edit').onclick=resetDraft;
  editor.querySelectorAll<HTMLButtonElement>('[data-edit-reference]').forEach(button=>button.onclick=()=>{const ref=scene.references!.find(r=>r.id===button.dataset.editReference)!;referenceEditingId=ref.id;(editor as HTMLDetailsElement).open=true;delete editor.dataset.dirty;$('reference-form-caption').textContent='编辑参考能级';$<HTMLInputElement>('reference-name').value=ref.name;setChoiceValue($<HTMLSelectElement>('reference-anchor'),ref.anchorId);mode.value=ref.anchorMode??'follow';referenceNumber!.setMHz(ref.offsetMHz);$('add-reference').textContent='保存参考能级';$('cancel-reference-edit').hidden=false;definition();$<HTMLInputElement>('reference-name').focus();});
  editor.querySelectorAll<HTMLButtonElement>('[data-remove-reference]').forEach(button=>{const action=()=>{const id=button.dataset.removeReference;for(const beam of scene.lasers)if(beam.referenceStateId===id)delete beam.referenceStateId;scene.references=scene.references!.filter(r=>r.id!==id);selectedStates=selectedStates.filter(id=>id!==button.dataset.removeReference);if(referenceEditingId===id)resetDraft();update();};const remove=command('×',action,{role:'destructive',glyph:true,square:true,title:button.getAttribute('aria-label')??'删除参考能级'}).root;remove.dataset.removeReference=button.dataset.removeReference;remove.setAttribute('aria-label',button.getAttribute('aria-label')??'删除参考能级');button.replaceWith(remove);});
}

// ---------------------------------------------------------------- solver
function dynamicsKey(steadyOnly=view==='levels'&&populationMode==='steady'):string{return dynamicsCacheKey(scene,dynamics(),steadyOnly);}
function cachedAnalysis():Analysis{
  const key=JSON.stringify({kind:view==='levels'?'spectrum':'bare',fields:scene.fields,lasers:scene.lasers,initial:initialId(),excluded:view==='levels'?activeLaser()?.id:undefined});
  const cached=analysisCache.get(key);if(cached)return cached;
  const start=performance.now(),analysis=view==='levels'?spectroscopyAnalysis(scene,activeLaser()?.id):analyze(scene);
  analysisCache.set(key,analysis);while(analysisCache.size>4)analysisCache.delete(analysisCache.keys().next().value!);
  root.dataset.spectrumBuildCount=String(++spectrumBuildCount);root.dataset.spectrumComputeMs=String(performance.now()-start);return analysis;
}
function invalidateObsoletePopulation():void{
  // Retain the displayed calculation only; revisiting earlier parameters must solve again.
  const physicalKey=dynamicsKey(false);displayedResults.activate(physicalKey);
  const key=dynamicsKey();if(requestedPhysicalKey&&requestedPhysicalKey!==physicalKey)cancelObsoleteSolve();
  if(renderedPopulationKey&&renderedPopulationKey!==key){
    if(view==='levels')patchLevelPopulations($('levels-chart'),{label:'布居待计算',mean:false,selectedLaser:!!activeLaser(),initialId:initialId()});
    const oldReadout=$('selection-readout').querySelector<HTMLElement>('[data-current-population]');if(oldReadout){oldReadout.hidden=true;const value=oldReadout.querySelector('b');if(value)value.textContent='—';}
    root.dataset.populationFresh='false';if(latestAnalysis&&view==='levels')populationToolbar();
  }
}
function refreshPopulationOnly():void{
  if(analysisError){refreshStatus();return;}
  if(view==='levels'){
    const snapshot=populationSnapshot(),steady=steadyPopulationsInView();populationToolbar();
    patchLevelPopulations($('levels-chart'),{populations:snapshot.populations,steady,label:snapshot.label,mean:isDisplayedMean(),selectedLaser:!!activeLaser(),initialId:initialId()});
    alignLevelReadouts($('levels-chart'));
    // The readout editor and its focusable state chips remain mounted.
    const p=$('selection-readout').querySelector<HTMLElement>('[data-current-population]'),value=selectedStates.length?snapshot.populations?.[selectedStates.at(-1)!]:undefined;
    if(p){p.hidden=value===undefined;const b=p.querySelector('b');if(b)b.textContent=value===undefined?'—':populationValue(value);}
    const details=$('levels-details').querySelector<HTMLDetailsElement>('.solver-details');if(details){const next=element('div');next.innerHTML=solverDetails();const summary=details.querySelector<HTMLElement>('summary');reconcileChildren(details,[...next.firstElementChild!.children].map(child=>child.tagName==='SUMMARY'&&summary?summary:child as HTMLElement));}
  }else if(view==='dynamics')renderDynamics(latestAnalysis,false);
  renderedPopulationKey=dynamicsKey();root.dataset.populationFresh=String(!!currentResult());root.dataset.populationPaintCount=String(++populationPaintCount);refreshStatus();
}
function requestSolve(force=false):void{
  const key=dynamicsKey(),physicalKey=dynamicsKey(false),steadyOnly=view==='levels'&&populationMode==='steady';
  if(force)stoppedKey='';else if(key===stoppedKey)return;
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
    }catch(e){if(!stillCurrent())return;solverBusy=false;solverError=message(e);refreshPopulationOnly();}
  },force?0:16);
  refreshPopulationOnly();
}
function cancelObsoleteSolve():void{solver.cancel();solverRevision++;clearTimeout(solverTimer);requestedKey='';requestedPhysicalKey='';requestedSteadyOnly=false;solverBusy=false;solverQueued=false;solverError='';}
/** 停止: the running solve ends here; a mean it already delivered stays. Editing the scene, or 计算, starts again. */
function stopSolve():void{
  if(!currentSolverBusy())return;
  const key=dynamicsKey();cancelObsoleteSolve();stoppedKey=key;requestedKey=key;
  if(view==='dynamics')renderDynamics(latestAnalysis,false);else refreshPopulationOnly();
  refreshStatus();
}
function currentRecord():SolvedRecord|undefined{
  const key=dynamicsKey(false);
  if(view==='levels'&&populationMode==='steady')return displayedResults.get(key,'mean');
  return displayedResults.get(key,'trajectory')??displayedResults.get(key,'progress');
}
function currentResult():DynamicsResult|undefined{return currentRecord()?.result;}
function currentSolverError():string{return requestedKey===dynamicsKey()?solverError:'';}
function currentSolverBusy():boolean{return solverBusy&&requestedPhysicalKey===dynamicsKey(false)&&(!requestedSteadyOnly||(view==='levels'&&populationMode==='steady'));}
function isDisplayedMean(result=currentResult(),a=latestAnalysis):boolean{return isTimeAverage(result?.steady?.status)||(view==='levels'&&a.spectroscopy?.requiresDensityProjection===true&&result?.steady?.status==='population-stationary');}
function steadyPopulationsInView(a=latestAnalysis):Record<string,number>|undefined{
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
  return {populations:Object.fromEntries(Object.entries(result.populations).map(([id,values])=>[id,values[index]??0])),label:`p(${formatNumber(timeUs,{digits:4})} μs)`,timeUs};
}
function solverDetails():string{
  const result=currentResult(),error=currentSolverError();
  const readout=(label:string,value:string)=>`<span class="field-label">${label}</span><span class="readout-edit">${esc(value)}</span>`;
  const small=(value:number)=>formatNumber(value,{digits:2});
  const accuracy=result?.populationAccuracy;
  const rows=result?[
    readout('求解耗时',formatElapsed(result.metrics.totalMs)),readout('含通信',formatElapsed(currentRecord()?.wallMs??solverWallMs)),
    ...(accuracy?[readout('布居误差估计',`≤ ${small(accuracy.estimatedMaximumError)}（目标 ${small(accuracy.requestedTolerance)}）`)]:[]),
    readout('概率守恒误差',small(result.validation.maxTraceError)),readout('最小布居',small(result.validation.minPopulation)),
    readout('长期解',`${steadyStatusText(result.steady?.status)}${result.steady?.residual==null?'':` · 残差 ${small(result.steady.residual)}`}`),
    // The limit is reached after a few of these; more than one conserved quantity makes it initial-dependent.
    ...(result.steady?.slowestRelaxationUs!=null?[readout('最慢弛豫时间',(()=>{const unit=timeDisplayUnit(result.steady.slowestRelaxationUs!);return withUnit(result.steady.slowestRelaxationUs!*unit.factor,unit.name,{digits:3});})())]:[]),
    ...(result.steady?.conservedQuantities!=null?[readout('守恒量个数',`${result.steady.conservedQuantities}（含迹）`)]:[]),
  ].join(''):'';
  const notes=result?[...(result.method?[`时间演化：${solverText(result.method)}`]:[]),...(result.steady.method?[`长期解：${solverText(result.steady.method)}`]:[]),...(result.steady.reason?[solverText(result.steady.reason)]:[]),...result.assumptions]:[];
  return `<details class="solver-details"><summary>计算详情</summary>${error?`<p class="solver-error">${esc(solverText(error))}</p>`:''}${result?`<div class="form solver-readouts">${rows}</div>${notes.map(text=>`<p class="note wrap">${esc(text)}</p>`).join('')}`:'<p class="note">没有当前参数的求解结果。</p>'}${failureRecords.length?'<button data-export-solver-failures>导出失败记录</button>':''}</details>`;
}
function showSolverDetails():void{
  const host=view==='dynamics'?$('dynamics-details'):$('levels-details'),detail=host.querySelector<HTMLDetailsElement>('.solver-details');
  if(detail){detail.open=true;detail.scrollIntoView({block:'nearest',behavior:'smooth'});}
}

// The levels page's population row: what the cards are coloured by, the scale, the F totals and 计算.
const populationMode$=element('select');populationMode$.id='population-mode';populationMode$.setAttribute('aria-label','能级布居着色');
const populationModeFace=comboFace();populationMode$.append(populationModeFace,new Option('稳态布居','steady'),new Option('指定时刻 p(t)','time'));
populationMode$.onchange=()=>{populationMode=populationMode$.value as 'steady'|'time';update();};
const populationTime=element('input');populationTime.type='range';populationTime.id='population-time';populationTime.setAttribute('aria-label','布居时间');
populationTime.oninput=()=>{populationTimeIndex=Number(populationTime.value);update();};
const populationTimeValue=element('span','population-time-value');populationTimeValue.id='population-time-value';
const populationLegend=element('span','population-scale');populationLegend.setAttribute('aria-label','固定布居色标 0 至 100%');
populationLegend.append(element('span','scale-end','0'),element('span','scale-bar'),element('span','scale-end','100%'));
const populationTotals=element('span','population-data');populationTotals.id='population-data';
const runLevels=command('计算',()=>requestSolve(true));runLevels.root.dataset.runDynamics='';
$('population-toolbar').append(populationMode$,populationTime,populationTimeValue,populationLegend,populationTotals,runLevels.root);
function populationToolbar():void{
  const snap=populationSnapshot(),result=currentResult(),p=snap.populations;
  // A base-select shows the picked option as it was picked: a renamed choice is a new option.
  const steadyWords=isDisplayedMean(result)?'长期平均布居':'稳态布居';
  if(populationMode$.options[0]!.text!==steadyWords){populationMode$.replaceChildren(populationModeFace,new Option(steadyWords,'steady'),new Option('指定时刻 p(t)','time'));populationMode$.value=populationMode;}
  else if(populationMode$.value!==populationMode)populationMode$.value=populationMode;
  const timed=populationMode==='time'&&!!result;populationTime.hidden=!timed;populationTimeValue.hidden=!timed;
  if(timed){const max=String(result.timesUs.length-1),value=String(populationTimeIndex<0?result.timesUs.length-1:Math.min(populationTimeIndex,result.timesUs.length-1));if(populationTime.max!==max)populationTime.max=max;if(populationTime.value!==value)populationTime.value=value;populationTimeValue.textContent=durationText(snap.timeUs??0,durationUnit(dynamics()));}
  const sum=(filter:(id:string)=>boolean)=>p?Object.entries(p).reduce((total,[id,value])=>total+(filter(id)?value:0),0):undefined;
  const f1=sum(id=>/^ground-1-/.test(id)),f2=sum(id=>/^ground-2-/.test(id)),pe=sum(id=>!id.startsWith('ground-'));
  const markup=`<span class="population-total">F=1 <b>${populationPercent(f1)}</b></span><span class="population-total">F=2 <b>${populationPercent(f2)}</b></span><span class="population-total">激发态 <b>${populationPercent(pe)}</b></span>`;
  if(populationTotals.innerHTML!==markup)populationTotals.innerHTML=markup;
  runLevels.disabled=currentSolverBusy()&&!solverQueued;
}

// ---------------------------------------------------------------- 布居演化
function renderDynamicsControls(a:Analysis):void{
  const d=dynamics(),enabled=scene.lasers.filter(l=>l.enabled&&laserIntensityMWcm2(l)>0);d.durationUnit??=durationUnit(d);if(!enabled.some(l=>l.id===d.laserId))d.laserId=enabled[0]?.id??'';
  const states=allLevels(a);d.monitoredGroupIds??=[];
  if(!states.some(l=>l.id===d.initialStateId))d.initialStateId=initialId();d.monitoredStateIds=d.monitoredStateIds.filter(id=>states.some(l=>l.id===id));
  setText('dynamics-beams',enabled.length?`驱动：${enabled.map(l=>l.name).join(' + ')}`:'驱动：无光束');
  runDynamics.disabled=currentSolverBusy()&&!solverQueued;
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
  const time=form('dynamics-config'),accuracy=form('dynamics-config');parent.append(time,accuracy);
  const after=()=>update();
  bindField(dynamicsFields,time,'dynamics',()=>({...numberSpec('duration','时长',durationValue(d),durationUnit(d),{spin:true,inexact:true}),units:DURATION_UNITS.map(u=>({value:u,label:u,group:'时间'}))}),v=>setDurationValue(d,Number(v)),{after,unit:unit=>{setDurationUnit(d,unit as DurationUnit);}});
  bindField(dynamicsFields,time,'dynamics',()=>({name:'samples',label:'采样点数',kind:'integer',value:d.samples,minimum:2,maximum:2001,step:1}),v=>{const n=Number(v);if(!Number.isInteger(n))throw new Error('采样点数必须为整数');d.samples=n;},{after});
  bindField(dynamicsFields,accuracy,'dynamics',()=>numberSpec('population-tolerance','布居误差目标',Number((d.populationTolerance*100).toPrecision(12)),'%',{maximum:MAX_POPULATION_TOLERANCE*100,step:.1}),v=>{d.populationTolerance=resolvePopulationTolerance(Number(v)/100);},{after});
  bindField(dynamicsFields,accuracy,'dynamics',()=>({name:'monitor-mode',label:'显示曲线',kind:'choice',segmented:true,choices:[['auto','初末各 3 态'],['manual','手动选择']],value:d.monitorMode??'auto'}),v=>{d.monitorMode=v as 'auto'|'manual';if(v==='auto')d.monitoredGroupIds=[];else parent.querySelector<HTMLDetailsElement>('.monitor-controls')!.open=true;},{after});
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
  requestAnimationFrame(()=>{shareLabelColumn(time);shareLabelColumn(accuracy);});
}
const monitorAll=command('全部磁子能级',()=>{const d=dynamics();d.monitorMode='manual';d.monitoredStateIds=allLevels().map(state=>state.id);update();},{role:'quiet'});monitorAll.root.id='monitor-all';
const monitorNone=command('清空',()=>{const d=dynamics();d.monitorMode='manual';d.monitoredStateIds=[];d.monitoredGroupIds=[];update();},{role:'quiet'});monitorNone.root.id='monitor-none';
function renderDynamics(a:Analysis,_allowSolve=true):void{
  const d=dynamics(),raw=currentResult();if((d.monitorMode??'auto')==='auto')d.monitoredStateIds=recommendedPopulationStates(raw,d.initialStateId);
  renderDynamicsControls(a);const ids=populationDisplayIds(d),result=raw?withPopulationGroups(raw,d.monitoredGroupIds??[]):undefined;
  const labels=Object.fromEntries([...allLevels(a).map(l=>[l.id,stateText(l)]),...POPULATION_GROUPS.map(g=>[g.id,g.label])]);
  unbindPopulation?.();unbindPopulation=undefined;
  const host=$('population-chart'),plotOptions={showTotalExcited:false,timeUnit:durationUnit(d),scale:populationScale,viewportWidth:host.clientWidth,initialId:d.initialStateId};
  const stopped=stoppedKey&&stoppedKey===dynamicsKey();
  host.innerHTML=result&&result.timesUs.length>1?populationChart(result,ids,labels,plotOptions):result?.steady.converged?`<div class="empty-state"><div><p>长期布居</p><div class="form solver-readouts">${ids.map(id=>`<span class="field-label">${esc(labels[id]??id)}</span><span class="readout-edit">${populationValue(result.steady.populations?.[id]??0)}</span>`).join('')}</div><p class="note">${currentSolverError()?'时间演化未完成，原因见状态栏。':stopped?'时间演化已停止。':'时间演化计算中…'}</p></div></div>`:`<div class="empty-state">${currentSolverError()?esc(solverText(currentSolverError())):currentSolverBusy()?'正在求解当前场景…':stopped?'计算已停止；按“计算”重新开始。':'按“计算”开始求解。'}</div>`;
  if(result&&result.timesUs.length>1)unbindPopulation=bindPopulationChart(host,result,ids,labels,plotOptions);
  const legend=ids.map(id=>`<span class="population-legend"><svg width="18" height="5" aria-hidden="true"><line x1="0" y1="2.5" x2="18" y2="2.5" stroke="${populationColor(id)}" stroke-width="2"/></svg>${esc(labels[id]??id)}${id===d.initialStateId?'<span class="initial-tag">初态</span>':''}</span>`).join('');
  $('population-caption').innerHTML=legend+(result&&result.timesUs.length>1?`<span class="sampling-note">${esc(populationSamplingInfo(result))}</span>`:'');
  renderDetails('dynamics-details',solverDetails());
}

function refreshDirectionStates(a:Analysis):void{
  const current=selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):scene.fields.find(f=>f.id===selection.id);if(!current)return;
  const currentAxis:AxisAngles=selection.kind==='laser'?laserDirectionAxis(current as Laser):current;
  const sameAxis=(axis:AxisAngles)=>currentAxis.thetaDeg===axis.thetaDeg&&(currentAxis.thetaResidualDeg??0)===(axis.thetaResidualDeg??0)&&(((axis.thetaDeg===0||axis.thetaDeg===180)&&!(axis.thetaResidualDeg??0))||((currentAxis.phiDeg===axis.phiDeg||Math.abs(currentAxis.phiDeg-axis.phiDeg)===360)&&(currentAxis.phiResidualDeg??0)===(axis.phiResidualDeg??0)));
  for(const {command:action,kind,key} of directionCommands){
    if(kind==='axes'){const [theta,phi]=key.split(',').map(Number);action.setActive(sameAxis({thetaDeg:theta!,phiDeg:phi!}));}
    else{action.disabled=a.fieldMagnitudeG===0;action.setActive(a.fieldMagnitudeG>0&&sameAxis(relativeDirectionAxis(a.fieldAxis,key as 'parallel'|'anti'|'perpendicular')));}
  }
}

/** The level diagram's one-line definition of its labels (ui-design.md §1.5). */
function levelsLegend(a:Analysis,laser:Laser|undefined):string{
  const background=[...(a.fieldMagnitudeG>0?['B']:[]),...(a.spectroscopy?.backgroundLaserIds??[]).map(id=>scene.lasers.find(l=>l.id===id)?.name??id)].join(' + ')||'无';
  if(a.spectroscopy?.status==='unresolved')return `标注：裸态能级，背景光移未确定；背景：${background}`;
  // Every card is how far its state lies above its anchor (levelReadoutMHz): + is higher.
  if(laser&&laserEmits(scene,laser,allLevels(a)))return `标注：激发态 (E − E参考)/h，基态为高出光子终点 E参考 − hν(${laser.name}) 的量；正值表示更高；背景：${background}`;
  if(laser)return `标注：激发态 ν(参考态→e) − ν(${laser.name})，基态 (E − E参考)/h；正值表示更高；背景：${background}`;
  return `标注：总能移（相对零外场）；背景：${background}`;
}
/** What the level diagram is showing now, for its ⓘ. */
function levelScope():{title:string;lines:string[]}|undefined{
  const a=latestAnalysis;if(!a)return undefined;
  const lines=[a.spectroscopy?.modelScope,...(a.spectroscopy?.warnings??[])].filter((line):line is string=>Boolean(line)).flatMap(line=>line.split('\n'));
  return lines.length?{title:'当前能级图',lines}:undefined;
}
function samplingScope():{title:string;lines:string[]}|undefined{
  const result=currentResult();return result&&result.timesUs.length>1?{title:'当前采样',lines:[populationSamplingInfo(result)]}:undefined;
}
/** 方向与偏振's legend: |B| and each field, every beam in its colour; the selected beam's ∠(k, B) and σ⁻/π/σ⁺ (as in the inspector). */
function renderSpaceLegend(a:Analysis):void{
  const fields=scene.fields.filter(f=>f.enabled&&f.magnitudeG>0),zero=a.fieldMagnitudeG===0;
  const swatch=(colour:string,thick=false)=>`<span class="legend-swatch${thick?' thick':''}" style="background:${colour}"></span>`;
  const isSelected=(kind:'field'|'laser',id:string)=>selection.kind===kind&&selection.id===id;
  const items=[`<span class="legend-item total">${swatch('var(--field-ink)',true)}<span>${fields.length>1?'合成 B':'B'}：<b>${zero?'0（参考轴 +z）':withUnit(a.fieldMagnitudeG,'G',{digits:4})}</b></span></span>`];
  if(fields.length>1)for(const f of fields)items.push(`<button class="quiet legend-item${isSelected('field',f.id)?' selected':''}" data-field-id="${esc(f.id)}">${swatch('var(--field-ink)')}<span>${esc(f.name)}：<b>${withUnit(f.magnitudeG,'G',{digits:4})}</b></span></button>`);
  for(const l of scene.lasers.filter(l=>l.enabled)){
    const p=a.polarizations[l.id],chosen=isSelected('laser',l.id),detail=chosen&&p?` · ∠(k, ${zero?'z':'B'}) <b>${formatNumber(p.angleToBDeg,{places:1})}°</b> · σ⁻ <b>${populationPercent(p.minus)}</b> · π <b>${populationPercent(p.pi)}</b> · σ⁺ <b>${populationPercent(p.plus)}</b>`:'';
    items.push(`<button class="quiet legend-item${chosen?' selected':''}" data-laser-id="${esc(l.id)}">${swatch(laserColour(scene,l.id))}<span>${esc(l.name)}${detail}</span></button>`);
  }
  const markup=items.join('');if($('space-legend').innerHTML!==markup)$('space-legend').innerHTML=markup;
}
function showAnalysisError(text:string):void{
  const markup=`<div class="empty-state error" role="alert">能级参数无效：${esc(text)}</div>`;
  if(view==='levels'){$('levels-chart').innerHTML=markup;$('selection-readout').hidden=true;$('population-toolbar').hidden=true;$('levels-details').replaceChildren();referenceEditorNode=undefined;}
  else if(view==='dynamics'){unbindPopulation?.();unbindPopulation=undefined;$('population-chart').innerHTML=markup;$('population-caption').replaceChildren();}
  else{spaceMount.remove();$('space-chart').innerHTML=markup;}
}

function renderResults(allowSolve=true):void{
  const focused=document.activeElement?.closest<HTMLElement>('[data-state],[data-laser-id],[data-transition]'),levelsChart$=$('levels-chart'),levelsDetails=$('levels-details');
  const focusHost=focused&&(levelsChart$.contains(focused)?levelsChart$:levelsDetails.contains(focused)?levelsDetails:undefined),focusAttribute=focused&&['data-state','data-laser-id','data-transition'].find(attr=>focused.hasAttribute(attr)),focusValue=focusAttribute?focused?.getAttribute(focusAttribute):undefined;
  if(tabs.currentKey!==view)tabs.setCurrent(view);
  invalidateObsoletePopulation();
  let a:Analysis;
  try{a=cachedAnalysis();latestAnalysis=a;analysisError='';selectedStates=selectedStates.filter(id=>allLevels(a).some(state=>state.id===id)||scene.references?.some(ref=>ref.id===id));if(comparisonSource&&!allLevels(a).some(s=>s.id===comparisonSource)&&!scene.references?.some(r=>r.id===comparisonSource))comparisonSource=undefined;}
  catch(error){analysisError=message(error);console.error(error);showAnalysisError(analysisError);setText('summary','—');refreshStatus();return;}
  refreshDirectionStates(a);refreshFields(inspectorFields);
  const laser=activeLaser(view==='levels'?spectral:undefined),pol=laser?a.polarizations[laser.id]:undefined;
  manifoldFocus.setCurrent(spectral);root.dataset.view=view;
  if(view!=='space'&&spaceMount.isConnected)spaceMount.remove();
  if(view!=='dynamics'&&unbindPopulation){unbindPopulation();unbindPopulation=undefined;}
  if($('inspector-polarization'))$('inspector-polarization').innerHTML=polarizationBars(a,selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):laser);
  if(selection.kind==='laser'){const l=scene.lasers.find(l=>l.id===selection.id);if(l&&laserReadout.readout.root.isConnected){try{laserReadout.show(laserValue(l),scene);}catch{laserReadout.show(undefined,scene);}}}
  const field=withUnit(a.fieldMagnitudeG,'G',{digits:4});
  if(view==='levels'){
    $('selection-readout').hidden=false;$('population-toolbar').hidden=false;
    populationToolbar();
    const initial=referenceStateId(laser),transitions=(a.spectroscopy?.status==='unresolved'?[]:[...incidentTransitions(a,initial,laser?.id,'D1'),...incidentTransitions(a,initial,laser?.id,'D2')]).sort((x,y)=>y.strength*y.weight-x.strength*x.weight),snapshot=populationSnapshot();
    const clockGap=a.spectroscopy?.status==='unresolved'?NaN:(a.ground.find(state=>state.id==='ground-2-0')!.energyMHz-a.ground.find(state=>state.id==='ground-1-0')!.energyMHz);
    setText('summary',`B ${field} · 钟跃迁 ${Number.isFinite(clockGap)?withUnit(clockGap/1000,'GHz',{places:6}):'未定义'} · 32 态 Lindblad 主方程`);
    setText('levels-heading',a.spectroscopy?.status==='unresolved'?'裸态能级':laser?'背景能级':'总场能级');
    setText('levels-legend',levelsLegend(a,laser));
    beamChip.hidden=!laser;if(laser)setMiddle(beamChipName,laser.name,'cut');
    setText('levels-badge',a.spectroscopy?.status==='unresolved'?(laser?'裸态失谐 · 背景光移未确定':'静态光移未定义'):populationMode==='time'&&a.spectroscopy?.requiresDensityProjection?'当前基底 p(t) 未提供':laser&&!laser.enabled?'已停用':'');
    renderSelection(a,laser);
    levelsChart$.innerHTML=levelsChart(scene,a,laser,{line:spectral,selectedIds:markedComparison(laser),initialId:dynamics().initialStateId,populations:snapshot.populations,steadyPopulations:steadyPopulationsInView(a),steadyPopulationLabel:isDisplayedMean(currentResult(),a)?'mean':'limit',populationLabel:snapshot.label,viewportWidth:levelsChart$.clientWidth});
    const table=transitionTable(transitions);
    renderDetails('levels-details',`<details class="transition-disclosure"><summary>跃迁矩元 <span>${laser?esc(laser.name):spectral}</span></summary><div class="table-scroll"><table class="transition-table"><thead><tr><th>参考态 → 末态</th><th>分量</th><th>${esc(table.heading)}</th><th>Ω/2π / MHz</th><th title="相对 D2 stretched cycling 跃迁的偶极强度，并乘当前偏振权重">耦合强度</th></tr></thead><tbody>${table.rows}</tbody></table></div></details>${referenceEditor()}${solverDetails()}`);bindReferences();
  }else if(view==='dynamics'){setText('summary',`B ${field} · 32 态 Lindblad 主方程`);renderDynamics(a,allowSolve);}
  else{
    setText('summary',`B ${field}`);
    spaceViews.setWithheld('k',!laser||!pol);
    const spaceSelection:SpaceSelection=selection.kind==='none'?{kind:'none',id:''}:selection;
    const host=$('space-chart');if(!spaceMount.isConnected)host.replaceChildren(spaceMount);if(!spaceRenderer)spaceRenderer=renderSpace3D(spaceMount,scene,a,spaceSelection,{onViewChange:name=>{spaceView=name;spaceViews.setCurrent(name);}});else spaceRenderer.update(scene,a,spaceSelection);
    renderSpaceLegend(a);
  }
  paintStateBadges();renderCount++;renderedPopulationKey=dynamicsKey();root.dataset.populationFresh=String(!!currentResult());refreshStatus();root.dataset.renderCount=String(renderCount);root.dataset.energyRenderedAt=String(performance.now());
  if(focusHost&&focusAttribute&&focusValue)focusHost.querySelector<HTMLElement>(`[${focusAttribute}="${CSS.escape(focusValue)}"]`)?.focus({preventScroll:true});
  if(allowSolve&&!loadFailure&&view!=='space')requestSolve();
}

// ---------------------------------------------------------------- the scene as a document
function replaceScene(next:Scene,text:string):void{
  loadFailure='';solver.cancel();solverRevision++;clearTimeout(solverTimer);requestedKey='';requestedPhysicalKey='';requestedSteadyOnly=false;stoppedKey='';solverError='';solverBusy=false;controlsKey='';displayedResults.clear();
  referenceEditingId=null;referenceEditorNode=undefined;referenceNumber=undefined;disposeTrees($('levels-details'));$('levels-details').replaceChildren();$('selection-readout').dataset.selectionKey='';
  normalizeScene(next);scene=next;selectedStates=selectedStates.filter(id=>!id.startsWith('reference-')||scene.references?.some(r=>r.id===id));populationTimeIndex=-1;
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
  const example=exampleScene(key);view=example.view;spectral=example.line;replaceScene(example.scene,`已加载示例「${example.scene.name}」`);
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
/** One line of text (v3 FluentInputDialog), its commands in the workbench's words. */
async function askText(title:string,label:string,value:string):Promise<string|null>{
  const dialog=new Dialog(title),edit=element('input');edit.type='text';edit.spellcheck=false;edit.value=value;
  const line=element('div','dialog-field');line.append(element('label','field-label',label),edit);dialog.content.append(line);
  dialog.addAction('取消','cancel');const ok=dialog.addAction('确定','ok','primary');
  edit.addEventListener('keydown',event=>{if(event.key==='Enter')ok.click();});
  const answer=dialog.open();edit.focus();edit.select();return await answer==='ok'?edit.value:null;
}
async function renameSceneDialog():Promise<void>{
  const chosen=await askText('重命名场景','名称',scene.name);if(chosen===null)return;const name=chosen.trim();
  if(!name||name.length>100){note('场景名称应包含 1 至 100 个字符','warning');return;}scene.name=name;renderObjects();persist();refreshStatus();
}
async function renameSelected():Promise<void>{
  const object=selection.kind==='laser'?scene.lasers.find(l=>l.id===selection.id):scene.fields.find(f=>f.id===selection.id);if(!object)return;
  const chosen=await askText('重命名','名称',object.name);if(chosen===null)return;const name=chosen.trim();
  if(!name||name.length>64){note('名称应包含 1 至 64 个字符','warning');return;}object.name=name;setText('inspector-title',name);renderObjects();update();
}
function openModel(section:ModelSection,current?:{title:string;lines:string[]}):void{
  const dialog=new Dialog('模型、单位与适用范围');dialog.root.classList.add('model-dialog');
  const body=element('div','model-notes readout-text');body.tabIndex=0;body.setAttribute('aria-label','模型说明内容');
  body.innerHTML=(current?`<section class="model-current"><h3>${esc(current.title)}</h3>${current.lines.map(line=>`<p>${esc(line)}</p>`).join('')}</section>`:'')+modelNotes();
  dialog.content.append(body);dialog.addAction('关闭','close','primary');void dialog.open();
  requestAnimationFrame(()=>{const target=current?body.querySelector<HTMLElement>('.model-current'):body.querySelector<HTMLElement>(`#model-${section}`);if(target)body.scrollTop=target.offsetTop-body.offsetTop;});
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
root.addEventListener('atom-select-laser',event=>{const id=(event as CustomEvent<{id:string}>).detail?.id;if(scene.lasers.some(l=>l.id===id))select({kind:'laser',id});});root.addEventListener('atom-select-field',event=>{const id=(event as CustomEvent<{id:string}>).detail?.id;if(scene.fields.some(f=>f.id===id))select({kind:'field',id});});
root.addEventListener('click',event=>{
  const target=event.target as Element;
  if(target.closest('[data-export-solver-failures]')){download(JSON.stringify(failureRecords,null,2),'rb87-solver-failures.json');return;}
  if(root.classList.contains('scene-open')&&!target.closest('.scene-panel')){root.classList.remove('scene-open');syncLayout();}
  if(target.closest('[data-show-solver-details]')){showSolverDetails();return;}
  const beam=target.closest<HTMLElement>('[data-laser-id]');if(beam){select({kind:'laser',id:beam.dataset.laserId!});return;}
  const fieldItem=target.closest<HTMLElement>('[data-field-id]');if(fieldItem){select({kind:'field',id:fieldItem.dataset.fieldId!});return;}
  const state=target.closest<HTMLElement>('[data-state]');if(state){selectState(state.dataset.state!);return;}
  const tr=target.closest<HTMLElement>('[data-transition]');if(tr){const t=latestAnalysis.transitions.find(t=>t.id===tr.dataset.transition);if(t){const endpoints=transitionEndpoints(t,referenceStateId(scene.lasers.find(l=>l.id===t.laserId)));if(endpoints){comparisonSource=undefined;selectedStates=[endpoints.fromId,endpoints.toId];persist();renderResults(false);}}}
});
root.addEventListener('keydown',event=>{
  // Escape closes only the topmost overlay; an edit's own Escape (v3 Field) never reaches here.
  if(event.key==='Escape'&&!event.defaultPrevented&&!document.querySelector('dialog[open]')){
    if(root.classList.contains('inspector-open')){root.classList.remove('inspector-open');syncLayout();inspectorToggle.root.focus();event.preventDefault();return;}
    if(root.classList.contains('scene-open')){root.classList.remove('scene-open');syncLayout();toggleScene.root.focus();event.preventDefault();return;}
  }
  if(event.key!=='Enter'&&event.key!==' ')return;
  const el=(event.target as Element).closest<HTMLElement>('g[data-state],g[data-laser-id],tr[data-transition]');if(el){event.preventDefault();el.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
});
window.addEventListener('pagehide',()=>{if(saveState==='pending')saveSceneNow();cancelObsoleteSolve();});
window.addEventListener('pageshow',event=>{if(event.persisted)renderResults();});
window.addEventListener('resize',syncLayout);
let chartWidth=0,resizeFrame=0;new ResizeObserver(entries=>{const width=Math.round(entries[0]?.contentRect.width??0);if(width!==chartWidth){chartWidth=width;cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(latestAnalysis&&(view==='levels'||view==='dynamics'))renderResults(false);});}}).observe(tabs.stack);

// ---------------------------------------------------------------- start
normalizeScene(scene);if(baseline===null&&!unreadableScene&&!loadFailure){let stored:string|null=null;try{stored=localStorage.getItem(STORAGE);}catch{}if(stored===null)markBaseline();}
syncLayout();latestAnalysis=analyze(scene);renderInitial();renderObjects();renderInspector();renderResults();
if(loadFailure)refreshStatus();
