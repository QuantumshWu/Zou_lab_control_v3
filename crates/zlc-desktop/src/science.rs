//! The scientific worker: the native Figure documents -- the live board's and
//! the Figure Viewer's (v2's two windows, each its own document). Their panels
//! prepare and present on the process's one bounded worker pool; no
//! interpreter is involved.
use std::{collections::{BTreeMap,BTreeSet,HashSet}, sync::{Arc,mpsc}};
use serde_json::{Value,json};
use zlc_core::runtime::{EventIdentity,Read,ReadKind,PublicationKey,Runtime};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::{error_text,DocumentRead,FigureDocument};
use zlc_plot::figure::{read_document,ArchiveSaver};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::Source;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer,ImagePointOverlay,SessionServices};
use crate::overlay::LiveOverlays;
use crate::panels::{PanelInput,BindingAuthority};
use crate::derivation::{Derivations,Effect};
use crate::archive::{ArchiveItem,DecodedArchive,MountedArchive};

pub struct Job { pub ticket:u64, pub command: Value, pub inputs: Vec<PanelInput>, pub authorities:BTreeMap<String,BindingAuthority>,
    pub intents:BTreeMap<String,BindingAuthority>, pub archive:Option<Arc<MountedArchive>>,
    /// Which document: 0 the live board's, n Figure Viewer n's.
    pub document: u32 }

/// One composed RGBA8 front, shared (never copied) up to the WebView slot.
pub type Front = zlc_plot::session::RgbaFront;

/// A Runtime read as the document sees it: the planes viewed in place.
#[derive(Clone)]
pub struct HostRead { pub read: Read, source: Arc<Source>, overlay: Option<Option<Arc<ImagePointOverlay>>> }

impl HostRead {
    pub fn new(read: Read) -> Result<Self,String> {
        let source = Source::from_read(&read).map_err(|error| error.to_string())?;
        Ok(Self { read, source: Arc::new(source), overlay: None })
    }
}

impl DocumentRead for HostRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { self.read.provenance().to_string() }
    fn image_overlay(&self) -> Option<Option<Arc<ImagePointOverlay>>> { self.overlay.clone() }
}

pub struct Completion {
    pub ticket:u64,
    pub document: u32,
    pub request: u64,
    pub target: Option<String>,
    pub authorities: BTreeMap<String,BindingAuthority>,
    pub intents:BTreeMap<String,BindingAuthority>,
    pub action:String,
    pub decision:Option<mpsc::Sender<PresentationDecision>>,
    pub sync: bool,
    pub publications: Vec<(String,PublicationKey)>,
    pub cohorts:Vec<Vec<String>>,
    pub imported:Option<DecodedArchive>,
    pub result: Result<(Value, BTreeMap<String, Front>), String>,
}

#[derive(Default)]
pub struct PresentationDecision { pub targets:BTreeSet<String>, pub fronts:BTreeSet<String> }

/// A live panel's region changed (v2 `_route_exact_panel_selection`): its document
/// (None: removed), the signal it was drawn on and the run-time facts of that read
/// (v2 `_selection_context`), for the producer of the signal to read.
pub struct SelectionNote { pub panel: String, pub signal: String, pub selection: Option<Value>, pub context: Value }

/// v2 `_selection_context`: the read's run record device snapshots (one camera's
/// facts spread at the top) and each axis's unit.
fn selection_context(read: &Read) -> Value {
    let snapshots = read.anchor().run().metadata["device_snapshots"].clone();
    let schema = read.schema();
    let units: serde_json::Map<String, Value> = [zlc_core::schema::DomainRole::Repeat, zlc_core::schema::DomainRole::Point, zlc_core::schema::DomainRole::Cell].into_iter()
        .flat_map(|role| schema.domain(role).axes().iter().map(|axis| (axis.id().to_owned(), json!(axis.unit().unwrap_or("1")))).collect::<Vec<_>>()).collect();
    let mut context = json!({"device_snapshots": if snapshots.is_object() { snapshots.clone() } else { json!({}) }, "axis_units": units});
    if let Some(only) = snapshots.as_object().filter(|items| items.len() == 1).and_then(|items| items.values().next()).and_then(Value::as_object) {
        for (key, value) in only { context[key] = value.clone(); }
    }
    context
}

/// v2 _NumberHandler: a plot's int-or-float control is a plain box that keeps
/// the number's own spelling (55.0 stays 55.0), never a spin box.
pub fn spell_numbers(panel: &mut Value) {
    for control in panel["controls"].as_array_mut().into_iter().flatten() {
        if control["section"] != "display" || control["kind"] != "number" { continue; }
        control["lossless"] = json!(true);
        let text = match &control["value"] {
            Value::Number(number) if number.is_f64() => number.as_f64().map(zlc_core::pyfmt::repr),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        };
        if let Some(text) = text { control["text"] = json!(text); }
    }
}

struct PendingEffects {effects:Vec<Effect>,intents:BTreeMap<String,BindingAuthority>,authorities:BTreeMap<String,BindingAuthority>}

/// One document and the reads it holds: the live board's (0) and each Figure Viewer's.
struct Document { figure: FigureDocument<HostRead>, reads: BTreeMap<(Arc<str>,usize),HostRead> }

pub struct Science { documents: BTreeMap<u32, Document>, services: SessionServices, current: u32,
    derivatives: Derivations, derivative_errors:BTreeMap<String,String>, overlays:LiveOverlays, overlay_errors:BTreeMap<String,String>,
    pending:Option<PendingEffects>, report:Arc<dyn Fn(String)+Send+Sync>, selected:Arc<dyn Fn(SelectionNote)+Send+Sync>,
    /// Each history panel's shots held at its last read.
    held:BTreeMap<String,usize> }

impl Science {
    pub fn new(runtime: Runtime, report: Arc<dyn Fn(String)+Send+Sync>, notify:Arc<dyn Fn()+Send+Sync>, selected:Arc<dyn Fn(SelectionNote)+Send+Sync>) -> Self {
        let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
            defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
        Self { documents: BTreeMap::new(), services, current: 0,
            derivatives: Derivations::new(runtime,report.clone(),notify), derivative_errors: BTreeMap::new(), overlays: LiveOverlays::default(),
            overlay_errors: BTreeMap::new(), pending: None, report, selected, held: BTreeMap::new() }
    }

    fn open(&self, path: &str) -> Result<DecodedArchive,String> {
        let opened = read_document(path, self.services.fit.clone(), self.services.defaults.clone(), self.services.registry.clone())
            .map_err(|error| error_text(&error))?;
        Ok(DecodedArchive { path: opened.path, info: opened.info, items: opened.items.into_iter().map(|item| ArchiveItem {
            key: item.key, plane: item.plane, target: item.target, overlay: item.overlay, overlay_plane: item.overlay_plane }).collect() })
    }

    pub fn execute(&mut self, job: &Job) -> Completion {
        assert!(self.pending.is_none(),"Previous presentation requires its acceptance decision");
        let request = job.command["id"].as_u64().unwrap_or(0);
        if job.command["action"]=="open" {
            let (imported,result)=match self.open(job.command["path"].as_str().unwrap_or("")) {
                Ok(value)=>(Some(value),Ok((Value::Null,BTreeMap::new()))),Err(error)=>(None,Err(error))};
            return Completion {ticket:job.ticket,document:job.document,request,target:None,authorities:job.authorities.clone(),intents:job.intents.clone(),
                action:"open".into(),decision:None,sync:false,publications:Vec::new(),cohorts:Vec::new(),imported,result};
        }
        if job.command["action"]=="drop_document" {
            // A closed Figure Viewer's document goes, with its sessions and reads.
            self.documents.remove(&job.document);
            self.current = job.document;
            self.pending=Some(PendingEffects{effects:Vec::new(),intents:BTreeMap::new(),authorities:BTreeMap::new()});
            return Completion {ticket:job.ticket,document:job.document,request,target:None,authorities:BTreeMap::new(),intents:BTreeMap::new(),
                action:"drop_document".into(),decision:None,sync:false,publications:Vec::new(),cohorts:Vec::new(),imported:None,result:Ok((Value::Null,BTreeMap::new()))};
        }
        self.derivatives.sync_sources();
        self.derivatives.set_published(&job.command["published_outputs"]);
        let action = job.command["action"].as_str().unwrap_or("").to_owned();
        let _whole = crate::trace::span("job.execute",||json!({"job":job.ticket,"action":action}));
        let mut publications=Vec::new();
        let mut cohorts=Vec::new();
        let mut effects=Vec::new();
        let result = self.run(job, &mut publications, &mut cohorts, &mut effects);
        // v2 panel_selection_output_catalog: what a region on each panel's surface would publish.
        let document = self.documents.get(&self.current);
        let catalog = |id: &str| document.and_then(|document| document.figure.session(id))
            .and_then(|session| session.lock().unwrap_or_else(|error| error.into_inner()).selection_subject().ok())
            .map(|subject| zlc_plot::roi::selection_outputs(&subject));
        let result = result.map(|(mut state,fronts)| {
            if let Some(items) = state["panels"].as_array_mut() {
                for item in items {
                    if let Some(outputs) = item["id"].as_str().and_then(catalog) { item["selection_outputs"] = json!(outputs); }
                    if let Some(error) = item["id"].as_str().and_then(|id|self.derivative_errors.get(id).or(self.overlay_errors.get(id))) {item["error"] = json!(error);}
                }
            }
            (state,fronts)
        });
        self.pending=Some(PendingEffects{effects,intents:job.intents.clone(),authorities:job.authorities.clone()});
        if matches!(action.as_str(),"close"|"close_panel") && result.is_ok() { if let Some(document) = self.documents.get_mut(&self.current) { document.reads.clear(); } }
        Completion { ticket:job.ticket, document:job.document, request, target:job.command["panel"].as_str().map(str::to_owned),
            authorities:job.authorities.clone(), sync:action == "sync_panels", intents:job.intents.clone(),action,decision:None,
            publications, cohorts, imported:None, result }
    }

    fn run(&mut self, job: &Job, publications: &mut Vec<(String,PublicationKey)>, cohorts: &mut Vec<Vec<String>>,
           effects: &mut Vec<Effect>) -> Result<(Value, BTreeMap<String, Front>), String> {
        self.current = job.document;
        let at = self.current;
        let services = self.services.clone();
        self.documents.entry(at).or_insert_with(|| Document { figure: FigureDocument::new(services, Some(Arc::new(ArchiveSaver))), reads: BTreeMap::new() });
        let binding_span = crate::trace::span("native.inputs",||json!({"job":job.ticket}));
        let mut command = job.command.clone();
        let mut overlays = None;
        if let Some(archive)=&job.archive {
            command["action"]=json!("sync_panels");
            command["reset_document"]=json!(true);
            command["initial_document"]=json!({"path":archive.path(),"info":archive.info()});
            let mut items = BTreeMap::new();
            for item in archive.items() {
                let mut overlay=item.overlay.clone();
                if let Some(read)=archive.overlay_read(item)? {
                    let geometry=overlay.as_ref().ok_or("Measured status has no overlay geometry")?;
                    let status=Source::from_read(&read).map_err(|error|error.to_string())?;
                    overlay=Some(Arc::new(geometry.with_status(Some(Arc::new(status))).map_err(|error|error_text(&error))?));
                }
                items.insert(item.panel.clone(),overlay);
            }
            overlays=Some(items);
        }
        let sync = command["action"] == "sync_panels";
        let mut reads = BTreeMap::new();
        if !job.inputs.is_empty() || sync {
            let mut used = BTreeSet::new();
            let mut inputs = Vec::with_capacity(job.inputs.len());
            for input in &job.inputs {
                inputs.push(json!({"panel":input.panel,"kind":input.kind,"window":input.window,"signal":input.signal.as_ref(),
                    "epoch":input.epoch,"reason":input.reason,"waiting":input.waiting}));
                let Some(source) = &input.read else { continue };
                let key = (input.signal.clone(),match source.kind() {ReadKind::Window(window)=>window.width,_=>0});
                let held = &mut self.documents.get_mut(&at).expect("made").reads;
                if !held.get(&key).is_some_and(|cached|cached.read.same_read(source)) {
                    held.insert(key.clone(),HostRead::new(source.clone())?);
                }
                let mut read = held[&key].clone();
                // A saved Figure keeps the layer it was saved with; a data copy's preview, the one its copy kept.
                if job.document != 0 { if let Some(layer) = &input.layer { read.overlay = Some(Some(layer.clone())); } }
                if job.document == 0 {
                    // A live image carries its point layer (v2 ImageFrame(snapshot, overlay)).
                    // A point review draws its own candidates.
                    read.overlay = Some(match (&input.layer, self.overlays.layer(&input.panel, source, input.overlay.as_ref())) {
                        (Some(layer), _) => Some(layer.clone()),
                        (None, Ok(layer)) => { self.overlay_errors.remove(&input.panel); layer }
                        (None, Err(error)) => { self.overlay_errors.insert(input.panel.clone(), format!("Overlay: {error}")); None }
                    });
                }
                reads.insert(input.panel.clone(),read);
                used.insert(key);
            }
            if sync {
                self.documents.get_mut(&at).expect("made").reads.retain(|key,_|used.contains(key));
            }
            if sync && job.document == 0 {
                let shown: BTreeSet<&str> = job.inputs.iter().filter(|input| input.read.is_some()).map(|input| input.panel.as_str()).collect();
                self.overlays.retain(|panel| shown.contains(panel));
                self.overlay_errors.retain(|panel, _| shown.contains(panel.as_str()));
            }
            command["inputs"] = json!(inputs);
        }
        drop(binding_span);
        // v2 counts a history by the shots it holds (its window fills as they arrive);
        // a panel inside its update interval keeps the read it shows.
        let kept: BTreeSet<&str> = command["held"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        for (panel,read) in reads.iter().filter(|(panel, _)| !kept.contains(panel.as_str())) {
            if let ReadKind::Window(window) = read.read.kind() { self.held.insert(panel.clone(), window.entries().len()); }
        }
        let publisher=self.derivatives.publisher(job.authorities.clone());
        let mut execution = {
            let _scientific = crate::trace::span("science.execute",||json!({"job":job.ticket}));
            self.documents.get_mut(&at).expect("made").figure.execute(&command.to_string(), reads, Some(&publisher), overlays).map_err(|error|error_text(&error))?
        };
        if sync {
            let mut groups:Vec<(HashSet<EventIdentity>,Vec<String>)>=Vec::new();
            for (panel,read) in &execution.admitted {
                let roots=read.read.shot_roots();
                if let Some((_,members))=groups.iter_mut().find(|(candidate,_)|*candidate==roots) {members.push(panel.clone());}
                else {groups.push((roots,vec![panel.clone()]));}
            }
            *cohorts=groups.into_iter().map(|(_,members)|members).collect();
        }
        for item in execution.state["panels"].as_array_mut().into_iter().flatten() {
            let Some(count) = item["id"].as_str().and_then(|id| self.held.get(id)) else { continue };
            for axis in item["structure"][1].as_array_mut().into_iter().flatten() {
                if axis[0] == "source index" { axis[1] = json!(count); }
            }
        }
        for (panel,read) in &execution.presented {
            crate::trace::point("job.presented_source",||json!({"job":job.ticket,"panel":panel,"source":crate::trace::key(read.read.key())}));
            publications.push((panel.clone(),read.read.key().clone()));
        }
        for effect in execution.effects {
            if let Effect::Roi{panel,source,..}|Effect::Fit{panel,source,..}=&effect {
                if job.inputs.iter().find(|input|input.panel==*panel).is_some_and(|input|input.signal!=source.read.key().signal) {
                    return Err("ValueError: Scientific plan input differs from its authored signal binding".into());
                }
            }
            effects.push(effect);
        }
        Ok((execution.state, execution.fronts))
    }

    pub fn finish_presentation(&mut self, decision:PresentationDecision) {
        let pending=self.pending.take();
        if let Some(document) = self.documents.get_mut(&self.current) { document.figure.finish_presentation(&decision.targets,&decision.fronts); }
        let Some(pending)=pending else {return};
        for effect in pending.effects {
            let panel=effect.panel().to_owned();
            if !decision.targets.contains(&panel) {continue;}
            if let Effect::Selection{source,selection,..}=effect {
                // The live board's regions only: an opened Figure's data has no producer to set.
                if self.current==0 { (self.selected)(SelectionNote{panel,signal:source.read.key().signal.to_string(),selection,context:selection_context(&source.read)}); }
                continue;
            }
            let authority=if matches!(&effect,Effect::Fit{..}) {
                // A manual/configure result is public only after its complete
                // front accepts, exactly like v2's deferred FitEvent.
                if !decision.fronts.contains(&panel) {continue;}
                pending.authorities.get(&panel).cloned()
            }else{pending.intents.get(&panel).cloned()};
            match self.derivatives.apply(effect,authority) {
                Ok(())=>{self.derivative_errors.remove(&panel);},
                Err(error)=>{(self.report)(format!("{panel}: {error}"));self.derivative_errors.insert(panel,error);},
            }
        }
    }
}

impl Drop for Science {
    fn drop(&mut self) { self.derivatives.clear(); }
}
