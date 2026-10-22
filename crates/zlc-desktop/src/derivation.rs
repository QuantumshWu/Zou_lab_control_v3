//! Derived outputs have one Runtime owner. ROI follows latest; a fit is an
//! admitted plot result, not an independent subscriber solving every source shot.
use std::{collections::{BTreeMap,BTreeSet}, sync::{Arc,Weak,Mutex,atomic::{AtomicBool,Ordering}},time::{Duration,Instant}};
use serde_json::{Value,json};
use zlc_core::{schema::DatasetSchema,runtime::{Runtime,Read,ReadKind,PublicationKey,Generation,GenerationId,
    Declaration,Storage,Bundle,Input,Subscription,DeliveryPolicy,DeliveryKind,ReadMode,ReadRequest,Start,RuntimeError}};
use zlc_core::store::Snapshot;
use zlc_plot::roi::RoiPlan;
use crate::panels::BindingAuthority;
use crate::science::HostRead;

/// What a document command leaves for the Runtime: ROI plans, fit outputs, removals.
pub type Effect = zlc_plot::document::Effect<HostRead>;

/// A panel's derived signals keep v2's spelling: `@logic/<panel>/roi_frame`,
/// `@logic/<panel>/roi_mean`, ... for its region, `@logic/<panel>/<parameter>` for its fit.
pub fn output_prefix(panel:&str)->String {format!("@logic/{panel}/")}

struct Output {
    runtime:Runtime,owner:String,prefix:String,generation:Option<Generation>,coherent:bool,
    source_identity:Option<(GenerationId,Arc<str>)>,records:bool,definition:Value,
    last_source:Option<PublicationKey>,schemas:BTreeMap<String,Arc<DatasetSchema>>,
}
impl Output {
    fn new(runtime:&Runtime,owner:String,prefix:String,coherent:bool)->Self {
        Self {runtime:runtime.clone(),owner,prefix,generation:None,coherent,source_identity:None,
            records:false,definition:Value::Null,last_source:None,schemas:BTreeMap::new()}
    }
    fn publish(&mut self,source:&Read,values:BTreeMap<String,Snapshot>,records:bool,metadata:Value,
               authority:&BindingAuthority)->Result<bool,String> {
        let _span=crate::trace::span("derived.commit",||json!({"owner":self.owner,"source":crate::trace::key(source.key()),"records":records}));
        let allowed=authority.lock().unwrap_or_else(|error|error.into_inner());
        if !*allowed {return Ok(false);}
        let records=records || source.key().ordinal.is_none();
        let schemas=values.iter().map(|(name,data)|(name.clone(),data.schema().clone())).collect::<BTreeMap<_,_>>();
        let identity=(source.key().generation,source.key().signal.clone());
        let ended=self.generation.as_ref().is_some_and(|token|
            self.runtime.outcome(token).map_or(true,|outcome|outcome.is_some()));
        let reconfigured=self.last_source.as_ref().is_some_and(|last|
            last.generation==source.key().generation && last.revision>=source.key().revision);
        if self.source_identity.as_ref()!=Some(&identity) || self.schemas!=schemas || self.records!=records
            || self.definition!=metadata || self.generation.is_none() || ended || reconfigured {
            if let Some(token)=&self.generation {
                match self.runtime.seal(token,json!({"reason":"replaced"})) {
                    Ok(())|Err(RuntimeError::UnknownGeneration)=>{},Err(error)=>return Err(error.to_string()),
                }
            }
            let outputs=schemas.iter().map(|(name,schema)|Declaration {signal:format!("{}{}",self.prefix,name).into(),
                schema:schema.clone(),storage:if records {Storage::Snapshot}else{Storage::Latest}}).collect();
            self.generation=Some(match self.runtime.begin_derived(self.owner.clone(),outputs,metadata.clone(),source.key(),self.coherent) {
                Ok(token)=>token,Err(RuntimeError::SourceSuperseded)=>return Ok(false),Err(error)=>return Err(error.to_string()),
            });
            self.source_identity=Some(identity);self.schemas=schemas;self.records=records;
            self.definition=metadata.clone();self.last_source=None;
        }
        let inputs=values.into_iter().map(|(name,snapshot)| {
            let signal=format!("{}{}",self.prefix,name).into();
            if records {return Ok(Input::Dataset {signal,ordinal:source.key().ordinal,snapshot});}
            let part=snapshot.segments().next().filter(|part|snapshot.segment_count()==1
                && part.origin()==[0,0] && part.data().schema()==snapshot.schema())
                .ok_or("A sample output must contain one complete event block")?;
            Ok(Input::Event {signal,ordinal:source.key().ordinal.ok_or("A sample requires an actual source ordinal")?,
                data:part.data().clone(),origin:None})
        }).collect::<Result<Vec<_>,String>>()?;
        // A history view has one current shot anchor, not N simultaneous roots.
        let accepted=match self.runtime.commit(self.generation.as_ref().unwrap(),&Bundle {inputs,metadata,parents:vec![source.anchor().clone()]}) {
            Ok(value)=>value,Err(RuntimeError::SourceSuperseded)=>return Ok(false),Err(error)=>return Err(error.to_string()),
        };
        crate::trace::point("derived.accepted",||json!({"owner":self.owner,"source":crate::trace::key(source.key()),
            "outputs":accepted.publications.iter().map(|event|crate::trace::key(event.key())).collect::<Vec<_>>()}));
        self.last_source=Some(source.key().clone());Ok(true)
    }
    fn seal(&self,error:Option<&str>) {
        if let Some(token)=&self.generation {let _=self.runtime.seal(token,json!({"error":error}));}
    }
}
impl Drop for Output {
    fn drop(&mut self) {if let Some(token)=self.generation.take() {let _=self.runtime.retire(&token);}}
}

/// A panel's region outputs, published under its own owner.
fn roi_output(runtime:&Runtime,panel:&str)->Output {Output::new(runtime,format!("{panel}/roi"),output_prefix(panel),true)}

struct RoiWorker {cancel:Arc<AtomicBool>,subscription:Weak<Subscription>,worker:Option<std::thread::JoinHandle<Output>>}
impl RoiWorker {
    /// The panel's region derived from every new read of its source: the outputs the
    /// board publishes (`published`, read at each answer; a change of it answers the
    /// last read again, v2 `configure_outputs`).
    fn start(runtime:&Runtime,panel:String,source:Read,plan:Arc<RoiPlan>,metadata:Value,published:Arc<FitOutputs>,
             authority:BindingAuthority,report:Arc<dyn Fn(String)+Send+Sync>,notify:Arc<dyn Fn()+Send+Sync>,
             previous:Option<Output>)->Result<Self,String> {
        let owner=format!("{panel}/roi");
        let mode=match source.kind() {ReadKind::Event(_)=>ReadMode::Event,ReadKind::Canonical(_)=>ReadMode::Dataset,
            ReadKind::Window(window)=>ReadMode::Window(window.width)};
        let subscription=Arc::new(runtime.follow_reads(source.anchor().run().owner.clone(),
            vec![ReadRequest {signal:source.key().signal.clone(),mode}],DeliveryPolicy::Latest,Start::Current)
            .map_err(|error|error.to_string())?);
        let cancel=Arc::new(AtomicBool::new(false));let stopped=cancel.clone();let wake=Arc::downgrade(&subscription);
        let source_schema=source.schema().clone();
        let mut output=previous.unwrap_or_else(||roi_output(runtime,&panel));
        let runtime=runtime.clone();
        // v2 routes a region's processor the newest source read at each display beat
        // (BoardScheduler.on_tick freezes the plane): one answer per base interval, on the newest read.
        let beat=Duration::from_millis(zlc_workbench::INTERVALS[0]);
        let worker=std::thread::Builder::new().name(format!("{owner}-processor")).spawn(move || {
            let result=(||->Result<(),String> {
                let mut initial=Some(source);
                let (mut last,mut answered,mut answered_at):(Option<Read>,Option<Vec<&str>>,Option<Instant>)=(None,None,None);
                // The newest source read not answered yet, waiting for the beat.
                let mut newest:Option<Read>=None;
                loop {
                    if stopped.load(Ordering::Acquire) || !*authority.lock().unwrap_or_else(|error|error.into_inner()) {return Ok(());}
                    let wanted:Vec<&str>={
                        let published=published.published.lock().unwrap_or_else(|error|error.into_inner());
                        plan.outputs().iter().map(|(name,_)|*name).filter(|name|wanted(&published,&panel,name)).collect()
                    };
                    // First answer the exact displayed read on which the box was
                    // drawn; only afterwards catch up to the latest publication.
                    let current=if let Some(read)=initial.take() {read} else {
                        let due=answered_at.map_or(Duration::ZERO,|at|(at+beat).saturating_duration_since(Instant::now()));
                        match subscription.next(if newest.is_some() {due} else {Duration::from_secs(3600)}) {
                            Ok(Some(delivery))=>{
                                if delivery.kind!=DeliveryKind::Data {
                                    if output.last_source.as_ref().is_some_and(|last|last.generation==delivery.generation) {output.seal(None);}
                                    continue;
                                }
                                let read=delivery.reads[0].clone();
                                if output.last_source.as_ref().is_some_and(|last|last.generation==read.key().generation && last.revision>=read.key().revision) {continue;}
                                newest=Some(read);
                                continue;
                            }
                            // The beat: the newest read is answered.
                            Ok(None) if newest.is_some() && answered_at.is_none_or(|at|at.elapsed()>=beat) => newest.take().expect("a newest read"),
                            // Woken: the published outputs may have changed; the last read is answered again.
                            Ok(None)=>match &last { Some(read) if newest.is_none() && answered.as_ref()!=Some(&wanted)=>read.clone(), _=>continue },
                            Err(RuntimeError::Closed)=>return Ok(()),Err(error)=>return Err(error.to_string()),
                        }
                    };
                    if current.schema()!=&source_schema {notify();return Ok(());}
                    last=Some(current.clone());answered=Some(wanted.clone());answered_at=Some(Instant::now());
                    // Nothing published: the region's signals go (their names are free again).
                    if wanted.is_empty() {output=roi_output(&runtime,&panel);continue;}
                    let _span=crate::trace::span("roi.apply",||json!({"owner":owner,"source":crate::trace::key(current.key())}));
                    // Native cut and reductions, no interpreter: every output in the plan's schemas.
                    let data=plan.derive(&current,&wanted).map_err(|error|format!("{}: {error}",error.kind.python_name()))?;
                    if stopped.load(Ordering::Acquire) {return Ok(());}
                    if !output.publish(&current,data,current.is_complete_dataset(),metadata.clone(),&authority)?
                        && !*authority.lock().unwrap_or_else(|error|error.into_inner()) {return Ok(());}
                }
            })();
            if let Err(error)=&result {report(format!("{owner}: {error}"));}
            output.seal(result.as_ref().err().map(String::as_str));output
        }).map_err(|error|error.to_string())?;
        Ok(Self {cancel,subscription:wake,worker:Some(worker)})
    }
    fn finish(&mut self)->Result<Option<Output>,String> {
        self.cancel.store(true,Ordering::Release);
        if let Some(subscription)=self.subscription.upgrade() {let _=subscription.wake();}
        self.worker.take().map(|worker|worker.join().map_err(|_|"ROI processor panicked".into())).transpose()
    }
}
impl Drop for RoiWorker {fn drop(&mut self) {if let Err(error)=self.finish() {eprintln!("Processor cleanup: {error}");}}}

/// Whether a panel publishes `name` (the board's `published_outputs`; absent = yes).
/// A panel's Edit, or a point review, publishes nothing: its frozen fit and ROI are its own.
fn wanted(published:&Value,panel:&str,name:&str)->bool {!crate::panels::is_edit(panel) && !crate::panels::is_review(panel) && published[panel][name]!=Value::Bool(false)}

struct FitOutputs {runtime:Runtime,outputs:Mutex<BTreeMap<String,Output>>,published:Mutex<Value>}
impl FitOutputs {
    fn sync_sources(&self) {
        let sources=self.runtime.signals();
        self.outputs.lock().unwrap_or_else(|error|error.into_inner()).retain(|_,output| {
            let Some((generation,signal))=&output.source_identity else{return true;};
            let Some(source)=sources.iter().find(|item|item.signal==*signal && item.generation.id()==*generation) else{return false;};
            if source.sealed {
                let current=self.runtime.read_views(&[(source.generation.clone(),signal.clone(),None)]).remove(0);
                if current.as_ref().is_ok_and(|read|output.last_source.as_ref()==Some(read.key())) {output.seal(None);}
            }
            true
        });
    }
    fn publish(&self,panel:&str,source:&Read,mut values:BTreeMap<String,Snapshot>,records:bool,metadata:Value,
               authority:&BindingAuthority)->Result<bool,String> {
        let mut outputs=self.outputs.lock().unwrap_or_else(|error|error.into_inner());
        {
            let published=self.published.lock().unwrap_or_else(|error|error.into_inner());
            values.retain(|name,_|wanted(&published,panel,name));
        }
        // Nothing this panel publishes: its fit signals go away.
        if values.is_empty() {outputs.remove(panel);return Ok(false);}
        let output=outputs.entry(panel.into()).or_insert_with(||Output::new(&self.runtime,format!("{panel}/fit"),output_prefix(panel),false));
        let accepted=output.publish(source,values,records,metadata,authority)?;
        if self.runtime.signals().iter().any(|item|item.signal==source.key().signal && item.generation.id()==source.key().generation && item.sealed) {
            output.seal(None);
        }
        Ok(accepted)
    }
}

/// Admission-scoped bridge: numeric publish precedes raster; the same batch's
/// follower can read that exact result without another display beat.
pub struct FitPublisher {outputs:Arc<FitOutputs>,authorities:BTreeMap<String,BindingAuthority>,published:Mutex<BTreeSet<String>>}
impl zlc_plot::document::FitPublisher<HostRead> for FitPublisher {
    fn publish(&self,panel:&str,read:&HostRead,values:BTreeMap<String,Snapshot>,records:bool,metadata:Value)->Result<(),String> {
        let Some(authority)=self.authorities.get(panel) else{return Ok(());};
        let names=values.keys().map(|name|format!("{}{name}",output_prefix(panel))).collect::<Vec<_>>();
        if self.outputs.publish(panel,&read.read,values,records,metadata,authority)? {
            self.published.lock().unwrap_or_else(|error|error.into_inner()).extend(names);
        }
        Ok(())
    }
    fn read(&self,signal:&str,window:usize,records:bool)->Result<Option<HostRead>,String> {
        if !self.published.lock().unwrap_or_else(|error|error.into_inner()).contains(signal) {return Ok(None);}
        let runtime=&self.outputs.runtime;
        let Some(source)=runtime.signals().into_iter().find(|source|source.signal.as_ref()==signal) else{return Ok(None);};
        let width=(source.history_capable && (window>1 || records)).then_some(window);
        let _lease=if width.is_some_and(|width|width>1) {Some(runtime.retain_history(&source.generation,signal,window)
            .map_err(|error|error.to_string())?)}else{None};
        let read=runtime.read_views(&[(source.generation,signal.into(),width)]).remove(0).map_err(|error|error.to_string())?;
        HostRead::new(read).map(Some)
    }
}

pub struct Derivations {runtime:Runtime,workers:BTreeMap<String,RoiWorker>,dependencies:BTreeMap<String,String>,
    fits:Arc<FitOutputs>,report:Arc<dyn Fn(String)+Send+Sync>,notify:Arc<dyn Fn()+Send+Sync>}
impl Derivations {
    pub fn new(runtime:Runtime,report:Arc<dyn Fn(String)+Send+Sync>,notify:Arc<dyn Fn()+Send+Sync>)->Self {
        Self {fits:Arc::new(FitOutputs{runtime:runtime.clone(),outputs:Mutex::new(BTreeMap::new()),published:Mutex::new(Value::Null)}),runtime,
            workers:BTreeMap::new(),dependencies:BTreeMap::new(),report,notify}
    }
    pub fn publisher(&self,authorities:BTreeMap<String,BindingAuthority>)->FitPublisher {
        FitPublisher{outputs:self.fits.clone(),authorities,published:Mutex::new(BTreeSet::new())}
    }
    pub fn sync_sources(&self) {self.fits.sync_sources();}
    /// The board's choice of which ROI/fit outputs each panel publishes.
    pub fn set_published(&self,published:&Value) {
        let mut held=self.fits.published.lock().unwrap_or_else(|error|error.into_inner());
        if *held==*published {return;}
        *held=published.clone();
        drop(held);
        // Each region answers again with what it now publishes.
        for worker in self.workers.values() {if let Some(subscription)=worker.subscription.upgrade() {let _=subscription.wake();}}
    }
    /// A panel's ROI reads `source`; refuse a chain of panels that reaches itself.
    fn dependency(&mut self,panel:&str,source:&str)->Result<(),String> {
        let producer=|signal:&str|signal.strip_prefix("@logic/").and_then(|rest|rest.rsplit_once('/')).map(|(producer,_)|producer.to_owned());
        let mut next=producer(source);
        while let Some(current)=next {
            if current==panel {return Err("A derived signal cannot depend on itself".into());}
            next=self.dependencies.get(&current).and_then(|input|producer(input));
        }
        self.dependencies.insert(panel.into(),source.into());Ok(())
    }
    pub fn apply(&mut self,effect:Effect,authority:Option<BindingAuthority>)->Result<(),String> {
        match effect {
            Effect::Remove{panel,output}=>{
                let owner=format!("{panel}/{output}");self.workers.remove(&owner);
                if output=="roi" {self.dependencies.remove(&panel);}
                if output=="fit" {self.fits.outputs.lock().unwrap_or_else(|error|error.into_inner()).remove(&panel);}
            },
            Effect::Roi{panel,source,plan,metadata}=>{
                let Some(authority)=authority else{return Ok(());};
                if !*authority.lock().unwrap_or_else(|error|error.into_inner()) {return Ok(());}
                let owner=format!("{panel}/roi");
                self.dependency(&panel,&source.read.key().signal)?;
                let previous=self.workers.remove(&owner).map(|mut worker|worker.finish()).transpose()?.flatten();
                let worker=RoiWorker::start(&self.runtime,panel,source.read,plan,metadata,self.fits.clone(),authority,self.report.clone(),self.notify.clone(),previous)?;
                self.workers.insert(owner,worker);
            },
            Effect::Fit{panel,source,values,records,metadata}=>{
                if let Some(authority)=authority {self.fits.publish(&panel,&source.read,values,records,metadata,&authority)?;}
            },
            // A region's meaning to its producer is the application's (Science routes it).
            Effect::Selection{..}=>{},
        }Ok(())
    }
    pub fn clear(&mut self) {self.workers.clear();self.dependencies.clear();self.fits.outputs.lock().unwrap_or_else(|error|error.into_inner()).clear();}
}
