//! One atomic publication owner. Reads own immutable data; history is a lease,
//! not a plot feature, and display coalescing never rewrites scientific order.
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, Weak, atomic::{AtomicBool, Ordering}, mpsc::{self, Receiver, SyncSender}};
use std::time::{Duration, Instant};
use serde_json::Value;
use uuid::Uuid;
use crate::dataset::DataBlock;
use crate::schema::{Axis, CodeBase, CodeMapping, Coordinates, DatasetSchema, Domain, DomainRole};
use crate::store::{DatasetStore, Placement, Snapshot};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GenerationId(Uuid);
impl GenerationId { pub fn as_uuid(&self) -> Uuid { self.0 } }

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PublicationKey {
    pub generation: GenerationId,
    pub signal: Arc<str>,
    pub revision: u64,
    /// A sample index exists only for an actual acquisition event. Publishing
    /// a finite prefix (including the empty prefix) does not invent a shot.
    pub ordinal: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Storage { Latest, Finite, Snapshot }
pub struct Declaration { pub signal: Arc<str>, pub schema: Arc<DatasetSchema>, pub storage: Storage }

pub struct SignalDescription {
    pub generation: Generation, pub signal: Arc<str>, pub schema: Arc<DatasetSchema>,
    pub finite: bool, pub published: bool, pub sealed: bool,
    pub history_capable: bool, pub history_capacity: usize, pub retained_events: usize,
}

#[derive(Clone)]
pub struct Generation { id: GenerationId, owner: Arc<str> }
impl Generation {
    pub fn id(&self) -> GenerationId { self.id }
    pub fn owner(&self) -> &str { &self.owner }
}

pub struct RunRecord { pub generation: GenerationId, pub owner: Arc<str>, pub metadata: Value }
pub struct ParentRecord {
    pub key: PublicationKey,
    pub run: Arc<RunRecord>,
    pub event: Arc<EventRecord>,
}
pub struct EventRecord { pub metadata: Value, pub parents: Arc<[ParentRecord]> }

pub struct Publication {
    key: PublicationKey,
    data: Option<Arc<DataBlock>>,
    run: Arc<RunRecord>,
    event: Arc<EventRecord>,
    canonical: Option<Snapshot>,
    storage: Storage,
    // History owns event pixels, not every upstream camera buffer. Only a
    // current head or an acquired Read pins this short-lived causal payloads.
    lineage: Weak<PayloadLineage>,
}
struct PayloadLineage { publications: HashMap<PublicationKey, Arc<Publication>> }
impl Publication {
    pub fn key(&self) -> &PublicationKey { &self.key }
    pub fn data(&self) -> Option<&Arc<DataBlock>> { self.data.as_ref() }
    pub fn run(&self) -> &Arc<RunRecord> { &self.run }
    pub fn event(&self) -> &Arc<EventRecord> { &self.event }
    /// Finite display reads use the complete authored layout at this exact
    /// publication. Scientific event consumers still use data().
    pub fn canonical(&self) -> Option<&Snapshot> { self.canonical.as_ref() }
    pub fn storage(&self) -> Storage { self.storage }
}

pub enum Input {
    Event {
        signal: Arc<str>, ordinal: u64, data: Arc<DataBlock>,
        /// Finite acquisition placement in flat Repeat and Point carriers.
        origin: Option<[usize; 2]>,
    },
    /// Publish the store as it stands, without writing data or sampling an
    /// ordinal. Dataset readers observe it; raw event readers do not.
    FiniteSnapshot { signal: Arc<str> },
    /// A complete immutable dataset replaces a Snapshot output while keeping
    /// its real acquired placements. It is not an acquisition delta.
    Dataset { signal:Arc<str>, ordinal:Option<u64>, snapshot:Snapshot },
}
impl Input {
    fn signal(&self) -> &Arc<str> {match self {Self::Event{signal,..}|Self::FiniteSnapshot{signal}|Self::Dataset{signal,..}=>signal}}
    fn ordinal(&self) -> Option<u64> {match self {Self::Event{ordinal,..}=>Some(*ordinal),Self::FiniteSnapshot{..}=>None,Self::Dataset{ordinal,..}=>*ordinal}}
}
pub struct Bundle {
    pub inputs: Vec<Input>,
    pub metadata: Value,
    /// The actual immutable parents are the proof; no late lookup in a live
    /// history is needed. Outputs retain causal metadata, not ancestor pixels.
    pub parents: Vec<Arc<Publication>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DeliveryKind { Data, Sealed { outcome: Arc<Value> }, Retired }

pub struct Delivery {
    pub kind: DeliveryKind,
    pub generation: GenerationId,
    /// Data has its commit revision, or None for a current snapshot. Lifecycle
    /// markers carry the run's final revision without inventing a publication.
    pub revision: Option<u64>,
    pub publications: Arc<[Arc<Publication>]>,
    /// Reads are frozen under the commit lock, not resolved when dequeued.
    pub reads: Arc<[Read]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidDeclaration(String), OwnerRunning, SignalOwned(String), UnknownGeneration,
    Sealed, UnknownSignal(String), NoData, SchemaChanged(String), OutOfOrder(String),
    EmptyBundle, DuplicateSignal(String), InvalidMetadata, CounterExhausted,
    HistoryNotRetained, PublicationExpired, InvalidWindow,
    SourceSuperseded,
    SubscriptionOverflow { events: usize, referenced_bytes: usize }, InvalidLayout(String), Closed,
}
impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for RuntimeError {}
type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Clone, Copy)]
pub enum DeliveryPolicy {
    Latest,
    /// Payload bounds deduplicate DataBlock owners; finite prefixes use their
    /// largest queued prefix per generation/signal, conservatively summed.
    /// Overflow fails this subscription without stalling the producer.
    Exact { max_events: usize, max_referenced_bytes: usize },
}
#[derive(Clone, Copy)]
pub enum Start { Next, Current }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadMode { Event, Dataset, Window(usize) }
pub struct ReadRequest { pub signal: Arc<str>, pub mode: ReadMode }

/// Fixed relative-index window. Missing ordinals remain holes; acquiring a
/// lease or enlarging one cannot manufacture earlier samples.
pub struct WindowRead {
    pub latest: PublicationKey,
    pub width: usize,
    pub view_epoch: u64,
    pub schema: Arc<DatasetSchema>,
    entries: Arc<[Arc<Publication>]>,
}

/// A self-owned scientific read. Display selects canonical finite data, while
/// exact processors can explicitly keep the individual Event.
#[derive(Clone)]
pub enum ReadKind {
    Event(Arc<Publication>), Canonical(Arc<Publication>), Window(Arc<WindowRead>),
}
#[derive(Clone)]
pub struct Read { kind: ReadKind, _lineage: Option<Arc<PayloadLineage>> }
impl Read {
    fn new(kind: ReadKind) -> Self {
        let anchor = match &kind {ReadKind::Event(event)|ReadKind::Canonical(event)=>event,
            ReadKind::Window(window)=>window.entries.last().expect("window has an anchor")};
        Self {_lineage:anchor.lineage.upgrade(),kind}
    }
    pub fn from_event(event:Arc<Publication>) -> Result<Self> {
        if event.data.is_none() {return Err(RuntimeError::NoData);}
        Ok(Self::new(ReadKind::Event(event)))
    }
    pub fn from_canonical(event:Arc<Publication>) -> Self {Self::new(ReadKind::Canonical(event))}
    pub fn from_window(window:Arc<WindowRead>) -> Self {Self::new(ReadKind::Window(window))}
    pub fn kind(&self) -> &ReadKind {&self.kind}
    pub fn anchor(&self) -> &Arc<Publication> {match &self.kind {
        ReadKind::Event(event)|ReadKind::Canonical(event)=>event,
        ReadKind::Window(window)=>window.entries.last().expect("window has an anchor"),
    }}
    /// Scientific shot identity for presentation cohorts. A history read is
    /// anchored at its newest event, not every retained row of the window.
    pub fn shot_roots(&self) -> HashSet<EventIdentity> {
        causal_events(&self.anchor().key,&self.anchor().event,true)
    }
    pub fn schema(&self) -> &Arc<DatasetSchema> {
        match &self.kind {
            ReadKind::Event(event) => event.data.as_ref().expect("checked event payload").schema(),
            ReadKind::Canonical(event) => event.canonical.as_ref().expect("canonical read").schema(),
            ReadKind::Window(window) => &window.schema,
        }
    }
    pub fn key(&self) -> &PublicationKey {
        self.anchor().key()
    }
    pub fn same_read(&self, other: &Self) -> bool {
        match (&self.kind,&other.kind) {
            (ReadKind::Event(left),ReadKind::Event(right)) | (ReadKind::Canonical(left),ReadKind::Canonical(right)) => Arc::ptr_eq(left,right),
            (ReadKind::Window(left),ReadKind::Window(right)) => Arc::ptr_eq(left,right)
                || (left.latest==right.latest && left.width==right.width && left.view_epoch==right.view_epoch
                    && Arc::ptr_eq(&left.schema,&right.schema) && left.entries.len()==right.entries.len()
                    && left.entries.iter().zip(right.entries.iter()).all(|(left,right)|Arc::ptr_eq(left,right))),
            _ => false,
        }
    }
    pub fn is_complete_dataset(&self) -> bool {
        match &self.kind {
            ReadKind::Event(event) => event.storage == Storage::Snapshot,
            ReadKind::Canonical(_) | ReadKind::Window(_) => true,
        }
    }
    /// The causal facts a saved Figure keeps: each publication with its run and
    /// event records and parents; a window also lists every member's record.
    pub fn provenance(&self) -> Value {
        fn record(key:&PublicationKey, run:&RunRecord, event:&EventRecord) -> Value {
            serde_json::json!({"publication":{"stream":run.owner.as_ref(),"generation":key.generation.as_uuid().to_string(),"signal":key.signal.as_ref(),
                "revision":key.revision,"ordinal":key.ordinal},"run":run.metadata,"event":event.metadata,
                "parents":event.parents.iter().map(|parent|record(&parent.key,&parent.run,&parent.event)).collect::<Vec<_>>()})
        }
        match &self.kind {
            ReadKind::Event(event)|ReadKind::Canonical(event) => record(&event.key,&event.run,&event.event),
            ReadKind::Window(window) => {
                let latest = window.entries.last().expect("window has an anchor");
                let mut value = record(&latest.key,&latest.run,&latest.event);
                value["window"] = serde_json::json!({"width":window.width,"members":window.entries.iter()
                    .map(|event|record(&event.key,&event.run,&event.event)).collect::<Vec<_>>()});
                value
            }
        }
    }
}
impl WindowRead {
    pub fn entries(&self) -> &[Arc<Publication>] { &self.entries }
    pub fn relative_index(&self, publication: &Publication) -> Option<i128> {
        (publication.key.generation == self.latest.generation && publication.key.signal == self.latest.signal)
            .then_some(())?;
        Some(i128::from(publication.key.ordinal?) - i128::from(self.latest.ordinal?))
    }
    pub fn at(&self, relative: i128) -> Option<&Arc<Publication>> {
        if relative > 0 || relative <= -(self.width as i128) { return None; }
        let ordinal = i128::from(self.latest.ordinal?) + relative;
        let ordinal = u64::try_from(ordinal).ok()?;
        self.entries.binary_search_by_key(&Some(ordinal), |p| p.key.ordinal).ok().map(|i| &self.entries[i])
    }
    /// Repeat/Point origins for this immutable segment in the effective schema.
    /// The scalar source index is already an ordinary logical axis there.
    pub fn placement(&self, publication: &Publication) -> Option<[usize; 2]> {
        let relative = self.relative_index(publication)?;
        if relative > 0 || relative <= -(self.width as i128) { return None; }
        let slot = (self.width as i128 - 1 + relative) as usize;
        Some([0, slot * publication.data.as_ref()?.schema().domain(DomainRole::Point).size()])
    }
}

struct Signal {
    schema: Arc<DatasetSchema>, latest: Option<Arc<Publication>>,
    last_ordinal: Option<u64>,
    head: Option<Arc<PayloadLineage>>,
    history: VecDeque<Arc<Publication>>, demands: BTreeMap<u64, usize>,
    views: HashMap<usize, Weak<WindowRead>>, view_epoch: u64,
    window_schemas: HashMap<usize, Arc<DatasetSchema>>, history_axis: Arc<str>,
    store: Option<DatasetStore>, history_capable: bool,
}
impl Signal {
    fn capacity(&self) -> usize { self.demands.values().copied().max().unwrap_or(0) }
    fn trim(&mut self, garbage: &mut Vec<Arc<Publication>>) -> bool {
        let capacity = self.capacity();
        let Some(latest) = &self.latest else { return false; };
        let count = self.history.len();
        let start = latest.key.ordinal.map(|ordinal|ordinal.saturating_sub(capacity.saturating_sub(1) as u64));
        while self.history.front().is_some_and(|p| capacity == 0 || p.key.ordinal < start) {
            garbage.push(self.history.pop_front().unwrap());
        }
        let trimmed = count != self.history.len();
        if trimmed { self.views.clear(); }
        self.views.retain(|width, _| *width == 1 || self.demands.values().any(|value| value == width));
        self.window_schemas.retain(|width, _| *width == 1 || self.demands.values().any(|value| value == width));
        trimmed
    }
}

fn history_schema(signal: &mut Signal, width: usize) -> Result<Arc<DatasetSchema>> {
    if let Some(schema) = signal.window_schemas.get(&width) { return Ok(schema.clone()); }
    let build = || -> std::result::Result<Arc<DatasetSchema>, String> {
        let source = &signal.schema;
        let point = source.domain(DomainRole::Point);
        let size = point.size().checked_mul(width).ok_or("History point carrier size overflows")?;
        let index = Axis::new(signal.history_axis.clone(), "source index".into(), "primary-index".into(),
            Coordinates::Index { origin: 1 - width as i128, len: width }, None, None, None, None)?;
        let mut axes = vec![index];
        axes.extend(point.axes().iter().cloned());
        let mut mappings = vec![CodeMapping::new(CodeBase::Range {start:0,step:1,len:width},point.size(),1)?];
        for axis in point.axes() {
            let mapping = point.mapping(axis.id()).ok_or("Source axis lacks its mapping")?;
            let base = match mapping.base() {
                CodeBase::Range {start,step,len} => CodeBase::Range {start:*start,step:*step,len:*len},
                CodeBase::Explicit(values) => CodeBase::Explicit(values.clone()),
            };
            mappings.push(CodeMapping::new(base, mapping.inner(),
                mapping.outer().checked_mul(width).ok_or("History mapping size overflows")?)?);
        }
        DatasetSchema::new(source.domain(DomainRole::Repeat).clone(), Domain::new(vec![size],axes,mappings)?,
            source.domain(DomainRole::Cell).clone(), source.value().clone())
    };
    let schema = build().map_err(RuntimeError::InvalidLayout)?;
    signal.window_schemas.insert(width,schema.clone());
    Ok(schema)
}

fn validate_retention(signal: &Signal, width: usize) -> Result<()> {
    if !signal.history_capable {
        return Err(RuntimeError::InvalidLayout("This signal publishes a complete dataset; indexed history requires a latest-event declaration".into()));
    }
    // A lease must not succeed and start retaining pixels if its effective
    // layout can never be read. This check needs no coordinate expansion.
    if signal.schema.value_len().checked_mul(width).is_none()
        || signal.schema.component_mask_len().checked_mul(width).is_none() {
        return Err(RuntimeError::InvalidLayout("History shape overflows addressable storage".into()));
    }
    if [DomainRole::Repeat,DomainRole::Point,DomainRole::Cell].iter().any(|role|
        signal.schema.domain(*role).axis(&signal.history_axis).is_some()) {
        return Err(RuntimeError::InvalidLayout("History axis identity collides with a declared source axis".into()));
    }
    Ok(())
}

fn window_read(signal: &mut Signal, width: usize) -> Result<Arc<WindowRead>> {
    validate_window(width)?;
    validate_retention(signal,width)?;
    if width > 1 && !signal.demands.values().any(|requested| *requested == width) { return Err(RuntimeError::HistoryNotRetained); }
    if let Some(view) = signal.views.get(&width).and_then(Weak::upgrade) { return Ok(view); }
    let latest = signal.latest.as_ref().ok_or(RuntimeError::NoData)?.key.clone();
    let start = Some(latest.ordinal.ok_or(RuntimeError::NoData)?.saturating_sub(width as u64 - 1));
    let schema = history_schema(signal,width)?;
    let entries = if width == 1 { signal.latest.iter().cloned().collect() }
        else { signal.history.iter().filter(|p| p.key.ordinal >= start).cloned().collect() };
    let view = Arc::new(WindowRead { latest, width, view_epoch: signal.view_epoch, schema, entries });
    signal.views.insert(width, Arc::downgrade(&view));
    Ok(view)
}

fn window_read_at(signal:&mut Signal,width:usize,anchor:&Arc<Publication>) -> Result<Arc<WindowRead>> {
    if signal.latest.as_ref().is_some_and(|latest|latest.key==anchor.key) {return window_read(signal,width);}
    validate_window(width)?;
    validate_retention(signal,width)?;
    if width>1 && !signal.demands.values().any(|requested|*requested==width) {return Err(RuntimeError::HistoryNotRetained);}
    let start=Some(anchor.key.ordinal.ok_or(RuntimeError::NoData)?.saturating_sub(width as u64-1));
    // The historical anchor itself must still be materializable. A retained
    // display picture is not permission to silently substitute a newer shot.
    if width>1 && !signal.history.iter().any(|event|event.key==anchor.key) {return Err(RuntimeError::PublicationExpired);}
    let entries=if width==1 {vec![anchor.clone()]} else {
        signal.history.iter().filter(|event|event.key.ordinal>=start && event.key.ordinal<=anchor.key.ordinal).cloned().collect()
    };
    Ok(Arc::new(WindowRead {latest:anchor.key.clone(),width,view_epoch:signal.view_epoch,
        schema:history_schema(signal,width)?,entries:entries.into()}))
}

fn dataset_read(event:Arc<Publication>) -> Read {
    if event.canonical.is_some() {Read::from_canonical(event)} else {Read::from_event(event).expect("noncanonical publication has data")}
}

pub type EventIdentity=(GenerationId,u64);
fn event_identity(key:&PublicationKey) -> EventIdentity {(key.generation,key.revision)}
fn causal_events(key:&PublicationKey,event:&EventRecord,roots_only:bool) -> HashSet<EventIdentity> {
    let mut seen=HashSet::new();let mut result=HashSet::new();let mut pending=vec![(key,event)];
    while let Some((key,event))=pending.pop() {
        let identity=event_identity(key);
        if !seen.insert(identity) {continue;}
        if !roots_only || event.parents.is_empty() {result.insert(identity);}
        pending.extend(event.parents.iter().map(|parent|(&parent.key,parent.event.as_ref())));
    }
    result
}

fn ancestry(publication:&Arc<Publication>) -> Option<HashMap<Arc<str>,Read>> {
    let mut candidates:HashMap<Arc<str>,Vec<Arc<Publication>>>=HashMap::new();
    if let Some(lineage)=publication.lineage.upgrade() {
        for value in lineage.publications.values() {candidates.entry(value.key.signal.clone()).or_default().push(value.clone());}
    } else {candidates.insert(publication.key.signal.clone(),vec![publication.clone()]);}
    candidates.into_iter().map(|(name,values)| {
        let chosen=if values.len()==1 {values.first()} else {values.iter().find(|candidate| {
            let reachable=causal_events(&candidate.key,&candidate.event,false);
            values.iter().all(|value|reachable.contains(&event_identity(&value.key)))
        })}?;
        Some((name,dataset_read(chosen.clone())))
    }).collect()
}

/// One surface's last complete same-shot selection (v2 `previous_front`): what
/// it keeps showing while a component's new shot is still being derived. Each
/// board owns its own, so what another surface reads never changes this one.
#[derive(Default)]
pub struct Front(HashMap<Arc<str>,Read>);

/// Pure requested-component selection, following v2 front.build_front. The
/// registry owns payload lifetime; this algorithm never acquires history.
fn coherent_front(state:&State,requests:&[(Generation,Arc<str>,Option<usize>)],previous:&Front) -> HashMap<Arc<str>,Read> {
    let mut latest=HashMap::new();
    let mut adjacency:HashMap<Arc<str>,HashSet<Arc<str>>>=HashMap::new();
    let mut source_by_output=HashMap::new();
    let mut bundles:HashMap<EventIdentity,HashSet<Arc<str>>>=HashMap::new();
    let mut bundle_of=HashMap::new();
    for run in state.runs.values() {
        // v2 uses the owner's active declarations for ancestor membership:
        // a processor of counts is also a descendant of its occupied sibling,
        // even while only one of those outputs has a newer partial commit.
        let names=Arc::new(run.signals.iter().filter(|(_,signal)|!run.sealed || signal.latest.is_some())
            .map(|(name,_)|name.clone()).collect::<HashSet<_>>());
        for name in names.iter() {bundle_of.insert(name.clone(),names.clone());}
        for (name,signal) in &run.signals {
            if let Some(event)=&signal.latest {
                latest.insert(name.clone(),dataset_read(event.clone()));
                bundles.entry(event_identity(&event.key)).or_default().insert(name.clone());
            }
            // A finished route of a replaced source run is history, not a
            // pending member of the current shot (v2 retires it).
            if let Some(source)=run.source.as_ref().filter(|source|source.coherent && (!run.sealed
                || source.generation.is_some_and(|generation|state.signals.get(&source.signal)==Some(&generation)))) {
                source_by_output.insert(name.clone(),source.signal.clone());
                adjacency.entry(name.clone()).or_default().insert(source.signal.clone());
                adjacency.entry(source.signal.clone()).or_default().insert(name.clone());
            }
        }
    }
    for names in bundles.values() {
        for name in names {adjacency.entry(name.clone()).or_default().extend(names.iter().filter(|other|*other!=name).cloned());}
    }
    let requested:HashSet<_>=requests.iter().filter(|(generation,name,_)|state.signals.get(name)==Some(&generation.id))
        .map(|(_,name,_)|name.clone()).collect();
    let mut pending=requested.clone();let mut selected=latest.clone();
    let is_ancestor=|ancestor:&Arc<str>,descendant:&Arc<str>| {
        let mut seen=HashSet::new();let mut current=descendant;
        while let Some(parent)=source_by_output.get(current) {
            if !seen.insert(current.clone()) {break;}
            if parent==ancestor || bundle_of.get(parent).is_some_and(|names|names.contains(ancestor)) {return true;}
            current=parent;
        }
        false
    };
    while let Some(seed)=pending.iter().next().cloned() {
        let mut component=HashSet::new();let mut stack=vec![seed];
        while let Some(name)=stack.pop() {
            if !state.signals.contains_key(&name) || !component.insert(name.clone()) {continue;}
            if let Some(neighbours)=adjacency.get(&name) {stack.extend(neighbours.iter().cloned());}
        }
        pending.retain(|name|!component.contains(name));
        let required:Vec<_>=component.intersection(&requested).cloned().collect();
        let leaves:Vec<_>=required.iter().filter(|name|!required.iter().any(|other|other!=*name && is_ancestor(name,other))).collect();
        let mut roots=None;let mut joined=HashMap::new();let mut coherent=!leaves.is_empty();
        for name in leaves {
            let Some(read)=latest.get(name) else {coherent=false;break;};
            let event=read.anchor();let current=causal_events(&event.key,&event.event,true);
            if roots.as_ref().is_some_and(|previous|*previous!=current) {coherent=false;break;}
            roots=Some(current);
            let Some(ancestors)=ancestry(event) else {coherent=false;break;};
            for (name,ancestor) in ancestors {
                if joined.get(&name).is_some_and(|old:&Read|old.key()!=ancestor.key()) {coherent=false;break;}
                joined.insert(name,ancestor);
            }
            if !coherent {break;}
        }
        coherent &= required.iter().all(|name|joined.get(name)
            .is_some_and(|read|state.signals.get(name)==Some(&read.key().generation)));
        if !coherent {
            joined=previous.0.iter().filter(|(name,read)|component.contains(*name)
                && state.signals.get(*name)==Some(&read.key().generation)).map(|(name,read)|(name.clone(),read.clone())).collect();
            let roots:HashSet<_>=required.iter().filter_map(|name|joined.get(name)).map(|read| {
                let mut roots:Vec<_>=causal_events(&read.anchor().key,&read.anchor().event,true).into_iter().collect();
                roots.sort_unstable_by_key(|(generation,revision)|(generation.as_uuid(),*revision));roots
            }).collect();
            if !required.iter().all(|name|joined.contains_key(name)) || roots.len()!=1 {joined.clear();}
        }
        for name in component {selected.remove(&name);if let Some(read)=joined.remove(&name) {selected.insert(name,read);}}
    }
    // Keeping an unrelated, undisplayed signal here would create retention
    // without a consumer. Each chosen Read already pins its causal closure.
    selected.retain(|name,_|requested.contains(name));
    selected
}
struct Run {
    token: Generation, record: Arc<RunRecord>, sealed: bool, revision: u64,
    signals: HashMap<Arc<str>, Signal>, outcome: Option<Arc<Value>>,
    subscribers: Arc<[u64]>,
    source: Option<DerivedSource>,
}
/// The signal a run's commits derive from (v2 `source_name`): one run of it
/// (`generation`), or the signal across its runs. A guarded route is a derived
/// owner's: it ends when that source run is replaced.
struct DerivedSource { signal: Arc<str>, generation: Option<GenerationId>, coherent: bool, guarded: bool }
struct Subscriber {
    generation: Option<GenerationId>, follow_owner: Option<Arc<str>>,
    requests: Vec<ReadRequest>, policy: DeliveryPolicy,
    queue: VecDeque<Arc<Delivery>>, bytes: u128, terminal: Option<RuntimeError>,
    blocks: HashMap<usize, (Arc<DataBlock>, usize)>,
    prefixes: HashMap<(GenerationId, Arc<str>), (usize, usize)>,
    changed: Arc<Condvar>, interrupted: bool,
}
impl Subscriber {
    fn account_publication(&mut self, publication: &Publication, add: bool) {
        if publication.storage==Storage::Finite {
            let prefix=publication.canonical.as_ref().expect("finite publication has canonical prefix");
            let key = (publication.key.generation,publication.key.signal.clone());
            if add {
                let held = self.prefixes.entry(key).or_default();
                let bytes = prefix.payload_bytes();
                self.bytes += bytes.saturating_sub(held.1) as u128;
                held.0 += 1; held.1 = held.1.max(bytes);
            } else {
                let held = self.prefixes.get_mut(&key).unwrap();
                held.0 -= 1;
                if held.0 == 0 {
                    self.bytes -= held.1 as u128;
                    self.prefixes.remove(&key);
                }
            }
        } else if let Some(dataset)=&publication.canonical {
            // Snapshot replacements are independent datasets, not append-only
            // prefixes. Deduplicate actual owners rather than taking one max.
            for part in dataset.segments() {self.account_data(part.data(),add);}
        } else {
            self.account_data(publication.data.as_ref().expect("event payload"),add);
        }
    }

    fn account_data(&mut self,data:&Arc<DataBlock>,add:bool) {
        let identity = Arc::as_ptr(data) as usize;
        if add {
            let held = self.blocks.entry(identity).or_insert_with(|| (data.clone(), 0));
            if held.1 == 0 { self.bytes += data.payload_bytes() as u128; }
            held.1 += 1;
        } else {
            let held = self.blocks.get_mut(&identity).unwrap();
            held.1 -= 1;
            if held.1 == 0 {
                self.bytes -= data.payload_bytes() as u128;
                self.blocks.remove(&identity);
            }
        }
    }

    fn account(&mut self, delivery: &Delivery, add: bool) {
        if matches!(self.policy, DeliveryPolicy::Latest) { return; }
        for read in delivery.reads.iter() {
            match read.kind() {
                ReadKind::Event(event) | ReadKind::Canonical(event) => self.account_publication(event, add),
                ReadKind::Window(window) => for event in window.entries.iter() { self.account_publication(event, add); },
            }
            if let Some(lineage)=&read._lineage {
                for event in lineage.publications.values() {self.account_publication(event,add);}
            }
        }
    }

    fn clear_queue(&mut self, garbage: &mut Vec<Arc<Delivery>>) {
        garbage.extend(self.queue.drain(..));
        self.blocks.clear(); self.prefixes.clear(); self.bytes = 0;
    }

    fn enqueue(&mut self, delivery: Arc<Delivery>, garbage: &mut Vec<Arc<Delivery>>) -> Result<()> {
        if matches!(self.policy, DeliveryPolicy::Latest) {
            if delivery.kind == DeliveryKind::Data
                || self.queue.back().is_some_and(|old|old.generation!=delivery.generation) {
                self.clear_queue(garbage);
            } else {
                // Seal/retire follows the last Data; it must not erase the
                // final frame. At most Data + Sealed + Retired are pending.
                let mut retained=VecDeque::new();
                for old in self.queue.drain(..) {
                    if old.kind==delivery.kind {garbage.push(old);} else {retained.push_back(old);}
                }
                self.queue=retained;
            }
        }
        self.account(&delivery, true);
        self.queue.push_back(delivery);
        if let DeliveryPolicy::Exact { max_events, max_referenced_bytes } = self.policy {
            if self.queue.len() > max_events || self.bytes > max_referenced_bytes as u128 {
                let error = RuntimeError::SubscriptionOverflow {
                    events: self.queue.len(), referenced_bytes: self.bytes.min(usize::MAX as u128) as usize,
                };
                self.terminal = Some(error.clone());
                self.clear_queue(garbage);
                self.changed.notify_all();
                return Err(error);
            }
        }
        self.changed.notify_one();
        Ok(())
    }

    fn attach(&mut self, run: &mut Run, ticket: u64, start: Start,
              garbage: &mut Vec<Arc<Delivery>>, publications: &mut Vec<Arc<Publication>>) -> Result<()> {
        for request in &self.requests {
            let signal = run.signals.get(&request.signal).ok_or_else(|| RuntimeError::UnknownSignal(request.signal.to_string()))?;
            if let ReadMode::Window(width) = request.mode { validate_retention(signal,width)?; }
        }
        let mut current = Vec::new();
        for request in &self.requests {
            let signal = run.signals.get_mut(&request.signal).unwrap();
            if let ReadMode::Window(width) = request.mode {
                signal.demands.insert(ticket,width);
                if signal.history.is_empty() { signal.history.extend(signal.latest.iter().cloned()); }
            }
            if matches!(start,Start::Current) {
                current.extend(signal.latest.iter().filter(|event|!matches!(request.mode,ReadMode::Event) || event.data.is_some()).cloned());
            }
        }
        let result = (|| {
            if !current.is_empty() {
                let reads = capture_reads(run,&self.requests,&current)?;
                self.enqueue(Arc::new(Delivery {kind:DeliveryKind::Data,generation:run.token.id,
                    revision:None,publications:current.into(),reads}),garbage)?;
            }
            if run.sealed {
                self.end_run(run,DeliveryKind::Sealed {outcome:run.outcome.clone().unwrap()},garbage)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            release_demands(run,&self.requests,ticket,publications);
            return Err(error);
        }
        self.generation = Some(run.token.id);
        let mut ids = run.subscribers.to_vec(); ids.push(ticket); run.subscribers = ids.into();
        Ok(())
    }

    fn end_run(&mut self, run: &Run, kind: DeliveryKind, garbage: &mut Vec<Arc<Delivery>>) -> Result<()> {
        if self.follow_owner.is_some() {
            if matches!(kind,DeliveryKind::Retired) { self.generation = None; }
            self.enqueue(Arc::new(Delivery {kind,generation:run.token.id,revision:Some(run.revision),
                publications:Arc::from([]),reads:Arc::from([])}),garbage)
        } else {
            self.terminal = Some(if matches!(kind,DeliveryKind::Sealed {..}) {RuntimeError::Sealed} else {RuntimeError::UnknownGeneration});
            self.changed.notify_all();
            Ok(())
        }
    }
}

fn capture_reads(run: &mut Run, requests: &[ReadRequest], publications: &[Arc<Publication>]) -> Result<Arc<[Read]>> {
    publications.iter().map(|event| {
        let request = requests.iter().find(|request| request.signal == event.key.signal).unwrap();
        match request.mode {
            ReadMode::Event => Read::from_event(event.clone()),
            ReadMode::Dataset => Ok(dataset_read(event.clone())),
            ReadMode::Window(width) => window_read(run.signals.get_mut(&request.signal).unwrap(),width).map(Read::from_window),
        }
    }).collect()
}

fn release_demands(run: &mut Run, requests: &[ReadRequest], ticket: u64, garbage: &mut Vec<Arc<Publication>>) {
    for request in requests {
        if matches!(request.mode, ReadMode::Window(_)) {
            let signal = run.signals.get_mut(&request.signal).unwrap();
            signal.demands.remove(&ticket);
            if signal.trim(garbage) { signal.view_epoch += 1; }
        }
    }
}
struct State {
    runs: HashMap<GenerationId, Run>, owners: HashMap<Arc<str>, GenerationId>,
    signals: HashMap<Arc<str>, GenerationId>, subscribers: HashMap<u64, Subscriber>, next_ticket: u64, closed: bool,
    watches: HashMap<u64, (SyncSender<()>, Arc<AtomicBool>)>,
}
struct Inner { state: Mutex<State> }

#[derive(Clone)]
pub struct Runtime { inner: Arc<Inner> }
impl Default for Runtime { fn default() -> Self { Self::new() } }
impl Runtime {
    pub fn new() -> Self {
        Self { inner: Arc::new(Inner { state: Mutex::new(State {
            runs: HashMap::new(), owners: HashMap::new(), signals: HashMap::new(),
            subscribers: HashMap::new(), next_ticket: 1, closed: false, watches: HashMap::new(),
        }) }) }
    }

    pub fn watch_changes(&self) -> Result<ChangeWatch> {
        let mut state = self.inner.state.lock().unwrap();
        if state.closed { return Err(RuntimeError::Closed); }
        let ticket = take_ticket(&mut state)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let pending = Arc::new(AtomicBool::new(false));
        state.watches.insert(ticket, (sender, pending.clone()));
        Ok(ChangeWatch { inner: Arc::downgrade(&self.inner), ticket, receiver, pending })
    }

    pub fn begin(&self, owner: impl Into<Arc<str>>, outputs: Vec<Declaration>, metadata: Value) -> Result<Generation> {
        self.begin_run(owner.into(),outputs,metadata,None)
    }

    /// Declare an actual causal route, not a naming convention. Presentation
    /// followers keep lineage but do not hold their source's coherent front.
    pub fn begin_derived(&self, owner: impl Into<Arc<str>>, outputs: Vec<Declaration>, metadata: Value,
                         source:&PublicationKey, coherent:bool) -> Result<Generation> {
        self.begin_run(owner.into(),outputs,metadata,Some(DerivedSource {signal:source.signal.clone(),
            generation:Some(source.generation),coherent,guarded:true}))
    }

    /// A plugin run that consumes `source` and commits each result with the
    /// source event it is of: a same-shot route of that signal like a derived
    /// owner (v2 processors and source-reading producers), so a picture and a
    /// judgement of it are selected as one shot whoever else consumes either.
    /// It is bound to the one run of `source` it subscribed (`generation`), or
    /// follows the signal across its runs (None), and keeps committing a
    /// replaced run's remaining queue.
    pub fn begin_consuming(&self, owner: impl Into<Arc<str>>, outputs: Vec<Declaration>, metadata: Value,
                           source: Arc<str>, generation: Option<GenerationId>) -> Result<Generation> {
        self.begin_run(owner.into(),outputs,metadata,Some(DerivedSource {signal:source,generation,coherent:true,guarded:false}))
    }

    fn begin_run(&self, owner:Arc<str>, outputs:Vec<Declaration>, metadata:Value,
                 source:Option<DerivedSource>) -> Result<Generation> {
        if owner.is_empty() || owner.trim() != owner.as_ref() || outputs.is_empty() {
            return Err(RuntimeError::InvalidDeclaration("An owner and at least one output are required".into()));
        }
        if !metadata.is_object() { return Err(RuntimeError::InvalidMetadata); }
        let mut names = HashSet::new();
        for output in &outputs {
            if output.signal.is_empty() || output.signal.trim() != output.signal.as_ref() || !names.insert(output.signal.clone()) {
                return Err(RuntimeError::InvalidDeclaration("Output names must be canonical and unique".into()));
            }
        }
        let token = Generation { id: GenerationId(Uuid::new_v4()), owner: owner.clone() };
        let old;
        let mut deliveries = Vec::new();
        let mut garbage = Vec::new();
        {
            let mut state = self.inner.state.lock().unwrap();
            if state.closed { return Err(RuntimeError::Closed); }
            if let Some((signal,generation))=source.as_ref().filter(|source|source.guarded).and_then(|source|Some((&source.signal,source.generation?))) {
                if state.signals.get(signal)!=Some(&generation) {return Err(RuntimeError::SourceSuperseded);}
                let run=state.runs.get(&generation).ok_or(RuntimeError::SourceSuperseded)?;
                if run.token.owner==owner {return Err(RuntimeError::InvalidDeclaration("A derived owner cannot follow itself".into()));}
            }
            let previous = state.owners.get(&owner).copied();
            if previous.is_some_and(|id| !state.runs[&id].sealed) { return Err(RuntimeError::OwnerRunning); }
            for name in &names {
                if state.signals.get(name).is_some_and(|id| Some(*id) != previous) {
                    return Err(RuntimeError::SignalOwned(name.to_string()));
                }
            }
            old = previous.and_then(|id| remove_run(&mut state, id, &mut deliveries));
            let signals = outputs.into_iter().map(|output| {
                // Axis identity belongs to the logical signal, not to a run.
                // A deterministic namespace needs no forever-growing registry.
                let history_axis: Arc<str> = format!("history.{}", Uuid::new_v5(
                    &Uuid::NAMESPACE_URL, format!("zlc:history:{}",output.signal).as_bytes())).into();
                state.signals.insert(output.signal.clone(), token.id);
                let store = matches!(output.storage,Storage::Finite).then(|| DatasetStore::new(output.schema.clone()));
                (output.signal, Signal { schema: output.schema, latest: None, last_ordinal:None, head:None, history: VecDeque::new(),
                    demands: BTreeMap::new(), views: HashMap::new(), view_epoch: 0,
                    window_schemas: HashMap::new(), history_axis, store,
                    history_capable: matches!(output.storage,Storage::Latest) })
            }).collect();
            state.owners.insert(owner.clone(), token.id);
            let mut run = Run { token: token.clone(), sealed: false, revision: 0, signals,
                record: Arc::new(RunRecord { generation: token.id, owner:owner.clone(), metadata }), outcome: None, subscribers: Arc::from([]),source };
            // Explicit followers and their retention are installed before the
            // producer can commit its first event, without a notification race.
            for (&ticket,subscriber) in state.subscribers.iter_mut().filter(|(_,subscriber)|
                subscriber.follow_owner.as_ref()==Some(&owner) && subscriber.terminal.is_none()) {
                if let Err(error) = subscriber.attach(&mut run,ticket,Start::Next,&mut deliveries,&mut garbage) {
                    subscriber.terminal=Some(error);
                    subscriber.changed.notify_all();
                }
            }
            state.runs.insert(token.id,run);
            notify_changes(&state);
        }
        drop((old,deliveries,garbage)); // Large buffers are released outside the state lock.
        Ok(token)
    }

    pub fn commit(&self, generation: &Generation, bundle: &Bundle) -> Result<Arc<Delivery>> {
        if bundle.inputs.is_empty() { return Err(RuntimeError::EmptyBundle); }
        if !bundle.metadata.is_object() { return Err(RuntimeError::InvalidMetadata); }
        let event = Arc::new(EventRecord { metadata: bundle.metadata.clone(),
            parents: bundle.parents.iter().map(|p| ParentRecord {
                key: p.key.clone(), run: p.run.clone(), event: p.event.clone(),
            }).collect() });
        let mut garbage = Vec::new();
        let mut deliveries = Vec::new();
        let mut heads = Vec::new();
        let committed;
        {
            let mut state = self.inner.state.lock().unwrap();
            if state.closed { return Err(RuntimeError::Closed); }
            if let Some(source)=state.runs.get(&generation.id).ok_or(RuntimeError::UnknownGeneration)?.source.as_ref().filter(|source|source.guarded) {
                if state.signals.get(&source.signal)!=source.generation.as_ref() {return Err(RuntimeError::SourceSuperseded);}
                if !bundle.parents.first().is_some_and(|parent|Some(parent.key.generation)==source.generation
                    && parent.key.signal==source.signal) {
                    return Err(RuntimeError::InvalidLayout("Derived bundle anchor does not belong to its declared source".into()));
                }
            }
            let run = state.runs.get_mut(&generation.id).ok_or(RuntimeError::UnknownGeneration)?;
            if run.sealed { return Err(RuntimeError::Sealed); }
            let revision = run.revision.checked_add(1).ok_or(RuntimeError::CounterExhausted)?;
            let mut names = HashSet::new();
            let mut prepared = Vec::with_capacity(bundle.inputs.len());
            // Every sibling is checked before any state or subscriber changes.
            for input in &bundle.inputs {
                let name=input.signal();
                if !names.insert(name.clone()) { return Err(RuntimeError::DuplicateSignal(name.to_string())); }
                let signal = run.signals.get(name).ok_or_else(|| RuntimeError::UnknownSignal(name.to_string()))?;
                let (data, placement, snapshot) = match input {
                    Input::Event {data,origin,..} => match (&signal.store,origin) {
                        (None,None) => (Some(data.with_schema(signal.schema.clone())
                            .map_err(|_| RuntimeError::SchemaChanged(name.to_string()))?), None,None),
                        (Some(store),Some(origin)) => {
                            let extent = [data.schema().domain(DomainRole::Repeat).size(),
                                data.schema().domain(DomainRole::Point).size()];
                            let placement = Placement::new(*origin,extent,data.clone()).map_err(RuntimeError::InvalidLayout)?;
                            store.validate(std::slice::from_ref(&placement)).map_err(RuntimeError::InvalidLayout)?;
                            (Some(data.clone()),Some(placement),None)
                        }
                        _ => return Err(RuntimeError::InvalidLayout("Only finite outputs require a Repeat/Point placement".into())),
                    },
                    Input::FiniteSnapshot {..} => {
                        if signal.store.is_none() {return Err(RuntimeError::InvalidLayout("A prefix publication requires a finite dataset store".into()));}
                        (None,None,None)
                    }
                    Input::Dataset {snapshot,..} => {
                        if signal.store.is_some() || signal.history_capable {
                            return Err(RuntimeError::InvalidLayout("Complete datasets require a Snapshot declaration".into()));
                        }
                        (None,None,Some(snapshot.with_schema(signal.schema.clone())
                            .map_err(|_|RuntimeError::SchemaChanged(name.to_string()))?))
                    }
                };
                if input.ordinal().is_some_and(|ordinal|signal.last_ordinal.is_some_and(|last|last>ordinal)) {
                    return Err(RuntimeError::OutOfOrder(name.to_string()));
                }
                prepared.push((input,data,placement,snapshot));
            }
            let mut prepared_publications = Vec::with_capacity(prepared.len());
            for (input,data,placement,snapshot) in prepared {
                let signal = run.signals.get_mut(input.signal()).unwrap();
                // The same lock still protects the prevalidated stores. No
                // fallible sibling installation remains after preflight.
                let canonical = snapshot.or_else(||signal.store.as_mut().map(|store|match placement {
                    Some(placement)=>store.commit(vec![placement]).expect("finite store preflight under Runtime lock"),
                    None=>store.snapshot(),
                }));
                prepared_publications.push(Publication {
                    key: PublicationKey { generation: generation.id, signal: input.signal().clone(), revision, ordinal: input.ordinal() },
                    data, run: run.record.clone(), event: event.clone(), canonical,
                    storage: if signal.store.is_some() { Storage::Finite }
                        else if signal.history_capable { Storage::Latest } else { Storage::Snapshot },
                    lineage:Weak::new(),
                });
            }
            // A current head and an in-flight Read own this flattened payload
            // closure. Publications retained in history carry only a Weak
            // link, so forty ROI samples do not pin forty full camera frames.
            let head=Arc::new_cyclic(|weak| {
                let mut closure=HashMap::new();
                if run.source.as_ref().is_some_and(|source|source.coherent) {
                    for parent in &bundle.parents {
                        if let Some(lineage)=parent.lineage.upgrade() {
                            closure.extend(lineage.publications.iter().map(|(key,value)|(key.clone(),value.clone())));
                        }
                        closure.insert(parent.key.clone(),parent.clone());
                    }
                }
                for mut publication in prepared_publications {
                    publication.lineage=weak.clone();
                    closure.insert(publication.key.clone(),Arc::new(publication));
                }
                PayloadLineage {publications:closure}
            });
            let publications: Arc<[_]> = bundle.inputs.iter().map(|input|head.publications[&PublicationKey {
                generation:generation.id,signal:input.signal().clone(),revision,ordinal:input.ordinal(),
            }].clone()).collect();
            for publication in publications.iter() {
                let signal = run.signals.get_mut(&publication.key.signal).unwrap();
                if let Some(ordinal)=publication.key.ordinal {signal.last_ordinal=Some(ordinal);}
                if let Some(previous)=signal.head.replace(head.clone()) {heads.push(previous);}
                if let Some(previous) = signal.latest.replace(publication.clone()) { garbage.push(previous); }
                if signal.capacity() > 0 {
                    // Recomputing the same source ordinal revises that sample;
                    // it is not an invented extra camera shot.
                    if signal.history.back().is_some_and(|p| p.key.ordinal == publication.key.ordinal) {
                        garbage.push(signal.history.pop_back().unwrap());
                    }
                    signal.history.push_back(publication.clone());
                }
                signal.trim(&mut garbage);
                signal.views.clear();
            }
            run.revision = revision;
            let subscriber_ids = run.subscribers.clone();
            let reads = publications.iter().cloned().map(|event|if event.data.is_some() {
                Read::from_event(event).expect("actual event")
            } else {Read::from_canonical(event)}).collect();
            committed = Arc::new(Delivery { kind:DeliveryKind::Data, generation: generation.id, revision: Some(revision), publications, reads });
            let State { runs, subscribers, .. } = &mut *state;
            let run = runs.get_mut(&generation.id).unwrap();
            for ticket in subscriber_ids.iter() {
                let Some(subscriber) = subscribers.get_mut(ticket).filter(|s| s.terminal.is_none()) else { continue; };
                let publications: Arc<[_]> = committed.publications.iter().filter(|p|
                    subscriber.requests.iter().any(|request| request.signal == p.key.signal
                        && (!matches!(request.mode,ReadMode::Event) || p.data.is_some()))).cloned().collect();
                if publications.is_empty() { continue; }
                let result = capture_reads(run,&subscriber.requests,&publications).and_then(|reads|
                    subscriber.enqueue(Arc::new(Delivery { kind:DeliveryKind::Data, generation: generation.id, revision: Some(revision), publications, reads }), &mut deliveries));
                if let Err(error) = result {
                    subscriber.terminal = Some(error);
                    subscriber.clear_queue(&mut deliveries);
                    release_demands(run,&subscriber.requests,*ticket,&mut garbage);
                    subscriber.changed.notify_all();
                }
            }
            notify_changes(&state);
        }
        drop((garbage, deliveries, heads));
        Ok(committed)
    }

    pub fn read_latest(&self, generation: &Generation, signal: &str) -> Result<Arc<Publication>> {
        let state = self.inner.state.lock().unwrap();
        signal_ref(&state, generation.id, signal)?.latest.clone().ok_or(RuntimeError::NoData)
    }

    pub fn schema(&self, generation: &Generation, signal: &str) -> Result<Arc<DatasetSchema>> {
        let state = self.inner.state.lock().unwrap();
        Ok(signal_ref(&state,generation.id,signal)?.schema.clone())
    }

    pub fn signals(&self) -> Vec<SignalDescription> {
        let state = self.inner.state.lock().unwrap();
        let mut descriptions: Vec<_> = state.runs.values().flat_map(|run| run.signals.iter().map(|(name,signal)| SignalDescription {
            generation:run.token.clone(),signal:name.clone(),schema:signal.schema.clone(),finite:signal.store.is_some(),
            published:signal.latest.is_some(),sealed:run.sealed,
            history_capable:signal.history_capable,history_capacity:signal.capacity(),retained_events:signal.history.len(),
        })).collect();
        descriptions.sort_unstable_by(|left,right|left.signal.cmp(&right.signal));
        descriptions
    }

    /// Acquire a group under one lock, so panels of one signal cannot mix shots
    /// just because another publication arrived halfway through their reads.
    pub fn read_views(&self, requests: &[(Generation,Arc<str>,Option<usize>)]) -> Vec<Result<Read>> {
        let mut state = self.inner.state.lock().unwrap();
        requests.iter().map(|(generation,name,width)| {
            let signal = signal_mut(&mut state,generation.id,name)?;
            if let Some(width) = width { return window_read(signal,*width).map(Read::from_window); }
            let event = signal.latest.clone().ok_or(RuntimeError::NoData)?;
            Ok(dataset_read(event))
        }).collect()
    }

    /// Select the same causal front as v2: a camera and its displayed ROI
    /// use the ROI's exact parent, not two independently captured latest values.
    /// Unrelated components never wait for one another; presentation followers
    /// are excluded from the source component by their declaration. `front` is
    /// the asking surface's own last selection, and becomes this one.
    pub fn read_coherent_views(&self, requests:&[(Generation,Arc<str>,Option<usize>)], front:&mut Front) -> Vec<Result<Read>> {
        let mut state=self.inner.state.lock().unwrap();
        let selected=coherent_front(&state,requests,front);
        let answers=requests.iter().map(|(generation,name,width)| {
            let read=selected.get(name).filter(|read|read.key().generation==generation.id)
                .ok_or(RuntimeError::NoData)?;
            let signal=signal_mut(&mut state,generation.id,name)?;
            if let Some(width)=width {window_read_at(signal,*width,read.anchor()).map(Read::from_window)}
            else {Ok(read.clone())}
        }).collect();
        drop(state);
        front.0=selected;
        answers
    }

    /// Metadata is available before the first sample. Waiting for data must
    /// not hide a known signal's axes or masquerade as a disconnected source.
    pub fn window_schema(&self, generation: &Generation, signal: &str, width: usize) -> Result<Arc<DatasetSchema>> {
        validate_window(width)?;
        let mut state = self.inner.state.lock().unwrap();
        let signal = signal_mut(&mut state,generation.id,signal)?;
        if width > 1 && !signal.demands.values().any(|value| *value == width) { return Err(RuntimeError::HistoryNotRetained); }
        validate_retention(signal,width)?;
        history_schema(signal,width)
    }

    pub fn read_at(&self, key: &PublicationKey) -> Result<Arc<Publication>> {
        let state = self.inner.state.lock().unwrap();
        let signal = signal_ref(&state, key.generation, &key.signal)?;
        signal.latest.iter().chain(signal.history.iter()).find(|p| p.key == *key).cloned()
            .ok_or(RuntimeError::PublicationExpired)
    }

    pub fn retain_history(&self, generation: &Generation, signal: &str, width: usize) -> Result<HistoryLease> {
        validate_window(width)?;
        let mut state = self.inner.state.lock().unwrap();
        let ticket = take_ticket(&mut state)?;
        let signal_state = signal_mut(&mut state, generation.id, signal)?;
        validate_retention(signal_state,width)?;
        signal_state.demands.insert(ticket, width);
        if signal_state.history.is_empty() {
            if let Some(latest) = &signal_state.latest { signal_state.history.push_back(latest.clone()); }
        }
        notify_changes(&state);
        Ok(HistoryLease { inner: Arc::downgrade(&self.inner), generation: generation.id,
            signal: Arc::from(signal), ticket })
    }

    pub fn read_window(&self, generation: &Generation, signal: &str, width: usize) -> Result<Arc<WindowRead>> {
        let mut state = self.inner.state.lock().unwrap();
        window_read(signal_mut(&mut state, generation.id, signal)?,width)
    }

    pub fn subscribe(&self, generation: &Generation, signals: Vec<Arc<str>>, policy: DeliveryPolicy, start: Start) -> Result<Subscription> {
        self.subscribe_reads(generation, signals.into_iter().map(|signal| ReadRequest {signal, mode:ReadMode::Event}).collect(),policy,start)
    }

    pub fn subscribe_reads(&self, generation: &Generation, requests: Vec<ReadRequest>, policy: DeliveryPolicy, start: Start) -> Result<Subscription> {
        self.subscribe_source(Some(generation.id),None,requests,policy,start)
    }

    /// Follow one explicitly named owner across runs. Lifecycle markers count
    /// toward the same exact queue bounds as data; no future run is polled.
    pub fn follow_reads(&self, owner: impl Into<Arc<str>>, requests: Vec<ReadRequest>, policy: DeliveryPolicy, start: Start) -> Result<Subscription> {
        let owner=owner.into();
        if owner.is_empty() || owner.trim()!=owner.as_ref() {
            return Err(RuntimeError::InvalidDeclaration("A canonical owner is required".into()));
        }
        self.subscribe_source(None,Some(owner),requests,policy,start)
    }

    fn subscribe_source(&self, generation: Option<GenerationId>, follow_owner: Option<Arc<str>>,
                        requests: Vec<ReadRequest>, policy: DeliveryPolicy, start: Start) -> Result<Subscription> {
        if requests.is_empty() { return Err(RuntimeError::EmptyBundle); }
        let mut seen = HashMap::new();
        let mut unique = Vec::new();
        for request in requests {
            if request.signal.is_empty() || request.signal.trim()!=request.signal.as_ref() {
                return Err(RuntimeError::InvalidDeclaration("A canonical signal is required".into()));
            }
            if let ReadMode::Window(width)=request.mode { validate_window(width)?; }
            match seen.insert(request.signal.clone(),request.mode) {
                None => unique.push(request),
                Some(mode) if mode == request.mode => {},
                Some(_) => return Err(RuntimeError::InvalidDeclaration("Each subscribed signal requires exactly one read mode".into())),
            }
        }
        if let DeliveryPolicy::Exact { max_events, max_referenced_bytes } = policy {
            if max_events == 0 || max_referenced_bytes == 0 { return Err(RuntimeError::InvalidWindow); }
        }
        let mut garbage = Vec::new();
        let mut deliveries = Vec::new();
        let result = (|| {
            let mut state = self.inner.state.lock().unwrap();
            if state.closed { return Err(RuntimeError::Closed); }
            let ticket = take_ticket(&mut state)?;
            let generation = follow_owner.as_ref().and_then(|owner|state.owners.get(owner).copied()).or(generation);
            let mut subscriber = Subscriber { generation:None, follow_owner, requests: unique, policy,
                queue: VecDeque::new(), bytes: 0, terminal:None, blocks:HashMap::new(),prefixes:HashMap::new(),
                changed: Arc::new(Condvar::new()), interrupted: false };
            if let Some(generation)=generation {
                let run=state.runs.get_mut(&generation).ok_or(RuntimeError::UnknownGeneration)?;
                subscriber.attach(run,ticket,start,&mut deliveries,&mut garbage)?;
            }
            let has_demand = subscriber.generation.is_some() && subscriber.requests.iter().any(|request| matches!(request.mode,ReadMode::Window(_)));
            state.subscribers.insert(ticket,subscriber);
            if has_demand { notify_changes(&state); }
            Ok(Subscription { inner: Arc::downgrade(&self.inner), ticket })
        })();
        drop((garbage,deliveries));
        result
    }

    pub fn seal(&self, generation: &Generation, outcome: Value) -> Result<()> {
        if !outcome.is_object() { return Err(RuntimeError::InvalidMetadata); }
        let mut garbage=Vec::new();
        let mut deliveries=Vec::new();
        let mut state = self.inner.state.lock().unwrap();
        let State {runs,subscribers,..}=&mut *state;
        let run = runs.get_mut(&generation.id).ok_or(RuntimeError::UnknownGeneration)?;
        if !run.sealed {
            run.sealed = true; run.outcome = Some(Arc::new(outcome));
            let ids = run.subscribers.clone();
            for ticket in ids.iter() {
                if let Some(subscriber) = subscribers.get_mut(ticket).filter(|s| s.terminal.is_none()) {
                    if subscriber.end_run(run,DeliveryKind::Sealed {outcome:run.outcome.clone().unwrap()},&mut deliveries).is_err() {
                        release_demands(run,&subscriber.requests,*ticket,&mut garbage);
                    }
                }
            }
            notify_changes(&state);
        }
        drop(state);
        drop((garbage,deliveries));
        Ok(())
    }

    pub fn retire(&self, generation: &Generation) -> Result<()> {
        let old;
        let mut deliveries=Vec::new();
        {
            let mut state = self.inner.state.lock().unwrap();
            old = remove_run(&mut state, generation.id, &mut deliveries).ok_or(RuntimeError::UnknownGeneration)?;
            notify_changes(&state);
        }
        drop((old,deliveries));
        Ok(())
    }

    pub fn outcome(&self, generation: &Generation) -> Result<Option<Arc<Value>>> {
        Ok(self.inner.state.lock().unwrap().runs.get(&generation.id)
            .ok_or(RuntimeError::UnknownGeneration)?.outcome.clone())
    }

    /// Explicit application shutdown wakes waiters and releases Runtime's
    /// references. Already acquired scientific reads remain independent.
    pub fn close(&self) {
        let mut deliveries = Vec::new();
        let runs;
        {
            let mut state = self.inner.state.lock().unwrap();
            state.closed = true;
            runs = std::mem::take(&mut state.runs);
            state.owners.clear(); state.signals.clear();
            for subscriber in state.subscribers.values_mut() {
                subscriber.terminal = Some(RuntimeError::Closed);
                subscriber.clear_queue(&mut deliveries);
                subscriber.changed.notify_all();
            }
            notify_changes(&state);
            state.watches.clear();
        }
        drop((runs, deliveries));
    }
}

fn notify_changes(state: &State) {
    for (sender, pending) in state.watches.values() {
        if !pending.swap(true, Ordering::AcqRel) { let _ = sender.try_send(()); }
    }
}

/// One coalesced wake token, never a publication queue. Clear pending before
/// acquiring current views, so a concurrent commit can schedule the next wake.
pub struct ChangeWatch {
    inner: Weak<Inner>, ticket: u64, receiver: Receiver<()>,
    pub pending: Arc<AtomicBool>,
}
impl ChangeWatch {
    pub fn wait(&self) -> Result<()> { self.receiver.recv().map_err(|_| RuntimeError::Closed) }
}
impl Drop for ChangeWatch {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() { inner.state.lock().unwrap().watches.remove(&self.ticket); }
    }
}

fn signal_ref<'a>(state: &'a State, generation: GenerationId, signal: &str) -> Result<&'a Signal> {
    state.runs.get(&generation).ok_or(RuntimeError::UnknownGeneration)?.signals.get(signal)
        .ok_or_else(|| RuntimeError::UnknownSignal(signal.into()))
}
fn signal_mut<'a>(state: &'a mut State, generation: GenerationId, signal: &str) -> Result<&'a mut Signal> {
    state.runs.get_mut(&generation).ok_or(RuntimeError::UnknownGeneration)?.signals.get_mut(signal)
        .ok_or_else(|| RuntimeError::UnknownSignal(signal.into()))
}
fn remove_run(state: &mut State, id: GenerationId, garbage: &mut Vec<Arc<Delivery>>) -> Option<Run> {
    let run = state.runs.remove(&id)?;
    state.owners.remove(&run.token.owner);
    for name in run.signals.keys() { state.signals.remove(name); }
    for ticket in run.subscribers.iter() {
        if let Some(subscriber) = state.subscribers.get_mut(ticket).filter(|s| s.terminal.is_none()) {
            let _=subscriber.end_run(&run,DeliveryKind::Retired,garbage);
        }
    }
    Some(run)
}
fn take_ticket(state: &mut State) -> Result<u64> {
    let ticket = state.next_ticket;
    state.next_ticket = ticket.checked_add(1).ok_or(RuntimeError::CounterExhausted)?;
    Ok(ticket)
}
fn validate_window(width: usize) -> Result<()> {
    if width == 0 || width > i64::MAX as usize { Err(RuntimeError::InvalidWindow) } else { Ok(()) }
}

pub struct HistoryLease { inner: Weak<Inner>, generation: GenerationId, signal: Arc<str>, ticket: u64 }
impl HistoryLease {
    pub fn resize(&self, width: usize) -> Result<()> {
        validate_window(width)?;
        let inner = self.inner.upgrade().ok_or(RuntimeError::Closed)?;
        let mut garbage = Vec::new();
        {
            let mut state = inner.state.lock().unwrap();
            let signal = signal_mut(&mut state, self.generation, &self.signal)?;
            validate_retention(signal,width)?;
            let demand = signal.demands.get_mut(&self.ticket).ok_or(RuntimeError::Closed)?;
            if *demand == width { return Ok(()); }
            *demand = width;
            if signal.trim(&mut garbage) { signal.view_epoch += 1; }
            notify_changes(&state);
        }
        drop(garbage);
        Ok(())
    }
}
impl Drop for HistoryLease {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else { return; };
        let mut garbage = Vec::new();
        {
            let mut state = inner.state.lock().unwrap();
            if let Ok(signal) = signal_mut(&mut state, self.generation, &self.signal) {
                signal.demands.remove(&self.ticket);
                if signal.trim(&mut garbage) { signal.view_epoch += 1; }
                notify_changes(&state);
            }
        }
        drop(garbage);
    }
}

pub struct Subscription { inner: Weak<Inner>, ticket: u64 }
impl Subscription {
    /// Interrupt one wait for control work without consuming an exact delivery.
    pub fn wake(&self) -> Result<()> {
        let inner = self.inner.upgrade().ok_or(RuntimeError::Closed)?;
        let mut state = inner.state.lock().unwrap();
        if state.closed { return Err(RuntimeError::Closed); }
        let subscriber = state.subscribers.get_mut(&self.ticket).ok_or(RuntimeError::Closed)?;
        subscriber.interrupted = true;
        subscriber.changed.notify_all();
        Ok(())
    }

    pub fn next(&self, timeout: Duration) -> Result<Option<Arc<Delivery>>> {
        let inner = self.inner.upgrade().ok_or(RuntimeError::Closed)?;
        let deadline = Instant::now().checked_add(timeout)
            .ok_or_else(|| RuntimeError::InvalidDeclaration("Timeout exceeds the monotonic clock range".into()))?;
        let mut state = inner.state.lock().unwrap();
        let changed = state.subscribers.get(&self.ticket).ok_or(RuntimeError::Closed)?.changed.clone();
        loop {
            let subscriber = state.subscribers.get_mut(&self.ticket).ok_or(RuntimeError::Closed)?;
            if std::mem::take(&mut subscriber.interrupted) { return Ok(None); }
            if let Some(delivery) = subscriber.queue.pop_front() {
                subscriber.account(&delivery, false);
                return Ok(Some(delivery));
            }
            if let Some(error) = &subscriber.terminal { return Err(error.clone()); }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else { return Ok(None); };
            if remaining.is_zero() { return Ok(None); }
            state = changed.wait_timeout(state, remaining).unwrap().0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::{DType, OwnedBuffer};
    use crate::dataset::Validity;
    use crate::schema::{Axis, Coordinates, Domain, ValueSchema, ValidityContract};

    fn schema() -> Arc<DatasetSchema> {
        let singleton = Domain::new(vec![1], vec![], vec![]).unwrap();
        let axis = |id: &str, len| Axis::new(id.into(), id.into(), "coordinate".into(),
            Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
        let cell = Domain::dense(vec![2, 3], vec![axis("y", 2), axis("x", 3)]).unwrap();
        DatasetSchema::new(singleton.clone(), singleton, cell,
            ValueSchema::new(DType::U16, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap()
    }
    fn block(schema: &Arc<DatasetSchema>, value: u16) -> Arc<DataBlock> {
        DataBlock::new(schema.clone(), OwnedBuffer::from_vec(vec![value; 6]).into(), Validity::Uniform(true), None).unwrap()
    }
    fn bundle(schema: &Arc<DatasetSchema>, signals: &[&str], ordinal: u64) -> Bundle {
        Bundle { inputs: signals.iter().map(|name| Input::Event { signal: (*name).into(), ordinal,
            data: block(schema, ordinal as u16), origin: None }).collect(), metadata: json!({}), parents: vec![] }
    }

    #[test]
    fn demand_history_gaps_frozen_derivation_and_restart_are_one_lifetime() {
        let plane = Runtime::new();
        let schema = schema();
        let generation = plane.begin("camera", vec![Declaration { signal: "image".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
        assert!(matches!(plane.retain_history(&generation,"image",i64::MAX as usize),Err(RuntimeError::InvalidLayout(_))));
        let first = plane.commit(&generation, &bundle(&schema, &["image"], 0)).unwrap();
        let old_key = first.publications[0].key.clone();
        let weak_first = Arc::downgrade(first.publications[0].data().unwrap());
        drop(first);
        plane.commit(&generation, &bundle(&schema, &["image"], 1)).unwrap();
        assert!(weak_first.upgrade().is_none()); // No demand, no hidden history.
        assert!(matches!(plane.read_at(&old_key), Err(RuntimeError::PublicationExpired)));
        let long = plane.retain_history(&generation, "image", 4).unwrap();
        plane.commit(&generation, &bundle(&schema, &["image"], 3)).unwrap();
        plane.commit(&generation, &bundle(&schema, &["image"], 4)).unwrap();
        let frozen = plane.read_window(&generation, "image", 4).unwrap();
        for _ in 0..4 { assert!(Arc::ptr_eq(&frozen, &plane.read_window(&generation, "image", 4).unwrap())); }
        let second_consumer = plane.retain_history(&generation,"image",4).unwrap();
        assert!(Arc::ptr_eq(&frozen,&plane.read_window(&generation,"image",4).unwrap()));
        drop(second_consumer);
        assert!(Arc::ptr_eq(&frozen,&plane.read_window(&generation,"image",4).unwrap()));
        assert_eq!(frozen.entries().len(), 3);
        assert_eq!(frozen.relative_index(frozen.at(-3).unwrap()), Some(-3));
        assert!(frozen.at(-2).is_none()); // Missing ordinal 2 is a real invalid hole.
        assert_eq!(frozen.schema.physical_shape(), &[1,4,2,3]);
        assert_eq!(frozen.placement(frozen.at(0).unwrap()), Some([0,3]));
        assert_eq!(frozen.schema.domain(DomainRole::Point).axes()[0].coordinate(0),
            Some(crate::schema::Coordinate::Integer(-3)));
        let parent = frozen.at(0).unwrap().clone();
        let parent_key = parent.key.clone();
        let parent_weak = Arc::downgrade(parent.data().unwrap());
        let replacement = bundle(&schema, &["image"], 4);
        let replaced = plane.commit(&generation, &replacement).unwrap();
        assert_eq!(replaced.publications[0].key.ordinal, Some(4));
        assert!(replaced.revision.unwrap() > parent_key.revision);
        let new_window = plane.read_window(&generation, "image", 4).unwrap();
        assert_eq!(new_window.entries().len(), 3);
        assert!(Arc::ptr_eq(&new_window.schema,&frozen.schema));
        let short = plane.retain_history(&generation, "image", 1).unwrap();
        drop(long);
        assert_eq!(plane.read_window(&generation, "image", 1).unwrap().entries().len(), 1);
        drop(short);
        assert_eq!(plane.read_window(&generation, "image", 1).unwrap().entries().len(),1);
        assert!(plane.inner.state.lock().unwrap().runs[&generation.id].signals["image"].history.is_empty());
        assert_eq!(frozen.entries().len(), 3); // Lease release cannot revoke acquired reads.
        plane.seal(&generation, json!({"reason":"operator stop"})).unwrap();
        assert_eq!(plane.read_latest(&generation, "image").unwrap().key.ordinal, Some(4));
        let restarted = plane.begin("camera", vec![Declaration { signal: "image".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
        assert_ne!(generation.id, restarted.id);
        assert!(matches!(plane.commit(&generation, &bundle(&schema, &["image"], 5)), Err(RuntimeError::UnknownGeneration)));
        assert!(matches!(plane.seal(&generation, json!({})), Err(RuntimeError::UnknownGeneration)));
        plane.commit(&restarted, &bundle(&schema, &["image"], 0)).unwrap();
        let restarted_lease = plane.retain_history(&restarted,"image",4).unwrap();
        let restarted_view = plane.read_window(&restarted,"image",4).unwrap();
        assert_eq!(restarted_view.schema.domain(DomainRole::Point).axes()[0].id(),
            frozen.schema.domain(DomainRole::Point).axes()[0].id());
        drop(restarted_lease);
        let detached = plane.begin("frozen analysis", vec![Declaration { signal: "answer".into(), schema: schema.clone(), storage: Storage::Snapshot }], json!({})).unwrap();
        assert!(!plane.signals().iter().find(|signal| signal.signal.as_ref() == "answer").unwrap().history_capable);
        assert!(matches!(plane.retain_history(&detached,"answer",4),Err(RuntimeError::InvalidLayout(_))));
        assert!(matches!(plane.read_window(&detached,"answer",1),Err(RuntimeError::InvalidLayout(_))));
        let derived = Bundle { parents: vec![parent.clone()], ..bundle(&schema, &["answer"], 0) };
        let answer = plane.commit(&detached, &derived).unwrap();
        plane.seal(&detached, json!({"complete":true})).unwrap();
        assert_eq!(answer.publications[0].event.parents[0].key, parent_key);
        assert_eq!(answer.publications[0].event.parents[0].run.owner.as_ref(), "camera");
        drop((derived, parent, frozen));
        assert!(parent_weak.upgrade().is_none()); // Causal records retain metadata, not pixels.
        plane.close();
        assert_eq!(answer.publications[0].data().unwrap().values().len(), 6);

        // The v2 same-shot contract follows a displayed ROI's parent, even
        // when raw acquisition is ahead. A nested ROI pins its in-flight
        // input lineage, while retained ROI history does not pin old cameras.
        let plane=Runtime::new();
        let mut board=Front::default();
        let declaration=|name:&str|vec![Declaration {signal:name.into(),schema:schema.clone(),storage:Storage::Latest}];
        let camera=plane.begin("camera",declaration("image"),json!({})).unwrap();
        let shot0=plane.commit(&camera,&bundle(&schema,&["image"],0)).unwrap();
        let camera0=Arc::downgrade(shot0.publications[0].data().unwrap());
        let roi=plane.begin_derived("roi",declaration("roi"),json!({}),shot0.reads[0].key(),true).unwrap();
        let roi_history=plane.retain_history(&roi,"roi",4).unwrap();
        plane.commit(&roi,&Bundle {parents:vec![shot0.reads[0].anchor().clone()],..bundle(&schema,&["roi"],0)}).unwrap();
        drop(shot0);
        let nested_input=plane.read_views(&[(roi.clone(),"roi".into(),None)]).pop().unwrap().unwrap();
        let nested=plane.begin_derived("nested",declaration("nested"),json!({}),nested_input.key(),true).unwrap();
        let shot1=plane.commit(&camera,&bundle(&schema,&["image"],1)).unwrap();
        let front=plane.read_coherent_views(&[(camera.clone(),"image".into(),None),(roi.clone(),"roi".into(),None)],&mut board);
        assert!(front.iter().all(|read|read.as_ref().unwrap().key().ordinal==Some(0)));
        drop(front);
        let window_requests=vec![(camera.clone(),"image".into(),None),(roi.clone(),"roi".into(),Some(4))];
        let first=plane.read_coherent_views(&window_requests,&mut board);
        let repeated=plane.read_coherent_views(&window_requests,&mut board);
        assert!(first[1].as_ref().unwrap().same_read(repeated[1].as_ref().unwrap()));
        drop((first,repeated));
        plane.commit(&roi,&Bundle {parents:vec![shot1.reads[0].anchor().clone()],..bundle(&schema,&["roi"],1)}).unwrap();
        drop(shot1);
        plane.commit(&camera,&bundle(&schema,&["image"],2)).unwrap();
        plane.commit(&nested,&Bundle {parents:vec![nested_input.anchor().clone()],..bundle(&schema,&["nested"],0)}).unwrap();
        let requests=vec![(camera.clone(),"image".into(),None),(roi.clone(),"roi".into(),None),(nested.clone(),"nested".into(),None)];
        let front=plane.read_coherent_views(&requests,&mut board);
        assert!(front.iter().all(|read|read.as_ref().unwrap().key().ordinal==Some(0)));
        drop((front,nested_input));
        let new_input=plane.read_views(&[(roi.clone(),"roi".into(),None)]).pop().unwrap().unwrap();
        plane.commit(&nested,&Bundle {parents:vec![new_input.anchor().clone()],..bundle(&schema,&["nested"],1)}).unwrap();
        let front=plane.read_coherent_views(&requests,&mut board);
        assert!(front.iter().all(|read|read.as_ref().unwrap().key().ordinal==Some(1)));
        assert_eq!(plane.read_window(&roi,"roi",4).unwrap().entries().len(),2);
        assert!(camera0.upgrade().is_none());
        drop((front,new_input,roi_history));
        // Presentation-paced fit followers must not hold the raw component.
        let current=plane.read_views(&[(camera.clone(),"image".into(),None)]).pop().unwrap().unwrap();
        let fit=plane.begin_derived("fit",declaration("fit"),json!({}),current.key(),false).unwrap();
        plane.commit(&fit,&Bundle {parents:vec![current.anchor().clone()],..bundle(&schema,&["fit"],2)}).unwrap();
        plane.commit(&camera,&bundle(&schema,&["image"],3)).unwrap();
        let front=plane.read_coherent_views(&[(camera.clone(),"image".into(),None),(fit,"fit".into(),None)],&mut board);
        assert_eq!(front[0].as_ref().unwrap().key().ordinal,Some(3));
        assert_eq!(front[1].as_ref().unwrap().key().ordinal,Some(2));
        plane.seal(&camera,json!({})).unwrap();
        let final_shot=plane.read_views(&[(camera.clone(),"image".into(),None)]).pop().unwrap().unwrap();
        let terminal=plane.begin_derived("terminal",declaration("terminal"),json!({}),final_shot.key(),true).unwrap();
        let terminal_bundle=Bundle {parents:vec![final_shot.anchor().clone()],..bundle(&schema,&["terminal"],3)};
        plane.commit(&terminal,&terminal_bundle).unwrap();
        let mismatched=Bundle {parents:vec![plane.read_latest(&roi,"roi").unwrap()],..bundle(&schema,&["terminal"],4)};
        assert!(matches!(plane.commit(&terminal,&mismatched),Err(RuntimeError::InvalidLayout(_))));
        assert_eq!(plane.read_latest(&terminal,"terminal").unwrap().key.ordinal,Some(3));
        let next_camera=plane.begin("camera",declaration("image"),json!({})).unwrap();
        plane.commit(&next_camera,&bundle(&schema,&["image"],0)).unwrap();
        // Ownership is checked inside the same lock as the actual mutation:
        // both an old pending commit and a new old-source route are superseded.
        assert!(matches!(plane.commit(&terminal,&terminal_bundle),Err(RuntimeError::SourceSuperseded)));
        assert!(matches!(plane.begin_derived("expired",declaration("expired"),json!({}),final_shot.key(),true),Err(RuntimeError::SourceSuperseded)));
        assert_eq!(plane.read_latest(&terminal,"terminal").unwrap().key.ordinal,Some(3));
        let front=plane.read_coherent_views(&[(next_camera,"image".into(),None),(roi,"roi".into(),None)],&mut board);
        assert!(front.iter().all(|read|matches!(read,Err(RuntimeError::NoData))));
    }

    #[test]
    fn a_judgement_and_its_picture_are_one_shot_whoever_else_consumes_them() {
        // v2's same-shot group through plugin processors: Occupancy lags the
        // camera, so the frames it judged -- not the camera's newest -- go with
        // its judgement; an undisplayed Frame Survival and another board's
        // reads change nothing on this board.
        let plane=Runtime::new();let schema=schema();
        let declaration=|name:&str|vec![Declaration {signal:name.into(),schema:schema.clone(),storage:Storage::Latest}];
        let judged=|source:&Arc<Delivery>,name:&str,ordinal|Bundle {parents:vec![source.publications[0].clone()],..bundle(&schema,&[name],ordinal)};
        let ordinals=|reads:Vec<Result<Read>>|reads.into_iter().map(|read|read.unwrap().key().ordinal.unwrap()).collect::<Vec<_>>();
        let camera=plane.begin("camera",declaration("frames"),json!({})).unwrap();
        let shot0=plane.commit(&camera,&bundle(&schema,&["frames"],0)).unwrap();
        let occupancy=plane.begin_consuming("occupancy",declaration("occupied"),json!({}),"frames".into(),Some(camera.id())).unwrap();
        let occupied0=plane.commit(&occupancy,&judged(&shot0,"occupied",0)).unwrap();
        let survival=plane.begin_consuming("survival",declaration("survival"),json!({}),"occupied".into(),Some(occupancy.id())).unwrap();
        plane.commit(&survival,&judged(&occupied0,"survival",0)).unwrap();
        let shot1=plane.commit(&camera,&bundle(&schema,&["frames"],1)).unwrap();
        plane.commit(&occupancy,&judged(&shot1,"occupied",1)).unwrap();
        let shot2=plane.commit(&camera,&bundle(&schema,&["frames"],2)).unwrap();
        let (mut board,mut other)=(Front::default(),Front::default());
        let overlay=vec![(camera.clone(),"frames".into(),None),(occupancy.clone(),"occupied".into(),None)];
        let pair=plane.read_coherent_views(&overlay,&mut board);
        assert_eq!(pair[0].as_ref().unwrap().shot_roots(),pair[1].as_ref().unwrap().shot_roots());
        assert_eq!(ordinals(pair),[1,1]);
        // A board that shows Frame Survival too shows its shot: one group, as v2.
        let mut chain=overlay.clone();chain.push((survival.clone(),"survival".into(),None));
        assert_eq!(ordinals(plane.read_coherent_views(&chain,&mut other)),[0,0,0]);
        assert_eq!(ordinals(plane.read_coherent_views(&overlay,&mut board)),[1,1]);
        // A processor is no derived owner: it still commits its replaced source
        // run's queue, and once finished it holds none of the next run's pictures.
        plane.seal(&camera,json!({})).unwrap();
        let next=plane.begin("camera",declaration("frames"),json!({})).unwrap();
        plane.commit(&next,&bundle(&schema,&["frames"],0)).unwrap();
        plane.commit(&occupancy,&judged(&shot2,"occupied",2)).unwrap();
        plane.seal(&occupancy,json!({})).unwrap();
        let pair=vec![(next,"frames".into(),None),(occupancy,"occupied".into(),None)];
        assert_eq!(ordinals(plane.read_coherent_views(&pair,&mut board)),[0,2]);
    }

    #[test]
    fn siblings_commit_atomically_and_exact_overflow_does_not_stall_latest() {
        let plane = Runtime::new(); let schema = schema();
        let mut board=Front::default();
        let generation = plane.begin("paired", ["a","b"].into_iter().map(|name|
            Declaration { signal: name.into(), schema: schema.clone(), storage: Storage::Latest }).collect(), json!({})).unwrap();
        let exact = plane.subscribe(&generation, vec!["a".into(),"b".into()],
            DeliveryPolicy::Exact { max_events: 2, max_referenced_bytes: 1024 }, Start::Next).unwrap();
        let latest = plane.subscribe(&generation, vec!["a".into()], DeliveryPolicy::Latest, Start::Next).unwrap();
        let first = plane.commit(&generation, &bundle(&schema, &["a","b"], 0)).unwrap();
        assert!(Arc::ptr_eq(first.publications[0].event(), first.publications[1].event()));
        let child=plane.begin_derived("child",vec![Declaration {signal:"child".into(),schema:schema.clone(),storage:Storage::Latest}],
            json!({}),first.publications[0].key(),true).unwrap();
        plane.commit(&child,&Bundle {parents:vec![first.publications[0].clone()],..bundle(&schema,&["child"],0)}).unwrap();
        let mut invalid = bundle(&schema, &["a","missing"], 1);
        let same_data=block(&self::schema(),1); // Equal, separately rebuilt schema.
        if let Input::Event {data,..}=&mut invalid.inputs[0] {*data=same_data.clone();}
        assert!(matches!(plane.commit(&generation, &invalid), Err(RuntimeError::UnknownSignal(_))));
        assert_eq!(plane.read_latest(&generation,"a").unwrap().key.revision, 1);
        if let Input::Event {signal,..}=&mut invalid.inputs[1] {*signal="b".into();}
        assert_eq!(plane.commit(&generation, &invalid).unwrap().revision, Some(2));
        let canonical = plane.read_latest(&generation,"a").unwrap();
        assert!(Arc::ptr_eq(canonical.data().unwrap().schema(),&schema));
        assert_eq!(canonical.data().unwrap().values().as_bytes().as_ptr(), same_data.values().as_bytes().as_ptr());
        assert_eq!(plane.commit(&generation, &bundle(&schema, &["a","b"], 2)).unwrap().revision, Some(3));
        let sibling_front=plane.read_coherent_views(&[(generation.clone(),"b".into(),None),(child,"child".into(),None)],&mut board);
        assert!(sibling_front.iter().all(|read|read.as_ref().unwrap().key().ordinal==Some(0)));
        assert!(matches!(exact.next(Duration::ZERO), Err(RuntimeError::SubscriptionOverflow {events:3,..})));
        let delivered = latest.next(Duration::ZERO).unwrap().unwrap();
        assert_eq!(delivered.revision, Some(3)); assert_eq!(delivered.publications.len(), 1);
        let current = plane.subscribe(&generation, vec!["a".into()], DeliveryPolicy::Latest, Start::Current).unwrap();
        assert_eq!(current.next(Duration::ZERO).unwrap().unwrap().revision, None);
        plane.commit(&generation, &bundle(&schema, &["a"], 3)).unwrap();
        assert_eq!(plane.read_latest(&generation,"b").unwrap().key.ordinal, Some(2));
        let snapshot = plane.subscribe(&generation,vec!["a".into(),"b".into(),"a".into()],DeliveryPolicy::Latest,Start::Current).unwrap()
            .next(Duration::ZERO).unwrap().unwrap();
        assert_eq!(snapshot.revision,None);
        assert_eq!(snapshot.publications.len(),2);
        assert_eq!(snapshot.publications[0].key.revision,4);
        assert_eq!(snapshot.publications[1].key.revision,3);
        plane.seal(&generation, json!({"complete":true})).unwrap();
        assert_eq!(latest.next(Duration::ZERO).unwrap().unwrap().revision, Some(4));
        assert!(matches!(latest.next(Duration::ZERO), Err(RuntimeError::Sealed)));

        // The same atomic contract applies when each signal is an append-only
        // finite scan: a bad sibling cannot install half a scan point.
        let axis = Axis::new("scan".into(),"scan".into(),"coordinate".into(),
            Coordinates::Index {origin:0,len:3},None,None,None,None).unwrap();
        let full = DatasetSchema::new(schema.domain(DomainRole::Repeat).clone(),
            Domain::dense(vec![3],vec![axis]).unwrap(),schema.domain(DomainRole::Cell).clone(),schema.value().clone()).unwrap();
        let finite = plane.begin("paired", ["a","b"].into_iter().map(|name|
            Declaration {signal:name.into(),schema:full.clone(),storage:Storage::Finite}).collect(),json!({})).unwrap();
        let finite_reads = plane.subscribe_reads(&finite,vec![ReadRequest {signal:"a".into(),mode:ReadMode::Dataset}],
            DeliveryPolicy::Exact {max_events:3,max_referenced_bytes:36},Start::Next).unwrap();
        let finite_events = plane.subscribe(&finite,vec!["a".into()],
            DeliveryPolicy::Exact {max_events:3,max_referenced_bytes:36},Start::Next).unwrap();
        let mut sample = bundle(&schema,&["a","b"],0);
        for input in &mut sample.inputs {if let Input::Event {origin,..}=input {*origin=Some([0,0]);}}
        let frozen = plane.commit(&finite,&sample).unwrap();
        let prefix = frozen.publications[0].canonical().unwrap();
        assert_eq!(prefix.schema().physical_shape(),&[1,3,2,3]);
        assert_eq!(prefix.holes(),2);
        if let Input::Event {origin,..}=&mut sample.inputs[0] {*origin=Some([0,1]);}
        for input in &mut sample.inputs {if let Input::Event {ordinal,..}=input {*ordinal=1;}}
        assert!(matches!(plane.commit(&finite,&sample),Err(RuntimeError::InvalidLayout(_))));
        assert_eq!(plane.read_latest(&finite,"a").unwrap().key.revision,1);
        if let Input::Event {origin,..}=&mut sample.inputs[1] {*origin=Some([0,1]);}
        if let Input::Event {data,..}=&mut sample.inputs[0] {*data=block(&schema,1);}
        let second = plane.commit(&finite,&sample).unwrap();
        assert_eq!(second.revision,Some(2));
        assert_eq!(second.publications[0].canonical().unwrap().holes(),1);
        assert_eq!(prefix.holes(),2);
        assert_eq!(prefix.payload_bytes(),12);
        assert_eq!(second.publications[0].canonical().unwrap().payload_bytes(),24);
        let cross_run = plane.follow_reads("paired",vec![ReadRequest {signal:"a".into(),mode:ReadMode::Dataset}],
            DeliveryPolicy::Exact {max_events:8,max_referenced_bytes:36},Start::Current).unwrap();
        plane.seal(&finite,json!({"reason":"operator stop"})).unwrap();
        assert_eq!(plane.read_latest(&finite,"a").unwrap().canonical().unwrap().written_cells(),2);
        plane.retire(&finite).unwrap();
        assert!(Arc::ptr_eq(prefix.segments().next().unwrap().data(),frozen.publications[0].data().unwrap()));
        for holes in [2,1] {
            let delivery = finite_reads.next(Duration::ZERO).unwrap().unwrap();
            assert!(matches!(delivery.reads[0].kind(),ReadKind::Canonical(event) if event.canonical().unwrap().holes()==holes));
            assert!(delivery.reads[0].is_complete_dataset());
            let event = finite_events.next(Duration::ZERO).unwrap().unwrap();
            assert!(matches!(event.reads[0].kind(),ReadKind::Event(_)));
            assert!(!event.reads[0].is_complete_dataset());
            assert_eq!(event.publications[0].storage(),Storage::Finite);
        }
        assert!(matches!(finite_reads.next(Duration::ZERO),Err(RuntimeError::Sealed)));
        let restarted = plane.begin("paired",vec![Declaration {signal:"a".into(),schema:full.clone(),storage:Storage::Finite}],json!({})).unwrap();
        let mut next = bundle(&schema,&["a"],0);
        if let Input::Event {origin,..}=&mut next.inputs[0] {*origin=Some([0,0]);}
        plane.commit(&restarted,&next).unwrap();
        assert!(matches!(cross_run.next(Duration::ZERO),Err(RuntimeError::SubscriptionOverflow {events:4,referenced_bytes:48})));

        // Declared, zero-sample finite data is a real canonical prefix, not
        // a dense invalid shot. Raw event consumers must not receive it.
        let plane=Runtime::new();
        let declared=plane.begin("empty",vec![Declaration {signal:"scan".into(),schema:full,storage:Storage::Finite}],json!({})).unwrap();
        let declared_schema=plane.schema(&declared,"scan").unwrap();
        let datasets=plane.subscribe_reads(&declared,vec![ReadRequest {signal:"scan".into(),mode:ReadMode::Dataset}],
            DeliveryPolicy::Exact {max_events:8,max_referenced_bytes:128},Start::Next).unwrap();
        let events=plane.subscribe(&declared,vec!["scan".into()],DeliveryPolicy::Exact {max_events:8,max_referenced_bytes:128},Start::Next).unwrap();
        let prefix_bundle=Bundle {inputs:vec![Input::FiniteSnapshot {signal:"scan".into()}],metadata:json!({}),parents:vec![]};
        let empty=plane.commit(&declared,&prefix_bundle).unwrap();
        assert!(empty.publications[0].data().is_none());
        assert_eq!(empty.publications[0].key.ordinal,None);
        let prefix=empty.publications[0].canonical().unwrap();
        assert_eq!((prefix.segment_count(),prefix.written_cells(),prefix.holes()),(0,0,3));
        assert!(matches!(Read::from_event(empty.publications[0].clone()),Err(RuntimeError::NoData)));
        assert!(events.next(Duration::ZERO).unwrap().is_none());
        let late_raw=plane.subscribe(&declared,vec!["scan".into()],DeliveryPolicy::Latest,Start::Current).unwrap();
        assert!(late_raw.next(Duration::ZERO).unwrap().is_none());
        let frozen_empty=datasets.next(Duration::ZERO).unwrap().unwrap();
        assert!(matches!(frozen_empty.reads[0].kind(),ReadKind::Canonical(_)));
        let invalid=DataBlock::new(schema.clone(),OwnedBuffer::from_vec(vec![1u16;6]).into(),Validity::Uniform(false),None).unwrap();
        let actual=plane.commit(&declared,&Bundle {inputs:vec![Input::Event {signal:"scan".into(),ordinal:0,
            data:invalid,origin:Some([0,0])}],metadata:json!({}),parents:vec![]}).unwrap();
        assert_eq!(actual.publications[0].key.ordinal,Some(0));
        assert_eq!(actual.publications[0].canonical().unwrap().written_cells(),1);
        assert_eq!(events.next(Duration::ZERO).unwrap().unwrap().publications[0].key.ordinal,Some(0));
        plane.commit(&declared,&prefix_bundle).unwrap();
        assert!(events.next(Duration::ZERO).unwrap().is_none());
        plane.seal(&declared,json!({})).unwrap();
        let held=plane.read_views(&[(declared.clone(),"scan".into(),None)]).pop().unwrap().unwrap();
        assert_eq!(held.anchor().canonical().unwrap().written_cells(),1);
        plane.retire(&declared).unwrap();
        assert_eq!(held.anchor().canonical().unwrap().written_cells(),1);
        assert_eq!(frozen_empty.reads[0].anchor().canonical().unwrap().written_cells(),0);

        // The same complete-Dataset input atomically carries a partial image
        // and its empty companion without pretending either is a new shot.
        let replacement=plane.begin("snapshot",["image","status"].into_iter().map(|signal|Declaration {
            signal:signal.into(),schema:declared_schema.clone(),storage:Storage::Snapshot}).collect(),json!({})).unwrap();
        let event_reader=plane.subscribe(&replacement,vec!["image".into(),"status".into()],DeliveryPolicy::Latest,Start::Next).unwrap();
        let data=held.anchor().canonical().unwrap().clone().with_window(Some((-3,0))).unwrap();
        let zero=frozen_empty.reads[0].anchor().canonical().unwrap().clone();
        let publish=|image:Snapshot,status:Snapshot|plane.commit(&replacement,&Bundle {inputs:vec![
            Input::Dataset {signal:"image".into(),ordinal:None,snapshot:image},
            Input::Dataset {signal:"status".into(),ordinal:None,snapshot:status}],metadata:json!({}),parents:vec![]});
        let replaced=publish(data.clone(),zero.clone()).unwrap();
        assert!(replaced.publications.iter().all(|value|value.data().is_none() && value.key.ordinal.is_none()));
        assert!(Arc::ptr_eq(replaced.publications[0].event(),replaced.publications[1].event()));
        assert_eq!(replaced.publications[0].canonical().unwrap().window(),Some((-3,0)));
        assert!(event_reader.next(Duration::ZERO).unwrap().is_none());
        assert_eq!(plane.read_latest(&replacement,"image").unwrap().canonical().unwrap().written_cells(),1);
        publish(zero.clone(),data).unwrap();
        assert_eq!(plane.read_latest(&replacement,"image").unwrap().canonical().unwrap().written_cells(),0);
        let invalid=DatasetStore::new(schema.clone()).snapshot();
        assert!(matches!(publish(zero.clone(),invalid),Err(RuntimeError::SchemaChanged(_))));
        assert_eq!(plane.read_latest(&replacement,"status").unwrap().canonical().unwrap().written_cells(),1);
        let bounded=plane.subscribe_reads(&replacement,vec![ReadRequest {signal:"image".into(),mode:ReadMode::Dataset}],
            DeliveryPolicy::Exact {max_events:4,max_referenced_bytes:24},Start::Next).unwrap();
        for value in 0..3 {
            let mut store=DatasetStore::new(declared_schema.clone());
            let data=store.commit(vec![Placement::new([0,0],[1,1],block(&schema,value)).unwrap()]).unwrap();
            publish(data,zero.clone()).unwrap();
        }
        assert!(matches!(bounded.next(Duration::ZERO),Err(RuntimeError::SubscriptionOverflow {events:3,referenced_bytes:36})));
    }

    #[test]
    fn exact_delivery_order_survives_concurrent_publication_and_seal() {
        let plane = Runtime::new(); let schema = schema();
        let changes = plane.watch_changes().unwrap();
        let pending = changes.pending.clone();
        let transient = plane.watch_changes().unwrap();
        let registration = Arc::downgrade(&transient.pending);
        drop(transient);
        assert!(registration.upgrade().is_none());
        let acknowledge = || {
            assert_eq!(changes.receiver.try_recv(), Ok(()));
            assert!(matches!(changes.receiver.try_recv(), Err(mpsc::TryRecvError::Empty)));
            assert!(pending.swap(false, Ordering::AcqRel));
        };
        let generation = plane.begin("source", vec![Declaration {signal:"value".into(),schema:schema.clone(),storage:Storage::Latest}],json!({})).unwrap();
        let history = plane.retain_history(&generation,"value",4).unwrap();
        acknowledge(); // Begin and demand changes coalesce into one token.
        history.resize(4).unwrap();
        assert!(matches!(changes.receiver.try_recv(), Err(mpsc::TryRecvError::Empty)));
        history.resize(3).unwrap(); history.resize(4).unwrap();
        acknowledge();
        let metadata = plane.window_schema(&generation,"value",4).unwrap();
        assert!(Arc::ptr_eq(&plane.schema(&generation,"value").unwrap(),&schema));
        assert!(matches!(plane.read_latest(&generation,"value"),Err(RuntimeError::NoData)));
        let exact = plane.subscribe(&generation,vec!["value".into()],DeliveryPolicy::Exact {max_events:128,max_referenced_bytes:4096},Start::Next).unwrap();
        plane.commit(&generation, &bundle(&schema, &["value"], 0)).unwrap();
        exact.wake().unwrap(); exact.wake().unwrap();
        assert!(exact.next(Duration::ZERO).unwrap().is_none()); // Control wake leaves ordinal 0 queued.
        let producer = plane.clone(); let token = generation.clone();
        let thread = std::thread::spawn(move || {
            for ordinal in 1..128 { producer.commit(&token,&bundle(&schema,&["value"],ordinal)).unwrap(); }
            producer.seal(&token,json!({"complete":true})).unwrap();
        });
        for ordinal in 0..128 {
            let delivery = exact.next(Duration::from_secs(2)).unwrap().unwrap();
            assert_eq!(delivery.revision,Some(ordinal+1));
            assert_eq!(delivery.publications[0].key.ordinal,Some(ordinal));
        }
        thread.join().unwrap();
        assert!(matches!(exact.next(Duration::ZERO),Err(RuntimeError::Sealed)));
        acknowledge(); // All 128 commits and seal leave one bounded, payload-free wake.
        plane.seal(&generation, json!({})).unwrap();
        assert!(matches!(changes.receiver.try_recv(), Err(mpsc::TryRecvError::Empty)));
        let held = plane.read_latest(&generation,"value").unwrap();
        assert!(Arc::ptr_eq(&metadata,&plane.read_window(&generation,"value",4).unwrap().schema));
        drop(history);
        acknowledge();
        plane.retire(&generation).unwrap();
        acknowledge();
        plane.close();
        changes.wait().unwrap();
        assert!(pending.load(Ordering::Acquire));
        assert!(matches!(changes.wait(), Err(RuntimeError::Closed)));
        drop(plane);
        assert_eq!(held.key.ordinal,Some(127));
        assert!(matches!(exact.next(Duration::ZERO),Err(RuntimeError::Closed)));

        // Adjacent exact windows share their forty owners, but each retains
        // its original revision even after source history and the run disappear.
        let plane = Runtime::new();
        let schema = self::schema();
        let generation = plane.begin("source",vec![Declaration {signal:"value".into(),schema:schema.clone(),storage:Storage::Latest}],json!({})).unwrap();
        let warm = plane.retain_history(&generation,"value",40).unwrap();
        let first = plane.commit(&generation,&bundle(&schema,&["value"],0)).unwrap();
        let weak = Arc::downgrade(first.publications[0].data().unwrap());
        drop(first);
        for ordinal in 1..40 { plane.commit(&generation,&bundle(&schema,&["value"],ordinal)).unwrap(); }
        let request = || ReadRequest {signal:"value".into(),mode:ReadMode::Window(40)};
        let windows = plane.subscribe_reads(&generation,vec![request(),request()],
            DeliveryPolicy::Exact {max_events:82,max_referenced_bytes:81*12},Start::Current).unwrap();
        let overflow = plane.subscribe_reads(&generation,vec![request()],
            DeliveryPolicy::Exact {max_events:82,max_referenced_bytes:41*12},Start::Current).unwrap();
        drop(warm);
        assert_eq!(plane.signals()[0].history_capacity,40);
        for ordinal in 40..81 { plane.commit(&generation,&bundle(&schema,&["value"],ordinal)).unwrap(); }
        assert!(matches!(overflow.next(Duration::ZERO),Err(RuntimeError::SubscriptionOverflow {events:3,referenced_bytes:504})));
        drop(overflow);
        plane.seal(&generation,json!({})).unwrap();
        plane.retire(&generation).unwrap();
        windows.wake().unwrap();
        assert!(windows.next(Duration::ZERO).unwrap().is_none());
        let mut frozen = None;
        for ordinal in 39..81 {
            let delivery = windows.next(Duration::ZERO).unwrap().unwrap();
            assert_eq!(delivery.reads.len(),1);
            assert_eq!(delivery.revision,if ordinal==39 {None} else {Some(ordinal+1)});
            let ReadKind::Window(window) = delivery.reads[0].kind() else { panic!("exact window") };
            assert_eq!(window.latest.ordinal,Some(ordinal));
            assert_eq!(window.at(-39).unwrap().key.ordinal,Some(ordinal-39));
            assert_eq!(window.at(0).unwrap().key,delivery.publications[0].key);
            assert!(delivery.reads[0].is_complete_dataset());
            if ordinal==39 { frozen=Some(window.clone()); }
        }
        assert!(matches!(windows.next(Duration::ZERO),Err(RuntimeError::Sealed)));
        assert!(weak.upgrade().is_some());
        drop(frozen);
        assert!(weak.upgrade().is_none());

        let plane=Runtime::new();
        let follow=plane.follow_reads("camera",vec![ReadRequest {signal:"image".into(),mode:ReadMode::Window(3)}],
            DeliveryPolicy::Exact {max_events:12,max_referenced_bytes:60},Start::Current).unwrap();
        assert!(follow.next(Duration::ZERO).unwrap().is_none());
        let declaration=|| vec![Declaration {signal:"image".into(),schema:schema.clone(),storage:Storage::Latest}];
        let first=plane.begin("camera",declaration(),json!({})).unwrap();
        assert_eq!(plane.signals()[0].history_capacity,3);
        for ordinal in 0..3 { plane.commit(&first,&bundle(&schema,&["image"],ordinal)).unwrap(); }
        plane.seal(&first,json!({"run":1})).unwrap();
        let second=plane.begin("camera",declaration(),json!({})).unwrap();
        assert_eq!(plane.signals()[0].history_capacity,3);
        for ordinal in 0..2 { plane.commit(&second,&bundle(&schema,&["image"],ordinal)).unwrap(); }
        plane.seal(&second,json!({"run":2})).unwrap();
        for (index,(generation,count)) in [(&first,3),(&second,2)].into_iter().enumerate() {
            for ordinal in 0..count {
                let delivery=follow.next(Duration::ZERO).unwrap().unwrap();
                assert_eq!(delivery.kind,DeliveryKind::Data);
                assert_eq!(delivery.generation,generation.id);
                let ReadKind::Window(window)=delivery.reads[0].kind() else {panic!("followed window")};
                assert_eq!(window.latest.ordinal,Some(ordinal));
                assert_eq!(window.entries().len(),ordinal as usize+1);
                assert!(window.entries().iter().all(|entry|entry.key.generation==generation.id));
            }
            let sealed=follow.next(Duration::ZERO).unwrap().unwrap();
            assert_eq!(sealed.generation,generation.id);
            assert_eq!(sealed.kind,DeliveryKind::Sealed {outcome:Arc::new(json!({"run":index+1}))});
            assert!(sealed.reads.is_empty() && sealed.publications.is_empty());
            if index==0 {
                let retired=follow.next(Duration::ZERO).unwrap().unwrap();
                assert_eq!(retired.generation,first.id);
                assert_eq!(retired.kind,DeliveryKind::Retired);
            }
        }
        assert!(follow.next(Duration::ZERO).unwrap().is_none());
        drop(follow);
        assert_eq!(plane.signals()[0].history_capacity,0);
        let third=plane.begin("camera",declaration(),json!({})).unwrap();
        plane.commit(&third,&bundle(&schema,&["image"],0)).unwrap();
        assert_eq!(plane.signals()[0].retained_events,0);
        let latest=plane.follow_reads("camera",vec![ReadRequest {signal:"image".into(),mode:ReadMode::Event}],
            DeliveryPolicy::Latest,Start::Current).unwrap();
        for ordinal in 1..20 {plane.commit(&third,&bundle(&schema,&["image"],ordinal)).unwrap();}
        plane.seal(&third,json!({})).unwrap();
        assert_eq!(latest.next(Duration::ZERO).unwrap().unwrap().reads[0].key().ordinal,Some(19));
        assert!(matches!(latest.next(Duration::ZERO).unwrap().unwrap().kind,DeliveryKind::Sealed {..}));
        // An unattended Latest follower cannot build a lifecycle FIFO over
        // repeated restarts. The final Data still precedes that run's seal.
        for _ in 0..20 {
            let run=plane.begin("camera",declaration(),json!({})).unwrap();
            for ordinal in 0..2 {plane.commit(&run,&bundle(&schema,&["image"],ordinal)).unwrap();}
            plane.seal(&run,json!({})).unwrap();
        }
        assert_eq!(plane.inner.state.lock().unwrap().subscribers[&latest.ticket].queue.len(),2);
        assert_eq!(latest.next(Duration::ZERO).unwrap().unwrap().reads[0].key().ordinal,Some(1));
        assert!(matches!(latest.next(Duration::ZERO).unwrap().unwrap().kind,DeliveryKind::Sealed {..}));
        assert!(latest.next(Duration::ZERO).unwrap().is_none());
    }
}
impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            let retired;
            let mut garbage = Vec::new();
            {
                let mut state = inner.state.lock().unwrap();
                retired = state.subscribers.remove(&self.ticket);
                if let Some(run) = retired.as_ref().and_then(|subscriber| subscriber.generation.and_then(|generation|state.runs.get_mut(&generation))) {
                    run.subscribers = run.subscribers.iter().copied().filter(|id| *id != self.ticket).collect();
                    release_demands(run,&retired.as_ref().unwrap().requests,self.ticket,&mut garbage);
                    if retired.as_ref().unwrap().requests.iter().any(|request| matches!(request.mode,ReadMode::Window(_))) {
                        notify_changes(&state);
                    }
                }
            }
            drop((retired,garbage));
        }
    }
}
