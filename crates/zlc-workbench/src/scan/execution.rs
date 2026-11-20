//! Application-owned scan execution. The plugin host supplies only generic
//! run resources, exact input and publication capabilities through Context.
use std::any::Any;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::runtime::{Bundle, Generation, GenerationId, Input, Read, ReadKind};
use zlc_core::schema::{DatasetSchema, DomainRole};
use zlc_core::scan::Plan;

pub type Resource = Arc<dyn Any + Send + Sync>;
const RESOURCE: &str = "zlc.workbench.scan";

/// Object-safe capabilities provided by the application adapter. No transport
/// type, window, device, mutable global store or file operation is exposed.
/// install_resource must refuse first installation after outputs have begun;
/// begin_output binds the currently consumed exact-input subscription.
pub trait Context {
    fn check_active(&self) -> Result<(), String>;
    fn input_next(&self, timeout: Duration, terminal_errors: bool) -> Result<Option<Read>, String>;
    fn input_drain(&self, quiet: Duration) -> Result<usize, String>;
    fn input_await_new_source(&self) -> Result<(), String>;
    fn input_generation(&self) -> Result<Option<GenerationId>, String>;
    fn begin_output(&self, name: &str, schema: Arc<DatasetSchema>, record: Value) -> Result<(Generation, Arc<str>), String>;
    fn commit(&self, generation: &Generation, bundle: &Bundle) -> Result<(), String>;
    fn resource(&self, key: &str) -> Result<Option<Resource>, String>;
    fn install_resource(&self, key: &str, value: Resource) -> Result<Resource, String>;
}

fn invalid(message: impl Into<String>) -> String { message.into() }
fn integer(params: &Value, name: &str) -> Result<usize, String> {
    params[name].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| format!("{name} must be a nonnegative integer"))
}
fn seconds(params: &Value, name: &str, default: f64, maximum: f64) -> Result<Duration, String> {
    let value = match params.get(name) { None => default, Some(value) => value.as_f64().ok_or_else(|| format!("{name} must be seconds"))? };
    if !value.is_finite() || value < 0.0 || value > maximum { return Err(format!("{name} must be between 0 and {maximum} s")); }
    Ok(Duration::from_secs_f64(value))
}

struct Configuration { output: String, record: Value }
struct Publication { generation: Generation, signal: Arc<str>, source_schema: Arc<DatasetSchema>, source_generation: GenerationId }
struct Segment {
    row_offset: usize,
    scan_repeat_base: usize,
    total: usize,
    written: usize,
    previous_ordinal: Option<u64>,
    actual_devices: Value,
}

/// What anyone may read of a scan at any time (the Pulse bridge on the window's turn): its
/// handle, its plan and whether it failed. Held only to read or swap them, never across a wait.
struct Facts {
    handle: String,
    plan: Arc<Plan>,
    configured: bool,
    failure: Option<String>,
    pulse_applied: Option<Value>,
}

/// The collection's own state, used by one scan operation at a time (they wait on the source).
struct Work {
    configuration: Option<Configuration>,
    publication: Option<Publication>,
    segment: Option<Segment>,
    written: usize,
}

struct Scan { facts: Mutex<Facts>, work: Mutex<Work> }

impl Scan {
    fn facts(&self) -> std::sync::MutexGuard<'_, Facts> { self.facts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }
    /// The plan of the current handle, unless the scan failed.
    fn current(&self, handle: &str) -> Result<Arc<Plan>, String> {
        let facts = self.facts();
        if facts.handle != handle { return Err("the scan handle is not current for this run".into()); }
        if let Some(error) = &facts.failure { return Err(error.clone()); }
        Ok(facts.plan.clone())
    }
}

fn scan(context: &dyn Context) -> Result<Arc<Scan>, String> {
    context.check_active()?;
    context.resource(RESOURCE)?.ok_or("prepare a scan before using its plan")?
        .downcast::<Scan>().map_err(|_| "the run scan resource has a different type".into())
}

/// Immutable geometry passed directly to the native Pulse bridge. Never waits for a scan
/// operation in progress (an arm waiting for its source, a collect waiting for events).
pub fn plan(handle: &str, context: &dyn Context) -> Result<Arc<Plan>, String> {
    scan(context)?.current(handle)
}

/// Full LOAD/FIRE facts stay native; configure freezes the run's first receipt.
pub fn accept_pulse(handle: &str, effective_inner: Option<Vec<Vec<f64>>>, applied: Value, context: &dyn Context) -> Result<(), String> {
    let scan = scan(context)?;
    let mut facts = scan.facts();
    if facts.handle != handle { return Err("the scan handle is not current for this run".into()); }
    if let Some(error) = &facts.failure { return Err(error.clone()); }
    if let Some(rows) = effective_inner {
        let plan = facts.plan.with_effective_inner(&rows)?;
        if facts.configured && plan.axes() != facts.plan.axes() {
            return Err("the board changed effective scan coordinates during the run".into());
        }
        facts.plan = Arc::new(plan);
    }
    if !facts.configured { facts.pulse_applied = Some(applied); }
    else if facts.pulse_applied.as_ref().is_some_and(|first| first["config"] != applied["config"]) {
        return Err("the Config file changed during the scan".into());
    }
    Ok(())
}

pub fn dispatch(op: &str, params: &Value, context: &dyn Context) -> Result<Value, String> {
    context.check_active()?;
    if op == "run.scan.prepare" {
        // Only admission touches a run resource registry; geometry is validated
        // before the atomic installation, outside any host registry lock.
        context.input_generation()?;
        if context.resource(RESOURCE)?.is_some() { return Err("this run already prepared its scan or outputs".into()); }
        let plan = Plan::new(super::wire::plan_spec(&params["spec"])?)?;
        let handle = uuid::Uuid::new_v4().to_string();
        let answer = json!({"handle":handle,"outer_count":plan.outer_count(),"inner_count":plan.inner_count(),
            "point_count":plan.point_count(),"total_shots":plan.total_shots(),"outer_width":plan.outer_axis_count(),
            "inner_width":plan.axes().len()-plan.outer_axis_count(),"max_row_values":zlc_core::scan::MAX_ROW_VALUES,
            "ports":plan.axes().iter().map(|axis|axis.port.as_str()).collect::<Vec<_>>()});
        let proposed: Resource = Arc::new(Scan {
            facts: Mutex::new(Facts { handle, plan: Arc::new(plan), configured: false, failure: None, pulse_applied: None }),
            work: Mutex::new(Work { configuration: None, publication: None, segment: None, written: 0 }) });
        let installed = context.install_resource(RESOURCE, proposed.clone())?;
        if !Arc::ptr_eq(&installed,&proposed) { return Err("this run already prepared its scan or outputs".into()); }
        return Ok(answer);
    }
    let scan = scan(context)?;
    let mut work = scan.work.try_lock().map_err(|_| "another operation is using this scan")?;
    let plan = scan.current(params["handle"].as_str().unwrap_or(""))?;
    match op {
        "run.scan.rows" => {
            let scope = super::wire::scope(&params["scope"])?;
            let rows = plan.rows(scope, integer(params, "start")?, integer(params, "count")?).map_err(invalid)?;
            Ok(json!({"rows": rows}))
        }
        "run.scan.configure" => {
            if work.configuration.is_some() || work.publication.is_some() { return Err(invalid("the scan configuration is already frozen")); }
            let output = params["output"].as_str().filter(|s| !s.is_empty()).ok_or_else(|| invalid("the scan needs an output name"))?.to_owned();
            let mut record = params["record"].as_object().cloned().ok_or_else(|| invalid("the scan needs a run record"))?;
            let effective: Option<Vec<Vec<f64>>> = match params.get("effective_inner").filter(|v| !v.is_null()) {
                Some(rows) => Some(serde_json::from_value(rows.clone()).map_err(|e| invalid(format!("effective scan rows: {e}")))?),
                None => None,
            };
            // The plan and the board's first receipt are frozen together: a receipt the bridge
            // gives meanwhile is either in this record or checked against it.
            let mut facts = scan.facts();
            let plan = match &effective {
                Some(rows) => facts.plan.with_effective_inner(rows).map_err(invalid)?,
                None => Plan::new(facts.plan.spec().clone()).map_err(invalid)?,
            };
            if let Some(applied) = &facts.pulse_applied {
                let snapshots = record.entry("device_snapshots").or_insert_with(|| json!({}));
                if !snapshots.is_object() { return Err(invalid("device_snapshots must be an object")); }
                snapshots["sequencer"] = applied.clone();
            }
            let axes: Vec<Value> = plan.axes().iter().map(|axis| {
                let mut value = json!({"port": axis.port, "values": axis.values, "unit": axis.unit});
                if let Some(parent) = &axis.coordinate_of { value["coordinate_of"] = json!(parent); }
                value
            }).collect();
            record.insert("plan".into(), json!({"axes": axes}));
            record.insert("scan_shape".into(), json!(plan.axes().iter().filter(|axis| axis.coordinate_of.is_none())
                .map(|axis| axis.values.len()).collect::<Vec<_>>()));
            let answer = json!({"axes": super::wire::axes(plan.axes())});
            work.configuration = Some(Configuration { output, record: Value::Object(record) });
            facts.plan = Arc::new(plan);
            facts.configured = true;
            Ok(answer)
        }
        "run.scan.arm" => {
            if work.segment.as_ref().is_some_and(|segment| segment.written != segment.total) {
                return Err(invalid("the previous scan segment has not completed"));
            }
            if work.written >= plan.total_shots() { return Err(invalid("the scan has already collected every planned shot")); }
            let inner = plan.inner_count();
            let shots = plan.spec().run_repeats;
            let index = work.written / (inner * shots);
            let expected = if plan.outer_axis_count() == 0 {
                (0, 0, plan.spec().scan_repeats)
            } else { (index % plan.outer_count() * inner, index / plan.outer_count(), 1) };
            let actual = (integer(params, "row_offset")?, integer(params, "scan_repeat_base")?, integer(params, "sweeps")?);
            if actual != expected { return Err(invalid(format!("scan segment {:?} is out of order; expected {:?}", actual, expected))); }
            let quiet = seconds(params, "quiet_seconds", 0.05, 0.25)?;
            let new_source = match params.get("new_source") {
                None => false, Some(value) => value.as_bool().ok_or_else(|| invalid("new_source must be a bool"))?,
            };
            if new_source && work.publication.is_some() { return Err(invalid("a scan cannot replace its source after collection began")); }
            let actual_devices = params.get("actual_devices").cloned().unwrap_or_else(|| json!({}));
            if !actual_devices.is_object() { return Err(invalid("actual_devices must be an object")); }
            let dropped = context.input_drain(quiet)?;
            if new_source { context.input_await_new_source()?; }
            // `dropped`: source events that had arrived before this segment was armed and belong to no shot of it.
            zlc_core::journal::note("scan", "segment_armed", json!({"row_offset": actual.0, "scan_repeat_base": actual.1, "sweeps": actual.2,
                "shots": inner * shots * actual.2, "written_before": work.written, "dropped": dropped, "new_source": new_source,
                "actual_devices": actual_devices}));
            work.segment = Some(Segment { row_offset: actual.0, scan_repeat_base: actual.1,
                total: inner * shots * actual.2, written: 0, previous_ordinal: None, actual_devices });
            Ok(json!({"dropped": dropped}))
        }
        "run.scan.collect" => {
            let answer = collect(params, &plan, &mut work, context);
            if let Err(error) = &answer { scan.facts().failure = Some(error.clone()); }
            answer
        }
        _ => Err(format!("unknown scan operation {op}")),
    }
}

fn collect(params: &Value, plan: &Plan, work: &mut Work, context: &dyn Context) -> Result<Value, String> {
    if work.configuration.is_none() || work.segment.is_none() { return Err(invalid("configure and arm the scan before collecting")); }
    let timeout = seconds(params, "timeout_seconds", 0.1, 1.0)?;
    let maximum = integer(params, "max_events")?;
    if maximum == 0 || maximum > 4096 { return Err(invalid("max_events must be between 1 and 4096")); }
    let deadline = Instant::now() + timeout;
    let mut collected = 0;
    while collected < maximum {
        context.check_active()?;
        let segment = work.segment.as_ref().unwrap();
        if segment.written == segment.total { break; }
        // Wait only for the first event; drain already available events as a
        // batch. A busy source cannot hold the caller past its control slice.
        let wait = if collected == 0 { deadline.saturating_duration_since(Instant::now()) } else { Duration::ZERO };
        let next = context.input_next(wait, true)?;
        if let Some(publication) = &work.publication {
            if context.input_generation()?.is_some_and(|generation| generation != publication.source_generation) {
                return Err(invalid("the source run changed during the scan"));
            }
        }
        let Some(read) = next else { break; };
        let ReadKind::Event(parent) = read.kind() else { return Err(invalid("scan collection requires source events, not canonical snapshots")); };
        let data = parent.data().ok_or_else(|| invalid("the source event has no data"))?.clone();
        let source_ordinal = read.key().ordinal.ok_or_else(|| invalid("the source event has no ordinal"))?;
        if segment.previous_ordinal.is_some_and(|previous| previous.checked_add(1) != Some(source_ordinal)) {
            return Err(invalid(format!("scan source ordinal jumped from {:?} to {source_ordinal}; shots cannot be reassigned by arrival", segment.previous_ordinal)));
        }
        if let Some(publication) = &work.publication {
            if publication.source_generation != read.key().generation { return Err(invalid("the source run changed during the scan")); }
            if *publication.source_schema != **data.schema() { return Err(invalid("the source dataset schema changed during the scan")); }
        } else {
            let schema = plan.dataset_schema(data.schema()).map_err(invalid)?;
            let configuration = work.configuration.as_ref().unwrap();
            let (generation, signal) = context.begin_output(&configuration.output, schema, configuration.record.clone())?;
            work.publication = Some(Publication { generation, signal,
                source_schema: data.schema().clone(), source_generation: read.key().generation });
        }
        let publication = work.publication.as_ref().unwrap();
        let segment = work.segment.as_mut().unwrap();
        let shots = plan.spec().run_repeats;
        let per_sweep = plan.inner_count() * shots;
        let row = segment.row_offset + (segment.written % per_sweep) / shots;
        let origin = plan.placement(row, segment.scan_repeat_base + segment.written / per_sweep,
            segment.written % shots, data.schema().domain(DomainRole::Point).size()).map_err(invalid)?;
        let name = read.key().signal.rsplit('/').next().unwrap_or("");
        let metadata = &parent.event().metadata;
        let mut record = if metadata.get("records").is_some() { metadata["records"][name].clone() } else { metadata.clone() };
        if segment.actual_devices.as_object().is_some_and(|values| !values.is_empty()) {
            if !record.is_object() { record = json!({}); }
            let mut devices = record["actual_devices"].as_object().cloned().unwrap_or_default();
            devices.extend(segment.actual_devices.as_object().unwrap().clone());
            record["actual_devices"] = Value::Object(devices);
        }
        let output = &work.configuration.as_ref().unwrap().output;
        let metadata = if record.is_null() { json!({}) } else { json!({"records": {output: record}}) };
        // Same immutable DataBlock and real parent: no array encoding, copy,
        // materialization, or re-read of a live Runtime history.
        context.commit(&publication.generation, &Bundle {
            inputs: vec![Input::Event { signal: publication.signal.clone(), ordinal: source_ordinal, data,
                origin: Some(origin), shot_time: None }], metadata, parents: vec![parent.clone()],
        })?;
        segment.previous_ordinal = Some(source_ordinal);
        segment.written += 1;
        work.written += 1;
        collected += 1;
        if segment.written == segment.total {
            zlc_core::journal::note("scan", "segment_filled", json!({"row_offset": segment.row_offset, "shots": segment.total,
                "last_source_ordinal": source_ordinal, "written": work.written}));
        }
        if Instant::now() >= deadline { break; }
    }
    let segment = work.segment.as_ref().unwrap();
    Ok(json!({"collected": collected, "segment_written": segment.written, "segment_total": segment.total,
        "written": work.written, "total": plan.total_shots(), "complete": segment.written == segment.total}))
}