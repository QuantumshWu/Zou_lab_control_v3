//! Run-scoped scan plans and native placement of unchanged source publications.
//! Device changes, operator interaction and Pulse policy stay with the caller;
//! source ownership, layout and commits use the existing Runtime exclusively.
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::runtime::{Bundle, Generation, GenerationId, Input, ReadKind};
use zlc_core::schema::{DatasetSchema, DomainRole};
use zlc_scan::{Plan, PlanSpec, Scope};

use crate::{CallError, ErrorKind, Inner, native_input::{CoreInput, check_stop}};

fn invalid(message: impl Into<String>) -> CallError { CallError::new(ErrorKind::Invalid, message) }
fn integer(params: &Value, name: &str) -> Result<usize, CallError> {
    params[name].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid(format!("{name} must be a nonnegative integer")))
}
fn seconds(params: &Value, name: &str, default: f64, maximum: f64) -> Result<Duration, CallError> {
    let value = match params.get(name) { None => default, Some(value) => value.as_f64().ok_or_else(|| invalid(format!("{name} must be seconds")))? };
    if !value.is_finite() || value < 0.0 || value > maximum { return Err(invalid(format!("{name} must be between 0 and {maximum} s"))); }
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

pub(crate) struct Session {
    handle: String,
    plan: Plan,
    input: Arc<Mutex<CoreInput>>,
    configuration: Option<Configuration>,
    publication: Option<Publication>,
    segment: Option<Segment>,
    written: usize,
    failure: Option<CallError>,
}

impl Inner {
    pub(crate) fn scan_request(&self, pid: u32, op: &str, params: &Value) -> Result<Value, CallError> {
        // Plan validation can be substantial for long authored vectors; it
        // must not hold the global device/run registry while doing that work.
        let prepared_plan = if op == "run.scan.prepare" {
            let spec: PlanSpec = serde_json::from_value(params["spec"].clone()).map_err(|e| invalid(format!("scan plan: {e}")))?;
            Some(Plan::new(spec).map_err(invalid)?)
        } else { None };
        // Both the run and its opaque plan handle are checked. Another process,
        // a restarted run, or a late response cannot use this session's plan.
        let (session, stop) = {
            let mut state = self.state.lock().unwrap();
            let instance = params["instance"].as_str().unwrap_or("");
            if state.instances.get(instance).is_none_or(|entry| entry.serving.pid() != pid) {
                return Err(CallError::new(ErrorKind::StaleRun, "the scan does not belong to this plugin process"));
            }
            let run = Self::run_of(&mut state, params)?;
            check_stop(&run.stop.0)?;
            let input = run.core_input.clone().ok_or_else(|| invalid("scan services require an exact input with consumer='core'"))?;
            if op == "run.scan.prepare" {
                if run.scan.is_some() || run.generation.is_some() { return Err(invalid("this run already prepared its scan or outputs")); }
                let plan = prepared_plan.expect("prepare parsed a plan");
                let handle = uuid::Uuid::new_v4().to_string();
                let answer = json!({"handle": handle, "outer_count": plan.outer_count(), "inner_count": plan.inner_count(),
                    "point_count": plan.point_count(), "total_shots": plan.total_shots(),
                    "outer_width": plan.outer_axis_count(), "inner_width": plan.axes().len() - plan.outer_axis_count(),
                    "max_row_values": zlc_scan::MAX_ROW_VALUES,
                    "ports": plan.axes().iter().map(|axis| axis.port.as_str()).collect::<Vec<_>>()});
                run.scan = Some(Arc::new(Mutex::new(Session { handle, plan, input, configuration: None,
                    publication: None, segment: None, written: 0, failure: None })));
                return Ok(answer);
            }
            (run.scan.clone().ok_or_else(|| invalid("prepare a scan before using its plan"))?, run.stop.0.clone())
        };
        // A malicious/reentrant caller cannot queue many long operations behind
        // one scan. Sequential SDK calls never contend for this lock.
        let mut session = session.try_lock().map_err(|_| CallError::new(ErrorKind::Refused, "another operation is using this scan"))?;
        if params["handle"].as_str() != Some(session.handle.as_str()) {
            return Err(CallError::new(ErrorKind::StaleRun, "the scan handle is not current for this run"));
        }
        if let Some(error) = &session.failure { return Err(error.clone()); }
        match op {
            "run.scan.rows" => {
                let scope: Scope = serde_json::from_value(params["scope"].clone()).map_err(|e| invalid(e.to_string()))?;
                let rows = session.plan.rows(scope, integer(params, "start")?, integer(params, "count")?).map_err(invalid)?;
                Ok(json!({"rows": rows}))
            }
            "run.scan.configure" => {
                if session.configuration.is_some() || session.publication.is_some() { return Err(invalid("the scan configuration is already frozen")); }
                let plan = if let Some(rows) = params.get("effective_inner").filter(|v| !v.is_null()) {
                    let rows: Vec<Vec<f64>> = serde_json::from_value(rows.clone()).map_err(|e| invalid(format!("effective scan rows: {e}")))?;
                    session.plan.with_effective_inner(&rows).map_err(invalid)?
                } else { Plan::new(session.plan.spec().clone()).map_err(invalid)? };
                let output = params["output"].as_str().filter(|s| !s.is_empty()).ok_or_else(|| invalid("the scan needs an output name"))?.to_owned();
                let mut record = params["record"].as_object().cloned().ok_or_else(|| invalid("the scan needs a run record"))?;
                let axes: Vec<Value> = plan.axes().iter().map(|axis| {
                    let mut value = json!({"port": axis.port, "values": axis.values, "unit": axis.unit});
                    if let Some(parent) = &axis.coordinate_of { value["coordinate_of"] = json!(parent); }
                    value
                }).collect();
                record.insert("plan".into(), json!({"axes": axes}));
                record.insert("scan_shape".into(), json!(plan.axes().iter().filter(|axis| axis.coordinate_of.is_none())
                    .map(|axis| axis.values.len()).collect::<Vec<_>>()));
                let answer = json!({"axes": plan.axes()});
                session.configuration = Some(Configuration { output, record: Value::Object(record) });
                session.plan = plan;
                Ok(answer)
            }
            "run.scan.arm" => {
                if session.segment.as_ref().is_some_and(|segment| segment.written != segment.total) {
                    return Err(invalid("the previous scan segment has not completed"));
                }
                if session.written >= session.plan.total_shots() { return Err(invalid("the scan has already collected every planned shot")); }
                let inner = session.plan.inner_count();
                let shots = session.plan.spec().run_repeats;
                let index = session.written / (inner * shots);
                let expected = if session.plan.outer_axis_count() == 0 {
                    (0, 0, session.plan.spec().scan_repeats)
                } else { (index % session.plan.outer_count() * inner, index / session.plan.outer_count(), 1) };
                let actual = (integer(params, "row_offset")?, integer(params, "scan_repeat_base")?, integer(params, "sweeps")?);
                if actual != expected { return Err(invalid(format!("scan segment {:?} is out of order; expected {:?}", actual, expected))); }
                let quiet = seconds(params, "quiet_seconds", 0.05, 0.25)?;
                let new_source = match params.get("new_source") {
                    None => false, Some(value) => value.as_bool().ok_or_else(|| invalid("new_source must be a bool"))?,
                };
                if new_source && session.publication.is_some() { return Err(invalid("a scan cannot replace its source after collection began")); }
                let actual_devices = params.get("actual_devices").cloned().unwrap_or_else(|| json!({}));
                if !actual_devices.is_object() { return Err(invalid("actual_devices must be an object")); }
                let mut input = session.input.lock().unwrap();
                let dropped = input.drain(quiet, &stop)?;
                if new_source { input.await_new_source(); }
                drop(input);
                session.segment = Some(Segment { row_offset: actual.0, scan_repeat_base: actual.1,
                    total: inner * shots * actual.2, written: 0, previous_ordinal: None, actual_devices });
                Ok(json!({"dropped": dropped}))
            }
            "run.scan.collect" => {
                let answer = self.collect_scan(params, &mut session, &stop);
                if let Err(error) = &answer { session.failure = Some(error.clone()); }
                answer
            }
            _ => Err(CallError::new(ErrorKind::UnknownOp, format!("unknown scan operation {op}"))),
        }
    }

    fn collect_scan(&self, params: &Value, session: &mut Session, stop: &std::sync::atomic::AtomicBool) -> Result<Value, CallError> {
        if session.configuration.is_none() || session.segment.is_none() { return Err(invalid("configure and arm the scan before collecting")); }
        let timeout = seconds(params, "timeout_seconds", 0.1, 1.0)?;
        let maximum = integer(params, "max_events")?;
        if maximum == 0 || maximum > 4096 { return Err(invalid("max_events must be between 1 and 4096")); }
        let deadline = Instant::now() + timeout;
        let mut collected = 0;
        while collected < maximum {
            check_stop(stop)?;
            let segment = session.segment.as_ref().unwrap();
            if segment.written == segment.total { break; }
            let mut input = session.input.lock().unwrap();
            // Wait only for the first event; drain already available events as a
            // batch. A busy source cannot hold the caller past its control slice.
            let wait = if collected == 0 { deadline.saturating_duration_since(Instant::now()) } else { Duration::ZERO };
            let next = input.next(wait, stop, true)?;
            if let Some(publication) = &session.publication {
                if input.generation().is_some_and(|generation| generation != publication.source_generation) {
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
            if let Some(publication) = &session.publication {
                if publication.source_generation != read.key().generation { return Err(invalid("the source run changed during the scan")); }
                if *publication.source_schema != **data.schema() { return Err(invalid("the source dataset schema changed during the scan")); }
            } else {
                let schema = session.plan.dataset_schema(data.schema()).map_err(invalid)?;
                let configuration = session.configuration.as_ref().unwrap();
                let mut begin = params.clone();
                begin["record"] = configuration.record.clone();
                {
                    let mut state = self.state.lock().unwrap();
                    let run = Self::run_of(&mut state, params)?;
                    check_stop(&run.stop.0)?;
                    run.consumer = input.ticket();
                }
                self.begin_outputs(&begin, vec![(configuration.output.clone(), schema, true)])?;
                let mut state = self.state.lock().unwrap();
                let run = Self::run_of(&mut state, params)?;
                session.publication = Some(Publication { generation: run.generation.clone().ok_or_else(|| invalid("the scan output did not begin"))?,
                    signal: run.outputs[&configuration.output].signal.clone(), source_schema: data.schema().clone(), source_generation: read.key().generation });
            }
            drop(input);
            let publication = session.publication.as_ref().unwrap();
            let segment = session.segment.as_mut().unwrap();
            let shots = session.plan.spec().run_repeats;
            let per_sweep = session.plan.inner_count() * shots;
            let row = segment.row_offset + (segment.written % per_sweep) / shots;
            let origin = session.plan.placement(row, segment.scan_repeat_base + segment.written / per_sweep,
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
            let output = &session.configuration.as_ref().unwrap().output;
            let metadata = if record.is_null() { json!({}) } else { json!({"records": {output: record}}) };
            // Same immutable DataBlock and real parent: no array encoding, copy,
            // materialization, or re-read of a live Runtime history.
            self.runtime.commit(&publication.generation, &Bundle {
                inputs: vec![Input::Event { signal: publication.signal.clone(), ordinal: source_ordinal, data,
                    origin: Some(origin), shot_time: None }], metadata, parents: vec![parent.clone()],
            }).map_err(|e| CallError::new(ErrorKind::Failed, e.to_string()))?;
            segment.previous_ordinal = Some(source_ordinal);
            segment.written += 1;
            session.written += 1;
            collected += 1;
            if Instant::now() >= deadline { break; }
        }
        let segment = session.segment.as_ref().unwrap();
        Ok(json!({"collected": collected, "segment_written": segment.written, "segment_total": segment.total,
            "written": session.written, "total": session.plan.total_shots(), "complete": segment.written == segment.total}))
    }
}
