//! Calibration coordinates of a discrete SITE axis. Acquisition records bind
//! axis coordinates to stable point IDs; the order in a projected array is local.
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde_json::{Value, json};
use zlc_core::runtime::{EventRecord, GenerationId, Read, ReadKind, RunRecord, Storage};
use zlc_core::schema::{Coordinate, CoordinateKey, DomainRole};

use super::nd::Scalar;
use super::{ProjectionError, Result};

const RECORDED: &str = "image_point_overlay_geometry";
/// Per-dataset section in a saved Figure. Imported runs carry the same binding.
pub const SAVED_GEOMETRIES: &str = "site_geometries";

/// `_point_ring_radius(points, fraction=..., fallback=...)`: a fraction of the
/// median nearest-neighbour distance.
pub(crate) fn point_ring_radius(points: &[[f64; 2]], fraction: f64, fallback: f64) -> f64 {
    if points.len() < 2 || !points.iter().all(|p| p[0].is_finite() && p[1].is_finite()) { return fallback; }
    let mut nearest: Vec<f64> = points.iter().enumerate().map(|(i, a)| {
        let mut best = f64::INFINITY;
        for (j, b) in points.iter().enumerate() {
            let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
            let squared = if i == j { f64::INFINITY } else { dx * dx + dy * dy };
            if squared < best || best.is_nan() { best = squared; }
        }
        best.sqrt()
    }).filter(|d| d.is_finite() && *d > 0.0).collect();
    if nearest.is_empty() { return fallback; }
    nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = nearest.len();
    let median = if n % 2 == 1 { nearest[n / 2] } else { (nearest[n / 2 - 1] + nearest[n / 2]) / 2.0 };
    fraction * median
}

#[derive(Debug, PartialEq)]
pub struct SitePoint {
    pub id: String,
    pub label: String,
    pub xy: [f64; 2],
    coordinate: Coordinate,
}

#[derive(Debug)]
pub struct SiteGeometry {
    pub frame: Option<String>,
    pub x_axis_id: String,
    pub y_axis_id: String,
    pub points: Vec<SitePoint>,
    axis: String,
    positions: HashMap<CoordinateKey, usize>,
}

fn invalid(msg: impl Into<String>) -> ProjectionError { ProjectionError::view(msg) }

impl SiteGeometry {
    pub fn index(&self, coordinate: &Scalar) -> Result<usize> {
        let value = match coordinate {
            Scalar::Int(n) => Coordinate::Integer(*n),
            Scalar::Float(x) => Coordinate::Real(*x),
            Scalar::Text(s) => Coordinate::Text(s.clone()),
            _ => return Err(invalid("Site coordinates must be numbers or text")),
        };
        self.positions.get(&value.key().map_err(invalid)?).copied().ok_or_else(||
            invalid(format!("Site coordinate {} has no position in the recorded geometry for {}", coordinate.python_str(), self.axis)))
    }

    fn parse(doc: &Value) -> Result<Self> {
        let text = |field: &str| doc[field].as_str().filter(|s| !s.trim().is_empty())
            .map(str::to_owned).ok_or_else(|| invalid(format!("Site geometry requires {field}")));
        let array = |field: &str| doc[field].as_array().ok_or_else(|| invalid(format!("Site geometry requires {field}")));
        let (ids, xy, coordinates, labels) = (array("point_ids")?, array("coordinates_xy")?, array("status_coordinates")?, array("labels")?);
        if ids.is_empty() || [xy.len(), coordinates.len(), labels.len()].iter().any(|n| *n != ids.len()) {
            return Err(invalid("Site IDs, positions, labels and coordinates must have the same nonzero length"));
        }
        let mut points = Vec::with_capacity(ids.len());
        let mut positions = HashMap::with_capacity(ids.len());
        let mut unique_ids = HashSet::new();
        for (i, id) in ids.iter().enumerate() {
            let id = id.as_str().filter(|v| !v.trim().is_empty() && unique_ids.insert(*v))
                .ok_or_else(|| invalid("Site IDs must be unique nonempty text"))?;
            let pair = xy[i].as_array().filter(|pair| pair.len() == 2).ok_or_else(|| invalid("Site position must be an x/y pair"))?;
            let number = |v: &Value| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| invalid("Site positions must be finite"));
            let v = &coordinates[i];
            let coordinate = if let Some(n) = v.as_i64() { Coordinate::Integer(n.into()) }
                else if let Some(n) = v.as_u64() { Coordinate::Integer(n.into()) }
                else if let Some(x) = v.as_f64() { Coordinate::Real(x) }
                else if let Some(s) = v.as_str() { Coordinate::Text(s.into()) }
                else { return Err(invalid("Invalid site coordinate")); };
            if positions.insert(coordinate.key().map_err(invalid)?, i).is_some() {
                return Err(invalid("Site geometry contains a duplicate axis coordinate"));
            }
            let label = labels[i].as_str().ok_or_else(|| invalid("Site geometry labels must be text"))?;
            points.push(SitePoint { id: id.to_owned(), label: label.to_owned(), xy: [number(&pair[0])?, number(&pair[1])?], coordinate });
        }
        let frame = match doc.get("coordinate_frame") {
            Some(Value::Null) => None, Some(Value::String(name)) if !name.trim().is_empty() => Some(name.clone()),
            _ => return Err(invalid("Invalid site coordinate frame")),
        };
        Ok(Self { points, positions, frame, axis: text("status_axis_id")?, x_axis_id: text("x_axis_id")?, y_axis_id: text("y_axis_id")? })
    }

    fn agrees(&self, other: &Self) -> bool {
        self.frame == other.frame && self.x_axis_id == other.x_axis_id && self.y_axis_id == other.y_axis_id
            && self.points.len() == other.points.len() && self.positions.iter().all(|(coordinate, i)| {
                other.positions.get(coordinate).is_some_and(|j| self.points[*i] == other.points[*j])
            })
    }

    /// Freeze only the geometry of the selected dataset; optional image overlay
    /// paths and occupancy statuses describe the image, not the site scalar.
    pub fn document(&self) -> Value {
        let coordinates: Vec<Value> = self.points.iter().map(|point| match &point.coordinate {
            Coordinate::Integer(n) => json!(n), Coordinate::Real(x) => json!(x),
            Coordinate::Text(s) => json!(s.as_ref()), Coordinate::Missing => unreachable!("validated coordinate"),
        }).collect();
        json!({"status_axis_id":self.axis, "coordinate_frame":self.frame, "x_axis_id":self.x_axis_id, "y_axis_id":self.y_axis_id,
            "point_ids":self.points.iter().map(|p| &p.id).collect::<Vec<_>>(),
            "labels":self.points.iter().map(|p| &p.label).collect::<Vec<_>>(),
            "coordinates_xy":self.points.iter().map(|p| p.xy).collect::<Vec<_>>(), "status_coordinates":coordinates})
    }
}

#[derive(Clone, Default)]
struct Candidates(HashMap<String, Result<Arc<SiteGeometry>>>);

impl Candidates {
    fn add(&mut self, axis: &str, value: Result<Arc<SiteGeometry>>) {
        use std::collections::hash_map::Entry;
        match self.0.entry(axis.to_owned()) {
            Entry::Vacant(at) => { at.insert(value); }
            Entry::Occupied(mut at) => {
                if let Ok(before) = at.get() {
                    match value {
                        Ok(next) if Arc::ptr_eq(before, &next) || before.agrees(&next) => {}
                        Ok(_) => { *at.get_mut() = Err(invalid(format!("Conflicting site geometries in the source records for {axis}"))); }
                        Err(error) => { *at.get_mut() = Err(error); }
                    }
                }
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        for (axis, result) in &other.0 { self.add(axis, result.clone()); }
    }

    fn document(&mut self, doc: &Value) {
        if let Some(axis) = doc["status_axis_id"].as_str() { self.add(axis, SiteGeometry::parse(doc).map(Arc::new)); }
    }

    fn record(&mut self, record: &Value, signal: &str) {
        if let Some(doc) = record.get(RECORDED) { self.document(doc); }
        // A bundled event may describe outputs on different calibrations.
        let name = signal.rsplit('/').next().unwrap_or(signal);
        if let Some(outputs) = record.get("records") {
            if let Some(doc) = outputs.get(signal).or_else(|| outputs.get(name)).and_then(|output| output.get(RECORDED)) {
                self.document(doc);
            }
        }
    }

    fn archive_acquisition(&mut self, table: &Value) -> Result<()> {
        let mut seen = HashSet::new();
        zlc_core::acquisition::visit_acquisition_records(table, |index, signal, run, event| {
            if seen.insert((index, signal.to_owned())) { self.record(run, signal); }
            self.record(event, signal);
        }).map_err(invalid)
    }

    fn archived_tree(&mut self, root: &Value) {
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            let signal = node["publication"]["signal"].as_str().unwrap_or("");
            self.record(&node["run"], signal);
            self.record(&node["event"], signal);
            pending.extend(node["parents"].as_array().into_iter().flatten());
        }
    }

    fn archived(&mut self, lineage: &Value) -> Result<()> {
        let raw = lineage["nodes"].as_array().ok_or_else(|| invalid("Saved lineage has no node table"))?;
        if lineage["root"].is_null() && raw.is_empty() { return Ok(()); }
        let root = lineage["root"].as_str().ok_or_else(|| invalid("Saved lineage has no publication root"))?;
        let mut nodes = HashMap::with_capacity(raw.len());
        for node in raw {
            let id = node["id"].as_str().ok_or_else(|| invalid("Invalid saved lineage node ID"))?;
            if nodes.insert(id, node).is_some() { return Err(invalid("Duplicate saved lineage node ID")); }
        }
        let mut pending = vec![root];
        let mut seen = HashSet::new();
        while let Some(id) = pending.pop() {
            if !seen.insert(id) { continue; }
            let node = nodes.get(id).ok_or_else(|| invalid("Saved site lineage refers to an unknown publication"))?;
            let event = &node["event_record"];
            let mut signals = node["signals"].as_array().into_iter().flatten().filter_map(Value::as_str).peekable();
            // Old custom figure records can carry run-level facts without a signal list.
            if signals.peek().is_none() {
                self.record(&node["record"], "");
                self.record(event, "");
            } else {
                for signal in signals { self.record(&node["record"], signal); self.record(event, signal); }
            }
            if let Some(table) = event.get("acquisition") { self.archive_acquisition(table)?; }
            if let Some(window) = event.get("window") {
                if let Some(table) = window.get("acquisition") { self.archive_acquisition(table)?; }
                else { for member in window["members"].as_array().into_iter().flatten() { self.archived_tree(member); } }
            }
            pending.extend(node["parents"].as_array().into_iter().flatten().filter_map(Value::as_str));
        }
        Ok(())
    }
}

type Prefix = (GenerationId, Arc<str>, Weak<EventRecord>, Candidates);

#[derive(Default)]
struct RecordCache {
    runs: HashMap<(GenerationId, Arc<str>), (Weak<RunRecord>, Candidates)>,
    prefix: Option<Prefix>,
}

impl RecordCache {
    fn live(&mut self, kind: &ReadKind) -> Result<Candidates> {
        self.runs.retain(|_, (run, _)| run.strong_count() > 0);
        let canonical = match kind {
            ReadKind::Canonical(p) if p.storage() == Storage::Finite => Some(p),
            _ => None,
        };
        let previous = canonical.and_then(|p| self.prefix.as_ref().filter(|(g, s, _, _)| *g == p.key().generation && *s == p.key().signal));
        let mut result = Candidates::default();
        let mut seen_runs = HashSet::new();
        let mut seen_events = HashSet::new();
        let mut failure = None;
        kind.visit_record_roots(|signal, run, event| {
            if let Some((_, _, head, facts)) = previous {
                if head.ptr_eq(&Arc::downgrade(event)) { result.merge(facts); return false; }
            }
            let mut pending = vec![(signal, run, event)];
            while let Some((signal, run, event)) = pending.pop() {
                let key = (run.generation, signal.clone());
                if seen_runs.insert(key.clone()) {
                    let candidates = match self.runs.entry(key) {
                        std::collections::hash_map::Entry::Occupied(at) => &at.into_mut().1,
                        std::collections::hash_map::Entry::Vacant(at) => {
                            let mut run_facts = Candidates::default();
                            run_facts.record(&run.metadata, signal);
                            if let Some(saved) = run.metadata.get(SAVED_GEOMETRIES) {
                                if let Some(fields) = saved.as_object() {
                                    for (axis, document) in fields {
                                        if document["status_axis_id"].as_str() == Some(axis) { run_facts.document(document); }
                                        else { run_facts.add(axis, Err(invalid("Saved site geometry names a different axis"))); }
                                    }
                                }
                            } else if let Some(imported) = &run.imported {
                                let saved = imported.saved_history();
                                let loaded = saved.get("lineage").ok_or_else(|| invalid("The saved source lineage could not be read"))
                                    .and_then(|lineage| run_facts.archived(lineage));
                                if let Err(error) = loaded { failure = Some(error); return false; }
                            }
                            &at.insert((Arc::downgrade(run), run_facts)).1
                        }
                    };
                    result.merge(candidates);
                }
                if seen_events.insert((Arc::as_ptr(event), signal.clone())) {
                    result.record(&event.metadata, signal);
                    pending.extend(event.parents.iter().map(|parent| (&parent.key.signal, &parent.run, &parent.event)));
                }
            }
            true
        });
        if let Some(error) = failure { return Err(error); }
        if let Some(p) = canonical { self.prefix = Some((p.key().generation, p.key().signal.clone(), Arc::downgrade(p.event()), result.clone())); }
        Ok(result)
    }
}

#[derive(Clone, Default)]
pub(super) struct GeometryCache(Arc<Mutex<RecordCache>>);

enum Origin { Live(ReadKind), Archived(Value), Recorded(Value) }
struct Evidence { origin: Origin, facts: OnceLock<Result<Candidates>> }

/// Only site-bearing inputs retain a lazy handle. Non-map projections never walk
/// record history. The projection owner shares decoded run facts across reads.
#[derive(Clone, Default)]
pub struct SiteGeometries { evidence: Option<Arc<Evidence>>, cache: Option<GeometryCache> }

impl SiteGeometries {
    fn new(origin: Origin) -> Self {
        Self { evidence: Some(Arc::new(Evidence { origin, facts: OnceLock::new() })), cache: None }
    }
    pub fn from_read(read: &Read) -> Self {
        if [DomainRole::Repeat, DomainRole::Point, DomainRole::Cell].iter()
            .any(|d| read.schema().domain(*d).axes().iter().any(|a| a.role() == "site")) {
            Self::new(Origin::Live(read.kind().clone()))
        } else { Self::default() }
    }
    pub fn from_lineage(lineage: Value) -> Self { Self::new(Origin::Archived(lineage)) }
    pub fn from_document(document: Value) -> Self { Self::new(Origin::Recorded(document)) }
    pub(super) fn cached(&self, cache: &GeometryCache) -> Self {
        Self { evidence: self.evidence.clone(), cache: self.evidence.as_ref().map(|_| cache.clone()) }
    }

    pub fn resolve(&self, axis: &str) -> Result<Arc<SiteGeometry>> {
        if let Some(evidence) = &self.evidence {
            let facts = evidence.facts.get_or_init(|| {
                let mut facts = Candidates::default();
                match &evidence.origin {
                    Origin::Recorded(document) => facts.document(document),
                    Origin::Archived(lineage) => facts.archived(lineage)?,
                    Origin::Live(kind) => return match &self.cache {
                        Some(cache) => cache.0.lock().unwrap_or_else(|p| p.into_inner()).live(kind),
                        None => RecordCache::default().live(kind),
                    },
                }
                Ok(facts)
            });
            if let Some(candidate) = facts.as_ref().map_err(Clone::clone)?.0.get(axis) { return candidate.clone(); }
        }
        Err(invalid(format!("No recorded site geometry for axis {axis}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zlc_core::{DType, OwnedBuffer};
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, Generation, Input, Publication, Runtime};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValueSchema, ValidityContract};

    const SITE: &str = "calibration.site";

    fn geometry(left: f64) -> Value {
        json!({"status_axis_id":SITE, "coordinate_frame":"sensor_pixel_xy", "x_axis_id":"camera.x", "y_axis_id":"camera.y",
            "point_ids":["left","right"], "labels":["S10","S20"], "coordinates_xy":[[left,11.0],[42.0,11.0]],
            "status_coordinates":[10,20]})
    }

    #[test]
    fn archived_geometry_joins_coordinates_and_only_uses_reachable_sources() {
        let lineage = json!({"root":"fit","nodes":[
            {"id":"fit","parents":["occupied"],"record":{},"event_record":{}},
            {"id":"occupied","signals":["@logic/occupancy/survival"],"record":{}, "event_record":{"records":{
                "survival":{RECORDED:geometry(3.0)}, "unrelated":{RECORDED:geometry(99.0)}}}},
            {"id":"other-calibration","record":{RECORDED:geometry(25.0)}}]});
        let got = SiteGeometries::from_lineage(lineage).resolve(SITE).unwrap();
        let i = got.index(&Scalar::Float(20.0)).unwrap();
        assert_eq!((got.points[i].id.as_str(), got.points[i].xy), ("right", [42.0, 11.0]));
        assert_eq!(got.points[got.index(&Scalar::Int(10)).unwrap()].xy, [3.0, 11.0]);
        assert!(got.index(&Scalar::Int(1)).is_err(), "array offsets are not site coordinates");
        let reopened = SiteGeometries::from_document(got.document()).resolve(SITE).unwrap();
        assert!(got.agrees(&reopened));
    }

    #[test]
    fn saved_acquisition_finds_geometry_on_the_actual_output_of_an_earlier_shot() {
        let mut table = zlc_core::acquisition::Builder::default();
        let run = table.run("camera", "saved-generation", &json!({}));
        let record = json!({"records":{"survival":{RECORDED:geometry(12.0)},"other":{RECORDED:geometry(99.0)}}});
        let first = table.event(run,"@logic/occupancy/survival",1,Some(0),&record,&[]);
        table.range(first,[0,0],[1,1]);
        let lineage = json!({"root":"finite","nodes":[{"id":"finite","event_record":{"acquisition":table.finish()}}]});
        let resolved = SiteGeometries::from_lineage(lineage).resolve(SITE).unwrap();
        assert_eq!(resolved.points[0].xy,[12.0,11.0]);
    }

    fn schema(records: usize) -> Arc<DatasetSchema> {
        let domain = |name: &str, role: &str, count, origin| Domain::dense(vec![count], vec![
            Axis::new(name.into(), name.into(), role.into(), Coordinates::Index { origin, len:count }, None, None, None, None).unwrap(),
        ]).unwrap();
        DatasetSchema::new(domain("repeat","repeat",records,0), Domain::new(vec![1], vec![], vec![]).unwrap(),
            domain(SITE, "site", 2, 10), ValueSchema::new(DType::F64,None,None,ValidityContract::Value).unwrap()).unwrap()
    }
    fn begin(rt: &Runtime, signal: &str, records: usize, storage: Storage, meta: Value) -> Generation {
        rt.begin(signal,vec![Declaration{ signal:signal.into(), schema:schema(records), storage }],meta).unwrap()
    }
    fn publish(rt: &Runtime, run: &Generation, signal: &str, ordinal: u64, parents: Vec<Arc<Publication>>, finite: bool) -> Arc<Publication> {
        let block = DataBlock::new(schema(1),OwnedBuffer::from_vec(vec![1.0_f64,2.0]).into(), Validity::Uniform(true),None).unwrap();
        rt.commit(run,&Bundle{ inputs:vec![Input::Event{ signal:signal.into(), ordinal, data:block,
            origin:finite.then_some([ordinal as usize,0]), shot_time:None }], metadata:json!({}), parents }).unwrap().publications[0].clone()
    }

    #[test]
    fn all_finite_input_records_participate_in_geometry_resolution() {
        let rt = Runtime::new();
        let first = begin(&rt,"first",1,Storage::Latest,json!({RECORDED:geometry(3.0)}));
        let a = publish(&rt,&first,"first",0,vec![],false);
        let run = begin(&rt,"parameter",2,Storage::Finite,json!({}));
        let p1 = publish(&rt,&run,"parameter",0,vec![a],true);
        let cache = GeometryCache::default();
        let first_binding = SiteGeometries::from_read(&Read::from_canonical(p1)).cached(&cache);
        assert_eq!(first_binding.resolve(SITE).unwrap().points[0].xy,[3.0,11.0]);
        let second = begin(&rt,"second",1,Storage::Latest,json!({RECORDED:geometry(18.0)}));
        let b = publish(&rt,&second,"second",0,vec![],false);
        let p2 = publish(&rt,&run,"parameter",1,vec![b],true);
        let latest = SiteGeometries::from_read(&Read::from_event(p2.clone()).unwrap()).cached(&cache);
        assert_eq!(latest.resolve(SITE).unwrap().points[0].xy,[18.0,11.0]);
        let complete = SiteGeometries::from_read(&Read::from_canonical(p2)).cached(&cache);
        assert!(complete.resolve(SITE).unwrap_err().message.contains("Conflicting"));
        assert_eq!(first_binding.resolve(SITE).unwrap().points[0].xy,[3.0,11.0]);
    }
}
