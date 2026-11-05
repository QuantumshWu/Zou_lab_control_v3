//! An archive is an ordinary sealed Runtime source. Mounting is a candidate:
//! the caller keeps the old document until every replacement front accepts.
use std::{collections::HashSet, sync::{Arc, OnceLock, atomic::{AtomicBool, AtomicU64, Ordering}}};

use zlc_plot::session::ImagePointOverlay;
use serde_json::{Map, Value, json};
use zlc_core::{store::Snapshot,
    runtime::{Bundle, Declaration, Generation, ImportedRun, Input, Read, Runtime, Storage}};

use crate::panels::PanelInput;
use crate::record::Record;

static NEXT_ARCHIVE: AtomicU64 = AtomicU64::new(1);

/// Whether a signal belongs to the Figure Viewer's document: an archive's
/// mounted data or a viewer panel's output. The live board never offers them.
pub fn viewer_owned(signal: &str) -> bool {
    signal.starts_with(FIGURE_SIGNALS) || signal.starts_with(&format!("@logic/{}-", crate::panels::VIEWER_PREFIX))
}
/// v2 `_ArchiveDatasetProducer`: an archive's datasets publish as `@figure/<serial>/<name>`.
const FIGURE_SIGNALS: &str = "@figure/";

/// The dataset's key as a signal segment (v2: runs outside `[A-Za-z0-9_.-]`
/// become `_`, edge dots and underscores go), else its ordinal name.
fn dataset_segment(key: &str, index: usize) -> String {
    let (mut safe, mut replacing) = (String::with_capacity(key.len()), false);
    for character in key.chars() {
        let kept = character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-');
        if kept { safe.push(character) } else if !replacing { safe.push('_') }
        replacing = !kept;
    }
    let safe = safe.trim_matches(|character| character == '.' || character == '_');
    if safe.is_empty() { format!("dataset-{}", index + 1) } else { safe.to_owned() }
}

/// An opened Figure's saved metadata: one, whatever the number of its datasets. Drawing and
/// working with its panels never touch it. What the viewer's record tabs show of it is built
/// when they are first looked at, a descendant's saved history when one is saved.
pub struct SavedRecord { info: Value, lineage: zlc_archive::SavedLineage, read: OnceLock<Record>, asked: AtomicBool }

impl SavedRecord {
    fn read(&self) -> &Record {
        self.read.get_or_init(|| match self.lineage.read() {
            Ok(Some(lineage)) => {
                let mut info = self.info.clone();
                info["sections"]["lineage"] = lineage.to_serde();
                crate::record::describe(&info)
            }
            Ok(None) => crate::record::describe(&self.info),
            Err(error) => crate::record::unreadable(&self.info, &error.message),
        })
    }
}

impl ImportedRun for SavedRecord {
    fn saved_history(&self) -> Value {
        // A lineage that cannot be read fails the capture: this is no history (the save is refused).
        match self.lineage.read() {
            Ok(lineage) => json!({"source": self.info["sections"].get("source").cloned().unwrap_or_else(|| json!({})),
                                  "lineage": lineage.map_or_else(|| json!({"root":null,"nodes":[],"device_settings":[]}), |lineage| lineage.to_serde())}),
            Err(error) => json!(error.message),
        }
    }
}

pub struct ArchiveItem {
    pub key: String,
    /// One complete dataset state, including partial or empty acquired layout.
    pub plane: Snapshot,
    pub target: Value,
    /// Passive typed geometry/static annotations. Measured status is native.
    pub overlay: Option<Arc<ImagePointOverlay>>,
    pub overlay_plane: Option<Snapshot>,
}

pub struct DecodedArchive {
    pub path: String,
    pub info: Value,
    pub lineage: zlc_archive::SavedLineage,
    pub items: Vec<ArchiveItem>,
}

pub struct MountedItem {
    pub panel: String,
    pub key: String,
    pub signal: Arc<str>,
    pub generation: Generation,
    /// The dataset as saved, and the event that published it: what Edit data copies and descends from.
    pub plane: Snapshot,
    pub publication: Option<Arc<zlc_core::runtime::Publication>>,
    pub kind: String,
    pub window: usize,
    pub target: Value,
    pub overlay: Option<Arc<ImagePointOverlay>>,
    /// The actual status companion, in this image's generation and event.
    pub overlay_signal: Option<Arc<str>>,
}

pub struct MountedArchive {
    runtime: Runtime,
    path: String,
    items: Vec<MountedItem>,
    saved: Arc<SavedRecord>,
    /// How its datasets read in the Data chooser.
    label: String,
}

impl MountedArchive {
    pub fn mount(runtime: Runtime, decoded: DecodedArchive, panel_ids: Vec<String>) -> Result<Self, String> {
        if decoded.items.is_empty() { return Err("Figure archive contains no datasets".into()); }
        if decoded.items.len() != panel_ids.len() { return Err("Archive datasets and panel identities differ".into()); }
        let mut identities = HashSet::new();
        for panel in &panel_ids {
            if panel.is_empty() || panel.trim() != panel || !identities.insert(panel.as_str()) {
                return Err("Archive panel identities must be canonical and unique".into());
            }
        }
        if !decoded.info.is_object() { return Err("Figure archive metadata must be an object".into()); }
        let recipes = decoded.info["sections"]["plot"].clone();
        let prepared = decoded.items.into_iter().map(|mut item| {
            // v2 `_add_recipe_panel(initial_recipe=recipe)`: the panel starts from the
            // spec it was saved with, so its authored labels (the title) are drawn.
            if let (Some(target), Some(spec)) = (item.target.as_object_mut(), recipes[item.key.as_str()].get("spec")) {
                target.insert("spec".into(), spec.clone());
            }
            if item.overlay_plane.is_some() && item.overlay.is_none() {
                return Err("Measured archive status requires its typed overlay geometry".into());
            }
            let kind = item.target["semantic"]["kind"].as_str()
                .filter(|kind| !kind.is_empty() && kind.trim() == *kind)
                .ok_or("Archive target has no canonical plot kind")?.to_owned();
            let window = match item.target["parameters"].get("window") {
                None => 1,
                Some(value) => value.as_u64().filter(|width| *width > 0 && *width <= i64::MAX as u64)
                    .and_then(|width| usize::try_from(width).ok()).ok_or("Archive window must be a positive int64 integer")?,
            };
            Ok((item, kind, window))
        }).collect::<Result<Vec<_>, String>>()?;
        let serial = NEXT_ARCHIVE.try_update(Ordering::Relaxed, Ordering::Relaxed, |next| next.checked_add(1))
            .map_err(|_| "Archive import identity counter exhausted")?;
        let sections = &decoded.info["sections"];
        if !sections.get("source").is_none_or(Value::is_object) { return Err("Imported Figure source must be an object".into()); }
        let label = crate::record::label(&sections["source"]);
        let saved = Arc::new(SavedRecord { info: decoded.info, lineage: decoded.lineage, read: OnceLock::new(), asked: AtomicBool::new(false) });
        let mut mounted = Self { runtime, path: decoded.path, items: Vec::new(), saved, label };
        for (index, ((item, kind, window), panel)) in prepared.into_iter().zip(panel_ids).enumerate() {
            // Each dataset is its own sealed producer, as in v2. Independent
            // archived datasets must not become one artificial atomic event.
            let owner = format!("figure-{serial}-{}", index + 1);
            let mut signal: Arc<str> = format!("{FIGURE_SIGNALS}{serial}/{}", dataset_segment(&item.key, index)).into();
            // Two keys that spell one segment keep their ordinals apart.
            if mounted.items.iter().any(|mounted| mounted.signal == signal) { signal = format!("{FIGURE_SIGNALS}{serial}/dataset-{}", index + 1).into(); }
            let overlay_signal: Option<Arc<str>> = item.overlay_plane.as_ref()
                .map(|_| format!("{signal}/overlay").into());
            let metadata = json!({"operation":"figure-import", "archive":mounted.path, "dataset":item.key});
            let mut declarations = vec![Declaration { signal:signal.clone(),
                schema:item.plane.schema().clone(), storage:Storage::Snapshot }];
            let plane = item.plane.clone();
            let mut inputs = vec![Input::Dataset { signal:signal.clone(), ordinal:None, snapshot:item.plane }];
            if let Some(status) = item.overlay_plane {
                let status_signal = overlay_signal.as_ref().unwrap().clone();
                declarations.push(Declaration { signal:status_signal.clone(),
                    schema:status.schema().clone(), storage:Storage::Snapshot });
                inputs.push(Input::Dataset { signal:status_signal, ordinal:None, snapshot:status });
            }
            let generation = mounted.runtime.begin_imported(owner, declarations,
                metadata, mounted.saved.clone()).map_err(|error| error.to_string())?;
            // Register ownership before either fallible write. On any error
            // Drop retires only these candidate generations, never the old UI.
            mounted.items.push(MountedItem { panel, key:item.key, signal:signal.clone(), generation:generation.clone(), plane, publication:None,
                kind, window, target:item.target, overlay:item.overlay, overlay_signal });
            // A saved dataset is one state, not a replay of fabricated shots.
            // Image and measured status become visible in the same atomic event;
            // the complete native snapshots also admit empty prefixes unchanged.
            let delivery = mounted.runtime.commit(&generation, &Bundle {
                inputs, metadata:json!({"operation":"figure-import"}), parents:vec![],
            }).map_err(|error| error.to_string())?;
            let published = delivery.publications.iter().find(|publication| publication.key().signal == signal).cloned();
            if let Some(item) = mounted.items.last_mut() { item.publication = published; }
            mounted.runtime.seal(&generation, json!({"reason":"archive snapshot"}))
                .map_err(|error| error.to_string())?;
        }
        Ok(mounted)
    }

    pub fn inputs(&self) -> Vec<PanelInput> {
        let requests = self.items.iter().map(|item| (item.generation.clone(), item.signal.clone(), None)).collect::<Vec<_>>();
        self.items.iter().zip(self.runtime.read_views(&requests)).map(|(item, read)| {
            let (read, reason) = match read { Ok(read) => (Some(read), None), Err(error) => (None, Some(error.to_string())) };
            PanelInput { panel:item.panel.clone(), kind:item.kind.clone(), window:item.window, signal:item.signal.clone(), layer:None,
                epoch:1, read, overlay:None, reason, waiting:false }
        }).collect()
    }

    pub fn targets(&self) -> Map<String, Value> {
        self.items.iter().map(|item| (item.panel.clone(), item.target.clone())).collect()
    }

    pub fn overlay_read(&self, item: &MountedItem) -> Result<Option<Read>, String> {
        let Some(signal) = &item.overlay_signal else { return Ok(None); };
        let mut reads = self.runtime.read_views(&[
            (item.generation.clone(), item.signal.clone(), None),
            (item.generation.clone(), signal.clone(), None),
        ]).into_iter();
        let image = reads.next().unwrap().map_err(|error| error.to_string())?;
        let status = reads.next().unwrap().map_err(|error| error.to_string())?;
        if image.key().generation != status.key().generation
            || image.key().revision != status.key().revision
            || !Arc::ptr_eq(image.anchor().event(), status.anchor().event()) {
            return Err("Archive image and status do not share one atomic publication".into());
        }
        Ok(Some(status))
    }

    pub fn items(&self) -> &[MountedItem] { &self.items }
    /// What the viewer's Data chooser offers: this archive's datasets (not their
    /// status companions) and the viewer panels' outputs.
    pub fn offers(&self, signal: &str, viewer: u32) -> bool {
        self.items.iter().any(|item| &*item.signal == signal) || signal.starts_with(&format!("@logic/{}-", crate::panels::viewer_prefix(viewer)))
    }
    /// How a signal this archive mounted reads in the Data chooser.
    pub fn label(&self, signal: &str) -> Option<&str> {
        (!self.label.is_empty() && self.items.iter().any(|item| &*item.signal == signal)).then_some(self.label.as_str())
    }
    pub fn path(&self) -> &str { &self.path }
    /// The record, when something has already read it.
    pub fn record_if_read(&self) -> Option<&Record> { self.saved.read.get() }
    /// The record, read here if nothing has yet.
    pub fn record(&self) -> &Record { self.saved.read() }
    /// Read the record on a thread of its own, once; `done` is told when it can be shown.
    pub fn read_record(&self, done: impl FnOnce(Arc<SavedRecord>) + Send + 'static) {
        if self.saved.asked.swap(true, Ordering::Relaxed) { return; }
        let saved = self.saved.clone();
        std::thread::spawn(move || { saved.read(); done(saved); });
    }
    /// Whether this archive is the one that `saved` is the record of.
    pub fn holds(&self, saved: &Arc<SavedRecord>) -> bool { Arc::ptr_eq(&self.saved, saved) }
    /// The info pane's tabs, its Flow and how far the record is read ("unread", "reading", "read").
    pub fn record_view(&self) -> (Value, Value, &'static str) {
        match self.record_if_read() {
            Some(record) => (record.tabs.clone(), record.flow.clone(), "read"),
            None => (crate::record::unread_tabs(&self.saved.info), json!({"nodes": [], "edges": []}),
                     if self.saved.asked.load(Ordering::Relaxed) { "reading" } else { "unread" }),
        }
    }
}

impl Drop for MountedArchive {
    fn drop(&mut self) {
        for item in &self.items { let _ = self.runtime.retire(&item.generation); }
    }
}
