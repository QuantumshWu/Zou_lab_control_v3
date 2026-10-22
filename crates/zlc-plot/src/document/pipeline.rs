//! `sync_panels`: every bound panel resolves, prepares and presents on its
//! own worker of the pool; a follower of a panel's fit starts once its
//! producer has prepared (and published). The document takes back each
//! panel's entries in the order the sequential executor wrote them:
//! preparation in dependency scan order, presentation in binding order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use serde_json::{Map, Value as Json};

use crate::configuration::error::{PlotError, Result};
use crate::configuration::handlers::HANDLERS;
use crate::configuration::value::Value;
use crate::session::gesture::monotonic;
use crate::session::RgbaFront;

use super::edit::Edit;
use super::panel::{Panel, Shared};
use super::{truthy_json, Binding, DocumentRead, FigureDocument, FitPublisher};

/// The producers each binding follows: a producer panel's fit outputs,
/// `@logic/<producer>/<parameter>` (its area selector's `roi_` outputs are
/// published by their own processor, not while the producer prepares).
fn dependencies(bindings: &[Binding]) -> Vec<Vec<usize>> {
    bindings.iter().map(|binding| (0..bindings.len())
        .filter(|producer| binding.signal.strip_prefix(&format!("@logic/{}/", bindings[*producer].panel))
            .is_some_and(|output| !output.contains('/') && !output.starts_with("roi_"))).collect()).collect()
}

/// The order the sequential executor prepares in: each pass over the
/// bindings takes every panel whose producers are done; a panel in a
/// dependency cycle never prepares.
fn preparation_order(dependencies: &[Vec<usize>]) -> Vec<usize> {
    let mut done = vec![false; dependencies.len()];
    let mut order = Vec::with_capacity(dependencies.len());
    while order.len() < dependencies.len() {
        let before = order.len();
        for index in 0..dependencies.len() {
            if !done[index] && dependencies[index].iter().all(|producer| done[*producer]) {
                done[index] = true;
                order.push(index);
            }
        }
        if order.len() == before { break; }
    }
    order
}

/// One bound panel's work and what it answers; `held`: inside its update
/// interval, it keeps the read it shows.
struct Job<R> { panel: Panel<R>, read: Option<R>, held: bool, target: Option<Json>, front: Option<RgbaFront>, unchanged: bool }

/// Everything the workers of one `sync_panels` share.
struct Schedule<'a, R> {
    jobs: Vec<Mutex<Job<R>>>,
    bindings: &'a [Binding],
    followers: Vec<Vec<usize>>,
    waiting: Vec<AtomicUsize>,
    command: &'a Json,
    shared: Shared<'a>,
    publisher: Option<&'a dyn FitPublisher<R>>,
    live_data: bool,
}

/// Prepare panel `index`, release the followers it was the last producer
/// of, then present it.
fn run<'s, R: DocumentRead>(scope: &rayon::Scope<'s>, schedule: &'s Schedule<'s, R>, index: usize) {
    let mut job = schedule.jobs[index].lock().expect("panel job");
    let span = crate::render::profile::span("panel.prepare");
    prepare_binding(&mut job, &schedule.bindings[index], schedule.command, &schedule.shared, schedule.publisher, schedule.live_data);
    drop(span);
    for &follower in &schedule.followers[index] {
        if schedule.waiting[follower].fetch_sub(1, Ordering::AcqRel) == 1 { scope.spawn(move |scope| run(scope, schedule, follower)); }
    }
    let _span = crate::render::profile::span("panel.present");
    (job.front, job.unchanged) = job.panel.present(&schedule.shared);
}

/// `_prepare_binding`: resolve and prepare one bound panel; a failure is
/// the panel's error, never the command's.
fn prepare_binding<R: DocumentRead>(job: &mut Job<R>, binding: &Binding, command: &Json, shared: &Shared<'_>,
                                    publisher: Option<&dyn FitPublisher<R>>, live_data: bool) {
    let outcome = (|| -> Result<()> {
        if let Some(publisher) = publisher.filter(|_| !binding.signal.is_empty() && !job.held) {
            let records = HANDLERS.iter().find(|handler| Some(handler.kind.value()) == binding.kind()).map(|handler| handler.record_input)
                .ok_or_else(|| PlotError::runtime(""))?;
            if let Some(fresh) = publisher.read(&binding.signal, binding.window as usize, records).map_err(PlotError::runtime)? { job.read = Some(fresh); }
        }
        let Some(read) = job.read.clone() else {
            let own = command.get("panel").and_then(Json::as_str) == Some(binding.panel.as_str());
            let edit = if own { Some(Edit::of(&binding.panel, binding, command, None, Vec::new())?) } else { None };
            job.panel.defer(binding, job.target.as_ref(), edit)?;
            let status = job.panel.status_mut();
            if binding.waiting() {
                status.error = None;
                status.message = Some(binding.reason().unwrap_or("Waiting for data").to_owned());
            } else {
                status.error = Some(binding.reason().unwrap_or("No source read is available").to_owned());
                status.message = None;
            }
            return Ok(());
        };
        // A failure of this command's own edit after queued edits is already
        // the panel's edit error; the queued edits are presented.
        let resolution = job.panel.resolve(shared, binding, &read, job.target.as_ref(), command, None, monotonic())?;
        job.target = Some(Value::Map(resolution.configuration.target()?).to_json());
        job.panel.apply(shared, &read, resolution.configuration, resolution.changed, live_data, publisher, resolution.shown)?;
        job.panel.status_mut().error = None;
        Ok(())
    })();
    if let Err(error) = outcome {
        let status = job.panel.status_mut();
        status.error = Some(error.message);
        status.authoring_pending = Some(false);
    }
}

impl<R: DocumentRead> FigureDocument<R> {
    /// `_sync_panels`.
    pub(crate) fn sync_panels(&mut self, command: &Json, reads: &mut BTreeMap<String, R>, targets: &mut Map<String, Json>,
                              publisher: Option<&dyn FitPublisher<R>>) -> Result<BTreeMap<String, RgbaFront>> {
        let reset = match command.get("reset_document") {
            None => false,
            Some(Json::Bool(reset)) => *reset,
            Some(_) => return Err(PlotError::type_error("reset_document must be a boolean")),
        };
        if reset {
            self.close();
            let initial = command.get("initial_document");
            self.path = initial.and_then(|initial| initial.get("path")).and_then(Json::as_str).unwrap_or("").to_owned();
            self.info = initial.and_then(|initial| initial.get("info")).cloned().unwrap_or_else(|| serde_json::json!({}));
        }
        let bindings = Self::bindings(command)?;
        for binding in &bindings {
            let request = command.get("target_requests").and_then(|requests| requests.get(&binding.panel)).cloned();
            let status = self.status_mut(&binding.panel);
            status.binding = Some(binding.clone());
            status.target_request = request;
        }
        let live_data = command.get("id").is_none_or(|id| id.as_i64() == Some(0) && (id.is_i64() || id.is_u64()))
            && !command.get("initial_document").is_some_and(truthy_json);
        // v2 HarmonicClock: a panel inside its update interval (`held`) keeps the
        // read it shows, so nothing it shows or publishes advances, while its
        // binding is the one that read was shown for.
        let held: BTreeSet<&str> = command.get("held").and_then(Json::as_array).into_iter().flatten().filter_map(Json::as_str).collect();
        let shown = |binding: &Binding| held.contains(binding.panel.as_str()).then(|| self.panel_sources.get(&binding.panel)).flatten()
            .filter(|source| source.signal == binding.signal && source.window == binding.window && source.epoch == binding.epoch)
            .and_then(|_| self.inputs.get(&binding.panel).cloned());
        let dependencies = dependencies(&bindings);
        let order = preparation_order(&dependencies);
        let mut followers = vec![Vec::new(); bindings.len()];
        for (follower, producers) in dependencies.iter().enumerate() { for producer in producers { followers[*producer].push(follower); } }
        let schedule = Schedule {
            jobs: bindings.iter().map(|binding| {
                let kept = shown(binding);
                Mutex::new(Job { panel: self.panel(&binding.panel), held: kept.is_some(), read: kept.or_else(|| reads.get(&binding.panel).cloned()),
                    target: targets.get(&binding.panel).cloned(), front: None, unchanged: false })
            }).collect(),
            bindings: &bindings, followers, waiting: dependencies.iter().map(|producers| AtomicUsize::new(producers.len())).collect(),
            command, shared: Shared { services: &self.services, dpr: self.dpr }, publisher, live_data,
        };
        rayon::scope(|scope| {
            let schedule = &schedule;
            for index in (0..bindings.len()).filter(|index| dependencies[*index].is_empty()) {
                scope.spawn(move |scope| run(scope, schedule, index));
            }
        });
        let mut jobs: Vec<Job<R>> = schedule.jobs.into_iter().map(|job| job.into_inner().expect("panel job")).collect();
        for index in order {
            let job = &mut jobs[index];
            if let Some(read) = job.read.clone() { reads.insert(job.panel.key.clone(), read); }
            if let Some(target) = job.target.clone() { targets.insert(job.panel.key.clone(), target); }
            self.store_prepared(&mut job.panel);
        }
        let mut results = Vec::with_capacity(jobs.len());
        for job in jobs {
            results.push((job.panel.key.clone(), job.front, job.unchanged));
            self.store_presented(job.panel);
        }
        Ok(self.collect_fronts(results))
    }
}
