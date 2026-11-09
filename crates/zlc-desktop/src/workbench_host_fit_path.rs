//! A real plugin process running the real Derive node, its `run.fit` and `fit_models`
//! answered by the application's own `Builtin::host_event`. Needs ZLC_PLUGIN_PYTHON.
//! The source is the test plugin `scanner` (zlc-plugin-host's test plugins).
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use zlc_bus::mcp::ToolOutput;
use zlc_core::control::SafeActions;
use zlc_plugin_host::HostConfig;

use super::*;

const SIGNAL: &str = "@logic/scanner/survival";

struct Stage { host: PluginHost, runtime: Runtime, builtin: Builtin, events: mpsc::Receiver<HostEvent>, ended: RefCell<BTreeMap<String, (String, Option<Value>)>> }

fn stage() -> Option<Stage> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("../zlc-plugin-host/tests/plugins")],
        packages: vec!["zlc_atom".into(), "zlc_test_plugins".into()], memory_limit_mib: Some(8192) };
    let runtime = Runtime::new();
    let (sender, events) = mpsc::channel();
    let sender = Mutex::new(sender);
    let host = PluginHost::new(config, runtime.clone(), SafeActions::new(), move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
        crate::plugin_services::install(&host).unwrap();
    // What main.rs's `fit_models` arm answers; any other command is not this path's.
    let core: Arc<dyn Fn(crate::remote::RemoteCall) -> Result<(), String> + Send + Sync> = Arc::new(|call| {
        assert_eq!(call.name, "fit_models");
        call.reply.send(Ok(ToolOutput { json: crate::run_fit::fit_models(), png: None })).map_err(|error| error.to_string())
    });
    let builtin = Builtin::new(runtime.clone(), Some(host.clone()), Arc::new(|_| {}), core);
    Some(Stage { host, runtime, builtin, events, ended: RefCell::default() })
}

fn request(values: Value, source: Option<&str>, outputs: &[(String, String)]) -> RunRequest {
    RunRequest { prepared: None, device_labels: Default::default(), device_units: Default::default(), values, devices: Default::default(), shared: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: source.map(str::to_owned), run_directory: None, outputs: outputs.iter().cloned().collect(), record_devices: false }
}

impl Stage {
    /// A run, known to the application under a workbench id of its own (`node#n`), as a
    /// started workbench run is: never the host's id, so a value of one used for the other fails here.
    fn start(&self, node: &str, kind: &str, request: RunRequest) -> String {
        if self.host.instances().iter().all(|(name, _, _)| name != node) { self.host.open(node, kind, json!({}), json!({})).unwrap(); }
        let hosted = self.host.start_run(node, request).unwrap();
        let runs = self.builtin.0.borrow().runs.clone();
        let mut runs = runs.lock().unwrap();
        let ours = format!("{node}#{}", &hosted[..8]);
        runs.mapped.insert(hosted, (node.to_owned(), ours.clone()));
        ours
    }

    /// The host's id of a workbench run.
    fn hosted(&self, ours: &str) -> String {
        let runs = self.builtin.0.borrow().runs.clone();
        let runs = runs.lock().unwrap();
        runs.mapped.iter().find(|(_, (_, mapped))| mapped == ours).map(|(hosted, _)| hosted.clone()).expect("a started run")
    }

    fn scan(&self, values: Value) -> String {
        let run = self.start("scanner", "scanner", request(values, None, &[("survival".to_owned(), SIGNAL.to_owned())]));
        self.published(SIGNAL);
        run
    }

    /// Derive over the scan; `publishes` are its outputs, named `@logic/<node>/<output>`.
    fn derive(&self, node: &str, variables: &Value, publishes: &[&str]) -> String {
        let outputs: Vec<_> = publishes.iter().map(|name| ((*name).to_owned(), format!("@logic/{node}/{name}"))).collect();
        self.start(node, "derive", request(json!({"variables": variables}), Some(SIGNAL), &outputs))
    }

    fn published(&self, signal: &str) -> zlc_core::runtime::SignalDescription {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            // A build's draft read waits on the application: keep answering while waiting.
            while let Ok(event) = self.events.try_recv() { self.answer(event); }
            if let Some(item) = self.runtime.signals().into_iter().rev().find(|item| &*item.signal == signal && item.published) { return item; }
            assert!(Instant::now() < deadline, "{signal} was not published");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// One host event through the application; a run's end is kept.
    fn answer(&self, event: HostEvent) {
        for event in self.builtin.host_event(event) {
            match event {
                Event::RunFinished { run, outcome, error, .. } => { self.ended.borrow_mut().insert(run, (outcome, error)); }
                Event::Failed { instance, error } => panic!("{instance} failed: {error}"),
                _ => {}
            }
        }
    }

    /// Host events through the application until these runs have ended: (outcome, error) of each.
    fn finish(&self, runs: &[&str], limit: Duration) -> BTreeMap<String, (String, Option<Value>)> {
        let deadline = Instant::now() + limit;
        while !runs.iter().all(|run| self.ended.borrow().contains_key(*run)) {
            self.answer(self.events.recv_timeout(deadline.saturating_duration_since(Instant::now())).expect("the runs end in time"));
        }
        runs.iter().map(|run| ((*run).to_owned(), self.ended.borrow()[*run].clone())).collect()
    }
}

fn group_variables(every: usize, axis: &str) -> Value {
    json!([{"name": "a", "publish": false, "terms": [{"of": "survival"}], "statistic": "mean", "over": ["repeat", "Site"], "by": "repeat", "every": every},
           {"name": "rabi", "terms": [{"of": "a"}], "statistic": "fit", "over": [axis], "model": "damped_sine"}])
}

const PARAMETERS: [&str; 5] = ["rabi_amplitude", "rabi_offset", "rabi_baseband_frequency", "rabi_decay_time", "rabi_phase"];

/// One parameter by group, as Runtime holds it: (unit, value, sigma, valid).
fn parameter(runtime: &Runtime, signal: &str) -> (String, Vec<f64>, Vec<f64>, Vec<bool>) {
    let source = runtime.signals().into_iter().rev().find(|item| &*item.signal == signal && item.published).unwrap_or_else(|| panic!("{signal}"));
    let latest = runtime.read_latest(&source.generation, signal).unwrap();
    let dataset = latest.canonical().expect("a finite output");
    let unit = dataset.schema().value().unit().unwrap_or("").to_owned();
    let mut rows = Vec::new();
    for placement in dataset.segments() {
        let zlc_core::AnyBuffer::F64(values) = placement.data().values() else { panic!("dtype") };
        let sigma = placement.data().sigma().expect("a standard error");
        for (k, (value, error)) in values.as_slice().iter().zip(sigma.as_slice()).enumerate() {
            let valid = match placement.data().validity() { zlc_core::dataset::Validity::Uniform(valid) => *valid, zlc_core::dataset::Validity::Mask(mask) => mask.as_slice()[k] };
            rows.push((placement.origin()[0] + k, *value, *error, valid));
        }
    }
    rows.sort_by_key(|row| row.0);
    (unit, rows.iter().map(|row| row.1).collect(), rows.iter().map(|row| row.2).collect(), rows.iter().map(|row| row.3).collect())
}

#[test]
fn a_derive_fit_runs_through_the_application_arm() {
    let Some(stage) = stage() else { return };
    let (repeats, every, points, sites) = (400, 50, 60, 35);
    let source = stage.scan(json!({"repeats": repeats, "points": points, "sites": sites, "block": points}));
    // Live over the running source, then over what it sealed: one answer.
    let live = stage.derive("live", &group_variables(every, "duration"), &PARAMETERS);
    stage.finish(&[&source, &live], Duration::from_secs(120)).into_iter().for_each(|(run, (outcome, error))| assert_eq!(outcome, "completed", "{run}: {error:?}"));
    let sealed = stage.derive("sealed", &group_variables(every, "duration"), &PARAMETERS);
    let ended = stage.finish(&[&sealed], Duration::from_secs(120));
    assert_eq!(ended[&sealed].0, "completed", "{:?}", ended[&sealed].1);

    let (unit, values, sigmas, valid) = parameter(&stage.runtime, "@logic/live/rabi_baseband_frequency");
    let (_unit, others, _sigmas, _valid) = parameter(&stage.runtime, "@logic/sealed/rabi_baseband_frequency");
    assert_eq!(unit, "MHz", "the unit a panel's fit names over µs");
    assert_eq!(values.len(), repeats / every);
    for group in 0..values.len() {
        // The scanner's law at the group's middle repeat.
        let middle = (group * every) as f64 + (every as f64 - 1.0) / 2.0;
        let truth = 0.026 + 0.009 * middle / (repeats as f64 - 1.0);
        assert!(valid[group], "group {group}");
        assert!((values[group] - truth).abs() < 5.0 * sigmas[group], "group {group}: {} ± {}, law {truth}", values[group], sigmas[group]);
        assert!(sigmas[group] < 2e-3, "group {group}: sigma {}", sigmas[group]);
        assert!((values[group] - others[group]).abs() <= 1e-3 * sigmas[group], "group {group}: live {}, sealed {}", values[group], others[group]);
    }
    assert_eq!(parameter(&stage.runtime, "@logic/live/rabi_decay_time").0, "µs");
    assert_eq!(parameter(&stage.runtime, "@logic/live/rabi_phase").0, "rad");
    stage.host.shutdown();
}

fn column(runtime: &Runtime, node: &str, name: &str) -> Value {
    let (unit, values, sigmas, valid) = parameter(runtime, &format!("@logic/{node}/{name}"));
    let number = |values: Vec<f64>| values.into_iter().map(|value| if value.is_finite() { json!(value) } else { Value::Null }).collect::<Vec<_>>();
    json!({"unit": unit, "value": number(values), "sigma": number(sigmas), "valid": valid})
}

/// ZLC_DERIVE_FILE names a saved scan of `repeat` x `rabi.duration` x `Site`, ZLC_DERIVE_OUT the JSON
/// of what came out, live and sealed, for the comparison with an independent analysis.
#[test]
fn a_saved_scan_through_the_application_arm() {
    let (Some(path), Some(out)) = (std::env::var("ZLC_DERIVE_FILE").ok(), std::env::var_os("ZLC_DERIVE_OUT")) else {
        eprintln!("ZLC_DERIVE_FILE and ZLC_DERIVE_OUT are not set: skipping the saved scan");
        return;
    };
    let Some(stage) = stage() else { return };
    let variables = group_variables(50, "rabi.duration");
    let started = Instant::now();
    let source = stage.scan(json!({"path": path, "block": 50}));
    let live = stage.derive("live", &variables, &PARAMETERS);
    let ended = stage.finish(&[&source, &live], Duration::from_secs(240));
    assert!(ended.values().all(|(outcome, _)| outcome == "completed"), "{ended:?}");
    let live_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let sealed = stage.derive("sealed", &variables, &PARAMETERS);
    let ended = stage.finish(&[&sealed], Duration::from_secs(120));
    assert_eq!(ended[&sealed].0, "completed", "{:?}", ended[&sealed].1);
    let sealed_seconds = started.elapsed().as_secs_f64();
    let side = |node: &str| PARAMETERS.iter().map(|name| ((*name).to_owned(), column(&stage.runtime, node, name))).collect::<serde_json::Map<_, _>>();
    let written = json!({"live": side("live"), "sealed": side("sealed"), "live_seconds": live_seconds, "sealed_seconds": sealed_seconds});
    std::fs::write(out, written.to_string()).unwrap();
    println!("saved scan: live {live_seconds:.2} s, sealed {sealed_seconds:.2} s");
    stage.host.shutdown();
}

fn per_site(every: usize) -> Value {
    json!([{"name": "a", "publish": false, "terms": [{"of": "survival"}], "statistic": "mean", "over": ["repeat"], "by": "repeat", "every": every},
           {"name": "rabi", "terms": [{"of": "a"}], "statistic": "fit", "over": ["duration"], "model": "damped_sine"}])
}

/// One batch with a cell per repeat: no groups, so nothing splits it.
fn one_batch() -> Value {
    json!([{"name": "a", "publish": false, "terms": [{"of": "survival"}], "statistic": "mean", "over": ["Site"]},
           {"name": "rabi", "terms": [{"of": "a"}], "statistic": "fit", "over": ["duration"], "model": "damped_sine"}])
}

/// Ten times the 20 Hz of a real finite scan, on a 1000 x 60 x 35 schema, a publication per
/// point: the whole first group and part of the next. Falling behind by a Follow queue fails the run.
#[test]
fn a_source_publishing_while_fits_run_keeps_its_follow_queue() {
    let Some(stage) = stage() else { return };
    let (every, points, publications) = (50, 60, 3300);
    let started = Instant::now();
    let source = stage.scan(json!({"repeats": 1000, "points": points, "sites": 35, "block": 1, "pause": 0.005, "limit": publications}));
    let pooled = stage.derive("pooled", &group_variables(every, "duration"), &PARAMETERS);
    let sites = stage.derive("sites", &per_site(every), &PARAMETERS);
    let ended = stage.finish(&[&source, &pooled, &sites], Duration::from_secs(240));
    assert!(ended.values().all(|(outcome, _)| outcome == "completed"), "{ended:?}");
    println!("{publications} publications and their fits: {:.1} s", started.elapsed().as_secs_f64());
    let (_unit, values, _sigmas, valid) = parameter(&stage.runtime, "@logic/pooled/rabi_baseband_frequency");
    assert_eq!(values.len(), 2, "the group that closed, and the partial one the seal gave");
    assert!(valid[0]);
    stage.host.shutdown();
}

/// Stop while its first batch is out: seconds until the run ended.
fn stopped_during_a_fit(stage: &Stage, node: &str, variables: &Value) -> f64 {
    let run = stage.derive(node, variables, &PARAMETERS);
    let mut stopped: Option<Instant> = None;
    let mut asked: Option<Instant> = None;
    loop {
        if stopped.is_none() && asked.is_some_and(|when| when.elapsed() > Duration::from_millis(50)) {
            stage.host.stop_run(node, &stage.hosted(&run)).unwrap();
            stopped = Some(Instant::now());
        }
        let Ok(event) = stage.events.recv_timeout(Duration::from_millis(5)) else { continue };
        if matches!(&event, HostEvent::Request { op, .. } if op == "run.fit") { asked.get_or_insert_with(Instant::now); }
        for event in stage.builtin.host_event(event) {
            if let Event::RunFinished { run: ended, outcome, error, .. } = event {
                if ended == run {
                    assert_eq!(outcome, "stopped", "{error:?}");
                    return stopped.expect("the run ended before its Stop").elapsed().as_secs_f64();
                }
            }
        }
    }
}

#[test]
fn stop_during_a_large_batch_answers_within_a_second() {
    let Some(stage) = stage() else { return };
    let source = stage.scan(json!({"repeats": 20000, "points": 60, "sites": 35}));
    let ended = stage.finish(&[&source], Duration::from_secs(60));
    assert_eq!(ended[&source].0, "completed", "{:?}", ended[&source].1);
    let batch = stopped_during_a_fit(&stage, "batch", &one_batch());
    let groups = stopped_during_a_fit(&stage, "groups", &per_site(1));
    println!("stopped in {batch:.3} s inside one batch of 20000, {groups:.3} s between groups");
    assert!(batch < 1.0 && groups < 1.0, "{batch} {groups}");
    stage.host.shutdown();
}

/// `... a_sealed_scan_takes -- --ignored --nocapture`.
#[test]
#[ignore = "a timing, read by hand"]
fn a_sealed_scan_takes() {
    let Some(stage) = stage() else { return };
    let source = stage.scan(json!({"repeats": 1000, "points": 60, "sites": 35}));
    let ended = stage.finish(&[&source], Duration::from_secs(60));
    assert_eq!(ended[&source].0, "completed", "{:?}", ended[&source].1);
    for (node, variables, what) in [("pooled", group_variables(50, "duration"), "20 groups pooled"), ("sites", per_site(50), "20 groups x 35 sites"),
                                    ("each", per_site(1), "1000 groups x 35 sites"), ("batch", one_batch(), "one batch of 1000")] {
        let started = Instant::now();
        let run = stage.derive(node, &variables, &PARAMETERS);
        let ended = stage.finish(&[&run], Duration::from_secs(280));
        assert_eq!(ended[&run].0, "completed", "{:?}", ended[&run].1);
        println!("sealed 1000 x 60 x 35, {what}: {:.2} s", started.elapsed().as_secs_f64());
    }
    stage.host.shutdown();
}

/// Survival pooled over a group's repeats and its sites, point by point, as a panel's curve holds
/// it: (x, y, sem, sem_dof) of the valid points, by `sem_from_moments` and `CurveSeries::sem_dof`.
fn panel_series(runtime: &Runtime, group: std::ops::Range<usize>, sites: usize) -> [Vec<f64>; 4] {
    let source = runtime.signals().into_iter().rev().find(|item| &*item.signal == SIGNAL && item.published).unwrap();
    let latest = runtime.read_latest(&source.generation, SIGNAL).unwrap();
    let dataset = latest.canonical().unwrap();
    let placement = dataset.segments().next().unwrap();
    let zlc_core::AnyBuffer::Bool(values) = placement.data().values() else { panic!("dtype") };
    let zlc_core::dataset::Validity::Mask(mask) = placement.data().validity() else { panic!("a mask") };
    let schema = dataset.schema();
    let axis = schema.domain(zlc_core::schema::DomainRole::Point).axes()[0].clone();
    let zlc_core::schema::Coordinates::Numeric(zlc_core::AnyBuffer::F64(x)) = axis.coordinates() else { panic!("coordinates") };
    let points = x.as_slice().len();
    let mut series: [Vec<f64>; 4] = Default::default();
    for point in 0..points {
        let samples: Vec<f64> = group.clone().flat_map(|row| (0..sites).map(move |site| (row * points + point) * sites + site))
            .filter(|&at| mask.as_slice()[at]).map(|at| f64::from(u8::from(values.as_slice()[at]))).collect();
        if samples.is_empty() { continue; }
        let n = samples.len() as f64;
        let (mean, square) = (samples.iter().sum::<f64>() / n, samples.iter().map(|sample| sample * sample).sum::<f64>() / n);
        let sem = zlc_plot::projection::reduce::sem_from_moments(&[mean], &[square], &[samples.len() as i64], None)[0];
        series[0].push(x.as_slice()[point]);
        series[1].push(mean);
        series[2].push(sem);
        series[3].push(if samples.len() > 1 { n - 1.0 } else { f64::INFINITY });
    }
    series
}

#[test]
fn a_small_group_fits_as_a_panel_fit_does() {
    use zlc_plot::fit::param_units::{FitParameterUnits, FitQuantityUnits};
    use zlc_plot::fit::{CellData, FitCell, FitEngine, FitRequest};
    let Some(stage) = stage() else { return };
    let (repeats, every, points, sites) = (12, 2, 60, 8);
    let source = stage.scan(json!({"repeats": repeats, "points": points, "sites": sites, "unit": "mV"}));
    let scanned = stage.finish(&[&source], Duration::from_secs(60));
    assert_eq!(scanned[&source].0, "completed", "{:?}", scanned[&source].1);
    let run = stage.derive("small", &group_variables(every, "duration"), &PARAMETERS);
    let ended = stage.finish(&[&run], Duration::from_secs(60));
    assert_eq!(ended[&run].0, "completed", "{:?}", ended[&run].1);

    // The engine as a panel's fit calls it, each group warm from the one before as Derive's is.
    let engine = FitEngine::default();
    let model = engine.registry.get("damped_sine").unwrap();
    let columns: Vec<_> = PARAMETERS.iter().map(|name| parameter(&stage.runtime, &format!("@logic/small/{name}"))).collect();
    let (mut warm, mut widened): (Option<Vec<f64>>, f64) = (None, 0.0);
    for group in 0..repeats / every {
        let [x, y, sem, dof] = panel_series(&stage.runtime, group * every..(group + 1) * every, sites);
        let cell = |dof: Option<Arc<[f64]>>| FitCell { data: CellData::Points { coordinates: vec![x.clone().into()], observations: Some(y.clone().into()) },
            sigma: Some(sem.clone().into()), sigma_dof: dof, selected_indices: None, data_revision: 0, warm_start: warm.clone() };
        let panel = engine.fit(&model, &cell(Some(dof.clone().into())), &FitRequest::default()).unwrap();
        let gaussian = engine.fit(&model, &cell(None), &FitRequest::default()).unwrap();
        assert!(panel.success, "group {group}: {}", panel.message);
        for (k, column) in columns.iter().enumerate() {
            let (value, error) = (panel.parameter_values[k], panel.standard_errors[k]);
            assert!((column.1[group] - value).abs() <= 1e-3 * error, "group {group} {}: {} against the panel's {value} ± {error}", PARAMETERS[k], column.1[group]);
            // A decay far beyond the scan is a flat direction: its error, larger than itself, is rounding's to move.
            let determined = error < value.abs();
            assert!(!determined || (column.2[group] - error).abs() <= 1e-3 * error, "group {group} {}: error {} against the panel's {error}", PARAMETERS[k], column.2[group]);
            if determined { widened = widened.max((gaussian.standard_errors[k] / error - 1.0).abs()); }
        }
        warm = Some(panel.parameter_values.clone());
    }
    assert!(widened > 1e-2, "without its degrees of freedom the fit would read otherwise: {widened}");

    // The unit: what a panel's fit names over the same axis.
    let axis = zlc_core::units::DEFAULT_UNITS.resolve("mV").unwrap();
    let units = FitParameterUnits { registry: None, histogram: false, rolling_primary: false,
        quantities: Box::new(|_| Some(FitQuantityUnits { canonical_unit: axis.clone(), display_unit: axis.clone() })) };
    let frequency = model.parameters.iter().find(|spec| spec.name == "baseband_frequency").unwrap();
    let named = units.parameter_conversion(frequency, false, None).unwrap().symbol;
    assert_eq!(named, "mV^-1");
    assert_eq!(parameter(&stage.runtime, "@logic/small/rabi_baseband_frequency").0, named);
    stage.host.shutdown();
}
