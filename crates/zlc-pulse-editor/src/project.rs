//! The pages' view of one pulse (`project_schedule` and its parts): plain
//! JSON records in v2's view-model shapes (`zlc_ui.pulse.models`), the one
//! place a period becomes a card, a port a row and a binding a field state.
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use serde_json::{Value, json};
use zlc_plot::style::Palette;
use zlc_pulse::binding::tick_in;
use zlc_pulse::components::extract_subpulse;
use zlc_pulse::model::{
    Binding, FieldKind, FieldRef, MAXIMUM_REPEAT_COUNT, MINIMUM_BRACKET_COUNT, Period, PortKind, PortSpec, Sequence, Source, Target,
};
use zlc_pulse::units::TIME_UNIT_CHOICES;

use crate::edits::HOLD_MODE;
use crate::target::{pins_of, programmable_ports};
use crate::text::{SEPARATOR, counted, number, readable};

pub type ConfigValues = [(String, (f64, String))];

pub const VALIDATOR_NONE: &str = "none";
pub const VALIDATOR_INT: &str = "int";
pub const VALIDATOR_FLOAT: &str = "float";

/// One editable value (`FieldVM`), its binding's state merged in.
#[derive(Clone, Debug)]
pub struct Field {
    pub text: String,
    pub editable: bool,
    pub scan: bool,
    pub source: String,
    pub can_scan: bool,
    pub can_api: bool,
    pub effective_text: String,
    pub source_text: String,
    pub config_key: String,
    pub validator_kind: &'static str,
    /// Python numbers: an int stays an int (a bracket count's limits).
    pub validator_lo: Value,
    pub validator_hi: Value,
    pub resolution: Value,
    pub allow_any: bool,
    pub field_id: String,
    pub config_names: Vec<String>,
    pub config_active: bool,
    pub config_enabled: bool,
    pub default_text: String,
    pub binding_editable: bool,
}

impl Field {
    pub fn new(text: impl Into<String>) -> Field {
        Field { text: text.into(), editable: true, scan: false, source: "default".into(), can_scan: true, can_api: true,
                effective_text: String::new(), source_text: String::new(), config_key: String::new(), validator_kind: VALIDATOR_NONE,
                validator_lo: json!(0.0), validator_hi: json!(0.0), resolution: json!(0.0), allow_any: true, field_id: String::new(),
                config_names: Vec::new(), config_active: false, config_enabled: false, default_text: String::new(), binding_editable: true }
    }

    pub fn to_json(&self) -> Value {
        json!({"text": self.text, "editable": self.editable, "scan": self.scan, "source": self.source, "can_scan": self.can_scan,
               "can_api": self.can_api, "effective_text": self.effective_text, "source_text": self.source_text,
               "config_key": self.config_key, "validator_kind": self.validator_kind, "validator_lo": self.validator_lo,
               "validator_hi": self.validator_hi, "resolution": self.resolution, "allow_any": self.allow_any,
               "field_id": self.field_id, "config_names": self.config_names, "config_active": self.config_active,
               "config_enabled": self.config_enabled, "default_text": self.default_text, "binding_editable": self.binding_editable})
    }

    /// `_binding_field_state`: the source of the next execution, distinct from its default.
    fn bound(mut self, binding: Option<&Binding>, config: &ConfigValues, scanned: &BTreeSet<String>, config_enabled: bool,
             default_text: String, actual_text: String) -> Field {
        self.config_enabled = config_enabled;
        self.default_text = default_text;
        let Some(binding) = binding else { return self };
        let mut effective = String::new();
        let mut active = false;
        let status = if scanned.contains(&binding.field_id()) {
            effective = "Scan table".into();
            "The Scan table supplies this field; the default remains editable."
        } else if binding.source == Source::Config {
            let value = (!binding.config_key.is_empty() && config_enabled)
                .then(|| config.iter().find(|(key, _)| *key == binding.config_key)).flatten();
            if value.is_none() {
                if !config_enabled { "Config is off; using the Pulse default." }
                else if !binding.config_key.is_empty() { "That name is not supplied by Config; using the Pulse default." }
                else { "Config name is unassigned; using the Pulse default." }
            } else {
                active = true;
                effective = actual_text;
                "The saved Config supplies this value. Edit it on the Value Config page."
            }
        } else if binding.source == Source::Api {
            "An API caller may override it; otherwise the Pulse default is used."
        } else if !binding.config_key.is_empty() {
            "Using the Pulse default; the Config name is kept for switching back."
        } else { "Using the Pulse default." };
        self.scan = binding.scan;
        self.source = binding.source.name().into();
        self.config_active = active;
        self.effective_text = effective;
        self.source_text = status.into();
        self.config_key = binding.config_key.clone();
        self
    }
}

fn config_names(config: &ConfigValues) -> Vec<String> { config.iter().map(|(key, _)| key.clone()).collect() }

fn binding_for<'a>(sequence: &'a Sequence, reference: &FieldRef) -> Option<&'a Binding> { sequence.binding(reference) }

/// `project_ports`: the programmable ports as rows; a one-lane port shows its
/// package pin, a DAC its bus and width with every lane in the tooltip.
pub fn project_ports(target: &Target, pins: Option<&[(String, String)]>, visible: Option<&BTreeSet<String>>) -> Vec<Value> {
    let pin_by_lane = pins_of(target, pins);
    let clock_lane = |key: &str| target.port(key).filter(|p| p.kind == PortKind::Clock).map(|p| p.lanes[0].clone());
    let wired = |lane: &str| match pin_by_lane.get(lane).filter(|pin| !pin.is_empty()) { Some(pin) => format!("{lane} -> {pin}"), None => lane.to_owned() };
    programmable_ports(target).into_iter().map(|port| {
        let endpoint = if port.kind == PortKind::Dac {
            format!("bus{} {}b+clk", port.bus_index.unwrap_or(0), port.width)
        } else {
            pin_by_lane.get(&port.lanes[0]).filter(|pin| !pin.is_empty()).cloned().unwrap_or_else(|| port.lanes[0].clone())
        };
        let mut lines = vec![format!("{} {}", port.kind.name(), port.key)];
        lines.extend(port.lanes.iter().map(|lane| format!("    {}", wired(lane))));
        if port.kind == PortKind::Dac {
            if let Some(clock) = &port.latch_clock {
                let latch = clock_lane(clock).unwrap_or_else(|| clock.clone());
                lines.push(format!("    latch clock {clock}: {}", wired(&latch)));
                let (low, high) = port.signed_range().expect("a DAC");
                lines.push(format!("    code range {low}..{high}"));
            }
        }
        let (lo, hi) = port.signed_range().unwrap_or((0, 0));
        json!({"key": port.key, "kind": port.kind.name(), "label": port.display(), "endpoint_text": endpoint,
               "endpoint_tooltip": lines.join("\n"), "width": if port.width > 0 { port.width } else { port.lanes.len() as u64 },
               "lo": lo, "hi": hi, "visible": visible.is_none_or(|shown| shown.contains(&port.key))})
    }).collect()
}

/// `_analog_mode`: edge, ramp, or hold (no step).
fn analog_mode(period: &Period, port: &PortSpec) -> &'static str { period.step(&port.key).map_or(HOLD_MODE, |s| s.mode.name()) }

/// What a projection reads beside the pulse it shows.
#[derive(Clone, Copy, Default)]
pub struct Context<'a> {
    pub visible_ports: Option<&'a BTreeSet<String>>,
    pub config: &'a ConfigValues,
    pub scanned: Option<&'a BTreeSet<String>>,
    pub authored: Option<&'a Sequence>,
}

fn no_scan() -> &'static BTreeSet<String> { static EMPTY: BTreeSet<String> = BTreeSet::new(); &EMPTY }

/// What every card of one pulse reads, looked up once per projection: the
/// shown ports, the authored period and owning Subpulse of each id, and each
/// DAC's held level before each period. Per card these were scans of the whole
/// pulse (a held level walked every earlier period), quadratic in its length:
/// a 500-period pulse took ~50 ms per state, now a few.
pub struct Periods<'a> {
    sequence: &'a Sequence,
    context: Context<'a>,
    digital: Vec<(&'a str, usize)>,
    analog: Vec<&'a PortSpec>,
    authored: HashMap<&'a str, &'a Period>,
    owners: HashMap<&'a str, &'a str>,
    index: HashMap<&'a str, usize>,
    /// Per shown DAC (in `analog`'s order): the level held before each period, and after the last.
    held: Vec<(Vec<i64>, i64)>,
}

impl<'a> Periods<'a> {
    pub fn new(sequence: &'a Sequence, context: Context<'a>) -> Periods<'a> {
        let target = &sequence.target;
        let shown = |key: &str| context.visible_ports.is_none_or(|ports| ports.contains(key));
        let offered = programmable_ports(target);
        let digital = offered.iter().filter(|p| p.kind == PortKind::Digital && shown(&p.key))
            .map(|port| (port.key.as_str(), target.lane_index(&port.lanes[0]).expect("a lane"))).collect();
        let analog: Vec<&PortSpec> = offered.into_iter().filter(|p| p.kind == PortKind::Dac && shown(&p.key)).collect();
        // The first of each id, as the linear lookups found it.
        let mut authored = HashMap::new();
        for period in &context.authored.unwrap_or(sequence).periods { authored.entry(period.period_id.as_str()).or_insert(period); }
        let mut owners = HashMap::new();
        for component in &sequence.components {
            for id in &component.period_ids { owners.entry(id.as_str()).or_insert(component.component_id.as_str()); }
        }
        let mut index = HashMap::new();
        for (at, period) in sequence.periods.iter().enumerate() { index.entry(period.period_id.as_str()).or_insert(at); }
        let held = analog.iter().map(|port| {
            let mut level = 0;
            let before = sequence.periods.iter().map(|period| {
                let was = level;
                if let Some(step) = period.step(&port.key) { level = step.value; }
                was
            }).collect();
            (before, level)
        }).collect();
        Periods { sequence, context, digital, analog, authored, owners, index, held }
    }

    /// `_held_value`: the last level set before this period, else rest (0).
    fn held(&self, dac: usize, period_id: &str) -> i64 {
        let (before, after) = &self.held[dac];
        self.index.get(period_id).map_or(*after, |at| before[*at])
    }

    /// `_analog_field`: one DAC's box on one card; a holding output shows the
    /// level it holds and cannot be typed into.
    fn analog_field(&self, period: &Period, dac: usize, readonly: bool) -> Field {
        let (sequence, context, port) = (self.sequence, self.context, self.analog[dac]);
        let step = period.step(&port.key);
        let (low, high) = port.signed_range().unwrap_or((0, 0));
        let value = step.map_or_else(|| self.held(dac, &period.period_id), |s| s.value);
        let reference = FieldRef::dac(&period.period_id, &port.key);
        let original = self.authored.get(period.period_id.as_str()).copied().unwrap_or(period);
        let default_value = original.step(&port.key).map_or(value, |s| s.value);
        let mut field = Field::new(value.to_string());
        field.field_id = reference.key();
        field.config_names = config_names(context.config);
        field = field.bound(binding_for(sequence, &reference), context.config, context.scanned.unwrap_or(no_scan()), sequence.config_enabled,
                            default_value.to_string(), value.to_string());
        field.editable = step.is_some();
        field.validator_kind = VALIDATOR_INT;
        field.validator_lo = json!(low as f64);
        field.validator_hi = json!(high as f64);
        field.allow_any = true;
        field.binding_editable = !readonly;
        field
    }

    /// Whether the card's Subpulse is read from its file while Config is on (its bindings are the file's).
    fn readonly(&self, period: &Period) -> bool {
        let authored = self.context.authored.unwrap_or(self.sequence);
        self.sequence.config_enabled && self.owners.get(period.period_id.as_str()).is_some_and(|owner| {
            authored.components.iter().any(|c| c.component_id == *owner && c.from_file())
        })
    }

    /// What `project` reads of this period beside what every card reads alike.
    fn made_of(&self, period: &Period) -> MadeOf {
        let id = &period.period_id;
        let references = std::iter::once(FieldRef::duration(id)).chain(self.analog.iter().map(|port| FieldRef::dac(id, &port.key)));
        MadeOf {
            period: period.clone(), authored: self.authored.get(id.as_str()).copied().unwrap_or(period).clone(), readonly: self.readonly(period),
            tick: tick_in(self.sequence, &period.unit).unwrap_or(0.0),
            bindings: references.map(|reference| binding_for(self.sequence, &reference).cloned()).collect(),
            held: (0..self.analog.len()).map(|dac| self.held(dac, id)).collect(),
        }
    }

    /// Every card, each taken back from `cards` when it was made of the same.
    pub fn cards(&self, cards: &mut Cards) -> Vec<Value> {
        let alike = Alike {
            digital: self.digital.iter().map(|(key, lane)| ((*key).to_owned(), *lane)).collect(), analog: self.analog.iter().map(|port| (*port).clone()).collect(),
            config: self.context.config.to_vec(), scanned: self.context.scanned.cloned(), config_enabled: self.sequence.config_enabled,
        };
        if cards.alike.as_ref() != Some(&alike) { cards.kept.clear(); cards.alike = Some(alike); }
        cards.lent.clear();
        self.sequence.periods.iter().map(|period| {
            let made_of = self.made_of(period);
            let card = match cards.kept.remove(&period.period_id) {
                Some((kept, card)) if kept == made_of => card,
                _ => self.project(period),
            };
            cards.lent.push((period.period_id.clone(), made_of));
            card
        }).collect()
    }

    /// `project_period`: one period as its card.
    pub fn project(&self, period: &Period) -> Value {
        let (sequence, context) = (self.sequence, self.context);
        let spacer = period.is_spacer();
        let default_period = self.authored.get(period.period_id.as_str()).copied().unwrap_or(period);
        let readonly = self.readonly(period);
        let reference = FieldRef::duration(&period.period_id);
        let tick = tick_in(sequence, &period.unit).unwrap_or(0.0);
        let mut duration = Field::new(number(period.duration.to_f64()));
        duration.field_id = reference.key();
        duration.config_names = config_names(context.config);
        duration = duration.bound(binding_for(sequence, &reference), context.config, context.scanned.unwrap_or(no_scan()), sequence.config_enabled,
                                  format!("{} {}", number(default_period.duration.to_f64()), default_period.unit),
                                  format!("{} {}", number(period.duration.to_f64()), period.unit));
        duration.binding_editable = !readonly;
        duration.can_scan = !spacer;
        duration.can_api = !spacer;
        duration.validator_kind = VALIDATOR_FLOAT;
        duration.validator_lo = json!(tick);
        duration.validator_hi = json!(0.0);
        duration.resolution = json!(tick);
        duration.allow_any = false;
        let digital: Vec<Value> = self.digital.iter().map(|(key, lane)| json!([key, period.states[*lane] != 0])).collect();
        let analog: Vec<Value> = self.analog.iter().enumerate().map(|(dac, port)| {
            Value::Array(vec![json!(port.key), json!(analog_mode(period, port)), self.analog_field(period, dac, readonly).to_json()])
        }).collect();
        with_parts(json!({"period_id": period.period_id, "name": period.display_name(), "unit": period.unit, "unit_choices": TIME_UNIT_CHOICES,
                          "kind": period.kind.name(), "readonly": readonly}),
                   [("duration", duration.to_json()), ("digital", Value::Array(digital)), ("analog", Value::Array(analog))])
    }
}

/// What one card is made of beside what every card reads alike: its period,
/// the authored period it defaults to, whether it is read only, its tick, its
/// fields' bindings and each shown DAC's held level.
#[derive(PartialEq)]
struct MadeOf { period: Period, authored: Period, readonly: bool, tick: f64, bindings: Vec<Option<Binding>>, held: Vec<i64> }

/// What every card reads alike: the shown ports, the Config values, the scanned fields and Config mode.
#[derive(PartialEq)]
struct Alike { digital: Vec<(String, usize)>, analog: Vec<PortSpec>, config: Vec<(String, (f64, String))>, scanned: Option<BTreeSet<String>>, config_enabled: bool }

/// The cards of the schedule last shown, each with what it was made of. A
/// card made of the same is taken back instead of made again: an edit of one
/// period of a 500-period pulse makes one card, not 500. The cards go out in
/// the state and come back once it is sent (`recycle`); a state that does not
/// come back leaves none to take.
#[derive(Default)]
pub struct Cards { alike: Option<Alike>, kept: HashMap<String, (MadeOf, Value)>, lent: Vec<(String, MadeOf)> }

impl Cards {
    /// The cards of the state last made, sent.
    pub fn recycle(&mut self, cards: Vec<Value>) {
        self.kept.clear();
        let lent = std::mem::take(&mut self.lent);
        if lent.len() != cards.len() { return; }
        for ((id, made_of), card) in lent.into_iter().zip(cards) {
            if card["period_id"] != id.as_str() { self.kept.clear(); return; }
            self.kept.insert(id, (made_of, card));
        }
    }
}

/// `_delay_of`: one output's delay as the pair it is stored as.
fn delay_of(sequence: &Sequence, port: &str) -> (f64, String) {
    sequence.delays.iter().find(|d| d.port == port).map_or((0.0, "ns".into()), |d| (d.value.to_f64(), d.unit.clone()))
}

/// `_delay_row`: one output's delay row, whether or not a pulse is open.
pub fn delay_row(sequence: Option<&Sequence>, port: &str, context: Context) -> Value {
    let (value, unit) = sequence.map_or((0.0, "ns".into()), |s| delay_of(s, port));
    let (default, default_unit) = sequence.map_or((value, unit.clone()), |s| delay_of(context.authored.unwrap_or(s), port));
    let reference = FieldRef::delay(port);
    let mut field = Field::new(if sequence.is_some() { number(value) } else { "0".into() });
    field.field_id = reference.key();
    field.config_names = config_names(context.config);
    field.can_scan = false;
    field = field.bound(sequence.and_then(|s| binding_for(s, &reference)), context.config, no_scan(),
                        sequence.is_some_and(|s| s.config_enabled),
                        if sequence.is_some() { format!("{} {default_unit}", number(default)) } else { String::new() },
                        format!("{} {unit}", number(value)));
    field.editable = sequence.is_some();
    field.allow_any = false;
    json!({"port_key": port, "value": field.to_json(), "unit": unit, "units": TIME_UNIT_CHOICES})
}

/// `_bracket_vms`: each bracket numbered and inked outermost first.
pub fn bracket_records(sequence: &Sequence, context: Context) -> Vec<Value> {
    let palette = Palette::default();
    let authored = context.authored.unwrap_or(sequence);
    sequence.brackets.iter().enumerate().map(|(index, b)| {
        let reference = FieldRef::bracket_count(&b.bracket_id);
        let mut field = Field::new(b.count.to_string());
        field.can_scan = false;
        field.validator_kind = VALIDATOR_INT;
        field.field_id = reference.key();
        field.config_names = config_names(context.config);
        field.validator_lo = json!(MINIMUM_BRACKET_COUNT);
        field.validator_hi = json!(MAXIMUM_REPEAT_COUNT);
        let binding = sequence.bindings.iter().find(|binding| binding.field_ref.bracket_id.as_deref() == Some(&b.bracket_id));
        let default = authored.brackets.iter().find(|old| old.bracket_id == b.bracket_id).map_or(b.count, |old| old.count);
        field = field.bound(binding, context.config, no_scan(), sequence.config_enabled, default.to_string(), b.count.to_string());
        let readonly = sequence.config_enabled && authored.components.iter().any(|c| {
            c.from_file() && sequence.components.iter().any(|resolved| resolved.component_id == c.component_id
                && resolved.bracket_ids.contains(&b.bracket_id))
        });
        json!({"bracket_id": b.bracket_id, "start_period_id": b.start_period_id, "end_period_id": b.end_period_id, "count": b.count,
               "ordinal": index + 1, "color": palette.bracket_color(index), "count_field": field.to_json(), "readonly": readonly})
    }).collect()
}

/// `_period_counts`: periods, then spacers only when there are some.
pub fn period_counts(periods: &[Period]) -> String {
    let spacers = periods.iter().filter(|p| p.is_spacer()).count();
    let mut parts = vec![counted(periods.len() - spacers, "period")];
    if spacers > 0 { parts.push(counted(spacers, "spacer")); }
    parts.join(SEPARATOR)
}

fn visible_text(ports: &[Value]) -> String {
    format!("{}/{} ports", ports.iter().filter(|p| p["visible"] == true).count(), ports.len())
}

/// `ANALOG_MODE_ROWS`: the model's modes, titled, and the authoring-only Hold.
pub fn analog_mode_choices() -> Value {
    json!([{"label": "Edge", "value": "edge"}, {"label": "Ramp", "value": "ramp"}, {"label": "Hold", "value": HOLD_MODE}])
}

/// Everything `project_schedule` reads beside the pulse.
#[derive(Clone, Default)]
pub struct ScheduleRequest<'a> {
    pub path: &'a str,
    pub generation: u64,
    pub revision: u64,
    pub pins: Option<&'a [(String, String)]>,
    pub scan_points: usize,
    pub config_error: &'a str,
    pub context: Context<'a>,
    /// The cards last shown, to take back the unchanged ones from.
    pub cards: Option<&'a std::cell::RefCell<Cards>>,
}

/// `project_schedule(sequence, ...)`: the schedule page of one pulse.
pub fn project_schedule(sequence: &Sequence, request: &ScheduleRequest) -> Value {
    let context = request.context;
    let ports = project_ports(&sequence.target, request.pins, context.visible_ports);
    let cards = Periods::new(sequence, context);
    let periods: Vec<Value> = match request.cards {
        Some(kept) => cards.cards(&mut kept.borrow_mut()),
        None => sequence.periods.iter().map(|period| cards.project(period)).collect(),
    };
    let pass_ns = sequence.periods.iter().fold(0.0, |total, p| total + crate::text::nanoseconds(p.duration, &p.unit));
    let total_ns = sequence.played_nanoseconds();
    let authored = sequence.periods.iter().filter(|p| !p.is_spacer()).count();
    let spacers = sequence.periods.len() - authored;
    let tooltip = if total_ns != pass_ns {
        format!("{} ns as the board plays it; {} ns in one pass through ", number(total_ns), number(pass_ns))
    } else { format!("{} ns over ", number(total_ns)) } + &counted(authored, "period")
        + &if spacers > 0 { format!(" and {}", counted(spacers, "spacer")) } else { String::new() };
    let scanned = context.scanned.unwrap_or(no_scan());
    let slots = sequence.scan_bindings().into_iter().filter(|b| scanned.is_empty() || scanned.contains(&b.field_id())).count();
    let original = context.authored.unwrap_or(sequence);
    let components: Vec<Value> = sequence.components.iter().map(|component| {
        let authored_component = original.components.iter().find(|c| c.component_id == component.component_id).unwrap_or(component);
        let (total, summary, bracket_ids) = match extract_subpulse(sequence, &component.component_id) {
            Ok(fragment) => {
                let inner = fragment.to_sequence();
                let mut summary = vec![period_counts(&inner.periods)];
                if !inner.brackets.is_empty() { summary.push(counted(inner.brackets.len(), "bracket")); }
                (readable(inner.played_nanoseconds()), summary.join(SEPARATOR), inner.brackets.iter().map(|b| b.bracket_id.clone()).collect())
            }
            Err(_) => (String::new(), String::new(), Vec::<String>::new()),
        };
        json!({"component_id": component.component_id, "name": component.name, "period_ids": component.period_ids,
               "total_text": total, "summary_text": summary, "bracket_ids": bracket_ids, "source": authored_component.source.name(),
               "config_path": authored_component.config_path,
               "config_active": sequence.config_enabled && authored_component.from_file() && request.config_error.is_empty(),
               "error": if authored_component.from_file() { request.config_error } else { "" }})
    }).collect();
    let delay_rows = programmable_ports(&sequence.target).into_iter().filter(|p| matches!(p.kind, PortKind::Digital | PortKind::Dac))
        .map(|p| delay_row(Some(sequence), &p.key, context)).collect();
    let schedule = json!({
        "document_generation": request.generation, "revision": request.revision, "document_name": sequence.name,
        "clock_text": format!("{} ns/tick", number(sequence.time_step_ns)), "total_text": readable(total_ns),
        "total_tooltip": tooltip, "period_count": authored, "visible_text": visible_text(&ports),
        "summary_text": if request.path.is_empty() { sequence.name.clone() } else {
            Path::new(request.path).file_name().map_or_else(|| request.path.to_owned(), |name| name.to_string_lossy().into_owned())
        },
        "analog_mode_choices": analog_mode_choices(), "run_repeats": sequence.run_repeats,
        "scan_summary_text": if slots > 0 { format!("{}{SEPARATOR}{}", counted(slots, "slot"), counted(request.scan_points, "point")) }
                             else { "no scan slots".into() },
        "min_bracket_count": MINIMUM_BRACKET_COUNT, "default_bracket_count": MINIMUM_BRACKET_COUNT,
    });
    with_parts(schedule, [("ports", Value::Array(ports)), ("periods", Value::Array(periods)), ("brackets", Value::Array(bracket_records(sequence, context))),
                          ("delay_rows", Value::Array(delay_rows)), ("components", Value::Array(components))])
}

/// `object` with `parts` moved in. `json!` copies every value it is given (it
/// serializes a reference): a projection's large parts -- a pulse's cards --
/// go in this way, or one state copies its whole schedule two or three times.
pub fn with_parts<const N: usize>(mut object: Value, parts: [(&str, Value); N]) -> Value {
    let map = object.as_object_mut().expect("an object");
    for (key, value) in parts { map.insert(key.to_owned(), value); }
    object
}

/// `project_schedule(None, target=, time_step_ns=)`: the bare board a first
/// period would land on, its ports and delays real and nothing authored.
pub fn project_board(target: &Target, time_step_ns: f64, revision: u64, pins: Option<&[(String, String)]>) -> Value {
    let ports = project_ports(target, pins, None);
    json!({
        "document_generation": 0, "revision": revision, "document_name": "(no pulse)",
        "clock_text": format!("{} ns/tick", number(time_step_ns)), "total_text": "", "total_tooltip": "", "period_count": 0,
        "visible_text": visible_text(&ports), "summary_text": "Add a period to start a pulse on this board",
        "ports": ports, "periods": [], "analog_mode_choices": analog_mode_choices(), "brackets": [], "run_repeats": 0,
        "delay_rows": programmable_ports(target).into_iter().filter(|p| matches!(p.kind, PortKind::Digital | PortKind::Dac))
            .map(|p| delay_row(None, &p.key, Context::default())).collect::<Vec<_>>(),
        "scan_summary_text": "no scan slots", "min_bracket_count": MINIMUM_BRACKET_COUNT,
        "default_bracket_count": MINIMUM_BRACKET_COUNT, "components": [],
    })
}

/// `EMPTY_SCHEDULE`: an editor holding no pulse and attached to no board.
pub fn empty_schedule(revision: u64) -> Value {
    json!({
        "document_generation": 0, "revision": revision, "document_name": "(no pulse)", "clock_text": "", "total_text": "",
        "total_tooltip": "", "period_count": 0, "visible_text": "",
        "summary_text": "Open a pulse, or add a period to start one on this board", "ports": [], "periods": [],
        "analog_mode_choices": analog_mode_choices(), "brackets": [], "run_repeats": 0, "delay_rows": [],
        "scan_summary_text": "", "min_bracket_count": 1, "default_bracket_count": 2, "components": [],
    })
}

/// Which kinds of field a binding names, the way the field-kind argument says it.
pub fn field_kind_word(kind: FieldKind) -> &'static str {
    match kind { FieldKind::Duration => "duration", FieldKind::Dac => "analog", FieldKind::Delay => "delay", FieldKind::BracketCount => "bracket_count" }
}
