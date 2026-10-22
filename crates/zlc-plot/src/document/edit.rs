//! One command's own edit of a panel, and the edits a panel queues while it
//! has no read (`_edit`, `_start`, `_defer`, `_declared_start`).

use serde_json::Value as Json;

use crate::configuration::error::{PlotError, Result};
use crate::configuration::resolve::InteractionUpdates;
use crate::configuration::value::{Value, ValueMap};

use super::{json_key_error, truthy_json, Binding, PanelStatus};

/// The target fields a command states by value (main.rs also writes them
/// into the target it sends).
const COMMAND_TARGET_FIELDS: [&str; 6] = ["fit", "size", "selectors", "viewport", "facet_focus", "classifier_thresholds"];

/// What one command states for one panel, on the source it was made for.
#[derive(Clone)]
pub(crate) struct Edit {
    pub signal: String,
    pub window: i64,
    pub kind: Option<String>,
    pub target: ValueMap,
    pub semantic: Option<Json>,
    pub parameters: Option<Json>,
    pub scope: Option<(String, i64)>,
    /// A gesture already drawn: what it changed in the recipe.
    pub interaction: Option<InteractionUpdates>,
    /// A gesture's pointer commands with their times, replayed at its turn.
    pub pointer: Vec<(Json, f64)>,
}

/// The edits a panel without a read keeps, in order, and the start of a
/// panel that has no configuration yet.
#[derive(Clone)]
pub(crate) struct Queue { pub start: Option<Edit>, pub edits: Vec<Edit> }

const BUTTONS: [&str; 3] = ["left", "middle", "right"];

/// A pointer move with no button held.
pub(crate) fn hover(command: &Json) -> bool {
    command.get("event").and_then(Json::as_str) == Some("move")
        && !command.get("button").and_then(Json::as_str).is_some_and(|button| BUTTONS.contains(&button))
}

impl Edit {
    /// `_edit(key, binding, command, interaction_updates, pointer)`.
    pub fn of(key: &str, binding: &Binding, command: &Json, interaction: Option<InteractionUpdates>, pointer: Vec<(Json, f64)>) -> Result<Edit> {
        let own = command.get("panel").and_then(Json::as_str) == Some(key);
        let action = command.get("action").and_then(Json::as_str);
        let mut target = ValueMap::new();
        if own {
            for name in COMMAND_TARGET_FIELDS {
                if let Some(value) = command.get(name) { target.insert(name, Value::from_json(value)); }
            }
            match action {
                Some("focus") => target.insert("facet_focus", command.get("index").map_or(Value::None, Value::from_json)),
                Some("reset_view") => target.insert("viewport", Value::None),
                _ => {}
            }
        }
        let given = |name: &str| command.get(name).filter(|value| own && !value.is_null()).cloned();
        let scope = if own && action == Some("scope_step") {
            let name = command.get("name").and_then(Json::as_str).ok_or_else(|| json_key_error("name"))?;
            let steps = command.get("steps").and_then(Json::as_f64).ok_or_else(|| json_key_error("steps"))? as i64;
            Some((name.to_owned(), steps))
        } else { None };
        Ok(Edit {
            signal: binding.signal.clone(), window: binding.window, kind: binding.kind().map(str::to_owned), target,
            semantic: given("semantic"), parameters: given("parameters"), scope, interaction, pointer,
        })
    }

    /// `_start(key, binding, target)`: the complete target a panel with no
    /// configuration starts from, as one edit.
    pub fn start(key: &str, binding: &Binding, target: Option<&Json>) -> Result<Edit> {
        let mut start = Edit::of(key, binding, &Json::Object(Default::default()), None, Vec::new())?;
        start.target = target.map(Value::from_json).and_then(|target| target.as_map().cloned()).unwrap_or_default();
        Ok(start)
    }

    /// `_changes(edit)`.
    pub fn changes(&self) -> bool {
        !self.target.is_empty() || self.semantic.as_ref().is_some_and(truthy_json) || self.parameters.as_ref().is_some_and(truthy_json)
            || self.scope.is_some() || self.interaction.as_ref().is_some_and(|updates| !updates.recipe.is_empty()) || !self.pointer.is_empty()
    }

    pub fn source(&self) -> (String, i64) { (self.signal.clone(), self.window) }

    /// The semantic edit as resolver input; a value that is no mapping is refused.
    pub fn semantic_map(&self) -> Result<Option<ValueMap>> {
        match &self.semantic {
            None => Ok(None),
            Some(semantic @ Json::Object(_)) => Ok(Value::from_json(semantic).as_map().cloned()),
            Some(_) => Err(PlotError::type_error("semantic_updates must be a mapping or None")),
        }
    }

    pub fn names_kind(&self) -> bool { self.semantic.as_ref().and_then(|semantic| semantic.get("kind")).is_some() }
}

impl Queue {
    /// `_defer`: one more edit; pointer commands in a row are one gesture,
    /// and of hover moves in a row only the last one matters.
    pub fn push(&mut self, edit: Edit) {
        if let Some(last) = self.edits.last_mut().filter(|last| !edit.pointer.is_empty() && !last.pointer.is_empty()
                                                          && last.source() == edit.source()) {
            if last.pointer.last().is_some_and(|(command, _)| hover(command)) && hover(&edit.pointer[0].0) { last.pointer.pop(); }
            last.pointer.extend(edit.pointer);
        } else {
            self.edits.push(edit);
        }
    }
}

/// `_defer`: queue `edit` until the panel has a read to resolve it on; a
/// panel with no configuration yet also keeps its start.
pub(crate) fn defer(key: &str, status: &mut PanelStatus, configured: bool, binding: &Binding, target: Option<&Json>, edit: Option<Edit>) -> Result<()> {
    let mut queue = match status.queue.take() {
        Some(queue) => queue,
        None => Queue { start: if configured { None } else { Some(Edit::start(key, binding, target)?) }, edits: Vec::new() },
    };
    if let Some(edit) = edit { queue.push(edit); }
    if queue.start.is_some() || !queue.edits.is_empty() { status.queue = Some(queue); }
    status.authoring_pending = Some(true);
    Ok(())
}

/// `_declared_start(start, edits)`: before its first read a panel has no
/// configuration to edit, so an edit naming a kind declares what the panel
/// starts as: its semantic joins the start's target, the rest stays in place.
pub(crate) fn declared_start(mut start: Edit, edits: Vec<Edit>) -> (Edit, Vec<Edit>) {
    let kept = edits.into_iter().map(|mut edit| {
        if edit.names_kind() {
            let mut semantic = start.target.get("semantic").and_then(Value::as_map).cloned().unwrap_or_default();
            if let Some(Value::Map(stated)) = edit.semantic.as_ref().map(Value::from_json) { semantic.update(&stated); }
            start.target.insert("semantic", Value::Map(semantic));
            edit.semantic = None;
        }
        edit
    }).collect();
    (start, kept)
}
