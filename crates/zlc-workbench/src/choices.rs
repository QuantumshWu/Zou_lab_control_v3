//! v2's keyed choice trees (zlc_ui choice_tree_groups, ConsolePresenter.signal_groups):
//! every choice filed under the heading a person reads, as FluentTreeComboBox shows them.
use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::logic::split_signal;
use crate::{PanelContent, SignalInfo, Workbench};

/// One offered choice: its key, its row's words, the heading it is filed under
/// ("": none) and what the closed box says when it is picked ("": "<heading> · <label>").
pub struct Choice { pub key: String, pub label: String, pub heading: String, pub text: String }

/// v2 choice_tree_groups: each key once (the first offer files it), the headings in
/// case-blind order with `unfiled` last, each one's choices by key; a current key
/// nobody offers stands under `unfiled`. `[{heading, leaves: [{key, label, text}]}]`.
pub fn choice_tree(choices: Vec<Choice>, current: &str, unfiled: &str) -> Value {
    let mut offered: BTreeMap<String, Choice> = BTreeMap::new();
    for choice in choices { offered.entry(choice.key.clone()).or_insert(choice); }
    if !current.is_empty() && !offered.contains_key(current) {
        offered.insert(current.to_owned(), Choice { key: current.to_owned(), label: String::new(), heading: String::new(), text: String::new() });
    }
    let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
    for choice in offered.into_values() {
        let heading = if choice.heading.is_empty() { unfiled.to_owned() } else { choice.heading };
        let label = if choice.label.is_empty() { choice.key.clone() } else { choice.label };
        let text = if choice.text.is_empty() { format!("{heading} · {label}") } else { choice.text };
        let leaf = json!({"key": choice.key, "label": label, "text": text});
        match groups.iter_mut().find(|(name, _)| *name == heading) { Some((_, leaves)) => leaves.push(leaf), None => groups.push((heading, vec![leaf])) }
    }
    groups.sort_by_key(|(heading, _)| (heading == unfiled, heading.to_lowercase()));
    Value::Array(groups.into_iter().map(|(heading, leaves)| json!({"heading": heading, "leaves": leaves})).collect())
}

/// v2 signal_groups(panel) of a Figure Viewer (its panel_only ConsolePresenter): every signal the
/// viewer holds, under its producer -- a saved dataset (`@figure/<serial>/<name>`) under the figure
/// panel it was saved from, its generation's owner -- read by its output and shape, closed as v2
/// spells it (`@figure/1/data`, `figure-1-2/roi_sum`); never what `panel` publishes or anything cut
/// from that. `signals`: (key, owner, shape); `shown`: the viewer's (panel, signal) bindings.
pub fn viewer_signal_tree(panel: &str, current: &str, signals: &[(String, String, String)], shown: &[(String, String)]) -> Value {
    let own = |key: &str| {
        let mut name = key.to_owned();
        for _ in 0..32 {
            let Some((producer, _)) = split_signal(&name) else { return false };
            if producer == panel { return true; }
            match shown.iter().find(|(id, _)| id == producer) { Some((_, next)) => name = next.clone(), None => return false }
        }
        false
    };
    let mut choices: Vec<Choice> = signals.iter().filter(|(key, ..)| !own(key)).map(|(key, owner, shape)| {
        let (heading, output, text) = match split_signal(key) {
            Some((producer, output)) => (producer.to_owned(), output, format!("{producer}/{output}")),
            None => (owner.clone(), key.rsplit('/').next().unwrap_or(key), key.clone()),
        };
        Choice { key: key.clone(), label: format!("{output} · {shape}"), heading, text }
    }).collect();
    if !current.is_empty() && !choices.iter().any(|choice| choice.key == current) {
        choices.push(Choice { key: current.to_owned(), label: String::new(), heading: String::new(), text: current.to_owned() });
    }
    choice_tree(choices, current, "Unresolved")
}

/// v2 UNGROUPED_HEADING: what a signal with no producer is filed under.
const UNGROUPED: &str = "signals";

impl Workbench {
    /// v2 producer_text: a node by its name, what it owns by its name and the owned
    /// part, any other producer (a panel) by its id.
    pub(crate) fn producer_text(&self, owner: &str) -> String {
        if let Some(node) = self.node(owner) { return node.name.clone(); }
        match owner.split_once('/') {
            Some((head, tail)) if self.node(head).is_some() => format!("{}/{tail}", self.node(head).expect("a node").name),
            _ => owner.to_owned(),
        }
    }

    /// A signal as a picker's row reads it (v2 signal_label): its output and its shape.
    fn signal_row(&self, signal: &str, signals: &[SignalInfo]) -> Choice {
        let (heading, output) = split_signal(signal).map_or((UNGROUPED.to_owned(), signal.rsplit('/').next().unwrap_or(signal)), |(producer, output)| (self.producer_text(producer), output));
        let shape = signals.iter().find(|info| info.name == signal).map_or("—", |info| info.shape.as_str());
        Choice { key: signal.to_owned(), label: format!("{output} · {shape}"), heading, text: self.signal_label(signal) }
    }

    /// v2 signal_groups(panel): every signal `panel` may show, under its producer --
    /// never what the panel itself publishes or anything cut from it, however many
    /// hops away (the pick would withdraw the very source it picked).
    pub(crate) fn signal_tree(&self, panel: &str, current: &str, signals: &[SignalInfo], contents: &[PanelContent]) -> Value {
        // What a signal was cut from: a panel's outputs from its signal, a node's from its source.
        let source = |signal: &str| -> Option<String> {
            let head = split_signal(signal)?.0.split('/').next()?;
            match self.node(head) {
                Some(node) => (!node.draft.source_signal.is_empty()).then(|| node.draft.source_signal.clone()),
                None => contents.iter().find(|content| content.panel == head).map(|content| content.signal.clone()),
            }
        };
        let own = |signal: &str| {
            let mut name = signal.to_owned();
            for _ in 0..32 {
                if split_signal(&name).is_some_and(|(producer, _)| producer == panel) { return true; }
                match source(&name) { Some(next) => name = next, None => return false }
            }
            false
        };
        let mut choices: Vec<Choice> = signals.iter().filter(|info| !own(&info.name)).map(|info| self.signal_row(&info.name, signals)).collect();
        if !current.is_empty() && !choices.iter().any(|choice| choice.key == current) {
            choices.push(Choice { key: current.to_owned(), label: String::new(), heading: String::new(), text: self.signal_label(current) });
        }
        choice_tree(choices, current, "Unresolved")
    }

    /// v2 overlay_signal_groups: the judgements of this image's measurement, under their producers.
    pub(crate) fn overlay_tree(&self, offered: &[(String, String)], current: &str, signals: &[SignalInfo]) -> Value {
        let mut choices: Vec<Choice> = offered.iter().map(|(signal, _)| self.signal_row(signal, signals)).collect();
        if !current.is_empty() && !choices.iter().any(|choice| choice.key == current) {
            choices.push(Choice { key: current.to_owned(), label: String::new(), heading: String::new(), text: self.signal_label(current) });
        }
        choice_tree(choices, current, "Unresolved")
    }

    /// v2 _source_choices: a node's sources under their producers, each read by its output, closed by its signal text.
    pub(crate) fn source_tree(&self, offered: &[String], current: &str) -> Value {
        let mut choices: Vec<Choice> = offered.iter().map(|signal| {
            let (heading, label) = split_signal(signal).map_or((String::new(), signal.clone()), |(producer, output)| (self.producer_text(producer), output.to_owned()));
            Choice { key: signal.clone(), label, heading, text: self.signal_label(signal) }
        }).collect();
        if let Some((producer, output)) = split_signal(current).filter(|_| !offered.iter().any(|signal| signal == current)) {
            choices.push(Choice { key: current.to_owned(), label: output.to_owned(), heading: self.producer_text(producer), text: self.signal_label(current) });
        }
        choice_tree(choices, current, UNGROUPED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(key: &str, label: &str, heading: &str) -> Choice { Choice { key: key.into(), label: label.into(), heading: heading.into(), text: String::new() } }

    #[test]
    fn choices_are_filed_as_v2_files_them() {
        let tree = choice_tree(vec![choice("@logic/panel-1/roi_sum", "roi_sum", "panel-1"), choice("@logic/cam/frames", "frames", "Camera"),
            choice("@logic/panel-1/roi_frame", "roi_frame", "panel-1"), choice("@logic/cam/frames", "again", "else")], "@logic/gone/x", "Unresolved");
        assert_eq!(tree, json!([
            {"heading": "Camera", "leaves": [{"key": "@logic/cam/frames", "label": "frames", "text": "Camera · frames"}]},
            {"heading": "panel-1", "leaves": [{"key": "@logic/panel-1/roi_frame", "label": "roi_frame", "text": "panel-1 · roi_frame"},
                                              {"key": "@logic/panel-1/roi_sum", "label": "roi_sum", "text": "panel-1 · roi_sum"}]},
            {"heading": "Unresolved", "leaves": [{"key": "@logic/gone/x", "label": "@logic/gone/x", "text": "Unresolved · @logic/gone/x"}]},
        ]));
    }

    #[test]
    fn a_viewer_files_its_data_under_the_panel_it_was_saved_from() {
        let row = |key: &str, owner: &str| (key.to_owned(), owner.to_owned(), "(1) × (1) × (96 × 128)".to_owned());
        let signals = [row("@figure/1/data", "figure-1-1"), row("@logic/figure-1-1/roi_frame", "figure-1-1/roi"), row("@logic/figure-1-2/roi_sum", "figure-1-2/roi")];
        let shown = [("figure-1-1".to_owned(), "@figure/1/data".to_owned()), ("figure-1-2".to_owned(), "@logic/figure-1-1/roi_frame".to_owned())];
        let leaf = |key: &str, label: &str, text: &str| json!({"key": key, "label": label, "text": text});
        assert_eq!(viewer_signal_tree("figure-1-1", "@figure/1/data", &signals, &shown), json!([
            {"heading": "figure-1-1", "leaves": [leaf("@figure/1/data", "data · (1) × (1) × (96 × 128)", "@figure/1/data")]}]),
            "its own ROI, and a ROI cut from that, are not offered");
        assert_eq!(viewer_signal_tree("figure-1-2", "@logic/figure-1-1/roi_frame", &signals, &shown), json!([
            {"heading": "figure-1-1", "leaves": [leaf("@figure/1/data", "data · (1) × (1) × (96 × 128)", "@figure/1/data"),
                leaf("@logic/figure-1-1/roi_frame", "roi_frame · (1) × (1) × (96 × 128)", "figure-1-1/roi_frame")]}]));
    }
}
