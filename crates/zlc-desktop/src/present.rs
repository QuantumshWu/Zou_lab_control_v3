//! When the windows are told what changed: once per presentation beat, not once per event.
//!
//! An event changes what the application holds -- the live document, the workbench, the pulse
//! board -- and says here whose view may have changed (`mark`), but tells no window itself. What
//! is pending is presented when the window's loop has handled every event it was sent (tao's
//! `MainEventsCleared`: it is about to wait) or at the board's beat, and never more often than
//! that beat: the live board's own presentation cadence (`zlc_workbench::INTERVALS[0]`, v2's
//! fastest panel refresh, the board's beat). A burst of plugin facts, publications and board jobs
//! is then one workbench view, one build and one message per page, and an event that changed
//! nothing anyone shows sends nothing.
//!
//! A reply is the exception: it is told at the end of the turn that made it (`reply`), with the
//! state as it then stands (reply.rs), never held for the beat. A board job's new pictures are
//! presented with the rows that frame them, the newest of each panel.
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{reply::Reply, science::Front};

/// Whose view of the application may have changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    /// The workbench's sections of the live document (devices, logic, runs, the board,
    /// the apparatus): made again (workbench_host::decorate), then sent.
    Workbench,
    /// The live board's signal list while it has no panel (panels::Panels::decorate), then sent.
    Signals,
    /// The live document as it is held: sent as it stands.
    Live,
    /// The pulse board and its editor (pulse_host): its view made again, then sent.
    Pulse,
}

/// What is due at a presentation.
#[derive(Default)]
pub struct Due {
    pub workbench: bool,
    pub signals: bool,
    pub live: bool,
    pub pulse: bool,
    /// A board job's answer was accepted: the windows let go of the panels it no longer has.
    pub board: bool,
    /// The replies to tell, each with its own message (the first carries what changed).
    pub replies: Vec<Reply>,
    /// Each live panel's newest picture not yet presented, with the sequence its row names.
    pub fronts: BTreeMap<String, (u64, Front)>,
}

impl Due {
    fn pending(&self) -> bool { self.workbench || self.signals || self.live || self.pulse || self.board || !self.replies.is_empty() || !self.fronts.is_empty() }
}

pub struct Presentation {
    due: Due,
    /// A reply, or a page just up, waits for nothing: presented at the end of this turn.
    now: bool,
    /// When the windows were last told.
    last: Option<Instant>,
    beat: Duration,
}

impl Default for Presentation {
    fn default() -> Self { Self { due: Due::default(), now: false, last: None, beat: Duration::from_millis(zlc_workbench::INTERVALS[0]) } }
}

impl Presentation {
    /// `owner`'s view may have changed.
    pub fn mark(&mut self, owner: Owner) {
        match owner {
            Owner::Workbench => { self.due.workbench = true; self.due.live = true; }
            Owner::Signals => { self.due.signals = true; self.due.live = true; }
            Owner::Live => self.due.live = true,
            Owner::Pulse => self.due.pulse = true,
        }
    }

    /// `owner`'s view was just told whole by whoever changed it: nothing of it is pending.
    pub fn told(&mut self, owner: Owner) {
        match owner {
            Owner::Pulse => self.due.pulse = false,
            Owner::Workbench | Owner::Signals | Owner::Live => {}
        }
    }

    /// The live document goes out at the end of this turn: a page is up and holds nothing yet.
    pub fn now(&mut self) { self.due.live = true; self.now = true; }

    /// A reply, told at the end of this turn with the live document as it then stands. A message
    /// that answers nothing (`Reply::default()`) is no reply: it waits for the beat with the rest.
    pub fn reply(&mut self, reply: Reply) {
        self.due.live = true;
        if reply != Reply::default() { self.due.replies.push(reply); self.now = true; }
    }

    /// A board job's answer is the live document now; its pictures (`fronts`, by panel) are presented
    /// after the rows that frame them, each panel's newest.
    pub fn board(&mut self, fronts: impl IntoIterator<Item = (String, (u64, Front))>) {
        self.due.board = true;
        self.front(fronts);
    }

    /// Pictures of live panels to present after the live document (a page just up is given its panels' last).
    pub fn front(&mut self, fronts: impl IntoIterator<Item = (String, (u64, Front))>) {
        self.due.live = true;
        for (panel, (sequence, front)) in fronts {
            if self.due.fronts.get(&panel).is_none_or(|(pending, _)| *pending <= sequence) { self.due.fronts.insert(panel, (sequence, front)); }
        }
    }

    /// Whether to present now: a reply at once; anything else at a `beat` (the end of a batch of
    /// events, or the board's beat), a beat after the last presentation.
    pub fn due(&self, now: Instant, beat: bool) -> bool {
        self.now || (beat && self.due.pending() && self.last.is_none_or(|last| now >= last + self.beat))
    }

    /// When the loop must turn for what is pending, should no batch of events end before then.
    pub fn wake_at(&self) -> Option<Instant> {
        if !self.due.pending() { return None; }
        self.last.map(|last| last + self.beat)
    }

    /// Everything pending, presented now.
    pub fn take(&mut self, now: Instant) -> Due {
        self.now = false;
        self.last = Some(now);
        std::mem::take(&mut self.due)
    }

    /// The live document's views that are stale, to be made now for a reader of the held state
    /// (MCP get_state); the document itself is still told at the beat.
    pub fn take_stale(&mut self) -> Due {
        let stale = Due { workbench: self.due.workbench, signals: self.due.signals, ..Due::default() };
        self.due.workbench = false;
        self.due.signals = false;
        stale
    }
}

/// The live document replaced by a board job's `answer`, which holds only the job's own sections:
/// the sections the presentation made (the workbench's views, the diagnostics) stay as they are
/// until the next beat makes them again; a section the last answer had and this one has not goes.
/// `answered` is the last answer's sections.
pub fn replace_answer(document: &mut Value, mut answer: Value, answered: &mut BTreeSet<String>) {
    let sections: BTreeSet<String> = answer.as_object().map(|object| object.keys().cloned().collect()).unwrap_or_default();
    if let (Some(held), Some(object)) = (document.as_object_mut(), answer.as_object_mut()) {
        for (key, value) in std::mem::take(held) {
            if !object.contains_key(&key) && !answered.contains(&key) { object.insert(key, value); }
        }
    }
    *document = answer;
    *answered = sections;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn front() -> Front { Front { width: 1, height: 1, pixels: zlc_core::OwnedBuffer::from_vec(vec![0u8; 4]) } }

    #[test]
    fn events_in_one_batch_are_one_presentation_and_a_reply_goes_at_once() {
        let mut presentation = Presentation::default();
        let start = Instant::now();
        // Nothing changed: nothing is due, whenever the loop asks.
        assert!(!presentation.due(start, true));
        assert_eq!(presentation.wake_at(), None);
        // Several events of one batch mark their views; none is presented mid-batch.
        presentation.mark(Owner::Workbench);
        presentation.mark(Owner::Live);
        presentation.board([("panel-1".to_owned(), (1, front()))]);
        presentation.board([("panel-1".to_owned(), (2, front()))]);
        presentation.mark(Owner::Pulse);
        assert!(!presentation.due(start, false), "events are presented when the batch ends");
        // The batch ends: one presentation of everything, the panel's newest picture.
        assert!(presentation.due(start, true));
        let due = presentation.take(start);
        assert!(due.workbench && due.live && due.pulse && due.board && due.replies.is_empty());
        assert_eq!(due.fronts.iter().map(|(panel, (sequence, _))| (panel.as_str(), *sequence)).collect::<Vec<_>>(), [("panel-1", 2)]);
        assert!(!presentation.due(start, true), "and nothing is left");
        // More within the beat waits for it; the loop is woken then.
        presentation.mark(Owner::Workbench);
        let beat = Duration::from_millis(zlc_workbench::INTERVALS[0]);
        assert!(!presentation.due(start + beat / 2, true));
        assert_eq!(presentation.wake_at(), Some(start + beat));
        assert!(presentation.due(start + beat, true));
        // A reply does not wait for the beat: it goes at the end of its own turn, with what is pending.
        let _ = presentation.take(start + beat);
        presentation.mark(Owner::Live);
        presentation.reply(Reply::done(&json!(7)));
        assert!(presentation.due(start + beat, false));
        let due = presentation.take(start + beat);
        assert_eq!((due.replies, due.live), (vec![Reply::done(&json!(7))], true));
        // A message that answers nothing is no reply: it waits for the beat.
        presentation.reply(Reply::default());
        assert!(!presentation.due(start + beat, false));
        // A view told whole by its owner is not pending any more.
        let _ = presentation.take(start + beat * 2);
        presentation.mark(Owner::Pulse);
        presentation.told(Owner::Pulse);
        assert!(!presentation.due(start + beat * 4, true));
        // A reader of the held state takes the stale views; the document is still told at the beat.
        presentation.mark(Owner::Workbench);
        let stale = presentation.take_stale();
        assert!(stale.workbench && !stale.live);
        let due = presentation.take(start + beat * 4);
        assert!(!due.workbench && due.live);
    }

    #[test]
    fn a_board_answer_keeps_the_presented_sections_until_they_are_made_again() {
        let mut answered = BTreeSet::new();
        let mut document = json!({"type": "state", "panels": [], "acquisition": {}, "logic": [1], "diagnostics": {}});
        replace_answer(&mut document, json!({"type": "state", "panels": [1], "saved": {"path": "a"}, "acquisition": {"produced": 1}}), &mut answered);
        assert_eq!(document, json!({"type": "state", "panels": [1], "saved": {"path": "a"}, "acquisition": {"produced": 1}, "logic": [1], "diagnostics": {}}));
        // A section the last answer had and this one lacks goes; the presented ones stay.
        replace_answer(&mut document, json!({"type": "state", "panels": [2], "acquisition": {"produced": 2}}), &mut answered);
        assert_eq!(document, json!({"type": "state", "panels": [2], "acquisition": {"produced": 2}, "logic": [1], "diagnostics": {}}));
    }
}
