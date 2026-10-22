//! The selector controller (`selectors._SelectorController`): the committed
//! selectors, their revisions, and one isolated transient pointer candidate.
//!
//! The Python controller locks itself; here the owner (the session) holds it
//! behind its own lock, so the controller is a plain value and `fork` is
//! `clone`.  Python's `_rollback_install` asks whether the stored object IS
//! the one it installed; that object identity is carried by
//! [`HeldSelector`], which every call that stores, removes or restores a
//! committed state takes or returns.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::{committed_selectors, CrosshairPoint, SelectorKind, SelectorState};

use super::drag::{dragged_value, stored_state, DragFrame, DragHandle, GestureState};

static NEXT_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// One committed selector together with the identity of the stored object.
#[derive(Clone, Copy, Debug)]
pub struct HeldSelector { state: SelectorState, identity: u64 }

impl HeldSelector {
    /// A new object holding `state` (Python's constructor or `replace`).
    pub fn new(state: SelectorState) -> Self { Self { state, identity: NEXT_IDENTITY.fetch_add(1, Ordering::Relaxed) } }
    pub fn state(&self) -> SelectorState { self.state }
    pub fn kind(&self) -> SelectorKind { self.state.kind() }
    /// `self is other`.
    pub fn is(&self, other: &HeldSelector) -> bool { self.identity == other.identity }
}

/// The committed selector front plus one pointer gesture.
#[derive(Clone, Debug, Default)]
pub struct SelectorController {
    states: [Option<HeldSelector>; 4],
    revisions: [Option<u64>; 4],
    draft: Option<SelectorState>,
    gesture: Option<GestureState>,
}

fn slot(kind: SelectorKind) -> usize { kind as usize }

fn missing(kind: SelectorKind) -> PlotError { PlotError::key(format!("<SelectorKind.{}: '{}'>", kind_name(kind), kind.value())) }

fn kind_name(kind: SelectorKind) -> &'static str {
    match kind { SelectorKind::XRange => "X_RANGE", SelectorKind::Area => "AREA", SelectorKind::Crosshair => "CROSSHAIR", SelectorKind::Threshold => "THRESHOLD" }
}

impl SelectorController {
    pub fn new() -> Self { Self::default() }

    /// Capture complete pointer state for a provisional front rollback.
    pub fn fork(&self) -> Self { self.clone() }

    pub fn active_gesture(&self) -> Option<GestureState> { self.gesture }

    /// The committed states in kind order.
    pub fn states(&self) -> Vec<SelectorState> { self.states.iter().flatten().map(HeldSelector::state).collect() }

    /// The committed state objects in kind order.
    pub fn held(&self) -> Vec<HeldSelector> { self.states.iter().flatten().copied().collect() }

    /// Committed and transient state captured together.
    pub fn snapshot(&self, include_candidate: bool) -> Result<SelectorSnapshot> {
        let candidate = if include_candidate { self.candidate_state()? } else { None };
        Ok(SelectorSnapshot { committed: self.states(), candidate })
    }

    pub fn state(&self, kind: SelectorKind) -> Result<SelectorState> {
        self.states[slot(kind)].map(|held| held.state).ok_or_else(|| missing(kind))
    }

    /// The highest revision each kind has reached (`_revisions`).
    pub fn revision(&self, kind: SelectorKind) -> Option<u64> { self.revisions[slot(kind)] }

    /// The state a gesture drags from instead of the committed one (`_draft`).
    pub fn draft(&self) -> Option<SelectorState> { self.draft }

    /// Atomically replace the one committed state for the state's kind.
    pub fn install(&mut self, state: SelectorState) -> Result<(Option<HeldSelector>, HeldSelector)> {
        if self.gesture.is_some() { return Err(PlotError::runtime("cannot install a selector during a pointer gesture")); }
        let index = slot(state.kind());
        let previous = self.states[index];
        let stored = HeldSelector::new(stored_state(state, self.revisions[index]));
        self.revisions[index] = Some(stored.state.revision);
        self.states[index] = Some(stored);
        Ok((previous, stored))
    }

    /// Shared geometry on another cell, without a numeric revision: a new
    /// controller, this one untouched, so a refused presentation keeps it.
    pub fn retarget_facet(&self, facet_index: u64, kinds: &[SelectorKind]) -> Result<Self> {
        if self.gesture.is_some() { return Err(PlotError::runtime("cannot retarget a selector during a pointer gesture")); }
        let mut moved = Self { revisions: self.revisions, ..Self::default() };
        for (index, held) in self.states.iter().enumerate() {
            moved.states[index] = held.map(|held| if kinds.contains(&held.kind()) {
                HeldSelector::new(SelectorState { facet_index: Some(facet_index), ..held.state })
            } else { held });
        }
        Ok(moved)
    }

    /// Promote the exact final gesture geometry without another revision
    /// (`_commit_finished`); the promoted state is a new stored object.
    pub fn commit_finished(&mut self, state: SelectorState) -> Result<(Option<HeldSelector>, HeldSelector)> {
        if self.gesture.is_some() || self.draft.is_some() { return Err(PlotError::runtime("cannot promote an unfinished pointer gesture")); }
        let index = slot(state.kind());
        let known = self.revisions[index].unwrap_or(state.revision);
        if state.revision < known { return Err(PlotError::runtime("finished selector revision is stale")); }
        let previous = self.states[index];
        let stored = HeldSelector::new(state);
        self.revisions[index] = Some(state.revision);
        self.states[index] = Some(stored);
        Ok((previous, stored))
    }

    /// Take the committed geometry of `kind` away; a gesture on that kind
    /// goes on from its own copy.
    pub fn remove(&mut self, kind: SelectorKind) -> Result<HeldSelector> {
        let removed = self.states[slot(kind)].take().ok_or_else(|| missing(kind))?;
        if self.gesture.is_some_and(|gesture| gesture.kind == kind) && self.draft.is_none() {
            self.draft = Some(removed.state);
        }
        Ok(removed)
    }

    /// Restore an exact state when its enclosing presentation rolls back.
    pub fn restore_removed(&mut self, state: HeldSelector) -> Result<()> {
        let index = slot(state.kind());
        if self.states[index].is_some() { return Err(PlotError::runtime("cannot restore a selector kind that is present")); }
        self.states[index] = Some(state);
        Ok(())
    }

    /// Restore the selector front preceding one failed presentation.
    pub fn rollback_install(&mut self, installed: &HeldSelector, previous: Option<HeldSelector>) -> Result<()> {
        let index = slot(installed.kind());
        if !self.states[index].is_some_and(|held| held.is(installed)) {
            return Err(PlotError::runtime("installed selector is no longer current"));
        }
        match previous {
            None => self.states[index] = None,
            Some(previous) => {
                if previous.kind() != installed.kind() { return Err(PlotError::value("previous selector kind differs from installed kind")); }
                self.states[index] = Some(previous);
            }
        }
        Ok(())
    }

    /// Arm a drag of `kind` from `(x, y)`; a gesture already held is dropped
    /// first.  `draft` replaces the committed state as what is dragged.
    pub fn pointer_down(&mut self, kind: SelectorKind, x: f64, y: f64, handle: DragHandle, draft: Option<SelectorState>) -> Result<()> {
        let point = CrosshairPoint::new(x, y)?;
        if self.gesture.is_some() {
            self.gesture = None;
            self.draft = None;
        }
        let state = match draft {
            None => self.state(kind)?,
            Some(draft) => {
                if draft.kind() != kind { return Err(PlotError::value("draft selector kind does not match pointer target")); }
                self.draft = Some(draft);
                draft
            }
        };
        self.gesture = Some(GestureState { kind: state.kind(), handle, origin: point, original_value: state.value,
                                           candidate_value: state.value, revision: state.revision });
        Ok(())
    }

    fn current(&self, gesture: &GestureState) -> Result<SelectorState> {
        match self.draft { Some(draft) => Ok(draft), None => self.state(gesture.kind) }
    }

    /// Move the held selector; the new candidate (with a new revision when it
    /// changed), or `None` without a gesture.
    pub fn pointer_move(&mut self, x: f64, y: f64, frame: &DragFrame) -> Result<Option<SelectorState>> {
        let point = CrosshairPoint::new(x, y)?;
        let Some(gesture) = self.gesture else { return Ok(None) };
        let current = self.current(&gesture)?;
        let value = dragged_value(current.kind(), &gesture, point, frame)?;
        if value == gesture.candidate_value {
            return Ok(Some(SelectorState::new(gesture.candidate_value, gesture.revision, current.facet_index)?));
        }
        let index = slot(current.kind());
        let revision = gesture.revision.max(current.revision).max(self.revisions[index].unwrap_or(current.revision)) + 1;
        self.revisions[index] = Some(revision);
        self.gesture = Some(GestureState { candidate_value: value, revision, ..gesture });
        Ok(Some(SelectorState::new(value, revision, current.facet_index)?))
    }

    /// The final transient value; the gesture ends whatever happens, and
    /// nothing is committed.
    pub fn finish_gesture(&mut self, x: f64, y: f64, frame: &DragFrame) -> Result<Option<SelectorState>> {
        let result = self.pointer_move(x, y, frame);
        self.gesture = None;
        self.draft = None;
        result
    }

    /// The transient pointer value, committed state untouched.
    pub fn candidate_state(&self) -> Result<Option<SelectorState>> {
        let Some(gesture) = self.gesture else { return Ok(None) };
        let current = self.current(&gesture)?;
        Ok(Some(SelectorState::new(gesture.candidate_value, gesture.revision, current.facet_index)?))
    }

    /// End the gesture; the state it dragged from, if there was one.
    pub fn cancel(&mut self) -> Result<Option<SelectorState>> {
        let result = match self.gesture { None => Ok(None), Some(gesture) => self.current(&gesture).map(Some) };
        self.gesture = None;
        self.draft = None;
        result
    }

    pub fn lost_pointer_capture(&mut self) -> Result<Option<SelectorState>> { self.cancel() }
}

/// One atomic selector front with at most one state of each kind
/// (`selectors.SelectorSnapshot`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SelectorSnapshot {
    /// In kind order.
    pub committed: Vec<SelectorState>,
    pub candidate: Option<SelectorState>,
}

impl SelectorSnapshot {
    pub fn new(committed: &[SelectorState], candidate: Option<SelectorState>) -> Result<Self> {
        Ok(Self { committed: committed_selectors(committed)?, candidate })
    }

    /// The scene states after the transient candidate takes its kind's place.
    pub fn states(&self) -> Vec<SelectorState> {
        let mut states = self.committed.clone();
        if let Some(candidate) = self.candidate {
            match states.iter_mut().find(|state| state.kind() == candidate.kind()) {
                Some(state) => *state = candidate,
                None => { states.push(candidate); states.sort_by_key(SelectorState::kind); }
            }
        }
        states
    }
}
