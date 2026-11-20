//! Exact Runtime input for core services. Retains the existing bounded Runtime
//! subscription rather than adding a second queue or a Python delivery/ACK.
use std::sync::{atomic::{AtomicBool, Ordering}, mpsc::Receiver};
use std::time::{Duration, Instant};

use zlc_core::runtime::{DeliveryKind, GenerationId, Read, RuntimeError};
use crate::{CallError, ErrorKind, Feed, Source};

const CONTROL_SLICE: Duration = Duration::from_millis(20);
/// Why a source that ends during collection fails the scan rather than being followed to its next run.
const FIXED_SOURCE: &str = "a scan's source cannot change once collection began";

pub(crate) struct CoreInput {
    sources: Receiver<Source>,
    current: Option<Source>,
}

pub(crate) fn check_stop(stop: &AtomicBool) -> Result<(), CallError> {
    if stop.load(Ordering::Acquire) { Err(CallError::new(ErrorKind::Cancelled, "the run was stopped")) } else { Ok(()) }
}

impl CoreInput {
    pub fn new(sources: Receiver<Source>) -> Self { Self { sources, current: None } }

    pub fn ticket(&self) -> Option<zlc_core::runtime::SubscriptionTicket> {
        self.current.as_ref().and_then(|source| match &source.feed {
            Feed::Live(subscription) => Some(subscription.ticket()), Feed::Sealed(_) => None,
        })
    }

    pub fn generation(&self) -> Option<GenerationId> { self.current.as_ref().map(|source| source.generation) }

    /// Called before requesting a new acquisition. The already-registered
    /// follower will deliver its generation before that generation's first
    /// publication; the replaced source's terminal must not end the new scan.
    pub fn await_new_source(&mut self) {
        self.current = None;
        while self.sources.try_recv().is_ok() {}
    }

    /// One event, with its immutable payload and provenance still owned natively.
    /// Between segments, terminal markers can be drained; during collection they
    /// are failures rather than a promise that a replacement source will arrive.
    pub fn next(&mut self, timeout: Duration, stop: &AtomicBool, terminal_errors: bool) -> Result<Option<Read>, CallError> {
        let deadline = Instant::now() + timeout;
        loop {
            check_stop(stop)?;
            while let Ok(source) = self.sources.try_recv() { self.current = Some(source); }
            let wait = deadline.saturating_duration_since(Instant::now()).min(CONTROL_SLICE);
            let Some(source) = &self.current else {
                if wait.is_zero() { return Ok(None); }
                match self.sources.recv_timeout(wait) {
                    Ok(source) => self.current = Some(source),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err(CallError::new(ErrorKind::StaleRun, "the input owner ended")),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {},
                }
                continue;
            };
            let Feed::Live(subscription) = &source.feed else {
                self.current = None;
                if terminal_errors { return Err(CallError::new(ErrorKind::Failed, format!("the source run had already ended; {FIXED_SOURCE}"))); }
                continue;
            };
            match subscription.next(wait) {
                Ok(Some(delivery)) if matches!(delivery.kind, DeliveryKind::Data) => {
                    if delivery.reads.len() != 1 {
                        return Err(CallError::new(ErrorKind::Invalid, "a core input needs one source publication per event"));
                    }
                    return Ok(Some(delivery.reads[0].clone()));
                }
                Ok(None) => if Instant::now() >= deadline { return Ok(None); },
                Ok(Some(_)) | Err(RuntimeError::Sealed | RuntimeError::UnknownGeneration) => {
                    self.current = None;
                    if terminal_errors { return Err(CallError::new(ErrorKind::Failed, format!("the source run ended during collection; {FIXED_SOURCE}"))); }
                }
                Err(error) => return Err(CallError::new(ErrorKind::Failed, format!("scan input: {error}"))),
            }
        }
    }

    /// Clear only already-delivered history at a segment boundary. The first
    /// segment can request a bounded quiet period for legacy sources without an
    /// arm/drain acknowledgement. Later completed fires need no silence timer.
    pub fn drain(&mut self, quiet: Duration, stop: &AtomicBool) -> Result<usize, CallError> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut dropped = 0;
        loop {
            check_stop(stop)?;
            if Instant::now() >= deadline {
                return Err(CallError::new(ErrorKind::Timeout, "the source did not become quiet before the scan segment"));
            }
            match self.next(quiet, stop, false)? {
                Some(_) => dropped += 1,
                None => return Ok(dropped),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use serde_json::json;
    use zlc_core::{DType, OwnedBuffer};
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, DeliveryPolicy, Input, Runtime, Start, Storage};
    use zlc_core::schema::{DatasetSchema, Domain, ValidityContract, ValueSchema};

    #[test]
    fn a_requested_new_acquisition_does_not_inherit_the_old_sources_terminal() {
        let runtime = Runtime::new();
        let scalar = Domain::new(vec![1], vec![], vec![]).unwrap();
        let schema = DatasetSchema::new(scalar.clone(), scalar.clone(), scalar,
            ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
        let begin = |owner: &str| runtime.begin(owner, vec![Declaration { signal: format!("{owner}/data").into(),
            schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
        let source = |generation: &zlc_core::runtime::Generation, owner: &str| Source {
            feed: Feed::Live(runtime.subscribe(generation, vec![format!("{owner}/data").into()], DeliveryPolicy::Follow, Start::Current).unwrap()),
            bundle: false, input: "source".into(), generation: generation.id(),
        };
        let old = begin("old");
        let (sender, receiver) = mpsc::channel();
        assert!(sender.send(source(&old, "old")).is_ok());
        let mut input = CoreInput::new(receiver);
        let stop = AtomicBool::new(false);
        assert!(input.next(Duration::ZERO, &stop, false).unwrap().is_none());
        input.await_new_source();
        runtime.seal(&old, json!({"status":"stopped"})).unwrap();
        assert!(input.next(Duration::ZERO, &stop, true).unwrap().is_none());
        let new = begin("new");
        assert!(sender.send(source(&new, "new")).is_ok());
        let data = DataBlock::new(schema, OwnedBuffer::from_vec(vec![7.0]).into(), Validity::Uniform(true), None).unwrap();
        runtime.commit(&new, &Bundle { inputs: vec![Input::Event { signal: "new/data".into(), ordinal: 0,
            data: data.clone(), origin: None, shot_time: None }], metadata: json!({}), parents: vec![] }).unwrap();
        let read = input.next(Duration::ZERO, &stop, true).unwrap().unwrap();
        assert_eq!(read.key().generation, new.id());
        assert!(Arc::ptr_eq(read.anchor().data().unwrap(), &data));
        runtime.seal(&new, json!({"status":"completed"})).unwrap();
        let ended = input.next(Duration::ZERO, &stop, true).err().unwrap().message;
        assert!(ended.contains("source run ended during collection") && ended.contains("cannot change once collection began"), "{ended}");
    }

    #[test]
    fn stop_interrupts_waiting_for_a_generation_that_has_not_published() {
        let (_sender, receiver) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let observed = stop.clone();
        let started = Instant::now();
        let waiter = std::thread::spawn(move || CoreInput::new(receiver).next(Duration::from_secs(1), &observed, true));
        std::thread::sleep(Duration::from_millis(30));
        stop.store(true, Ordering::Release);
        let error = match waiter.join().unwrap() { Err(error) => error, Ok(_) => panic!("a stopped input kept waiting") };
        assert_eq!(error.kind, ErrorKind::Cancelled);
        assert!(started.elapsed() < Duration::from_millis(700));
    }
}
