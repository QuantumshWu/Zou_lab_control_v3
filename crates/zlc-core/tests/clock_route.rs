//! A board with a live source and a consumer of it, through the consumer's own deliveries:
//! its group waits for it only for the shot it owes.
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Delivery, DeliveryPolicy, Front, Generation, Input, Publication, Read, Runtime, RuntimeError, Start, Storage, Subscription};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{DType, OwnedBuffer};

fn schema() -> Arc<DatasetSchema> {
    let one = |id: &str, role: &str| Domain::dense(vec![1], vec![Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len: 1 }, None, None, None, None).unwrap()]).unwrap();
    DatasetSchema::new(one("r", "repeat"), one("p", "point"), one("c", "scalar"), ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap()
}

struct Rig { plane: Runtime, schema: Arc<DatasetSchema>, source: Generation, derived: Generation, feed: Subscription, front: Front, answers: u64 }

impl Rig {
    fn new(policy: DeliveryPolicy) -> Self {
        let (plane, schema) = (Runtime::new(), schema());
        let source = plane.begin("scan", vec![Declaration { signal: "scan".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
        let feed = plane.subscribe(&source, vec!["scan".into()], policy, Start::Next).unwrap();
        let derived = plane.begin_consuming("fit", vec![Declaration { signal: "fit".into(), schema: schema.clone(), storage: Storage::Latest }], json!({}), "scan".into(), Some(source.id())).unwrap();
        plane.bind_consumer(&derived, feed.ticket()).unwrap();
        Self { plane, schema, source, derived, feed, front: Front::default(), answers: 0 }
    }

    fn block(&self) -> Arc<DataBlock> { DataBlock::new(self.schema.clone(), OwnedBuffer::from_vec(vec![0.0]).into(), Validity::Uniform(true), None).unwrap() }

    fn publish(&self, ordinal: u64) {
        self.plane.commit(&self.source, &Bundle { inputs: vec![Input::Event { signal: "scan".into(), ordinal, data: self.block(), origin: None, shot_time: None }], metadata: json!({}), parents: vec![] }).unwrap();
    }

    /// The consumer takes its next delivery, which also says it finished the last.
    fn take(&self) -> Arc<Delivery> { self.feed.next(Duration::ZERO).unwrap().expect("a delivery") }

    /// It finished the last with nothing to take next.
    fn finish(&self) { assert!(self.feed.next(Duration::ZERO).unwrap().is_none()); }

    fn answer(&mut self, delivery: &Delivery) {
        let parent: Arc<Publication> = delivery.publications[0].clone();
        let ordinal = self.answers;
        self.answers += 1;
        let data = self.block();
        self.plane.commit(&self.derived, &Bundle { inputs: vec![Input::Event { signal: "fit".into(), ordinal, data, origin: None, shot_time: None }], metadata: json!({}), parents: vec![parent] }).unwrap();
    }

    /// (the source's shown ordinal, the consumer's shown ordinal, None: it has none).
    fn board(&mut self) -> (Option<u64>, Option<u64>) {
        let requests = [(self.source.clone(), Arc::<str>::from("scan"), None), (self.derived.clone(), Arc::<str>::from("fit"), None)];
        let shown = |read: &Result<Read, RuntimeError>| match read { Ok(read) => read.key().ordinal, Err(RuntimeError::NoData) => None, Err(error) => panic!("{error:?}") };
        let reads = self.plane.read_coherent_views(&requests, &mut self.front);
        (shown(&reads[0]), shown(&reads[1]))
    }
}

const FOLLOW: DeliveryPolicy = DeliveryPolicy::Follow;

/// Answers once every ten events: the source stays live, and the answer and its shot show together.
#[test]
fn a_consumer_with_no_answer_owes_nothing_and_an_answer_is_shown_with_its_shot() {
    let mut rig = Rig::new(FOLLOW);
    for ordinal in 0..35u64 {
        rig.publish(ordinal);
        let delivery = rig.take();
        if ordinal % 10 == 9 {
            // It has the newest in hand: the source waits, nothing goes back or blanks.
            let before = rig.board();
            assert_eq!(before.0, Some(ordinal - 1), "the source waits for the answer of its newest");
            rig.answer(&delivery);
            assert_eq!(rig.board(), (Some(ordinal), Some(ordinal / 10)), "together");
        }
        rig.finish();
        assert_eq!(rig.board().0, Some(ordinal), "nothing owed: the source is live");
    }
    assert_eq!(rig.board(), (Some(34), Some(2)), "and the answer stands");
}

/// An answer of every event: same-shot as ever while it keeps up.
#[test]
fn a_consumer_that_keeps_up_is_shown_with_its_shot() {
    let mut rig = Rig::new(FOLLOW);
    for ordinal in 0..6u64 {
        rig.publish(ordinal);
        let delivery = rig.take();
        if ordinal > 0 { assert_eq!(rig.board(), (Some(ordinal - 1), Some(ordinal - 1)), "the pair waits for the answer"); }
        rig.answer(&delivery);
        assert_eq!(rig.board(), (Some(ordinal), Some(ordinal)));
    }
}

/// Behind by more than the newest: the source does not wait, and the consumer shows
/// what it has answered; once caught up the pair are one shot again.
#[test]
fn a_consumer_catching_up_holds_nothing_back() {
    let mut rig = Rig::new(FOLLOW);
    for ordinal in 0..12u64 { rig.publish(ordinal); }
    assert_eq!(rig.board(), (Some(11), None), "it is far behind, with nothing to show yet");
    let mut last = None;
    for caught in 0..10u64 {
        let delivery = rig.take();
        rig.answer(&delivery);
        last = Some(caught);
        assert_eq!(rig.board(), (Some(11), last), "the source stays on its newest");
    }
    // Only the newest is left.
    let delivery = rig.take();
    rig.answer(&delivery);
    let delivery = rig.take();
    assert_eq!(rig.board().0, Some(11), "a source shown never goes back");
    rig.answer(&delivery);
    rig.finish();
    assert_eq!(rig.board(), (Some(11), Some(11)));
    rig.publish(12);
    let delivery = rig.take();
    assert_eq!(rig.board(), (Some(11), Some(11)), "keeping up again: it waits");
    rig.answer(&delivery);
    assert_eq!(rig.board(), (Some(12), Some(12)));
    assert_eq!(last, Some(9));
}

/// A chain (frames -> occupancy -> survival): a consumer that has answered the newest and
/// waits for more is caught up, not gone. While its own consumer still owes that shot, the
/// source waits with them: the picture and its judgement are never split by the slower stage
/// (the lab's occupied overlay blinked off for a moment on every shot).
#[test]
fn a_caught_up_consumer_keeps_its_source_with_its_own_consumers_shot() {
    let (plane, schema) = (Runtime::new(), schema());
    let declare = |signal: &str| vec![Declaration { signal: signal.into(), schema: schema.clone(), storage: Storage::Latest }];
    let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(vec![0.0]).into(), Validity::Uniform(true), None).unwrap();
    let commit = |run: &Generation, signal: &str, ordinal: u64, parents: Vec<Arc<Publication>>|
        plane.commit(run, &Bundle { inputs: vec![Input::Event { signal: signal.into(), ordinal, data: block.clone(), origin: None, shot_time: None }], metadata: json!({}), parents }).unwrap();
    let source = plane.begin("frames", declare("frames"), json!({})).unwrap();
    let first_feed = plane.subscribe(&source, vec!["frames".into()], FOLLOW, Start::Next).unwrap();
    let first = plane.begin_consuming("occupancy", declare("occupied"), json!({}), "frames".into(), Some(source.id())).unwrap();
    plane.bind_consumer(&first, first_feed.ticket()).unwrap();
    let second_feed = plane.subscribe(&first, vec!["occupied".into()], FOLLOW, Start::Next).unwrap();
    let second = plane.begin_consuming("survival", declare("survival"), json!({}), "occupied".into(), Some(first.id())).unwrap();
    plane.bind_consumer(&second, second_feed.ticket()).unwrap();
    let mut front = Front::default();
    let mut board = || {
        let requests = [(source.clone(), Arc::<str>::from("frames"), None), (first.clone(), Arc::<str>::from("occupied"), None),
                        (second.clone(), Arc::<str>::from("survival"), None)];
        plane.read_coherent_views(&requests, &mut front).iter().map(|read| read.as_ref().ok().and_then(|read| read.key().ordinal)).collect::<Vec<_>>()
    };
    let take = |feed: &Subscription| feed.next(Duration::ZERO).unwrap().expect("a delivery");
    for ordinal in 0..4u64 {
        commit(&source, "frames", ordinal, vec![]);
        let delivery = take(&first_feed);
        commit(&first, "occupied", ordinal, vec![delivery.publications[0].clone()]);
        assert!(first_feed.next(Duration::ZERO).unwrap().is_none(), "occupancy is caught up");
        let delivery = take(&second_feed);
        if ordinal > 0 { assert_eq!(board(), vec![Some(ordinal - 1); 3], "survival owes the newest: the three wait at one shot"); }
        commit(&second, "survival", ordinal, vec![delivery.publications[0].clone()]);
        assert!(second_feed.next(Duration::ZERO).unwrap().is_none());
        assert_eq!(board(), vec![Some(ordinal); 3], "together");
    }
}

/// A consumer whose feed is not told keeps its group, as it always did.
#[test]
fn a_consumer_with_no_known_feed_keeps_its_group() {
    let (plane, schema) = (Runtime::new(), schema());
    let source = plane.begin("scan", vec![Declaration { signal: "scan".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
    let derived = plane.begin_consuming("fit", vec![Declaration { signal: "fit".into(), schema: schema.clone(), storage: Storage::Latest }], json!({}), "scan".into(), Some(source.id())).unwrap();
    let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(vec![0.0]).into(), Validity::Uniform(true), None).unwrap();
    let publish = |ordinal: u64| plane.commit(&source, &Bundle { inputs: vec![Input::Event { signal: "scan".into(), ordinal, data: block.clone(), origin: None, shot_time: None }], metadata: json!({}), parents: vec![] }).unwrap();
    publish(0);
    let first = plane.read_latest(&source, "scan").unwrap();
    plane.commit(&derived, &Bundle { inputs: vec![Input::Event { signal: "fit".into(), ordinal: 0, data: block.clone(), origin: None, shot_time: None }], metadata: json!({}), parents: vec![first] }).unwrap();
    publish(1);
    let requests = [(source.clone(), Arc::<str>::from("scan"), None), (derived.clone(), Arc::<str>::from("fit"), None)];
    let reads = plane.read_coherent_views(&requests, &mut Front::default());
    assert_eq!(reads[0].as_ref().unwrap().key().ordinal, Some(0));
    assert_eq!(reads[1].as_ref().unwrap().key().ordinal, Some(0));
}
