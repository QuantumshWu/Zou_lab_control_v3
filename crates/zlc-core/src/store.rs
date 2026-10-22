//! Append-only finite data. Unwritten carrier positions are holes, not values.
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::dataset::DataBlock;
use crate::schema::{DatasetSchema, DomainRole};

const PAGE_SIZE: usize = 64;

#[derive(Clone)]
pub struct Placement {
    origin: [usize; 2],
    extent: [usize; 2],
    data: Arc<DataBlock>,
}

impl Placement {
    pub fn new(origin: [usize; 2], extent: [usize; 2], data: Arc<DataBlock>) -> Result<Self, String> {
        let schema = data.schema();
        if extent != [schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size()] {
            return Err("Placement extent must match the block's flat Repeat and Point carriers".into());
        }
        Ok(Self { origin, extent, data })
    }

    pub fn origin(&self) -> [usize; 2] { self.origin }
    pub fn extent(&self) -> [usize; 2] { self.extent }
    pub fn data(&self) -> &Arc<DataBlock> { &self.data }
}

#[derive(Clone)]
struct Page {
    previous: Option<Arc<Page>>,
    entries: Vec<Placement>,
}

impl Drop for Page {
    fn drop(&mut self) {
        // Release long, unshared directory tails without recursive stack growth.
        let mut previous = self.previous.take();
        while let Some(page) = previous {
            match Arc::try_unwrap(page) {
                Ok(mut page) => previous = page.previous.take(),
                Err(_) => break,
            }
        }
    }
}

#[derive(Clone)]
pub struct Snapshot {
    schema: Arc<DatasetSchema>,
    // Immutable source-window facts inherited from a scientific DataBlock.
    // This does not acquire or maintain any Runtime history.
    window: Option<(i128,u64)>,
    tail: Option<Arc<Page>>,
    segment_count: usize,
    written_cells: usize,
    total_cells: usize,
    payload_bytes: usize,
}

impl Snapshot {
    pub fn schema(&self) -> &Arc<DatasetSchema> { &self.schema }
    pub fn window(&self) -> Option<(i128,u64)> {self.window}
    pub fn with_window(mut self, window:Option<(i128,u64)>) -> Result<Self,String> {
        if window.is_some_and(|(start,latest)|start>i128::from(latest)) {
            return Err("Indexed window start exceeds its latest source ordinal".into());
        }
        self.window=window;Ok(self)
    }
    /// Normalize an equal declaration's schema owner without copying any
    /// acquired segment or turning unstarted positions into measurements.
    pub fn with_schema(&self, schema:Arc<DatasetSchema>) -> Result<Self,String> {
        if schema!=self.schema {return Err("Snapshot schema differs from its declaration".into());}
        let mut snapshot=self.clone();snapshot.schema=schema;Ok(snapshot)
    }
    pub fn segment_count(&self) -> usize { self.segment_count }
    /// Acquired Repeat × Point positions, independent of sample validity.
    pub fn written_cells(&self) -> usize { self.written_cells }
    pub fn total_cells(&self) -> usize { self.total_cells }
    pub fn holes(&self) -> usize { self.total_cells - self.written_cells }
    /// Distinct DataBlock owners in this immutable prefix, including all planes.
    pub fn payload_bytes(&self) -> usize { self.payload_bytes }

    /// Commit-order segments borrow their original immutable data planes.
    pub fn segments(&self) -> impl Iterator<Item = &Placement> {
        let mut pages = Vec::with_capacity(self.segment_count.div_ceil(PAGE_SIZE));
        let mut page = self.tail.as_deref();
        while let Some(current) = page {
            pages.push(current.entries.as_slice());
            page = current.previous.as_deref();
        }
        pages.into_iter().rev().flat_map(|entries| entries.iter())
    }
}

pub struct DatasetStore {
    current: Snapshot,
    occupied: BTreeMap<(usize, usize), usize>,
    // The append-only directory keeps every indexed owner alive.
    owners: HashSet<usize>,
}

impl DatasetStore {
    pub fn new(schema: Arc<DatasetSchema>) -> Self {
        let total_cells = schema.domain(DomainRole::Repeat).size() * schema.domain(DomainRole::Point).size();
        Self { current: Snapshot { schema, window:None, tail: None, segment_count: 0, written_cells: 0, total_cells, payload_bytes: 0 },
            occupied: BTreeMap::new(), owners: HashSet::new() }
    }

    pub fn snapshot(&self) -> Snapshot { self.current.clone() }

    /// Runtime validates every sibling under its state lock before installing any.
    pub fn validate(&self, placements: &[Placement]) -> Result<(), String> {
        self.additions(placements).map(|_| ())
    }

    fn additions(&self, placements: &[Placement]) -> Result<(BTreeMap<(usize, usize), usize>, usize), String> {
        let canonical = &self.current.schema;
        let bounds = [canonical.domain(DomainRole::Repeat).size(), canonical.domain(DomainRole::Point).size()];
        let mut added = BTreeMap::new();
        let mut written = self.current.written_cells;
        for placement in placements {
            let schema = placement.data.schema();
            if schema.domain(DomainRole::Cell) != canonical.domain(DomainRole::Cell)
                || schema.value() != canonical.value() {
                return Err("Placed block Cell and Value contracts differ from the declared dataset".into());
            }
            let mut end = [0; 2];
            for dimension in 0..2 {
                end[dimension] = placement.origin[dimension].checked_add(placement.extent[dimension])
                    .filter(|end| *end <= bounds[dimension]).ok_or("Placement lies outside declared dataset carriers")?;
            }
            for repeat in placement.origin[0]..end[0] {
                let start = placement.origin[1];
                if overlaps(&self.occupied, repeat, start, end[1]) || overlaps(&added, repeat, start, end[1]) {
                    return Err("Finite placement overlaps already written cells or another batch member".into());
                }
                insert_range(&mut added, repeat, start, end[1]);
            }
            written = written.checked_add(placement.extent[0] * placement.extent[1]).ok_or("Written coverage overflows usize")?;
        }
        Ok((added, written))
    }

    /// Precheck the complete batch before changing coverage or the persistent directory.
    pub fn commit(&mut self, placements: Vec<Placement>) -> Result<Snapshot, String> {
        let (added, written) = self.additions(&placements)?;
        for ((repeat, start), end) in added { insert_range(&mut self.occupied, repeat, start, end); }
        for placement in placements {
            if self.owners.insert(Arc::as_ptr(&placement.data) as usize) {
                self.current.payload_bytes = self.current.payload_bytes.saturating_add(placement.data.payload_bytes());
            }
            if self.current.tail.as_ref().is_none_or(|page| page.entries.len() == PAGE_SIZE) {
                self.current.tail = Some(Arc::new(Page {
                    previous: self.current.tail.take(), entries: Vec::with_capacity(PAGE_SIZE),
                }));
            }
            // Only this bounded tail page is copied when an older snapshot holds it.
            Arc::make_mut(self.current.tail.as_mut().unwrap()).entries.push(placement);
            self.current.segment_count += 1;
        }
        self.current.written_cells = written;
        Ok(self.snapshot())
    }
}

fn overlaps(ranges: &BTreeMap<(usize, usize), usize>, repeat: usize, start: usize, end: usize) -> bool {
    ranges.range((repeat, 0)..(repeat, end)).next_back().is_some_and(|(_, previous_end)| *previous_end > start)
}

fn insert_range(ranges: &mut BTreeMap<(usize, usize), usize>, repeat: usize, mut start: usize, mut end: usize) {
    if let Some((&(_, previous_start), &previous_end)) = ranges.range((repeat, 0)..(repeat, start)).next_back() {
        if previous_end == start {
            ranges.remove(&(repeat, previous_start));
            start = previous_start;
        }
    }
    if let Some((&(next_repeat, next_start), &next_end)) = ranges.range((repeat, end)..).next() {
        if next_repeat == repeat && next_start == end {
            ranges.remove(&(repeat, next_start));
            end = next_end;
        }
    }
    ranges.insert((repeat, start), end);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, OwnedBuffer};
    use crate::dataset::Validity;
    use crate::schema::{Axis, CodeBase, CodeMapping, Coordinates, Domain, ValidityContract, ValueSchema};

    #[test]
    fn mapped_finite_placement_is_atomic_shared_and_snapshot_stable() {
        let axis = |id: &str, len| Axis::new(id.into(), id.into(), "generic".into(),
            Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
        let scans = vec![axis("scan-a", 10), axis("scan-b", 10), axis("scan-c", 10)];
        let point = Domain::new(vec![1000], scans.clone(), [(100, 1), (10, 10), (1, 100)].into_iter()
            .map(|(inner, outer)| CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 10 }, inner, outer).unwrap()).collect()).unwrap();
        let repeat = Domain::dense(vec![20], vec![axis("repeat", 20)]).unwrap();
        let cell = Domain::dense(vec![3, 35], vec![axis("frame", 3), axis("site", 35)]).unwrap();
        let value = ValueSchema::new(DType::Bool, None, None, ValidityContract::Components(vec!["site".into()])).unwrap();
        let schema = DatasetSchema::new(repeat, point, cell.clone(), value.clone()).unwrap();
        let scalar = Domain::new(vec![1], vec![], vec![]).unwrap();
        let block_point = Domain::new(vec![10], scans, vec![
            CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 1 }, 10, 1).unwrap(),
            CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 1 }, 10, 1).unwrap(),
            CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 10 }, 1, 1).unwrap(),
        ]).unwrap();
        let block_schema = DatasetSchema::new(scalar, block_point, cell, value).unwrap();
        let block = DataBlock::new(block_schema, OwnedBuffer::from_vec(vec![true; 10 * 3 * 35]).into(),
            Validity::Uniform(false), None).unwrap();
        let weak = Arc::downgrade(&block);
        let mut store = DatasetStore::new(schema.clone());
        let empty = store.snapshot();
        assert_eq!(empty.holes(), 20_000);
        let first = store.commit(vec![Placement::new([0, 0], [1, 10], block.clone()).unwrap()]).unwrap();
        for index in 1..66 {
            store.commit(vec![Placement::new([0, index * 10], [1, 10], block.clone()).unwrap()]).unwrap();
        }
        let current = store.snapshot();
        assert_eq!((current.segment_count(), current.written_cells(), current.holes()), (66, 660, 19_340));
        assert_eq!(current.payload_bytes(), block.payload_bytes());
        assert_eq!(first.payload_bytes(), block.payload_bytes());
        assert_eq!(empty.payload_bytes(), 0);
        assert!(Arc::ptr_eq(current.schema(), &schema));
        assert_eq!(first.segments().count(), 1);
        assert_eq!(first.written_cells(), 10);
        assert_eq!(empty.segments().count(), 0);
        for (index, segment) in current.segments().enumerate() {
            assert_eq!(segment.origin(), [0, index * 10]);
            assert!(Arc::ptr_eq(segment.data(), &block));
        }
        assert!(store.commit(vec![
            Placement::new([1, 0], [1, 10], block.clone()).unwrap(),
            Placement::new([0, 5], [1, 10], block.clone()).unwrap(),
        ]).is_err());
        assert_eq!(store.snapshot().written_cells(), 660);
        let accepted = store.commit(vec![Placement::new([1, 0], [1, 10], block.clone()).unwrap()]).unwrap();
        assert_eq!(accepted.written_cells(), 670);
        drop(block); drop(store); drop(accepted); drop(current);
        assert!(weak.upgrade().is_some());
        assert_eq!(first.segments().next().unwrap().data().values().len(), 1050);
        drop(first);
        assert!(weak.upgrade().is_none());
    }
}
