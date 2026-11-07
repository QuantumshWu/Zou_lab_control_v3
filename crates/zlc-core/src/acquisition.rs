//! A saved Figure's event history in columns: what `Read::provenance` writes for a
//! finite acquisition and a history window, and the one expansion every reader goes
//! through. Lossless: `expand(encode(old)) == old`, an irregular entry staying literal.
//!
//! * Integer columns (run, revision, ordinal, parents; a range's event, origin, extent)
//!   are rows under [rules](Chunk): a template of rows repeated with a constant step per
//!   cell, over one or more nested counts. A row that follows no rule is a literal chunk,
//!   so dropped and retried shots only end one rule and begin the next.
//! * Signals sit in a table, referred to by index.
//! * Records sit in a table of shapes. Records alike in their fields and their nesting
//!   share a template; a leaf they all agree on is written in it, a leaf that varies is a
//!   column under its path (no key name repeats per event). An event names its shape and
//!   its position in the shape's distinct records.
use std::borrow::Cow;
use std::collections::HashMap;

use serde_json::{json, Map, Value};

const ACQUISITION_FORMAT: &str = "zlc.acquisition/2";
const WINDOW_FORMAT: &str = "zlc.window/2";
/// What identifies a period's first row: an event's run, signal and record shape.
const EVENT_KEY: usize = 3;
/// A range's extent.
const RANGE_KEY: usize = 2;
const RANGE_CELLS: usize = 5;

/// An integer, or the null an ordinal may be.
type Cell = Option<i128>;

fn number(value: i128) -> Value {
    match (i64::try_from(value), u64::try_from(value)) {
        (Ok(signed), _) => signed.into(),
        (_, Ok(unsigned)) => unsigned.into(),
        _ => unreachable!("cells come from JSON integers"),
    }
}
fn integer(value: &Value) -> Option<i128> {
    value.as_i64().map(i128::from).or_else(|| value.as_u64().map(i128::from))
}
fn cell_value(cell: Cell) -> Value { cell.map_or(Value::Null, number) }

// ---------------------------------------------------------------- rules

/// `rows` repeated over the nested counts `repeat`, the last varying fastest; the n-th
/// row of an expansion at indices `i` is `rows[n] + sum(i[d] * step[d][n])`. No counts: literal.
#[derive(Clone, Debug, PartialEq)]
struct Chunk { rows: Vec<Vec<Cell>>, repeat: Vec<usize>, step: Vec<Vec<Vec<i128>>> }

/// `to - from` cell by cell, when the two rows can be one rule's.
fn delta(from: &[Cell], to: &[Cell]) -> Option<Vec<i128>> {
    if from.len() != to.len() { return None; }
    from.iter().zip(to).map(|cells| match cells {
        (Some(from), Some(to)) => Some(to - from),
        (None, None) => Some(0),
        _ => None,
    }).collect()
}

fn follows(row: &[Cell], start: &[Cell], step: &[i128], times: usize) -> bool {
    row.len() == start.len() && row.iter().zip(start).zip(step).all(|((cell, start), step)| match (cell, start) {
        (Some(cell), Some(start)) => *cell == start + step * times as i128,
        (None, None) => true,
        _ => false,
    })
}

/// The rows as rules, greedily in order. A rule's period runs from a row to the next row
/// with the same first `key` cells, so finding them is linear in the rows.
fn rules(rows: &[Vec<Cell>], key: usize) -> Vec<Chunk> {
    let mut next = vec![None; rows.len()];
    let mut seen: HashMap<&[Cell], usize> = HashMap::with_capacity(rows.len().min(1024));
    for (index, row) in rows.iter().enumerate().rev() { next[index] = seen.insert(&row[..key], index); }
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut at = 0;
    while at < rows.len() {
        let period = next[at].map_or(1, |following: usize| following - at);
        let whole = |repeat: usize| at + repeat * period <= rows.len();
        let step: Option<Vec<Vec<i128>>> = if whole(2) { (0..period).map(|row| delta(&rows[at + row], &rows[at + period + row])).collect() } else { None };
        let Some(step) = step else {
            chunks.push(Chunk { rows: rows[at..at + period].to_vec(), repeat: Vec::new(), step: Vec::new() });
            at += period;
            continue;
        };
        let mut repeat = 2;
        while whole(repeat + 1) && (0..period).all(|row| follows(&rows[at + repeat * period + row], &rows[at + row], &step[row], repeat)) { repeat += 1; }
        chunks.push(Chunk { rows: rows[at..at + period].to_vec(), repeat: vec![repeat], step: vec![step] });
        at += repeat * period;
    }
    nest(chunks)
}

/// Rules that themselves progress become one rule with one more count (a raster's lines),
/// until none does; adjacent literal rows become one literal chunk.
fn nest(mut chunks: Vec<Chunk>) -> Vec<Chunk> {
    loop {
        let before = chunks.len();
        let mut nested: Vec<Chunk> = Vec::with_capacity(before);
        let mut items = chunks.into_iter().peekable();
        while let Some(first) = items.next() {
            if first.repeat.is_empty() {
                match nested.last_mut() {
                    Some(previous) if previous.repeat.is_empty() => previous.rows.extend(first.rows),
                    _ => nested.push(first),
                }
                continue;
            }
            let alike = |chunk: &Chunk| chunk.repeat == first.repeat && chunk.step == first.step && chunk.rows.len() == first.rows.len();
            let progress: Option<Vec<Vec<i128>>> = items.peek().filter(|chunk| alike(chunk))
                .and_then(|chunk| first.rows.iter().zip(&chunk.rows).map(|(from, to)| delta(from, to)).collect());
            let Some(progress) = progress else { nested.push(first); continue };
            let mut count = 1;
            while items.peek().is_some_and(|chunk| alike(chunk) && chunk.rows.iter().zip(&first.rows).zip(&progress)
                .all(|((row, start), step)| follows(row, start, step, count))) {
                items.next();
                count += 1;
            }
            let Chunk { rows, mut repeat, mut step } = first;
            repeat.insert(0, count);
            step.insert(0, progress);
            nested.push(Chunk { rows, repeat, step });
        }
        chunks = nested;
        if chunks.len() == before { return chunks; }
    }
}

fn chunks_value(chunks: &[Chunk]) -> Value {
    Value::Array(chunks.iter().map(|chunk| {
        let mut wire = Map::new();
        wire.insert("rows".into(), Value::Array(chunk.rows.iter().map(|row| Value::Array(row.iter().copied().map(cell_value).collect())).collect()));
        if !chunk.repeat.is_empty() {
            wire.insert("repeat".into(), json!(chunk.repeat));
            wire.insert("step".into(), Value::Array(chunk.step.iter().map(|level| Value::Array(level.iter()
                .map(|row| Value::Array(row.iter().copied().map(number).collect())).collect())).collect()));
        }
        Value::Object(wire)
    }).collect())
}

/// Each row of the stored rules, in order.
fn for_each_row(chunks: &Value, mut visit: impl FnMut(&[Cell]) -> Result<(), String>) -> Result<(), String> {
    let malformed = |what: &str| format!("acquisition rule {what} is malformed");
    for chunk in chunks.as_array().ok_or_else(|| malformed("list"))? {
        let cells = |row: &Value| -> Result<Vec<Cell>, String> {
            row.as_array().ok_or_else(|| malformed("row"))?.iter().map(|cell| if cell.is_null() { Ok(None) } else { integer(cell).map(Some).ok_or_else(|| malformed("cell")) }).collect()
        };
        let starts: Vec<Vec<Cell>> = chunk["rows"].as_array().ok_or_else(|| malformed("rows"))?.iter().map(cells).collect::<Result<_, _>>()?;
        let counts: Vec<usize> = chunk.get("repeat").and_then(Value::as_array).into_iter().flatten()
            .map(|count| count.as_u64().map(|count| count as usize).ok_or_else(|| malformed("count"))).collect::<Result<_, _>>()?;
        let levels: Vec<Vec<Vec<i128>>> = chunk.get("step").and_then(Value::as_array).into_iter().flatten().map(|level| {
            level.as_array().ok_or_else(|| malformed("step"))?.iter().map(|row| {
                row.as_array().ok_or_else(|| malformed("step"))?.iter().map(|step| integer(step).ok_or_else(|| malformed("step"))).collect::<Result<Vec<_>, _>>()
            }).collect::<Result<Vec<_>, _>>()
        }).collect::<Result<_, _>>()?;
        let shaped = levels.len() == counts.len() && levels.iter().all(|level| level.len() == starts.len()
            && level.iter().zip(&starts).all(|(steps, row)| steps.len() == row.len()));
        if !shaped || counts.contains(&0) { return Err(malformed("shape")); }
        let mut indices = vec![0usize; counts.len()];
        let mut row: Vec<Cell> = Vec::new();
        'expansion: loop {
            for (position, start) in starts.iter().enumerate() {
                row.clear();
                row.extend(start.iter().enumerate().map(|(column, cell)| cell.map(|cell| {
                    cell + indices.iter().zip(&levels).map(|(index, level)| *index as i128 * level[position][column]).sum::<i128>()
                })));
                visit(&row)?;
            }
            for dimension in (0..counts.len()).rev() {
                indices[dimension] += 1;
                if indices[dimension] < counts[dimension] { continue 'expansion; }
                indices[dimension] = 0;
            }
            break;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- records

/// A value's nesting and field names, its leaves unnamed: what records of one shape share.
fn shape_of(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (name, item) in map { out.push_str(&Value::String(name.clone()).to_string()); out.push(':'); shape_of(item, out); out.push(','); }
            out.push('}');
        }
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for item in items { shape_of(item, out); out.push(','); }
            out.push(']');
        }
        Value::Object(_) => out.push_str("{}"),
        Value::Array(_) => out.push_str("[]"),
        _ => out.push('_'),
    }
}

/// Exactly equal, a float's bits included (`-0.0` is not `0.0`; `1` is not `1.0`).
fn identical(left: &Value, right: &Value) -> bool {
    left == right && (!left.is_f64() || left.as_f64().map(f64::to_bits) == right.as_f64().map(f64::to_bits))
}

fn leaf_paths(value: &Value, path: &mut Vec<Value>, out: &mut Vec<Vec<Value>>) {
    match value {
        Value::Object(map) if !map.is_empty() => for (name, item) in map {
            path.push(Value::String(name.clone()));
            leaf_paths(item, path, out);
            path.pop();
        },
        Value::Array(items) if !items.is_empty() => for (index, item) in items.iter().enumerate() {
            path.push(json!(index));
            leaf_paths(item, path, out);
            path.pop();
        },
        _ => out.push(path.clone()),
    }
}

fn at<'a>(mut value: &'a Value, path: &[Value]) -> &'a Value {
    for step in path { value = match step { Value::String(name) => &value[name.as_str()], step => &value[step.as_u64().unwrap_or(u64::MAX) as usize] }; }
    value
}

fn at_mut<'a>(mut value: &'a mut Value, path: &[Value]) -> Option<&'a mut Value> {
    for step in path {
        value = match step {
            Value::String(name) => value.as_object_mut()?.get_mut(name.as_str())?,
            step => value.as_array_mut()?.get_mut(usize::try_from(step.as_u64()?).ok()?)?,
        };
    }
    Some(value)
}

#[derive(Default)]
struct Records { known: HashMap<String, (usize, usize)>, shapes: HashMap<String, usize>, distinct: Vec<Vec<Value>> }

impl Records {
    /// (shape, position) of a record, its text telling exact values apart.
    fn index(&mut self, record: &Value) -> (usize, usize) {
        let text = record.to_string();
        if let Some(found) = self.known.get(&text) { return *found; }
        let mut shape = String::new();
        shape_of(record, &mut shape);
        let group = *self.shapes.entry(shape).or_insert_with(|| { self.distinct.push(Vec::new()); self.distinct.len() - 1 });
        let position = self.distinct[group].len();
        self.distinct[group].push(record.clone());
        self.known.insert(text, (group, position));
        (group, position)
    }

    fn value(&self) -> Value {
        Value::Array(self.distinct.iter().map(|records| {
            let mut template = records[0].clone();
            let mut paths = Vec::new();
            leaf_paths(&records[0], &mut Vec::new(), &mut paths);
            let columns: Vec<Value> = paths.into_iter().filter(|path| {
                let first = at(&records[0], path);
                records[1..].iter().any(|record| !identical(first, at(record, path)))
            }).map(|path| {
                *at_mut(&mut template, &path).expect("a path of the template") = Value::Null;
                let values: Vec<Value> = records.iter().map(|record| at(record, &path).clone()).collect();
                json!({"path": path, "values": values})
            }).collect();
            let mut wire = Map::new();
            wire.insert("template".into(), template);
            if !columns.is_empty() { wire.insert("columns".into(), Value::Array(columns)); }
            Value::Object(wire)
        }).collect())
    }
}

/// The record at `(shape, position)` of a stored table.
fn stored_record(table: &[Value], shape: i128, position: i128) -> Result<Value, String> {
    let group = usize::try_from(shape).ok().and_then(|shape| table.get(shape)).ok_or("acquisition event names an unknown record shape")?;
    let mut record = group.get("template").cloned().ok_or("acquisition record shape has no template")?;
    for column in group.get("columns").and_then(Value::as_array).into_iter().flatten() {
        let path = column["path"].as_array().ok_or("acquisition record column has no path")?;
        let value = usize::try_from(position).ok().and_then(|position| column["values"].get(position)).ok_or("acquisition event names an unknown record")?;
        *at_mut(&mut record, path).ok_or("acquisition record column does not fit its template")? = value.clone();
    }
    Ok(record)
}

// ---------------------------------------------------------------- encoding

/// A finite acquisition or window's events as they are gathered: what `finish` stores.
#[derive(Default)]
pub struct Builder { runs: Vec<Value>, signals: Vec<String>, signal_index: HashMap<String, usize>, records: Records, events: Vec<Vec<Cell>>, ranges: Vec<Vec<Cell>> }

impl Builder {
    pub fn run(&mut self, stream: &str, generation: &str, record: &Value) -> usize {
        self.runs.push(json!({"stream": stream, "generation": generation, "record": record}));
        self.runs.len() - 1
    }
    /// One event, with its parents' indices; its index.
    pub fn event(&mut self, run: usize, signal: &str, revision: u64, ordinal: Option<u64>, record: &Value, parents: &[usize]) -> usize {
        self.event_cells(run, signal, Some(i128::from(revision)), ordinal.map(i128::from), record, parents.iter().map(|parent| Some(*parent as i128)))
    }
    fn event_cells(&mut self, run: usize, signal: &str, revision: Cell, ordinal: Cell, record: &Value, parents: impl Iterator<Item = Cell>) -> usize {
        let table = match self.signal_index.get(signal) {
            Some(index) => *index,
            None => {
                self.signals.push(signal.to_owned());
                self.signal_index.insert(signal.to_owned(), self.signals.len() - 1);
                self.signals.len() - 1
            }
        };
        let (shape, position) = self.records.index(record);
        let mut row = vec![Some(run as i128), Some(table as i128), Some(shape as i128), Some(position as i128), revision, ordinal];
        row.extend(parents);
        self.events.push(row);
        self.events.len() - 1
    }
    pub fn range(&mut self, event: usize, origin: [usize; 2], extent: [usize; 2]) {
        self.ranges.push(vec![Some(extent[0] as i128), Some(extent[1] as i128), Some(event as i128), Some(origin[0] as i128), Some(origin[1] as i128)]);
    }
    pub fn finish(self) -> Value {
        json!({"format": ACQUISITION_FORMAT, "runs": self.runs, "signals": self.signals, "records": self.records.value(),
               "events": chunks_value(&rules(&self.events, EVENT_KEY)), "ranges": chunks_value(&rules(&self.ranges, RANGE_KEY))})
    }
}

/// A window: its width, its members' indices and the events they and their ancestors are.
pub fn window(width: usize, members: &[usize], events: Builder) -> Value {
    json!({"format": WINDOW_FORMAT, "width": width, "members": members, "acquisition": events.finish()})
}

/// Keep the columns only when they read back as exactly `original` (a hand-made or foreign
/// entry that does not fit them stays as it was written).
fn verified(original: &Value, encoded: Value, expand: impl Fn(&Value) -> Result<Cow<'_, Value>, String>) -> Option<Value> {
    (expand(&encoded).ok()?.as_ref() == original).then_some(encoded)
}

/// The columns of an acquisition stored event by event; `None` when it does not fit them.
pub fn encode_acquisition(acquisition: &Value) -> Option<Value> {
    let mut builder = Builder::default();
    for run in acquisition["runs"].as_array()? { builder.runs.push(run.clone()); }
    for event in acquisition["events"].as_array()? {
        let cell = |value: &Value| if value.is_null() { Some(None) } else { integer(value).map(Some) };
        let run = event["run"].as_u64()? as usize;
        let parents: Vec<Cell> = event["parents"].as_array()?.iter().map(|parent| integer(parent).map(Some)).collect::<Option<_>>()?;
        builder.event_cells(run, event["signal"].as_str()?, Some(integer(&event["revision"])?), cell(&event["ordinal"])?, &event["record"], parents.into_iter());
    }
    for range in acquisition["ranges"].as_array()? {
        let pair = |value: &Value| -> Option<[i128; 2]> { let items = value.as_array().filter(|items| items.len() == 2)?; Some([integer(&items[0])?, integer(&items[1])?]) };
        let (origin, extent) = (pair(&range["origin"])?, pair(&range["extent"])?);
        builder.ranges.push(vec![Some(extent[0]), Some(extent[1]), Some(integer(&range["event"])?), Some(origin[0]), Some(origin[1])]);
    }
    verified(acquisition, builder.finish(), expand_acquisition)
}

/// The columns of a window stored as a tree per member; `None` when it does not fit them.
pub fn encode_window(stored: &Value) -> Option<Value> {
    #[derive(Default)]
    struct Walk { builder: Builder, runs: HashMap<(String, String), usize>, events: HashMap<String, usize> }
    fn visit(walk: &mut Walk, tree: &Value) -> Option<usize> {
        let publication = &tree["publication"];
        let identity = publication.to_string();
        if let Some(found) = walk.events.get(&identity) { return Some(*found); }
        let (stream, generation) = (publication["stream"].as_str()?, publication["generation"].as_str()?);
        let run = match walk.runs.get(&(stream.to_owned(), generation.to_owned())) {
            Some(run) => *run,
            None => {
                let run = walk.builder.run(stream, generation, &tree["run"]);
                walk.runs.insert((stream.to_owned(), generation.to_owned()), run);
                run
            }
        };
        let parents: Vec<Cell> = tree["parents"].as_array()?.iter().map(|parent| visit(walk, parent).map(|index| Some(index as i128))).collect::<Option<_>>()?;
        let ordinal = &publication["ordinal"];
        let index = walk.builder.event_cells(run, publication["signal"].as_str()?, Some(integer(&publication["revision"])?),
            if ordinal.is_null() { None } else { Some(integer(ordinal)?) }, &tree["event"], parents.into_iter());
        walk.events.insert(identity, index);
        Some(index)
    }
    let mut walk = Walk::default();
    let members: Vec<usize> = stored["members"].as_array()?.iter().map(|member| visit(&mut walk, member)).collect::<Option<_>>()?;
    verified(stored, window(usize::try_from(stored["width"].as_u64()?).ok()?, &members, walk.builder), expand_window)
}

// ---------------------------------------------------------------- reading

fn stored_events(acquisition: &Value, mut visit: impl FnMut(Value) -> Result<(), String>) -> Result<(), String> {
    let signals = acquisition["signals"].as_array().ok_or("acquisition has no signal table")?;
    let records = acquisition["records"].as_array().ok_or("acquisition has no record table")?;
    for_each_row(&acquisition["events"], |row| {
        let [Some(run), Some(signal), Some(shape), Some(position), revision, ordinal, parents @ ..] = row else { return Err("acquisition event row is malformed".into()) };
        let signal = usize::try_from(*signal).ok().and_then(|signal| signals.get(signal)).ok_or("acquisition event names an unknown signal")?;
        visit(json!({"run": number(*run), "signal": signal, "revision": cell_value(*revision), "ordinal": cell_value(*ordinal),
            "record": stored_record(records, *shape, *position)?, "parents": parents.iter().copied().map(cell_value).collect::<Vec<_>>()}))
    })
}

/// Inspect scientific records from a saved acquisition in either wire format.
/// The compressed form decodes one event at a time; no copy of its complete
/// event or placement tables is needed for metadata-only readers.
pub fn visit_acquisition_records(table: &Value, mut visit: impl FnMut(usize, &str, &Value, &Value)) -> Result<(), String> {
    let runs = table["runs"].as_array().ok_or("acquisition has no run table")?;
    let mut event = |entry: &Value| -> Result<(), String> {
        let index = entry["run"].as_u64().and_then(|i| usize::try_from(i).ok()).ok_or("invalid saved run index")?;
        let run = runs.get(index).ok_or("acquisition event names an unknown run")?;
        let signal = entry["signal"].as_str().ok_or("acquisition event has no signal")?;
        visit(index, signal, &run["record"], &entry["record"]);
        Ok(())
    };
    match table.get("format") {
        None => table["events"].as_array().ok_or("acquisition has no events")?.iter().try_for_each(event),
        Some(format) if format == ACQUISITION_FORMAT => stored_events(table, |entry| event(&entry)),
        Some(format) => Err(format!("unknown acquisition format {format}")),
    }
}

/// An acquisition as readers take it, `{runs, events: [{run, signal, revision, ordinal,
/// record, parents}], ranges: [{event, origin, extent}]}`, however it is stored.
pub fn expand_acquisition(stored: &Value) -> Result<Cow<'_, Value>, String> {
    if stored.get("format").is_none() { return Ok(Cow::Borrowed(stored)); }
    if stored["format"] != ACQUISITION_FORMAT { return Err(format!("unknown acquisition format {}", stored["format"])); }
    let mut events = Vec::new();
    stored_events(stored, |event| { events.push(event); Ok(()) })?;
    let mut ranges = Vec::new();
    for_each_row(&stored["ranges"], |row| {
        if row.len() != RANGE_CELLS { return Err("acquisition range row is malformed".into()); }
        let value = |cell: usize| cell_value(row[cell]);
        ranges.push(json!({"event": value(2), "origin": [value(3), value(4)], "extent": [value(0), value(1)]}));
        Ok(())
    })?;
    Ok(Cow::Owned(json!({"runs": stored["runs"], "events": events, "ranges": ranges})))
}

/// A window as readers take it, `{width, members: [{publication, run, event, parents}]}`
/// with each parent again such a tree, however it is stored.
pub fn expand_window(stored: &Value) -> Result<Cow<'_, Value>, String> {
    if stored.get("format").is_none() { return Ok(Cow::Borrowed(stored)); }
    if stored["format"] != WINDOW_FORMAT { return Err(format!("unknown window format {}", stored["format"])); }
    let acquisition = &stored["acquisition"];
    if acquisition["format"] != ACQUISITION_FORMAT { return Err(format!("unknown acquisition format {}", acquisition["format"])); }
    let runs = acquisition["runs"].as_array().ok_or("window has no run table")?;
    let mut trees: Vec<Value> = Vec::new();
    stored_events(acquisition, |event| {
        let run = event["run"].as_u64().and_then(|run| runs.get(run as usize)).ok_or("window event names an unknown run")?;
        // A parent was gathered before its child.
        let parents: Vec<Value> = event["parents"].as_array().into_iter().flatten().map(|parent| {
            parent.as_u64().and_then(|parent| trees.get(parent as usize)).cloned().ok_or("window event names an unknown parent")
        }).collect::<Result<_, _>>()?;
        trees.push(json!({"publication": {"stream": run["stream"], "generation": run["generation"], "signal": event["signal"],
            "revision": event["revision"], "ordinal": event["ordinal"]}, "run": run["record"], "event": event["record"], "parents": parents}));
        Ok(())
    })?;
    let members: Vec<&Value> = stored["members"].as_array().ok_or("window has no members")?.iter().map(|member| {
        member.as_u64().and_then(|member| trees.get(member as usize)).ok_or("window names an unknown member")
    }).collect::<Result<_, _>>()?;
    Ok(Cow::Owned(json!({"width": stored["width"], "members": members})))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (xorshift64*): the same irregular cases every run.
    struct Random(u64);
    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        }
        fn below(&mut self, bound: u64) -> u64 { self.next() % bound.max(1) }
        fn chance(&mut self, percent: u64) -> bool { self.below(100) < percent }
    }

    fn event(run: usize, signal: &str, revision: i64, ordinal: Option<i64>, record: Value, parents: Vec<usize>) -> Value {
        json!({"run": run, "signal": signal, "revision": revision, "ordinal": ordinal, "record": record, "parents": parents})
    }

    fn range(event: usize, origin: [usize; 2], extent: [usize; 2]) -> Value { json!({"event": event, "origin": origin, "extent": extent}) }

    fn runs(count: usize) -> Vec<Value> {
        (0..count).map(|run| json!({"stream": format!("logic-{run}"), "generation": format!("generation-{run}"), "record": {"node": format!("logic-{run}"), "parameters": {"gain": run}}})).collect()
    }

    fn acquisition(runs: Vec<Value>, events: Vec<Value>, ranges: Vec<Value>) -> Value { json!({"runs": runs, "events": events, "ranges": ranges}) }

    fn round_trips(old: &Value) -> Value {
        let encoded = encode_acquisition(old).expect("the acquisition fits the columns");
        assert_eq!(expand_acquisition(&encoded).unwrap().to_string(), old.to_string(), "exactly the old structure");
        encoded
    }

    /// A camera, occupancy, survival, scan chain per shot, each producing its fixed record.
    fn chain(shots: usize, points: usize) -> Value {
        let (mut events, mut ranges) = (Vec::new(), Vec::new());
        let records = [json!({"records": {"frames": {}}}), json!({}), json!({}), json!({"records": {"scan": {}}})];
        for shot in 0..shots {
            for step in 0..4 {
                let parents = if step == 0 { vec![] } else { vec![events.len() - 1] };
                events.push(event(3 - step, &format!("@logic/{step}/out"), shot as i64 + 1, Some(shot as i64), records[step].clone(), parents));
            }
            ranges.push(range(events.len() - 1, [shot / points, shot % points], [1, 1]));
        }
        acquisition(runs(4), events, ranges)
    }

    #[test]
    fn a_regular_chain_is_a_handful_of_rules_whatever_its_length() {
        let (short, long) = (chain(1000, 50), chain(50_000, 50));
        let (short, long) = (round_trips(&short), round_trips(&long));
        assert_eq!(long["events"].as_array().unwrap().len(), 1);
        assert_eq!(long["events"][0]["repeat"], json!([50_000]));
        assert_eq!(long["ranges"].as_array().unwrap().len(), 1, "a raster is one rule of lines");
        assert_eq!(long["ranges"][0]["repeat"], json!([1000, 50]));
        assert_eq!(long["records"].as_array().unwrap().len(), 3, "three distinct records");
        assert_eq!(long.to_string().len(), short.to_string().len() + 3, "only the counts grow, by three digits");
    }

    #[test]
    fn a_varying_record_is_columns_under_its_paths_and_names_no_key_per_event() {
        let mut events = Vec::new();
        for shot in 0..1000usize {
            events.push(event(0, "@logic/wave/trace", shot as i64 + 1, Some(shot as i64),
                json!({"records": {"trace": {"record_timing": {"origin": shot as f64 * 0.125 + 0.0625, "step": 1e-9}, "channel": "ch1"}}}), vec![]));
        }
        let encoded = round_trips(&acquisition(runs(1), events, vec![]));
        assert_eq!(encoded["records"].as_array().unwrap().len(), 1);
        let shape = &encoded["records"][0];
        assert_eq!(shape["columns"].as_array().unwrap().len(), 1, "only the leaf that varies");
        assert_eq!(shape["columns"][0]["path"], json!(["records", "trace", "record_timing", "origin"]));
        assert_eq!(shape["template"]["records"]["trace"]["record_timing"]["step"], json!(1e-9));
        assert_eq!(encoded.to_string().matches("record_timing").count(), 2, "the template and the column's path");
    }

    #[test]
    fn exact_values_stay_apart() {
        let records = [json!({"a": 1}), json!({"a": 1.0}), json!({"a": 0.0}), json!({"a": -0.0}), json!({"a": null}), json!({"a": "1"}), json!({"a": []}), json!({"a": {}}), json!({"a": [1, 2]}), json!({"b": 1})];
        let events = records.iter().enumerate().map(|(index, record)| event(0, "s", index as i64, None, record.clone(), vec![])).collect();
        round_trips(&acquisition(runs(1), events, vec![]));
    }

    #[test]
    fn what_does_not_fit_the_columns_stays_as_written() {
        let fits = chain(3, 2);
        for spoil in [
            json!({"runs": [], "events": [{"run": 0, "signal": "s", "revision": 1.5, "ordinal": 0, "record": {}, "parents": []}], "ranges": []}),
            json!({"runs": [], "events": [{"run": 0, "signal": "s", "revision": 1, "ordinal": 0, "record": {}, "parents": [], "extra": 1}], "ranges": []}),
            json!({"runs": [], "events": [], "ranges": [{"event": 0, "origin": [0, 0, 0], "extent": [1, 1]}]}),
            json!({"runs": [], "events": [], "ranges": [], "extra": true}),
        ] { assert!(encode_acquisition(&spoil).is_none(), "{spoil}"); }
        assert!(matches!(expand_acquisition(&fits).unwrap(), Cow::Borrowed(_)), "the old form is read as it is");
        assert!(matches!(expand_acquisition(&json!({"format": "zlc.acquisition/99"})), Err(_)));
    }

    #[test]
    fn random_irregular_chains_expand_to_exactly_the_old_structure() {
        for seed in 1..=200u64 {
            let mut random = Random(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
            let run_count = 1 + random.below(5) as usize;
            let signals: Vec<String> = (0..1 + random.below(6)).map(|signal| format!("@logic/{signal}/out")).collect();
            let chain_length = 1 + random.below(5) as usize;
            let shots = random.below(if seed % 10 == 0 { 3000 } else { 200 }) as usize;
            let (drop_percent, retry_percent, record_percent, multi_percent, null_percent) =
                (random.below(30), random.below(20), random.below(60), random.below(40), random.below(10));
            let (mut events, mut ranges): (Vec<Value>, Vec<Value>) = (Vec::new(), Vec::new());
            let mut revision = 1i64;
            let mut last_of_shot = None;
            for shot in 0..shots {
                if random.chance(drop_percent) { revision += random.below(3) as i64; continue; }
                let mut previous: Option<usize> = None;
                for step in 0..chain_length {
                    if random.chance(drop_percent / 2) { continue; }
                    let record = if random.chance(record_percent) {
                        match random.below(4) {
                            0 => json!({"records": {"frames": {"index": shot}}}),
                            1 => json!({"records": {"scan": {"record_timing": {"origin": random.next() as f64 / 7.0, "shots": [shot, shot + 1]}}}}),
                            2 => json!({"records": {"scan": {"actual_devices": {"rf": {"values": {"frequency": 80.0 + (shot % 5) as f64}, "units": {"frequency": "MHz"}}}}}}),
                            _ => json!({"note": format!("shot {shot}"), "flag": shot % 2 == 0, "nothing": null, "empty": {}, "list": []}),
                        }
                    } else { json!({}) };
                    let mut parents: Vec<usize> = previous.into_iter().collect();
                    if random.chance(multi_percent) { if let Some(earlier) = last_of_shot { parents.push(earlier); } }
                    if random.chance(multi_percent) && !events.is_empty() { parents.push(random.below(events.len() as u64) as usize); }
                    let ordinal = (!random.chance(null_percent)).then_some(shot as i64);
                    let signal = &signals[(step + if random.chance(retry_percent) { 1 } else { 0 }) % signals.len()];
                    events.push(event((step + shot % 2) % run_count, signal, revision, ordinal, record, parents));
                    previous = Some(events.len() - 1);
                }
                if let Some(last) = previous {
                    last_of_shot = Some(last);
                    ranges.push(range(last, [shot / 7, shot % 7], [1, 1]));
                    if random.chance(retry_percent) { ranges.push(range(last, [shot / 7, shot % 7], [2, 1])); }
                }
                revision += 1;
                if random.chance(retry_percent) { revision -= 1; }
            }
            round_trips(&acquisition(runs(run_count), events, ranges));
        }
    }

    #[test]
    fn a_window_keeps_each_run_once_and_expands_to_its_trees() {
        let run = |stream: &str| json!({"node": stream, "device_snapshots": {"camera": {"exposure_seconds": 0.02}}});
        let publication = |stream: &str, signal: &str, revision: i64, ordinal: Option<i64>| json!({"stream": stream, "generation": format!("{stream}-generation"), "signal": signal, "revision": revision, "ordinal": ordinal});
        let tree = |pub_: Value, stream: &str, event: Value, parents: Vec<Value>| json!({"publication": pub_, "run": run(stream), "event": event, "parents": parents});
        let member = |shot: i64| {
            let camera = tree(publication("camera", "frames", shot + 1, Some(shot)), "camera", json!({"records": {"frames": {"index": shot}}}), vec![]);
            let shared = tree(publication("bench", "state", 1, None), "bench", json!({}), vec![]);
            tree(publication("occupancy", "occupied", shot + 1, Some(shot)), "occupancy", json!({}), vec![camera, shared])
        };
        let old = json!({"width": 8, "members": (0..8).map(member).collect::<Vec<_>>()});
        let encoded = encode_window(&old).expect("the window fits the columns");
        assert_eq!(expand_window(&encoded).unwrap().to_string(), old.to_string());
        assert_eq!(encoded["acquisition"]["runs"].as_array().unwrap().len(), 3);
        assert_eq!(encoded.to_string().matches("exposure_seconds").count(), 3, "once per run, not per member");
        // A window's run that differs between members is not one run: it stays as written.
        let mut differing = old.clone();
        differing["members"][3]["run"]["node"] = json!("another");
        assert!(encode_window(&differing).is_none());
        assert!(matches!(expand_window(&old).unwrap(), Cow::Borrowed(_)));
    }
}
