//! How a loop table plays: the arithmetic the model, the compiler and the
//! schedule share. A loop table is `(first_row, last_row, count)` per bracket,
//! outermost first; how long one Pulse lasts and which rows it enters in what
//! order are facts about that table alone, so there is one walk.
use crate::error::{Error, Kind, Result};
use crate::exact::BigUint;

pub type Loop = (usize, usize, u64);

/// One loop of the table with the loops it encloses, in row order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopNode { pub start: usize, pub end: usize, pub count: u64, pub children: Vec<LoopNode> }

/// The loop table as a forest, outermost loops at the top level. The table is
/// stored outer-first, the order the board's walker pushes loops, so a stack
/// rebuilds the nesting without searching.
pub fn loop_tree(loops: &[Loop]) -> Vec<LoopNode> {
    let mut roots = Vec::new();
    let mut pending: Vec<LoopNode> = Vec::new();
    fn close(node: LoopNode, pending: &mut Vec<LoopNode>, roots: &mut Vec<LoopNode>) {
        match pending.last_mut() { Some(parent) => parent.children.push(node), None => roots.push(node) }
    }
    for &(start, end, count) in loops {
        while pending.last().is_some_and(|top| !(top.start <= start && end <= top.end)) {
            let node = pending.pop().expect("non-empty");
            close(node, &mut pending, &mut roots);
        }
        pending.push(LoopNode { start, end, count, children: Vec::new() });
    }
    while let Some(node) = pending.pop() { close(node, &mut pending, &mut roots); }
    roots
}

/// How many loops the board must hold on its stack at once.
pub fn loop_nesting_depth(loops: &[Loop]) -> usize {
    loops.iter().enumerate().map(|(index, &(start, end, _))| {
        1 + loops.iter().enumerate().filter(|&(other, &(s, e, _))| other != index && s <= start && end <= e).count()
    }).max().unwrap_or(0)
}

fn span_big(node: &LoopNode, durations: &[BigUint]) -> BigUint {
    let sum = |rows: std::ops::Range<usize>| rows.fold(BigUint::zero(), |acc, row| acc.add(&durations[row]));
    let mut total = BigUint::zero();
    let mut row = node.start;
    for child in &node.children {
        total = total.add(&sum(row..child.start)).add(&span_big(child, durations).mul_small(child.count));
        row = child.end + 1;
    }
    total.add(&sum(row..node.end + 1))
}

/// How long one complete Pulse lasts, exactly, for any row durations.
pub fn frame_ticks_big(durations: &[BigUint], loops: &[Loop]) -> BigUint {
    if durations.is_empty() { return BigUint::zero(); }
    span_big(&LoopNode { start: 0, end: durations.len() - 1, count: 1, children: loop_tree(loops) }, durations)
}

fn overflow() -> Error { Error::new(Kind::Value, "this program lasts longer than the schedule can count") }

fn span(node: &LoopNode, durations: &[u64]) -> Result<u128> {
    let sum = |rows: std::ops::Range<usize>| rows.map(|row| u128::from(durations[row])).sum::<u128>();
    let mut total: u128 = 0;
    let mut row = node.start;
    for child in &node.children {
        let body = span(child, durations)?.checked_mul(u128::from(child.count)).ok_or_else(overflow)?;
        total = total.checked_add(sum(row..child.start)).and_then(|t| t.checked_add(body)).ok_or_else(overflow)?;
        row = child.end + 1;
    }
    total.checked_add(sum(row..node.end + 1)).ok_or_else(overflow)
}

/// How many ticks one complete Pulse of row durations lasts.
pub fn frame_ticks(durations: &[u64], loops: &[Loop]) -> Result<u128> {
    if durations.is_empty() { return Ok(0); }
    span(&LoopNode { start: 0, end: durations.len() - 1, count: 1, children: loop_tree(loops) }, durations)
}

/// Which replays of a Bracket to walk: every one, or a bounded first-and-last.
/// A delay window meets only the few bodies its length spans, so the first and
/// last `bodies` replays (at their true ticks) see every window there is.
pub fn bracket_iterations(count: u64, bodies: Option<u64>) -> Vec<u64> {
    match bodies {
        Some(bodies) if count > 2 * bodies => (0..bodies).chain(count - bodies..count).collect(),
        _ => (0..count).collect(),
    }
}

/// Every row one Pulse enters, as `(row, start tick)`, in the board's order.
/// `bracket_bodies` answers, from one replay's length in ticks, how many first
/// and last replays of that bracket to walk; None walks every replay.
pub fn frame_visits(durations: &[u64], loops: &[Loop], bracket_bodies: Option<&dyn Fn(u128) -> u64>) -> Result<Vec<(usize, u128)>> {
    let mut visits = Vec::new();
    if durations.is_empty() { return Ok(visits); }
    walk_rows(&loop_tree(loops), 0, durations.len() - 1, 0, durations, bracket_bodies, &mut visits)?;
    Ok(visits)
}

fn walk_rows(nodes: &[LoopNode], first_row: usize, last_row: usize, mut tick: u128, durations: &[u64],
             bracket_bodies: Option<&dyn Fn(u128) -> u64>, visits: &mut Vec<(usize, u128)>) -> Result<u128> {
    let mut row = first_row;
    for node in nodes {
        for index in row..node.start {
            visits.push((index, tick));
            tick = tick.checked_add(u128::from(durations[index])).ok_or_else(overflow)?;
        }
        let body = span(node, durations)?;
        let bodies = bracket_bodies.map(|choose| choose(body));
        let mut previous: i128 = -1;
        for iteration in bracket_iterations(node.count, bodies) {
            let skipped = (i128::from(iteration) - previous - 1) as u128;
            tick = tick.checked_add(skipped.checked_mul(body).ok_or_else(overflow)?).ok_or_else(overflow)?;
            tick = walk_rows(&node.children, node.start, node.end, tick, durations, bracket_bodies, visits)?;
            previous = i128::from(iteration);
        }
        let rest = (i128::from(node.count) - 1 - previous) as u128;
        tick = tick.checked_add(rest.checked_mul(body).ok_or_else(overflow)?).ok_or_else(overflow)?;
        row = node.end + 1;
    }
    for index in row..=last_row {
        visits.push((index, tick));
        tick = tick.checked_add(u128::from(durations[index])).ok_or_else(overflow)?;
    }
    Ok(tick)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_loops_expand_in_board_order() {
        let loops = [(0, 3, 2), (1, 2, 3)];
        let durations = [1, 2, 3, 4];
        assert_eq!(frame_ticks(&durations, &loops).unwrap(), 2 * (1 + 3 * 5 + 4));
        assert_eq!(loop_nesting_depth(&loops), 2);
        let visits = frame_visits(&durations, &loops, None).unwrap();
        assert_eq!(visits.len(), 2 * (2 + 3 * 2));
        assert_eq!(&visits[..4], &[(0, 0), (1, 1), (2, 3), (1, 6)]);
        let bounded = frame_visits(&durations, &[(0, 3, 10)], Some(&|_body| 1)).unwrap();
        assert_eq!(bounded.iter().map(|v| v.1).collect::<Vec<_>>(), vec![0, 1, 3, 6, 90, 91, 93, 96]);
    }
}
