//! Algebraic witnesses for independent stationary modes. This uses the full
//! Hermitian Lindblad coordinate graph, including coherences, not a rate model.
use crate::Generator;

pub(crate) fn certified_zero_modes(g: &Generator, indices: &[usize]) -> usize {
    let n = indices.len();
    let mut local = vec![usize::MAX; g.matrix.len()];
    for (i, &global) in indices.iter().enumerate() { local[global] = i; }
    let mut edges = vec![Vec::new(); n];
    let mut reverse = edges.clone();
    for (i, &global) in indices.iter().enumerate() {
        for &(row, value) in &g.matrix[global] {
            if value == 0. { continue; }
            let j = local[row];
            // A reachable coordinate set must be closed.
            if j == usize::MAX { return 0; }
            edges[i].push(j); reverse[j].push(i);
        }
    }
    fn visit(i: usize, graph: &[Vec<usize>], seen: &mut [bool], order: &mut Vec<usize>) {
        if seen[i] { return; }
        seen[i] = true;
        for &j in &graph[i] { visit(j, graph, seen, order); }
        order.push(i);
    }
    let mut order = Vec::new(); let mut seen = vec![false; n];
    for i in 0..n { visit(i, &edges, &mut seen, &mut order); }
    seen.fill(false);
    let mut groups = Vec::new(); let mut membership = vec![0; n];
    for &i in order.iter().rev() {
        if seen[i] { continue; }
        let mut group = Vec::new(); visit(i, &reverse, &mut seen, &mut group);
        for &j in &group { membership[j] = groups.len(); }
        groups.push(group);
    }
    groups.iter().enumerate().filter(|(id, group)| {
        if group.iter().any(|&i| edges[i].iter().any(|&j| membership[j] != *id)) { return false; }
        // A terminal trace-bearing block has a nonzero left trace functional
        // annihilating its generator, hence at least one exact zero mode. This
        // includes a driven cycling transition, not just isolated dark levels.
        // Exact-zero coherence columns supply additional conserved modes.
        group.iter().any(|&i| indices[i] < g.n)
            || (group.len() == 1 && edges[group[0]].is_empty())
    }).count()
}
