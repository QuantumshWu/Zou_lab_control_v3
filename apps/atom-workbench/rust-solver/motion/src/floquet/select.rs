//! The rounding rule's elements (mdlat.select_all with L alone, RUSTPLAN correction "select by L"): best first from the
//! k = 0 populations, each (harmonic, element) by one rule. An element x at amplitude a_x reaches y = (m + Δm, n + Δn, r)
//! through L's entry c = |L_rx| of the component (Δm, Δn) with the first-order amplitude a_y = a_x c / max(|λ₀_y|, c),
//! λ₀_y = i f(m + Δm, n + Δn) + L_rr(0) (the entry's own resolvent, capped at one: a coupling stronger than its detuning
//! carries the whole amplitude); y is admitted iff a_y ≥ ε, ε the format's precision (a_y² ≥ ε², the prototype's form).
//! Only L propagates: the system solved is L's (ρ alone); the force's K1 reads ρ.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use super::{Element, Elements, Liouville};
use crate::linalg::C64;

/// An amplitude ordered for the heap.
#[derive(Clone, Copy, PartialEq)]
struct Amplitude(f64);
impl Eq for Amplitude {}
impl PartialOrd for Amplitude {
    fn partial_cmp(&self, o: &Amplitude) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Amplitude {
    fn cmp(&self, o: &Amplitude) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// The admitted elements of a Liouvillian, in the order the rule took them (the largest amplitude first, ties by the
/// smallest element).
pub fn select(parts: &Liouville) -> Elements {
    let n = parts.states;
    let d = n * n;
    let rule = f64::EPSILON * f64::EPSILON;
    // Per column x: (component, row, |L_rx|).
    let keys: Vec<&(Vec<i64>, [i64; 3])> = parts.components.keys().collect();
    let mut columns: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); d];
    for (k, key) in keys.iter().enumerate() {
        let Some(l) = parts.components[*key].l.as_ref() else { continue };
        for r in 0..l.rows {
            for p in l.indptr[r]..l.indptr[r + 1] {
                let c = l.values[p].abs();
                if c != 0.0 {
                    columns[l.indices[p]].push((k, r, c));
                }
            }
        }
    }
    let m0 = vec![0i64; parts.lattice.slow.len()];
    let mut amplitude: HashMap<Element, f64> = HashMap::new();
    let mut heap: BinaryHeap<(Amplitude, Reverse<Element>)> = BinaryHeap::new();
    for a in 0..n {
        let y = Element { m: m0.clone(), n: [0; 3], x: a * n + a };
        amplitude.insert(y.clone(), 1.0);
        heap.push((Amplitude(1.0), Reverse(y)));
    }
    let mut done: HashMap<Element, usize> = HashMap::new();
    let mut out = Elements::default();
    while let Some((Amplitude(ax), Reverse(x))) = heap.pop() {
        if done.contains_key(&x) {
            continue;
        }
        done.insert(x.clone(), out.order.len());
        for &(k, r, c) in &columns[x.x] {
            let (dm, dn) = keys[k];
            let m: Vec<i64> = x.m.iter().zip(dm).map(|(a, b)| a + b).collect();
            let nn: [i64; 3] = std::array::from_fn(|i| x.n[i] + dn[i]);
            let lambda = (C64::new(0.0, parts.lattice.frequency(&m, &nn)) + parts.diagonal[r]).abs();
            let ay = ax * c / lambda.max(c);
            if ay * ay < rule {
                continue;
            }
            let y = Element { m, n: nn, x: r };
            if !done.contains_key(&y) && amplitude.get(&y).is_none_or(|&a| ay > a) {
                amplitude.insert(y.clone(), ay);
                heap.push((Amplitude(ay), Reverse(y)));
            }
        }
        out.order.push(x);
        out.amplitudes.push(ax);
    }
    out
}
