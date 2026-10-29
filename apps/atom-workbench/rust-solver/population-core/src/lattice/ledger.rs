//! The population tolerance and the error metric. What the fold leaves out
//! is no ledger entry: the truncation rule keeps or drops every piece by its
//! induced trace norm times the run's longest output time, and bounds what it
//! dropped on the reduced solution (`fold`, `pipeline`).

/// The population tolerance: absolute, on the probability scale, over every
/// state and requested group. It is fixed; no other accuracy is served.
pub const POPULATION_TOLERANCE:f64=5e-3;
/// Largest population difference over the states and the requested groups.
pub fn metric(a:&[f64],b:&[f64],groups:&[Vec<usize>])->f64{
    let mut m=a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0_f64,f64::max);
    for g in groups{m=m.max(g.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
    m
}
