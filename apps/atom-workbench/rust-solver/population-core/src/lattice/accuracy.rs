//! The population tolerance and the error metric every output is accepted
//! by.

/// The population tolerance: absolute, on the probability scale, over every
/// state and requested group. It is fixed; no other accuracy is served.
pub const POPULATION_TOLERANCE:f64=5e-3;
/// Largest population difference over the states and the requested groups.
pub fn metric(a:&[f64],b:&[f64],groups:&[Vec<usize>])->f64{
    let mut m=a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0_f64,f64::max);
    for g in groups{m=m.max(g.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
    m
}
