//! The population tolerance and the error metric every output is accepted
//! by.

/// The population tolerance: absolute, on the probability scale, over every
/// state and requested group. It is fixed; no other accuracy is served.
pub const POPULATION_TOLERANCE:f64=1e-2;
/// The share of the tolerance each a-priori truncation may take: what the
/// convergence check cannot see, because it compares the run's model with
/// itself. Each rule leaves out at most A_PRIORI_SHARE x tolerance: the
/// lattice (the translation gate and the cover), the fold's qualification
/// (a far step's micromotion), the run's fold (its stored rests and the far
/// shell's rings, half each), the conserved modes and the slowest
/// relaxation's candidates. A deliberate margin: every rule at its limit at
/// once still leaves two thirds of the tolerance to the convergence.
pub const A_PRIORI_SHARE:f64=0.1;
/// Largest population difference over the states and the requested groups.
pub fn metric(a:&[f64],b:&[f64],groups:&[Vec<usize>])->f64{
    let mut m=a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0_f64,f64::max);
    for g in groups{m=m.max(g.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
    m
}
