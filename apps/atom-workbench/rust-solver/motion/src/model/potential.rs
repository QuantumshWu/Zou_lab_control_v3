//! The trap's descent (SPEC §1.3): from the focus to the local minimum of a potential given by its second-order
//! expansion at a point, and the trap's centre, analytic Hessian and frequencies there. The one algorithm of every trap:
//! the trajectory model's prepared wells and its reported trap (solve.rs `Prepared`, `reported`) are each descended here.
//!
//! # The descent
//! The descent from the focus to the local minimum of V_L(r; p) -- the prepared lights' for the initial wells, the running ones'
//! for the reported trap (SPEC §1.3) --: Newton's step along each eigenvector of the analytic Hessian
//! curved upward, downhill by the trust radius along the others (along an eigenvector without slope at a top: as it is), all
//! within the trust radius, from the light's shortest length 1/max k_l up; doubled after a cut step that lowered V, halved
//! where V did not fall. It ends where a step's predicted gain is within the rounding of V's sum; the point is the trap's
//! centre where every curvature there is positive beyond that rounding over the light's shortest length (a bowl), and the
//! request is refused where it is not, or where the descent leaves the trap's region: no minimum holds the atom ("光镊不约束
//! 原子"), and the focus is never taken instead. A focus that is a bowl's bottom is the centre as it stands, bit for bit.

use super::{Quadratic, Trap};
use crate::linalg::{dot3, symmetric3_eigen};

/// The trap of a potential (module documentation, # The trap): the descent from the focus to its local minimum, on its
/// second-order expansion with the rounding of its value at a point (`expand`), from the light's shortest length `length`
/// (µm) up, within the trap's region (`outside`: the trap light's region left); ħ/m `hbar_over_mass` makes the
/// frequencies.
pub(crate) fn descend(length: f64, hbar_over_mass: f64, outside: &dyn Fn([f64; 3]) -> bool, expand: &dyn Fn([f64; 3]) -> (Quadratic, f64)) -> Result<Trap, String> {
    let mut r = [0.0; 3];
    let mut here = expand(r);
    let mut radius = length;
    loop {
        let (values, vectors) = symmetric3_eigen(&here.0.hessian);
        let along = vectors.map(|e| dot3(e, here.0.gradient));
        let noise = here.1;
        let bowl = values.iter().all(|&l| l * length * length > noise);
        let steps: [f64; 3] = std::array::from_fn(|i| {
            if values[i] * length * length > noise {
                -along[i] / values[i]
            } else if along[i] != 0.0 {
                -along[i].signum() * radius
            } else if values[i] * length * length < -noise {
                radius
            } else {
                0.0
            }
        });
        let size = steps.iter().map(|x| x * x).sum::<f64>().sqrt();
        loop {
            let scale = if size > radius { radius / size } else { 1.0 };
            let gain = -(0..3).map(|i| along[i] * steps[i] * scale + 0.5 * values[i] * (steps[i] * scale).powi(2)).sum::<f64>();
            if !(gain > noise) {
                if !bowl {
                    return Err("光镊不约束原子：原子的势能从焦点往下走，找不到局部极小".into());
                }
                let frequencies = values.map(|l| (l * hbar_over_mass).sqrt());
                let lab = std::array::from_fn(|a| (here.0.hessian[a][a] * hbar_over_mass).sqrt());
                return Ok(Trap { centre: r, minimum: here.0.value, axes: vectors, frequencies, lab });
            }
            let next: [f64; 3] = std::array::from_fn(|a| r[a] + (0..3).map(|i| steps[i] * scale * vectors[i][a]).sum::<f64>());
            if outside(next) {
                return Err("光镊不约束原子：原子的势能从焦点往下走，一直走出了光镊的范围".into());
            }
            let there = expand(next);
            if there.0.value < here.0.value {
                r = next;
                here = there;
                if scale < 1.0 {
                    radius *= 2.0;
                }
                break;
            }
            radius = crate::linalg::smaller(radius, size) * 0.5;
        }
    }
}
