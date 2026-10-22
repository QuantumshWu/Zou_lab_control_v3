//! How an axis divides the space between its ends (`zlc_plot._axis_scale`).
//! Every mapping between a position and a value is an interpolation in the
//! space the axis draws linearly: the value itself, its decimal exponent on a
//! log axis, or the sample index of a nonuniform scan.

use std::sync::Arc;

use crate::units::Unit;

#[derive(Clone, Debug, PartialEq)]
pub enum AxisScale {
    Linear,
    Log,
    /// A nonuniform coordinate lattice drawn at its sample index.
    Samples(Arc<[f64]>),
    /// A lattice authored in `canonical` and shown in `display`.
    Converted { mapping: Box<AxisScale>, canonical: Arc<Unit>, display: Arc<Unit> },
}

/// Interpolate and linearly extend the two endpoint segments.
fn piecewise(value: f64, coordinates: &[f64], positions: &[f64]) -> f64 {
    if coordinates.len() == 1 { return value - coordinates[0] + positions[0]; }
    let (coordinates, positions): (Vec<f64>, Vec<f64>) = if coordinates[coordinates.len() - 1] < coordinates[0] {
        (coordinates.iter().rev().copied().collect(), positions.iter().rev().copied().collect())
    } else { (coordinates.to_vec(), positions.to_vec()) };
    // searchsorted(side="right") - 1, clipped to a segment.
    let right = coordinates.partition_point(|&c| c <= value);
    let index = (right as isize - 1).clamp(0, coordinates.len() as isize - 2) as usize;
    let fraction = (value - coordinates[index]) / (coordinates[index + 1] - coordinates[index]);
    positions[index] + fraction * (positions[index + 1] - positions[index])
}

fn sample_positions(scale: &[f64]) -> Vec<f64> {
    let mut positions: Vec<f64> = (0..scale.len()).map(|i| i as f64).collect();
    if scale[scale.len() - 1] < scale[0] { positions.reverse(); }
    positions
}

impl AxisScale {
    /// The coordinate in which this axis is linear on screen.
    pub fn space(&self, value: f64) -> f64 {
        match self {
            AxisScale::Linear => value,
            AxisScale::Log => if value > 0.0 { value.log10() } else { f64::NEG_INFINITY },
            AxisScale::Samples(lattice) => piecewise(value, lattice, &sample_positions(lattice)),
            AxisScale::Converted { mapping, canonical, display } => {
                mapping.space(canonical.convert_value_to(value, display).unwrap_or(f64::NAN))
            }
        }
    }

    /// Undo [`AxisScale::space`].
    pub fn value(&self, position: f64) -> f64 {
        match self {
            AxisScale::Linear => position,
            AxisScale::Log => 10f64.powf(position),
            AxisScale::Samples(lattice) => piecewise(position, &sample_positions(lattice), lattice),
            AxisScale::Converted { mapping, canonical, display } => {
                display.convert_value_to(mapping.value(position), canonical).unwrap_or(f64::NAN)
            }
        }
    }

    /// The value `fraction` of the way from `low` to `high` on screen.
    pub fn interpolate(&self, low: f64, high: f64, fraction: f64) -> f64 {
        let (start, stop) = (self.space(low), self.space(high));
        self.value(start + fraction * (stop - start))
    }

    /// Where `value` sits between the limits, as a fraction of the box.
    pub fn fraction_of(&self, value: f64, low: f64, high: f64) -> f64 {
        let (start, stop) = (self.space(low), self.space(high));
        if stop == start { return 0.0; }
        (self.space(value) - start) / (stop - start)
    }

    /// `value` where the axis draws it: an end the scale cannot place lies at the low wall.
    pub fn placed(&self, value: f64, limits: (f64, f64)) -> f64 {
        if self.space(value).is_finite() { value } else { limits.0.min(limits.1) }
    }

    pub fn midpoint(&self, low: f64, high: f64) -> f64 { self.interpolate(low, high, 0.5) }
}
