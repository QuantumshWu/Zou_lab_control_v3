//! A field's value domain: the bounds its values keep and the grid they lie
//! on, in the field's own unit. One quantizer places a value on it, whoever
//! writes the field: the binding layer an edit, a scan plan each of its points.
use std::cmp::Ordering;

use crate::exact::{BigUint, Decimal, Ratio};

/// Where a value exactly halfway between two grid points goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ties {
    /// To the larger magnitude: the clock grid's `Fraction + 1/2` (`align_to_grid`).
    Away,
    /// To the even point: Python's `round`, as a DAC code is rounded.
    Even,
}

impl Ties {
    pub fn name(self) -> &'static str { match self { Ties::Away => "away", Ties::Even => "even" } }
    pub fn parse(text: &str) -> Option<Ties> { match text { "away" => Some(Ties::Away), "even" => Some(Ties::Even), _ => None } }
}

/// The whole multiples of `step`, the values a field can hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid { pub step: f64, pub ties: Ties }

/// A DAC code or a repeat count: a whole number, halves to even.
pub const WHOLE: Grid = Grid { step: 1.0, ties: Ties::Even };

impl Grid {
    /// The whole number of steps nearest `value` (`Fraction(str(step))` of them), and its sign.
    pub fn steps(&self, value: &Decimal) -> (bool, BigUint) {
        let ratio = Ratio::of(value, &Decimal::from_f64(self.step));
        (value.negative, match self.ties { Ties::Away => ratio.round_half_up(), Ties::Even => ratio.round_half_even() })
    }

    /// The grid point `steps` steps from zero, on the side `negative` says.
    fn point(&self, negative: bool, steps: BigUint) -> Decimal {
        Decimal { negative, mantissa: steps, exponent: 0 }.mul(&Decimal::from_f64(self.step))
    }

    /// The grid point nearest an exact `value`.
    pub fn nearest_to(&self, value: &Decimal) -> Decimal {
        let (negative, steps) = self.steps(value);
        self.point(negative, steps)
    }

    /// The grid point nearest `value`; a value that is not finite has none and stays as it is.
    pub fn nearest(&self, value: f64) -> f64 {
        if !value.is_finite() { return value; }
        self.nearest_to(&Decimal::from_f64(value)).to_f64()
    }

    /// The grid point at or above `value` (`up`), else at or below it.
    fn toward(&self, value: &Decimal, up: bool) -> Decimal {
        let ratio = Ratio::of(value, &Decimal::from_f64(self.step));
        let (whole, part) = ratio.numerator.divrem(&ratio.denominator);
        // Up from a positive value, or down from a negative one, is away from zero.
        let steps = if up != value.negative && !part.is_zero() { whole.add(&BigUint::from_u128(1)) } else { whole };
        self.point(value.negative, steps)
    }
}

/// Two exact values in order, signs included.
fn order(a: &Decimal, b: &Decimal) -> Ordering {
    match (a.negative && !a.is_zero(), b.negative && !b.is_zero()) {
        (false, true) => Ordering::Greater,
        (true, false) => Ordering::Less,
        (false, false) => a.cmp_abs(b),
        (true, true) => b.cmp_abs(a),
    }
}

/// `value` within `lo..=hi`.
fn within(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal {
    if order(&value, &lo) == Ordering::Less { lo } else if order(&value, &hi) == Ordering::Greater { hi } else { value }
}

/// The values a field may take: `lo..=hi`, on `grid` (None: any value between).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain { pub lo: f64, pub hi: f64, pub grid: Option<Grid> }

impl Domain {
    pub fn holds(&self, value: f64) -> bool { self.lo <= value && value <= self.hi }

    /// An exact `value` as the field takes it: the nearest grid point, kept within the bounds --
    /// where that point lies past one, the innermost grid point inside it (a domain whose
    /// bounds hold no grid point keeps the bound).
    pub fn place(&self, value: &Decimal) -> Decimal {
        let (lo, hi) = (Decimal::from_f64(self.lo), Decimal::from_f64(self.hi));
        let Some(grid) = self.grid else { return within(value.clone(), lo, hi) };
        let (first, last) = (grid.toward(&lo, true), grid.toward(&hi, false));
        let nearest = grid.nearest_to(value);
        if order(&first, &last) == Ordering::Greater { within(nearest, lo, hi) } else { within(nearest, first, last) }
    }

    /// `value` as the field takes it ([`Domain::place`]); a value that is not finite stays as it is.
    pub fn quantize(&self, value: f64) -> f64 {
        if !value.is_finite() { return value; }
        self.place(&Decimal::from_f64(value)).to_f64()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::binding::{pulse_field_value, replace_pulse_field, tick_in};
    use crate::json::Json;
    use crate::model::{AnalogStep, Binding, Bracket, FieldKind, FieldRef, Sequence};
    use crate::num::{Num, number_for};
    use crate::run::{describe_sequence, read_document};
    use crate::scan::api_parameter_columns_for;

    /// The imaging template with an API parameter of every kind: its three durations,
    /// a DAC code, an output delay and a bracket's repeat count.
    fn every_kind() -> Sequence {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pulses/imaging_template.json");
        let (_path, sequence) = read_document(&path).unwrap();
        sequence.with(|parts| {
            parts.periods[0].analog_steps.push(AnalogStep::new("da_dipole", "edge", Num::Int(0)).unwrap());
            parts.bindings.push(Binding::new(FieldRef::dac(&parts.periods[0].period_id, "da_dipole"), "value", false, "api", "").unwrap());
            parts.bindings.push(Binding::new(FieldRef::delay("cooling"), "ns", false, "api", "").unwrap());
            parts.brackets.push(Bracket::new("loop", Some("long_before"), Some("long_after"), 2).unwrap());
            parts.bindings.push(Binding::new(FieldRef::bracket_count("loop"), "count", false, "api", "").unwrap());
        }).unwrap()
    }

    /// Values across a column's grid in its own unit: whole steps, quarter steps and the halfway
    /// points between, each written as a decimal the way an operator types it.
    fn across(sequence: &Sequence, kind: FieldKind, unit: &str) -> Vec<f64> {
        match kind {
            FieldKind::Dac => (-14..=14).map(|quarter| f64::from(quarter) / 4.0).collect(),
            FieldKind::BracketCount => (4..=24).map(|quarter| f64::from(quarter) / 4.0).collect(),
            // Every quarter tick (5 ns of a 20 ns clock), written in the field's unit.
            _ => {
                let quarter = sequence.time_step_ns / 4.0;
                let exponent = crate::units::nanoseconds_exponent(unit).unwrap();
                let quarters = if kind == FieldKind::Delay { -21..=21 } else { 4..=21 };
                quarters.map(|count| format!("{}e{}", f64::from(count) * quarter, -exponent).parse().unwrap()).collect()
            }
        }
    }

    #[test]
    fn a_field_takes_from_a_scan_exactly_what_the_binding_layer_makes_of_it() {
        let sequence = every_kind();
        let columns = api_parameter_columns_for(&sequence).unwrap();
        let kinds: Vec<FieldKind> = sequence.api_bindings().iter().map(|binding| binding.kind()).collect();
        for kind in [FieldKind::Duration, FieldKind::Dac, FieldKind::Delay, FieldKind::BracketCount] { assert!(kinds.contains(&kind), "{kind:?}"); }
        for binding in sequence.api_bindings() {
            let column = columns.iter().find(|column| column.name == binding.field_id()).unwrap();
            let domain = Domain { lo: column.limit_lo, hi: column.limit_hi, grid: Some(column.grid) };
            let write = |value: f64| replace_pulse_field(&sequence, &binding.field_ref, number_for(value), &binding.unit, &column.name)
                .and_then(|written| pulse_field_value(&written, &binding.field_ref, &binding.unit));
            let values = across(&sequence, binding.kind(), &binding.unit);
            assert!(values.iter().any(|value| domain.quantize(*value) != *value), "{}: some values move", column.name);
            for value in values.into_iter().filter(|value| domain.holds(*value)) {
                let played = domain.quantize(value);
                // What a scan sends is on the grid already: the binding layer keeps it as it is.
                assert_eq!(write(played).unwrap(), played, "{} {value}", column.name);
                if binding.kind() == FieldKind::BracketCount {
                    // A count is never rounded there: a fraction is a program's error.
                    if value.fract() != 0.0 { assert!(write(value).unwrap_err().message.contains("must be an integer count"), "{value}"); }
                } else {
                    // A value as typed lands where the quantizer puts it.
                    assert_eq!(write(value).unwrap(), played, "{} {value}", column.name);
                }
            }
        }
    }

    #[test]
    fn each_field_states_its_grid_from_the_rules_the_binding_layer_keeps() {
        let sequence = every_kind();
        let description = describe_sequence(&sequence).unwrap();
        let tick = |unit: &str| tick_in(&sequence, unit).unwrap();
        for field in description.get("api_fields").unwrap().as_array().unwrap() {
            let grid = field.get("grid").unwrap();
            let (step, ties) = (grid.get("step").and_then(Json::as_f64).unwrap(), grid.get("ties").and_then(Json::as_str).unwrap());
            let unit = field.get("unit").and_then(Json::as_str).unwrap();
            let low = field.get("limit_lo").and_then(Json::as_f64).unwrap();
            match field.get("kind").and_then(Json::as_str).unwrap() {
                "duration" => assert_eq!((step, ties, low), (tick(unit), "away", tick(unit)), "a duration is at least one tick"),
                "delay" => assert_eq!((step, ties), (tick(unit), "away")),
                "dac" => assert_eq!((step, ties), (1.0, "even")),
                "bracket_count" => assert_eq!((step, ties, low), (1.0, "even", crate::model::MINIMUM_BRACKET_COUNT as f64)),
                other => panic!("an API field of kind {other}"),
            }
        }
    }

    #[test]
    fn halfway_goes_as_each_grid_says_and_a_value_stays_within_its_bounds() {
        let clock = Grid { step: 0.02, ties: Ties::Away };
        // 0.03 us is 1.5 ticks exactly, though 0.03 / 0.02 is not 1.5 in binary.
        assert_eq!([clock.nearest(0.03), clock.nearest(-0.03), clock.nearest(0.029), clock.nearest(0.0)], [0.04, -0.04, 0.02, 0.0]);
        assert_eq!([2.5, 3.5, -2.5, 0.4, -0.4].map(|value| WHOLE.nearest(value)), [2.0, 4.0, -2.0, 0.0, 0.0]);
        assert_eq!(WHOLE.nearest(-0.4).to_bits(), 0f64.to_bits(), "no negative zero");
        // A scan table's DAC column rounds through WHOLE as it rounded before (Python's round).
        for value in (-40..=40).map(|quarter| f64::from(quarter) / 4.0).chain([2.4999999999999996, 4503599627370495.5, -511.5]) {
            assert_eq!(WHOLE.nearest(value), value.round_ties_even() + 0.0, "{value}");
        }
        let ends = Domain { lo: 0.5, hi: 9.5, grid: Some(Grid { step: 1.0, ties: Ties::Away }) };
        assert_eq!([0.6, 9.4, 4.5, 9.5, 0.5].map(|value| ends.quantize(value)), [1.0, 9.0, 5.0, 9.0, 1.0],
                   "a grid point past a bound is the innermost one inside it");
        let negative = Domain { lo: -9.5, hi: -0.5, grid: Some(Grid { step: 2.0, ties: Ties::Away }) };
        assert_eq!([-9.4, -0.6, -5.0].map(|value| negative.quantize(value)), [-8.0, -2.0, -6.0]);
        let none = Domain { lo: 0.2, hi: 0.4, grid: Some(WHOLE) };
        assert_eq!(none.quantize(0.3), 0.2, "bounds that hold no grid point keep the bound");
        let continuous = Domain { lo: -1.0, hi: 1.0, grid: None };
        assert_eq!(continuous.quantize(0.123), 0.123);
        assert!(!continuous.holds(1.5) && continuous.holds(-1.0));
        assert!(WHOLE.nearest(f64::NAN).is_nan() && clock.nearest(f64::INFINITY) == f64::INFINITY, "not finite: no grid point");
        assert_eq!(WHOLE.nearest(f64::NAN) as i64, f64::NAN.round_ties_even() as i64, "a DAC write of NaN codes as it did");
    }
}
