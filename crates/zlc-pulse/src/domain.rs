//! A field's value domain: the bounds its values keep and the grid they lie
//! on, in the field's own unit. One quantizer places a value on it, whoever
//! writes the field: the binding layer an edit, a scan plan each of its points.
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

    /// The grid point nearest a finite `value`.
    pub fn nearest(&self, value: f64) -> f64 {
        let (negative, steps) = self.steps(&Decimal::from_f64(value));
        Decimal { negative, mantissa: steps, exponent: 0 }.mul(&Decimal::from_f64(self.step)).to_f64()
    }
}

/// The values a field may take: `lo..=hi`, on `grid` (None: any value between).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain { pub lo: f64, pub hi: f64, pub grid: Option<Grid> }

impl Domain {
    pub fn holds(&self, value: f64) -> bool { self.lo <= value && value <= self.hi }

    /// A finite `value` as the field takes it: the nearest grid point, kept within the bounds.
    pub fn quantize(&self, value: f64) -> f64 {
        self.grid.map_or(value, |grid| grid.nearest(value)).clamp(self.lo, self.hi)
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
        let ends = Domain { lo: 0.5, hi: 9.5, grid: Some(Grid { step: 1.0, ties: Ties::Away }) };
        assert_eq!([0.6, 9.4, 4.5].map(|value| ends.quantize(value)), [1.0, 9.0, 5.0]);
        let off_grid = Domain { lo: 0.0, hi: 9.5, grid: Some(Grid { step: 2.0, ties: Ties::Away }) };
        assert_eq!(off_grid.quantize(9.4), 9.5, "a grid point past a bound is the bound");
        let continuous = Domain { lo: -1.0, hi: 1.0, grid: None };
        assert_eq!(continuous.quantize(0.123), 0.123);
        assert!(!continuous.holds(1.5) && continuous.holds(-1.0));
    }
}
