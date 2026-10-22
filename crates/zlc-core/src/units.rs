//! What a number means and the one way it is shown: `zlc_data.units`.
//!
//! The registry holds base units, one per dimension, plus the few that are
//! their own family; every prefixed spelling is derived on resolution.
//! Showing a number shifts its decimal point and never multiplies a float, so
//! what is displayed is what is held, digit for digit.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, LazyLock};

use crate::decimal::Decimal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitError(pub String);

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}

impl std::error::Error for UnitError {}

fn error<T>(message: impl Into<String>) -> Result<T, UnitError> { Err(UnitError(message.into())) }

/// Python's `repr` of a text, as error messages quote spellings.
fn quoted(text: &str) -> String { format!("'{text}'") }

/// One decimal step: the spelling shown and the spellings accepted.
#[derive(Debug, PartialEq, Eq)]
pub struct Prefix {
    pub symbol: &'static str,
    pub exponent: i32,
    pub accepts: &'static [&'static str],
}

impl Prefix {
    pub fn spellings(&self) -> impl Iterator<Item = &'static str> + '_ {
        std::iter::once(self.symbol).chain(self.accepts.iter().copied())
    }
}

/// Every prefix some unit's ladder uses, largest first.
pub static PREFIXES: [Prefix; 7] = [
    Prefix { symbol: "G", exponent: 9, accepts: &[] },
    Prefix { symbol: "M", exponent: 6, accepts: &[] },
    Prefix { symbol: "k", exponent: 3, accepts: &[] },
    Prefix { symbol: "", exponent: 0, accepts: &[] },
    Prefix { symbol: "m", exponent: -3, accepts: &[] },
    Prefix { symbol: "µ", exponent: -6, accepts: &["u", "μ"] },
    Prefix { symbol: "n", exponent: -9, accepts: &[] },
];

/// The identity step.
pub static NO_PREFIX: &Prefix = &PREFIXES[3];
const PREFIX_STEP: i64 = 3;

fn prefix_by_symbol(symbol: &str) -> Option<&'static Prefix> {
    PREFIXES.iter().find(|prefix| !prefix.symbol.is_empty() && prefix.symbol == symbol)
}

fn prefix_by_exponent(exponent: i64) -> Option<&'static Prefix> {
    PREFIXES.iter().find(|prefix| prefix.exponent as i64 == exponent)
}

/// The non-empty spellings in table order (`_PREFIX_BY_SPELLING`).
fn prefix_spellings() -> impl Iterator<Item = (&'static str, &'static Prefix)> {
    PREFIXES.iter().flat_map(|prefix| prefix.spellings().filter(|s| !s.is_empty()).map(move |s| (s, prefix)))
}

fn prefix_by_spelling(spelling: &str) -> Option<&'static Prefix> {
    prefix_spellings().find(|(s, _)| *s == spelling).map(|(_, prefix)| prefix)
}

/// `10.0 ** n` as Python computes it (C `pow`).
fn ten_to(exponent: i64) -> f64 { 10f64.powf(exponent as f64) }

/// How a unit's numbers map onto its dimension's base unit.
#[derive(Clone, Debug, PartialEq)]
pub enum Conversion {
    /// `base = value * factor`.
    Scaled(f64),
    /// `base = value + offset`.
    Offset(f64),
    /// `base = reference * 10 ** (value / factor)`.
    Decibel { reference: f64, factor: f64 },
    /// `base = value ** 2 / (divisor * load)`; negative amplitudes have no power.
    VoltageIntoLoad { load_ohms: f64, divisor: f64 },
    /// A prefix in front of a non-linear unit.
    Prefixed { inner: Box<Conversion>, factor: f64 },
}

impl Conversion {
    pub fn scaled(factor: f64) -> Result<Self, UnitError> {
        if !factor.is_finite() || factor == 0.0 { return error("unit factor must be finite and non-zero"); }
        Ok(Conversion::Scaled(factor))
    }

    pub fn to_base(&self, value: f64) -> f64 {
        match self {
            Conversion::Scaled(factor) => value * factor,
            Conversion::Offset(offset) => value + offset,
            Conversion::Decibel { reference, factor } => reference * 10f64.powf(value / factor),
            Conversion::VoltageIntoLoad { load_ohms, divisor } => {
                (if value < 0.0 { f64::NAN } else { value * value }) / (divisor * load_ohms)
            }
            Conversion::Prefixed { inner, factor } => inner.to_base(value * factor),
        }
    }

    pub fn from_base(&self, value: f64) -> f64 {
        match self {
            Conversion::Scaled(factor) => value / factor,
            Conversion::Offset(offset) => value - offset,
            Conversion::Decibel { reference, factor } => factor * (value / reference).log10(),
            Conversion::VoltageIntoLoad { load_ohms, divisor } => (divisor * load_ohms * value).sqrt(),
            Conversion::Prefixed { inner, factor } => inner.from_base(value) / factor,
        }
    }
}

/// The whole power of ten a conversion is, when it is exactly one.
fn exact_decade(conversion: &Conversion) -> Option<i32> {
    let Conversion::Scaled(factor) = conversion else { return None };
    if *factor <= 0.0 { return None; }
    let candidate = factor.log10().round_ties_even();
    (ten_to(candidate as i64) == *factor).then_some(candidate as i32)
}

/// One way of writing a quantity of one dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub symbol: String,
    pub dimension: String,
    pub conversion: Conversion,
    pub prefixes: Vec<&'static str>,
    pub aliases: Vec<String>,
    pub inverse_dimension: Option<String>,
    pub decade: Option<i32>,
}

fn trimmed(text: &str) -> bool { !text.is_empty() && text.trim() == text }

impl Unit {
    pub fn new(symbol: &str, dimension: &str, conversion: Conversion, prefixes: &[&str], aliases: &[&str],
               inverse_dimension: Option<&str>) -> Result<Self, UnitError> {
        if !trimmed(symbol) { return error("unit symbol must be a non-empty, trimmed string"); }
        if !trimmed(dimension) { return error("unit dimension must be a non-empty, trimmed string"); }
        let mut resolved_prefixes = Vec::new();
        for prefix in prefixes {
            match prefix_by_symbol(prefix) {
                Some(found) => {
                    if resolved_prefixes.contains(&found.symbol) { return error(format!("unit {} names a prefix twice", quoted(symbol))); }
                    resolved_prefixes.push(found.symbol);
                }
                None => return error(format!("unit {} names prefixes the table lacks", quoted(symbol))),
            }
        }
        if aliases.iter().any(|alias| !trimmed(alias)) { return error("unit aliases must be non-empty, trimmed strings"); }
        let mut names: Vec<&str> = vec![symbol];
        for alias in aliases {
            if names.contains(alias) { return error(format!("unit {} contains duplicate names", quoted(symbol))); }
            names.push(alias);
        }
        let decade = exact_decade(&conversion);
        let nonlinear = matches!(conversion, Conversion::Offset(_) | Conversion::Decibel { .. } | Conversion::Prefixed { .. });
        if !resolved_prefixes.is_empty() && (nonlinear || decade.is_some_and(|d| d != 0 && d as i64 % PREFIX_STEP == 0)) {
            return error(format!("this unit cannot take a prefix: {}", quoted(symbol)));
        }
        if let Some(inverse) = inverse_dimension {
            if !trimmed(inverse) { return error("unit inverse_dimension must be a non-empty, trimmed string or None"); }
        }
        Ok(Unit {
            symbol: symbol.to_owned(), dimension: dimension.to_owned(), conversion, prefixes: resolved_prefixes,
            aliases: aliases.iter().map(|a| (*a).to_owned()).collect(), inverse_dimension: inverse_dimension.map(str::to_owned), decade,
        })
    }

    /// The rungs this unit is shown and read in, largest first.
    pub fn ladder(&self) -> impl Iterator<Item = &'static Prefix> + '_ {
        PREFIXES.iter().filter(|prefix| prefix.exponent == 0 || self.prefixes.contains(&prefix.symbol))
    }

    pub fn in_ladder(&self, prefix: &Prefix) -> bool { self.ladder().any(|rung| std::ptr::eq(rung, prefix)) }

    pub fn is_linear(&self) -> bool { matches!(self.conversion, Conversion::Scaled(_)) }

    /// Group and signed-number scale, distinct from conversion to base.
    pub fn coordinate_scale(&self) -> Option<(String, f64)> {
        let mut conversion = &self.conversion;
        let mut factor = 1.0;
        while let Conversion::Prefixed { inner, factor: step } = conversion {
            factor *= step;
            conversion = inner;
        }
        match conversion {
            Conversion::Scaled(scale) => Some((self.dimension.clone(), factor * scale)),
            Conversion::Offset(_) => Some((self.dimension.clone(), factor)),
            Conversion::VoltageIntoLoad { load_ohms, divisor } => {
                Some((format!("coordinate:{}:voltage", self.dimension), factor / (divisor * load_ohms).sqrt()))
            }
            _ => None,
        }
    }

    /// How many base units one of this unit is; only linear units have one.
    pub fn scale(&self) -> Result<f64, UnitError> {
        match self.conversion {
            Conversion::Scaled(factor) => Ok(factor),
            _ => error(format!("{} is not a linear unit and has no scale", quoted(&self.symbol))),
        }
    }

    pub fn is_base(&self) -> bool { self.conversion == Conversion::Scaled(1.0) }

    pub fn compatible_with(&self, other: &Unit) -> bool { self.dimension == other.dimension }

    pub fn to_base(&self, value: f64) -> f64 { self.conversion.to_base(value) }
    pub fn from_base(&self, value: f64) -> f64 { self.conversion.from_base(value) }

    /// One value in this unit, in `target`.
    pub fn convert_value_to(&self, value: f64, target: &Unit) -> Result<f64, UnitError> {
        if !self.compatible_with(target) {
            return error(format!("incompatible units: {} ({}) and {} ({})", quoted(&self.symbol), self.dimension,
                                 quoted(&target.symbol), target.dimension));
        }
        if self == target { return Ok(value); }
        Ok(target.from_base(self.to_base(value)))
    }

    fn display_order(&self) -> f64 {
        match self.conversion { Conversion::Scaled(scale) => scale.log10(), _ => f64::NEG_INFINITY }
    }
}

/// The unit that is `prefix` applied to `base`.
fn prefixed(base: &Arc<Unit>, prefix: &'static Prefix) -> Arc<Unit> {
    if prefix.exponent == 0 { return base.clone(); }
    let factor = ten_to(prefix.exponent as i64);
    let conversion = match (&base.conversion, base.decade) {
        (Conversion::Scaled(_), Some(decade)) => Conversion::Scaled(ten_to(decade as i64 + prefix.exponent as i64)),
        (Conversion::Scaled(scale), None) => Conversion::Scaled(scale * factor),
        (other, _) => Conversion::Prefixed { inner: Box::new(other.clone()), factor },
    };
    let mut aliases = Vec::new();
    for spelling in prefix.spellings() {
        for name in std::iter::once(&base.symbol).chain(&base.aliases) {
            if spelling != prefix.symbol || *name != base.symbol { aliases.push(format!("{spelling}{name}")); }
        }
    }
    Arc::new(Unit {
        symbol: format!("{}{}", prefix.symbol, base.symbol), dimension: base.dimension.clone(), decade: exact_decade(&conversion),
        conversion, prefixes: Vec::new(), aliases, inverse_dimension: base.inverse_dimension.clone(),
    })
}

/// The registered base units and every prefixed spelling they imply.
#[derive(Clone, Default)]
pub struct UnitRegistry {
    order: Vec<String>,
    units: HashMap<String, Arc<Unit>>,
    base_units: HashMap<String, Arc<Unit>>,
    revision: u64,
}

impl UnitRegistry {
    pub fn new(units: impl IntoIterator<Item = Unit>) -> Result<Self, UnitError> {
        let mut registry = UnitRegistry::default();
        for unit in units { registry.register(unit, false)?; }
        Ok(registry)
    }

    pub fn revision(&self) -> u64 { self.revision }

    pub fn register(&mut self, unit: Unit, replace: bool) -> Result<Arc<Unit>, UnitError> {
        let unit = Arc::new(unit);
        let names: Vec<String> = std::iter::once(unit.symbol.clone()).chain(unit.aliases.iter().cloned()).collect();
        let mut collisions: Vec<String> = names.iter().filter(|name| self.units.contains_key(*name)).cloned().collect();
        if !collisions.is_empty() && !replace {
            return error(format!("unit name already registered: {}", quoted(&collisions[0])));
        }
        if unit.is_base() {
            if let Some(existing) = self.base_units.get(&unit.dimension).filter(|existing| ***existing != *unit).cloned() {
                if !replace {
                    return error(format!("dimension {} already has base unit {}", quoted(&unit.dimension), quoted(&existing.symbol)));
                }
                collisions.extend(self.order.iter().filter(|name| *self.units[*name] == *existing).cloned());
            }
        }
        if replace && !collisions.is_empty() {
            let displaced: Vec<Arc<Unit>> = collisions.iter().filter_map(|name| self.units.get(name).cloned()).collect();
            let units = &mut self.units;
            self.order.retain(|name| {
                let keep = !displaced.iter().any(|gone| **gone == *units[name]);
                if !keep { units.remove(name); }
                keep
            });
            self.base_units.retain(|_, base| !displaced.iter().any(|gone| **gone == **base));
        }
        for name in names {
            if self.units.insert(name.clone(), unit.clone()).is_none() { self.order.push(name); }
        }
        if unit.is_base() { self.base_units.insert(unit.dimension.clone(), unit.clone()); }
        self.revision += 1;
        Ok(unit)
    }

    fn registered(&self, name: &str) -> Option<&Arc<Unit>> { self.units.get(name) }

    fn split_prefix(&self, text: &str) -> Option<(Arc<Unit>, &'static Prefix)> {
        for (spelling, prefix) in prefix_spellings() {
            let Some(rest) = text.strip_prefix(spelling) else { continue };
            if let Some(base) = self.units.get(rest) {
                if base.in_ladder(prefix) { return Some((base.clone(), prefix)); }
            }
        }
        None
    }

    /// One unit, by any spelling this registry accepts.
    pub fn resolve(&self, unit: &str) -> Result<Arc<Unit>, UnitError> {
        if let Some(found) = self.registered(unit) { return Ok(found.clone()); }
        if let Some((base, prefix)) = self.split_prefix(unit) { return Ok(prefixed(&base, prefix)); }
        if unit.contains('*') || unit.contains('^') { return self.compound(unit); }
        error(format!("unknown unit {}", quoted(unit)))
    }

    /// A product of powers of resolvable units, `a*b^2*c^-1`.
    fn compound(&self, unit: &str) -> Result<Arc<Unit>, UnitError> {
        let mut dimensions: BTreeMap<String, Decimal> = BTreeMap::new();
        let (mut symbols, mut scales) = (Vec::new(), Vec::new());
        for part in unit.split('*') {
            let (spelling, raw_exponent) = match part.split_once('^') { Some((s, e)) => (s, Some(e)), None => (part, None) };
            let exponent = match raw_exponent {
                Some(raw) => Decimal::parse(raw).ok_or_else(|| UnitError(format!("invalid unit exponent {}", quoted(raw))))?,
                None => Decimal::from_i128(1),
            };
            let exponent_value = exponent.to_f64();
            if !exponent_value.is_finite() { return error("unit exponent must be finite"); }
            let factor = self.resolve(spelling)?;
            let (dimension, scale) = match factor.coordinate_scale() {
                Some(found) => found,
                None => {
                    let (family, prefix) = self.family_and_prefix(&factor)?;
                    (format!("coordinate:{}", family.symbol), ten_to(prefix.exponent as i64))
                }
            };
            if dimension != "dimensionless" {
                let entry = dimensions.entry(dimension).or_insert_with(|| Decimal::from_i128(0));
                *entry = entry.add(&exponent);
            }
            let powered = scale.powf(exponent_value);
            if !powered.is_finite() {
                return error("unit exponent produces an invalid scale");
            }
            scales.push(powered);
            symbols.push(if exponent.equals(&Decimal::from_i128(1)) { factor.symbol.clone() }
                         else { format!("{}^{}", factor.symbol, exponent.normalize().to_sci_string()) });
        }
        let one = Decimal::from_i128(1);
        let dimension = dimensions.iter().filter(|(_, power)| !power.is_zero())
            .map(|(name, power)| if power.equals(&one) { name.clone() } else { format!("{name}^{}", power.normalize().to_sci_string()) })
            .collect::<Vec<_>>().join("*");
        let product = scales.iter().fold(1.0, |acc, scale| acc * scale);
        Ok(Arc::new(Unit::new(&symbols.join("*"), if dimension.is_empty() { "dimensionless" } else { &dimension },
                              Conversion::scaled(product)?, &[], &[], None)?))
    }

    /// The base unit of a unit's dimension.
    pub fn base_for_unit(&self, unit: &Unit) -> Result<Arc<Unit>, UnitError> {
        if unit.is_base() { return Ok(Arc::new(unit.clone())); }
        self.base_for_dimension(&unit.dimension)
    }

    /// The base unit of a dimension, or of the dimension of a spelling.
    pub fn base_for(&self, unit_or_dimension: &str) -> Result<Arc<Unit>, UnitError> {
        let dimension = if self.base_units.contains_key(unit_or_dimension) { unit_or_dimension.to_owned() }
                        else { self.resolve(unit_or_dimension)?.dimension.clone() };
        self.base_for_dimension(&dimension)
    }

    fn base_for_dimension(&self, dimension: &str) -> Result<Arc<Unit>, UnitError> {
        self.base_units.get(dimension).cloned().ok_or_else(|| UnitError(format!("dimension {} has no base unit", quoted(dimension))))
    }

    /// The registered unit this spelling is a prefixed form of, or itself.
    pub fn family_of(&self, unit: &Unit) -> Result<Arc<Unit>, UnitError> { Ok(self.family_and_prefix(unit)?.0) }

    /// The family a unit belongs to and the prefix it carries.
    pub fn family_and_prefix(&self, unit: &Unit) -> Result<(Arc<Unit>, &'static Prefix), UnitError> {
        if let Some(registered) = self.registered(&unit.symbol) { return Ok((registered.clone(), NO_PREFIX)); }
        Ok(match self.split_prefix(&unit.symbol) {
            Some((base, prefix)) => (base, prefix),
            None => (Arc::new(unit.clone()), NO_PREFIX),
        })
    }

    pub fn convert(&self, value: f64, source: &str, target: &str) -> Result<f64, UnitError> {
        self.resolve(source)?.convert_value_to(value, &*self.resolve(target)?)
    }

    /// Convert one authored number without a binary round trip for prefixes.
    pub fn convert_decimal(&self, number: &Decimal, source: &Unit, target: &Unit) -> Result<Decimal, UnitError> {
        if !source.compatible_with(target) {
            return error(format!("incompatible units: {} and {}", quoted(&source.symbol), quoted(&target.symbol)));
        }
        let (own_family, own_prefix) = self.family_and_prefix(source)?;
        let (shown_family, shown_prefix) = self.family_and_prefix(target)?;
        if own_family == shown_family { return Ok(number.scaleb((own_prefix.exponent - shown_prefix.exponent) as i64)); }
        if let (Some(a), Some(b)) = (source.decade, target.decade) { return Ok(number.scaleb((a - b) as i64)); }
        let converted = source.convert_value_to(number.to_f64(), target)?;
        Decimal::from_repr(converted).ok_or_else(|| UnitError("unit conversion produced a non-finite value".into()))
    }

    /// The inverse-dimension unit with reciprocal scale, or None.
    pub fn inverse_for(&self, unit: &Unit) -> Option<Arc<Unit>> {
        let dimension = unit.inverse_dimension.as_deref()?;
        let Conversion::Scaled(scale) = unit.conversion else { return None };
        let base = self.base_for_dimension(dimension).ok()?;
        let wanted = 1.0 / scale;
        let exponent = (wanted / base.scale().ok()?).log10().round_ties_even();
        let prefix = prefix_by_exponent(exponent as i64)?;
        if !base.in_ladder(prefix) { return None; }
        let candidate = prefixed(&base, prefix);
        is_close(candidate.scale().ok()?, wanted, 1e-12).then_some(candidate)
    }

    /// One symbol per registered unit, sorted.
    pub fn distinct_symbols(&self) -> Vec<String> {
        let mut symbols: Vec<String> = self.units.values().map(|unit| unit.symbol.clone()).collect();
        symbols.sort();
        symbols.dedup();
        symbols
    }

    /// Registered units in registration order, one entry per unit.
    fn distinct_units(&self) -> Vec<Arc<Unit>> {
        let mut seen: Vec<Arc<Unit>> = Vec::new();
        for name in &self.order {
            let unit = &self.units[name];
            if let Some(slot) = seen.iter_mut().find(|known| known.symbol == unit.symbol) { *slot = unit.clone(); } else { seen.push(unit.clone()); }
        }
        seen
    }

    /// Every unit this one may be shown in, largest first.
    pub fn display_choices(&self, unit: &Unit) -> Result<Vec<String>, UnitError> {
        let mut choices: Vec<String> = Vec::new();
        for candidate in self.distinct_units().iter().filter(|candidate| candidate.dimension == unit.dimension) {
            for prefix in candidate.ladder() {
                let symbol = prefixed(candidate, prefix).symbol.clone();
                if !choices.contains(&symbol) { choices.push(symbol); }
            }
        }
        if choices.is_empty() { choices.push(unit.symbol.clone()); }
        let mut keyed = Vec::with_capacity(choices.len());
        for symbol in choices { keyed.push((-self.resolve(&symbol)?.display_order(), symbol)); }
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(keyed.into_iter().map(|(_, symbol)| symbol).collect())
    }

    /// Every spelling `unit` may be read in, grouped under the family each
    /// belongs to, the unit's own family first (zlc_ui `unit_choice_tree`: W
    /// with its prefixes, dBm on its own, Vpp with its prefixes). Empty when
    /// there is nothing to choose: an unknown or dimensionless unit, or one
    /// whose ladder is only itself.
    pub fn choice_tree(&self, unit: &str) -> Vec<(String, Vec<String>)> {
        let symbol = unit.trim();
        if symbol.is_empty() || symbol == "1" { return Vec::new(); }
        let family = |spelling: &str| self.resolve(spelling).and_then(|unit| self.family_of(&unit)).map(|family| family.symbol.clone());
        let (Ok(resolved), Ok(own)) = (self.resolve(symbol), family(symbol)) else { return Vec::new() };
        let choices = self.display_choices(&resolved).unwrap_or_default();
        if choices.len() < 2 { return Vec::new(); }
        let mut tree: Vec<(String, Vec<String>)> = vec![(own, Vec::new())];
        for spelling in choices {
            let Ok(trunk) = family(&spelling) else { continue };
            match tree.iter_mut().find(|(name, _)| *name == trunk) {
                Some((_, leaves)) => leaves.push(spelling),
                None => tree.push((trunk, vec![spelling])),
            }
        }
        tree.retain(|(_, leaves)| !leaves.is_empty());
        tree
    }
}

/// `math.isclose(a, b, rel_tol=rel)` with no absolute tolerance.
fn is_close(a: f64, b: f64, rel: f64) -> bool {
    a == b || ((a - b).abs() <= (rel * b.abs()).max(rel * a.abs()) && a.is_finite() && b.is_finite())
}

/// The load an RF amplitude is read into, in ohms.
pub const RF_LOAD_OHMS: f64 = 50.0;

fn builtin_units() -> Vec<Unit> {
    let scaled = |factor| Conversion::Scaled(factor);
    let unit = |symbol: &str, dimension: &str, conversion, prefixes: &[&str], aliases: &[&str], inverse: Option<&str>| {
        Unit::new(symbol, dimension, conversion, prefixes, aliases, inverse).expect("builtin unit")
    };
    vec![
        unit("1", "dimensionless", scaled(1.0), &[], &["arb"], None),
        unit("s", "time", scaled(1.0), &["m", "µ", "n"], &[], Some("frequency")),
        unit("m", "length", scaled(1.0), &["k", "m", "µ", "n"], &[], None),
        unit("Hz", "frequency", scaled(1.0), &["k", "M", "G"], &[], Some("time")),
        unit("V", "voltage", scaled(1.0), &["m", "µ"], &[], None),
        unit("A", "current", scaled(1.0), &["m", "µ"], &[], None),
        unit("W", "power", scaled(1.0), &["m", "µ", "n"], &[], None),
        unit("K", "temperature", scaled(1.0), &["m", "µ", "n"], &[], None),
        unit("°C", "temperature", Conversion::Offset(273.15), &[], &["degC"], None),
        unit("T", "magnetic_flux_density", scaled(1.0), &["m", "µ", "n"], &[], None),
        unit("G", "magnetic_flux_density", scaled(1e-4), &["m"], &[], None),
        unit("rad", "angle", scaled(1.0), &["m"], &[], None),
        unit("°", "angle", scaled(std::f64::consts::PI / 180.0), &[], &["deg"], None),
        unit("dBm", "power", Conversion::Decibel { reference: 1.0e-3, factor: 10.0 }, &[], &[], None),
        unit("dB", "log_gain", scaled(1.0), &[], &[], None),
        unit("Vpp", "power", Conversion::VoltageIntoLoad { load_ohms: RF_LOAD_OHMS, divisor: 8.0 }, &["m", "µ"], &[], None),
        unit("Vrms", "power", Conversion::VoltageIntoLoad { load_ohms: RF_LOAD_OHMS, divisor: 1.0 }, &["m", "µ"], &[], None),
        unit("count", "count", scaled(1.0), &[], &[], None),
        unit("point", "point", scaled(1.0), &[], &[], None),
        unit("code", "code", scaled(1.0), &[], &[], None),
        unit("pixel", "pixel", scaled(1.0), &[], &[], None),
    ]
}

/// The application's registry (`DEFAULT_UNITS`).
pub static DEFAULT_UNITS: LazyLock<UnitRegistry> = LazyLock::new(|| UnitRegistry::new(builtin_units()).expect("builtin registry"));

pub fn resolve_unit(unit: &str, registry: Option<&UnitRegistry>) -> Result<Arc<Unit>, UnitError> {
    registry.unwrap_or(&DEFAULT_UNITS).resolve(unit)
}

/// A number as Python holds it: an integer is read as itself, never a float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number { Int(i128), Float(f64) }

impl Number {
    /// `decimal_of`: the exact decimal the value is, or None when not finite.
    pub fn decimal(self) -> Option<Decimal> {
        match self { Number::Int(value) => Some(Decimal::from_i128(value)), Number::Float(value) => Decimal::from_repr(value) }
    }

    /// `str(value)`.
    pub fn text(self) -> String {
        match self { Number::Int(value) => value.to_string(), Number::Float(value) => crate::pyfmt::repr(value) }
    }
}

/// The one prefix that suits every value here.
pub fn prefix_for(values: &[Number], unit: &Unit) -> &'static Prefix {
    let ladder: Vec<&'static Prefix> = unit.ladder().collect();
    if ladder.len() == 1 { return NO_PREFIX; }
    let Some(largest) = values.iter().filter_map(|value| value.decimal()).filter(|d| !d.is_zero()).map(|d| d.adjusted()).max()
        else { return NO_PREFIX };
    let exponent = PREFIX_STEP * largest.div_euclid(PREFIX_STEP);
    ladder.iter().copied().find(|rung| rung.exponent as i64 <= exponent).unwrap_or(ladder[ladder.len() - 1])
}

/// The number with nothing after the point that says nothing.
fn plain_digits(value: &Decimal) -> String {
    let text = value.format_fixed();
    if !text.contains('.') { return text; }
    let stripped = text.trim_end_matches('0').trim_end_matches('.');
    if stripped.is_empty() { "0".into() } else { stripped.to_owned() }
}

/// `value` padded on the left with figure spaces to the width of `widest`.
pub fn figure_padded(value: &str, widest: &str) -> String {
    let width = widest.chars().count();
    let length = value.chars().count();
    format!("{}{value}", "\u{2007}".repeat(width.saturating_sub(length)))
}

/// One number and its unit, as a person should read them.
pub fn format_quantity(value: Number, unit: &Unit, prefix: Option<&'static Prefix>) -> Result<String, UnitError> {
    let Some(decimal) = value.decimal() else { return Ok(format!("{} {}", value.text(), unit.symbol).trim().to_owned()) };
    let step = prefix.unwrap_or_else(|| prefix_for(&[value], unit));
    if !unit.in_ladder(step) { return error(format!("{} cannot take the prefix {}", quoted(&unit.symbol), quoted(step.symbol))); }
    let shifted = decimal.scaleb(-step.exponent as i64);
    let symbol = if unit.symbol != "1" { format!("{}{}", step.symbol, unit.symbol) } else { step.symbol.to_owned() };
    Ok(format!("{} {symbol}", plain_digits(&shifted)).trim().to_owned())
}

/// Split `text` into the number and unit spellings `_QUANTITY` matches.
fn split_quantity(text: &str) -> Option<(&str, &str)> {
    let body = text.trim_start();
    let bytes = body.as_bytes();
    let mut at = 0;
    if matches!(bytes.first(), Some(b'+' | b'-')) { at += 1; }
    let integer_start = at;
    while at < bytes.len() && bytes[at].is_ascii_digit() { at += 1; }
    let integer_digits = at - integer_start;
    if integer_digits > 0 {
        if at < bytes.len() && bytes[at] == b'.' {
            at += 1;
            while at < bytes.len() && bytes[at].is_ascii_digit() { at += 1; }
        }
    } else {
        if at >= bytes.len() || bytes[at] != b'.' { return None; }
        at += 1;
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() { at += 1; }
        if at == start { return None; }
    }
    if at < bytes.len() && matches!(bytes[at], b'e' | b'E') {
        let mut probe = at + 1;
        if probe < bytes.len() && matches!(bytes[probe], b'+' | b'-') { probe += 1; }
        let start = probe;
        while probe < bytes.len() && bytes[probe].is_ascii_digit() { probe += 1; }
        if probe > start { at = probe; }
    }
    let number = &body[..at];
    let rest = body[at..].trim_start();
    let unit_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let (written, tail) = rest.split_at(unit_end);
    tail.trim().is_empty().then_some((number, written))
}

/// Read a typed quantity back, in `unit`.
pub fn parse_quantity(text: &str, unit: &Unit, registry: Option<&UnitRegistry>) -> Result<f64, UnitError> {
    let units = registry.unwrap_or(&DEFAULT_UNITS);
    let Some((number, written)) = split_quantity(text) else {
        return error(format!("not a number with an optional unit: {}", quoted(text.trim())));
    };
    if written.is_empty() { return Ok(number.parse().expect("matched number")); }
    let (family, carried) = units.family_and_prefix(unit)?;
    let known = units.resolve(written).ok();
    let prefix = prefix_by_spelling(written);
    let spelled = match (&known, prefix) {
        (Some(found), _) if *units.family_of(found)? == *family => found.clone(),
        (_, Some(prefix)) if family.in_ladder(prefix) => prefixed(&family, prefix),
        (Some(found), _) => found.clone(),
        (None, None) => return error(format!("unknown unit or prefix {}", quoted(written))),
        (None, Some(_)) => return error(format!("{} cannot take the prefix {}", quoted(&unit.symbol), quoted(written))),
    };
    if !spelled.compatible_with(unit) {
        return error(format!("{} is {}, and this value is {}", quoted(written), spelled.dimension, unit.dimension));
    }
    let (spelled_family, spelled_prefix) = units.family_and_prefix(&spelled)?;
    if *spelled_family == *family {
        let decimal = Decimal::parse(number).expect("matched number");
        return Ok(decimal.scaleb((spelled_prefix.exponent - carried.exponent) as i64).to_f64());
    }
    spelled.convert_value_to(number.parse().expect("matched number"), unit)
}
