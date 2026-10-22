//! The decimal arithmetic the tick lattice is laid out in: Python's `decimal`
//! with a 32-digit context and half-even rounding, for the operations it uses.
//! Every result is the exactly computed value rounded once, as the General
//! Decimal Arithmetic specification requires, so values and comparisons agree.

use std::cmp::Ordering;

/// Unsigned magnitude, little-endian base 2^32 limbs, no trailing zero limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Natural(Vec<u32>);

impl Natural {
    fn zero() -> Self { Natural(Vec::new()) }
    fn from_u128(mut value: u128) -> Self {
        let mut limbs = Vec::new();
        while value != 0 { limbs.push(value as u32); value >>= 32; }
        Natural(limbs)
    }
    fn is_zero(&self) -> bool { self.0.is_empty() }
    fn trim(mut self) -> Self { while self.0.last() == Some(&0) { self.0.pop(); } self }
    fn to_u128(&self) -> Option<u128> {
        if self.0.len() > 4 { return None; }
        Some(self.0.iter().rev().fold(0u128, |acc, &limb| (acc << 32) | limb as u128))
    }
    fn mul_small(&self, factor: u32) -> Self {
        let mut carry = 0u64;
        let mut limbs = Vec::with_capacity(self.0.len() + 1);
        for &limb in &self.0 {
            let product = limb as u64 * factor as u64 + carry;
            limbs.push(product as u32);
            carry = product >> 32;
        }
        if carry != 0 { limbs.push(carry as u32); }
        Natural(limbs).trim()
    }
    fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() { return Self::zero(); }
        let mut limbs = vec![0u32; self.0.len() + other.0.len()];
        for (i, &a) in self.0.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.0.iter().enumerate() {
                let sum = limbs[i + j] as u64 + a as u64 * b as u64 + carry;
                limbs[i + j] = sum as u32;
                carry = sum >> 32;
            }
            let mut k = i + other.0.len();
            while carry != 0 {
                let sum = limbs[k] as u64 + carry;
                limbs[k] = sum as u32;
                carry = sum >> 32;
                k += 1;
            }
        }
        Natural(limbs).trim()
    }
    fn add(&self, other: &Self) -> Self {
        let (long, short) = if self.0.len() >= other.0.len() { (self, other) } else { (other, self) };
        let mut limbs = Vec::with_capacity(long.0.len() + 1);
        let mut carry = 0u64;
        for i in 0..long.0.len() {
            let sum = long.0[i] as u64 + short.0.get(i).copied().unwrap_or(0) as u64 + carry;
            limbs.push(sum as u32);
            carry = sum >> 32;
        }
        if carry != 0 { limbs.push(carry as u32); }
        Natural(limbs)
    }
    /// self - other, requiring self >= other.
    fn sub(&self, other: &Self) -> Self {
        let mut limbs = Vec::with_capacity(self.0.len());
        let mut borrow = 0i64;
        for i in 0..self.0.len() {
            let mut difference = self.0[i] as i64 - other.0.get(i).copied().unwrap_or(0) as i64 - borrow;
            borrow = if difference < 0 { difference += 1 << 32; 1 } else { 0 };
            limbs.push(difference as u32);
        }
        debug_assert_eq!(borrow, 0);
        Natural(limbs).trim()
    }
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.len().cmp(&other.0.len()).then_with(|| {
            for i in (0..self.0.len()).rev() {
                match self.0[i].cmp(&other.0[i]) { Ordering::Equal => continue, order => return order }
            }
            Ordering::Equal
        })
    }
    fn divmod_small(&self, divisor: u64) -> (Self, u64) {
        let mut limbs = vec![0u32; self.0.len()];
        let mut remainder = 0u128;
        for i in (0..self.0.len()).rev() {
            let current = (remainder << 32) | self.0[i] as u128;
            limbs[i] = (current / divisor as u128) as u32;
            remainder = current % divisor as u128;
        }
        (Natural(limbs).trim(), remainder as u64)
    }
    fn pow10(exponent: u32) -> Self {
        let mut value = Natural::from_u128(1);
        let mut left = exponent;
        while left >= 9 { value = value.mul_small(1_000_000_000); left -= 9; }
        for _ in 0..left { value = value.mul_small(10); }
        value
    }
    fn mul_pow10(&self, exponent: u32) -> Self { if exponent == 0 { self.clone() } else { self.mul(&Natural::pow10(exponent)) } }
    /// Quotient and remainder by 10^exponent.
    fn divmod_pow10(&self, exponent: u32) -> (Self, Self) {
        let divisor = Natural::pow10(exponent);
        if self.cmp(&divisor) == Ordering::Less { return (Natural::zero(), self.clone()); }
        // Long division by repeated base-10^9 steps keeps this exact and simple.
        let mut quotient = self.clone();
        let mut left = exponent;
        let mut remainder_parts: Vec<(u64, u32)> = Vec::new();
        while left > 0 {
            let step = left.min(9);
            let (q, r) = quotient.divmod_small(10u64.pow(step));
            remainder_parts.push((r, step));
            quotient = q;
            left -= step;
        }
        let mut remainder = Natural::zero();
        for &(part, step) in remainder_parts.iter().rev() {
            remainder = remainder.mul_small(10u32.pow(step)).add(&Natural::from_u128(part as u128));
        }
        (quotient, remainder)
    }
    fn to_decimal(&self) -> String {
        if self.is_zero() { return "0".into(); }
        let mut chunks = Vec::new();
        let mut value = self.clone();
        while !value.is_zero() {
            let (q, r) = value.divmod_small(1_000_000_000);
            chunks.push(r);
            value = q;
        }
        let mut text = chunks.last().unwrap().to_string();
        for chunk in chunks.iter().rev().skip(1) { text.push_str(&format!("{chunk:09}")); }
        text
    }
    fn digits(&self) -> u32 { if self.is_zero() { 1 } else { self.to_decimal().len() as u32 } }
    fn is_odd(&self) -> bool { self.0.first().is_some_and(|limb| limb & 1 == 1) }
}

/// A finite decimal value: (-1)^negative * coefficient * 10^exponent.
#[derive(Clone, Debug)]
pub struct Decimal { negative: bool, coefficient: Natural, exponent: i64 }

pub const PRECISION: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integral { Floor, Ceiling }

impl Decimal {
    pub fn from_i128(value: i128) -> Self {
        Decimal { negative: value < 0, coefficient: Natural::from_u128(value.unsigned_abs()), exponent: 0 }
    }

    /// Exactly the binary value, as `Decimal.from_float`.
    pub fn from_f64(value: f64) -> Self {
        assert!(value.is_finite(), "only finite floats have a decimal value");
        let bits = value.to_bits();
        let negative = bits >> 63 == 1;
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1u64 << 52) - 1);
        let (mantissa, binary) = if biased == 0 { (fraction, -1074) } else { (fraction | (1u64 << 52), biased - 1075) };
        if mantissa == 0 { return Decimal { negative, coefficient: Natural::zero(), exponent: 0 }; }
        if binary >= 0 {
            let mut coefficient = Natural::from_u128(mantissa as u128);
            for _ in 0..binary { coefficient = coefficient.mul_small(2); }
            Decimal { negative, coefficient, exponent: 0 }
        } else {
            let mut coefficient = Natural::from_u128(mantissa as u128);
            for _ in 0..(-binary) { coefficient = coefficient.mul_small(5); }
            Decimal { negative, coefficient, exponent: binary }
        }
    }

    pub fn is_zero(&self) -> bool { self.coefficient.is_zero() }

    /// Round an exact (or sticky-truncated) value to the context precision.
    fn rounded(negative: bool, coefficient: Natural, exponent: i64, sticky: bool) -> Self {
        let digits = coefficient.digits();
        if digits <= PRECISION {
            debug_assert!(!sticky, "a sticky quotient must carry more digits than the precision");
            return Decimal { negative, coefficient, exponent };
        }
        let drop = digits - PRECISION;
        let (mut kept, removed) = coefficient.divmod_pow10(drop);
        let half = Natural::pow10(drop).divmod_small(2).0;
        let up = match removed.cmp(&half) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => sticky || kept.is_odd(),
        };
        let mut exponent = exponent + drop as i64;
        if up {
            kept = kept.add(&Natural::from_u128(1));
            if kept.cmp(&Natural::pow10(PRECISION)) == Ordering::Equal {
                kept = Natural::pow10(PRECISION - 1);
                exponent += 1;
            }
        }
        Decimal { negative, coefficient: kept, exponent }
    }

    /// `scaleb`: the value times 10^shift, rounded to the context.
    pub fn scaleb(&self, shift: i64) -> Self {
        Self::rounded(self.negative, self.coefficient.clone(), self.exponent + shift, false)
    }

    fn aligned(a: &Self, b: &Self) -> (Natural, Natural, i64) {
        let exponent = a.exponent.min(b.exponent);
        (a.coefficient.mul_pow10((a.exponent - exponent) as u32), b.coefficient.mul_pow10((b.exponent - exponent) as u32), exponent)
    }

    pub fn add(&self, other: &Self) -> Self {
        let (a, b, exponent) = Self::aligned(self, other);
        if self.negative == other.negative {
            return Self::rounded(self.negative, a.add(&b), exponent, false);
        }
        match a.cmp(&b) {
            Ordering::Equal => Decimal { negative: false, coefficient: Natural::zero(), exponent },
            Ordering::Greater => Self::rounded(self.negative, a.sub(&b), exponent, false),
            Ordering::Less => Self::rounded(other.negative, b.sub(&a), exponent, false),
        }
    }

    pub fn neg(&self) -> Self { Decimal { negative: !self.negative && !self.is_zero(), ..self.clone() } }
    pub fn sub(&self, other: &Self) -> Self { self.add(&other.neg()) }
    pub fn abs(&self) -> Self { Decimal { negative: false, ..self.clone() } }

    pub fn mul(&self, other: &Self) -> Self {
        Self::rounded(self.negative != other.negative && !self.is_zero() && !other.is_zero(),
                      self.coefficient.mul(&other.coefficient), self.exponent + other.exponent, false)
    }

    /// Correctly rounded quotient by a divisor whose coefficient fits in 64 bits.
    pub fn div(&self, divisor: &Self) -> Self {
        let small = divisor.coefficient.to_u128().filter(|&value| value != 0 && value <= u64::MAX as u128)
            .expect("divisor coefficient must be a nonzero 64-bit integer") as u64;
        if self.is_zero() { return Decimal { negative: false, coefficient: Natural::zero(), exponent: self.exponent - divisor.exponent }; }
        let (da, db) = (self.coefficient.digits() as i64, Natural::from_u128(small as u128).digits() as i64);
        let shift = (PRECISION as i64 + 2 - da + db).max(0) as u32;
        let (quotient, remainder) = self.coefficient.mul_pow10(shift).divmod_small(small);
        let negative = self.negative != divisor.negative;
        if remainder == 0 {
            return Self::rounded(negative, quotient, self.exponent - divisor.exponent - shift as i64, false);
        }
        Self::rounded(negative, quotient, self.exponent - divisor.exponent - shift as i64, true)
    }

    /// `to_integral_value` with the given rounding, as a machine integer.
    pub fn to_integral(&self, rounding: Integral) -> i128 {
        let magnitude = if self.exponent >= 0 {
            let value = self.coefficient.mul_pow10(self.exponent as u32).to_u128().expect("integral fits 128 bits");
            return if self.negative { -(value as i128) } else { value as i128 };
        } else {
            self.coefficient.divmod_pow10((-self.exponent) as u32)
        };
        let (quotient, remainder) = magnitude;
        let quotient = quotient.to_u128().expect("integral fits 128 bits") as i128;
        let inexact = !remainder.is_zero();
        match (self.negative, rounding) {
            (false, Integral::Floor) => quotient,
            (false, Integral::Ceiling) => quotient + inexact as i128,
            (true, Integral::Floor) => -(quotient + inexact as i128),
            (true, Integral::Ceiling) => -quotient,
        }
    }

    /// The nearest binary64, as `float(Decimal)`.
    pub fn to_f64(&self) -> f64 {
        let text = format!("{}{}e{}", if self.negative { "-" } else { "" }, self.coefficient.to_decimal(), self.exponent);
        text.parse().expect("a finite decimal is a valid float literal")
    }

    pub fn cmp(&self, other: &Self) -> Ordering {
        if self.is_zero() && other.is_zero() { return Ordering::Equal; }
        match (self.negative && !self.is_zero(), other.negative && !other.is_zero()) {
            (false, true) => return Ordering::Greater,
            (true, false) => return Ordering::Less,
            _ => {}
        }
        let (a, b, _) = Self::aligned(self, other);
        let order = a.cmp(&b);
        if self.negative { order.reverse() } else { order }
    }
}

/// Python `decimal` text forms and parsing, for values whose coefficient
/// already fits the context (they are exact, never rounded here).
impl Decimal {
    /// `Decimal(text)` for a finite numeric string (surrounding whitespace allowed).
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let (negative, body) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let (mantissa, exponent) = match body.find(['e', 'E']) {
            Some(at) => (&body[..at], body[at + 1..].parse::<i64>().ok()?),
            None => (body, 0),
        };
        let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if integer.is_empty() && fraction.is_empty() { return None; }
        if !integer.bytes().chain(fraction.bytes()).all(|b| b.is_ascii_digit()) { return None; }
        let mut coefficient = Natural::zero();
        for chunk in integer.as_bytes().iter().chain(fraction.as_bytes()).collect::<Vec<_>>().chunks(9) {
            let digits: String = chunk.iter().map(|&&b| b as char).collect();
            coefficient = coefficient.mul_pow10(digits.len() as u32).add(&Natural::from_u128(digits.parse().unwrap()));
        }
        Some(Decimal { negative, coefficient: coefficient.trim(), exponent: exponent - fraction.len() as i64 })
    }

    /// `Decimal(repr(x))`: the shortest decimal that reads back as `x`.
    pub fn from_repr(value: f64) -> Option<Self> {
        if !value.is_finite() { return None; }
        Decimal::parse(&crate::pyfmt::repr(value))
    }

    pub fn is_negative(&self) -> bool { self.negative }
    pub fn exponent(&self) -> i64 { self.exponent }
    pub fn coefficient_digits(&self) -> String { self.coefficient.to_decimal() }

    /// `adjusted()`: the exponent of the most significant digit.
    pub fn adjusted(&self) -> i64 { self.exponent + self.coefficient.digits() as i64 - 1 }

    /// `normalize()`: trailing zeros removed; zero becomes `0`.
    pub fn normalize(&self) -> Self {
        if self.is_zero() { return Decimal { negative: self.negative, coefficient: Natural::zero(), exponent: 0 }; }
        let mut coefficient = self.coefficient.clone();
        let mut exponent = self.exponent;
        loop {
            let (quotient, remainder) = coefficient.divmod_small(10);
            if remainder != 0 { break; }
            coefficient = quotient;
            exponent += 1;
        }
        Decimal { negative: self.negative, coefficient, exponent }
    }

    /// `format(d, "f")`.
    pub fn format_fixed(&self) -> String {
        let digits = self.coefficient.to_decimal();
        // A zero with a positive exponent has no fixed-point form; it is 0e0.
        let exponent = if self.is_zero() { self.exponent.min(0) } else { self.exponent };
        let dotplace = exponent + digits.len() as i64;
        let (integer, fraction) = if dotplace < 0 {
            ("0".to_owned(), "0".repeat((-dotplace) as usize) + &digits)
        } else if dotplace as usize > digits.len() {
            (digits.clone() + &"0".repeat(dotplace as usize - digits.len()), String::new())
        } else {
            let (i, f) = digits.split_at(dotplace as usize);
            (if i.is_empty() { "0".to_owned() } else { i.to_owned() }, f.to_owned())
        };
        let sign = if self.negative { "-" } else { "" };
        if fraction.is_empty() { format!("{sign}{integer}") } else { format!("{sign}{integer}.{fraction}") }
    }

    /// `str(d)`: the to-scientific-string conversion.
    pub fn to_sci_string(&self) -> String {
        let digits = self.coefficient.to_decimal();
        let leftdigits = self.exponent + digits.len() as i64;
        let dotplace = if self.exponent <= 0 && leftdigits > -6 { leftdigits } else { 1 };
        let (integer, fraction) = if dotplace <= 0 {
            ("0".to_owned(), format!(".{}{}", "0".repeat((-dotplace) as usize), digits))
        } else if dotplace as usize >= digits.len() {
            (digits.clone() + &"0".repeat(dotplace as usize - digits.len()), String::new())
        } else {
            (digits[..dotplace as usize].to_owned(), format!(".{}", &digits[dotplace as usize..]))
        };
        let exponent = if leftdigits == dotplace { String::new() } else { format!("E{:+}", leftdigits - dotplace) };
        format!("{}{integer}{fraction}{exponent}", if self.negative { "-" } else { "" })
    }

    /// Numerical equality, as `==` between Decimals.
    pub fn equals(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal }
}
