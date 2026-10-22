//! Exact arithmetic where v2 used Python's `Fraction` and unbounded `int`.
//!
//! The clock grid is decided on `Fraction(str(value))`: the decimal a value is
//! written as, not its binary approximation, so `0.02 us` is exactly one 20 ns
//! tick. A loop table's duration is an integer that nested 32-bit counts can
//! make wider than any machine word. Both are reproduced here exactly, and every
//! conversion to a float is correctly rounded (half to even), as Python's
//! `float(Fraction)` and `float(int)` are.
use std::cmp::Ordering;

/// An unsigned integer of any width, least significant 32-bit limb first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BigUint { limbs: Vec<u32> }

impl BigUint {
    pub fn zero() -> Self { BigUint { limbs: Vec::new() } }

    pub fn from_u128(mut value: u128) -> Self {
        let mut limbs = Vec::new();
        while value != 0 { limbs.push(value as u32); value >>= 32; }
        BigUint { limbs }
    }

    fn trim(mut self) -> Self {
        while self.limbs.last() == Some(&0) { self.limbs.pop(); }
        self
    }

    pub fn is_zero(&self) -> bool { self.limbs.is_empty() }

    pub fn bits(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(top) => (self.limbs.len() as u64 - 1) * 32 + u64::from(32 - top.leading_zeros()),
        }
    }

    pub fn to_u128(&self) -> Option<u128> {
        if self.limbs.len() > 4 { return None; }
        Some(self.limbs.iter().rev().fold(0u128, |acc, &limb| (acc << 32) | u128::from(limb)))
    }

    pub fn add(&self, other: &BigUint) -> BigUint {
        let (long, short) = if self.limbs.len() >= other.limbs.len() { (self, other) } else { (other, self) };
        let mut limbs = Vec::with_capacity(long.limbs.len() + 1);
        let mut carry = 0u64;
        for (index, &limb) in long.limbs.iter().enumerate() {
            let sum = u64::from(limb) + u64::from(*short.limbs.get(index).unwrap_or(&0)) + carry;
            limbs.push(sum as u32);
            carry = sum >> 32;
        }
        if carry != 0 { limbs.push(carry as u32); }
        BigUint { limbs }
    }

    /// `self - other`; `other` must not exceed `self`.
    pub fn sub(&self, other: &BigUint) -> BigUint {
        debug_assert!(self.cmp(other) != Ordering::Less);
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i64;
        for (index, &limb) in self.limbs.iter().enumerate() {
            let mut difference = i64::from(limb) - i64::from(*other.limbs.get(index).unwrap_or(&0)) - borrow;
            borrow = if difference < 0 { difference += 1 << 32; 1 } else { 0 };
            limbs.push(difference as u32);
        }
        BigUint { limbs }.trim()
    }

    pub fn mul_small(&self, factor: u64) -> BigUint {
        if factor == 0 || self.is_zero() { return BigUint::zero(); }
        self.mul(&BigUint::from_u128(u128::from(factor)))
    }

    pub fn mul(&self, other: &BigUint) -> BigUint {
        if self.is_zero() || other.is_zero() { return BigUint::zero(); }
        let mut limbs = vec![0u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let product = u64::from(a) * u64::from(b) + u64::from(limbs[i + j]) + carry;
                limbs[i + j] = product as u32;
                carry = product >> 32;
            }
            let mut k = i + other.limbs.len();
            while carry != 0 {
                let sum = u64::from(limbs[k]) + carry;
                limbs[k] = sum as u32;
                carry = sum >> 32;
                k += 1;
            }
        }
        BigUint { limbs }.trim()
    }

    pub fn pow10(exponent: u32) -> BigUint {
        let mut result = BigUint::from_u128(1);
        let mut left = exponent;
        while left > 0 {
            let step = left.min(19);
            result = result.mul_small(10u64.pow(step));
            left -= step;
        }
        result
    }

    pub fn shl(&self, bits: u64) -> BigUint {
        if self.is_zero() { return BigUint::zero(); }
        let (whole, part) = ((bits / 32) as usize, (bits % 32) as u32);
        let mut limbs = vec![0u32; whole];
        let mut carry = 0u32;
        for &limb in &self.limbs {
            if part == 0 { limbs.push(limb); } else {
                limbs.push((limb << part) | carry);
                carry = limb >> (32 - part);
            }
        }
        if carry != 0 { limbs.push(carry); }
        BigUint { limbs }.trim()
    }

    fn bit(&self, index: u64) -> bool {
        self.limbs.get((index / 32) as usize).is_some_and(|limb| (limb >> (index % 32)) & 1 == 1)
    }

    /// `(self // divisor, self % divisor)`; the divisor is not zero.
    pub fn divrem(&self, divisor: &BigUint) -> (BigUint, BigUint) {
        assert!(!divisor.is_zero(), "division by zero");
        if self.cmp(divisor) == Ordering::Less { return (BigUint::zero(), self.clone()); }
        if let (Some(a), Some(b)) = (self.to_u128(), divisor.to_u128()) {
            return (BigUint::from_u128(a / b), BigUint::from_u128(a % b));
        }
        let mut quotient = vec![0u32; self.limbs.len()];
        let mut remainder = BigUint::zero();
        for index in (0..self.bits()).rev() {
            remainder = remainder.shl(1);
            if self.bit(index) {
                if remainder.limbs.is_empty() { remainder.limbs.push(1); } else { remainder.limbs[0] |= 1; }
            }
            if remainder.cmp(divisor) != Ordering::Less {
                remainder = remainder.sub(divisor);
                quotient[(index / 32) as usize] |= 1 << (index % 32);
            }
        }
        (BigUint { limbs: quotient }.trim(), remainder)
    }

    /// Decimal digits, as Python's `str(int)` writes them.
    pub fn to_decimal(&self) -> String {
        if let Some(value) = self.to_u128() { return value.to_string(); }
        let chunk = BigUint::from_u128(1_000_000_000_000_000_000);
        let mut parts = Vec::new();
        let mut rest = self.clone();
        while !rest.is_zero() {
            let (quotient, remainder) = rest.divrem(&chunk);
            parts.push(remainder.to_u128().unwrap_or(0));
            rest = quotient;
        }
        let mut text = parts.last().copied().unwrap_or(0).to_string();
        for part in parts.iter().rev().skip(1) { text.push_str(&format!("{part:018}")); }
        text
    }

    pub fn from_decimal(digits: &str) -> Option<BigUint> {
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) { return None; }
        let mut result = BigUint::zero();
        for chunk in digits.as_bytes().chunks(18) {
            let text = std::str::from_utf8(chunk).ok()?;
            result = result.mul_small(10u64.pow(chunk.len() as u32)).add(&BigUint::from_u128(text.parse().ok()?));
        }
        Some(result)
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs.len().cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

/// `numerator / denominator` correctly rounded to a float, half to even
/// (Python's true division of two ints). The denominator is not zero.
pub fn ratio_to_f64(numerator: &BigUint, denominator: &BigUint) -> f64 {
    if numerator.is_zero() { return 0.0; }
    // Scale so the quotient has exactly 55 bits: 53 kept, a guard bit and a
    // round bit, with the remainder as the sticky bit.
    let mut shift = 55i64 - (numerator.bits() as i64 - denominator.bits() as i64);
    let (mut quotient, mut remainder);
    loop {
        (quotient, remainder) = if shift >= 0 { numerator.shl(shift as u64).divrem(denominator) }
            else { numerator.divrem(&denominator.shl((-shift) as u64)) };
        match quotient.bits() {
            55 => break,
            bits if bits < 55 => shift += 1,
            _ => shift -= 1,
        }
    }
    let q = quotient.to_u128().expect("55 bits") as u64;
    let mut mantissa = q >> 2;
    let low = q & 3;
    let sticky = !remainder.is_zero();
    if low > 2 || (low == 2 && (sticky || mantissa & 1 == 1)) { mantissa += 1; }
    // value = mantissa * 2^(2 - shift); a carry out of 53 bits stays exact.
    scale_by_power_of_two(mantissa as f64, 2 - shift)
}

fn scale_by_power_of_two(mut value: f64, mut exponent: i64) -> f64 {
    while exponent > 1000 { value *= 2f64.powi(1000); exponent -= 1000; }
    while exponent < -1000 { value *= 2f64.powi(-1000); exponent += 1000; }
    value * 2f64.powi(exponent as i32)
}

/// `float(int)`, correctly rounded.
pub fn big_to_f64(value: &BigUint) -> f64 { ratio_to_f64(value, &BigUint::from_u128(1)) }

/// A signed decimal `mantissa * 10^exponent`: exactly what `Fraction(str(x))` holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decimal { pub negative: bool, pub mantissa: BigUint, pub exponent: i64 }

impl Decimal {
    pub fn zero() -> Self { Decimal { negative: false, mantissa: BigUint::zero(), exponent: 0 } }

    pub fn from_i128(value: i128) -> Self {
        Decimal { negative: value < 0, mantissa: BigUint::from_u128(value.unsigned_abs()), exponent: 0 }.normal()
    }

    /// `Fraction(str(value))` of a float: the shortest digits that round-trip.
    pub fn from_f64(value: f64) -> Self {
        Decimal::parse(&zlc_core::pyfmt::repr(value)).expect("a finite float's repr is a decimal")
    }

    /// A decimal literal: optional sign, digits with an optional point, optional exponent.
    pub fn parse(text: &str) -> Option<Self> {
        let (negative, body) = match text.as_bytes().first()? {
            b'-' => (true, &text[1..]),
            b'+' => (false, &text[1..]),
            _ => (false, text),
        };
        let (number, exponent) = match body.find(['e', 'E']) {
            Some(at) => (&body[..at], body[at + 1..].parse::<i64>().ok()?),
            None => (body, 0),
        };
        let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
        if whole.is_empty() && fraction.is_empty() { return None; }
        let digits = format!("{whole}{fraction}");
        let mantissa = BigUint::from_decimal(&digits)?;
        Some(Decimal { negative: negative && !mantissa.is_zero(), mantissa, exponent: exponent - fraction.len() as i64 }.normal())
    }

    /// Trailing decimal zeros moved into the exponent (zero has exponent 0).
    fn normal(mut self) -> Self {
        if self.mantissa.is_zero() { return Decimal::zero(); }
        let ten = BigUint::from_u128(10);
        loop {
            let (quotient, remainder) = self.mantissa.divrem(&ten);
            if !remainder.is_zero() { break; }
            self.mantissa = quotient;
            self.exponent += 1;
        }
        self
    }

    pub fn is_zero(&self) -> bool { self.mantissa.is_zero() }

    pub fn scale10(&self, power: i64) -> Decimal {
        if self.is_zero() { return Decimal::zero(); }
        Decimal { negative: self.negative, mantissa: self.mantissa.clone(), exponent: self.exponent + power }
    }

    pub fn mul(&self, other: &Decimal) -> Decimal {
        Decimal { negative: self.negative != other.negative, mantissa: self.mantissa.mul(&other.mantissa),
                  exponent: self.exponent + other.exponent }.normal()
    }

    /// `(numerator, denominator)` of the absolute value as integers.
    fn parts(&self) -> (BigUint, BigUint) {
        if self.exponent >= 0 { (self.mantissa.mul(&BigUint::pow10(self.exponent as u32)), BigUint::from_u128(1)) }
        else { (self.mantissa.clone(), BigUint::pow10((-self.exponent) as u32)) }
    }

    /// `float(Fraction(...))`.
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() { return 0.0; }
        if self.exponent > 400 { return if self.negative { f64::NEG_INFINITY } else { f64::INFINITY }; }
        if self.exponent < -800 && self.mantissa.bits() < 2000 { return if self.negative { -0.0 } else { 0.0 }; }
        let (numerator, denominator) = self.parts();
        let magnitude = ratio_to_f64(&numerator, &denominator);
        if self.negative { -magnitude } else { magnitude }
    }

    pub fn cmp_abs(&self, other: &Decimal) -> Ordering {
        let (a, b) = (self.parts(), other.parts());
        a.0.mul(&b.1).cmp(&b.0.mul(&a.1))
    }
}

/// An exact non-negative rational `numerator / denominator`, as `Fraction` holds it.
#[derive(Clone, Debug)]
pub struct Ratio { pub numerator: BigUint, pub denominator: BigUint }

impl Ratio {
    /// `|value| / divisor` of two decimals.
    pub fn of(value: &Decimal, divisor: &Decimal) -> Ratio {
        let shift = value.exponent - divisor.exponent;
        let (numerator, denominator) = if shift >= 0 {
            (value.mantissa.mul(&BigUint::pow10(shift as u32)), divisor.mantissa.clone())
        } else {
            (value.mantissa.clone(), divisor.mantissa.mul(&BigUint::pow10((-shift) as u32)))
        };
        Ratio { numerator, denominator }
    }

    /// The integer it equals, if it is one.
    pub fn integer(&self) -> Option<BigUint> {
        let (quotient, remainder) = self.numerator.divrem(&self.denominator);
        remainder.is_zero().then_some(quotient)
    }

    /// `int(ratio + 1/2)`: the nearest integer, halves rounded up.
    pub fn round_half_up(&self) -> BigUint {
        self.numerator.shl(1).add(&self.denominator).divrem(&self.denominator.shl(1)).0
    }

    pub fn to_f64(&self) -> f64 { ratio_to_f64(&self.numerator, &self.denominator) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_round_like_python() {
        let cases: [(u128, u128, f64); 5] = [(1, 3, 1.0 / 3.0), (2, 3, 2.0 / 3.0), (10, 1, 10.0), (1, 10, 0.1), ((1 << 60) + 1, 1, 1152921504606846976.0)];
        for (n, d, expected) in cases {
            assert_eq!(ratio_to_f64(&BigUint::from_u128(n), &BigUint::from_u128(d)).to_bits(), expected.to_bits(), "{n}/{d}");
        }
        // Halfway between two floats rounds to the even mantissa.
        let halfway = (1u128 << 53) + 1;
        assert_eq!(big_to_f64(&BigUint::from_u128(halfway)), 9007199254740992.0);
        assert_eq!(big_to_f64(&BigUint::from_u128(halfway + 2)), 9007199254740996.0);
        assert_eq!(Decimal::parse("1e-07").unwrap().to_f64(), 1e-7);
        assert_eq!(Decimal::parse("0.0151").unwrap().scale10(9).to_f64(), 15100000.0);
        let big = BigUint::pow10(40).add(&BigUint::from_u128(7));
        assert_eq!(big.to_decimal(), format!("1{}7", "0".repeat(39)));
        assert_eq!(BigUint::from_decimal(&big.to_decimal()), Some(big.clone()));
        assert_eq!(big.divrem(&BigUint::pow10(20)).1, BigUint::from_u128(7));
    }
}
