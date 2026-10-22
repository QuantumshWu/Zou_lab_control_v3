//! Linear `rgba`, `rgba8` and `gray8` with Agg's exact integer arithmetic.
use crate::basics::uround;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rgba { pub r: f64, pub g: f64, pub b: f64, pub a: f64 }

impl Rgba {
    pub const fn new(r: f64, g: f64, b: f64, a: f64) -> Self { Self { r, g, b, a } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Rgba8 { pub r: u8, pub g: u8, pub b: u8, pub a: u8 }

impl Rgba8 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self { Self { r, g, b, a } }

    /// `rgba8(const rgba&)`: `value_type(uround(v * 255))`.
    #[inline]
    pub fn from_rgba(c: &Rgba) -> Self {
        Self { r: uround(c.r * 255.0) as u8, g: uround(c.g * 255.0) as u8,
               b: uround(c.b * 255.0) as u8, a: uround(c.a * 255.0) as u8 }
    }

    #[inline(always)]
    pub fn is_transparent(&self) -> bool { self.a == 0 }
    #[inline(always)]
    pub fn is_opaque(&self) -> bool { self.a == 255 }
}

/// `rgba8::multiply`, exact over int8u.
#[inline(always)]
pub fn multiply(a: u8, b: u8) -> u8 {
    let t = u32::from(a) * u32::from(b) + 128;
    (((t >> 8) + t) >> 8) as u8
}

/// `gray8::lerp`.
#[inline(always)]
pub fn lerp(p: u8, q: u8, a: u8) -> u8 {
    let t = (i32::from(q) - i32::from(p)) * i32::from(a) + 128 - i32::from(p > q);
    (i32::from(p) + (((t >> 8) + t) >> 8)) as u8
}

/// `ceil(2^48 / (65280 + alpha))`: over an opaque pixel the plain blend
/// divides by `65280 + alpha`, and every dividend is below 2^26, so a
/// multiply by this and a 48-bit shift is that division exactly.
const OPAQUE_RECIPROCALS: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut alpha = 0;
    while alpha < 256 {
        let d = 65280 + alpha as u64;
        table[alpha] = ((1u64 << 48) + d - 1) / d;
        alpha += 1;
    }
    table
};

/// Matplotlib's `fixed_blender_rgba_plain::blend_pix` without cover.
#[inline(always)]
pub fn blend_pix_plain(p: &mut [u8], cr: u8, cg: u8, cb: u8, alpha: u8) {
    if alpha == 0 { return; }
    if p[3] == 255 {
        // The general form below with a = 255: its denominator is 65280 + alpha.
        let reciprocal = OPAQUE_RECIPROCALS[usize::from(alpha)];
        let alpha = u32::from(alpha);
        let channel = |c: u8, v: u8| {
            let v = u32::from(v) * 255;
            (((u64::from((u32::from(c) << 8).wrapping_sub(v).wrapping_mul(alpha).wrapping_add(v << 8))) * reciprocal) >> 48) as u8
        };
        p[0] = channel(cr, p[0]);
        p[1] = channel(cg, p[1]);
        p[2] = channel(cb, p[2]);
        return;
    }
    let alpha = u32::from(alpha);
    let a = u32::from(p[3]);
    let r = u32::from(p[0]) * a;
    let g = u32::from(p[1]) * a;
    let b = u32::from(p[2]) * a;
    let a = ((alpha + a) << 8).wrapping_sub(alpha * a);
    p[3] = (a >> 8) as u8;
    p[0] = ((u32::from(cr) << 8).wrapping_sub(r).wrapping_mul(alpha).wrapping_add(r << 8) / a) as u8;
    p[1] = ((u32::from(cg) << 8).wrapping_sub(g).wrapping_mul(alpha).wrapping_add(g << 8) / a) as u8;
    p[2] = ((u32::from(cb) << 8).wrapping_sub(b).wrapping_mul(alpha).wrapping_add(b << 8) / a) as u8;
}

/// `fixed_blender_rgba_plain::blend_pix` with a cover.
#[inline(always)]
pub fn blend_pix_plain_cover(p: &mut [u8], c: &Rgba8, cover: u8) {
    blend_pix_plain(p, c.r, c.g, c.b, multiply(c.a, cover));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_rounds_exactly() {
        for a in 0..=255u32 {
            for b in 0..=255u32 {
                let exact = (a * b + 127) / 255;
                assert_eq!(u32::from(multiply(a as u8, b as u8)), exact);
            }
        }
    }

    /// The opaque-pixel shortcut against the general division, every case.
    #[test]
    fn plain_blend_over_opaque_divides_exactly() {
        for alpha in 1..=255u32 {
            for c in 0..=255u32 {
                for v in 0..=255u32 {
                    let r = v * 255;
                    let a = ((alpha + 255) << 8).wrapping_sub(alpha * 255);
                    let want = ((c << 8).wrapping_sub(r).wrapping_mul(alpha).wrapping_add(r << 8) / a) as u8;
                    let mut p = [v as u8, 0, 0, 255];
                    blend_pix_plain(&mut p, c as u8, 0, 0, alpha as u8);
                    assert_eq!(p[0], want, "alpha {alpha} colour {c} under {v}");
                    assert_eq!(p[3], (a >> 8) as u8);
                }
            }
        }
    }

    #[test]
    fn plain_blend_over_transparent_takes_the_colour() {
        let mut p = [255, 255, 255, 0];
        blend_pix_plain(&mut p, 10, 20, 30, 128);
        assert_eq!(p, [10, 20, 30, 128]);
    }
}
