//! Agg's numeric conventions as the x86-64 build of Matplotlib executes them.
//!
//! A double converts to `int` by truncation (`cvttsd2si`): NaN or a value
//! outside the 32-bit range gives `i32::MIN`. A double converts to `unsigned`
//! through the 64-bit truncation, keeping its low 32 bits.

pub const PATH_CMD_STOP: u32 = 0;
pub const PATH_CMD_MOVE_TO: u32 = 1;
pub const PATH_CMD_LINE_TO: u32 = 2;
pub const PATH_CMD_CURVE3: u32 = 3;
pub const PATH_CMD_CURVE4: u32 = 4;
pub const PATH_CMD_END_POLY: u32 = 0x0F;
pub const PATH_CMD_MASK: u32 = 0x0F;
pub const PATH_FLAGS_CCW: u32 = 0x10;
pub const PATH_FLAGS_CW: u32 = 0x20;
pub const PATH_FLAGS_CLOSE: u32 = 0x40;
/// Matplotlib's CLOSEPOLY code.
pub const CLOSEPOLY: u32 = PATH_CMD_END_POLY | PATH_FLAGS_CLOSE;

pub const POLY_SUBPIXEL_SHIFT: i32 = 8;
pub const POLY_SUBPIXEL_SCALE: i32 = 1 << POLY_SUBPIXEL_SHIFT;
pub const POLY_SUBPIXEL_MASK: i32 = POLY_SUBPIXEL_SCALE - 1;
pub const COVER_FULL: u8 = 255;
pub const PI: f64 = 3.14159265358979323846;

#[inline(always)]
pub fn is_vertex(c: u32) -> bool { (PATH_CMD_MOVE_TO..PATH_CMD_END_POLY).contains(&c) }
#[inline(always)]
pub fn is_stop(c: u32) -> bool { c == PATH_CMD_STOP }
#[inline(always)]
pub fn is_move_to(c: u32) -> bool { c == PATH_CMD_MOVE_TO }
#[inline(always)]
pub fn is_end_poly(c: u32) -> bool { c & PATH_CMD_MASK == PATH_CMD_END_POLY }
#[inline(always)]
pub fn is_close(c: u32) -> bool { c & !(PATH_FLAGS_CW | PATH_FLAGS_CCW) == CLOSEPOLY }
#[inline(always)]
pub fn is_closed(c: u32) -> bool { c & PATH_FLAGS_CLOSE != 0 }
#[inline(always)]
pub fn get_close_flag(c: u32) -> u32 { c & PATH_FLAGS_CLOSE }

/// `int(v)`.
#[inline(always)]
pub fn trunc_i32(v: f64) -> i32 {
    if v > -2147483649.0 && v < 2147483648.0 { v as i32 } else { i32::MIN }
}

/// `unsigned(v)`.
#[inline(always)]
pub fn trunc_u32(v: f64) -> u32 {
    if v > -9223372036854775809.0 && v < 9223372036854775808.0 { (v as i64) as u32 } else { 0 }
}

#[inline(always)]
pub fn iround(v: f64) -> i32 { trunc_i32(if v < 0.0 { v - 0.5 } else { v + 0.5 }) }

/// Agg's `uround`, which returns the unsigned conversion as an `int`.
#[inline(always)]
pub fn uround(v: f64) -> i32 { trunc_u32(v + 0.5) as i32 }

#[inline(always)]
pub fn ifloor(v: f64) -> i32 {
    let i = trunc_i32(v);
    i - i32::from(f64::from(i) > v)
}

#[inline(always)]
pub fn uceil(v: f64) -> u32 { trunc_u32(v.ceil()) }

#[inline(always)]
pub fn mpl_round_to_int(v: f64) -> i32 { trunc_i32(v + if v >= 0.0 { 0.5 } else { -0.5 }) }

#[inline(always)]
pub fn mpl_round(v: f64) -> f64 { f64::from(mpl_round_to_int(v)) }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RectI { pub x1: i32, pub y1: i32, pub x2: i32, pub y2: i32 }

impl RectI {
    pub const fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self { Self { x1, y1, x2, y2 } }
    pub fn normalize(&mut self) {
        if self.x1 > self.x2 { std::mem::swap(&mut self.x1, &mut self.x2); }
        if self.y1 > self.y2 { std::mem::swap(&mut self.y1, &mut self.y2); }
    }
    pub fn clip(&mut self, r: &RectI) -> bool {
        if self.x2 > r.x2 { self.x2 = r.x2; }
        if self.y2 > r.y2 { self.y2 = r.y2; }
        if self.x1 < r.x1 { self.x1 = r.x1; }
        if self.y1 < r.y1 { self.y1 = r.y1; }
        self.x1 <= self.x2 && self.y1 <= self.y2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RectD { pub x1: f64, pub y1: f64, pub x2: f64, pub y2: f64 }

impl RectD {
    pub const fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self { Self { x1, y1, x2, y2 } }
    pub fn normalize(&mut self) {
        if self.x1 > self.x2 { std::mem::swap(&mut self.x1, &mut self.x2); }
        if self.y1 > self.y2 { std::mem::swap(&mut self.y1, &mut self.y2); }
    }
    pub fn hit_test(&self, x: f64, y: f64) -> bool { x >= self.x1 && x <= self.x2 && y >= self.y1 && y <= self.y2 }
}

/// A vertex pipeline stage. `vertex` writes `x`/`y` exactly where Agg's
/// pointer-based stage does; a stage that leaves them untouched keeps the
/// caller's previous values, as downstream stages observe in C++.
pub trait VertexSource {
    fn rewind(&mut self, path_id: u32);
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32;
}

impl<T: VertexSource + ?Sized> VertexSource for &mut T {
    #[inline(always)]
    fn rewind(&mut self, path_id: u32) { (**self).rewind(path_id) }
    #[inline(always)]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 { (**self).vertex(x, y) }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PointD { pub x: f64, pub y: f64 }

impl PointD {
    pub const fn new(x: f64, y: f64) -> Self { Self { x, y } }
}

#[inline(always)]
pub fn calc_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}

#[inline(always)]
pub fn calc_sq_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    dx * dx + dy * dy
}

#[inline(always)]
pub fn cross_product(x1: f64, y1: f64, x2: f64, y2: f64, x: f64, y: f64) -> f64 {
    (x - x2) * (y2 - y1) - (y - y2) * (x2 - x1)
}

pub const INTERSECTION_EPSILON: f64 = 1.0e-30;
pub const VERTEX_DIST_EPSILON: f64 = 1e-14;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn calc_intersection(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, dx: f64, dy: f64,
                         x: &mut f64, y: &mut f64) -> bool {
    let num = (ay - cy) * (dx - cx) - (ax - cx) * (dy - cy);
    let den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
    if den.abs() < INTERSECTION_EPSILON { return false; }
    let r = num / den;
    *x = ax + r * (bx - ax);
    *y = ay + r * (by - ay);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_follow_cvttsd2si() {
        assert_eq!(trunc_i32(-2.7), -2);
        assert_eq!(trunc_i32(f64::NAN), i32::MIN);
        assert_eq!(trunc_i32(3.0e9), i32::MIN);
        assert_eq!(trunc_u32(4294967296.0 + 5.0), 5);
        assert_eq!(iround(-0.5), -1);
        assert_eq!(mpl_round_to_int(2.5), 3);
        assert_eq!(ifloor(-1.5), -2);
    }
}
