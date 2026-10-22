//! Span generators: the spline36 gray filter that rotates text, nearest
//! neighbour RGBA images, the repeating hatch pattern and Gouraud triangles.
use crate::basics::*;
use crate::color::Rgba8;
use crate::pixfmt::{ImageRgba, SpanGenerator};
use crate::trans_affine::TransAffine;

const IMAGE_SUBPIXEL_SHIFT: i32 = 8;
const IMAGE_SUBPIXEL_SCALE: i32 = 1 << IMAGE_SUBPIXEL_SHIFT;
const IMAGE_SUBPIXEL_MASK: i32 = IMAGE_SUBPIXEL_SCALE - 1;
const IMAGE_FILTER_SHIFT: i32 = 14;
const IMAGE_FILTER_SCALE: i32 = 1 << IMAGE_FILTER_SHIFT;

/// `image_filter_lut` calculated from `image_filter_spline36` and normalized.
pub struct ImageFilterLut { diameter: u32, start: i32, weights: Vec<i16> }

fn spline36(x: f64) -> f64 {
    if x < 1.0 { return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0; }
    if x < 2.0 { return ((-6.0 / 11.0 * (x - 1.0) + 270.0 / 209.0) * (x - 1.0) - 156.0 / 209.0) * (x - 1.0); }
    ((1.0 / 11.0 * (x - 2.0) - 45.0 / 209.0) * (x - 2.0) + 26.0 / 209.0) * (x - 2.0)
}

impl ImageFilterLut {
    pub fn spline36() -> Self {
        let radius = 3.0;
        let diameter = uceil(radius) * 2;
        let start = -((diameter / 2 - 1) as i32);
        let size = (diameter << IMAGE_SUBPIXEL_SHIFT) as usize;
        let mut weights = vec![0i16; size];
        let pivot = (diameter << (IMAGE_SUBPIXEL_SHIFT - 1)) as usize;
        for i in 0..pivot {
            let x = i as f64 / f64::from(IMAGE_SUBPIXEL_SCALE);
            let y = spline36(x);
            let w = iround(y * f64::from(IMAGE_FILTER_SCALE)) as i16;
            weights[pivot + i] = w;
            weights[pivot - i] = w;
        }
        let end = ((diameter << IMAGE_SUBPIXEL_SHIFT) - 1) as usize;
        weights[0] = weights[end];
        let mut lut = Self { diameter, start, weights };
        lut.normalize();
        lut
    }

    fn normalize(&mut self) {
        let d = self.diameter as usize;
        let s = IMAGE_SUBPIXEL_SCALE as usize;
        let mut flip = 1;
        for i in 0..s {
            loop {
                let mut sum: i32 = (0..d).map(|j| i32::from(self.weights[j * s + i])).sum();
                if sum == IMAGE_FILTER_SCALE { break; }
                let k = f64::from(IMAGE_FILTER_SCALE) / f64::from(sum);
                sum = 0;
                for j in 0..d {
                    let w = iround(f64::from(self.weights[j * s + i]) * k) as i16;
                    self.weights[j * s + i] = w;
                    sum += i32::from(w);
                }
                sum -= IMAGE_FILTER_SCALE;
                let inc: i16 = if sum > 0 { -1 } else { 1 };
                let mut j = 0;
                while j < d && sum != 0 {
                    flip ^= 1;
                    let idx = if flip != 0 { d / 2 + j / 2 } else { d / 2 - j / 2 };
                    let v = i32::from(self.weights[idx * s + i]);
                    if v < IMAGE_FILTER_SCALE {
                        self.weights[idx * s + i] += inc;
                        sum += i32::from(inc);
                    }
                    j += 1;
                }
            }
        }
        let pivot = (self.diameter << (IMAGE_SUBPIXEL_SHIFT - 1)) as usize;
        for i in 0..pivot { self.weights[pivot + i] = self.weights[pivot - i]; }
        let end = ((self.diameter << IMAGE_SUBPIXEL_SHIFT) - 1) as usize;
        self.weights[0] = self.weights[end];
    }
}

/// `dda2_line_interpolator`, forward-adjusted.
#[derive(Clone, Copy, Default)]
struct Dda2 { cnt: i32, lft: i32, rem: i32, modulo: i32, y: i32 }

impl Dda2 {
    fn new(y1: i32, y2: i32, count: i32) -> Self {
        let cnt = if count <= 0 { 1 } else { count };
        let mut d = Self { cnt, lft: y2.wrapping_sub(y1) / cnt, rem: y2.wrapping_sub(y1) % cnt, modulo: 0, y: y1 };
        d.modulo = d.rem;
        if d.modulo <= 0 {
            d.modulo += count;
            d.rem += count;
            d.lft -= 1;
        }
        d.modulo -= count;
        d
    }

    #[inline(always)]
    fn inc(&mut self) {
        self.modulo += self.rem;
        self.y = self.y.wrapping_add(self.lft);
        if self.modulo > 0 {
            self.modulo -= self.cnt;
            self.y = self.y.wrapping_add(1);
        }
    }
}

/// `span_interpolator_linear<trans_affine, 8>`.
pub struct InterpolatorLinear { trans: TransAffine, li_x: Dda2, li_y: Dda2 }

impl InterpolatorLinear {
    pub fn new(trans: TransAffine) -> Self { Self { trans, li_x: Dda2::default(), li_y: Dda2::default() } }

    fn begin(&mut self, x: f64, y: f64, len: u32) {
        let (mut tx, mut ty) = (x, y);
        self.trans.transform(&mut tx, &mut ty);
        let x1 = iround(tx * 256.0);
        let y1 = iround(ty * 256.0);
        tx = x + f64::from(len);
        ty = y;
        self.trans.transform(&mut tx, &mut ty);
        let x2 = iround(tx * 256.0);
        let y2 = iround(ty * 256.0);
        self.li_x = Dda2::new(x1, x2, len as i32);
        self.li_y = Dda2::new(y1, y2, len as i32);
    }

    #[inline(always)]
    fn coordinates(&self) -> (i32, i32) { (self.li_x.y, self.li_y.y) }

    #[inline(always)]
    fn next(&mut self) {
        self.li_x.inc();
        self.li_y.inc();
    }
}

/// `font_to_rgba<span_image_filter_gray<image_accessor_clip<pixfmt_gray8>>>`:
/// the spline36-filtered glyph coverage scales the text colour's alpha.
pub struct TextSpans<'a> {
    pub image: &'a [u8],
    pub width: i32,
    pub height: i32,
    pub interpolator: InterpolatorLinear,
    pub filter: ImageFilterLut,
    pub color: Rgba8,
    /// Inked-pixel counts summed over the image (`(height + 1) x (width + 1)`):
    /// a filter window over no ink sums to zero, so it is not evaluated.
    ink: Vec<u32>,
    /// The filtered coverage this draw generates, when it is recorded.
    pub record: Option<TextFilter>,
    /// A recording of the same draw: its spans are copied, not filtered.
    pub replay: Option<(TextFilter, usize, usize)>,
}

/// A rotated text draw's filtered coverage, span by span as generated.
#[derive(Clone, Debug, Default)]
pub struct TextFilter { pub spans: Vec<(i32, i32, u32)>, pub coverage: Vec<u8> }

impl<'a> TextSpans<'a> {
    pub fn new(image: &'a [u8], width: i32, height: i32, interpolator: InterpolatorLinear, filter: ImageFilterLut, color: Rgba8)
        -> TextSpans<'a> {
        let (w, h) = (width.max(0) as usize, height.max(0) as usize);
        let mut ink = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                row += u32::from(image[y * w + x] != 0);
                ink[(y + 1) * (w + 1) + x + 1] = ink[y * (w + 1) + x + 1] + row;
            }
        }
        TextSpans { image, width, height, interpolator, filter, color, ink, record: None, replay: None }
    }

    /// Whether any pixel of the `size` x `size` window at (`x0`, `y0`) is inked.
    #[inline(always)]
    fn inked(&self, x0: i32, y0: i32, size: i32) -> bool {
        let clamp = |v: i32, limit: i32| v.clamp(0, limit) as usize;
        let (xa, xb) = (clamp(x0, self.width), clamp(x0 + size, self.width));
        let (ya, yb) = (clamp(y0, self.height), clamp(y0 + size, self.height));
        if xa >= xb || ya >= yb { return false; }
        let stride = self.width as usize + 1;
        self.ink[yb * stride + xb] + self.ink[ya * stride + xa] != self.ink[ya * stride + xb] + self.ink[yb * stride + xa]
    }

    #[inline(always)]
    fn pixel(&self, x: i32, y: i32) -> u8 {
        if y >= 0 && y < self.height && x >= 0 && x < self.width { self.image[(y * self.width + x) as usize] } else { 0 }
    }
}

impl SpanGenerator for TextSpans<'_> {
    fn prepare(&mut self) {}

    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32, len: u32) {
        if let Some((filter, index, at)) = &mut self.replay {
            if filter.spans.get(*index) == Some(&(x, y, len)) {
                for (out, &fg) in span.iter_mut().zip(&filter.coverage[*at..*at + len as usize]) {
                    *out = self.color;
                    out.a = ((u32::from(self.color.a) * u32::from(fg)) >> 8) as u8;
                }
                *index += 1;
                *at += len as usize;
                return;
            }
            self.replay = None;
        }
        if let Some(record) = &mut self.record { record.spans.push((x, y, len)); }
        self.interpolator.begin(f64::from(x) + 0.5, f64::from(y) + 0.5, len);
        let diameter = self.filter.diameter as i32;
        let start = self.filter.start;
        for out in span.iter_mut().take(len as usize) {
            let (mut x, mut y) = self.interpolator.coordinates();
            x -= IMAGE_SUBPIXEL_SCALE / 2;
            y -= IMAGE_SUBPIXEL_SCALE / 2;
            let x_lr = x >> IMAGE_SUBPIXEL_SHIFT;
            let y_lr = y >> IMAGE_SUBPIXEL_SHIFT;
            let mut fg: i32 = 0;
            let x_fract = x & IMAGE_SUBPIXEL_MASK;
            let mut y_hr = IMAGE_SUBPIXEL_MASK - (y & IMAGE_SUBPIXEL_MASK);
            // image_accessor_clip::span/next_x/next_y read the same pixels:
            // inside the image directly, outside it the zero background.
            let x0 = x_lr + start;
            let y0 = y_lr + start;
            if !self.inked(x0, y0, diameter) {
                // Every tap reads the zero background: the sum is zero.
                *out = self.color;
                out.a = 0;
                if let Some(record) = &mut self.record { record.coverage.push(0); }
                self.interpolator.next();
                continue;
            }
            for row in 0..diameter {
                let weight_y = i32::from(self.filter.weights[y_hr as usize]);
                let mut x_hr = IMAGE_SUBPIXEL_MASK - x_fract;
                for column in 0..diameter {
                    let value = i32::from(self.pixel(x0 + column, y0 + row));
                    fg += value * ((weight_y * i32::from(self.filter.weights[x_hr as usize]) + IMAGE_FILTER_SCALE / 2)
                        >> IMAGE_FILTER_SHIFT);
                    x_hr += IMAGE_SUBPIXEL_SCALE;
                }
                y_hr += IMAGE_SUBPIXEL_SCALE;
            }
            fg >>= IMAGE_FILTER_SHIFT;
            if fg < 0 { fg = 0; }
            if fg > 255 { fg = 255; }
            *out = self.color;
            out.a = ((u32::from(self.color.a) * fg as u32) >> 8) as u8;
            if let Some(record) = &mut self.record { record.coverage.push(fg as u8); }
            self.interpolator.next();
        }
    }
}

/// `span_converter<span_image_filter_rgba_nn<image_accessor_clip<pixfmt>>,
/// span_conv_alpha>` over an image attached with a negative stride.
pub struct ImageSpansNn<'a> { pub image: ImageRgba<'a>, pub interpolator: InterpolatorLinear, pub alpha: f64 }

impl SpanGenerator for ImageSpansNn<'_> {
    fn prepare(&mut self) {}

    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32, len: u32) {
        self.interpolator.begin(f64::from(x) + 0.5, f64::from(y) + 0.5, len);
        for out in span.iter_mut().take(len as usize) {
            let (x, y) = self.interpolator.coordinates();
            let (px, py) = (x >> IMAGE_SUBPIXEL_SHIFT, y >> IMAGE_SUBPIXEL_SHIFT);
            *out = if py >= 0 && py < self.image.height as i32 && px >= 0 && px < self.image.width as i32 {
                let o = self.image.offset(px, py) as usize;
                let d = self.image.data;
                Rgba8::new(d[o], d[o + 1], d[o + 2], d[o + 3])
            } else {
                Rgba8::new(0, 0, 0, 0)
            };
            self.interpolator.next();
        }
        for out in span.iter_mut().take(len as usize) {
            out.a = (f64::from(out.a) * self.alpha) as u8;
        }
    }
}

/// `wrap_mode_repeat_auto_pow2`.
struct WrapPow2 { size: u32, add: u32, mask: u32, value: u32 }

impl WrapPow2 {
    fn new(size: u32) -> Self {
        Self { size, add: size.wrapping_mul(0x3FFFFFFF / size), mask: if size & (size.wrapping_sub(1)) != 0 { 0 } else { size - 1 },
               value: 0 }
    }
    #[inline(always)]
    fn at(&mut self, v: i32) -> u32 {
        self.value = if self.mask != 0 { (v as u32) & self.mask } else { (v as u32).wrapping_add(self.add) % self.size };
        self.value
    }
    #[inline(always)]
    fn inc(&mut self) -> u32 {
        self.value += 1;
        if self.value >= self.size { self.value = 0; }
        self.value
    }
}

/// `span_pattern_rgba<image_accessor_wrap<pixfmt, repeat_auto_pow2 ×2>>`
/// with zero offsets.
pub struct PatternSpans<'a> { pub pattern: &'a [u8], pub size: u32 }

impl SpanGenerator for PatternSpans<'_> {
    fn prepare(&mut self) {}

    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32, len: u32) {
        let mut wrap_x = WrapPow2::new(self.size);
        let mut wrap_y = WrapPow2::new(self.size);
        let row = wrap_y.at(y) as usize * self.size as usize * 4;
        let mut column = wrap_x.at(x) as usize;
        for (i, out) in span.iter_mut().take(len as usize).enumerate() {
            if i != 0 { column = wrap_x.inc() as usize; }
            let o = row + column * 4;
            *out = Rgba8::new(self.pattern[o], self.pattern[o + 1], self.pattern[o + 2], self.pattern[o + 3]);
        }
    }
}

/// `dda_line_interpolator<14>`.
#[derive(Clone, Copy)]
struct Dda14 { y: i32, inc: i32, dy: i32 }

impl Dda14 {
    fn new(y1: i32, y2: i32, count: u32) -> Self {
        Self { y: y1, inc: (y2.wrapping_sub(y1) << 14) / (count as i32), dy: 0 }
    }
    #[inline(always)]
    fn add(&mut self, n: i32) { self.dy = self.dy.wrapping_add(self.inc.wrapping_mul(n)); }
    #[inline(always)]
    fn sub(&mut self, n: i32) { self.dy = self.dy.wrapping_sub(self.inc.wrapping_mul(n)); }
    #[inline(always)]
    fn value(&self) -> i32 { self.y.wrapping_add(self.dy >> 14) }
}

#[derive(Clone, Copy, Default)]
struct GouraudCoord { x: f64, y: f64, color: Rgba8 }

#[derive(Clone, Copy, Default)]
struct RgbaCalc {
    x1: f64, y1: f64, dx: f64, one_dy: f64,
    r1: i32, g1: i32, b1: i32, a1: i32, dr: i32, dg: i32, db: i32, da: i32,
    r: i32, g: i32, b: i32, a: i32, x: i32,
}

impl RgbaCalc {
    fn init(c1: &GouraudCoord, c2: &GouraudCoord) -> Self {
        let dy = c2.y - c1.y;
        let (r1, g1, b1, a1) = (i32::from(c1.color.r), i32::from(c1.color.g), i32::from(c1.color.b), i32::from(c1.color.a));
        Self { x1: c1.x - 0.5, y1: c1.y - 0.5, dx: c2.x - c1.x, one_dy: if dy < 1e-5 { 1e5 } else { 1.0 / dy },
               r1, g1, b1, a1, dr: i32::from(c2.color.r) - r1, dg: i32::from(c2.color.g) - g1,
               db: i32::from(c2.color.b) - b1, da: i32::from(c2.color.a) - a1, ..Default::default() }
    }

    fn calc(&mut self, y: f64) {
        let mut k = (y - self.y1) * self.one_dy;
        if k < 0.0 { k = 0.0; }
        if k > 1.0 { k = 1.0; }
        self.r = self.r1 + iround(f64::from(self.dr) * k);
        self.g = self.g1 + iround(f64::from(self.dg) * k);
        self.b = self.b1 + iround(f64::from(self.db) * k);
        self.a = self.a1 + iround(f64::from(self.da) * k);
        self.x = iround((self.x1 + self.dx * k) * 16.0);
    }
}

fn calc_orthogonal(thickness: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let d = (dx * dx + dy * dy).sqrt();
    (thickness * dy / d, -thickness * dx / d)
}

/// `span_gouraud_rgba<rgba8>` with its 6-vertex dilated triangle outline.
pub struct GouraudSpans {
    coord: [GouraudCoord; 3],
    vx: [f64; 8],
    vy: [f64; 8],
    cmd: [u32; 8],
    vertex: usize,
    swap: bool,
    y2: i32,
    rgba1: RgbaCalc,
    rgba2: RgbaCalc,
    rgba3: RgbaCalc,
}

impl GouraudSpans {
    #[allow(clippy::too_many_arguments)]
    pub fn new(c1: Rgba8, c2: Rgba8, c3: Rgba8, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, d: f64) -> Self {
        let mut s = Self { coord: [GouraudCoord::default(); 3], vx: [0.0; 8], vy: [0.0; 8], cmd: [PATH_CMD_STOP; 8],
                           vertex: 0, swap: false, y2: 0, rgba1: RgbaCalc::default(), rgba2: RgbaCalc::default(),
                           rgba3: RgbaCalc::default() };
        s.coord[0].color = c1;
        s.coord[1].color = c2;
        s.coord[2].color = c3;
        s.coord[0].x = x1; s.vx[0] = x1;
        s.coord[0].y = y1; s.vy[0] = y1;
        s.coord[1].x = x2; s.vx[1] = x2;
        s.coord[1].y = y2; s.vy[1] = y2;
        s.coord[2].x = x3; s.vx[2] = x3;
        s.coord[2].y = y3; s.vy[2] = y3;
        s.cmd[0] = PATH_CMD_MOVE_TO;
        s.cmd[1] = PATH_CMD_LINE_TO;
        s.cmd[2] = PATH_CMD_LINE_TO;
        s.cmd[3] = PATH_CMD_STOP;
        if d != 0.0 {
            s.dilate_triangle(d);
            let (vx, vy) = (s.vx, s.vy);
            calc_intersection(vx[4], vy[4], vx[5], vy[5], vx[0], vy[0], vx[1], vy[1], &mut s.coord[0].x, &mut s.coord[0].y);
            calc_intersection(vx[0], vy[0], vx[1], vy[1], vx[2], vy[2], vx[3], vy[3], &mut s.coord[1].x, &mut s.coord[1].y);
            calc_intersection(vx[2], vy[2], vx[3], vy[3], vx[4], vy[4], vx[5], vy[5], &mut s.coord[2].x, &mut s.coord[2].y);
            s.cmd[3] = PATH_CMD_LINE_TO;
            s.cmd[4] = PATH_CMD_LINE_TO;
            s.cmd[5] = PATH_CMD_LINE_TO;
            s.cmd[6] = PATH_CMD_STOP;
        }
        s
    }

    fn dilate_triangle(&mut self, mut d: f64) {
        let [c0, c1, c2] = self.coord;
        let (x1, y1, x2, y2, x3, y3) = (c0.x, c0.y, c1.x, c1.y, c2.x, c2.y);
        let (mut dx1, mut dy1, mut dx2, mut dy2, mut dx3, mut dy3) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let loc = cross_product(x1, y1, x2, y2, x3, y3);
        if loc.abs() > INTERSECTION_EPSILON {
            if cross_product(x1, y1, x2, y2, x3, y3) > 0.0 { d = -d; }
            (dx1, dy1) = calc_orthogonal(d, x1, y1, x2, y2);
            (dx2, dy2) = calc_orthogonal(d, x2, y2, x3, y3);
            (dx3, dy3) = calc_orthogonal(d, x3, y3, x1, y1);
        }
        self.vx[0] = x1 + dx1; self.vy[0] = y1 + dy1;
        self.vx[1] = x2 + dx1; self.vy[1] = y2 + dy1;
        self.vx[2] = x2 + dx2; self.vy[2] = y2 + dy2;
        self.vx[3] = x3 + dx2; self.vy[3] = y3 + dy2;
        self.vx[4] = x3 + dx3; self.vy[4] = y3 + dy3;
        self.vx[5] = x1 + dx3; self.vy[5] = y1 + dy3;
    }
}

impl VertexSource for GouraudSpans {
    fn rewind(&mut self, _path_id: u32) { self.vertex = 0; }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        *x = self.vx[self.vertex];
        *y = self.vy[self.vertex];
        let cmd = self.cmd[self.vertex];
        self.vertex += 1;
        cmd
    }
}

impl SpanGenerator for GouraudSpans {
    fn prepare(&mut self) {
        let mut coord = self.coord;
        if self.coord[0].y > self.coord[2].y {
            coord[0] = self.coord[2];
            coord[2] = self.coord[0];
        }
        if coord[0].y > coord[1].y { coord.swap(0, 1); }
        if coord[1].y > coord[2].y { coord.swap(1, 2); }
        self.y2 = trunc_i32(coord[1].y);
        self.swap = cross_product(coord[0].x, coord[0].y, coord[2].x, coord[2].y, coord[1].x, coord[1].y) < 0.0;
        self.rgba1 = RgbaCalc::init(&coord[0], &coord[2]);
        self.rgba2 = RgbaCalc::init(&coord[0], &coord[1]);
        self.rgba3 = RgbaCalc::init(&coord[1], &coord[2]);
    }

    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32, len: u32) {
        self.rgba1.calc(f64::from(y));
        let mut pc1 = self.rgba1;
        let mut pc2;
        if y <= self.y2 {
            self.rgba2.calc(f64::from(y) + self.rgba2.one_dy);
            pc2 = self.rgba2;
        } else {
            self.rgba3.calc(f64::from(y) - self.rgba3.one_dy);
            pc2 = self.rgba3;
        }
        if self.swap { std::mem::swap(&mut pc1, &mut pc2); }
        let mut nlen = (pc2.x - pc1.x).wrapping_abs();
        if nlen <= 0 { nlen = 1; }
        let mut r = Dda14::new(pc1.r, pc2.r, nlen as u32);
        let mut g = Dda14::new(pc1.g, pc2.g, nlen as u32);
        let mut b = Dda14::new(pc1.b, pc2.b, nlen as u32);
        let mut a = Dda14::new(pc1.a, pc2.a, nlen as u32);
        let mut start = pc1.x - (x << 4);
        r.sub(start);
        g.sub(start);
        b.sub(start);
        a.sub(start);
        nlen += start;
        let clamp = |v: i32| v.clamp(0, 255) as u8;
        let mut i = 0usize;
        let mut len = len;
        while len != 0 && start > 0 {
            span[i] = Rgba8::new(clamp(r.value()), clamp(g.value()), clamp(b.value()), clamp(a.value()));
            r.add(16); g.add(16); b.add(16); a.add(16);
            nlen -= 16;
            start -= 16;
            i += 1;
            len -= 1;
        }
        while len != 0 && nlen > 0 {
            span[i] = Rgba8::new(r.value() as u8, g.value() as u8, b.value() as u8, a.value() as u8);
            r.add(16); g.add(16); b.add(16); a.add(16);
            nlen -= 16;
            i += 1;
            len -= 1;
        }
        while len != 0 {
            span[i] = Rgba8::new(clamp(r.value()), clamp(g.value()), clamp(b.value()), clamp(a.value()));
            r.add(16); g.add(16); b.add(16); a.add(16);
            i += 1;
            len -= 1;
        }
    }
}
