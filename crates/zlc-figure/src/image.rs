//! `matplotlib.image.AxesImage.draw`: `_make_image` for scalar data (its
//! 'rgba' and 'data' interpolation stages), `_resample`, and the C++
//! `_image.resample` for nearest-neighbour affine resampling of the dtypes
//! those stages use (gray64, rgba64, gray32), on zlc-agg's rasterizer.
use zlc_agg::rasterizer::RasterizerScanlineAa;
use zlc_agg::scanline::{ScanlineU8, SpanScanline};
use zlc_agg::trans_affine::TransAffine;

use crate::backend::RgbaImage;
use crate::colors::{process_dtype, Precision};
use crate::draw::Ctx;
use crate::ir::Image;
use crate::transform::{transformed_points, M3, Points};
use crate::wire::DType;
use crate::{unsupported, Result};

/// `Bbox.intersection(a, b)`: `xmin`/`xmax` are numpy reductions and the
/// combination `np.maximum`/`np.minimum`, all NaN-propagating (a NaN
/// coordinate makes no intersection).
fn intersection(a: &Points, b: &Points) -> Option<Points> {
    let nan_or = |u: f64, v: f64, pick: fn(f64, f64) -> f64| if u.is_nan() || v.is_nan() { f64::NAN } else { pick(u, v) };
    let amin = |p: &Points, i: usize| nan_or(p[i], p[i + 2], |u, v| if u < v { u } else { v });
    let amax = |p: &Points, i: usize| nan_or(p[i], p[i + 2], |u, v| if u > v { u } else { v });
    let x0 = nan_or(amin(a, 0), amin(b, 0), f64::max);
    let x1 = nan_or(amax(a, 0), amax(b, 0), f64::min);
    let y0 = nan_or(amin(a, 1), amin(b, 1), f64::max);
    let y1 = nan_or(amax(a, 1), amax(b, 1), f64::min);
    if x0 <= x1 && y0 <= y1 { Some([x0, y0, x1, y1]) } else { None }
}

// ---------------------------------------------------------------- resample
/// `dda2_line_interpolator`.
#[derive(Clone, Copy, Default)]
struct Dda2 { cnt: i32, lft: i32, rem: i32, modulo: i32, y: i32 }

impl Dda2 {
    fn new(y1: i32, y2: i32, count: i32) -> Self {
        let cnt = if count <= 0 { 1 } else { count };
        let mut d = Dda2 { cnt, lft: y2.wrapping_sub(y1) / cnt, rem: y2.wrapping_sub(y1) % cnt, modulo: 0, y: y1 };
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

fn iround(v: f64) -> i32 { (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i32 }

/// `span_interpolator_linear<trans_affine, 8>`.
struct Interpolator { trans: TransAffine, x: Dda2, y: Dda2 }

impl Interpolator {
    fn begin(&mut self, x: f64, y: f64, len: u32) {
        let (mut tx, mut ty) = (x, y);
        self.trans.transform(&mut tx, &mut ty);
        let x1 = iround(tx * 256.0);
        let y1 = iround(ty * 256.0);
        let (mut tx, mut ty) = (x + f64::from(len), y);
        self.trans.transform(&mut tx, &mut ty);
        let x2 = iround(tx * 256.0);
        let y2 = iround(ty * 256.0);
        self.x = Dda2::new(x1, x2, len as i32);
        self.y = Dda2::new(y1, y2, len as i32);
    }
}

/// `wrap_mode_reflect`.
struct Reflect { size: u32, size2: u32, add: u32 }

impl Reflect {
    fn new(size: u32) -> Self {
        let size2 = size.wrapping_mul(2);
        Reflect { size, size2, add: size2.wrapping_mul(0x3FFF_FFFF / size2) }
    }
    #[inline(always)]
    fn at(&self, v: i32) -> u32 {
        // `add` is a multiple of 2*size: inside the image the wrap is the identity.
        if (v as u32) < self.size { return v as u32; }
        let value = (v as u32).wrapping_add(self.add) % self.size2;
        if value >= self.size { self.size2 - value - 1 } else { value }
    }
}

/// A pixel type of `_image_resample.h` with its plain blender.
trait Pixel: Copy {
    /// The source sample as the NN span generator reads it.
    fn opaque(&self) -> bool;
    fn transparent(&self) -> bool;
    /// `blend_pix(p, c, cover)` (cover < 255 or not opaque).
    fn blend(dst: &mut Self, src: &Self, cover: u8);
}

#[derive(Clone, Copy, Debug)]
struct Gray64 { v: f64, a: f64 }
#[derive(Clone, Copy, Debug)]
struct Gray32 { v: f32, a: f32 }
#[derive(Clone, Copy, Debug)]
struct Rgba64 { r: f64, g: f64, b: f64, a: f64 }

impl Pixel for Gray64 {
    fn opaque(&self) -> bool { self.a >= 1.0 }
    fn transparent(&self) -> bool { self.a <= 0.0 }
    fn blend(dst: &mut Self, src: &Self, cover: u8) {
        let alpha = src.a * f64::from(cover) / 255.0;
        dst.v = (1.0 - alpha) * dst.v + alpha * src.v;
    }
}

impl Pixel for Gray32 {
    fn opaque(&self) -> bool { self.a >= 1.0 }
    fn transparent(&self) -> bool { self.a <= 0.0 }
    fn blend(dst: &mut Self, src: &Self, cover: u8) {
        let alpha = src.a * f32::from(cover) / 255.0;
        dst.v = (1.0 - alpha) * dst.v + alpha * src.v;
    }
}

impl Pixel for Rgba64 {
    fn opaque(&self) -> bool { self.a >= 1.0 }
    fn transparent(&self) -> bool { self.a <= 0.0 }
    fn blend(p: &mut Self, c: &Self, cover: u8) {
        let alpha = c.a * f64::from(cover) / 255.0;
        if alpha > 0.0 {
            let a = p.a;
            let r = p.r * a;
            let g = p.g * a;
            let b = p.b * a;
            p.r = (1.0 - alpha) * r + alpha * c.r;
            p.g = (1.0 - alpha) * g + alpha * c.g;
            p.b = (1.0 - alpha) * b + alpha * c.b;
            p.a = (1.0 - alpha) * a + alpha;
            let a = p.a;
            p.r = if a == 0.0 { 0.0 } else { p.r / a };
            p.g = if a == 0.0 { 0.0 } else { p.g / a };
            p.b = if a == 0.0 { 0.0 } else { p.b / a };
        }
    }
}

fn mul8(a: u32, b: u32) -> u32 { let t = a * b + 128; ((t >> 8) + t) >> 8 }

fn lerp8(p: u8, q: u8, a: u8) -> u8 {
    let t: i32 = (i32::from(q) - i32::from(p)) * i32::from(a) + 128 - i32::from(p > q);
    (i32::from(p) + (((t >> 8) + t) >> 8)) as u8
}

#[derive(Clone, Copy, Debug)]
struct Gray8 { v: u8, a: u8 }
#[derive(Clone, Copy, Debug)]
struct Rgba8 { r: u8, g: u8, b: u8, a: u8 }

impl Pixel for Gray8 {
    fn opaque(&self) -> bool { self.a == 255 }
    fn transparent(&self) -> bool { self.a == 0 }
    fn blend(dst: &mut Self, src: &Self, cover: u8) {
        let alpha = mul8(u32::from(src.a), u32::from(cover)) as u8;
        dst.v = lerp8(dst.v, src.v, alpha);
    }
}

impl Pixel for Rgba8 {
    fn opaque(&self) -> bool { self.a == 255 }
    fn transparent(&self) -> bool { self.a == 0 }
    /// matplotlib's `fixed_blender_rgba_plain` (unsigned 32-bit arithmetic).
    fn blend(p: &mut Self, c: &Self, cover: u8) {
        let alpha = mul8(u32::from(c.a), u32::from(cover));
        if alpha == 0 { return; }
        let a = u32::from(p.a);
        let r = u32::from(p.r) * a;
        let g = u32::from(p.g) * a;
        let b = u32::from(p.b) * a;
        let a = ((alpha + a) << 8).wrapping_sub(alpha * a);
        p.a = (a >> 8) as u8;
        let ch = |cv: u8, v: u32| ((((u32::from(cv) << 8).wrapping_sub(v)).wrapping_mul(alpha)).wrapping_add(v << 8) / a) as u8;
        p.r = ch(c.r, r);
        p.g = ch(c.g, g);
        p.b = ch(c.b, b);
    }
}

/// The input `_resample` hands `_image.resample`: Agg's coordinates hold
/// fewer than 2**23 columns and 2**24 rows, so larger data is decimated first
/// (`data[:, ::step]`, then `data[::step, :]`, `step = ceil(n / 2**k)`) with
/// the transform pre-scaled to match (`Affine2D().scale(step, 1) +
/// transform`, then `scale(1, step)`).
struct Decimated { cols: usize, rows: usize, row_stride: usize, col_step: usize, trans: M3 }

fn decimate(cols: usize, rows: usize, trans: &M3) -> Decimated {
    let mut d = Decimated { cols, rows, row_stride: cols, col_step: 1, trans: *trans };
    if cols > 1 << 23 {
        let step = (cols as f64 / f64::from(1u32 << 23)).ceil() as usize;
        d.cols = cols.div_ceil(step);
        d.col_step = step;
        d.trans = d.trans.dot(&M3::IDENTITY.scale(step as f64, 1.0));
    }
    if rows > 1 << 24 {
        let step = (rows as f64 / f64::from(1u32 << 24)).ceil() as usize;
        d.rows = rows.div_ceil(step);
        d.row_stride = cols * step;
        d.trans = d.trans.dot(&M3::IDENTITY.scale(1.0, step as f64));
    }
    d
}

/// `resample<color_type>` for `NEAREST` with an affine transform, as a visit:
/// the transformed input rectangle is rasterized into the output and every
/// covered output pixel is reported once, with the source pixel the span
/// interpolator samples for it (its flat index in the undecimated data) and
/// its coverage, in Agg's order. The caller blends (onto the zero-filled
/// output, as the C++ does).
fn nn_visit(input: &Decimated, out_w: usize, out_h: usize, mut visit: impl FnMut(usize, usize, u8)) -> Result<()> {
    let (in_w, in_h, affine) = (input.cols, input.rows, &input.trans);
    let (row_stride, col_step) = (input.row_stride, input.col_step);
    let m = affine.0;
    let fwd = TransAffine { sx: m[0][0], shx: m[0][1], tx: m[0][2], shy: m[1][0], sy: m[1][1], ty: m[1][2] };
    let mut inverted = fwd;
    inverted.invert();
    let mut ras = RasterizerScanlineAa::new(1024);
    ras.clip_box(0.0, 0.0, out_w as f64, out_h as f64);
    let corners = [(0.0, 0.0), (in_w as f64, 0.0), (in_w as f64, in_h as f64), (0.0, in_h as f64)];
    let overflow = |_| crate::Error::Backend("Exceeded cell block limit in image resample".into());
    for (k, &(x, y)) in corners.iter().enumerate() {
        let (mut tx, mut ty) = (x, y);
        fwd.transform(&mut tx, &mut ty);
        if k == 0 { ras.move_to_d(tx, ty).map_err(overflow)?; } else { ras.line_to_d(tx, ty).map_err(overflow)?; }
    }
    ras.close_polygon().map_err(overflow)?;
    let wrap_x = Reflect::new(in_w as u32);
    let wrap_y = Reflect::new(in_h as u32);
    let mut interp = Interpolator { trans: inverted, x: Dda2::default(), y: Dda2::default() };
    let mut sl = ScanlineU8::new();
    if !ras.rewind_scanlines().map_err(overflow)? { return Ok(()); }
    use zlc_agg::rasterizer::ScanlineSink;
    sl.reset(ras.min_x(), ras.max_x());
    let mut sources: Vec<usize> = Vec::new();
    let (clip_x2, clip_y2) = (out_w as i32 - 1, out_h as i32 - 1);
    while ras.sweep_scanline(&mut sl) {
        let y = sl.y();
        let covers = sl.covers();
        for span in sl.spans() {
            let len = span.len.unsigned_abs();
            sources.clear();
            interp.begin(f64::from(span.x) + 0.5, f64::from(y) + 0.5, len);
            for _ in 0..len {
                let (sx, sy) = (interp.x.y >> 8, interp.y.y >> 8);
                sources.push(wrap_y.at(sy) as usize * row_stride + wrap_x.at(sx) as usize * col_step);
                interp.x.inc();
                interp.y.inc();
            }
            // renderer_base::blend_color_hspan clipping to the output.
            if y > clip_y2 || y < 0 { continue; }
            let mut x = span.x;
            let mut n = len as i32;
            let mut first = 0usize;
            if x < 0 {
                let d = -x;
                n -= d;
                if n <= 0 { continue; }
                first = d as usize;
                x = 0;
            }
            if x + n > clip_x2 {
                n = clip_x2 - x + 1;
                if n <= 0 { continue; }
            }
            let cover_base = span.cover + first;
            let row = y as usize * out_w;
            for i in 0..n as usize {
                visit(row + x as usize + i, sources[first + i], covers[cover_base + i]);
            }
        }
    }
    Ok(())
}

/// The output pixel an Agg pixel format leaves for source pixel `c` at
/// coverage `cover`, on the zero-filled output (each pixel is written once).
fn from_zero<P: Pixel>(zero: P, c: P, cover: u8) -> P {
    if c.transparent() { return zero; }
    if c.opaque() && cover == 255 { return c; }
    let mut p = zero;
    P::blend(&mut p, &c, cover);
    p
}

// ---------------------------------------------------------------- make_image
struct Made { image: RgbaImage, x: f64, y: f64 }

/// `_ImageBase._make_image(A, bbox, transformed_bbox, clip, magnification)`.
fn make_image(ctx: &mut Ctx<'_>, im: &Image, magnification: f64) -> Result<Option<Made>> {
    if im.interpolation != "nearest" { return unsupported(format!("image interpolation {}", im.interpolation)); }
    let shape = &im.data.shape;
    let colour = shape.len() == 3;
    if !(shape.len() == 2 || (colour && shape[2] == 4 && im.data.dtype == DType::U8)) {
        return unsupported(format!("image data {:?} {:?}", im.data.dtype, shape));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let trans = ctx.transform(&im.artist)?;
    let [x1, x2, y1, y2] = im.extent;
    let in_bbox: Points = [x1, y1, x2, y2];
    let out_bbox = transformed_points(&trans, &in_bbox);
    let clip_bbox = if im.artist.clip_on {
        match im.artist.clip_box {
            Some(id) => ctx.eval.points(id)?,
            None => return unsupported("clipped image without a clip box"),
        }
    } else {
        ctx.figure_bbox()?
    };
    let Some(clipped) = intersection(&out_bbox, &clip_bbox) else { return Ok(None) };
    let out_width_base = (clipped[2] - clipped[0]) * magnification;
    let out_height_base = (clipped[3] - clipped[1]) * magnification;
    if out_width_base == 0.0 || out_height_base == 0.0 { return Ok(None); }
    if !(out_width_base.is_finite() && out_height_base.is_finite()) {
        // math.ceil of an infinite size raises OverflowError.
        return crate::invalid("an image of infinite output size");
    }
    let t0 = if im.origin == "upper" {
        Some(M3::IDENTITY.translate(0.0, -(rows as f64)).scale(1.0, -1.0))
    } else if im.origin == "lower" {
        None
    } else {
        return unsupported(format!("origin {}", im.origin));
    };
    let s = M3::IDENTITY.scale((in_bbox[2] - in_bbox[0]) / cols as f64, (in_bbox[3] - in_bbox[1]) / rows as f64)
        .translate(in_bbox[0], in_bbox[1]);
    let x = trans.dot(&s);
    let t0 = match t0 { Some(t0) => x.dot(&t0), None => x };
    let a2 = M3::IDENTITY.translate(-clipped[0], -clipped[1]).scale(magnification, magnification);
    let mut t = a2.dot(&t0);
    let (out_width, out_height) = if out_width_base % 1.0 != 0.0 || out_height_base % 1.0 != 0.0 {
        let ow = out_width_base.ceil();
        let oh = out_height_base.ceil();
        let extra_width = (ow - out_width_base) / out_width_base;
        let extra_height = (oh - out_height_base) / out_height_base;
        t = M3::IDENTITY.scale(1.0 + extra_width, 1.0 + extra_height).dot(&t);
        (ow as usize, oh as usize)
    } else {
        (out_width_base as usize, out_height_base as usize)
    };
    let stage = match im.interpolation_stage.as_str() {
        "antialiased" | "auto" => {
            let (ax, ay) = t.apply(0.0, 0.0);
            let (bx, by) = t.apply(cols as f64, rows as f64);
            let dispx = (bx - ax).abs() / cols as f64;
            let dispy = (by - ay).abs() / rows as f64;
            if dispx < 3.0 || dispy < 3.0 { "rgba" } else { "data" }
        }
        "rgba" => "rgba",
        "data" => "data",
        other => return unsupported(format!("interpolation stage {other}")),
    };
    if im.artist.alpha.is_some() { return unsupported("image with an alpha"); }
    // Every `_resample` of this draw decimates alike (the stage above saw
    // the full shape).
    let input = decimate(cols, rows, &t);
    let mask: Option<Vec<bool>> = crate::colors::mask_bits(im.mask.as_ref(), &im.data)?;
    let masked_any = mask.as_ref().is_some_and(|m| m.iter().any(|&b| b));
    let mut out = vec![0u8; out_width * out_height * 4];
    if colour {
        // An RGBA uint8 image: alpha resampled as gray8, colour as rgba8 with alpha 255.
        let px = &im.data.bytes;
        nn_visit(&input, out_width, out_height, |o, s, cover| {
            let a = from_zero(Gray8 { v: 0, a: 0 }, Gray8 { v: px[s * 4 + 3], a: 255 }, cover);
            let c = from_zero(Rgba8 { r: 0, g: 0, b: 0, a: 0 },
                              Rgba8 { r: px[s * 4], g: px[s * 4 + 1], b: px[s * 4 + 2], a: 255 }, cover);
            out[o * 4..o * 4 + 4].copy_from_slice(&[c.r, c.g, c.b, a.v]);
        })?;
    } else if stage == "rgba" {
        // norm.autoscale_None(A) (a no-op with set limits); A = to_rgba(A),
        // computed per sampled pixel; its alpha resampled as gray64, its
        // colour as rgba64 with alpha 1.
        let data = &im.data;
        let precision = process_dtype(im.data.dtype)?;
        let map = crate::colors::PixelMap::new(&im.norm, &im.cmap, precision)?;
        let lut = &im.cmap.lut;
        nn_visit(&input, out_width, out_height, |o, s, cover| {
            let bad = if masked_any { Some(mask.as_ref().is_some_and(|m| m[s])) } else { None };
            let c = lut[map.index(map.normalize(crate::colors::round_to(precision, data.get_f64(s))), bad)];
            let a = from_zero(Gray64 { v: 0.0, a: 0.0 }, Gray64 { v: c[3], a: 1.0 }, cover);
            let p = from_zero(Rgba64 { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, Rgba64 { r: c[0], g: c[1], b: c[2], a: 1.0 }, cover);
            let px = [p.r, p.g, p.b, a.v];
            let nan = px.iter().any(|v| v.is_nan());
            for ch in 0..4 {
                out[o * 4 + ch] = if nan { 0 } else { (px[ch] * 255.0) as u8 };
            }
        })?;
    } else {
        // 'data': the data resampled at float32/64 with its mask, then norm
        // and the colormap's bytes; the resampled mask scales the alpha.
        let data = &im.data;
        let precision = if im.data.dtype == DType::F64 { Precision::F64 } else if im.data.dtype == DType::F32 {
            Precision::F32
        } else {
            let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
            for k in 0..rows * cols {
                if mask.as_ref().is_some_and(|m| m[k]) { continue; }
                let v = data.get_f64(k);
                min = min.min(v);
                max = max.max(v);
            }
            if min > max { return unsupported("fully masked image"); }
            if max - min > 1e8 { Precision::F64 } else { Precision::F32 }
        };
        let map = crate::colors::PixelMap::new(&im.norm, &im.cmap, precision)?;
        // Every output pixel's resampled value and mask (zero where the
        // image does not reach, as the zero-filled outputs say).
        let mut resampled = vec![0.0f64; out_width * out_height];
        let mut mask_out = vec![0.0f32; out_width * out_height];
        nn_visit(&input, out_width, out_height, |o, s, cover| {
            resampled[o] = if precision == Precision::F32 {
                f64::from(from_zero(Gray32 { v: 0.0, a: 0.0 }, Gray32 { v: data.get_f64(s) as f32, a: 1.0 }, cover).v)
            } else {
                from_zero(Gray64 { v: 0.0, a: 0.0 }, Gray64 { v: data.get_f64(s), a: 1.0 }, cover).v
            };
            let m = if mask.as_ref().is_some_and(|m| m[s]) { f32::NAN } else { 1.0 };
            mask_out[o] = from_zero(Gray32 { v: 0.0, a: 0.0 }, Gray32 { v: m, a: 1.0 }, cover).v;
        })?;
        let out_masked_any = mask_out.iter().any(|a| a.is_nan());
        let lut8: Vec<[u8; 4]> = im.cmap.lut.iter().map(|c| c.map(|v| (v * 255.0) as u8)).collect();
        for (k, (&v, &m)) in resampled.iter().zip(&mask_out).enumerate() {
            let bad = if out_masked_any { Some(m.is_nan()) } else { None };
            let mut px = lut8[map.index(map.normalize(v), bad)];
            let out_alpha = if m.is_nan() { 1.0f32 } else { m };
            px[3] = ((f32::from(px[3]) * out_alpha) * 1.0f32) as u8;
            out[k * 4..k * 4 + 4].copy_from_slice(&px);
        }
    }
    Ok(Some(Made { image: RgbaImage { rows: out_height, cols: out_width, data: out }, x: clipped[0], y: clipped[1] }))
}

/// `_make_image(..., unsampled=True)` for a renderer that scales images:
/// the data coloured once (`to_rgba(A, bytes=True)`), cut to the rows and
/// columns the clip box shows, and its pixel grid's map to display space.
///
/// Matplotlib takes this path only for interpolation 'none' and resamples
/// 'nearest' at `magnification` (the export dpi); a nearest-neighbour
/// picture is the same picture at the data's own resolution, without the
/// resampling grid's edges -- unless its pixels are finer than the export's,
/// where the export's resolution is all a reader can see (and the file
/// would only grow): that image is [`Unsampled::Finer`].
fn unsampled_image(ctx: &mut Ctx<'_>, im: &Image, magnification: f64) -> Result<Unsampled> {
    let shape = &im.data.shape;
    let colour = shape.len() == 3;
    if !(shape.len() == 2 || (colour && shape[2] == 4 && im.data.dtype == DType::U8)) {
        return unsupported(format!("image data {:?} {:?}", im.data.dtype, shape));
    }
    if im.artist.alpha.is_some() { return unsupported("image with an alpha"); }
    let (rows, cols) = (shape[0], shape[1]);
    let trans = ctx.transform(&im.artist)?;
    let [x1, x2, y1, y2] = im.extent;
    let in_bbox: Points = [x1, y1, x2, y2];
    let out_bbox = transformed_points(&trans, &in_bbox);
    let clip_bbox = if im.artist.clip_on {
        match im.artist.clip_box {
            Some(id) => ctx.eval.points(id)?,
            None => return unsupported("clipped image without a clip box"),
        }
    } else {
        ctx.figure_bbox()?
    };
    let Some(clipped) = intersection(&out_bbox, &clip_bbox) else { return Ok(Unsampled::Nothing) };
    if clipped[2] - clipped[0] == 0.0 || clipped[3] - clipped[1] == 0.0 { return Ok(Unsampled::Nothing); }
    let flip = match im.origin.as_str() {
        "upper" => M3::IDENTITY.translate(0.0, -(rows as f64)).scale(1.0, -1.0),
        "lower" => M3::IDENTITY,
        other => return unsupported(format!("origin {other}")),
    };
    let s = M3::IDENTITY.scale((in_bbox[2] - in_bbox[0]) / cols as f64, (in_bbox[3] - in_bbox[1]) / rows as f64)
        .translate(in_bbox[0], in_bbox[1]);
    let t0 = trans.dot(&s).dot(&flip);
    let (ox, oy) = t0.apply(0.0, 0.0);
    let ((cx, cy), (rx, ry)) = (t0.apply(1.0, 0.0), t0.apply(0.0, 1.0));
    if (cx - ox).hypot(cy - oy) * magnification < 1.0 || (rx - ox).hypot(ry - oy) * magnification < 1.0 { return Ok(Unsampled::Finer); }
    // The part the clip box shows: `TransformedBbox(clip_bbox, t0.inverted())`.
    let inverse = invert(&t0)?;
    let subset = transformed_points(&inverse, &clip_bbox);
    let (sx0, sx1) = (subset[0].min(subset[2]), subset[0].max(subset[2]));
    let (sy0, sy1) = (subset[1].min(subset[3]), subset[1].max(subset[3]));
    let r0 = sy0.max(0.0) as usize;
    let r1 = ((sy1 + 1.0).min(rows as f64)).max(0.0) as usize;
    let c0 = sx0.max(0.0) as usize;
    let c1 = ((sx1 + 1.0).min(cols as f64)).max(0.0) as usize;
    if r0 >= r1 || c0 >= c1 { return Ok(Unsampled::Nothing); }
    let (h, w) = (r1 - r0, c1 - c0);
    let mut out = vec![0u8; w * h * 4];
    if colour {
        let px = &im.data.bytes;
        for r in 0..h {
            let from = ((r0 + r) * cols + c0) * 4;
            out[r * w * 4..(r + 1) * w * 4].copy_from_slice(&px[from..from + w * 4]);
        }
    } else {
        let precision = process_dtype(im.data.dtype)?;
        let map = crate::colors::PixelMap::new(&im.norm, &im.cmap, precision)?;
        let mask: Option<Vec<bool>> = crate::colors::mask_bits(im.mask.as_ref(), &im.data)?;
        let masked_any = mask.as_ref().is_some_and(|m| m.iter().any(|&b| b));
        let lut8: Vec<[u8; 4]> = im.cmap.lut.iter().map(|c| c.map(|v| (v * 255.0) as u8)).collect();
        for r in 0..h {
            for c in 0..w {
                let s = (r0 + r) * cols + c0 + c;
                let bad = if masked_any { Some(mask.as_ref().is_some_and(|m| m[s])) } else { None };
                let px = lut8[map.index(map.normalize(crate::colors::round_to(precision, im.data.get_f64(s))), bad)];
                out[(r * w + c) * 4..(r * w + c) * 4 + 4].copy_from_slice(&px);
            }
        }
    }
    Ok(Unsampled::Image(RgbaImage { rows: h, cols: w, data: out }, t0.dot(&M3::IDENTITY.translate(c0 as f64, r0 as f64))))
}

enum Unsampled {
    Nothing,
    /// The data's pixels are finer than the export's.
    Finer,
    /// The image and its pixel grid's map to display space.
    Image(RgbaImage, M3),
}

/// The inverse of an affine matrix.
fn invert(m: &M3) -> Result<M3> {
    let [a, b, c, d, e, f] = m.to6();
    let det = a * e - b * d;
    if det == 0.0 || !det.is_finite() { return crate::invalid("a singular image transform"); }
    Ok(M3::from6([e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det]))
}

/// `AxesImage.draw(renderer)`.
pub fn draw_image(ctx: &mut Ctx<'_>, im: &Image) -> Result<()> {
    if !im.artist.visible { return Ok(()); }
    if im.data.is_empty() { return Ok(()); }
    let mut gc = ctx.new_gc();
    ctx.set_gc_clip(&mut gc, &im.artist)?;
    gc.set_alpha(Some(im.artist.alpha.unwrap_or(1.0)));
    let magnification = ctx.backend.image_magnification();
    if ctx.vector {
        match unsampled_image(ctx, im, magnification)? {
            Unsampled::Nothing => return Ok(()),
            Unsampled::Image(image, pixel_to_display) => return ctx.backend.draw_image_transformed(&gc, &image, &pixel_to_display),
            Unsampled::Finer => {}
        }
    }
    if let Some(made) = make_image(ctx, im, magnification)? {
        ctx.backend.draw_image(&gc, made.x, made.y, &made.image)?;
    }
    Ok(())
}
