//! Pixel formats and `renderer_base`: Matplotlib's plain RGBA32 blender, the
//! gray8 alpha-mask surface and `pixfmt_amask_adaptor`, plus the scanline
//! renderers that drive them.
use crate::basics::*;
use crate::color::*;
use crate::rasterizer::{CellBlockLimit, RasterizerScanlineAa, ScanlineSink};
use crate::scanline::SpanScanline;

pub trait PixFmt {
    type Color: Copy;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Self::Color);
    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Self::Color, cover: u8);
    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Self::Color, covers: &[u8]);
    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Self::Color], covers: Option<&[u8]>, cover: u8);
}

/// Store one colour into every pixel of a row run.
#[inline]
fn fill_pixels(row: &mut [u8], c: &Rgba8) {
    let value = u32::from_ne_bytes([c.r, c.g, c.b, c.a]);
    // Every byte pattern is a valid u32; unaligned ends take the bytewise loop.
    let (head, words, tail) = unsafe { row.align_to_mut::<u32>() };
    if head.is_empty() && tail.is_empty() {
        words.fill(value);
    } else {
        for p in row.chunks_exact_mut(4) { p.copy_from_slice(&[c.r, c.g, c.b, c.a]); }
    }
}

/// `pixfmt_alpha_blend_rgba<fixed_blender_rgba_plain>` over a positive-stride buffer.
pub struct PixfmtRgba32<'a> { pub buf: &'a mut [u8], pub width: u32, pub height: u32 }

impl<'a> PixfmtRgba32<'a> {
    pub fn new(buf: &'a mut [u8], width: u32, height: u32) -> Self { Self { buf, width, height } }

    #[inline(always)]
    fn offset(&self, x: i32, y: i32) -> usize { (y as usize * self.width as usize + x as usize) * 4 }

    #[inline(always)]
    fn copy_or_blend_pix(p: &mut [u8], c: &Rgba8) {
        if !c.is_transparent() {
            if c.is_opaque() { p.copy_from_slice(&[c.r, c.g, c.b, c.a]); }
            else { blend_pix_plain(p, c.r, c.g, c.b, c.a); }
        }
    }

    #[inline(always)]
    fn copy_or_blend_pix_cover(p: &mut [u8], c: &Rgba8, cover: u8) {
        if !c.is_transparent() {
            if c.is_opaque() && cover == COVER_FULL { p.copy_from_slice(&[c.r, c.g, c.b, c.a]); }
            else { blend_pix_plain_cover(p, c, cover); }
        }
    }

    /// `copy_from`: one row segment from another RGBA buffer.
    pub fn copy_from(&mut self, src: &[u8], src_width: u32, xdst: i32, ydst: i32, xsrc: i32, ysrc: i32, len: u32) {
        let d = self.offset(xdst, ydst);
        let s = (ysrc as usize * src_width as usize + xsrc as usize) * 4;
        self.buf[d..d + len as usize * 4].copy_from_slice(&src[s..s + len as usize * 4]);
    }

    /// `blend_from` an RGBA source image row segment.
    pub fn blend_from(&mut self, src: &ImageRgba<'_>, xdst: i32, ydst: i32, xsrc: i32, ysrc: i32, len: u32, cover: u8) {
        let mut ps = src.offset(xsrc, ysrc);
        let mut pd = self.offset(xdst, ydst) as isize;
        let (mut sinc, mut dinc) = (4isize, 4isize);
        if xdst > xsrc {
            ps += (len as isize - 1) * 4;
            pd += (len as isize - 1) * 4;
            sinc = -4;
            dinc = -4;
        }
        for _ in 0..len {
            let s = ps as usize;
            let c = Rgba8::new(src.data[s], src.data[s + 1], src.data[s + 2], src.data[s + 3]);
            let p = &mut self.buf[pd as usize..pd as usize + 4];
            if cover == COVER_FULL { Self::copy_or_blend_pix(p, &c); } else { Self::copy_or_blend_pix_cover(p, &c, cover); }
            ps += sinc;
            pd += dinc;
        }
    }
}

impl PixFmt for PixfmtRgba32<'_> {
    type Color = Rgba8;
    fn width(&self) -> u32 { self.width }
    fn height(&self) -> u32 { self.height }

    #[inline]
    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        let o = self.offset(x, y);
        fill_pixels(&mut self.buf[o..o + len as usize * 4], c);
    }

    #[inline]
    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: u8) {
        if c.is_transparent() { return; }
        let o = self.offset(x, y);
        let row = &mut self.buf[o..o + len as usize * 4];
        if c.is_opaque() && cover == COVER_FULL {
            fill_pixels(row, c);
        } else if cover == COVER_FULL {
            for p in row.chunks_exact_mut(4) { blend_pix_plain(p, c.r, c.g, c.b, c.a); }
        } else {
            for p in row.chunks_exact_mut(4) { blend_pix_plain_cover(p, c, cover); }
        }
    }

    #[inline]
    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]) {
        if c.is_transparent() { return; }
        let o = self.offset(x, y);
        let row = &mut self.buf[o..o + len as usize * 4];
        for (p, &cover) in row.chunks_exact_mut(4).zip(&covers[..len as usize]) {
            if c.is_opaque() && cover == COVER_FULL { p.copy_from_slice(&[c.r, c.g, c.b, c.a]); }
            else { blend_pix_plain_cover(p, c, cover); }
        }
    }

    #[inline]
    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        let o = self.offset(x, y);
        let row = &mut self.buf[o..o + len as usize * 4];
        let colors = &colors[..len as usize];
        match covers {
            Some(covers) => for ((p, c), &k) in row.chunks_exact_mut(4).zip(colors).zip(covers) {
                Self::copy_or_blend_pix_cover(p, c, k);
            },
            None if cover == COVER_FULL => for (p, c) in row.chunks_exact_mut(4).zip(colors) {
                Self::copy_or_blend_pix(p, c);
            },
            None => for (p, c) in row.chunks_exact_mut(4).zip(colors) {
                Self::copy_or_blend_pix_cover(p, c, cover);
            },
        }
    }
}

/// A read-only RGBA32 image, possibly attached with a negative stride.
pub struct ImageRgba<'a> { pub data: &'a [u8], pub width: u32, pub height: u32, pub stride: isize }

impl ImageRgba<'_> {
    #[inline(always)]
    pub fn offset(&self, x: i32, y: i32) -> isize {
        let start = if self.stride < 0 { -(self.height as isize - 1) * self.stride } else { 0 };
        start + y as isize * self.stride + x as isize * 4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gray8 { pub v: u8, pub a: u8 }

/// `pixfmt_gray8` with `blender_gray`.
pub struct PixfmtGray8<'a> { pub buf: &'a mut [u8], pub width: u32, pub height: u32 }

impl PixfmtGray8<'_> {
    #[inline(always)]
    fn blend_pix(p: &mut u8, c: &Gray8, cover: u8) { *p = lerp(*p, c.v, multiply(c.a, cover)); }
}

impl PixFmt for PixfmtGray8<'_> {
    type Color = Gray8;
    fn width(&self) -> u32 { self.width }
    fn height(&self) -> u32 { self.height }

    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Gray8) {
        let o = y as usize * self.width as usize + x as usize;
        self.buf[o..o + len as usize].fill(c.v);
    }

    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Gray8, cover: u8) {
        if c.a == 0 { return; }
        let o = y as usize * self.width as usize + x as usize;
        let row = &mut self.buf[o..o + len as usize];
        if c.a == 255 && cover == COVER_FULL { row.fill(c.v); }
        else { for p in row { Self::blend_pix(p, c, cover); } }
    }

    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Gray8, covers: &[u8]) {
        if c.a == 0 { return; }
        let o = y as usize * self.width as usize + x as usize;
        for (p, &cover) in self.buf[o..o + len as usize].iter_mut().zip(covers) {
            if c.a == 255 && cover == COVER_FULL { *p = c.v; } else { Self::blend_pix(p, c, cover); }
        }
    }

    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Gray8], covers: Option<&[u8]>, cover: u8) {
        let o = y as usize * self.width as usize + x as usize;
        for (i, p) in self.buf[o..o + len as usize].iter_mut().enumerate() {
            let c = &colors[i];
            let k = covers.map_or(cover, |covers| covers[i]);
            if c.a != 0 {
                if c.a == 255 && k == COVER_FULL { *p = c.v; } else { Self::blend_pix(p, c, k); }
            }
        }
    }
}

/// `pixfmt_amask_adaptor<pixfmt, amask_no_clip_gray8>`. Like Agg, its
/// `blend_hline` ignores the given cover and applies only the mask.
pub struct AmaskAdaptor<'a, 'm> { pub pixf: PixfmtRgba32<'a>, pub mask: &'m [u8], pub span: Vec<u8> }

impl<'a, 'm> AmaskAdaptor<'a, 'm> {
    pub fn new(pixf: PixfmtRgba32<'a>, mask: &'m [u8]) -> Self { Self { pixf, mask, span: Vec::new() } }

    #[inline(always)]
    fn mask_row(&self, x: i32, y: i32, len: u32) -> &[u8] {
        let o = y as usize * self.pixf.width as usize + x as usize;
        &self.mask[o..o + len as usize]
    }

    #[inline(always)]
    fn combine(span: &mut [u8], mask: &[u8]) {
        for (d, &m) in span.iter_mut().zip(mask) { *d = ((255 + u32::from(*d) * u32::from(m)) >> 8) as u8; }
    }
}

impl PixFmt for AmaskAdaptor<'_, '_> {
    type Color = Rgba8;
    fn width(&self) -> u32 { self.pixf.width }
    fn height(&self) -> u32 { self.pixf.height }

    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        let mut span = std::mem::take(&mut self.span);
        span.clear();
        span.extend_from_slice(self.mask_row(x, y, len));
        self.pixf.blend_solid_hspan(x, y, len, c, &span);
        self.span = span;
    }

    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, _cover: u8) {
        let mut span = std::mem::take(&mut self.span);
        span.clear();
        span.resize(len as usize, 255);
        Self::combine(&mut span, self.mask_row(x, y, len));
        self.pixf.blend_solid_hspan(x, y, len, c, &span);
        self.span = span;
    }

    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]) {
        let mut span = std::mem::take(&mut self.span);
        span.clear();
        span.extend_from_slice(&covers[..len as usize]);
        Self::combine(&mut span, self.mask_row(x, y, len));
        self.pixf.blend_solid_hspan(x, y, len, c, &span);
        self.span = span;
    }

    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        let mut span = std::mem::take(&mut self.span);
        span.clear();
        match covers {
            Some(covers) => {
                span.extend_from_slice(&covers[..len as usize]);
                Self::combine(&mut span, self.mask_row(x, y, len));
            }
            None => span.extend_from_slice(self.mask_row(x, y, len)),
        }
        self.pixf.blend_color_hspan(x, y, len, colors, Some(&span), cover);
        self.span = span;
    }
}

/// `renderer_base`: clipping to an inclusive box before the pixel format.
pub struct RendererBase<P> { pub ren: P, pub clip: RectI }

impl<P: PixFmt> RendererBase<P> {
    /// `renderer_base(pixfmt&)`: the whole surface.
    pub fn new(ren: P) -> Self {
        let clip = RectI::new(0, 0, ren.width() as i32 - 1, ren.height() as i32 - 1);
        Self { ren, clip }
    }

    pub fn with_clip(ren: P, clip: RectI) -> Self { Self { ren, clip } }

    pub fn reset_clipping(&mut self, visibility: bool) {
        self.clip = if visibility {
            RectI::new(0, 0, self.ren.width() as i32 - 1, self.ren.height() as i32 - 1)
        } else {
            RectI::new(1, 1, 0, 0)
        };
    }

    pub fn clip_box(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) -> bool {
        let mut cb = RectI::new(x1, y1, x2, y2);
        cb.normalize();
        if cb.clip(&RectI::new(0, 0, self.ren.width() as i32 - 1, self.ren.height() as i32 - 1)) {
            self.clip = cb;
            return true;
        }
        self.clip = RectI::new(1, 1, 0, 0);
        false
    }

    pub fn clear(&mut self, c: &P::Color) {
        let w = self.ren.width();
        if w != 0 {
            for y in 0..self.ren.height() { self.ren.copy_hline(0, y as i32, w, c); }
        }
    }

    #[inline]
    pub fn blend_hline(&mut self, mut x1: i32, y: i32, mut x2: i32, c: &P::Color, cover: u8) {
        if x1 > x2 { std::mem::swap(&mut x1, &mut x2); }
        let b = self.clip;
        if y > b.y2 || y < b.y1 || x1 > b.x2 || x2 < b.x1 { return; }
        if x1 < b.x1 { x1 = b.x1; }
        if x2 > b.x2 { x2 = b.x2; }
        self.ren.blend_hline(x1, y, (x2 - x1 + 1) as u32, c, cover);
    }

    #[inline]
    pub fn blend_solid_hspan(&mut self, mut x: i32, y: i32, mut len: i32, c: &P::Color, mut covers: &[u8]) {
        let b = self.clip;
        if y > b.y2 || y < b.y1 { return; }
        if x < b.x1 {
            len -= b.x1 - x;
            if len <= 0 { return; }
            covers = &covers[(b.x1 - x) as usize..];
            x = b.x1;
        }
        if x + len > b.x2 {
            len = b.x2 - x + 1;
            if len <= 0 { return; }
        }
        self.ren.blend_solid_hspan(x, y, len as u32, c, covers);
    }

    #[inline]
    pub fn blend_color_hspan(&mut self, mut x: i32, y: i32, mut len: i32, mut colors: &[P::Color],
                             mut covers: Option<&[u8]>, cover: u8) {
        let b = self.clip;
        if y > b.y2 || y < b.y1 { return; }
        if x < b.x1 {
            let d = b.x1 - x;
            len -= d;
            if len <= 0 { return; }
            covers = covers.map(|covers| &covers[d as usize..]);
            colors = &colors[d as usize..];
            x = b.x1;
        }
        if x + len > b.x2 {
            len = b.x2 - x + 1;
            if len <= 0 { return; }
        }
        self.ren.blend_color_hspan(x, y, len as u32, colors, covers, cover);
    }

    /// `clip_rect_area`, for copies between surfaces.
    pub fn clip_rect_area(&self, dst: &mut RectI, src: &mut RectI, wsrc: i32, hsrc: i32) -> RectI {
        let mut rc = RectI::new(0, 0, 0, 0);
        let mut cb = self.clip;
        cb.x2 += 1;
        cb.y2 += 1;
        if src.x1 < 0 { dst.x1 -= src.x1; src.x1 = 0; }
        if src.y1 < 0 { dst.y1 -= src.y1; src.y1 = 0; }
        if src.x2 > wsrc { src.x2 = wsrc; }
        if src.y2 > hsrc { src.y2 = hsrc; }
        if dst.x1 < cb.x1 { src.x1 += cb.x1 - dst.x1; dst.x1 = cb.x1; }
        if dst.y1 < cb.y1 { src.y1 += cb.y1 - dst.y1; dst.y1 = cb.y1; }
        if dst.x2 > cb.x2 { dst.x2 = cb.x2; }
        if dst.y2 > cb.y2 { dst.y2 = cb.y2; }
        rc.x2 = dst.x2 - dst.x1;
        rc.y2 = dst.y2 - dst.y1;
        if rc.x2 > src.x2 - src.x1 { rc.x2 = src.x2 - src.x1; }
        if rc.y2 > src.y2 - src.y1 { rc.y2 = src.y2 - src.y1; }
        rc
    }
}

impl<'a> RendererBase<PixfmtRgba32<'a>> {
    /// `renderer_base::copy_from` a rendering buffer of `src_width` × `src_height`.
    pub fn copy_from(&mut self, src: &[u8], src_width: u32, src_height: u32, rect_src: Option<&RectI>, dx: i32, dy: i32) {
        let mut rsrc = RectI::new(0, 0, src_width as i32, src_height as i32);
        if let Some(r) = rect_src {
            rsrc.x1 = r.x1;
            rsrc.y1 = r.y1;
            rsrc.x2 = r.x2 + 1;
            rsrc.y2 = r.y2 + 1;
        }
        let mut rdst = RectI::new(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);
        let mut rc = self.clip_rect_area(&mut rdst, &mut rsrc, src_width as i32, src_height as i32);
        if rc.x2 > 0 {
            let mut incy = 1;
            if rdst.y1 > rsrc.y1 {
                rsrc.y1 += rc.y2 - 1;
                rdst.y1 += rc.y2 - 1;
                incy = -1;
            }
            while rc.y2 > 0 {
                self.ren.copy_from(src, src_width, rdst.x1, rdst.y1, rsrc.x1, rsrc.y1, rc.x2 as u32);
                rdst.y1 += incy;
                rsrc.y1 += incy;
                rc.y2 -= 1;
            }
        }
    }

    /// `renderer_base::blend_from` with no source rectangle.
    pub fn blend_from(&mut self, src: &ImageRgba<'_>, dx: i32, dy: i32, cover: u8) {
        let mut rsrc = RectI::new(0, 0, src.width as i32, src.height as i32);
        let mut rdst = RectI::new(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);
        let mut rc = self.clip_rect_area(&mut rdst, &mut rsrc, src.width as i32, src.height as i32);
        if rc.x2 > 0 {
            let mut incy = 1;
            if rdst.y1 > rsrc.y1 {
                rsrc.y1 += rc.y2 - 1;
                rdst.y1 += rc.y2 - 1;
                incy = -1;
            }
            while rc.y2 > 0 {
                // row(): x1 = 0, x2 = width - 1.
                let mut x1src = rsrc.x1;
                let mut x1dst = rdst.x1;
                let mut len = rc.x2;
                if 0 > x1src {
                    x1dst -= x1src;
                    len += x1src;
                    x1src = 0;
                }
                if len > 0 {
                    let rw_x2 = src.width as i32 - 1;
                    if x1src + len - 1 > rw_x2 { len -= x1src + len - rw_x2 - 1; }
                    if len > 0 { self.ren.blend_from(src, x1dst, rdst.y1, x1src, rsrc.y1, len as u32, cover); }
                }
                rdst.y1 += incy;
                rsrc.y1 += incy;
                rc.y2 -= 1;
            }
        }
    }
}

/// A color source for `render_scanlines_aa`.
pub trait SpanGenerator {
    fn prepare(&mut self);
    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32, len: u32);
}

#[inline]
pub fn render_scanline_aa_solid<P: PixFmt, S: SpanScanline>(sl: &S, ren: &mut RendererBase<P>, color: &P::Color) {
    let y = sl.y();
    let covers = sl.covers();
    for span in sl.spans() {
        if span.len > 0 {
            ren.blend_solid_hspan(span.x, y, span.len, color, &covers[span.cover..]);
        } else {
            ren.blend_hline(span.x, y, span.x - span.len - 1, color, covers[span.cover]);
        }
    }
}

#[inline]
pub fn render_scanline_bin_solid<P: PixFmt, S: SpanScanline>(sl: &S, ren: &mut RendererBase<P>, color: &P::Color) {
    let y = sl.y();
    for span in sl.spans() {
        ren.blend_hline(span.x, y, span.x - 1 + span.len.wrapping_abs(), color, COVER_FULL);
    }
}

pub fn render_scanlines_aa_solid<P, S>(ras: &mut RasterizerScanlineAa, sl: &mut S, ren: &mut RendererBase<P>,
                                       color: &P::Color) -> Result<(), CellBlockLimit>
where P: PixFmt, S: ScanlineSink + SpanScanline {
    if ras.rewind_scanlines()? {
        sl.reset(ras.min_x(), ras.max_x());
        while ras.sweep_scanline(sl) { render_scanline_aa_solid(sl, ren, color); }
    }
    Ok(())
}

pub fn render_scanlines_bin_solid<P, S>(ras: &mut RasterizerScanlineAa, sl: &mut S, ren: &mut RendererBase<P>,
                                        color: &P::Color) -> Result<(), CellBlockLimit>
where P: PixFmt, S: ScanlineSink + SpanScanline {
    if ras.rewind_scanlines()? {
        sl.reset(ras.min_x(), ras.max_x());
        while ras.sweep_scanline(sl) { render_scanline_bin_solid(sl, ren, color); }
    }
    Ok(())
}

pub fn render_scanlines_aa<P, S, G>(ras: &mut RasterizerScanlineAa, sl: &mut S, ren: &mut RendererBase<P>,
                                    colors: &mut Vec<Rgba8>, span_gen: &mut G) -> Result<(), CellBlockLimit>
where P: PixFmt<Color = Rgba8>, S: ScanlineSink + SpanScanline, G: SpanGenerator {
    if ras.rewind_scanlines()? {
        sl.reset(ras.min_x(), ras.max_x());
        span_gen.prepare();
        while ras.sweep_scanline(sl) {
            let y = sl.y();
            let covers = sl.covers();
            for span in sl.spans() {
                let len = span.len.wrapping_abs();
                if colors.len() < len as usize { colors.resize(len as usize, Rgba8::default()); }
                span_gen.generate(&mut colors[..len as usize], span.x, y, len as u32);
                let span_covers = if span.len < 0 { None } else { Some(&covers[span.cover..]) };
                ren.blend_color_hspan(span.x, y, len, colors, span_covers, covers[span.cover]);
            }
        }
    }
    Ok(())
}
