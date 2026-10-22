//! Scanline containers: `scanline32_p8`, `scanline32_u8`, `scanline32_bin`
//! and `scanline_storage_aa8`, with Agg's span merging rules.
use crate::rasterizer::ScanlineSink;

/// One span: `len < 0` is a solid run whose single cover is `covers[cover]`.
#[derive(Clone, Copy, Debug)]
pub struct Span { pub x: i32, pub len: i32, pub cover: usize }

/// Read access shared by the scanline kinds that renderers consume.
pub trait SpanScanline {
    fn y(&self) -> i32;
    fn spans(&self) -> &[Span];
    fn covers(&self) -> &[u8];
}

#[derive(Default)]
pub struct ScanlineP8 { last_x: i32, y: i32, covers: Vec<u8>, spans: Vec<Span> }

impl ScanlineP8 {
    pub fn new() -> Self { Self { last_x: 0x7FFFFFF0, ..Default::default() } }
}

impl ScanlineSink for ScanlineP8 {
    fn reset(&mut self, _min_x: i32, _max_x: i32) { self.reset_spans(); }
    fn reset_spans(&mut self) {
        self.last_x = 0x7FFFFFF0;
        self.covers.clear();
        self.spans.clear();
    }
    #[inline]
    fn add_cell(&mut self, x: i32, cover: u32) {
        let index = self.covers.len();
        self.covers.push(cover as u8);
        match self.spans.last_mut() {
            Some(last) if x == self.last_x.wrapping_add(1) && last.len > 0 => last.len += 1,
            _ => self.spans.push(Span { x, len: 1, cover: index }),
        }
        self.last_x = x;
    }
    #[inline]
    fn add_span(&mut self, x: i32, len: u32, cover: u32) {
        let merge = match self.spans.last() {
            Some(last) => x == self.last_x.wrapping_add(1) && last.len < 0 && cover == u32::from(self.covers[last.cover]),
            None => false,
        };
        if merge {
            self.spans.last_mut().unwrap().len -= len as i32;
        } else {
            let index = self.covers.len();
            self.covers.push(cover as u8);
            self.spans.push(Span { x, len: -(len as i32), cover: index });
        }
        self.last_x = x + len as i32 - 1;
    }
    fn num_spans(&self) -> usize { self.spans.len() }
    fn finalize(&mut self, y: i32) { self.y = y; }
}

impl SpanScanline for ScanlineP8 {
    fn y(&self) -> i32 { self.y }
    fn spans(&self) -> &[Span] { &self.spans }
    fn covers(&self) -> &[u8] { &self.covers }
}

/// `scanline32_u8` (and `scanline32_u8_am` without a mask, as Matplotlib
/// constructs it): covers are indexed by x relative to the sweep's minimum.
#[derive(Default)]
pub struct ScanlineU8 { min_x: i32, last_x: i32, y: i32, covers: Vec<u8>, spans: Vec<Span> }

impl ScanlineU8 {
    pub fn new() -> Self { Self { last_x: 0x7FFFFFF0, ..Default::default() } }
}

impl ScanlineSink for ScanlineU8 {
    fn reset(&mut self, min_x: i32, max_x: i32) {
        let max_len = (max_x.wrapping_sub(min_x).wrapping_add(2)) as u32 as usize;
        if max_len > self.covers.len() { self.covers.resize(max_len, 0); }
        self.last_x = 0x7FFFFFF0;
        self.min_x = min_x;
        self.spans.clear();
    }
    fn reset_spans(&mut self) {
        self.last_x = 0x7FFFFFF0;
        self.spans.clear();
    }
    #[inline]
    fn add_cell(&mut self, x: i32, cover: u32) {
        let x = x - self.min_x;
        self.covers[x as usize] = cover as u8;
        if x == self.last_x.wrapping_add(1) {
            self.spans.last_mut().unwrap().len += 1;
        } else {
            self.spans.push(Span { x: x + self.min_x, len: 1, cover: x as usize });
        }
        self.last_x = x;
    }
    #[inline]
    fn add_span(&mut self, x: i32, len: u32, cover: u32) {
        let x = x - self.min_x;
        self.covers[x as usize..x as usize + len as usize].fill(cover as u8);
        if x == self.last_x.wrapping_add(1) {
            self.spans.last_mut().unwrap().len += len as i32;
        } else {
            self.spans.push(Span { x: x + self.min_x, len: len as i32, cover: x as usize });
        }
        self.last_x = x + len as i32 - 1;
    }
    fn num_spans(&self) -> usize { self.spans.len() }
    fn finalize(&mut self, y: i32) { self.y = y; }
}

impl SpanScanline for ScanlineU8 {
    fn y(&self) -> i32 { self.y }
    fn spans(&self) -> &[Span] { &self.spans }
    fn covers(&self) -> &[u8] { &self.covers }
}

/// `scanline32_bin`: spans without covers; every span length is positive.
#[derive(Default)]
pub struct ScanlineBin { last_x: i32, y: i32, spans: Vec<Span> }

impl ScanlineBin {
    pub fn new() -> Self { Self { last_x: 0x7FFFFFF0, ..Default::default() } }
}

impl ScanlineSink for ScanlineBin {
    fn reset(&mut self, _min_x: i32, _max_x: i32) { self.reset_spans(); }
    fn reset_spans(&mut self) {
        self.last_x = 0x7FFFFFF0;
        self.spans.clear();
    }
    #[inline]
    fn add_cell(&mut self, x: i32, _cover: u32) {
        if x == self.last_x.wrapping_add(1) {
            self.spans.last_mut().unwrap().len += 1;
        } else {
            self.spans.push(Span { x, len: 1, cover: 0 });
        }
        self.last_x = x;
    }
    #[inline]
    fn add_span(&mut self, x: i32, len: u32, _cover: u32) {
        if x == self.last_x.wrapping_add(1) {
            self.spans.last_mut().unwrap().len += len as i32;
        } else {
            self.spans.push(Span { x, len: len as i32, cover: 0 });
        }
        self.last_x = x + len as i32 - 1;
    }
    fn num_spans(&self) -> usize { self.spans.len() }
    fn finalize(&mut self, y: i32) { self.y = y; }
}

impl SpanScanline for ScanlineBin {
    fn y(&self) -> i32 { self.y }
    fn spans(&self) -> &[Span] { &self.spans }
    fn covers(&self) -> &[u8] { &[] }
}

#[derive(Clone, Copy, Debug)]
pub struct StoredScanline { pub y: i32, pub first_span: usize, pub num_spans: usize }

/// `scanline_storage_aa8`. A solid span keeps its one cover; a cell run its
/// covers. Bounds start at Agg's sentinels and widen with every span.
pub struct ScanlineStorageAa8 {
    pub covers: Vec<u8>,
    pub spans: Vec<Span>,
    pub scanlines: Vec<StoredScanline>,
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

impl Default for ScanlineStorageAa8 {
    fn default() -> Self { Self::new() }
}

impl ScanlineStorageAa8 {
    pub fn new() -> Self {
        Self { covers: Vec::new(), spans: Vec::new(), scanlines: Vec::new(),
               min_x: 0x7FFFFFFF, min_y: 0x7FFFFFFF, max_x: -0x7FFFFFFF, max_y: -0x7FFFFFFF }
    }

    pub fn prepare(&mut self) {
        self.covers.clear();
        self.spans.clear();
        self.scanlines.clear();
        self.min_x = 0x7FFFFFFF;
        self.min_y = 0x7FFFFFFF;
        self.max_x = -0x7FFFFFFF;
        self.max_y = -0x7FFFFFFF;
    }

    pub fn render<S: SpanScanline>(&mut self, sl: &S) {
        let y = sl.y();
        if y < self.min_y { self.min_y = y; }
        if y > self.max_y { self.max_y = y; }
        let first_span = self.spans.len();
        let covers = sl.covers();
        for span in sl.spans() {
            let len = span.len.wrapping_abs();
            let start = self.covers.len();
            if span.len < 0 {
                self.covers.push(covers[span.cover]);
            } else {
                self.covers.extend_from_slice(&covers[span.cover..span.cover + len as usize]);
            }
            self.spans.push(Span { x: span.x, len: span.len, cover: start });
            let x1 = span.x;
            let x2 = span.x + len - 1;
            if x1 < self.min_x { self.min_x = x1; }
            if x2 > self.max_x { self.max_x = x2; }
        }
        self.scanlines.push(StoredScanline { y, first_span, num_spans: sl.spans().len() });
    }
}
