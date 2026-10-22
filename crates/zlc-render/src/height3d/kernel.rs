//! The per-column scanline materializer (`_height3d_scanline._materialize`).
//!
//! Every pixel column shares one footprint line through the ground grid, so
//! a column is one front-to-back walk: crossings merged in closed form, one
//! running silhouette for occlusion, span records with float bounds, and a
//! float32 prefix walk down the rows. Dtypes, conversion points and the
//! order of every sum follow the numba kernel (and through it the numpy
//! reference) operation for operation. Columns are independent, so output
//! column blocks run on the crate pool; each block walks its columns into a
//! tile and lands it row by row.

use rayon::prelude::*;

use super::pymath::{py_max, trunc_i64};

/// The per-render-column footprint line, as `render_height_bars` builds it.
pub(crate) struct Columns {
    pub a_at: Vec<f64>,
    pub t_enter: Vec<f64>,
    pub t_exit: Vec<f64>,
    pub ia0: Vec<f64>,
    pub ib0: Vec<f64>,
    pub t_a0: Vec<f64>,
    pub t_b0: Vec<f64>,
    pub g0: Vec<f32>,
    /// `k / sa` and `m / ca`: the same quotient in every column's merge.
    pub k_over_sa: Vec<f64>,
    pub m_over_ca: Vec<f64>,
}

/// One folded cell's elevation-scaled planes and validity.
#[derive(Clone, Copy, Default)]
pub(crate) struct CellZ {
    pub top: f32,
    pub bot: f32,
    pub finite: bool,
}

pub(crate) struct Kernel<'a> {
    pub columns: &'a Columns,
    pub cells: &'a [CellZ],
    pub rgb: &'a [[f32; 3]],
    /// The zero-end colour of every side face; `None` is the bar's own.
    pub base: Option<[f32; 3]>,
    pub nx: usize,
    pub ny: usize,
    pub shade_x: f32,
    pub shade_y: f32,
    pub ca: f64,
    pub se32: f32,
    pub y_high: f64,
    pub scale: f64,
    pub pane_high32: f32,
    pub pane_low32: f32,
    pub background32: [f32; 3],
    pub taps: usize,
    pub render_h: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub mid_tap: usize,
}

/// One span: its partial-pixel rows and fractions, and its 255-scaled
/// gradient (anchor at the midpoint, slope, anchor).
#[derive(Clone, Copy, Default)]
struct Record {
    a_mid: [f64; 3],
    s255: [f64; 3],
    a255: [f64; 3],
    f_top: f64,
    m_top: f64,
    f_bot: f64,
    m_bot: f64,
    id: i64,
    c0i: u32,
    c1i: u32,
    p_top: u32,
    p_bot: u32,
    ic0: u32,
    ic1: u32,
    first: u32,
}

#[derive(Clone, Copy, Default)]
struct Row {
    diff_a: [f64; 3],
    diff_s: [f64; 3],
    diff_cov: f64,
    extra_rgb: [f64; 3],
    extra_cov: f64,
    diff_id: i64,
}

/// Columns walked together: one SIMD lane per column. Lanes never mix, so
/// every lane's arithmetic is its own column's, operation for operation.
const LANES: usize = 8;

/// One output row's float32 diffs (and integer id diff) for every lane,
/// converted from the column's float64 bincount sums exactly as the
/// reference casts them before its prefix sums.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct LaneRow {
    a: [[f32; LANES]; 3],
    s: [[f32; LANES]; 3],
    extra: [[f32; LANES]; 3],
    cov: [f32; LANES],
    extra_cov: [f32; LANES],
    id: [i32; LANES],
}

struct Scratch {
    sorted_t32: Vec<f32>,
    is_b: Vec<bool>,
    used: Vec<bool>,
    sides: Vec<Record>,
    tops: Vec<Record>,
    floors: Vec<Record>,
    rows: Vec<Row>,
    touched: Vec<u64>,
    lanes: Vec<LaneRow>,
    lane_touched: Vec<u64>,
    rgb_acc: Vec<[[f32; LANES]; 3]>,
    cov_acc: Vec<[f32; LANES]>,
    tile_rgba: Vec<[u32; LANES]>,
    tile_id: Vec<[i32; LANES]>,
    clean: bool,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Option<Scratch>> = const { std::cell::RefCell::new(None) };
}

impl Scratch {
    fn fits(&self, s: usize, out_h: usize) -> bool {
        self.clean && self.sorted_t32.len() == s && self.rows.len() == out_h + 1
    }

    fn new(s: usize, out_h: usize) -> Self {
        let words = (out_h + 1).div_ceil(64);
        Self {
            sorted_t32: vec![0.0; s],
            is_b: vec![false; s],
            used: vec![false; s],
            sides: Vec::with_capacity(s),
            tops: Vec::with_capacity(s),
            floors: Vec::with_capacity(s),
            rows: vec![Row::default(); out_h + 1],
            touched: vec![0; words],
            lanes: vec![LaneRow::default(); out_h + 1],
            lane_touched: vec![0; words],
            rgb_acc: Vec::new(),
            cov_acc: Vec::new(),
            tile_rgba: vec![[0; LANES]; out_h],
            tile_id: vec![[0; LANES]; out_h],
            clean: true,
        }
    }

    /// Move one column's touched float64 rows into its lane, as float32,
    /// and leave the column rows zeroed for the next column.
    #[inline(always)]
    fn finalize(&mut self, lane: usize) {
        for word in 0..self.touched.len() {
            let mut bits = self.touched[word];
            self.lane_touched[word] |= bits;
            self.touched[word] = 0;
            while bits != 0 {
                let row = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let d = std::mem::take(&mut self.rows[row]);
                let l = &mut self.lanes[row];
                for ch in 0..3 {
                    l.a[ch][lane] = d.diff_a[ch] as f32;
                    l.s[ch][lane] = d.diff_s[ch] as f32;
                    l.extra[ch][lane] = d.extra_rgb[ch] as f32;
                }
                l.cov[lane] = d.diff_cov as f32;
                l.extra_cov[lane] = d.extra_cov as f32;
                // i32 wrapping sums are the i64 sums cast at the end
                l.id[lane] = d.diff_id as i32;
            }
        }
    }

    fn clear_lanes(&mut self) {
        for word in 0..self.lane_touched.len() {
            let mut bits = self.lane_touched[word];
            self.lane_touched[word] = 0;
            while bits != 0 {
                let row = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.lanes[row] = LaneRow::default();
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Target {
    rgba: *mut u8,
    ids: *mut i32,
}
// SAFETY: every block writes only its own disjoint column range.
unsafe impl Send for Target {}
unsafe impl Sync for Target {}

#[inline]
fn clamp_rows(value: f64, render_h: f64) -> f64 {
    if value < 0.0 {
        0.0
    } else if value > render_h {
        render_h
    } else {
        value
    }
}

/// What a walk needs besides the scratch.
#[derive(Clone, Copy)]
struct Walk {
    start: usize,
    out_h: usize,
    mid: bool,
    inv_taps32: f32,
    bg255: [f32; 3],
}

#[inline(always)]
fn clamp_unit(c: f32) -> f32 {
    if c < 0.0 {
        0.0
    } else if c > 1.0 {
        1.0
    } else {
        c
    }
}

/// The reference's final conversion of one lane-row: average the taps,
/// complete with the background, clip, round half up. Pixels come back as
/// little-endian RGBA words.
#[inline(always)]
fn combine(rgb: &[[f32; LANES]; 3], cov: &[f32; LANES], inv_taps32: f32, bg255: [f32; 3]) -> [u32; LANES] {
    let mut pixels = [0xff00_0000u32; LANES];
    let mut remainder = [0.0f32; LANES];
    for l in 0..LANES {
        remainder[l] = 1.0f32 - cov[l] * inv_taps32;
    }
    for ch in 0..3 {
        for l in 0..LANES {
            let mut r = rgb[ch][l] * inv_taps32;
            r = r + remainder[l] * bg255[ch];
            if r < 0.0 {
                r = 0.0;
            } else if r > 255.0 {
                r = 255.0;
            }
            // r + 0.5 lies in [0.5, 255.5]: truncation is the uint8 cast
            pixels[l] |= (((r + 0.5f32) as i32 as u32) & 0xff) << (8 * ch);
        }
    }
    pixels
}

/// The float32 prefix walk of one tap over all lanes. `SINGLE`: one tap per
/// pixel, so the walk converts each row straight to pixels (the tap sum of
/// one is `0.0 + v`, kept literally).
///
/// A row no lane touched holds zeros, and adding a zero changes none of
/// these sums: an accumulator starts at +0 and a round-to-nearest sum is
/// never -0 unless both terms are, so `x + 0.0` is `x` throughout. Such rows
/// skip their loads and adds and keep only the row-dependent arithmetic.
#[inline(always)]
fn walk_body<const SINGLE: bool>(scratch: &mut Scratch, walk: Walk) {
    let mut acc_a = [[0.0f32; LANES]; 3];
    let mut acc_s = [[0.0f32; LANES]; 3];
    let mut acc_c = [0.0f32; LANES];
    let mut acc_id = [0i32; LANES];
    if SINGLE {
        let sky = combine(&[[0.0f32 + 0.0f32; LANES]; 3], &[0.0f32 + 0.0f32; LANES], walk.inv_taps32, walk.bg255);
        for row in 0..walk.start {
            scratch.tile_rgba[row] = sky;
            scratch.tile_id[row] = [0; LANES];
        }
    } else if walk.mid {
        for row in 0..walk.start {
            scratch.tile_id[row] = [0; LANES];
        }
    }
    for row in walk.start..walk.out_h {
        let touched = scratch.lane_touched[row >> 6] >> (row & 63) & 1 != 0;
        let mut value = [[0.0f32; LANES]; 3];
        let mut cover = [0.0f32; LANES];
        let row32 = row as f32;
        if touched {
            let d = &scratch.lanes[row];
            for ch in 0..3 {
                for l in 0..LANES {
                    acc_a[ch][l] += d.a[ch][l];
                    acc_s[ch][l] += d.s[ch][l];
                }
            }
            for l in 0..LANES {
                acc_c[l] += d.cov[l];
                acc_id[l] = acc_id[l].wrapping_add(d.id[l]);
            }
            for ch in 0..3 {
                for l in 0..LANES {
                    let v = acc_a[ch][l] + acc_s[ch][l] * row32;
                    value[ch][l] = v + d.extra[ch][l];
                }
            }
            for l in 0..LANES {
                cover[l] = clamp_unit(acc_c[l] + d.extra_cov[l]);
            }
        } else {
            for ch in 0..3 {
                for l in 0..LANES {
                    value[ch][l] = acc_a[ch][l] + acc_s[ch][l] * row32;
                }
            }
            for l in 0..LANES {
                cover[l] = clamp_unit(acc_c[l]);
            }
        }
        if walk.mid {
            scratch.tile_id[row] = acc_id;
        }
        if SINGLE {
            for ch in 0..3 {
                for l in 0..LANES {
                    value[ch][l] = 0.0f32 + value[ch][l];
                }
            }
            for l in 0..LANES {
                cover[l] = 0.0f32 + cover[l];
            }
            scratch.tile_rgba[row] = combine(&value, &cover, walk.inv_taps32, walk.bg255);
        } else {
            let acc = &mut scratch.rgb_acc[row];
            for ch in 0..3 {
                for l in 0..LANES {
                    acc[ch][l] += value[ch][l];
                }
            }
            for l in 0..LANES {
                scratch.cov_acc[row][l] += cover[l];
            }
        }
    }
}

fn walk_plain_single(scratch: &mut Scratch, walk: Walk) { walk_body::<true>(scratch, walk) }
fn walk_plain_taps(scratch: &mut Scratch, walk: Walk) { walk_body::<false>(scratch, walk) }

/// Test hook: walk with the portable loops even where AVX2 is present.
pub(crate) static FORCE_PORTABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `walk_body` in AVX2 registers, one lane per column. Every operation is
/// the portable one per lane: adds, multiplies, the same compare-and-select
/// clamps (NaN passes through both), truncating converts masked to a byte.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn walk_avx2<const SINGLE: bool>(scratch: &mut Scratch, walk: Walk) {
    use std::arch::x86_64::*;
    // SAFETY (all intrinsics): AVX2 is enabled for this function, and every
    // load/store addresses a whole [f32; 8] / [i32; 8] / [u32; 8] array.
    unsafe {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let top = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);
        let inv = _mm256_set1_ps(walk.inv_taps32);
        let bg = [_mm256_set1_ps(walk.bg255[0]), _mm256_set1_ps(walk.bg255[1]), _mm256_set1_ps(walk.bg255[2])];
        let byte = _mm256_set1_epi32(0xff);
        let alpha = _mm256_set1_epi32(0xff00_0000u32 as i32);
        // Macros, not closures: a closure would not carry this function's
        // target features, and every intrinsic in it would become a call.
        macro_rules! clamp {
            ($x:expr, $high:expr) => {{
                let x = $x;
                let below = _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero);
                let above = _mm256_cmp_ps::<_CMP_GT_OQ>(x, $high);
                _mm256_blendv_ps(_mm256_blendv_ps(x, $high, above), zero, below)
            }};
        }
        macro_rules! channel {
            ($rgb:expr, $remainder:expr, $bg:expr) => {{
                let r = _mm256_add_ps(_mm256_mul_ps($rgb, inv), _mm256_mul_ps($remainder, $bg));
                let r = clamp!(r, top);
                _mm256_and_si256(_mm256_cvttps_epi32(_mm256_add_ps(r, half)), byte)
            }};
        }
        macro_rules! combine {
            ($rgb:expr, $cov:expr) => {{
                let rgb: [__m256; 3] = $rgb;
                let remainder = _mm256_sub_ps(one, _mm256_mul_ps($cov, inv));
                let red = channel!(rgb[0], remainder, bg[0]);
                let green = _mm256_slli_epi32::<8>(channel!(rgb[1], remainder, bg[1]));
                let blue = _mm256_slli_epi32::<16>(channel!(rgb[2], remainder, bg[2]));
                _mm256_or_si256(_mm256_or_si256(alpha, red), _mm256_or_si256(green, blue))
            }};
        }
        macro_rules! load {
            ($v:expr) => {
                _mm256_loadu_ps($v.as_ptr())
            };
        }
        let mut acc_a = [zero; 3];
        let mut acc_s = [zero; 3];
        let mut acc_c = zero;
        let mut acc_id = _mm256_setzero_si256();
        if SINGLE {
            let mut sky = [0u32; LANES];
            let zero_sum = _mm256_add_ps(zero, zero);
            _mm256_storeu_si256(sky.as_mut_ptr() as *mut __m256i, combine!([zero_sum; 3], zero_sum));
            for row in 0..walk.start {
                scratch.tile_rgba[row] = sky;
                scratch.tile_id[row] = [0; LANES];
            }
        } else if walk.mid {
            for row in 0..walk.start {
                scratch.tile_id[row] = [0; LANES];
            }
        }
        for row in walk.start..walk.out_h {
            let touched = scratch.lane_touched[row >> 6] >> (row & 63) & 1 != 0;
            let row32 = _mm256_set1_ps(row as f32);
            let mut value = [zero; 3];
            let cover;
            if touched {
                let d = &scratch.lanes[row];
                for ch in 0..3 {
                    acc_a[ch] = _mm256_add_ps(acc_a[ch], load!(d.a[ch]));
                    acc_s[ch] = _mm256_add_ps(acc_s[ch], load!(d.s[ch]));
                }
                acc_c = _mm256_add_ps(acc_c, load!(d.cov));
                acc_id = _mm256_add_epi32(acc_id, _mm256_loadu_si256(d.id.as_ptr() as *const __m256i));
                for ch in 0..3 {
                    let v = _mm256_add_ps(acc_a[ch], _mm256_mul_ps(acc_s[ch], row32));
                    value[ch] = _mm256_add_ps(v, load!(d.extra[ch]));
                }
                cover = clamp!(_mm256_add_ps(acc_c, load!(d.extra_cov)), one);
            } else {
                for ch in 0..3 {
                    value[ch] = _mm256_add_ps(acc_a[ch], _mm256_mul_ps(acc_s[ch], row32));
                }
                cover = clamp!(acc_c, one);
            }
            if walk.mid {
                _mm256_storeu_si256(scratch.tile_id[row].as_mut_ptr() as *mut __m256i, acc_id);
            }
            if SINGLE {
                let rgb = [_mm256_add_ps(zero, value[0]), _mm256_add_ps(zero, value[1]), _mm256_add_ps(zero, value[2])];
                let pixels = combine!(rgb, _mm256_add_ps(zero, cover));
                _mm256_storeu_si256(scratch.tile_rgba[row].as_mut_ptr() as *mut __m256i, pixels);
            } else {
                let acc = &mut scratch.rgb_acc[row];
                for ch in 0..3 {
                    let sum = _mm256_add_ps(_mm256_loadu_ps(acc[ch].as_ptr()), value[ch]);
                    _mm256_storeu_ps(acc[ch].as_mut_ptr(), sum);
                }
                let cov = &mut scratch.cov_acc[row];
                _mm256_storeu_ps(cov.as_mut_ptr(), _mm256_add_ps(_mm256_loadu_ps(cov.as_ptr()), cover));
            }
        }
    }
}

/// The walk, in AVX2 registers when the CPU has them.
fn walk_tap(scratch: &mut Scratch, walk: Walk, single: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && !FORCE_PORTABLE.load(std::sync::atomic::Ordering::Relaxed) {
            // SAFETY: the feature was detected on this CPU.
            unsafe {
                return if single { walk_avx2::<true>(scratch, walk) } else { walk_avx2::<false>(scratch, walk) };
            }
        }
    }
    if single { walk_plain_single(scratch, walk) } else { walk_plain_taps(scratch, walk) }
}

impl Kernel<'_> {
    pub(crate) fn run(&self, rgba: &mut [u8], ids: &mut [i32]) {
        let (out_h, out_w) = (self.out_h, self.out_w);
        assert_eq!(rgba.len(), out_h * out_w * 4);
        assert_eq!(ids.len(), out_h * out_w);
        let s = self.nx + self.ny;
        let target = Target { rgba: rgba.as_mut_ptr(), ids: ids.as_mut_ptr() };
        let blocks = out_w.div_ceil(LANES);
        (0..blocks).into_par_iter().for_each(|block| {
            let col0 = block * LANES;
            let col1 = (col0 + LANES).min(out_w);
            // One scratch per pool thread, kept across frames: every
            // walk leaves its rows and lanes zeroed, and the tiles are
            // written in full, so nothing stale reaches a column.
            SCRATCH.with(|cell| {
                let mut slot = cell.borrow_mut();
                if !slot.as_ref().is_some_and(|scratch: &Scratch| scratch.fits(s, out_h)) {
                    *slot = Some(Scratch::new(s, out_h));
                }
                let scratch = slot.as_mut().expect("scratch");
                scratch.clean = false; // an unwinding block leaves it dirty
                self.dispatch_block(scratch, col0, col1, target);
                scratch.clean = true;
            });
        });
    }

    /// The block compiled for AVX2 where present: the prelude's ceil/floor
    /// become single instructions (the baseline target calls the C library
    /// for them) and its loops may use the wider registers -- the same IEEE
    /// operations either way, no FMA.
    fn dispatch_block(&self, scratch: &mut Scratch, col0: usize, col1: usize, target: Target) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && !FORCE_PORTABLE.load(std::sync::atomic::Ordering::Relaxed) {
                // SAFETY: the feature was detected on this CPU.
                return unsafe { self.block_avx2(scratch, col0, col1, target) };
            }
        }
        self.block(scratch, col0, col1, target)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn block_avx2(&self, scratch: &mut Scratch, col0: usize, col1: usize, target: Target) {
        self.block(scratch, col0, col1, target)
    }

    #[inline(always)]
    fn block(&self, scratch: &mut Scratch, col0: usize, col1: usize, target: Target) {
        let width = col1 - col0;
        let out_h = self.out_h;
        let single = self.taps == 1;
        let inv_taps32 = (1.0 / self.taps as f64) as f32;
        let bg255 = [
            self.background32[0] * 255.0f32,
            self.background32[1] * 255.0f32,
            self.background32[2] * 255.0f32,
        ];
        if !single {
            // the tap sums exist only where there are taps to sum
            scratch.rgb_acc.clear();
            scratch.rgb_acc.resize(out_h, [[0.0; LANES]; 3]);
            scratch.cov_acc.clear();
            scratch.cov_acc.resize(out_h, [0.0; LANES]);
        }
        for tap in 0..self.taps {
            // Rows above every lane's first span add zero to zero: the walk
            // starts at the earliest of them.
            let mut start = out_h;
            for lane in 0..width {
                let first = self.prelude(scratch, (col0 + lane) * self.taps + tap);
                start = start.min(first);
                scratch.finalize(lane);
            }
            walk_tap(scratch, Walk { start, out_h, mid: tap == self.mid_tap, inv_taps32, bg255 }, single);
            scratch.clear_lanes();
        }
        if !single {
            for row in 0..out_h {
                scratch.tile_rgba[row] = combine(&scratch.rgb_acc[row], &scratch.cov_acc[row], inv_taps32, bg255);
            }
        }
        for row in 0..out_h {
            // SAFETY: rows of [col0, col1) belong to this block alone.
            unsafe {
                let destination = target.rgba.add((row * self.out_w + col0) * 4);
                let words = scratch.tile_rgba[row].map(u32::to_le);
                std::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, destination, width * 4);
                let destination = target.ids.add(row * self.out_w + col0);
                std::ptr::copy_nonoverlapping(scratch.tile_id[row].as_ptr(), destination, width);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    /// One render column up to its row diffs: crossings, the segment sweep
    /// (which emits each segment's side, top and floor records as it goes,
    /// into their class lists), and the bincount-order scatter. Returns the
    /// first row any span touches; every row written is marked in `touched`.
    #[inline(always)]
    fn prelude(&self, scratch: &mut Scratch, column: usize) -> usize {
        let (nx, ny) = (self.nx, self.ny);
        let s_count = nx + ny;
        let render_h = self.render_h as f64;
        let inv_taps64 = 1.0 / self.taps as f64;
        let t255 = 255.0 * self.taps as f64;
        let last = self.out_h as i64 - 1;
        let (ca, se32) = (self.ca, self.se32);
        let (y_high, scale) = (self.y_high, self.scale);
        let cols = self.columns;
        let enter = cols.t_enter[column];
        let texit = cols.t_exit[column];
        let ia0c = trunc_i64(cols.ia0[column]);
        let ib0c = trunc_i64(cols.ib0[column]);
        let ta0 = cols.t_a0[column];
        let tb0 = cols.t_b0[column];
        let g0c = cols.g0[column];

        // ---- crossings: a slots by the closed form, b slots as the integer
        // complement (a permutation by construction).
        let sorted = &mut scratch.sorted_t32;
        let used = &mut scratch.used;
        let is_b = &mut scratch.is_b;
        used.fill(false);
        is_b.fill(false);
        for k in 0..nx {
            let t_cross = ta0 + cols.k_over_sa[k];
            let mut before = ((t_cross - tb0) * ca).ceil();
            if before < 0.0 {
                before = 0.0;
            } else if before > ny as f64 {
                before = ny as f64;
            }
            let slot = k + trunc_i64(before) as usize;
            sorted[slot] = t_cross as f32;
            used[slot] = true;
        }
        let mut cursor = 0usize;
        for m in 0..ny {
            while used[cursor] {
                cursor += 1;
            }
            sorted[cursor] = (tb0 + cols.m_over_ca[m]) as f32;
            used[cursor] = true;
            is_b[cursor] = true;
            cursor += 1;
        }

        // ---- segment sweep, emitting span records with FLOAT bounds
        let rows_of = |hi: f64, lo: f64| -> Option<(f64, f64)> {
            let rr_top = clamp_rows((y_high - hi) * scale, render_h);
            let rr_bot = clamp_rows((y_high - lo) * scale, render_h);
            (rr_bot > rr_top).then_some((rr_top * inv_taps64, rr_bot * inv_taps64))
        };
        let background = [
            self.background32[0] as f64,
            self.background32[1] as f64,
            self.background32[2] as f64,
        ];
        // A flat face: slope zero, so s255 = 0 * t255 and the anchor is the
        // colour, a_mid = a * 255 + s255 * 0.5, a255 = a * 255.
        let flat = |colour: [f64; 3]| -> ([f64; 3], [f64; 3], [f64; 3]) {
            let mut a_mid = [0.0; 3];
            let mut s255 = [0.0; 3];
            let mut a255 = [0.0; 3];
            for ch in 0..3 {
                s255[ch] = 0.0 * t255;
                a255[ch] = colour[ch] * 255.0;
                a_mid[ch] = a255[ch] + s255[ch] * 0.5;
            }
            (a_mid, s255, a255)
        };
        let (bg_mid, bg_s255, bg_a255) = flat(background);
        let a_at = cols.a_at[column];
        let first_entered = (a_at - a_at.round_ties_even()).abs() < 1e-9;
        let cells = self.cells;
        let sides = &mut scratch.sides;
        let tops = &mut scratch.tops;
        let floors = &mut scratch.floors;
        sides.clear();
        tops.clear();
        floors.clear();
        let mut previous32 = enter as f32;
        let mut running = f64::NEG_INFINITY;
        let mut before_b: i64 = 0;
        for s in 0..s_count {
            let mut v0 = previous32 as f64;
            if v0 > texit {
                v0 = texit;
            }
            let t0 = v0 as f32;
            let mut t1 = sorted[s] as f64;
            if t1 > texit {
                t1 = texit;
            }
            previous32 = sorted[s];
            let cell_ia = ia0c - (s as i64 - before_b);
            let cell_ib = ib0c + before_b;
            if cell_ia < 0 || cell_ib >= ny as i64 {
                // cell_ia only falls and cell_ib only rises along the walk:
                // from here no segment is inside, so none adds a record or
                // raises the silhouette.
                break;
            }
            let inside = cell_ia >= 0
                && cell_ia < nx as i64
                && cell_ib >= 0
                && cell_ib < ny as i64
                && t1 > t0 as f64;
            let cai = cell_ia.clamp(0, nx as i64 - 1) as usize;
            let cbi = cell_ib.clamp(0, ny as i64 - 1) as usize;
            let index = cbi * nx + cai;
            let cell = cells[index];
            let ok = inside && cell.finite;
            let glo = g0c + t0 * se32;
            let ghi = g0c as f64 + t1 * se32 as f64;
            let ent = if s == 0 { first_entered } else { !is_b[s - 1] };
            let wm = running;
            if ok {
                let sil = ghi + cell.top as f64;
                if sil > running {
                    running = sil;
                }
            }
            if is_b[s] {
                before_b += 1;
            }
            if ok {
                // side face (without a bar its hi is -inf: no record)
                let mut lo = (glo + cell.bot) as f64;
                if lo < wm {
                    lo = wm;
                }
                let hi = (glo + cell.top) as f64;
                if hi > lo {
                    if let Some((q0, q1)) = rows_of(hi, lo) {
                        let full_lo = glo + cell.bot;
                        let full_hi = glo + cell.top;
                        let fr_bottom = (y_high - full_lo as f64) * scale;
                        let fr_top = (y_high - full_hi as f64) * scale;
                        let mut height = fr_bottom - fr_top;
                        if height < 1e-6 {
                            height = 1e-6;
                        }
                        let shade = if ent { self.shade_x } else { self.shade_y };
                        let positive = cell.top > 0.0f32;
                        let own = self.rgb[index];
                        let base = self.base.unwrap_or(own);
                        let mut a_mid = [0.0; 3];
                        let mut s255 = [0.0; 3];
                        let mut a255 = [0.0; 3];
                        for ch in 0..3 {
                            let (c_hi, c_lo) = if positive {
                                (own[ch] * shade, base[ch] * shade)
                            } else {
                                (base[ch] * shade, own[ch] * shade)
                            };
                            let d = c_lo - c_hi;
                            let slp = d as f64 / height;
                            let intercept = c_hi as f64 - slp * fr_top;
                            s255[ch] = slp * t255;
                            a255[ch] = intercept * 255.0;
                            a_mid[ch] = a255[ch] + s255[ch] * 0.5;
                        }
                        let id = (index * 4 + 4 + 1 + usize::from(ent)) as i64;
                        sides.push(Record::new(q0, q1, a_mid, s255, a255, id, last));
                    }
                }
                // top face
                let mut lo = (glo + cell.top) as f64;
                if lo < wm {
                    lo = wm;
                }
                let hi = ghi + cell.top as f64;
                if hi > lo {
                    if let Some((q0, q1)) = rows_of(hi, lo) {
                        let own = self.rgb[index];
                        let (a_mid, s255, a255) = flat([own[0] as f64, own[1] as f64, own[2] as f64]);
                        tops.push(Record::new(q0, q1, a_mid, s255, a255, (index * 4 + 4 + 3) as i64, last));
                    }
                }
            } else if inside {
                // the floor under a cell without a bar
                let mut lo = glo as f64;
                if lo < wm {
                    lo = wm;
                }
                let hi = ghi;
                if hi > lo {
                    if let Some((q0, q1)) = rows_of(hi, lo) {
                        floors.push(Record::new(q0, q1, bg_mid, bg_s255, bg_a255, 1, last));
                    }
                }
            }
        }
        let final_wm = running;
        let g_exit = g0c + (texit as f32) * se32;
        let mut pane = None;
        if texit > enter {
            let mut pane_lo = (g_exit + self.pane_low32) as f64;
            if pane_lo < final_wm {
                pane_lo = final_wm;
            }
            let pane_hi = (g_exit + self.pane_high32) as f64;
            if pane_hi > pane_lo {
                if let Some((q0, q1)) = rows_of(pane_hi, pane_lo) {
                    pane = Some(Record::new(q0, q1, bg_mid, bg_s255, bg_a255, 2, last));
                }
            }
        }

        // ---- scatter in the bincount slot order of the reference. Each
        // plane keeps its own order -- every + entry (record order) before
        // every - entry -- and planes never share an accumulator, so one
        // pass carries the + entries of all of them and a second the -.
        let rows = &mut scratch.rows;
        let touched = &mut scratch.touched;
        let records = || sides.iter().chain(tops.iter()).chain(floors.iter()).chain(pane.iter());
        let mut first_row = self.out_h;
        for record in records() {
            let row = &mut rows[record.c0i as usize];
            for ch in 0..3 {
                row.diff_a[ch] += record.a_mid[ch];
                row.diff_s[ch] += record.s255[ch];
            }
            row.diff_cov += 1.0;
            let row = &mut rows[record.p_top as usize];
            for ch in 0..3 {
                row.extra_rgb[ch] += record.f_top * (record.a255[ch] + record.s255[ch] * record.m_top);
            }
            row.extra_cov += record.f_top;
            rows[record.ic0 as usize].diff_id += record.id;
            for row in [record.c0i, record.c1i, record.p_top, record.p_bot, record.ic0, record.ic1] {
                touched[row as usize >> 6] |= 1u64 << (row & 63);
            }
            first_row = first_row.min(record.first as usize);
        }
        for record in records() {
            let row = &mut rows[record.c1i as usize];
            for ch in 0..3 {
                row.diff_a[ch] += -record.a_mid[ch];
                row.diff_s[ch] += -record.s255[ch];
            }
            row.diff_cov += -1.0;
            let row = &mut rows[record.p_bot as usize];
            for ch in 0..3 {
                row.extra_rgb[ch] += record.f_bot * (record.a255[ch] + record.s255[ch] * record.m_bot);
            }
            row.extra_cov += record.f_bot;
            rows[record.ic1 as usize].diff_id += -record.id;
        }
        first_row
    }
}

impl Record {
    /// A span's rows and partial fractions, decided once.
    #[inline(always)]
    fn new(q0: f64, q1: f64, a_mid: [f64; 3], s255: [f64; 3], a255: [f64; 3], id: i64, last: i64) -> Self {
        let c0 = q0.ceil();
        let c1 = q1.floor();
        let c0i = trunc_i64(c0);
        let c1i = trunc_i64(c1).max(c0i);
        let floor_q0 = trunc_i64(q0.floor());
        let p_top = floor_q0.min(last);
        let top_end = if q1 < c0 { q1 } else { c0 };
        let f_top = py_max(top_end - q0, 0.0);
        let m_top = (q0 + top_end) * 0.5;
        // The bottom partial lands on floor(q1) itself, not raised to c0i;
        // where the two differ its fraction is zero.
        let p_bot = trunc_i64(c1).min(last);
        let f_bot = if c1 >= c0 {
            let f = q1 - c1;
            if f < 0.0 { 0.0 } else { f }
        } else {
            0.0
        };
        let m_bot = (c1 + q1) * 0.5;
        let ic0 = trunc_i64((q0 - 0.5).ceil());
        let ic1 = trunc_i64((q1 - 0.5).ceil()).max(ic0);
        let first = c0i.min(floor_q0).min(ic0).max(0);
        Self {
            a_mid,
            s255,
            a255,
            f_top,
            m_top,
            f_bot,
            m_bot,
            id,
            c0i: c0i as u32,
            c1i: c1i as u32,
            p_top: p_top as u32,
            p_bot: p_bot as u32,
            ic0: ic0 as u32,
            ic1: ic1 as u32,
            first: first as u32,
        }
    }
}
