//! The v2 CPU stroke kernels, with their original coverage and rounding.
//!
//! Scheduling alone changes: the crate's shared pool owns disjoint row bands.
//! Each band visits lines, samples and pieces in the original painter order.
//! NaN/Inf vertices break a path; negative coordinates are ordinary clipped
//! geometry. No Python owner, per-kernel pool, or full-canvas scratch is used.

use rayon::prelude::*;

use crate::Canvas;

pub struct PolylineBatch<'a> {
    pub vertices: &'a [[f64; 2]],
    pub offsets: &'a [i64],
    pub colours: &'a [[u8; 4]],
    pub widths: &'a [f64],
    pub clips: &'a [[i32; 4]],
}

pub struct ErrorBarBatch<'a> {
    pub x: &'a [f64],
    pub y_low: &'a [f64],
    pub y_high: &'a [f64],
    pub offsets: &'a [i64],
    pub colours: &'a [[u8; 4]],
    pub widths: &'a [f64],
    pub cap_widths: &'a [f64],
    pub clips: &'a [[i32; 4]],
}

fn validate_canvas(canvas: &Canvas<'_>) -> Result<(), String> {
    let bottom = canvas.top.checked_add(canvas.height).ok_or("Canvas row extent overflow")?;
    if canvas.width.checked_mul(canvas.height).and_then(|n| n.checked_mul(4)) != Some(canvas.pixels.len())
        || canvas.width > i64::MAX as usize || bottom > i64::MAX as usize
    {
        return Err("Canvas must contain its complete contiguous RGBA row band".into());
    }
    Ok(())
}

fn validate_offsets(offsets: &[i64], length: usize, name: &str) -> Result<usize, String> {
    if offsets.is_empty() {
        return Err(format!("{name} requires at least one boundary"));
    }
    let mut previous = 0usize;
    for &offset in offsets {
        let offset = usize::try_from(offset).map_err(|_| format!("{name} must be nonnegative"))?;
        if offset < previous || offset > length {
            return Err(format!("{name} must be ordered inside its input"));
        }
        previous = offset;
    }
    Ok(offsets.len() - 1)
}

fn validate_groups(
    count: usize, colours: &[[u8; 4]], widths: &[f64], clips: &[[i32; 4]],
) -> Result<(), String> {
    if colours.len() != count || widths.len() != count || clips.len() != count {
        return Err("Stroke colours, widths and clips must match group count".into());
    }
    if widths.iter().any(|width| !width.is_finite()) {
        return Err("Stroke widths must be finite".into());
    }
    Ok(())
}

/// Batch affine conversion in exactly the operand order of the v2 kernel.
/// An affine flag is a byte, not a Rust bool borrowed from foreign memory.
pub fn transform_polylines(
    vertices: &[[f64; 2]], offsets: &[i64], matrices: &[[f64; 6]], affine: &[u8],
    canvas_height: f64, out: &mut [[f64; 2]],
) -> Result<(), String> {
    let count = validate_offsets(offsets, vertices.len(), "polyline offsets")?;
    if matrices.len() != count || affine.len() != count || out.len() != vertices.len() {
        return Err("Transform arrays must match their polyline/vertex counts".into());
    }
    if !canvas_height.is_finite() {
        return Err("Canvas height must be finite".into());
    }
    for line in 0..count {
        let start = offsets[line] as usize;
        let stop = offsets[line + 1] as usize;
        if affine[line] != 0 {
            let [a, c, e, b, d, f] = matrices[line];
            for point in start..stop {
                let [x, y] = vertices[point];
                // Do not contract these into fused multiply-add operations.
                out[point][0] = a * x + c * y + e;
                out[point][1] = canvas_height - (b * x + d * y + f);
            }
        } else {
            for point in start..stop {
                out[point][0] = vertices[point][0];
                out[point][1] = canvas_height - vertices[point][1];
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn clamped_line_integral(value: f64) -> f64 {
    if value <= 0.0 { 0.0 }
    else if value <= 1.0 { 0.5 * value * value }
    else { value - 0.5 }
}

#[inline(always)]
fn slanted_cover(depth: f64, grade: f64) -> f64 {
    if grade <= 1.0e-9 { return depth.max(0.0).min(1.0); }
    let half = 0.5 * grade;
    (clamped_line_integral(depth + half) - clamped_line_integral(depth - half)) / grade
}

#[inline(always)]
fn disc_cover(px: f64, py: f64, x: f64, y: f64, radius: f64) -> f64 {
    let dx = px - x;
    let dy = py - y;
    (radius + 0.5 - (dx * dx + dy * dy).sqrt()).max(0.0).min(1.0)
}

#[derive(Clone, Copy)]
struct Piece {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    cap_start: bool,
    cap_end: bool,
}

/// Where one sloped piece can cover a pixel column, as lines in that column.
///
/// `segment_cover` is nonzero only where the pixel centre projects onto the
/// piece (or within a cap's reach past a capped end), or inside the join
/// wedge past an uncapped end and before the next piece. Projection is
/// linear in the row, so each condition bounds the row on one side; the
/// coefficients are fixed per piece, not per column. Every other row
/// contributes exactly zero and is not evaluated.
#[derive(Clone, Copy)]
struct Reach {
    flat: bool,
    slope: f64,
    low: f64,
    high: f64,
    join: bool,
    past_side: i8,
    past: f64,
    before_side: i8,
    before_slope: f64,
    before: f64,
}

const FLAT: f64 = 1.0e-6;
/// Rows kept beyond each analytic bound, against rounding.
const MARGIN: i64 = 2;

fn reach_of(piece: Piece, next: Option<Piece>, radius: f64) -> Reach {
    let dx = piece.x1 - piece.x0;
    let dy = piece.y1 - piece.y0;
    let length2 = dx * dx + dy * dy;
    let flat = !(dy.abs() > FLAT && length2 > 0.0);
    let (slope, base, along) = if flat { (0.0, 0.0, 0.0) } else { (dx / dy, piece.y0 + piece.x0 * (dx / dy), length2 / dy) };
    let extent = (radius + 0.5) / length2.sqrt();
    let low = base + (if piece.cap_start { -extent } else { 0.0 }) * along;
    let high = base + (if piece.cap_end { 1.0 + extent } else { 1.0 }) * along;
    let mut reach = Reach { flat, slope, low, high, join: !piece.cap_end, past_side: 0, past: base + along,
                            before_side: 0, before_slope: 0.0, before: 0.0 };
    if !flat { reach.past_side = if dy > 0.0 { 1 } else { -1 }; }
    if let Some(next) = next {
        let nx = next.x1 - piece.x1;
        let ny = next.y1 - piece.y1;
        if nx * nx + ny * ny > 0.0 && ny.abs() > FLAT {
            reach.before_slope = nx / ny;
            reach.before = piece.y1 + piece.x1 * reach.before_slope;
            reach.before_side = if ny > 0.0 { -1 } else { 1 };
        }
    }
    reach
}

#[inline(always)]
fn row_span(low: f64, high: f64, rows: [i64; 2]) -> [i64; 2] {
    [rows[0].max(floor_i64(low - 0.5).saturating_sub(MARGIN)), rows[1].min(ceil_i64(high - 0.5).saturating_add(MARGIN + 1))]
}

/// The candidate rows of one column: the body, and the join wedge if any.
#[inline(always)]
fn reached_rows(cx: f64, piece: Piece, reach: &Reach, radius: f64, rows: [i64; 2]) -> ([i64; 2], Option<[i64; 2]>) {
    let body = if reach.flat { rows } else {
        let first = reach.low - cx * reach.slope;
        let last = reach.high - cx * reach.slope;
        row_span(first.min(last), first.max(last), rows)
    };
    let disc = radius + 0.5;
    if !reach.join || (cx - piece.x1).abs() > disc + 0.01 { return (body, None); }
    let mut low = piece.y1 - disc;
    let mut high = piece.y1 + disc;
    let past = reach.past - cx * reach.slope;
    if reach.past_side > 0 { low = low.max(past); } else if reach.past_side < 0 { high = high.min(past); }
    let before = reach.before - cx * reach.before_slope;
    if reach.before_side > 0 { low = low.max(before); } else if reach.before_side < 0 { high = high.min(before); }
    let join = row_span(low, high, rows);
    (body, (join[0] < join[1]).then_some(join))
}

#[inline(always)]
fn segment_cover(px: f64, py: f64, piece: Piece, next: Option<Piece>, radius: f64) -> f64 {
    let Piece { x0, y0, x1, y1, cap_start, cap_end } = piece;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let length2 = dx * dx + dy * dy;
    if length2 <= 0.0 { return 0.0; }
    let along = ((px - x0) * dx + (py - y0) * dy) / length2;
    let length = length2.sqrt();
    let mut beyond = 0.0;
    if along < 0.0 {
        if !cap_start { return 0.0; }
        beyond = -along * length;
    } else if along > 1.0 {
        if !cap_end {
            if let Some(next) = next {
                let nx = next.x1 - x1;
                let ny = next.y1 - y1;
                let next2 = nx * nx + ny * ny;
                if next2 > 0.0 {
                    let along_next = ((px - x1) * nx + (py - y1) * ny) / next2;
                    if along_next >= 0.0 { return 0.0; }
                }
            }
            return disc_cover(px, py, x1, y1, radius);
        }
        beyond = (along - 1.0) * length;
    }
    if beyond > radius + 0.5 { return 0.0; }
    let mut cover;
    if dy.abs() > dx.abs() {
        let grade = (dx / dy).abs();
        let centre = x0 + (py - y0) * dx / dy;
        let half = radius * (1.0 + grade * grade).sqrt();
        cover = slanted_cover(px - (centre - half) + 0.5, grade)
            + slanted_cover((centre + half) - px + 0.5, grade) - 1.0;
    } else {
        let grade = (dy / dx).abs();
        let centre = y0 + (px - x0) * dy / dx;
        let half = radius * (1.0 + grade * grade).sqrt();
        cover = slanted_cover(py - (centre - half) + 0.5, grade)
            + slanted_cover((centre + half) - py + 0.5, grade) - 1.0;
    }
    if cover <= 0.0 { return 0.0; }
    if beyond > 0.0 {
        cover *= (radius + 0.5 - beyond).min(1.0);
    }
    cover
}

#[inline]
fn finite(vertex: [f64; 2]) -> bool {
    vertex[0].is_finite() && vertex[1].is_finite()
}

fn cut_pieces(vertices: &[[f64; 2]], snap: bool, snap_offset: f64, pieces: &mut Vec<Piece>) {
    for point in 0..vertices.len().saturating_sub(1) {
        let [mut x0, mut y0] = vertices[point];
        let [mut x1, mut y1] = vertices[point + 1];
        if !(finite(vertices[point]) && finite(vertices[point + 1])) { continue; }
        if snap {
            x0 = (x0 + 0.5).floor() + snap_offset;
            y0 = (y0 + 0.5).floor() + snap_offset;
            x1 = (x1 + 0.5).floor() + snap_offset;
            y1 = (y1 + 0.5).floor() + snap_offset;
        }
        pieces.push(Piece { x0, y0, x1, y1,
            cap_start: point == 0 || !finite(vertices[point - 1]),
            cap_end: point + 2 >= vertices.len() || !finite(vertices[point + 2]),
        });
    }
}

/// Analytic antialiased polyline coverage, with one blend per line and pixel.
pub fn raster_polylines(batch: &PolylineBatch<'_>, canvas: &mut Canvas<'_>) -> Result<(), String> {
    validate_canvas(canvas)?;
    let count = validate_offsets(batch.offsets, batch.vertices.len(), "polyline offsets")?;
    validate_groups(count, batch.colours, batch.widths, batch.clips)?;
    if count == 0 || canvas.width == 0 || canvas.height == 0 { return Ok(()); }

    // The immutable piece store is shared by all row bands. Allocation is
    // bounded by input vertices, not by the output canvas area.
    let mut pieces = Vec::with_capacity(batch.vertices.len());
    let mut piece_offsets = Vec::with_capacity(count + 1);
    piece_offsets.push(0);
    for line in 0..count {
        let vertices = &batch.vertices[batch.offsets[line] as usize..batch.offsets[line + 1] as usize];
        let mut snap = (2..=1024).contains(&vertices.len());
        if snap {
            for pair in vertices.windows(2) {
                if !(finite(pair[0]) && finite(pair[1])) { continue; }
                if (pair[0][0] - pair[1][0]).abs() >= 1.0e-4
                    && (pair[0][1] - pair[1][1]).abs() >= 1.0e-4 {
                    snap = false;
                    break;
                }
            }
        }
        let snap_offset = if (batch.widths[line] + 0.5).floor().rem_euclid(2.0) == 1.0 { 0.5 } else { 0.0 };
        cut_pieces(vertices, snap, snap_offset, &mut pieces);
        piece_offsets.push(pieces.len());
    }
    // Numba targets the host CPU; a portable Rust binary otherwise lowers
    // floor/ceil through its baseline instruction set. Dispatch without fast
    // math or FMA: the coverage arithmetic and integer rounding stay exact.
    #[cfg(target_arch = "x86_64")]
    let sse41 = std::is_x86_feature_detected!("sse4.1");
    #[cfg(not(target_arch = "x86_64"))]
    let sse41 = false;
    let buckets = if sse41 {
        #[cfg(target_arch = "x86_64")]
        { unsafe { prepare_buckets_sse41(batch, &pieces, &piece_offsets, canvas) } }
        #[cfg(not(target_arch = "x86_64"))]
        { unreachable!() }
    } else { prepare_buckets(batch, &pieces, &piece_offsets, canvas) };
    let paint = |lines: &[usize], rows: Rows| {
        #[cfg(target_arch = "x86_64")]
        if sse41 { return unsafe { paint_polylines_sse41(batch, &pieces, &buckets, lines, rows) }; }
        paint_polylines(batch, &pieces, &buckets, lines, rows)
    };
    // Lines of pairwise disjoint clip boxes (a grid's cells) paint apart in
    // parallel, each box's lines in their order; otherwise row bands do.
    if let Some(groups) = disjoint_clip_groups(&buckets) {
        let rows = Rows::of(canvas);
        groups.par_iter().for_each(|(lines, [top, bottom])| {
            paint(lines, rows.band(*top as usize, *bottom as usize));
        });
        return Ok(());
    }
    let lines: Vec<usize> = (0..buckets.len()).collect();
    canvas.for_bands(|mut band| paint(&lines, Rows::of(&mut band)))
}

/// The drawn lines grouped by clip box (in order) with each box's rows, when
/// there are at least two boxes and no two of them meet; None otherwise.
#[allow(clippy::type_complexity)]
fn disjoint_clip_groups(buckets: &[Option<LineBuckets>]) -> Option<Vec<(Vec<usize>, [i64; 2])>> {
    let mut groups: Vec<([i64; 4], Vec<usize>)> = Vec::new();
    for (line, bucket) in buckets.iter().enumerate() {
        let Some(bucket) = bucket else { continue };
        match groups.iter_mut().find(|(clip, _)| *clip == bucket.clip) {
            Some((_, lines)) => lines.push(line),
            None => groups.push((bucket.clip, vec![line])),
        }
        if groups.len() > MAX_CLIP_GROUPS { return None; }
    }
    if groups.len() < 2 { return None; }
    for (index, (a, _)) in groups.iter().enumerate() {
        for (b, _) in &groups[index + 1..] {
            if a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3] { return None; }
        }
    }
    Some(groups.into_iter().map(|(clip, lines)| (lines, [clip[1], clip[3]])).collect())
}

/// Beyond this many clip boxes the row bands paint (the overlap test is quadratic).
const MAX_CLIP_GROUPS: usize = 256;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn prepare_buckets_sse41(batch:&PolylineBatch<'_>, pieces:&[Piece], offsets:&[usize], canvas:&Canvas<'_>) -> Vec<Option<LineBuckets>> {
    prepare_buckets(batch,pieces,offsets,canvas)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn paint_polylines_sse41(batch:&PolylineBatch<'_>, pieces:&[Piece], buckets:&[Option<LineBuckets>], lines:&[usize], canvas:Rows) {
    paint_polylines(batch,pieces,buckets,lines,canvas);
}

#[inline(always)]
fn floor_i64(value: f64) -> i64 { value.floor() as i64 }

#[inline(always)]
fn ceil_i64(value: f64) -> i64 { value.ceil() as i64 }

/// The canvas rows one paint writes, as a raw view: paints of lines with
/// pairwise disjoint clip boxes share rows, each writing only inside its own
/// lines' clips (no pixel is read or written by two of them).
#[derive(Clone, Copy)]
struct Rows { pixels: *mut u8, width: usize, top: usize, height: usize }

// SAFETY: a `Rows` is handed to one paint of rows nobody else holds (a band),
// or of clip boxes no other paint touches.
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}

impl Rows {
    fn of(canvas: &mut Canvas<'_>) -> Rows {
        Rows { pixels: canvas.pixels.as_mut_ptr(), width: canvas.width, top: canvas.top, height: canvas.height }
    }

    /// Rows `top..bottom` (global) of these rows.
    fn band(self, top: usize, bottom: usize) -> Rows {
        // SAFETY: `top..bottom` lies within these rows.
        Rows { pixels: unsafe { self.pixels.add((top - self.top) * self.width * 4) }, width: self.width, top, height: bottom - top }
    }
}

/// Original float coverage blend, not Agg's different integer fill blend.
#[inline(always)]
fn blend(rows: Rows, row: i64, column: i64, colour: [u8; 4], alpha: f64) {
    let pixel = ((row as usize - rows.top) * rows.width + column as usize) * 4;
    let inverse = 1.0 - alpha;
    // SAFETY: (row, column) lies inside the rows and inside this paint's clip.
    unsafe {
        let at = rows.pixels.add(pixel);
        for channel in 0..3 {
            let value = f64::from(colour[channel]) * alpha + f64::from(*at.add(channel)) * inverse;
            *at.add(channel) = (value + 0.5).floor().min(255.0) as u8;
        }
        *at.add(3) = 255;
    }
}

struct LineBuckets {
    piece_first: usize,
    piece_last: usize,
    clip: [i64; 4],
    rows: [i64; 2],
    radius: f64,
    reach: f64,
    reach_columns: i64,
    first_column: Vec<i64>,
    piece_rows: Vec<[i64; 2]>,
    reaches: Vec<Reach>,
    counts: Vec<usize>,
    entries: Vec<usize>,
}

/// X buckets depend on the line and its horizontal clip, never on a worker's
/// row band. Build them once; all bands borrow this immutable frame-local plan.
#[inline(always)]
fn prepare_buckets(batch: &PolylineBatch<'_>, pieces: &[Piece], piece_offsets: &[usize], canvas: &Canvas<'_>) -> Vec<Option<LineBuckets>> {
    let width = canvas.width as i64;
    let top = canvas.top as i64;
    let bottom = top + canvas.height as i64;
    (0..batch.widths.len()).map(|line| {
        let piece_first = piece_offsets[line];
        let piece_last = piece_offsets[line + 1];
        if piece_last <= piece_first { return None; }
        let clip = batch.clips[line];
        let clip_left = 0.max(i64::from(clip[0]));
        let clip_top = top.max(i64::from(clip[1]));
        let clip_right = width.min(i64::from(clip[2]));
        let clip_bottom = bottom.min(i64::from(clip[3]));
        if clip_right <= clip_left || clip_bottom <= clip_top { return None; }
        let radius = (batch.widths[line] * 0.5).max(0.5);
        let reach = radius * 1.4142135623730951 + 1.0;
        let reach_columns = ceil_i64(reach);
        let fill_left = clip_left;
        let fill_right = clip_right;
        let keep_left = fill_left as f64 - reach;
        let keep_right = fill_right as f64 + reach;
        let piece_count = piece_last - piece_first;
        let columns = (fill_right - fill_left) as usize;
        let mut first_column = vec![0i64; piece_count];
        let mut last_column = vec![0i64; piece_count];
        let mut piece_rows = vec![[0i64; 2]; piece_count];
        let mut counts = vec![0usize; columns + 1];
        let mut low_y = f64::INFINITY;
        let mut high_y = f64::NEG_INFINITY;
        for index in 0..piece_count {
            let piece = pieces[piece_first + index];
            let low = piece.x0.min(piece.x1);
            let high = piece.x0.max(piece.x1);
            if high < keep_left || low > keep_right {
                first_column[index] = 1;
                last_column[index] = 0;
                continue;
            }
            let low_column = fill_left.max(floor_i64(low));
            let high_column = (fill_right - 1).min(floor_i64(high));
            first_column[index] = low_column;
            last_column[index] = high_column;
            if high_column >= low_column {
                low_y = low_y.min(piece.y0.min(piece.y1));
                high_y = high_y.max(piece.y0.max(piece.y1));
                piece_rows[index] = [floor_i64(piece.y0.min(piece.y1) - reach).saturating_sub(1),
                                     ceil_i64(piece.y0.max(piece.y1) + reach).saturating_add(2)];
            }
            for column in low_column..=high_column {
                counts[(column - fill_left) as usize] += 1;
            }
        }
        if low_y > high_y { return None; }
        // Conservative ink bounds only reject wholly empty bands. They do not
        // change the per-column row ranges or the coverage/blend arithmetic.
        let rows = [floor_i64(low_y - reach).saturating_sub(1),
                    ceil_i64(high_y + reach).saturating_add(2)];
        if rows[1] <= clip_top || rows[0] >= clip_bottom { return None; }
        let mut total = 0usize;
        for count in &mut counts[..columns] {
            let held = *count;
            *count = total;
            total += held;
        }
        counts[columns] = total;
        let mut entries = vec![0usize; total];
        for index in 0..piece_count {
            for column in first_column[index]..=last_column[index] {
                let column = (column - fill_left) as usize;
                entries[counts[column]] = index;
                counts[column] += 1;
            }
        }
        for column in (1..=columns).rev() { counts[column] = counts[column - 1]; }
        counts[0] = 0;
        let reaches = (piece_first..piece_last).map(|index| {
            let piece = pieces[index];
            reach_of(piece, (!piece.cap_end && index + 1 < piece_last).then(|| pieces[index + 1]), radius)
        }).collect();
        Some(LineBuckets { piece_first, piece_last, clip:[clip_left, clip_top, clip_right, clip_bottom],
            rows, radius, reach, reach_columns, first_column, piece_rows, reaches, counts, entries })
    }).collect()
}

#[inline(always)]
fn paint_polylines(batch: &PolylineBatch<'_>, pieces: &[Piece], buckets: &[Option<LineBuckets>], lines: &[usize], canvas: Rows) {
    let top = canvas.top as i64;
    let bottom = top + canvas.height as i64;
    let mut amount = vec![0.0; canvas.height];
    for &line in lines {
        let Some(bucket) = &buckets[line] else { continue; };
        let [clip_left, line_top, clip_right, line_bottom] = bucket.clip;
        let clip_top = top.max(line_top);
        let clip_bottom = bottom.min(line_bottom);
        if clip_bottom <= clip_top || bucket.rows[1] <= top || bucket.rows[0] >= bottom { continue; }
        let piece_first = bucket.piece_first;
        let piece_last = bucket.piece_last;
        let radius = bucket.radius;
        let reach = bucket.reach;
        let reach_columns = bucket.reach_columns;
        let fill_left = clip_left;
        let fill_right = clip_right;
        let first_column = &bucket.first_column;
        let counts = &bucket.counts;
        let entries = &bucket.entries;

        let alpha_code = f64::from(batch.colours[line][3]) / 255.0;
        for column in clip_left..clip_right {
            let cx = column as f64 + 0.5;
            let window_left = cx - reach;
            let window_right = cx + reach;
            let mut rows_low = bottom;
            let mut rows_high = top;
            let source_first = fill_left.max(column.saturating_sub(reach_columns));
            let source_last = (fill_right - 1).min(column.saturating_add(reach_columns));
            for source in source_first..=source_last {
                let source_column = (source - fill_left) as usize;
                for slot in counts[source_column]..counts[source_column + 1] {
                    let index = entries[slot];
                    if source != first_column[index].max(source_first) { continue; }
                    let [piece_top, piece_bottom] = bucket.piece_rows[index];
                    if piece_bottom <= clip_top || piece_top >= clip_bottom { continue; }
                    let piece_index = piece_first + index;
                    let piece = pieces[piece_index];
                    let next = if !piece.cap_end && piece_index + 1 < piece_last {
                        Some(pieces[piece_index + 1])
                    } else { None };
                    let dx = piece.x1 - piece.x0;
                    let row_first;
                    let row_last;
                    if dx == 0.0 {
                        let level = (radius + 0.5 - (cx - piece.x0).abs()).max(0.0).min(1.0);
                        let y_low = piece.y0.min(piece.y1);
                        let y_high = piece.y0.max(piece.y1);
                        let inner_first = clip_top.max(ceil_i64(y_low - 0.5));
                        let inner_last = clip_bottom.min(floor_i64(y_high - 0.5).saturating_add(1));
                        if level > 0.0 {
                            for row in inner_first..inner_last {
                                amount[row as usize - canvas.top] += level;
                            }
                        }
                        row_first = clip_top.max(floor_i64(y_low - reach));
                        row_last = clip_bottom.min(ceil_i64(y_high + reach).saturating_add(1));
                        for row in row_first..row_last {
                            if inner_first <= row && row < inner_last { continue; }
                            amount[row as usize - canvas.top] += segment_cover(cx, row as f64 + 0.5, piece, next, radius);
                        }
                    } else {
                        // A piece reaching no row of this band adds nothing to any
                        // pixel here; nor does it widen the rows blended below.
                        let ([body_first, body_last], join) =
                            reached_rows(cx, piece, &bucket.reaches[index], radius, [clip_top, clip_bottom]);
                        if body_first >= body_last && join.is_none() { continue; }
                        let mut t_low = (window_left - piece.x0) / dx;
                        let mut t_high = (window_right - piece.x0) / dx;
                        if t_low > t_high { std::mem::swap(&mut t_low, &mut t_high); }
                        t_low = t_low.max(0.0).min(1.0);
                        t_high = t_high.max(0.0).min(1.0);
                        let y_a = piece.y0 + t_low * (piece.y1 - piece.y0);
                        let y_b = piece.y0 + t_high * (piece.y1 - piece.y0);
                        row_first = clip_top.max(floor_i64(y_a.min(y_b) - reach));
                        row_last = clip_bottom.min(ceil_i64(y_a.max(y_b) + reach).saturating_add(1));
                        let [body_first, body_last] = [body_first.max(row_first), body_last.min(row_last)];
                        let join = join.map(|[first, last]| [first.max(row_first), last.min(row_last)])
                            .filter(|[first, last]| first < last);
                        let mut paint = |first: i64, last: i64| for row in first..last {
                            amount[row as usize - canvas.top] += segment_cover(cx, row as f64 + 0.5, piece, next, radius);
                        };
                        match join {
                            Some([first, last]) if first <= body_last && body_first <= last =>
                                paint(first.min(body_first), last.max(body_last)),
                            Some([first, last]) => { paint(body_first, body_last); paint(first, last); }
                            None => paint(body_first, body_last),
                        }
                    }
                    if row_last > row_first {
                        rows_low = rows_low.min(row_first);
                        rows_high = rows_high.max(row_last);
                    }
                }
            }
            for row in rows_low..rows_high {
                let offset = row as usize - canvas.top;
                let covered = amount[offset].min(1.0);
                amount[offset] = 0.0;
                if covered <= 1.0e-6 { continue; }
                blend(canvas, row, column, batch.colours[line], alpha_code * covered);
            }
        }
    }
}

/// One stem plus two caps form an analytic union, blended once per sample.
pub fn raster_error_bars(batch: &ErrorBarBatch<'_>, canvas: &mut Canvas<'_>) -> Result<(), String> {
    validate_canvas(canvas)?;
    if batch.y_low.len() != batch.x.len() || batch.y_high.len() != batch.x.len() {
        return Err("Error-bar coordinate arrays must have equal lengths".into());
    }
    let count = validate_offsets(batch.offsets, batch.x.len(), "error-bar offsets")?;
    validate_groups(count, batch.colours, batch.widths, batch.clips)?;
    if batch.cap_widths.len() != count || batch.cap_widths.iter().any(|value| !value.is_finite()) {
        return Err("Error-bar cap widths must be finite and match group count".into());
    }
    if count == 0 || canvas.width == 0 || canvas.height == 0 { return Ok(()); }
    canvas.for_bands(|band| paint_error_bars(batch, band))
}

fn paint_error_bars(batch: &ErrorBarBatch<'_>, mut canvas: Canvas<'_>) {
    let width = canvas.width as i64;
    let top = canvas.top as i64;
    let bottom = top + canvas.height as i64;
    for group in 0..batch.widths.len() {
        let clip = batch.clips[group];
        let clip_left = 0.max(i64::from(clip[0]));
        let clip_top = top.max(i64::from(clip[1]));
        let clip_right = width.min(i64::from(clip[2]));
        let clip_bottom = bottom.min(i64::from(clip[3]));
        if clip_right <= clip_left || clip_bottom <= clip_top { continue; }
        let radius = (batch.widths[group] * 0.5).max(0.5);
        let cap_half = (batch.cap_widths[group] * 0.5).max(0.0);
        let alpha_code = f64::from(batch.colours[group][3]) / 255.0;
        for point in batch.offsets[group] as usize..batch.offsets[group + 1] as usize {
            let px = batch.x[point];
            let mut low = batch.y_low[point];
            let mut high = batch.y_high[point];
            if !(px.is_finite() && low.is_finite() && high.is_finite()) { continue; }
            if high < low { std::mem::swap(&mut low, &mut high); }
            if high <= low { continue; }
            let half = radius.max(cap_half);
            let bar_top = if cap_half > 0.0 { low - radius } else { low };
            let bar_bottom = if cap_half > 0.0 { high + radius } else { high };
            let first_column = clip_left.max(floor_i64(px - half));
            let last_column = clip_right.min(ceil_i64(px + half));
            let first_row = clip_top.max(floor_i64(bar_top));
            let last_row = clip_bottom.min(ceil_i64(bar_bottom));
            for column in first_column..last_column {
                let x = column as f64;
                let mut stem_x = ((x + 1.0).min(px + radius) - x.max(px - radius)).max(0.0);
                let mut cap_x = ((x + 1.0).min(px + cap_half) - x.max(px - cap_half)).max(0.0);
                let shared_x = stem_x.min(cap_x);
                stem_x -= shared_x;
                cap_x -= shared_x;
                let mut row = first_row;
                while row < last_row {
                    let y = row as f64;
                    if shared_x == 0.0 && stem_x == 0.0 && y >= low + radius && y + 1.0 <= high - radius {
                        row = floor_i64(high - radius);
                        continue;
                    }
                    let full_y = ((y + 1.0).min(bar_bottom) - y.max(bar_top)).max(0.0);
                    let mut covered = shared_x * full_y;
                    if stem_x > 0.0 {
                        covered += stem_x * ((y + 1.0).min(high) - y.max(low)).max(0.0);
                    }
                    if cap_x > 0.0 {
                        let mut caps_y = full_y;
                        if low + radius < high - radius {
                            caps_y = ((y + 1.0).min(low + radius) - y.max(low - radius)).max(0.0)
                                + ((y + 1.0).min(high + radius) - y.max(high - radius)).max(0.0);
                        }
                        covered += cap_x * caps_y;
                    }
                    if covered > 0.0 {
                        blend(Rows::of(&mut canvas), row, column, batch.colours[group], alpha_code * covered.min(1.0));
                    }
                    row += 1;
                }
            }
        }
    }
}
