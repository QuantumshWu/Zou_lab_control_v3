//! CPU composition with the existing Agg-compatible coverage and rounding.
//! Inputs borrow prepared scientific planes; only disjoint canvas bands run
//! concurrently, and every band retains the original painter order.
use crate::Canvas;

pub struct Mask<'a> {
    pub coverage: &'a [u8],
    pub top: i64,
    pub left: i64,
    pub width: usize,
    pub height: usize,
    pub color: [u8; 4],
}

pub struct HistogramPlan<'a> {
    pub edges: &'a [f64],
    pub tops: &'a [f64],
    pub bases: &'a [f64],
    pub offsets: &'a [usize],
    pub surface_offsets: &'a [usize],
    pub colours: &'a [[u8; 4]],
    pub clips: &'a [[i64; 4]],
}

fn canvas_bounds(canvas: &Canvas<'_>) -> Result<(i128, i128), String> {
    if canvas.width == 0 || canvas.height == 0 { return Err("Canvas extent must be nonempty".into()); }
    let length = canvas.width.checked_mul(canvas.height).and_then(|size| size.checked_mul(4))
        .ok_or("Canvas extent overflows addressable storage")?;
    if canvas.pixels.len() != length { return Err("Canvas pixels differ from its extent".into()); }
    let bottom = canvas.top.checked_add(canvas.height).ok_or("Canvas row extent overflows")?;
    Ok((canvas.top as i128, bottom as i128))
}

/// Masks are already listed in painter order. Static masks and individual
/// text glyph masks remain separate borrowed blocks, never a concatenated atlas.
pub fn replay_foreground_masks(canvas: &mut Canvas<'_>, masks: &[Mask<'_>]) -> Result<(), String> {
    canvas_bounds(canvas)?;
    for mask in masks {
        if mask.width.checked_mul(mask.height) != Some(mask.coverage.len()) {
            return Err("Foreground mask coverage differs from its extent".into());
        }
    }
    if masks.is_empty() { return Ok(()); }
    let work = masks.iter().map(|mask| mask.coverage.len()).sum();
    canvas.for_bands_sized(work, |band| {
        let top = band.top as i128;
        let bottom = top + band.height as i128;
        for mask in masks {
            let left = i128::from(mask.left).max(0);
            let right = (i128::from(mask.left) + mask.width as i128).min(band.width as i128);
            let first = i128::from(mask.top).max(top);
            let last = (i128::from(mask.top) + mask.height as i128).min(bottom);
            if right <= left || last <= first { continue; }
            for y in first..last {
                let coverage_row = (y - i128::from(mask.top)) as usize * mask.width;
                let pixel_row = (y - top) as usize * band.width * 4;
                for x in left..right {
                    let cover = u32::from(mask.coverage[coverage_row + (x - i128::from(mask.left)) as usize]);
                    let product = u32::from(mask.color[3]) * cover + 128;
                    let alpha = ((product >> 8) + product) >> 8;
                    if alpha == 0 { continue; }
                    let pixel = &mut band.pixels[pixel_row + x as usize * 4..][..4];
                    if alpha == 255 { pixel[..3].copy_from_slice(&mask.color[..3]); pixel[3] = 255; continue; }
                    let old_alpha = u32::from(pixel[3]);
                    let denominator = ((alpha + old_alpha) << 8) - alpha * old_alpha;
                    for channel in 0..3 {
                        let premultiplied = u32::from(pixel[channel]) * old_alpha;
                        pixel[channel] = ((premultiplied * (256 - alpha)
                            + (u32::from(mask.color[channel]) << 8) * alpha) / denominator) as u8;
                    }
                    pixel[3] = (denominator >> 8) as u8;
                }
            }
        }
    })?;
    Ok(())
}

fn validate_offsets(offsets: &[usize], count: usize, end: usize, name: &str) -> Result<(), String> {
    if count.checked_add(1) != Some(offsets.len()) || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || offsets.last().is_some_and(|last| *last > end) {
        return Err(format!("{name} offsets do not partition their input"));
    }
    Ok(())
}

#[inline]
fn fill_channel(destination: u8, source: u8, alpha: u32) -> u8 {
    if alpha >= 255 { return source; }
    let held = u32::from(destination) * 255;
    let scale = ((alpha + 255) << 8) - alpha * 255;
    // Algebraically the original signed Agg numerator, without unsigned
    // underflow when the source channel is darker than the destination.
    (((u32::from(source) << 8) * alpha + held * (256 - alpha)) / scale) as u8
}

#[inline]
fn snapped(value: f64) -> i128 { (value + 0.5).floor() as i128 }

pub fn raster_histogram_bars(canvas: &mut Canvas<'_>, plan: &HistogramPlan<'_>) -> Result<(), String> {
    canvas_bounds(canvas)?;
    let groups = plan.colours.len();
    if plan.tops.len() != plan.edges.len() || plan.bases.len() != groups {
        return Err("Histogram planes differ from their distributions".into());
    }
    validate_offsets(plan.offsets, groups, plan.edges.len(), "Histogram distribution")?;
    validate_offsets(plan.surface_offsets, plan.clips.len(), groups, "Histogram surface")?;
    if groups == 0 { return Ok(()); }
    canvas.for_bands(|band| {
        let band_top = band.top as i128;
        let band_bottom = band_top + band.height as i128;
        for (surface, clip) in plan.clips.iter().enumerate() {
            let clip_left = i128::from(clip[0]).max(0);
            let clip_top = i128::from(clip[1]).max(band_top);
            let clip_right = i128::from(clip[2]).min(band.width as i128);
            let clip_bottom = i128::from(clip[3]).min(band_bottom);
            if clip_right <= clip_left || clip_bottom <= clip_top { continue; }
            for group in plan.surface_offsets[surface]..plan.surface_offsets[surface + 1] {
                let color = plan.colours[group];
                let alpha = u32::from(color[3]);
                let base = plan.bases[group];
                if alpha == 0 || !base.is_finite() { continue; }
                let base_row = snapped(base);
                for bar in plan.offsets[group]..plan.offsets[group + 1].saturating_sub(1) {
                    let left = plan.edges[bar];
                    let right = plan.edges[bar + 1];
                    let top = plan.tops[bar];
                    if !left.is_finite() || !right.is_finite() || !top.is_finite() { continue; }
                    let (left, right, top) = (snapped(left), snapped(right), snapped(top));
                    let x0 = left.min(right).max(clip_left);
                    let x1 = left.max(right).min(clip_right);
                    let y0 = top.min(base_row).max(clip_top);
                    let y1 = top.max(base_row).min(clip_bottom);
                    if x1 <= x0 || y1 <= y0 { continue; }
                    for y in y0..y1 {
                        let start = ((y - band_top) as usize * band.width + x0 as usize) * 4;
                        let end = start + (x1 - x0) as usize * 4;
                        for pixel in band.pixels[start..end].chunks_exact_mut(4) {
                            for channel in 0..3 { pixel[channel] = fill_channel(pixel[channel], color[channel], alpha); }
                            pixel[3] = 255;
                        }
                    }
                }
            }
        }
    })?;
    Ok(())
}

#[derive(Clone, Copy)]
pub enum ImageValues<'a> {
    Bool(&'a [u8]), U8(&'a [u8]), U16(&'a [u16]), U32(&'a [u32]), U64(&'a [u64]),
    I8(&'a [i8]), I16(&'a [i16]), I32(&'a [i32]), I64(&'a [i64]),
    F32(&'a [f32]), F64(&'a [f64]),
}
impl ImageValues<'_> {
    fn len(self) -> usize {
        match self {
            Self::Bool(values) | Self::U8(values) => values.len(),
            Self::U16(values) => values.len(), Self::U32(values) => values.len(), Self::U64(values) => values.len(),
            Self::I8(values) => values.len(), Self::I16(values) => values.len(),
            Self::I32(values) => values.len(), Self::I64(values) => values.len(),
            Self::F32(values) => values.len(), Self::F64(values) => values.len(),
        }
    }
}

pub struct ImagePlan<'a> {
    pub values: ImageValues<'a>,
    pub valid: Option<&'a [u8]>,
    pub shape: [usize; 3],
    pub blits: &'a [[i64; 4]],
    pub clips: &'a [[i64; 4]],
    pub affines: &'a [[f64; 6]],
    pub lut: &'a [[u8; 4]; 256],
    pub vmin: f64,
    pub span: f64,
    pub single: bool,
}

fn image_rectangle(blit: &[i64; 4], clip: &[i64; 4], width: usize, top: i128, bottom: i128)
    -> Option<[i128; 4]>
{
    if blit[2] <= 0 || blit[3] <= 0 { return None; }
    let left = i128::from(blit[0]).max(i128::from(clip[0])).max(0);
    let right = (i128::from(blit[0]) + i128::from(blit[2])).min(i128::from(clip[2])).min(width as i128);
    let first = i128::from(blit[1]).max(i128::from(clip[1])).max(top);
    let last = (i128::from(blit[1]) + i128::from(blit[3])).min(i128::from(clip[3])).min(bottom);
    (right > left && last > first).then_some([left, first, right, last])
}

#[inline]
fn fixed_coordinate(value: f64) -> Option<i128> {
    // Agg's fixed-coordinate conversion is only defined in its integer
    // domain. Reject an unrepresentable transform instead of overflowing it.
    if !value.is_finite() || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value) {
        return None;
    }
    Some(if value < 0.0 { (value - 0.5) as i128 } else { (value + 0.5) as i128 })
}

#[inline]
fn transformed_span(affine: &[f64; 6], y_up: f64, width: i64) -> [f64; 4] {
    let last = width as f64 + 0.5;
    [
        (0.5 * affine[0] + y_up * affine[2] + affine[4]) * 256.0,
        (0.5 * affine[1] + y_up * affine[3] + affine[5]) * 256.0,
        (last * affine[0] + y_up * affine[2] + affine[4]) * 256.0,
        (last * affine[1] + y_up * affine[3] + affine[5]) * 256.0,
    ]
}

struct Dda { value: i128, left: i128, remainder: i128, modulus: i128, count: i128 }
impl Dda {
    fn new(first: i128, last: i128, count: i64) -> Self {
        let count = i128::from(count);
        let delta = last - first;
        let mut left = delta / count;
        let mut remainder = delta - left * count;
        let mut modulus = remainder;
        if modulus <= 0 { modulus += count; remainder += count; left -= 1; }
        Self { value:first, left, remainder, modulus:modulus - count, count }
    }

    fn skip(&mut self, steps: i128) {
        // Exact closed form of the original integer DDA steps. Clipped-away
        // pixels need not be visited to establish the first visible sample.
        let advanced = self.modulus + steps * self.remainder;
        let carries = (advanced + self.count - 1) / self.count;
        self.value += steps * self.left + carries;
        self.modulus = advanced - carries * self.count;
    }

    #[inline]
    fn step(&mut self) {
        self.modulus += self.remainder;
        self.value += self.left;
        if self.modulus > 0 { self.modulus -= self.count; self.value += 1; }
    }
}

fn paint_images<T: Copy + Sync>(canvas: &mut Canvas<'_>, plan: &ImagePlan<'_>, values: &[T],
                              value_of: impl Fn(T) -> f64 + Sync) -> Result<(), String> {
    let [cells, source_rows, source_columns] = plan.shape;
    canvas.for_bands(|band| {
        let band_top = band.top as i128;
        let band_bottom = band_top + band.height as i128;
        for cell in 0..cells {
            let blit = plan.blits[cell];
            let Some([left, top, right, bottom]) = image_rectangle(&blit, &plan.clips[cell], band.width, band_top, band_bottom)
                else { continue; };
            for row in top..bottom {
                let v = row - i128::from(blit[1]);
                let y_up = (i128::from(blit[3]) - 1 - v) as f64 + 0.5;
                let span = transformed_span(&plan.affines[cell], y_up, blit[2]);
                // Visible affine extrema were checked before any canvas write.
                let coordinates = span.map(|value| fixed_coordinate(value).expect("validated image transform"));
                let mut x = Dda::new(coordinates[0], coordinates[2], blit[2]);
                let mut y = Dda::new(coordinates[1], coordinates[3], blit[2]);
                let first = left - i128::from(blit[0]);
                x.skip(first); y.skip(first);
                let pixel_row = (row - band_top) as usize * band.width * 4;
                for column in left..right {
                    let source_x = x.value >> 8;
                    let source_y = y.value >> 8;
                    x.step(); y.step();
                    if source_x < 0 || source_y < 0 || source_x >= source_columns as i128 || source_y >= source_rows as i128 {
                        continue;
                    }
                    let index = (cell * source_rows + source_y as usize) * source_columns + source_x as usize;
                    if plan.valid.is_some_and(|valid| valid[index] == 0) { continue; }
                    let value = value_of(values[index]);
                    if !value.is_finite() { continue; }
                    let scaled = if plan.single {
                        let shifted = (f64::from(value as f32) - plan.vmin) as f32;
                        let unit = (f64::from(shifted) / plan.span) as f32;
                        f64::from(unit * 256.0_f32)
                    } else { ((value - plan.vmin) / plan.span) * 256.0 };
                    let code = if scaled < 0.0 { 0 } else if scaled >= 256.0 { 255 } else { scaled as usize };
                    let start = pixel_row + column as usize * 4;
                    band.pixels[start..start + 4].copy_from_slice(&plan.lut[code]);
                }
            }
        }
    })
}

pub fn raster_prepared_images(canvas: &mut Canvas<'_>, plan: &ImagePlan<'_>) -> Result<(), String> {
    let (top, bottom) = canvas_bounds(canvas)?;
    let [cells, rows, columns] = plan.shape;
    let count = cells.checked_mul(rows).and_then(|size| size.checked_mul(columns))
        .ok_or("Image source extent overflows addressable storage")?;
    if plan.values.len() != count || plan.valid.is_some_and(|valid| valid.len() != count)
        || plan.blits.len() != cells || plan.clips.len() != cells || plan.affines.len() != cells {
        return Err("Prepared image buffers differ from their declared extents".into());
    }
    if count == 0 { return Ok(()); }
    if !plan.vmin.is_finite() || !plan.span.is_finite() || plan.span <= 0.0 {
        return Err("Prepared image normalization requires a finite minimum and positive span".into());
    }
    for cell in 0..cells {
        let blit = plan.blits[cell];
        let Some([_, first, _, last]) = image_rectangle(&blit, &plan.clips[cell], canvas.width, top, bottom)
            else { continue; };
        for row in [first, last - 1] {
            let y_up = (i128::from(blit[3]) - 1 - (row - i128::from(blit[1]))) as f64 + 0.5;
            if transformed_span(&plan.affines[cell], y_up, blit[2]).into_iter().any(|value| fixed_coordinate(value).is_none()) {
                return Err("Prepared image transform exceeds finite fixed-point coordinates".into());
            }
        }
    }
    match plan.values {
        ImageValues::Bool(values) => paint_images(canvas, plan, values, |value| if value != 0 {1.0} else {0.0}),
        ImageValues::U8(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::U16(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::U32(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::U64(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::I8(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::I16(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::I32(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::I64(values) => paint_images(canvas, plan, values, |value| value as f64),
        ImageValues::F32(values) => paint_images(canvas, plan, values, |value| f64::from(value)),
        ImageValues::F64(values) => paint_images(canvas, plan, values, |value| value),
    }
}
