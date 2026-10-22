//! The drawn surface's own creases (`_stroke_rims`): find every pixel beside
//! a face change that involves a bar, stamp a round stroke's coverage around
//! it (the strongest stamp reaching a pixel wins), and blend the rim colour
//! in float32. A maximum over one finite set is the same number in any
//! order, so bands of rows run in parallel, each re-reading its halo rows.

use rayon::prelude::*;

/// `_rim_stamp`: coverage weights of a round stroke `width` pixels wide.
pub fn rim_stamp(width: f64) -> (Vec<f32>, usize) {
    let half = width * 0.5;
    let radius = ((half - 1.0).ceil() as i64).max(1) as usize;
    let side = 2 * radius + 1;
    let mut weights = Vec::with_capacity(side * side);
    for dy in 0..side {
        for dx in 0..side {
            let oy = dy as f64 - radius as f64;
            let ox = dx as f64 - radius as f64;
            let distance = oy.hypot(ox);
            let value = half - distance;
            let clipped = if value < 0.0 { 0.0 } else if value > 1.0 { 1.0 } else { value };
            weights.push(clipped as f32);
        }
    }
    (weights, radius)
}

/// Whether a pixel sits beside a crease: a 4-neighbour of another face,
/// either of the two being a bar's.
#[inline(always)]
fn crease(id_plane: &[i32], width: usize, height: usize, sy: usize, sx: usize) -> bool {
    let face = id_plane[sy * width + sx];
    let differs = |other: i32| other != face && (face >= 4 || other >= 4);
    (sx + 1 < width && differs(id_plane[sy * width + sx + 1]))
        || (sx > 0 && differs(id_plane[sy * width + sx - 1]))
        || (sy + 1 < height && differs(id_plane[(sy + 1) * width + sx]))
        || (sy > 0 && differs(id_plane[(sy - 1) * width + sx]))
}

/// One row's crease mask as bits (bit x of word x / 64).
fn crease_row(id_plane: &[i32], width: usize, height: usize, sy: usize, bits: &mut [u64]) {
    bits.fill(0);
    let mut start = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the feature was detected on this CPU.
            start = unsafe { crease_row_avx2(id_plane, width, height, sy, bits) };
        }
    }
    for sx in start..width {
        if crease(id_plane, width, height, sy, sx) {
            bits[sx >> 6] |= 1u64 << (sx & 63);
        }
    }
}

/// The interior of a row eight pixels at a time; returns where the scalar
/// tail takes over. The same integer predicate as `crease`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn crease_row_avx2(id_plane: &[i32], width: usize, height: usize, sy: usize, bits: &mut [u64]) -> usize {
    use std::arch::x86_64::*;
    if width < 10 {
        return 0;
    }
    // pixel 0 has no left neighbour: the scalar test takes it
    if crease(id_plane, width, height, sy, 0) {
        bits[0] |= 1;
    }
    let row = id_plane[sy * width..(sy + 1) * width].as_ptr();
    let up = (sy > 0).then(|| id_plane[(sy - 1) * width..sy * width].as_ptr());
    let down = (sy + 1 < height).then(|| id_plane[(sy + 1) * width..(sy + 2) * width].as_ptr());
    let mut sx = 1;
    // Macros, not closures: a closure would not carry the target features.
    macro_rules! load {
        ($base:expr, $at:expr) => {
            _mm256_loadu_si256($base.add($at) as *const __m256i)
        };
    }
    // SAFETY: every load reads [sx - 1, sx + 9) inside a row of `width`.
    unsafe {
        let three = _mm256_set1_epi32(3);
        macro_rules! differs {
            ($face:expr, $face_bar:expr, $other:expr) => {{
                let other = $other;
                let same = _mm256_cmpeq_epi32($face, other);
                let bar = _mm256_or_si256($face_bar, _mm256_cmpgt_epi32(other, three));
                _mm256_andnot_si256(same, bar)
            }};
        }
        while sx + 9 <= width {
            let face = load!(row, sx);
            let face_bar = _mm256_cmpgt_epi32(face, three);
            let mut any = _mm256_or_si256(differs!(face, face_bar, load!(row, sx + 1)), differs!(face, face_bar, load!(row, sx - 1)));
            if let Some(down) = down {
                any = _mm256_or_si256(any, differs!(face, face_bar, load!(down, sx)));
            }
            if let Some(up) = up {
                any = _mm256_or_si256(any, differs!(face, face_bar, load!(up, sx)));
            }
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(any)) as u32 as u64;
            if mask != 0 {
                let word = sx >> 6;
                let shift = sx & 63;
                bits[word] |= mask << shift;
                if shift > 56 {
                    bits[word + 1] |= mask >> (64 - shift);
                }
            }
            sx += 8;
        }
    }
    sx
}

/// Draw the creases of `id_plane` into `out` (H×W×4) at `width` pixels.
pub fn stroke_rims(out: &mut [u8], id_plane: &[i32], width_px: usize, height: usize, rim_rgb: [f64; 3], width: f64) {
    // Python returns at `width <= 0.0` (and raises on NaN in the stamp).
    if !(width > 0.0) {
        return;
    }
    let (weights, radius) = rim_stamp(width);
    let side = 2 * radius + 1;
    // The stamp offsets that can ever win: a zero weight never beats the
    // zero a pixel starts at. weights[sy - ty + r, sx - tx + r] reaches
    // (ty, tx) = (sy - oy, sx - ox) for the offset (oy, ox) of its index.
    let taps: Vec<(isize, isize, f32)> = (0..side * side)
        .filter(|&index| weights[index] > 0.0)
        .map(|index| ((index / side) as isize - radius as isize, (index % side) as isize - radius as isize, weights[index]))
        .collect();
    let target = [
        rim_rgb[0] as f32 * 255.0f32,
        rim_rgb[1] as f32 * 255.0f32,
        rim_rgb[2] as f32 * 255.0f32,
    ];
    let w = width_px;
    let words = w.div_ceil(64);
    let band_rows = 32usize;
    let stamp = |(band, pixels): (usize, &mut [u8])| {
        let lo = band * band_rows;
        let hi = lo + pixels.len() / (4 * w);
        BAND.with(|cell| {
            let mut buffers = cell.borrow_mut();
            let (coverage, bits) = &mut *buffers;
            coverage.clear();
            coverage.resize((hi - lo) * w, 0.0);
            bits.resize(words, 0);
            let sy0 = lo.saturating_sub(radius);
            let sy1 = (hi + radius).min(height);
            for sy in sy0..sy1 {
                crease_row(id_plane, w, height, sy, bits);
                for (word, &mask) in bits.iter().enumerate() {
                    let mut mask = mask;
                    while mask != 0 {
                        let sx = word * 64 + mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        for &(oy, ox, weight) in &taps {
                            let ty = sy as isize - oy;
                            let tx = sx as isize - ox;
                            if ty < lo as isize || ty >= hi as isize || tx < 0 || tx >= w as isize {
                                continue;
                            }
                            let cover = &mut coverage[(ty as usize - lo) * w + tx as usize];
                            if weight > *cover {
                                *cover = weight;
                            }
                        }
                    }
                }
            }
            // Coverage is never negative: a group of eight whose maximum is
            // zero has nothing to blend, and most of a frame is such groups.
            for (group, fractions) in coverage.chunks(8).enumerate() {
                if fractions.iter().fold(0.0f32, |m, &f| if f > m { f } else { m }) <= 0.0 {
                    continue;
                }
                for (offset, &fraction) in fractions.iter().enumerate() {
                    if fraction <= 0.0f32 {
                        continue;
                    }
                    let index = group * 8 + offset;
                    let pixel = &mut pixels[index * 4..index * 4 + 3];
                    for ch in 0..3 {
                        let painted = pixel[ch] as f32;
                        let value = painted + (target[ch] - painted) * fraction + 0.5f32;
                        pixel[ch] = value as u8;
                    }
                }
            }
        });
    };
    out.par_chunks_mut(band_rows * w * 4).enumerate().for_each(stamp);
}

thread_local! {
    static BAND: std::cell::RefCell<(Vec<f32>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector crease rows are the scalar predicate, pixel for pixel.
    #[test]
    fn crease_rows_match_the_scalar_predicate() {
        let (width, height) = (203usize, 37usize);
        let mut state = 12345u64;
        let ids: Vec<i32> = (0..width * height)
            .map(|i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // long runs with occasional changes, background and bars
                if (state >> 60) < 3 { ((state >> 33) % 9) as i32 } else { (i / 7 % 5) as i32 + 3 }
            })
            .collect();
        let mut bits = vec![0u64; width.div_ceil(64)];
        for sy in 0..height {
            crease_row(&ids, width, height, sy, &mut bits);
            for sx in 0..width {
                assert_eq!(bits[sx >> 6] >> (sx & 63) & 1 == 1, crease(&ids, width, height, sy, sx), "({sy}, {sx})");
            }
        }
    }
}
