//! CPU scientific rasterization. Coordinates, coverage and painter order are
//! inherited from the validated v2 kernels; this crate has no Python/UI owner.
use rayon::prelude::*;

pub mod composition;
pub mod height3d;
pub mod strokes;

pub struct Canvas<'a> {
    pub pixels: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    /// Global row corresponding to pixels[0]; bands keep global coordinates.
    pub top: usize,
}

impl<'a> Canvas<'a> {
    pub fn new(pixels: &'a mut [u8], width: usize, height: usize) -> Result<Self, String> {
        if width == 0 || height == 0 || width.checked_mul(height).and_then(|n| n.checked_mul(4)) != Some(pixels.len()) {
            return Err("Canvas must be a nonempty contiguous H×W×4 byte plane".into());
        }
        Ok(Self { pixels, width, height, top: 0 })
    }

    pub fn for_bands<F>(&mut self, paint: F) -> Result<(), String>
    where F: Fn(Canvas<'_>) + Send + Sync {
        // Bands run on the caller's pool, which every panel shares.
        let width = self.width;
        let top = self.top;
        let rows = self.height.div_ceil(rayon::current_num_threads());
        let stride = width * 4;
        self.pixels.par_chunks_mut(rows * stride).enumerate().for_each(|(index, pixels)| {
            let height = pixels.len() / stride;
            paint(Canvas { pixels, width, height, top: top + index * rows });
        });
        Ok(())
    }

    /// `for_bands`, or the whole canvas as one band on this thread when the
    /// work (pixels to paint) is too little to repay waking the pool: the
    /// bands paint the same pixels either way.
    pub fn for_bands_sized<F>(&mut self, work: usize, paint: F) -> Result<(), String>
    where F: Fn(Canvas<'_>) + Send + Sync {
        if work < SERIAL_WORK {
            paint(Canvas { pixels: &mut *self.pixels, width: self.width, height: self.height, top: self.top });
            return Ok(());
        }
        self.for_bands(paint)
    }
}

/// Below this many pixels a paint runs on the calling thread.
const SERIAL_WORK: usize = 1 << 16;
