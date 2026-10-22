//! One immutable axes transform shared by native and raster interaction
//! (`_axis_transform.AxisTransform`): the axes box the window sends, in
//! top-origin fractions of the canvas, with the drawn (display) limits and
//! scales and the canonical limits and unit mappings a pointer is read in.

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::CrosshairPoint;
use crate::render::scale::AxisScale;

use super::pymath::{axis_value, divide, interpolate};

/// One of the two axes of a box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis { X, Y }

/// The physical-pixel extent Matplotlib pointer events are measured in
/// (`canvas_physical_size`): both sides positive and finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasSize { pub width: f64, pub height: f64 }

impl CanvasSize {
    pub fn new(width: f64, height: f64) -> Result<Self> {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(PlotError::value("interactive canvas physical size must be positive and finite"));
        }
        Ok(Self { width, height })
    }
}

/// Exact top-origin plot box plus display and canonical coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisTransform {
    pub role: String,
    pub cell_index: Option<u64>,
    /// `(left, top, right, bottom)` as fractions of the canvas, top origin.
    pub bounds: (f64, f64, f64, f64),
    pub x_limits: (f64, f64),
    pub y_limits: (f64, f64),
    pub canonical_x_limits: (f64, f64),
    pub canonical_y_limits: (f64, f64),
    /// How each drawn axis maps value to position.
    pub x_scale: AxisScale,
    pub y_scale: AxisScale,
    /// A display-unit axis' mapping of a canonical value (an
    /// [`AxisScale::Converted`]); `None` where the axis is read as drawn.
    pub canonical_x_scale: Option<AxisScale>,
    pub canonical_y_scale: Option<AxisScale>,
}

/// The canonical value `fraction` of the way along one drawn axis (`_read`).
fn read(fraction: f64, canonical: (f64, f64), display: (f64, f64), scale: &AxisScale, canonical_scale: Option<&AxisScale>) -> Result<f64> {
    match canonical_scale {
        None => interpolate(scale, canonical.0, canonical.1, fraction),
        Some(canonical_scale) => {
            // Drawn straight in its display unit: interpolated between the
            // drawn limits, only the value under the pointer converted.
            let (start, stop) = (scale.space(display.0), scale.space(display.1));
            axis_value(canonical_scale, start + fraction * (stop - start))
        }
    }
}

impl AxisTransform {
    /// A transform with linear scales and no display-unit mapping.
    pub fn linear(role: &str, cell_index: Option<u64>, bounds: (f64, f64, f64, f64), x_limits: (f64, f64), y_limits: (f64, f64)) -> Self {
        Self { role: role.to_owned(), cell_index, bounds, x_limits, y_limits, canonical_x_limits: x_limits, canonical_y_limits: y_limits,
               x_scale: AxisScale::Linear, y_scale: AxisScale::Linear, canonical_x_scale: None, canonical_y_scale: None }
    }

    fn axis(&self, axis: Axis) -> ((f64, f64), (f64, f64), &AxisScale, Option<&AxisScale>) {
        match axis {
            Axis::X => (self.x_limits, self.canonical_x_limits, &self.x_scale, self.canonical_x_scale.as_ref()),
            Axis::Y => (self.y_limits, self.canonical_y_limits, &self.y_scale, self.canonical_y_scale.as_ref()),
        }
    }

    /// How a canonical value on one axis is placed where it is straight:
    /// hitting, moving and letting go of a selector all measure here.
    pub fn interaction_scale(&self, axis: Axis) -> &AxisScale {
        let (_, _, scale, canonical_scale) = self.axis(axis);
        canonical_scale.unwrap_or(scale)
    }

    /// One drawn axis' length in [`AxisTransform::interaction_scale`]'s space.
    pub fn drawn_span(&self, axis: Axis) -> f64 {
        let (display, canonical, scale, canonical_scale) = self.axis(axis);
        let ends = if canonical_scale.is_none() { canonical } else { display };
        (scale.space(ends.1) - scale.space(ends.0)).abs()
    }

    /// Display-space axes data into top-origin widget fractions.
    pub fn display_to_normalized(&self, x: f64, y: f64) -> (f64, f64) {
        let (left, top, right, bottom) = self.bounds;
        let nx = left + self.x_scale.fraction_of(x, self.x_limits.0, self.x_limits.1) * (right - left);
        let ny = top + self.y_scale.fraction_of(y, self.y_limits.1, self.y_limits.0) * (bottom - top);
        (nx, ny)
    }

    /// Top-origin widget fractions into a canonical data point.
    pub fn canonical_from_normalized(&self, nx: f64, ny: f64) -> Result<CrosshairPoint> {
        let (x, y) = self.canonical_values(nx, ny)?;
        CrosshairPoint::new(x, y)
    }

    fn fractions(&self, nx: f64, ny: f64) -> Result<(f64, f64)> {
        let (left, top, right, bottom) = self.bounds;
        Ok((divide(nx - left, right - left)?, divide(ny - top, bottom - top)?))
    }

    /// The canonical coordinates under widget fractions, unvalidated: over
    /// what an axis' canonical unit cannot express they read NaN or infinite.
    pub fn canonical_values(&self, nx: f64, ny: f64) -> Result<(f64, f64)> {
        let (tx, ty) = self.fractions(nx, ny)?;
        let (x0, x1) = self.canonical_x_limits;
        let (y0, y1) = self.canonical_y_limits;
        if self.role == "distribution" {
            // The rail is the value axis stood on its side.
            let x_scale = self.canonical_x_scale.as_ref().unwrap_or(&self.x_scale);
            return Ok((interpolate(x_scale, x1, x0, ty)?, 0.0));
        }
        let y_display = (self.y_limits.1, self.y_limits.0);
        Ok((read(tx, (x0, x1), self.x_limits, &self.x_scale, self.canonical_x_scale.as_ref())?,
            read(ty, (y1, y0), y_display, &self.y_scale, self.canonical_y_scale.as_ref())?))
    }

    /// Top-origin widget fractions into a display-space data point.
    pub fn display_from_normalized(&self, nx: f64, ny: f64) -> Result<CrosshairPoint> {
        let (tx, ty) = self.fractions(nx, ny)?;
        let (x0, x1) = self.x_limits;
        let (y0, y1) = self.y_limits;
        if self.role == "distribution" {
            return CrosshairPoint::new(interpolate(&self.y_scale, y1, y0, ty)?, 0.0);
        }
        CrosshairPoint::new(interpolate(&self.x_scale, x0, x1, tx)?, interpolate(&self.y_scale, y1, y0, ty)?)
    }

    /// A bottom-origin physical pixel as top-origin canvas fractions.
    pub fn event_normalized(pixel: Option<(f64, f64)>, canvas: CanvasSize) -> Option<(f64, f64)> {
        pixel.map(|(x, y)| (x / canvas.width, 1.0 - y / canvas.height))
    }

    /// The canonical reading of one pointer event at `pixel` (`None` without
    /// a position).  Off the axis' canonical unit a coordinate reads NaN or
    /// infinite; the colour rail validates its reading as a point.
    pub fn canonical_reading(&self, pixel: Option<(f64, f64)>, canvas: CanvasSize) -> Result<Option<(f64, f64)>> {
        let Some((nx, ny)) = Self::event_normalized(pixel, canvas) else { return Ok(None) };
        if self.role == "distribution" {
            let point = self.canonical_from_normalized(nx, ny)?;
            return Ok(Some((point.x, point.y)));
        }
        self.canonical_values(nx, ny).map(Some)
    }

    /// The display-space point under one pointer event.
    pub fn display_point(&self, pixel: Option<(f64, f64)>, canvas: CanvasSize) -> Result<Option<CrosshairPoint>> {
        match Self::event_normalized(pixel, canvas) { None => Ok(None), Some((nx, ny)) => self.display_from_normalized(nx, ny).map(Some) }
    }

    /// Display-space data into bottom-origin physical canvas pixels.
    pub fn display_to_pixel(&self, x: f64, y: f64, canvas: CanvasSize) -> (f64, f64) {
        let (nx, ny) = self.display_to_normalized(x, y);
        (nx * canvas.width, (1.0 - ny) * canvas.height)
    }
}
