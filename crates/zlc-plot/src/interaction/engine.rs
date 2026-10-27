//! Backend-neutral pointer gesture state (`_gesture_engine`): the hit tests
//! of a range end and an area handle, the pan translation, the colour-limit
//! drag, and the record of each pointer gesture a session holds.  Every
//! helper works on transforms and values alone.

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState, Viewport};
use crate::render::scale::AxisScale;

use super::drag::{drag_numeric_range, DragHandle};
use super::pymath::{axis_value, divide, isclose, py_max, py_min};
use super::transform::AxisTransform;

/// Resolve a bounded numeric drag against its nearest endpoint
/// (`range_endpoint_hit`): the score is the distance in `tolerance`s where
/// the axis is straight, LOW winning a tie; `None` beyond one tolerance.
pub fn range_endpoint_hit(value: NumericRange, coordinate: f64, tolerance: f64, scale: &AxisScale) -> Result<Option<(f64, DragHandle)>> {
    let position = scale.space(coordinate);
    let low = (divide((position - scale.space(value.low)).abs(), tolerance)?, DragHandle::Low);
    let high = (divide((position - scale.space(value.high)).abs(), tolerance)?, DragHandle::High);
    let best = if high.0 < low.0 { high } else { low };
    Ok((best.0 <= 1.0).then_some(best))
}

/// Choose the area body or handle under one pixel-space pointer
/// (`area_drag_handle`).  `pixel_point` maps a data coordinate to canvas
/// pixels and is the only frontend-supplied operation; edge handles sit at
/// the on-screen middles of their corners.
pub fn area_drag_handle(
    value: RectangleRange,
    mouse: (f64, f64),
    mut pixel_point: impl FnMut((f64, f64)) -> Result<(f64, f64)>,
    handle_radius: f64,
) -> Result<Option<(f64, DragHandle)>> {
    let (x0, x1, y0, y1) = (value.x.low, value.x.high, value.y.low, value.y.high);
    let bottom_left = pixel_point((x0, y0))?;
    let bottom_right = pixel_point((x1, y0))?;
    let top_right = pixel_point((x1, y1))?;
    let top_left = pixel_point((x0, y1))?;
    let middle = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
    let handles = [
        (bottom_left, DragHandle::BottomLeft),
        (middle(bottom_left, bottom_right), DragHandle::Bottom),
        (bottom_right, DragHandle::BottomRight),
        (middle(bottom_right, top_right), DragHandle::Right),
        (top_right, DragHandle::TopRight),
        (middle(top_left, top_right), DragHandle::Top),
        (top_left, DragHandle::TopLeft),
        (middle(bottom_left, top_left), DragHandle::Left),
    ];
    let mut best: Option<(f64, DragHandle)> = None;
    for (point, handle) in handles {
        let (dx, dy) = (point.0 - mouse.0, point.1 - mouse.1);
        let score = divide((dx * dx + dy * dy).sqrt(), handle_radius)?;
        if best.is_none_or(|(held, _)| score < held) { best = Some((score, handle)); }
    }
    let best = best.expect("eight handles");
    if best.0 <= 1.0 { return Ok(Some(best)); }
    let corners = [bottom_left, bottom_right, top_right, top_left];
    let low = (np_min(corners.map(|c| c.0)), np_min(corners.map(|c| c.1)));
    let high = (np_max(corners.map(|c| c.0)), np_max(corners.map(|c| c.1)));
    let inside = low.0 <= mouse.0 && low.1 <= mouse.1 && mouse.0 <= high.0 && mouse.1 <= high.1;
    Ok(inside.then_some((1.0, DragHandle::Body)))
}

fn np_min(values: [f64; 4]) -> f64 {
    values.into_iter().reduce(|a, b| if a.is_nan() || b.is_nan() { f64::NAN } else if b < a { b } else { a }).unwrap()
}

fn np_max(values: [f64; 4]) -> f64 {
    values.into_iter().reduce(|a, b| if a.is_nan() || b.is_nan() { f64::NAN } else if b > a { b } else { a }).unwrap()
}

/// The translated viewport for one pointer position (`pan_rectangle`), or
/// `None` while the hand has not moved the view; only an image pans y.
pub fn pan_rectangle(
    origin: CrosshairPoint,
    point: CrosshairPoint,
    x: NumericRange,
    y: NumericRange,
    image_like: bool,
    x_scale: &AxisScale,
    y_scale: &AxisScale,
) -> Result<Option<RectangleRange>> {
    let dx = x_scale.space(origin.x) - x_scale.space(point.x);
    let dy = y_scale.space(origin.y) - y_scale.space(point.y);
    if isclose(dx, 0.0, 0.0, 1.0e-15) && (!image_like || isclose(dy, 0.0, 0.0, 1.0e-15)) { return Ok(None); }
    let shift = |range: NumericRange, scale: &AxisScale, delta: f64| -> Result<NumericRange> {
        let low = axis_value(scale, scale.space(range.low) + delta)?;
        let high = axis_value(scale, scale.space(range.high) + delta)?;
        NumericRange::new(low, high)
    };
    let x = shift(x, x_scale, dx)?;
    let y = if image_like { shift(y, y_scale, dy)? } else { y };
    Ok(Some(RectangleRange { x, y }))
}

/// One canonical colour-scale gesture, separate from data selectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorLimitDrag {
    pub original: NumericRange,
    pub candidate: NumericRange,
    pub handle: DragHandle,
    pub origin: f64,
    pub bounds: NumericRange,
    pub minimum_span: f64,
}

impl ColorLimitDrag {
    /// The candidate differs from the original beyond binary noise:
    /// `allclose(rtol=1e-12)` with the absolute part relative to the rail
    /// (v2's `atol=1e-15` was the whole rail of a small-valued image, whose
    /// every drag then let go unchanged).
    pub fn changed(&self) -> bool {
        let atol = 1.0e-12 * self.bounds.span();
        !(isclose(self.candidate.low, self.original.low, 1.0e-12, atol) && isclose(self.candidate.high, self.original.high, 1.0e-12, atol))
    }

    /// The drag with the pointer at `position`.
    pub fn moved(&self, position: f64) -> Result<Self> {
        let candidate = drag_numeric_range(self.original, self.handle, self.origin, position, Some(self.minimum_span),
                                           Some(self.bounds), &AxisScale::Linear)?
            .ok_or_else(|| PlotError::type_error("a linear colour-limit drag always has a candidate"))?;
        Ok(Self { candidate, ..*self })
    }
}

/// Which backend axes a gesture lives on: the `role[:cell]` the renderer
/// names each axes with.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AxesId { pub role: String, pub cell_index: Option<u64> }

impl AxesId {
    /// The axes a transform was taken from.
    pub fn of(transform: &AxisTransform) -> Self { Self { role: transform.role.clone(), cell_index: transform.cell_index } }

    /// The renderer's axes id `role[:cell]` read as the session reads it: an
    /// empty id or role is `main`, a suffix of digits the cell index.
    pub fn from_gid(gid: &str) -> Self {
        let gid = if gid.is_empty() { "main" } else { gid };
        let (role, suffix) = match gid.split_once(':') { Some((role, suffix)) => (role, Some(suffix)), None => (gid, None) };
        let cell_index = suffix.filter(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())).and_then(|s| s.parse().ok());
        Self { role: if role.is_empty() { "main".into() } else { role.into() }, cell_index }
    }
}

/// A selector drag: armed by the press, started by the first move that
/// leaves the press pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectorGesture {
    pub axes: AxesId,
    pub transform: AxisTransform,
    pub kind: SelectorKind,
    pub state: SelectorState,
    pub handle: DragHandle,
    pub origin: CrosshairPoint,
    pub origin_px: (f64, f64),
    /// Where the hand was last read, the press to begin with.  A coordinate
    /// it cannot read is taken from here.
    pub reading: CrosshairPoint,
    pub started: bool,
    /// A move has left the press pixel.
    pub moved: bool,
}

/// A colour-limit drag on the distribution rail.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorGesture { pub axes: AxesId, pub transform: AxisTransform, pub drag: ColorLimitDrag }

/// A middle-button pan of the view.
#[derive(Clone, Debug, PartialEq)]
pub struct PanGesture {
    pub axes: AxesId,
    pub transform: AxisTransform,
    pub origin: CrosshairPoint,
    pub x: NumericRange,
    pub y: NumericRange,
    pub candidate: Option<Viewport>,
}

/// Height-bar camera orbit angles (`HeightBarCamera`'s fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraAngles { pub azimuth_deg: f64, pub elevation_deg: f64, pub zoom: f64 }

/// A height-bar camera drag: pixels in, orbit angles out.
#[derive(Clone, Debug, PartialEq)]
pub struct OrbitGesture {
    pub axes: AxesId,
    pub transform: AxisTransform,
    pub origin_px: (f64, f64),
    pub start: CameraAngles,
    pub current: CameraAngles,
}

impl CameraAngles {
    /// `HeightBarCamera(...)`: finite numbers, the elevation held to
    /// [8, 80] degrees and the zoom to [0.4, 6].
    pub fn new(azimuth_deg: f64, elevation_deg: f64, zoom: f64) -> Result<Self> {
        for (name, value) in [("azimuth_deg", azimuth_deg), ("elevation_deg", elevation_deg), ("zoom", zoom)] {
            if !value.is_finite() { return Err(PlotError::type_error(format!("{name} must be a finite number"))); }
        }
        Ok(Self { azimuth_deg, elevation_deg: py_min(py_max(elevation_deg, 8.0), 80.0), zoom: py_min(py_max(zoom, 0.4), 6.0) })
    }
}

impl OrbitGesture {
    /// Follow the hand to `pixel` (`_on_motion`'s orbit): a drag right turns
    /// the azimuth down, a drag up tips the camera toward the horizon, 0.35
    /// degree per pixel.  The camera becomes `current`; the session commits
    /// its azimuth and elevation as display parameters.
    pub fn follow(&mut self, pixel: (f64, f64)) -> Result<CameraAngles> {
        let dx = pixel.0 - self.origin_px.0;
        let dy = pixel.1 - self.origin_px.1;
        let camera = CameraAngles::new(self.start.azimuth_deg - dx * 0.35, self.start.elevation_deg - dy * 0.35, self.start.zoom)?;
        self.current = camera;
        Ok(camera)
    }
}

/// A left press on the 3D scene: a click picks a bar, a drag is inert.
#[derive(Clone, Debug, PartialEq)]
pub struct PickGesture { pub axes: AxesId, pub transform: AxisTransform, pub origin_px: (f64, f64) }

/// The one pointer gesture a session holds.
#[derive(Clone, Debug, PartialEq)]
pub enum PointerGesture {
    Selector(SelectorGesture),
    Color(ColorGesture),
    Pan(PanGesture),
    Orbit(OrbitGesture),
    Pick(PickGesture),
}

impl PointerGesture {
    pub fn axes(&self) -> &AxesId {
        match self {
            PointerGesture::Selector(g) => &g.axes, PointerGesture::Color(g) => &g.axes, PointerGesture::Pan(g) => &g.axes,
            PointerGesture::Orbit(g) => &g.axes, PointerGesture::Pick(g) => &g.axes,
        }
    }
    pub fn transform(&self) -> &AxisTransform {
        match self {
            PointerGesture::Selector(g) => &g.transform, PointerGesture::Color(g) => &g.transform, PointerGesture::Pan(g) => &g.transform,
            PointerGesture::Orbit(g) => &g.transform, PointerGesture::Pick(g) => &g.transform,
        }
    }
}
