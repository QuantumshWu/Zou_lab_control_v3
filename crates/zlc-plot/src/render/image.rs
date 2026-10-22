//! The Image kind (`_update_image`): one scalar plane on its sample lattice
//! (a FuncScale where the coordinates are nonuniform), letterboxed in a
//! square frame at its cell aspect, coloured through a colormap between the
//! resolved colour limits; the distribution rail beside it
//! (`_update_horizontal_histogram`, image profile, with guides at the data
//! range) and the colorbar sharing the rail's axis (`_update_image_chrome`).
//!
//! Live, the plane is a prepared scene (`image:prepared`) the compose paints
//! with zlc-render's `raster_prepared_images` at the geometry
//! `_image_scene_geometry` computes; the artist keeps the scalar data, cmap
//! and clim every reader takes from it, and an export materializes it.
use std::sync::Arc;

use super::colormaps;
use super::core::{text_value, Kind, Renderer};
use super::geometry::{centers_extent, image_cell_aspect, image_coordinate_scale, square_image_limits};
use super::labels::{has_explicit_unit_suffix, quantity_label, state_label};
use super::limits::{autoscaled_limits, scalar_close};
use super::mpl::axes::{nonsingular_by, Child, ChildId, LineArtist};
use super::mpl::color;
use super::mpl::image::{colormap_rgba, ImageArtist, ImagePixels, LineCollectionArtist, MeshArtist, RectArtist};
use super::scale::AxisScale;
use crate::configuration::selectors::NumericRange;
use crate::configuration::{DisplayState, Value};
use crate::projection::histogram::{aligned_histogram_edges, histogram_counts};
use crate::projection::nd::{Array, DType};
use crate::projection::output::{ImageData, Payload};

/// The colormap's uint8 RGBA table, as `cmap((i + 0.5) / 256, bytes=True)`.
pub fn color_lut(name: &str) -> Option<&'static [u8; 1024]> { colormaps::table(name).map(|(_, bytes, _)| bytes) }

/// The same table as float32 RGB (`_image_color_lut_rgb`).
pub fn color_lut_rgb(lut: &[u8; 1024]) -> Vec<[f32; 3]> {
    lut.chunks(4).map(|rgba| [rgba[0] as f32 / 255.0f32, rgba[1] as f32 / 255.0f32, rgba[2] as f32 / 255.0f32]).collect()
}

/// One prepared image surface (`image:prepared`): the plane the kernel paints.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub key: String,
    pub state: Arc<DisplayState>,
    pub values: Array,
    /// None: every sample valid.
    pub valid: Option<Array>,
    /// (left, right, bottom, top) on the sample lattice.
    pub extent: [f64; 4],
    pub color_limits: (f64, f64),
    pub cmap: &'static str,
}

/// The colorbar a `figure.colorbar(mappable, cax=...)` built, and zlc's end fills.
#[derive(Clone, Debug)]
pub struct Colorbar {
    pub axes: usize,
    pub dividers: ChildId,
    pub solids: ChildId,
    pub ends: [ChildId; 2],
    /// `colorbar_state`: (cmap, colour limits, value label, rail limits).
    pub state: Option<(&'static str, (f64, f64), String, (f64, f64))>,
    /// The proxy mappable's norm.
    pub norm: (f64, f64),
    pub cmap: &'static str,
    /// The colorbar's `_boundaries` (its vmin and vmax are their ends).
    pub boundaries: Vec<f64>,
    /// `_zlc_tick_signature` of `declare_colorbar_ticks`.
    pub ticks_declared: bool,
}

/// The Image renderer's retained tables (`_artists` of one image surface).
#[derive(Clone, Debug, Default)]
pub struct ImageTables {
    /// `_coordinate_scales[id(axes)]`.
    pub scales: Option<(usize, AxisScale, AxisScale)>,
    /// `_artists[key]`: the image artist.
    pub image: Option<(usize, ChildId)>,
    /// `{key}:mapping_state`.
    pub mapping: Option<(&'static str, (f64, f64))>,
    /// `_image_ranges[key]`: the plane and validity it was measured on, and the range.
    pub range: Option<(Array, Option<Array>, u64, Option<(f64, f64)>)>,
    /// `{key}:distribution_data_limits`.
    pub distribution_limits: Option<(f64, f64)>,
    /// `{key}:distribution_cache`: (data revision, bins, histogram limits) and the bars.
    pub distribution_cache: Option<((u64, usize, (u64, u64)), Vec<i64>, Vec<f64>)>,
    /// `{key}:distribution` and its count ceiling.
    pub distribution: (Option<(usize, ChildId)>, f64),
    /// `{key}:guides`.
    pub guides: Option<(usize, [ChildId; 2])>,
    pub colorbar: Option<Colorbar>,
    /// `image:prepared`.
    pub prepared: Option<PreparedImage>,
    /// `_color_limit_candidate`.
    pub candidate: Option<NumericRange>,
    /// A colour-limit gesture is held (`_selector_gesture_kind is COLOR_LIMITS`).
    pub color_gesture: bool,
    /// `{key}:aspect_box`: the aspect, limits and original box `apply_aspect` last resolved.
    pub aspect_box: Option<[u64; 9]>,
    /// The point overlay layer.
    pub points: super::points::PointTables,
}

pub struct ImageKind;

impl Kind for ImageKind {
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, axes: usize, key: &str) {
        renderer.update_image(axes, payload, state, key);
    }
    fn key(&self) -> &'static str { "image" }
}

fn bool_value(state: &DisplayState, name: &str) -> bool { matches!(state.get(name), Some(Value::Bool(true))) }

/// A colormap name the renderer draws (`str(state["colormap"])`).
pub fn colormap_name(state: &DisplayState) -> &'static str {
    let name = text_value(state.get("colormap")).unwrap_or_default();
    colormaps::NAMES.iter().copied().find(|known| *known == name).unwrap_or_else(|| panic!("colormap {name:?}"))
}

/// Every sample valid (`_all_true`).
pub(crate) fn all_valid(valid: &Array) -> bool {
    match valid.constant_value() {
        Some(value) => value.as_f64().is_some_and(|v| v != 0.0),
        None => valid.bytes().iter().all(|&b| b != 0),
    }
}

/// The extrema of the kept elements, compared exactly in the plane's own type.
fn integer_extrema<T: Copy + Ord + Into<i128>>(values: &[T], mask: Option<&[u8]>) -> Option<(i128, i128)> {
    let (low, high) = match mask {
        None => (values.iter().copied().min()?, values.iter().copied().max()?),
        Some(mask) => {
            let mut kept = values.iter().zip(mask).filter(|(_, m)| **m != 0).map(|(v, _)| *v);
            let first = kept.next()?;
            kept.fold((first, first), |(low, high), v| (low.min(v), high.max(v)))
        }
    };
    Some((low.into(), high.into()))
}

/// `finite_extrema`: count, and the first of each extreme in index order.
fn float_extrema<T: Copy + crate::projection::nd::Elem>(values: &[T], mask: Option<&[u8]>) -> Option<(f64, f64)> {
    let (mut low, mut high, mut count) = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for (index, value) in values.iter().enumerate() {
        let value = value.f64();
        if mask.is_some_and(|m| m[index] == 0) || !value.is_finite() { continue; }
        count += 1;
        if value < low { low = value; }
        if value > high { high = value; }
    }
    (count > 0).then_some((low, high))
}

/// `_image_data_range(values, valid)`: the exact finite range, or None.
pub fn image_data_range(values: &Array, valid: Option<&Array>) -> Option<(f64, f64)> {
    let mask: Option<&[u8]> = valid.map(|v| v.bytes());
    let integral = match values.dtype() {
        DType::Bool => integer_extrema(values.slice::<bool>(), mask),
        DType::U8 => integer_extrema(values.slice::<u8>(), mask),
        DType::I8 => integer_extrema(values.slice::<i8>(), mask),
        DType::U16 => integer_extrema(values.slice::<u16>(), mask),
        DType::I16 => integer_extrema(values.slice::<i16>(), mask),
        DType::U32 => integer_extrema(values.slice::<u32>(), mask),
        DType::I32 => integer_extrema(values.slice::<i32>(), mask),
        DType::U64 => integer_extrema(values.slice::<u64>(), mask),
        DType::I64 => integer_extrema(values.slice::<i64>(), mask),
        DType::F16 => return float_extrema(values.slice::<zlc_core::F16>(), mask),
        DType::F32 => return float_extrema(values.slice::<f32>(), mask),
        DType::F64 => return float_extrema(values.slice::<f64>(), mask),
        DType::Object => panic!("image values must be a real numeric two-dimensional array"),
    };
    let (low, high) = integral?;
    let (float_low, float_high) = (low as f64, high as f64);
    if low != high && float_low == float_high { panic!("integer image range is not distinguishable in float display space"); }
    Some((float_low, float_high))
}

/// `_bounded_image_distribution_values(values, valid, sample_target)`.
pub fn bounded_distribution_values(values: &Array, valid: Option<&Array>, sample_target: usize) -> Vec<f64> {
    fn sampled<T: Copy + crate::projection::nd::Elem>(values: &[T], mask: Option<&[u8]>, float: bool, target: usize) -> Vec<f64> {
        let pick = |step: usize| -> Vec<f64> {
            values.iter().enumerate().step_by(step).filter_map(|(i, v)| {
                let v = v.f64();
                (mask.is_none_or(|m| m[i] != 0) && (!float || v.is_finite())).then_some(v)
            }).collect()
        };
        if values.len() > target {
            let kept = pick(values.len() / target + 1);
            if !kept.is_empty() || (mask.is_none() && !float) { return kept; }
        }
        pick(1)
    }
    let (float, mask) = (values.kind() == 'f', valid.map(|v| v.bytes()));
    match values.dtype() {
        DType::Bool => sampled(values.bytes(), mask, false, sample_target),
        DType::U8 => sampled(values.slice::<u8>(), mask, false, sample_target),
        DType::I8 => sampled(values.slice::<i8>(), mask, false, sample_target),
        DType::U16 => sampled(values.slice::<u16>(), mask, false, sample_target),
        DType::I16 => sampled(values.slice::<i16>(), mask, false, sample_target),
        DType::U32 => sampled(values.slice::<u32>(), mask, false, sample_target),
        DType::I32 => sampled(values.slice::<i32>(), mask, false, sample_target),
        DType::U64 => sampled(values.slice::<u64>(), mask, false, sample_target),
        DType::I64 => sampled(values.slice::<i64>(), mask, false, sample_target),
        DType::F16 => sampled(values.slice::<zlc_core::F16>(), mask, float, sample_target),
        DType::F32 => sampled(values.slice::<f32>(), mask, float, sample_target),
        DType::F64 => sampled(values.slice::<f64>(), mask, float, sample_target),
        DType::Object => panic!("image values must be a real numeric two-dimensional array"),
    }
}

/// A colorbar's boundaries over a Normalize(vmin, vmax): `norm.inverse(linspace(0, 1, N + 1))`.
fn colorbar_boundaries(vmin: f64, vmax: f64) -> Vec<f64> {
    crate::ticks::linspace(0.0, 1.0, 257).iter().map(|&y| vmin + y * (vmax - vmin)).collect()
}

/// `_mesh()`'s boundary coordinates for boundaries `b` over the norm (vmin, vmax).
fn colorbar_mesh(boundaries: &[f64], norm: (f64, f64)) -> Vec<f64> {
    let (vmin, vmax) = norm;
    let y: Vec<f64> = boundaries.iter().map(|&b| (b - vmin) / (vmax - vmin)).collect();
    let (first, last) = (y[0], y[y.len() - 1]);
    let y: Vec<f64> = y.iter().map(|&v| (v - first) / (last - first)).collect();
    let (low, high) = (boundaries[0], boundaries[boundaries.len() - 1]);
    y.iter().map(|&v| low + v * (high - low)).collect()
}

impl Renderer {
    /// `_update_image(axes, payload, state, key)`.
    pub fn update_image(&mut self, axes: usize, payload: &Arc<Payload>, state: &DisplayState, key: &str) {
        let Payload::Image(data) = &**payload else { panic!("an Image surface needs an image payload") };
        self.update_image_data(axes, data, state, key, None, true);
    }

    /// `_update_image(axes, payload, state, key, color_limits=..., paint_labels=...)`:
    /// a grid's cell is told the grid's colour scale.
    pub(crate) fn update_image_data(&mut self, axes: usize, data: &ImageData, state: &DisplayState, key: &str,
                                    color_limits: Option<(f64, f64)>, paint_labels: bool) {
        let x = data.x.display.to_f64();
        let y = data.y.display.to_f64();
        let scales = (image_coordinate_scale(&x).unwrap_or_else(|e| panic!("{e}")),
                      image_coordinate_scale(&y).unwrap_or_else(|e| panic!("{e}")));
        let previous = self.image.scales.as_ref().filter(|s| s.0 == axes).map(|s| (s.1.clone(), s.2.clone()));
        // The scene draws on the sample lattice itself: its axes stay linear.
        let height_bars = self.height_bars_active(key, state);
        for (x_axis, scale, before) in [(true, &scales.0, previous.as_ref().map(|p| &p.0)), (false, &scales.1, previous.as_ref().map(|p| &p.1))] {
            let wanted = match scale { AxisScale::Samples(lattice) if !height_bars => Some(lattice.clone()), _ => None };
            let actual = self.fig.axes[axes].axis(x_axis).function.clone();
            if before != Some(scale) || actual.is_some() != wanted.is_some() {
                if wanted.is_some() || actual.is_some() { self.set_axis_function(axes, x_axis, wanted); }
                self.chrome_dirty.insert(axes);
            }
        }
        self.image.scales = Some((axes, scales.0.clone(), scales.1.clone()));
        let (x_label, y_label, value_label) = self.image_labels(data, state);
        let aspect = image_cell_aspect(&x, &y, (&scales.0, &scales.1));
        self.mutate_image_artists(axes, data, state, key, (&scales.0, &scales.1), &x_label, &y_label, &value_label, aspect,
                                  color_limits, paint_labels);
    }

    /// The x, y and value (colorbar) labels of one image (`_effective_labels`).
    pub(crate) fn image_labels(&self, data: &ImageData, state: &DisplayState) -> (String, String, String) {
        // The semantic spec's labels: a grid's cell labels its axes.
        let labels = self.spec.semantic().labels();
        let nonempty = |v: &Option<Arc<str>>| v.as_deref().filter(|s| !s.is_empty()).map(str::to_owned);
        let explicit = |name: &str, fallback: Option<String>| state_label(text_value(state.get(name)).as_deref(), fallback.as_deref());
        let explicit_x = explicit("x_label", nonempty(&labels.x));
        let explicit_y = explicit("y_label", nonempty(&labels.y));
        let explicit_value = explicit("value_label", nonempty(&labels.value));
        let label = |q: &crate::projection::output::QuantityArray, fallback: &str, explicit: Option<&str>| {
            quantity_label(Some(&q.label), Some(&q.display_unit.symbol), fallback, explicit)
        };
        let value_label = match &explicit_value {
            Some(text) if !text.is_empty() && has_explicit_unit_suffix(text) => text.clone(),
            _ => label(&data.z, "value", explicit_value.as_deref()),
        };
        (label(&data.x, "x", explicit_x.as_deref()), label(&data.y, "y", explicit_y.as_deref()), value_label)
    }

    /// `set_xscale("function", ...)` / `set_xscale("linear")`: matplotlib's
    /// default locator and formatter come back with the new scale.
    fn set_axis_function(&mut self, axes: usize, x: bool, lattice: Option<Arc<[f64]>>) {
        let axis = self.fig.axes[axes].axis_mut(x);
        axis.function = lattice;
        axis.scale_serial += 1;
        axis.policy = super::mpl::axes::TickPolicy::Default;
        axis.offset_pin = None;
    }

    #[allow(clippy::too_many_arguments)]
    fn mutate_image_artists(&mut self, axes: usize, data: &ImageData, state: &DisplayState, key: &str,
                            scales: (&AxisScale, &AxisScale), x_label: &str, y_label: &str, value_label: &str,
                            coordinate_aspect: Option<f64>, color_limits: Option<(f64, f64)>, paint_labels: bool) {
        let values = data.z.display.clone();
        if values.ndim() != 2 || !"biuf".contains(values.kind()) { panic!("image values must be a real numeric two-dimensional array"); }
        let (rows, columns) = (values.shape()[0], values.shape()[1]);
        let x = data.x.display.to_f64();
        let y = data.y.display.to_f64();
        if (rows, columns) != (y.len(), x.len()) { panic!("image values must match the y/x coordinate grid"); }
        let valid = (!all_valid(&data.valid)).then(|| data.valid.clone());
        let (x0, x1, y0, y1) = centers_extent(&x, &y, scales);
        let extent = [x0, x1, y1, y0];
        let data_range = { let _span = super::profile::span("present.image.range"); self.cached_image_range(&values, valid.as_ref()) };
        // Bars stand on the colour scale: it is the z axis of the scene.
        let height_bars = self.height_bars_active(key, state);
        let (mut vmin, mut vmax) = match color_limits {
            Some(limits) => limits,
            None => self.resolve_limits(key, data_range, state, "color", false, true, height_bars),
        };
        if let Some(candidate) = self.image.candidate { (vmin, vmax) = (candidate.low, candidate.high); }
        let cmap = colormap_name(state);
        // A grid's overview paints its cells as one scene of its own; only
        // the scene THIS surface installed goes.
        if !height_bars && !self.facet_overview() {
            self.image.prepared = Some(PreparedImage {
                key: key.to_owned(), state: Arc::new(state.clone()), values: values.clone(), valid: valid.clone(), extent,
                color_limits: (vmin, vmax), cmap,
            });
        } else if self.image.prepared.as_ref().is_some_and(|p| p.key == key) {
            self.image.prepared = None;
        }
        {
            let _span = super::profile::span("present.image.artist");
            self.update_image_artist(axes, &values, valid.as_ref(), extent, state, key, (vmin, vmax), cmap, coordinate_aspect, scales);
        }
        if paint_labels { self.set_axis_labels(axes, x_label, y_label); }
        let _span = super::profile::span("present.image.chrome");
        self.update_image_chrome(axes, &values, valid.as_ref(), state, data_range, (vmin, vmax), cmap, value_label);
    }

    /// `_materialize_prepared_images` for an export: the artist's plane takes
    /// the mask a draw needs (invalid samples, non-finite values).
    pub(crate) fn materialize_prepared_image(&mut self) {
        let Some(prepared) = self.image.prepared.take() else { return };
        self.mask_image_artist(&prepared.values, prepared.valid.as_ref());
    }

    /// The image artist's plane takes the mask a draw needs: invalid samples, non-finite values.
    pub(crate) fn mask_image_artist(&mut self, values: &Array, valid: Option<&Array>) {
        let Some((axes, id)) = self.image.image else { return };
        let mut mask: Option<Vec<bool>> = valid.map(|v| v.bytes().iter().map(|&b| b == 0).collect());
        if values.kind() == 'f' {
            let finite = values.finite_mask();
            if finite.iter().any(|f| !f) {
                let held = mask.unwrap_or_else(|| vec![false; finite.len()]);
                mask = Some(held.iter().zip(&finite).map(|(m, f)| *m || !f).collect());
            }
        }
        if let Some(Child::Image(image)) = self.fig.axes[axes].child_mut(id) {
            if let ImagePixels::Scalar { mask: slot, .. } = &mut image.pixels { *slot = mask.map(Arc::new); }
        }
    }

    /// `_cached_image_range`: measured once per (plane, validity, revision).
    pub(crate) fn cached_image_range(&mut self, values: &Array, valid: Option<&Array>) -> Option<(f64, f64)> {
        let revision = self.data_revision.unwrap_or(0);
        if let Some((held, held_valid, held_revision, range)) = &self.image.range {
            let same_valid = match (held_valid, valid) { (None, None) => true, (Some(a), Some(b)) => a.same_storage(b), _ => false };
            if held.same_storage(values) && same_valid && *held_revision == revision { return *range; }
        }
        let range = image_data_range(values, valid);
        self.image.range = Some((values.clone(), valid.cloned(), revision, range));
        range
    }

    /// `_update_image_artist` (the heatmap; `materialize=False`).
    #[allow(clippy::too_many_arguments)]
    fn update_image_artist(&mut self, axes: usize, values: &Array, valid: Option<&Array>, extent: [f64; 4],
                           state: &DisplayState, key: &str, color_limits: (f64, f64), cmap: &'static str,
                           coordinate_aspect: Option<f64>, scales: (&AxisScale, &AxisScale)) {
        let policy = self.style.render.clone();
        let previous_mapping = self.image.mapping;
        if self.height_bars_active(key, state) {
            self.image.mapping = Some((cmap, color_limits));
            self.update_height_bars_artist(axes, values, valid, extent, state, Some(color_limits), cmap);
            return;
        }
        if self.height_bars.call.take().is_some() {
            // Back from the scene: its chrome hides and the 2D chrome returns, once.
            self.hide_height_bars_chrome();
            // An overview cell draws no chrome of its own: the grid's group paints it.
            if !self.facet_overview() { self.fig.axes[axes].axison = true; }
            self.chrome_dirty.insert(axes);
        }
        let coordinate_aspect = coordinate_aspect.unwrap_or_else(|| panic!("image cell geometry has no finite coordinate aspect"));
        let home = square_image_limits((extent[0], extent[1], extent[2], extent[3]), coordinate_aspect).unwrap_or_else(|e| panic!("{e}"));
        let home = ((scales.0.value(home.0), scales.0.value(home.1)), (scales.1.value(home.2), scales.1.value(home.3)));
        self.home_limits.insert(axes, home);
        let requested = if self.painted_surfaces().iter().any(|(_, a, _)| *a == axes) { self.requested_view } else { None };
        let (x_limits, y_limits) = self.resolved_view_limits(axes, requested);
        self.set_xlim(axes, x_limits);
        self.set_ylim(axes, y_limits);
        let model = &mut self.fig.axes[axes];
        model.anchor = policy.image_anchor;
        model.aspect = Some(coordinate_aspect);
        // The drawable box follows the aspect: Matplotlib's `apply_aspect` resolves it now.
        let (l, b, w, h) = zlc_figure::transform::bounds(&model.position);
        let signature = [coordinate_aspect, x_limits.0, x_limits.1, y_limits.0, y_limits.1, l, b, w, h].map(f64::to_bits);
        if self.image.aspect_box != Some(signature) {
            model.apply_aspect(self.fig.size_inches, self.fig.dpi);
            self.image.aspect_box = Some(signature);
        }
        let lattice = scales.0 != &AxisScale::Linear || scales.1 != &AxisScale::Linear;
        let artist = ImageArtist {
            visible: bool_value(state, "show_image"), pixels: ImagePixels::Scalar { values: values.clone(), mask: None }, extent,
            origin: policy.image_origin, cmap, clim: Some(color_limits), lattice,
        };
        match self.image.image {
            None => {
                let id = self.fig.axes[axes].add(Child::Image(Box::new(artist)));
                self.image.image = Some((axes, id));
            }
            Some((owner, id)) => {
                if let Some(Child::Image(image)) = self.fig.axes[owner].child_mut(id) {
                    let clim = image.clim;
                    **image = artist;
                    // `set_clim` only on a new mapping or one the artist no longer paints.
                    if previous_mapping.is_some_and(|p| p.1 == color_limits)
                        && clim.is_some_and(|c| scalar_close(c.0, color_limits.0) && scalar_close(c.1, color_limits.1)) {
                        image.clim = clim;
                    }
                }
            }
        }
        self.image.mapping = Some((cmap, color_limits));
        self.set_xlim(axes, x_limits);
        self.set_ylim(axes, y_limits);
    }

    /// `_update_image_chrome`: distribution, colorbar and spatial ticks of one image axes.
    #[allow(clippy::too_many_arguments)]
    fn update_image_chrome(&mut self, axes: usize, values: &Array, valid: Option<&Array>,
                           state: &DisplayState, data_range: Option<(f64, f64)>, color_limits: (f64, f64), cmap: &'static str,
                           value_label: &str) {
        let policy = self.style.render.clone();
        let (vmin, vmax) = color_limits;
        let (histogram_limits, rail_limits) = self.resolve_distribution_limits(data_range, color_limits);
        if let Some(&distribution) = self.axes_of("distribution").first() {
            let bins = policy.image_distribution_min_bins.max((values.len().max(1) / policy.distribution_bin_divisor).min(policy.distribution_max_bins));
            let cache_key = (self.data_revision.unwrap_or(0), bins, (histogram_limits.0.to_bits(), histogram_limits.1.to_bits()));
            let (counts, edges) = match &self.image.distribution_cache {
                Some((held, counts, edges)) if *held == cache_key => (counts.clone(), edges.clone()),
                _ => {
                    let _span = super::profile::span("present.image.histogram");
                    let samples = Array::vector(bounded_distribution_values(values, valid, policy.image_distribution_sample_target));
                    let edges = aligned_histogram_edges(&samples, bins, Some(histogram_limits));
                    let counts = histogram_counts(&samples, &edges, None);
                    self.image.distribution_cache = Some((cache_key, counts.clone(), edges.clone()));
                    (counts, edges)
                }
            };
            let slot = self.image.distribution;
            if slot.0.is_none() {
                // `add_collection(bars)`: their extent is the rail's dataLim.
                let peak = counts.iter().copied().max().map_or(0.0, |c| c as f64);
                self.fig.axes[distribution].update_datalim(Some((0.0, peak)), Some((edges[0], edges[edges.len() - 1])));
            }
            self.image.distribution = self.update_horizontal_histogram(distribution, slot, &edges, &counts, rail_limits, true);
            if self.image.guides.is_none() {
                let guide = |value: f64| Child::AxLine(LineArtist {
                    visible: true, zorder: 2.0, alpha: Some(policy.distribution_guide_alpha), color: self.style.palette.guide.to_owned(),
                    linewidth: self.style.artists.side_distribution_linewidth, linestyle: "-".into(), marker: "None".into(),
                    markersize: self.rc.lines_markersize, markeredgewidth: self.rc.lines_markeredgewidth, markevery: None,
                    marker_colors: None, label: String::new(), x: vec![0.0, 1.0], y: vec![value, value],
                });
                // `axhline(value)`: its y joins dataLim, and a value outside
                // the y bound asks an autoscale (the rail's x limits are set).
                let made = [(guide(vmin), vmin), (guide(vmax), vmax)];
                let mut ids = [0; 2];
                for (slot, (artist, value)) in ids.iter_mut().zip(made) {
                    let model = &mut self.fig.axes[distribution];
                    let (low, high) = if model.ylim.0 < model.ylim.1 { model.ylim } else { (model.ylim.1, model.ylim.0) };
                    *slot = model.add(artist);
                    model.update_datalim(None, Some((value, value)));
                    if value < low || value > high { self.plotted(distribution); }
                }
                self.image.guides = Some((distribution, ids));
            }
            let (owner, guides) = self.image.guides.unwrap();
            for (index, id) in guides.into_iter().enumerate() {
                if let Some(Child::AxLine(line)) = self.fig.axes[owner].child_mut(id) {
                    line.visible = data_range.is_some();
                    if let Some(range) = data_range {
                        let value = if index == 0 { range.0 } else { range.1 };
                        line.y = vec![value, value];
                    }
                }
            }
        }
        if let Some(&colorbar_axes) = self.axes_of("colorbar").first() {
            let _span = super::profile::span("present.image.colorbar");
            self.update_colorbar(colorbar_axes, cmap, color_limits, value_label, rail_limits);

        }
        self.apply_colorbar_visibility(state);
        // A grid's overview owns its cells' ticks.
        if !self.facet_overview() {
            let tick_pt = self.style.fonts.tick_pt;
            super::curve::apply_smart_ticks(&mut self.fig.axes[axes], true, tick_pt);
            super::curve::apply_smart_ticks(&mut self.fig.axes[axes], false, tick_pt);
        }
    }

    /// `_resolve_distribution_limits`: (histogram limits, rail limits).
    fn resolve_distribution_limits(&mut self, data_range: Option<(f64, f64)>, color_limits: (f64, f64)) -> ((f64, f64), (f64, f64)) {
        let current = self.image.distribution_limits;
        let selected = match data_range {
            None => current.unwrap_or((0.0, 1.0)),
            Some(range) => {
                let policy = &self.style.render;
                autoscaled_limits(range, current, policy.image_color_padding_fraction, policy.image_color_deadband_fraction, false, true)
            }
        };
        self.image.distribution_limits = Some(selected);
        let rail = (if color_limits.0 < selected.0 { color_limits.0 } else { selected.0 },
                    if color_limits.1 > selected.1 { color_limits.1 } else { selected.1 });
        (selected, rail)
    }

    /// `_apply_colorbar_visibility(state)`.
    pub(crate) fn apply_colorbar_visibility(&mut self, state: &DisplayState) {
        let Some(value) = state.get("show_colorbar") else { return };
        let visible = matches!(value, Value::Bool(true));
        for axes in self.axes_of("colorbar").to_vec() {
            if self.fig.axes[axes].visible != visible {
                self.fig.axes[axes].visible = visible;
                self.chrome_dirty.insert(axes);
            }
        }
    }

    /// The colorbar half of `_update_image_chrome`.
    fn update_colorbar(&mut self, axes: usize, cmap: &'static str, color_limits: (f64, f64), value_label: &str, rail: (f64, f64)) {
        let (vmin, vmax) = color_limits;
        if self.image.colorbar.is_none() { self.create_colorbar(axes, cmap, color_limits); }
        let mut wanted = (cmap, color_limits, value_label.to_owned(), rail);
        let previous = self.image.colorbar.as_ref().unwrap().state.clone();
        if let (Some(_), Some(previous)) = (self.image.candidate, &previous) { wanted = previous.clone(); }
        if Some(&wanted) == previous.as_ref() { return; }
        let first = previous.is_none();
        if first || previous.as_ref().is_some_and(|p| p.0 != cmap) {
            self.colorbar_set_cmap(cmap);
        }
        if first || previous.as_ref().is_some_and(|p| p.1 != color_limits) {
            if first {
                self.colorbar_set_clim(color_limits);
            } else {
                // The ramp is invariant under a clim move: only its coordinates and values follow.
                let colorbar = self.image.colorbar.as_mut().unwrap();
                colorbar.norm = color_limits;
                let solids = colorbar.solids;
                if let Some(Child::Mesh(mesh)) = self.fig.axes[axes].child_mut(solids) {
                    let ramp = crate::ticks::linspace(vmin, vmax, mesh.coordinates.len());
                    for (row, &value) in mesh.coordinates.iter_mut().zip(&ramp) { row[0][1] = value; row[1][1] = value; }
                    mesh.values = ramp.windows(2).map(|w| 0.5 * (w[1] + w[0])).collect();
                    mesh.flat = true;
                    mesh.norm = color_limits;
                }
            }
        }
        if first || previous.as_ref().is_some_and(|p| p.2 != value_label) {
            self.fig.axes[axes].yaxis.label = value_label.to_owned();
        }
        if first || previous.as_ref().is_some_and(|p| p.1 != color_limits) {
            self.declare_colorbar_ticks(axes, color_limits);
        }
        let (rail_low, rail_high) = wanted.3;
        if self.fig.axes[axes].ylim != (rail_low, rail_high) { self.fig.axes[axes].set_ylim(rail_low, rail_high); }
        let colorbar = self.image.colorbar.as_mut().unwrap();
        let [below, above] = colorbar.ends;
        let model = &mut self.fig.axes[axes];
        if let Some(Child::Rect(rect)) = model.child_mut(below) {
            rect.xy.1 = rail_low;
            rect.height = 0.0f64.max(vmin - rail_low);
            rect.facecolor = Some(colormap_rgba(cmap, 0.0));
        }
        if let Some(Child::Rect(rect)) = model.child_mut(above) {
            rect.xy.1 = vmax;
            rect.height = 0.0f64.max(rail_high - vmax);
            rect.facecolor = Some(colormap_rgba(cmap, 1.0));
        }
        colorbar.state = Some(wanted);
    }

    /// `figure.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=axes)` and the two end fills.
    fn create_colorbar(&mut self, axes: usize, cmap: &'static str, limits: (f64, f64)) {
        let rc = self.rc.clone();
        let edge = color::rgba(&rc.axes_edgecolor);
        let model = &mut self.fig.axes[axes];
        model.colorbar_locator = true;
        model.outline = Some(vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0]]);
        let dividers = model.add(Child::LineCollection(LineCollectionArtist { visible: true, zorder: 2.0, color: edge, linewidth: 0.5 * rc.axes_linewidth }));
        model.yaxis.label = String::new();
        // `_draw_all`: the long axis' label and ticks on the right, no ticks on the short axis.
        model.yaxis.label_position = "right";
        for params in [&mut model.yaxis.major, &mut model.yaxis.minor] {
            params.tick1on = false;
            params.tick2on = true;
            params.label1on = false;
            params.label2on = true;
        }
        model.xaxis.policy = super::mpl::axes::TickPolicy::Named { ticks: Vec::new(), texts: Vec::new() };
        let norm = nonsingular_by(limits.0, limits.1, 0.1);
        let boundaries = colorbar_boundaries(norm.0, norm.1);
        model.set_xlim(0.0, 1.0);
        model.set_ylim(boundaries[0], boundaries[boundaries.len() - 1]);
        let solids = model.add(Child::Mesh(Box::new(colorbar_solids(&boundaries, norm, cmap, None))));
        let end = |model: &mut super::mpl::axes::AxesModel| model.add(Child::Rect(RectArtist {
            visible: true, zorder: 1.0 - 0.1, xy: (0.0, 0.0), width: 1.0, height: 0.0, facecolor: None,
        }));
        let ends = [end(model), end(model)];
        self.image.colorbar = Some(Colorbar { axes, dividers, solids, ends, state: None, norm, cmap, boundaries, ticks_declared: false });
    }

    /// `update_normal` after a cmap or norm change: `_draw_all` rebuilds the ramp.
    fn colorbar_draw_all(&mut self) {
        let colorbar = self.image.colorbar.as_mut().unwrap();
        colorbar.norm = nonsingular_by(colorbar.norm.0, colorbar.norm.1, 0.1);
        colorbar.boundaries = colorbar_boundaries(colorbar.norm.0, colorbar.norm.1);
        let (axes, old) = (colorbar.axes, colorbar.solids);
        let mesh = colorbar_solids(&colorbar.boundaries, colorbar.norm, colorbar.cmap, Some(1.0));
        let (low, high) = (colorbar.boundaries[0], colorbar.boundaries[colorbar.boundaries.len() - 1]);
        let model = &mut self.fig.axes[axes];
        model.set_ylim(low, high);
        model.remove(old);
        let solids = model.add(Child::Mesh(Box::new(mesh)));
        self.image.colorbar.as_mut().unwrap().solids = solids;
    }

    /// `mappable.set_cmap(cmap)`.
    fn colorbar_set_cmap(&mut self, cmap: &'static str) {
        self.image.colorbar.as_mut().unwrap().cmap = cmap;
        self.colorbar_draw_all();
    }

    /// `mappable.set_clim(vmin, vmax)`: a change notifies the colorbar.
    fn colorbar_set_clim(&mut self, limits: (f64, f64)) {
        let colorbar = self.image.colorbar.as_mut().unwrap();
        if colorbar.norm == limits { return; }
        colorbar.norm = limits;
        self.colorbar_draw_all();
    }

    /// `declare_colorbar_ticks(colorbar, label_pt=tick_pt, label_chars=..., span=limits)`.
    fn declare_colorbar_ticks(&mut self, axes: usize, span: (f64, f64)) {
        use super::mpl::axes::{DeclaredState, DeclaredText, TickPolicy};
        let size_pt = self.style.fonts.tick_pt;
        let shortest = self.style.render.colorbar_limit_label_chars.max(1);
        let lengths: Vec<Option<usize>> = (shortest..shortest + 4).map(Some).collect();
        let colorbar = self.image.colorbar.as_mut().unwrap();
        let model = &mut self.fig.axes[axes];
        if !colorbar.ticks_declared {
            colorbar.ticks_declared = true;
            model.yaxis.policy = TickPolicy::Declared { locator: Box::new(DeclaredState::new(2, DeclaredText::Compact, lengths, false, size_pt, Some(span))) };
            // `colorbar.ax.tick_params(labelsize=...)`: both axes, major ticks.
            model.xaxis.major.set_labelsize(size_pt);
            model.yaxis.major.set_labelsize(size_pt);
        }
        if let TickPolicy::Declared { locator } = &mut model.yaxis.policy {
            // The span is part of what the memoized ticks answered.
            if locator.span != Some(span) { locator.memo = None; }
            locator.span = Some(span);
        }
    }
}

/// The ramp `_add_solids` draws: `pcolormesh(X, Y, values[:, None])` over the boundaries.
fn colorbar_solids(boundaries: &[f64], norm: (f64, f64), cmap: &'static str, alpha: Option<f64>) -> MeshArtist {
    let y = colorbar_mesh(boundaries, norm);
    MeshArtist {
        visible: true, zorder: 1.0, alpha,
        coordinates: y.iter().map(|&v| [[0.0, v], [1.0, v]]).collect(),
        values: boundaries.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect(),
        flat: false, cmap, norm,
    }
}

/// `Bbox.intersection(a, b)`.
fn intersection(a: &[f64; 4], b: &[f64; 4]) -> Option<[f64; 4]> {
    let low = |p: &[f64; 4], i: usize| if p[i] < p[i + 2] { p[i] } else { p[i + 2] };
    let high = |p: &[f64; 4], i: usize| if p[i] > p[i + 2] { p[i] } else { p[i + 2] };
    let (x0, x1) = (low(a, 0).max(low(b, 0)), high(a, 0).min(high(b, 0)));
    let (y0, y1) = (low(a, 1).max(low(b, 1)), high(a, 1).min(high(b, 1)));
    (x0 <= x1 && y0 <= y1).then_some([x0, y0, x1, y1])
}

/// `_image_scene_geometry`: where `imshow` puts one picture -- the transform
/// built as `_make_image` builds it, inverted as Agg inverts it -- as the
/// kernel's (blit, clip, inverse affine); None for a picture out of view.
pub fn image_scene_geometry(bbox: [f64; 4], transform: &zlc_figure::transform::M3, extent: [f64; 4], rows: usize,
                            columns: usize, height: usize, upper: bool) -> Option<([i64; 4], [i64; 4], [f64; 6])> {
    use zlc_figure::transform::{transformed_points, M3};
    let [x1, x2, y1, y2] = extent;
    let in_bbox = [x1, y1, x2, y2];
    let out_bbox = transformed_points(transform, &in_bbox);
    let clipped = intersection(&out_bbox, &bbox)?;
    let (width_base, height_base) = (clipped[2] - clipped[0], clipped[3] - clipped[1]);
    if width_base == 0.0 || height_base == 0.0 { return None; }
    let placed = transform.dot(&M3::IDENTITY.scale((x2 - x1) / columns as f64, (y2 - y1) / rows as f64).translate(x1, y1));
    let t0 = if upper { placed.dot(&M3::IDENTITY.translate(0.0, -(rows as f64)).scale(1.0, -1.0)) } else { placed };
    let mut t = M3::IDENTITY.translate(-clipped[0], -clipped[1]).scale(1.0, 1.0).dot(&t0);
    let (out_width, out_height) = if width_base % 1.0 != 0.0 || height_base % 1.0 != 0.0 {
        let (w, h) = (width_base.ceil(), height_base.ceil());
        t = M3::IDENTITY.scale(1.0 + (w - width_base) / width_base, 1.0 + (h - height_base) / height_base).dot(&t);
        (w as i64, h as i64)
    } else { (width_base as i64, height_base as i64) };
    // Agg's trans_affine(sx, shy, shx, sy, tx, ty) and its invert(), in its order.
    let m = &t.0;
    let (sx, shy, shx, sy, tx, ty) = (m[0][0], m[1][0], m[0][1], m[1][1], m[0][2], m[1][2]);
    let determinant = sx * sy - shy * shx;
    if determinant == 0.0 || !determinant.is_finite() { return None; }
    let d = 1.0 / determinant;
    let t0_ = sy * d;
    let (sy, shy, shx) = (sx * d, -shy * d, -shx * d);
    let t4 = -tx * t0_ - ty * shx;
    let ty = -tx * shy - ty * sy;
    let (sx, tx) = (t0_, t4);
    let h = height as f64;
    let bottom = (h - clipped[1] + 0.5).floor() as i64;
    let blit = [(clipped[0] + 0.5).floor() as i64, bottom - out_height, out_width, out_height];
    let clip = [(bbox[0] + 0.5).floor() as i64, (h - bbox[3] + 0.5).floor() as i64, (bbox[2] + 0.5).floor() as i64 + 1,
                (h - bbox[1] - 0.5).ceil() as i64];
    Some((blit, clip, [sx, shy, shx, sy, tx, ty]))
}

impl Renderer {
    /// `_raster_prepared_images`: paint the prepared plane into the canvas;
    /// false when there is no prepared scene to paint.
    pub(crate) fn raster_prepared_image(&self, pixels: &mut [u8], width: usize, height: usize, figure: &zlc_figure::ir::Figure) -> bool {
        use zlc_render::composition::{raster_prepared_images, ImagePlan, ImageValues};
        let (Some(prepared), Some((axes, id))) = (&self.image.prepared, self.image.image) else { return false };
        if !bool_value(&prepared.state, "show_image") { return true; }
        let (low, high) = prepared.color_limits;
        let span = high - low;
        if !span.is_finite() || span <= 0.0 { return false; }
        let model = &self.fig.axes[axes];
        let Some(index) = model.children.iter().position(|(child, _)| *child == id) else { return false };
        let zlc_figure::ir::Artist::Image(image) = &figure.axes[axes].children[index] else { return false };
        let mut evaluator = zlc_figure::transform::Evaluator::new(figure);
        let Ok(transform) = evaluator.matrix(image.artist.transform.expect("an image transform")) else { return false };
        let values = &prepared.values;
        let (rows, columns) = (values.shape()[0], values.shape()[1]);
        let bbox = model.bbox_exact(self.fig.size_inches, self.fig.dpi);
        let upper = self.style.render.image_origin == "upper";
        let Some((blit, clip, affine)) = image_scene_geometry(bbox, &transform, prepared.extent, rows, columns, height, upper) else { return true };
        let plane = match values.dtype() {
            DType::Bool => ImageValues::Bool(values.bytes()),
            DType::U8 => ImageValues::U8(values.slice()),
            DType::U16 => ImageValues::U16(values.slice()),
            DType::U32 => ImageValues::U32(values.slice()),
            DType::U64 => ImageValues::U64(values.slice()),
            DType::I8 => ImageValues::I8(values.slice()),
            DType::I16 => ImageValues::I16(values.slice()),
            DType::I32 => ImageValues::I32(values.slice()),
            DType::I64 => ImageValues::I64(values.slice()),
            DType::F32 => ImageValues::F32(values.slice()),
            DType::F64 => ImageValues::F64(values.slice()),
            other => panic!("the image kernel paints no {} plane", other.name()),
        };
        let table = color_lut(prepared.cmap).expect("an image colormap");
        let mut lut = [[0u8; 4]; 256];
        for (slot, rgba) in lut.iter_mut().zip(table.chunks_exact(4)) { slot.copy_from_slice(rgba); }
        // `_normalize_arithmetic`: a plane that promotes with float32 to float32 scales in float32.
        let single = matches!(values.dtype(), DType::Bool | DType::U8 | DType::U16 | DType::I8 | DType::I16 | DType::F16 | DType::F32);
        let plan = ImagePlan {
            values: plane, valid: prepared.valid.as_ref().map(|v| v.bytes()), shape: [1, rows, columns],
            blits: &[blit], clips: &[clip], affines: &[affine], lut: &lut, vmin: low, span, single,
        };
        // Only the rows the picture covers are split among the raster threads.
        let first = blit[1].max(clip[1]).clamp(0, height as i64) as usize;
        let last = (blit[1] + blit[3]).min(clip[3]).clamp(0, height as i64) as usize;
        if last <= first { return true; }
        let mut canvas = zlc_render::Canvas { pixels: &mut pixels[first * width * 4..last * width * 4], width, height: last - first, top: first };
        raster_prepared_images(&mut canvas, &plan).expect("prepared image");

        true
    }
}

impl Renderer {
    /// `resolved_color_limits()`: the clim the active image artist paints.
    pub fn image_clim(&self) -> Option<(f64, f64)> {
        let (axes, id) = self.image.image?;
        match self.fig.axes[axes].child(id) { Some(Child::Image(image)) => image.clim, _ => None }
    }
}

impl Renderer {
    /// `preview_color_limits(low, high)`: recolour the picture (or re-render
    /// the height-bar scene) at candidate limits without committing them.
    pub fn preview_color_limits(&mut self, low: f64, high: f64) -> crate::configuration::error::Result<()> {
        use crate::configuration::error::PlotError;
        if !(low.is_finite() && high.is_finite()) || low >= high {
            return Err(PlotError::value("preview color limits must be finite and increasing"));
        }
        let Some((axes, id)) = self.image.image else { return Err(PlotError::type_error("color-limit preview requires an Image")) };
        if self.height_bars.active {
            if let Some(call) = self.height_bars.call.clone() {
                self.in_style = true;
                self.update_height_bars_artist(call.axes, &call.values, call.valid.as_ref(), call.extent, &call.state, Some((low, high)), call.cmap);
                self.in_style = false;
            }
        } else if let Some(prepared) = self.image.prepared.as_mut() {
            prepared.color_limits = (low, high);
        }
        if let Some(Child::Image(image)) = self.fig.axes[axes].child_mut(id) { image.clim = Some((low, high)); }
        Ok(())
    }
}

impl Renderer {
    /// The image parts of one selector's scene context (`_make_selector_scene_owner`):
    /// handles in the colormap's high colour, the crosshair's sampled value and
    /// its mark on the rail, and each axis' scale.
    pub(crate) fn image_selector_context(&self, state: &crate::configuration::selectors::SelectorState, axes: usize,
                                         context: &mut crate::interaction::scene::SelectorItemContext) {
        use crate::configuration::selectors::{SelectorKind, SelectorValue};
        use crate::interaction::scene::SampleValue;
        let model = &self.fig.axes[axes];
        let lattice = |axis: &super::mpl::axes::AxisModel| axis.function.clone().map_or(AxisScale::Linear, AxisScale::Samples);
        context.x_scale = lattice(&model.xaxis);
        context.y_scale = lattice(&model.yaxis);
        let Some((image_axes, id)) = self.image_artist_on(axes) else { return };
        let Some(Child::Image(image)) = self.fig.axes[image_axes].child(id) else { return };
        let policy = &self.style.render;
        let tuple = |c: [f64; 4]| (c[0], c[1], c[2], c[3]);
        let selector = tuple(colormap_rgba(image.cmap, policy.colormap_high_fraction));
        context.selector_rgba = selector;
        context.color_limit_rgba = (selector, selector);
        context.sample_span = image.clim.map(|(low, high)| (high - low).abs());
        let payload = self.image_selector_payload(state.facet_index.map(|i| i as usize));
        if let (SelectorValue::Crosshair(point), Some(Payload::Image(data))) = (state.value, payload.as_deref()) {
            let nearest = |values: &[f64], at: f64| (0..values.len()).fold(0, |best, i| if (values[i] - at).abs() < (values[best] - at).abs() { i } else { best });
            let (x, y) = (data.x.display.to_f64(), data.y.display.to_f64());
            let z = &data.z.display;
            if !x.is_empty() && !y.is_empty() && z.shape() == [y.len(), x.len()] {
                let index = nearest(&y, point.y) * x.len() + nearest(&x, point.x);
                let usable = data.valid.get_f64(if data.valid.len() == z.len() { index } else { 0 }) != 0.0;
                let value = z.get_f64(index);
                context.sample_value = Some(if usable && value.is_finite() { SampleValue::Number(value) } else { SampleValue::Text("NaN".into()) });
            }
        }
        if state.kind() == SelectorKind::Crosshair {
            if let Some(&rail) = self.axes_of("distribution").first() {
                context.sample_target = Some(self.selector_target(rail));
                context.sample_x_limits = Some(self.fig.axes[rail].xlim);
                let mut sample = colormap_rgba(image.cmap, policy.selector_sample_colormap_fraction);
                sample[3] = policy.selector_sample_alpha;
                context.sample_rgba = Some(tuple(sample));
            }
        }
    }

    /// The colour-limit handles' context: on the rail, labelled on the picture.
    pub(crate) fn color_limit_context(&self, threshold_rgba: (f64, f64, f64, f64), threshold_label_rgba: (f64, f64, f64, f64))
                                      -> Option<crate::interaction::scene::SelectorItemContext> {
        use crate::interaction::scene::{SceneKind, SelectorItemContext};
        let rail = *self.axes_of("distribution").first()?;
        let (axes, id) = self.image.image?;
        let Some(Child::Image(image)) = self.fig.axes[axes].child(id) else { return None };
        let policy = &self.style.render;
        let tuple = |c: [f64; 4]| (c[0], c[1], c[2], c[3]);
        let model = &self.fig.axes[rail];
        Some(SelectorItemContext {
            kind: SceneKind::ColorLimits,
            target: self.selector_target(rail),
            label_target: self.selector_target(self.primary_axes()),
            x_limits: model.xlim,
            y_limits: model.ylim,
            selector_rgba: tuple(color::rgba(self.style.palette.line_single)),
            color_limit_rgba: (tuple(colormap_rgba(image.cmap, policy.colormap_low_fraction)),
                               tuple(colormap_rgba(image.cmap, policy.colormap_high_fraction))),
            threshold_rgba,
            threshold_label_rgba,
            threshold_uses_x: false,
            threshold_text: String::new(),
            x_scale: AxisScale::Linear,
            y_scale: AxisScale::Linear,
            x_label_factor: 1.0,
            sample_value: None,
            sample_span: None,
            sample_target: None,
            sample_x_limits: None,
            sample_rgba: None,
        })
    }

    /// The colour limits the handles show: a gesture's candidate, else the
    /// picture's valid clim (`_resolved_color_limit_state`).
    pub(crate) fn color_limit_state(&self) -> Option<crate::interaction::scene::ColorLimitState> {
        if let Some(value) = self.image.candidate { return Some(crate::interaction::scene::ColorLimitState { value, candidate: true }); }
        self.axes_of("distribution").first()?;
        let (low, high) = self.image_clim()?;
        let (low, high) = if high < low { (high, low) } else { (low, high) };
        if !(low.is_finite() && high.is_finite()) || low >= high { return None; }
        Some(crate::interaction::scene::ColorLimitState { value: NumericRange { low, high }, candidate: false })
    }
}
