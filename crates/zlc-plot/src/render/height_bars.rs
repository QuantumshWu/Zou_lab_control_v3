//! The Image kind's height-bar presentation (`_update_height_bars_artist`,
//! its chrome, the crosshair cage and readout): zlc-render's `height3d`
//! scene rendered into the image artist at the axes box, its room's grid,
//! axis lines and labels as matplotlib artists on `transAxes`.
use std::sync::{Arc, Mutex};

use zlc_figure::ir::{CapStyle, HAlign, VAlign};
use zlc_render::height3d::chrome::{self as h3d, ChromeLayout, DataFrame};
use zlc_render::height3d::{self, HeightBarCamera, HeightBarFrame, HeightBarRenderer, HeightBarRequest, Origin};

use super::core::Renderer;
use super::fonts::family_file;
use super::image::color_lut;
use super::mpl::axes::{Child, ChildId, TextArtist, TextFrame};
use super::mpl::color;
use super::mpl::image::{ImageArtist, ImagePixels, PolylineArtist};
use super::scale::AxisScale;
use crate::configuration::selectors::{SelectorState, SelectorValue};
use crate::configuration::{DisplayState, Value};
use crate::projection::nd::Array;

/// The scene's retained state (`_height_bars_*` and `image:h3d_*`).
#[derive(Clone, Default)]
pub struct HeightBars {
    /// `_height_bars_scene`: the current frame shows the scene.
    pub active: bool,
    /// `_height_bars_calls[key]`: the last call that painted it (for a colour-limit preview).
    pub call: Option<HeightBarCall>,
    /// `image:h3d_cache`: the renderer that keeps derived planes across camera moves.
    pub engine: Arc<Mutex<HeightBarRenderer>>,
    pub frame: Option<HeightBarFrame>,
    /// `_height_bars_rendered_camera`.
    pub camera: Option<HeightBarCamera>,
    /// `_height_bars_values`.
    pub heights: Option<Arc<Vec<f64>>>,
    pub data_frame: Option<DataFrame>,
    pub floor: f64,
    /// `_height_bars_axes_id`.
    pub axes: Option<usize>,
    /// `{key}:h3d_chrome`: grid, axis lines, labels.
    pub chrome: Option<(usize, ChildId, ChildId, Vec<ChildId>)>,
    pub cage: Option<(usize, ChildId)>,
    pub readout: Option<(usize, ChildId)>,
}

/// What painted the scene, kept so a colour-limit drag can repaint it.
#[derive(Clone)]
pub struct HeightBarCall {
    pub axes: usize,
    pub values: Array,
    pub valid: Option<Array>,
    pub extent: [f64; 4],
    pub state: Arc<DisplayState>,
    pub cmap: &'static str,
}

fn real(state: &DisplayState, name: &str) -> f64 {
    state.get(name).and_then(Value::real).unwrap_or_else(|| panic!("{name} must be a number"))
}

impl Renderer {
    /// `_height_bars_active(key, state)`: the presentation of the ONE examined
    /// image surface -- the standalone image, or a focused grid's cell (an
    /// overview stays a heatmap grid).
    pub(crate) fn height_bars_active(&self, key: &str, state: &DisplayState) -> bool {
        let wanted = matches!(state.get("presentation"), Some(Value::Text(text)) if &**text == "height_bars");
        wanted && (key == "image" || (key.starts_with("facet:") && self.facet.focus.is_some()))
    }

    /// The families a style `FontProperties` names (`style.fonts.sans_serif`).
    fn sans_serif(&self) -> Vec<String> {
        let fonts = &self.style.fonts;
        let declared: Vec<&str> = std::iter::once(fonts.family).chain(fonts.fallbacks.iter().copied()).collect();
        let installed: Vec<String> = declared.iter().filter(|f| family_file(f).is_some()).map(|f| (*f).to_owned()).collect();
        if installed.is_empty() { declared.iter().map(|f| (*f).to_owned()).collect() } else { installed }
    }

    /// `_update_height_bars_artist`: render the value grid as boxes into the image artist.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_height_bars_artist(&mut self, axes: usize, values: &Array, valid: Option<&Array>, extent: [f64; 4],
                                            state: &DisplayState, color_limits: Option<(f64, f64)>, cmap: &'static str) {
        let policy = self.style.render.clone();
        self.height_bars.call = Some(HeightBarCall {
            axes, values: values.clone(), valid: valid.cloned(), extent, state: Arc::new(state.clone()), cmap,
        });
        let model = &mut self.fig.axes[axes];
        if model.axison {
            model.axison = false;
            self.chrome_dirty.insert(axes);
        }
        self.fig.axes[axes].aspect = None;
        let (rows, columns) = (values.shape()[0], values.shape()[1]);
        let mut heights = values.to_f64();
        if let Some(valid) = valid {
            for (height, &keep) in heights.iter_mut().zip(valid.bytes()) { if keep == 0 { *height = f64::NAN; } }
        }
        // The range this surface measured for this very revision.
        let data_range = self.image.range.as_ref().and_then(|range| range.3);
        let (low, high) = height3d::bar_limits(color_limits, data_range);
        let lut = color_lut(cmap).expect("an image colormap");
        let rgb: Vec<[u8; 3]> = lut.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect();
        let table = height3d::colour_table(&rgb);
        let zero_rgb = height3d::zero_rgb(&rgb, low, high).unwrap_or_else(|e| panic!("{e}"));
        let lowest = data_range.map_or(0.0, |range| range.0);
        let camera = HeightBarCamera::new(real(state, "camera_azimuth"), real(state, "camera_elevation"), real(state, "camera_zoom"))
            .unwrap_or_else(|e| panic!("{e}"));
        self.height_bars.camera = Some(camera);
        let bbox = self.fig.axes[axes].bbox_exact(self.fig.size_inches, self.fig.dpi);
        let box_w = height3d::box_pixels(bbox[2] - bbox[0]);
        let box_h = height3d::box_pixels(bbox[3] - bbox[1]);
        let z_ticks = zlc_render::height3d::ticks::z_ticks(low, high, false);
        let [left, right, bottom, top] = extent;
        let (x_scale, y_scale) = self.image.scales.as_ref().map(|s| (s.1.clone(), s.2.clone())).unwrap_or((AxisScale::Linear, AxisScale::Linear));
        let mut coordinate_ticks: Vec<Vec<(i64, String)>> = Vec::new();
        let mut offsets: Vec<String> = Vec::new();
        for ((count, start, end), scale) in [((columns, left, right), &x_scale), ((rows, top, bottom), &y_scale)] {
            let indices = height3d::coordinate_tick_indices(count);
            let located: Vec<f64> = indices.iter().map(|&i| scale.value(height3d::coordinate_tick_value(i, start, end, count))).collect();
            let (labels, offset) = scene_axis_labels(&located);
            coordinate_ticks.push(indices.into_iter().zip(labels).collect());
            offsets.push(offset);
        }
        let font = super::mpl::artist::font_of(&self.rc, self.style.fonts.tick_pt, &self.sans_serif());
        let (widths, line_height) = {
            let figure = self.ir();
            let mut fonts = self.fonts.lock().unwrap();
            let (mut null, mut math) = (zlc_figure::query::NullBackend, super::fonts::Math);
            let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut *fonts, &mut math).expect("scene");
            let mut metric = |text: &str| zlc_figure::text::text_width_height_descent(&mut scene.ctx, text, &font, false).expect("text metrics");
            let labels: Vec<(f64, f64)> = z_ticks.iter().map(|t| &t.1).chain(coordinate_ticks.iter().flatten().map(|t| &t.1))
                .map(|label| { let (w, h, _) = metric(label); (w, h) }).collect();
            let (_, line, _) = metric("lp");
            (labels, line)
        };
        let dots_per_point = self.fig.dpi / 72.0;
        let tick_length_px = self.rc.xtick.major_size * dots_per_point;
        let label_gap_px = tick_length_px + self.rc.xtick.major_pad * dots_per_point;
        let widest = widths.iter().map(|m| m.0).fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        let tallest = widths.iter().map(|m| m.1).fold(line_height, |a, b| if b > a { b } else { a });
        let inset_px = (widest + label_gap_px, tallest + label_gap_px);
        let supersample = height3d::frame_supersample(self.plan.device_pixel_ratio, box_w, box_h, rows, columns);
        let rim = color::rgba(policy.height_bars_axis_color);
        let mut request = HeightBarRequest::new(&heights, rows, columns, &table, box_w, box_h);
        request.camera = camera;
        request.value_limits = (low, high);
        request.supersample = supersample;
        request.side_shades = policy.height_bars_side_shades;
        let background = policy.height_bars_background_rgb;
        request.background_rgb = [background.0, background.1, background.2];
        request.zero_rgb = Some(zero_rgb);
        request.z_fraction = policy.height_bars_z_fraction;
        request.rim_rgb = [rim[0], rim[1], rim[2]];
        request.rim_width_px = height3d::rim_width_px(self.rc.axes_linewidth, self.plan.dpi);
        request.origin = if policy.image_origin == "upper" { Origin::Upper } else { Origin::Lower };
        request.inset_px = inset_px;
        let frame = self.height_bars.engine.lock().unwrap().render(&request).unwrap_or_else(|e| panic!("{e}"));
        self.height_bars.heights = Some(Arc::new(heights));
        self.height_bars.axes = Some(axes);
        self.height_bars.floor = h3d::floor_value(lowest, low);
        self.height_bars.data_frame = Some(DataFrame { extent: (left, right, bottom, top), source_nx: columns, source_ny: rows });
        let pixels = ImagePixels::Rgba { rows: box_h, columns: box_w, pixels: frame.rgba.clone() };
        match self.image.image {
            None => {
                let image = ImageArtist {
                    visible: true, pixels: ImagePixels::Rgba { rows: box_h, columns: box_w, pixels: frame.rgba.clone() },
                    extent: [0.0, 1.0, 0.0, 1.0], origin: policy.image_origin, cmap, clim: Some((low, high)), lattice: false,
                };
                let id = self.fig.axes[axes].add(Child::Image(Box::new(image)));
                self.image.image = Some((axes, id));
            }
            Some((owner, id)) => {
                if let Some(Child::Image(image)) = self.fig.axes[owner].child_mut(id) {
                    image.pixels = pixels;
                    image.lattice = false;
                    image.extent = [0.0, 1.0, 0.0, 1.0];
                    image.cmap = cmap;
                    image.clim = Some((low, high));
                }
            }
        }
        self.home_limits.insert(axes, ((0.0, 1.0), (0.0, 1.0)));
        self.set_xlim(axes, (0.0, 1.0));
        self.set_ylim(axes, (0.0, 1.0));
        let scene = frame.scene.clone();
        self.height_bars.frame = Some(frame);
        let layout = ChromeLayout {
            z_ticks, x_ticks: coordinate_ticks[0].clone(), y_ticks: coordinate_ticks[1].clone(),
            x_offset: offsets[0].clone(), y_offset: offsets[1].clone(), tick_length_px, label_gap_px, box_w, box_h,
            floor_value: Some(self.height_bars.floor),
        };
        self.update_height_bars_chrome(axes, &h3d::chrome(&scene, &layout));
    }

    /// The scene's chrome artists (`_update_height_bars_chrome`'s artists half).
    fn update_height_bars_chrome(&mut self, axes: usize, geometry: &h3d::ChromeGeometry) {
        let policy = self.style.render.clone();
        let (grid_rgb, axis_rgba) = (policy.height_bars_grid_rgb, color::rgba(policy.height_bars_axis_color));
        if self.height_bars.chrome.is_none() {
            let model = &mut self.fig.axes[axes];
            let grid = model.add(Child::Polyline(PolylineArtist {
                visible: true, zorder: 5.0, alpha: Some(policy.height_bars_grid_alpha), color: [grid_rgb.0, grid_rgb.1, grid_rgb.2, 1.0],
                linewidth: policy.height_bars_grid_line_pt, capstyle: CapStyle::Butt, x: Vec::new(), y: Vec::new(),
            }));
            let lines = model.add(Child::Polyline(PolylineArtist {
                visible: true, zorder: 6.0, alpha: None, color: axis_rgba, linewidth: self.rc.axes_linewidth,
                capstyle: CapStyle::Projecting, x: Vec::new(), y: Vec::new(),
            }));
            self.height_bars.chrome = Some((axes, grid, lines, Vec::new()));
        }
        let (owner, grid, lines, mut texts) = self.height_bars.chrome.clone().unwrap();
        let model = &mut self.fig.axes[owner];
        if let Some(Child::Polyline(line)) = model.child_mut(grid) {
            line.x = geometry.grid_x.clone();
            line.y = geometry.grid_y.clone();
            line.visible = true;
        }
        if let Some(Child::Polyline(line)) = model.child_mut(lines) {
            line.x = geometry.line_x.clone();
            line.y = geometry.line_y.clone();
            line.visible = true;
        }
        let families = self.sans_serif();
        let size = self.style.fonts.tick_pt;
        for (index, wanted) in geometry.texts.iter().enumerate() {
            if index == texts.len() {
                let text = TextArtist {
                    visible: true, zorder: 6.0, x: 0.0, y: 0.0, text: String::new(), color: self.rc.text_color.clone(), size,
                    ha: HAlign::Left, va: VAlign::Baseline, ma: None, clip_on: true, frame: TextFrame::Axes, family: Some(families.clone()),
                };
                texts.push(self.fig.axes[owner].add(Child::Text(text)));
            }
            if let Some(Child::Text(text)) = self.fig.axes[owner].child_mut(texts[index]) {
                text.x = wanted.x;
                text.y = wanted.y;
                text.text = wanted.text.clone();
                text.ha = match wanted.ha { "left" => HAlign::Left, "right" => HAlign::Right, _ => HAlign::Center };
                text.va = match wanted.va { "bottom" => VAlign::Bottom, "top" => VAlign::Top, _ => VAlign::Center };
                text.visible = true;
                text.clip_on = true;
            }
        }
        for &id in &texts[geometry.texts.len()..] {
            if let Some(child) = self.fig.axes[owner].child_mut(id) { child.set_visible(false); }
        }
        let projected = geometry.projected_axes.len();
        self.height_bars.chrome = Some((owner, grid, lines, texts.clone()));
        self.thin_overlapping_chrome(owner, &texts[..projected], &geometry.projected_axes);
    }

    /// `_thin_overlapping_chrome`: drop projected labels that would print
    /// across one already kept, every axis' ends first.
    fn thin_overlapping_chrome(&mut self, axes: usize, texts: &[ChildId], axes_of: &[&'static str]) {
        let visible: Vec<(ChildId, &'static str)> = texts.iter().zip(axes_of)
            .filter(|(id, _)| self.fig.axes[axes].child(**id).is_some_and(Child::visible)).map(|(id, a)| (*id, *a)).collect();
        if visible.len() < 2 { return; }
        let mut ends: Vec<usize> = Vec::new();
        let mut inner: Vec<usize> = Vec::new();
        let mut names: Vec<&str> = Vec::new();
        for (_, name) in &visible { if !names.contains(name) { names.push(name); } }
        for name in names {
            let members: Vec<usize> = (0..visible.len()).filter(|&i| visible[i].1 == name).collect();
            ends.push(members[0]);
            if members[members.len() - 1] != members[0] { ends.push(members[members.len() - 1]); }
            inner.extend(&members[1..members.len().saturating_sub(1).max(1)]);
        }
        let figure = self.ir();
        let extents: Vec<[f64; 4]> = {
            let mut fonts = self.fonts.lock().unwrap();
            let (mut null, mut math) = (zlc_figure::query::NullBackend, super::fonts::Math);
            let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut *fonts, &mut math).expect("scene");
            let children = &self.fig.axes[axes].children;
            visible.iter().map(|(id, _)| {
                let index = children.iter().position(|(c, _)| c == id).unwrap();
                let zlc_figure::ir::Artist::Text(text) = &figure.axes[axes].children[index] else { unreachable!() };
                scene.text_extent(text).expect("label extent")
            }).collect()
        };
        let overlaps = |a: &[f64; 4], b: &[f64; 4]| {
            // `Bbox.overlaps`: strict on every side, either orientation.
            let (ax0, ax1) = (a[0].min(a[2]), a[0].max(a[2]));
            let (ay0, ay1) = (a[1].min(a[3]), a[1].max(a[3]));
            let (bx0, bx1) = (b[0].min(b[2]), b[0].max(b[2]));
            let (by0, by1) = (b[1].min(b[3]), b[1].max(b[3]));
            ax0 < bx1 && bx0 < ax1 && ay0 < by1 && by0 < ay1
        };
        let mut kept: Vec<[f64; 4]> = Vec::new();
        for position in ends.into_iter().chain(inner) {
            let extent = extents[position];
            if kept.iter().any(|other| overlaps(&extent, other)) {
                if let Some(child) = self.fig.axes[axes].child_mut(visible[position].0) { child.set_visible(false); }
                continue;
            }
            kept.push(extent);
        }
    }

    /// `_hide_height_bars_chrome`.
    pub(crate) fn hide_height_bars_chrome(&mut self) {
        if let Some((axes, grid, lines, texts)) = self.height_bars.chrome.clone() {
            for id in [grid, lines].into_iter().chain(texts) {
                if let Some(child) = self.fig.axes[axes].child_mut(id) { child.set_visible(false); }
            }
        }
        if let Some((axes, cage)) = self.height_bars.cage {
            if let Some(child) = self.fig.axes[axes].child_mut(cage) { child.set_visible(false); }
        }
    }

    /// The present half: `scene_3d` against the last frame's (a change retires
    /// the composition epoch, and leaving the scene drops its working set).
    pub(crate) fn settle_height_bars_scene(&mut self, state: &DisplayState) -> bool {
        let scene = self.height_bars_active(&self.primary_surface().0, state);
        if scene != self.height_bars.active {
            self.background = None;
            if !scene {
                self.height_bars.engine = Arc::new(Mutex::new(HeightBarRenderer::default()));
                self.height_bars.heights = None;
                self.height_bars.frame = None;
                self.height_bars.data_frame = None;
            }
        }
        self.height_bars.active = scene;
        scene
    }

    /// `_update_height_bars_cage(snapshot)`: a wireframe over the crosshair's bar.
    pub(crate) fn update_height_bars_cage(&mut self, selectors: &[SelectorState]) {
        let crosshair = selectors.iter().find_map(|s| match &s.value { SelectorValue::Crosshair(point) => Some((point.x, point.y)), _ => None });
        let hide = |r: &mut Renderer| {
            for slot in [r.height_bars.cage, r.height_bars.readout].into_iter().flatten() {
                if let Some(child) = r.fig.axes[slot.0].child_mut(slot.1) { child.set_visible(false); }
            }
        };
        let (Some(frame), Some(data_frame), Some((x, y))) = (self.height_bars.frame.clone(), self.height_bars.data_frame, crosshair) else {
            hide(self);
            return;
        };
        let (x_scale, y_scale) = self.image.scales.as_ref().map(|s| (s.1.clone(), s.2.clone())).unwrap_or((AxisScale::Linear, AxisScale::Linear));
        let Some(cell) = h3d::cell_of(&data_frame, x_scale.space(x), y_scale.space(y)) else { hide(self); return };
        let (xs, ys) = h3d::cage(&frame.scene, cell);
        let axes = self.primary_axes();
        let policy = self.style.render.clone();
        if self.height_bars.cage.is_none() {
            let grid = policy.height_bars_grid_rgb;
            let id = self.fig.axes[axes].add(Child::Polyline(PolylineArtist {
                visible: true, zorder: 7.0, alpha: Some(policy.height_bars_cage_alpha), color: [grid.0, grid.1, grid.2, 1.0],
                linewidth: policy.height_bars_grid_line_pt, capstyle: CapStyle::Round, x: Vec::new(), y: Vec::new(),
            }));
            self.height_bars.cage = Some((axes, id));
        }
        let (owner, cage) = self.height_bars.cage.unwrap();
        if let Some(Child::Polyline(line)) = self.fig.axes[owner].child_mut(cage) {
            line.x = xs;
            line.y = ys;
            line.visible = true;
        }
        self.update_height_bars_readout(axes, (x, y), cell, &x_scale, &y_scale);
    }

    /// `_update_height_bars_readout`: (x, y, z) of the selected bar in the axes corner.
    fn update_height_bars_readout(&mut self, axes: usize, at: (f64, f64), cell: (i64, i64), x_scale: &AxisScale, y_scale: &AxisScale) {
        let (Some(heights), Some(frame), Some(scene)) = (self.height_bars.heights.clone(), self.height_bars.data_frame, self.height_bars.frame.clone()) else { return };
        let (left, right, bottom, top) = frame.extent;
        let z = heights[cell.0 as usize * frame.source_nx + cell.1 as usize];
        let precision = |span: f64| h3d::selector_precision(span).unwrap_or_else(|e| panic!("{e}"));
        let xp = precision((x_scale.value(right) - x_scale.value(left)).abs());
        let yp = precision((y_scale.value(top) - y_scale.value(bottom)).abs());
        let zp = precision((scene.scene.value_high - scene.scene.value_low).abs());
        let text = format!("({}, {}, {})", h3d::selector_number(at.0, xp), h3d::selector_number(at.1, yp), h3d::selector_number(z, zp));
        let inset = self.style.render.axes_text_inset_fraction;
        if self.height_bars.readout.is_none() {
            let readout = TextArtist {
                visible: true, zorder: 8.0, x: 1.0 - inset, y: 1.0 - inset, text: text.clone(), color: self.style.palette.line_single.to_owned(),
                size: self.style.fonts.annotation_pt, ha: HAlign::Left, va: VAlign::Top, ma: None, clip_on: false, frame: TextFrame::Axes,
                family: Some(vec![self.sans_serif()[0].clone()]),
            };
            let id = self.fig.axes[axes].add(Child::Text(readout));
            self.height_bars.readout = Some((axes, id));
        }
        let (owner, id) = self.height_bars.readout.unwrap();
        if let Some(Child::Text(readout)) = self.fig.axes[owner].child_mut(id) { readout.text = text; }
        self.pin_readout_held(owner, id, 1.0 - inset, false);
        if let Some(child) = self.fig.axes[owner].child_mut(id) { child.set_visible(true); }
    }

    /// `height_bars_pick(x, y)`: the picked bar's data coordinates.
    pub fn height_bars_pick(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let (frame, data_frame) = (self.height_bars.frame.as_ref()?, self.height_bars.data_frame.as_ref()?);
        let axes = &self.fig.axes[self.primary_axes()];
        let bbox = axes.bbox_exact(self.fig.size_inches, self.fig.dpi);
        let (x_space, y_space) = h3d::pick_axis_space(&frame.scene, data_frame, bbox[0], bbox[3], x, y)?;
        let (x_scale, y_scale) = self.image.scales.as_ref().map(|s| (s.1.clone(), s.2.clone())).unwrap_or((AxisScale::Linear, AxisScale::Linear));
        Some((x_scale.value(x_space), y_scale.value(y_space)))
    }
}

/// `_scene_axis_labels(values)`: a ScalarFormatter's labels and shared offset.
fn scene_axis_labels(values: &[f64]) -> (Vec<String>, String) {
    let low = values.iter().copied().fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    let high = values.iter().copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let formatter = super::mpl::ticker::ScalarFormatter::set_locs(values, (low, high));
    (values.iter().map(|&v| formatter.format(v)).collect(), formatter.offset_text())
}
