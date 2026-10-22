//! The image point overlay (`_update_image_point_overlay`): site rings in
//! their status colours (an `EllipseCollection`), or -- with movement paths --
//! one filled coverage union per status colour (`_point_path_union_paths`),
//! the timebase and timing labels placed clear of the site labels, and the
//! site labels.
use std::sync::Arc;

use zlc_figure::ir::{HAlign, VAlign};

use super::core::Renderer;
use super::mpl::axes::{Child, ChildId, TextArtist, TextFrame};
use super::mpl::color;
use super::mpl::image::{EllipsesArtist, UnionArtist};
use crate::configuration::{spec::Surface, DisplayState, PlotSpec, Value};
use crate::kinds::AxisDomain;
use crate::projection::{nd::Scalar, output::{ImageData, Payload}};
use crate::session::overlay::{ImagePointOverlay, PointStatus};

/// The overlay's artists (`{key}:points*`) and the signature they were made for.
#[derive(Clone, Debug, Default)]
pub struct PointTables {
    pub signature: Option<PointSignature>,
    pub axes: usize,
    pub rings: Option<ChildId>,
    pub joined: Option<ChildId>,
    pub labels: Vec<ChildId>,
    pub path_labels: Vec<ChildId>,
    pub timebase: Option<ChildId>,
    /// Whether the label lists exist yet (`setdefault` on the first update).
    pub listed: bool,
}

/// `{key}:points-signature`.
#[derive(Clone, Debug, PartialEq)]
pub struct PointSignature {
    overlay: Option<(u64, usize)>,
    facet_value: Option<Scalar>,
    show_labels: bool,
    units: (String, String),
    view: Option<([u64; 4], [u64; 4], u64)>,
}

/// `_point_ring_radius(points, fraction=..., fallback=...)`: a fraction of the
/// median nearest-neighbour distance.
fn point_ring_radius(points: &[[f64; 2]], fraction: f64, fallback: f64) -> f64 {
    if points.len() < 2 || !points.iter().all(|p| p[0].is_finite() && p[1].is_finite()) { return fallback; }
    let mut nearest: Vec<f64> = points.iter().enumerate().map(|(i, a)| {
        let mut best = f64::INFINITY;
        for (j, b) in points.iter().enumerate() {
            let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
            let squared = if i == j { f64::INFINITY } else { dx * dx + dy * dy };
            if squared < best || best.is_nan() { best = squared; }
        }
        best.sqrt()
    }).filter(|d| d.is_finite() && *d > 0.0).collect();
    if nearest.is_empty() { return fallback; }
    nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = nearest.len();
    let median = if n % 2 == 1 { nearest[n / 2] } else { (nearest[n / 2 - 1] + nearest[n / 2]) / 2.0 };
    fraction * median
}

/// `ImagePointOverlay.statuses_for(spec, facet_value)`: the site judgements of
/// the one exact (repeat, point) cell the image's scope (and a grid cell's facet
/// coordinate) names, or None.
fn statuses_for(overlay: &ImagePointOverlay, spec: &PlotSpec, facet_value: Option<&Scalar>) -> Option<Vec<PointStatus>> {
    use crate::configuration::data_contract::{resolve_axis, PlotSchema};
    use crate::configuration::semantics::{projection_scope, ScopeCoordinate};
    use crate::projection::plan::ScopeValue;
    if let Some(statuses) = &overlay.static_statuses { return Some(statuses.clone()); }
    let status = overlay.status.as_ref()?;
    let schema = PlotSchema::new(status.schema.clone());
    let scope = projection_scope(&schema, spec, Some(&[AxisDomain::Repeat, AxisDomain::Point])).ok()?;
    let mut terms: Vec<(Arc<str>, ScopeValue)> = Vec::new();
    for (reference, coordinate) in scope {
        if reference.domain == AxisDomain::CellData { return None; }
        let axis = resolve_axis(&schema, &reference).ok()?;
        let value = match coordinate {
            ScopeCoordinate::Latest => ScopeValue::Latest,
            ScopeCoordinate::Value(c) => ScopeValue::Coordinate(crate::session::projected::coordinate_scalar(&c)),
        };
        let id: Arc<str> = Arc::from(axis.axis.id());
        match terms.iter_mut().find(|(held, _)| *held == id) { Some(term) => term.1 = value, None => terms.push((id, value)) }
    }
    if let (PlotSpec::FacetGrid(grid), Some(value)) = (spec, facet_value) {
        if let Some(facet) = &grid.facet {
            if facet.domain == AxisDomain::CellData { return None; }
            let id: Arc<str> = Arc::from(resolve_axis(&schema, facet).ok()?.axis.id());
            let value = ScopeValue::Coordinate(value.clone());
            match terms.iter_mut().find(|(held, _)| *held == id) { Some(term) => term.1 = value, None => terms.push((id, value)) }
        }
    }
    let mut cache = crate::projection::SchemaCache::default();
    let snapshot = crate::projection::Snapshot::from_source(status, &mut cache).ok()?;
    let (repeat, point) = crate::projection::scope::selected_cells(&snapshot.schema, &terms).ok()?;
    if repeat.len() != 1 || point.len() != 1 { return None; }
    let (acquired, values) = cell_statuses(&snapshot, repeat.at(0), point.at(0));
    if !acquired.iter().any(|v| *v) { return None; }
    Some(acquired.iter().zip(&values).map(|(&ok, &value)| {
        if !ok { PointStatus::Invalid } else if value { PointStatus::Occupied } else { PointStatus::Empty }
    }).collect())
}

impl Renderer {
    /// `_update_image_point_overlay` of every painted image surface: an Image's
    /// one, or each cell of a grid of images with its own facet coordinate.
    pub(crate) fn update_image_point_overlay(&mut self, overlay: Option<&Arc<ImagePointOverlay>>, state: &DisplayState) {
        if !matches!(self.spec.semantic(), Surface::Image(_)) { return; }
        let Some(payload) = self.last_payload.clone() else { return };
        match &*payload {
            Payload::Image(data) => { let axes = self.primary_axes(); self.update_surface_points(axes, data, overlay, state, None); }
            Payload::Facet(facet) => {
                for (key, axes, index) in self.painted_surfaces() {
                    let Some(cell) = index.and_then(|index| facet.cells.get(index)) else { continue };
                    let Payload::Image(data) = &cell.payload else { continue };
                    self.select_image(&key);
                    self.update_surface_points(axes, data, overlay, state, Some(&cell.facet_value_canonical));
                }
                // The renderer's image tables are the primary surface's between frames.
                let (key, _) = self.primary_surface();
                if key.starts_with("facet:") { self.select_image(&key); }
            }
            _ => {}
        }
    }

    /// `_update_image_point_overlay(axes, payload, overlay, state, key, facet_value)`.
    fn update_surface_points(&mut self, axes: usize, data: &ImageData, overlay: Option<&Arc<ImagePointOverlay>>, state: &DisplayState, facet_value: Option<&Scalar>) {
        let show_labels = matches!(state.get("show_point_labels"), Some(Value::Bool(true)));
        let view = overlay.filter(|o| o.paths_xy.is_some()).map(|_| {
            let a = &self.fig.axes[axes];
            let b = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
            ([a.xlim.0, a.xlim.1, a.ylim.0, a.ylim.1].map(f64::to_bits), [b[0], b[1], b[2] - b[0], b[3] - b[1]].map(f64::to_bits), self.fig.dpi.to_bits())
        });
        let signature = PointSignature {
            overlay: overlay.map(|o| (o.revision, Arc::as_ptr(o) as usize)), facet_value: facet_value.cloned(), show_labels,
            units: (data.x.display_unit.symbol.to_string(), data.y.display_unit.symbol.to_string()), view,
        };
        if self.image.points.signature.as_ref() == Some(&signature) { return; }
        self.image.points.signature = Some(signature);
        self.image.points.axes = axes;
        self.image.points.listed = true;
        let points = self.image.points.clone();
        for &id in points.path_labels.iter().chain(&points.timebase) { self.set_point_child_visible(id, false); }
        let statuses = overlay.and_then(|o| statuses_for(o, &self.spec, facet_value));
        let Some(overlay) = overlay.filter(|o| o.count() > 0 && !(o.status.is_some() && statuses.is_none())) else {
            for &id in points.rings.iter().chain(&points.joined).chain(&points.labels) { self.set_point_child_visible(id, false); }
            return;
        };
        self.paint_point_overlay(data, overlay, statuses, show_labels);
    }

    fn set_point_child_visible(&mut self, id: ChildId, visible: bool) {
        let axes = self.image.points.axes;
        if let Some(child) = self.fig.axes[axes].child_mut(id) { child.set_visible(visible); }
    }

    fn paint_point_overlay(&mut self, data: &ImageData, overlay: &ImagePointOverlay, statuses: Option<Vec<PointStatus>>, show_labels: bool) {
        let axes = self.image.points.axes;
        let (xq, yq) = (&data.x, &data.y);
        let convert = |q: &crate::projection::output::QuantityArray, v: f64| q.canonical_unit.convert_value_to(v, &q.display_unit)
            .unwrap_or_else(|e| panic!("{e}"));
        let count = overlay.count();
        let canonical: Vec<[f64; 2]> = overlay.coordinates.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
        let points: Vec<[f64; 2]> = canonical.iter().map(|p| [convert(xq, p[0]), convert(yq, p[1])]).collect();
        let span = |q: &crate::projection::output::QuantityArray| {
            let finite: Vec<f64> = q.canonical.to_f64().into_iter().filter(|v| v.is_finite()).collect();
            (!finite.is_empty()).then(|| finite.iter().copied().fold(f64::NEG_INFINITY, f64::max) - finite.iter().copied().fold(f64::INFINITY, f64::min))
        };
        let spans: Vec<f64> = [span(xq), span(yq)].into_iter().flatten().filter(|s| *s > 0.0).collect();
        let visual = self.style.artists.clone();
        let fallback = spans.iter().copied().fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| if s < m { s } else { m }))).unwrap_or(1.0)
            * visual.point_single_radius_fraction;
        let radius = point_ring_radius(&canonical, visual.point_auto_radius_fraction, fallback);
        let display_radius = |q| (convert(q, radius) - convert(q, 0.0)).abs();
        let radii = (display_radius(xq), display_radius(yq));
        let statuses = statuses.filter(|s| !s.is_empty()).unwrap_or_else(|| vec![PointStatus::Unknown; count]);
        let dynamic = overlay.status.is_some();
        let token = |s: PointStatus| match s {
            PointStatus::Unknown => &visual.point_unknown, PointStatus::Empty => &visual.point_empty,
            PointStatus::Occupied => &visual.point_occupied, PointStatus::Invalid => &visual.point_invalid,
        };
        let colours: Vec<[f64; 4]> = statuses.iter().map(|&s| {
            let t = token(s);
            color::to_rgba_alpha(t.color, Some(if dynamic && s == PointStatus::Invalid { 0.0 } else { t.alpha })).expect("point colour")
        }).collect();
        let widths: Vec<f64> = statuses.iter().map(|&s| token(s).linewidth).collect();
        let dashed: Vec<bool> = statuses.iter().map(|&s| s == PointStatus::Invalid).collect();
        let fit_pt = self.style.fonts.fit_annotation_pt;
        match &overlay.paths_xy {
            None => {
                if let Some(id) = self.image.points.joined { self.set_point_child_visible(id, false); }
                let rings = EllipsesArtist {
                    visible: true, zorder: visual.point_zorder, offsets: points.clone(), widths: vec![2.0 * radii.0; count],
                    heights: vec![2.0 * radii.1; count], edgecolors: colours.clone(), linewidths: widths.clone(), dashed: dashed.clone(),
                };
                match self.image.points.rings {
                    None => self.image.points.rings = Some(self.fig.axes[axes].add(Child::Ellipses(Box::new(rings)))),
                    Some(id) => if let Some(Child::Ellipses(held)) = self.fig.axes[axes].child_mut(id) { **held = rings; },
                }
            }
            Some((xy, steps)) => {
                if let Some(id) = self.image.points.rings { self.set_point_child_visible(id, false); }
                let paths: Vec<Vec<[f64; 2]>> = xy.chunks_exact(steps * 2).map(|path| {
                    path.chunks_exact(2).map(|p| [convert(xq, p[0]), convert(yq, p[1])]).collect()
                }).collect();
                let raw: Vec<Vec<[f64; 2]>> = xy.chunks_exact(steps * 2).map(|path| path.chunks_exact(2).map(|p| [p[0], p[1]]).collect()).collect();
                let union = UnionArtist {
                    visible: true, zorder: visual.point_zorder, points: points.clone(), paths: paths.clone(), radii,
                    colours: colours.clone(), widths: widths.clone(), dashed: dashed.clone(),
                };
                match self.image.points.joined {
                    None => self.image.points.joined = Some(self.fig.axes[axes].add(Child::Union(Box::new(union)))),
                    Some(id) => if let Some(Child::Union(held)) = self.fig.axes[axes].child_mut(id) { **held = union; },
                }
                if self.image.points.timebase.is_none() {
                    let text = TextArtist {
                        visible: true, zorder: visual.point_label_zorder, x: 0.015, y: 0.015, text: String::new(),
                        color: self.style.palette.annotation.to_owned(), size: fit_pt, ha: HAlign::Left, va: VAlign::Bottom,
                        ma: None, clip_on: true, frame: TextFrame::Axes, family: None,
                    };
                    self.image.points.timebase = Some(self.fig.axes[axes].add(Child::Text(text)));
                }
                let timebase = self.image.points.timebase.unwrap();
                if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(timebase) {
                    text.text = format!("f0 \u{2192} f{}", steps - 1);
                    text.visible = show_labels;
                }
                let mut used = 0;
                for (index, path) in paths.iter().enumerate() {
                    if colours[index][3] <= 0.0 { continue; }
                    for (step, text) in timing_labels(&raw[index], path) {
                        if used == self.image.points.path_labels.len() {
                            let label = TextArtist {
                                visible: true, zorder: visual.point_label_zorder, x: -2.0, y: -2.0, text: String::new(),
                                color: self.rc.text_color.clone(), size: fit_pt, ha: HAlign::Right, va: VAlign::Top, ma: None,
                                clip_on: true, frame: TextFrame::Annotation(path[step][0], path[step][1], None), family: None,
                            };
                            let id = self.fig.axes[axes].add(Child::Text(label));
                            self.image.points.path_labels.push(id);
                        }
                        let id = self.image.points.path_labels[used];
                        if let Some(Child::Text(label)) = self.fig.axes[axes].child_mut(id) {
                            if let TextFrame::Annotation(_, _, placed) = label.frame { label.frame = TextFrame::Annotation(path[step][0], path[step][1], placed); }
                            label.text = text;
                            label.color = color::tuple_spec(colours[index]);
                            label.visible = show_labels;
                        }
                        used += 1;
                    }
                }
            }
        }
        while self.image.points.labels.len() < count {
            let label = TextArtist {
                visible: true, zorder: visual.point_label_zorder, x: 0.0, y: 0.0, text: String::new(), color: self.rc.text_color.clone(),
                size: fit_pt, ha: HAlign::Right, va: VAlign::Bottom, ma: None, clip_on: true, frame: TextFrame::Data, family: None,
            };
            let id = self.fig.axes[axes].add(Child::Text(label));
            self.image.points.labels.push(id);
        }
        let (x_inverted, y_inverted) = { let a = &self.fig.axes[axes]; (a.xlim.0 > a.xlim.1, a.ylim.0 > a.ylim.1) };
        for (index, &id) in self.image.points.labels.clone().iter().enumerate() {
            let visible = show_labels && index < count && colours[index][3] > 0.0;
            let Some(Child::Text(label)) = self.fig.axes[axes].child_mut(id) else { continue };
            label.visible = visible;
            if !visible { continue; }
            label.size = fit_pt;
            label.ha = HAlign::Right;
            label.va = VAlign::Bottom;
            label.x = points[index][0] + if x_inverted { radii.0 } else { -radii.0 };
            label.y = points[index][1] + if y_inverted { -radii.1 } else { radii.1 };
            label.color = color::tuple_spec(colours[index]);
            label.text = overlay.labels.as_ref().and_then(|l| l[index].clone()).filter(|t| !t.is_empty())
                .or_else(|| overlay.point_ids.as_ref().map(|ids| ids[index].clone()).filter(|t| !t.is_empty())).unwrap_or_default();
        }
        if overlay.paths_xy.is_some() && show_labels { self.place_timing_labels(axes, &points, radii); }
    }

    /// The timing labels' placement: of eight directions round each anchor,
    /// the one least outside the axes and least over the labels placed before.
    fn place_timing_labels(&mut self, axes: usize, points: &[[f64; 2]], radii: (f64, f64)) {
        let dpi = self.fig.dpi;
        let gap = 2.0 * dpi / 72.0;
        let directions = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        let fit_pt = self.style.fonts.fit_annotation_pt;
        let model = &self.fig.axes[axes];
        let bbox = model.bbox_exact(self.fig.size_inches, dpi);
        let visible_text = |id: &ChildId| match model.child(*id) { Some(Child::Text(t)) if t.visible => Some(t.clone()), _ => None };
        let sources: Vec<ChildId> = self.image.points.labels.iter().copied()
            .filter(|id| visible_text(id).is_some_and(|t| !t.text.is_empty())).collect();
        let ordered: Vec<ChildId> = self.image.points.path_labels.iter().copied().filter(|id| visible_text(id).is_some()).collect();
        let mut placed: Vec<[f64; 4]> = sources.iter().map(|&id| self.text_window_extent(axes, id)).collect();
        for id in ordered {
            if let Some(Child::Text(label)) = self.fig.axes[axes].child_mut(id) { label.size = fit_pt; }
            let bounds = self.text_window_extent(axes, id);
            let (width, height) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
            let TextFrame::Annotation(px, py, _) = (match self.fig.axes[axes].child(id) { Some(Child::Text(t)) => t.frame, _ => unreachable!() }) else { unreachable!() };
            let model = &self.fig.axes[axes];
            let pixel = model.to_display(self.fig.size_inches, dpi, px, py);
            let at_site = points.iter().any(|p| p[0] == px && p[1] == py);
            let site_radius = if at_site {
                let edge_x = model.to_display(self.fig.size_inches, dpi, px + radii.0, py);
                let edge_y = model.to_display(self.fig.size_inches, dpi, px, py + radii.1);
                let norm = |e: (f64, f64)| { let (dx, dy) = (e.0 - pixel.0, e.1 - pixel.1); (dx * dx + dy * dy).sqrt() };
                (norm(edge_x), norm(edge_y))
            } else { (0.0, 0.0) };
            let mut best: Option<(bool, f64, f64, f64, f64, f64, f64, [f64; 4])> = None;
            for (dx, dy) in directions {
                let (offx, offy) = (dx * (gap + site_radius.0), dy * (gap + site_radius.1));
                self.set_timing_label(axes, id, dx, dy, offx * 72.0 / dpi, offy * 72.0 / dpi);
                let b = self.text_window_extent(axes, id);
                let area: Vec<f64> = placed.iter().map(|p| {
                    let ox = (b[2].min(p[2]) - b[0].max(p[0])).max(0.0);
                    let oy = (b[3].min(p[3]) - b[1].max(p[1])).max(0.0);
                    ox * oy
                }).collect();
                let area = crate::numeric::sum_f64(&area);
                let inside = (b[2].min(bbox[2]) - b[0].max(bbox[0])).max(0.0) * (b[3].min(bbox[3]) - b[1].max(bbox[1])).max(0.0);
                let outside = (width * height - inside).max(0.0);
                let candidate = (outside > 1e-6, if outside > 1e-6 { outside } else { 0.0 }, area, offx, offy, dx, dy, b);
                let better = best.as_ref().is_none_or(|held| (candidate.0, candidate.1, candidate.2).partial_cmp(&(held.0, held.1, held.2)) == Some(std::cmp::Ordering::Less));
                if better { best = Some(candidate); }
            }
            let (_, _, _, offx, offy, dx, dy, b) = best.unwrap();
            placed.push(b);
            self.set_timing_label(axes, id, dx, dy, offx * 72.0 / dpi, offy * 72.0 / dpi);
        }
        self.settle_path_labels();
    }

    /// What a draw or measure leaves on the visible path labels whose anchor is
    /// in the axes (`update_positions`): the transform at the anchor's pixel.
    pub(crate) fn settle_path_labels(&mut self) {
        let axes = self.image.points.axes;
        let shown: Vec<(usize, ChildId)> = self.image.points.path_labels.iter().filter_map(|&id| {
            let index = self.fig.axes[axes].children.iter().position(|(c, _)| *c == id)?;
            matches!(&self.fig.axes[axes].children[index].1, Child::Text(t) if t.visible).then_some((index, id))
        }).collect();
        if shown.is_empty() { return; }
        let figure = self.ir();
        let inside: Vec<bool> = {
            let mut fonts = self.fonts.lock().unwrap();
            let (mut null, mut math) = (zlc_figure::query::NullBackend, super::fonts::Math);
            let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut *fonts, &mut math).expect("scene");
            shown.iter().map(|&(index, _)| match &figure.axes[axes].children[index] {
                zlc_figure::ir::Artist::Text(t) => zlc_figure::text::check_xy(&mut scene.ctx, t).expect("annotation clip"),
                _ => false,
            }).collect()
        };
        let (size, dpi) = (self.fig.size_inches, self.fig.dpi);
        for (&(_, id), inside) in shown.iter().zip(inside) {
            if !inside { continue; }
            let model = &mut self.fig.axes[axes];
            let Some(Child::Text(TextArtist { frame: TextFrame::Annotation(x, y, _), .. })) = model.child(id).cloned() else { continue };
            let (px, py) = model.to_display(size, dpi, x, y);
            if let Some(Child::Text(label)) = model.child_mut(id) { label.frame = TextFrame::Annotation(x, y, Some([dpi / 72.0, px, py])); }
        }
    }

    fn set_timing_label(&mut self, axes: usize, id: ChildId, dx: f64, dy: f64, x: f64, y: f64) {
        if let Some(Child::Text(label)) = self.fig.axes[axes].child_mut(id) {
            label.ha = if dx < 0.0 { HAlign::Right } else if dx > 0.0 { HAlign::Left } else { HAlign::Center };
            label.va = if dy < 0.0 { VAlign::Top } else if dy > 0.0 { VAlign::Bottom } else { VAlign::Center };
            label.x = x;
            label.y = y;
        }
    }

    /// `text.get_window_extent(renderer)` of one text child as it stands.
    fn text_window_extent(&self, axes: usize, id: ChildId) -> [f64; 4] {
        let figure = self.ir();
        let index = self.fig.axes[axes].children.iter().position(|(c, _)| *c == id).expect("a text child");
        let zlc_figure::ir::Artist::Text(text) = &figure.axes[axes].children[index] else { unreachable!() };
        let mut fonts = self.fonts.lock().unwrap();
        let (mut null, mut math) = (zlc_figure::query::NullBackend, super::fonts::Math);
        let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut *fonts, &mut math).expect("scene");
        scene.text_extent(text).expect("text extent")
    }
}

/// The timing annotations of one path: its return to the start, up to two
/// waits (index ranges), up to two turns.
fn timing_labels(canonical: &[[f64; 2]], display: &[[f64; 2]]) -> Vec<(usize, String)> {
    let n = canonical.len();
    let delta: Vec<[f64; 2]> = canonical.windows(2).map(|w| [w[1][0] - w[0][0], w[1][1] - w[0][1]]).collect();
    let moving: Vec<usize> = (0..delta.len()).filter(|&i| delta[i][0] != 0.0 || delta[i][1] != 0.0).collect();
    if moving.is_empty() { return Vec::new(); }
    let mut waiting = vec![false; delta.len() + 2];
    for (i, d) in delta.iter().enumerate() { waiting[i + 1] = d[0] == 0.0 && d[1] == 0.0; }
    let edges: Vec<usize> = (0..waiting.len() - 1).filter(|&i| waiting[i + 1] != waiting[i]).collect();
    let waits: Vec<(usize, usize)> = edges.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    let mut entries: Vec<(usize, String)> = Vec::new();
    if display[0] == display[n - 1] { entries.push((n - 1, format!("f{}", n - 1))); }
    let pick = |length: usize| -> Vec<usize> {
        // np.linspace(0, length - 1, min(2, length), dtype=int)
        match length.min(2) { 0 => Vec::new(), 1 => vec![0], _ => vec![0, length - 1] }
    };
    for j in pick(waits.len()) { entries.push((waits[j].0, format!("f{}\u{2013}{}", waits[j].0, waits[j].1))); }
    if moving.len() > 1 {
        let mut turns = Vec::new();
        for k in 0..moving.len() - 1 {
            let (p, f) = (delta[moving[k]], delta[moving[k + 1]]);
            let cross = p[0] * f[1] - p[1] * f[0];
            let scale = (p[0] * p[0] + p[1] * p[1]).sqrt() * (f[0] * f[0] + f[1] * f[1]).sqrt();
            let turning = cross.abs() > 1e-7 * scale || p[0] * f[0] + p[1] * f[1] < 0.0;
            let step = moving[k + 1];
            if turning && !waits.iter().any(|&(start, end)| start <= step && step <= end) { turns.push(step); }
        }
        for j in pick(turns.len()) { entries.push((turns[j], format!("f{}", turns[j]))); }
    }
    entries
}

/// One cell's `expanded_validity()[repeat, point]` and `values[repeat, point]`
/// as booleans; a cell no segment holds measured nothing.
fn cell_statuses(snapshot: &crate::projection::Snapshot, repeat: usize, point: usize) -> (Vec<bool>, Vec<bool>) {
    use crate::projection::{model::Block, nd::{Array, Mask}};
    let schema = &snapshot.schema;
    let sizes: Vec<usize> = schema.cell.axes.iter().map(|a| a.size).collect();
    let spread = schema.mask_spread(&[]);
    let sites: usize = sizes.iter().product();
    let compact: usize = spread.iter().product();
    let mask_offset = |site: usize| {
        let (mut rest, mut offset, mut stride) = (site, 0, 1);
        for k in (0..sizes.len()).rev() {
            if spread[k] > 1 { offset += rest % sizes[k] * stride; }
            rest /= sizes[k];
            stride *= spread[k];
        }
        offset
    };
    let read = |values: &Array, mask: &Mask, cell: usize| -> (Vec<bool>, Vec<bool>) {
        ((0..sites).map(|i| mask.get(cell * compact + mask_offset(i))).collect(),
         (0..sites).map(|i| values.get_f64(cell * sites + i) != 0.0).collect())
    };
    match &snapshot.block {
        Block::Values { values, mask, .. } => read(values, mask, repeat * schema.point.size() + point),
        Block::Segments(segments) => segments.iter().filter(|s| {
            (s.origin[0]..s.origin[0] + s.extent[0]).contains(&repeat) && (s.origin[1]..s.origin[1] + s.extent[1]).contains(&point)
        }).last().map_or_else(|| (vec![false; sites], vec![false; sites]), |s| {
            read(&s.values, &s.mask, (repeat - s.origin[0]) * s.extent[1] + point - s.origin[1])
        }),
    }
}
