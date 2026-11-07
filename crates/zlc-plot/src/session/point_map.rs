//! A Point Map binds its projected value plane and calibrated sites to the
//! same Image renderer used by camera panels and saved figures.
use std::sync::Arc;

use crate::configuration::{DisplayState, ImagePlot, PlotSpec, SeriesReadout, SeriesTarget, Value};
use crate::interaction::engine::AxesId;
use crate::projection::output::{Payload, PointMapData};
use crate::pyfmt::format_g;

use super::core::{PlotSession, Scene};
use super::overlay::{ImagePointOverlay, PointStatus};
use super::projected::convert;
use super::renderer::SeriesAction;

impl Scene {
    pub(super) fn render_spec(&self) -> PlotSpec {
        match (&self.spec, self.projection.payload().as_ref()) {
            (PlotSpec::PointMap(spec), Payload::PointMap(map)) => PlotSpec::Image(ImagePlot {
                x: map.image.x_ref.clone(), y: map.image.y_ref.clone(), reduction: spec.reduction,
                labels: spec.labels.clone(), scope: Vec::new(), coordinates: Vec::new(),
            }),
            _ => self.spec.clone(),
        }
    }
}

#[derive(Default)]
pub(super) struct SiteSelection {
    image: Option<(Arc<Payload>, Arc<Payload>)>,
    highlighted: Option<(Arc<ImagePointOverlay>, usize, Arc<ImagePointOverlay>)>,
    press: Option<(Arc<str>, (f64, f64), Option<SeriesTarget>)>,
}

impl SiteSelection {
    pub(super) fn image(&mut self, payload: &Arc<Payload>, map: &PointMapData) -> Arc<Payload> {
        if let Some((source, image)) = &self.image {
            if Arc::ptr_eq(source, payload) { return image.clone(); }
        }
        let image = Arc::new(Payload::Image(map.image.clone()));
        self.image = Some((payload.clone(), image.clone()));
        image
    }

    pub(super) fn overlay(&mut self, map: &PointMapData, index: Option<usize>) -> Arc<ImagePointOverlay> {
        let Some(index) = index else { self.highlighted = None; return map.overlay.clone() };
        if let Some((source, selected, overlay)) = &self.highlighted {
            if *selected == index && Arc::ptr_eq(source, &map.overlay) { return overlay.clone(); }
        }
        let mut overlay = map.overlay.as_ref().clone();
        if let Some(status) = overlay.static_statuses.as_mut().and_then(|statuses| statuses.get_mut(index)) {
            *status = PointStatus::Occupied;
        }
        let overlay = Arc::new(overlay);
        self.highlighted = Some((map.overlay.clone(), index, overlay.clone()));
        overlay
    }
}

/// Two boolean states keep their colour even when this frame contains only one of them.
/// Use the Image limit controls on the presentation copy, leaving the authored choice intact.
pub(super) fn boolean_scale(state: &mut Arc<DisplayState>, boolean: bool) {
    if boolean && state.get("relim_mode").and_then(Value::as_str) != Some("fixed") {
        let values = &mut Arc::make_mut(state).values;
        values.insert("relim_mode", Value::text("fixed"));
        values.insert("color_min", Value::Float(0.0));
        values.insert("color_max", Value::Float(1.0));
    }
}

fn target_at(map: &PointMapData, index: usize) -> SeriesTarget {
    let id = &map.geometry.points[map.indices[index]].id;
    SeriesTarget { key: vec![[map.site.domain.value().into(), map.site.axis_id.as_str().into(), id.as_str().into()]], facet_index: None }
}

pub(super) fn readout(map: &PointMapData, target: &SeriesTarget, mode: &str) -> Option<(usize, SeriesReadout)> {
    let [part] = target.key.as_slice() else { return None };
    if target.facet_index.is_some() || part[0].as_ref() != map.site.domain.value() || part[1].as_ref() != map.site.axis_id { return None; }
    let index = map.indices.iter().position(|&i| map.geometry.points[i].id == part[2].as_ref())?;
    let point = &map.geometry.points[map.indices[index]];
    let value = map.values.display.get_f64(index);
    let number = if point.label.is_empty() { &point.id } else { &point.label };
    let mut label = format!("Site {number}\n");
    if map.valid.get_f64(index) == 0.0 || !value.is_finite() {
        label.push_str("invalid");
    } else if map.boolean {
        label.push_str(if value == 0.0 { "false" } else { "true" });
    } else {
        label.push_str(&format_g(value, 6));
        if let Some(sigma) = map.sigma.as_ref().map(|s| s.get_f64(index)).filter(|s| s.is_finite() && *s >= 0.0) {
            let central = map.values.canonical.get_f64(index);
            let display = |v| convert(v, &map.values.canonical_unit, &map.values.display_unit);
            let (a,b) = (display(central - sigma), display(central + sigma));
            if a.is_finite() && b.is_finite() {
                let (minus,plus) = (value - a.min(b), a.max(b) - value);
                if (minus - plus).abs() <= 1e-12 * (1.0 + plus.abs()) {
                    label.push_str(&format!(" ± {}", format_g(plus, 3)));
                } else {
                    label.push_str(&format!(" +{}/−{}", format_g(plus, 3), format_g(minus, 3)));
                }
            }
        }
        let unit = map.values.display_unit.symbol.as_str();
        if !unit.is_empty() && unit != "1" { label.push(' '); label.push_str(unit); }
    }
    Some((index, SeriesReadout { target: target.clone(), label: label.into(), mode: mode.into() }))
}

impl PlotSession {
    fn site_under_pointer(&self, map: &PointMapData, pixel: (f64,f64), radius: f64) -> Option<usize> {
        let axes = self.renderer.geometry(&self.renderer.primary_axes())?;
        if !axes.contains(pixel.0, pixel.1) { return None; }
        // Native height-bar picking resolves visibility and returns the actual carrier cell.
        let picked = self.renderer.height_bars_scene().then(|| self.renderer.height_bars_pick(pixel.0,pixel.1));
        if picked == Some(None) { return None; }
        let (left,bottom,right,top) = axes.bbox;
        let (x0,x1) = (axes.x_scale.space(axes.x_limits.0), axes.x_scale.space(axes.x_limits.1));
        let (y0,y1) = (axes.y_scale.space(axes.y_limits.0), axes.y_scale.space(axes.y_limits.1));
        let sx = (right-left)/(x1-x0);
        let sy = (top-bottom)/(y1-y0);
        let display_x = |v| convert(v, &map.image.x.canonical_unit, &map.image.x.display_unit);
        let display_y = |v| convert(v, &map.image.y.canonical_unit, &map.image.y.display_unit);
        let mut nearest = None;
        let mut best = f64::INFINITY;
        for (index, &site) in map.indices.iter().enumerate() {
            let [x,y] = map.geometry.points[site].xy;
            let distance = if let Some(Some((hit_x,hit_y))) = picked {
                let cx = convert(hit_x, &map.image.x.display_unit, &map.image.x.canonical_unit);
                let cy = convert(hit_y, &map.image.y.display_unit, &map.image.y.canonical_unit);
                let distance = (cx-x).hypot(cy-y);
                if distance > map.radius { continue; }
                distance
            } else {
                let x_px = left + (axes.x_scale.space(display_x(x))-x0)*sx;
                let y_px = bottom + (axes.y_scale.space(display_y(y))-y0)*sy;
                let ring_px = ((axes.x_scale.space(display_x(x+map.radius))-axes.x_scale.space(display_x(x)))*sx).abs();
                let distance = (x_px-pixel.0).hypot(y_px-pixel.1);
                if distance > ring_px.max(radius*self.device_pixel_ratio) { continue; }
                distance
            };
            if distance < best { nearest = Some(index); best = distance; }
        }
        nearest
    }

    pub(super) fn site_focus(&mut self, action: SeriesAction, axes: Option<&AxesId>, pixel: (f64,f64),
                             hit_radius: f64, click_radius: f64) -> bool {
        let Payload::PointMap(map) = self.scene.projection.payload().as_ref() else { return false };
        let prior = self.renderer.series_presentation().series_readout;
        let locked = prior.as_ref().is_some_and(|r| r.mode.as_ref() == "locked");
        let over_map = axes == Some(&self.renderer.primary_axes());
        let hit = if over_map && (matches!(action, SeriesAction::Press | SeriesAction::Release) || action == SeriesAction::Move && !locked) {
            self.site_under_pointer(map, pixel, hit_radius).map(|i| target_at(map,i))
        } else { None };
        let next = match action {
            SeriesAction::Press => {
                self.sites.press = over_map.then(|| (map.generation.clone(), pixel, hit));
                return false;
            }
            SeriesAction::Release => {
                let Some((generation, start, pressed)) = self.sites.press.take() else { return false };
                if !over_map || generation != map.generation || (start.0-pixel.0).hypot(start.1-pixel.1) > click_radius*self.device_pixel_ratio {
                    return false;
                }
                if hit != pressed { return false; }
                hit.filter(|site| !locked || prior.as_ref().is_none_or(|r| &r.target != site))
                    .and_then(|target| readout(map,&target,"locked").map(|(_, r)| r))
            }
            SeriesAction::Move if !locked => hit.and_then(|target| readout(map,&target,"hover").map(|(_, r)| r)),
            SeriesAction::Leave | SeriesAction::Clear => {
                self.sites.press = None;
                if action == SeriesAction::Leave && locked { return false; }
                None
            }
            SeriesAction::Move => return false,
        };
        if next == prior { return false; }
        self.renderer.show_resolved_series_readout(next.as_ref());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::{RenderEffect, ValueMap};

    #[test]
    fn boolean_colours_are_stable_without_changing_authored_numeric_or_fixed_limits() {
        let authored = Arc::new(DisplayState {
            revision: 1, values: ValueMap::from_pairs([("relim_mode", Value::text("tight"))]),
            changed_names: Default::default(), effects: RenderEffect::NONE, interaction: Default::default(),
        });
        let mut rendered = authored.clone();
        boolean_scale(&mut rendered, true);
        assert_eq!(rendered.get("color_min"), Some(&Value::Float(0.0)));
        assert_eq!(rendered.get("color_max"), Some(&Value::Float(1.0)));
        assert_eq!(authored.get("relim_mode").and_then(Value::as_str), Some("tight"));
        let mut fixed = rendered.clone();
        Arc::make_mut(&mut fixed).values.insert("color_max", Value::Float(2.0));
        boolean_scale(&mut fixed, true);
        assert_eq!(fixed.get("color_max"), Some(&Value::Float(2.0)));
        let mut numeric = authored.clone();
        boolean_scale(&mut numeric, false);
        assert!(Arc::ptr_eq(&authored, &numeric));
    }
}
