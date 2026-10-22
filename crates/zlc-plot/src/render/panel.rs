//! The native renderer behind the session's [`PanelRenderer`] seam.
//!
//! Curve, Rolling and Histogram surfaces are drawn by this crate's core; a spec of
//! another kind is accepted (`set_spec`, `relayout`) and answers presents with
//! a NotImplemented-class error until that kind's module is ported on top of
//! the core. Selector, colour-limit and height-bar gestures belong to the
//! overlay layers ported separately: the core keeps their state so a session
//! can drive it, and draws no overlay for them yet.
use std::sync::Arc;


use super::core::Renderer;
use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::selectors::{NumericRange, SelectorKind, SelectorState};
use crate::configuration::spec::PlotSpec;
use crate::configuration::state::{Interaction, Presentation};
use crate::interaction::engine::{AxesId, CameraAngles};
use crate::interaction::transform::CanvasSize;
use crate::layout::SurfacePlan;
use crate::projection::output::Payload;
use crate::session::renderer::{AxesGeometry, PanelRenderer, RenderFrame, RendererFactory, RgbaFront, SeriesAction};
use crate::style::PlotStyle;

impl Renderer {
    fn axes_id(&self, index: usize) -> AxesId { AxesId::from_gid(&self.fig.axes[index].gid) }

    fn axes_index(&self, id: &AxesId) -> Option<usize> {
        (0..self.fig.axes.len()).find(|&index| self.axes_id(index) == *id)
    }

    fn axes_geometry(&self, index: usize) -> AxesGeometry {
        let a = &self.fig.axes[index];
        let b = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
        AxesGeometry {
            id: self.axes_id(index),
            bbox: (b[0], b[1], b[2], b[3]),
            x_limits: a.xlim,
            y_limits: a.ylim,
            x_scale: self.axis_scale(index, true),
            y_scale: self.axis_scale(index, false),
        }
    }

    fn unsupported_kind(&self) -> PlotError {
        PlotError::new(ErrorKind::Runtime, format!("the native renderer does not draw {:?} yet", self.spec.kind()))
    }
}

/// A draw the fonts or the backend refused: the panel's error.
fn draw_error(error: zlc_figure::Error) -> PlotError { PlotError::runtime(format!("Drawing failed: {error}")) }

impl PanelRenderer for Renderer {
    fn plan(&self) -> &SurfacePlan { &self.plan }
    fn spec(&self) -> &PlotSpec { &self.spec }
    fn set_spec(&mut self, spec: PlotSpec) { Renderer::set_spec(self, spec) }

    fn relayout(&mut self, plan: SurfacePlan, facet_index: Option<usize>, facet_focus_index: Option<usize>) -> Result<()> {
        Renderer::relayout(self, plan, facet_index, facet_focus_index)
    }

    fn present(&mut self, frame: &RenderFrame, compose: bool) -> Result<()> {
        if !self.draws_kind() { return Err(self.unsupported_kind()); }
        let visible = match &*frame.payload { Payload::Facet(facet) => Some(facet.cells.len()), _ => self.is_facet().then_some(0) };
        self.check_facet_presentation(frame.facet_index, frame.facet_focus_index, visible)?;
        Renderer::present(self, frame);
        if let Some(error) = self.pulse.failure.take() { return Err(error); }
        if compose { self.compose().map_err(draw_error)?; }
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        if !self.draws_kind() { return Err(self.unsupported_kind()); }
        self.compose().map_err(draw_error)?;
        Ok(())
    }

    fn canvas_size(&self) -> CanvasSize {
        let (w, h) = self.fig.size_px();
        CanvasSize { width: w.trunc(), height: h.trunc() }
    }

    fn axes(&self, role: &str) -> Vec<AxesId> { self.axes_of(role).iter().map(|&i| self.axes_id(i)).collect() }

    fn visible_axes(&self) -> Vec<AxesGeometry> {
        (0..self.fig.axes.len()).filter(|&i| self.fig.axes[i].visible).map(|i| self.axes_geometry(i)).collect()
    }

    fn geometry(&self, axes: &AxesId) -> Option<AxesGeometry> { self.axes_index(axes).map(|i| self.axes_geometry(i)) }

    fn primary_axes(&self) -> AxesId { self.axes_id(Renderer::primary_axes(self)) }

    fn interactive_axes_at(&self, x: f64, y: f64) -> Option<AxesId> {
        Renderer::interactive_axes_at(self, x, y).map(|i| self.axes_id(i))
    }

    fn facet_index_for_axes(&self, axes: &AxesId) -> Option<usize> {
        self.axes_index(axes).and_then(|index| Renderer::facet_index_for_axes(self, index))
    }

    fn resolved_color_limits(&self) -> Option<(f64, f64)> { self.image_clim() }

    fn begin_selector_gesture(&mut self, kind: SelectorKind, compose: bool) -> Result<()> {
        Renderer::begin_selector_gesture(self, kind, compose);
        self.image.color_gesture = false;
        if compose { self.compose().map_err(draw_error)?; }
        Ok(())
    }

    fn begin_color_limit_gesture(&mut self, candidate: NumericRange) -> Result<()> {
        self.selector_gesture = None;
        self.image.color_gesture = true;
        self.image.candidate = Some(candidate);
        self.update_selectors();
        self.compose().map_err(draw_error)?;
        Ok(())
    }

    fn preview_selector(&mut self, state: &SelectorState) -> Result<bool> {
        if !Renderer::preview_selector(self, state) { return Ok(false); }
        self.compose().map_err(draw_error)?;
        Ok(true)
    }

    fn preview_color_limit_candidate(&mut self, candidate: NumericRange) -> Result<bool> {
        self.composed_generation = -1;
        if !self.image.color_gesture { return Ok(false); }
        self.image.candidate = Some(candidate);
        self.preview_color_limits(candidate.low, candidate.high)?;
        self.update_selectors();
        self.compose().map_err(draw_error)?;
        Ok(true)
    }

    fn end_selector_gesture(&mut self) {
        Renderer::end_selector_gesture(self);
        self.image.color_gesture = false;
        self.image.candidate = None;
    }

    fn set_classifier_labels(&mut self, labels: &[String]) {
        for (entry, label) in self.classifier.labels.iter_mut().zip(labels) { entry.1 = label.clone(); }
    }

    fn set_view_dragging(&mut self, axes: Option<&AxesId>) -> Result<()> {
        let index = axes.and_then(|id| self.axes_index(id));
        Renderer::set_view_dragging(self, index);
        if index.is_some() { self.compose().map_err(draw_error)?; }
        Ok(())
    }

    fn capture_gesture_background(&mut self) -> Result<()> {
        self.compose().map_err(draw_error)?;
        Ok(())
    }

    fn set_height_bars_dragging(&mut self, dragging: bool) -> Result<()> {
        let axes = if dragging { self.height_bars.axes } else { None };
        Renderer::set_view_dragging(self, axes);
        if axes.is_some() { self.compose().map_err(draw_error)?; }
        Ok(())
    }
    fn height_bars_scene(&self) -> bool { self.height_bars.active }
    fn height_bars_camera(&self) -> Option<CameraAngles> {
        self.height_bars.camera.map(|c| CameraAngles { azimuth_deg: c.azimuth_deg, elevation_deg: c.elevation_deg, zoom: c.zoom })
    }
    fn height_bars_pick(&self, x: f64, y: f64) -> Option<(f64, f64)> { Renderer::height_bars_pick(self, x, y) }

    fn series_interaction(&self) -> Interaction { Renderer::series_interaction(self) }
    fn series_presentation(&self) -> Presentation { Renderer::series_presentation(self) }

    fn series_focus(&mut self, action: SeriesAction, axes: Option<&AxesId>, x: f64, y: f64, hit_radius: f64,
                    click_radius: f64) -> Result<bool> {
        let index = axes.and_then(|id| self.axes_index(id));
        Ok(Renderer::series_focus(self, action.value(), index, x, y, hit_radius, click_radius))
    }

    fn series_focus_scroll(&mut self, axes: Option<&AxesId>, step: f64) -> Result<bool> {
        let index = axes.and_then(|id| self.axes_index(id));
        Ok(Renderer::series_focus_scroll(self, index, step))
    }

    fn raster_generation(&self) -> u64 { self.raster_generation }

    fn rgba(&mut self) -> Result<RgbaFront> {
        let fresh = self.composed_generation >= 0 && self.composed_generation as u64 == self.raster_generation;
        if !fresh || self.front.is_none() {
            if !self.draws_kind() { return Err(self.unsupported_kind()); }
            self.compose().map_err(draw_error)?;
        }
        let front = self.front.as_ref().unwrap();
        // Shared, not copied: the buffer returns to the renderer's pool when its readers drop it.
        Ok(RgbaFront { width: front.width as u32, height: front.height as u32, pixels: front.pixels.clone() })
    }

    fn save(&mut self, format: &str, dpi: f64, _restore_display: bool) -> Result<Vec<u8>> {
        if format == "png" { return self.export_png(dpi).map_err(PlotError::runtime); }
        let vector = crate::vector::Format::parse(format)
            .ok_or_else(|| PlotError::value(format!("the native renderer exports png, svg or pdf, not {format}")))?;
        self.export_vector(dpi, vector).map_err(PlotError::runtime)
    }
}

/// Makes the native renderer of a new session.
pub struct NativeRendererFactory;

impl RendererFactory for NativeRendererFactory {
    fn create(&self, _panel: Option<&str>, spec: &PlotSpec, plan: &SurfacePlan, style: &PlotStyle) -> Result<Box<dyn PanelRenderer>> {
        Ok(Box::new(Renderer::new(spec.clone(), plan.clone(), Arc::new(style.clone()))))
    }
}
