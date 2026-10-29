//! Series selection must hit the geometry actually painted, including axes
//! scales retained by cell-kind edits and source dots shown beneath a fit.
use std::sync::Arc;

use zlc_plot::configuration::{parameter_schema_for, CurvePlot, DisplayState, HistogramPlot, Interaction,
    PlotLabels, PlotSpec, Presentation, RenderEffect, RollingPlot, SeriesReadout, SeriesTarget};
use zlc_plot::interaction::controller::SelectorSnapshot;
use zlc_plot::kinds::{AxisRef, Reduction};
use zlc_plot::layout::{resolve_surface, Kind, LayoutConfig, SurfaceOptions};
use zlc_plot::projection::nd::{Array, Scalar};
use zlc_plot::projection::output::{AxisValue, CurveData, CurveSeries, HistogramData, Payload, QuantityArray};
use zlc_plot::render::core::{Layer, Renderer};
use zlc_plot::render::fit::FitLayer;
use zlc_plot::render::mpl::axes::Child;
use zlc_plot::session::{FitOverlay, RenderFrame};
use zlc_plot::style::PlotStyle;
use zlc_plot::units::resolve_unit;

fn quantity(values: &[f64], unit: &str) -> QuantityArray {
    let unit = resolve_unit(unit, None).unwrap();
    QuantityArray::new(Array::from_vec(values.to_vec(), vec![values.len()]), unit.clone(), unit, String::new())
}

fn group(index: usize) -> Vec<AxisValue> {
    vec![AxisValue { axis: AxisRef::point("probe.group").unwrap(), index: Some(index),
        canonical: Scalar::Int(index as i128), display: Scalar::Int(index as i128), label: format!("group {index}") }]
}

fn make_renderer(spec: PlotSpec, kind: Kind, ratio: f64) -> (Renderer, Arc<DisplayState>) {
    let style = Arc::new(PlotStyle::default());
    let plan = resolve_surface(Some("2x2"), kind, &SurfaceOptions { device_pixel_ratio: Some(ratio),
        rolling_side_distribution: (kind == Kind::Rolling).then_some(false), ..Default::default() },
        &LayoutConfig::default(), &style.fonts).unwrap();
    let state = Arc::new(DisplayState { revision: 0, values: parameter_schema_for(&spec, &style).initial_values(None).unwrap(),
        changed_names: Default::default(), effects: RenderEffect::NONE, interaction: Interaction::default() });
    (Renderer::new(spec, plan, style), state)
}

fn curves(rolling: bool, ratio: f64) -> (Renderer, Arc<Payload>, Arc<DisplayState>) {
    let x_ref = AxisRef::point("probe.frequency").unwrap();
    let group_ref = AxisRef::point("probe.group").unwrap();
    let spec = if rolling {
        PlotSpec::Rolling(RollingPlot::new(Some(group_ref.clone()), Some(x_ref.clone()), Reduction::Mean, vec![]).unwrap())
    } else {
        PlotSpec::Curve(CurvePlot::new(x_ref.clone(), Some(group_ref.clone()), Reduction::Mean,
            PlotLabels::default(), vec![], vec![]).unwrap())
    };
    let (mut renderer, state) = make_renderer(spec, if rolling { Kind::Rolling } else { Kind::Curve }, ratio);
    let series = [[100.0, 200.0, 100.0], [400.0, 500.0, 400.0], [800.0, 900.0, 800.0]].into_iter().enumerate()
        .map(|(index, y)| CurveSeries { x: quantity(&[1.0, 10.0, 100.0], "MHz"), y: quantity(&y, "count"),
            valid: Array::full(true, vec![3]), sem: None, x_labels: None, group_key: group(index), label: format!("group {index}") }).collect();
    let payload = Arc::new(Payload::Curve(CurveData { revision: 0, generation: "series-hit".into(), x_ref: Some(x_ref),
        group_by: vec![group_ref], series, x_role: None }));
    renderer.last_payload = Some(payload.clone());
    renderer.last_state = Some(state.clone());
    let axes = renderer.primary_axes();
    if rolling { renderer.update_rolling(axes, &payload, &state, "rolling"); }
    else { renderer.update_curve(axes, &payload, &state, "curve", None, None, true); }
    renderer.requested_view = Some((Some((1.0, 100.0)), Some((50.0, 1000.0))));
    renderer.fig.axes[axes].xlim = (1.0, 100.0);
    renderer.fig.axes[axes].ylim = (50.0, 1000.0);
    (renderer, payload, state)
}

fn pixels(renderer: &Renderer, x: f64, y: f64) -> (f64, f64) {
    renderer.fig.axes[renderer.primary_axes()].to_display(renderer.fig.size_inches, renderer.fig.dpi, x, y)
}

fn hover(renderer: &mut Renderer, point: (f64, f64)) -> Option<String> {
    renderer.series_focus("move", Some(renderer.primary_axes()), point.0, point.1, 10.0, 6.0);
    renderer.series_presentation().series_readout.map(|readout| readout.label.to_string())
}

fn click(renderer: &mut Renderer, point: (f64, f64)) {
    let axes = Some(renderer.primary_axes());
    renderer.series_focus("press", axes, point.0, point.1, 10.0, 6.0);
    assert!(renderer.series_focus("release", axes, point.0, point.1, 10.0, 6.0));
    assert!(renderer.series_interaction().series_lock.is_some());
}

#[test]
fn curve_and_rolling_hit_scaled_lines_and_keep_lock_wheel_and_clear() {
    for rolling in [false, true] {
        for ratio in [1.0, 2.0] {
            for scale in ["linear", "log-x", "log-y", "log-both", "function-x", "function-y", "function-both"] {
                let (mut renderer, _, _) = curves(rolling, ratio);
                let axes = renderer.primary_axes();
                if matches!(scale, "log-x" | "log-both") { renderer.fig.axes[axes].xaxis.log = true; }
                if matches!(scale, "log-y" | "log-both") { renderer.fig.axes[axes].yaxis.log = true; }
                if matches!(scale, "function-x" | "function-both") {
                    renderer.fig.axes[axes].xaxis.function = Some(vec![1.0, 5.0, 10.0, 100.0].into());
                }
                if matches!(scale, "function-y" | "function-both") {
                    renderer.fig.axes[axes].yaxis.function = Some(vec![50.0, 100.0, 200.0, 1000.0].into());
                }
                // A vertex transformed by the painter, not a point on an
                // unscaled approximation of its line.
                let point = pixels(&renderer, 10.0, 200.0);
                assert_eq!(hover(&mut renderer, point).as_deref(), Some("group 0"), "rolling={rolling} dpr={ratio} {scale}");
                click(&mut renderer, point);
                assert!(renderer.series_focus_scroll(Some(axes), -1.0));
                assert_eq!(renderer.series_presentation().series_readout.unwrap().label.as_ref(), "group 1");
                renderer.series_focus("clear", None, point.0, point.1, 10.0, 6.0);
                assert!(renderer.series_interaction().series_lock.is_none());
                assert!(renderer.series_presentation().series_readout.is_none());
            }
        }
    }
}

#[test]
fn fit_source_scatter_is_selectable_only_at_its_visible_dots() {
    for rolling in [false, true] {
        let (mut renderer, payload, state) = curves(rolling, 1.0);
        let axes = renderer.primary_axes();
        let frame = RenderFrame { payload, state, effects: RenderEffect::OVERLAY, data_revision: 0,
            fit_overlays: vec![FitOverlay { group_key: group(0), ..Default::default() }].into(), fit_model_id: None,
            classifier_overlays: vec![].into(), classifier_thresholds: vec![], classifier_labels: vec![], image_overlay: None,
            selectors: SelectorSnapshot::new(&[], None).unwrap(), facet_index: None, facet_focus_index: None,
            view_limits: renderer.requested_view, presentation: None };
        FitLayer.update(&mut renderer, &frame);
        assert_eq!(renderer.fit.hidden_source_lines.len(), 1);
        let (_, line, _) = renderer.fit.hidden_source_lines[0];
        assert!(!renderer.fig.axes[axes].line(line).visible);
        let between = pixels(&renderer, 5.5, 150.0);
        assert_eq!(hover(&mut renderer, between), None, "a scatter has no connecting stroke");
        let point = pixels(&renderer, 10.0, 200.0);
        assert_eq!(hover(&mut renderer, point).as_deref(), Some("group 0"));
        click(&mut renderer, point);
        assert!(renderer.series_focus_scroll(Some(axes), -1.0));
        assert_eq!(renderer.series_presentation().series_readout.unwrap().label.as_ref(), "group 1");
        renderer.series_focus("clear", None, 0.0, 0.0, 10.0, 6.0);
        let (_, scatter) = renderer.fit.source_scatter.unwrap();
        if let Some(Child::Scatter(scatter)) = renderer.fig.axes[axes].child_mut(scatter) { scatter.visible = false; }
        assert_eq!(hover(&mut renderer, point), None, "an actually hidden scatter must stay unpickable");
    }
}

#[test]
fn histogram_bars_invert_both_axis_scales_before_selecting_a_bin() {
    for scale in ["linear", "log-x", "log-y", "function-x", "function-y"] {
        let spec = PlotSpec::Histogram(HistogramPlot { group: Some(AxisRef::point("probe.group").unwrap()), ..Default::default() });
        let (mut renderer, state) = make_renderer(spec, Kind::Histogram, 1.0);
        let data = HistogramData { revision: 0, generation: "series-hit".into(), edges: quantity(&[1.0, 10.0, 100.0], "count"),
            centers: quantity(&[5.5, 55.0], "count"), counts: Array::from_vec(vec![100i64, 0, 0, 100], vec![2, 2]),
            group_keys: vec![group(0), group(1)] };
        let axes = renderer.primary_axes();
        renderer.update_histogram(axes, &data, &state, "histogram", None, Some(((1.0, 100.0), (1.0, 1000.0))), true);
        renderer.fig.axes[axes].xaxis.log = scale == "log-x";
        renderer.fig.axes[axes].yaxis.log = scale == "log-y";
        if scale == "function-x" { renderer.fig.axes[axes].xaxis.function = Some(vec![1.0, 10.0, 100.0].into()); }
        if scale == "function-y" { renderer.fig.axes[axes].yaxis.function = Some(vec![1.0, 90.0, 1000.0].into()); }
        let above = pixels(&renderer, 55.0, 500.0);
        assert_eq!(hover(&mut renderer, above), None, "{scale}: the space above a bar is not a bar");
        let point = pixels(&renderer, 55.0, 50.0);
        assert_eq!(hover(&mut renderer, point).as_deref(), Some("group 1"), "{scale}");
        click(&mut renderer, point);
        assert!(renderer.series_focus_scroll(Some(axes), 1.0));
        assert_eq!(renderer.series_presentation().series_readout.unwrap().label.as_ref(), "group 0");
    }
}

fn target(index: usize, facet_index: Option<i128>) -> SeriesTarget {
    SeriesTarget { key: vec![["point".into(), "probe.group".into(), index.to_string().into()]], facet_index }
}

#[test]
fn a_press_cannot_lock_an_identical_group_on_another_axes() {
    let (mut renderer, payload, state) = curves(false, 1.0);
    let first = renderer.primary_axes();
    let point = pixels(&renderer, 10.0, 200.0);
    hover(&mut renderer, point); // Materialize the original native scene.
    renderer.series_focus("clear", None, 0.0, 0.0, 10.0, 6.0);
    let mut other = renderer.fig.axes[first].clone();
    other.children.clear();
    other.role = "other".into();
    other.gid = "other".into();
    let second = renderer.fig.axes.len();
    renderer.fig.axes.push(other);
    renderer.fit_overlays = 1; // Both surfaces have materialized artists.
    renderer.update_curve(second, &payload, &state, "other", Some(((1.0, 100.0), (50.0, 1000.0))), None, true);
    renderer.series_focus("press", Some(first), point.0, point.1, 10.0, 6.0);
    assert!(!renderer.series_focus("release", Some(second), point.0, point.1, 10.0, 6.0));
    assert!(renderer.series_interaction().series_lock.is_none());
    assert!(renderer.series_focus("move", Some(first), point.0, point.1, 10.0, 6.0));
    assert!(renderer.series_focus("move", Some(second), point.0, point.1, 10.0, 6.0), "same group on another axes is a different hover target");
    assert!(!renderer.series_focus("move", Some(second), point.0, point.1, 10.0, 6.0), "the hover now belongs to the second axes");
    renderer.series_focus("press", Some(first), point.0, point.1, 10.0, 6.0);
    assert!(renderer.series_focus("release", Some(first), point.0, point.1, 10.0, 6.0));
    renderer.series_focus("press", Some(second), point.0, point.1, 10.0, 6.0);
    assert!(renderer.series_focus("release", Some(second), point.0, point.1, 10.0, 6.0));
    assert!(renderer.series_interaction().series_lock.is_some(), "clicking the other axes must switch targets, not unlock the first");
    assert!(renderer.series_focus_scroll(Some(second), 1.0), "the lock and its wheel now belong to the second axes");
}

#[test]
fn same_target_restore_preserves_press_and_wheel_but_unlock_discards_remainder() {
    let (mut renderer, _, _) = curves(false, 1.0);
    let axes = Some(renderer.primary_axes());
    let point = pixels(&renderer, 10.0, 500.0);
    renderer.series_focus("press", axes, point.0, point.1, 10.0, 6.0);
    renderer.restore_series_interaction(&Interaction::default(), None);
    assert!(renderer.series_focus("release", axes, point.0, point.1, 10.0, 6.0), "a same-source frame cannot cancel an in-progress click");
    let held = renderer.series_interaction();
    assert_eq!(held.series_lock, Some(target(1, None)));
    renderer.series_focus_scroll(axes, 0.5);
    renderer.restore_series_interaction(&held, None);
    renderer.series_focus_scroll(axes, 0.5);
    assert_eq!(renderer.series_interaction().series_lock, Some(target(0, None)), "same target kept its two half-notches");
    renderer.restore_series_interaction(&held, None);
    renderer.series_focus_scroll(axes, 0.5);
    renderer.restore_series_interaction(&Interaction::default(), None);
    renderer.restore_series_interaction(&held, None);
    renderer.series_focus_scroll(axes, 0.5);
    assert_eq!(renderer.series_interaction().series_lock, Some(target(1, None)), "unlock discarded the old half-notch");
}

#[test]
fn restoring_a_different_target_invalidates_a_pending_press() {
    let (mut renderer, _, _) = curves(false, 1.0);
    let axes = Some(renderer.primary_axes());
    let point = pixels(&renderer, 10.0, 200.0);
    renderer.series_focus("press", axes, point.0, point.1, 10.0, 6.0);
    let selected = Interaction { series_lock: Some(target(2, None)) };
    renderer.restore_series_interaction(&selected, None);
    assert!(!renderer.series_focus("release", axes, point.0, point.1, 10.0, 6.0));
    assert_eq!(renderer.series_interaction().series_lock, selected.series_lock);
}

#[test]
fn a_readout_label_is_restored_only_for_the_complete_target() {
    let (mut renderer, _, _) = curves(false, 1.0);
    let selected = Interaction { series_lock: Some(target(1, None)) };
    let wrong_facet = Presentation { series_readout: Some(SeriesReadout {
        target: target(1, Some(3)), label: "another facet's stale label".into(), mode: "locked".into(),
    }) };
    renderer.restore_series_interaction(&selected, Some(&wrong_facet));
    assert_eq!(renderer.series_presentation().series_readout.unwrap().label.as_ref(), "group 1");
    let same = Presentation { series_readout: Some(SeriesReadout {
        target: target(1, None), label: "current series label".into(), mode: "locked".into(),
    }) };
    renderer.restore_series_interaction(&selected, Some(&same));
    assert_eq!(renderer.series_presentation().series_readout.unwrap().label.as_ref(), "current series label");
}
