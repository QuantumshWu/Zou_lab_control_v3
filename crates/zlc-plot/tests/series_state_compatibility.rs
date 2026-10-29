//! A series key is meaningful within its source. Implicit live continuity,
//! explicit saved state and an explicit unlock have different contracts.
use std::sync::Arc;

use zlc_core::{DType, schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema}};
use zlc_plot::configuration::{CurvePlot, Interaction, InteractionUpdate, PlotLabels, PlotSchema, PlotSpec,
    Presentation, PresentationUpdate, SeriesReadout, SeriesTarget};
use zlc_plot::configuration::fit::EmptyFitLayer;
use zlc_plot::configuration::resolve::{ConfigurationUpdates, ResolveRequest, ResolvedConfiguration, resolve_configuration};
use zlc_plot::kinds::{AxisRef, Reduction};
use zlc_plot::layout::{Kind, SurfaceOptions, resolve_surface};
use zlc_plot::projection::nd::{Array, Scalar};
use zlc_plot::projection::output::{AxisValue, CurveData, CurveSeries, Payload, QuantityArray};
use zlc_plot::render::core::Renderer;
use zlc_plot::units::resolve_unit;

fn axis(id: &str, count: usize) -> Arc<Axis> {
    Axis::new(id.into(), id.into(), "coordinate".into(), Coordinates::Index { origin: 0, len: count },
        None, None, None, None).unwrap()
}

fn target() -> SeriesTarget {
    SeriesTarget { key: vec![["cell_data".into(), "probe.group".into(), "1".into()]], facet_index: None }
}

fn readout(mode: &str) -> SeriesReadout {
    SeriesReadout { target: target(), label: "group 1".into(), mode: mode.into() }
}

fn authored(mode: &str) -> ConfigurationUpdates {
    ConfigurationUpdates {
        interaction: Some(InteractionUpdate { series_lock: Some((mode == "locked").then(target)) }),
        presentation: Some(PresentationUpdate { series_readout: Some(Some(readout(mode))) }),
        ..Default::default()
    }
}

fn base(mode: &str) -> Arc<ResolvedConfiguration> {
    let repeat = Domain::dense(vec![1], vec![axis("probe.repeat", 1)]).unwrap();
    let point = Domain::dense(vec![1], vec![axis("probe.point", 1)]).unwrap();
    let cell = Domain::dense(vec![3, 3], vec![axis("probe.group", 3), axis("probe.x", 3)]).unwrap();
    let schema = PlotSchema::new(DatasetSchema::new(repeat, point, cell,
        ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap());
    let spec = PlotSpec::Curve(CurvePlot::new(AxisRef::cell_data("probe.x").unwrap(), Some(AxisRef::cell_data("probe.group").unwrap()),
        Reduction::Mean, PlotLabels::default(), vec![], vec![]).unwrap());
    resolve_configuration(ResolveRequest { schema: Some(schema), spec: Some(spec), fit: Some(Arc::new(EmptyFitLayer)),
        configuration_updates: Some(authored(mode)), ..Default::default() }).unwrap()
}

fn rendered(configuration: &ResolvedConfiguration) -> (Interaction, Presentation) {
    let style = Arc::new(configuration.defaults.style.clone());
    let plan = resolve_surface(Some("2x2"), Kind::Curve, &SurfaceOptions::default(), &configuration.defaults.layout, &style.fonts).unwrap();
    let mut renderer = Renderer::new(configuration.spec.clone(), plan, style);
    let unit = resolve_unit("1", None).unwrap();
    let quantity = |values: Vec<f64>| QuantityArray::new(Array::vector(values), unit.clone(), unit.clone(), String::new());
    let group = AxisRef::cell_data("probe.group").unwrap();
    let series = (0..3).map(|index| CurveSeries {
        x: quantity(vec![0.0, 1.0, 2.0]), y: quantity(vec![index as f64; 3]), valid: Array::full(true, vec![3]),
        sem: None, x_labels: None,
        group_key: vec![AxisValue { axis: group.clone(), index: Some(index), canonical: Scalar::Int(index as i128),
            display: Scalar::Int(index as i128), label: format!("group {index}") }],
        label: format!("group {index}"),
    }).collect();
    let payload = Arc::new(Payload::Curve(CurveData { revision: 1, generation: "source-state-test".into(),
        x_ref: Some(AxisRef::cell_data("probe.x").unwrap()), group_by: vec![group], series, x_role: None }));
    renderer.last_payload = Some(payload.clone());
    renderer.last_state = Some(Arc::new(configuration.display_state.clone()));
    let axes = renderer.primary_axes();
    renderer.update_curve(axes, &payload, &configuration.display_state, "curve", None, None, true);
    renderer.restore_series_interaction(&configuration.interaction, Some(&configuration.presentation));
    (renderer.series_interaction(), renderer.series_presentation())
}

#[test]
fn live_frames_keep_series_state_but_another_source_does_not_inherit_it() {
    for mode in ["locked", "hover"] {
        let base = base(mode);
        let same = resolve_configuration(ResolveRequest { schema: base.schema.clone(), base: Some(&base), ..Default::default() }).unwrap();
        assert_eq!(same.interaction, base.interaction, "same-source {mode}");
        assert_eq!(same.presentation, base.presentation, "same-source {mode}");
        assert_eq!(rendered(&same).1.series_readout.unwrap().mode.as_ref(), mode);

        let other = resolve_configuration(ResolveRequest { schema: base.schema.clone(), base: Some(&base), different_source: true,
            ..Default::default() }).unwrap();
        assert_eq!(other.interaction, Interaction::default(), "different-source {mode}");
        assert_eq!(other.presentation, Presentation::default(), "different-source {mode}");
        assert_eq!(other.display_state.interaction, other.interaction);
        assert_eq!(rendered(&other), (Interaction::default(), Presentation::default()));
    }
}

#[test]
fn explicitly_authored_series_state_still_restores_on_another_source() {
    for mode in ["locked", "hover"] {
        let base = base(mode);
        let typed = resolve_configuration(ResolveRequest { schema: base.schema.clone(), base: Some(&base), different_source: true,
            configuration_updates: Some(authored(mode)), ..Default::default() }).unwrap();
        assert_eq!(typed.interaction, base.interaction);
        assert_eq!(typed.presentation, base.presentation);
        assert_eq!(rendered(&typed).1.series_readout.unwrap().mode.as_ref(), mode);

        let wire = base.target().unwrap();
        let restored = resolve_configuration(ResolveRequest { schema: base.schema.clone(), base: Some(&base), target: Some(&wire),
            different_source: true, ..Default::default() }).unwrap();
        assert_eq!(restored.interaction, base.interaction);
        assert_eq!(restored.presentation, base.presentation);
        assert_eq!(rendered(&restored).1.series_readout.unwrap().mode.as_ref(), mode);
    }
}

#[test]
fn explicit_unlock_cannot_be_resurrected_by_the_previous_locked_readout() {
    let base = base("locked");
    let unlocked = resolve_configuration(ResolveRequest { schema: base.schema.clone(), base: Some(&base),
        configuration_updates: Some(ConfigurationUpdates {
            interaction: Some(InteractionUpdate { series_lock: Some(None) }), ..Default::default()
        }), ..Default::default() }).unwrap();
    assert!(unlocked.interaction.series_lock.is_none());
    assert!(unlocked.presentation.series_readout.is_none(), "the stale locked readout must not restore the cleared lock");
    assert_eq!(rendered(&unlocked), (Interaction::default(), Presentation::default()));
}

#[test]
fn explicit_unlock_can_publish_an_explicit_hover_without_turning_it_into_a_lock() {
    let base = base("locked");
    let hovering = resolve_configuration(ResolveRequest { schema: base.schema.clone(), base: Some(&base),
        configuration_updates: Some(authored("hover")), ..Default::default() }).unwrap();
    assert!(hovering.interaction.series_lock.is_none());
    assert_eq!(hovering.presentation.series_readout.as_ref().unwrap().mode.as_ref(), "hover");
    let (interaction, presentation) = rendered(&hovering);
    assert!(interaction.series_lock.is_none());
    assert_eq!(presentation.series_readout.as_ref().unwrap().mode.as_ref(), "hover");
}
