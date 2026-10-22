//! Every golden case of the scientific layout, compared bit for bit.
use serde_json::Value;
use zlc_plot::layout::{resolve_surface, AxesPlan, FacetTopology, Kind, LayoutConfig, SurfaceOptions};
use zlc_plot::style::FontStyle;

fn float(value: &Value) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }
fn optional(value: &Value) -> Option<f64> { (!value.is_null()).then(|| float(value)) }

fn axes(plans: &[AxesPlan]) -> Value {
    Value::Array(plans.iter().map(|a| serde_json::json!({
        "role": a.role,
        "box": [a.r#box.left.to_bits(), a.r#box.top.to_bits(), a.r#box.right.to_bits(), a.r#box.bottom.to_bits()],
        "cell": a.cell_index,
        "room": [a.room.left.to_bits(), a.room.right.to_bits(), a.room.top.to_bits(), a.room.bottom.to_bits()],
    })).collect())
}

#[test]
fn resolve_surface_matches_the_scientific_layout() {
    let cases: Value = serde_json::from_str(include_str!("data/layout.json")).unwrap();
    let (layout, fonts) = (LayoutConfig::default(), FontStyle::default());
    let mut compared = 0;
    for case in cases.as_array().unwrap() {
        let o = &case["options"];
        let options = SurfaceOptions {
            facet_topology: o["cells"].as_u64().map(|cells| FacetTopology {
                cell_count: cells as u32, cell_height_over_width: optional(&o["cell_aspect"]) }),
            device_pixel_ratio: optional(&o["dpr"]),
            export_scale: optional(&o["export"]),
            rolling_side_distribution: o["rolling"].as_bool(),
            image_height_over_width: optional(&o["aspect"]),
            image_scene: o["scene"].as_bool().unwrap(),
            image_side_distribution: o["side"].as_bool(),
        };
        let kind = Kind::parse(case["kind"].as_str().unwrap()).unwrap();
        let result = resolve_surface(case["preset"].as_str(), kind, &options, &layout, &fonts);
        let expected = &case["result"];
        if expected.get("error").is_some() {
            assert!(result.is_err(), "{case}: expected an error, got {result:?}");
            compared += 1;
            continue;
        }
        let plan = result.unwrap_or_else(|error| panic!("{case}: {error}"));
        let mut got = serde_json::json!({
            "preset": plan.preset,
            "logical": [plan.logical_size.0, plan.logical_size.1],
            "raster": [plan.raster_size.0, plan.raster_size.1],
            "inches": [plan.figure_size_inches.0.to_bits(), plan.figure_size_inches.1.to_bits()],
            "logical_dpi": plan.logical_dpi.to_bits(), "dpi": plan.dpi.to_bits(),
            "dpr": plan.device_pixel_ratio.to_bits(), "export": plan.export_scale.to_bits(),
            "axes": axes(&plan.axes),
            "shape": plan.facet_shape.map(|(r, c)| vec![r, c]),
            "focus": plan.facet_focus_axes.as_deref().map(axes),
            "rolling": plan.rolling_side_distribution,
            "aspect": plan.image_height_over_width.map(f64::to_bits),
        });
        got["typography"] = match &plan.facet_typography {
            None => Value::Null,
            Some(t) => serde_json::json!({"tier": t.tier, "scale": t.scale.to_bits(), "title": t.cell_title_pt.to_bits(),
                "tick": t.tick_pt.to_bits(), "outer": t.outer_axis_label_pt.to_bits(),
                "title_min": t.cell_title_min_pt.to_bits(), "title_width": t.cell_title_max_width_pt.to_bits()}),
        };
        assert_eq!(&got, expected, "case {}", case["options"]);
        compared += 1;
    }
    assert_eq!(compared, cases.as_array().unwrap().len());
}
