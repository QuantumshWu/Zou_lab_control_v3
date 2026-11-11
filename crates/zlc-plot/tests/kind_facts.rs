//! What the plotting owner alone says of a plot kind, and every other owner reads: which kinds
//! a board offers and in what order, which are drawn with a chosen overlay, how many shots a
//! new panel reads, and which declare a Window (a grid declares its cells').
use zlc_plot::configuration::{parameter_schema_for_kind, plot_kinds};
use zlc_plot::kinds::PlotKind;
use zlc_plot::style::PlotStyle;

#[test]
fn a_board_offers_six_kinds_in_one_order_and_two_take_an_overlay() {
    let kinds = plot_kinds();
    let mut offered: Vec<(u64, &str)> = kinds.as_array().unwrap().iter()
        .filter_map(|kind| Some((kind["board"].as_u64()?, kind["id"].as_str().unwrap()))).collect();
    offered.sort();
    assert_eq!(offered.iter().map(|(_, id)| *id).collect::<Vec<_>>(), ["image", "curve", "rolling", "histogram", "facet_grid", "point_map"]);
    assert_eq!(offered.iter().map(|(place, _)| *place).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5], "one place each");
    let fact = |id: &str, name: &str| kinds.as_array().unwrap().iter().find(|kind| kind["id"] == id).unwrap()[name].clone();
    assert!(fact("pulse_timeline", "board").is_null(), "no board adds the pulse editor's timeline");
    let overlaid: Vec<&str> = kinds.as_array().unwrap().iter().filter(|kind| kind["overlay"] == true).map(|kind| kind["id"].as_str().unwrap()).collect();
    assert_eq!(overlaid, ["image", "facet_grid"]);
    assert_eq!((fact("rolling", "default_window"), fact("curve", "default_window"), fact("facet_grid", "default_window")), (100.into(), 1.into(), 1.into()));
}

#[test]
fn a_grid_declares_the_window_its_cells_do() {
    let style = PlotStyle::default();
    let declares = |kind: PlotKind, cell: Option<PlotKind>| parameter_schema_for_kind(kind, &style, cell).unwrap().names().any(|name| name == "window");
    for kind in [PlotKind::Curve, PlotKind::Histogram, PlotKind::Rolling] { assert!(declares(kind, None), "{kind:?}"); }
    for kind in [PlotKind::Image, PlotKind::PointMap] { assert!(!declares(kind, None), "{kind:?}"); }
    assert!(declares(PlotKind::FacetGrid, Some(PlotKind::Histogram)), "a grid of histogram cells");
    assert!(declares(PlotKind::FacetGrid, Some(PlotKind::Curve)), "a grid of curve cells");
    assert!(!declares(PlotKind::FacetGrid, Some(PlotKind::Image)), "a grid of image cells");
}
