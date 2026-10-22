//! Render helpers against the real zlc_plot.rendering functions: box
//! quantization, image geometry, envelopes, limits, labels and axis scales,
//! values bit for bit.
use serde_json::Value;
use zlc_plot::kinds::RelimMode;
use zlc_plot::render::{envelope, geometry, labels, limits, scale::AxisScale};

fn f(value: &Value) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }
fn fs(value: &Value) -> Vec<f64> { value.as_array().unwrap().iter().map(f).collect() }
fn bits(values: &[f64]) -> Vec<u64> { values.iter().map(|v| v.to_bits()).collect() }
fn want_bits(value: &Value) -> Vec<u64> { value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect() }
fn same(a: f64, b: f64) -> bool { a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) }
fn same_all(a: &[f64], b: &[f64]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(*x, *y)) }
fn opt_pair(value: &Value) -> Option<(f64, f64)> { (!value.is_null()).then(|| { let v = fs(value); (v[0], v[1]) }) }

fn scale_of(kind: &Value, lattice: &Value) -> AxisScale {
    match kind.as_str().unwrap() { "linear" => AxisScale::Linear, "log" => AxisScale::Log, _ => AxisScale::Samples(fs(lattice).into()) }
}

#[test]
fn render_helpers_match_the_renderer() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/render_helpers.json")).unwrap();
    let data: Value = serde_json::from_str(&text).unwrap();
    for c in data["exact_box"].as_array().unwrap() {
        let (o, s, e) = geometry::exact_box_fractions(c[0].as_i64().unwrap(), c[1].as_i64().unwrap(), f(&c[2]));
        assert_eq!((o.to_bits(), s.to_bits(), e), (c[3].as_u64().unwrap(), c[4].as_u64().unwrap(), c[5].as_bool().unwrap()), "exact {c}");
    }
    for c in data["box_on_aspect"].as_array().unwrap() {
        let got = geometry::box_on_aspect(c[0].as_i64().unwrap(), c[1].as_i64().unwrap(), f(&c[2]));
        let want = c[3].as_array().map(|p| (p[0].as_i64().unwrap(), p[1].as_i64().unwrap()));
        assert_eq!(got, want, "aspect {c}");
    }
    for c in data["quantized"].as_array().unwrap() {
        let b = fs(&c[0]);
        let ratio = (!c[3].is_null()).then(|| f(&c[3]));
        let (snapped, exact) = geometry::pixel_quantized_bounds([b[0], b[1], b[2], b[3]], f(&c[1]), f(&c[2]), ratio,
                                                               c[4].as_bool().unwrap(), c[5].as_str().unwrap());
        assert_eq!(bits(&snapped), want_bits(&c[6]), "quantized {c}");
        assert_eq!(exact, c[7].as_bool().unwrap(), "quantized exact {c}");
    }
    for c in data["coordinate_scale"].as_array().unwrap() {
        let got = geometry::image_coordinate_scale(&fs(&c[0])).ok();
        match &c[1] {
            Value::Null => assert!(got.is_none(), "scale {c}"),
            Value::String(name) => assert_eq!(got, Some(if name == "linear" { AxisScale::Linear } else { AxisScale::Log }), "scale {c}"),
            lattice => assert_eq!(got, Some(AxisScale::Samples(fs(lattice).into())), "scale {c}"),
        }
    }
    for c in data["extent"].as_array().unwrap() {
        let (x, y) = (fs(&c[0]), fs(&c[1]));
        let (sx, sy) = (geometry::image_coordinate_scale(&x).unwrap(), geometry::image_coordinate_scale(&y).unwrap());
        let e = geometry::centers_extent(&x, &y, (&sx, &sy));
        assert_eq!(bits(&[e.0, e.1, e.2, e.3]), want_bits(&c[2]), "extent {c}");
        let aspect = geometry::image_cell_aspect(&x, &y, (&sx, &sy));
        assert_eq!(aspect.map(f64::to_bits), c[3].as_u64(), "aspect {c}");
        let span = geometry::image_axis_span(&x);
        assert_eq!(span.map(f64::to_bits), c[4].as_u64(), "span {c}");
    }
    for c in data["nearest"].as_array().unwrap() {
        let (view, source) = (fs(&c[2]), fs(&c[3]));
        let got = geometry::view_nearest_map(c[0].as_u64().unwrap() as usize, c[1].as_u64().unwrap() as usize, (view[0], view[1]), (source[0], source[1]));
        let want = c[4].as_array().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize,
                                            r[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect::<Vec<_>>()));
        assert_eq!(got, want, "nearest {c}");
    }
    for c in data["destination"].as_array().unwrap() {
        let (bb, e, xl, yl) = (fs(&c[0]), fs(&c[1]), fs(&c[2]), fs(&c[3]));
        let got = geometry::image_destination_rect([bb[0], bb[1], bb[2], bb[3]], (e[0], e[1], e[2], e[3]), (xl[0], xl[1]), (yl[0], yl[1]));
        let want = c[4].as_array().map(|r| (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(), r[2].as_i64().unwrap(), r[3].as_i64().unwrap()));
        assert_eq!(got, want, "destination {c}");
    }
    for c in data["square"].as_array().unwrap() {
        let e = fs(&c[0]);
        let got = geometry::square_image_limits((e[0], e[1], e[2], e[3]), f(&c[1])).ok().map(|r| bits(&[r.0, r.1, r.2, r.3]));
        assert_eq!(got, c[2].as_array().map(|_| want_bits(&c[2])), "square {c}");
    }
    for c in data["histogram"].as_array().unwrap() {
        let got: Vec<f64> = geometry::histogram_vertices(&fs(&c[0]), &fs(&c[1])).unwrap().iter().flat_map(|bar| bar.iter().flatten().copied()).collect();
        assert_eq!(bits(&got), want_bits(&c[2]), "histogram");
    }
    for (index, c) in data["envelope"].as_array().unwrap().iter().enumerate() {
        let (x, y) = (fs(&c[0]), fs(&c[1]));
        let columns = envelope::pixel_columns(f(&c[2]));
        assert_eq!(columns as u64, c[4].as_u64().unwrap(), "columns {index}");
        let window = fs(&c[3]);
        let (dx, dy) = envelope::thinned_to_columns(&x, &y, (window[0], window[1]), columns);
        assert!(same_all(&dx, &fs(&c[5])) && same_all(&dy, &fs(&c[6])), "envelope {index}: {} vs {} points", dx.len(), fs(&c[5]).len());
        let (ix, iy) = envelope::bounded_isolated_points(&x, &y, (window[0], window[1]), columns);
        assert!(same_all(&ix, &fs(&c[7])) && same_all(&iy, &fs(&c[8])), "isolated {index}");
    }
    for c in data["data_limits"].as_array().unwrap() {
        let values = fs(&c[0]);
        assert_eq!(limits::curve_x_limits(&values).map(|r| bits(&[r.0, r.1])), c[1].as_array().map(|_| want_bits(&c[1])), "curve x {c}");
        assert_eq!(limits::data_limits(&values).map(|r| bits(&[r.0, r.1])), c[2].as_array().map(|_| want_bits(&c[2])), "data {c}");
    }
    for c in data["autoscaled"].as_array().unwrap() {
        let range = fs(&c[0]);
        let got = limits::autoscaled_limits((range[0], range[1]), opt_pair(&c[1]), f(&c[2]), f(&c[3]), c[4].as_bool().unwrap(), c[5].as_bool().unwrap());
        assert_eq!(bits(&[got.0, got.1]), want_bits(&c[6]), "autoscaled {c}");
    }
    for c in data["display_limits"].as_array().unwrap() {
        let mode = RelimMode::parse(c[0].as_str().unwrap()).unwrap();
        let authored = ((!c[2].is_null()).then(|| f(&c[2])), (!c[3].is_null()).then(|| f(&c[3])));
        let got = limits::select_display_limits(mode, opt_pair(&c[1]), authored, c[4].as_bool().unwrap()).ok().map(|r| bits(&[r.0, r.1]));
        assert_eq!(got, c[5].as_array().map(|_| want_bits(&c[5])), "display limits {c}");
    }
    for c in data["quantity_label"].as_array().unwrap() {
        let got = labels::quantity_label(c[0].as_str(), c[1].as_str(), "value", c[2].as_str());
        assert_eq!(got, c[3].as_str().unwrap(), "label {c}");
    }
    for c in data["suffix"].as_array().unwrap() {
        assert_eq!(labels::has_explicit_unit_suffix(c[0].as_str().unwrap()), c[1].as_bool().unwrap(), "suffix {c}");
    }
    for c in data["scales"].as_array().unwrap() {
        let scale = scale_of(&c[0], &c[1]);
        let values = fs(&c[2]);
        let spaces: Vec<f64> = values.iter().map(|&v| scale.space(v)).collect();
        assert!(same_all(&spaces, &fs(&c[3])), "space {c}");
        for (space, back) in spaces.iter().zip(c[4].as_array().unwrap()) {
            if let Some(back) = back.as_u64() { assert!(same(scale.value(*space), f64::from_bits(back)), "value {c}"); }
        }
        let p = fs(&c[5]);
        assert!(same(scale.interpolate(p[0], p[1], p[2]), f(&c[6])), "interpolate {c}");
        assert!(same(scale.fraction_of(values[0], p[0], p[1]), f(&c[7])), "fraction {c}");
        assert!(same(scale.placed(values[2], (p[0], p[1])), f(&c[8])), "placed {c}");
        assert!(same(scale.midpoint(p[0], p[1]), f(&c[9])), "midpoint {c}");
    }
}

/// Error bars on an axes that kept an image cell's FuncScale (a grid whose
/// cells turned curve keeps it, as matplotlib does): `_SegmentBuffered` takes
/// each bar through transData -- the scale, then its affine -- where the IR
/// once refused (panicked on) the non-affine transData.
#[test]
fn error_bars_on_a_kept_function_scale() {
    use zlc_plot::render::mpl::{axes::{AxesModel, Child, SegmentArtist}, figure::FigureModel, rc::Rc};
    let rc = Rc::new(&zlc_plot::style::PlotStyle::default());
    let mut figure = FigureModel::new([4.0, 3.0], 100.0);
    let mut axes = AxesModel::new(&rc, [0.125, 0.11, 0.775, 0.77]);
    axes.xaxis.function = Some(vec![0.0, 1.0, 4.0, 9.0].into());
    axes.xlim = (-0.5, 3.5);
    axes.ylim = (0.0, 10.0);
    axes.add(Child::Segments(SegmentArtist {
        visible: true, zorder: 2.0, alpha: 1.0, facecolor: [0.0, 0.0, 1.0, 1.0], linewidth: 1.0, capsize: 0.0,
        segments: vec![[4.0, 2.0, 6.0]],
    }));
    figure.axes.push(axes);
    let ir = figure.to_ir(&rc);
    let zlc_figure::ir::Artist::Collection(bars) = &ir.axes[0].children[0] else { panic!("the error bars") };
    let glyph = &bars.paths[0].vertices;
    let centre = (glyph[0][0] + glyph[1][0]) * 0.5;
    let (x, _) = figure.axes[0].to_display(figure.size_inches, figure.dpi, 4.0, 2.0);
    assert!((centre - x).abs() < 1e-9, "bar centre {centre} vs transData {x}");
}
