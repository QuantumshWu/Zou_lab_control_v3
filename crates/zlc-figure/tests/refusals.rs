//! What matplotlib would not draw as the IR says, or what this port does not
//! implement, is refused (`Error::Unsupported` / `Error::Invalid`), never
//! drawn some other way: golden figures with one thing changed.
mod support;

use serde_json::{json, Value};
use zlc_figure::ir::{Dash, Figure, PatchShape};
use zlc_figure::Error;

/// A golden's IR and a draw of it (fonts answered from its record).
fn figure(name: &str) -> (Figure, Value) {
    zlc_figure::wire::set_blob_root(Some(support::golden_blob_root()));
    let golden = support::golden(name);
    let figure: Figure = serde_json::from_value(golden["ir"].clone()).unwrap();
    (figure, json!({"fonts": golden["fonts"]}))
}

fn draw(figure: &Figure, record: &Value) -> zlc_figure::Result<()> {
    let mut fonts = support::RecordedFonts::new(record);
    let mut math = support::RecordedMath::new(record);
    zlc_figure::draw(figure, &mut zlc_figure::record::Recorder::default(), &mut fonts, &mut math)
}

const LINE: &str = "direct.chunksize_dashed_line.json.gz";

#[derive(Clone)]
enum Step { Key(String), Index(usize) }

/// Every object of `v` with its concrete path and its kind: the path with
/// indices folded, and the variant tag of a tagged object.
fn objects(v: &Value, path: &mut Vec<Step>, kind: &str, out: &mut Vec<(String, Vec<Step>)>) {
    match v {
        Value::Object(map) => {
            let tag = ["t", "b", "kind", "patch_type"].iter().filter_map(|t| map.get(*t).and_then(Value::as_str)).next();
            let kind = match tag { Some(t) => format!("{kind}<{t}>"), None => kind.to_owned() };
            out.push((kind.clone(), path.clone()));
            for (k, x) in map {
                path.push(Step::Key(k.clone()));
                objects(x, path, &format!("{kind}.{k}"), out);
                path.pop();
            }
        }
        Value::Array(items) => {
            for (i, x) in items.iter().enumerate() {
                path.push(Step::Index(i));
                objects(x, path, &format!("{kind}[]"), out);
                path.pop();
            }
        }
        _ => {}
    }
}

fn at<'v>(v: &'v mut Value, path: &[Step]) -> &'v mut Value {
    path.iter().fold(v, |v, step| match step { Step::Key(k) => &mut v[k.as_str()], Step::Index(i) => &mut v[*i] })
}

/// The wire form is closed: in every kind of object the goldens hold (every
/// struct, every tagged variant, every wire array), a field the IR does not
/// know fails the load instead of being dropped.
#[test]
fn an_unknown_field_anywhere_fails_the_load() {
    zlc_figure::wire::set_blob_root(Some(support::golden_blob_root()));
    let mut names: Vec<String> = std::fs::read_dir(support::data_dir().join("goldens")).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    let mut seen = std::collections::BTreeSet::new();
    let mut accepted = Vec::new();
    for name in &names {
        let ir = support::golden(name)["ir"].clone();
        let mut found = Vec::new();
        objects(&ir, &mut Vec::new(), "", &mut found);
        for (kind, path) in found {
            if !seen.insert(kind.clone()) { continue; }
            let mut bad = ir.clone();
            at(&mut bad, &path).as_object_mut().unwrap().insert("zz_unknown".into(), json!(0));
            if serde_json::from_value::<Figure>(bad).is_ok() { accepted.push(format!("{name}: {kind}")); }
        }
    }
    eprintln!("{} kinds of object, each refusing an unknown field", seen.len());
    assert!(seen.len() > 100, "{}", seen.len());
    assert!(accepted.is_empty(), "unknown field accepted in {accepted:#?}");
}

/// A scalar-mapped collection (the extractor refuses it too) and a collection
/// class whose draw is not Collection.draw are refused.
#[test]
fn a_mapped_or_foreign_collection_is_refused() {
    let names: Vec<String> = std::fs::read_dir(support::data_dir().join("goldens")).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    let name = names.iter().find(|n| {
        let ir = &support::golden(n)["ir"];
        ir["axes"].as_array().unwrap().iter().any(|a| a["children"].as_array().unwrap().iter().any(|c| c["kind"] == "collection"))
    }).expect("a golden with a collection");
    for change in ["mapped", "cls"] {
        let (mut figure, record) = figure(name);
        let collection = figure.axes.iter_mut().flat_map(|a| a.children.iter_mut())
            .find_map(|c| match c { zlc_figure::ir::Artist::Collection(c) => Some(c), _ => None }).unwrap();
        if change == "mapped" { collection.mapped = true; } else { collection.cls = "TriMesh".into(); }
        let result = draw(&figure, &record);
        assert!(matches!(result, Err(Error::Unsupported(_))), "{change}: {result:?}");
    }
}

#[test]
fn the_golden_draws() {
    let (figure, record) = figure(LINE);
    draw(&figure, &record).unwrap();
}

/// With a NaN zorder `<` is no order: Python's sort and a stable sort here
/// place the artists differently, so it is refused.
#[test]
fn a_nan_zorder_is_refused() {
    let (mut figure, record) = figure(LINE);
    let line = figure.axes[0].children.iter_mut()
        .find_map(|c| match c { zlc_figure::ir::Artist::Line(l) => Some(l), _ => None }).unwrap();
    line.artist.zorder = f64::NAN;
    let result = draw(&figure, &record);
    assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}");
}

/// The ticks are what the draw's `_update_ticks` produced: an axis whose
/// update the extractor did not see has none to draw.
#[test]
fn ticks_of_an_axis_the_draw_did_not_update_are_invalid() {
    let (mut figure, record) = figure(LINE);
    figure.axes[0].xaxis.drawn = false;
    let result = draw(&figure, &record);
    assert!(matches!(result, Err(Error::Invalid(_))), "{result:?}");
}

/// An image's mask is booleans of the data's shape (`np.ma.getmaskarray`).
#[test]
fn an_image_mask_not_of_its_data_is_invalid() {
    use zlc_figure::wire::{DType, NdArray};
    let names: Vec<String> = std::fs::read_dir(support::data_dir().join("goldens")).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    let name = names.iter().find(|n| {
        let ir = &support::golden(n)["ir"];
        ir["axes"].as_array().unwrap().iter().any(|a| a["children"].as_array().unwrap().iter().any(|c| {
            c["kind"] == "image" && c["data"]["shape"].as_array().unwrap().len() == 2
        }))
    }).expect("a golden with a scalar image");
    let (figure, record) = figure(name);
    draw(&figure, &record).unwrap();
    for (dtype, transpose) in [(DType::F64, false), (DType::Bool, true)] {
        let mut changed = figure.clone();
        let image = changed.axes.iter_mut().flat_map(|a| a.children.iter_mut())
            .find_map(|c| match c { zlc_figure::ir::Artist::Image(i) => Some(i), _ => None }).unwrap();
        let mut shape = image.data.shape.clone();
        if transpose { shape.reverse(); shape[0] += 1; }
        let count: usize = shape.iter().product();
        let size = if dtype == DType::Bool { 1 } else { 8 };
        image.mask = Some(NdArray { dtype, shape, bytes: vec![0; count * size] });
        let result = draw(&changed, &record);
        assert!(matches!(result, Err(Error::Invalid(_))), "{dtype:?}: {result:?}");
    }
}

/// `Spine._adjust_location` raises for a 'line' spine of any other type,
/// with bounds as without ('circle' returns before reading them).
#[test]
fn a_spine_of_another_type_is_refused_with_or_without_bounds() {
    for bounds in [None, Some([0.25, 0.75])] {
        for kind in ["polar", "circle", "outline"] {
            let (mut figure, record) = figure(LINE);
            let PatchShape::Spine { spine_type, bounds: b, .. } = &mut figure.axes[0].spines[0].patch.shape else {
                panic!("not a spine")
            };
            *spine_type = kind.into();
            *b = bounds;
            let result = draw(&figure, &record);
            assert!(matches!(result, Err(Error::Unsupported(_))), "{kind} {bounds:?}: {result:?}");
        }
    }
}

fn first_line(figure: &mut Figure) -> &mut zlc_figure::ir::Line2D {
    figure.axes[0].children.iter_mut()
        .find_map(|c| match c { zlc_figure::ir::Artist::Line(l) => Some(l), _ => None }).unwrap()
}

/// `GraphicsContextBase.set_dashes` raises for a dash list with a negative
/// value, or a non-empty one without a positive value (a dashed line or
/// patch at linewidth 0 scales its pattern to zeros): a dashed line's stroke
/// and a patch are refused; a line that strokes nothing never sets them.
#[test]
fn dash_lists_set_dashes_rejects_are_invalid() {
    for list in [vec![0.0, 0.0], vec![0.0], vec![-2.0, 4.0], vec![f64::NAN, 0.0]] {
        let (mut figure, record) = figure(LINE);
        first_line(&mut figure).dash_pattern = Dash(0.0, Some(list.clone()));
        let result = draw(&figure, &record);
        assert!(matches!(result, Err(Error::Invalid(_))), "line {list:?}: {result:?}");
        first_line(&mut figure).linestyle = "None".into();
        draw(&figure, &record).unwrap();
        figure.axes[0].patch.dash_pattern = Dash(0.0, Some(list.clone()));
        let result = draw(&figure, &record);
        assert!(matches!(result, Err(Error::Invalid(_))), "patch {list:?}: {result:?}");
    }
    for list in [vec![], vec![0.0, 3.0], vec![f64::NAN, 2.0]] {
        let (mut figure, record) = figure(LINE);
        first_line(&mut figure).dash_pattern = Dash(0.0, Some(list));
        draw(&figure, &record).unwrap();
    }
}

/// A collection's dashes reach Agg unchecked (`draw_path_collection`): a
/// pattern Agg's dasher would loop on forever is refused there, at once.
#[test]
fn a_collection_dashed_never_to_advance_is_refused_at_once() {
    const NAME: &str = "direct.collection_paths_made_by_transform.json.gz";
    let (mut figure, _) = figure(NAME);
    let pixels: Value = serde_json::from_str(&std::fs::read_to_string(support::data_dir().join("pixels.json")).unwrap()).unwrap();
    for c in figure.axes[0].children.iter_mut() {
        if let zlc_figure::ir::Artist::Collection(c) = c { c.linestyles = vec![Dash(0.0, Some(vec![0.0, 0.0]))]; }
    }
    let started = std::time::Instant::now();
    let result = support::paint(&figure, &pixels[NAME]["canvas"], &mut support::native_fonts(), &mut support::native_math());
    assert!(matches!(&result, Err(Error::Backend(m)) if m.contains("never advances")), "{:?}", result.err());
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "{:?}", started.elapsed());
}

/// numpy raises for a markevery mask not of the points' length, or an index
/// out of range (so does `_mark_every_path`): refused.
#[test]
fn a_markevery_numpy_rejects_is_invalid() {
    use zlc_figure::ir::MarkEvery;
    for index in [MarkEvery::Mask(vec![true, false, true]), MarkEvery::Indices(vec![12]), MarkEvery::Indices(vec![-13])] {
        let (mut figure, record) = figure("direct.markevery_bool_mask.json.gz");
        first_line(&mut figure).markevery = Some(index.clone());
        let result = draw(&figure, &record);
        assert!(matches!(result, Err(Error::Invalid(_))), "{index:?}: {result:?}");
    }
}

/// A hatch path simplifies only at 128 vertices or more (`Path._update_values`):
/// a '/' hatch (14) said to simplify is no cached path matplotlib made.
#[test]
fn a_short_hatch_path_that_simplifies_is_invalid() {
    let (mut figure, record) = figure("direct.hatch_paths_keep_their_flags.json.gz");
    draw(&figure, &record).unwrap();
    let hatch = figure.axes[0].children.iter_mut().find_map(|c| match c {
        zlc_figure::ir::Artist::Patch(p) => p.hatch.as_mut().filter(|h| h.pattern == "/"),
        _ => None,
    }).unwrap();
    hatch.path_simplify.should_simplify = true;
    let result = draw(&figure, &record);
    assert!(matches!(result, Err(Error::Invalid(_))), "{result:?}");
}

/// A title the draw would place automatically is refused by the draw and by
/// the Scene's query of its extent alike.
#[test]
fn a_title_placed_automatically_is_refused_by_queries_too() {
    let (mut figure, record) = figure(LINE);
    figure.axes[0].autotitlepos = Some(true);
    figure.axes[0].titles[0].text = "title".into();
    let result = draw(&figure, &record);
    assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}");
    let mut fonts = support::RecordedFonts::new(&record);
    let mut math = support::RecordedMath::new(&record);
    let mut null = zlc_figure::query::NullBackend;
    let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut fonts, &mut math).unwrap();
    let result = scene.text_extent_at("axes/0/titles/0");
    assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}");
}

fn collection_of(ir: &mut Value) -> &mut Value {
    ir["axes"][0]["children"].as_array_mut().unwrap().iter_mut().find(|c| c["kind"] == "collection").unwrap()
}

/// The wire arrays load exactly as the extractor writes them -- f8, u1 or b1,
/// 1-d, (n, N) or (n, 3, 3), a path's codes one per vertex -- and anything
/// else fails the load instead of being converted or reshaped.
#[test]
fn a_wire_array_of_another_dtype_or_shape_fails_the_load() {
    const COLLECTION: &str = "direct.collection_paths_made_by_transform.json.gz";
    zlc_figure::wire::set_blob_root(Some(support::golden_blob_root()));
    let changes: [(&str, &str, fn(&mut Value)); 7] = [
        (LINE, "tick locations as f4", |ir| ir["axes"][0]["xaxis"]["major"]["locs"]["dtype"] = json!("f4")),
        (LINE, "tick locations 2-d", |ir| ir["axes"][0]["xaxis"]["major"]["locs"]["shape"] = json!([2, 3])),
        (LINE, "path vertices as i8", |ir| ir["axes"][0]["spines"][0]["patch"]["shape"]["path"]["vertices"]["dtype"] = json!("i8")),
        (LINE, "path vertices (4,)", |ir| ir["axes"][0]["spines"][0]["patch"]["shape"]["path"]["vertices"]["shape"] = json!([4])),
        (LINE, "one code for two vertices",
         |ir| ir["axes"][0]["spines"][0]["patch"]["shape"]["path"]["codes"] = json!({"dtype": "u1", "shape": [1], "data": [1]})),
        (COLLECTION, "transforms (0, 9)", |ir| collection_of(ir)["transforms"]["shape"] = json!([0, 9])),
        (COLLECTION, "antialiaseds as u1", |ir| collection_of(ir)["antialiaseds"]["dtype"] = json!("u1")),
    ];
    for (name, what, change) in changes {
        let mut ir = support::golden(name)["ir"].clone();
        serde_json::from_value::<Figure>(ir.clone()).unwrap();
        change(&mut ir);
        assert!(serde_json::from_value::<Figure>(ir).is_err(), "{what} loaded");
    }
}
