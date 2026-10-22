//! Matplotlib artists as their constructors leave them (Text, Line2D,
//! PolyCollection, Rectangle), resolved into IR values.
use zlc_figure::ir::{self, ArtistProps, CapStyle, Color, Dash, HAlign, JoinStyle, NodeId, RotationMode, VAlign};

use super::color;
use super::rc::Rc;
use crate::render::fonts::font_files;

/// rcParams['path.simplify_threshold'] as matplotlibrc spells it.
pub const SIMPLIFY_THRESHOLD: f64 = 0.111111111111;

pub fn props(zorder: f64, transform: Option<NodeId>) -> ArtistProps {
    ArtistProps { visible: true, zorder, alpha: None, clip_on: true, clip_box: None, transform, snap: None }
}

pub fn colour(spec: &str) -> Color {
    Color { rgba: color::rgba(spec), none: spec.eq_ignore_ascii_case("none") }
}

/// `FontProperties` resolved for the IR.
pub fn font(rc: &Rc, size: f64) -> ir::Font { font_of(rc, size, &rc.font_family) }

/// A font of explicit `family` entries; a generic family resolves through
/// the rc's sans-serif list (`findfont` per entry).
pub fn font_of(rc: &Rc, size: f64, family: &[String]) -> ir::Font {
    let files: Vec<String> = family.iter().filter_map(|name| match name.as_str() {
        "sans-serif" | "sans" | "sans serif" => font_files(&rc.font_family).into_iter().next(),
        other => crate::render::fonts::family_file(other).map(str::to_owned),
    }).collect();
    ir::Font {
        family: family.to_vec(),
        style: "normal".into(),
        variant: "normal".into(),
        weight: ir::FontWeight::Name(rc.font_weight.clone()),
        stretch: ir::FontStretch::Name("normal".into()),
        size,
        math_fontfamily: rc.math_fontfamily.clone(),
        file: files.first().cloned().unwrap_or_default(),
        files,
    }
}

/// `Text(x, y, text, fontsize=size)` as matplotlib constructs it.
pub fn text(rc: &Rc, x: f64, y: f64, content: &str, size: f64, transform: NodeId) -> ir::Text {
    ir::Text {
        artist: props(3.0, Some(transform)),
        x,
        y,
        text: content.to_owned(),
        color: color::rgba(&rc.text_color),
        font: font(rc, size),
        ha: HAlign::Left,
        va: VAlign::Baseline,
        ma: None,
        rotation: 0.0,
        rotation_mode: RotationMode::Default,
        linespacing: 1.2,
        antialiased: rc.ir.text_antialiased,
        parse_math: true,
        span_label: None,
        bbox_patch: None,
        annotation: None,
    }
}

pub fn path(vertices: Vec<[f64; 2]>, codes: Option<Vec<u8>>, should_simplify: bool) -> ir::Path {
    ir::Path { vertices, codes, should_simplify, simplify_threshold: SIMPLIFY_THRESHOLD }
}

/// What `Path.__init__` makes of `n` straight vertices under the rcParams:
/// simplified when `path.simplify`, a positive threshold and at least 128.
pub fn simplify(rc: &Rc, n: usize) -> ir::Simplify {
    let threshold = rc.ir.path_simplify_threshold;
    ir::Simplify { should_simplify: threshold > 0.0 && rc.ir.path_simplify && n >= 128, threshold }
}

/// `_path.is_sorted_and_has_non_nan(x)`: non-decreasing over its non-NaN
/// values, of which there is at least one.
fn sorted_with_non_nan(x: &[f64]) -> bool {
    let mut last: Option<f64> = None;
    for &v in x {
        if v.is_nan() { continue; }
        if last.is_some_and(|l| v < l) { return false; }
        last = Some(v);
    }
    last.is_some()
}

/// `np.interp(x, xp, fp)` for increasing `xp` (numpy's formula, NaN retry included).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] { return fp[0]; }
    if x >= xp[n - 1] { return fp[n - 1]; }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if xp[j] == x { return fp[j]; }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut value = slope * (x - xp[j]) + fp[j];
    if value.is_nan() {
        value = slope * (x - xp[j + 1]) + fp[j + 1];
        if value.is_nan() && fp[j] == fp[j + 1] { value = fp[j]; }
    }
    value
}

/// `Line2D.recache` for a line on a rectilinear, linear-x axes through
/// `transData` (`on_data`): whether it subslices (more than 1000 sorted x,
/// clipped, no markevery), its NaN-filled x then, and otherwise the
/// simplification of its cached path.
pub fn recache(rc: &Rc, line: &mut ir::Line2D, on_data: bool) {
    let x: Vec<f64> = line.xy.iter().map(|p| p[0]).collect();
    let subslice = line.axes.is_some() && on_data && x.len() > 1000 && sorted_with_non_nan(&x) && line.markevery.is_none()
        && line.artist.clip_on;
    line.subslice = subslice;
    line.x_filled = None;
    line.stroke_simplify = None;
    line.points_simplify = None;
    if subslice {
        if x.iter().any(|v| v.is_nan()) {
            let (xp, fp): (Vec<f64>, Vec<f64>) = x.iter().enumerate().filter(|(_, v)| !v.is_nan()).map(|(i, v)| (i as f64, *v)).unzip();
            line.x_filled = Some(x.iter().enumerate().map(|(i, &v)| if v.is_nan() { interp(i as f64, &xp, &fp) } else { v }).collect());
        }
    } else {
        let simplified = simplify(rc, line.xy.len());
        line.stroke_simplify = Some(simplified);
        line.points_simplify = Some(simplified);
    }
}

/// A marker as `MarkerStyle(marker)` resolves it (the subset zlc draws).
pub fn marker(kind: &str) -> Option<ir::Marker> {
    let (id, vertices, transform, _filled, snap) = match kind {
        "None" | "none" | "" | " " => return None,
        // TICKLEFT / TICKRIGHT: `_tickhoriz_path`, scaled (-1, 1) / (1, 1).
        "0" => (ir::MarkerId::Code(0), vec![[0.0, 0.0], [1.0, 0.0]], [-1.0, -0.0, -0.0, 0.0, 1.0, 0.0], false, Some(1.0)),
        "1" => (ir::MarkerId::Code(1), vec![[0.0, 0.0], [1.0, 0.0]], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], false, Some(1.0)),
        // TICKUP / TICKDOWN: `_tickvert_path`, scaled (1, 1) / (1, -1).
        "2" => (ir::MarkerId::Code(2), vec![[-0.0, 0.0], [-0.0, 1.0]], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], false, Some(1.0)),
        "3" => (ir::MarkerId::Code(3), vec![[-0.0, 0.0], [-0.0, 1.0]], [1.0, 0.0, 0.0, -0.0, -1.0, -0.0], false, Some(1.0)),
        // '_': `_set_vline` (`_line_marker_path`, scale 0.5) rotated by 90 degrees.
        "_" => (ir::MarkerId::Name("_".into()), vec![[0.0, -1.0], [0.0, 1.0]],
                // math.cos(math.radians(90)) * 0.5, as CPython's libm gives it.
                [3.061616997868383e-17, -0.5, 0.0, 0.5, 3.061616997868383e-17, 0.0], false, Some(1.0)),
        // 'o': `_set_circle`, the unit circle scaled by 0.5, never snapped.
        "o" => {
            let (vertices, codes) = unit_circle();
            return Some(ir::Marker {
                marker: ir::MarkerId::Name("o".into()), path: path(vertices, Some(codes), false),
                transform: [0.5, 0.0, 0.0, 0.0, 0.5, 0.0], alt: false, joinstyle: JoinStyle::Round, capstyle: CapStyle::Butt,
                snap_threshold: Some(f64::INFINITY),
            });
        }
        other => panic!("marker {other:?} is not part of the renderer's subset"),
    };
    Some(ir::Marker {
        marker: id,
        path: path(vertices, None, false),
        transform,
        alt: false,
        joinstyle: JoinStyle::Round,
        capstyle: CapStyle::Butt,
        snap_threshold: snap,
    })
}

/// `Path.unit_circle()`: four cubic Bezier quadrants (26 vertices).
pub fn unit_circle() -> (Vec<[f64; 2]>, Vec<u8>) {
    const MAGIC: f64 = 0.2652031;
    let sqrthalf = 0.5f64.sqrt();
    let magic45 = sqrthalf * MAGIC;
    let vertices = vec![
        [0.0, -1.0], [MAGIC, -1.0], [sqrthalf - magic45, -sqrthalf - magic45], [sqrthalf, -sqrthalf],
        [sqrthalf + magic45, -sqrthalf + magic45], [1.0, -MAGIC], [1.0, 0.0], [1.0, MAGIC],
        [sqrthalf + magic45, sqrthalf - magic45], [sqrthalf, sqrthalf], [sqrthalf - magic45, sqrthalf + magic45], [MAGIC, 1.0],
        [0.0, 1.0], [-MAGIC, 1.0], [-sqrthalf + magic45, sqrthalf + magic45], [-sqrthalf, sqrthalf],
        [-sqrthalf - magic45, sqrthalf - magic45], [-1.0, MAGIC], [-1.0, 0.0], [-1.0, -MAGIC],
        [-sqrthalf - magic45, -sqrthalf + magic45], [-sqrthalf, -sqrthalf], [-sqrthalf + magic45, -sqrthalf - magic45], [-MAGIC, -1.0],
        [0.0, -1.0], [0.0, -1.0],
    ];
    let mut codes = vec![4u8; 26];
    codes[0] = 1;
    codes[25] = 79;
    (vertices, codes)
}

/// A scatter marker's path, `MarkerStyle(m).get_path().transformed(get_transform())`.
pub fn marker_path(marker: &str) -> ir::Path {
    let (vertices, codes, m) = match marker {
        // `_set_circle`: the unit circle scaled by 0.5.
        "o" => { let (v, c) = unit_circle(); (v, c, [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]) }
        // `_set_square`: the unit rectangle translated by -0.5.
        "s" => (vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], vec![1, 2, 2, 2, 79], [1.0, 0.0, -0.5, 0.0, 1.0, -0.5]),
        other => panic!("scatter marker {other:?} is not part of the renderer's subset"),
    };
    let affine = zlc_figure::transform::M3::from6(m);
    let vertices = vertices.iter().map(|v| { let (x, y) = affine.apply(v[0], v[1]); [x, y] }).collect();
    path(vertices, Some(codes), false)
}

/// `_get_dash_pattern(style)` scaled by the linewidth (`_scale_dashes`).
pub fn dash(rc: &Rc, linestyle: &str, linewidth: f64) -> Dash {
    let pattern: Option<Vec<f64>> = match linestyle {
        "-" | "solid" | "None" | "none" | "" | " " => None,
        "--" | "dashed" => Some(rc.lines_dashed_pattern.to_vec()),
        "-." | "dashdot" => Some(rc.lines_dashdot_pattern.to_vec()),
        ":" | "dotted" => Some(rc.lines_dotted_pattern.to_vec()),
        other => panic!("linestyle {other:?}"),
    };
    match pattern {
        None => Dash(0.0, None),
        Some(pattern) => {
            Dash(0.0, Some(pattern.iter().map(|v| v * linewidth).collect()))
        }
    }
}

/// matplotlib's stored spelling of a linestyle (`ls_mapper_r`).
pub fn linestyle_name(linestyle: &str) -> &'static str {
    match linestyle {
        "-" | "solid" => "-",
        "--" | "dashed" => "--",
        "-." | "dashdot" => "-.",
        ":" | "dotted" => ":",
        _ => "None",
    }
}

/// `Line2D(x, y, color=..., linewidth=..., ...)` before any axes adds it.
#[derive(Clone, Debug)]
pub struct LineSpec<'a> {
    pub color: &'a str,
    pub linewidth: f64,
    pub linestyle: &'a str,
    pub alpha: Option<f64>,
    pub marker: &'a str,
    pub markersize: f64,
    pub markeredgewidth: f64,
    /// None: 'auto' (the line colour).
    pub markeredgecolor: Option<&'a str>,
    /// None: the line colour.
    pub markerfacecolor: Option<&'a str>,
    pub zorder: f64,
}

/// A Line2D as its constructor and a recache off any axes leave it (see
/// [`recache`] for one on an axes).
pub fn line(rc: &Rc, spec: &LineSpec<'_>, xy: Vec<[f64; 2]>, transform: NodeId) -> ir::Line2D {
    let colour = colour(spec.color);
    let edge = spec.markeredgecolor.map(self::colour).unwrap_or(colour);
    let linestyle = linestyle_name(spec.linestyle);
    let mut artist = props(spec.zorder, Some(transform));
    artist.alpha = spec.alpha;
    let simplified = simplify(rc, xy.len());
    ir::Line2D {
        artist,
        xy,
        subslice: false,
        x_filled: None,
        stroke_simplify: Some(simplified),
        points_simplify: Some(simplified),
        linestyle: linestyle.into(),
        linewidth: spec.linewidth,
        dash_pattern: dash(rc, linestyle, spec.linewidth),
        dash_capstyle: CapStyle::Butt,
        dash_joinstyle: JoinStyle::Round,
        solid_capstyle: CapStyle::Projecting,
        solid_joinstyle: JoinStyle::Round,
        antialiased: true,
        color: colour,
        marker: marker(spec.marker),
        markersize: spec.markersize,
        markeredgewidth: spec.markeredgewidth,
        markeredgecolor: edge,
        mec_auto: spec.markeredgecolor.is_none(),
        markerfacecolor: spec.markerfacecolor.map(self::colour).unwrap_or(colour),
        mfc_none: false,
        markevery: None,
        axes: None,
        bracket: None,
    }
}

/// A Rectangle patch at the unit box (figure and axes backgrounds).
pub fn rectangle(facecolor: [f64; 4], edgecolor: [f64; 4], linewidth: f64, antialiased: bool, transform: NodeId) -> ir::Patch {
    ir::Patch {
        artist: props(1.0, Some(transform)),
        facecolor,
        edgecolor,
        linewidth,
        linestyle: "solid".into(),
        dash_pattern: Dash(0.0, None),
        capstyle: CapStyle::Butt,
        joinstyle: JoinStyle::Miter,
        antialiased,
        hatch: None,
        // `_set_edgecolor`: an explicit edge colour is also the hatch colour.
        hatch_color: edgecolor,
        hatch_linewidth: 1.0,
        shape: ir::PatchShape::Rectangle {
            x0: 0.0, y0: 0.0, width: 1.0, height: 1.0, angle: 0.0,
            rotation_point: ir::RotationPoint::Named("xy".into()),
            // `Path.unit_rectangle()`: five vertices, never simplified.
            path_simplify: ir::Simplify { should_simplify: false, threshold: SIMPLIFY_THRESHOLD },
        },
    }
}
