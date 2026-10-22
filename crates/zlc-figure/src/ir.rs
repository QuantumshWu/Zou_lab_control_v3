//! The resolved artist tree a figure is drawn from.
//!
//! Semantic level: the properties matplotlib's artist `draw` methods READ
//! (colours already resolved to RGBA, dash patterns already scaled by the
//! linewidth as matplotlib stores them, fonts with their resolved files,
//! data arrays, tick locations with their formatted labels). Never draw
//! calls, never callables. Everything matplotlib COMPUTES at draw time --
//! `apply_aspect`, tick placement, label/offset/title positions, spine
//! extents, text layout, image resampling, scalar mapping -- is computed
//! from this by the draw code.
//!
//! Transforms and boxes form one DAG ([`Node`]) whose leaves are the live
//! boxes -- an axes' active position, its view limits, the figure's inch box,
//! its dpi -- so a box that moves at draw time (apply_aspect) moves every
//! transform built on it, as in matplotlib. Matrices are evaluated with
//! matplotlib's operation order ([`crate::transform`]).
//!
//! The wire form (JSON) stores floats as IEEE-754 bits; see [`crate::wire`].
//! Every object of it is closed (`deny_unknown_fields`): a field this port
//! does not know is an error, never dropped.
use serde::{Deserialize, Serialize};

use crate::wire::{self, arr, NdArray};

/// Index into [`Figure::nodes`].
pub type NodeId = u32;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Figure {
    pub version: u32,
    #[serde(with = "wire::f")]
    pub dpi: f64,
    /// `figure.bbox_inches` width and height (its origin is (0, 0)).
    #[serde(with = "wire::fa")]
    pub size_inches: [f64; 2],
    pub visible: bool,
    pub suppress_composite: Option<bool>,
    pub rc: Rc,
    /// The figure background (`figure.patch`, a Rectangle).
    pub patch: Patch,
    /// `figure.axes`, in order; [`Artist::Axes`] refers to these by index.
    pub axes: Vec<Axes>,
    /// `figure.get_children()` without the patch, in that order (the draw
    /// sorts them by zorder, stably).
    pub children: Vec<Artist>,
    pub trans_figure: NodeId,
    pub figure_bbox: NodeId,
    pub nodes: Vec<Node>,
}

/// The rcParams of the draw (zlc's style layered over the user's live ones).
/// The draw reads the path, chunk, text, hatch and `mathtext.default` ones;
/// `image.resample`, `_internal.classic_mode` and `path.snap` are already
/// resolved into the artists, and a font's `math_fontfamily` stands for
/// `mathtext.fontset`. The mathtext font families (`mathtext.rm` .. `bfit`,
/// `mathtext.fallback`) are read only by the fontsets that use them, which
/// [`crate::mathtext::NativeMathText`] checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rc {
    #[serde(rename = "path.simplify")]
    pub path_simplify: bool,
    #[serde(rename = "path.simplify_threshold", with = "wire::f")]
    pub path_simplify_threshold: f64,
    #[serde(rename = "agg.path.chunksize")]
    pub agg_path_chunksize: i64,
    #[serde(rename = "text.hinting")]
    pub text_hinting: String,
    #[serde(rename = "text.hinting_factor")]
    pub text_hinting_factor: i64,
    #[serde(rename = "text.kerning_factor")]
    pub text_kerning_factor: i64,
    #[serde(rename = "text.antialiased")]
    pub text_antialiased: bool,
    #[serde(rename = "mathtext.fontset")]
    pub mathtext_fontset: String,
    #[serde(rename = "mathtext.fallback")]
    pub mathtext_fallback: Option<String>,
    #[serde(rename = "mathtext.default")]
    pub mathtext_default: String,
    #[serde(rename = "mathtext.rm")]
    pub mathtext_rm: String,
    #[serde(rename = "mathtext.it")]
    pub mathtext_it: String,
    #[serde(rename = "mathtext.bf")]
    pub mathtext_bf: String,
    #[serde(rename = "mathtext.sf")]
    pub mathtext_sf: String,
    #[serde(rename = "mathtext.tt")]
    pub mathtext_tt: String,
    #[serde(rename = "mathtext.cal")]
    pub mathtext_cal: String,
    #[serde(rename = "mathtext.bfit")]
    pub mathtext_bfit: String,
    #[serde(rename = "image.resample")]
    pub image_resample: bool,
    #[serde(rename = "_internal.classic_mode")]
    pub classic_mode: bool,
    #[serde(rename = "path.snap")]
    pub path_snap: bool,
    #[serde(rename = "hatch.color", with = "wire::fa")]
    pub hatch_color: [f64; 4],
    #[serde(rename = "hatch.linewidth", with = "wire::f")]
    pub hatch_linewidth: f64,
}

/// One transform or bbox node of the DAG.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Node {
    Transform(TransformNode),
    Bbox(BboxNode),
}

/// The tagged enums' variants are all structs (an empty one for no fields),
/// so `deny_unknown_fields` reaches every variant (a unit variant of an
/// internally tagged enum would ignore extra fields).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case", deny_unknown_fields)]
pub enum TransformNode {
    Identity {},
    /// `figure.dpi_scale_trans`: `Affine2D().scale(dpi)`.
    DpiScale {},
    /// A frozen `Affine2D`: `[a, b, c, d, e, f]` of `[[a, b, c], [d, e, f], [0, 0, 1]]`.
    Affine {
        #[serde(with = "wire::fa")]
        m: [f64; 6],
    },
    /// `a + b` (apply `a`, then `b`).
    Composite { a: NodeId, b: NodeId },
    /// `blended_transform_factory(x, y)`; `x_eq_y` is matplotlib's `x == y`.
    Blended { x: NodeId, y: NodeId, x_eq_y: bool },
    BboxTo { bbox: NodeId },
    BboxFrom { bbox: NodeId },
    /// `matplotlib.scale.LogTransform` (1-D, non-affine).
    Log {
        #[serde(with = "wire::f")]
        base: f64,
        clip: bool,
    },
    /// zlc's FuncTransform `axis_space` over a nonuniform coordinate table
    /// (1-D, non-affine).
    Piecewise {
        #[serde(with = "wire::vf")]
        coordinates: Vec<f64>,
    },
    ScaledTranslation {
        #[serde(with = "wire::f")]
        xt: f64,
        #[serde(with = "wire::f")]
        yt: f64,
        scale: NodeId,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "b", rename_all = "snake_case", deny_unknown_fields)]
pub enum BboxNode {
    /// The ACTIVE position of `figure.axes[axes]` (`axes._position`), set by
    /// `apply_aspect` at draw time.
    AxesPosition { axes: usize },
    /// `figure.axes[axes].viewLim`.
    ViewLim { axes: usize },
    /// `figure.bbox_inches`.
    FigureInches {},
    Transformed { bbox: NodeId, transform: NodeId },
    /// A plain `Bbox` by its points `[x0, y0, x1, y1]`.
    Literal {
        #[serde(with = "wire::fa")]
        p: [f64; 4],
    },
}

/// What every artist carries.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtistProps {
    pub visible: bool,
    #[serde(with = "wire::f")]
    pub zorder: f64,
    #[serde(with = "wire::of")]
    pub alpha: Option<f64>,
    pub clip_on: bool,
    pub clip_box: Option<NodeId>,
    /// `Artist.get_transform(artist)` (for patches, without the patch transform).
    pub transform: Option<NodeId>,
    pub snap: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Artist {
    /// `figure.axes[index]`.
    Axes { index: usize },
    Text(Text),
    Line(Line2D),
    Patch(Patch),
    Collection(Collection),
    QuadMesh(QuadMesh),
    Image(Image),
    /// A figure artist that draws its texts in order (zlc's `_FacetChromeLabels`).
    Group(Group),
}

impl Artist {
    pub fn props(&self) -> Option<&ArtistProps> {
        match self {
            Artist::Axes { .. } => None,
            Artist::Text(a) => Some(&a.artist),
            Artist::Line(a) => Some(&a.artist),
            Artist::Patch(a) => Some(&a.artist),
            Artist::Collection(a) => Some(&a.artist),
            Artist::QuadMesh(a) => Some(&a.artist),
            Artist::Image(a) => Some(&a.artist),
            Artist::Group(a) => Some(&a.artist),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapStyle { Butt, Round, Projecting }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinStyle { Miter, Round, Bevel }

/// `(offset, on-off sequence or None)`, in points, as matplotlib stores it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dash(
    #[serde(with = "wire::f")] pub f64,
    #[serde(with = "wire::ovf")] pub Option<Vec<f64>>,
);

/// A colour spec resolved by `to_rgba`; `none` remembers the 'none' spelling,
/// which ignores an alpha override.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Color {
    #[serde(with = "wire::fa")]
    pub rgba: [f64; 4],
    pub none: bool,
}

/// A `Path`; its codes, when it has them, are one per vertex (a load error otherwise).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "PathWire")]
pub struct Path {
    #[serde(with = "arr::rows")]
    pub vertices: Vec<[f64; 2]>,
    #[serde(with = "arr::ou8s")]
    pub codes: Option<Vec<u8>>,
    pub should_simplify: bool,
    #[serde(with = "wire::f")]
    pub simplify_threshold: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathWire {
    #[serde(with = "arr::rows")]
    vertices: Vec<[f64; 2]>,
    #[serde(with = "arr::ou8s")]
    codes: Option<Vec<u8>>,
    should_simplify: bool,
    #[serde(with = "wire::f")]
    simplify_threshold: f64,
}

impl TryFrom<PathWire> for Path {
    type Error = String;
    fn try_from(w: PathWire) -> Result<Path, String> {
        if let Some(codes) = &w.codes {
            if codes.len() != w.vertices.len() {
                return Err(format!("a path of {} vertices with {} codes", w.vertices.len(), codes.len()));
            }
        }
        Ok(Path { vertices: w.vertices, codes: w.codes, should_simplify: w.should_simplify,
                  simplify_threshold: w.simplify_threshold })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FontWeight { Name(String), Number(i64) }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FontStretch { Name(String), Number(i64) }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Font {
    pub family: Vec<String>,
    pub style: String,
    pub variant: String,
    pub weight: FontWeight,
    pub stretch: FontStretch,
    /// Size in points.
    #[serde(with = "wire::f")]
    pub size: f64,
    pub math_fontfamily: String,
    /// The fonts `font_manager._find_fonts_by_props` resolves (file names):
    /// the first is the FT2Font, the rest its fallbacks.
    pub files: Vec<String>,
    /// `findfont(props)`: mathtext's default font.
    pub file: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HAlign { Left, Center, Right }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VAlign { Top, Bottom, Center, Baseline, CenterBaseline }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotationMode { Default, Anchor }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Text {
    #[serde(flatten)]
    pub artist: ArtistProps,
    #[serde(with = "wire::f")]
    pub x: f64,
    #[serde(with = "wire::f")]
    pub y: f64,
    pub text: String,
    #[serde(with = "wire::fa")]
    pub color: [f64; 4],
    pub font: Font,
    pub ha: HAlign,
    pub va: VAlign,
    pub ma: Option<HAlign>,
    /// Degrees (`get_rotation()`).
    #[serde(with = "wire::f")]
    pub rotation: f64,
    pub rotation_mode: RotationMode,
    #[serde(with = "wire::f")]
    pub linespacing: f64,
    pub antialiased: bool,
    pub parse_math: bool,
    /// zlc's pulse `SpanLabel`: drawn only when the span is wide enough on screen.
    pub span_label: Option<SpanLabel>,
    /// `text.get_bbox_patch()` (a FancyBboxPatch).
    pub bbox_patch: Option<Box<Patch>>,
    /// An `Annotation` (no arrow): its transform is set at draw time from `xy`.
    pub annotation: Option<AnnotationIr>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationIr {
    #[serde(with = "wire::fa")]
    pub xy: [f64; 2],
    pub xycoords: String,
    pub anncoords: String,
    pub annotation_clip: Option<bool>,
    pub axes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanLabel {
    #[serde(with = "wire::fa")]
    pub span: [f64; 2],
    #[serde(with = "wire::f")]
    pub pad_pt: f64,
}

/// zlc's pulse `RepeatBracketSide`: a loop bracket side whose data the draw computes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bracket {
    #[serde(with = "wire::f")]
    pub start: f64,
    #[serde(with = "wire::f")]
    pub stop: f64,
    pub at_start: bool,
    #[serde(with = "wire::f")]
    pub y_low: f64,
    #[serde(with = "wire::f")]
    pub y_high: f64,
    #[serde(with = "wire::f")]
    pub lift_pt: f64,
    #[serde(with = "wire::f")]
    pub foot_fraction: f64,
    #[serde(with = "wire::f")]
    pub max_foot_fraction: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MarkerId { Code(i64), Name(String) }

/// A resolved `MarkerStyle` (its fill style is in the path, the alt path
/// and the line's resolved face colour).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Marker {
    pub marker: MarkerId,
    pub path: Path,
    /// `marker.get_transform()` as `[a, b, c, d, e, f]`.
    #[serde(with = "wire::fa")]
    pub transform: [f64; 6],
    pub alt: bool,
    pub joinstyle: JoinStyle,
    pub capstyle: CapStyle,
    #[serde(with = "wire::of")]
    pub snap_threshold: Option<f64>,
}

/// A path's `should_simplify` and `simplify_threshold`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Simplify {
    pub should_simplify: bool,
    #[serde(with = "wire::f")]
    pub threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Line2D {
    #[serde(flatten)]
    pub artist: ArtistProps,
    /// The points the draw reads: a subsliced line's `_xy` (sliced at draw
    /// time), any other's cached `_transformed_path` source -- the recache's
    /// `_path`, a copy of `_xy` that an in-place change of `_xy` since does
    /// not reach.
    #[serde(with = "arr::rows")]
    pub xy: Vec<[f64; 2]>,
    /// `_subslice and axes` as the last recache left it: the draw strokes only
    /// the view's span of the line. Stale when the scale, the transform or
    /// `markevery` changed since (none of them recaches), as in matplotlib.
    pub subslice: bool,
    /// `_x_filled` when it is not `_x` itself (NaN x interpolated), for a
    /// subsliced line.
    #[serde(with = "arr::of64s")]
    pub x_filled: Option<Vec<f64>>,
    /// Not subsliced: the simplification of the paths the line's cached
    /// `_transformed_path` hands the draw -- the stroke's
    /// (`get_transformed_path_and_affine`) and the points'
    /// (`get_transformed_points_and_affine`): the recached `_path`'s, or a
    /// new Path's where the transform built one, under the rcParams of the
    /// draw that cached it. None for a subsliced line (new paths each draw).
    pub stroke_simplify: Option<Simplify>,
    pub points_simplify: Option<Simplify>,
    pub linestyle: String,
    #[serde(with = "wire::f")]
    pub linewidth: f64,
    /// `_dash_pattern` (already scaled by the linewidth).
    pub dash_pattern: Dash,
    pub dash_capstyle: CapStyle,
    pub dash_joinstyle: JoinStyle,
    pub solid_capstyle: CapStyle,
    pub solid_joinstyle: JoinStyle,
    pub antialiased: bool,
    pub color: Color,
    pub marker: Option<Marker>,
    #[serde(with = "wire::f")]
    pub markersize: f64,
    #[serde(with = "wire::f")]
    pub markeredgewidth: f64,
    /// `get_markeredgecolor()` resolved ('auto' resolves to the line colour).
    pub markeredgecolor: Color,
    pub mec_auto: bool,
    /// `_get_markerfacecolor()` resolved (markers with an alt path are refused).
    pub markerfacecolor: Color,
    /// `get_markerfacecolor()` is 'none'.
    pub mfc_none: bool,
    pub markevery: Option<MarkEvery>,
    /// Index of `line.axes` in `figure.axes`.
    pub axes: Option<usize>,
    pub bracket: Option<Bracket>,
}

/// `markevery` as `_mark_every_path` indexes the points with it (numpy
/// fancy indexing): integers (negative from the end) or a boolean mask.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MarkEvery { Indices(Vec<i64>), Mask(Vec<bool>) }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RotationPoint { Named(String), Point(#[serde(with = "wire::fa")] [f64; 2]) }

/// A patch's shape, under `shape` (a flattened enum would keep its fields
/// from `deny_unknown_fields`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "patch_type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PatchShape {
    Rectangle {
        #[serde(with = "wire::f")]
        x0: f64,
        #[serde(with = "wire::f")]
        y0: f64,
        #[serde(with = "wire::f")]
        width: f64,
        #[serde(with = "wire::f")]
        height: f64,
        #[serde(with = "wire::f")]
        angle: f64,
        rotation_point: RotationPoint,
        /// The simplification of its path, `Path.unit_rectangle()`: a
        /// class-level cache keeping the rcParams of its making.
        path_simplify: Simplify,
    },
    /// A text's `FancyBboxPatch`: its box is set by the text's draw.
    FancyBbox {
        boxstyle: BoxStyle,
        #[serde(with = "wire::f")]
        mutation_aspect: f64,
    },
    /// `Ellipse`: the unit circle through scale, rotation and translation.
    Ellipse {
        #[serde(with = "wire::fa")]
        center: [f64; 2],
        #[serde(with = "wire::f")]
        width: f64,
        #[serde(with = "wire::f")]
        height: f64,
        #[serde(with = "wire::f")]
        angle: f64,
        #[serde(with = "wire::f")]
        aspect_ratio_correction: f64,
        /// The simplification of its path, `Path.unit_circle()` (cached alike).
        path_simplify: Simplify,
    },
    /// PathPatch / Polygon: the path in the artist transform's input space.
    Path { path: Path },
    /// A 'line' Spine at outward 0; its varying coordinate follows the view limits.
    Spine {
        spine_type: String,
        path: Path,
        #[serde(with = "wire::ofa")]
        bounds: Option<[f64; 2]>,
        axis: Option<String>,
    },
    /// A colorbar's outline (`_ColorbarSpine`): a closed path in axes coordinates.
    ColorbarSpine { path: Path },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    #[serde(flatten)]
    pub artist: ArtistProps,
    /// `_facecolor`/`_edgecolor` (alpha already applied).
    #[serde(with = "wire::fa")]
    pub facecolor: [f64; 4],
    #[serde(with = "wire::fa")]
    pub edgecolor: [f64; 4],
    #[serde(with = "wire::f")]
    pub linewidth: f64,
    pub linestyle: String,
    pub dash_pattern: Dash,
    pub capstyle: CapStyle,
    pub joinstyle: JoinStyle,
    pub antialiased: bool,
    pub hatch: Option<Hatch>,
    #[serde(with = "wire::fa")]
    pub hatch_color: [f64; 4],
    #[serde(with = "wire::f")]
    pub hatch_linewidth: f64,
    pub shape: PatchShape,
}

/// A patch's hatch: the pattern, and the simplification of the path its
/// draw hands Agg -- `Path.hatch(pattern, 6.0)`, lru-cached, which keeps the
/// rcParams of its first making in the process.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hatch {
    pub pattern: String,
    pub path_simplify: Simplify,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxStyle {
    /// "Round" or "Circle".
    pub name: String,
    #[serde(with = "wire::f")]
    pub pad: f64,
    #[serde(with = "wire::of")]
    pub rounding_size: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedSpine {
    pub name: String,
    pub patch: Patch,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collection {
    #[serde(flatten)]
    pub artist: ArtistProps,
    /// The matplotlib class (`PolyCollection`, `PathCollection`,
    /// `LineCollection`, zlc's `_SegmentBuffered`/`_BarBuffered`).
    pub cls: String,
    /// `get_paths()` at draw time.
    pub paths: Vec<Path>,
    #[serde(with = "arr::rows")]
    pub offsets: Vec<[f64; 2]>,
    pub offset_transform: NodeId,
    #[serde(with = "arr::rows")]
    pub facecolors: Vec<[f64; 4]>,
    /// `get_edgecolor()` ('face' already resolved).
    #[serde(with = "arr::rows")]
    pub edgecolors: Vec<[f64; 4]>,
    #[serde(with = "arr::f64s")]
    pub linewidths: Vec<f64>,
    /// `_linestyles` (scaled dash patterns).
    pub linestyles: Vec<Dash>,
    #[serde(with = "arr::bools")]
    pub antialiaseds: Vec<bool>,
    pub joinstyle: Option<JoinStyle>,
    pub capstyle: Option<CapStyle>,
    /// `get_array() is not None`: a scalar-mapped collection (refused; its
    /// colours would come from the mapping at draw time).
    pub mapped: bool,
    /// `_CollectionWithSizes`: the areas its draw turns into transforms at the
    /// figure dpi (with `factor`); otherwise None and `transforms` holds
    /// `get_transforms()`.
    #[serde(with = "arr::of64s")]
    pub sizes: Option<Vec<f64>>,
    #[serde(with = "wire::of")]
    pub factor: Option<f64>,
    #[serde(with = "arr::omat3s")]
    pub transforms: Option<Vec<[[f64; 3]; 3]>>,
    /// `EllipseCollection`: the half widths/heights, angles (radians) and units
    /// its draw turns into transforms.
    pub ellipses: Option<Ellipses>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ellipses {
    #[serde(with = "arr::f64s")]
    pub widths: Vec<f64>,
    #[serde(with = "arr::f64s")]
    pub heights: Vec<f64>,
    #[serde(with = "arr::f64s")]
    pub angles: Vec<f64>,
    pub units: String,
    pub axes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Colormap {
    pub name: String,
    #[serde(rename = "N")]
    pub n: usize,
    /// `_lut`: N colours, then under, over, bad.
    #[serde(with = "arr::rows")]
    pub lut: Vec<[f64; 4]>,
    pub i_under: usize,
    pub i_over: usize,
    pub i_bad: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Norm {
    pub class: String,
    #[serde(with = "wire::of")]
    pub vmin: Option<f64>,
    #[serde(with = "wire::of")]
    pub vmax: Option<f64>,
    pub clip: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuadMesh {
    #[serde(flatten)]
    pub artist: ArtistProps,
    pub cls: String,
    /// (rows + 1, columns + 1, 2).
    pub coordinates: NdArray,
    #[serde(with = "arr::rows")]
    pub offsets: Vec<[f64; 2]>,
    pub offset_transform: NodeId,
    #[serde(with = "arr::f64s")]
    pub linewidths: Vec<f64>,
    pub antialiased: bool,
    pub array: Option<NdArray>,
    pub array_mask: Option<NdArray>,
    pub cmap: Colormap,
    pub norm: Norm,
    pub face_is_mapped: bool,
    pub edge_is_mapped: bool,
    #[serde(with = "arr::orows")]
    pub facecolors: Option<Vec<[f64; 4]>>,
    #[serde(with = "arr::orows")]
    pub edgecolors: Option<Vec<[f64; 4]>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    #[serde(flatten)]
    pub artist: ArtistProps,
    /// `_A`'s data (2-D scalar, or (M, N, 3|4) colour).
    pub data: NdArray,
    /// The mask when it is per element.
    pub mask: Option<NdArray>,
    #[serde(with = "wire::fa")]
    pub extent: [f64; 4],
    pub origin: String,
    /// Only 'nearest' is drawn (resample, filternorm and filterrad, which
    /// only the filtered interpolations read, are not carried).
    pub interpolation: String,
    pub interpolation_stage: String,
    pub cmap: Colormap,
    pub norm: Norm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    #[serde(flatten)]
    pub artist: ArtistProps,
    pub cls: String,
    pub texts: Vec<Text>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TickTemplate {
    /// `XTick` or `YTick`.
    pub cls: String,
    pub visible: bool,
    /// `_size` and `_tickdir`, which pad a spine's extent (the tick's pad
    /// and width are in its labels' transforms and its lines' markers).
    #[serde(with = "wire::f")]
    pub size: f64,
    pub tickdir: String,
    pub tick1line: Line2D,
    pub tick2line: Line2D,
    pub gridline: Line2D,
    pub label1: Text,
    pub label2: Text,
}

/// One axis' major or minor ticks: the locator's locations, the formatter's
/// labels, and the tick every one of them is drawn as.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ticks {
    #[serde(with = "arr::f64s")]
    pub locs: Vec<f64>,
    pub labels: Vec<String>,
    pub tick: Option<TickTemplate>,
    /// Every tick's own template, when the ticks are not alike (zlc colours
    /// some ticks individually); None when `tick` serves them all.
    pub per_tick: Option<Vec<TickTemplate>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axis {
    /// "x" or "y".
    pub name: String,
    pub visible: bool,
    #[serde(with = "wire::f")]
    pub zorder: f64,
    pub label: Text,
    pub label_position: String,
    #[serde(with = "wire::f")]
    pub labelpad: f64,
    pub autolabelpos: bool,
    pub offset_text: Text,
    /// The offset text is pinned where `offset_text` says (zlc's
    /// SmartOffsetFormatter re-places it after matplotlib's placement).
    pub offset_text_pinned: bool,
    /// XAxis `_tick_position` ('bottom' unless set); None for a y axis.
    pub tick_position: Option<String>,
    /// The scale transform (identity for linear; the extractor admits the
    /// linear, log and zlc's function scales).
    pub transform: NodeId,
    /// Whether this axis is drawn (its ticks below are only meaningful then).
    pub drawn: bool,
    pub major: Option<Ticks>,
    pub minor: Option<Ticks>,
    /// The major formatter's offset string, as the draw sets it.
    pub offset_string: Option<String>,
    /// The ticks a spine of this axis pads its extent by
    /// (`Spine.get_window_extent`): the first major and the first minor tick
    /// of the drawn ticks' SET intersection with the axis' ticks, in set
    /// order (object hashes), as indices into the drawn ticks.
    pub spine_ticks: [Option<usize>; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Aspect { Auto(String), Value(#[serde(with = "wire::f")] f64) }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Anchor { Named(String), Point(#[serde(with = "wire::fa")] [f64; 2]) }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axes {
    pub index: usize,
    pub visible: bool,
    #[serde(with = "wire::f")]
    pub zorder: f64,
    #[serde(with = "wire::fa")]
    pub position_original: [f64; 4],
    /// The active position before the draw (apply_aspect recomputes it).
    #[serde(with = "wire::fa")]
    pub position_active: [f64; 4],
    pub aspect: Aspect,
    pub adjustable: String,
    pub anchor: Anchor,
    #[serde(with = "wire::of")]
    pub box_aspect: Option<f64>,
    /// "matplotlib": `Axes.apply_aspect`; "position": zlc's override that
    /// makes the original position active as is.
    pub apply_aspect: String,
    /// None, or "colorbar_neither" (a colorbar locator that returns the
    /// original position).
    pub axes_locator: Option<String>,
    #[serde(with = "wire::fa")]
    pub view_lim: [f64; 4],
    pub axison: bool,
    pub frameon: bool,
    pub patch: Patch,
    pub spines: Vec<NamedSpine>,
    pub xaxis: Axis,
    pub yaxis: Axis,
    /// title, _left_title, _right_title.
    pub titles: Vec<Text>,
    pub autotitlepos: Option<bool>,
    /// `axes._children`, in order.
    pub children: Vec<Artist>,
    pub trans_data: NodeId,
    pub trans_axes: NodeId,
    pub bbox: NodeId,
}
