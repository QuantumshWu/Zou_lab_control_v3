//! The artists an image surface adds to matplotlib's: the `AxesImage`, a
//! colorbar's parts (the `pcolormesh` ramp, its end fills, its empty divider
//! `LineCollection` and `_ColorbarSpine` outline) -- and their IR.
use std::sync::Arc;

use zlc_figure::ir::{self, Artist};
use zlc_figure::wire::{DType, NdArray};

use super::artist;
use super::axes::{closed_path, EmitContext};
use super::color;
use super::nodes::AxesNodes;
use crate::projection::nd::{self, Array};

/// What an image artist holds (`_A`).
#[derive(Clone, Debug)]
pub enum ImagePixels {
    /// A scalar plane (any real dtype) and, for an export, its mask
    /// (`np.ma.array(values, mask=...)`).
    Scalar { values: Array, mask: Option<Arc<Vec<bool>>> },
    /// A precomposed RGBA frame (the height-bar scene), rows top first.
    Rgba { rows: usize, columns: usize, pixels: Arc<Vec<u8>> },
}

/// `axes.imshow(..., interpolation="nearest")` as zlc keeps it.
#[derive(Clone, Debug)]
pub struct ImageArtist {
    pub visible: bool,
    pub pixels: ImagePixels,
    /// (left, right, bottom, top) as `get_extent()` returns it.
    pub extent: [f64; 4],
    pub origin: &'static str,
    /// The colormap (its bad colour 'none') and the norm's limits.
    pub cmap: &'static str,
    pub clim: Option<(f64, f64)>,
    /// `transLimits + transAxes`: the picture stands on its sample lattice.
    pub lattice: bool,
}

/// The colorbar's `pcolormesh` ramp: (n + 1) boundary rows of two points,
/// n values, mapped through the norm.
#[derive(Clone, Debug)]
pub struct MeshArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    pub coordinates: Vec<[[f64; 2]; 2]>,
    pub values: Vec<f64>,
    /// `set_array` of a flat ramp (one column otherwise).
    pub flat: bool,
    pub cmap: &'static str,
    pub norm: (f64, f64),
}

/// A Rectangle on the y-axis transform (x in axes, y in data).
#[derive(Clone, Debug)]
pub struct RectArtist {
    pub visible: bool,
    pub zorder: f64,
    pub xy: (f64, f64),
    pub width: f64,
    pub height: f64,
    /// None: 'none'.
    pub facecolor: Option<[f64; 4]>,
}

/// A colorbar's divider `LineCollection` (no segments unless drawedges).
#[derive(Clone, Debug)]
pub struct LineCollectionArtist {
    pub visible: bool,
    pub zorder: f64,
    pub color: [f64; 4],
    pub linewidth: f64,
}

/// A colormap with its bad colour set to 'none', as the IR spells it.
pub fn colormap_ir(name: &str) -> ir::Colormap {
    let (floats, _, extremes) = crate::render::colormaps::table(name).unwrap_or_else(|| panic!("colormap {name:?}"));
    let mut lut: Vec<[f64; 4]> = floats.chunks(4).map(|c| [c[0], c[1], c[2], c[3]].map(f64::from_bits)).collect();
    lut.extend(extremes.iter().map(|c| c.map(f64::from_bits)));
    ir::Colormap { name: name.into(), n: 256, lut, i_under: 256, i_over: 257, i_bad: 258 }
}

/// `cmap(fraction)` for a fraction in [0, 1].
pub fn colormap_rgba(name: &str, fraction: f64) -> [f64; 4] {
    let (floats, _, _) = crate::render::colormaps::table(name).unwrap_or_else(|| panic!("colormap {name:?}"));
    let slot = ((fraction * 256.0) as usize).min(255);
    [0, 1, 2, 3].map(|k| f64::from_bits(floats[slot * 4 + k]))
}

fn norm_ir(limits: Option<(f64, f64)>) -> ir::Norm {
    ir::Norm { class: "Normalize".into(), vmin: limits.map(|l| l.0), vmax: limits.map(|l| l.1), clip: false }
}

fn wire_dtype(dtype: nd::DType) -> DType {
    match dtype {
        nd::DType::Bool => DType::Bool, nd::DType::U8 => DType::U8, nd::DType::I8 => DType::I8, nd::DType::U16 => DType::U16,
        nd::DType::I16 => DType::I16, nd::DType::U32 => DType::U32, nd::DType::I32 => DType::I32, nd::DType::U64 => DType::U64,
        nd::DType::I64 => DType::I64, nd::DType::F32 => DType::F32, nd::DType::F64 => DType::F64,
        other => panic!("an image plane of dtype {} is not drawn", other.name()),
    }
}

pub fn image_ir(ctx: &EmitContext<'_>, an: &AxesNodes, image: &ImageArtist) -> Artist {
    let mut props = artist::props(0.0, Some(if image.lattice { an.limits_axes } else { an.trans_data }));
    props.visible = image.visible;
    props.clip_box = Some(an.patch_clip);
    let (data, mask) = match &image.pixels {
        ImagePixels::Scalar { values, mask } if ctx.materialize => {
            let data = NdArray { dtype: wire_dtype(values.dtype()), shape: values.shape().to_vec(), bytes: values.bytes().to_vec() };
            let mask = mask.as_ref().map(|m| NdArray { dtype: DType::Bool, shape: values.shape().to_vec(), bytes: m.iter().map(|&b| u8::from(b)).collect() });
            (data, mask)
        }
        // Live, the kernel paints the plane: the artist's data is not drawn.
        ImagePixels::Scalar { values, .. } => (NdArray { dtype: wire_dtype(values.dtype()), shape: vec![0, 0], bytes: Vec::new() }, None),
        ImagePixels::Rgba { rows, columns, pixels } => {
            let shape = vec![*rows, *columns, 4];
            (NdArray { dtype: DType::U8, shape, bytes: pixels.to_vec() }, None)
        }
    };
    Artist::Image(ir::Image {
        artist: props,
        data,
        mask,
        extent: image.extent,
        origin: image.origin.into(),
        interpolation: "nearest".into(),
        interpolation_stage: "auto".into(),
        cmap: colormap_ir(image.cmap),
        norm: norm_ir(image.clim),
    })
}

pub fn mesh_ir(ctx: &EmitContext<'_>, an: &AxesNodes, mesh: &MeshArtist) -> Artist {
    let mut props = artist::props(mesh.zorder, Some(an.trans_data));
    props.visible = mesh.visible;
    props.alpha = mesh.alpha;
    props.clip_box = Some(an.patch_clip);
    props.snap = Some(true);
    let flat: Vec<f64> = mesh.coordinates.iter().flat_map(|row| [row[0][0], row[0][1], row[1][0], row[1][1]]).collect();
    Artist::QuadMesh(ir::QuadMesh {
        artist: props,
        cls: "QuadMesh".into(),
        coordinates: NdArray::from_f64(vec![mesh.coordinates.len(), 2, 2], &flat),
        offsets: vec![[0.0, 0.0]],
        offset_transform: ctx.figure.identity,
        linewidths: vec![ctx.rc.patch_linewidth],
        antialiased: false,
        array: Some(NdArray::from_f64(if mesh.flat { vec![mesh.values.len()] } else { vec![mesh.values.len(), 1] }, &mesh.values)),
        array_mask: None,
        cmap: colormap_ir(mesh.cmap),
        norm: norm_ir(Some(mesh.norm)),
        face_is_mapped: true,
        edge_is_mapped: false,
        facecolors: None,
        edgecolors: Some(Vec::new()),
    })
}

pub fn rect_ir(ctx: &EmitContext<'_>, an: &AxesNodes, rect: &RectArtist) -> Artist {
    let facecolor = rect.facecolor.unwrap_or([0.0; 4]);
    let mut patch = artist::rectangle(facecolor, [0.0; 4], ctx.rc.patch_linewidth, true, an.yaxis);
    patch.artist.visible = rect.visible;
    patch.artist.zorder = rect.zorder;
    patch.artist.clip_box = Some(an.patch_clip);
    patch.shape = ir::PatchShape::Rectangle {
        x0: rect.xy.0, y0: rect.xy.1, width: rect.width, height: rect.height, angle: 0.0,
        rotation_point: ir::RotationPoint::Named("xy".into()),
        path_simplify: ir::Simplify { should_simplify: false, threshold: artist::SIMPLIFY_THRESHOLD },
    };
    Artist::Patch(patch)
}

pub fn line_collection_ir(ctx: &EmitContext<'_>, an: &AxesNodes, lines: &LineCollectionArtist) -> Artist {
    let mut props = artist::props(lines.zorder, Some(an.trans_data));
    props.visible = lines.visible;
    props.clip_on = false;
    props.clip_box = Some(an.patch_clip);
    Artist::Collection(ir::Collection {
        artist: props,
        cls: "LineCollection".into(),
        paths: Vec::new(),
        offsets: vec![[0.0, 0.0]],
        offset_transform: ctx.figure.identity,
        facecolors: Vec::new(),
        edgecolors: vec![lines.color],
        linewidths: vec![lines.linewidth],
        linestyles: vec![ir::Dash(0.0, None)],
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: None,
        factor: None,
        transforms: Some(Vec::new()),
        ellipses: None,
    })
}

/// The `_ColorbarSpine` outline, after the four spines in `axes.spines`.
pub fn outline_spine(ctx: &EmitContext<'_>, an: &AxesNodes, outline: &[[f64; 2]]) -> ir::NamedSpine {
    let rc = ctx.rc;
    let mut patch = artist::rectangle([0.0; 4], color::rgba(&rc.axes_edgecolor), rc.axes_linewidth, true, an.trans_axes);
    patch.artist.zorder = 2.5;
    patch.capstyle = ir::CapStyle::Projecting;
    let mut path = closed_path(outline[..outline.len() - 1].to_vec());
    path.should_simplify = false;
    patch.shape = ir::PatchShape::ColorbarSpine { path };
    ir::NamedSpine { name: "outline".into(), patch }
}

/// `axes.plot(x, y, transform=axes.transAxes, ...)`: a Line2D in axes fractions.
#[derive(Clone, Debug)]
pub struct PolylineArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    pub color: [f64; 4],
    pub linewidth: f64,
    pub capstyle: ir::CapStyle,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

pub fn polyline_ir(ctx: &EmitContext<'_>, an: &AxesNodes, index: usize, line: &PolylineArtist) -> Artist {
    // The colour is an RGBA tuple: set below on every field that reads it.
    let spec = artist::LineSpec {
        color: "black", linewidth: line.linewidth, linestyle: "-", alpha: line.alpha, marker: "None",
        markersize: ctx.rc.lines_markersize, markeredgewidth: ctx.rc.lines_markeredgewidth, markerfacecolor: None, markeredgecolor: None,
        zorder: line.zorder,
    };
    let xy: Vec<[f64; 2]> = line.x.iter().zip(&line.y).map(|(&x, &y)| [x, y]).collect();
    let mut out = artist::line(ctx.rc, &spec, xy, an.trans_axes);
    out.color = ir::Color { rgba: line.color, none: false };
    out.markeredgecolor = out.color;
    out.markerfacecolor = out.color;
    out.solid_capstyle = line.capstyle;
    out.artist.visible = line.visible;
    out.artist.clip_box = Some(an.patch_clip);
    out.axes = Some(index);
    artist::recache(ctx.rc, &mut out, an.data_is_axes);
    Artist::Line(out)
}

/// The point overlay's rings: `EllipseCollection(units="xy", transOffset=transData, facecolors="none")`.
#[derive(Clone, Debug)]
pub struct EllipsesArtist {
    pub visible: bool,
    pub zorder: f64,
    pub offsets: Vec<[f64; 2]>,
    /// Full widths and heights (the collection keeps halves).
    pub widths: Vec<f64>,
    pub heights: Vec<f64>,
    pub edgecolors: Vec<[f64; 4]>,
    pub linewidths: Vec<f64>,
    /// `"--"` where true, `"-"` elsewhere.
    pub dashed: Vec<bool>,
}

pub fn ellipses_ir(ctx: &EmitContext<'_>, nodes: &mut super::nodes::Nodes, an: &AxesNodes, index: usize,
                   model: &super::axes::AxesModel, e: &EllipsesArtist) -> Artist {
    // `_set_transforms` (units 'xy'): transData's affine without its translation.
    let m = model.trans_data(ctx.size_inches, ctx.dpi).0;
    let transform = nodes.affine([m[0][0], m[0][1], 0.0, m[1][0], m[1][1], 0.0]);
    let mut props = artist::props(e.zorder, Some(transform));
    props.visible = e.visible;
    props.clip_box = Some(an.patch_clip);
    let (vertices, codes) = artist::unit_circle();
    let dashes: Vec<ir::Dash> = e.dashed.iter().zip(&e.linewidths).map(|(&dashed, &lw)| artist::dash(ctx.rc, if dashed { "--" } else { "-" }, lw)).collect();
    Artist::Collection(ir::Collection {
        artist: props,
        cls: "EllipseCollection".into(),
        paths: vec![artist::path(vertices, Some(codes), false)],
        offsets: e.offsets.clone(),
        offset_transform: an.trans_data,
        facecolors: Vec::new(),
        edgecolors: e.edgecolors.clone(),
        linewidths: e.linewidths.clone(),
        linestyles: dashes,
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: None,
        factor: None,
        transforms: None,
        ellipses: Some(ir::Ellipses {
            widths: e.widths.iter().map(|w| 0.5 * w).collect(), heights: e.heights.iter().map(|h| 0.5 * h).collect(),
            angles: vec![0.0; e.widths.len()], units: "xy".into(), axes: index,
        }),
    })
}

/// The point overlay with movement paths: one filled coverage union per
/// status colour (`_point_path_union_paths`), laid out in display pixels when
/// the figure is drawn.
#[derive(Clone, Debug)]
pub struct UnionArtist {
    pub visible: bool,
    pub zorder: f64,
    /// Sites and their paths in display units.
    pub points: Vec<[f64; 2]>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub radii: (f64, f64),
    pub colours: Vec<[f64; 4]>,
    pub widths: Vec<f64>,
    pub dashed: Vec<bool>,
}

/// `Path(vstack((v, v[0])), [MOVETO, LINETO..., CLOSEPOLY])`, appended to a compound path.
fn polygon(vertices: &[[f64; 2]], out: &mut (Vec<[f64; 2]>, Vec<u8>)) {
    out.0.extend_from_slice(vertices);
    out.0.push(vertices[0]);
    out.1.push(1);
    out.1.extend(std::iter::repeat_n(2, vertices.len() - 1));
    out.1.push(79);
}

fn norm(v: [f64; 2]) -> f64 { (v[0] * v[0] + v[1] * v[1]).sqrt() }

/// The compound paths of `_point_path_union_paths` and their colours, at the
/// axes' current transData.
fn union_paths(ctx: &EmitContext<'_>, model: &super::axes::AxesModel, u: &UnionArtist) -> (Vec<ir::Path>, Vec<[f64; 4]>) {
    let dpi = ctx.dpi;
    let to_display = |p: [f64; 2]| { let (x, y) = model.to_display(ctx.size_inches, dpi, p[0], p[1]); [x, y] };
    let unit: Vec<[f64; 2]> = crate::ticks::linspace(0.0, 2.0 * std::f64::consts::PI, 65)[..64].iter().map(|t| [t.cos(), t.sin()]).collect();
    let circle = artist::unit_circle();
    let (rx, ry) = u.radii;
    let mut groups: Vec<([f64; 4], (Vec<[f64; 2]>, Vec<u8>))> = Vec::new();
    for (index, (point, source)) in u.points.iter().zip(&u.paths).enumerate() {
        let colour = u.colours[index];
        if colour[3] <= 0.0 { continue; }
        let slot = match groups.iter().position(|(c, _)| *c == colour) {
            Some(slot) => slot,
            None => { groups.push((colour, (Vec::new(), Vec::new()))); groups.len() - 1 }
        };
        let polygons = &mut groups[slot].1;
        let ring: Vec<[f64; 2]> = unit.iter().map(|c| to_display([point[0] + c[0] * rx, point[1] + c[1] * ry])).collect();
        let path: Vec<[f64; 2]> = source.iter().map(|&p| to_display(p)).collect();
        let n = ring.len();
        let stroke = u.widths[index] * dpi / 144.0;
        let mut normal: Vec<[f64; 2]> = (0..n).map(|i| {
            let t = [ring[(i + 1) % n][0] - ring[(i + n - 1) % n][0], ring[(i + 1) % n][1] - ring[(i + n - 1) % n][1]];
            let v = [t[1], -t[0]];
            let length = norm(v);
            [v[0] / length, v[1] / length]
        }).collect();
        let terms: Vec<f64> = (0..n).map(|i| ring[i][0] * ring[(i + 1) % n][1] - ring[i][1] * ring[(i + 1) % n][0]).collect();
        let area = crate::numeric::sum_f64(&terms);
        if area < 0.0 { for v in &mut normal { *v = [v[0] * -1.0, v[1] * -1.0]; } }
        let outer: Vec<[f64; 2]> = (0..n).map(|i| [ring[i][0] + normal[i][0] * stroke, ring[i][1] + normal[i][1] * stroke]).collect();
        let inner: Vec<[f64; 2]> = (0..n).map(|i| [ring[i][0] - normal[i][0] * stroke, ring[i][1] - normal[i][1] * stroke]).collect();
        let reversed = |v: &[[f64; 2]]| v.iter().rev().copied().collect::<Vec<_>>();
        if u.dashed[index] {
            let mut distance = vec![0.0];
            for i in 0..n {
                let next = ring[(i + 1) % n];
                let step = norm([next[0] - ring[i][0], next[1] - ring[i][1]]);
                distance.push(distance[i] + step);
            }
            let dash = (u.widths[index] * dpi / 72.0 * 3.7).max(1.0);
            for part in 0..n {
                if ((distance[part] / dash) as i64) % 2 == 0 {
                    let quad = [outer[part], outer[(part + 1) % n], inner[(part + 1) % n], inner[part]];
                    polygon(&if area > 0.0 { quad.to_vec() } else { reversed(&quad) }, polygons);
                }
            }
        } else {
            polygon(&if area > 0.0 { outer.clone() } else { reversed(&outer) }, polygons);
            polygon(&if area > 0.0 { reversed(&inner) } else { inner.clone() }, polygons);
        }
        let delta: Vec<[f64; 2]> = path.windows(2).map(|w| [w[1][0] - w[0][0], w[1][1] - w[0][1]]).collect();
        let length: Vec<f64> = delta.iter().map(|d| norm(*d)).collect();
        let moving: Vec<usize> = (0..length.len()).filter(|&i| length[i] > 0.0).collect();
        if moving.is_empty() { continue; }
        let last = path.len() - 1;
        let mut paint: Vec<[f64; 2]> = moving.iter().chain(std::iter::once(&last)).map(|&i| path[i]).collect();
        let directions: Vec<[f64; 2]> = paint.windows(2).zip(&moving)
            .map(|(w, &m)| [(w[1][0] - w[0][0]) / length[m], (w[1][1] - w[0][1]) / length[m]]).collect();
        let mut keep = vec![0usize];
        for k in 0..directions.len().saturating_sub(1) {
            let (a, b) = (directions[k], directions[k + 1]);
            let cross = a[0] * b[1] - a[1] * b[0];
            if cross.abs() > 1e-9 || a[0] * b[0] + a[1] * b[1] < 0.0 { keep.push(k + 1); }
        }
        keep.push(paint.len() - 1);
        paint = keep.iter().map(|&i| paint[i]).collect();
        let delta: Vec<[f64; 2]> = paint.windows(2).map(|w| [w[1][0] - w[0][0], w[1][1] - w[0][1]]).collect();
        let paint_length: Vec<f64> = delta.iter().map(|d| norm(*d)).collect();
        let half = stroke * 0.5;
        let end = delta.len() - 1;
        let mut direction = [delta[end][0] / paint_length[end], delta[end][1] / paint_length[end]];
        let tail = source[source.len() - 1];
        let end_radius = unit.iter().map(|c| {
            let e = to_display([tail[0] + c[0] * rx, tail[1] + c[1] * ry]);
            norm([e[0] - path[last][0], e[1] - path[last][1]])
        }).fold(f64::NEG_INFINITY, |m, v| if v > m { v } else { m });
        let gap = (end_radius + stroke).min(paint_length[end] * 0.4);
        let head_length = (u.widths[index] * dpi / 72.0 * 3.0).min((paint_length[end] - gap) * 0.4);
        let tip = [path[last][0] - direction[0] * gap, path[last][1] - direction[1] * gap];
        let base = [tip[0] - direction[0] * head_length, tip[1] - direction[1] * head_length];
        let corners = paint.len() - 1;
        paint[corners] = base;
        for step in 0..delta.len() {
            direction = [delta[step][0] / paint_length[step], delta[step][1] / paint_length[step]];
            let normal = [-direction[1] * half, direction[0] * half];
            polygon(&[[paint[step][0] - normal[0], paint[step][1] - normal[1]], [paint[step + 1][0] - normal[0], paint[step + 1][1] - normal[1]],
                      [paint[step + 1][0] + normal[0], paint[step + 1][1] + normal[1]], [paint[step][0] + normal[0], paint[step][1] + normal[1]]],
                    polygons);
        }
        for vertex in &paint[..paint.len() - 1] {
            if *vertex == path[last] { continue; }
            polygons.0.extend(circle.0.iter().map(|c| [c[0] * half + vertex[0], c[1] * half + vertex[1]]));
            polygons.1.extend_from_slice(&circle.1);
        }
        let normal = [-direction[1] * head_length * 0.4, direction[0] * head_length * 0.4];
        polygon(&[tip, [base[0] + normal[0], base[1] + normal[1]], [base[0] - normal[0], base[1] - normal[1]]], polygons);
    }
    let colours = groups.iter().map(|(c, _)| *c).collect();
    (groups.into_iter().map(|(_, (vertices, codes))| artist::path(vertices, Some(codes), false)).collect(), colours)
}

pub fn union_ir(ctx: &EmitContext<'_>, an: &AxesNodes, model: &super::axes::AxesModel, u: &UnionArtist) -> Artist {
    let mut props = artist::props(u.zorder, Some(ctx.figure.identity));
    props.visible = u.visible;
    props.snap = Some(false);
    props.clip_box = Some(an.patch_clip);
    let (paths, facecolors) = union_paths(ctx, model, u);
    Artist::Collection(ir::Collection {
        artist: props,
        cls: "PolyCollection".into(),
        paths,
        offsets: vec![[0.0, 0.0]],
        offset_transform: ctx.figure.identity,
        facecolors,
        edgecolors: Vec::new(),
        linewidths: vec![0.0],
        linestyles: vec![ir::Dash(0.0, None)],
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: Some(Vec::new()),
        factor: Some(1.0),
        transforms: None,
        ellipses: None,
    })
}
