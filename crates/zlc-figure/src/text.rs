//! `matplotlib.text.Text`: layout (`_get_layout`), window extent and draw,
//! and `RendererAgg.draw_text`/`draw_mathtext`/`get_text_width_height_descent`.
use crate::draw::Ctx;
use crate::ir::{HAlign, RotationMode, Text, VAlign};
use crate::transform::{M3, Points};
use crate::{unsupported, Result};

/// `cbook.is_math_text(s)`: an even, positive number of unescaped dollars.
pub fn is_math_text(s: &str) -> bool {
    let dollars = s.matches('$').count() as i64 - s.matches("\\$").count() as i64;
    dollars > 0 && dollars % 2 == 0
}

/// `Text._preprocess_math(s)` (usetex is refused at extraction).
pub fn preprocess_math(t: &Text, s: &str) -> (String, bool) {
    if !t.parse_math {
        (s.to_owned(), false)
    } else if is_math_text(s) {
        (s.to_owned(), true)
    } else {
        (s.replace("\\$", "$"), false)
    }
}

/// `TextToPath.FONT_SCALE`: the size a vector renderer measures text at.
pub const FONT_SCALE: f64 = 100.0;

/// `renderer.get_text_width_height_descent(s, prop, ismath)`: Agg's, or for
/// a vector renderer `TextToPath`'s (unhinted at [`FONT_SCALE`] and 72 dpi,
/// scaled to the size).
pub fn text_width_height_descent(ctx: &mut Ctx<'_>, s: &str, font: &crate::ir::Font, ismath: bool) -> Result<(f64, f64, f64)> {
    if ctx.vector {
        if ismath {
            let parse = ctx.math.vector(&ctx.fig.rc, s, font)?;
            return Ok((parse.width, parse.height, parse.descent));
        }
        let id = ctx.fonts.get_font(&font.files, ctx.fig.rc.text_hinting_factor, ctx.fig.rc.text_kerning_factor)?;
        ctx.fonts.set_size(id, FONT_SCALE, 72.0)?;
        ctx.fonts.set_text(id, s, 0.0, crate::fonts::load_flags::NO_HINTING)?;
        let (w, h) = ctx.fonts.get_width_height(id)?;
        let d = ctx.fonts.get_descent(id)?;
        let scale = font.size / FONT_SCALE / 64.0;
        return Ok((w as f64 * scale, h as f64 * scale, d as f64 * scale));
    }
    if ismath {
        let parse = ctx.math.parse(&ctx.fig.rc, s, ctx.dpi, font, None)?;
        return Ok((parse.width, parse.height, parse.descent));
    }
    let id = prepare_font(ctx, font)?;
    ctx.fonts.set_text(id, s, 0.0, ctx.hinting)?;
    let (w, h) = ctx.fonts.get_width_height(id)?;
    let d = ctx.fonts.get_descent(id)?;
    Ok((w as f64 / 64.0, h as f64 / 64.0, d as f64 / 64.0))
}

/// `RendererAgg._prepare_font(prop)`.
fn prepare_font(ctx: &mut Ctx<'_>, font: &crate::ir::Font) -> Result<usize> {
    let id = ctx.fonts.get_font(&font.files, ctx.fig.rc.text_hinting_factor, ctx.fig.rc.text_kerning_factor)?;
    ctx.fonts.clear(id)?;
    ctx.fonts.set_size(id, font.size, ctx.dpi)?;
    Ok(id)
}

/// One laid-out line: text, (w, h), offset from the anchor.
#[derive(Clone, Debug)]
pub struct Line { pub text: String, pub w: f64, pub h: f64, pub x: f64, pub y: f64 }

#[derive(Clone, Debug)]
pub struct Layout { pub bbox: Points, pub lines: Vec<Line>, pub descent: f64 }

/// numpy `.min()` / `.max()` of a column (first of equals).
fn col_min(v: &[f64; 4]) -> f64 { let mut m = v[0]; for &x in &v[1..] { if x < m { m = x; } } m }
fn col_max(v: &[f64; 4]) -> f64 { let mut m = v[0]; for &x in &v[1..] { if x > m { m = x; } } m }

/// `Text._get_layout(renderer)`.
pub fn layout(ctx: &mut Ctx<'_>, t: &Text) -> Result<Layout> {
    let lines: Vec<&str> = t.text.split('\n').collect();
    let (_, lp_h, lp_d) = text_width_height_descent(ctx, "lp", &t.font, false)?;
    let min_dy = (lp_h - lp_d) * t.linespacing;
    let mut ws = Vec::with_capacity(lines.len());
    let mut hs = Vec::with_capacity(lines.len());
    let mut ys = Vec::with_capacity(lines.len());
    let mut thisy = 0.0f64;
    let thisx = 0.0f64;
    let mut baseline = 0.0;
    let mut d = 0.0;
    for (i, line) in lines.iter().enumerate() {
        let (clean, ismath) = preprocess_math(t, line);
        let (w, mut h, mut dd) = if !clean.is_empty() {
            text_width_height_descent(ctx, &clean, &t.font, ismath)?
        } else {
            (0.0, 0.0, 0.0)
        };
        h = h.max(lp_h);
        dd = dd.max(lp_d);
        ws.push(w);
        hs.push(h);
        baseline = (h - dd) - thisy;
        if i == 0 {
            thisy = -(h - dd);
        } else {
            thisy -= min_dy.max((h - dd) * t.linespacing);
        }
        ys.push(thisy);
        thisy -= dd;
        d = dd;
    }
    let descent = d;
    let width = ws.iter().copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let (xmin0, xmax0, ymax0) = (0.0f64, width, 0.0f64);
    let ymin0 = ys[ys.len() - 1] - descent;
    let m = M3::IDENTITY.rotate_deg(t.rotation);
    let malign = t.ma.unwrap_or(t.ha);
    let offset_layout: Vec<(f64, f64)> = match malign {
        HAlign::Left => ys.iter().map(|&y| (thisx, y)).collect(),
        HAlign::Center => ys.iter().zip(&ws).map(|(&y, &w)| (thisx + width / 2.0 - w / 2.0, y)).collect(),
        HAlign::Right => ys.iter().zip(&ws).map(|(&y, &w)| (thisx + width - w, y)).collect(),
    };
    let corners = [(xmin0, ymin0), (xmin0, ymax0), (xmax0, ymax0), (xmax0, ymin0)];
    let mut cx = [0.0; 4];
    let mut cy = [0.0; 4];
    for (k, &(x, y)) in corners.iter().enumerate() {
        let (a, b) = m.apply(x, y);
        cx[k] = a;
        cy[k] = b;
    }
    let mut xmin = col_min(&cx);
    let xmax = col_max(&cx);
    let mut ymin = col_min(&cy);
    let ymax = col_max(&cy);
    let width = xmax - xmin;
    let height = ymax - ymin;
    let (offsetx, offsety) = if t.rotation_mode != RotationMode::Anchor {
        let ox = match t.ha { HAlign::Center => (xmin + xmax) / 2.0, HAlign::Right => xmax, HAlign::Left => xmin };
        let oy = match t.va {
            VAlign::Center => (ymin + ymax) / 2.0,
            VAlign::Top => ymax,
            VAlign::Baseline => ymin + descent,
            VAlign::CenterBaseline => ymin + height - baseline / 2.0,
            VAlign::Bottom => ymin,
        };
        (ox, oy)
    } else {
        let (xmin1, ymin1) = corners[0];
        let (xmax1, ymax1) = corners[2];
        let ox = match t.ha { HAlign::Center => (xmin1 + xmax1) / 2.0, HAlign::Right => xmax1, HAlign::Left => xmin1 };
        let oy = match t.va {
            VAlign::Center => (ymin1 + ymax1) / 2.0,
            VAlign::Top => ymax1,
            VAlign::Baseline => ymax1 - baseline,
            VAlign::CenterBaseline => ymax1 - baseline / 2.0,
            VAlign::Bottom => ymin1,
        };
        m.apply(ox, oy)
    };
    xmin -= offsetx;
    ymin -= offsety;
    let bbox = [xmin, ymin, xmin + width, ymin + height];
    let mut out = Vec::with_capacity(lines.len());
    for (k, line) in lines.iter().enumerate() {
        let (x, y) = m.apply(offset_layout[k].0, offset_layout[k].1);
        out.push(Line { text: (*line).to_owned(), w: ws[k], h: hs[k], x: x - offsetx, y: y - offsety });
    }
    Ok(Layout { bbox, lines: out, descent })
}

/// `Text.get_window_extent(renderer)` at position `xy` (the text's own
/// `(_x, _y)` unless the draw moved it).
pub fn window_extent(ctx: &mut Ctx<'_>, t: &Text, xy: Option<(f64, f64)>) -> Result<Points> {
    let (x, y) = xy.unwrap_or((t.x, t.y));
    if !t.artist.visible || !check_xy(ctx, t)? {
        return Ok([0.0, 0.0, 1.0, 1.0]);
    }
    if t.text.is_empty() {
        let (tx, ty) = text_point(ctx, t, x, y)?;
        return Ok([tx, ty, tx + 0.0, ty + 0.0]);
    }
    let lay = layout(ctx, t)?;
    let (tx, ty) = text_point(ctx, t, x, y)?;
    let b = lay.bbox;
    Ok([b[0] + tx, b[1] + ty, b[2] + tx, b[3] + ty])
}

/// `Text.draw(renderer)` at position `xy` (default: the text's own).
pub fn draw_text(ctx: &mut Ctx<'_>, t: &Text, xy: Option<(f64, f64)>) -> Result<()> {
    if !t.artist.visible || !check_xy(ctx, t)? || t.text.is_empty() {
        return Ok(());
    }
    if let Some(span) = &t.span_label {
        // SpanLabel.draw: only where the span holds the text and its padding.
        let x0 = ctx.point(&t.artist, span.span[0], 0.0)?.0;
        let x1 = ctx.point(&t.artist, span.span[1], 0.0)?.0;
        let e = window_extent(ctx, t, xy)?;
        let width = e[2] - e[0];
        if width + 2.0 * ctx.points_to_pixels(span.pad_pt) > (x1 - x0).abs() {
            return Ok(());
        }
    }
    let (x, y) = xy.unwrap_or((t.x, t.y));
    let lay = layout(ctx, t)?;
    if x.is_nan() || y.is_nan() { return Ok(()); }
    let (posx, posy) = text_point(ctx, t, x, y)?;
    if posx.is_nan() || posy.is_nan() || !posx.is_finite() || !posy.is_finite() {
        return Ok(());
    }
    let canvash = ctx.height;
    if let Some(patch) = &t.bbox_patch {
        // update_bbox_position_size, then the patch draws.
        let (x_box, y_box, w_box, h_box) = textbox(t, &lay);
        let transform = M3::IDENTITY.rotate_deg(t.rotation).translate(posx + x_box, posy + y_box);
        let scale = ctx.points_to_pixels(t.font.size);
        crate::patches::draw_fancy_bbox(ctx, patch, w_box, h_box, &transform, scale)?;
    }
    let mut gc = ctx.new_gc();
    gc.set_foreground_color(t.color, false);
    gc.set_alpha(t.artist.alpha);
    gc.antialiased = t.antialiased;
    ctx.set_gc_clip(&mut gc, &t.artist)?;
    let angle = t.rotation;
    for line in &lay.lines {
        if ctx.vector {
            // A vector renderer is not flipped (or flips itself).
            let (clean, ismath) = preprocess_math(t, &line.text);
            let (x, y) = (line.x + posx, line.y + posy);
            if ismath {
                let parse = ctx.math.vector(&ctx.fig.rc, &clean, &t.font)?;
                ctx.backend.draw_text(&gc, x, y, crate::backend::TextRun::Math { parse: &parse, font: &t.font }, angle)?;
            } else {
                ctx.backend.draw_text(&gc, x, y, crate::backend::TextRun::Plain { text: &clean, font: &t.font }, angle)?;
            }
            continue;
        }
        let lx = line.x + posx;
        let ly = canvash - (line.y + posy);
        let (clean, ismath) = preprocess_math(t, &line.text);
        if ismath {
            draw_mathtext(ctx, &gc, lx, ly, &clean, t, angle)?;
        } else {
            draw_plain(ctx, &gc, lx, ly, &clean, t, angle)?;
        }
    }
    Ok(())
}

/// Python's `round(x)` for a float: nearest integer, ties to even.
fn py_round(v: f64) -> Result<i32> {
    let r = v.round_ties_even();
    if !r.is_finite() || r < i32::MIN as f64 || r > i32::MAX as f64 {
        return unsupported(format!("text position {v} out of int range"));
    }
    Ok(r as i32)
}

/// `RendererAgg.draw_text` (not math).
fn draw_plain(ctx: &mut Ctx<'_>, gc: &crate::backend::Gc, x: f64, y: f64, s: &str, t: &Text, angle: f64) -> Result<()> {
    let id = prepare_font(ctx, &t.font)?;
    ctx.fonts.set_text(id, s, 0.0, ctx.hinting)?;
    ctx.fonts.draw_glyphs_to_bitmap(id, gc.antialiased)?;
    let d = ctx.fonts.get_descent(id)? as f64 / 64.0;
    let (xo, yo) = ctx.fonts.get_bitmap_offset(id)?;
    let xo = xo as f64 / 64.0;
    let yo = yo as f64 / 64.0;
    let rad = angle.to_radians();
    let xd = d * rad.sin();
    let yd = d * rad.cos();
    let xi = py_round(x + xo + xd)?;
    let yi = py_round(y + yo + yd)?;
    let image = ctx.fonts.image(id)?;
    ctx.backend.draw_text_image(image, xi, yi + 1, angle, gc)
}

/// `RendererAgg.draw_mathtext`.
fn draw_mathtext(ctx: &mut Ctx<'_>, gc: &crate::backend::Gc, x: f64, y: f64, s: &str, t: &Text, angle: f64) -> Result<()> {
    let parse = ctx.math.parse(&ctx.fig.rc, s, ctx.dpi, &t.font, Some(gc.antialiased))?;
    let rad = angle.to_radians();
    let xd = parse.descent * rad.sin();
    let yd = parse.descent * rad.cos();
    let xi = py_round(x + parse.ox + xd)?;
    let yi = py_round(y - parse.oy + yd)?;
    ctx.backend.draw_text_image(parse.image.as_ref(), xi, yi + 1, angle, gc)
}

/// `text._get_textbox(text, renderer)`: the unrotated box of the layout, its
/// corner rotated back: (x_box, y_box, w_box, h_box).
pub fn textbox(t: &Text, lay: &Layout) -> (f64, f64, f64, f64) {
    let theta = t.rotation.to_radians();
    let tr = M3::IDENTITY.rotate(-theta);
    let mut xs = Vec::with_capacity(2 * lay.lines.len());
    let mut ys = Vec::with_capacity(2 * lay.lines.len());
    for line in &lay.lines {
        let (xt1, mut yt1) = tr.apply(line.x, line.y);
        yt1 -= lay.descent;
        xs.push(xt1);
        xs.push(xt1 + line.w);
        ys.push(yt1);
        ys.push(yt1 + line.h);
    }
    let first_min = |v: &[f64]| v[1..].iter().fold(v[0], |m, &x| if x < m { x } else { m });
    let first_max = |v: &[f64]| v[1..].iter().fold(v[0], |m, &x| if x > m { x } else { m });
    let (xt_box, yt_box) = (first_min(&xs), first_min(&ys));
    let (w_box, h_box) = (first_max(&xs) - xt_box, first_max(&ys) - yt_box);
    let (x_box, y_box) = M3::IDENTITY.rotate(theta).apply(xt_box, yt_box);
    (x_box, y_box, w_box, h_box)
}

/// A text position in display space: an Annotation's transform comes from
/// its anchor at draw time (`update_positions`), any other text's is its own.
pub fn text_point(ctx: &mut Ctx<'_>, t: &Text, x: f64, y: f64) -> Result<(f64, f64)> {
    if t.annotation.is_none() { return ctx.point(&t.artist, x, y); }
    Ok(annotation_transform(ctx, t)?.apply(x, y))
}

fn annotation_transform(ctx: &mut Ctx<'_>, t: &Text) -> Result<M3> {
    let Some(a) = &t.annotation else { return ctx.transform(&t.artist) };
    let td_id = ctx.fig.axes[a.axes].trans_data;
    match a.anncoords.as_str() {
        "data" => ctx.eval.matrix(td_id),
        "offset points" | "offset pixels" => {
            let (x0, y0) = ctx.eval.transform(td_id, a.xy[0], a.xy[1])?;
            let s = if a.anncoords == "offset points" { ctx.dpi / 72.0 } else { 1.0 };
            let base = if a.anncoords == "offset points" { M3::IDENTITY.scale(s, s) } else { M3::IDENTITY };
            Ok(base.translate(x0, y0))
        }
        other => unsupported(format!("annotation coords {other}")),
    }
}

/// `_AnnotationBase._check_xy`: an annotation of data coordinates is drawn
/// only while its anchor is in the axes (with a 1 px margin).
pub fn check_xy(ctx: &mut Ctx<'_>, t: &Text) -> Result<bool> {
    let Some(a) = &t.annotation else { return Ok(true) };
    let clip = a.annotation_clip;
    if clip == Some(true) || (clip.is_none() && a.xycoords == "data") {
        let point = ctx.eval.transform(ctx.fig.axes[a.axes].trans_data, a.xy[0], a.xy[1])?;
        let patch = &ctx.fig.axes[a.axes].patch;
        return crate::patches::rectangle_contains_point(ctx, patch, point, 1.0);
    }
    Ok(true)
}
