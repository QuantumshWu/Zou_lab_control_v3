//! A page as SVG (`RendererSVG` with matplotlib's default `svg.fonttype:
//! path`): points as user units with y down, text as the outlines of its
//! glyphs (one `<path>` each in `<defs>`, placed by `<use>`), so it looks the
//! same in every reader whatever fonts it has; images as PNG.
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

use base64::Engine;
use zlc_figure::ir::{CapStyle, JoinStyle};
use zlc_figure::text::FONT_SCALE;

use super::{byte, num, png, Item, Page, Paint, Rect, Seg};

fn colour(c: [f64; 4]) -> String { format!("#{:02x}{:02x}{:02x}", byte(c[0]), byte(c[1]), byte(c[2])) }

/// Path data with y flipped (`flip` is the page height; 0 keeps a symbol's frame).
fn path_data(segs: &[Seg], flip: Option<f64>) -> String {
    let y = |v: f64| num(match flip { Some(h) => h - v, None => -v }, 3);
    let x = |v: f64| num(v, 3);
    let mut d = String::new();
    for seg in segs {
        if !d.is_empty() { d.push(' '); }
        match *seg {
            Seg::Move(a, b) => { let _ = write!(d, "M {} {}", x(a), y(b)); }
            Seg::Line(a, b) => { let _ = write!(d, "L {} {}", x(a), y(b)); }
            Seg::Quad(a, b, c, e) => { let _ = write!(d, "Q {} {} {} {}", x(a), y(b), x(c), y(e)); }
            Seg::Cubic(a, b, c, e, f, g) => { let _ = write!(d, "C {} {} {} {} {} {}", x(a), y(b), x(c), y(e), x(f), y(g)); }
            Seg::Close => d.push('z'),
        }
    }
    d
}

/// The style of a paint (defaults: `stroke-linejoin: round; stroke-linecap: butt`).
fn style(paint: &Paint) -> String {
    let mut s = String::new();
    match paint.fill {
        None => s.push_str("fill:none"),
        Some(c) => {
            let _ = write!(s, "fill:{}", colour(c));
            if c[3] != 1.0 { let _ = write!(s, ";fill-opacity:{}", num(c[3], 4)); }
        }
    }
    if let Some(stroke) = &paint.stroke {
        let _ = write!(s, ";stroke:{};stroke-width:{}", colour(stroke.color), num(stroke.width, 4));
        if stroke.color[3] != 1.0 { let _ = write!(s, ";stroke-opacity:{}", num(stroke.color[3], 4)); }
        match stroke.cap { CapStyle::Butt => {}, CapStyle::Round => s.push_str(";stroke-linecap:round"), CapStyle::Projecting => s.push_str(";stroke-linecap:square") }
        match stroke.join { JoinStyle::Round => {}, JoinStyle::Miter => s.push_str(";stroke-linejoin:miter"), JoinStyle::Bevel => s.push_str(";stroke-linejoin:bevel") }
        if let Some((offset, list)) = &stroke.dashes {
            let list: Vec<String> = list.iter().map(|v| num(*v, 4)).collect();
            let _ = write!(s, ";stroke-dasharray:{};stroke-dashoffset:{}", list.join(","), num(*offset, 4));
        }
    }
    s
}

/// The page's clip boxes, one id each.
struct Clips { ids: HashMap<[u64; 4], usize>, defs: String, height: f64 }

impl Clips {
    fn attr(&mut self, clip: &Option<Rect>) -> String {
        let Some(r) = clip else { return String::new() };
        let key = r.map(f64::to_bits);
        let next = self.ids.len();
        let id = *self.ids.entry(key).or_insert_with(|| {
            let _ = write!(self.defs, "  <clipPath id=\"c{next}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath>\n",
                           num(r[0], 3), num(self.height - r[3], 3), num(r[2] - r[0], 3), num(r[3] - r[1], 3));
            next
        });
        format!(" clip-path=\"url(#c{id})\"")
    }
}

/// A glyph's outline as `TextToPath` takes it (unhinted, at `FONT_SCALE`
/// points and 72 dpi, y up), from a face set to that size.
fn outline(font: &mut zlc_text::FT2Font, gid: u16) -> Result<Vec<Seg>, String> {
    font.load_glyph(u32::from(gid), zlc_figure::fonts::load_flags::NO_HINTING, false).map_err(|e| e.to_string())?;
    let (xy, codes) = font.get_path().map_err(|e| e.to_string())?;
    let mut segs = Vec::new();
    let mut k = 0;
    while k < codes.len() {
        let (x, y) = (xy[2 * k], xy[2 * k + 1]);
        match codes[k] {
            zlc_text::MOVETO => segs.push(Seg::Move(x, y)),
            zlc_text::LINETO => segs.push(Seg::Line(x, y)),
            zlc_text::CURVE3 if k + 1 < codes.len() => { segs.push(Seg::Quad(x, y, xy[2 * k + 2], xy[2 * k + 3])); k += 1; }
            zlc_text::CURVE4 if k + 2 < codes.len() => {
                segs.push(Seg::Cubic(x, y, xy[2 * k + 2], xy[2 * k + 3], xy[2 * k + 4], xy[2 * k + 5]));
                k += 2;
            }
            zlc_text::CLOSEPOLY => segs.push(Seg::Close),
            other => return Err(format!("glyph {gid}: path code {other}")),
        }
        k += 1;
    }
    Ok(segs)
}

pub(crate) fn write(page: &mut Page) -> Result<Vec<u8>, String> {
    let (w, h) = (page.width, page.height);
    // Every glyph drawn, once, in `<defs>`.
    let mut glyphs = String::new();
    let mut drawn = BTreeSet::new();
    for (index, face) in page.faces.iter_mut().enumerate() {
        if face.used.is_empty() { continue; }
        face.font.clear().map_err(|e| e.to_string())?;
        face.font.set_size(FONT_SCALE, 72.0).map_err(|e| e.to_string())?;
        for &gid in face.used.keys() {
            let segs = outline(&mut face.font, gid)?;
            if segs.is_empty() { continue; }
            let _ = writeln!(glyphs, "  <path id=\"g{index}_{gid}\" d=\"{}\"/>", path_data(&segs, None));
            drawn.insert((index, gid));
        }
    }
    let mut clips = Clips { ids: HashMap::new(), defs: String::new(), height: h };
    let mut body = String::new();
    for item in &page.items {
        match item {
            Item::Path { segs, paint, clip } => {
                let clip = clips.attr(clip);
                let _ = writeln!(body, "  <path d=\"{}\" style=\"{}\"{clip}/>", path_data(segs, Some(h)), style(paint));
            }
            Item::Uses { symbol, at, clip } => {
                let _ = writeln!(body, "  <g{}>", clips.attr(clip));
                for &(x, y) in at {
                    let _ = writeln!(body, "   <use xlink:href=\"#m{symbol}\" x=\"{}\" y=\"{}\"/>", num(x, 3), num(h - y, 3));
                }
                body.push_str("  </g>
");
            }
            Item::Image { cols, rows, rgba, matrix, opacity, clip } => {
                let m = matrix.0;
                // Page y down: the pixel grid's map, flipped.
                let transform = format!("matrix({} {} {} {} {} {})", num(m[0][0], 6), num(-m[1][0], 6), num(m[0][1], 6), num(-m[1][1], 6),
                                        num(m[0][2], 4), num(h - m[1][2], 4));
                let data = base64::engine::general_purpose::STANDARD.encode(png(*cols, *rows, rgba)?);
                let opacity = if *opacity != 1.0 { format!(" opacity=\"{}\"", num(*opacity, 4)) } else { String::new() };
                let _ = writeln!(body, "  <g{}><image width=\"{cols}\" height=\"{rows}\" transform=\"{transform}\" preserveAspectRatio=\"none\"{opacity}                     style=\"image-rendering:crisp-edges;image-rendering:pixelated\" xlink:href=\"data:image/png;base64,{data}\"/></g>", clips.attr(clip));
            }
            Item::Text { x, y, angle, color, glyphs: placed, rules, clip } => {
                let mut fill = format!("fill:{}", colour(*color));
                if color[3] != 1.0 { let _ = write!(fill, ";fill-opacity:{}", num(color[3], 4)); }
                let rotate = if *angle != 0.0 { format!(" rotate({})", num(-angle, 4)) } else { String::new() };
                // The clip box is in page space: on a group outside the text's transform.
                let _ = writeln!(body, "  <g{}><g transform=\"translate({} {}){rotate}\" style=\"{fill}\">", clips.attr(clip), num(*x, 3), num(h - y, 3));
                for g in placed.iter().filter(|g| drawn.contains(&(g.face, g.gid))) {
                    let _ = writeln!(body, "   <use xlink:href=\"#g{}_{}\" transform=\"translate({} {}) scale({})\"/>", g.face, g.gid,
                                     num(g.x, 3), num(-g.y, 3), num(g.size / FONT_SCALE, 6));
                }
                for r in rules {
                    let _ = writeln!(body, "   <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>", num(r[0], 3), num(-r[3], 3),
                                     num(r[2] - r[0], 3), num(r[3] - r[1], 3));
                }
                body.push_str("  </g></g>
");
            }
        }
    }
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>
");
    out.push_str("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">
");
    let _ = writeln!(out, "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{}pt\" height=\"{}pt\" viewBox=\"0 0 {} {}\"         xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">", num(w, 4), num(h, 4), num(w, 4), num(h, 4));
    out.push_str(" <defs>
  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>
");
    out.push_str(&glyphs);
    for (index, symbol) in page.symbols.iter().enumerate() {
        let _ = writeln!(out, "  <path id=\"m{index}\" d=\"{}\" style=\"{}\"/>", path_data(&symbol.segs, None), style(&symbol.paint));
    }
    out.push_str(&clips.defs);
    out.push_str(" </defs>
");
    out.push_str(&body);
    out.push_str("</svg>
");
    Ok(out.into_bytes())
}
