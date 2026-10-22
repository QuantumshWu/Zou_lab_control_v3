//! A page as a one-page PDF (`RendererPdf` with TrueType fonts embedded as
//! CID fonts, `pdf.fonttype: 42`): text stays text, in the figure's own
//! fonts cut to the glyphs used; images are Flate RGB with a soft mask.
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::Write as _;

use zlc_figure::ir::{CapStyle, JoinStyle};
use zlc_figure::transform::M3;

use super::{num, sfnt, Item, Page, Paint, Rect, Seg};

fn deflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(data).map_err(|e| e.to_string())?;
    z.finish().map_err(|e| e.to_string())
}

/// A compressed stream object with extra dictionary entries.
fn stream(dictionary: &str, data: &[u8]) -> Result<Vec<u8>, String> {
    let packed = deflate(data)?;
    let mut out = format!("<< {dictionary} /Filter /FlateDecode /Length {} >>\nstream\n", packed.len()).into_bytes();
    out.extend_from_slice(&packed);
    out.extend_from_slice(b"\nendstream");
    Ok(out)
}

fn rgb(c: [f64; 4]) -> String { format!("{} {} {}", num(c[0], 4), num(c[1], 4), num(c[2], 4)) }

fn path_ops(out: &mut String, segs: &[Seg]) {
    let n = |v: f64| num(v, 3);
    let mut current = (0.0, 0.0);
    for seg in segs {
        match *seg {
            Seg::Move(x, y) => { let _ = writeln!(out, "{} {} m", n(x), n(y)); current = (x, y); }
            Seg::Line(x, y) => { let _ = writeln!(out, "{} {} l", n(x), n(y)); current = (x, y); }
            Seg::Quad(cx, cy, x, y) => {
                // The quadratic as its cubic.
                let (x0, y0) = current;
                let (c1x, c1y) = (x0 + 2.0 / 3.0 * (cx - x0), y0 + 2.0 / 3.0 * (cy - y0));
                let (c2x, c2y) = (x + 2.0 / 3.0 * (cx - x), y + 2.0 / 3.0 * (cy - y));
                let _ = writeln!(out, "{} {} {} {} {} {} c", n(c1x), n(c1y), n(c2x), n(c2y), n(x), n(y));
                current = (x, y);
            }
            Seg::Cubic(a, b, c, d, x, y) => { let _ = writeln!(out, "{} {} {} {} {} {} c", n(a), n(b), n(c), n(d), n(x), n(y)); current = (x, y); }
            Seg::Close => out.push_str("h\n"),
        }
    }
}

/// The page's transparency states: (stroke alpha, fill alpha) → name.
#[derive(Default)]
struct States(BTreeMap<(u64, u64), usize>);

impl States {
    fn set(&mut self, out: &mut String, stroke: f64, fill: f64) {
        if stroke == 1.0 && fill == 1.0 { return; }
        let next = self.0.len();
        let id = *self.0.entry((stroke.to_bits(), fill.to_bits())).or_insert(next);
        let _ = writeln!(out, "/A{id} gs");
    }
}

/// Colours, alphas and line state of a paint; the operator that paints it.
fn paint_ops(out: &mut String, states: &mut States, paint: &Paint) -> &'static str {
    let fill_alpha = paint.fill.map_or(1.0, |c| c[3]);
    let stroke_alpha = paint.stroke.as_ref().map_or(1.0, |s| s.color[3]);
    states.set(out, stroke_alpha, fill_alpha);
    if let Some(c) = paint.fill { let _ = writeln!(out, "{} rg", rgb(c)); }
    if let Some(s) = &paint.stroke {
        let cap = match s.cap { CapStyle::Butt => 0, CapStyle::Round => 1, CapStyle::Projecting => 2 };
        let join = match s.join { JoinStyle::Miter => 0, JoinStyle::Round => 1, JoinStyle::Bevel => 2 };
        let _ = writeln!(out, "{} RG {} w {cap} J {join} j", rgb(s.color), num(s.width, 4));
        if let Some((offset, list)) = &s.dashes {
            let list: Vec<String> = list.iter().map(|v| num(*v, 4)).collect();
            let _ = writeln!(out, "[{}] {} d", list.join(" "), num(*offset, 4));
        }
    }
    match (paint.fill.is_some(), paint.stroke.is_some()) { (true, true) => "B", (true, false) => "f", (false, true) => "S", (false, false) => "n" }
}

fn clip_ops(out: &mut String, clip: &Option<Rect>) {
    if let Some(r) = clip {
        let _ = writeln!(out, "{} {} {} {} re W n", num(r[0], 3), num(r[1], 3), num(r[2] - r[0], 3), num(r[3] - r[1], 3));
    }
}

/// Appends an object; its number.
fn add(objects: &mut Vec<Vec<u8>>, body: Vec<u8>) -> usize { objects.push(body); objects.len() }

fn hex16(value: u16) -> String { format!("{value:04X}") }

/// The ToUnicode CMap of a subset: each new glyph → its text.
fn to_unicode(map: &BTreeMap<u16, String>) -> String {
    let mut out = String::from("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
        /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
        1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n");
    let entries: Vec<(&u16, &String)> = map.iter().filter(|(_, text)| !text.is_empty()).collect();
    for chunk in entries.chunks(100) {
        let _ = writeln!(out, "{} beginbfchar", chunk.len());
        for (gid, text) in chunk {
            let units: String = text.encode_utf16().map(hex16).collect();
            let _ = writeln!(out, "<{}> <{units}>", hex16(**gid));
        }
        out.push_str("endbfchar\n");
    }
    out.push_str("endcmap\nCMapName currentdict /CMapName get exch defineresource pop\nend\nend\n");
    out
}

/// A subset font's tag: six capitals from its glyph set.
fn tag(seed: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.bytes() { hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3); }
    (0..6).map(|k| char::from(b'A' + ((hash >> (5 * k)) % 26) as u8)).collect()
}

pub(crate) fn write(page: &Page) -> Result<Vec<u8>, String> {
    // Objects 1-4: catalog, pages, page, contents; the rest follow.
    let mut objects: Vec<Vec<u8>> = vec![Vec::new(); 4];
    let mut fonts = String::new();
    let mut xobjects = String::new();
    // Fonts: the subsets, their widths and new glyph numbers.
    let mut subsets: HashMap<usize, sfnt::Subset> = HashMap::new();
    for (index, face) in page.faces.iter().enumerate() {
        if face.used.is_empty() { continue; }
        let subset = sfnt::subset(&face.data, &face.used)?;
        let metrics = sfnt::Sfnt::parse(&face.data)?.metrics()?;
        let base = format!("{}+{}", tag(&format!("{}{:?}", face.postscript, face.used.keys().collect::<Vec<_>>())),
                           face.postscript.replace(' ', ""));
        let file = add(&mut objects, stream(&format!("/Length1 {}", subset.font.len()), &subset.font)?);
        let flags = 32 | if face.italic || metrics.italic_angle != 0.0 { 64 } else { 0 } | u32::from(metrics.fixed);
        let b = metrics.bbox;
        let descriptor = add(&mut objects, format!(
            "<< /Type /FontDescriptor /FontName /{base} /Flags {flags} /FontBBox [{} {} {} {}] /ItalicAngle {} /Ascent {} /Descent {} \
             /CapHeight {} /StemV {} /FontFile2 {file} 0 R >>", num(b[0], 0), num(b[1], 0), num(b[2], 0), num(b[3], 0),
            num(metrics.italic_angle, 3), num(metrics.ascent, 0), num(metrics.descent, 0), num(metrics.cap_height, 0),
            if metrics.weight >= 600 { 120 } else { 80 }).into_bytes());
        let widths: Vec<String> = subset.widths.iter().map(|w| num(*w, 2)).collect();
        let cid = add(&mut objects, format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{base} /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor {descriptor} 0 R /W [0 [{}]] /CIDToGIDMap /Identity >>", widths.join(" ")).into_bytes());
        let text: BTreeMap<u16, String> = face.used.iter().map(|(old, text)| (subset.new_of_old[old], text.clone())).collect();
        let unicode = add(&mut objects, stream("", to_unicode(&text).as_bytes())?);
        let font = add(&mut objects, format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /{base} /Encoding /Identity-H /DescendantFonts [{cid} 0 R] /ToUnicode {unicode} 0 R >>").into_bytes());
        let _ = write!(fonts, " /F{index} {font} 0 R");
        subsets.insert(index, subset);
    }
    // Symbols: forms that paint with the state of their use.
    for (index, symbol) in page.symbols.iter().enumerate() {
        let mut ops = String::new();
        path_ops(&mut ops, &symbol.segs);
        ops.push_str(match (symbol.paint.fill.is_some(), symbol.paint.stroke.is_some()) { (true, true) => "B\n", (true, false) => "f\n", (false, true) => "S\n", _ => "n\n" });
        let reach = symbol.segs.iter().flat_map(|seg| match *seg {
            Seg::Move(x, y) | Seg::Line(x, y) => vec![x, y], Seg::Quad(a, b, x, y) => vec![a, b, x, y],
            Seg::Cubic(a, b, c, d, x, y) => vec![a, b, c, d, x, y], Seg::Close => vec![],
        }).fold(0.0f64, |m, v| m.max(v.abs())) + symbol.paint.stroke.as_ref().map_or(0.0, |s| s.width) + 1.0;
        let form = add(&mut objects, stream(&format!("/Type /XObject /Subtype /Form /BBox [{} {} {} {}]", num(-reach, 3), num(-reach, 3),
                                                    num(reach, 3), num(reach, 3)), ops.as_bytes())?);
        let _ = write!(xobjects, " /S{index} {form} 0 R");
    }
    let mut states = States::default();
    let mut content = String::new();
    let mut images = 0;
    for item in &page.items {
        content.push_str("q\n");
        match item {
            Item::Path { segs, paint, clip } => {
                clip_ops(&mut content, clip);
                let op = paint_ops(&mut content, &mut states, paint);
                path_ops(&mut content, segs);
                content.push_str(op);
                content.push('\n');
            }
            Item::Uses { symbol, at, clip } => {
                clip_ops(&mut content, clip);
                paint_ops(&mut content, &mut states, &page.symbols[*symbol].paint);
                for &(x, y) in at {
                    let _ = writeln!(content, "q 1 0 0 1 {} {} cm /S{symbol} Do Q", num(x, 3), num(y, 3));
                }
            }
            Item::Image { cols, rows, rgba, matrix, opacity, clip } => {
                let pixels = cols * rows;
                let rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
                let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
                let mask = if alpha.iter().any(|&a| a != 255) {
                    let smask = add(&mut objects, stream(&format!("/Type /XObject /Subtype /Image /Width {cols} /Height {rows} \
                        /ColorSpace /DeviceGray /BitsPerComponent 8"), &alpha)?);
                    format!(" /SMask {smask} 0 R")
                } else { String::new() };
                debug_assert_eq!(rgb.len(), 3 * pixels);
                let image = add(&mut objects, stream(&format!("/Type /XObject /Subtype /Image /Width {cols} /Height {rows} \
                    /ColorSpace /DeviceRGB /BitsPerComponent 8{mask}"), &rgb)?);
                let name = format!("I{images}");
                images += 1;
                let _ = write!(xobjects, " /{name} {image} 0 R");
                clip_ops(&mut content, clip);
                states.set(&mut content, *opacity, *opacity);
                // Unit square (row 0 at the top) → pixel grid → page.
                let m = matrix.dot(&M3::from6([*cols as f64, 0.0, 0.0, 0.0, -(*rows as f64), *rows as f64])).0;
                let _ = writeln!(content, "{} {} {} {} {} {} cm /{name} Do", num(m[0][0], 6), num(m[1][0], 6), num(m[0][1], 6),
                                 num(m[1][1], 6), num(m[0][2], 4), num(m[1][2], 4));
            }
            Item::Text { x, y, angle, color, glyphs, rules, clip } => {
                clip_ops(&mut content, clip);
                states.set(&mut content, 1.0, color[3]);
                let (sin, cos) = angle.to_radians().sin_cos();
                let _ = writeln!(content, "{} {} {} {} {} {} cm", num(cos, 6), num(sin, 6), num(-sin, 6), num(cos, 6), num(*x, 3), num(*y, 3));
                let _ = writeln!(content, "{} rg", rgb(*color));
                content.push_str("BT\n");
                let mut start = 0;
                while start < glyphs.len() {
                    let (face, size, gy) = (glyphs[start].face, glyphs[start].size, glyphs[start].y);
                    let end = (start..glyphs.len()).find(|&k| glyphs[k].face != face || glyphs[k].size != size || glyphs[k].y != gy).unwrap_or(glyphs.len());
                    let subset = subsets.get(&face).ok_or("a glyph of a font without a subset")?;
                    let _ = writeln!(content, "/F{face} {} Tf 1 0 0 1 {} {} Tm", num(size, 4), num(glyphs[start].x, 3), num(gy, 3));
                    // TJ: each glyph advances by its width; the layout's
                    // position is the adjustment.
                    let mut tj = String::from("[<");
                    for k in start..end {
                        let gid = subset.new_of_old[&glyphs[k].gid];
                        if k > start {
                            let previous = subset.new_of_old[&glyphs[k - 1].gid];
                            let adjust = subset.widths[previous as usize] - (glyphs[k].x - glyphs[k - 1].x) * 1000.0 / size;
                            if adjust.abs() > 0.001 { let _ = write!(tj, "> {} <", num(adjust, 3)); }
                        }
                        tj.push_str(&hex16(gid));
                    }
                    tj.push_str(">] TJ");
                    content.push_str(&tj);
                    content.push('\n');
                    start = end;
                }
                content.push_str("ET\n");
                for r in rules {
                    let _ = writeln!(content, "{} {} {} {} re f", num(r[0], 3), num(r[1], 3), num(r[2] - r[0], 3), num(r[3] - r[1], 3));
                }
            }
        }
        content.push_str("Q\n");
    }
    let mut gstates = String::new();
    for ((stroke, fill), id) in &states.0 {
        let _ = write!(gstates, " /A{id} << /Type /ExtGState /CA {} /ca {} >>", num(f64::from_bits(*stroke), 4), num(f64::from_bits(*fill), 4));
    }
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    objects[1] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
    objects[2] = format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Contents 4 0 R /Resources << /ProcSet [/PDF /Text /ImageB /ImageC] \
        /ExtGState <<{gstates} >> /Font <<{fonts} >> /XObject <<{xobjects} >> >> >>", num(page.width, 4), num(page.height, 4)).into_bytes();
    objects[3] = stream("", content.as_bytes())?;
    let info = add(&mut objects, b"<< /Producer (Zou lab control) /Creator (zlc-plot) >>".to_vec());
    let mut out = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
    for offset in offsets { out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes()); }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R /Info {info} 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1).as_bytes());
    Ok(out)
}
