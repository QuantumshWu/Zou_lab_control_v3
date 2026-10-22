//! TrueType fonts cut to the glyphs a figure uses, for PDF embedding:
//! `.notdef` stays glyph 0 and the used glyphs (with the components of
//! composite ones) follow in their original order; a fresh cmap maps the
//! characters they stand for, and the tables a renderer needs are kept.
use std::collections::{BTreeMap, BTreeSet};

fn u16_at(d: &[u8], at: usize) -> Result<u16, String> {
    d.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]])).ok_or_else(|| "truncated font table".into())
}
fn i16_at(d: &[u8], at: usize) -> Result<i16, String> { u16_at(d, at).map(|v| v as i16) }
fn u32_at(d: &[u8], at: usize) -> Result<u32, String> {
    d.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| "truncated font table".into())
}

/// The sum of a table's big-endian words (zero padded).
fn checksum(data: &[u8]) -> u32 {
    data.chunks(4).fold(0u32, |sum, word| {
        let mut b = [0u8; 4];
        b[..word.len()].copy_from_slice(word);
        sum.wrapping_add(u32::from_be_bytes(b))
    })
}

/// An sfnt's table directory over its bytes (TrueType outlines only).
pub(crate) struct Sfnt<'a> { tables: BTreeMap<[u8; 4], &'a [u8]> }

impl<'a> Sfnt<'a> {
    pub(crate) fn parse(data: &'a [u8]) -> Result<Sfnt<'a>, String> {
        let version = u32_at(data, 0)?;
        if version != 0x0001_0000 && version != u32::from_be_bytes(*b"true") {
            return Err(format!("not a TrueType font (sfnt version {version:#x})"));
        }
        let count = u16_at(data, 4)? as usize;
        let mut tables = BTreeMap::new();
        for k in 0..count {
            let record = 12 + 16 * k;
            let tag: [u8; 4] = data.get(record..record + 4).ok_or("truncated table directory")?.try_into().unwrap();
            let (offset, length) = (u32_at(data, record + 8)? as usize, u32_at(data, record + 12)? as usize);
            tables.insert(tag, data.get(offset..offset + length).ok_or("a table outside the font")?);
        }
        for tag in [b"head", b"hhea", b"hmtx", b"maxp", b"loca", b"glyf"] {
            if !tables.contains_key(tag) { return Err(format!("font without a {} table", String::from_utf8_lossy(tag))); }
        }
        Ok(Sfnt { tables })
    }

    fn table(&self, tag: &[u8; 4]) -> Option<&'a [u8]> { self.tables.get(tag).copied() }

    pub(crate) fn units_per_em(&self) -> Result<u16, String> { u16_at(self.table(b"head").unwrap(), 18) }

    fn num_glyphs(&self) -> Result<usize, String> { Ok(u16_at(self.table(b"maxp").unwrap(), 4)? as usize) }

    /// The descriptor values a PDF font needs, in 1000ths of the em.
    pub(crate) fn metrics(&self) -> Result<Metrics, String> {
        let head = self.table(b"head").unwrap();
        let hhea = self.table(b"hhea").unwrap();
        let scale = 1000.0 / f64::from(self.units_per_em()?);
        let em = |v: i16| (f64::from(v) * scale).round();
        let bbox = [em(i16_at(head, 36)?), em(i16_at(head, 38)?), em(i16_at(head, 40)?), em(i16_at(head, 42)?)];
        let (mut ascent, mut descent) = (em(i16_at(hhea, 4)?), em(i16_at(hhea, 6)?));
        let mut cap_height = bbox[3];
        let mut weight = 400;
        if let Some(os2) = self.table(b"OS/2") {
            if os2.len() >= 72 { ascent = em(i16_at(os2, 68)?); descent = em(i16_at(os2, 70)?); }
            if u16_at(os2, 0)? >= 2 && os2.len() >= 90 { cap_height = em(i16_at(os2, 88)?); }
            weight = u16_at(os2, 4)?;
        }
        let italic_angle = match self.table(b"post") { Some(post) => f64::from(u32_at(post, 4)? as i32) / 65536.0, None => 0.0 };
        let fixed = self.table(b"post").is_some_and(|post| u32_at(post, 12).is_ok_and(|v| v != 0));
        Ok(Metrics { bbox, ascent, descent, cap_height, italic_angle, weight, fixed })
    }

    /// One glyph's outline bytes.
    fn glyph(&self, gid: usize) -> Result<&'a [u8], String> {
        let (start, end) = self.loca(gid)?;
        self.table(b"glyf").unwrap().get(start..end).ok_or_else(|| "a glyph outside glyf".into())
    }

    fn loca(&self, gid: usize) -> Result<(usize, usize), String> {
        let loca = self.table(b"loca").unwrap();
        if i16_at(self.table(b"head").unwrap(), 50)? == 0 {
            Ok((2 * u16_at(loca, 2 * gid)? as usize, 2 * u16_at(loca, 2 * gid + 2)? as usize))
        } else {
            Ok((u32_at(loca, 4 * gid)? as usize, u32_at(loca, 4 * gid + 4)? as usize))
        }
    }

    /// (advance, left side bearing) of a glyph, in font units.
    fn horizontal(&self, gid: usize) -> Result<(u16, i16), String> {
        let hmtx = self.table(b"hmtx").unwrap();
        let long = u16_at(self.table(b"hhea").unwrap(), 34)? as usize;
        if gid < long { return Ok((u16_at(hmtx, 4 * gid)?, i16_at(hmtx, 4 * gid + 2)?)); }
        Ok((u16_at(hmtx, 4 * (long - 1))?, i16_at(hmtx, 4 * long + 2 * (gid - long))?))
    }

    /// A glyph's advance in 1000ths of the em.
    pub(crate) fn advance(&self, gid: u16) -> Result<f64, String> {
        Ok(f64::from(self.horizontal(gid as usize)?.0) * 1000.0 / f64::from(self.units_per_em()?))
    }
}

pub(crate) struct Metrics {
    pub bbox: [f64; 4],
    pub ascent: f64,
    pub descent: f64,
    pub cap_height: f64,
    pub italic_angle: f64,
    pub weight: u16,
    pub fixed: bool,
}

/// A composite glyph's component records: (offset of the glyph index, glyph index).
fn components(glyph: &[u8]) -> Result<Vec<(usize, u16)>, String> {
    let mut out = Vec::new();
    if glyph.len() < 10 || i16_at(glyph, 0)? >= 0 { return Ok(out); }
    let mut at = 10;
    loop {
        let flags = u16_at(glyph, at)?;
        out.push((at + 2, u16_at(glyph, at + 2)?));
        at += 4 + if flags & 0x1 != 0 { 4 } else { 2 };
        at += if flags & 0x8 != 0 { 2 } else if flags & 0x40 != 0 { 4 } else if flags & 0x80 != 0 { 8 } else { 0 };
        if flags & 0x20 == 0 { return Ok(out); }
    }
}

/// A font cut to some glyphs.
pub(crate) struct Subset {
    /// The TrueType file.
    pub font: Vec<u8>,
    /// New glyph index of each used original one.
    pub new_of_old: BTreeMap<u16, u16>,
    /// Advance of each new glyph, in 1000ths of the em.
    pub widths: Vec<f64>,
}

/// Cuts `data` to the glyphs `used` names (original index → the characters it
/// stands for, mapped by the new cmap).
pub(crate) fn subset(data: &[u8], used: &BTreeMap<u16, String>) -> Result<Subset, String> {
    let font = Sfnt::parse(data)?;
    let count = font.num_glyphs()?;
    // The closure under composite components, in original order.
    let mut keep: BTreeSet<u16> = BTreeSet::from([0]);
    let mut pending: Vec<u16> = used.keys().copied().collect();
    while let Some(gid) = pending.pop() {
        if gid as usize >= count { return Err(format!("glyph {gid} beyond the font's {count}")); }
        if !keep.insert(gid) && gid != 0 { continue; }
        for (_, component) in components(font.glyph(gid as usize)?)? {
            if !keep.contains(&component) { pending.push(component); }
        }
    }
    let new_of_old: BTreeMap<u16, u16> = keep.iter().enumerate().map(|(new, &old)| (old, new as u16)).collect();
    let mut glyf = Vec::new();
    let mut loca = Vec::with_capacity(4 * (keep.len() + 1));
    let mut hmtx = Vec::with_capacity(4 * keep.len());
    let mut widths = Vec::with_capacity(keep.len());
    for &old in &keep {
        loca.extend_from_slice(&(glyf.len() as u32).to_be_bytes());
        let mut glyph = font.glyph(old as usize)?.to_vec();
        for (at, component) in components(&glyph)? {
            glyph[at..at + 2].copy_from_slice(&new_of_old[&component].to_be_bytes());
        }
        glyf.extend_from_slice(&glyph);
        while glyf.len() % 4 != 0 { glyf.push(0); }
        let (advance, lsb) = font.horizontal(old as usize)?;
        hmtx.extend_from_slice(&advance.to_be_bytes());
        hmtx.extend_from_slice(&lsb.to_be_bytes());
        widths.push(font.advance(old)?);
    }
    loca.extend_from_slice(&(glyf.len() as u32).to_be_bytes());
    let n = keep.len() as u16;
    let mut head = font.table(b"head").unwrap().to_vec();
    head[8..12].copy_from_slice(&[0; 4]);
    head[50..52].copy_from_slice(&1i16.to_be_bytes());
    let mut hhea = font.table(b"hhea").unwrap().to_vec();
    hhea[34..36].copy_from_slice(&n.to_be_bytes());
    let mut maxp = font.table(b"maxp").unwrap().to_vec();
    maxp[4..6].copy_from_slice(&n.to_be_bytes());
    // post format 3: no glyph names.
    let mut post = font.table(b"post").map_or_else(|| vec![0; 32], |post| post[..32.min(post.len())].to_vec());
    post.resize(32, 0);
    post[0..4].copy_from_slice(&0x0003_0000u32.to_be_bytes());
    let chars: BTreeMap<u32, u16> = used.iter().rev()
        .flat_map(|(old, text)| text.chars().map(move |c| (u32::from(c), *old)))
        .map(|(code, old)| (code, new_of_old[&old])).collect();
    let mut tables: Vec<([u8; 4], Vec<u8>)> = vec![
        (*b"cmap", cmap(&chars)), (*b"glyf", glyf), (*b"head", head), (*b"hhea", hhea), (*b"hmtx", hmtx),
        (*b"loca", loca), (*b"maxp", maxp), (*b"post", post),
    ];
    for tag in [b"OS/2", b"cvt ", b"fpgm", b"gasp", b"name", b"prep"] {
        if let Some(table) = font.table(tag) { tables.push((*tag, table.to_vec())); }
    }
    tables.sort_by_key(|(tag, _)| *tag);
    let mut font = sfnt(&tables);
    let adjustment = 0xB1B0_AFBAu32.wrapping_sub(checksum(&font));
    let head_at = head_offset(&font)?;
    font[head_at + 8..head_at + 12].copy_from_slice(&adjustment.to_be_bytes());
    Ok(Subset { font, new_of_old, widths })
}

fn head_offset(font: &[u8]) -> Result<usize, String> {
    let count = u16_at(font, 4)? as usize;
    (0..count).find(|k| &font[12 + 16 * k..16 + 16 * k] == b"head")
        .map(|k| u32_at(font, 12 + 16 * k + 8).unwrap() as usize).ok_or_else(|| "no head".into())
}

/// An sfnt of tables sorted by tag.
fn sfnt(tables: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let n = tables.len();
    let power = (n as f64).log2().floor() as u32;
    let range = (1u16 << power) * 16;
    let mut out = Vec::new();
    out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    out.extend_from_slice(&(n as u16).to_be_bytes());
    out.extend_from_slice(&range.to_be_bytes());
    out.extend_from_slice(&(power as u16).to_be_bytes());
    out.extend_from_slice(&((n as u16) * 16 - range).to_be_bytes());
    let mut offset = 12 + 16 * n;
    for (tag, data) in tables {
        out.extend_from_slice(tag);
        out.extend_from_slice(&checksum(data).to_be_bytes());
        out.extend_from_slice(&(offset as u32).to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        offset += data.len().div_ceil(4) * 4;
    }
    for (_, data) in tables {
        out.extend_from_slice(data);
        while out.len() % 4 != 0 { out.push(0); }
    }
    out
}

/// A cmap of (3, 1) format 4 for the BMP and, when needed, (3, 10) format 12.
fn cmap(chars: &BTreeMap<u32, u16>) -> Vec<u8> {
    let bmp: Vec<(u16, u16)> = chars.iter().filter(|(c, _)| **c < 0xFFFF).map(|(&c, &g)| (c as u16, g)).collect();
    let mut format4 = Vec::new();
    let segments = bmp.len() + 1;
    let power = (segments as f64).log2().floor() as u32;
    let search = (1u16 << power) * 2;
    format4.extend_from_slice(&4u16.to_be_bytes());
    format4.extend_from_slice(&((16 + 8 * segments) as u16).to_be_bytes());
    format4.extend_from_slice(&0u16.to_be_bytes());
    format4.extend_from_slice(&((2 * segments) as u16).to_be_bytes());
    format4.extend_from_slice(&search.to_be_bytes());
    format4.extend_from_slice(&(power as u16).to_be_bytes());
    format4.extend_from_slice(&((2 * segments) as u16 - search).to_be_bytes());
    for &(c, _) in &bmp { format4.extend_from_slice(&c.to_be_bytes()); }
    format4.extend_from_slice(&0xFFFFu16.to_be_bytes());
    format4.extend_from_slice(&0u16.to_be_bytes());
    for &(c, _) in &bmp { format4.extend_from_slice(&c.to_be_bytes()); }
    format4.extend_from_slice(&0xFFFFu16.to_be_bytes());
    for &(c, g) in &bmp { format4.extend_from_slice(&g.wrapping_sub(c).to_be_bytes()); }
    format4.extend_from_slice(&1u16.to_be_bytes());
    for _ in 0..segments { format4.extend_from_slice(&0u16.to_be_bytes()); }
    let wide: Vec<(u32, u16)> = chars.iter().filter(|(c, _)| **c > 0xFFFF).map(|(&c, &g)| (c, g)).collect();
    let mut subtables = vec![(1u16, format4)];
    if !wide.is_empty() {
        let mut format12 = Vec::new();
        format12.extend_from_slice(&12u16.to_be_bytes());
        format12.extend_from_slice(&0u16.to_be_bytes());
        format12.extend_from_slice(&((16 + 12 * wide.len()) as u32).to_be_bytes());
        format12.extend_from_slice(&0u32.to_be_bytes());
        format12.extend_from_slice(&(wide.len() as u32).to_be_bytes());
        for &(c, g) in &wide {
            format12.extend_from_slice(&c.to_be_bytes());
            format12.extend_from_slice(&c.to_be_bytes());
            format12.extend_from_slice(&u32::from(g).to_be_bytes());
        }
        subtables.push((10, format12));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&0u16.to_be_bytes());
    out.extend_from_slice(&(subtables.len() as u16).to_be_bytes());
    let mut offset = 4 + 8 * subtables.len();
    for (encoding, table) in &subtables {
        out.extend_from_slice(&3u16.to_be_bytes());
        out.extend_from_slice(&encoding.to_be_bytes());
        out.extend_from_slice(&(offset as u32).to_be_bytes());
        offset += table.len();
    }
    for (_, table) in subtables { out.extend_from_slice(&table); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cut of DejaVu Sans keeps `.notdef`, the used glyphs and the
    /// components of composite ones (é is e + acute), renumbered, with a cmap
    /// and metrics a reader finds again.
    #[test]
    fn a_subset_reads_back() {
        let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/fonts/DejaVuSans.ttf")).unwrap();
        let font = Sfnt::parse(&data).unwrap();
        let cmap_of = |c: char| -> u16 {
            let face = zlc_text::FT2Font::new(data.clone(), 8, Vec::new(), 0).unwrap();
            face.get_char_index(u32::from(c), false).unwrap() as u16
        };
        let used: BTreeMap<u16, String> = ['A', 'é', '±'].iter().map(|&c| (cmap_of(c), c.to_string())).collect();
        let cut = subset(&data, &used).unwrap();
        let back = Sfnt::parse(&cut.font).unwrap();
        let count = back.num_glyphs().unwrap();
        assert!(count > used.len() + 1, "components kept: {count} glyphs");
        assert_eq!(cut.widths.len(), count);
        for gid in 0..count {
            for (_, component) in components(back.glyph(gid).unwrap()).unwrap() { assert!((component as usize) < count); }
        }
        for (old, text) in &used {
            let new = cut.new_of_old[old] as usize;
            assert_eq!(back.horizontal(new).unwrap(), font.horizontal(*old as usize).unwrap(), "{text}: advance and bearing");
            let face = zlc_text::FT2Font::new(cut.font.clone(), 8, Vec::new(), 0).unwrap();
            assert_eq!(face.get_char_index(u32::from(text.chars().next().unwrap()), false).unwrap() as usize, new, "{text}: cmap");
        }
    }
}
