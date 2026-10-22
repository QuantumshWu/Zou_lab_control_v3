//! The TeX box model of `_mathtext.py` (`Node` ... `ship`), operation for operation:
//! the same f64 expressions in the same order, Python's `max`/`min`/`round`, and the
//! mutations `ship` makes (a Vlist widens its child lists while shipping).

use crate::Error;
use crate::fonts::{FontMetrics, FontName, Fontset, InfoId};

pub(crate) const SHRINK_FACTOR: f64 = 0.7;
pub(crate) const NUM_SIZE_LEVELS: u32 = 6;

/// Python's `max(a, b)`: the first argument unless the second is greater.
pub(crate) fn pymax(a: f64, b: f64) -> f64 { if b > a { b } else { a } }
/// Python's `min(a, b)`.
pub(crate) fn pymin(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

#[derive(Clone, Copy, Debug)]
pub(crate) struct GlueSpec { pub width: f64, pub stretch: f64, pub stretch_order: usize, pub shrink: f64, pub shrink_order: usize }

impl GlueSpec {
    pub(crate) const FILL: Self = Self { width: 0.0, stretch: 1.0, stretch_order: 2, shrink: 0.0, shrink_order: 0 };
    pub(crate) const SS: Self = Self { width: 0.0, stretch: 1.0, stretch_order: 1, shrink: -1.0, shrink_order: 1 };
}

pub(crate) struct CharNode {
    pub c: String,
    pub font: FontName,
    pub font_class: FontName,
    pub fontsize: f64,
    pub dpi: f64,
    pub metrics: FontMetrics,
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub size: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ListKind { H, V }

pub(crate) struct ListNode {
    pub kind: ListKind,
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub size: u32,
    pub shift_amount: f64,
    pub children: Vec<Node>,
    pub glue_set: f64,
    pub glue_sign: i32,
    pub glue_order: usize,
    /// `Hlist.function_name`, set by `Parser.function`.
    pub function_name: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BoxKind { Plain, Rule }

pub(crate) struct BoxNode { pub kind: BoxKind, pub width: f64, pub height: f64, pub depth: f64, pub size: u32 }

pub(crate) enum Node {
    Char(CharNode),
    List(ListNode),
    Box(BoxNode),
    Kern { width: f64, size: u32 },
    Glue { spec: GlueSpec, size: u32 },
}

impl Node {
    pub(crate) fn kern(width: f64) -> Self { Node::Kern { width, size: 0 } }
    pub(crate) fn glue(spec: GlueSpec) -> Self { Node::Glue { spec, size: 0 } }
    pub(crate) fn plain_box(width: f64, height: f64, depth: f64) -> Self {
        Node::Box(BoxNode { kind: BoxKind::Plain, width, height, depth, size: 0 })
    }
    pub(crate) fn hbox(width: f64) -> Self { Self::plain_box(width, 0.0, 0.0) }
    pub(crate) fn vbox(height: f64, depth: f64) -> Self { Self::plain_box(0.0, height, depth) }
    /// `Hrule(state, thickness)`.
    pub(crate) fn hrule(thickness: f64) -> Self {
        let height = thickness * 0.5;
        Node::Box(BoxNode { kind: BoxKind::Rule, width: f64::INFINITY, height, depth: height, size: 0 })
    }

    /// `Char(c, state)`.
    pub(crate) fn char(fonts: &mut Fontset<'_>, c: &str, font: FontName, font_class: FontName, fontsize: f64, dpi: f64)
        -> Result<Self, Error> {
        let metrics = fonts.metrics(font, font_class, c, fontsize, dpi)?;
        let mut node = CharNode { c: c.to_string(), font, font_class, fontsize, dpi, metrics,
            width: 0.0, height: 0.0, depth: 0.0, size: 0 };
        node.apply_metrics();
        Ok(Node::Char(node))
    }

    pub(crate) fn width(&self) -> f64 {
        match self {
            Node::Char(c) => c.width,
            Node::List(l) => l.width,
            Node::Box(b) => b.width,
            Node::Kern { width, .. } => *width,
            Node::Glue { .. } => unreachable!("Glue has no width attribute"),
        }
    }

    /// `node.height`; Kern's is its class attribute 0, Glue has none.
    pub(crate) fn height(&self) -> Result<f64, Error> {
        Ok(match self {
            Node::Char(c) => c.height,
            Node::List(l) => l.height,
            Node::Box(b) => b.height,
            Node::Kern { .. } => 0.0,
            Node::Glue { .. } => return Err(Error::MatplotlibRaises("AttributeError: height of a Glue node".into())),
        })
    }

    pub(crate) fn depth(&self) -> Result<f64, Error> {
        Ok(match self {
            Node::Char(c) => c.depth,
            Node::List(l) => l.depth,
            Node::Box(b) => b.depth,
            Node::Kern { .. } => 0.0,
            Node::Glue { .. } => return Err(Error::MatplotlibRaises("AttributeError: depth of a Glue node".into())),
        })
    }

    /// `getattr(node, 'shift_amount', 0.)`.
    pub(crate) fn shift_amount(&self) -> f64 { if let Node::List(l) = self { l.shift_amount } else { 0.0 } }

    pub(crate) fn shrink(&mut self) {
        match self {
            Node::Char(c) => {
                c.size += 1;
                if c.size < NUM_SIZE_LEVELS {
                    c.fontsize *= SHRINK_FACTOR;
                    c.width *= SHRINK_FACTOR;
                    c.height *= SHRINK_FACTOR;
                    c.depth *= SHRINK_FACTOR;
                }
            }
            Node::List(l) => l.shrink(),
            Node::Box(b) => {
                b.size += 1;
                if b.size < NUM_SIZE_LEVELS {
                    b.width *= SHRINK_FACTOR;
                    b.height *= SHRINK_FACTOR;
                    b.depth *= SHRINK_FACTOR;
                }
            }
            Node::Kern { width, size } => {
                *size += 1;
                if *size < NUM_SIZE_LEVELS { *width *= SHRINK_FACTOR }
            }
            Node::Glue { spec, size } => {
                *size += 1;
                if *size < NUM_SIZE_LEVELS { spec.width *= SHRINK_FACTOR }
            }
        }
    }

    /// `get_kerning(next)`: a Char's advance correction plus the font kerning.
    fn get_kerning(&self, next: Option<&Node>, fonts: &mut Fontset<'_>) -> Result<f64, Error> {
        let Node::Char(c) = self else { return Ok(0.0) };
        let advance = c.metrics.advance - c.width;
        let mut kern = 0.0;
        if let Some(Node::Char(n)) = next {
            kern = fonts.get_kern(c.font, c.font_class, &c.c, c.fontsize, n.font, n.font_class, &n.c, n.fontsize, c.dpi)?;
        }
        Ok(advance + kern)
    }
}

impl CharNode {
    /// `Char._update_metrics` from the stored metrics.
    fn apply_metrics(&mut self) {
        let metrics = self.metrics;
        self.width = if self.c == " " { metrics.advance } else { metrics.width };
        self.height = metrics.iceberg;
        self.depth = -(metrics.iceberg - metrics.height);
    }
}

impl ListNode {
    /// `List.shrink`.
    pub(crate) fn shrink(&mut self) {
        for child in &mut self.children { child.shrink() }
        self.size += 1;
        if self.size < NUM_SIZE_LEVELS {
            self.width *= SHRINK_FACTOR;
            self.height *= SHRINK_FACTOR;
            self.depth *= SHRINK_FACTOR;
        }
        if self.size < NUM_SIZE_LEVELS {
            self.shift_amount *= SHRINK_FACTOR;
            self.glue_set *= SHRINK_FACTOR;
        }
    }

    fn new(kind: ListKind, children: Vec<Node>) -> Self {
        Self { kind, width: 0.0, height: 0.0, depth: 0.0, size: 0, shift_amount: 0.0, children,
            glue_set: 0.0, glue_sign: 0, glue_order: 0, function_name: None }
    }

    /// `Hlist(elements, w, m, do_kern)`.
    pub(crate) fn hlist(children: Vec<Node>, do_kern: bool, fonts: &mut Fontset<'_>) -> Result<Self, Error> {
        let mut list = Self::new(ListKind::H, children);
        if do_kern { list.kern(fonts)? }
        list.hpack(0.0, false);
        Ok(list)
    }

    /// `Vlist(elements)`.
    pub(crate) fn vlist(children: Vec<Node>) -> Result<Self, Error> {
        let mut list = Self::new(ListKind::V, children);
        list.vpack(0.0, false, f64::INFINITY)?;
        Ok(list)
    }

    /// `HCentered(elements)`.
    pub(crate) fn hcentered(elements: Vec<Node>, fonts: &mut Fontset<'_>) -> Result<Self, Error> {
        let mut children = Vec::with_capacity(elements.len() + 2);
        children.push(Node::glue(GlueSpec::SS));
        children.extend(elements);
        children.push(Node::glue(GlueSpec::SS));
        Self::hlist(children, false, fonts)
    }

    /// `Hlist.kern`.
    pub(crate) fn kern(&mut self, fonts: &mut Fontset<'_>) -> Result<(), Error> {
        let count = self.children.len();
        if count == 0 { return Ok(()) }
        let mut distances = Vec::with_capacity(count);
        for i in 0..count {
            let next = if i < count - 1 { Some(&self.children[i + 1]) } else { None };
            distances.push(self.children[i].get_kerning(next, fonts)?);
        }
        let old = std::mem::take(&mut self.children);
        for (child, distance) in old.into_iter().zip(distances) {
            self.children.push(child);
            if distance != 0.0 { self.children.push(Node::kern(distance)) }
        }
        Ok(())
    }

    /// `Hlist.hpack(w, m)`; `exactly` is m == 'exactly'.
    pub(crate) fn hpack(&mut self, w: f64, exactly: bool) {
        let (mut h, mut d, mut x) = (0.0f64, 0.0f64, 0.0f64);
        let mut total_stretch = [0.0f64; 4];
        let mut total_shrink = [0.0f64; 4];
        for p in &self.children {
            match p {
                Node::Char(c) => {
                    x += c.width;
                    h = pymax(h, c.height);
                    d = pymax(d, c.depth);
                }
                Node::List(_) | Node::Box(_) => {
                    x += p.width();
                    let (height, depth) = match p { Node::List(l) => (l.height, l.depth), Node::Box(b) => (b.height, b.depth), _ => unreachable!() };
                    if !height.is_infinite() && !depth.is_infinite() {
                        let s = p.shift_amount();
                        h = pymax(h, height - s);
                        d = pymax(d, depth + s);
                    }
                }
                Node::Glue { spec, .. } => {
                    x += spec.width;
                    total_stretch[spec.stretch_order] += spec.stretch;
                    total_shrink[spec.shrink_order] += spec.shrink;
                }
                Node::Kern { width, .. } => x += width,
            }
        }
        self.height = h;
        self.depth = d;
        let mut w = w;
        if !exactly { w += x }
        self.width = w;
        let x = w - x;
        if x == 0.0 {
            self.glue_sign = 0;
            self.glue_order = 0;
            return;
        }
        if x > 0.0 { self.set_glue(x, 1, &total_stretch) } else { self.set_glue(x, -1, &total_shrink) }
    }

    /// `Vlist.vpack(h, m, l)`.
    pub(crate) fn vpack(&mut self, h: f64, exactly: bool, l: f64) -> Result<(), Error> {
        let (mut w, mut d, mut x) = (0.0f64, 0.0f64, 0.0f64);
        let mut total_stretch = [0.0f64; 4];
        let mut total_shrink = [0.0f64; 4];
        for p in &self.children {
            match p {
                Node::List(_) | Node::Box(_) => {
                    let (height, depth, width) = match p {
                        Node::List(n) => (n.height, n.depth, n.width),
                        Node::Box(b) => (b.height, b.depth, b.width),
                        _ => unreachable!(),
                    };
                    x += d + height;
                    d = depth;
                    if !width.is_infinite() {
                        let s = p.shift_amount();
                        w = pymax(w, width + s);
                    }
                }
                Node::Glue { spec, .. } => {
                    x += d;
                    d = 0.0;
                    x += spec.width;
                    total_stretch[spec.stretch_order] += spec.stretch;
                    total_shrink[spec.shrink_order] += spec.shrink;
                }
                Node::Kern { width, .. } => {
                    x += d + width;
                    d = 0.0;
                }
                Node::Char(_) => return Err(Error::MatplotlibRaises("RuntimeError: Internal mathtext error: Char node found in Vlist".into())),
            }
        }
        self.width = w;
        if d > l {
            x += d - l;
            self.depth = l;
        } else {
            self.depth = d;
        }
        let mut h = h;
        if !exactly { h += x }
        self.height = h;
        let x = h - x;
        if x == 0.0 {
            self.glue_sign = 0;
            self.glue_order = 0;
            return Ok(());
        }
        if x > 0.0 { self.set_glue(x, 1, &total_stretch) } else { self.set_glue(x, -1, &total_shrink) }
        Ok(())
    }

    fn set_glue(&mut self, x: f64, sign: i32, totals: &[f64; 4]) {
        let o = (0..4).rev().find(|&i| totals[i] != 0.0).unwrap_or(0);
        self.glue_order = o;
        self.glue_sign = sign;
        if totals[o] != 0.0 { self.glue_set = x / totals[o] } else { self.glue_sign = 0 }
    }

    pub(crate) fn into_node(self) -> Node { Node::List(self) }
}

/// One shipped glyph: (ox, oy, info).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShippedGlyph { pub ox: f64, pub oy: f64, pub info: InfoId }

/// `Output`: positioned glyphs and rectangles (x1, y1, x2, y2).
#[derive(Default)]
pub(crate) struct Output { pub glyphs: Vec<ShippedGlyph>, pub rects: Vec<[f64; 4]> }

fn clamp(value: f64) -> f64 { if value < -1e9 { -1e9 } else if value > 1e9 { 1e9 } else { value } }

/// Python's `round(x)` of a float (half to even), as the float of the int it returns.
fn pyround(value: f64) -> f64 { value.round_ties_even() }

struct Shipper<'s, 'f> {
    fonts: &'s mut Fontset<'f>,
    output: Output,
    cur_v: f64,
    cur_h: f64,
    off_h: f64,
    off_v: f64,
}

/// `ship(box, xy)`.
pub(crate) fn ship(root: &mut ListNode, xy: (f64, f64), fonts: &mut Fontset<'_>) -> Result<Output, Error> {
    let (ox, oy) = xy;
    let mut shipper = Shipper { fonts, output: Output::default(), cur_v: 0.0, cur_h: 0.0, off_h: ox, off_v: oy + root.height };
    debug_assert_eq!(root.kind, ListKind::H);
    shipper.hlist_out(root)?;
    Ok(shipper.output)
}

impl Shipper<'_, '_> {
    fn render_char(&mut self, c: &CharNode, x: f64, y: f64) -> Result<(), Error> {
        let info = self.fonts.get_info(c.font, c.font_class, &c.c, c.fontsize, c.dpi)?;
        self.output.glyphs.push(ShippedGlyph { ox: x, oy: y, info });
        Ok(())
    }

    fn hlist_out(&mut self, list: &mut ListNode) -> Result<(), Error> {
        let mut cur_g = 0.0f64;
        let mut cur_glue = 0.0f64;
        let glue_order = list.glue_order;
        let glue_sign = list.glue_sign;
        let base_line = self.cur_v;
        let (box_height, box_depth, glue_set) = (list.height, list.depth, list.glue_set);
        for p in &mut list.children {
            match p {
                Node::Char(c) => {
                    let (x, y) = (self.cur_h + self.off_h, self.cur_v + self.off_v);
                    self.render_char(c, x, y)?;
                    self.cur_h += c.width;
                }
                Node::Kern { width, .. } => self.cur_h += *width,
                Node::List(child) => {
                    if child.children.is_empty() {
                        self.cur_h += child.width;
                    } else {
                        let edge = self.cur_h;
                        self.cur_v = base_line + child.shift_amount;
                        match child.kind {
                            ListKind::H => self.hlist_out(child)?,
                            ListKind::V => self.vlist_out(child)?,
                        }
                        self.cur_h = edge + child.width;
                        self.cur_v = base_line;
                    }
                }
                Node::Box(b) => {
                    let mut rule_height = b.height;
                    let mut rule_depth = b.depth;
                    let rule_width = b.width;
                    if rule_height.is_infinite() { rule_height = box_height }
                    if rule_depth.is_infinite() { rule_depth = box_depth }
                    if rule_height > 0.0 && rule_width > 0.0 {
                        self.cur_v = base_line + rule_depth;
                        if b.kind == BoxKind::Rule {
                            let (x, y) = (self.cur_h + self.off_h, self.cur_v + self.off_v);
                            self.output.rects.push([x, y, x + rule_width, y + rule_height]);
                        }
                        self.cur_v = base_line;
                    }
                    self.cur_h += rule_width;
                }
                Node::Glue { spec, .. } => {
                    let mut rule_width = spec.width - cur_g;
                    if glue_sign != 0 {
                        if glue_sign == 1 {
                            if spec.stretch_order == glue_order {
                                cur_glue += spec.stretch;
                                cur_g = pyround(clamp(glue_set * cur_glue));
                            }
                        } else if spec.shrink_order == glue_order {
                            cur_glue += spec.shrink;
                            cur_g = pyround(clamp(glue_set * cur_glue));
                        }
                    }
                    rule_width += cur_g;
                    self.cur_h += rule_width;
                }
            }
        }
        Ok(())
    }

    fn vlist_out(&mut self, list: &mut ListNode) -> Result<(), Error> {
        let mut cur_g = 0.0f64;
        let mut cur_glue = 0.0f64;
        let glue_order = list.glue_order;
        let glue_sign = list.glue_sign;
        let left_edge = self.cur_h;
        self.cur_v -= list.height;
        let (box_width, glue_set) = (list.width, list.glue_set);
        for p in &mut list.children {
            match p {
                Node::Kern { width, .. } => self.cur_v += *width,
                Node::List(child) => {
                    if child.children.is_empty() {
                        self.cur_v += child.height + child.depth;
                    } else {
                        self.cur_v += child.height;
                        self.cur_h = left_edge + child.shift_amount;
                        let save_v = self.cur_v;
                        child.width = box_width;
                        match child.kind {
                            ListKind::H => self.hlist_out(child)?,
                            ListKind::V => self.vlist_out(child)?,
                        }
                        self.cur_v = save_v + child.depth;
                        self.cur_h = left_edge;
                    }
                }
                Node::Box(b) => {
                    let mut rule_height = b.height;
                    let rule_depth = b.depth;
                    let mut rule_width = b.width;
                    if rule_width.is_infinite() { rule_width = box_width }
                    rule_height += rule_depth;
                    if rule_height > 0.0 && rule_depth > 0.0 {
                        self.cur_v += rule_height;
                        if b.kind == BoxKind::Rule {
                            let (x, y) = (self.cur_h + self.off_h, self.cur_v + self.off_v);
                            self.output.rects.push([x, y, x + rule_width, y + rule_height]);
                        }
                    }
                }
                Node::Glue { spec, .. } => {
                    let mut rule_height = spec.width - cur_g;
                    if glue_sign != 0 {
                        if glue_sign == 1 {
                            if spec.stretch_order == glue_order {
                                cur_glue += spec.stretch;
                                cur_g = pyround(clamp(glue_set * cur_glue));
                            }
                        } else if spec.shrink_order == glue_order {
                            cur_glue += spec.shrink;
                            cur_g = pyround(clamp(glue_set * cur_glue));
                        }
                    }
                    rule_height += cur_g;
                    self.cur_v += rule_height;
                }
                Node::Char(_) => return Err(Error::MatplotlibRaises("RuntimeError: Internal mathtext error: Char node found in vlist".into())),
            }
        }
        Ok(())
    }
}
