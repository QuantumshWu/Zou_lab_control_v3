//! Matplotlib's mathtext grammar (`_mathtext.Parser`) as pyparsing 3.3 runs it.
//!
//! Each function below is one grammar element. Where pyparsing skips whitespace
//! (an element whose `skipWhitespace` and `callPreparse` are set, called with
//! `callPreParse`), the function calls `skip`; where it does not, it does not.
//! Parse actions run once, when their element completes, in pyparsing's order; the
//! one element pyparsing re-parses at the same place (the `subsuper` nucleus) is
//! parsed once and reused, as its packrat cache does. Elements outside the ported
//! subset report `Unsupported` as soon as they match, never an approximation.

use std::collections::HashMap;

use crate::Error;
use crate::data;
use crate::fonts::{FontName, Fontset};
use crate::node::{GlueSpec, ListKind, ListNode, Node, pymax};

/// Why an element did not match.
pub(crate) enum Fail {
    /// `ParseException`: the caller may try something else.
    Soft,
    /// `ParseFatalException` / `ParseSyntaxException`: Matplotlib raises ValueError.
    Fatal(String),
    /// Anything that is not a parse failure (unsupported construct, font error).
    Hard(Error),
}

impl From<Error> for Fail {
    fn from(error: Error) -> Self { Fail::Hard(error) }
}

type P<T> = Result<T, Fail>;

fn unsupported<T>(what: &str) -> P<T> { Err(Fail::Hard(Error::Unsupported(what.to_string()))) }
/// Matplotlib itself raises an exception other than ValueError here.
fn raises<T>(what: &str) -> P<T> { Err(Fail::Hard(Error::MatplotlibRaises(what.to_string()))) }

/// After an ErrorStop (`-`), a soft failure is fatal.
fn stop<T>(result: P<T>, message: &str) -> P<T> {
    match result {
        Err(Fail::Soft) => Err(Fail::Fatal(message.to_string())),
        other => other,
    }
}

#[derive(Clone, Copy)]
struct State { font: FontName, font_class: FontName, fontsize: f64, dpi: f64 }

impl State {
    fn set_font(&mut self, name: FontName) {
        if name.sets_class() { self.font_class = name }
        self.font = name;
    }
}

const WHITE: [char; 4] = [' ', '\n', '\t', '\r'];
const MINUS_SIGN: &str = "\u{2212}";

fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let code = c as u32;
    ranges.binary_search_by(|&(lo, hi)| if hi < code { std::cmp::Ordering::Less } else if lo > code { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal }).is_ok()
}
fn py_isspace(c: char) -> bool { in_ranges(data::ISSPACE_RANGES, c) }
fn py_isdigit(c: char) -> bool { in_ranges(data::ISDIGIT_RANGES, c) }

fn contains(set: &[&str], item: &str) -> bool { set.binary_search(&item).is_ok() }

/// A token value: a node, or (in `subsuper`'s script list) an operator.
pub(crate) struct Parser<'p, 'f> {
    fonts: &'p mut Fontset<'f>,
    states: Vec<State>,
    in_subscript_or_superscript: bool,
    em_width_cache: HashMap<(FontName, u64, u64), f64>,
    /// The string the current (outer or math) parse reads, as code points.
    s: Vec<char>,
}

impl<'p, 'f> Parser<'p, 'f> {
    pub(crate) fn new(fonts: &'p mut Fontset<'f>) -> Self {
        Self { fonts, states: Vec::new(), in_subscript_or_superscript: false, em_width_cache: HashMap::new(), s: Vec::new() }
    }

    /// `Parser.parse(s, fonts, fontsize, dpi)`.
    pub(crate) fn parse(&mut self, text: &str, fontsize: f64, dpi: f64) -> Result<ListNode, Error> {
        self.states = vec![State { font: FontName::Default, font_class: FontName::Rm, fontsize, dpi }];
        self.em_width_cache.clear();
        self.in_subscript_or_superscript = false;
        let result = self.main(text);
        self.states.clear();
        self.in_subscript_or_superscript = false;
        match result {
            Ok(list) => Ok(list),
            Err(Fail::Soft) => Err(Error::Syntax(format!("Matplotlib cannot parse {text:?}"))),
            Err(Fail::Fatal(message)) => Err(Error::Syntax(message)),
            Err(Fail::Hard(error)) => Err(error),
        }
    }

    fn state(&mut self) -> &mut State { self.states.last_mut().expect("parser state") }
    fn push_state(&mut self) { let top = *self.state(); self.states.push(top) }
    fn pop_state(&mut self) { self.states.pop(); }

    fn skip(&self, mut loc: usize) -> usize {
        while loc < self.s.len() && WHITE.contains(&self.s[loc]) { loc += 1 }
        loc
    }

    fn starts(&self, loc: usize, literal: &str) -> bool {
        let mut i = loc;
        for c in literal.chars() {
            if i >= self.s.len() || self.s[i] != c { return false }
            i += 1;
        }
        true
    }

    fn alpha_end(&self, mut loc: usize) -> usize {
        while loc < self.s.len() && self.s[loc].is_ascii_alphabetic() { loc += 1 }
        loc
    }

    fn text(&self, from: usize, to: usize) -> String { self.s[from..to].iter().collect() }

    /// `\\name` whose whole letter run is in `names` (csnames); the name's end.
    fn csname(&self, loc: usize, names: &[&str]) -> Option<(String, usize)> {
        if loc >= self.s.len() || self.s[loc] != '\\' { return None }
        let end = self.alpha_end(loc + 1);
        let name = self.text(loc + 1, end);
        (end > loc + 1 && contains(names, &name)).then_some((name, end))
    }

    /// Longest of `names` starting at `loc` (pyparsing `one_of`).
    fn one_of<'n>(&self, loc: usize, names: impl Iterator<Item = &'n str>) -> Option<(&'n str, usize)> {
        let mut best: Option<(&str, usize)> = None;
        for name in names {
            if self.starts(loc, name) {
                let length = name.chars().count();
                if best.is_none_or(|(_, best_length)| length > best_length) { best = Some((name, length)) }
            }
        }
        best.map(|(name, length)| (name, loc + length))
    }

    // ------------------------------------------------------------------ main

    fn main(&mut self, text: &str) -> P<ListNode> {
        self.s = text.chars().collect();
        let mut tokens = Vec::new();
        let mut loc = self.non_math(0, &mut tokens)?;
        loop {
            // ZeroOrMore(math_string + non_math): the And skips whitespace.
            let at = self.skip(loc);
            let Some(end) = self.quoted_math(at) else { break };
            let inner: Vec<char> = self.s[at + 1..end - 1].to_vec();
            let outer = std::mem::replace(&mut self.s, inner);
            let result = self.math();
            self.s = outer;
            match result {
                Ok(node) => tokens.push(node),
                Err(Fail::Soft) => break,
                Err(other) => return Err(other),
            }
            loc = self.non_math(end, &mut tokens)?;
        }
        if self.skip(loc) != self.s.len() { return Err(Fail::Soft) }
        Ok(ListNode::hlist(tokens, true, self.fonts)?)
    }

    /// `QuotedString('$', '\\', unquote_results=False)`: the end of `$...$` at `loc`.
    fn quoted_math(&self, loc: usize) -> Option<usize> {
        let s = &self.s;
        if loc >= s.len() || s[loc] != '$' { return None }
        let mut i = loc + 1;
        loop {
            if i < s.len() && s[i] == '\\' && i + 1 < s.len() && s[i + 1] != '\n' { i += 2 }
            else if i < s.len() && !matches!(s[i], '$' | '\n' | '\r' | '\\') { i += 1 }
            else { break }
        }
        (i < s.len() && s[i] == '$').then_some(i + 1)
    }

    /// `non_math`: literal text up to the next unescaped `$`; then math state.
    fn non_math(&mut self, loc: usize, tokens: &mut Vec<Node>) -> P<usize> {
        let mut end = loc;
        loop {
            if end + 1 < self.s.len() && self.s[end] == '\\' && self.s[end + 1] == '$' { end += 2 }
            else if end < self.s.len() && self.s[end] != '$' { end += 1 }
            else { break }
        }
        let literal = self.text(loc, end).replace("\\$", "$");
        let state = *self.state();
        let mut symbols = Vec::new();
        let mut buffer = [0u8; 4];
        for c in literal.chars() {
            symbols.push(Node::char(self.fonts, c.encode_utf8(&mut buffer), state.font, state.font_class, state.fontsize, state.dpi)?);
        }
        tokens.push(ListNode::hlist(symbols, true, self.fonts)?.into_node());
        self.push_state();
        self.state().set_font(FontName::It); // mathtext.default
        Ok(end)
    }

    /// `math` (`OneOrMore(token)` with `parse_all`) over the string between `$`s.
    fn math(&mut self) -> P<Node> {
        let mut loc = self.skip(0);
        let mut nodes = Vec::new();
        loc = self.token(loc, &mut nodes)?;
        loop {
            match self.token(loc, &mut nodes) {
                Ok(end) => loc = end,
                Err(Fail::Soft) => break,
                Err(other) => return Err(other),
            }
        }
        let hlist = ListNode::hlist(nodes, true, self.fonts)?;
        self.pop_state();
        if self.skip(loc) != self.s.len() { return Err(Fail::Soft) }
        Ok(hlist.into_node())
    }

    // ----------------------------------------------------------------- token

    /// `token` (called with `callPreParse`): space | customspace | font | subsuper |
    /// auto_delim | unclosed_group | unknown_symbol.
    fn token(&mut self, loc: usize, out: &mut Vec<Node>) -> P<usize> {
        let loc = self.skip(loc);
        if let Some(end) = self.simple(loc, out)? { return Ok(end) }
        match self.auto_delim(loc) {
            Ok((end, node)) => { out.push(node); return Ok(end) }
            Err(Fail::Soft) => {}
            Err(other) => return Err(other),
        }
        // unclosed_group: an opening group that the alternatives above could not
        // close always ends the parse with "Expected '}'" (or an unknown symbol).
        let at = self.skip(loc);
        if self.start_group_end(at).is_some() { return Err(Fail::Fatal("Expected '}'".into())) }
        // unknown_symbol
        if at < self.s.len() && self.s[at] == '\\' && self.alpha_end(at + 1) > at + 1 {
            let end = self.alpha_end(at + 1);
            return Err(Fail::Fatal(format!("Unknown symbol: {}", self.text(at, end))));
        }
        Err(Fail::Soft)
    }

    /// `simple` = space | customspace | font | subsuper (each called with
    /// `callPreParse`); None for a soft failure.
    fn simple(&mut self, loc: usize, out: &mut Vec<Node>) -> P<Option<usize>> {
        // space
        let at = self.skip(loc);
        if let Some((name, end)) = self.one_of(at, data::SPACE_WIDTHS.iter().map(|(name, _)| *name)) {
            let width = data::SPACE_WIDTHS.iter().find(|(key, _)| *key == name).expect("space").1;
            out.push(self.make_space(width)?);
            return Ok(Some(end));
        }
        // customspace
        if self.starts(at, "\\hspace") {
            let (end, value) = stop(self.hspace_argument(at + 7), "Expected \\hspace{space}")?;
            out.push(self.make_space(value)?);
            return Ok(Some(end));
        }
        // font
        if let Some((name, end)) = self.csname(at, data::FONTNAMES) {
            let font = FontName::from_grammar(&name).expect("font name");
            self.state().set_font(font);
            return Ok(Some(end));
        }
        // subsuper
        match self.subsuper(loc) {
            Ok((end, node)) => { out.push(node); Ok(Some(end)) }
            Err(Fail::Soft) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// `"{" + float_literal + "}"` after `\hspace`.
    fn hspace_argument(&mut self, loc: usize) -> P<(usize, f64)> {
        let at = self.skip(loc);
        if !self.starts(at, "{") { return Err(Fail::Soft) }
        let at = self.skip(at + 1);
        let (end, value) = self.float_literal(at).ok_or(Fail::Soft)?;
        let at = self.skip(end);
        if !self.starts(at, "}") { return Err(Fail::Soft) }
        Ok((at + 1, value))
    }

    /// `[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)` converted with `float`.
    fn float_literal(&self, loc: usize) -> Option<(usize, f64)> {
        let s = &self.s;
        let mut i = loc;
        if i < s.len() && (s[i] == '-' || s[i] == '+') { i += 1 }
        let digits = |mut i: usize| { while i < s.len() && s[i].is_ascii_digit() { i += 1 } i };
        let start = i;
        let after = digits(i);
        let end = if after > start {
            let mut j = after;
            if j < s.len() && s[j] == '.' { j = digits(j + 1) }
            j
        } else if i < s.len() && s[i] == '.' && digits(i + 1) > i + 1 {
            digits(i + 1)
        } else {
            return None;
        };
        self.text(loc, end).parse::<f64>().ok().map(|value| (end, value))
    }

    // -------------------------------------------------------------- subsuper

    /// `subsuper` (a MatchFirst: never skips whitespace itself).
    fn subsuper(&mut self, loc: usize) -> P<(usize, Node)> {
        let at = self.skip(loc);
        // Alternative 1 and 3 both parse the nucleus placeable at `at` (packrat).
        let nucleus = match self.placeable(at) {
            Ok(found) => Some(found),
            Err(Fail::Soft) => None,
            Err(other) => return Err(other),
        };
        let after_nucleus = nucleus.as_ref().map_or(at, |(end, _)| *end);
        // OneOrMore(("_" | "^") - placeable)
        let mut scripts: Vec<(char, Node)> = Vec::new();
        let mut loc1 = after_nucleus;
        loop {
            let op_at = self.skip(loc1);
            if op_at >= self.s.len() || !matches!(self.s[op_at], '_' | '^') { break }
            let op = self.s[op_at];
            let (end, arg) = stop(self.placeable(op_at + 1), &format!("Expected placeable after {op:?}"))?;
            scripts.push((op, arg));
            loc1 = end;
        }
        if !scripts.is_empty() {
            let apostrophes_at = self.skip(loc1);
            let end = self.apostrophes(apostrophes_at);
            let node = self.subsuper_action(at, nucleus.map(|(_, node)| node), scripts, end - apostrophes_at)?;
            return Ok((end, node));
        }
        // Alternative 2: Regex("'+")
        let primes = self.apostrophes(at);
        if primes > at {
            let node = self.subsuper_action(at, None, Vec::new(), primes - at)?;
            return Ok((primes, node));
        }
        // Alternative 3: named_placeable + Regex("'*")
        let Some((end, node)) = nucleus else { return Err(Fail::Soft) };
        let apostrophes_at = self.skip(end);
        let after = self.apostrophes(apostrophes_at);
        let node = self.subsuper_action(at, Some(node), Vec::new(), after - apostrophes_at)?;
        Ok((after, node))
    }

    fn apostrophes(&self, mut loc: usize) -> usize {
        while loc < self.s.len() && self.s[loc] == '\'' { loc += 1 }
        loc
    }

    fn is_overunder(nucleus: &Node) -> bool {
        match nucleus {
            Node::Char(c) => contains(data::OVERUNDER_SYMBOLS, &c.c),
            Node::List(l) if l.kind == ListKind::H => l.function_name.as_deref().is_some_and(|name| contains(data::OVERUNDER_FUNCTIONS, name)),
            _ => false,
        }
    }

    fn is_dropsub(node: &Node) -> bool { matches!(node, Node::Char(c) if contains(data::DROPSUB_SYMBOLS, &c.c)) }
    fn is_slanted(node: &Node) -> bool { matches!(node, Node::Char(c) if c.metrics.slanted) }

    /// `Parser.subsuper`'s action.
    fn subsuper_action(&mut self, loc: usize, nucleus: Option<Node>, scripts: Vec<(char, Node)>, napostrophes: usize)
        -> P<Node> {
        let nucleus = nucleus.unwrap_or_else(|| Node::hbox(0.0));
        if scripts.is_empty() && napostrophes == 0 { return Ok(nucleus) }
        let (mut sub, mut sup): (Option<Node>, Option<Node>) = (None, None);
        for (op, arg) in scripts {
            if op == '_' {
                if sub.is_some() { return Err(Fail::Fatal("Double subscript".into())) }
                sub = Some(arg);
            } else {
                if sup.is_some() { return Err(Fail::Fatal("Double superscript".into())) }
                sup = Some(arg);
            }
        }
        let state = *self.state();
        let rule_thickness = Fontset::underline_thickness(state.fontsize, state.dpi);
        let x_height = self.fonts.get_xheight(state.font, state.fontsize, state.dpi)?;

        if napostrophes > 0 {
            let mut list = match sup.take() {
                None => ListNode::hlist(Vec::new(), true, self.fonts)?,
                Some(Node::List(list)) if list.kind == ListKind::H => list,
                Some(_) => return raises("AttributeError: apostrophes after a superscript that is not a group"),
            };
            for _ in 0..napostrophes {
                let prime = self.symbol_action(loc, "\\prime")?;
                list.children.push(prime);
            }
            list.kern(self.fonts)?;
            list.hpack(0.0, false);
            sup = Some(list.into_node());
        }

        if Self::is_overunder(&nucleus) {
            let mut vlist = Vec::new();
            let mut shift = 0.0;
            let mut width = nucleus.width();
            if let Some(sup) = sup.as_mut() { sup.shrink(); width = pymax(width, sup.width()) }
            if let Some(sub) = sub.as_mut() { sub.shrink(); width = pymax(width, sub.width()) }
            let vgap = rule_thickness * 3.0;
            if let Some(sup) = sup {
                let mut hlist = ListNode::hcentered(vec![sup], self.fonts)?;
                hlist.hpack(width, true);
                vlist.push(hlist.into_node());
                vlist.push(Node::vbox(0.0, vgap));
            }
            let nucleus_depth = nucleus.depth()?;
            let mut hlist = ListNode::hcentered(vec![nucleus], self.fonts)?;
            hlist.hpack(width, true);
            vlist.push(hlist.into_node());
            if let Some(sub) = sub {
                let mut hlist = ListNode::hcentered(vec![sub], self.fonts)?;
                hlist.hpack(width, true);
                shift = hlist.height + vgap + nucleus_depth;
                vlist.push(Node::vbox(0.0, vgap));
                vlist.push(hlist.into_node());
            }
            let mut vlt = ListNode::vlist(vlist)?;
            vlt.shift_amount = shift;
            return Ok(ListNode::hlist(vec![vlt.into_node()], true, self.fonts)?.into_node());
        }

        // Drop the last kern and use the last character's advance; `last_char` is
        // the nucleus's last child, or the nucleus itself (an empty Hlist included).
        let facts = |node: &Node| -> P<(f64, f64, bool, bool)> {
            Ok((node.height()?, node.depth()?, Self::is_dropsub(node), Self::is_slanted(node)))
        };
        let (nucleus, (lc_height, lc_depth, dropsub, slanted)) = match nucleus {
            Node::List(list) if list.kind == ListKind::H => {
                let mut children = list.children;
                let last = if children.is_empty() {
                    (list.height, list.depth, false, false)
                } else {
                    let count = children.len();
                    if matches!(children[count - 1], Node::Kern { .. }) {
                        // Matplotlib indexes children[-2] and its IndexError fails `subsuper` at
                        // a group, which nothing else can then consume.
                        if count < 2 { return Err(Fail::Fatal("exception raised in parse action".into())) }
                        if matches!(children[count - 2], Node::Char(_)) { children.pop(); }
                    }
                    let index = children.len() - 1;
                    if let Node::Char(c) = &mut children[index] { c.width = c.metrics.advance }
                    facts(&children[index])?
                };
                (ListNode::hlist(children, false, self.fonts)?, last)
            }
            mut other => {
                if let Node::Char(c) = &mut other { c.width = c.metrics.advance }
                let last = facts(&other)?;
                (ListNode::hlist(vec![other], true, self.fonts)?, last)
            }
        };

        let family = self.fonts.family_name(state.font)?;
        let constants = Constants::for_family(&family);
        let lc_baseline = if dropsub { lc_depth } else { 0.0 };
        let mut superkern = constants.delta * x_height;
        let mut subkern = constants.delta * x_height;
        if slanted {
            superkern += constants.delta * x_height;
            superkern += constants.delta_slanted * (lc_height - x_height * 2. / 3.);
            if dropsub {
                subkern = (3.0 * constants.delta - constants.delta_integral) * lc_height;
                superkern = (3.0 * constants.delta + constants.delta_integral) * lc_height;
            } else {
                subkern = 0.0;
            }
        }

        let mut x = match sup {
            None => {
                let sub = sub.expect("a script");
                let mut x = ListNode::hlist(vec![Node::kern(subkern), sub], true, self.fonts)?;
                x.shrink();
                let shift_down = if dropsub { lc_baseline + constants.subdrop * x_height } else { constants.sub1 * x_height };
                x.shift_amount = shift_down;
                x
            }
            Some(sup) => {
                let mut x = ListNode::hlist(vec![Node::kern(superkern), sup], true, self.fonts)?;
                x.shrink();
                let mut shift_up = if dropsub { lc_height - constants.subdrop * x_height } else { constants.sup1 * x_height };
                match sub {
                    None => {
                        x.shift_amount = -shift_up;
                        x
                    }
                    Some(sub) => {
                        let mut y = ListNode::hlist(vec![Node::kern(subkern), sub], true, self.fonts)?;
                        y.shrink();
                        let shift_down = if dropsub { lc_baseline + constants.subdrop * x_height } else { constants.sub2 * x_height };
                        let clr = 2.0 * rule_thickness - ((shift_up - x.depth) - (y.height - shift_down));
                        if clr > 0.0 { shift_up += clr }
                        let gap = (shift_up - x.depth) - (y.height - shift_down);
                        let mut v = ListNode::vlist(vec![x.into_node(), Node::kern(gap), y.into_node()])?;
                        v.shift_amount = shift_down;
                        v
                    }
                }
            }
        };
        if !dropsub { x.width += constants.script_space * x_height }
        let mut spaced = vec![nucleus.into_node(), x.into_node()];
        if self.in_subscript_or_superscript {
            spaced.push(self.make_space(0.16667)?);
            self.in_subscript_or_superscript = false;
        }
        Ok(ListNode::hlist(spaced, true, self.fonts)?.into_node())
    }

    // ------------------------------------------------------------- placeable

    /// `placeable` (never skips whitespace itself; its alternatives do).
    fn placeable(&mut self, loc: usize) -> P<(usize, Node)> {
        let at = self.skip(loc);
        // accent
        if at < self.s.len() && self.s[at] == '\\' {
            let end = self.alpha_end(at + 1);
            let name = self.text(at + 1, end);
            let alpha = end > at + 1 && (contains(data::ACCENT_NAMES, &name) || contains(data::WIDE_ACCENTS, &name));
            let symbolic = at + 1 < self.s.len() && matches!(self.s[at + 1], '"' | '`' | '\'' | '~' | '.' | '^');
            if alpha || symbolic { return unsupported("accents") }
        }
        // symbol (no whitespace skipping)
        if let Some((sym, end)) = self.symbol_regex(loc) {
            let node = self.symbol_action(loc, &sym)?;
            return Ok((end, node));
        }
        // function
        if let Some((name, end)) = self.csname(at, data::FUNCTION_NAMES) {
            let node = self.function_action(at, &name)?;
            return Ok((end, node));
        }
        for (literal, what) in [("\\operatorname", "\\operatorname")] {
            if self.starts(at, literal) { return unsupported(what) }
        }
        // group
        if let Some((font, open_end)) = self.start_group_end(at) {
            self.push_state();
            if let Some(font) = font { self.state().set_font(font) }
            let mut nodes = Vec::new();
            let mut loc = open_end;
            loop {
                match self.token(loc, &mut nodes) {
                    Ok(end) => loc = end,
                    Err(Fail::Soft) => break,
                    Err(other) => return Err(other),
                }
            }
            let close = self.skip(loc);
            if !self.starts(close, "}") { return Err(Fail::Soft) }
            self.pop_state();
            return Ok((close + 1, ListNode::hlist(nodes, true, self.fonts)?.into_node()));
        }
        if self.starts(at, "\\frac") {
            let (end, (num, den)) = stop(self.two_groups(at + 5), "Expected \\frac{num}{den}")?;
            let thickness = self.underline_thickness();
            return Ok((end, self.genfrac(thickness, 1, num, den)?));
        }
        for literal in ["\\dfrac", "\\binom", "\\genfrac", "\\overset", "\\underset"] {
            if self.starts(at, literal) { return unsupported(literal) }
        }
        if self.starts(at, "\\sqrt") {
            let (end, (root, body)) = stop(self.sqrt_arguments(at + 5), "Expected \\sqrt{value}")?;
            return Ok((end, self.sqrt_action(root, body)?));
        }
        for literal in ["\\overline", "\\text", "\\boldsymbol", "\\substack"] {
            if self.starts(at, literal) { return unsupported(literal) }
        }
        Err(Fail::Soft)
    }

    /// `start_group` (`Optional(\math<font>) + "{"`) at `loc`: its font and end.
    fn start_group_end(&self, loc: usize) -> Option<(Option<FontName>, usize)> {
        let mut font = None;
        let mut at = loc;
        if self.starts(loc, "\\math") {
            let name_at = self.skip(loc + 5);
            if let Some((name, end)) = self.one_of(name_at, data::FONTNAMES.iter().copied()) {
                font = FontName::from_grammar(name);
                at = end;
            }
        }
        let open = self.skip(at);
        if !self.starts(open, "{") { return None }
        Some((font, open + 1))
    }

    /// The symbol regex at `loc` (no whitespace skipping): (sym, end).
    fn symbol_regex(&self, loc: usize) -> Option<(String, usize)> {
        let s = &self.s;
        if loc >= s.len() { return None }
        let c = s[loc];
        let single = c.is_ascii_alphanumeric() || " +-*/<>=:,.;!?&'@()[]|".contains(c)
            || ('\u{80}'..='\u{1ffff}').contains(&c);
        if single { return Some((c.to_string(), loc + 1)) }
        if c != '\\' { return None }
        if loc + 1 < s.len() && "%${}[]_|".contains(s[loc + 1]) {
            return Some((self.text(loc, loc + 2), loc + 2));
        }
        // `\\(?:names in tex2uni order)(?![A-Za-z])`: a name made of letters must be
        // the whole letter run; the other names start with a non-letter, so the two
        // kinds never both match.
        let run_end = self.alpha_end(loc + 1);
        if run_end > loc + 1 {
            let run = self.text(loc + 1, run_end);
            if crate::fonts::tex2uni(&run).is_some() { return Some((self.text(loc, run_end), run_end)) }
        }
        for (name, _) in data::TEX2UNI.iter().filter(|(name, _)| !name.chars().all(|c| c.is_ascii_alphabetic())) {
            if self.starts(loc + 1, name) {
                let end = loc + 1 + name.chars().count();
                if end >= s.len() || !s[end].is_ascii_alphabetic() { return Some((self.text(loc, end), end)) }
            }
        }
        None
    }

    fn underline_thickness(&mut self) -> f64 {
        let state = *self.state();
        Fontset::underline_thickness(state.fontsize, state.dpi)
    }

    /// `_make_space(percentage)`.
    fn make_space(&mut self, percentage: f64) -> P<Node> {
        let state = *self.state();
        let key = (state.font, state.fontsize.to_bits(), state.dpi.to_bits());
        let width = match self.em_width_cache.get(&key) {
            Some(&width) => width,
            None => {
                let width = self.fonts.metrics(FontName::It, FontName::It, "m", state.fontsize, state.dpi)?.advance;
                self.em_width_cache.insert(key, width);
                width
            }
        };
        Ok(Node::kern(width * percentage))
    }

    /// `Parser.symbol`'s action for `c` at `loc`.
    fn symbol_action(&mut self, loc: usize, sym: &str) -> P<Node> {
        let c = if sym == "-" { MINUS_SIGN } else { sym };
        let state = *self.state();
        let char = match Node::char(self.fonts, c, state.font, state.font_class, state.fontsize, state.dpi) {
            Ok(node) => node,
            Err(Error::UnknownSymbol(_)) => return Err(Fail::Fatal(format!("Unknown symbol: {c}"))),
            Err(other) => return Err(other.into()),
        };
        let spaced = contains(data::BINARY_OPERATORS, c) || contains(data::RELATION_SYMBOLS, c) || contains(data::ARROW_SYMBOLS, c);
        let s = &self.s;
        let prev_char = s[..loc].iter().rev().find(|&&ch| ch != ' ').copied();
        if spaced {
            let leading = s[..loc].iter().all(|&ch| py_isspace(ch));
            let after_opening = prev_char.is_some_and(|p| {
                let p = p.to_string();
                p == "{" || contains(data::LEFT_DELIMS, &p) || contains(data::RELATION_SYMBOLS, &p)
            });
            if self.in_subscript_or_superscript || (contains(data::BINARY_OPERATORS, c) && (leading || after_opening)) {
                return Ok(char);
            }
            let before = self.make_space(0.2)?;
            let after = self.make_space(0.2)?;
            return Ok(ListNode::hlist(vec![before, char, after], true, self.fonts)?.into_node());
        }
        if contains(data::PUNCTUATION_SYMBOLS, c) {
            let next_char = s.get(loc + 1..).unwrap_or(&[]).iter().find(|&&ch| ch != ' ').copied();
            if c == "," && prev_char == Some('{') && next_char == Some('}') { return Ok(char) }
            if c == "." && prev_char.is_some_and(py_isdigit) && next_char.is_some_and(py_isdigit) { return Ok(char) }
            let space = self.make_space(0.2)?;
            return Ok(ListNode::hlist(vec![char, space], true, self.fonts)?.into_node());
        }
        Ok(char)
    }

    /// `Parser.function` (`operatorname` with the name as text) at `loc`.
    fn function_action(&mut self, loc: usize, name: &str) -> P<Node> {
        self.push_state();
        self.state().set_font(FontName::Rm);
        let state = *self.state();
        let mut list = Vec::new();
        let mut buffer = [0u8; 4];
        for c in name.chars() {
            list.push(Node::char(self.fonts, c.encode_utf8(&mut buffer), state.font, state.font_class, state.fontsize, state.dpi)?);
        }
        let next_char_loc = loc + name.chars().count() + 1;
        let next_char = self.s.get(next_char_loc..).unwrap_or(&[]).iter().find(|&&ch| ch != ' ').copied();
        let is_delimiter = next_char.is_some_and(|ch| {
            let ch = ch.to_string();
            ch == "^" || ch == "_" || contains(data::LEFT_DELIMS, &ch) || contains(data::RIGHT_DELIMS, &ch) || contains(data::AMBI_DELIMS, &ch)
        });
        if !is_delimiter && !contains(data::OVERUNDER_FUNCTIONS, name) {
            list.push(self.make_space(0.16667)?);
        }
        self.pop_state();
        self.in_subscript_or_superscript = matches!(next_char, Some('^' | '_'));
        let mut hlist = ListNode::hlist(list, true, self.fonts)?;
        hlist.function_name = Some(name.to_string());
        Ok(hlist.into_node())
    }

    /// `required_group` (Forward: skips whitespace) = "{" + OneOrMore(token) + "}".
    fn required_group(&mut self, loc: usize) -> P<(usize, ListNode)> {
        let at = self.skip(loc);
        if !self.starts(at, "{") { return Err(Fail::Soft) }
        let mut nodes = Vec::new();
        let mut loc = self.token(at + 1, &mut nodes)?;
        loop {
            match self.token(loc, &mut nodes) {
                Ok(end) => loc = end,
                Err(Fail::Soft) => break,
                Err(other) => return Err(other),
            }
        }
        let close = self.skip(loc);
        if !self.starts(close, "}") { return Err(Fail::Soft) }
        Ok((close + 1, ListNode::hlist(nodes, true, self.fonts)?))
    }

    fn two_groups(&mut self, loc: usize) -> P<(usize, (ListNode, ListNode))> {
        let (end, num) = self.required_group(loc)?;
        let (end, den) = self.required_group(end)?;
        Ok((end, (num, den)))
    }

    /// `_genfrac("", "", rule, style, num, den)` with `style` shrink steps.
    fn genfrac(&mut self, rule: f64, style: usize, num: ListNode, den: ListNode) -> P<Node> {
        let state = *self.state();
        let thickness = Fontset::underline_thickness(state.fontsize, state.dpi);
        let (mut num, mut den) = (num.into_node(), den.into_node());
        for _ in 0..style { num.shrink(); den.shrink(); }
        let (num_width, den_width) = (num.width(), den.width());
        let mut cnum = ListNode::hcentered(vec![num], self.fonts)?;
        let mut cden = ListNode::hcentered(vec![den], self.fonts)?;
        let width = pymax(num_width, den_width);
        cnum.hpack(width, true);
        cden.hpack(width, true);
        let cden_height = cden.height;
        let mut vlist = ListNode::vlist(vec![
            cnum.into_node(),
            Node::vbox(0.0, thickness * 2.0),
            Node::hrule(rule),
            Node::vbox(0.0, thickness * 2.0),
            cden.into_node(),
        ])?;
        let metrics = self.fonts.metrics(state.font, FontName::It, "=", state.fontsize, state.dpi)?;
        let shift = cden_height - ((metrics.ymax + metrics.ymin) / 2.0 - thickness * 3.0);
        vlist.shift_amount = shift;
        Ok(ListNode::hlist(vec![vlist.into_node(), Node::hbox(thickness * 2.0)], true, self.fonts)?.into_node())
    }

    /// `Optional("[" + OneOrMore(~"]" + token)("root") + "]") + required_group("value")`.
    fn sqrt_arguments(&mut self, loc: usize) -> P<(usize, (Option<Vec<Node>>, ListNode))> {
        let at = self.skip(loc);
        let mut root = None;
        let mut after_root = at;
        if self.starts(at, "[") {
            let mut nodes = Vec::new();
            let mut loc = at + 1;
            let mut count = 0;
            loop {
                if self.starts(self.skip(loc), "]") { break }
                match self.token(loc, &mut nodes) {
                    Ok(end) => { loc = end; count += 1 }
                    Err(Fail::Soft) => break,
                    Err(other) => return Err(other),
                }
            }
            let close = self.skip(loc);
            if count > 0 && self.starts(close, "]") {
                root = Some(nodes);
                after_root = close + 1;
            }
        }
        let (end, body) = self.required_group(after_root)?;
        Ok((end, (root, body)))
    }

    /// `Parser.sqrt`'s action.
    fn sqrt_action(&mut self, root: Option<Vec<Node>>, body: ListNode) -> P<Node> {
        let state = *self.state();
        let thickness = Fontset::underline_thickness(state.fontsize, state.dpi);
        let height = body.height - body.shift_amount + thickness * 5.0;
        let depth = body.depth + body.shift_amount;
        let check = self.auto_height_char("\\__sqrt__", height, depth, None)?;
        let height = check.height - check.shift_amount;
        let depth = check.depth + check.shift_amount;
        let check_width = check.width;
        let padded = ListNode::hlist(vec![Node::hbox(2.0 * thickness), body.into_node(), Node::hbox(2.0 * thickness)], true, self.fonts)?;
        let mut rightside = ListNode::vlist(vec![Node::hrule(thickness), Node::glue(GlueSpec::FILL), padded.into_node()])?;
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0), true, depth)?;
        let root = match root {
            None => Node::plain_box(check_width * 0.5, 0.0, 0.0),
            Some(nodes) if nodes.is_empty() => Node::plain_box(check_width * 0.5, 0.0, 0.0),
            Some(nodes) => {
                let mut root = ListNode::hlist(nodes, true, self.fonts)?.into_node();
                root.shrink();
                root.shrink();
                root
            }
        };
        let inner = ListNode::hlist(vec![root], true, self.fonts)?;
        let mut root_vlist = ListNode::vlist(vec![inner.into_node()])?;
        root_vlist.shift_amount = -height * 0.6;
        Ok(ListNode::hlist(vec![root_vlist.into_node(), Node::kern(-check_width * 0.5), check.into_node(), rightside.into_node()], true, self.fonts)?.into_node())
    }

    /// `AutoHeightChar(c, height, depth, state, factor=factor)`.
    fn auto_height_char(&mut self, c: &str, height: f64, depth: f64, factor: Option<f64>) -> P<ListNode> {
        let base = *self.state();
        let alternatives = self.fonts.sized_alternatives(base.font, c)?;
        let x_height = self.fonts.get_xheight(base.font, base.fontsize, base.dpi)?;
        let mut state = base;
        let target_total = height + depth;
        let mut chosen = None;
        for (fontname, sym) in &alternatives {
            state.set_font(*fontname);
            let char = Node::char(self.fonts, sym, state.font, state.font_class, state.fontsize, state.dpi)?;
            let total = char.height()? + char.depth()?;
            chosen = Some((char, sym.clone()));
            if total >= target_total - 0.2 * x_height { break }
        }
        let Some((mut char, sym)) = chosen else { return raises("UnboundLocalError: a delimiter without sized alternatives") };
        let mut shift = 0.0;
        if state.font != FontName::Size(0) || alternatives.len() == 1 {
            let factor = match factor { Some(factor) => factor, None => target_total / (char.height()? + char.depth()?) };
            state.fontsize *= factor;
            char = Node::char(self.fonts, &sym, state.font, state.font_class, state.fontsize, state.dpi)?;
            shift = depth - char.depth()?;
        }
        let mut list = ListNode::hlist(vec![char], true, self.fonts)?;
        list.shift_amount = shift;
        Ok(list)
    }

    // ------------------------------------------------------------ auto_delim

    fn delim(&self, loc: usize) -> Option<(String, usize)> {
        let at = self.skip(loc);
        self.one_of(at, data::LEFT_DELIMS.iter().chain(data::RIGHT_DELIMS).chain(data::AMBI_DELIMS).copied())
            .map(|(name, end)| (name.to_string(), end))
    }

    /// `auto_delim` = `\left` - delim + ZeroOrMore(simple | auto_delim | mdelim) + `\right` - delim.
    fn auto_delim(&mut self, loc: usize) -> P<(usize, Node)> {
        let at = self.skip(loc);
        if !self.starts(at, "\\left") { return Err(Fail::Soft) }
        let Some((left, mut loc)) = self.delim(at + 5) else { return Err(Fail::Fatal("Expected a delimiter".into())) };
        let mut middle = Vec::new();
        loop {
            match self.simple(loc, &mut middle)? {
                Some(end) => { loc = end; continue }
                None => {}
            }
            match self.auto_delim(loc) {
                Ok((end, node)) => { middle.push(node); loc = end; continue }
                Err(Fail::Soft) => {}
                Err(other) => return Err(other),
            }
            if self.starts(self.skip(loc), "\\middle") { return unsupported("\\middle") }
            break;
        }
        let right_at = self.skip(loc);
        if !self.starts(right_at, "\\right") { return Err(Fail::Fatal("Expected \\right".into())) }
        let Some((right, end)) = self.delim(right_at + 6) else { return Err(Fail::Fatal("Expected a delimiter".into())) };
        Ok((end, self.auto_sized_delimiter(&left, middle, &right)?))
    }

    /// `_auto_sized_delimiter(front, middle, back)` (no `\middle`).
    fn auto_sized_delimiter(&mut self, front: &str, middle: Vec<Node>, back: &str) -> P<Node> {
        let (height, depth, factor) = if middle.is_empty() {
            (0.0, 0.0, Some(1.0))
        } else {
            let mut height = middle[0].height()?;
            let mut depth = middle[0].depth()?;
            for node in &middle[1..] {
                height = pymax(height, node.height()?);
                depth = pymax(depth, node.depth()?);
            }
            (height, depth, None)
        };
        let mut parts = Vec::new();
        if front != "." { parts.push(self.auto_height_char(front, height, depth, factor)?.into_node()) }
        parts.extend(middle);
        if back != "." { parts.push(self.auto_height_char(back, height, depth, factor)?.into_node()) }
        Ok(ListNode::hlist(parts, true, self.fonts)?.into_node())
    }
}

/// `FontConstantsBase` and the subclasses `_font_constant_mapping` picks.
struct Constants { script_space: f64, subdrop: f64, sup1: f64, sub1: f64, sub2: f64, delta: f64, delta_slanted: f64, delta_integral: f64 }

impl Constants {
    fn for_family(family: &str) -> Self {
        const BASE: Constants = Constants { script_space: 0.05, subdrop: 0.4, sup1: 0.7, sub1: 0.3, sub2: 0.5, delta: 0.025, delta_slanted: 0.2, delta_integral: 0.1 };
        match family {
            "cmb10" | "cmex10" | "cmmi10" | "cmr10" | "cmss10" | "cmsy10" | "cmtt10" => Constants {
                script_space: 0.075, subdrop: 0.2, sup1: 0.45, sub1: 0.2, sub2: 0.3, delta: 0.075, delta_slanted: 0.3, delta_integral: 0.3 },
            "STIXGeneral" | "STIXNonUnicode" | "STIXSizeFiveSym" | "STIXSizeFourSym" | "STIXSizeThreeSym"
            | "STIXSizeTwoSym" | "STIXSizeOneSym" => Constants {
                script_space: 0.1, subdrop: 0.4, sup1: 0.8, sub1: 0.3, sub2: 0.6, delta: 0.05, delta_slanted: 0.3, delta_integral: 0.3 },
            _ => BASE,
        }
    }
}
