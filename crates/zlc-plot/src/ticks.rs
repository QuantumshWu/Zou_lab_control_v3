//! One numeric tick ladder: the normal then compact label tier, a sparser
//! lattice before a smaller font, labels on their tick anchors, explicit names
//! kept. The decisions, their order, the hysteresis of a settled unit and the
//! single-slot memo that a final text layout writes through are the scientific
//! tick policy's own.

use crate::decimal::{Decimal, Integral};
use crate::pyfmt::{format_e, format_g};

pub const TICKS_FLOOR: usize = 2;
pub const MIN_TICK_LABEL_PT: f64 = 3.25;

/// How text is priced: one line's width and height in points at a size, and
/// the height of a line of labels ("lp", ascender over descender).
pub trait Measure {
    fn size(&self, text: &str, size_pt: f64) -> (f64, f64);
    fn line(&self, size_pt: f64) -> f64;
}

/// The axis scale the labels are priced through.
pub enum Scale<'a> {
    Linear,
    Log10,
    Function(&'a dyn Fn(&[f64]) -> Option<Vec<f64>>),
}

impl Scale<'_> {
    fn is_log(&self) -> bool { matches!(self, Scale::Log10) }
    fn map(&self, values: &[f64]) -> Vec<f64> {
        let mapped = match self {
            Scale::Linear => None,
            Scale::Log10 => Some(values.iter().map(|&v| if v > 0.0 { v.log10() } else { f64::NEG_INFINITY }).collect::<Vec<_>>()),
            Scale::Function(function) => function(values),
        };
        match mapped {
            Some(mapped) if mapped.len() == values.len() && mapped.iter().all(|v| v.is_finite()) => mapped,
            _ => values.to_vec(),
        }
    }
}

/// Where an axis is painted, in points: its extent, the room before its low
/// end and past its high end, the room beyond its label side, and direction.
/// `inward`: a label whose tick sits at an end of the axis aligns inside the
/// axes, and every label must fit within the axis' extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry { pub extent: f64, pub before: f64, pub after: f64, pub across: Option<f64>, pub horizontal: bool, pub inward: bool }

#[derive(Clone, Debug, PartialEq)]
pub struct LatticePayload {
    pub step: i64, pub decade: i64, pub indices: Vec<i128>, pub offset_int: i128,
    pub exponent: i64, pub scale: i64, pub label_decade: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate { pub ticks: Vec<f64>, pub texts: Vec<String>, pub payload: Option<LatticePayload> }

#[derive(Clone, Debug, PartialEq)]
/// `centred`: an inward axis' labels that do not fit inside it, drawn centred on their ticks (v2's way).
pub struct Placement { pub ticks: Vec<f64>, pub texts: Vec<String>, pub size_pt: f64, pub payload: Option<LatticePayload>, pub centred: bool }

fn empty() -> Placement { Placement { ticks: Vec::new(), texts: Vec::new(), size_pt: MIN_TICK_LABEL_PT, payload: None, centred: false } }

/// `compact_number`: as many significant digits as `length` characters allow.
pub fn compact_number(value: f64, length: Option<usize>) -> String {
    if !value.is_finite() { return "nan".into(); }
    fn normalized(text: String) -> String {
        if !text.to_lowercase().contains('e') { return text; }
        let lower = text.to_lowercase();
        let (mantissa, exponent) = lower.split_once('e').unwrap();
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{mantissa}e{}", exponent.parse::<i64>().unwrap())
    }
    let Some(length) = length else {
        let general = format_g(value, 4);
        return if general.to_lowercase().contains('e') { normalized(format_e(value, 1)) } else { general };
    };
    let maximum = length.max(1);
    for significant in (1..=maximum.max(4)).rev() {
        let general = normalized(format_g(value, significant));
        if general.chars().count() <= maximum { return general; }
        let scientific = normalized(format_e(value, significant - 1));
        if scientific.chars().count() <= maximum { return scientific; }
    }
    normalized(format_e(value, 0))
}

pub fn count_label(value: f64, length: Option<usize>) -> String {
    match length { None => format!("{value:.0}"), Some(_) => compact_number(value, length) }
}

/// `_fmt_scaled_int`: an integer times 10^exp10, spelled exactly.
pub fn fmt_scaled_int(value: i128, exp10: i64, force_sign: bool) -> String {
    if value == 0 { return if force_sign { "+0" } else { "0" }.into(); }
    let sign = if value < 0 { "-" } else if force_sign { "+" } else { "" };
    let base = value.unsigned_abs().to_string();
    if exp10 >= 0 { return format!("{sign}{base}{}", "0".repeat(exp10 as usize)); }
    let places = (-exp10) as usize;
    let padded = if base.len() <= places { format!("{}{base}", "0".repeat(places - base.len() + 1)) } else { base };
    let (quotient, remainder) = padded.split_at(padded.len() - places);
    let fraction = remainder.trim_end_matches('0');
    if fraction.is_empty() { format!("{sign}{quotient}") } else { format!("{sign}{quotient}.{fraction}") }
}

/// The ladder both locators share.
#[derive(Clone, Debug)]
pub struct Ladder {
    pub floor: usize,
    pub max_ticks: usize,
    pub label_pt: f64,
    pub drawn_pt: f64,
    pub ticks: Vec<f64>,
    pub texts: Vec<String>,
    /// The labels last stored are drawn centred although the axis is inward (they do not fit inside it).
    pub centred: bool,
    price_across: bool,
}

impl Ladder {
    fn new(max_ticks: usize, label_pt: f64, floor: usize, price_across: bool) -> Self {
        Ladder { floor, max_ticks: max_ticks.max(2), label_pt, drawn_pt: label_pt, ticks: Vec::new(), texts: Vec::new(), centred: false, price_across }
    }

    fn positions(&self, lower: f64, upper: f64, ticks: &[f64], extent: f64, scale: &Scale) -> Vec<f64> {
        let mut values = vec![lower, upper];
        values.extend_from_slice(ticks);
        let mapped = scale.map(&values);
        let (low, high) = (mapped[0], mapped[1]);
        if high == low { return vec![0.0; ticks.len()]; }
        mapped[2..].iter().map(|&value| (value - low) / (high - low) * extent).collect()
    }

    fn size_ladder(&self) -> Vec<f64> {
        if self.label_pt > MIN_TICK_LABEL_PT { vec![self.label_pt, MIN_TICK_LABEL_PT] } else { vec![self.label_pt] }
    }

    #[allow(clippy::too_many_arguments)]
    fn judge(&self, lower: f64, upper: f64, candidate: &Candidate, size_pt: f64, room: bool, clear: bool,
             geometry: Option<Geometry>, scale: &Scale, measure: &dyn Measure) -> Option<Placement> {
        let (ticks, texts) = (&candidate.ticks, &candidate.texts);
        let count = ticks.len();
        let floor = if count > 0 { self.floor.min(count) } else { self.floor };
        let distinct = texts.iter().collect::<std::collections::HashSet<_>>().len();
        if count < self.floor || distinct < floor { return None; }
        let placement = || Placement { ticks: ticks.clone(), texts: texts.clone(), size_pt, payload: candidate.payload.clone(), centred: false };
        let Some(geometry) = geometry else { return Some(placement()); };
        if upper <= lower || geometry.extent <= 0.0 { return Some(placement()); }
        let widths: Vec<f64> = texts.iter().map(|text| measure.size(text, size_pt).0).collect();
        let line = measure.line(size_pt);
        let along: Vec<f64> = if geometry.horizontal { widths.clone() } else { vec![line; count] };
        if room && self.price_across {
            if let Some(across) = geometry.across {
                let largest = widths.iter().map(|&width| if geometry.horizontal { line } else { width })
                    .fold(f64::NEG_INFINITY, f64::max);
                if largest > across + 1e-9 { return None; }
            }
        }
        let positions = self.positions(lower, upper, ticks, geometry.extent, scale);
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| positions[a].partial_cmp(&positions[b]).unwrap());
        let end = |position: f64| if !geometry.inward { 0 } else if position <= 1e-9 * geometry.extent { -1 }
            else if position >= geometry.extent * (1.0 - 1e-9) { 1 } else { 0 };
        let boxes: Vec<(f64, f64)> = order.iter().map(|&i| match end(positions[i]) {
            -1 => (positions[i], positions[i] + along[i]),
            1 => (positions[i] - along[i], positions[i]),
            _ => (positions[i] - along[i] / 2.0, positions[i] + along[i] / 2.0),
        }).collect();
        if room && geometry.inward && boxes.iter().any(|&(low, high)| low < -1e-9 || high > geometry.extent + 1e-9) { return None; }
        if clear {
            let digit = measure.size("0", size_pt).0;
            for pair in boxes.windows(2) {
                if pair[1].0 - pair[0].1 < digit - 1e-9 { return None; }
            }
        }
        Some(placement())
    }

    /// The labels are v2's (judged centred); an inward axis hangs them inside its ends
    /// only where they fit there, else draws them centred as v2 did -- never a
    /// coarser short form for the sake of the inward hang.
    fn settle(&self, lower: f64, upper: f64, tiers: &[Vec<Candidate>], geometry: Option<Geometry>,
              scale: &Scale, measure: &dyn Measure) -> Placement {
        let Some(inward) = geometry.filter(|geometry| geometry.inward) else { return self.settle_as(lower, upper, tiers, geometry, scale, measure) };
        let chosen = self.settle_as(lower, upper, tiers, Some(Geometry { inward: false, ..inward }), scale, measure);
        let candidate = Candidate { ticks: chosen.ticks.clone(), texts: chosen.texts.clone(), payload: chosen.payload.clone() };
        let fits = self.judge(lower, upper, &candidate, chosen.size_pt, true, true, Some(inward), scale, measure).is_some();
        Placement { centred: !fits, ..chosen }
    }

    fn settle_as(&self, lower: f64, upper: f64, tiers: &[Vec<Candidate>], geometry: Option<Geometry>,
                 scale: &Scale, measure: &dyn Measure) -> Placement {
        let sizes = self.size_ladder();
        let ranked: Vec<Vec<&Candidate>> = tiers.iter().map(|tier| tier.iter()
            .filter(|c| self.floor <= c.ticks.len() && c.ticks.len() <= self.max_ticks).collect()).collect();
        for &size in &sizes {
            for tier in &ranked {
                let mut chosen = None;
                for candidate in tier {
                    if let Some(placement) = self.judge(lower, upper, candidate, size, true, true, geometry, scale, measure) {
                        chosen = Some(placement);
                    }
                }
                if let Some(chosen) = chosen { return chosen; }
            }
        }
        for tier in &ranked {
            let Some(candidate) = tier.first() else { continue; };
            let last = *sizes.last().unwrap();
            return self.judge(lower, upper, candidate, last, true, false, geometry, scale, measure)
                .or_else(|| self.judge(lower, upper, candidate, last, false, false, geometry, scale, measure))
                .unwrap_or_else(empty);
        }
        empty()
    }

    fn store(&mut self, placement: &Placement, reverse: bool) {
        let mut ticks = placement.ticks.clone();
        let mut texts = placement.texts.clone();
        if reverse { ticks.reverse(); texts.reverse(); }
        self.ticks = ticks;
        self.texts = texts;
        if (placement.size_pt - self.drawn_pt).abs() >= 1.0e-9 { self.drawn_pt = placement.size_pt; }
        self.centred = placement.centred;
    }

    fn nearest(&self, value: f64) -> Option<usize> {
        if self.ticks.is_empty() || !value.is_finite() { return None; }
        let mut best = 0;
        for (index, &tick) in self.ticks.iter().enumerate() {
            if (value - tick).abs() < (value - self.ticks[best]).abs() { best = index; }
        }
        Some(best)
    }

    /// What the tick nearest `value` says.
    pub fn text_for(&self, value: f64) -> String {
        match self.nearest(value) { Some(index) if index < self.texts.len() => self.texts[index].clone(), _ => String::new() }
    }
}

/// Inputs that decide one locator answer; equal keys return the stored answer.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoKey { pub vmin: u64, pub vmax: u64, pub geometry: Option<Geometry>, pub scale_identity: u64 }

impl MemoKey {
    pub fn new(vmin: f64, vmax: f64, geometry: Option<Geometry>, scale_identity: u64) -> Self {
        MemoKey { vmin: vmin.to_bits(), vmax: vmax.to_bits(), geometry, scale_identity }
    }
}

/// Large common offsets apart from short coordinate labels, on the (1, 2, 5)
/// lattice; a logarithmic axis offers its decade families first.
#[derive(Clone, Debug)]
pub struct SmartOffsetLocator {
    pub ladder: Ladder,
    pub steps: Vec<i64>,
    pub oom: i64,
    pub settled: Option<(i64, i64)>,
    pub k: i64,
    pub m: i64,
    pub c: f64,
    pub c_int: i128,
    pub c_exp: i64,
    pub step: i64,
    pub n_array: Vec<i128>,
    memo: Option<MemoKey>,
}

impl SmartOffsetLocator {
    pub fn new(label_pt: f64) -> Self {
        SmartOffsetLocator { ladder: Ladder::new(8, label_pt, TICKS_FLOOR, false), steps: vec![1, 2, 5], oom: 3,
            settled: None, k: 0, m: 0, c: 0.0, c_int: 0, c_exp: 0, step: 1, n_array: Vec::new(), memo: None }
    }

    /// `_lay_out`: ticks, their indices and the offset of one unit.
    pub fn lay_out(&self, lower: f64, upper: f64, step: i64, decade: i64, use_offset: bool, max_count: Option<usize>)
        -> (Vec<f64>, Vec<i128>, i128, i64, i64, i64) {
        let lower_decimal = Decimal::from_f64(lower);
        let upper_decimal = Decimal::from_f64(upper);
        let unit = Decimal::from_i128(step as i128).scaleb(decade);
        let mut exponent = decade;
        let mut offset_int: i128 = 0;
        let mut offset = Decimal::from_i128(0);
        if use_offset {
            let multiple = lower_decimal.div(&unit).to_integral(Integral::Floor);
            offset_int = multiple * step as i128;
            offset = Decimal::from_i128(offset_int).scaleb(exponent);
            if !offset.to_f64().is_finite() { offset_int = 0; exponent = 0; offset = Decimal::from_i128(0); }
        }
        let first = lower_decimal.sub(&offset).div(&unit).to_integral(Integral::Ceiling);
        let last = upper_decimal.sub(&offset).div(&unit).to_integral(Integral::Floor);
        if let Some(maximum) = max_count {
            if last - first + 1 > maximum as i128 { return (Vec::new(), Vec::new(), 0, 0, 0, decade); }
        }
        let mut ticks: Vec<f64> = Vec::new();
        let mut entries: Vec<(Decimal, i128)> = Vec::new();
        let mut n = first;
        while n <= last {
            let target = Decimal::from_i128(n).mul(&unit).add(&offset);
            let tick = target.to_f64();
            let error = Decimal::from_f64(tick).sub(&target).abs();
            match ticks.iter().position(|&existing| existing == tick) {
                None => { ticks.push(tick); entries.push((error, n)); }
                Some(index) => if error.cmp(&entries[index].0) == std::cmp::Ordering::Less { entries[index] = (error, n); },
            }
            n += 1;
        }
        let indices: Vec<i128> = entries.iter().map(|entry| entry.1).collect();
        let largest = indices.iter().map(|index| index.abs()).max().unwrap_or(0);
        let residual = Decimal::from_i128(largest * step as i128).scaleb(decade);
        let mut scale = 0;
        let mut label_decade = decade;
        if !indices.is_empty()
            && (decade <= -self.oom || residual.cmp(&Decimal::from_i128(1).scaleb(self.oom + 1)) != std::cmp::Ordering::Less) {
            scale = decade;
            label_decade = 0;
        }
        (ticks, indices, offset_int, exponent, scale, label_decade)
    }

    fn carries_a_common_part(&self, indices: &[i128], step: i64) -> bool {
        indices.iter().map(|index| index.abs()).max().unwrap_or(0) * step as i128 >= 10i128.pow(self.oom as u32)
    }

    fn lattice_candidate(&self, lower: f64, upper: f64, step: i64, decade: i64) -> Option<Candidate> {
        let mut laid = self.lay_out(lower, upper, step, decade, false, Some(self.ladder.max_ticks));
        if laid.0.len() < self.ladder.floor { return None; }
        if self.carries_a_common_part(&laid.1, step) {
            let offset = self.lay_out(lower, upper, step, decade, true, Some(self.ladder.max_ticks));
            if offset.0.len() >= self.ladder.floor { laid = offset; }
        }
        let (ticks, indices, offset_int, exponent, scale, label_decade) = laid;
        let texts = indices.iter().map(|&index| fmt_scaled_int(index * step as i128, label_decade, false)).collect();
        Some(Candidate { ticks, texts, payload: Some(LatticePayload { step, decade, indices, offset_int, exponent, scale, label_decade }) })
    }

    fn lattice(&self, lower: f64, upper: f64) -> Vec<Candidate> {
        let span = upper - lower;
        let exponent = span.log10().floor() as i64;
        let mut lattice: Vec<(i64, i64)> = Vec::new();
        let mut decade = exponent + 1;
        while decade > exponent - 4 {
            for &step in &self.steps { lattice.push((step, decade)); }
            decade -= 1;
        }
        lattice.sort_by(|a, b| {
            let ka = -(a.0 as f64) * 10f64.powf(a.1 as f64);
            let kb = -(b.0 as f64) * 10f64.powf(b.1 as f64);
            ka.partial_cmp(&kb).unwrap()
        });
        let mut candidates = Vec::new();
        for (step, decade) in lattice {
            let unit = step as f64 * 10f64.powf(decade as f64);
            if unit <= 0.0 || !unit.is_finite() { continue; }
            match self.lattice_candidate(lower, upper, step, decade) {
                None => if !candidates.is_empty() { break; },
                Some(candidate) => candidates.push(candidate),
            }
        }
        candidates
    }

    fn decades(&self, lower: f64, upper: f64) -> Vec<Candidate> {
        if lower <= 0.0 { return Vec::new(); }
        let first = lower.log10().floor() as i64;
        let last = upper.log10().ceil() as i64;
        let decades: Vec<i64> = (first..=last).filter(|&k| {
            let value = 10f64.powf(k as f64);
            lower - 1e-12 * upper <= value && value <= upper + 1e-12 * upper
        }).collect();
        let families: [(usize, &[i64]); 5] = [(3, &[1]), (2, &[1]), (1, &[1]), (1, &[1, 2, 5]), (1, &[1, 2, 3, 4, 5, 6, 7, 8, 9])];
        let mut candidates: Vec<Candidate> = Vec::new();
        for (stride, subs) in families {
            let strided: Vec<i64> = decades.iter().copied().step_by(stride).collect();
            let mut ticks = Vec::new();
            let mut texts = Vec::new();
            for k in first..=last {
                for &sub in subs {
                    let value = sub as f64 * 10f64.powf(k as f64);
                    if !(lower <= value && value <= upper) { continue; }
                    if sub == 1 && !strided.contains(&k) { continue; }
                    if sub != 1 && stride != 1 { continue; }
                    ticks.push(value);
                    texts.push(fmt_scaled_int(sub as i128, k, false));
                }
            }
            if ticks.len() < self.ladder.floor || ticks.len() > self.ladder.max_ticks { continue; }
            if candidates.last().is_some_and(|previous| previous.ticks == ticks) { continue; }
            candidates.push(Candidate { ticks, texts, payload: None });
        }
        candidates
    }

    fn unit(&self, lower: f64, upper: f64, geometry: Option<Geometry>, scale: &Scale, measure: &dyn Measure) -> Placement {
        let logarithmic = scale.is_log();
        if let (Some(settled), false) = (self.settled, logarithmic) {
            if let Some(held) = self.lattice_candidate(lower, upper, settled.0, settled.1) {
                if let Some(placement) = self.ladder.judge(lower, upper, &held, self.ladder.label_pt, true, true, geometry, scale, measure) {
                    return placement;
                }
            }
        }
        let tier = if logarithmic {
            let mut tier = self.decades(lower, upper);
            tier.extend(self.lattice(lower, upper));
            tier
        } else { self.lattice(lower, upper) };
        self.ladder.settle(lower, upper, &[tier], geometry, scale, measure)
    }

    /// `tick_values`, given the axis' painted geometry and scale.
    pub fn tick_values(&mut self, vmin: f64, vmax: f64, geometry: Option<Geometry>, scale: &Scale,
                       scale_identity: u64, measure: &dyn Measure) -> Vec<f64> {
        let key = MemoKey::new(vmin, vmax, geometry, scale_identity);
        if self.memo.as_ref() == Some(&key) { return self.ladder.ticks.clone(); }
        let (lower, upper) = if vmin <= vmax { (vmin, vmax) } else { (vmax, vmin) };
        if !(lower.is_finite() && upper.is_finite()) || lower == upper {
            self.ladder.store(&empty(), false);
            self.k = 0; self.m = 0; self.c_int = 0; self.c_exp = 0; self.c = 0.0; self.step = 1; self.n_array.clear();
            self.memo = Some(key);
            return self.ladder.ticks.clone();
        }
        let placement = self.unit(lower, upper, geometry, scale, measure);
        self.adopt_layout(&placement, vmin > vmax, Some(key));
        self.settled = placement.payload.as_ref().map(|p| (p.step, p.decade));
        self.ladder.ticks.clone()
    }

    /// `layout_candidates`: sparser choices at the drawn size for final text layout.
    pub fn layout_candidates(&self, view: (f64, f64), geometry: Option<Geometry>, scale: &Scale, measure: &dyn Measure) -> Vec<Placement> {
        let (lower, upper) = if view.0 <= view.1 { view } else { (view.1, view.0) };
        let mut candidates = if scale.is_log() { self.decades(lower, upper) } else { Vec::new() };
        candidates.extend(self.lattice(lower, upper));
        let mut indexed: Vec<(usize, Candidate)> = candidates.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.ticks.len().cmp(&a.1.ticks.len()).then(a.0.cmp(&b.0)));
        indexed.into_iter().filter(|(_, candidate)| candidate.ticks.len() < self.ladder.ticks.len())
            .filter_map(|(_, candidate)| self.ladder.judge(lower, upper, &candidate, self.ladder.drawn_pt, true, true, geometry, scale, measure))
            .collect()
    }

    /// `adopt_layout`: accept final ticks and their formatter operands.
    pub fn adopt_layout(&mut self, placement: &Placement, reverse: bool, memo: Option<MemoKey>) {
        match &placement.payload {
            None => { self.step = 1; self.m = 0; self.k = 0; self.c_int = 0; self.c_exp = 0; self.c = 0.0; self.n_array.clear(); }
            Some(p) => {
                self.step = p.step; self.m = p.label_decade; self.k = p.scale; self.c_int = p.offset_int; self.c_exp = p.exponent;
                self.c = Decimal::from_i128(p.offset_int).scaleb(p.exponent).to_f64();
                self.n_array = p.indices.clone();
                if reverse { self.n_array.reverse(); }
            }
        }
        self.ladder.store(placement, reverse);
        self.memo = memo;
    }

    /// Record which inputs the adopted answer belongs to (the caller's current view).
    pub fn remember(&mut self, key: MemoKey) { self.memo = Some(key); }

    /// `_tick_cache_key = None`: the next `tick_values` asks the ladder again.
    pub fn forget_memo(&mut self) { self.memo = None; }

    pub fn forget_settled_step(&mut self) { self.settled = None; }

    pub const PLAIN_OFFSET_CHARACTERS: usize = 8;

    fn format_constant(&self) -> String {
        let value = self.c_int;
        if value == 0 { return String::new(); }
        let plain = fmt_scaled_int(value, self.c_exp, true);
        if plain.chars().count() <= Self::PLAIN_OFFSET_CHARACTERS { return plain; }
        let magnitude = value.unsigned_abs().to_string();
        let significant = magnitude.trim_end_matches('0');
        let power = self.c_exp + (magnitude.len() - significant.len()) as i64 + significant.len() as i64 - 1;
        let mantissa = format!("{}{}", &significant[..1], if significant.len() > 1 { format!(".{}", &significant[1..]) } else { String::new() });
        let scientific = format!("{}{mantissa}e{power}", if value < 0 { '-' } else { '+' });
        if scientific.chars().count() < plain.chars().count() { scientific } else { plain }
    }

    /// The formatter's offset text for the `x` or `y` axis.
    pub fn offset_text(&self, horizontal: bool) -> String {
        let mut parts = Vec::new();
        if self.k != 0 { parts.push(format!("×1e{}", self.k)); }
        let constant = self.format_constant();
        if !constant.is_empty() { parts.push(constant); }
        match parts.len() {
            0 => String::new(),
            2 => parts.join(if horizontal { "\n" } else { " " }),
            _ => parts.concat(),
        }
    }
}

/// Declared, evenly spaced ticks whose spelling and size the ladder decides.
pub struct DeclaredLocator<'a> {
    pub ladder: Ladder,
    pub count: usize,
    pub text: &'a dyn Fn(f64, Option<usize>) -> String,
    pub text_lengths: Vec<Option<usize>>,
    pub zero_optional: bool,
    pub span: Option<(f64, f64)>,
    memo: Option<MemoKey>,
}

pub(crate) fn linspace(first: f64, last: f64, count: usize) -> Vec<f64> {
    // numpy.linspace: start + step * i, the last sample set to stop exactly.
    let divisions = (count - 1) as f64;
    let step = (last - first) / divisions;
    let mut values: Vec<f64> = (0..count).map(|i| if step == 0.0 { first + (i as f64 / divisions) * (last - first) } else { first + i as f64 * step }).collect();
    if count > 1 { values[count - 1] = last; }
    values
}

impl<'a> DeclaredLocator<'a> {
    pub fn new(count: usize, text: &'a dyn Fn(f64, Option<usize>) -> String, text_lengths: Vec<Option<usize>>,
               zero_optional: bool, label_pt: f64, span: Option<(f64, f64)>) -> Self {
        DeclaredLocator { ladder: Ladder::new(count.max(2), label_pt, if zero_optional { 1 } else { 2 }, true),
            count, text, text_lengths: if text_lengths.is_empty() { vec![None] } else { text_lengths },
            zero_optional, span, memo: None }
    }

    pub fn set_memo(&mut self, memo: Option<MemoKey>) { self.memo = memo; }

    pub fn memo(&self) -> Option<MemoKey> { self.memo.clone() }

    pub fn omit_optional_zero(&mut self) {
        if self.zero_optional && self.ladder.ticks.contains(&0.0) {
            let kept: Vec<(f64, String)> = self.ladder.ticks.iter().zip(&self.ladder.texts)
                .filter(|(tick, _)| **tick != 0.0).map(|(tick, text)| (*tick, text.clone())).collect();
            let placement = Placement { ticks: kept.iter().map(|k| k.0).collect(), texts: kept.into_iter().map(|k| k.1).collect(),
                                        size_pt: self.ladder.drawn_pt, payload: None, centred: self.ladder.centred };
            self.ladder.store(&placement, false);
        }
    }

    pub fn tick_values(&mut self, vmin: f64, vmax: f64, geometry: Option<Geometry>, scale: &Scale,
                       scale_identity: u64, measure: &dyn Measure) -> Vec<f64> {
        let key = MemoKey::new(vmin, vmax, geometry, scale_identity);
        if self.memo.as_ref() == Some(&key) { return self.ladder.ticks.clone(); }
        let (lower, upper) = if vmin <= vmax { (vmin, vmax) } else { (vmax, vmin) };
        if !(lower.is_finite() && upper.is_finite()) || lower == upper {
            self.ladder.store(&empty(), false);
            self.memo = Some(key);
            return self.ladder.ticks.clone();
        }
        let (first, last) = match self.span { None => (lower, upper), Some((a, b)) => if a <= b { (a, b) } else { (b, a) } };
        if !(first.is_finite() && last.is_finite()) || first == last {
            self.ladder.store(&empty(), false);
            self.memo = Some(key);
            return self.ladder.ticks.clone();
        }
        let ticks = linspace(first, last, self.count);
        let mut tiers = Vec::new();
        for &length in &self.text_lengths {
            let texts: Vec<String> = ticks.iter().map(|&tick| (self.text)(tick, length)).collect();
            let mut tier = Vec::new();
            if self.zero_optional && ticks[0] == 0.0 {
                tier.push(Candidate { ticks: ticks[1..].to_vec(), texts: texts[1..].to_vec(), payload: None });
            }
            tier.push(Candidate { ticks: ticks.clone(), texts, payload: None });
            tiers.push(tier);
        }
        let placement = self.ladder.settle(lower, upper, &tiers, geometry, scale, measure);
        self.ladder.store(&placement, vmin > vmax);
        self.memo = Some(key);
        self.ladder.ticks.clone()
    }
}
