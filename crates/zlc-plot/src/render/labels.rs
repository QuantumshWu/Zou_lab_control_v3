//! Label text rules (`rendering.py` `_literal_text`, `_quantity_label`,
//! `_state_label`, `_EXPLICIT_UNIT_SUFFIX`).
//!
//! Producer and operator text is a NAME: its dollar signs are escaped so the
//! text layer never reads them as math. Only the fit catalogue writes math.

/// Producer and operator text, made safe for a text artist: `$` becomes `\$`.
pub fn literal_text(text: &str) -> String { text.replace('$', r"\$") }

/// `(?:\[[^\[\]]+\]|\([^()]+\))\s*$`: the label already ends in a bracketed unit.
pub fn has_explicit_unit_suffix(label: &str) -> bool {
    let trimmed = label.trim_end_matches(|c: char| c.is_whitespace());
    let Some(last) = trimmed.chars().last() else { return false };
    let (open, close, forbidden): (char, char, &[char]) = match last {
        ']' => ('[', ']', &['[', ']']),
        ')' => ('(', ')', &['(', ')']),
        _ => return false,
    };
    let body = &trimmed[..trimmed.len() - close.len_utf8()];
    match body.rfind(open) {
        Some(at) => {
            let inner = &body[at + open.len_utf8()..];
            !inner.is_empty() && !inner.contains(forbidden)
        }
        None => false,
    }
}

/// The axis text for one quantity: an explicit label, else the quantity's own
/// name, with the display unit appended unless the label carries one.
/// `name` is the quantity's label (or None for the fallback), `unit` its display unit symbol.
pub fn quantity_label(name: Option<&str>, unit: Option<&str>, fallback: &str, explicit: Option<&str>) -> String {
    let label = match explicit {
        Some(explicit) => {
            if explicit.is_empty() { return String::new(); }
            explicit.to_owned()
        }
        None => literal_text(name.filter(|name| !name.is_empty()).unwrap_or(fallback)),
    };
    let unit = literal_text(unit.unwrap_or(""));
    if unit.is_empty() || unit == "1" { return label; }
    if explicit.is_some() && has_explicit_unit_suffix(&label) { return label; }
    let suffix = format!("({unit})");
    if label.trim_end().ends_with(&suffix) { label } else { format!("{label} {suffix}") }
}

/// The literal text one label shows: the operator's edit, else the authored spec text.
pub fn state_label(edited: Option<&str>, fallback: Option<&str>) -> Option<String> {
    edited.or(fallback).map(literal_text)
}
