//! Canonical scan authoring: dormant input banks remain authored data; only
//! the selected banks compile. No device access or Cartesian expansion occurs.
use std::collections::{BTreeMap, HashSet};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value, json};
use zlc_core::scan::{AxisSpec, Plan, PlanSpec, Result};
use zlc_core::units::DEFAULT_UNITS;

const MAX_RANGE_POINTS: usize = 100_000;

struct Strict(Value);
impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> std::result::Result<Self, D::Error> {
        struct Read;
        impl<'de> Visitor<'de> for Read {
            type Value = Strict;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("strict JSON") }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Strict, E> { Ok(Strict(v.into())) }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Strict, E> { Ok(Strict(v.into())) }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Strict, E> { Ok(Strict(v.into())) }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Strict, E> {
                serde_json::Number::from_f64(v).map(|v| Strict(Value::Number(v))).ok_or_else(|| E::custom("numbers must be finite"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Strict, E> { Ok(Strict(v.into())) }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Strict, E> { Ok(Strict(Value::Null)) }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Strict, A::Error> {
                let mut values = Vec::new();
                while let Some(Strict(value)) = seq.next_element()? { values.push(value); }
                Ok(Strict(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Strict, A::Error> {
                let mut values = Map::new();
                while let Some((key, Strict(value))) = map.next_entry::<String, Strict>()? {
                    if values.insert(key.clone(), value).is_some() { return Err(de::Error::custom(format!("duplicate key {key:?}"))); }
                }
                Ok(Strict(Value::Object(values)))
            }
        }
        decoder.deserialize_any(Read)
    }
}

fn text<'a>(value: &'a Value, what: &str) -> Result<&'a str> { value.as_str().ok_or_else(|| format!("{what} must be text")) }
fn numbers(value: &Value, what: &str) -> Result<Vec<f64>> {
    value.as_array().ok_or_else(|| format!("{what} must be a numeric list"))?.iter()
        .map(|v| v.as_f64().filter(|v| v.is_finite()).ok_or_else(|| format!("{what} must be a finite numeric list"))).collect()
}
fn host(port: &str) -> bool { ["manual:", "device:", "api:"].iter().any(|prefix| port.starts_with(prefix)) }

/// A generated Range keeps NumPy/previous UI ordering of operations and its
/// exact final endpoint; linked legacy ranges retain their own operation order.
pub fn range_values(start: f64, stop: f64, count: usize) -> Result<Vec<f64>> {
    if count == 0 || count > MAX_RANGE_POINTS { return Err(format!("Range points must be between 1 and {MAX_RANGE_POINTS}")); }
    if !start.is_finite() || !stop.is_finite() { return Err("Range endpoints must be finite".into()); }
    if count == 1 { return Ok(vec![start]); }
    let step = (stop - start) / (count - 1) as f64;
    let values: Vec<_> = (0..count).map(|i| if i == count - 1 { stop } else { i as f64 * step + start }).collect();
    if values.iter().any(|v| !v.is_finite()) { return Err("Range must produce finite numbers".into()); }
    Ok(values)
}

fn range(row: &Value) -> Result<Option<(f64, f64, usize)>> {
    let Some(range) = row.get("range") else { return Ok(None); };
    let fields = range.as_object().ok_or("Range must contain start, stop and count")?;
    if fields.len() != 3 || !["start", "stop", "count"].iter().all(|key| fields.contains_key(*key)) {
        return Err("Range must contain only start, stop and count".into());
    }
    let start = range["start"].as_f64().filter(|v| v.is_finite()).ok_or("Range start must be finite")?;
    let stop = range["stop"].as_f64().filter(|v| v.is_finite()).ok_or("Range stop must be finite")?;
    let count = range["count"].as_u64().and_then(|v| usize::try_from(v).ok()).filter(|v| (1..=MAX_RANGE_POINTS).contains(v))
        .ok_or_else(|| format!("Range points must be between 1 and {MAX_RANGE_POINTS}"))?;
    Ok(Some((start, stop, count)))
}

/// A linked parameter: it is not scanned, it follows its axis by a formula of `x` (a
/// coordinate of that axis, one value per point). Its `formula` and unit are all it holds.
fn normalize_link(entry: &Value) -> Result<Value> {
    let fields = entry.as_object().ok_or("a linked parameter is its port, its formula of x and an optional unit")?;
    if !fields.contains_key("port") || fields.keys().any(|key| !["port", "unit", "mode", "formula", "values", "value_text", "range"].contains(&key.as_str())) {
        return Err("a linked parameter is its port, its formula of x and an optional unit".into());
    }
    if let Some(mode) = entry.get("mode") {
        if text(mode, "linked mode")? != "formula" { return Err("a linked parameter follows its axis by a formula of x; Range and Values links are not offered".into()); }
    }
    let port = text(&entry["port"], "scan port")?;
    let unit = entry.get("unit").map(|v| text(v, "scan unit")).transpose()?.unwrap_or("");
    let formula = entry.get("formula").map(|v| text(v, "formula")).transpose()?.unwrap_or("x");
    Ok(json!({"port":port,"unit":unit,"mode":"formula","formula":formula,"values":[],"value_text":""}))
}

fn normalize_row(entry: &Value, linked: bool, index: usize) -> Result<Value> {
    if linked { return normalize_link(entry); }
    let fields = entry.as_object().ok_or("scan axis fields must be port, values, and optional unit, mode, value_text")?;
    let allowed: HashSet<&str> = ["port", "values", "range", "unit", "mode", "value_text", "formula", "links", "link_enabled", "coordinate_of"].into_iter().collect();
    if !fields.contains_key("port") || (!fields.contains_key("values") && !fields.contains_key("range")) || fields.keys().any(|key| !allowed.contains(key.as_str())) {
        return Err("scan axis fields must be port, values or range, and optional unit, mode, value_text, links".into());
    }
    let port = text(&entry["port"], "scan port")?;
    let unit = entry.get("unit").map(|v| text(v, "scan unit")).transpose()?.unwrap_or("");
    let mode = entry.get("mode").map(|v| text(v, "scan mode")).transpose()?.unwrap_or("range");
    if !["range", "values"].contains(&mode) { return Err(format!("scan axis {index} ({port:?}): mode must be range or values")); }
    let values = entry.get("values").map(|v| numbers(v, "Range values")).transpose()?.unwrap_or_default();
    let value_text = entry.get("value_text").map(|v| text(v, "value_text")).transpose()?.unwrap_or("");
    let mut row = json!({"port":port,"values":values,"unit":unit,"mode":mode,"value_text":value_text});
    if let Some((start, stop, count)) = range(entry)? { row["range"] = json!({"start":start,"stop":stop,"count":count}); }
    if let Some(parent) = entry.get("coordinate_of") {
        if !parent.is_null() && !parent.is_string() { return Err("coordinate_of must be text or null".into()); }
        row["coordinate_of"] = parent.clone();
    }
    if fields.contains_key("links") || fields.contains_key("link_enabled") {
        let enabled = entry.get("link_enabled").map(|v| v.as_bool().ok_or("link_enabled must be boolean")).transpose()?.unwrap_or(false);
        let links = entry.get("links").map(|v| v.as_array().ok_or("linked parameters must be a list")).transpose()?;
        row["link_enabled"] = json!(enabled);
        row["links"] = Value::Array(links.into_iter().flatten().map(|link| normalize_row(link, true, index)).collect::<Result<_>>()?);
    }
    Ok(row)
}

fn normalize(document: &Value) -> Result<Vec<Value>> {
    let parsed;
    let document = if let Some(source) = document.as_str() {
        if source.trim().is_empty() { return Err("the scan plan is empty; add an axis".into()); }
        parsed = serde_json::from_str::<Strict>(source).map_err(|e| format!("the scan plan: {e}"))?.0;
        &parsed
    } else { document };
    if document.as_object().is_none_or(|fields| fields.len() != 1 || !fields.contains_key("axes")) {
        return Err("a scan plan document carries only its axes".into());
    }
    let entries = document["axes"].as_array().ok_or("scan plan axes must be a list")?;
    let mut rows: Vec<Value> = entries.iter().enumerate().map(|(i, row)| normalize_row(row, false, i+1)).collect::<Result<_>>()?;
    rows.sort_by_key(|row| !(host(row["port"].as_str().unwrap()) || (row["link_enabled"] == true && row["links"].as_array().is_some_and(|links|
        links.iter().any(|link| host(link["port"].as_str().unwrap()))))));
    Ok(rows)
}

fn float_text(source: &str) -> Result<f64> {
    let source = source.trim();
    let bytes = source.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'_' && (i == 0 || i+1 == bytes.len() || !bytes[i-1].is_ascii_digit() || !bytes[i+1].is_ascii_digit()) {
            return Err("not a number".into());
        }
    }
    source.replace('_', "").parse::<f64>().map_err(|_| "not a number".into())
}

pub fn parse_values(source: &str) -> Result<Vec<f64>> {
    if source.trim().is_empty() { return Err("Values is empty; enter comma-separated numbers".into()); }
    source.split(',').enumerate().map(|(i, item)| {
        let value = float_text(item).map_err(|_| format!("Values item {} is not a number: {item:?}", i+1))?;
        if !value.is_finite() { return Err(format!("Values item {} must be finite", i+1)); }
        Ok(value)
    }).collect()
}

#[derive(Clone, Debug)]
enum Expr { Number(f64), Variable, Unary(bool, Box<Expr>), Binary(u8, Box<Expr>, Box<Expr>) }
impl Expr {
    fn nodes(&self) -> usize { match self { Self::Number(_) => 1, Self::Variable => 2, Self::Unary(_, child) => 2+child.nodes(), Self::Binary(_, a,b) => 2+a.nodes()+b.nodes() } }
    fn evaluate(&self, x: f64) -> Result<f64> {
        Ok(match self {
            Self::Number(v) => *v, Self::Variable => x,
            Self::Unary(negative, child) => child.evaluate(x)? * if *negative { -1.0 } else { 1.0 },
            Self::Binary(op,a,b) => {
                let (a,b) = (a.evaluate(x)?,b.evaluate(x)?);
                match op {
                    b'+' => a+b, b'-' => a-b, b'*' => a*b,
                    b'/' => { if b == 0.0 { return Err("cannot evaluate formula: float division by zero".into()); } a/b },
                    b'^' => { if b.abs() > 64.0 { return Err("formula exponent exceeds 64".into()); }
                        if a == 0.0 && b < 0.0 { return Err("cannot evaluate formula: 0.0 cannot be raised to a negative power".into()); }
                        a.powf(b) },
                    _ => unreachable!(),
                }
            }
        })
    }
}

struct Parser<'a> { source: &'a [u8], at: usize, depth: usize }
impl Parser<'_> {
    fn skip(&mut self) { while self.at < self.source.len() && self.source[self.at].is_ascii_whitespace() { self.at += 1; } }
    fn take(&mut self, token: &[u8]) -> bool {
        self.skip();
        if self.source[self.at..].starts_with(token) { self.at += token.len(); true } else { false }
    }
    fn sum(&mut self) -> Result<Expr> {
        let mut value = self.product()?;
        loop { let op = if self.take(b"+") { b'+' } else if self.take(b"-") { b'-' } else { break };
            value = Expr::Binary(op, Box::new(value), Box::new(self.product()?)); }
        Ok(value)
    }
    fn product(&mut self) -> Result<Expr> {
        let mut value = self.factor()?;
        loop { self.skip(); if self.source[self.at..].starts_with(b"**") { break; }
            let op = if self.take(b"*") { b'*' } else if self.take(b"/") { b'/' } else { break };
            value = Expr::Binary(op, Box::new(value), Box::new(self.factor()?)); }
        Ok(value)
    }
    fn factor(&mut self) -> Result<Expr> {
        self.depth += 1;
        if self.depth > 128 { return Err("formula is too complex".into()); }
        let result = if self.take(b"+") { Expr::Unary(false, Box::new(self.factor()?)) }
            else if self.take(b"-") { Expr::Unary(true, Box::new(self.factor()?)) }
            else { let atom = self.atom()?;
                if self.take(b"**") { Expr::Binary(b'^', Box::new(atom), Box::new(self.factor()?)) } else { atom } };
        self.depth -= 1;
        Ok(result)
    }
    fn atom(&mut self) -> Result<Expr> {
        if self.take(b"(") { let value = self.sum()?; if !self.take(b")") { return Err("invalid formula".into()); } return Ok(value); }
        if self.take(b"x") { return Ok(Expr::Variable); }
        self.skip(); let start = self.at;
        if self.source.get(start) == Some(&b'0') {
            if let Some(radix) = self.source.get(start+1).and_then(|v| match v.to_ascii_lowercase() { b'x'=>Some(16u32),b'o'=>Some(8),b'b'=>Some(2),_=>None }) {
                self.at += 2; let digits = self.at;
                while self.at < self.source.len() && (self.source[self.at].is_ascii_hexdigit() || self.source[self.at] == b'_') { self.at += 1; }
                let source = &self.source[digits..self.at];
                if source.is_empty() || source.last() == Some(&b'_') || source.windows(2).any(|v|v==b"__") { return Err("invalid formula".into()); }
                let mut bits = Vec::new();
                for &digit in source.iter().filter(|v| **v != b'_') {
                    let value = (digit as char).to_digit(radix).ok_or("invalid formula")?;
                    for shift in (0..radix.trailing_zeros()).rev() { bits.push(value & (1 << shift) != 0); }
                }
                let first = bits.iter().position(|bit|*bit).unwrap_or(bits.len());
                let bits = &bits[first..];
                if bits.len()>1024 { return Err("cannot evaluate formula: integer too large to convert to float".into()); }
                let mut top = bits.iter().take(53).fold(0u64,|v,bit| v*2+u64::from(*bit));
                if bits.len()>53 && bits[53] && (bits[54..].iter().any(|bit|*bit) || top%2==1) { top += 1; }
                return Ok(Expr::Number(top as f64 * 2f64.powi(bits.len().saturating_sub(53) as i32)));
            }
        }
        while self.at < self.source.len() {
            let c = self.source[self.at];
            if c.is_ascii_digit() || c == b'.' || c == b'_' || c == b'e' || c == b'E'
                || ((c == b'+' || c == b'-') && self.at > start && matches!(self.source[self.at-1], b'e'|b'E')) { self.at += 1; } else { break; }
        }
        if start == self.at { return Err("formula accepts numbers, x, parentheses and + - * / **".into()); }
        let source = std::str::from_utf8(&self.source[start..self.at]).map_err(|_| "invalid formula")?;
        if !source.contains(['.','e','E']) && source.starts_with('0') && source.bytes().any(|v|matches!(v,b'1'..=b'9')) {
            return Err("invalid formula".into());
        }
        Ok(Expr::Number(float_text(source).map_err(|_| "invalid formula")?))
    }
}

pub fn formula_values(source: &str, values: &[f64]) -> Result<Vec<f64>> {
    if source.chars().count() > 2048 { return Err("formula is too long".into()); }
    let mut parser = Parser { source:source.as_bytes(), at:0, depth:0 };
    let expression = parser.sum()?; parser.skip();
    if parser.at != parser.source.len() { return Err("formula accepts numbers, x, parentheses and + - * / **".into()); }
    if expression.nodes()+1 > 128 { return Err("formula is too complex".into()); }
    values.iter().map(|&x| { let value = expression.evaluate(x)?;
        if !value.is_finite() { return Err("formula must produce finite numbers".into()); } Ok(value) }).collect()
}

fn bank(row: &Value) -> Result<Vec<f64>> {
    if row["mode"] == "values" { return parse_values(row["value_text"].as_str().unwrap()); }
    if let Some((start,stop,count)) = range(row)? { return range_values(start,stop,count); }
    numbers(&row["values"], "Range values")
}

fn axis(row: &Value, values: Vec<f64>, parent: Option<String>) -> Result<AxisSpec> {
    let port = row["port"].as_str().unwrap();
    if let Some(name) = port.strip_prefix("manual:") { if name.trim().is_empty() { return Err("a manual axis carries a name".into()); } }
    let identity = if let Some(name) = port.strip_prefix("pulse:param:").or_else(||port.strip_prefix("api:")) { name.to_owned() }
        else if let Some(name) = port.strip_prefix("manual:") { name.trim().to_owned() }
        else if let Some(name) = port.strip_prefix("device:") { name.replacen(':', ".", 1) } else { port.to_owned() };
    Ok(AxisSpec { port:port.into(), identity:identity.clone(), label:identity, unit:row["unit"].as_str().unwrap().trim().into(),
        values, coordinate_of:parent, host_advanced:host(port) })
}

fn compile(rows: &[Value]) -> Result<Plan> {
    let mut axes = Vec::new();
    for (index,row) in rows.iter().enumerate() {
        let compile_row = || -> Result<Vec<AxisSpec>> {
            let values = bank(row)?;
            let mut axes = vec![axis(row, values.clone(), row["coordinate_of"].as_str().map(str::to_owned))?];
            if row["link_enabled"] == true {
                for linked in row["links"].as_array().into_iter().flatten() {
                    let played = formula_values(linked["formula"].as_str().unwrap(), &values)?;
                    if linked["port"].as_str().unwrap().starts_with("manual:") { return Err("a linked parameter must be a scan slot, API or device parameter".into()); }
                    axes.push(axis(linked, played, Some(row["port"].as_str().unwrap().into()))?);
                }
            }
            Ok(axes)
        };
        axes.extend(compile_row().map_err(|error| format!("scan axis {} ({:?}): {error}", index+1, row["port"].as_str().unwrap()))?);
    }
    Plan::new(PlanSpec { axes, scan_repeats:1, run_repeats:1 })
}

fn facts(row: &Value) -> Value {
    let explicit = row["values"].as_array().unwrap();
    let generated = range(row).unwrap();
    let (start,stop,count) = generated.unwrap_or_else(|| (explicit.first().and_then(Value::as_f64).unwrap_or(0.0),
        explicit.last().and_then(Value::as_f64).unwrap_or(0.0), explicit.len().max(1)));
    let magnitude = start.abs().max(stop.abs());
    let ulp = if magnitude == 0.0 { f64::from_bits(1) } else {
        let next = f64::from_bits(magnitude.to_bits()+1); next-magnitude
    };
    let uniform = generated.is_some() || explicit.len()<2 || range_values(start,stop,count).is_ok_and(|grid|
        grid.iter().zip(explicit).all(|(a,b)| (a-b.as_f64().unwrap()).abs() <= 4.0*ulp));
    let active = bank(row);
    let bounds = active.as_ref().ok().filter(|values| !values.is_empty()).map(|values|
        [values.iter().copied().fold(f64::INFINITY,f64::min),values.iter().copied().fold(f64::NEG_INFINITY,f64::max)]);
    let values_bounds=parse_values(row["value_text"].as_str().unwrap()).ok().map(|values|
        [values.iter().copied().fold(f64::INFINITY,f64::min),values.iter().copied().fold(f64::NEG_INFINITY,f64::max)]);
    json!({"port":row["port"],"range_start":start,"range_stop":stop,"range_count":count,"range_uniform":uniform,
        "mode":row["mode"],"value_text":row["value_text"],"active_bounds":bounds,"values_bounds":values_bounds,"problem":active.err().unwrap_or_default()})
}

/// Ordering/binding may inspect a filtered view whose linked parent is outside
/// that view. Full parent/cycle/coordinate admission remains Plan::new's job.
fn view_axes(args: &Value) -> Result<Vec<Value>> {
    let axes=args["axes"].as_array().ok_or("plan axes must be a list")?;
    if axes.is_empty() { return Err("a scan plan needs at least one axis".into()); }
    let mut names=HashSet::new();
    axes.iter().enumerate().map(|(index,axis)| {
        let axis=normalize_row(axis,false,index+1)?;
        let port=axis["port"].as_str().unwrap();
        if !names.insert(port.to_owned()) { return Err("a port may appear on one axis only".into()); }
        let values=numbers(&axis["values"],"axis values")?;
        if values.is_empty() { return Err(format!("axis {port:?} has no values to play")); }
        Ok(axis)
    }).collect()
}

fn order_view(args: &Value) -> Result<Value> {
    let axes=view_axes(args)?;
    let group=|index:usize| axes[index]["coordinate_of"].as_str().filter(|s|!s.is_empty()).unwrap_or(axes[index]["port"].as_str().unwrap());
    let host_groups:HashSet<_>=(0..axes.len()).filter(|index|host(axes[*index]["port"].as_str().unwrap())).map(group).collect();
    let mut order:Vec<_>=(0..axes.len()).collect();
    order.sort_by_key(|index|!host_groups.contains(group(*index)));
    Ok(json!({"order":order}))
}

fn bind_view(args: &Value) -> Result<Value> {
    let axes=view_axes(args)?;
    let ports=args["ports"].as_array().ok_or("scan ports must be a list")?;
    let ports:BTreeMap<_,_>=ports.iter().map(|port|Ok((text(&port["port"],"scan port")?,port))).collect::<Result<_>>()?;
    let mut bound=Vec::new();
    for axis in &axes {
        let name=axis["port"].as_str().unwrap();
        let Some(port)=ports.get(name) else {
            if let Some(reason)=args["withheld"].get(name).and_then(Value::as_str) {
                return Err(format!("{reason}; remove axis {name:?} or put a period inside the bracket"));
            }
            let offered=if ports.is_empty(){"nothing".into()}else{ports.keys().copied().collect::<Vec<_>>().join(", ")};
            return Err(format!("this pulse offers no scan port named {name:?}; it offers {offered}"));
        };
        let target=text(&port["unit"],"port unit")?;
        let source=axis["unit"].as_str().unwrap().trim();
        let source=if source.is_empty(){target}else{source};
        let low=port["lo"].as_f64().ok_or("port low bound must be numeric")?;
        let high=port["hi"].as_f64().ok_or("port high bound must be numeric")?;
        for value in numbers(&axis["values"],"axis values")? {
            let effective=if source==target {value}else{DEFAULT_UNITS.convert(value,source,target).map_err(|error|error.to_string())?};
            if !effective.is_finite(){return Err(format!("axis {name:?} converts to a non-finite port value"));}
            if effective<low || effective>high {
                return Err(format!("axis {name:?} plays {value:?} {source}, outside the port's range [{}, {}] {target}",
                    zlc_core::pyfmt::format_g(low,6),zlc_core::pyfmt::format_g(high,6)));
            }
        }
        bound.push(name);
    }
    Ok(json!({"ports":bound}))
}

/// The rows of a plan as written, read leniently: each axis's port, mode, unit, value text,
/// numeric values and links, with no rule applied (what an editor shows of a plan that
/// does not compile). None: not a readable plan document at all.
fn authored_rows(document: &Value) -> Option<Vec<Value>> {
    let parsed;
    let document = if let Some(source) = document.as_str() { parsed = serde_json::from_str::<Value>(source).ok()?; &parsed } else { document };
    fn authored_row(entry: &Value) -> Option<Value> {
        let text = |key: &str| entry.get(key).and_then(Value::as_str).unwrap_or("").to_owned();
        let mut row = json!({"port": entry.get("port")?.as_str()?, "unit": text("unit"), "value_text": text("value_text"),
            "mode": entry.get("mode").and_then(Value::as_str).unwrap_or("range"),
            "values": entry.get("values").and_then(Value::as_array).map(|values| values.iter().filter_map(Value::as_f64).collect::<Vec<_>>()).unwrap_or_default()});
        for key in ["range", "coordinate_of", "formula", "link_enabled"] { if let Some(value) = entry.get(key) { row[key] = value.clone(); } }
        if let Some(links) = entry.get("links").and_then(Value::as_array) { row["links"] = Value::Array(links.iter().filter_map(authored_row).collect()); }
        Some(row)
    }
    Some(document.get("axes")?.as_array()?.iter().filter_map(authored_row).collect())
}

/// Global read-only native operation used by draft validation, editor preview
/// and execution admission. The offline helper dispatches this same function.
pub fn dispatch(operation: &str, args: &Value) -> Result<Value> {
    match operation {
        "scan.order" => return order_view(args),
        "scan.bind" => return bind_view(args),
        "scan.values" => return Ok(json!({"values":parse_values(text(&args["text"], "Values")?)?})),
        "scan.formula" => return Ok(json!({"values":formula_values(text(&args["source"], "formula")?, &numbers(&args["values"], "formula values")?)?})),
        "scan.range" => { let row = json!({"range":{"start":args["start"],"stop":args["stop"],"count":args["count"]}});
            let (start,stop,count) = range(&row)?.unwrap(); return Ok(json!({"values":range_values(start,stop,count)?})); },
        "scan.compile" => {},
        _ => return Err(format!("unknown scan authoring operation {operation}")),
    }
    let mode = args.get("mode").and_then(Value::as_str).unwrap_or("compile");
    if !["normalize","compile","preview"].contains(&mode) { return Err("scan compile mode must be normalize, compile or preview".into()); }
    let rows = match normalize(&args["document"]) {
        Ok(rows) => rows,
        // An editor's preview of a plan one of whose rows is wrong (a count past the limit, a
        // value it cannot read) still shows every row as written, the reason beside them:
        // rows that vanished would be authored again from what is left.
        Err(error) if mode == "preview" => match authored_rows(&args["document"]) {
            Some(rows) => return Ok(json!({"document":{"axes":rows},"rows":rows,"axis_facts":[],"problem":error})),
            None => return Err(error),
        },
        Err(error) => return Err(error),
    };
    let mut result = json!({"document":{"axes":rows},"rows":rows});
    if mode == "normalize" { return Ok(result); }
    result["axis_facts"] = Value::Array(rows.iter().map(facts).collect());
    match compile(&rows) {
        Ok(plan) => {
            result["axes"] = Value::Array(plan.axes().iter().map(|axis| { let mut value = json!({"port":axis.port,"values":axis.values,"unit":axis.unit});
                if let Some(parent) = &axis.coordinate_of { value["coordinate_of"] = json!(parent); } value }).collect());
            result["shape"] = json!(plan.axes().iter().filter(|axis| axis.coordinate_of.is_none()).map(|axis|axis.values.len()).collect::<Vec<_>>());
            result["point_count"] = json!(plan.point_count()); result["outer_count"] = json!(plan.outer_count()); result["inner_count"] = json!(plan.inner_count());
            result["problem"] = json!("");
        },
        Err(error) if mode == "preview" => result["problem"] = json!(error),
        Err(error) => return Err(error),
    }
    Ok(result)
}
