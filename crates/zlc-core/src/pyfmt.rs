//! Python's `format(x, ".{p}e")` and `format(x, ".{p}g")` for binary64, from
//! the correctly rounded digits Rust's exponent formatting produces.

/// Sign, decimal digits and decimal exponent of `x` rounded to `significant` digits.
fn digits(x: f64, significant: usize) -> (bool, String, i32) {
    let text = format!("{:.*e}", significant.saturating_sub(1), x.abs());
    let (mantissa, exponent) = text.split_once('e').expect("exponent form");
    (x.is_sign_negative(), mantissa.replace('.', ""), exponent.parse().expect("exponent"))
}

fn special(x: f64) -> Option<String> {
    if x.is_nan() { Some("nan".into()) } else if x.is_infinite() { Some(if x < 0.0 { "-inf" } else { "inf" }.into()) } else { None }
}

fn exponent_suffix(exponent: i32) -> String {
    format!("e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.unsigned_abs())
}

/// `format(x, f".{precision}e")`.
pub fn format_e(x: f64, precision: usize) -> String {
    if let Some(text) = special(x) { return text; }
    let (negative, digits, exponent) = digits(x, precision + 1);
    let mut text = String::new();
    if negative { text.push('-'); }
    text.push_str(&digits[..1]);
    if precision > 0 { text.push('.'); text.push_str(&digits[1..]); }
    text + &exponent_suffix(exponent)
}

/// `format(x, f".{precision}g")`.
pub fn format_g(x: f64, precision: usize) -> String {
    if let Some(text) = special(x) { return text; }
    let precision = precision.max(1);
    let (negative, digits, exponent) = digits(x, precision);
    let mut text = String::new();
    if negative { text.push('-'); }
    if -4 <= exponent && exponent < precision as i32 {
        if exponent >= 0 {
            let integer = exponent as usize + 1;
            text.push_str(&digits[..integer]);
            let fraction = digits[integer..].trim_end_matches('0');
            if !fraction.is_empty() { text.push('.'); text.push_str(fraction); }
        } else {
            let fraction = format!("{}{}", "0".repeat((-exponent - 1) as usize), digits).trim_end_matches('0').to_owned();
            text.push('0');
            if !fraction.is_empty() { text.push('.'); text.push_str(&fraction); }
        }
    } else {
        text.push_str(&digits[..1]);
        let fraction = digits[1..].trim_end_matches('0');
        if !fraction.is_empty() { text.push('.'); text.push_str(fraction); }
        text.push_str(&exponent_suffix(exponent));
    }
    text
}

/// `repr(x)` for a float: the shortest round-tripping digits, fixed notation
/// for decimal-point positions in (-4, 16], exponent notation otherwise.
pub fn repr(x: f64) -> String {
    if x.is_nan() { return "nan".into(); }
    if x.is_infinite() { return if x < 0.0 { "-inf" } else { "inf" }.into(); }
    // The shortest length, then that many correctly rounded digits: where two
    // candidates of that length are equally near, Python keeps the even one.
    let shortest = format!("{:e}", x.abs());
    let length = shortest.split_once('e').expect("exponent form").0.replace('.', "").len();
    let text = format!("{:.*e}", length - 1, x.abs());
    let (mantissa, exponent) = text.split_once('e').expect("exponent form");
    let digits = mantissa.replace('.', "");
    let decpt = exponent.parse::<i32>().expect("exponent") + 1;
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if decpt <= -4 || decpt > 16 {
        let tail = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        return format!("{sign}{}{tail}{}", &digits[..1], exponent_suffix(decpt - 1));
    }
    if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{sign}{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    }
}
