//! `Normalize.__call__` and `Colormap.__call__` with numpy's dtype rules:
//! integer and bool data normalize in float32 (two bytes or less) or float64,
//! float data in its own precision; an in-place op against a float64 scalar
//! computes in float64 and rounds into the array's dtype.
use crate::ir::{Colormap, Norm};
use crate::wire::{DType, NdArray};
use crate::{unsupported, Result};

/// The working precision of normalized data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision { F32, F64 }

/// `Normalize.process_value`'s dtype for an array of `dtype`.
pub fn process_dtype(dtype: DType) -> Result<Precision> {
    Ok(match dtype {
        DType::Bool | DType::U8 | DType::I8 | DType::U16 | DType::I16 => Precision::F32,
        DType::U32 | DType::I32 | DType::U64 | DType::I64 => Precision::F64,
        DType::F32 => Precision::F32,
        DType::F64 => Precision::F64,
    })
}

pub struct ScalarInput<'a> { pub data: &'a NdArray, pub mask: Option<&'a NdArray> }

/// Normalized values (in their precision, widened exactly to f64) and the
/// colormap's bad mask: the input mask when any element is masked (empty
/// otherwise, meaning "isnan").
pub struct Normalized { pub values: Vec<f64>, pub precision: Precision, pub bad: Vec<bool> }

pub fn round_to(p: Precision, v: f64) -> f64 { if p == Precision::F32 { f64::from(v as f32) } else { v } }

/// `norm(value)` for values already in their working dtype, with `mask`
/// the masked array's mask (or empty).
pub fn normalize_values(values: &[f64], precision: Precision, mask: &[bool], norm: &Norm) -> Result<Normalized> {
    if norm.class != "Normalize" { return unsupported(format!("norm {}", norm.class)); }
    if norm.clip { return unsupported("Normalize(clip=True)"); }
    let (Some(vmin), Some(vmax)) = (norm.vmin, norm.vmax) else {
        return unsupported("autoscaling an unset norm at draw time");
    };
    let mut out: Vec<f64> = values.to_vec();
    if vmin == vmax {
        out.iter_mut().for_each(|v| *v = 0.0);
    } else if vmin > vmax {
        return crate::invalid("minvalue must be less than or equal to maxvalue");
    } else {
        let span = vmax - vmin;
        for v in &mut out {
            *v = round_to(precision, *v - vmin);
            *v = round_to(precision, *v / span);
        }
    }
    let masked_any = mask.iter().any(|&b| b);
    Ok(Normalized { values: out, precision, bad: if masked_any { mask.to_vec() } else { Vec::new() } })
}

/// A masked array's mask (`np.ma.getmaskarray`): booleans of the data's shape.
pub fn mask_bits(mask: Option<&NdArray>, data: &NdArray) -> Result<Option<Vec<bool>>> {
    let Some(mask) = mask else { return Ok(None) };
    if mask.dtype != DType::Bool || mask.shape != data.shape {
        return crate::invalid(format!("mask {:?} {:?} for data {:?}", mask.dtype, mask.shape, data.shape));
    }
    Ok(Some(mask.to_bool()))
}

/// `norm(A)` for a (masked) data array.
pub fn normalize(input: &ScalarInput<'_>, norm: &Norm) -> Result<Normalized> {
    let p = process_dtype(input.data.dtype)?;
    let values: Vec<f64> = input.data.to_f64().into_iter().map(|v| round_to(p, v)).collect();
    let mask: Vec<bool> = mask_bits(input.mask, input.data)?.unwrap_or_default();
    normalize_values(&values, p, &mask, norm)
}

/// The LUT row of every normalized value (`Colormap._get_rgba_and_mask`).
pub fn lut_indices(n: &Normalized, cmap: &Colormap) -> Result<Vec<usize>> {
    if cmap.lut.len() != cmap.n + 3 { return crate::invalid("colormap lut size"); }
    let nn = cmap.n as f64;
    let masked = !n.bad.is_empty();
    let last = cmap.lut.len() as i64 - 1;
    Ok(n.values.iter().enumerate().map(|(k, &v)| {
        let mut xa = round_to(n.precision, v * nn);
        if xa == nn { xa = nn - 1.0; }
        let under = xa < 0.0;
        let over = xa >= nn;
        let bad = if masked { n.bad[k] } else { xa.is_nan() };
        let mut index: i64 = if xa.is_nan() { i64::MIN } else { xa as i64 };
        if under { index = cmap.i_under as i64; }
        if over { index = cmap.i_over as i64; }
        if bad { index = cmap.i_bad as i64; }
        index.clamp(0, last) as usize
    }).collect())
}

/// `Colormap.__call__(X, alpha)` for normalized data; RGBA rows (floats).
pub fn colormap(n: &Normalized, cmap: &Colormap, alpha: Option<f64>) -> Result<Vec<[f64; 4]>> {
    let indices = lut_indices(n, cmap)?;
    let masked = !n.bad.is_empty();
    let mut out: Vec<[f64; 4]> = indices.iter().map(|&i| {
        let mut rgba = cmap.lut[i];
        if let Some(a) = alpha { rgba[3] = a.clamp(0.0, 1.0); }
        rgba
    }).collect();
    if alpha.is_some() && cmap.lut[cmap.lut.len() - 1].iter().all(|&c| c == 0.0) {
        for (k, rgba) in out.iter_mut().enumerate() {
            let bad = if masked { n.bad[k] } else { round_to(n.precision, n.values[k] * cmap.n as f64).is_nan() };
            if bad { *rgba = [0.0, 0.0, 0.0, 0.0]; }
        }
    }
    Ok(out)
}

/// One value's `norm` and LUT row, for callers mapping values one at a time
/// (the arithmetic of [`normalize_values`] and [`lut_indices`]).
pub struct PixelMap { precision: Precision, vmin: f64, span: f64, flat: bool, n: f64, under: i64, over: i64, bad: i64, last: i64 }

impl PixelMap {
    pub fn new(norm: &Norm, cmap: &Colormap, precision: Precision) -> Result<PixelMap> {
        if norm.class != "Normalize" { return unsupported(format!("norm {}", norm.class)); }
        if norm.clip { return unsupported("Normalize(clip=True)"); }
        let (Some(vmin), Some(vmax)) = (norm.vmin, norm.vmax) else {
            return unsupported("autoscaling an unset norm at draw time");
        };
        if vmin > vmax { return crate::invalid("minvalue must be less than or equal to maxvalue"); }
        if cmap.lut.len() != cmap.n + 3 { return crate::invalid("colormap lut size"); }
        Ok(PixelMap { precision, vmin, span: vmax - vmin, flat: vmin == vmax, n: cmap.n as f64,
                      under: cmap.i_under as i64, over: cmap.i_over as i64, bad: cmap.i_bad as i64,
                      last: cmap.lut.len() as i64 - 1 })
    }

    /// `norm(v)` for a value already in the working precision.
    #[inline]
    pub fn normalize(&self, v: f64) -> f64 {
        if self.flat { return 0.0; }
        round_to(self.precision, round_to(self.precision, v - self.vmin) / self.span)
    }

    /// The LUT row of a normalized value; `bad` is its mask bit when the
    /// array is masked anywhere (`None`: "isnan").
    #[inline]
    pub fn index(&self, normalized: f64, bad: Option<bool>) -> usize {
        let mut xa = round_to(self.precision, normalized * self.n);
        if xa == self.n { xa = self.n - 1.0; }
        let is_bad = bad.unwrap_or(xa.is_nan());
        let mut index: i64 = if xa.is_nan() { i64::MIN } else { xa as i64 };
        if xa < 0.0 { index = self.under; }
        if xa >= self.n { index = self.over; }
        if is_bad { index = self.bad; }
        index.clamp(0, self.last) as usize
    }
}

/// `ScalarMappable.to_rgba(A, alpha)` (norm then colormap).
pub fn map_scalars(input: &ScalarInput<'_>, cmap: &Colormap, norm: &Norm, alpha: Option<f64>) -> Result<Vec<[f64; 4]>> {
    let n = normalize(input, norm)?;
    colormap(&n, cmap, alpha)
}
