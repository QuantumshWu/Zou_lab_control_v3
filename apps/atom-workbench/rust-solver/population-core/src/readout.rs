//! Population readout of trial columns.
//!
//! The population of original state `s` at time `t` is a fixed bilinear
//! function of the envelope field (see `PopulationObservation`):
//! `p_s(t) = Re sum_mu e^{i omega_mu t} sum_j W[mu, s, j] a_j(t)`,
//! where `a(t)` are the reduced coefficients. `W` is stored once per column
//! and output-mode; phases are applied only when evaluating at output times,
//! so memory does not grow with the number of samples.

use super::*;
use super::coordinates::PopulationObservation;
use crate::preparation::FrequencyParts;

/// Output times evaluated per GEMM batch.
const TIME_CHUNK: usize = 256;

pub(crate) struct Readout {
    observer: Arc<PopulationObservation>,
    domain_modes: Vec<Charge>,
    modes: Vec<Charge>,
    frequencies: Vec<FrequencyParts>,
    n: usize,
    /// Column-major `[(mode * n + state), column]` real and imaginary parts.
    re: Vec<f64>,
    im: Vec<f64>,
    columns: usize,
}

/// Populations at output times plus the largest discarded imaginary part,
/// which must vanish for Hermitian-symmetric trial columns.
pub(crate) struct Populations {
    pub values: Vec<f64>,
    pub maximum_imaginary: f64,
}

impl Readout {
    pub fn new(
        observer: &Arc<PopulationObservation>, prepared: &OwnedPrepared, domain_modes: &[Charge],
    ) -> Result<Self, String> {
        let modes = observer.output_modes(domain_modes)?;
        let frequencies = modes.iter().map(|q| observer.frequency(q)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            observer: Arc::clone(observer),
            domain_modes: domain_modes.to_vec(),
            modes,
            frequencies,
            n: prepared.primitives().hilbert.source.n,
            re: vec![],
            im: vec![],
            columns: 0,
        })
    }

    fn rows(&self) -> usize {
        self.modes.len() * self.n
    }
    pub fn bytes(&self) -> usize {
        (self.re.capacity() + self.im.capacity()) * 8
    }

    /// Appends the readout coefficients of `k` signed columns `[row, k]`.
    pub fn append(&mut self, signed: &[C], k: usize) -> Result<(), String> {
        let basis = self.observer.project(&self.domain_modes, signed, k)?;
        if basis.modes() != self.modes.as_slice() {
            return Err("Readout output modes changed within one domain".into());
        }
        let coefficients = basis.coefficients();
        let rows = self.rows();
        for j in 0..k {
            for r in 0..rows {
                let z = coefficients[r * k + j];
                self.re.push(z.re);
                self.im.push(z.im);
            }
        }
        self.columns += k;
        Ok(())
    }

    /// The same readout for a larger domain. Old columns vanish on the added
    /// modes, so their coefficients on newly reachable output modes are zero.
    pub fn transferred(&self, domain_modes: &[Charge]) -> Result<Self, String> {
        let modes = self.observer.output_modes(domain_modes)?;
        let index: BTreeMap<&Charge, usize> = modes.iter().enumerate().map(|(i, q)| (q, i)).collect();
        let n = self.n;
        let old_rows = self.rows();
        let new_rows = modes.len() * n;
        let mut re = vec![0.; new_rows * self.columns];
        let mut im = vec![0.; new_rows * self.columns];
        for (old, q) in self.modes.iter().enumerate() {
            let new = *index.get(q).ok_or("Readout mode lost in a domain transfer")?;
            for j in 0..self.columns {
                let (from, to) = (j * old_rows + old * n, j * new_rows + new * n);
                re[to..to + n].copy_from_slice(&self.re[from..from + n]);
                im[to..to + n].copy_from_slice(&self.im[from..from + n]);
            }
        }
        let frequencies = modes.iter().map(|q| self.observer.frequency(q)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            observer: Arc::clone(&self.observer),
            domain_modes: domain_modes.to_vec(),
            modes,
            frequencies,
            n,
            re,
            im,
            columns: self.columns,
        })
    }

    /// Populations `[time, state]` for reduced coefficients `a` given
    /// column-major `[m, times]`.
    pub fn evaluate(&self, lapack: &Lapack, coefficients: &[f64], times: &[f64]) -> Result<Populations, String> {
        let m = self.columns;
        let n = self.n;
        if coefficients.len() != m * times.len() {
            return Err("Reduced coefficients do not match the readout".into());
        }
        let rows = self.rows();
        let mut values = vec![0.; times.len() * n];
        let mut maximum_imaginary = 0_f64;
        let mut real_part = vec![0.; rows * TIME_CHUNK];
        let mut imaginary_part = vec![0.; rows * TIME_CHUNK];
        for start in (0..times.len()).step_by(TIME_CHUNK) {
            let count = TIME_CHUNK.min(times.len() - start);
            let block = &coefficients[start * m..(start + count) * m];
            lapack.gemm(false, false, rows, count, m, 1., &self.re, rows, block, m, 0., &mut real_part, rows)?;
            lapack.gemm(false, false, rows, count, m, 1., &self.im, rows, block, m, 0., &mut imaginary_part, rows)?;
            for t in 0..count {
                let time = times[start + t];
                let output = &mut values[(start + t) * n..(start + t + 1) * n];
                let mut imaginary = vec![0.; n];
                for (mode, frequency) in self.frequencies.iter().enumerate() {
                    let phase = frequency.phase(time)?;
                    let at = t * rows + mode * n;
                    for s in 0..n {
                        let (a, b) = (real_part[at + s], imaginary_part[at + s]);
                        output[s] += phase.re * a - phase.im * b;
                        imaginary[s] += phase.re * b + phase.im * a;
                    }
                }
                for value in imaginary {
                    maximum_imaginary = maximum_imaginary.max(value.abs());
                }
            }
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite population readout".into());
        }
        Ok(Populations { values, maximum_imaginary })
    }
}
