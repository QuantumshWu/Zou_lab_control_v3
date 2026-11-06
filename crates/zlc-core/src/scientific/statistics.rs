//! Uncertainty and mergeable moments shared by live derivation and plotting.

/// Standard error from population first/second moments. A single sample can
/// stand on its stated uncertainty; an unobserved bucket has no uncertainty.
pub fn sem_from_moments(mean: &[f64], mean_of_squares: &[f64], counts: &[i64], mean_sigma_square: Option<&[f64]>) -> Vec<f64> {
    (0..mean.len()).map(|i| {
        let n = counts[i] as f64;
        let squared_mean = mean[i] * mean[i];
        let mut spread = mean_of_squares[i] - squared_mean;
        let roundoff = ((mean_of_squares[i].abs() + squared_mean) * f64::EPSILON) * (n.max(1.0) + 3.0);
        if spread <= roundoff { spread = 0.0; }
        if !spread.is_nan() && spread < 0.0 { spread = 0.0; }
        let mut variance = if n > 1.0 { spread * n / (n - 1.0) } else { f64::NAN };
        if let Some(stated) = mean_sigma_square { if variance.is_nan() { variance = stated[i]; } }
        if n < 1.0 { f64::NAN } else { (variance / n).sqrt() }
    }).collect()
}

/// Standard error from a centred sum of squares, retaining the operation
/// order used for live grouped observations.
pub fn sem_from_centred(spread: &[f64], counts: &[u64], stated_square: Option<&[f64]>) -> Vec<f64> {
    (0..counts.len()).map(|i| {
        let n = counts[i] as f64;
        let mut second = if n > 0.0 { spread[i] / n } else { 0.0 };
        if second <= second.abs() * f64::EPSILON * (n.max(1.0) + 3.0) { second = 0.0; }
        let mut variance = if n > 1.0 { second * n / (n - 1.0) } else { f64::NAN };
        if let Some(stated) = stated_square { if variance.is_nan() { variance = if n > 0.0 { stated[i] / n } else { f64::NAN }; } }
        if n >= 1.0 { (variance / n).sqrt() } else { f64::NAN }
    }).collect()
}

pub fn sem_dof(count: u64) -> f64 { if count > 1 { (count - 1) as f64 } else { f64::INFINITY } }

#[derive(Clone, Copy, Debug, Default)]
pub struct Moment { pub count: u64, pub mean: f64, pub m2: f64 }

impl Moment {
    /// Chan/Welford merging; zero-sized increments do not change an estimate.
    pub fn merge(&mut self, other: Self) -> Result<(), String> {
        let total = self.count.checked_add(other.count).ok_or("sample count overflows u64")?;
        let delta = other.mean - self.mean;
        if total > 0 {
            self.m2 += other.m2 + delta * delta * ((self.count as f64 * other.count as f64) / total as f64);
            self.mean += delta * other.count as f64 / total as f64;
        }
        self.count = total;
        Ok(())
    }
}
