//! Constants and derived values (SPEC §0, §2.2): the only numbers of the crate that no request gives. Each is one of the
//! allowed kinds, its source or its derivation beside it: the user's targets, the scheme's order p, the statistics'
//! confidence z, physical constants, the floating-point format, values derived from these, the validated regime's
//! boundary, and the contract's definitions and caps (contract §1-§2, mirrored in the app).
//!
//! Units of the whole crate (SPEC §0): length µm, time µs, energy and frequency rad/µs (an energy E as the rate E/ħ),
//! momentum as the wave number κ = p/ħ (1/µm), temperature µK, field V/m, wave number rad/µm. A velocity in µm/µs is
//! the same number in m/s.

use std::f64::consts::TAU;

// ---------------------------------------------------------------------------------------------------------------------
// The user's targets, the scheme, the statistics.

/// The user's population and loss target, absolute (SPEC §0): every displayed population row and the loss
/// (the result's `requestedTolerance`, contract §2.2).
pub const POPULATION_TARGET: f64 = 0.05;
/// The user's temperature target, relative (SPEC §0).
pub const TEMPERATURE_TARGET: f64 = 0.10;
/// The scheme's order p (SPEC §2.1): the symmetric (Strang) splitting A(τ/2) B(τ) A(τ/2), whose local error is
/// O(τ^{p+1}) and global error O(τ^p).
pub const ORDER: i32 = 2;
/// The statistics' confidence z (SPEC §0, §5.1): an error is read at z standard errors (the normal's two-sided 95.4 %).
pub const CONFIDENCE: f64 = 2.0;

/// The step error's share s of a target (SPEC §2.2): s = √(1/(1 + 2p)), √(1/5) ≈ 0.447 at p = 2.
///
/// Derivation: a run's error is E² = b² + S², the step bias b and the statistics S. The work goes as the trajectories
/// times the steps, W ∝ S⁻² τ⁻¹, and b ∝ τ^p, so W ∝ S⁻² b^{−1/p}. At a fixed E put b = E sin θ, S = E cos θ:
/// d ln W/dθ = 2 tan θ − (1/p) cot θ vanishes at tan²θ = 1/(2p), i.e. sin²θ = 1/(1 + 2p). The least work for a
/// given accuracy gives the step b/E = √(1/(1 + 2p)) of the target.
pub const STEP_SHARE: f64 = square_root(1.0 / (1.0 + 2.0 * ORDER as f64));

/// √a for a ≥ 0 in a constant (f64::sqrt is not const): Heron's iteration x ← (x + a/x)/2 from max(a, 1) ≥ √a, which
/// decreases monotonically to √a and is stopped by the floating-point format, where a step no longer decreases x.
const fn square_root(a: f64) -> f64 {
    let mut x = if a > 1.0 { a } else { 1.0 };
    loop {
        let next = 0.5 * (x + a / x);
        if !(next < x) {
            return x;
        }
        x = next;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The contract's definitions and caps.

/// The validated regime's boundary (SPEC §0, §3): k_BT ≥ CLASSICAL·ħω, axis by axis, mirrored in the app (the result's
/// `classical`: every axis's; the UI's grey band, src/model-notes.ts `SEMICLASSICAL_REGIME`). Reported per output and
/// axis (solve.rs `classical`); it decides nothing in the solve. An empirical boundary, not a derived one.
pub const CLASSICAL: f64 = 4.0;
/// The trap's region (SPEC §1.3, §1.7): an atom is outside it where every beam of the trap light (the tweezer's
/// frequency) is at the atom below this share of its value at the focus -- the contract's region, the UI's text word for
/// word (src/model-notes.ts: "与光镊同频的各束光在原子处的强度都低于各自焦点处的 1/20"). A lost atom is one first outside it
/// with its loss energy above the far value (model.rs `Model::lost`), as the same sentence of the UI goes on.
pub const REGION_INTENSITY: f64 = 1.0 / 20.0;
/// The most trajectories a solve runs (SPEC §3; contract §2.3, whose reason text names 16384): reached, the solve
/// returns as usual, partial. A cap of the contract, not derived; beside the work it sets the least statistics a
/// solve books, z²/(N + z²) (estimate.rs `unseen`), and through it the reach's share and the tables' shares
/// (model/tabulate.rs `shares`).
pub const MOST_TRAJECTORIES: u64 = 1 << 14;
/// The most steps of the run's constant step a trajectory's run may hold (SPEC §2.2; contract §2.3): the step's ladder
/// goes no further down than the run's length over it; where its error still lies clearly beyond what it may take
/// there, it is booked as measured and flagged. (A trajectory's jumps and the halves of refused steps come on top;
/// they are not capped.) A cap of the contract, not derived: it bounds the work alone.
pub const MOST_STEPS: u64 = 1 << 24;
/// The longest trajectory a request asks for (µs): 10 s (contract §1.3, the population core's `LONGEST_TIME_US`; the
/// app's translated message names "10 s").
pub const LONGEST_TIME_US: f64 = 1e7;
/// The most output times a request asks for (contract §1.3, the population core's `MOST_SAMPLES`).
pub const MOST_SAMPLES: usize = 2000;

// ---------------------------------------------------------------------------------------------------------------------
// Physical constants (SI 2019: h, k_B and c are exact) and the crate's unit conversions.

/// Planck's constant h (J s).
pub const PLANCK: f64 = 6.626_070_15e-34;
/// The reduced Planck constant ħ = h/2π (J s).
pub const HBAR: f64 = PLANCK / TAU;
/// Boltzmann's constant k_B (J/K).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// The speed of light c (m/s, the same number in µm/µs).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// k_B/ħ in rad/µs per µK: a temperature T as the rate k_BT/ħ = `KT_PER_UK`·T
/// ((J/K)/(J s) = 1/(s K); × 10⁻⁶ K/µK × 10⁻⁶ s/µs).
pub const KT_PER_UK: f64 = BOLTZMANN / HBAR * 1e-12;
/// rad/µs per MHz: a frequency ν (MHz) as the angular frequency 2πν (rad/µs).
pub const RAD_PER_US_PER_MHZ: f64 = TAU;

/// ħ/m (µm²/µs) of a mass m (kg): (J s/kg = m²/s) × 10¹² µm²/m² × 10⁻⁶ s/µs. The kinetic energy (rad/µs) of the
/// wave number κ is (ħ/m) κ²/2, the velocity (µm/µs) (ħ/m) κ.
pub fn hbar_over_mass(mass_kg: f64) -> f64 {
    HBAR / mass_kg * 1e6
}
/// The wave number k = 2πν/c (rad/µm) of light of frequency ν (MHz = 1/µs): c in µm/µs.
pub fn wave_number(frequency_mhz: f64) -> f64 {
    TAU * frequency_mhz / SPEED_OF_LIGHT
}

// ---------------------------------------------------------------------------------------------------------------------
// The floating-point format.

/// The unit round-off u of f64, 2⁻⁵³: a computed operation is the exact one times (1 + δ), |δ| ≤ u.
pub const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Higham's γ_n = n u/(1 − n u) (Accuracy and Stability of Numerical Algorithms, §3.1): the bound of the relative
/// rounding error of n successive operations -- a sum of n + 1 terms errs by at most γ_n Σ|term|, an inner product of
/// length n by γ_n Σ|x_i y_i|. A quantity within this bound of its operands' scale is that computation's rounding.
pub fn rounding(n: usize) -> f64 {
    let nu = n as f64 * UNIT_ROUNDOFF;
    nu / (1.0 - nu)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived values: s = √(1/(1 + 2p)) = √(1/5) to round-off; k_B/ħ = 0.130920… rad/µs per µK; ħ/m of Rb87 (1.4432e-25 kg)
    /// 7.3072e-4 µm²/µs; the D2 line's wave number 2π·384230.484 GHz/c = 8.0528 rad/µm; γ_n against n u.
    #[test]
    fn the_derived_values_are_their_formulas() {
        assert!((STEP_SHARE - 0.2_f64.sqrt()).abs() <= f64::EPSILON * STEP_SHARE, "{STEP_SHARE}");
        assert!((KT_PER_UK - 0.130_920_3).abs() <= 1e-7, "{KT_PER_UK}");
        assert!((hbar_over_mass(1.4432e-25) - 7.307_2e-4).abs() <= 1e-8);
        assert!((wave_number(384_230_484.468_5) - 8.052_8).abs() <= 1e-4);
        assert!(rounding(0) == 0.0 && (rounding(10) / (10.0 * UNIT_ROUNDOFF) - 1.0).abs() <= 1e-14);
    }
}
