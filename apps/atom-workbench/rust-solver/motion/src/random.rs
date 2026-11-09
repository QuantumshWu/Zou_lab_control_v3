//! Counter-based random streams (SPEC §2.4): Philox4x32-10 (Salmon, Moraes, Dror & Shaw, "Parallel random numbers: as
//! easy as 1, 2, 3", SC11; the Random123 library). A stream is keyed by the run's seed and its purpose and counted by
//! the trajectory and the event: a draw depends on these alone, not on the order trajectories run in or on the thread
//! that runs them.
//!
//! Common random numbers: the j-th jump of a trajectory draws from its own stream (`Stream::jump`), whatever the steps
//! before it took and however many draws earlier jumps used (a rejection sampler draws as many as it needs), so two runs
//! that differ only in a tolerance or in the model draw the same numbers at the same jump. Within a stream the draws
//! are taken in a fixed order.

/// Philox4x32's round multipliers and the Weyl key increments (the golden ratio's and √3 − 1's 32-bit fractions):
/// the generator's definition (Random123 philox.h).
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;
/// Philox4x32-10's rounds, its definition (the 10 of its name).
const ROUNDS: usize = 10;

/// Philox4x32-10 of a counter and a key: four 32-bit words, a bijection of the counter for each key.
pub fn philox(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (mut c, mut k) = (counter, key);
    for round in 0..ROUNDS {
        if round > 0 {
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        let (p0, p1) = (M0 as u64 * c[0] as u64, M1 as u64 * c[2] as u64);
        c = [(p1 >> 32) as u32 ^ c[1] ^ k[0], p1 as u32, (p0 >> 32) as u32 ^ c[3] ^ k[1], p0 as u32];
    }
    c
}

/// What a stream is for: the tag in its key. Each purpose's draws are independent of every other's; within a
/// purpose, streams are told apart by the trajectory index alone (the pilot's trajectories, numbered from
/// `units::MOST_TRAJECTORIES` on, share no stream with any run's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// A trajectory's initial state (ensemble.rs; event 0).
    Initial = 1,
    /// A Latin-hypercube group's permutations (ensemble.rs; counted by the group's first trajectory, event 0).
    Latin = 2,
    /// A trajectory's jumps: event j ≥ 1 the j-th jump's draws, event 0 the first waiting time's threshold.
    Jump = 3,
}

/// A stream of uniform 32-bit words: Philox blocks at the counter (block, event, trajectory) under the key
/// (seed, purpose), four words a block.
#[derive(Clone, Debug)]
pub struct Stream {
    key: [u32; 2],
    counter: [u32; 4],
    block: [u32; 4],
    used: usize,
}

impl Stream {
    /// The stream of a purpose, a trajectory and an event of the run seeded `seed`.
    pub fn new(seed: u64, purpose: Purpose, trajectory: u64, event: u32) -> Stream {
        let key = [(seed as u32) ^ (purpose as u32).wrapping_mul(W0), (seed >> 32) as u32];
        Stream { key, counter: [0, event, trajectory as u32, (trajectory >> 32) as u32], block: [0; 4], used: 4 }
    }
    /// The stream of a trajectory's j-th jump (j ≥ 1; j = 0 the first waiting time's threshold).
    pub fn jump(seed: u64, trajectory: u64, j: u32) -> Stream {
        Stream::new(seed, Purpose::Jump, trajectory, j)
    }
    pub fn next_u32(&mut self) -> u32 {
        if self.used == 4 {
            self.block = philox(self.counter, self.key);
            self.counter[0] = self.counter[0].wrapping_add(1);
            self.used = 0;
        }
        self.used += 1;
        self.block[self.used - 1]
    }
    pub fn next_u64(&mut self) -> u64 {
        let high = self.next_u32() as u64;
        (high << 32) | self.next_u32() as u64
    }
    /// Uniform on [0, 1): the 53 high bits of a word pair over 2⁵³.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform on (0, 1]: a waiting time's threshold, never zero.
    pub fn open_uniform(&mut self) -> f64 {
        1.0 - self.uniform()
    }
    /// Uniform on (0, 1), both ends excluded: the middle of one of 2⁵³ equal cells.
    pub fn interior(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// A standard normal deviate: the quantile of one interior uniform (one draw per deviate, so the order of draws
    /// stays fixed).
    pub fn normal(&mut self) -> f64 {
        normal_quantile(self.interior())
    }
    /// A unit vector uniform on the sphere: z = cos θ uniform on [−1, 1] (Archimedes), the azimuth uniform.
    pub fn direction(&mut self) -> [f64; 3] {
        let z = 2.0 * self.uniform() - 1.0;
        let (s, c) = (std::f64::consts::TAU * self.uniform()).sin_cos();
        let rho = (1.0 - z * z).max(0.0).sqrt();
        [rho * c, rho * s, z]
    }
}

/// The standard normal's quantile Φ⁻¹(p) for p in (0, 1), to double precision: Wichura's algorithm AS 241 (PPND16,
/// Applied Statistics 37 (1988) 477), rational approximations in the centre |p − ½| ≤ 0.425 and in two tail ranges;
/// its coefficients are the published algorithm's (relative error about 1e-16).
pub fn normal_quantile(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((2.509_080_928_730_122_7e3 * r + 3.343_057_558_358_812_8e4) * r + 6.726_577_092_700_870_1e4) * r
                + 4.592_195_393_154_987_1e4)
                * r
                + 1.373_169_376_550_946_1e4)
                * r
                + 1.971_590_950_306_551_4e3)
                * r
                + 1.331_416_678_917_843_8e2)
                * r
                + 3.387_132_872_796_366_6)
            / (((((((5.226_495_278_852_854_6e3 * r + 2.872_908_573_572_194_3e4) * r + 3.930_789_580_009_271_1e4) * r
                + 2.121_379_430_158_659_6e4)
                * r
                + 5.394_196_021_424_751_1e3)
                * r
                + 6.871_870_074_920_579_1e2)
                * r
                + 4.231_333_070_160_091_1e1)
                * r
                + 1.0);
    }
    let r = (-(if q < 0.0 { p } else { 1.0 - p }).ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        (((((((7.745_450_142_783_414_1e-4 * r + 2.272_384_498_926_918_5e-2) * r + 2.417_807_251_774_506_1e-1) * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_6)
            * r
            + 5.769_497_221_460_691_4)
            * r
            + 4.630_337_846_156_545_3)
            * r
            + 1.423_437_110_749_683_6)
            / (((((((1.050_750_071_644_416_8e-9 * r + 5.475_938_084_995_345e-4) * r + 1.519_866_656_361_645_7e-2) * r
                + 1.481_039_764_274_800_7e-1)
                * r
                + 6.897_673_349_851_000_5e-1)
                * r
                + 1.676_384_830_183_803_8)
                * r
                + 2.053_191_626_637_758_8)
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((2.010_334_399_292_288_1e-7 * r + 2.711_555_568_743_487_6e-5) * r + 1.242_660_947_388_078_4e-3) * r
            + 2.653_218_952_657_612_3e-2)
            * r
            + 2.965_605_718_285_048_9e-1)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114_4)
            * r
            + 6.657_904_643_501_103_8)
            / (((((((2.044_263_103_389_939_8e-15 * r + 1.421_511_758_316_445_9e-7) * r + 1.846_318_317_510_054_7e-5)
                * r
                + 7.868_691_311_456_132_6e-4)
                * r
                + 1.487_536_129_085_061_5e-2)
                * r
                + 1.369_298_809_227_358_1e-1)
                * r
                + 5.998_322_065_558_879_4e-1)
                * r
                + 1.0)
    };
    if q < 0.0 {
        -x
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random123's known-answer vectors for Philox4x32-10 (kat_vectors): the zero counter and key, the all-ones ones,
    /// and π's digits.
    #[test]
    fn philox_gives_random123_s_known_answers() {
        assert_eq!(philox([0; 4], [0; 2]), [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
        assert_eq!(philox([u32::MAX; 4], [u32::MAX; 2]), [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]);
        assert_eq!(
            philox([0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344], [0xa409_3822, 0x299f_31d0]),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    /// A jump's stream is its own: the same (seed, trajectory, j) draws the same numbers whatever was drawn before,
    /// another j, trajectory, purpose or seed draws others.
    #[test]
    fn a_jump_s_stream_depends_on_its_keys_alone() {
        let draws = |mut s: Stream| -> Vec<u32> { (0..9).map(|_| s.next_u32()).collect() };
        let mut used = Stream::jump(7, 3, 2);
        for _ in 0..13 {
            used.next_u32();
        }
        assert_eq!(draws(Stream::jump(7, 3, 2)), draws(Stream::jump(7, 3, 2)));
        let base = draws(Stream::jump(7, 3, 2));
        for other in [Stream::jump(7, 3, 1), Stream::jump(7, 4, 2), Stream::jump(8, 3, 2), Stream::new(7, Purpose::Initial, 3, 2)] {
            assert_ne!(draws(other), base);
        }
    }

    /// Φ⁻¹ at known quantiles (each region of AS 241: the centre, the near tail r ≤ 5, the far tail): Φ(1) =
    /// 0.8413447460685429, Φ(2) = 0.9772498680518208, Φ⁻¹(0.975) = 1.959963984540054, Φ⁻¹(10⁻¹⁰) = −6.361340902404056,
    /// Φ⁻¹(10⁻³⁰⁰) = −37.0471 (its asymptotic −√(2 ln(1/p) − ln(4π ln(1/p))) within 0.01); the antisymmetry Φ⁻¹(½) = 0;
    /// and the deviates' first two moments over 2¹⁶ draws (mean 0 ± 4/256, variance 1 ± 4√2/256: z = 4 standard errors).
    #[test]
    fn the_normal_quantile_and_the_deviates_are_the_normal_s() {
        // Within the algorithm's relative precision of x plus the input's rounding, u·p moved through the slope
        // dΦ⁻¹/dp = 1/φ(x): γ₄ (|x| + p/φ(x)).
        let close = |p: f64, x: f64| {
            let density = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let within = crate::units::rounding(4) * (x.abs() + p / density);
            assert!((normal_quantile(p) - x).abs() <= within, "{p}: {} against {x} within {within:e}", normal_quantile(p));
        };
        close(0.841_344_746_068_542_9, 1.0);
        close(1.0 - 0.841_344_746_068_542_9, -1.0);
        close(0.977_249_868_051_820_8, 2.0);
        close(0.975, 1.959_963_984_540_054);
        close(1e-10, -6.361_340_902_404_056);
        let l = (1e300_f64).ln();
        let asymptotic = -(2.0 * l - (4.0 * std::f64::consts::PI * l).ln()).sqrt();
        assert!((normal_quantile(1e-300) - asymptotic).abs() <= 1e-2, "{}", normal_quantile(1e-300));
        assert_eq!(normal_quantile(0.5), 0.0);
        let mut s = Stream::new(1, Purpose::Initial, 0, 0);
        let n = 1 << 16;
        let x: Vec<f64> = (0..n).map(|_| s.normal()).collect();
        let mean = x.iter().sum::<f64>() / n as f64;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
        assert!(mean.abs() <= 4.0 / 256.0 && (var - 1.0).abs() <= 4.0 * 2f64.sqrt() / 256.0, "{mean} {var}");
    }
}
