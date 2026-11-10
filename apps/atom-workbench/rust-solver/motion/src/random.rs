//! Counter-based random streams (SPEC §2.4): Philox4x32-10 (Salmon, Moraes, Dror & Shaw, "Parallel random numbers: as
//! easy as 1, 2, 3", SC11; the Random123 library). A stream is keyed by the run's seed and its purpose and counted by
//! the trajectory and the event: a draw depends on these alone, not on the order trajectories run in or on the thread
//! that runs them.
//!
//! Common random numbers: the j-th jump of a trajectory draws from its own stream (`Stream::jump`), whatever the steps
//! before it took and however many draws earlier jumps used (a rejection sampler draws as many as it needs), so two runs
//! that differ only in the step or in the model draw the same numbers at the same jump. Within a stream the draws
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
    /// A trajectory's Brownian path (SPEC §1.6: the absorbed light's diffusion): event n the n-th step's deviates.
    Diffusion = 4,
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
    /// The stream of a trajectory's n-th step's Brownian deviates (step.rs `Brownian`).
    pub fn diffusion(seed: u64, trajectory: u64, n: u32) -> Stream {
        Stream::new(seed, Purpose::Diffusion, trajectory, n)
    }
    /// Three independent standard normal deviates: Box and Muller's (1958) two from a pair of uniforms, r (cos, sin) 2πu
    /// with r² = −2 ln v, v on (0, 1] (`interior`), twice; the fourth is not kept.
    pub fn normals(&mut self) -> [f64; 3] {
        let mut pair = || {
            let r = (-2.0 * self.interior().ln()).sqrt();
            let (s, c) = (std::f64::consts::TAU * self.uniform()).sin_cos();
            [r * c, r * s]
        };
        let (first, second) = (pair(), pair());
        [first[0], first[1], second[0]]
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
    /// Uniform on (0, 1]: the middle (k + ½)/2⁵³ of one of 2⁵³ equal cells, as the format holds it -- never 0; from
    /// k = 2⁵² on the middle is no number of the format and rounds to the even one of the cell's two edges, the last
    /// cell's to 1 itself (one draw in 2⁵³).
    pub fn interior(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// A standard normal deviate, for the tests alone (the engine's are `normals`): Φ⁻¹ of one interior uniform u -- one draw
    /// per deviate, so the order of draws stays fixed --, the normal's two-sided quantile at the tail 2 min(u, 1 − u)
    /// (estimate.rs `quantile`) on u's side of ½.
    #[cfg(test)]
    pub fn normal(&mut self) -> f64 {
        let u = self.interior();
        crate::estimate::quantile(2.0 * u.min(1.0 - u), None).copysign(u - 0.5)
    }
    /// A unit vector uniform on the sphere: z = cos θ uniform on [−1, 1] (Archimedes), the azimuth uniform.
    pub fn direction(&mut self) -> [f64; 3] {
        let z = 2.0 * self.uniform() - 1.0;
        let (s, c) = (std::f64::consts::TAU * self.uniform()).sin_cos();
        let rho = (1.0 - z * z).max(0.0).sqrt();
        [rho * c, rho * s, z]
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

    /// The deviates are the normal's: over 2¹⁶ draws their mean is 0 and their variance 1 within the standard errors of
    /// the two, 1/√n and √(2/(n − 1)), at the confidence z over the two looks (estimate.rs `Spread::interval_over`).
    #[test]
    fn the_deviates_are_the_normal_s() {
        let mut s = Stream::new(1, Purpose::Initial, 0, 0);
        let n = 1 << 16;
        let x: Vec<f64> = (0..n).map(|_| s.normal()).collect();
        let mean = x.iter().sum::<f64>() / n as f64;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
        let within = |error: f64| crate::estimate::Spread { error, groups: f64::INFINITY }.interval_over(2.0);
        assert!(mean.abs() <= within((1.0 / n as f64).sqrt()) && (var - 1.0).abs() <= within((2.0 / (n - 1) as f64).sqrt()), "{mean} {var}");
    }
}
