//! A tiny seeded pseudo-random number generator (`SplitMix64`) for reproducible example
//! inputs, so the examples need no `rand` dependency.
//!
//! Examples include this file as `pub mod prng`, so the methods one example leaves unused
//! do not count as dead code there.

/// `SplitMix64` (Steele, Lea and Flood 2014): a 64-bit state advanced by a constant and
/// mixed on output. Good enough to lay out cities, maps and floors; not for cryptography.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator starting from `seed`; equal seeds give equal sequences.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n`, with a negligible modulo bias for the small `n` used here.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn below(&mut self, n: usize) -> usize {
        let n = u64::try_from(n).expect("usize fits in u64");

        usize::try_from(self.next_u64() % n).expect("the remainder is below n")
    }

    /// A number in `low..high`.
    ///
    /// # Panics
    ///
    /// Panics unless `low < high`.
    pub fn range(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low)
    }

    /// A uniform float in `[-1, 1)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits fit an f64 exactly"
    )]
    pub fn signed(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }
}
