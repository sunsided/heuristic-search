//! A tiny seeded pseudo-random number generator (`SplitMix64`) for reproducible example
//! inputs, so the examples need no `rand` dependency.

/// `SplitMix64` (Steele, Lea and Flood 2014): a 64-bit state advanced by a constant and
/// mixed on output. Good enough to scatter rooms and boulders; not for cryptography.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator starting from `seed`; equal seeds give equal sequences.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A number in `0..n`, with a negligible modulo bias for the small `n` used here.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let n = u64::try_from(n).expect("usize fits in u64");

        usize::try_from(z % n).expect("the remainder is below n")
    }
}
