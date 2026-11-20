//! Priority arithmetic on costs ([`CostExt`]) and heuristic weights ([`Weight`]).
//!
//! The [`Cost`] trait itself comes from `search-core`.

use search_core::{Cost, Decimal};

/// Priority arithmetic that [`Cost`] lacks: subtraction and conversion to `f64`.
///
/// A few searches order nodes by priorities that subtract or divide costs, such as the
/// `(G - g) / h` order of [`ana_star`](crate::ana_star), the `2g + h - h_opposite` key of
/// [`dibbs`](crate::dibbs) and [`bae_star`](crate::bae_star), and the potential of
/// [`dps`](crate::dps). Only those searches (and [`bidirectional_dijkstra`], which shares the
/// bidirectional loop) require this trait; every other search needs [`Cost`] alone.
///
/// Implemented for every [`Cost`] type of `search-core`: `u8`, `u16`, `u32`, `u64`, `u128`,
/// `usize`, [`Decimal`] and (feature `float`) `NonNegF64`. A
/// user-defined cost type implements it next to [`Cost`].
///
/// [`bidirectional_dijkstra`]: crate::bidirectional_dijkstra
///
/// # Examples
///
/// ```
/// use heuristic_search::CostExt;
///
/// assert_eq!(3_u32.saturating_sub(5), 0);
/// assert_eq!(7_u64.to_f64(), 7.0);
/// ```
pub trait CostExt: Cost {
    /// Subtracts `other`, saturating at [`Cost::ZERO`].
    ///
    /// Used for priorities that subtract one estimate from another, such as the `G - g`
    /// slack of [`ana_star`](crate::ana_star).
    #[must_use]
    fn saturating_sub(self, other: Self) -> Self;

    /// The cost as a float, possibly rounded.
    ///
    /// Used only for priorities that divide costs, such as the `(G - g) / h` order of
    /// [`ana_star`](crate::ana_star). Rounding can change the expansion order among nearly
    /// equal priorities but never a returned cost.
    fn to_f64(self) -> f64;
}

/// Implements [`CostExt`] for unsigned integer types, each with its conversion to `f64`.
macro_rules! integer_cost_ext {
    ($($ty:ty => $to_f64:path),*) => {
        $(
            impl CostExt for $ty {
                fn saturating_sub(self, other: Self) -> Self {
                    <$ty>::saturating_sub(self, other)
                }

                fn to_f64(self) -> f64 {
                    $to_f64(self)
                }
            }
        )*
    };
}

integer_cost_ext!(
    u8 => f64::from,
    u16 => f64::from,
    u32 => f64::from,
    u64 => u64_to_f64,
    u128 => u128_to_f64,
    usize => usize_to_f64
);

/// Converts a `u64` cost to the nearest float, for priorities only.
#[expect(clippy::cast_precision_loss, reason = "only used for priorities")]
fn u64_to_f64(cost: u64) -> f64 {
    cost as f64
}

/// Converts a `u128` cost to the nearest float, for priorities only.
#[expect(clippy::cast_precision_loss, reason = "only used for priorities")]
fn u128_to_f64(cost: u128) -> f64 {
    cost as f64
}

/// Converts a `usize` cost to the nearest float, for priorities only.
#[expect(clippy::cast_precision_loss, reason = "only used for priorities")]
fn usize_to_f64(cost: usize) -> f64 {
    cost as f64
}

impl<const K: u32> CostExt for Decimal<K> {
    fn saturating_sub(self, other: Self) -> Self {
        Self::from_raw(self.raw().saturating_sub(other.raw()))
    }

    /// The raw unit count as a float: the value scaled by `10^K`.
    ///
    /// Priorities only compare quotients and differences of costs of one type, so the common
    /// factor `10^K` cancels and dividing it out would only add rounding.
    fn to_f64(self) -> f64 {
        u64_to_f64(self.raw())
    }
}

#[cfg(feature = "float")]
impl CostExt for search_core::NonNegF64 {
    /// Subtracts `other`, clamping at zero; `inf - inf` is zero.
    fn saturating_sub(self, other: Self) -> Self {
        // `f64::max` returns `0.0` for a NaN difference, which only `inf - inf` produces.
        Self::new((self.get() - other.get()).max(0.0)).unwrap_or(Self::ZERO)
    }

    fn to_f64(self) -> f64 {
        self.get()
    }
}

/// A rational heuristic weight `w = numerator / denominator` for Weighted A*.
///
/// Rational so that integer costs keep exact arithmetic: Weighted A* orders by
/// `denominator * g + numerator * h`, which sorts like `g + w * h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Weight {
    /// Numerator of the weight.
    numerator: u32,
    /// Denominator of the weight. Never zero.
    denominator: u32,
}

impl Weight {
    /// Weight `1`, which makes Weighted A* behave like A*.
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// Creates the weight `numerator / denominator`.
    ///
    /// Weights below `1` are allowed; they bias the search towards Dijkstra's order.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is zero.
    #[must_use]
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        assert!(denominator != 0, "weight denominator must not be zero");

        Self {
            numerator,
            denominator,
        }
    }

    /// Creates the integer weight `w`.
    #[must_use]
    pub const fn integer(w: u32) -> Self {
        Self::new(w, 1)
    }

    /// The numerator of the weight.
    #[must_use]
    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    /// The denominator of the weight. Never zero.
    #[must_use]
    pub const fn denominator(self) -> u32 {
        self.denominator
    }
}

/// Returns `true` if `cost <= weight * bound`, computed as `den * cost <= num * bound`
/// without division.
pub(crate) fn within<C: Cost>(cost: C, weight: Weight, bound: C) -> bool {
    cost.saturating_times(weight.denominator()) <= bound.saturating_times(weight.numerator())
}

#[cfg(test)]
mod tests {
    //! Unit tests for the [`CostExt`] implementations.

    use search_core::Decimal;

    use super::CostExt;

    /// Integer subtraction saturates at zero and converts exactly for small values.
    #[test]
    fn integers_saturate_and_convert() {
        assert_eq!(CostExt::saturating_sub(2_u8, 5), 0);
        assert_eq!(CostExt::saturating_sub(9_u128, 4), 5);
        assert_eq!(CostExt::saturating_sub(7_usize, 7), 0);
        assert!((CostExt::to_f64(3_u16) - 3.0).abs() < f64::EPSILON);
        assert!((CostExt::to_f64(u64::from(u32::MAX)) - f64::from(u32::MAX)).abs() < 1.0);
    }

    /// Decimals subtract on raw units and convert to the raw unit count.
    #[test]
    fn decimals_use_raw_units() {
        let a = Decimal::<2>::from_raw(150);
        let b = Decimal::<2>::from_raw(25);

        assert_eq!(a.saturating_sub(b), Decimal::from_raw(125));
        assert_eq!(b.saturating_sub(a), Decimal::from_raw(0));
        assert!((a.to_f64() - 150.0).abs() < f64::EPSILON);
    }

    /// Float subtraction clamps at zero, also for `inf - inf`.
    #[cfg(feature = "float")]
    #[test]
    fn floats_clamp_at_zero() {
        use search_core::{Cost, NonNegF64};

        let half = NonNegF64::new(0.5).expect("non-negative");
        let two = NonNegF64::new(2.0).expect("non-negative");

        assert_eq!(
            two.saturating_sub(half),
            NonNegF64::new(1.5).expect("non-negative")
        );
        assert_eq!(half.saturating_sub(two), NonNegF64::ZERO);
        assert_eq!(
            NonNegF64::INFINITY.saturating_sub(NonNegF64::INFINITY),
            NonNegF64::ZERO
        );
        assert_eq!(NonNegF64::INFINITY.saturating_sub(two), NonNegF64::INFINITY);
        assert!((two.to_f64() - 2.0).abs() < f64::EPSILON);
    }
}
