//! Edge and path costs ([`Cost`]) and heuristic weights ([`Weight`]).

use core::cmp::Ordering;
use core::ops::Add;

/// A non-negative edge or path cost.
///
/// Searches add costs along a path and compare them. Integers compare with [`Ord`], floats
/// with [`f64::total_cmp`], so a NaN cost never panics but orders after every number.
///
/// Implemented for `u32`, `u64`, `usize`, `i64` and `f64`. Integer additions follow the
/// usual overflow rules (panic in debug builds), so pick a type wide enough for the
/// longest path.
pub trait Cost: Copy + Add<Output = Self> {
    /// The cost of the empty path.
    const ZERO: Self;

    /// Total order on costs; smaller is cheaper.
    fn cmp_cost(&self, other: &Self) -> Ordering;

    /// Multiplies the cost by a small integer factor.
    ///
    /// Used by [`weighted_astar`](crate::weighted_astar) to compare `den * g + num * h`
    /// without division. Integer implementations saturate instead of overflowing.
    #[must_use]
    fn times(self, factor: u32) -> Self;

    /// Adds two costs, saturating at the maximum instead of overflowing for integers.
    ///
    /// Used where a sum only serves as a priority, such as the Weighted A* key.
    #[must_use]
    fn saturating_add(self, other: Self) -> Self;

    /// Subtracts `other`, saturating at the minimum (zero for unsigned integers).
    ///
    /// Used for priorities that subtract one estimate from another, such as the `G - g`
    /// slack of [`ana_star`](crate::ana_star).
    #[must_use]
    fn saturating_sub(self, other: Self) -> Self;

    /// The cost as a float, possibly rounded.
    ///
    /// Used only for priorities that divide costs, such as the `(G - g) / h` order of
    /// [`ana_star`](crate::ana_star). Rounding can
    /// change the expansion order among nearly equal priorities but never a returned cost.
    fn to_f64(self) -> f64;
}

/// Implements [`Cost`] for unsigned and signed integer types that `u32` converts into
/// losslessly, each with its conversion to `f64`.
macro_rules! integer_cost {
    ($($ty:ty => $to_f64:path),*) => {
        $(
            impl Cost for $ty {
                const ZERO: Self = 0;

                fn cmp_cost(&self, other: &Self) -> Ordering {
                    self.cmp(other)
                }

                fn times(self, factor: u32) -> Self {
                    self.saturating_mul(<$ty>::from(factor))
                }

                fn saturating_add(self, other: Self) -> Self {
                    <$ty>::saturating_add(self, other)
                }

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

integer_cost!(u32 => f64::from, u64 => u64_to_f64, i64 => i64_to_f64);

/// Converts a `u64` cost to the nearest float, for priorities only.
#[expect(clippy::cast_precision_loss, reason = "only used for priorities")]
fn u64_to_f64(cost: u64) -> f64 {
    cost as f64
}

/// Converts an `i64` cost to the nearest float, for priorities only.
#[expect(clippy::cast_precision_loss, reason = "only used for priorities")]
fn i64_to_f64(cost: i64) -> f64 {
    cost as f64
}

impl Cost for usize {
    const ZERO: Self = 0;

    fn cmp_cost(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn times(self, factor: u32) -> Self {
        usize::try_from(factor).map_or(usize::MAX, |factor| self.saturating_mul(factor))
    }

    fn saturating_add(self, other: Self) -> Self {
        usize::saturating_add(self, other)
    }

    fn saturating_sub(self, other: Self) -> Self {
        usize::saturating_sub(self, other)
    }

    #[expect(clippy::cast_precision_loss, reason = "only used for priorities")]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Cost for f64 {
    const ZERO: Self = 0.0;

    fn cmp_cost(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }

    fn times(self, factor: u32) -> Self {
        self * f64::from(factor)
    }

    fn saturating_add(self, other: Self) -> Self {
        self + other
    }

    fn saturating_sub(self, other: Self) -> Self {
        self - other
    }

    fn to_f64(self) -> f64 {
        self
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

/// A [`Cost`] with a total [`Ord`] from [`Cost::cmp_cost`], for ordered collections.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ordered<C>(pub(crate) C);

impl<C: Cost> PartialEq for Ordered<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0.cmp_cost(&other.0).is_eq()
    }
}

impl<C: Cost> Eq for Ordered<C> {}

impl<C: Cost> PartialOrd for Ordered<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Cost> Ord for Ordered<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_cost(&other.0)
    }
}

/// Returns `true` if `cost <= weight * bound`, computed as `den * cost <= num * bound`
/// without division.
pub(crate) fn within<C: Cost>(cost: C, weight: Weight, bound: C) -> bool {
    cost.times(weight.denominator())
        .cmp_cost(&bound.times(weight.numerator()))
        .is_le()
}
