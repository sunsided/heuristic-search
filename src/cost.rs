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
}

/// Implements [`Cost`] for unsigned and signed integer types that `u32` converts into
/// losslessly.
macro_rules! integer_cost {
    ($($ty:ty),*) => {
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
            }
        )*
    };
}

integer_cost!(u32, u64, i64);

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
