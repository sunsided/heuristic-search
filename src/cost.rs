//! Priority arithmetic on costs ([`CostExt`]) and heuristic weights ([`Weight`]).
//!
//! The [`Cost`] and [`CostExt`] traits come from `search-core`. [`CostExt`] adds the
//! subtraction and conversion to `f64` that a few searches need for their priorities, such
//! as the `(G - g) / h` order of [`ana_star`](crate::ana_star), the `2g + h - h_opposite` key
//! of [`dibbs`](crate::dibbs) and [`bae_star`](crate::bae_star), and the potential of
//! [`dps`](crate::dps).

use search_core::Cost;
pub use search_core::CostExt;

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

/// A cost or infinity, for bounds and backed-up values that must be able to exceed every
/// cost.
///
/// Ordered with every [`Extended::Finite`] value below [`Extended::Infinite`]. Saturated
/// costs are no substitute: a saturated sum ties with real costs at the maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Extended<C> {
    /// A finite cost.
    Finite(C),
    /// Larger than every cost: no bound, or a subtree without any goal.
    Infinite,
}

impl<C: Cost> Extended<C> {
    /// `g + h` for an estimate `h`, where a dead end (`None`) is infinite.
    pub(crate) fn f(g: C, h: Option<C>) -> Self {
        h.map_or(Self::Infinite, |h| Self::Finite(g.saturating_add(h)))
    }
}

/// Returns `true` if `cost <= weight * bound`, computed as `den * cost <= num * bound`
/// without division.
pub(crate) fn within<C: Cost>(cost: C, weight: Weight, bound: C) -> bool {
    cost.saturating_times(weight.denominator()) <= bound.saturating_times(weight.numerator())
}
