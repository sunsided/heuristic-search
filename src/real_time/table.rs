//! The learned heuristic table ([`LearnedHeuristic`]): learned values over a base
//! heuristic, kept by the caller across trials.

use core::hash::Hash;

use hashbrown::HashMap;
use search_core::{Cost, Heuristic};

/// A heuristic that real-time searches learn: a table of learned values in front of a base
/// heuristic.
///
/// A state's value is its learned value if it has one, else the base estimate. Values are
/// `Option<C>` as in [`Heuristic`]: `None` is a proven dead end, from which no goal is
/// reachable. The searches only learn `None` when they have proven it.
///
/// The table implements [`Heuristic`] itself, so after learning it can guide any other
/// search in this crate, such as [`a_star`](crate::a_star) (pass `&mut table`).
///
/// Learning keeps the guarantees of the initial heuristic: from an admissible base, the
/// values of [`lrta_star`](crate::lrta_star), [`lss_lrta_star`](crate::lss_lrta_star),
/// [`lhb_lrta_star`](crate::lhb_lrta_star) and [`rtaa_star`](crate::rtaa_star) stay
/// admissible; from a consistent base, they stay consistent. [`rta_star`](crate::rta_star)
/// learns inadmissible values, meant for the current trial only.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Heuristic, LearnedHeuristic, Zero};
///
/// let mut table = LearnedHeuristic::<u32, u32, _>::new(Zero);
/// assert_eq!(table.get(&7), Some(0));
///
/// assert!(table.set(7, Some(3)));
/// assert!(!table.set(7, Some(3)), "unchanged");
/// assert_eq!(table.estimate(&7), Some(3));
/// assert_eq!(table.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct LearnedHeuristic<S, C, H> {
    /// The initial estimate of every state without a learned value.
    base: H,
    /// Learned values. A state is in here once a search wrote it, even if the value equals
    /// the base estimate.
    learned: HashMap<S, Option<C>>,
}

impl<S: Hash + Eq, C: Cost, H: Heuristic<S, C>> LearnedHeuristic<S, C, H> {
    /// An empty table over `base`.
    pub fn new(base: H) -> Self {
        Self {
            base,
            learned: HashMap::new(),
        }
    }

    /// The value of `state`: its learned value, else the base estimate.
    pub fn get(&mut self, state: &S) -> Option<C> {
        match self.learned.get(state) {
            Some(&value) => value,
            None => self.base.estimate(state),
        }
    }

    /// Sets the learned value of `state` and returns `true` if its value changed.
    ///
    /// The value is stored even if it equals the current one.
    pub fn set(&mut self, state: S, value: Option<C>) -> bool {
        let changed = self.get(&state) != value;
        self.learned.insert(state, value);

        changed
    }

    /// The learned value of `state`, or `None` if it has none (and falls back to the base).
    #[must_use]
    pub fn learned(&self, state: &S) -> Option<Option<C>> {
        self.learned.get(state).copied()
    }

    /// The number of states with a learned value.
    #[must_use]
    pub fn len(&self) -> usize {
        self.learned.len()
    }

    /// Returns `true` if no state has a learned value.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.learned.is_empty()
    }

    /// The states with a learned value and their values, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&S, Option<C>)> {
        self.learned.iter().map(|(state, &value)| (state, value))
    }

    /// Forgets every learned value; the base heuristic remains.
    pub fn clear(&mut self) {
        self.learned.clear();
    }

    /// The base heuristic.
    pub fn base(&self) -> &H {
        &self.base
    }

    /// Forgets every learned dead end, so those states fall back to the base estimate.
    pub(crate) fn forget_dead_ends(&mut self) {
        self.learned.retain(|_, value| value.is_some());
    }

    /// The same learned values in front of another base heuristic.
    pub(crate) fn rebase<B: Heuristic<S, C>>(self, base: B) -> LearnedHeuristic<S, C, B> {
        LearnedHeuristic {
            base,
            learned: self.learned,
        }
    }
}

impl<S: Hash + Eq, C: Cost, H: Heuristic<S, C>> Heuristic<S, C> for LearnedHeuristic<S, C, H> {
    fn estimate(&mut self, state: &S) -> Option<C> {
        self.get(state)
    }
}
