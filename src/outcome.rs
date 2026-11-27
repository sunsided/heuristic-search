//! What a search returns: [`SearchResult`] with an [`Outcome`] and [`Stats`], and the
//! [`Limits`], the counter [`Budget`] that stops it early.

use alloc::vec::Vec;

use search_core::Budget;

/// Counters collected during one search run.
///
/// Non-exhaustive: later algorithms may add counters. Start from [`Stats::default`] where
/// a value is needed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Stats {
    /// States whose successors were generated. A goal state that ends the search is not
    /// counted. A state reopened through a cheaper path counts once per expansion.
    pub expanded: u64,
    /// Edges reported by [`Successors::successors`](crate::Successors::successors), whether
    /// or not their target was new.
    pub generated: u64,
    /// Closed states reopened because a cheaper path to them turned up. Always `0` for
    /// breadth-first and greedy best-first search, and for A* with a consistent heuristic.
    pub reopened: u64,
    /// True edge costs computed by an expensive edge evaluator. Only
    /// [`lazy_sp`](crate::lazy_sp) evaluates edges; `0` everywhere else.
    pub evaluations: u64,
}

impl Stats {
    /// The counters of `self` and `other` added up, for searches made of several runs.
    #[must_use]
    pub(crate) fn plus(self, other: Self) -> Self {
        Self {
            expanded: self.expanded + other.expanded,
            generated: self.generated + other.generated,
            reopened: self.reopened + other.reopened,
            evaluations: self.evaluations + other.evaluations,
        }
    }

    /// The counters of `self` beyond `earlier`, an earlier reading of the same counters.
    #[cfg(feature = "parallel")]
    #[must_use]
    pub(crate) fn since(self, earlier: Self) -> Self {
        Self {
            expanded: self.expanded - earlier.expanded,
            generated: self.generated - earlier.generated,
            reopened: self.reopened - earlier.reopened,
            evaluations: self.evaluations - earlier.evaluations,
        }
    }
}

/// A path from the start state to a goal state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution<S, A, C> {
    /// The states along the path, from the start to the goal. Never empty: a start state that
    /// already is a goal yields a single state.
    pub states: Vec<S>,
    /// The edge labels along the path; `actions[i]` leads from `states[i]` to
    /// `states[i + 1]`. Always one shorter than [`Solution::states`].
    pub actions: Vec<A>,
    /// The sum of the edge costs along the path.
    pub cost: C,
}

impl<S, A, C> Solution<S, A, C> {
    /// The number of edges on the path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns `true` if the start state is the goal, so the path has no edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// The goal state the path ends in.
    ///
    /// # Panics
    ///
    /// Never panics for solutions returned by a search, whose [`Solution::states`] is never
    /// empty.
    #[must_use]
    pub fn goal(&self) -> &S {
        self.states
            .last()
            .expect("a solution has at least one state")
    }
}

/// How a search ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome<S, A, C> {
    /// A path to a goal was found.
    Found(Solution<S, A, C>),
    /// The search space was exhausted without reaching a goal, so no goal is reachable
    /// (given a heuristic whose dead ends are real).
    Unsolvable,
    /// The [`Budget`] stopped the search before it could decide.
    LimitReached,
    /// A search with its own bound cut states off and found no goal within the bound: the
    /// depth limit of [`depth_limited_dfs`](crate::depth_limited_dfs), the beam width of
    /// [`beam_search`](crate::beam_search), or the memory of
    /// [`sma_star`](crate::sma_star). A goal may lie beyond it; a larger bound may find it.
    Cutoff,
}

/// The result of one search run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult<S, A, C> {
    /// How the search ended.
    pub outcome: Outcome<S, A, C>,
    /// Counters collected until the search ended.
    pub stats: Stats,
}

impl<S, A, C> SearchResult<S, A, C> {
    /// The solution, if one was found.
    #[must_use]
    pub fn solution(&self) -> Option<&Solution<S, A, C>> {
        match &self.outcome {
            Outcome::Found(solution) => Some(solution),
            Outcome::Unsolvable | Outcome::LimitReached | Outcome::Cutoff => None,
        }
    }

    /// Consumes the result and returns the solution, if one was found.
    #[must_use]
    pub fn into_solution(self) -> Option<Solution<S, A, C>> {
        match self.outcome {
            Outcome::Found(solution) => Some(solution),
            Outcome::Unsolvable | Outcome::LimitReached | Outcome::Cutoff => None,
        }
    }
}

/// Simple counter limits. The default is unlimited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    /// Stop once this many states have been expanded. `None` is unbounded.
    pub max_expansions: Option<u64>,
}

impl Limits {
    /// No limits: the search runs until it finds a goal or exhausts the space.
    pub const NONE: Self = Self {
        max_expansions: None,
    };

    /// Stops after `max` expansions.
    #[must_use]
    pub const fn expansions(max: u64) -> Self {
        Self {
            max_expansions: Some(max),
        }
    }
}

/// Asked before every expansion, with the counters so far, so a search stopped by
/// [`Limits::max_expansions`] `= n` has expanded exactly `n` states. Closures
/// `FnMut(&Stats) -> bool` are budgets too, which lets callers with a clock (this crate is
/// `no_std`) add a time limit; annotate the argument (`|stats: &Stats| ...`).
impl Budget<Stats> for Limits {
    fn exhausted(&mut self, stats: &Stats) -> bool {
        self.max_expansions.is_some_and(|max| stats.expanded >= max)
    }
}
