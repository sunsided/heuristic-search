//! Resumable best-first search ([`AStarSearch`]): A*, Weighted A* or GBFS that stops when
//! its budget runs out and continues where it stopped on the next call, so one search can
//! be spread over several frames.

use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use super::{Engine, Order, Workspace};
use crate::cost::Weight;
use crate::outcome::{Outcome, SearchResult, Stats};
use crate::problem::Successors;
use crate::space::NodeId;

/// A* search that keeps its open list and search space between calls, for a per-frame
/// budget: each [`AStarSearch::compute`] expands until its budget runs out and the next
/// call continues from there.
///
/// Expands the same states in the same order as [`a_star`](crate::a_star), so across all
/// calls it finds the same path with the same total counters. [`AStarSearch::new_weighted`]
/// and [`AStarSearch::new_gbfs`] resume [`weighted_a_star`](crate::weighted_a_star) and
/// [`gbfs`](crate::gbfs) the same way, and [`AStarSearch::new`] with the [`Zero`](crate::Zero)
/// heuristic resumes [`dijkstra`](crate::dijkstra). Like the incremental planners
/// ([`LpaStar`](crate::LpaStar) and so on), it takes the graph and the goal per call; pass
/// the same ones every time.
///
/// # Examples
///
/// ```
/// use heuristic_search::{AStarSearch, Limits, Outcome, Successors, heuristic_fn};
///
/// /// Counting up by one, at cost 1.
/// struct Up;
///
/// impl Successors for Up {
///     type State = u32;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
///         emit((), n + 1, 1);
///     }
/// }
///
/// let goal = |n: &u32| *n == 10;
/// let mut search = AStarSearch::new(&Up, 0, &goal, heuristic_fn(|n: &u32| 10_u32.checked_sub(*n)));
///
/// // Four expansions per frame.
/// let mut frames = 1;
/// while matches!(search.compute(&Up, &goal, Limits::expansions(4)).outcome, Outcome::LimitReached) {
///     frames += 1;
/// }
///
/// assert_eq!(frames, 3);
/// assert_eq!(search.stats().expanded, 10);
/// let result = search.compute(&Up, &goal, Limits::NONE);
/// assert_eq!(result.solution().map(|s| s.cost), Some(10));
/// ```
pub struct AStarSearch<S, A, C, H> {
    /// The heuristic, owned by the search.
    heuristic: H,
    /// The search space and open list, kept between calls.
    work: Workspace<S, A, C>,
    /// A start that is a goal, found when the search was created.
    start_goal: Option<NodeId>,
    /// Counters over every call so far.
    total: Stats,
    /// The open-list order, fixed at construction: the start keys depend on it.
    order: Order,
}

impl<S, A, C, H> AStarSearch<S, A, C, H>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
    H: Heuristic<S, C>,
{
    /// Prepares an A* search from `start` to the first state that satisfies `goal`.
    /// Evaluates the start, but expands nothing until [`AStarSearch::compute`].
    pub fn new<G, T>(graph: &G, start: S, goal: &T, heuristic: H) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        Self::new_multi(graph, [start], goal, heuristic)
    }

    /// Prepares an A* search from several start states at once, with the start handling of
    /// [`a_star_multi`](crate::a_star_multi).
    pub fn new_multi<G, T>(
        graph: &G,
        starts: impl IntoIterator<Item = S>,
        goal: &T,
        heuristic: H,
    ) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        Self::seeded(graph, starts, goal, heuristic, Order::Weighted(Weight::ONE))
    }

    /// Prepares a Weighted A* search with `weight` from `start`, resuming
    /// [`weighted_a_star`](crate::weighted_a_star).
    pub fn new_weighted<G, T>(graph: &G, start: S, goal: &T, heuristic: H, weight: Weight) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        Self::new_weighted_multi(graph, [start], goal, heuristic, weight)
    }

    /// Prepares a Weighted A* search from several start states at once, resuming
    /// [`weighted_a_star_multi`](crate::weighted_a_star_multi).
    pub fn new_weighted_multi<G, T>(
        graph: &G,
        starts: impl IntoIterator<Item = S>,
        goal: &T,
        heuristic: H,
        weight: Weight,
    ) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        Self::seeded(graph, starts, goal, heuristic, Order::Weighted(weight))
    }

    /// Prepares a greedy best-first search from `start`, resuming [`gbfs`](crate::gbfs).
    pub fn new_gbfs<G, T>(graph: &G, start: S, goal: &T, heuristic: H) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        Self::new_gbfs_multi(graph, [start], goal, heuristic)
    }

    /// Prepares a greedy best-first search from several start states at once, resuming
    /// [`gbfs_multi`](crate::gbfs_multi).
    pub fn new_gbfs_multi<G, T>(
        graph: &G,
        starts: impl IntoIterator<Item = S>,
        goal: &T,
        heuristic: H,
    ) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        Self::seeded(graph, starts, goal, heuristic, Order::H)
    }

    /// Seeds a search with open-list order `order` from `starts`.
    fn seeded<G, T>(
        graph: &G,
        starts: impl IntoIterator<Item = S>,
        goal: &T,
        mut heuristic: H,
        order: Order,
    ) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        let mut work = Workspace::new();
        let start_goal = Engine::new(graph, goal, &mut heuristic, order, &mut work).seed(starts);

        Self {
            heuristic,
            work,
            start_goal,
            total: Stats::default(),
            order,
        }
    }

    /// Continues the search until a goal is reached, the open list runs empty or `budget`
    /// is used up, and returns [`Outcome::LimitReached`] in the last case; the next call
    /// then continues where this one stopped.
    ///
    /// The budget and the returned stats cover this call only; see [`AStarSearch::stats`]
    /// for the totals. Once the search has finished, every further call returns the same
    /// outcome again without expanding anything.
    pub fn compute<G, T>(
        &mut self,
        graph: &G,
        goal: &T,
        budget: impl Budget<Stats>,
    ) -> SearchResult<S, A, C>
    where
        G: Successors<State = S, Action = A, Cost = C>,
        T: Goal<S> + ?Sized,
    {
        if let Some(found) = self.start_goal {
            return SearchResult {
                outcome: Outcome::Found(self.work.space.solution(found)),
                stats: Stats::default(),
            };
        }

        let mut engine = Engine::new(graph, goal, &mut self.heuristic, self.order, &mut self.work);
        let outcome = engine.advance(budget).unwrap_or(Outcome::LimitReached);
        let stats = engine.stats;
        self.total = self.total.plus(stats);

        SearchResult { outcome, stats }
    }

    /// Counters over every [`AStarSearch::compute`] so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.total
    }
}
