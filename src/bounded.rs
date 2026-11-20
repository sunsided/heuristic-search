//! Bounded-suboptimal search: [`focal_search`], [`ees()`] and [`dps()`] return a solution that
//! costs at most `w` times the optimum, and use the slack to search where a second
//! estimate says the goal is near.
//!
//! | Algorithm        | Expands                                              | Estimates           |
//! | ---------------- | ---------------------------------------------------- | ------------------- |
//! | [`focal_search`] | smallest focal key among `f <= w * f_min`            | `h`, focal key      |
//! | [`ees()`]          | smallest `d^` among `f^ <= w * f^_min`, else cleanup | `h`, `h^`, `d^`     |
//! | [`dps()`]          | largest `(w * f_min - g) / h`                        | `h`                 |
//!
//! All three reopen closed states when a cheaper path turns up. That keeps a state of an
//! optimal path open with its optimal `g`, which is what makes `f_min` a lower bound and
//! the `w` bound hold, for any admissible `h`, consistent or not. Goals are tested when
//! they are expanded.

mod dps;
mod ees;

pub use dps::dps;
pub use ees::{EesEstimates, ees};

use alloc::vec::Vec;
use core::hash::Hash;

use crate::best_first::GraphResult;
use crate::cost::Weight;
use crate::focal::FocalList;
use crate::outcome::{Budget, Outcome, SearchResult, Stats};
use crate::problem::{Goal, Heuristic, Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// The focal key of a search node, computed along the path that reaches it.
///
/// The key may depend on the whole path, not just the state: multi-agent path finding, for
/// example, counts the conflicts accumulated along a path. Each node keeps the key of its
/// best known path, recomputed with [`FocalKey::extend`] whenever that path improves. For
/// keys that depend on the state alone, wrap a closure with [`focal_fn`].
pub trait FocalKey<S, A, C> {
    /// The key; smaller is expanded first.
    type Key: Ord + Clone;

    /// The key of a start state.
    fn start(&mut self, state: &S) -> Self::Key;

    /// The key of `state`, reached from `parent` (whose key is `parent_key`) through
    /// `action` at edge cost `cost`.
    fn extend(
        &mut self,
        parent_key: &Self::Key,
        parent: &S,
        action: &A,
        cost: C,
        state: &S,
    ) -> Self::Key;
}

impl<S, A, C, F: FocalKey<S, A, C> + ?Sized> FocalKey<S, A, C> for &mut F {
    type Key = F::Key;

    fn start(&mut self, state: &S) -> Self::Key {
        (**self).start(state)
    }

    fn extend(
        &mut self,
        parent_key: &Self::Key,
        parent: &S,
        action: &A,
        cost: C,
        state: &S,
    ) -> Self::Key {
        (**self).extend(parent_key, parent, action, cost, state)
    }
}

/// Turns a closure `FnMut(&S) -> K` into a [`FocalKey`] that depends on the state alone.
///
/// # Examples
///
/// ```
/// use heuristic_search::{FocalKey, focal_fn};
///
/// let mut key = focal_fn(|n: &u32| 10_u32.abs_diff(*n));
///
/// assert_eq!(FocalKey::<u32, (), u32>::start(&mut key, &4), 6);
/// assert_eq!(FocalKey::<u32, (), u32>::extend(&mut key, &6, &4, &(), 1, &5), 5);
/// ```
pub fn focal_fn<S, K: Ord + Clone, F: FnMut(&S) -> K>(f: F) -> FocalFn<F> {
    FocalFn(f)
}

/// A closure used as a state-only [`FocalKey`]; created by [`focal_fn`].
#[derive(Debug, Clone, Copy)]
pub struct FocalFn<F>(F);

impl<S, A, C, K: Ord + Clone, F: FnMut(&S) -> K> FocalKey<S, A, C> for FocalFn<F> {
    type Key = K;

    fn start(&mut self, state: &S) -> K {
        (self.0)(state)
    }

    fn extend(&mut self, _: &K, _: &S, _: &A, _: C, state: &S) -> K {
        (self.0)(state)
    }
}

/// Focal search (A*<sub>ε</sub>, Pearl and Kim 1982): a solution within `weight` times the
/// optimum, expanding by a free secondary key.
///
/// Keeps open states ordered by `f = g + h` and expands, among those with
/// `f <= w * f_min`, the one with the smallest `focal` key (ties by smaller `f`, then
/// larger `g`, then first in, first out). With an admissible `heuristic` the solution costs
/// at most `w` times the optimum. With [`Weight::ONE`] it is A* with the focal key as
/// tie-breaker among equal `f`.
///
/// The open and focal lists are a [`FocalList`], which is public for searches that need
/// their own loop.
///
/// # Panics
///
/// Panics if `weight` is below `1`.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Weight, focal_fn, focal_search, heuristic_fn};
///
/// /// A line of integers; each step moves one to the left or right at cost 1.
/// struct Line;
///
/// impl Successors for Line {
///     type State = i32;
///     type Action = i32;
///     type Cost = u32;
///
///     fn successors(&self, state: &i32, mut emit: impl FnMut(i32, i32, u32)) {
///         emit(-1, state - 1, 1);
///         emit(1, state + 1, 1);
///     }
/// }
///
/// let to_goal = |state: &i32| 6_i32.abs_diff(*state);
/// let result = focal_search(
///     &Line,
///     0,
///     &|state: &i32| *state == 6,
///     heuristic_fn(|state: &i32| Some(to_goal(state))),
///     focal_fn(to_goal),
///     Weight::integer(2),
///     Limits::NONE,
/// );
///
/// assert_eq!(result.solution().map(|solution| solution.cost), Some(6));
/// assert_eq!(result.stats.expanded, 6);
/// ```
pub fn focal_search<G, T, H, F>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    focal: F,
    weight: Weight,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    F: FocalKey<G::State, G::Action, G::Cost>,
{
    let (space, start_goal) = SearchSpace::seed([start], goal);
    let search = FocalSearch {
        graph,
        heuristic,
        focal,
        space,
        list: FocalList::new(weight),
        stats: Stats::default(),
        h: Vec::new(),
        keys: Vec::new(),
        closed: Vec::new(),
        buffer: Vec::new(),
    };

    if let Some(found) = start_goal {
        let solution = search.space.solution(found);
        return search.finish(Outcome::Found(solution));
    }

    search.run(goal, budget)
}

/// The state of one focal search run.
struct FocalSearch<'a, G: Successors, H, F: FocalKey<G::State, G::Action, G::Cost>> {
    /// The graph being searched.
    graph: &'a G,
    /// The admissible heuristic, owned for the run.
    heuristic: H,
    /// The focal key, owned for the run.
    focal: F,
    /// Every state seen so far.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// The open and focal lists.
    list: FocalList<G::Cost, F::Key>,
    /// Counters for the result.
    stats: Stats,
    /// Heuristic value per node; `None` for proven dead ends, which are never opened.
    h: Vec<Option<G::Cost>>,
    /// Focal key per node along its best known path; `None` for dead ends.
    keys: Vec<Option<F::Key>>,
    /// Whether each node has been expanded at least once.
    closed: Vec<bool>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, H, F> FocalSearch<'_, G, H, F>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    H: Heuristic<G::State, G::Cost>,
    F: FocalKey<G::State, G::Action, G::Cost>,
{
    /// Expands from the focal list until a goal is expanded, the list is empty or the
    /// budget is used up.
    fn run<T: Goal<G::State> + ?Sized>(
        mut self,
        goal: &T,
        mut budget: impl Budget,
    ) -> GraphResult<G> {
        for start in 0..self.space.len() {
            self.add_node(start);
        }

        while let Some(node) = self.list.pop() {
            if goal.is_goal(self.space.state(node)) {
                let solution = self.space.solution(node);
                return self.finish(Outcome::Found(solution));
            }

            if budget.exhausted(&self.stats) {
                return self.finish(Outcome::LimitReached);
            }

            self.expand(node);
        }

        self.finish(Outcome::Unsolvable)
    }

    /// Generates the successors of `node` and opens the new and improved ones.
    fn expand(&mut self, node: NodeId) {
        self.stats.expanded += 1;
        self.closed[node] = true;
        self.stats.generated +=
            collect_successors(self.graph, self.space.state(node), &mut self.buffer);

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match self.space.reach(state, node, action, cost, true) {
                Reach::New(id) => self.add_node(id),
                Reach::Improved(id) => self.reopen(id),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
    }

    /// Evaluates the new node `id` and opens it, unless it is a dead end.
    fn add_node(&mut self, id: NodeId) {
        let h = self.heuristic.estimate(self.space.state(id));

        self.h.push(h);
        self.keys.push(None);
        self.closed.push(false);
        if h.is_some() {
            self.open(id);
        }
    }

    /// Opens the known node `id` again after its path cost improved.
    fn reopen(&mut self, id: NodeId) {
        if self.h[id].is_none() {
            return;
        }

        if self.closed[id] && !self.list.contains(id) {
            self.stats.reopened += 1;
        }

        self.open(id);
    }

    /// Computes the focal key of `id` along its best path and pushes it.
    fn open(&mut self, id: NodeId) {
        let state = self.space.state(id);
        let key = match self.space.parent(id) {
            None => self.focal.start(state),
            Some((parent, action, cost)) => {
                let parent_key = self.keys[parent]
                    .as_ref()
                    .expect("expanded nodes have keys");
                self.focal
                    .extend(parent_key, self.space.state(parent), action, cost, state)
            }
        };

        let g = self.space.g(id);
        let h = self.h[id].expect("only live nodes are opened");
        self.list.push(id, g + h, g, key.clone());
        self.keys[id] = Some(key);
    }

    /// Wraps up the run with `outcome`.
    fn finish(self, outcome: Outcome<G::State, G::Action, G::Cost>) -> GraphResult<G> {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
