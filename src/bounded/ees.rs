//! Explicit Estimation Search ([`ees`], Thayer and Ruml 2011).

use alloc::vec::Vec;
use core::cmp::Reverse;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::GraphResult;
use crate::cost::{Ordered, Weight, within};
use crate::focal::FocalList;
use crate::keyed_set::KeyedSet;
use crate::outcome::{Outcome, SearchResult, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// The three estimates [`ees`] combines.
#[derive(Debug, Clone, Copy, Default)]
pub struct EesEstimates<H, Hh, D> {
    /// `h`: an admissible estimate of the cost to go. Proves the bound.
    pub admissible: H,
    /// `h^`: an accurate but possibly inadmissible estimate of the cost to go, such as a
    /// corrected or learned heuristic. Decides which states may be expanded.
    pub cost_to_go: Hh,
    /// `d^`: an estimate of the number of edges (search effort) to go, of any [`Cost`]
    /// type. Picks the state to expand among those.
    pub distance_to_go: D,
}

/// Explicit Estimation Search (EES): bounded suboptimal search guided by inadmissible
/// estimates, within `weight` times the optimum.
///
/// EES keeps three orders over the open states: `f = g + h` (admissible, the *cleanup*
/// order), `f^ = g + h^`, and `d^` among the states with `f^ <= w * f^_min` (the *focal*
/// order, a [`FocalList`]). It expands
///
/// 1. the best `d^` state if its `f^ <= w * f_min`, else
/// 2. the best `f^` state if its `f^ <= w * f_min`, else
/// 3. the best `f` state, which raises `f_min`.
///
/// Every expanded state has `g <= w * f_min`, so with an admissible `h` the solution costs
/// at most `w` times the optimum. `h^` and `d^` only steer; they need no guarantees. A state is
/// a dead end if any of the three estimates returns `None`.
///
/// # Panics
///
/// Panics if `weight` is below `1`.
///
/// # Examples
///
/// ```
/// use heuristic_search::{EesEstimates, Limits, Successors, Weight, ees, heuristic_fn};
///
/// /// A line of integers; each step moves one to the left or right at cost 2.
/// struct Line;
///
/// impl Successors for Line {
///     type State = i32;
///     type Action = i32;
///     type Cost = u32;
///
///     fn successors(&self, state: &i32, mut emit: impl FnMut(i32, i32, u32)) {
///         emit(-1, state - 1, 2);
///         emit(1, state + 1, 2);
///     }
/// }
///
/// let steps = |state: &i32| 4_i32.abs_diff(*state);
/// let estimates = EesEstimates {
///     admissible: heuristic_fn(|state: &i32| Some(steps(state))),
///     cost_to_go: heuristic_fn(|state: &i32| Some(2 * steps(state))),
///     distance_to_go: heuristic_fn(|state: &i32| Some(steps(state))),
/// };
/// let result = ees(&Line, 0, &|s: &i32| *s == 4, estimates, Weight::integer(2), Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.cost), Some(8));
/// ```
pub fn ees<G, T, H, Hh, D, Dc>(
    graph: &G,
    start: G::State,
    goal: &T,
    estimates: EesEstimates<H, Hh, D>,
    weight: Weight,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    Hh: Heuristic<G::State, G::Cost>,
    D: Heuristic<G::State, Dc>,
    Dc: Cost,
{
    let (space, start_goal) = SearchSpace::seed([start], goal);
    let search = Ees {
        graph,
        estimates,
        weight,
        space,
        focal: FocalList::new(weight),
        cleanup: KeyedSet::new(),
        stats: Stats::default(),
        values: Vec::new(),
        closed: Vec::new(),
        buffer: Vec::new(),
    };

    if let Some(found) = start_goal {
        let solution = search.space.solution(found);
        return search.finish(Outcome::Found(solution));
    }

    search.run(goal, budget)
}

/// The estimates of one node: `h`, `h^` and `d^`.
type Values<C, Dc> = (C, C, Dc);

/// The cleanup order: smallest `f`, then larger `g`, then node id.
type CleanupKey<C> = (Ordered<C>, Reverse<Ordered<C>>);

/// The state of one EES run.
struct Ees<'a, G: Successors, H, Hh, D, Dc> {
    /// The graph being searched.
    graph: &'a G,
    /// The three estimates, owned for the run.
    estimates: EesEstimates<H, Hh, D>,
    /// The suboptimality bound.
    weight: Weight,
    /// Every state seen so far.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// Open states by `f^`, with the focal sublist by `d^`.
    focal: FocalList<G::Cost, Ordered<Dc>>,
    /// Open states by admissible `f`.
    cleanup: KeyedSet<CleanupKey<G::Cost>>,
    /// Counters for the result.
    stats: Stats,
    /// Estimates per node; `None` for dead ends, which are never opened.
    values: Vec<Option<Values<G::Cost, Dc>>>,
    /// Whether each node has been expanded at least once.
    closed: Vec<bool>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, H, Hh, D, Dc> Ees<'_, G, H, Hh, D, Dc>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    H: Heuristic<G::State, G::Cost>,
    Hh: Heuristic<G::State, G::Cost>,
    D: Heuristic<G::State, Dc>,
    Dc: Cost,
{
    /// Expands until a goal is expanded, nothing is open or the budget is used up.
    fn run<T: Goal<G::State> + ?Sized>(
        mut self,
        goal: &T,
        mut budget: impl Budget<Stats>,
    ) -> GraphResult<G> {
        for start in 0..self.space.len() {
            self.add_node(start);
        }

        while let Some(node) = self.select() {
            self.focal.remove(node);
            self.cleanup.remove(node);
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

    /// Picks the next node by EES's three rules.
    fn select(&self) -> Option<NodeId> {
        let (key, best_f) = self.cleanup.first()?;
        let f_min = key.0.0;
        let acceptable = |id: NodeId| {
            self.focal
                .f(id)
                .is_some_and(|f_hat| within(f_hat, self.weight, f_min))
        };

        [self.focal.peek(), self.focal.peek_min_f()]
            .into_iter()
            .flatten()
            .find(|&id| acceptable(id))
            .or(Some(best_f))
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

    /// Evaluates the new node `id` and opens it, unless an estimate proves it a dead end.
    fn add_node(&mut self, id: NodeId) {
        let state = self.space.state(id);
        let values = self.estimates.admissible.estimate(state).and_then(|h| {
            let h_hat = self.estimates.cost_to_go.estimate(state)?;
            let d_hat = self.estimates.distance_to_go.estimate(state)?;

            Some((h, h_hat, d_hat))
        });

        self.values.push(values);
        self.closed.push(false);
        self.open(id);
    }

    /// Opens the known node `id` again after its path cost improved.
    fn reopen(&mut self, id: NodeId) {
        if self.values[id].is_some() && self.closed[id] && self.cleanup.key(id).is_none() {
            self.stats.reopened += 1;
        }

        self.open(id);
    }

    /// Pushes `id` into both orders with its current `g`, unless it is a dead end.
    fn open(&mut self, id: NodeId) {
        let Some((h, h_hat, d_hat)) = self.values[id] else {
            return;
        };

        let g = self.space.g(id);
        self.focal.push(id, g + h_hat, g, Ordered(d_hat));
        self.cleanup
            .insert(id, (Ordered(g + h), Reverse(Ordered(g))));
    }

    /// Wraps up the run with `outcome`.
    fn finish(self, outcome: Outcome<G::State, G::Action, G::Cost>) -> GraphResult<G> {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
