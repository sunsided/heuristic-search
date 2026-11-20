//! Anytime Repairing A* ([`ara_star`], Likhachev, Gordon and Thrun 2003).

use alloc::vec::Vec;
use core::hash::Hash;
use core::iter::Peekable;

use crate::anytime::{AnytimeGraphResult, AnytimeResult};
use crate::cost::{Cost, Weight};
use crate::open_list::OpenList;
use crate::outcome::{Budget, Solution, Stats};
use crate::problem::{Goal, Heuristic, Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// Anytime Repairing A* (ARA*): a series of Weighted A* searches with falling weights that
/// reuse each other's work.
///
/// Runs one iteration per weight of `weights`, in order. An iteration expands by
/// `g + w * h` until the incumbent's cost is at most the smallest open key, so its solution
/// costs at most `w` times the optimum. States improved after their expansion in the
/// current iteration are not reopened but parked and carried into the next iteration,
/// which is what makes ARA* cheaper than restarting Weighted A*. The last iteration does
/// reopen them, so the final bound holds for every heuristic the last weight allows.
///
/// After each iteration that improved the incumbent, `on_solution` is called with the new
/// solution and a lower bound on the optimal cost (the smallest `g + h` among open and
/// parked states). The search stops early once the lower bound reaches the incumbent's cost.
///
/// Use a non-increasing schedule that ends with [`Weight::ONE`] for an optimal final
/// solution, such as `[3, 2, 3/2, 1]`. An empty schedule means `[1]` (plain A*). The
/// guarantees assume an admissible and consistent `heuristic`; with an inconsistent one
/// every iteration still returns valid paths, and the last stays within its weight.
/// Goal states are recognised when they are generated and are not expanded.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Weight, ara_star, heuristic_fn};
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
/// let distance = heuristic_fn(|state: &i32| Some(5_i32.abs_diff(*state)));
/// let weights = [Weight::integer(3), Weight::new(3, 2), Weight::ONE];
/// let mut costs = Vec::new();
/// let result = ara_star(
///     &Line,
///     0,
///     &|state: &i32| *state == 5,
///     distance,
///     weights,
///     |solution, _lower_bound| costs.push(solution.cost),
///     Limits::NONE,
/// );
///
/// assert!(result.complete);
/// assert_eq!(result.solution.map(|solution| solution.cost), Some(5));
/// assert_eq!(costs, [5]);
/// ```
pub fn ara_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    weights: impl IntoIterator<Item = Weight>,
    on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>, G::Cost),
    budget: impl Budget,
) -> AnytimeGraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let (space, start_goal) = SearchSpace::seed([start], goal);
    let ara = Ara {
        graph,
        goal,
        heuristic,
        space,
        stats: Stats::default(),
        h: Vec::new(),
        closed_in: Vec::new(),
        open_flag: Vec::new(),
        parked_flag: Vec::new(),
        parked: Vec::new(),
        open: OpenList::new(),
        incumbent: start_goal,
        iteration: 0,
        buffer: Vec::new(),
    };

    if start_goal.is_some() {
        return ara.finish(true);
    }

    ara.run(weights.into_iter().peekable(), on_solution, budget)
}

/// How one ARA* iteration ended.
enum Iteration {
    /// The incumbent is within the iteration's weight, or the open list ran empty.
    Done,
    /// The budget stopped the search.
    Stopped,
}

/// The state of one ARA* run.
struct Ara<'a, G: Successors, T: ?Sized, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic, owned for the run.
    heuristic: H,
    /// Every state seen so far.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// Counters for the result.
    stats: Stats,
    /// Heuristic value per node; `None` for proven dead ends, which are never opened.
    h: Vec<Option<G::Cost>>,
    /// The iteration in which each node was last expanded; `0` if never.
    closed_in: Vec<u32>,
    /// Whether each node is on the open list.
    open_flag: Vec<bool>,
    /// Whether each node is parked (ARA*'s `INCONS` list) for the next iteration.
    parked_flag: Vec<bool>,
    /// Nodes improved after their expansion in the current iteration.
    parked: Vec<NodeId>,
    /// The open list, keyed by `den * g + num * h` for the current weight.
    open: OpenList<G::Cost>,
    /// The cheapest goal node found so far.
    incumbent: Option<NodeId>,
    /// The current iteration, starting at `1`.
    iteration: u32,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, T, H> Ara<'_, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Runs one iteration per weight and reports each improvement.
    fn run<I: Iterator<Item = Weight>>(
        mut self,
        mut weights: Peekable<I>,
        mut on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>, G::Cost),
        mut budget: impl Budget,
    ) -> AnytimeGraphResult<G> {
        for start in 0..self.space.len() {
            self.add_node(start);
        }

        let mut reported: Option<G::Cost> = None;
        let mut weight = weights.next().unwrap_or(Weight::ONE);
        loop {
            let last = weights.peek().is_none();
            self.begin_iteration(weight);
            if matches!(
                self.improve_path(weight, last, &mut budget),
                Iteration::Stopped
            ) {
                return self.finish(false);
            }

            if let Some(cost) = self.incumbent_cost() {
                let lower_bound = self.lower_bound(cost);
                if reported.is_none_or(|old| cost.cmp_cost(&old).is_lt()) {
                    reported = Some(cost);
                    on_solution(&self.incumbent_solution(), lower_bound);
                }

                if lower_bound.cmp_cost(&cost).is_ge() {
                    return self.finish(true);
                }
            }

            match weights.next() {
                Some(next) if !last => weight = next,
                _ => return self.finish(true),
            }
        }
    }

    /// Starts an iteration: moves parked nodes back to the open list and re-keys it.
    fn begin_iteration(&mut self, weight: Weight) {
        self.iteration += 1;
        for node in core::mem::take(&mut self.parked) {
            self.parked_flag[node] = false;
            self.open_flag[node] = true;
        }

        self.open.clear();
        for node in 0..self.space.len() {
            if self.open_flag[node] {
                self.push(node, weight);
            }
        }
    }

    /// Expands until the incumbent is within `weight` of the smallest open key.
    fn improve_path(&mut self, weight: Weight, last: bool, budget: &mut impl Budget) -> Iteration {
        loop {
            let (space, open_flag) = (&self.space, &self.open_flag);
            let live =
                |node: NodeId, g: G::Cost| open_flag[node] && g.cmp_cost(&space.g(node)).is_eq();
            let Some((node, key)) = self.open.peek_where(live) else {
                return Iteration::Done;
            };

            let done = self
                .incumbent_cost()
                .is_some_and(|cost| cost.times(weight.denominator()).cmp_cost(&key).is_le());
            if done {
                return Iteration::Done;
            }

            if budget.exhausted(&self.stats) {
                return Iteration::Stopped;
            }

            self.open.pop_where(|_, _| true);
            self.open_flag[node] = false;
            self.expand(node, weight, last);
        }
    }

    /// Generates the successors of `node` and opens or parks the new and improved ones.
    fn expand(&mut self, node: NodeId, weight: Weight, last: bool) {
        self.stats.expanded += 1;
        self.closed_in[node] = self.iteration;
        self.stats.generated +=
            collect_successors(self.graph, self.space.state(node), &mut self.buffer);

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match self.space.reach(state, node, action, cost, true) {
                Reach::New(id) => {
                    self.add_node(id);
                    if self.open_flag[id] {
                        self.push(id, weight);
                    }
                }
                Reach::Improved(id) => self.improved(id, weight, last),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
    }

    /// Records the new node `id`: its heuristic value, whether it is a better goal, and
    /// marks it open unless it is a dead end or a goal.
    fn add_node(&mut self, id: NodeId) {
        let is_goal = self.goal.is_goal(self.space.state(id));
        let h = self.heuristic.estimate(self.space.state(id));

        self.h.push(h);
        self.closed_in.push(0);
        self.open_flag.push(h.is_some() && !is_goal);
        self.parked_flag.push(false);
        if is_goal {
            self.offer_incumbent(id);
        }
    }

    /// Handles a cheaper path to the known node `id`: a better goal, a parked node, or a
    /// node (re)opened for the current iteration.
    fn improved(&mut self, id: NodeId, weight: Weight, last: bool) {
        if self.goal.is_goal(self.space.state(id)) {
            self.offer_incumbent(id);
            return;
        }

        if self.h[id].is_none() {
            return;
        }

        if self.closed_in[id] == self.iteration && !last {
            if !self.parked_flag[id] {
                self.parked_flag[id] = true;
                self.parked.push(id);
            }

            return;
        }

        if self.closed_in[id] != 0 && !self.open_flag[id] {
            self.stats.reopened += 1;
        }

        self.open_flag[id] = true;
        self.push(id, weight);
    }

    /// Makes the goal node `id` the incumbent if it is cheaper than the current one.
    fn offer_incumbent(&mut self, id: NodeId) {
        let cheaper = self
            .incumbent_cost()
            .is_none_or(|cost| self.space.g(id).cmp_cost(&cost).is_lt());
        if cheaper {
            self.incumbent = Some(id);
        }
    }

    /// Pushes the open node `id` with its key under `weight`.
    fn push(&mut self, id: NodeId, weight: Weight) {
        let h = self.h[id].expect("only live nodes are opened");
        let g = self.space.g(id);
        let key = g
            .times(weight.denominator())
            .saturating_add(h.times(weight.numerator()));

        self.open.push(id, key, g);
    }

    /// The path cost of the incumbent, if there is one.
    fn incumbent_cost(&self) -> Option<G::Cost> {
        self.incumbent.map(|id| self.space.g(id))
    }

    /// The smallest `g + h` among open and parked nodes, capped at `cost`.
    ///
    /// Every cheaper solution passes through such a node, so this bounds the optimum from
    /// below when the heuristic is admissible.
    fn lower_bound(&self, cost: G::Cost) -> G::Cost {
        (0..self.space.len())
            .filter(|&id| self.open_flag[id] || self.parked_flag[id])
            .filter_map(|id| self.h[id].map(|h| self.space.g(id) + h))
            .fold(
                cost,
                |bound, f| if f.cmp_cost(&bound).is_lt() { f } else { bound },
            )
    }

    /// The path to the incumbent.
    fn incumbent_solution(&self) -> Solution<G::State, G::Action, G::Cost> {
        self.space
            .solution(self.incumbent.expect("called only with an incumbent"))
    }

    /// Wraps up the run.
    fn finish(self, complete: bool) -> AnytimeGraphResult<G> {
        AnytimeResult {
            solution: self.incumbent.map(|id| self.space.solution(id)),
            complete,
            stats: self.stats,
        }
    }
}
