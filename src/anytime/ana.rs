//! Anytime Nonparametric A* ([`ana_star`], van den Berg, Shah, Huang and Goldberg 2011).

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::anytime::{AnytimeGraphResult, AnytimeResult};
use crate::cost::CostExt;
use crate::open_list::OpenList;
use crate::outcome::{Solution, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// Anytime Nonparametric A* (ANA*): anytime search without a weight schedule.
///
/// With an incumbent of cost `G`, ANA* expands the open state with the largest
/// `e = (G - g) / h`, the state whose path could be completed below `G` with the most
/// slack, and prunes states with `g + h >= G`. Every goal it reaches is cheaper than the
/// one before; `on_solution` is called with each new solution and a lower bound on the
/// optimal cost (the smallest `g + h` left open). Before the first solution (`G` infinite)
/// it orders by `h` like greedy best-first search, ties towards larger `g`.
///
/// Closed states are reopened when a cheaper path to them turns up, so the final solution
/// is optimal for any admissible `heuristic`, consistent or not. Goals are tested when they
/// are expanded. The priority is computed in `f64` (see [`CostExt::to_f64`](crate::CostExt::to_f64)); rounding only
/// changes the expansion order, never the returned cost.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, ana_star, heuristic_fn};
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
/// let mut bounds = Vec::new();
/// let result = ana_star(
///     &Line,
///     0,
///     &|state: &i32| *state == 5,
///     distance,
///     |solution, lower_bound| bounds.push((solution.cost, lower_bound)),
///     Limits::NONE,
/// );
///
/// assert!(result.complete);
/// assert_eq!(result.solution.map(|solution| solution.cost), Some(5));
/// assert_eq!(bounds, [(5, 5)]);
/// ```
pub fn ana_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>, G::Cost),
    budget: impl Budget<Stats>,
) -> AnytimeGraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    G::Cost: CostExt,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let (space, start_goal) = SearchSpace::seed([start], goal);
    let ana = Ana {
        graph,
        goal,
        heuristic,
        space,
        stats: Stats::default(),
        h: Vec::new(),
        closed: Vec::new(),
        open_flag: Vec::new(),
        open: OpenList::new(),
        incumbent: start_goal,
        best: start_goal.map(|_| G::Cost::ZERO),
        buffer: Vec::new(),
    };

    if start_goal.is_some() {
        return ana.finish(true);
    }

    ana.run(on_solution, budget)
}

/// The state of one ANA* run.
struct Ana<'a, G: Successors, T: ?Sized, H> {
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
    /// Whether each node has been expanded at least once.
    closed: Vec<bool>,
    /// Whether each node is on the open list.
    open_flag: Vec<bool>,
    /// The open list, keyed by `-e` (or by `h` before the first solution).
    open: OpenList<Priority, G::Cost>,
    /// The cheapest goal node found so far.
    incumbent: Option<NodeId>,
    /// `G`, the cost of the incumbent when it was found.
    ///
    /// Kept apart from the incumbent's `g`, which drops as soon as a cheaper path to the
    /// same goal node is generated, before that path is expanded and reported.
    best: Option<G::Cost>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, T, H> Ana<'_, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    G::Cost: CostExt,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Expands until the open list is empty or the budget runs out.
    fn run(
        mut self,
        mut on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>, G::Cost),
        mut budget: impl Budget<Stats>,
    ) -> AnytimeGraphResult<G> {
        for start in 0..self.space.len() {
            self.add_node(start);
        }

        while let Some(node) = self.pop() {
            if self.goal.is_goal(self.space.state(node)) {
                self.incumbent = Some(node);
                self.best = Some(self.space.g(node));
                self.prune();
                on_solution(&self.space.solution(node), self.lower_bound());
                continue;
            }

            if budget.exhausted(&self.stats) {
                return self.finish(false);
            }

            self.expand(node);
        }

        self.finish(true)
    }

    /// Pops the open node with the best priority.
    fn pop(&mut self) -> Option<NodeId> {
        let (space, open_flag) = (&self.space, &self.open_flag);
        let live = |node: NodeId, g: G::Cost| open_flag[node] && g.cmp(&space.g(node)).is_eq();
        let (node, _) = self.open.pop_where(live)?;
        self.open_flag[node] = false;

        Some(node)
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
                Reach::Improved(id) => self.open_if_promising(id),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
    }

    /// Evaluates the new node `id` and opens it if it is promising.
    fn add_node(&mut self, id: NodeId) {
        let h = self.heuristic.estimate(self.space.state(id));

        self.h.push(h);
        self.closed.push(false);
        self.open_flag.push(false);
        self.open_if_promising(id);
    }

    /// Opens `id` (again) unless it is a dead end or cannot beat the incumbent.
    fn open_if_promising(&mut self, id: NodeId) {
        let Some(key) = self.key(id) else {
            return;
        };

        if self.closed[id] && !self.open_flag[id] {
            self.stats.reopened += 1;
        }

        self.open_flag[id] = true;
        self.open.push(id, key, self.space.g(id));
    }

    /// The open-list key of `id`, or `None` if it must stay closed: a dead end, or a node
    /// with `g + h >= G`.
    fn key(&self, id: NodeId) -> Option<Priority> {
        let h = self.h[id]?;
        let g = self.space.g(id);
        let Some(incumbent) = self.best else {
            return Some(Priority(h.to_f64()));
        };

        if (g + h).cmp(&incumbent).is_ge() {
            return None;
        }

        let slack = incumbent.saturating_sub(g).to_f64();
        let h = h.to_f64();

        Some(Priority(if h > 0.0 {
            -slack / h
        } else {
            f64::NEG_INFINITY
        }))
    }

    /// Re-keys the open list for the new incumbent and drops nodes that cannot beat it.
    fn prune(&mut self) {
        self.open.clear();
        for id in 0..self.space.len() {
            if !self.open_flag[id] {
                continue;
            }

            match self.key(id) {
                Some(key) => self.open.push(id, key, self.space.g(id)),
                None => self.open_flag[id] = false,
            }
        }
    }

    /// The smallest `g + h` among open nodes, capped at the incumbent's cost.
    fn lower_bound(&self) -> G::Cost {
        let incumbent = self.best.expect("called with an incumbent");

        (0..self.space.len())
            .filter(|&id| self.open_flag[id])
            .filter_map(|id| self.h[id].map(|h| self.space.g(id) + h))
            .fold(
                incumbent,
                |bound, f| {
                    if f.cmp(&bound).is_lt() { f } else { bound }
                },
            )
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

/// An open-list key of ANA*: `h` before the first solution, `-e` after it.
///
/// Ordered by [`f64::total_cmp`]. The keys are never NaN (`h > 0` guards the division and
/// costs are not NaN), but may be `-inf` for `h = 0`, so the `NonNegF64` cost type
/// cannot hold them.
#[derive(Debug, Clone, Copy)]
struct Priority(f64);

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Priority {}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
