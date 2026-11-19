//! Best-first search: [`astar`], [`dijkstra`], [`weighted_astar`] and [`gbfs`].
//!
//! All four run the same engine and differ only in the open-list key and in whether a
//! cheaper path to a known state is taken up (reopening):
//!
//! | Algorithm         | Key                         | Reopens | Optimal              |
//! | ----------------- | --------------------------- | ------- | -------------------- |
//! | [`dijkstra`]      | `g`                         | yes     | yes                  |
//! | [`astar`]         | `g + h`                     | yes     | if `h` is admissible |
//! | [`weighted_astar`]| `g + w * h`                 | yes     | within factor `w`    |
//! | [`gbfs`]          | `h`                         | no      | no                   |
//!
//! Ties are broken towards larger `g`, then first in, first out.
//!
//! ```mermaid
//! flowchart TD
//!     pop[pop cheapest live entry] -->|none| unsolvable[Unsolvable]
//!     pop --> goal{goal?}
//!     goal -->|yes| found[Found: extract path]
//!     goal -->|no| budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| expand[generate successors, push new or improved]
//!     expand --> pop
//! ```

mod open_list;

use alloc::vec::Vec;
use core::hash::Hash;

use crate::cost::{Cost, Weight};
use crate::outcome::{Budget, Outcome, SearchResult, Stats};
use crate::problem::{Goal, Heuristic, Successors, Zero, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};
use open_list::OpenList;

/// The result type of searches over graph `G`.
pub type GraphResult<G> =
    SearchResult<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;

/// A* search from `start` to the first state that satisfies `goal`.
///
/// Expands states in order of `f = g + h`. With an admissible `heuristic` the solution is
/// optimal; inconsistent heuristics are fine, because closed states are reopened when a
/// cheaper path to them turns up (counted in [`Stats::reopened`]).
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, astar};
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
/// let distance = |state: &i32| Some(7_i32.abs_diff(*state));
/// let result = astar(&Line, 0, &|state: &i32| *state == 7, &distance, Limits::NONE);
/// let solution = result.solution().expect("7 is reachable");
///
/// assert_eq!(solution.cost, 7);
/// assert_eq!(solution.actions, [1; 7]);
/// assert_eq!(result.stats.expanded, 7);
/// ```
pub fn astar<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: &H,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost> + ?Sized,
{
    let order = Order::Weighted(Weight::ONE);

    Engine::new(graph, goal, heuristic, order).run(start, budget)
}

/// Dijkstra's algorithm (uniform-cost search) from `start` to the cheapest goal.
///
/// A* with the [`Zero`] heuristic: always optimal for non-negative edge costs.
pub fn dijkstra<G, T>(graph: &G, start: G::State, goal: &T, budget: impl Budget) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    Engine::new(graph, goal, &Zero, Order::G).run(start, budget)
}

/// Weighted A*: A* ordered by `g + w * h`, trading solution quality for speed.
///
/// With an admissible `heuristic` and `w >= 1`, the solution costs at most `w` times the
/// optimum. Weight [`Weight::ONE`] is plain A*.
pub fn weighted_astar<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: &H,
    weight: Weight,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost> + ?Sized,
{
    Engine::new(graph, goal, heuristic, Order::Weighted(weight)).run(start, budget)
}

/// Greedy best-first search: always expands the state with the smallest heuristic value.
///
/// Ignores path cost when ordering and never reopens states, so it is fast but not optimal.
/// The reported [`Solution::cost`](crate::Solution::cost) is the true cost of the returned
/// path.
pub fn gbfs<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: &H,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost> + ?Sized,
{
    Engine::new(graph, goal, heuristic, Order::H).run(start, budget)
}

/// How the engine orders its open list.
#[derive(Clone, Copy)]
enum Order {
    /// By path cost `g` only (Dijkstra). The heuristic is still asked, for dead ends.
    G,
    /// By `denominator * g + numerator * h` ([`Weight`]), which sorts like `g + w * h`.
    Weighted(Weight),
    /// By heuristic value `h` only (greedy best-first). Never reopens.
    H,
}

impl Order {
    /// The open-list key of a node with path cost `g` and heuristic value `h`.
    fn key<C: Cost>(self, g: C, h: C) -> C {
        match self {
            Self::G => g,
            Self::Weighted(weight) => g.times(weight.denominator()) + h.times(weight.numerator()),
            Self::H => h,
        }
    }

    /// Returns `true` if a cheaper path to a known state replaces the old one.
    fn reopens(self) -> bool {
        !matches!(self, Self::H)
    }
}

/// One best-first search run over a graph, a goal and a heuristic.
struct Engine<'a, G: Successors, T: ?Sized, H: ?Sized> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic.
    heuristic: &'a H,
    /// The open-list order.
    order: Order,
    /// Counters for the result.
    stats: Stats,
    /// Heuristic value per node, by [`NodeId`]; computed once when the node is created.
    /// `None` for proven dead ends, which are never opened.
    h: Vec<Option<G::Cost>>,
    /// Whether each node, by [`NodeId`], has been expanded at least once.
    closed: Vec<bool>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<'a, G, T, H> Engine<'a, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost> + ?Sized,
{
    /// Prepares a run.
    fn new(graph: &'a G, goal: &'a T, heuristic: &'a H, order: Order) -> Self {
        Self {
            graph,
            goal,
            heuristic,
            order,
            stats: Stats::default(),
            h: Vec::new(),
            closed: Vec::new(),
            buffer: Vec::new(),
        }
    }

    /// Searches from `start` until a goal is expanded, the open list runs empty or `budget`
    /// is used up.
    fn run(mut self, start: G::State, mut budget: impl Budget) -> GraphResult<G> {
        let mut space = SearchSpace::new(start);
        let mut open = OpenList::new();
        self.open_if_alive(&space, &mut open, 0);

        while let Some(node) = open.pop_live(|id| space.g(id)) {
            if self.goal.is_goal(space.state(node)) {
                return self.finish(Outcome::Found(space.solution(node)));
            }

            if budget.exhausted(&self.stats) {
                return self.finish(Outcome::LimitReached);
            }

            self.expand(&mut space, &mut open, node);
        }

        self.finish(Outcome::Unsolvable)
    }

    /// Generates the successors of `node` and opens the new and improved ones.
    fn expand(
        &mut self,
        space: &mut SearchSpace<G::State, G::Action, G::Cost>,
        open: &mut OpenList<G::Cost>,
        node: NodeId,
    ) {
        self.stats.expanded += 1;
        self.closed[node] = true;
        self.stats.generated += collect_successors(self.graph, space.state(node), &mut self.buffer);

        let reopens = self.order.reopens();
        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match space.reach(state, node, action, cost, reopens) {
                Reach::New(id) => self.open_if_alive(space, open, id),
                Reach::Improved(id) => self.reopen(space, open, id),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
    }

    /// Evaluates the new node `id` and opens it, unless the heuristic proves it a dead end.
    ///
    /// Dead ends keep their node, so they are recognised as duplicates later, but never
    /// enter the open list.
    fn open_if_alive(
        &mut self,
        space: &SearchSpace<G::State, G::Action, G::Cost>,
        open: &mut OpenList<G::Cost>,
        id: NodeId,
    ) {
        let h = self.heuristic.estimate(space.state(id));
        self.record(id, h);

        if let Some(h) = h {
            let g = space.g(id);
            open.push(id, self.order.key(g, h), g);
        }
    }

    /// Records the heuristic value `h` of the new node `id`, which starts out not closed.
    fn record(&mut self, id: NodeId, h: Option<G::Cost>) {
        debug_assert_eq!(id, self.h.len(), "nodes are recorded in creation order");

        self.h.push(h);
        self.closed.push(false);
    }

    /// Pushes the known node `id` again after its path cost improved. Dead ends stay out.
    fn reopen(
        &mut self,
        space: &SearchSpace<G::State, G::Action, G::Cost>,
        open: &mut OpenList<G::Cost>,
        id: NodeId,
    ) {
        let Some(h) = self.h[id] else {
            return;
        };

        if self.closed[id] {
            self.stats.reopened += 1;
        }

        let g = space.g(id);
        open.push(id, self.order.key(g, h), g);
    }

    /// Wraps up the run with `outcome`.
    fn finish(self, outcome: Outcome<G::State, G::Action, G::Cost>) -> GraphResult<G> {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
