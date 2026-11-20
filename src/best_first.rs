//! Best-first search: [`astar`], [`dijkstra`], [`weighted_astar`] and [`gbfs`], each also
//! from several start states (`*_multi`).
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
/// use heuristic_search::{Limits, Successors, heuristic_fn, astar};
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
/// let distance = heuristic_fn(|state: &i32| Some(7_i32.abs_diff(*state)));
/// let result = astar(&Line, 0, &|state: &i32| *state == 7, distance, Limits::NONE);
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
    heuristic: H,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let order = Order::Weighted(Weight::ONE);

    Engine::new(graph, goal, heuristic, order).run([start], budget)
}

/// Dijkstra's algorithm (uniform-cost search) from `start` to the cheapest goal.
///
/// A* with the [`Zero`] heuristic: always optimal for non-negative edge costs.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, dijkstra};
///
/// /// Doubling costs 1, incrementing costs 3.
/// struct Numbers;
///
/// impl Successors for Numbers {
///     type State = u32;
///     type Action = char;
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut(char, u32, u32)) {
///         if *n < 100 {
///             emit('*', n * 2, 1);
///             emit('+', n + 1, 3);
///         }
///     }
/// }
///
/// let result = dijkstra(&Numbers, 1, &|n: &u32| *n == 10, Limits::NONE);
/// let solution = result.into_solution().expect("10 is reachable");
///
/// assert_eq!(solution.actions, ['*', '*', '+', '*']);
/// assert_eq!(solution.cost, 6);
/// ```
pub fn dijkstra<G, T>(graph: &G, start: G::State, goal: &T, budget: impl Budget) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    Engine::new(graph, goal, Zero, Order::G).run([start], budget)
}

/// Weighted A*: A* ordered by `g + w * h`, trading solution quality for speed.
///
/// With an admissible `heuristic` and `w >= 1`, the solution costs at most `w` times the
/// optimum. Weight [`Weight::ONE`] is plain A*.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, Weight, weighted_astar};
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
/// let distance = heuristic_fn(|state: &i32| Some(9_i32.abs_diff(*state)));
/// let weight = Weight::new(3, 2);
/// let result = weighted_astar(&Line, 0, &|s: &i32| *s == 9, distance, weight, Limits::NONE);
///
/// assert!(result.solution().is_some_and(|solution| solution.cost <= 9 * 3 / 2));
/// ```
pub fn weighted_astar<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    Engine::new(graph, goal, heuristic, Order::Weighted(weight)).run([start], budget)
}

/// Greedy best-first search: always expands the state with the smallest heuristic value.
///
/// Ignores path cost when ordering and never reopens states, so it is fast but not optimal.
/// The reported [`Solution::cost`](crate::Solution::cost) is the true cost of the returned
/// path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, gbfs};
///
/// /// Cells of a 1-D corridor `0..=20`; each step moves one cell at cost 1.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///         if *cell < 20 {
///             emit(1, cell + 1, 1);
///         }
///     }
/// }
///
/// let distance = heuristic_fn(|cell: &u8| Some(u32::from(15_u8.abs_diff(*cell))));
/// let result = gbfs(&Corridor, 5, &|cell: &u8| *cell == 15, distance, Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.len()), Some(10));
/// assert_eq!(result.stats.expanded, 10);
/// ```
pub fn gbfs<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    Engine::new(graph, goal, heuristic, Order::H).run([start], budget)
}

/// [`astar`] from several start states at once, to the cheapest goal from any of them.
///
/// Every start has path cost zero. Repeated starts count once, an empty set of starts is
/// [`Outcome::Unsolvable`], and a start that is a goal is returned as a zero-cost path
/// without expanding anything. The heuristic estimates the distance to the goal, so it
/// needs no change for several starts.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, astar_multi};
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
/// let distance = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let result = astar_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, distance, Limits::NONE);
/// let solution = result.into_solution().expect("8 is reachable");
///
/// assert_eq!(solution.states, [6, 7, 8]);
/// assert_eq!(solution.cost, 2);
/// ```
pub fn astar_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let order = Order::Weighted(Weight::ONE);

    Engine::new(graph, goal, heuristic, order).run(starts, budget)
}

/// [`dijkstra`] from several start states at once, to the cheapest goal from any of them.
///
/// Start handling as in [`astar_multi`].
pub fn dijkstra_multi<G, T>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    Engine::new(graph, goal, Zero, Order::G).run(starts, budget)
}

/// [`weighted_astar`] from several start states at once.
///
/// Start handling as in [`astar_multi`]; the cost bound is relative to the cheapest goal
/// from any start.
pub fn weighted_astar_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    Engine::new(graph, goal, heuristic, Order::Weighted(weight)).run(starts, budget)
}

/// [`gbfs`] from several start states at once.
///
/// Start handling as in [`astar_multi`].
pub fn gbfs_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    Engine::new(graph, goal, heuristic, Order::H).run(starts, budget)
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
    ///
    /// Saturates for integer costs, so keys near the maximum of the cost type tie instead of
    /// overflowing.
    fn key<C: Cost>(self, g: C, h: C) -> C {
        match self {
            Self::G => g,
            Self::Weighted(weight) => g
                .times(weight.denominator())
                .saturating_add(h.times(weight.numerator())),
            Self::H => h,
        }
    }

    /// Returns `true` if a cheaper path to a known state replaces the old one.
    fn reopens(self) -> bool {
        !matches!(self, Self::H)
    }
}

/// One best-first search run over a graph, a goal and a heuristic.
struct Engine<'a, G: Successors, T: ?Sized, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic, owned for the run (pass `&mut h` to lend one).
    heuristic: H,
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
    H: Heuristic<G::State, G::Cost>,
{
    /// Prepares a run.
    fn new(graph: &'a G, goal: &'a T, heuristic: H, order: Order) -> Self {
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

    /// Searches from `starts` until a goal is expanded, the open list runs empty or
    /// `budget` is used up. A start that is a goal ends the search before any expansion.
    fn run(
        mut self,
        starts: impl IntoIterator<Item = G::State>,
        mut budget: impl Budget,
    ) -> GraphResult<G> {
        let (mut space, start_goal) = SearchSpace::seed(starts, self.goal);
        if let Some(found) = start_goal {
            return self.finish(Outcome::Found(space.solution(found)));
        }

        let mut open = OpenList::new();
        for start in 0..space.len() {
            self.open_if_alive(&space, &mut open, start);
        }

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
