//! Best-first search: [`a_star`], [`dijkstra`], [`weighted_a_star`] and [`gbfs`], each also
//! from several start states (`*_multi`) and in a reusable [`Workspace`] (`*_in`).
//!
//! All four run the same engine and differ only in the open-list key and in whether a
//! cheaper path to a known state is taken up (reopening):
//!
//! | Algorithm         | Key                         | Reopens | Optimal              |
//! | ----------------- | --------------------------- | ------- | -------------------- |
//! | [`dijkstra`]      | `g`                         | yes     | yes                  |
//! | [`a_star`]         | `g + h`                     | yes     | if `h` is admissible |
//! | [`weighted_a_star`]| `g + w * h`                 | yes     | within factor `w`    |
//! | [`gbfs`]          | `h`                         | no      | no                   |
//!
//! Ties are broken towards larger `g`, then first in, first out.
//!
//! The engine runs in a [`Workspace`]; the `*_in` variants take one from the caller and
//! keep its allocations for the next search. [`AStarSearch`] keeps its own and resumes
//! after a budget stop: the next entry leaves the open list only when it is expanded.
//!
//! ```mermaid
//! flowchart TD
//!     peek[peek cheapest live entry] -->|none| unsolvable[Unsolvable]
//!     peek --> goal{goal?}
//!     goal -->|yes| found[Found: extract path]
//!     goal -->|no| budget{budget exhausted?}
//!     budget -->|yes| limit[LimitReached]
//!     budget -->|no| expand[pop it, generate successors, push new or improved]
//!     expand --> peek
//! ```

mod resumable;

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic, Zero};

use crate::cost::Weight;
use crate::open_list::OpenList;
use crate::outcome::{Outcome, SearchResult, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

pub use resumable::AStarSearch;

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
/// use heuristic_search::{Limits, Successors, heuristic_fn, a_star};
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
/// let result = a_star(&Line, 0, &|state: &i32| *state == 7, distance, Limits::NONE);
/// let solution = result.solution().expect("7 is reachable");
///
/// assert_eq!(solution.cost, 7);
/// assert_eq!(solution.actions, [1; 7]);
/// assert_eq!(result.stats.expanded, 7);
/// ```
pub fn a_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    a_star_in(&mut Workspace::new(), graph, start, goal, heuristic, budget)
}

/// A* from `start` that gives up once no path can cost less than `bound`: it stops with
/// [`Outcome::Unsolvable`] as soon as the smallest `f = g + h` on the open list reaches
/// `bound`, so a returned path always costs less than `bound`. Runs without a budget.
///
/// With a consistent heuristic, `f` never falls along a path, so no path found later could
/// cost less than `bound` either.
#[cfg(feature = "float")]
pub(crate) fn a_star_below<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    bound: G::Cost,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let mut work = Workspace::new();
    let mut engine = Engine::new(
        graph,
        goal,
        heuristic,
        Order::Weighted(Weight::ONE),
        &mut work,
    );
    engine.bound = Some(bound);
    engine.run([start], crate::outcome::Limits::NONE)
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
pub fn dijkstra<G, T>(
    graph: &G,
    start: G::State,
    goal: &T,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    dijkstra_in(&mut Workspace::new(), graph, start, goal, budget)
}

/// Weighted A*: A* ordered by `g + w * h`, trading solution quality for speed.
///
/// With an admissible `heuristic` and `w >= 1`, the solution costs at most `w` times the
/// optimum. Weight [`Weight::ONE`] is plain A*.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, Weight, weighted_a_star};
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
/// let result = weighted_a_star(&Line, 0, &|s: &i32| *s == 9, distance, weight, Limits::NONE);
///
/// assert!(result.solution().is_some_and(|solution| solution.cost <= 9 * 3 / 2));
/// ```
pub fn weighted_a_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    weighted_a_star_in(
        &mut Workspace::new(),
        graph,
        start,
        goal,
        heuristic,
        weight,
        budget,
    )
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
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    gbfs_in(&mut Workspace::new(), graph, start, goal, heuristic, budget)
}

/// [`a_star`] from several start states at once, to the cheapest goal from any of them.
///
/// Every start has path cost zero. Repeated starts count once, an empty set of starts is
/// [`Outcome::Unsolvable`], and a start that is a goal is returned as a zero-cost path
/// without expanding anything. The heuristic estimates the distance to the goal, so it
/// needs no change for several starts.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, a_star_multi};
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
/// let result = a_star_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, distance, Limits::NONE);
/// let solution = result.into_solution().expect("8 is reachable");
///
/// assert_eq!(solution.states, [6, 7, 8]);
/// assert_eq!(solution.cost, 2);
/// ```
pub fn a_star_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    a_star_multi_in(
        &mut Workspace::new(),
        graph,
        starts,
        goal,
        heuristic,
        budget,
    )
}

/// [`dijkstra`] from several start states at once, to the cheapest goal from any of them.
///
/// Start handling as in [`a_star_multi`].
pub fn dijkstra_multi<G, T>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    dijkstra_multi_in(&mut Workspace::new(), graph, starts, goal, budget)
}

/// [`weighted_a_star`] from several start states at once.
///
/// Start handling as in [`a_star_multi`]; the cost bound is relative to the cheapest goal
/// from any start.
pub fn weighted_a_star_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    weighted_a_star_multi_in(
        &mut Workspace::new(),
        graph,
        starts,
        goal,
        heuristic,
        weight,
        budget,
    )
}

/// [`gbfs`] from several start states at once.
///
/// Start handling as in [`a_star_multi`].
pub fn gbfs_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    gbfs_multi_in(
        &mut Workspace::new(),
        graph,
        starts,
        goal,
        heuristic,
        budget,
    )
}

/// [`a_star`] in a caller's [`Workspace`], which keeps its allocations for the next search.
///
/// Returns the same result as [`a_star`].
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Workspace, a_star_in, heuristic_fn};
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
/// let mut workspace = Workspace::new();
/// for (start, target) in [(0, 7_i32), (3, -2), (5, 5)] {
///     let distance = heuristic_fn(|state: &i32| Some(target.abs_diff(*state)));
///     let goal = |state: &i32| *state == target;
///     let result = a_star_in(&mut workspace, &Line, start, &goal, distance, Limits::NONE);
///     assert_eq!(result.solution().map(|s| s.cost), Some(start.abs_diff(target)));
/// }
/// ```
pub fn a_star_in<G, T, H>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let order = Order::Weighted(Weight::ONE);

    Engine::new(graph, goal, heuristic, order, workspace).run([start], budget)
}

/// [`dijkstra`] in a caller's [`Workspace`], which keeps its allocations for the next
/// search.
///
/// Returns the same result as [`dijkstra`]. See [`Workspace`] for an example.
pub fn dijkstra_in<G, T>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    start: G::State,
    goal: &T,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    Engine::new(graph, goal, Zero, Order::G, workspace).run([start], budget)
}

/// [`weighted_a_star`] in a caller's [`Workspace`], which keeps its allocations for the
/// next search.
///
/// Returns the same result as [`weighted_a_star`].
pub fn weighted_a_star_in<G, T, H>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let order = Order::Weighted(weight);

    Engine::new(graph, goal, heuristic, order, workspace).run([start], budget)
}

/// [`gbfs`] in a caller's [`Workspace`], which keeps its allocations for the next search.
///
/// Returns the same result as [`gbfs`].
pub fn gbfs_in<G, T, H>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    Engine::new(graph, goal, heuristic, Order::H, workspace).run([start], budget)
}

/// [`a_star_multi`] in a caller's [`Workspace`], which keeps its allocations for the next
/// search.
///
/// Returns the same result as [`a_star_multi`].
pub fn a_star_multi_in<G, T, H>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let order = Order::Weighted(Weight::ONE);

    Engine::new(graph, goal, heuristic, order, workspace).run(starts, budget)
}

/// [`dijkstra_multi`] in a caller's [`Workspace`], which keeps its allocations for the
/// next search.
///
/// Returns the same result as [`dijkstra_multi`].
pub fn dijkstra_multi_in<G, T>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    Engine::new(graph, goal, Zero, Order::G, workspace).run(starts, budget)
}

/// [`weighted_a_star_multi`] in a caller's [`Workspace`], which keeps its allocations for
/// the next search.
///
/// Returns the same result as [`weighted_a_star_multi`].
pub fn weighted_a_star_multi_in<G, T, H>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let order = Order::Weighted(weight);

    Engine::new(graph, goal, heuristic, order, workspace).run(starts, budget)
}

/// [`gbfs_multi`] in a caller's [`Workspace`], which keeps its allocations for the next
/// search.
///
/// Returns the same result as [`gbfs_multi`].
pub fn gbfs_multi_in<G, T, H>(
    workspace: &mut Workspace<G::State, G::Action, G::Cost>,
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    Engine::new(graph, goal, heuristic, Order::H, workspace).run(starts, budget)
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
                .saturating_times(weight.denominator())
                .saturating_add(h.saturating_times(weight.numerator())),
            Self::H => h,
        }
    }

    /// Returns `true` if a cheaper path to a known state replaces the old one.
    fn reopens(self) -> bool {
        !matches!(self, Self::H)
    }
}

/// The memory of a best-first search, kept by the caller across searches so repeated
/// queries reuse its allocations: the search space with its hash index and parent links,
/// the open list, the per-node heuristic values and closed flags, and the successor buffer.
///
/// Pass it to the `*_in` searches ([`a_star_in`], [`dijkstra_in`] and so on). A search
/// clears the workspace when it starts, not when it ends, and returns the same result as
/// the search without a workspace. Keep one workspace per thread: a search borrows it
/// mutably for its whole run.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Workspace, dijkstra_in};
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
/// let mut workspace = Workspace::new();
/// for target in [3, 8, 5] {
///     let result = dijkstra_in(&mut workspace, &Up, 0, &|n: &u32| *n == target, Limits::NONE);
///     assert_eq!(result.solution().map(|s| s.cost), Some(target));
/// }
/// ```
pub struct Workspace<S, A, C> {
    /// Every state seen by the current search, with its path cost and parent link.
    space: SearchSpace<S, A, C>,
    /// The open list.
    open: OpenList<C>,
    /// Heuristic value per node, by [`NodeId`]; computed once when the node is created.
    /// `None` for proven dead ends, which are never opened.
    h: Vec<Option<C>>,
    /// Whether each node, by [`NodeId`], has been expanded at least once.
    closed: Vec<bool>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(A, S, C)>,
}

impl<S: Hash + Eq, A, C: Cost> Workspace<S, A, C> {
    /// An empty workspace. Allocates nothing until the first search.
    #[must_use]
    pub fn new() -> Self {
        Self {
            space: SearchSpace::new(),
            open: OpenList::new(),
            h: Vec::new(),
            closed: Vec::new(),
            buffer: Vec::new(),
        }
    }

    /// Forgets the open list and the side tables, keeping their allocations. The search
    /// space is cleared when it is reseeded.
    fn clear(&mut self) {
        self.open.clear();
        self.h.clear();
        self.closed.clear();
        self.buffer.clear();
    }
}

impl<S, A, C> Workspace<S, A, C> {
    /// The search space of the last search.
    pub(crate) fn space(&self) -> &SearchSpace<S, A, C> {
        &self.space
    }

    /// Returns `true` if the last search expanded node `id`.
    pub(crate) fn is_closed(&self, id: NodeId) -> bool {
        self.closed[id]
    }
}

impl<S: Hash + Eq, A, C: Cost> Default for Workspace<S, A, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// One best-first search run over a graph, a goal and a heuristic, in a [`Workspace`].
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
    /// The search space, open list and side tables of the run.
    work: &'a mut Workspace<G::State, G::Action, G::Cost>,
    /// The open-list key at which the run gives up as unsolvable, if any.
    bound: Option<G::Cost>,
}

impl<'a, G, T, H> Engine<'a, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Prepares a run in `work`.
    fn new(
        graph: &'a G,
        goal: &'a T,
        heuristic: H,
        order: Order,
        work: &'a mut Workspace<G::State, G::Action, G::Cost>,
    ) -> Self {
        Self {
            graph,
            goal,
            heuristic,
            order,
            stats: Stats::default(),
            work,
            bound: None,
        }
    }

    /// Searches from `starts` until a goal is expanded, the open list runs empty or
    /// `budget` is used up. A start that is a goal ends the search before any expansion.
    fn run(
        mut self,
        starts: impl IntoIterator<Item = G::State>,
        budget: impl Budget<Stats>,
    ) -> GraphResult<G> {
        if let Some(found) = self.seed(starts) {
            let solution = self.work.space.solution(found);
            return self.finish(Outcome::Found(solution));
        }

        let outcome = self.advance(budget).unwrap_or(Outcome::LimitReached);
        self.finish(outcome)
    }

    /// Clears the workspace and opens `starts`. Returns the node of the first start that
    /// is a goal, if any; then nothing is opened.
    fn seed(&mut self, starts: impl IntoIterator<Item = G::State>) -> Option<NodeId> {
        self.work.clear();
        if let Some(found) = self.work.space.reseed(starts, self.goal) {
            return Some(found);
        }

        for start in 0..self.work.space.len() {
            self.open_if_alive(start);
        }

        None
    }

    /// Expands until a goal is next in the open list or the open list runs empty, and
    /// returns that outcome; returns `None` once `budget` is used up.
    ///
    /// The next node leaves the open list only when it is expanded, so a run stopped by
    /// its budget continues with the same node on the next call.
    fn advance(
        &mut self,
        mut budget: impl Budget<Stats>,
    ) -> Option<Outcome<G::State, G::Action, G::Cost>> {
        loop {
            let work = &mut *self.work;
            let space = &work.space;
            let Some((node, key)) = work.open.peek_where(|id, g| g.cmp(&space.g(id)).is_eq())
            else {
                return Some(Outcome::Unsolvable);
            };

            if self.bound.is_some_and(|bound| key >= bound) {
                return Some(Outcome::Unsolvable);
            }

            if self.goal.is_goal(space.state(node)) {
                return Some(Outcome::Found(space.solution(node)));
            }

            if budget.exhausted(&self.stats) {
                return None;
            }

            // The top entry was just found live, so this pops `node`.
            work.open.pop_where(|_, _| true);
            self.expand(node);
        }
    }

    /// Generates the successors of `node` and opens the new and improved ones.
    fn expand(&mut self, node: NodeId) {
        self.stats.expanded += 1;
        let work = &mut *self.work;
        work.closed[node] = true;
        self.stats.generated +=
            collect_successors(self.graph, work.space.state(node), &mut work.buffer);

        let reopens = self.order.reopens();
        let mut buffer = core::mem::take(&mut work.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match self.work.space.reach(state, node, action, cost, reopens) {
                Reach::New(id) => self.open_if_alive(id),
                Reach::Improved(id) => self.reopen(id),
                Reach::Stale => {}
            }
        }

        self.work.buffer = buffer;
    }

    /// Evaluates the new node `id` and opens it, unless the heuristic proves it a dead end.
    ///
    /// Dead ends keep their node, so they are recognised as duplicates later, but never
    /// enter the open list.
    fn open_if_alive(&mut self, id: NodeId) {
        let work = &mut *self.work;
        let h = self.heuristic.estimate(work.space.state(id));
        debug_assert_eq!(id, work.h.len(), "nodes are recorded in creation order");
        work.h.push(h);
        work.closed.push(false);

        if let Some(h) = h {
            let g = work.space.g(id);
            work.open.push(id, self.order.key(g, h), g);
        }
    }

    /// Pushes the known node `id` again after its path cost improved. Dead ends stay out.
    fn reopen(&mut self, id: NodeId) {
        let work = &mut *self.work;
        let Some(h) = work.h[id] else {
            return;
        };

        if work.closed[id] {
            self.stats.reopened += 1;
        }

        let g = work.space.g(id);
        work.open.push(id, self.order.key(g, h), g);
    }

    /// Wraps up the run with `outcome`.
    fn finish(self, outcome: Outcome<G::State, G::Action, G::Cost>) -> GraphResult<G> {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
