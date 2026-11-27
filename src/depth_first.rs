//! Depth-first search: [`dfs`], [`depth_limited_dfs`], [`iddfs`], [`ida_star`] and
//! [`ida_star_tt`].
//!
//! | Algorithm             | Memory            | Duplicates            | Returns                |
//! | --------------------- | ----------------- | --------------------- | ---------------------- |
//! | [`dfs`]               | every state seen  | closed set            | some path              |
//! | [`depth_limited_dfs`] | the current path  | cycles on the path    | a path of `<= limit`   |
//! | [`iddfs`]             | the current path  | cycles on the path    | fewest edges           |
//! | [`ida_star`]          | the current path  | cycles on the path    | cheapest (admissible)  |
//! | [`ida_star_tt`]       | path + `capacity` | path, bounded table   | cheapest (admissible)  |
//!
//! The last four are tree searches: they keep only the current path and skip a successor
//! already on it, so they need nothing but `Eq` on states but may revisit a state reached
//! along different paths ([`ida_star_tt`] bounds that with a transposition table, at the
//! price of `Hash`). All of them are iterative, so deep paths cannot overflow the call
//! stack. [`Stats::expanded`] counts every expansion, also repeated ones across iterations.

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::GraphResult;
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{Reach, SearchSpace};

mod transposition;

use transposition::{NoTable, Table, Transpositions};

/// Graph depth-first search: follows the newest successor first and finds some path, not
/// necessarily a short or cheap one.
///
/// Keeps every state it has seen, so each state is expanded at most once and the search
/// is complete on finite graphs. Goals are tested when they are generated. Successors are
/// tried in the order the graph reports them.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, dfs};
///
/// /// Numbers `0..10`; each steps to the next one and back to `0`.
/// struct Ring;
///
/// impl Successors for Ring {
///     type State = u8;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u8, mut emit: impl FnMut((), u8, u32)) {
///         emit((), (n + 1) % 10, 1);
///         emit((), 0, 1);
///     }
/// }
///
/// let result = dfs(&Ring, 0, &|n: &u8| *n == 4, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![0, 1, 2, 3, 4]));
/// ```
pub fn dfs<G, T>(graph: &G, start: G::State, goal: &T, budget: impl Budget<Stats>) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    dfs_multi(graph, [start], goal, budget)
}

/// [`dfs`] from several start states at once.
///
/// Searches the starts in order, sharing one closed set: a start reached from an earlier
/// start is not searched again.
///
/// Repeated starts count once, an empty set of starts finds nothing, and a start that is a
/// goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, dfs_multi};
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
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let result = dfs_multi(&Up, [9, 6, 3], &|n: &u32| *n == 8, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![6, 7, 8]));
/// ```
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
pub fn dfs_multi<G, T>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let (mut space, start_goal) = SearchSpace::seed(starts, goal);
    let mut stats = Stats::default();
    let finish = |outcome, stats| SearchResult { outcome, stats };
    if let Some(found) = start_goal {
        return finish(Outcome::Found(space.solution(found)), stats);
    }

    let mut stack: Vec<_> = (0..space.len()).rev().collect();
    let mut buffer = Vec::new();
    while let Some(node) = stack.pop() {
        if budget.exhausted(&stats) {
            return finish(Outcome::LimitReached, stats);
        }

        stats.expanded += 1;
        stats.generated += collect_successors(graph, space.state(node), &mut buffer);

        // Pushed in reverse, so the first reported successor is expanded first.
        let first_new = stack.len();
        for (action, state, cost) in buffer.drain(..) {
            let Reach::New(id) = space.reach(state, node, action, cost, false) else {
                continue;
            };

            if goal.is_goal(space.state(id)) {
                return finish(Outcome::Found(space.solution(id)), stats);
            }

            stack.push(id);
        }

        stack[first_new..].reverse();
    }

    finish(Outcome::Unsolvable, stats)
}

/// Depth-limited depth-first tree search: a path of at most `limit` edges.
///
/// Skips successors already on the current path. If no goal is found, the outcome is
/// [`Outcome::Unsolvable`] only if the limit never cut a branch off; otherwise it is
/// [`Outcome::Cutoff`], because a goal may lie deeper. [`Outcome::LimitReached`] means the
/// budget stopped it. Goals are tested when they are generated.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Outcome, Successors, depth_limited_dfs};
///
/// /// Counting up by one.
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
/// let is_five = |n: &u32| *n == 5;
/// assert_eq!(depth_limited_dfs(&Up, 0, &is_five, 4, Limits::NONE).outcome, Outcome::Cutoff);
/// assert_eq!(depth_limited_dfs(&Up, 0, &is_five, 5, Limits::NONE).solution().map(|s| s.len()), Some(5));
/// ```
pub fn depth_limited_dfs<G, T>(
    graph: &G,
    start: G::State,
    goal: &T,
    limit: usize,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    depth_limited_dfs_multi(graph, [start], goal, limit, budget)
}

/// [`depth_limited_dfs`] from several start states at once.
///
/// Probes the starts in order, each to the same depth limit.
///
/// Repeated starts count once, an empty set of starts finds nothing, and a start that is a
/// goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, depth_limited_dfs_multi};
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
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let result = depth_limited_dfs_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, 2, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
pub fn depth_limited_dfs_multi<G, T>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    limit: usize,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let roots = distinct(starts);
    let mut stats = Stats::default();
    if let Some(solution) = goal_root(&roots, goal) {
        return SearchResult {
            outcome: Outcome::Found(solution),
            stats,
        };
    }

    let probe = probe_roots(
        graph,
        &roots,
        goal,
        &mut stats,
        &mut budget,
        |_, _, depth| {
            if depth > limit {
                Bound::Prune(G::Cost::ZERO)
            } else {
                Bound::Open
            }
        },
    );

    let outcome = match probe {
        Probe::Found(solution) => Outcome::Found(solution),
        Probe::Exhausted(None) => Outcome::Unsolvable,
        Probe::Exhausted(Some(_)) => Outcome::Cutoff,
        Probe::Stopped => Outcome::LimitReached,
    };

    SearchResult { outcome, stats }
}

/// Iterative deepening depth-first search (IDDFS): the path with the fewest edges, in
/// memory linear in its length.
///
/// Runs [`depth_limited_dfs`] with limits `0, 1, 2, ...` until a goal is found or a limit
/// cuts nothing off ([`Outcome::Unsolvable`]). The budget sees the counters of all
/// iterations together.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, iddfs};
///
/// /// Doubling or adding one.
/// struct Numbers;
///
/// impl Successors for Numbers {
///     type State = u32;
///     type Action = char;
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut(char, u32, u32)) {
///         if *n < 100 {
///             emit('+', n + 1, 1);
///             emit('*', n * 2, 1);
///         }
///     }
/// }
///
/// let result = iddfs(&Numbers, 1, &|n: &u32| *n == 24, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.len()), Some(5));
/// ```
pub fn iddfs<G, T>(
    graph: &G,
    start: G::State,
    goal: &T,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    iddfs_multi(graph, [start], goal, budget)
}

/// [`iddfs`] from several start states at once.
///
/// Each iteration probes every start, in order, to the same depth limit, so the solution
/// has the fewest edges from any start.
///
/// Repeated starts count once, an empty set of starts finds nothing, and a start that is a
/// goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, iddfs_multi};
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
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let result = iddfs_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![6, 7, 8]));
/// ```
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
pub fn iddfs_multi<G, T>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let roots = distinct(starts);
    let mut stats = Stats::default();
    if let Some(solution) = goal_root(&roots, goal) {
        return SearchResult {
            outcome: Outcome::Found(solution),
            stats,
        };
    }

    for limit in 0.. {
        let probe = probe_roots(
            graph,
            &roots,
            goal,
            &mut stats,
            &mut budget,
            |_, _, depth| {
                if depth > limit {
                    Bound::Prune(G::Cost::ZERO)
                } else {
                    Bound::Open
                }
            },
        );

        let outcome = match probe {
            Probe::Exhausted(Some(_)) => continue,
            Probe::Found(solution) => Outcome::Found(solution),
            Probe::Exhausted(None) => Outcome::Unsolvable,
            Probe::Stopped => Outcome::LimitReached,
        };

        return SearchResult { outcome, stats };
    }

    unreachable!("depth limits never run out")
}

/// Iterative deepening A* (IDA*, Korf 1985): cheapest path with an admissible heuristic,
/// in memory linear in the path length.
///
/// Runs depth-first tree searches that prune states with `f = g + h` above a threshold,
/// starting at `h(start)` and raising it to the smallest pruned `f` after each iteration.
/// Optimal for any admissible `heuristic` (consistency is not needed); `None` prunes a dead
/// end. Skips successors already on the current path, so zero-cost cycles are harmless.
/// Expands states again in every iteration and along every path, so it suits trees and
/// graphs with few transpositions; it costs at least one iteration per distinct `f` value
/// up to the optimum, which hurts with many distinct float costs.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, ida_star};
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
/// let distance = heuristic_fn(|s: &i32| Some(6_i32.abs_diff(*s)));
/// let result = ida_star(&Line, 0, &|s: &i32| *s == 6, distance, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(6));
/// assert_eq!(result.stats.expanded, 6);
/// ```
pub fn ida_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    ida_star_multi(graph, [start], goal, heuristic, budget)
}

/// [`ida_star`] from several start states at once.
///
/// Each iteration probes every start, in order, under one threshold, which starts at the
/// smallest `h` of any start, so the solution is optimal over all starts for an admissible
/// heuristic.
///
/// Repeated starts count once, an empty set of starts finds nothing, and a start that is a
/// goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, ida_star_multi};
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
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let h = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let result = ida_star_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, h, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn ida_star_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    run_ida(graph, starts, goal, heuristic, NoTable, budget)
}

/// IDA* with a bounded transposition table of `capacity` states: [`ida_star`] that searches
/// below a state at most once per iteration and path cost.
///
/// Plain IDA* re-expands a state along every path that reaches it, which blows up on graphs
/// with many transpositions (grids, permutation puzzles). The table remembers, per
/// iteration, the smallest `g` each recorded state was searched from, and skips a state
/// reached again from no smaller `g`: the earlier visit had at least as much of the
/// threshold left. Solutions stay optimal for an admissible heuristic.
///
/// The table is direct-mapped: each state hashes to one of `capacity` slots, allocated up
/// front, and a new entry overwrites the old one. A lost entry only repeats work, so any
/// capacity is correct; `0` disables the table. Needs `Hash` on states, unlike
/// [`ida_star`].
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, ida_star, ida_star_tt};
///
/// /// An open 8 x 8 grid with unit moves right and down: many paths reach each cell.
/// struct Grid;
///
/// impl Successors for Grid {
///     type State = (u8, u8);
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, &(x, y): &(u8, u8), mut emit: impl FnMut((), (u8, u8), u32)) {
///         if x < 7 {
///             emit((), (x + 1, y), 1);
///         }
///         if y < 7 {
///             emit((), (x, y + 1), 1);
///         }
///     }
/// }
///
/// // The blind heuristic `0` makes every transposition count.
/// let blind = || heuristic_fn(|_: &(u8, u8)| Some(0_u32));
/// let goal = |cell: &(u8, u8)| *cell == (7, 7);
/// let plain = ida_star(&Grid, (0, 0), &goal, blind(), Limits::NONE);
/// let table = ida_star_tt(&Grid, (0, 0), &goal, blind(), 1 << 10, Limits::NONE);
///
/// assert_eq!(table.solution().map(|s| s.cost), Some(14));
/// assert_eq!(plain.solution().map(|s| s.cost), Some(14));
/// // States reach slots through a hasher seeded per process, so the count varies between
/// // runs: usually about 460 expansions against 22 878 without the table, rarely over
/// // ten thousand when hot states share a slot. A lost entry only repeats work the plain
/// // search does too.
/// assert!(table.stats.expanded < plain.stats.expanded);
/// ```
pub fn ida_star_tt<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    capacity: usize,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    ida_star_tt_multi(graph, [start], goal, heuristic, capacity, budget)
}

/// [`ida_star_tt`] from several start states at once, as [`ida_star_multi`] does.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, ida_star_tt_multi};
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
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let h = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let result = ida_star_tt_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, h, 64, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn ida_star_tt_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    capacity: usize,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    run_ida(graph, starts, goal, heuristic, Table::new(capacity), budget)
}

/// The IDA* iterations from `starts`, skipping states that `table` says this iteration has
/// already searched below from a path cost at most as large.
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
fn run_ida<G, T, H, X>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    mut heuristic: H,
    mut table: X,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    X: Transpositions<G::State, G::Cost>,
{
    let roots = distinct(starts);
    let mut stats = Stats::default();
    if let Some(solution) = goal_root(&roots, goal) {
        return SearchResult {
            outcome: Outcome::Found(solution),
            stats,
        };
    }

    let Some(mut threshold) = roots
        .iter()
        .filter_map(|root| heuristic.estimate(root))
        .min()
    else {
        return SearchResult {
            outcome: Outcome::Unsolvable,
            stats,
        };
    };

    loop {
        table.next_iteration();
        let probe = probe_roots(
            graph,
            &roots,
            goal,
            &mut stats,
            &mut budget,
            |state, g, _| {
                if table.covers(state, g) {
                    return Bound::Dead;
                }

                match heuristic.estimate(state) {
                    None => Bound::Dead,
                    Some(h) if g + h > threshold => Bound::Prune(g + h),
                    Some(_) => {
                        table.record(state, g);
                        Bound::Open
                    }
                }
            },
        );

        let outcome = match probe {
            Probe::Exhausted(Some(next)) => {
                threshold = next;
                continue;
            }
            Probe::Found(solution) => Outcome::Found(solution),
            Probe::Exhausted(None) => Outcome::Unsolvable,
            Probe::Stopped => Outcome::LimitReached,
        };

        return SearchResult { outcome, stats };
    }
}

/// The starts without repeats, in order of first appearance.
///
/// Compares by equality only, since tree searches do not require `Hash`: quadratic in the
/// number of starts, which is small in practice.
pub(crate) fn distinct<S: Eq>(starts: impl IntoIterator<Item = S>) -> Vec<S> {
    let mut roots: Vec<S> = Vec::new();
    for start in starts {
        if !roots.contains(&start) {
            roots.push(start);
        }
    }

    roots
}

/// The zero-cost solution at the first root that is a goal, if any.
pub(crate) fn goal_root<S: Clone, A, C: Cost, T: Goal<S> + ?Sized>(
    roots: &[S],
    goal: &T,
) -> Option<Solution<S, A, C>> {
    let root = roots.iter().find(|root| goal.is_goal(root))?;

    Some(Solution {
        states: alloc::vec![root.clone()],
        actions: Vec::new(),
        cost: C::ZERO,
    })
}

/// Probes from every root in order under one bound, as if they hung below a virtual root:
/// stops at the first goal or budget stop, and otherwise reports the smallest pruned
/// value over all roots.
fn probe_roots<G, T>(
    graph: &G,
    roots: &[G::State],
    goal: &T,
    stats: &mut Stats,
    budget: &mut impl Budget<Stats>,
    mut bound: impl FnMut(&G::State, G::Cost, usize) -> Bound<G::Cost>,
) -> Probe<G::State, G::Action, G::Cost>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let mut pruned: Option<G::Cost> = None;
    for root in roots {
        match probe(graph, root.clone(), goal, stats, budget, &mut bound) {
            Probe::Exhausted(value) => pruned = pruned.into_iter().chain(value).min(),
            ended => return ended,
        }
    }

    Probe::Exhausted(pruned)
}

/// What a depth-first probe does with a state.
enum Bound<C> {
    /// Search below it.
    Open,
    /// Cut it off with this value (the pruned `f` for IDA*); a deeper iteration may open it.
    Prune(C),
    /// Never search below it: a proven dead end.
    Dead,
}

/// How a depth-first probe ended.
enum Probe<S, A, C> {
    /// A goal was found.
    Found(Solution<S, A, C>),
    /// The tree within the bound holds no goal. `Some(v)` if something was pruned, with
    /// `v` the smallest pruned value; `None` if nothing was.
    Exhausted(Option<C>),
    /// The budget stopped the probe.
    Stopped,
}

/// One state on the current path of a probe.
struct Frame<S, A, C> {
    /// The state.
    state: S,
    /// The edge label and cost that led here; `None` for the start.
    via: Option<(A, C)>,
    /// The path cost from the start.
    g: C,
    /// The successors not tried yet, in reverse order (the next one is last). `None` until
    /// the state is expanded.
    children: Option<Vec<(A, S, C)>>,
}

/// A depth-first tree search from `origin`, cutting off states as `bound(state, g, depth)`
/// says. Goals are tested on states the bound keeps open, when they are generated (and on
/// the start).
fn probe<G, T>(
    graph: &G,
    origin: G::State,
    goal: &T,
    stats: &mut Stats,
    budget: &mut impl Budget<Stats>,
    mut bound: impl FnMut(&G::State, G::Cost, usize) -> Bound<G::Cost>,
) -> Probe<G::State, G::Action, G::Cost>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let mut pruned: Option<G::Cost> = None;
    let mut keep = |value: G::Cost| {
        pruned = Some(pruned.map_or(
            value,
            |old| {
                if value.cmp(&old).is_lt() { value } else { old }
            },
        ));
    };

    let root = Frame {
        state: origin,
        via: None,
        g: G::Cost::ZERO,
        children: None,
    };
    match bound(&root.state, root.g, 0) {
        Bound::Dead => return Probe::Exhausted(None),
        Bound::Prune(value) => return Probe::Exhausted(Some(value)),
        Bound::Open if goal.is_goal(&root.state) => return Probe::Found(path(&[root])),
        Bound::Open => {}
    }

    let mut stack = alloc::vec![root];
    while let Some(top) = stack.last_mut() {
        if top.children.is_none() {
            if budget.exhausted(stats) {
                return Probe::Stopped;
            }

            let mut children = Vec::new();
            stats.expanded += 1;
            stats.generated += collect_successors(graph, &top.state, &mut children);
            children.reverse();
            top.children = Some(children);
        }

        let parent_g = top.g;
        let Some((action, child, cost)) = top.children.as_mut().and_then(Vec::pop) else {
            stack.pop();
            continue;
        };

        if stack.iter().any(|frame| frame.state == child) {
            continue;
        }

        let g = parent_g + cost;
        match bound(&child, g, stack.len()) {
            Bound::Dead => {}
            Bound::Prune(value) => keep(value),
            Bound::Open => {
                let found = goal.is_goal(&child);
                stack.push(Frame {
                    state: child,
                    via: Some((action, cost)),
                    g,
                    children: None,
                });
                if found {
                    return Probe::Found(path(&stack));
                }
            }
        }
    }

    Probe::Exhausted(pruned)
}

/// The solution along the frames of the current path, start first.
fn path<S: Clone, A: Clone, C: Cost>(frames: &[Frame<S, A, C>]) -> Solution<S, A, C> {
    let states = frames.iter().map(|frame| frame.state.clone()).collect();
    let actions = frames
        .iter()
        .filter_map(|frame| frame.via.as_ref().map(|(action, _)| action.clone()))
        .collect();
    let cost = frames
        .iter()
        .filter_map(|frame| frame.via.as_ref().map(|&(_, cost)| cost))
        .fold(C::ZERO, |sum, cost| sum + cost);

    Solution {
        states,
        actions,
        cost,
    }
}
