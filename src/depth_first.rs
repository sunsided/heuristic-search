//! Depth-first search: [`dfs`], [`depth_limited_dfs`], [`iddfs`] and [`ida_star`].
//!
//! | Algorithm             | Memory            | Duplicates            | Returns                |
//! | --------------------- | ----------------- | --------------------- | ---------------------- |
//! | [`dfs`]               | every state seen  | closed set            | some path              |
//! | [`depth_limited_dfs`] | the current path  | cycles on the path    | a path of `<= limit`   |
//! | [`iddfs`]             | the current path  | cycles on the path    | fewest edges           |
//! | [`ida_star`]          | the current path  | cycles on the path    | cheapest (admissible)  |
//!
//! The last three are tree searches: they keep only the current path and skip a successor
//! already on it, so they need nothing but `Eq` on states but may revisit a state reached
//! along different paths. All four are iterative, so deep paths cannot overflow the call
//! stack. [`Stats::expanded`] counts every expansion, also repeated ones across iterations.

use alloc::vec::Vec;
use core::hash::Hash;

use crate::best_first::GraphResult;
use crate::cost::Cost;
use crate::outcome::{Budget, Outcome, SearchResult, Solution, Stats};
use crate::problem::{Goal, Heuristic, Successors, collect_successors};
use crate::space::{Reach, SearchSpace};

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
pub fn dfs<G, T>(graph: &G, start: G::State, goal: &T, mut budget: impl Budget) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let (mut space, start_goal) = SearchSpace::seed([start], goal);
    let mut stats = Stats::default();
    let finish = |outcome, stats| SearchResult { outcome, stats };
    if let Some(found) = start_goal {
        return finish(Outcome::Found(space.solution(found)), stats);
    }

    let mut stack = alloc::vec![0];
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
/// [`Outcome::LimitReached`], because a goal may lie deeper. Goals are tested when they are
/// generated.
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
/// assert_eq!(depth_limited_dfs(&Up, 0, &is_five, 4, Limits::NONE).outcome, Outcome::LimitReached);
/// assert_eq!(depth_limited_dfs(&Up, 0, &is_five, 5, Limits::NONE).solution().map(|s| s.len()), Some(5));
/// ```
pub fn depth_limited_dfs<G, T>(
    graph: &G,
    start: G::State,
    goal: &T,
    limit: usize,
    mut budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let mut stats = Stats::default();
    let probe = probe(
        graph,
        start,
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
        Probe::Exhausted(Some(_)) | Probe::Stopped => Outcome::LimitReached,
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
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the start by value like every other search; cloned once per iteration"
)]
pub fn iddfs<G, T>(graph: &G, start: G::State, goal: &T, mut budget: impl Budget) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let mut stats = Stats::default();
    for limit in 0.. {
        let probe = probe(
            graph,
            start.clone(),
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
    mut heuristic: H,
    mut budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let mut stats = Stats::default();
    let Some(mut threshold) = heuristic.estimate(&start) else {
        let outcome = if goal.is_goal(&start) {
            Outcome::Found(Solution {
                states: alloc::vec![start],
                actions: Vec::new(),
                cost: G::Cost::ZERO,
            })
        } else {
            Outcome::Unsolvable
        };

        return SearchResult { outcome, stats };
    };

    loop {
        let probe = probe(
            graph,
            start.clone(),
            goal,
            &mut stats,
            &mut budget,
            |state, g, _| match heuristic.estimate(state) {
                None => Bound::Dead,
                Some(h) if (g + h).cmp_cost(&threshold).is_gt() => Bound::Prune(g + h),
                Some(_) => Bound::Open,
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
    budget: &mut impl Budget,
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
        pruned = Some(pruned.map_or(value, |old| {
            if value.cmp_cost(&old).is_lt() {
                value
            } else {
                old
            }
        }));
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
