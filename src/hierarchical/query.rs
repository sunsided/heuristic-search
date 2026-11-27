//! The queries over a [`Hierarchy`]: [`hierarchical_a_star`] and its variants, which plan
//! on the abstract graph and refine every abstract edge in the graph, one after another or
//! (feature `parallel`) across threads.

use alloc::vec;
use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, heuristic_fn};

use super::{Hierarchy, Regions};
use crate::best_first::{GraphResult, a_star_multi};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::Successors;

/// Hierarchical A* (HPA*) from `start` to the state `goal` over a prebuilt [`Hierarchy`]
/// of `graph`.
///
/// The start and goal are connected to the entrances of their regions by A* inside the
/// region; then A* on the abstract graph picks the sequence of entrances, and every
/// abstract edge on it is refined into a path of `graph` by A* inside its region. The
/// refinements are independent; [`hierarchical_a_star`] runs them one after another, and
/// `hierarchical_a_star_par` (feature `parallel`) across threads.
///
/// `distance` is an admissible estimate of the cost between two states, used as the
/// heuristic of every search; `|_, _| 0` is fine. `graph` must be the graph the hierarchy
/// was built from.
///
/// The path is a cheapest one over the abstract graph. With [`Hierarchy::new`] that is a
/// cheapest path of the graph; with [`Hierarchy::grid`] it may cost more (the entrance
/// cells are a selection), and it is found whenever any path exists.
///
/// [`Stats`] add up the connecting, abstract and refining searches, but not the searches
/// that built the hierarchy. The `budget` sees those running totals while connecting and
/// on the abstract graph. Refinement is not budgeted: each refining search stays inside
/// one region and is known to succeed.
///
/// # Panics
///
/// Panics if an abstract edge cannot be refined, which happens only if `graph` differs
/// from the graph the hierarchy was built from.
///
/// # Examples
///
/// ```
/// use heuristic_search::{EightConnected, Hierarchy, Limits, Octile, UniformGrid, a_star, hierarchical_a_star, OctileDistance};
///
/// /// An open 16 x 16 grid with a wall at x = 8 that has a gap at y = 12.
/// struct Wall;
///
/// impl UniformGrid for Wall {
///     fn width(&self) -> usize {
///         16
///     }
///
///     fn height(&self) -> usize {
///         16
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         x != 8 || y == 12
///     }
/// }
///
/// let hierarchy = Hierarchy::grid(&Wall, 4);
/// let distance = |a: &(usize, usize), b: &(usize, usize)| Octile::between(*a, *b);
/// let graph = EightConnected(Wall);
///
/// let result = hierarchical_a_star(&graph, &hierarchy, (0, 0), (15, 0), distance, Limits::NONE);
/// let solution = result.solution().expect("the gap joins both halves");
/// let optimal = a_star(&graph, (0, 0), &|c: &(usize, usize)| *c == (15, 0), OctileDistance((15, 0)), Limits::NONE);
///
/// assert!(solution.states.contains(&(8, 12)));
/// assert!(solution.cost >= optimal.solution().expect("reachable").cost);
/// ```
pub fn hierarchical_a_star<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    start: G::State,
    goal: G::State,
    distance: D,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    R: Regions<G::State>,
    D: Fn(&G::State, &G::State) -> G::Cost,
{
    hierarchical_a_star_multi(graph, hierarchy, [start], goal, distance, budget)
}

/// [`hierarchical_a_star`] from several start states at once.
///
/// Every start is inserted into the abstract graph at path cost zero. Repeated starts are
/// harmless, an empty set of starts finds nothing, and the first start that is the goal is
/// returned at once.
///
/// # Panics
///
/// Panics if an abstract edge cannot be refined, which happens only if `graph` differs
/// from the graph the hierarchy was built from.
///
/// # Examples
///
/// ```
/// use heuristic_search::{EightConnected, Hierarchy, Limits, Octile, UniformGrid, hierarchical_a_star_multi};
///
/// /// An open 8 x 8 grid.
/// struct Open;
///
/// impl UniformGrid for Open {
///     fn width(&self) -> usize {
///         8
///     }
///
///     fn height(&self) -> usize {
///         8
///     }
///
///     fn is_passable(&self, _: usize, _: usize) -> bool {
///         true
///     }
/// }
///
/// let hierarchy = Hierarchy::grid(&Open, 4);
/// let distance = |a: &(usize, usize), b: &(usize, usize)| Octile::between(*a, *b);
/// let result = hierarchical_a_star_multi(&EightConnected(Open), &hierarchy, [(0, 0), (7, 6)], (7, 7), distance, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![(7, 6), (7, 7)]));
/// ```
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
pub fn hierarchical_a_star_multi<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    starts: impl IntoIterator<Item = G::State>,
    goal: G::State,
    distance: D,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    R: Regions<G::State>,
    D: Fn(&G::State, &G::State) -> G::Cost,
{
    match plan(graph, hierarchy, starts, goal, &distance, budget) {
        Err(result) => result,
        Ok((segments, stats)) => {
            let refined = segments
                .iter()
                .map(|(from, to)| refine(graph, hierarchy, from, to, &distance))
                .collect();
            join(refined, stats)
        }
    }
}

/// [`hierarchical_a_star`] with the abstract edges refined across threads.
///
/// The refinements are split into one contiguous share per available core and run under
/// [`std::thread::scope`]. The result, statistics included, is the same as that of
/// [`hierarchical_a_star`].
///
/// # Panics
///
/// Panics if an abstract edge cannot be refined, which happens only if `graph` differs
/// from the graph the hierarchy was built from.
#[cfg(feature = "parallel")]
pub fn hierarchical_a_star_par<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    start: G::State,
    goal: G::State,
    distance: D,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send + Sync,
    G::Action: Clone + Send,
    G::Cost: Send + Sync,
    R: Regions<G::State> + Sync,
    D: Fn(&G::State, &G::State) -> G::Cost + Sync,
{
    hierarchical_a_star_par_multi(graph, hierarchy, [start], goal, distance, budget)
}

/// [`hierarchical_a_star_multi`] with the abstract edges refined across threads, as in
/// `hierarchical_a_star_par`.
///
/// # Panics
///
/// Panics if an abstract edge cannot be refined, which happens only if `graph` differs
/// from the graph the hierarchy was built from.
#[cfg(feature = "parallel")]
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
pub fn hierarchical_a_star_par_multi<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    starts: impl IntoIterator<Item = G::State>,
    goal: G::State,
    distance: D,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send + Sync,
    G::Action: Clone + Send,
    G::Cost: Send + Sync,
    R: Regions<G::State> + Sync,
    D: Fn(&G::State, &G::State) -> G::Cost + Sync,
{
    let (segments, stats) = match plan(graph, hierarchy, starts, goal, &distance, budget) {
        Err(result) => return result,
        Ok(plan) => plan,
    };

    let threads = std::thread::available_parallelism().map_or(1, core::num::NonZeroUsize::get);
    let share = segments.len().div_ceil(threads);
    let refined = std::thread::scope(|scope| {
        let handles: Vec<_> = segments
            .chunks(share)
            .map(|part| {
                let distance = &distance;
                scope.spawn(move || {
                    part.iter()
                        .map(|(from, to)| refine(graph, hierarchy, from, to, distance))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        handles
            .into_iter()
            .flat_map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });

    join(refined, stats)
}

/// The abstract edges to refine, as `(from, to)` states in path order, with the counters
/// of the connecting and abstract searches; or the final result if the search ends before
/// refinement (a start is the goal, no abstract path, or the budget ran out).
type Plan<G> = Result<
    (
        Vec<(<G as Successors>::State, <G as Successors>::State)>,
        Stats,
    ),
    GraphResult<G>,
>;

/// Connects the starts and the goal to the abstract graph and runs A* on it.
fn plan<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    starts: impl IntoIterator<Item = G::State>,
    goal: G::State,
    distance: &D,
    mut budget: impl Budget<Stats>,
) -> Plan<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    R: Regions<G::State>,
    D: Fn(&G::State, &G::State) -> G::Cost,
{
    let starts: Vec<G::State> = starts.into_iter().collect();
    if starts.contains(&goal) {
        let solution = Solution {
            states: vec![goal],
            actions: Vec::new(),
            cost: G::Cost::ZERO,
        };
        return Err(ended(Outcome::Found(solution), Stats::default()));
    }

    let base = hierarchy.nodes.len();
    let goal_id = base + starts.len();
    let state = |id: usize| -> &G::State {
        if id < base {
            &hierarchy.nodes[id]
        } else if id < goal_id {
            &starts[id - base]
        } else {
            &goal
        }
    };

    let mut graph_view = Abstract {
        base: &hierarchy.edges,
        added: vec![Vec::new(); goal_id + 1],
    };
    let mut counters = Stats::default();
    for (from, to) in connections(hierarchy, &starts, &goal, base) {
        let inner = hierarchy.local(graph, state(from), state(to), distance, |inner: &Stats| {
            budget.exhausted(&counters.plus(*inner))
        });
        counters = counters.plus(inner.stats);
        match inner.outcome {
            Outcome::Found(solution) => graph_view.added[from].push((to, solution.cost)),
            Outcome::LimitReached => return Err(ended(Outcome::LimitReached, counters)),
            Outcome::Unsolvable | Outcome::Cutoff => {}
        }
    }

    let h = heuristic_fn(|id: &usize| Some(distance(state(*id), &goal)));
    let inner = a_star_multi(
        &graph_view,
        base..goal_id,
        &|id: &usize| *id == goal_id,
        h,
        |inner: &Stats| budget.exhausted(&counters.plus(*inner)),
    );
    counters = counters.plus(inner.stats);
    let path = match inner.outcome {
        Outcome::Found(path) => path,
        Outcome::LimitReached => return Err(ended(Outcome::LimitReached, counters)),
        Outcome::Unsolvable | Outcome::Cutoff => {
            return Err(ended(Outcome::Unsolvable, counters));
        }
    };

    let segments = path
        .states
        .windows(2)
        .map(|pair| (state(pair[0]).clone(), state(pair[1]).clone()))
        .collect();

    Ok((segments, counters))
}

/// The abstract edges a query adds, as `(from, to)` node ids: from every start (id
/// `base + i`) to the entrances of its region and to the goal (id `base + starts.len()`)
/// if it shares the region, and from the entrances of the goal's region to the goal.
fn connections<S, C, R: Regions<S>>(
    hierarchy: &Hierarchy<S, C, R>,
    starts: &[S],
    goal: &S,
    base: usize,
) -> Vec<(usize, usize)> {
    let goal_id = base + starts.len();
    let goal_region = hierarchy.regions.region(goal);
    let members = |region: usize| {
        hierarchy
            .members
            .get(&region)
            .map_or(&[][..], Vec::as_slice)
    };

    let mut pairs = Vec::new();
    for (i, start) in starts.iter().enumerate() {
        let region = hierarchy.regions.region(start);
        pairs.extend(members(region).iter().map(|&to| (base + i, to)));
        if region == goal_region {
            pairs.push((base + i, goal_id));
        }
    }
    pairs.extend(members(goal_region).iter().map(|&from| (from, goal_id)));

    pairs
}

/// A [`SearchResult`] from its parts.
fn ended<S, A, C>(outcome: Outcome<S, A, C>, stats: Stats) -> SearchResult<S, A, C> {
    SearchResult { outcome, stats }
}

/// A refined abstract edge: its path in the graph and the counters of finding it.
type Refined<G> = (
    Solution<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>,
    Stats,
);

/// The path of the abstract edge `from -> to` in `graph`, with the counters of finding it:
/// A* inside the region if both lie in one, else the cheapest graph edge between them
/// (the first of equally cheap ones).
fn refine<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    from: &G::State,
    to: &G::State,
    distance: &D,
) -> Refined<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    R: Regions<G::State>,
    D: Fn(&G::State, &G::State) -> G::Cost,
{
    /// Why refinement cannot fail on the graph the hierarchy was built from.
    const STALE: &str = "abstract edges refine in the graph the hierarchy was built from";

    let regions = &hierarchy.regions;
    if regions.region(from) == regions.region(to) {
        let result = hierarchy.local(graph, from, to, distance, |_: &Stats| false);
        let stats = result.stats;

        return (result.into_solution().expect(STALE), stats);
    }

    let mut stats = Stats {
        expanded: 1,
        ..Stats::default()
    };
    let mut best: Option<(G::Action, G::Cost)> = None;
    graph.successors(from, |action, next, cost| {
        stats.generated += 1;
        if next == *to && best.as_ref().is_none_or(|(_, known)| cost < *known) {
            best = Some((action, cost));
        }
    });
    let (action, cost) = best.expect(STALE);
    let solution = Solution {
        states: vec![from.clone(), to.clone()],
        actions: vec![action],
        cost,
    };

    (solution, stats)
}

/// Concatenates the refined segments, in path order, into one solution, and adds their
/// counters to `stats`.
fn join<S, A, C: Cost>(
    segments: Vec<(Solution<S, A, C>, Stats)>,
    mut stats: Stats,
) -> SearchResult<S, A, C> {
    let mut path = Solution {
        states: Vec::new(),
        actions: Vec::new(),
        cost: C::ZERO,
    };
    for (segment, counters) in segments {
        let skip = usize::from(!path.states.is_empty());
        path.states.extend(segment.states.into_iter().skip(skip));
        path.actions.extend(segment.actions);
        path.cost = path.cost + segment.cost;
        stats = stats.plus(counters);
    }

    ended(Outcome::Found(path), stats)
}

/// The abstract graph of one query: the hierarchy's edges between entrances plus the
/// edges that connect the starts and the goal. States are node ids.
struct Abstract<'a, C> {
    /// The abstract edges between entrances, by node id.
    base: &'a [Vec<(usize, C)>],
    /// The edges added for this query, by source node id (every id has an entry).
    added: Vec<Vec<(usize, C)>>,
}

impl<C: Cost> Successors for Abstract<'_, C> {
    type State = usize;
    type Action = ();
    type Cost = C;

    fn successors(&self, id: &usize, mut emit: impl FnMut((), usize, C)) {
        let base = self.base.get(*id).map_or(&[][..], Vec::as_slice);
        for &(to, cost) in base.iter().chain(&self.added[*id]) {
            emit((), to, cost);
        }
    }
}
