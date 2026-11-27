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
/// that built the hierarchy. The `budget` sees those running totals throughout, refinement
/// included, so a query never runs past it; when it runs out the query returns
/// [`Outcome::LimitReached`] and the partial path is dropped.
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
    let mut budget = budget;
    match plan(graph, hierarchy, starts, goal, &distance, &mut budget) {
        Err(result) => result,
        Ok((segments, stats, _)) => {
            let refined = refine_all(graph, hierarchy, &segments, &distance, |s: &Stats| {
                budget.exhausted(&stats.plus(*s))
            });
            join(refined, stats)
        }
    }
}

/// [`hierarchical_a_star`] with the abstract edges refined across threads.
///
/// The refinement work is estimated before it starts: every abstract edge inside one
/// region is assumed to cost as many expansions as the mean search that connected the
/// start and goal to their regions. If that fills two threads of a thousand expansions
/// each, the edges are split into one contiguous share per such thread, capped by the
/// available cores, and run under [`std::thread::scope`] with the first share on the
/// calling thread; otherwise the calling thread refines them all and no thread is
/// spawned. Unless the budget runs out, the result, statistics included, is the same as
/// that of [`hierarchical_a_star`].
///
/// The threads share the `budget` behind a lock, so it must be [`Send`]. Each asks it with
/// the counters summed over all threads before its first refining expansion and then every
/// 64 expansions, as [`hda_star`](crate::hda_star) does; once it is exhausted every thread
/// stops. On the threaded path a query may therefore run a little past the budget, and
/// where it stops depends on thread timing.
///
/// It pays off when refinement dominates a query: long paths through large regions, or a
/// weak `distance` (`|_, _| 0` makes every refinement Dijkstra's algorithm). With small
/// regions and a good `distance`, refinement is a few hundred expansions and this runs
/// sequentially, about as fast as [`hierarchical_a_star`]. Connecting the start and goal
/// and the search on the abstract graph run on the calling thread either way, which
/// bounds the speed-up.
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
    budget: impl Budget<Stats> + Send,
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
    budget: impl Budget<Stats> + Send,
) -> GraphResult<G>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send + Sync,
    G::Action: Clone + Send,
    G::Cost: Send + Sync,
    R: Regions<G::State> + Sync,
    D: Fn(&G::State, &G::State) -> G::Cost + Sync,
{
    let mut budget = budget;
    let (segments, stats, per_search) =
        match plan(graph, hierarchy, starts, goal, &distance, &mut budget) {
            Err(result) => return result,
            Ok(plan) => plan,
        };

    // Asked only when the work fills two threads: the call reads cgroup files on Linux.
    let cores = || std::thread::available_parallelism().map_or(1, core::num::NonZeroUsize::get);
    let refined = refine_par(
        graph,
        hierarchy,
        &segments,
        &distance,
        (stats, &mut budget),
        per_search,
        cores,
        EXPANSIONS_PER_THREAD,
    );

    join(refined, stats)
}

/// The least refinement work, in expansions, worth a thread of its own.
///
/// On grids an expansion costs about half a microsecond and spawning a scoped thread tens
/// to a hundred, so a thread that carries a thousand expansions spends at most about a
/// tenth of its time starting up.
#[cfg(feature = "parallel")]
const EXPANSIONS_PER_THREAD: u64 = 1000;

/// Refines `segments` in path order, across at most `cores()` threads (the calling thread
/// included), each carrying an estimated `per_thread` expansions or more. `budget` is the
/// query's budget with the counters spent before refinement.
///
/// The work is estimated as `per_search` expansions for every segment inside one region;
/// a segment between two regions is a single expansion. If the estimate does not fill two
/// threads, the calling thread refines everything and no thread is spawned; otherwise the
/// segments are split into contiguous shares, all but the first on spawned threads, which
/// share the budget as described at [`hierarchical_a_star_par`].
#[cfg(feature = "parallel")]
#[expect(
    clippy::too_many_arguments,
    reason = "the two tuning knobs are parameters so the tests can force threads"
)]
fn refine_par<G, R, D, B>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    segments: &[(G::State, G::State)],
    distance: &D,
    (spent, budget): (Stats, &mut B),
    per_search: u64,
    cores: impl FnOnce() -> usize,
    per_thread: u64,
) -> Vec<Refined<G>>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send + Sync,
    G::Action: Clone + Send,
    G::Cost: Send + Sync,
    R: Regions<G::State> + Sync,
    D: Fn(&G::State, &G::State) -> G::Cost + Sync,
    B: Budget<Stats> + Send,
{
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use crate::hda::{BUDGET_INTERVAL, lock};

    let regions = &hierarchy.regions;
    let inside = segments
        .iter()
        .filter(|(from, to)| regions.region(from) == regions.region(to))
        .count();
    let estimate = per_search.saturating_mul(inside as u64);
    let threads = segments
        .len()
        .min(usize::try_from(estimate / per_thread.max(1)).unwrap_or(usize::MAX));
    let threads = if threads < 2 { 1 } else { threads.min(cores()) };
    if threads < 2 {
        return refine_all(graph, hierarchy, segments, distance, |s: &Stats| {
            budget.exhausted(&spent.plus(*s))
        });
    }

    // The counters reported by all threads with the budget, and whether it ran out.
    let reports = Mutex::new((spent, budget));
    let stopped = AtomicBool::new(false);
    let refine_share = |part: &[(G::State, G::State)]| {
        // The share's counters last reported, `None` before the first report.
        let mut reported: Option<Stats> = None;
        let budget = |counters: &Stats| {
            if stopped.load(Ordering::Relaxed) {
                return true;
            }
            let last = reported.unwrap_or_default();
            if reported.is_some() && counters.expanded < last.expanded + BUDGET_INTERVAL {
                return false;
            }
            reported = Some(*counters);

            let mut reports = lock(&reports);
            let (total, budget) = &mut *reports;
            *total = total.plus(counters.since(last));
            let exhausted = budget.exhausted(total);
            if exhausted {
                stopped.store(true, Ordering::Relaxed);
            }

            exhausted
        };

        refine_all(graph, hierarchy, part, distance, budget)
    };

    let mut shares = segments.chunks(segments.len().div_ceil(threads));
    let first = shares.next().unwrap_or_default();
    let refine_share = &refine_share;
    std::thread::scope(|scope| {
        let handles: Vec<_> = shares
            .map(|part| scope.spawn(move || refine_share(part)))
            .collect();
        let mut refined = refine_share(first);
        for handle in handles {
            refined.extend(
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            );
        }

        refined
    })
}

/// The abstract edges to refine, as `(from, to)` states in path order, with the counters
/// of the connecting and abstract searches and the mean expansions of a connecting search
/// (which, like a refinement, stays inside one region); or the final result if the search ends before
/// refinement (a start is the goal, no abstract path, or the budget ran out).
type Plan<G> = Result<
    (
        Vec<(<G as Successors>::State, <G as Successors>::State)>,
        Stats,
        u64,
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
    budget: &mut impl Budget<Stats>,
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
    let connections = connections(hierarchy, &starts, &goal, base);
    for &(from, to) in &connections {
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

    let per_search = counters.expanded / (connections.len() as u64).max(1);

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

    Ok((segments, counters, per_search))
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

/// A refined abstract edge: its path in the graph, or `None` if the budget stopped the
/// refinement, and the counters of finding it.
type Segment<S, A, C> = (Option<Solution<S, A, C>>, Stats);

/// A [`Segment`] of the graph `G`.
type Refined<G> =
    Segment<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;

/// Refines `segments` in path order on the calling thread, until the first one that
/// `budget` stops. The budget is asked with the counters of all refinements so far.
fn refine_all<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    segments: &[(G::State, G::State)],
    distance: &D,
    mut budget: impl FnMut(&Stats) -> bool,
) -> Vec<Refined<G>>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    R: Regions<G::State>,
    D: Fn(&G::State, &G::State) -> G::Cost,
{
    let mut done = Stats::default();
    let mut refined = Vec::with_capacity(segments.len());
    for (from, to) in segments {
        let segment = refine(graph, hierarchy, from, to, distance, |inner: &Stats| {
            budget(&done.plus(*inner))
        });
        done = done.plus(segment.1);
        let stopped = segment.0.is_none();
        refined.push(segment);
        if stopped {
            break;
        }
    }

    refined
}

/// The path of the abstract edge `from -> to` in `graph`, with the counters of finding it:
/// A* inside the region if both lie in one, else the cheapest graph edge between them
/// (the first of equally cheap ones). `None` if `budget`, asked with the counters of this
/// refinement, runs out first.
fn refine<G, R, D>(
    graph: &G,
    hierarchy: &Hierarchy<G::State, G::Cost, R>,
    from: &G::State,
    to: &G::State,
    distance: &D,
    mut budget: impl Budget<Stats>,
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
        let result = hierarchy.local(graph, from, to, distance, budget);
        return match result.outcome {
            Outcome::Found(solution) => (Some(solution), result.stats),
            Outcome::LimitReached => (None, result.stats),
            Outcome::Unsolvable | Outcome::Cutoff => panic!("{STALE}"),
        };
    }
    if budget.exhausted(&Stats::default()) {
        return (None, Stats::default());
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

    (Some(solution), stats)
}

/// Concatenates the refined segments, in path order, into one solution, and adds their
/// counters to `stats`; [`Outcome::LimitReached`] if the budget stopped any of them.
fn join<S, A, C: Cost>(segments: Vec<Segment<S, A, C>>, mut stats: Stats) -> SearchResult<S, A, C> {
    let mut path = Solution {
        states: Vec::new(),
        actions: Vec::new(),
        cost: C::ZERO,
    };
    let mut complete = true;
    for (segment, counters) in segments {
        stats = stats.plus(counters);
        let Some(segment) = segment else {
            complete = false;
            continue;
        };
        let skip = usize::from(!path.states.is_empty());
        path.states.extend(segment.states.into_iter().skip(skip));
        path.actions.extend(segment.actions);
        path.cost = path.cost + segment.cost;
    }

    if complete {
        ended(Outcome::Found(path), stats)
    } else {
        ended(Outcome::LimitReached, stats)
    }
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

#[cfg(all(test, feature = "parallel"))]
mod tests {
    //! The threaded refinement against the sequential one, with the work threshold forced
    //! down so that even small maps use threads, with and without a budget.

    use crate::Limits;
    use proptest::prelude::*;

    use super::{Hierarchy, join, plan, refine_all, refine_par};
    use crate::Stats;
    use crate::grid::{EightConnected, GridCell, Octile, UniformGrid};

    /// A 12 x 12 grid whose walls are the set bits of `walls`, row-major.
    struct Map {
        /// Bit `y * 12 + x` set means the cell is a wall.
        walls: [u64; 3],
    }

    impl UniformGrid for Map {
        fn width(&self) -> usize {
            12
        }

        fn height(&self) -> usize {
            12
        }

        fn is_passable(&self, x: usize, y: usize) -> bool {
            let bit = y * 12 + x;
            self.walls[bit / 64] & (1 << (bit % 64)) == 0
        }
    }

    proptest! {
        #[test]
        fn threaded_refinement_matches_sequential(
            walls in proptest::array::uniform3(any::<u64>().prop_map(|w| w & w.rotate_left(7))),
            size in 1_usize..5,
            cores in 2_usize..6,
            start in (0_usize..12, 0_usize..12),
            goal in (0_usize..12, 0_usize..12),
        ) {
            let map = Map { walls };
            let hierarchy = Hierarchy::grid(&map, size);
            let graph = EightConnected(&map);
            let distance = |a: &GridCell, b: &GridCell| Octile::between(*a, *b);
            if let Ok((segments, stats, _)) = plan(&graph, &hierarchy, [start], goal, &distance, &mut |_: &Stats| false) {
                let sequential = refine_all(&graph, &hierarchy, &segments, &distance, |_: &Stats| false);
                let threaded = refine_par(&graph, &hierarchy, &segments, &distance, (stats, &mut |_: &Stats| false), u64::MAX, || cores, 1);
                prop_assert_eq!(threaded, sequential);
            }
        }

        /// Under an expansion budget that may run out during refinement, the threaded
        /// refinement either stops or, if it finds the path, finds the unlimited one; and
        /// it always finds it when the whole query fits the budget.
        #[test]
        fn threaded_refinement_respects_budget(
            walls in proptest::array::uniform3(any::<u64>().prop_map(|w| w & w.rotate_left(7))),
            size in 1_usize..5,
            cores in 2_usize..6,
            start in (0_usize..12, 0_usize..12),
            goal in (0_usize..12, 0_usize..12),
            percent in 0_u64..=120,
        ) {
            let map = Map { walls };
            let hierarchy = Hierarchy::grid(&map, size);
            let graph = EightConnected(&map);
            let distance = |a: &GridCell, b: &GridCell| Octile::between(*a, *b);
            if let Ok((segments, stats, _)) = plan(&graph, &hierarchy, [start], goal, &distance, &mut |_: &Stats| false) {
                let unlimited = join(refine_all(&graph, &hierarchy, &segments, &distance, |_: &Stats| false), stats);
                let mut limit = Limits::expansions(unlimited.stats.expanded * percent / 100);
                let threaded = join(
                    refine_par(&graph, &hierarchy, &segments, &distance, (stats, &mut limit), u64::MAX, || cores, 1),
                    stats,
                );
                if percent >= 100 {
                    prop_assert_eq!(threaded, unlimited);
                } else if threaded.solution().is_some() {
                    prop_assert_eq!(threaded, unlimited);
                } else {
                    prop_assert_eq!(threaded.outcome, crate::Outcome::LimitReached);
                }
            }
        }
    }
}
