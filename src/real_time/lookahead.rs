//! Real-time search with a lookahead of `n` expansions: [`lss_lrta_star`] (LSS-LRTA*),
//! [`lhb_lrta_star`] (the same agent with the limited-horizon Bellman update of Hadar et al.),
//! [`lhb_gbfs`] and [`lhb_bwa_star`] (that update after a GBFS or BWAS search phase),
//! [`rtaa_star`] (RTAA*), and the limited-horizon update on its own, [`lhb_update`].
//!
//! All of them share one engine, [`Lookahead`]: a bounded best-first search from the current
//! state (A*, GBFS or batch weighted A*) that keeps every edge it generated, so the update
//! rules in [`update`] can run shortest-path computations over the explored graph. The
//! rules differ only in what they write back.

mod update;

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::Order;
use crate::cost::{CostExt, Weight};
use crate::open_list::OpenList;
use crate::outcome::{Limits, Stats};
use crate::problem::{Successors, collect_successors};
use crate::real_time::{End, LearnedHeuristic, Trajectory, TrialResult, TrialStats};
use crate::space::{NodeId, Reach, SearchSpace};

/// The result type of real-time trials over graph `G`.
type GraphTrial<G> =
    TrialResult<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;

/// One trial of LSS-LRTA* (Koenig and Sun 2009) from `start` to a state that satisfies
/// `goal`, learning in `table`.
///
/// Each step runs A* from the current state for at most `lookahead` expansions (stopping
/// early when it selects a goal), then sets the value of every expanded state to its cost
/// to the frontier: `h(s) = min over open states o of dist(s, o) + h(o)`, computed by
/// Dijkstra's algorithm backwards from the open list over the explored edges. Expanded
/// states that cannot reach the open list become dead ends. The agent then moves along the
/// A* path to the open state with the smallest `f` (or to the goal A* selected) and plans
/// again.
///
/// The update does not take the maximum with the old value. From an admissible table it
/// stays admissible, from a consistent one consistent (then values only grow). With a
/// lookahead of one it is [`lrta_star`](crate::lrta_star) with a different move rule.
///
/// The [`Budget`] is asked before every expansion; [`TrialStats::max_step_expansions`]
/// never exceeds `lookahead`. The A* reopens states when a cheaper path turns up, so
/// inconsistent tables are fine; a state that was reopened and not expanded again counts
/// as open for the update.
///
/// # Panics
///
/// Panics if `lookahead` is zero: the agent could never move.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Zero, lss_lrta_star};
///
/// /// A corridor `0..=20` with unit moves both ways.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if *cell < 20 {
///             emit(1, cell + 1, 1);
///         }
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let result = lss_lrta_star(&Corridor, 0, &|cell: &u8| *cell == 20, &mut table, 4, Limits::NONE);
///
/// assert_eq!(result.solution().map(|path| path.cost), Some(20));
/// assert!(result.trial.max_step_expansions <= 4);
/// ```
pub fn lss_lrta_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    lookahead: usize,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let search = Lookahead::new(graph, goal, lookahead);

    search.trial(start, table, budget, update::dijkstra)
}

/// One trial of LSS-LRTA*'s agent with the limited-horizon Bellman update (LHB) of Hadar
/// et al. 2025 ("Beyond Single-Step Updates", arXiv 2511.10264), learning in `table`.
///
/// The lookahead and the moves are those of [`lss_lrta_star`]; the update is the paper's
/// Eq. 5, computed as in its Algorithm 1 (see [`lhb_update`]). On graphs where every state
/// has a successor and goals have value zero, the two updates write the same values, so the
/// two agents behave identically; they differ on states without successors (LHB keeps
/// their value, LSS-LRTA* proves them dead ends), on goals with a nonzero value (LHB sets
/// them to zero) and on states reopened by the lookahead (interior for LHB, open for
/// LSS-LRTA*). From an admissible table it stays admissible.
///
/// # Panics
///
/// Panics if `lookahead` is zero: the agent could never move.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Zero, lhb_lrta_star};
///
/// /// A corridor `0..=20` with unit moves both ways.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if *cell < 20 {
///             emit(1, cell + 1, 1);
///         }
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let result = lhb_lrta_star(&Corridor, 10, &|cell: &u8| *cell == 0, &mut table, 4, Limits::NONE);
///
/// assert_eq!(result.solution().map(|path| path.cost), Some(10));
/// ```
pub fn lhb_lrta_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    lookahead: usize,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let search = Lookahead::new(graph, goal, lookahead);

    search.trial(start, table, budget, update::limited_horizon)
}

/// One trial of the limited-horizon Bellman update (LHB) of Hadar et al. 2025 with greedy
/// best-first search (GBFS) as its search phase, learning in `table`.
///
/// Each step runs GBFS from the current state with `table` as the heuristic for at most
/// `lookahead` expansions, stopping early when it selects a goal: the open state with the
/// smallest value is expanded next, path costs only break ties, and a state is never
/// reopened. The update is [`lhb_update`]'s over the search graph GBFS built, so it reads
/// the cheapest paths off every generated edge, not off GBFS's parent links. The agent then
/// moves along GBFS's path to the open state with the smallest value (or to the goal GBFS
/// selected) and plans again.
///
/// GBFS can reach deeper than A* with the same `lookahead`, so a step can learn values
/// further from the agent, but the moves follow the heuristic, not the cheapest path in
/// the search graph, so a trial that learns nothing need not be a cheapest path. The
/// update does not depend on the search phase: from an admissible table it stays
/// admissible, from a consistent one consistent.
///
/// The [`Budget`] is asked before every expansion; [`TrialStats::max_step_expansions`]
/// never exceeds `lookahead`.
///
/// # Panics
///
/// Panics if `lookahead` is zero: the agent could never move.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Zero, lhb_gbfs};
///
/// /// A corridor `0..=20` with unit moves both ways.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if *cell < 20 {
///             emit(1, cell + 1, 1);
///         }
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let goal = |cell: &u8| *cell == 0;
///
/// // Repeated trials learn until one changes nothing; on this corridor, that one is a
/// // cheapest path.
/// let last = loop {
///     let result = lhb_gbfs(&Corridor, 10, &goal, &mut table, 4, Limits::NONE);
///     if result.trial.updates == 0 {
///         break result;
///     }
/// };
///
/// assert_eq!(last.solution().map(|path| path.cost), Some(10));
/// ```
pub fn lhb_gbfs<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    lookahead: usize,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let search = Lookahead::new(graph, goal, lookahead).phase(Order::H, 1);

    search.trial(start, table, budget, update::limited_horizon)
}

/// One trial of the limited-horizon Bellman update (LHB) of Hadar et al. 2025 with batch
/// weighted A* (BWAS, Agostinelli et al. 2019) as its search phase, learning in `table`.
///
/// Each step runs BWAS from the current state with `table` as the heuristic for at most
/// `lookahead` expansions. BWAS is Weighted A* ordered by `g + w * h` (as
/// [`weighted_a_star`](crate::weighted_a_star)) that takes the `batch` open states with the
/// smallest keys at once and then expands them all, as Agostinelli et al. do to evaluate a
/// network on a whole batch; a state reached by a strictly cheaper path is reopened. The
/// papers write the order as `λ g + h` with `λ` in `[0, 1]`, which sorts like `g + w * h`
/// for `w = 1 / λ`. If a batch contains a goal, the step stops before expanding any of it;
/// the last batch is cut so that the step makes at most `lookahead` expansions. The update
/// is [`lhb_update`]'s over the search graph, and the agent moves along the BWAS path to
/// the open state with the smallest key (or to the selected goal) and plans again.
///
/// With [`Weight::ONE`] and a batch of one this is [`lhb_lrta_star`]. The update does not
/// depend on the search phase: from an admissible table it stays admissible, from a
/// consistent one consistent. The moves do: a trial that learns nothing need not be a
/// cheapest path, as a batch expands states out of `g + w * h` order and the weight
/// trades cost for depth.
///
/// The [`Budget`] is asked before every expansion; [`TrialStats::max_step_expansions`]
/// never exceeds `lookahead`.
///
/// # Panics
///
/// Panics if `lookahead` or `batch` is zero.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Weight, Zero, lhb_bwa_star};
///
/// /// A corridor `0..=20` with unit moves both ways.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if *cell < 20 {
///             emit(1, cell + 1, 1);
///         }
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let goal = |cell: &u8| *cell == 0;
/// let weight = Weight::new(5, 3); // λ = 0.6
/// let result = lhb_bwa_star(&Corridor, 10, &goal, &mut table, 8, weight, 4, Limits::NONE);
///
/// assert_eq!(result.solution().map(|path| path.cost), Some(10));
/// assert!(result.trial.max_step_expansions <= 8);
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "the lookahead agents' arguments plus BWAS's weight and batch size"
)]
pub fn lhb_bwa_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    lookahead: usize,
    weight: Weight,
    batch: usize,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    assert!(batch > 0, "a batch needs at least one state");
    let search = Lookahead::new(graph, goal, lookahead).phase(Order::Weighted(weight), batch);

    search.trial(start, table, budget, update::limited_horizon)
}

/// One trial of RTAA* (Koenig and Likhachev 2006) from `start` to a state that satisfies
/// `goal`, learning in `table`.
///
/// The lookahead and the moves are those of [`lss_lrta_star`]. The update is cheaper: with
/// `f` the smallest `g + h` on the open list when A* stopped, every expanded state gets
/// `h(s) = f - g(s)`, without a second search. The values are lower than LSS-LRTA*'s or
/// equal, so learning is slower per step but each step is cheaper. From an admissible table
/// it stays admissible, from a consistent one consistent.
///
/// # Panics
///
/// Panics if `lookahead` is zero: the agent could never move.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Zero, rtaa_star};
///
/// /// A corridor `0..=20` with unit moves both ways.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if *cell < 20 {
///             emit(1, cell + 1, 1);
///         }
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let result = rtaa_star(&Corridor, 10, &|cell: &u8| *cell == 0, &mut table, 4, Limits::NONE);
///
/// assert_eq!(result.solution().map(|path| path.cost), Some(10));
/// ```
pub fn rtaa_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    lookahead: usize,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    G::Cost: CostExt,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let search = Lookahead::new(graph, goal, lookahead);

    search.trial(start, table, budget, update::rtaa)
}

/// The limited-horizon Bellman update (LHB) of Hadar et al. 2025, Algorithm 1, on the
/// states around `sample`; returns how many table values changed.
///
/// Runs A* from `sample` with `table` as the heuristic for at most `expansions` expansions,
/// stopping early when it selects a goal. That gives a search graph `G`: the generated
/// states and every generated edge. Its leaves are the states without children in `G`
/// (never expanded, or expanded without successors). Every other state `n` gets
///
/// ```text
/// h(n) = min over leaves l reachable from n in G of C(n, l) + h(l)       (Eq. 5)
/// ```
///
/// where `C(n, l)` is the cost of a cheapest path in `G` and `h(l)` the value before the
/// update, computed by Dijkstra's algorithm backwards from the leaves, as the paper's
/// auxiliary node `z` with an edge of cost `h(l)` from every leaf does. A state that
/// reaches no leaf becomes a dead end. Goal states get `0` (the first case of Eq. 5);
/// leaves keep their value.
///
/// The paper uses this offline, to label randomly sampled states for training a network;
/// with a table, calling it on sampled states is tabular learning from limited-horizon
/// search. [`lhb_lrta_star`] interleaves it with moving. The update does not need a
/// consistent table, and from an admissible one it stays admissible: a cheapest path from
/// `n` to a goal leaves `G` through some leaf.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Successors, Zero, lhb_update};
///
/// /// A chain `0 -> 1 -> ... -> 9`, each edge at cost 2.
/// struct Chain;
///
/// impl Successors for Chain {
///     type State = u8;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u8, mut emit: impl FnMut((), u8, u32)) {
///         if *n < 9 {
///             emit((), n + 1, 2);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let changed = lhb_update(&Chain, 0, &|n: &u8| *n == 9, &mut table, 3);
///
/// assert_eq!(changed, 3);
/// assert_eq!(table.get(&0), Some(6), "three edges to the leaf 3, whose value is 0");
/// assert_eq!(table.get(&2), Some(2));
/// ```
pub fn lhb_update<G, T, H>(
    graph: &G,
    sample: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    expansions: usize,
) -> u64
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    if goal.is_goal(&sample) {
        return u64::from(table.set(sample, Some(G::Cost::ZERO)));
    }

    let mut search = Lookahead::new(graph, goal, expansions);
    let mut stats = Stats::default();
    let mut unlimited = Limits::NONE;
    let frontier = search.search(sample, table, &mut unlimited, &mut stats);

    update::limited_horizon(&search, frontier, table)
}

/// An edge `(action, to, cost)` of graph `G`.
type Edge<G> = (
    <G as Successors>::Action,
    <G as Successors>::State,
    <G as Successors>::Cost,
);

/// One planning step of LSS-LRTA* from `start`, as the local search of
/// [`RealTimeDStar`](crate::RealTimeDStar): a lookahead of at most `lookahead` expansions
/// and the Dijkstra update, without moving.
///
/// Returns the first edge of the A* path to the state LSS-LRTA* would move to, or `None`
/// if no goal is reachable from `start` (or `start` is a goal), and the number of changed
/// table values. Expansions are added to `stats`.
pub(super) fn lss_lrta_step<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    lookahead: usize,
    stats: &mut Stats,
) -> (Option<Edge<G>>, u64)
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let mut search = Lookahead::new(graph, goal, lookahead);
    let mut unlimited = Limits::NONE;
    let frontier = search.search(start, table, &mut unlimited, stats);
    let changed = update::dijkstra(&search, frontier, table);
    let Frontier::Target(target) = frontier else {
        return (None, changed);
    };

    let mut first = None;
    let mut node = target;
    while let Some((parent, action, cost)) = search.space.parent(node) {
        first = Some((action.clone(), node, cost));
        node = parent;
    }

    let first = first.map(|(action, node, cost)| (action, search.space.state(node).clone(), cost));

    (first, changed)
}

/// Where a lookahead stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frontier {
    /// The state to move to: a goal A* selected, or the open state with the smallest `f`
    /// after the last allowed expansion.
    Target(NodeId),
    /// The open list ran empty: no goal is reachable from the start of the lookahead.
    Exhausted,
    /// The budget stopped the lookahead.
    Limit,
}

/// The lookahead status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// On the open list with its current `g`.
    Open,
    /// Expanded, and not reopened since.
    Closed,
    /// A proven dead end (value `None`); never opened.
    Dead,
}

/// One bounded best-first search over a graph, keeping every generated edge for the update
/// rules.
struct Lookahead<'a, G: Successors, T: ?Sized> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The most expansions of one lookahead.
    limit: usize,
    /// The open-list order: A* ([`Weight::ONE`]), Weighted A* or GBFS.
    order: Order,
    /// How many open states each iteration takes before expanding them (BWAS); one for
    /// plain best-first search.
    batch: usize,
    /// The batch taken off the open list for expansion; kept between batches and searches,
    /// also when a search ends inside one, so its allocation is reused.
    selected: Vec<NodeId>,
    /// The generated states, their `g` and A* parent links.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// Table value of each node, by [`NodeId`], read once when the node is created.
    h: Vec<Option<G::Cost>>,
    /// Lookahead status of each node, by [`NodeId`].
    status: Vec<Status>,
    /// Whether each node has children in the search graph: it was expanded and has at least
    /// one successor.
    children: Vec<bool>,
    /// Every generated edge `(from, to, cost)`, duplicates included.
    edges: Vec<(NodeId, NodeId, G::Cost)>,
    /// The open list, keyed by [`Lookahead::order`].
    open: OpenList<G::Cost>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<'a, G, T> Lookahead<'a, G, T>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    /// An empty A* lookahead of at most `limit` expansions per search.
    fn new(graph: &'a G, goal: &'a T, limit: usize) -> Self {
        Self {
            graph,
            goal,
            limit,
            order: Order::Weighted(Weight::ONE),
            batch: 1,
            selected: Vec::new(),
            space: SearchSpace::seed([], goal).0,
            h: Vec::new(),
            status: Vec::new(),
            children: Vec::new(),
            edges: Vec::new(),
            open: OpenList::new(),
            buffer: Vec::new(),
        }
    }

    /// The same lookahead with open-list `order`, taking `batch` states per iteration.
    fn phase(self, order: Order, batch: usize) -> Self {
        Self {
            order,
            batch,
            ..self
        }
    }

    /// Runs one trial from `start`: plan, apply `update`, move, until a goal, a proven dead
    /// end or the budget.
    ///
    /// `update` writes the learned values and returns how many changed.
    fn trial<H: Heuristic<G::State, G::Cost>>(
        mut self,
        start: G::State,
        table: &mut LearnedHeuristic<G::State, G::Cost, H>,
        mut budget: impl Budget<Stats>,
        update: impl Fn(&Self, Frontier, &mut LearnedHeuristic<G::State, G::Cost, H>) -> u64,
    ) -> GraphTrial<G> {
        assert!(
            self.limit > 0,
            "a real-time lookahead needs at least one expansion"
        );

        let mut path = Trajectory::new(start);
        let mut stats = Stats::default();
        let mut trial = TrialStats::default();

        let end = loop {
            if self.goal.is_goal(path.current()) {
                break End::Goal;
            }

            let before = stats.expanded;
            let frontier = self.search(path.current().clone(), table, &mut budget, &mut stats);
            if frontier == Frontier::Limit {
                break End::Limit;
            }

            let changed = update(&self, frontier, table);
            trial.record_step(stats.expanded - before, changed);

            let Frontier::Target(target) = frontier else {
                break End::Unsolvable;
            };

            trial.moves += self.travel(target, &mut path);
        };

        path.finish(end, stats, trial)
    }

    /// Runs the search from `start` for at most [`Lookahead::limit`] expansions, replacing
    /// the previous search graph.
    ///
    /// Each iteration takes up to [`Lookahead::batch`] open states and then expands them;
    /// a goal among them ends the search before any of them is expanded, so a goal is
    /// never interior to the search graph.
    fn search<H: Heuristic<G::State, G::Cost>>(
        &mut self,
        start: G::State,
        table: &mut LearnedHeuristic<G::State, G::Cost, H>,
        budget: &mut impl Budget<Stats>,
        stats: &mut Stats,
    ) -> Frontier {
        self.reset(start, table);

        let mut expansions = 0;
        loop {
            let Some(node) = self.peek() else {
                return Frontier::Exhausted;
            };

            if expansions == self.limit {
                return Frontier::Target(node);
            }

            let mut selected = core::mem::take(&mut self.selected);
            selected.clear();
            while selected.len() < self.batch && expansions + selected.len() < self.limit {
                let Some(node) = self.peek() else {
                    break;
                };

                if self.goal.is_goal(self.space.state(node)) {
                    self.selected = selected;
                    return Frontier::Target(node);
                }

                self.pop();
                selected.push(node);
            }

            for &node in &selected {
                if budget.exhausted(stats) {
                    self.selected = selected;
                    return Frontier::Limit;
                }

                self.expand(node, table, stats);
                expansions += 1;
            }

            self.selected = selected;
        }
    }

    /// Clears the previous search graph and opens `start`.
    fn reset<H: Heuristic<G::State, G::Cost>>(
        &mut self,
        start: G::State,
        table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    ) {
        self.h.clear();
        self.status.clear();
        self.children.clear();
        self.edges.clear();
        self.open.clear();

        let h = table.get(&start);
        self.space = SearchSpace::seed([start], self.goal).0;
        self.create(0, h);
    }

    /// The open node with the smallest key, left on the open list.
    fn peek(&mut self) -> Option<NodeId> {
        let (statuses, space) = (&self.status, &self.space);

        self.open
            .peek_where(|node, g| statuses[node] == Status::Open && g == space.g(node))
            .map(|(node, _)| node)
    }

    /// Takes the node [`Lookahead::peek`] returns off the open list; it stays
    /// [`Status::Open`] until it is expanded.
    fn pop(&mut self) {
        let (statuses, space) = (&self.status, &self.space);
        self.open
            .pop_where(|node, g| statuses[node] == Status::Open && g == space.g(node));
    }

    /// Closes `node`, taken off the open list by [`Lookahead::pop`], and generates its
    /// successors, recording every edge.
    fn expand<H: Heuristic<G::State, G::Cost>>(
        &mut self,
        node: NodeId,
        table: &mut LearnedHeuristic<G::State, G::Cost, H>,
        stats: &mut Stats,
    ) {
        self.status[node] = Status::Closed;

        stats.expanded += 1;
        stats.generated += collect_successors(self.graph, self.space.state(node), &mut self.buffer);
        self.children[node] = !self.buffer.is_empty();

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, next, cost) in buffer.drain(..) {
            self.generate(node, action, next, cost, table, stats);
        }

        self.buffer = buffer;
    }

    /// Records the edge from `parent` to `next` and opens `next`, or reopens it on a cheaper
    /// path if [`Lookahead::order`] reopens.
    fn generate<H: Heuristic<G::State, G::Cost>>(
        &mut self,
        parent: NodeId,
        action: G::Action,
        next: G::State,
        cost: G::Cost,
        table: &mut LearnedHeuristic<G::State, G::Cost, H>,
        stats: &mut Stats,
    ) {
        let id = if let Some(id) = self.space.find(&next) {
            let improve = self.order.reopens();
            if let Reach::Improved(_) = self.space.reach(next, parent, action, cost, improve) {
                self.reopen(id, stats);
            }

            id
        } else {
            let h = table.get(&next);
            let Reach::New(id) = self.space.reach(next, parent, action, cost, false) else {
                unreachable!("the state is not known yet");
            };

            self.create(id, h);
            id
        };

        self.edges.push((parent, id, cost));
    }

    /// Adds the lookahead data of the new node `id` with value `h`, opening it unless it is
    /// a dead end.
    fn create(&mut self, id: NodeId, h: Option<G::Cost>) {
        let status = if h.is_some() {
            Status::Open
        } else {
            Status::Dead
        };

        self.h.push(h);
        self.status.push(status);
        self.children.push(false);
        self.push_open(id);
    }

    /// Puts `id`, whose path just got cheaper, back on the open list.
    fn reopen(&mut self, id: NodeId, stats: &mut Stats) {
        match self.status[id] {
            Status::Dead => return,
            Status::Closed => {
                stats.reopened += 1;
                self.status[id] = Status::Open;
            }
            Status::Open => {}
        }

        self.push_open(id);
    }

    /// Pushes the open entry of `id` at its current `g`, if it has a value.
    fn push_open(&mut self, id: NodeId) {
        if let Some(h) = self.h[id] {
            let g = self.space.g(id);
            self.open.push(id, self.order.key(g, h), g);
        }
    }

    /// Moves the agent along the search path to `target` and returns the number of moves.
    fn travel(&self, target: NodeId, path: &mut Trajectory<G::State, G::Action, G::Cost>) -> u64 {
        let mut links = Vec::new();
        let mut node = target;
        while let Some((parent, action, cost)) = self.space.parent(node) {
            links.push((action.clone(), node, cost));
            node = parent;
        }

        let moves = links.len() as u64;
        for (action, node, cost) in links.into_iter().rev() {
            path.travel(action, self.space.state(node).clone(), cost);
        }

        moves
    }
}
