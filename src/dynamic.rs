//! Incremental search on graphs whose edge costs change: [`LpaStar`] (Lifelong Planning
//! A*), [`DStarLite`] and [`AdStar`] (Anytime D*), over a [`DynamicGraph`].
//!
//! Both keep their search tree between calls. After the caller changes edge costs in the
//! graph, it reports each changed edge (`edge_changed`) and asks for a new path
//! (`compute`); only the part of the tree the changes affect is repaired. Edges are
//! identified by their endpoints; parallel edges count with their cheapest cost.
//!
//! | Planner        | Searches          | Fixed       | Can move     |
//! | -------------- | ----------------- | ----------- | ------------ |
//! | [`LpaStar`]    | start to goal     | both ends   | nothing      |
//! | [`DStarLite`]  | goal to start     | the goal    | the start    |
//! | [`AdStar`]     | goal to start     | the goal    | the start    |
//!
//! [`AdStar`] is the anytime variant of [`DStarLite`]: it inflates the heuristic by a
//! weight the caller lowers over time. [`LpaStarMulti`], [`DStarLiteMulti`] and
//! [`AdStarMulti`] take several fixed starts, through a virtual super-start.
//!
//! Both need strictly positive edge costs, as in the papers: a zero-cost cycle can keep
//! supporting its own stale distances after the edge that made them real disappears. And
//! both need a consistent heuristic `h(a, b)` that estimates the cost from `a` to `b`, for
//! the costs at every point in time; `|_, _| 0` always works.

mod incremental;
mod multi;

use core::hash::Hash;

use search_core::{Budget, Cost};

use crate::cost::Weight;
use crate::outcome::{SearchResult, Stats};
use incremental::Incremental;
pub use multi::{AdStarMulti, DStarLiteMulti, LpaStarMulti};

/// A graph with stable vertices whose edge costs may change between searches.
///
/// Both directions must agree and reflect the current costs: `to` is reported as a
/// successor of `from` with cost `c` exactly when `from` is reported as a predecessor of
/// `to` with cost `c`.
pub trait DynamicGraph {
    /// A vertex. Cloned and hashed into the planner's tables.
    type Vertex: Clone + Hash + Eq;
    /// An edge cost. Must be strictly positive: [`LpaStar`] and [`DStarLite`] can keep
    /// stale distances alive around a zero-cost cycle.
    type Cost: Cost;

    /// Calls `emit(to, cost)` for every edge leaving `vertex`.
    fn successors(&self, vertex: &Self::Vertex, emit: impl FnMut(Self::Vertex, Self::Cost));

    /// Calls `emit(from, cost)` for every edge entering `vertex`.
    fn predecessors(&self, vertex: &Self::Vertex, emit: impl FnMut(Self::Vertex, Self::Cost));
}

/// The result of an incremental search: a path of vertices. Edges carry no labels, so the
/// solution's actions are `()`.
pub type DynamicResult<G> =
    SearchResult<<G as DynamicGraph>::Vertex, (), <G as DynamicGraph>::Cost>;

/// Lifelong Planning A* (LPA*, Koenig, Likhachev and Furcy 2004): A* that repairs its
/// search tree after edge-cost changes instead of starting over.
///
/// The first [`LpaStar::compute`] costs about as much as A*. After the caller changes the
/// graph and reports every changed edge with [`LpaStar::edge_changed`], the next
/// `compute` re-expands only vertices whose distance from the start may have changed.
/// Returns cheapest paths with a consistent `heuristic`, where `heuristic(v, goal)`
/// estimates the cost from `v` to the goal.
///
/// # Examples
///
/// ```
/// use heuristic_search::{DynamicGraph, Limits, LpaStar};
///
/// /// A directed graph as `(from, to, cost)` triples.
/// struct Edges(Vec<(u8, u8, u32)>);
///
/// impl DynamicGraph for Edges {
///     type Vertex = u8;
///     type Cost = u32;
///
///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
///         self.0.iter().filter(|e| e.0 == *v).for_each(|e| emit(e.1, e.2));
///     }
///
///     fn predecessors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
///         self.0.iter().filter(|e| e.1 == *v).for_each(|e| emit(e.0, e.2));
///     }
/// }
///
/// let mut graph = Edges(vec![(0, 1, 1), (1, 3, 1), (0, 2, 2), (2, 3, 2)]);
/// let mut planner = LpaStar::new(&graph, 0, 3, |_: &u8, _: &u8| 0);
/// assert_eq!(planner.compute(&graph, Limits::NONE).solution().map(|s| s.cost), Some(2));
///
/// // The edge 1 -> 3 gets expensive; report it and repair.
/// graph.0[1].2 = 10;
/// planner.edge_changed(&graph, &1, &3);
/// let solution = planner.compute(&graph, Limits::NONE).into_solution().expect("3 is reachable");
/// assert_eq!(solution.states, [0, 2, 3]);
/// assert_eq!(solution.cost, 4);
/// ```
pub struct LpaStar<V, C, D> {
    /// The incremental search from the start.
    core: Incremental<V, C, D>,
}

impl<V, C, D> LpaStar<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares a search from `start` to `goal`. Nothing is expanded until
    /// [`LpaStar::compute`].
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        start: V,
        goal: V,
        heuristic: D,
    ) -> Self {
        Self {
            core: Incremental::new(true, start, goal, heuristic, None),
        }
    }

    /// Repairs the search tree after the latest changes and returns a cheapest path, or
    /// [`Outcome::Unsolvable`](crate::Outcome::Unsolvable) if the goal is unreachable.
    ///
    /// The budget and the returned stats cover this call only; see [`LpaStar::stats`] for
    /// the totals. A search stopped by its budget continues where it stopped on the next
    /// call.
    pub fn compute<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        budget: impl Budget<Stats>,
    ) -> DynamicResult<G> {
        self.core.compute(graph, budget)
    }

    /// Reports that the edge `from -> to` changed its cost, appeared or disappeared in
    /// `graph`, which must already show the new state.
    pub fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        from: &V,
        to: &V,
    ) {
        self.core.edge_changed(graph, from, to);
    }

    /// Counters over every [`LpaStar::compute`] so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.core.total
    }
}

/// D* Lite (Koenig and Likhachev 2002): incremental search towards a fixed goal from a
/// start that moves, as when a robot replans while it drives and discovers obstacles.
///
/// Searches backwards from the goal, so the tree stays valid as the start moves. Move the
/// start with [`DStarLite::move_start`], report changed edges with
/// [`DStarLite::edge_changed`], then [`DStarLite::compute`] a path from the current start.
/// Returns cheapest paths with a consistent `heuristic`, where `heuristic(start, v)`
/// estimates the cost from the start to `v`.
///
/// # Examples
///
/// ```
/// use heuristic_search::{DStarLite, DynamicGraph, Limits};
///
/// /// A directed graph as `(from, to, cost)` triples.
/// struct Edges(Vec<(u8, u8, u32)>);
///
/// impl DynamicGraph for Edges {
///     type Vertex = u8;
///     type Cost = u32;
///
///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
///         self.0.iter().filter(|e| e.0 == *v).for_each(|e| emit(e.1, e.2));
///     }
///
///     fn predecessors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
///         self.0.iter().filter(|e| e.1 == *v).for_each(|e| emit(e.0, e.2));
///     }
/// }
///
/// let mut graph = Edges(vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (1, 3, 5)]);
/// let mut planner = DStarLite::new(&graph, 0, 3, |_: &u8, _: &u8| 0);
/// assert_eq!(planner.compute(&graph, Limits::NONE).solution().map(|s| s.cost), Some(3));
///
/// // Drive to 1, then find 2 -> 3 blocked.
/// planner.move_start(1);
/// graph.0.remove(2);
/// planner.edge_changed(&graph, &2, &3);
/// let solution = planner.compute(&graph, Limits::NONE).into_solution().expect("3 is reachable");
/// assert_eq!(solution.states, [1, 3]);
/// assert_eq!(solution.cost, 5);
/// ```
pub struct DStarLite<V, C, D> {
    /// The incremental search from the goal.
    core: Incremental<V, C, D>,
}

impl<V, C, D> DStarLite<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares a search from `start` to `goal`. Nothing is expanded until
    /// [`DStarLite::compute`].
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        start: V,
        goal: V,
        heuristic: D,
    ) -> Self {
        Self {
            core: Incremental::new(false, goal, start, heuristic, None),
        }
    }

    /// Repairs the search tree after the latest changes and returns a cheapest path from
    /// the current start, or [`Outcome::Unsolvable`](crate::Outcome::Unsolvable).
    ///
    /// The budget and the returned stats cover this call only; see [`DStarLite::stats`]
    /// for the totals.
    pub fn compute<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        budget: impl Budget<Stats>,
    ) -> DynamicResult<G> {
        self.core.compute(graph, budget)
    }

    /// Moves the start to `start`, typically one step along the last path.
    pub fn move_start(&mut self, start: V) {
        self.core.move_target(start);
    }

    /// Reports that the edge `from -> to` changed its cost, appeared or disappeared in
    /// `graph`, which must already show the new state.
    pub fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        from: &V,
        to: &V,
    ) {
        self.core.edge_changed(graph, from, to);
    }

    /// Counters over every [`DStarLite::compute`] so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.core.total
    }
}

/// Anytime Dynamic A* (AD*, Likhachev, Ferguson, Gordon, Stentz and Thrun 2005): D* Lite
/// with an inflated heuristic, so a first path comes fast and improves as time allows,
/// while the start moves and edges change.
///
/// Each [`AdStar::compute`] is one iteration: it returns a path that costs at most the
/// current weight times the optimum (with a consistent `heuristic`). Lower the weight with
/// [`AdStar::set_weight`] between calls to improve the path, ending with
/// [`Weight::ONE`](crate::Weight::ONE) for an optimal one; move the start and report
/// changed edges as with [`DStarLite`]. After large changes, raising the weight again
/// gets a new path faster. Vertices that become inconsistent after their expansion in an
/// iteration wait for the next one (AD*'s `INCONS`). Start moves shift keys by D* Lite's
/// `km`; only lowering the weight re-keys the whole queue, which is linear in the vertices
/// seen so far.
///
/// # Panics
///
/// [`AdStar::new`] and [`AdStar::set_weight`] panic if the weight is below `1`.
///
/// # Examples
///
/// ```
/// use heuristic_search::{AdStar, DynamicGraph, Limits, Weight};
///
/// /// A line `0..=9`; each step moves one to the left or right at cost 1.
/// struct Line;
///
/// impl DynamicGraph for Line {
///     type Vertex = u8;
///     type Cost = u32;
///
///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
///         if *v > 0 {
///             emit(v - 1, 1);
///         }
///         if *v < 9 {
///             emit(v + 1, 1);
///         }
///     }
///
///     fn predecessors(&self, v: &u8, emit: impl FnMut(u8, u32)) {
///         self.successors(v, emit);
///     }
/// }
///
/// let distance = |a: &u8, b: &u8| u32::from(a.abs_diff(*b));
/// let mut planner = AdStar::new(&Line, 2, 9, distance, Weight::integer(3));
/// assert_eq!(planner.compute(&Line, Limits::NONE).solution().map(|s| s.cost), Some(7));
///
/// planner.set_weight(Weight::ONE);
/// assert_eq!(planner.compute(&Line, Limits::NONE).solution().map(|s| s.cost), Some(7));
/// ```
pub struct AdStar<V, C, D> {
    /// The incremental search from the goal, with AD*'s anytime state.
    core: Incremental<V, C, D>,
}

impl<V, C, D> AdStar<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares a search from `start` to `goal` with initial inflation `weight`. Nothing
    /// is expanded until [`AdStar::compute`].
    ///
    /// # Panics
    ///
    /// Panics if `weight` is below `1`.
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        start: V,
        goal: V,
        heuristic: D,
        weight: Weight,
    ) -> Self {
        assert_weight(weight);

        Self {
            core: Incremental::new(false, goal, start, heuristic, Some(weight)),
        }
    }

    /// Runs one iteration with the current weight and returns a path from the current
    /// start within that weight of the optimum, or
    /// [`Outcome::Unsolvable`](crate::Outcome::Unsolvable).
    ///
    /// The budget and the returned stats cover this call only; see [`AdStar::stats`] for
    /// the totals.
    pub fn compute<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        budget: impl Budget<Stats>,
    ) -> DynamicResult<G> {
        self.core.compute(graph, budget)
    }

    /// Sets the inflation for the next iterations.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is below `1`.
    pub fn set_weight(&mut self, weight: Weight) {
        assert_weight(weight);
        self.core.set_weight(weight);
    }

    /// The current inflation.
    #[must_use]
    pub fn weight(&self) -> Weight {
        self.core.weight()
    }

    /// Moves the start to `start`, typically one step along the last path.
    pub fn move_start(&mut self, start: V) {
        self.core.move_target(start);
    }

    /// Reports that the edge `from -> to` changed its cost, appeared or disappeared in
    /// `graph`, which must already show the new state.
    pub fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        from: &V,
        to: &V,
    ) {
        self.core.edge_changed(graph, from, to);
    }

    /// Counters over every [`AdStar::compute`] so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.core.total
    }
}

/// Panics unless `weight` is at least `1`.
pub(crate) fn assert_weight(weight: Weight) {
    assert!(
        weight.numerator() >= weight.denominator(),
        "AD* weight must be at least 1"
    );
}
