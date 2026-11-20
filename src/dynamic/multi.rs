//! Several starts for the incremental planners: [`LpaStarMulti`], [`DStarLiteMulti`] and
//! [`AdStarMulti`] search from a virtual super-start joined to every start by a zero-cost
//! edge, and strip it from the returned path.
//!
//! The starts are fixed for the planner's lifetime: the super-start's heuristic is the
//! minimum over the starts, so changing them would change keys already queued. Edge changes
//! are reported as for the single-start planners.

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost};

use crate::cost::Weight;
use crate::dynamic::incremental::{Estimate, Incremental};
use crate::dynamic::{DynamicGraph, DynamicResult, assert_weight};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};

/// A vertex of the graph, or the virtual super-start.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Rooted<V> {
    /// The super-start, with a zero-cost edge to every start.
    Root,
    /// A vertex of the graph.
    Vertex(V),
}

/// The graph with the super-start added: edges `Root -> start` of cost zero.
struct RootedGraph<'g, G, V> {
    /// The real graph.
    graph: &'g G,
    /// The starts the super-start joins.
    starts: &'g [V],
}

impl<G, V> DynamicGraph for RootedGraph<'_, G, V>
where
    G: DynamicGraph<Vertex = V>,
    V: Clone + Hash + Eq,
{
    type Vertex = Rooted<V>;
    type Cost = G::Cost;

    fn successors(&self, vertex: &Rooted<V>, mut emit: impl FnMut(Rooted<V>, G::Cost)) {
        match vertex {
            Rooted::Root => {
                for start in self.starts {
                    emit(Rooted::Vertex(start.clone()), G::Cost::ZERO);
                }
            }
            Rooted::Vertex(vertex) => self
                .graph
                .successors(vertex, |next, cost| emit(Rooted::Vertex(next), cost)),
        }
    }

    fn predecessors(&self, vertex: &Rooted<V>, mut emit: impl FnMut(Rooted<V>, G::Cost)) {
        if let Rooted::Vertex(vertex) = vertex {
            self.graph.predecessors(vertex, |previous, cost| {
                emit(Rooted::Vertex(previous), cost);
            });
            if self.starts.contains(vertex) {
                emit(Rooted::Root, G::Cost::ZERO);
            }
        }
    }
}

/// The heuristic with the super-start: from it, the minimum over the starts (consistent if
/// the heuristic is); towards it, zero.
struct RootedH<D, V> {
    /// The heuristic between real vertices.
    inner: D,
    /// The starts, for estimates from the super-start.
    starts: Vec<V>,
}

impl<V, C: Cost, D: FnMut(&V, &V) -> C> Estimate<Rooted<V>, C> for RootedH<D, V> {
    fn estimate(&mut self, from: &Rooted<V>, to: &Rooted<V>) -> C {
        match (from, to) {
            (Rooted::Vertex(from), Rooted::Vertex(to)) => (self.inner)(from, to),
            (Rooted::Root, Rooted::Vertex(to)) => {
                let inner = &mut self.inner;
                self.starts
                    .iter()
                    .map(|start| inner(start, to))
                    .min()
                    .unwrap_or(C::ZERO)
            }
            (_, Rooted::Root) => C::ZERO,
        }
    }
}

/// The incremental search over the rooted graph, shared by the three planners.
struct Rooting<V, C, D> {
    /// The search; its source or target is the super-start.
    core: Incremental<Rooted<V>, C, RootedH<D, V>>,
    /// The starts, deduplicated, in order of first appearance.
    starts: Vec<V>,
}

impl<V, C, D> Rooting<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares the search: `forward` for LPA* (from the super-start), else D* Lite or
    /// AD* (towards it), with `weight` for AD*.
    fn new(
        forward: bool,
        starts: impl IntoIterator<Item = V>,
        goal: V,
        heuristic: D,
        weight: Option<Weight>,
    ) -> Self {
        let mut unique: Vec<V> = Vec::new();
        for start in starts {
            if !unique.contains(&start) {
                unique.push(start);
            }
        }

        let estimate = RootedH {
            inner: heuristic,
            starts: unique.clone(),
        };
        let goal = Rooted::Vertex(goal);
        let mut core = if forward {
            Incremental::new(true, Rooted::Root, goal, estimate, weight)
        } else {
            Incremental::new(false, goal, Rooted::Root, estimate, weight)
        };
        // The super-start's edges cost zero, so its key can tie with a start's.
        core.settle_ties = true;

        Self {
            core,
            starts: unique,
        }
    }

    /// Repairs the tree and returns a path from the best start.
    fn compute<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        budget: impl Budget<Stats>,
    ) -> DynamicResult<G> {
        let rooted = RootedGraph {
            graph,
            starts: &self.starts,
        };

        unroot(self.core.compute(&rooted, budget))
    }

    /// Reports a changed edge of the real graph.
    fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(&mut self, graph: &G, from: &V, to: &V) {
        let rooted = RootedGraph {
            graph,
            starts: &self.starts,
        };
        let (from, to) = (Rooted::Vertex(from.clone()), Rooted::Vertex(to.clone()));

        self.core.edge_changed(&rooted, &from, &to);
    }
}

/// The result without the super-start at the head of the path.
fn unroot<V, C>(result: SearchResult<Rooted<V>, (), C>) -> SearchResult<V, (), C> {
    let outcome = match result.outcome {
        Outcome::Found(solution) => {
            let states: Vec<V> = solution
                .states
                .into_iter()
                .filter_map(|vertex| match vertex {
                    Rooted::Vertex(vertex) => Some(vertex),
                    Rooted::Root => None,
                })
                .collect();

            Outcome::Found(Solution {
                actions: alloc::vec![(); states.len().saturating_sub(1)],
                states,
                cost: solution.cost,
            })
        }
        Outcome::Unsolvable => Outcome::Unsolvable,
        Outcome::LimitReached => Outcome::LimitReached,
        Outcome::Cutoff => Outcome::Cutoff,
    };

    SearchResult {
        outcome,
        stats: result.stats,
    }
}

/// Generates a multi-start planner type over [`Rooting`]; `$forward` picks LPA*.
macro_rules! multi_planner {
    ($(#[$doc:meta])* $name:ident, $single:literal, $forward:expr) => {
        $(#[$doc])*
        pub struct $name<V, C, D> {
            /// The rooted search.
            rooting: Rooting<V, C, D>,
        }

        impl<V, C, D> $name<V, C, D>
        where
            V: Clone + Hash + Eq,
            C: Cost,
            D: FnMut(&V, &V) -> C,
        {
            #[doc = concat!("Repairs the tree after the latest changes and returns a path from the best start, as [`", $single, "`] does for one.")]
            pub fn compute<G: DynamicGraph<Vertex = V, Cost = C>>(
                &mut self,
                graph: &G,
                budget: impl Budget<Stats>,
            ) -> DynamicResult<G> {
                self.rooting.compute(graph, budget)
            }

            /// Reports that the edge `from -> to` changed its cost, appeared or disappeared
            /// in `graph`, which must already show the new state.
            pub fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(
                &mut self,
                graph: &G,
                from: &V,
                to: &V,
            ) {
                self.rooting.edge_changed(graph, from, to);
            }

            /// Counters over every `compute` so far.
            #[must_use]
            pub fn stats(&self) -> Stats {
                self.rooting.core.total
            }

            /// The starts, deduplicated, in order of first appearance.
            #[must_use]
            pub fn starts(&self) -> &[V] {
                &self.rooting.starts
            }
        }
    };
}

multi_planner!(
    /// [`LpaStar`](crate::LpaStar) from several fixed starts to one goal: the cheapest path
    /// from any start, repaired after edge changes.
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{DynamicGraph, Limits, LpaStarMulti};
    ///
    /// /// A line `0 - 1 - ... - 9` with unit edges both ways.
    /// struct Line;
    ///
    /// impl DynamicGraph for Line {
    ///     type Vertex = u8;
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
    ///         if *v < 9 {
    ///             emit(v + 1, 1);
    ///         }
    ///         if *v > 0 {
    ///             emit(v - 1, 1);
    ///         }
    ///     }
    ///
    ///     fn predecessors(&self, v: &u8, emit: impl FnMut(u8, u32)) {
    ///         self.successors(v, emit);
    ///     }
    /// }
    ///
    /// let distance = |a: &u8, b: &u8| u32::from(a.abs_diff(*b));
    /// let mut planner = LpaStarMulti::new(&Line, [0, 7, 2], 9, distance);
    /// let path = planner.compute(&Line, Limits::NONE).into_solution().expect("reachable");
    ///
    /// assert_eq!(path.states, [7, 8, 9]);
    /// assert_eq!(path.cost, 2);
    /// ```
    LpaStarMulti,
    "crate::LpaStar::compute",
    true
);

multi_planner!(
    /// [`DStarLite`](crate::DStarLite) from several fixed starts: the search runs from the
    /// goal towards a super-start joined to every start, so it repairs the cheapest path
    /// from any start after edge changes. The starts cannot move.
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{DStarLiteMulti, DynamicGraph, Limits};
    ///
    /// /// A line `0 - 1 - ... - 9` with unit edges both ways.
    /// struct Line;
    ///
    /// impl DynamicGraph for Line {
    ///     type Vertex = u8;
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
    ///         if *v < 9 {
    ///             emit(v + 1, 1);
    ///         }
    ///         if *v > 0 {
    ///             emit(v - 1, 1);
    ///         }
    ///     }
    ///
    ///     fn predecessors(&self, v: &u8, emit: impl FnMut(u8, u32)) {
    ///         self.successors(v, emit);
    ///     }
    /// }
    ///
    /// let distance = |a: &u8, b: &u8| u32::from(a.abs_diff(*b));
    /// let mut planner = DStarLiteMulti::new(&Line, [0, 4], 6, distance);
    /// let path = planner.compute(&Line, Limits::NONE).into_solution().expect("reachable");
    ///
    /// assert_eq!(path.states, [4, 5, 6]);
    /// ```
    DStarLiteMulti,
    "crate::DStarLite::compute",
    false
);

multi_planner!(
    /// [`AdStar`](crate::AdStar) from several fixed starts: each `compute` returns a path
    /// from the best start within the current weight of the optimum. The starts cannot
    /// move.
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{AdStarMulti, DynamicGraph, Limits, Weight};
    ///
    /// /// A line `0 - 1 - ... - 9` with unit edges both ways.
    /// struct Line;
    ///
    /// impl DynamicGraph for Line {
    ///     type Vertex = u8;
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
    ///         if *v < 9 {
    ///             emit(v + 1, 1);
    ///         }
    ///         if *v > 0 {
    ///             emit(v - 1, 1);
    ///         }
    ///     }
    ///
    ///     fn predecessors(&self, v: &u8, emit: impl FnMut(u8, u32)) {
    ///         self.successors(v, emit);
    ///     }
    /// }
    ///
    /// let distance = |a: &u8, b: &u8| u32::from(a.abs_diff(*b));
    /// let mut planner = AdStarMulti::new(&Line, [0, 4], 6, distance, Weight::integer(2));
    /// assert!(planner.compute(&Line, Limits::NONE).solution().is_some());
    ///
    /// planner.set_weight(Weight::ONE);
    /// let path = planner.compute(&Line, Limits::NONE).into_solution().expect("reachable");
    /// assert_eq!(path.cost, 2);
    /// ```
    AdStarMulti,
    "crate::AdStar::compute",
    false
);

impl<V, C, D> LpaStarMulti<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares a search from `starts` to `goal`. Nothing is expanded until
    /// [`LpaStarMulti::compute`]. No starts means the goal is unreachable.
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        starts: impl IntoIterator<Item = V>,
        goal: V,
        heuristic: D,
    ) -> Self {
        Self {
            rooting: Rooting::new(true, starts, goal, heuristic, None),
        }
    }
}

impl<V, C, D> DStarLiteMulti<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares a search from `starts` to `goal`. Nothing is expanded until
    /// [`DStarLiteMulti::compute`]. No starts means the goal is unreachable.
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        starts: impl IntoIterator<Item = V>,
        goal: V,
        heuristic: D,
    ) -> Self {
        Self {
            rooting: Rooting::new(false, starts, goal, heuristic, None),
        }
    }
}

impl<V, C, D> AdStarMulti<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Prepares a search from `starts` to `goal` with initial inflation `weight`.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is below `1`.
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        starts: impl IntoIterator<Item = V>,
        goal: V,
        heuristic: D,
        weight: Weight,
    ) -> Self {
        assert_weight(weight);

        Self {
            rooting: Rooting::new(false, starts, goal, heuristic, Some(weight)),
        }
    }

    /// Sets the inflation for the next iterations.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is below `1`.
    pub fn set_weight(&mut self, weight: Weight) {
        assert_weight(weight);
        self.rooting.core.set_weight(weight);
    }

    /// The current inflation.
    #[must_use]
    pub fn weight(&self) -> Weight {
        self.rooting.core.weight()
    }
}
