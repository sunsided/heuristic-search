//! Flow fields ([`FlowField`]): one backward Dijkstra search from a goal set that settles
//! the cheapest cost to the goals and the next step for every state it reaches, so any
//! number of agents heading to the same goals share one search.

use core::hash::Hash;

use search_core::{Budget, Cost};

use crate::best_first::{Workspace, dijkstra_multi_in};
use crate::outcome::{Outcome, Solution, Stats};
use crate::problem::{Predecessors, Successors};
use crate::space::NodeId;

/// The cheapest cost to a goal set and the first step towards it, for every state from
/// which a goal is reachable: a distance field from a backward Dijkstra search.
///
/// [`FlowField::build`] searches backwards from the goals over [`Predecessors`] until
/// every state that reaches a goal is settled, or the budget runs out. Then
/// [`FlowField::cost`], [`FlowField::next`] and [`FlowField::path`] answer for any start
/// without another search. The steps are edges of the forward graph: by the
/// [`Predecessors`] contract, the action and cost of [`FlowField::next`] are those of a
/// successor edge of the state. Following `next` from any state reaches a goal along a
/// cheapest path.
///
/// A field stopped by its budget answers only for the states it settled: those are the
/// states cheapest to reach the goals, and their costs and steps are final.
///
/// # Examples
///
/// ```
/// use heuristic_search::{FlowField, Limits, Predecessors};
///
/// /// A directed graph as `(from, to, cost)` triples; the action is the target.
/// struct Edges(Vec<(u8, u8, u32)>);
///
/// impl Predecessors for Edges {
///     type Node = u8;
///     type Action = u8;
///     type Cost = u32;
///
///     fn predecessors(&self, node: &u8, mut emit: impl FnMut(u8, u8, u32)) {
///         self.0.iter().filter(|e| e.1 == *node).for_each(|e| emit(e.1, e.0, e.2));
///     }
/// }
///
/// let graph = Edges(vec![(0, 1, 1), (1, 3, 1), (0, 2, 1), (2, 3, 5), (4, 0, 2)]);
/// let field = FlowField::build(&graph, [3], Limits::NONE);
///
/// assert!(field.is_complete());
/// assert_eq!(field.cost(&4), Some(4));
/// assert_eq!(field.next(&0), Some((&1, &1, 1)));
/// assert_eq!(field.path(&4).map(|path| path.states), Some(vec![4, 0, 1, 3]));
/// assert_eq!(field.cost(&2), Some(5));
/// assert_eq!(field.cost(&3), Some(0));
/// assert_eq!(field.next(&3), None, "a goal has no next step");
/// assert_eq!(field.cost(&9), None, "9 is not in the graph");
/// ```
pub struct FlowField<S, A, C> {
    /// The finished backward search: its parent links point towards the goals.
    work: Workspace<S, A, C>,
    /// Counters of the backward search.
    stats: Stats,
    /// Whether the search settled every state that reaches a goal.
    complete: bool,
}

impl<S, A, C> FlowField<S, A, C>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
{
    /// Searches backwards from `goals` over `graph` until every state that reaches a goal
    /// is settled or `budget` is used up.
    ///
    /// Every goal has cost zero; repeated goals count once. An empty goal set gives an
    /// empty field.
    pub fn build<G>(
        graph: &G,
        goals: impl IntoIterator<Item = S>,
        budget: impl Budget<Stats>,
    ) -> Self
    where
        G: Predecessors<Node = S, Action = A, Cost = C>,
    {
        let mut work = Workspace::new();
        let never = |_: &S| false;
        let result = dijkstra_multi_in(&mut work, &Backward(graph), goals, &never, budget);

        Self {
            work,
            stats: result.stats,
            complete: matches!(result.outcome, Outcome::Unsolvable),
        }
    }

    /// The cheapest cost from `state` to a goal, or `None` if no goal is reachable from
    /// it (or the field did not settle it).
    #[must_use]
    pub fn cost(&self, state: &S) -> Option<C> {
        self.settled(state).map(|id| self.work.space().g(id))
    }

    /// The first edge of a cheapest path from `state` to a goal: its action, its target
    /// and its cost. `None` for a goal and for states the field does not answer for.
    #[must_use]
    pub fn next(&self, state: &S) -> Option<(&A, &S, C)> {
        let space = self.work.space();
        let (parent, action, cost) = space.parent(self.settled(state)?)?;

        Some((action, space.state(parent), cost))
    }

    /// A cheapest path from `start` to a goal, following [`FlowField::next`], or `None` as
    /// for [`FlowField::cost`].
    #[must_use]
    pub fn path(&self, start: &S) -> Option<Solution<S, A, C>> {
        self.settled(start)
            .map(|id| self.work.space().walk_to_root(id))
    }

    /// Returns `true` if the search settled every state that reaches a goal, so `None`
    /// from [`FlowField::cost`] means no goal is reachable. `false` after the budget ran
    /// out.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Counters of the backward search; [`Stats::expanded`] is the number of settled
    /// states.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The node of `state` if the search settled it.
    fn settled(&self, state: &S) -> Option<NodeId> {
        let id = self.work.space().find(state)?;

        self.work.is_closed(id).then_some(id)
    }
}

/// The edges of a [`Predecessors`] graph reversed, for a forward search towards its
/// goals. The parent links of that search are forward edges.
struct Backward<'a, G>(&'a G);

impl<G: Predecessors> Successors for Backward<'_, G> {
    type State = G::Node;
    type Action = G::Action;
    type Cost = G::Cost;

    fn successors(&self, node: &G::Node, emit: impl FnMut(G::Action, G::Node, G::Cost)) {
        self.0.predecessors(node, emit);
    }
}
