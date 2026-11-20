//! Bidirectional breadth-first search ([`bidirectional_bfs`]).

use core::hash::Hash;

use alloc::vec::Vec;

use search_core::{Budget, Cost, Heuristic, Zero};

use crate::best_first::GraphResult;
use crate::bidirectional::{BidirectionalGraph, Core, Direction};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::{Predecessors, Successors};
use crate::space::NodeId;

/// Bidirectional breadth-first search: the path with the fewest edges.
///
/// Grows the smaller of the two frontiers by one whole layer at a time and stops after the
/// first layer that meets the other side, returning the shortest of the meets found in it.
/// Edge costs are ignored for the search; the reported cost is the sum along the returned
/// path. On graphs with a large branching factor it expands far fewer states than
/// [`bfs`](crate::bfs()).
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, bidirectional_bfs};
///
/// /// Numbers; doubling or adding one.
/// struct Numbers;
///
/// impl Successors for Numbers {
///     type State = u32;
///     type Action = char;
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut(char, u32, u32)) {
///         emit('*', n * 2, 1);
///         emit('+', n + 1, 1);
///     }
/// }
///
/// impl Predecessors for Numbers {
///     type Node = u32;
///     type Action = char;
///     type Cost = u32;
///
///     fn predecessors(&self, n: &u32, mut emit: impl FnMut(char, u32, u32)) {
///         if n % 2 == 0 && *n > 0 {
///             emit('*', n / 2, 1);
///         }
///         if let Some(previous) = n.checked_sub(1) {
///             emit('+', previous, 1);
///         }
///     }
/// }
///
/// let result = bidirectional_bfs(&Numbers, 1, [24], Limits::NONE);
/// let solution = result.into_solution().expect("24 is reachable");
///
/// assert_eq!(solution.len(), 5);
/// assert_eq!(solution.goal(), &24);
/// ```
pub fn bidirectional_bfs<G>(
    graph: &G,
    start: G::State,
    goals: impl IntoIterator<Item = G::State>,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
{
    let unit = Unit(graph);
    let mut core = Core::new(&unit, start, goals, Zero, Zero);
    let mut frontiers = [Vec::new(), Vec::new()];
    for direction in [Direction::Forward, Direction::Backward] {
        frontiers[direction.index()] = (0..core.side(direction).space.len()).collect();
    }

    while core.meet.is_none() && frontiers.iter().all(|frontier| !frontier.is_empty()) {
        let direction = if frontiers[0].len() <= frontiers[1].len() {
            Direction::Forward
        } else {
            Direction::Backward
        };

        let layer = core::mem::take(&mut frontiers[direction.index()]);
        let Some(next) = expand_layer(&mut core, direction, &layer, &mut budget) else {
            return weighted(core.finish(false));
        };

        frontiers[direction.index()] = next;
    }

    weighted(core.finish(true))
}

/// Expands every node of `layer` on `direction`'s side and returns the next layer, or
/// `None` if the budget ran out.
fn expand_layer<G, HF, HB>(
    core: &mut Core<'_, G, HF, HB>,
    direction: Direction,
    layer: &[NodeId],
    budget: &mut impl Budget<Stats>,
) -> Option<Vec<NodeId>>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
{
    let mut next = Vec::new();
    for &node in layer {
        if budget.exhausted(&core.stats) {
            return None;
        }

        core.expand(direction, node, false, &mut next);
    }

    Some(next)
}

/// Converts a result over [`Unit`] back to the graph's actions and costs.
fn weighted<S, A, C: Cost>(result: SearchResult<S, (A, C), u64>) -> SearchResult<S, A, C> {
    let outcome = match result.outcome {
        Outcome::Found(solution) => {
            let (actions, costs): (Vec<A>, Vec<C>) = solution.actions.into_iter().unzip();
            Outcome::Found(Solution {
                states: solution.states,
                actions,
                cost: costs.into_iter().fold(C::ZERO, |sum, cost| sum + cost),
            })
        }
        Outcome::Unsolvable => Outcome::Unsolvable,
        Outcome::LimitReached => Outcome::LimitReached,
    };

    SearchResult {
        outcome,
        stats: result.stats,
    }
}

/// A view of a graph where every edge costs `1`, so path costs count edges. The real edge
/// cost travels in the action, for [`weighted`].
struct Unit<'a, G>(&'a G);

impl<G: Successors> Successors for Unit<'_, G> {
    type State = G::State;
    type Action = (<G as Successors>::Action, <G as Successors>::Cost);
    type Cost = u64;

    fn successors(&self, state: &G::State, mut emit: impl FnMut(Self::Action, G::State, u64)) {
        self.0
            .successors(state, |action, next, cost| emit((action, cost), next, 1));
    }
}

impl<G: Predecessors> Predecessors for Unit<'_, G> {
    type Node = G::Node;
    type Action = (<G as Predecessors>::Action, <G as Predecessors>::Cost);
    type Cost = u64;

    fn predecessors(&self, node: &G::Node, mut emit: impl FnMut(Self::Action, G::Node, u64)) {
        self.0.predecessors(node, |action, previous, cost| {
            emit((action, cost), previous, 1);
        });
    }
}
