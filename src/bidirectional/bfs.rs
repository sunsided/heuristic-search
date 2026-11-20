//! Bidirectional breadth-first search ([`bidirectional_bfs`], [`bidirectional_bfs_meet`]).

use core::hash::Hash;

use alloc::vec::Vec;

use search_core::{Budget, Cost, Heuristic, Zero};

use crate::best_first::GraphResult;
use crate::bidirectional::mixed::{Mixed, MixedGraph, Split, unmix};
use crate::bidirectional::{
    BidirectionalGraph, Core, Direction, Equality, MeetRelation, MeetResult, Meeting, Meets,
    RegressionGraph,
};
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
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
{
    bidirectional_bfs_multi(graph, [start], goals, budget)
}

/// [`bidirectional_bfs`] from several start states at once.
///
/// Every start has path cost zero on the forward side, just as every goal has on the
/// backward side. Repeated starts count once, an empty set of starts finds nothing, and a
/// start that is a goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, bidirectional_bfs_multi};
///
/// /// Counting up by one at cost 1, and back down for the backward search.
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
/// impl Predecessors for Up {
///     type Node = u32;
///     type Action = ();
///     type Cost = u32;
///
///     fn predecessors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
///         if let Some(previous) = n.checked_sub(1) {
///             emit((), previous, 1);
///         }
///     }
/// }
///
/// let result = bidirectional_bfs_multi(&Up, [0, 6, 3], [8], Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn bidirectional_bfs_multi<G>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goals: impl IntoIterator<Item = G::State>,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
{
    let unit = Unit(graph);
    let core = Core::new(
        &unit,
        starts,
        goals,
        Zero,
        Zero,
        None::<Meeting<Unit<'_, G>, Equality>>,
    );

    weighted(layers(core, budget))
}

/// [`bidirectional_bfs`] that meets by a [`MeetRelation`] instead of state equality, from
/// several starts at once: the path with the fewest edges through a meet.
///
/// `goals` are backward nodes of the graph's [`Predecessors::Node`] type, such as
/// subgoals, and `meet` decides which forward states meet which backward nodes. The path is
/// rebuilt by replaying the backward actions forwards through the first successor edge
/// with an equal label and cost, so every state on it is real.
///
/// # Errors
///
/// [`ReplayError`](crate::ReplayError) if a replay step finds no such edge.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, bidirectional_bfs_meet, meet_fn};
///
/// /// Three facts as bits; action `k` makes fact `k` true at cost 1.
/// struct Facts;
///
/// /// A subgoal: the facts that must hold.
/// #[derive(Clone, PartialEq, Eq, Hash)]
/// struct Subgoal(u8);
///
/// impl Successors for Facts {
///     type State = u8;
///     type Action = u8;
///     type Cost = u32;
///
///     fn successors(&self, state: &u8, mut emit: impl FnMut(u8, u8, u32)) {
///         for k in 0..3 {
///             if state & (1 << k) == 0 {
///                 emit(k, state | (1 << k), 1);
///             }
///         }
///     }
/// }
///
/// impl Predecessors for Facts {
///     type Node = Subgoal;
///     type Action = u8;
///     type Cost = u32;
///
///     fn predecessors(&self, subgoal: &Subgoal, mut emit: impl FnMut(u8, Subgoal, u32)) {
///         for k in 0..3 {
///             if subgoal.0 & (1 << k) != 0 {
///                 emit(k, Subgoal(subgoal.0 & !(1 << k)), 1);
///             }
///         }
///     }
/// }
///
/// let satisfies = meet_fn(|state: &u8, subgoal: &Subgoal| state & subgoal.0 == subgoal.0);
/// let result = bidirectional_bfs_meet(&Facts, [0], [Subgoal(0b110)], satisfies, Limits::NONE);
/// let solution = result.expect("the model replays").into_solution().expect("reachable");
///
/// assert_eq!(solution.len(), 2);
/// assert_eq!(solution.goal() & 0b110, 0b110);
/// ```
pub fn bidirectional_bfs_meet<G, M>(
    graph: &G,
    starts: impl IntoIterator<Item = <G as Successors>::State>,
    goals: impl IntoIterator<Item = <G as Predecessors>::Node>,
    meet: M,
    budget: impl Budget<Stats>,
) -> MeetResult<G>
where
    G: RegressionGraph,
    <G as Successors>::State: Clone + Hash + Eq,
    <G as Predecessors>::Node: Clone + Hash + Eq,
    <G as Successors>::Action: Clone + PartialEq,
    M: MeetRelation<<G as Successors>::State, <G as Predecessors>::Node>,
{
    let mixed = MixedGraph(graph);
    let unit = Unit(&mixed);
    let core = Core::new(
        &unit,
        starts.into_iter().map(Mixed::State),
        goals.into_iter().map(Mixed::Node),
        Zero,
        Zero,
        Some(Meeting::new(Split(meet))),
    );

    unmix(weighted(layers(core, budget)))
}

/// Grows the smaller frontier a layer at a time until a layer meets the other side, a
/// side runs empty or the budget is used up.
fn layers<G, M>(
    mut core: Core<'_, G, Zero, Zero, M>,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    M: Meets<G::State>,
{
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
            return core.finish(false);
        };

        frontiers[direction.index()] = next;
    }

    core.finish(true)
}

/// Expands every node of `layer` on `direction`'s side and returns the next layer, or
/// `None` if the budget ran out.
fn expand_layer<G, HF, HB, M>(
    core: &mut Core<'_, G, HF, HB, M>,
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
    M: Meets<G::State>,
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
        Outcome::Cutoff => Outcome::Cutoff,
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
