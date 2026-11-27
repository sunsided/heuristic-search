//! Bidirectional search that meets in the middle ([`mm`], Holte, Felner, Sharon and
//! Sturtevant 2016).

use core::hash::Hash;

use search_core::{Budget, Heuristic};

use crate::best_first::GraphResult;
use crate::bidirectional::frontier::{Frontier, Rule};
use crate::bidirectional::mixed::{Both, Mixed, MixedGraph, Split, unmix};
use crate::bidirectional::{
    BidirectionalGraph, Core, Equality, MeetRelation, MeetResult, Meeting, RegressionGraph,
};
use crate::cost::CostExt;
use crate::outcome::Stats;
use crate::problem::Predecessors;
use crate::problem::Successors;

/// MM: bidirectional heuristic search that is guaranteed to meet in the middle (Holte,
/// Felner, Sharon and Sturtevant 2016).
///
/// Orders each side by `pr = max(f, 2g)` with `f = g + h`, and expands the side whose
/// smallest `pr` is smaller. With `C` the smaller of the two, it stops once the cheapest
/// meet `mu <= max(C, f_min_F, f_min_B, g_min_F + g_min_B + epsilon)`. `epsilon` is the
/// smallest edge cost, which a generic [`Cost`](crate::Cost) cannot know; it is taken as
/// zero, which is correct for every cost type but may expand more states than a positive
/// `epsilon` would.
///
/// Every state MM expands has `pr <= C*`, the optimal cost, and since `pr >= 2g` neither
/// side expands a state whose `g` exceeds `C* / 2`: the frontiers meet in the middle.
/// `forward` estimates the cost to the nearest goal, `backward` the cost from the start;
/// with admissible estimates the solution is optimal. With [`Zero`](crate::Zero) for both,
/// this is MM0, a blind bidirectional search with the same guarantee.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, heuristic_fn, mm};
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
/// impl Predecessors for Line {
///     type Node = i32;
///     type Action = i32;
///     type Cost = u32;
///
///     fn predecessors(&self, state: &i32, mut emit: impl FnMut(i32, i32, u32)) {
///         emit(1, state - 1, 1);
///         emit(-1, state + 1, 1);
///     }
/// }
///
/// let to_goal = heuristic_fn(|s: &i32| Some(9_i32.abs_diff(*s)));
/// let from_start = heuristic_fn(|s: &i32| Some(0_i32.abs_diff(*s)));
/// let result = mm(&Line, 0, [9], to_goal, from_start, Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.cost), Some(9));
/// ```
pub fn mm<G, HF, HB>(
    graph: &G,
    start: G::State,
    goals: impl IntoIterator<Item = G::State>,
    forward: HF,
    backward: HB,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    <G as Successors>::Cost: CostExt,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
{
    mm_multi(graph, [start], goals, forward, backward, budget)
}

/// [`mm`] from several start states at once.
///
/// Every start has path cost zero on the forward side, just as every goal has on the
/// backward side. Repeated starts count once, an empty set of starts finds nothing, and a
/// start that is a goal is returned as a zero-cost path. The backward heuristic must estimate
/// the cost back to the nearest start, which a heuristic towards one start overestimates.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, heuristic_fn, mm_multi};
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
/// let h = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let result = mm_multi(&Up, [0, 6, 3], [8], h, Zero, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn mm_multi<G, HF, HB>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goals: impl IntoIterator<Item = G::State>,
    forward: HF,
    backward: HB,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    <G as Successors>::Cost: CostExt,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
{
    let core = Core::new(
        graph,
        starts,
        goals,
        forward,
        backward,
        None::<Meeting<G, Equality>>,
    );

    Frontier::new(core, Rule::Mm).run(budget)
}

/// [`mm`] that meets by a [`MeetRelation`] instead of state equality, from several starts
/// at once.
///
/// `goals` are backward nodes of the graph's [`Predecessors::Node`](crate::Predecessors)
/// type, such as subgoals (partial states), and `meet` decides which forward states meet
/// which backward nodes (see [`MeetRelation`] for the contract). The returned path is
/// rebuilt from the meeting state by replaying the backward actions forwards, through the
/// first successor edge with an equal label, so every state on it is a real state and its
/// cost the true cost.
///
/// Both heuristics are evaluated on states and on backward nodes, so they implement
/// [`Heuristic`] for both types ([`Zero`](crate::Zero) does). On a backward node such as a
/// subgoal, each must stay admissible when read as the cheapest value over the states
/// that satisfy the node.
///
/// # Errors
///
/// [`ReplayError`](crate::ReplayError) if a replay step finds no successor edge with the backward action's
/// label: the relation or the predecessors disagree with the successors.
///
/// The backward heuristic estimates the cost from the starts to a backward node.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, meet_fn, mm_meet};
///
/// /// Three facts as bits; action `k` makes fact `k` true at cost 1.
/// struct Facts;
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
/// // Regression: a subgoal is a set of facts that must hold; action `k` achieves fact `k`.
/// impl Predecessors for Facts {
///     type Node = u8;
///     type Action = u8;
///     type Cost = u32;
///
///     fn predecessors(&self, subgoal: &u8, mut emit: impl FnMut(u8, u8, u32)) {
///         for k in 0..3 {
///             if subgoal & (1 << k) != 0 {
///                 emit(k, subgoal & !(1 << k), 1);
///             }
///         }
///     }
/// }
///
/// // A state meets a subgoal it satisfies.
/// let satisfies = meet_fn(|state: &u8, subgoal: &u8| state & subgoal == *subgoal);
/// let result = mm_meet(&Facts, [0], [0b110], Zero, Zero, satisfies, Limits::NONE);
///
/// let solution = result.expect("the model replays").into_solution().expect("facts 1 and 2 are reachable");
/// assert_eq!(solution.cost, 2);
/// assert_eq!(solution.states.len(), 3);
/// assert_eq!(solution.goal() & 0b110, 0b110);
/// ```
pub fn mm_meet<G, HF, HB, M>(
    graph: &G,
    starts: impl IntoIterator<Item = <G as Successors>::State>,
    goals: impl IntoIterator<Item = <G as Predecessors>::Node>,
    forward: HF,
    backward: HB,
    meet: M,
    budget: impl Budget<Stats>,
) -> MeetResult<G>
where
    G: RegressionGraph,
    <G as Successors>::State: Clone + Hash + Eq,
    <G as Predecessors>::Node: Clone + Hash + Eq,
    <G as Successors>::Action: Clone + PartialEq,
    <G as Successors>::Cost: CostExt,
    HF: Heuristic<<G as Successors>::State, <G as Successors>::Cost>
        + Heuristic<<G as Predecessors>::Node, <G as Successors>::Cost>,
    HB: Heuristic<<G as Successors>::State, <G as Successors>::Cost>
        + Heuristic<<G as Predecessors>::Node, <G as Successors>::Cost>,
    M: MeetRelation<<G as Successors>::State, <G as Predecessors>::Node>,
{
    let mixed = MixedGraph(graph);
    let core = Core::new(
        &mixed,
        starts.into_iter().map(Mixed::State),
        goals.into_iter().map(Mixed::Node),
        Both(forward),
        Both(backward),
        Some(Meeting::new(Split(meet))),
    );

    unmix(Frontier::new(core, Rule::Mm).run(budget))
}
