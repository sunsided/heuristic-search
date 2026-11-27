//! Bidirectional best-first search with one open list per side:
//! [`bidirectional_dijkstra`], [`dibbs`] and [`bae_star`]; [`mm`](crate::mm) shares the loop.

use core::cmp::Reverse;
use core::hash::Hash;

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;

use search_core::{Budget, Cost, Heuristic, Zero};

use crate::best_first::GraphResult;
use crate::bidirectional::mixed::{Both, Mixed, MixedGraph, Split, unmix};
use crate::bidirectional::{
    BidirectionalGraph, Core, Direction, Equality, MeetRelation, MeetResult, Meeting, Meets,
    RegressionGraph,
};
use crate::cost::CostExt;
use crate::keyed_set::KeyedSet;
use crate::outcome::Stats;
use crate::problem::Predecessors;
use crate::problem::Successors;
use crate::space::NodeId;

/// Bidirectional Dijkstra: uniform-cost search from both ends, cheapest path.
///
/// Expands the side whose cheapest open state has the smaller `g` and stops once
/// `g_min_F + g_min_B` reaches the cheapest meet. Optimal for non-negative costs.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, bidirectional_dijkstra};
///
/// /// A cycle of 10 nodes; going up costs 1, going down costs 3.
/// struct Ring;
///
/// impl Successors for Ring {
///     type State = u8;
///     type Action = bool;
///     type Cost = u32;
///
///     fn successors(&self, n: &u8, mut emit: impl FnMut(bool, u8, u32)) {
///         emit(true, (n + 1) % 10, 1);
///         emit(false, (n + 9) % 10, 3);
///     }
/// }
///
/// impl Predecessors for Ring {
///     type Node = u8;
///     type Action = bool;
///     type Cost = u32;
///
///     fn predecessors(&self, n: &u8, mut emit: impl FnMut(bool, u8, u32)) {
///         emit(true, (n + 9) % 10, 1);
///         emit(false, (n + 1) % 10, 3);
///     }
/// }
///
/// let result = bidirectional_dijkstra(&Ring, 0, [8], Limits::NONE);
/// let solution = result.into_solution().expect("the ring is connected");
///
/// assert_eq!(solution.cost, 6);
/// assert_eq!(solution.actions, [false, false]);
/// ```
pub fn bidirectional_dijkstra<G>(
    graph: &G,
    start: G::State,
    goals: impl IntoIterator<Item = G::State>,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    <G as Successors>::Cost: CostExt,
{
    bidirectional_dijkstra_multi(graph, [start], goals, budget)
}

/// [`bidirectional_dijkstra`] from several start states at once.
///
/// Every start has path cost zero on the forward side, just as every goal has on the
/// backward side. Repeated starts count once, an empty set of starts finds nothing, and a
/// start that is a goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, bidirectional_dijkstra_multi};
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
/// let result = bidirectional_dijkstra_multi(&Up, [0, 6, 3], [8], Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn bidirectional_dijkstra_multi<G>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goals: impl IntoIterator<Item = G::State>,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    <G as Successors>::Cost: CostExt,
{
    let core = Core::new(
        graph,
        starts,
        goals,
        Zero,
        Zero,
        None::<Meeting<G, Equality>>,
    );

    Frontier::new(core, Rule::Dijkstra).run(budget)
}

/// Dynamically Improved Bounds Bidirectional Search (DIBBS, Sewell and Jacobson 2021).
///
/// Orders each side by `b = 2g + h - h_opposite`, where `h` estimates the distance to the
/// side's target and `h_opposite` the distance back to its own origin. It expands the side
/// with the smaller `b` and stops once `b_min_F + b_min_B >= 2 mu`. Optimal with
/// consistent heuristics; `forward` estimates the cost to the nearest goal, `backward` the
/// cost from the start.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, dibbs, heuristic_fn};
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
/// let result = dibbs(&Line, 0, [9], to_goal, from_start, Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.cost), Some(9));
/// ```
pub fn dibbs<G, HF, HB>(
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
    dibbs_multi(graph, [start], goals, forward, backward, budget)
}

/// [`dibbs`] from several start states at once.
///
/// Every start has path cost zero on the forward side, just as every goal has on the
/// backward side. Repeated starts count once, an empty set of starts finds nothing, and a
/// start that is a goal is returned as a zero-cost path. The backward heuristic must estimate
/// the cost back to the nearest start, which a heuristic towards one start overestimates.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, dibbs_multi, heuristic_fn};
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
/// let result = dibbs_multi(&Up, [0, 6, 3], [8], h, Zero, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn dibbs_multi<G, HF, HB>(
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

    Frontier::new(core, Rule::Dibbs).run(budget)
}

/// Bidirectional A* with error (BAE*, Sadhukhan 2013).
///
/// Uses the same `b = 2g + h - h_opposite` order and stopping rule as [`dibbs`] (`b` is
/// BAE*'s `f + d` with error term `d = g - h_opposite`) but picks the direction by the
/// cardinality criterion: the side with fewer open states expands. Optimal with consistent
/// heuristics.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, bae_star};
///
/// /// Counting up by one at cost 2; counting back down is not possible.
/// struct Up;
///
/// impl Successors for Up {
///     type State = u32;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
///         emit((), n + 1, 2);
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
///             emit((), previous, 2);
///         }
///     }
/// }
///
/// let result = bae_star(&Up, 3, [7], Zero, Zero, Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.states.clone()), Some(vec![3, 4, 5, 6, 7]));
/// ```
pub fn bae_star<G, HF, HB>(
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
    bae_star_multi(graph, [start], goals, forward, backward, budget)
}

/// [`bae_star`] from several start states at once.
///
/// Every start has path cost zero on the forward side, just as every goal has on the
/// backward side. Repeated starts count once, an empty set of starts finds nothing, and a
/// start that is a goal is returned as a zero-cost path. The backward heuristic must estimate
/// the cost back to the nearest start, which a heuristic towards one start overestimates.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, bae_star_multi, heuristic_fn};
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
/// let result = bae_star_multi(&Up, [0, 6, 3], [8], h, Zero, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn bae_star_multi<G, HF, HB>(
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

    Frontier::new(core, Rule::Bae).run(budget)
}

/// How a [`Frontier`] search orders, picks a side and stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Rule {
    /// Key `g`; smaller key expands; stop at `k_F + k_B >= mu`.
    Dijkstra,
    /// Key `b`; smaller key expands; stop at `k_F + k_B >= 2 mu`.
    Dibbs,
    /// Key `b`; the side with fewer open states expands; stop at `k_F + k_B >= 2 mu`.
    Bae,
    /// Key `pr = max(f, 2g)`; smaller key expands; stop at
    /// `mu <= max(min(k_F, k_B), f_min_F, f_min_B, g_min_F + g_min_B)`.
    Mm,
}

/// Open-list order of one side: smallest key, then larger `g`, then node id.
type Key<C> = (C, Reverse<C>);

/// A lazy-deletion min-heap of `(value, node)` for one term of [`Rule::Mm`]'s stopping rule.
type Bound<C> = BinaryHeap<Reverse<(C, NodeId)>>;

/// [`bidirectional_dijkstra`] that meets by a [`MeetRelation`] instead of state equality, from several
/// starts at once.
///
/// `goals` are backward nodes of the graph's [`Predecessors::Node`](crate::Predecessors)
/// type, such as subgoals (partial states), and `meet` decides which forward states meet
/// which backward nodes (see [`MeetRelation`] for the contract). The returned path is
/// rebuilt from the meeting state by replaying the backward actions forwards, through the
/// first successor edge with an equal label, so every state on it is a real state and its
/// cost the true cost.
///
/// Both heuristics are evaluated on states and on backward nodes, so they implement
/// [`Heuristic`] for both types ([`Zero`] does). On a backward node such as a subgoal,
/// each must keep its guarantee (admissible, and consistent where the search needs it)
/// when read as the cheapest value over the states that satisfy the node.
///
/// # Errors
///
/// [`ReplayError`](crate::ReplayError) if a replay step finds no successor edge with the backward action's
/// label: the relation or the predecessors disagree with the successors.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, bidirectional_dijkstra_meet, meet_fn};
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
/// let result = bidirectional_dijkstra_meet(&Facts, [0], [0b110], satisfies, Limits::NONE);
///
/// let solution = result.expect("the model replays").into_solution().expect("facts 1 and 2 are reachable");
/// assert_eq!(solution.cost, 2);
/// assert_eq!(solution.states.len(), 3);
/// assert_eq!(solution.goal() & 0b110, 0b110);
/// ```
pub fn bidirectional_dijkstra_meet<G, M>(
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
    <G as Successors>::Cost: CostExt,
    M: MeetRelation<<G as Successors>::State, <G as Predecessors>::Node>,
{
    let mixed = MixedGraph(graph);
    let core = Core::new(
        &mixed,
        starts.into_iter().map(Mixed::State),
        goals.into_iter().map(Mixed::Node),
        Zero,
        Zero,
        Some(Meeting::new(Split(meet))),
    );

    unmix(Frontier::new(core, Rule::Dijkstra).run(budget))
}

/// [`dibbs`] that meets by a [`MeetRelation`] instead of state equality, from several
/// starts at once.
///
/// `goals` are backward nodes of the graph's [`Predecessors::Node`](crate::Predecessors)
/// type, such as subgoals (partial states), and `meet` decides which forward states meet
/// which backward nodes (see [`MeetRelation`] for the contract). The returned path is
/// rebuilt from the meeting state by replaying the backward actions forwards, through the
/// first successor edge with an equal label, so every state on it is a real state and its
/// cost the true cost.
///
/// Both heuristics are evaluated on states and on backward nodes, so they implement
/// [`Heuristic`] for both types ([`Zero`] does). On a backward node such as a subgoal,
/// each must keep its guarantee (admissible, and consistent where the search needs it)
/// when read as the cheapest value over the states that satisfy the node.
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
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, dibbs_meet, meet_fn};
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
/// let result = dibbs_meet(&Facts, [0], [0b110], Zero, Zero, satisfies, Limits::NONE);
///
/// let solution = result.expect("the model replays").into_solution().expect("facts 1 and 2 are reachable");
/// assert_eq!(solution.cost, 2);
/// assert_eq!(solution.states.len(), 3);
/// assert_eq!(solution.goal() & 0b110, 0b110);
/// ```
pub fn dibbs_meet<G, HF, HB, M>(
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

    unmix(Frontier::new(core, Rule::Dibbs).run(budget))
}

/// [`bae_star`] that meets by a [`MeetRelation`] instead of state equality, from several
/// starts at once.
///
/// `goals` are backward nodes of the graph's [`Predecessors::Node`](crate::Predecessors)
/// type, such as subgoals (partial states), and `meet` decides which forward states meet
/// which backward nodes (see [`MeetRelation`] for the contract). The returned path is
/// rebuilt from the meeting state by replaying the backward actions forwards, through the
/// first successor edge with an equal label, so every state on it is a real state and its
/// cost the true cost.
///
/// Both heuristics are evaluated on states and on backward nodes, so they implement
/// [`Heuristic`] for both types ([`Zero`] does). On a backward node such as a subgoal,
/// each must keep its guarantee (admissible, and consistent where the search needs it)
/// when read as the cheapest value over the states that satisfy the node.
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
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, bae_star_meet, meet_fn};
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
/// let result = bae_star_meet(&Facts, [0], [0b110], Zero, Zero, satisfies, Limits::NONE);
///
/// let solution = result.expect("the model replays").into_solution().expect("facts 1 and 2 are reachable");
/// assert_eq!(solution.cost, 2);
/// assert_eq!(solution.states.len(), 3);
/// assert_eq!(solution.goal() & 0b110, 0b110);
/// ```
pub fn bae_star_meet<G, HF, HB, M>(
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

    unmix(Frontier::new(core, Rule::Bae).run(budget))
}

/// A bidirectional search with one keyed open list per side.
pub(super) struct Frontier<'a, G: BidirectionalGraph, HF, HB, M> {
    /// Both sides, the heuristics, the counters and the best meet.
    core: Core<'a, G, HF, HB, M>,
    /// The open lists, indexed by [`Direction::index`].
    open: [KeyedSet<Key<<G as Successors>::Cost>>; 2],
    /// For [`Rule::Mm`] only: min-heaps of `(f, node)` and `(g, node)` for the stopping
    /// rule, indexed by [`Direction::index`], then `0` for `f` and `1` for `g`. Empty
    /// until the first meet, when the stopping rule first needs them (see
    /// [`Frontier::bounded`]). Entries are deleted lazily: one whose node is closed or has
    /// been re-keyed is dropped when it reaches the top, so expanding a node costs no
    /// removal here.
    bounds: [[Bound<<G as Successors>::Cost>; 2]; 2],
    /// Whether [`Frontier::bounds`] hold an entry for every open node. Set at the first
    /// meet: before it the stopping rule cannot hold, so the heaps are not kept.
    bounded: bool,
    /// The ordering, side-selection and stopping rule.
    rule: Rule,
}

impl<'a, G, HF, HB, M> Frontier<'a, G, HF, HB, M>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    <G as Successors>::Cost: CostExt,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
    M: Meets<G::State>,
{
    /// Opens the live seeds of both sides.
    pub(super) fn new(core: Core<'a, G, HF, HB, M>, rule: Rule) -> Self {
        let mut search = Self {
            core,
            open: [KeyedSet::new(), KeyedSet::new()],
            bounds: [
                [BinaryHeap::new(), BinaryHeap::new()],
                [BinaryHeap::new(), BinaryHeap::new()],
            ],
            bounded: false,
            rule,
        };

        for direction in [Direction::Forward, Direction::Backward] {
            for id in 0..search.core.side(direction).space.len() {
                search.open(direction, id);
            }
        }

        search
    }

    /// Expands until the stopping rule holds, a side runs empty or the budget is used up.
    pub(super) fn run(mut self, mut budget: impl Budget<Stats>) -> GraphResult<G> {
        let mut touched = Vec::new();
        loop {
            let (Some(forward), Some(backward)) = (
                self.min_key(Direction::Forward),
                self.min_key(Direction::Backward),
            ) else {
                return self.core.finish(true);
            };

            if self.stops(forward, backward) {
                return self.core.finish(true);
            }

            if budget.exhausted(&self.core.stats) {
                return self.core.finish(false);
            }

            let direction = self.pick(forward, backward);
            let (_, node) = self.open[direction.index()]
                .first()
                .expect("the side is not empty");
            self.open[direction.index()].remove(node);
            self.core.expand(direction, node, true, &mut touched);
            for id in touched.drain(..) {
                self.open(direction, id);
            }
        }
    }

    /// The smallest key open on `direction`'s side.
    fn min_key(&self, direction: Direction) -> Option<<G as Successors>::Cost> {
        self.open[direction.index()].first().map(|(key, _)| key.0)
    }

    /// Returns `true` if the keys `forward` and `backward` prove the best meet optimal.
    fn stops(
        &mut self,
        forward: <G as Successors>::Cost,
        backward: <G as Successors>::Cost,
    ) -> bool {
        let sum = forward.saturating_add(backward);

        match self.rule {
            Rule::Dijkstra => self.core.proven(sum),
            Rule::Dibbs | Rule::Bae => self
                .core
                .mu()
                .is_some_and(|mu| sum.cmp(&mu.saturating_times(2)).is_ge()),
            // The stop is `mu <= max(C, f_min_F, f_min_B, g_min_F + g_min_B)`, tested term
            // by term so that the bounds are only looked up when the cheaper terms do not
            // decide. Epsilon, the smallest edge cost, is taken as zero: correct for every
            // cost type, though a positive one would stop sooner.
            Rule::Mm => {
                let Some(mu) = self.core.mu() else {
                    return false;
                };
                let reaches = |bound: <G as Successors>::Cost| bound.cmp(&mu).is_ge();
                if reaches(forward.min(backward)) {
                    return true;
                }

                self.track();
                reaches(self.bound(Direction::Forward, 0))
                    || reaches(self.bound(Direction::Backward, 0))
                    || reaches(
                        self.bound(Direction::Forward, 1)
                            .saturating_add(self.bound(Direction::Backward, 1)),
                    )
            }
        }
    }

    /// Fills the bound heaps with every open node, once; [`Frontier::open`] keeps them
    /// current from then on.
    fn track(&mut self) {
        if self.bounded {
            return;
        }

        self.bounded = true;
        for direction in [Direction::Forward, Direction::Backward] {
            for id in 0..self.core.side(direction).space.len() {
                if self.open[direction.index()].key(id).is_some() {
                    for which in 0..2 {
                        let value = self.value(direction, which, id);
                        self.bounds[direction.index()][which].push(Reverse((value, id)));
                    }
                }
            }
        }
    }

    /// The smallest `f` (`which == 0`) or `g` (`which == 1`) open on `direction`'s side;
    /// zero if the side is empty. Drops the stale entries above it from the heap.
    fn bound(&mut self, direction: Direction, which: usize) -> <G as Successors>::Cost {
        while let Some(&Reverse((key, id))) = self.bounds[direction.index()][which].peek() {
            if self.open[direction.index()].key(id).is_some()
                && self.value(direction, which, id) == key
            {
                return key;
            }
            self.bounds[direction.index()][which].pop();
        }

        <G as Successors>::Cost::ZERO
    }

    /// The current `f` (`which == 0`) or `g` (`which == 1`) of the open node `id` on
    /// `direction`'s side.
    fn value(&self, direction: Direction, which: usize, id: NodeId) -> <G as Successors>::Cost {
        let g = self.core.g(direction, id);
        if which == 0 {
            let (h, _) = self.core.h(direction, id).expect("open nodes are live");
            g.saturating_add(h)
        } else {
            g
        }
    }

    /// The direction to expand next.
    fn pick(
        &self,
        forward: <G as Successors>::Cost,
        backward: <G as Successors>::Cost,
    ) -> Direction {
        let forward_first = match self.rule {
            Rule::Dijkstra | Rule::Dibbs | Rule::Mm => forward.cmp(&backward).is_le(),
            Rule::Bae => self.open[0].len() <= self.open[1].len(),
        };

        if forward_first {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }

    /// Opens (or re-keys) the live node `id` on `direction`'s side.
    fn open(&mut self, direction: Direction, id: NodeId) {
        let Some((h, h_opposite)) = self.core.h(direction, id) else {
            return;
        };

        let g = self.core.g(direction, id);
        let key = match self.rule {
            Rule::Dijkstra => g,
            Rule::Dibbs | Rule::Bae => (g + g + h).saturating_sub(h_opposite),
            Rule::Mm => {
                let f = g.saturating_add(h);
                if self.bounded {
                    let bounds = &mut self.bounds[direction.index()];
                    bounds[0].push(Reverse((f, id)));
                    bounds[1].push(Reverse((g, id)));
                }
                f.max(g.saturating_times(2))
            }
        };

        self.open[direction.index()].insert(id, (key, Reverse(g)));
    }
}
