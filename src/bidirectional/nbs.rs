//! Near-Optimal Bidirectional Search ([`nbs`], Chen, Holte, Zilles and Sturtevant 2017).

use core::cmp::Reverse;
use core::hash::Hash;

use alloc::vec::Vec;

use search_core::{Budget, Cost, Heuristic};

use crate::best_first::GraphResult;
use crate::bidirectional::{BidirectionalGraph, Core, Direction, Equality, MeetRelation, Meeting};
use crate::keyed_set::KeyedSet;
use crate::outcome::Stats;
use crate::problem::Successors;
use crate::space::NodeId;

/// Near-Optimal Bidirectional Search (NBS): expands pairs of states, one per side, and
/// never expands more than twice the states any admissible front-to-end bidirectional
/// search must.
///
/// The lower bound of a pair `(u, v)` is `lb(u, v) = max(f_F(u), f_B(v), g_F(u) + g_B(v))`.
/// NBS keeps a current lower bound `CLB` and expands both states of a pair with
/// `lb <= CLB`, choosing the smallest `g` among the states with `f <= CLB` on each side
/// (the paper's waiting and ready queues). When no such pair is left it raises `CLB`
/// to the smallest bound any pair can have, and it stops once `CLB` reaches the cheapest
/// meet. `forward` estimates the cost to the nearest goal and `backward` the cost from the
/// start; with admissible estimates the solution is optimal.
///
/// ```mermaid
/// flowchart TD
///     check{"a side empty, or CLB >= mu?"} -->|yes| done[best meet, or Unsolvable]
///     check -->|no| promote["move f <= CLB from waiting to ready, both sides"]
///     promote --> pair{"g_F + g_B of the ready minima <= CLB?"}
///     pair -->|yes| expand[expand both, queue children as waiting]
///     pair -->|no| raise["CLB = smallest waiting f or ready g sum"]
///     expand --> check
///     raise --> check
/// ```
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, heuristic_fn, nbs};
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
/// let result = nbs(&Line, 0, [9], to_goal, from_start, Limits::NONE);
/// let solution = result.into_solution().expect("9 is reachable");
///
/// assert_eq!(solution.cost, 9);
/// assert_eq!(solution.actions, [1; 9]);
/// ```
pub fn nbs<G, HF, HB>(
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
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
{
    nbs_multi(graph, [start], goals, forward, backward, budget)
}

/// [`nbs`] from several start states at once.
///
/// Every start has path cost zero on the forward side, just as every goal has on the
/// backward side. Repeated starts count once, an empty set of starts finds nothing, and a
/// start that is a goal is returned as a zero-cost path. The backward heuristic must estimate
/// the cost back to the nearest start, which a heuristic towards one start overestimates.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, heuristic_fn, nbs_multi};
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
/// let result = nbs_multi(&Up, [0, 6, 3], [8], h, Zero, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(2));
/// ```
pub fn nbs_multi<G, HF, HB>(
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

    Nbs::new(core).run(budget)
}

/// [`nbs()`](crate::nbs()) that meets by a [`MeetRelation`] instead of state equality, from several
/// starts at once.
///
/// `goals` are backward nodes, such as subgoals (partial states), and `meet` decides which
/// forward states meet which backward nodes (see [`MeetRelation`] for the contract). The
/// returned path is rebuilt from the meeting state by replaying the backward actions
/// forwards, through the first successor edge with an equal label, so every state on it
/// is a real state and its cost the true cost. If a step finds no such edge (the relation
/// or the predecessors disagree with the successors), the rest of the backward nodes is
/// joined unchanged, with cost `g_F + g_B`.
///
/// Both heuristics are evaluated on nodes of both sides. On a backward node such as a
/// subgoal, each must keep its guarantee (admissible, and consistent where the search
/// needs it) when read as the cheapest value over the states that satisfy the node.
/// The backward heuristic estimates the cost from the starts to a backward node.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Predecessors, Successors, Zero, meet_fn, nbs_meet};
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
/// let result = nbs_meet(&Facts, [0], [0b110], Zero, Zero, satisfies, Limits::NONE);
///
/// let solution = result.into_solution().expect("facts 1 and 2 are reachable");
/// assert_eq!(solution.cost, 2);
/// assert_eq!(solution.states.len(), 3);
/// assert_eq!(solution.goal() & 0b110, 0b110);
/// ```
pub fn nbs_meet<G, HF, HB, M>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goals: impl IntoIterator<Item = G::State>,
    forward: HF,
    backward: HB,
    meet: M,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone + PartialEq,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
    M: MeetRelation<G::State>,
{
    let core = Core::new(
        graph,
        starts,
        goals,
        forward,
        backward,
        Some(Meeting::new(meet)),
    );

    Nbs::new(core).run(budget)
}

/// A queue order: smallest primary value, then the secondary value reversed, then node id.
type Key<C> = (C, Reverse<C>);

/// The state of one NBS run.
struct Nbs<'a, G: BidirectionalGraph, HF, HB, M> {
    /// Both sides, the heuristics, the counters and the best meet.
    core: Core<'a, G, HF, HB, M>,
    /// Per side, open states with `f > CLB`, by `f` (ties: larger `g`).
    waiting: [KeyedSet<Key<<G as Successors>::Cost>>; 2],
    /// Per side, open states with `f <= CLB`, by `g` (ties: larger `f`).
    ready: [KeyedSet<Key<<G as Successors>::Cost>>; 2],
    /// The current lower bound on the cost of any path not found yet.
    clb: <G as Successors>::Cost,
}

impl<'a, G, HF, HB, M> Nbs<'a, G, HF, HB, M>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
    M: MeetRelation<G::State>,
{
    /// Queues the live seeds of both sides.
    fn new(core: Core<'a, G, HF, HB, M>) -> Self {
        let mut search = Self {
            core,
            waiting: [KeyedSet::new(), KeyedSet::new()],
            ready: [KeyedSet::new(), KeyedSet::new()],
            clb: <G as Successors>::Cost::ZERO,
        };

        for direction in [Direction::Forward, Direction::Backward] {
            for id in 0..search.core.side(direction).space.len() {
                search.queue(direction, id);
            }
        }

        search
    }

    /// Expands pairs until `CLB` reaches the best meet, a side runs empty or the budget is
    /// used up.
    fn run(mut self, mut budget: impl Budget<Stats>) -> GraphResult<G> {
        let mut touched = Vec::new();
        while !self.side_exhausted() && !self.core.proven(self.clb) {
            self.promote(Direction::Forward);
            self.promote(Direction::Backward);

            let Some((u, v)) = self.ready_pair() else {
                self.raise_clb();
                continue;
            };

            for (direction, node) in [(Direction::Forward, u), (Direction::Backward, v)] {
                // Asked before each of the two expansions, so a limit of `n` stops at `n`.
                if budget.exhausted(&self.core.stats) {
                    return self.core.finish(false);
                }

                self.ready[direction.index()].remove(node);
                self.core.expand(direction, node, true, &mut touched);
                for id in touched.drain(..) {
                    self.queue(direction, id);
                }
            }
        }

        self.core.finish(true)
    }

    /// Returns `true` if either side has nothing open.
    fn side_exhausted(&self) -> bool {
        (0..2).any(|side| self.waiting[side].is_empty() && self.ready[side].is_empty())
    }

    /// Moves every waiting state with `f <= CLB` on `direction`'s side to the ready queue.
    fn promote(&mut self, direction: Direction) {
        let side = direction.index();
        while let Some((&(f, Reverse(g)), id)) = self.waiting[side].first() {
            if f.cmp(&self.clb).is_gt() {
                break;
            }

            self.waiting[side].remove(id);
            self.ready[side].insert(id, (g, Reverse(f)));
        }
    }

    /// The ready pair of smallest `g` on each side, if their `g` sum is within `CLB`.
    fn ready_pair(&self) -> Option<(NodeId, NodeId)> {
        let (forward, u) = self.ready[0].first()?;
        let (backward, v) = self.ready[1].first()?;

        (forward.0 + backward.0)
            .cmp(&self.clb)
            .is_le()
            .then_some((u, v))
    }

    /// Raises `CLB` to the smallest lower bound any pair can have now: the smallest waiting
    /// `f` of either side, or the smallest ready `g` sum.
    fn raise_clb(&mut self) {
        let waiting = self
            .waiting
            .iter()
            .filter_map(|queue| queue.first().map(|(key, _)| key.0));
        let ready = match (self.ready[0].first(), self.ready[1].first()) {
            (Some((forward, _)), Some((backward, _))) => Some(forward.0 + backward.0),
            _ => None,
        };

        let next = waiting
            .chain(ready)
            .reduce(|a, b| if b.cmp(&a).is_lt() { b } else { a })
            .expect("both sides have open states");
        debug_assert!(next.cmp(&self.clb).is_gt(), "CLB must grow");

        self.clb = next;
    }

    /// Queues the live node `id` on `direction`'s side as waiting, replacing any earlier
    /// entry; [`Nbs::promote`] moves it on if `f <= CLB`.
    fn queue(&mut self, direction: Direction, id: NodeId) {
        let Some((h, _)) = self.core.h(direction, id) else {
            return;
        };

        let g = self.core.g(direction, id);
        let side = direction.index();
        self.ready[side].remove(id);
        self.waiting[side].insert(id, (g + h, Reverse(g)));
    }
}
