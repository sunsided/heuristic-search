//! Bidirectional search from a start state forwards and from goal states backwards, until
//! the frontiers meet: [`bidirectional_bfs`], [`bidirectional_dijkstra`], [`nbs`],
//! [`dibbs`] and [`bae_star`].
//!
//! The graph must implement [`Successors`] and [`Predecessors`] over the same states,
//! actions and costs (the [`BidirectionalGraph`] bound). Goals are explicit states, not a
//! predicate, because the backward search starts from them. The frontiers meet when a state
//! is reached from both sides; the cheapest meet found so far, `mu`, is the incumbent, and
//! each algorithm stops once its lower bound on any undiscovered path reaches `mu`.
//!
//! | Algorithm                  | Expands                                        | Stops when                        | Needs          |
//! | -------------------------- | ---------------------------------------------- | --------------------------------- | -------------- |
//! | [`bidirectional_bfs`]      | the smaller frontier, a whole layer at a time   | a layer met the other side        | nothing        |
//! | [`bidirectional_dijkstra`] | the side with the smaller `g`                  | `g_min_F + g_min_B >= mu`         | nothing        |
//! | [`nbs`]                    | a pair `(u, v)` with `lb(u, v) <= CLB`, both    | `CLB >= mu`                       | admissible `h` |
//! | [`dibbs`]                  | the side with the smaller `b = 2g + h - h_opp` | `b_min_F + b_min_B >= 2 mu`       | consistent `h` |
//! | [`bae_star`]               | the side with fewer open states, by `b`        | `b_min_F + b_min_B >= 2 mu`       | consistent `h` |
//!
//! The forward heuristic estimates the cost from a state to the nearest goal, the backward
//! heuristic the cost from the start to a state. Each side reopens closed states when a
//! cheaper path turns up.

mod bfs;
mod frontier;
mod nbs;

pub use bfs::bidirectional_bfs;
pub use frontier::{bae_star, bidirectional_dijkstra, dibbs};
pub use nbs::nbs;

use alloc::vec::Vec;
use core::hash::Hash;

use crate::best_first::GraphResult;
use crate::cost::Cost;
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::{
    Heuristic, Predecessors, Successors, collect_predecessors, collect_successors,
};
use crate::space::{NodeId, Reach, SearchSpace};

/// A graph that can be searched in both directions over the same states.
///
/// Implemented for every graph whose [`Predecessors`] use the [`Successors`] state as
/// node, and the same action and cost types.
pub trait BidirectionalGraph:
    Successors
    + Predecessors<
        Node = <Self as Successors>::State,
        Action = <Self as Successors>::Action,
        Cost = <Self as Successors>::Cost,
    >
{
}

impl<G> BidirectionalGraph for G where
    G: Successors
        + Predecessors<
            Node = <G as Successors>::State,
            Action = <G as Successors>::Action,
            Cost = <G as Successors>::Cost,
        >
{
}

/// A search direction; also the index of its [`Side`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// From the start along successor edges.
    Forward = 0,
    /// From the goals along predecessor edges.
    Backward = 1,
}

impl Direction {
    /// The other direction.
    const fn opposite(self) -> Self {
        match self {
            Self::Forward => Self::Backward,
            Self::Backward => Self::Forward,
        }
    }

    /// The index of this direction's [`Side`].
    const fn index(self) -> usize {
        self as usize
    }
}

/// One direction's search space and per-node data.
struct Side<S, A, C> {
    /// Every state this side has seen. Forward, parent links point towards the start;
    /// backward, towards the goals.
    space: SearchSpace<S, A, C>,
    /// Per node, the estimate towards this side's target and the opposite side's estimate
    /// (`h` and `h_opposite`). `None` for dead ends, which are never opened.
    h: Vec<Option<(C, C)>>,
    /// Whether each node has been expanded at least once.
    closed: Vec<bool>,
}

/// The cheapest meeting of the two frontiers found so far.
#[derive(Debug, Clone, Copy)]
struct Meet<C> {
    /// `g_F + g_B` of the meeting state when it was found.
    cost: C,
    /// The meeting state's node in the forward space.
    forward: NodeId,
    /// The meeting state's node in the backward space.
    backward: NodeId,
}

/// What every bidirectional search shares: both sides, the heuristics, the counters and
/// the best meet.
struct Core<'a, G: Successors, HF, HB> {
    /// The graph being searched.
    graph: &'a G,
    /// Estimates the cost from a state to the nearest goal.
    forward_h: HF,
    /// Estimates the cost from the start to a state.
    backward_h: HB,
    /// The forward and backward sides, indexed by [`Direction::index`].
    sides: [Side<G::State, <G as Successors>::Action, <G as Successors>::Cost>; 2],
    /// Counters for the result, summed over both directions.
    stats: Stats,
    /// The cheapest meet found so far (`mu`).
    meet: Option<Meet<<G as Successors>::Cost>>,
    /// Neighbours of the node being expanded; reused between expansions.
    buffer: Vec<(<G as Successors>::Action, G::State, <G as Successors>::Cost)>,
}

impl<'a, G, HF, HB> Core<'a, G, HF, HB>
where
    G: BidirectionalGraph,
    G::State: Clone + Hash + Eq,
    <G as Successors>::Action: Clone,
    HF: Heuristic<G::State, <G as Successors>::Cost>,
    HB: Heuristic<G::State, <G as Successors>::Cost>,
{
    /// Seeds the forward side with `start` and the backward side with `goals`, evaluates
    /// them, and records a meet if the start is a goal.
    fn new(
        graph: &'a G,
        start: G::State,
        goals: impl IntoIterator<Item = G::State>,
        forward_h: HF,
        backward_h: HB,
    ) -> Self {
        let never = |_: &G::State| false;
        let side = |space: SearchSpace<_, _, _>| Side {
            space,
            h: Vec::new(),
            closed: Vec::new(),
        };

        let mut core = Self {
            graph,
            forward_h,
            backward_h,
            sides: [
                side(SearchSpace::seed([start], &never).0),
                side(SearchSpace::seed(goals, &never).0),
            ],
            stats: Stats::default(),
            meet: None,
            buffer: Vec::new(),
        };

        for direction in [Direction::Forward, Direction::Backward] {
            for id in 0..core.side(direction).space.len() {
                core.evaluate(direction, id);
                core.check_meet(direction, id);
            }
        }

        core
    }

    /// The side of `direction`.
    fn side(
        &self,
        direction: Direction,
    ) -> &Side<G::State, <G as Successors>::Action, <G as Successors>::Cost> {
        &self.sides[direction.index()]
    }

    /// The path cost of node `id` on `direction`'s side.
    fn g(&self, direction: Direction, id: NodeId) -> <G as Successors>::Cost {
        self.side(direction).space.g(id)
    }

    /// The estimates `(h, h_opposite)` of node `id` on `direction`'s side; `None` for a
    /// dead end.
    fn h(
        &self,
        direction: Direction,
        id: NodeId,
    ) -> Option<(<G as Successors>::Cost, <G as Successors>::Cost)> {
        self.side(direction).h[id]
    }

    /// The cost of the best meet so far (`mu`).
    fn mu(&self) -> Option<<G as Successors>::Cost> {
        self.meet.map(|meet| meet.cost)
    }

    /// Returns `true` if a meet exists and `bound` (a lower bound on every path not found
    /// yet) has reached it.
    fn proven(&self, bound: <G as Successors>::Cost) -> bool {
        self.mu().is_some_and(|mu| bound.cmp_cost(&mu).is_ge())
    }

    /// Expands `node` on `direction`'s side and appends the new and improved live nodes to
    /// `touched`. Known states are improved only if `improve` is set.
    fn expand(
        &mut self,
        direction: Direction,
        node: NodeId,
        improve: bool,
        touched: &mut Vec<NodeId>,
    ) {
        self.stats.expanded += 1;
        self.sides[direction.index()].closed[node] = true;
        self.generate(direction, node);

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            let side = &mut self.sides[direction.index()];
            let id = match side.space.reach(state, node, action, cost, improve) {
                Reach::New(id) => {
                    self.evaluate(direction, id);
                    id
                }
                Reach::Improved(id) => {
                    if self.side(direction).closed[id] {
                        self.stats.reopened += 1;
                    }

                    id
                }
                Reach::Stale => continue,
            };

            self.check_meet(direction, id);
            if self.h(direction, id).is_some() {
                touched.push(id);
            }
        }

        self.buffer = buffer;
    }

    /// Fills the buffer with the neighbours of `node` in `direction`.
    fn generate(&mut self, direction: Direction, node: NodeId) {
        let state = self.sides[direction.index()].space.state(node);
        self.stats.generated += match direction {
            Direction::Forward => collect_successors(self.graph, state, &mut self.buffer),
            Direction::Backward => collect_predecessors(self.graph, state, &mut self.buffer),
        };
    }

    /// Records both estimates of the new node `id` on `direction`'s side.
    fn evaluate(&mut self, direction: Direction, id: NodeId) {
        let side = &mut self.sides[direction.index()];
        let state = side.space.state(id);
        let forward = self.forward_h.estimate(state);
        let backward = self.backward_h.estimate(state);
        let h = match direction {
            Direction::Forward => forward.zip(backward),
            Direction::Backward => backward.zip(forward),
        };

        debug_assert_eq!(id, side.h.len(), "nodes are evaluated in creation order");
        side.h.push(h);
        side.closed.push(false);
    }

    /// Updates the best meet if the state of node `id` on `direction`'s side is known on
    /// the other side and the joined path is cheaper.
    fn check_meet(&mut self, direction: Direction, id: NodeId) {
        let other = direction.opposite();
        let state = self.side(direction).space.state(id);
        let Some(other_id) = self.side(other).space.find(state) else {
            return;
        };

        let cost = self.g(direction, id) + self.g(other, other_id);
        if self.mu().is_some_and(|mu| cost.cmp_cost(&mu).is_ge()) {
            return;
        }

        let (forward, backward) = match direction {
            Direction::Forward => (id, other_id),
            Direction::Backward => (other_id, id),
        };
        self.meet = Some(Meet {
            cost,
            forward,
            backward,
        });
    }

    /// The path through the best meet: forward from the start to the meeting state, then
    /// along the backward links to a goal.
    fn solution(
        &self,
    ) -> Option<Solution<G::State, <G as Successors>::Action, <G as Successors>::Cost>> {
        let meet = self.meet?;
        let mut path = self.side(Direction::Forward).space.solution(meet.forward);
        let tail = self
            .side(Direction::Backward)
            .space
            .walk_to_root(meet.backward);

        path.states.extend(tail.states.into_iter().skip(1));
        path.actions.extend(tail.actions);
        path.cost = path.cost + tail.cost;

        Some(path)
    }

    /// Wraps up the run: the best meet if `decided`, else [`Outcome::LimitReached`].
    ///
    /// A decided search without a meet is [`Outcome::Unsolvable`].
    fn finish(self, decided: bool) -> GraphResult<G> {
        let outcome = match (decided, self.solution()) {
            (false, _) => Outcome::LimitReached,
            (true, Some(solution)) => Outcome::Found(solution),
            (true, None) => Outcome::Unsolvable,
        };

        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
