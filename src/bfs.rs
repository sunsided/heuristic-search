//! Breadth-first search ([`bfs`]): fewest edges, ignoring edge costs.

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::hash::Hash;

use crate::best_first::GraphResult;
use crate::outcome::{Budget, Outcome, SearchResult, Stats};
use crate::problem::{Goal, Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// Breadth-first search from `start` to the goal with the fewest edges.
///
/// Edge costs are ignored for the order but summed into
/// [`Solution::cost`](crate::Solution::cost). The goal test runs when a state is generated,
/// not when it is expanded, which saves the last layer of expansions and keeps the result
/// shortest. Each state is expanded at most once.
pub fn bfs<G, T>(graph: &G, start: G::State, goal: &T, mut budget: impl Budget) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
{
    let mut stats = Stats::default();
    let mut space = SearchSpace::new(start);
    if goal.is_goal(space.state(0)) {
        return finish(Outcome::Found(space.solution(0)), stats);
    }

    let mut queue = VecDeque::from([0]);
    let mut buffer = Vec::new();
    while let Some(node) = queue.pop_front() {
        if budget.exhausted(&stats) {
            return finish(Outcome::LimitReached, stats);
        }

        stats.expanded += 1;

        stats.generated += collect_successors(graph, space.state(node), &mut buffer);
        if let Some(found) = enqueue_new(&mut space, &mut queue, &mut buffer, node, goal) {
            return finish(Outcome::Found(space.solution(found)), stats);
        }
    }

    finish(Outcome::Unsolvable, stats)
}

/// Adds the unseen successors in `buffer` (emptying it) to the search space and the back of
/// `queue`, and returns the first one that is a goal.
fn enqueue_new<S, A, C, T>(
    space: &mut SearchSpace<S, A, C>,
    queue: &mut VecDeque<NodeId>,
    buffer: &mut Vec<(A, S, C)>,
    parent: NodeId,
    goal: &T,
) -> Option<NodeId>
where
    S: Hash + Eq,
    C: crate::Cost,
    T: Goal<S> + ?Sized,
{
    for (action, state, cost) in buffer.drain(..) {
        let Reach::New(id) = space.reach(state, parent, action, cost, false) else {
            continue;
        };

        if goal.is_goal(space.state(id)) {
            return Some(id);
        }

        queue.push_back(id);
    }

    None
}

/// Builds the result from `outcome` and `stats`.
fn finish<S, A, C>(outcome: Outcome<S, A, C>, stats: Stats) -> SearchResult<S, A, C> {
    SearchResult { outcome, stats }
}
