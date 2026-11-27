//! Queries on a [`ContractionHierarchy`]: bidirectional Dijkstra over upward edges, and
//! shortcut unpacking.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::hash::Hash;

use hashbrown::HashMap;

use search_core::Cost;

use super::{BACKWARD, ContractionHierarchy, EdgeId, FORWARD, Via};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};

/// One side of a query: labels and open list. Labels are hashed rather than kept in arrays
/// of the hierarchy's size, so a query costs only what it touches.
struct Side<C> {
    /// Per reached state, the cheapest cost found and the edge it was reached through
    /// (`None` for a root).
    labels: HashMap<usize, (C, Option<EdgeId>)>,
    /// The open list, with lazy deletion.
    heap: BinaryHeap<Reverse<(C, usize)>>,
}

/// A running query.
struct Search<'a, S, A, C> {
    /// The hierarchy searched.
    hierarchy: &'a ContractionHierarchy<S, A, C>,
    /// The forward and backward sides, indexed by [`FORWARD`] and [`BACKWARD`].
    sides: [Side<C>; 2],
    /// The cheapest meeting so far: its cost and the state where the sides meet.
    best: Option<(C, usize)>,
    /// Counters for the result.
    stats: Stats,
}

/// Runs a query from `starts` to `goals`; see [`ContractionHierarchy::query_multi`].
pub(super) fn run<S, A, C>(
    hierarchy: &ContractionHierarchy<S, A, C>,
    starts: impl IntoIterator<Item = S>,
    goals: impl IntoIterator<Item = S>,
) -> SearchResult<S, A, C>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
{
    let starts: Vec<S> = starts.into_iter().collect();
    let goals: Vec<S> = goals.into_iter().collect();
    if let Some(both) = starts.iter().find(|start| goals.contains(start)) {
        let solution = Solution {
            states: vec![both.clone()],
            actions: Vec::new(),
            cost: C::ZERO,
        };

        return SearchResult {
            outcome: Outcome::Found(solution),
            stats: Stats::default(),
        };
    }

    let mut search = Search::new(hierarchy);
    for (side, roots) in [(FORWARD, &starts), (BACKWARD, &goals)] {
        for id in roots.iter().filter_map(|root| hierarchy.find(root)) {
            search.relax(side, id, C::ZERO, None);
        }
    }

    // Without a root on one side nothing can meet.
    if search.sides.iter().all(|side| !side.labels.is_empty()) {
        search.run();
    }

    let outcome = match search.best {
        Some((_, meet)) => Outcome::Found(search.solution(meet)),
        None => Outcome::Unsolvable,
    };

    SearchResult {
        outcome,
        stats: search.stats,
    }
}

/// The upward search space of `state` on `side`, in settling order; see
/// [`ContractionHierarchy::forward_search_space`].
pub(super) fn search_space<'a, S, A, C>(
    hierarchy: &'a ContractionHierarchy<S, A, C>,
    state: &S,
    side: usize,
) -> Vec<(&'a S, C)>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
{
    let mut space = Vec::new();
    let Some(root) = hierarchy.find(state) else {
        return space;
    };

    let mut search = Search::new(hierarchy);
    search.relax(side, root, C::ZERO, None);
    while search.top(side).is_some() {
        let (id, cost) = search.settle(side);
        space.push((&hierarchy.states[id], cost));
    }

    space
}

impl<'a, S, A, C> Search<'a, S, A, C>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
{
    /// A query on `hierarchy` without roots.
    fn new(hierarchy: &'a ContractionHierarchy<S, A, C>) -> Self {
        let side = || Side {
            labels: HashMap::new(),
            heap: BinaryHeap::new(),
        };

        Self {
            hierarchy,
            sides: [side(), side()],
            best: None,
            stats: Stats::default(),
        }
    }

    /// Settles states, always on the side with the smaller live key, until neither side
    /// has one below the best meeting's cost.
    fn run(&mut self) {
        loop {
            let side = match (self.top(FORWARD), self.top(BACKWARD)) {
                (None, None) => break,
                (Some(forward), backward)
                    if backward.is_none_or(|backward| forward <= backward) =>
                {
                    FORWARD
                }
                _ => BACKWARD,
            };
            self.settle(side);
        }
    }

    /// The smallest live key on `side`, if it is below the best meeting's cost; drops
    /// stale entries on the way.
    fn top(&mut self, side: usize) -> Option<C> {
        let open = &mut self.sides[side];
        while let Some(&Reverse((cost, id))) = open.heap.peek() {
            if open.labels[&id].0 < cost {
                open.heap.pop();
                continue;
            }

            return self.best.is_none_or(|(mu, _)| cost < mu).then_some(cost);
        }

        None
    }

    /// Settles the top state of `side` (which [`Search::top`] left live) and relaxes its
    /// upward edges. Returns the state and its cost.
    fn settle(&mut self, side: usize) -> (usize, C) {
        let Reverse((cost, id)) = self.sides[side]
            .heap
            .pop()
            .expect("`top` found a live entry");
        self.stats.expanded += 1;

        let hierarchy = self.hierarchy;
        for &edge_id in &hierarchy.up[side][id] {
            self.stats.generated += 1;
            let edge = &hierarchy.edges[edge_id];
            let next = if side == FORWARD { edge.to } else { edge.from };
            self.relax(side, next, cost + edge.cost, Some(edge_id));
        }

        (id, cost)
    }

    /// Labels `id` on `side` with `cost` through `edge` if that is cheaper than its label,
    /// and records a meeting with the other side.
    fn relax(&mut self, side: usize, id: usize, cost: C, edge: Option<EdgeId>) {
        let open = &mut self.sides[side];
        if open.labels.get(&id).is_some_and(|&(old, _)| old <= cost) {
            return;
        }

        open.labels.insert(id, (cost, edge));
        open.heap.push(Reverse((cost, id)));

        if let Some(&(other, _)) = self.sides[1 - side].labels.get(&id) {
            let total = cost + other;
            if self.best.is_none_or(|(mu, _)| total < mu) {
                self.best = Some((total, id));
            }
        }
    }

    /// The path through the meeting at `meet`, with every shortcut unpacked.
    fn solution(&self, meet: usize) -> Solution<S, A, C> {
        let mut packed = self.walk(FORWARD, meet);
        packed.reverse();
        packed.extend(self.walk(BACKWARD, meet));

        let edges = &self.hierarchy.edges;
        let first = packed.first().map_or(meet, |&edge_id| edges[edge_id].from);
        let mut path = Solution {
            states: vec![self.hierarchy.states[first].clone()],
            actions: Vec::new(),
            cost: C::ZERO,
        };

        let mut stack: Vec<EdgeId> = Vec::new();
        for edge_id in packed {
            stack.push(edge_id);
            while let Some(edge_id) = stack.pop() {
                let edge = &edges[edge_id];
                match &edge.via {
                    Via::Original(action) => {
                        path.actions.push(action.clone());
                        path.states.push(self.hierarchy.states[edge.to].clone());
                        path.cost = path.cost + edge.cost;
                    }
                    Via::Shortcut(first, second) => stack.extend([*second, *first]),
                }
            }
        }

        path
    }

    /// The edges from `meet` back to a root of `side`, in walking order.
    fn walk(&self, side: usize, meet: usize) -> Vec<EdgeId> {
        let mut walked = Vec::new();
        let mut id = meet;
        while let Some(edge_id) = self.sides[side].labels[&id].1 {
            walked.push(edge_id);
            let edge = &self.hierarchy.edges[edge_id];
            id = if side == FORWARD { edge.from } else { edge.to };
        }

        walked
    }
}
