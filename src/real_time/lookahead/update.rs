//! The learning rules that run after a lookahead: LSS-LRTA*'s Dijkstra update
//! ([`dijkstra`]), the limited-horizon Bellman update of Hadi et al.
//! ([`limited_horizon`]) and RTAA*'s update ([`rtaa`]).
//!
//! Each writes learned values into the table and returns how many changed.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::hash::Hash;

use search_core::{Cost, Goal, Heuristic};

use super::{Frontier, Lookahead, Status};
use crate::cost::CostExt;
use crate::problem::Successors;
use crate::real_time::LearnedHeuristic;
use crate::space::NodeId;

/// LSS-LRTA*'s update: every closed node gets its cost to the open list, `min over open o
/// of dist(s, o) + h(o)`, or `None` if it reaches no open node.
///
/// Koenig and Sun's formulation: closed values start at infinity, and Dijkstra's algorithm
/// runs from the open nodes, keyed by their values, along reversed edges into closed nodes.
pub(super) fn dijkstra<G, T, H>(
    search: &Lookahead<'_, G, T>,
    _frontier: Frontier,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
) -> u64
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    T: ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let seeds = search
        .status
        .iter()
        .zip(&search.h)
        .map(|(&status, &h)| if status == Status::Open { h } else { None })
        .collect();
    let values =
        backward_shortest_paths(search, seeds, |node| search.status[node] == Status::Closed);

    write(search, table, &values, |node| {
        search.status[node] == Status::Closed
    })
}

/// The limited-horizon Bellman update (Hadi et al. 2025, Eq. 5 by Algorithm 1): every node
/// with children in the search graph gets `min over leaves l reachable from it of
/// C(n, l) + h(l)`; goal nodes get zero; leaves keep their value.
///
/// The leaves are the nodes without children (never expanded, or expanded without
/// successors), seeded with their value before the update, as the paper's auxiliary node
/// `z` does. Only nodes with children have outgoing edges, so only they can improve.
pub(super) fn limited_horizon<G, T, H>(
    search: &Lookahead<'_, G, T>,
    _frontier: Frontier,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
) -> u64
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let seeds = search
        .children
        .iter()
        .zip(&search.h)
        .map(|(&children, &h)| if children { None } else { h })
        .collect();
    let mut values = backward_shortest_paths(search, seeds, |_| true);

    for (node, value) in values.iter_mut().enumerate() {
        if search.goal.is_goal(search.space.state(node)) {
            *value = Some(G::Cost::ZERO);
        }
    }

    write(search, table, &values, |node| {
        search.children[node] || search.goal.is_goal(search.space.state(node))
    })
}

/// RTAA*'s update: with `f` the value of the lookahead's target, every closed node gets
/// `f - g(s)`. If the lookahead exhausted the reachable states, every closed node is a
/// proven dead end.
pub(super) fn rtaa<G, T, H>(
    search: &Lookahead<'_, G, T>,
    frontier: Frontier,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
) -> u64
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Cost: CostExt,
    T: ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let f = match frontier {
        Frontier::Target(target) => search.h[target].map(|h| search.space.g(target) + h),
        Frontier::Exhausted | Frontier::Limit => None,
    };
    let values: Vec<_> = (0..search.status.len())
        .map(|node| f.map(|f| f.saturating_sub(search.space.g(node))))
        .collect();

    write(search, table, &values, |node| {
        search.status[node] == Status::Closed
    })
}

/// Shortest distances to the seeds along the reversed explored edges: `values[n] = min over
/// seeds s reachable from n of dist(n, s) + seeds[s]`, where only nodes with `relax(n)`
/// may take a value through an edge.
///
/// Dijkstra's algorithm from every seed at once, with lazy deletion.
fn backward_shortest_paths<G, T>(
    search: &Lookahead<'_, G, T>,
    mut values: Vec<Option<G::Cost>>,
    relax: impl Fn(NodeId) -> bool,
) -> Vec<Option<G::Cost>>
where
    G: Successors,
    T: ?Sized,
{
    let predecessors = reversed_edges(search);
    let mut heap: BinaryHeap<_> = values
        .iter()
        .enumerate()
        .filter_map(|(node, value)| value.map(|value| Reverse((value, node))))
        .collect();

    while let Some(Reverse((value, node))) = heap.pop() {
        if values[node] != Some(value) {
            continue;
        }

        for &(previous, cost) in &predecessors[node] {
            let via = cost.saturating_add(value);
            if relax(previous) && values[previous].is_none_or(|old| via < old) {
                values[previous] = Some(via);
                heap.push(Reverse((via, previous)));
            }
        }
    }

    values
}

/// The explored edges reversed: `result[to]` lists `(from, cost)` for every edge.
fn reversed_edges<G: Successors, T: ?Sized>(
    search: &Lookahead<'_, G, T>,
) -> Vec<Vec<(NodeId, G::Cost)>> {
    let mut predecessors = alloc::vec![Vec::new(); search.status.len()];
    for &(from, to, cost) in &search.edges {
        predecessors[to].push((from, cost));
    }

    predecessors
}

/// Writes `values[n]` into the table for every node with `selected(n)`, and returns how
/// many values changed.
fn write<G, T, H>(
    search: &Lookahead<'_, G, T>,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    values: &[Option<G::Cost>],
    selected: impl Fn(NodeId) -> bool,
) -> u64
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    T: ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let mut changed = 0;
    for (node, &value) in values.iter().enumerate() {
        if selected(node) && table.set(search.space.state(node).clone(), value) {
            changed += 1;
        }
    }

    changed
}
