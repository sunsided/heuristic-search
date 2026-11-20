//! Capability traits that describe a search graph: [`Successors`] and [`Predecessors`].
//!
//! The goal test [`Goal`] and the estimate [`Heuristic`] come from `search-core` and are
//! re-exported at the crate root.
//!
//! There is no single "search problem" trait. Each algorithm asks for the capabilities it
//! needs, and the goal test and heuristic are separate values, so one graph can be searched
//! with many goals and heuristics. [`Goal`] is implemented for plain closures, and
//! [`heuristic_fn`] turns a closure into a [`Heuristic`].
//!
//! [`Goal`] keeps `&self` and is taken by reference: goal tests are pure predicates that need
//! no workspace, and keeping the closure blanket impl lets callers pass `&|s| ...` directly.

use alloc::vec::Vec;

use search_core::Cost;
#[cfg(doc)]
use search_core::{Goal, Heuristic, heuristic_fn};

/// A graph that can enumerate the successors of a state.
///
/// Successors are reported through a callback, so implementations need no
/// allocation.
pub trait Successors {
    /// A node of the search graph.
    ///
    /// Searches clone states into their search space and hash them to detect duplicates.
    type State;
    /// The label of an edge, such as an operator id. Returned in solutions.
    type Action;
    /// The cost of an edge. Must be non-negative for the optimality guarantees to hold.
    type Cost: Cost;

    /// Calls `emit` once for every outgoing edge of `state`.
    fn successors(
        &self,
        state: &Self::State,
        emit: impl FnMut(Self::Action, Self::State, Self::Cost),
    );
}

/// A graph that can enumerate the predecessors of a node, for backward and bidirectional
/// search.
///
/// The backward node type may differ from the forward state type: planning regresses over
/// partial states (subgoals), not full states. The bidirectional searches in this crate
/// need `Node = State`, the same `Action` and `Cost` as [`Successors`], and detect a meeting
/// of the two frontiers by state equality.
pub trait Predecessors {
    /// A node of the backward search graph.
    type Node;
    /// The label of an edge, the same as the forward label of that edge.
    type Action;
    /// The cost of an edge. Must be non-negative.
    type Cost: Cost;

    /// Calls `emit(action, predecessor, cost)` once for every edge
    /// `predecessor --action--> node`.
    ///
    /// Must agree with the forward edges: `p` is reported as a predecessor of `n` with
    /// `action` and `cost` exactly when `n` is a successor of `p` with them.
    fn predecessors(
        &self,
        node: &Self::Node,
        emit: impl FnMut(Self::Action, Self::Node, Self::Cost),
    );
}

/// Appends the predecessors of `node` to `buffer` and returns how many there were.
pub(crate) fn collect_predecessors<G: Predecessors>(
    graph: &G,
    node: &G::Node,
    buffer: &mut Vec<(G::Action, G::Node, G::Cost)>,
) -> u64 {
    let before = buffer.len();
    graph.predecessors(node, |action, previous, cost| {
        buffer.push((action, previous, cost));
    });

    (buffer.len() - before) as u64
}

/// Appends the successors of `state` to `buffer` and returns how many there were.
pub(crate) fn collect_successors<G: Successors>(
    graph: &G,
    state: &G::State,
    buffer: &mut Vec<(G::Action, G::State, G::Cost)>,
) -> u64 {
    let before = buffer.len();
    graph.successors(state, |action, next, cost| {
        buffer.push((action, next, cost));
    });

    (buffer.len() - before) as u64
}
