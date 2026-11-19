//! Capability traits that describe a search problem: [`Successors`], [`Goal`] and
//! [`Heuristic`].
//!
//! There is no single "search problem" trait. Each algorithm asks for the capabilities it
//! needs, and the goal test and heuristic are separate values, so one graph can be searched
//! with many goals and heuristics. [`Goal`] and [`Heuristic`] are implemented for plain
//! closures.

use alloc::vec::Vec;

use crate::cost::Cost;

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

/// A goal test over states of type `S`.
///
/// Implemented for every `Fn(&S) -> bool`.
pub trait Goal<S> {
    /// Returns `true` if `state` is a goal state.
    fn is_goal(&self, state: &S) -> bool;
}

impl<S, F: Fn(&S) -> bool> Goal<S> for F {
    fn is_goal(&self, state: &S) -> bool {
        self(state)
    }
}

/// An estimate of the cost from a state to the nearest goal.
///
/// Implemented for every `Fn(&S) -> Option<C>`.
///
/// A* and Dijkstra return optimal solutions only if the heuristic is admissible (never
/// overestimates). They stay optimal with inconsistent heuristics, because states are
/// reopened when a cheaper path to them turns up.
pub trait Heuristic<S, C> {
    /// Estimates the remaining cost from `state` to a goal.
    ///
    /// `None` means no goal is reachable from `state` (a proven dead end). Searches drop such
    /// states, so a heuristic must only return `None` when that is certain.
    fn estimate(&self, state: &S) -> Option<C>;
}

impl<S, C, F: Fn(&S) -> Option<C>> Heuristic<S, C> for F {
    fn estimate(&self, state: &S) -> Option<C> {
        self(state)
    }
}

/// The blind heuristic: estimates `0` everywhere. Turns A* into Dijkstra's algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Zero;

impl<S, C: Cost> Heuristic<S, C> for Zero {
    fn estimate(&self, _state: &S) -> Option<C> {
        Some(C::ZERO)
    }
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
