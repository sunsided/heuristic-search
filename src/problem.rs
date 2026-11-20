//! Capability traits that describe a search problem: [`Successors`], [`Goal`] and
//! [`Heuristic`].
//!
//! There is no single "search problem" trait. Each algorithm asks for the capabilities it
//! needs, and the goal test and heuristic are separate values, so one graph can be searched
//! with many goals and heuristics. [`Goal`] is implemented for plain closures, and
//! [`heuristic_fn`] turns a closure into a [`Heuristic`].
//!
//! [`Goal`] keeps `&self` and is taken by reference: goal tests are pure predicates that need
//! no workspace, and keeping the closure blanket impl lets callers pass `&|s| ...` directly.

use alloc::boxed::Box;
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
/// `estimate` takes `&mut self`, so a heuristic can keep a reusable workspace (scratch
/// buffers, caches) without interior mutability. Searches take the heuristic by value and
/// own it for the run; pass `&mut h` to keep `h` and its workspace afterwards. Implemented
/// for `&mut H` and `Box<H>` of any heuristic `H`, for [`Zero`], and for closures through
/// [`heuristic_fn`].
///
/// A* and Dijkstra return optimal solutions only if the heuristic is admissible (never
/// overestimates). They stay optimal with inconsistent heuristics, because states are
/// reopened when a cheaper path to them turns up.
pub trait Heuristic<S, C> {
    /// Estimates the remaining cost from `state` to a goal.
    ///
    /// `None` means no goal is reachable from `state` (a proven dead end). Searches drop such
    /// states, so a heuristic must only return `None` when that is certain.
    fn estimate(&mut self, state: &S) -> Option<C>;
}

impl<S, C, H: Heuristic<S, C> + ?Sized> Heuristic<S, C> for &mut H {
    fn estimate(&mut self, state: &S) -> Option<C> {
        (**self).estimate(state)
    }
}

impl<S, C, H: Heuristic<S, C> + ?Sized> Heuristic<S, C> for Box<H> {
    fn estimate(&mut self, state: &S) -> Option<C> {
        (**self).estimate(state)
    }
}

/// Turns a closure `FnMut(&S) -> Option<C>` into a [`Heuristic`].
///
/// A blanket impl for closures is not possible next to the one for `&mut H`: `&mut F` is
/// itself a closure, so the two impls would overlap.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Heuristic, heuristic_fn};
///
/// let mut calls = 0;
/// let mut h = heuristic_fn(|n: &u32| {
///     calls += 1;
///     10_u32.checked_sub(*n)
/// });
///
/// assert_eq!(h.estimate(&3), Some(7));
/// assert_eq!(h.estimate(&12), None);
/// drop(h);
/// assert_eq!(calls, 2);
/// ```
pub fn heuristic_fn<S, C, F: FnMut(&S) -> Option<C>>(f: F) -> HeuristicFn<F> {
    HeuristicFn(f)
}

/// A closure used as a [`Heuristic`]; created by [`heuristic_fn`].
#[derive(Debug, Clone, Copy)]
pub struct HeuristicFn<F>(F);

impl<S, C, F: FnMut(&S) -> Option<C>> Heuristic<S, C> for HeuristicFn<F> {
    fn estimate(&mut self, state: &S) -> Option<C> {
        (self.0)(state)
    }
}

/// The blind heuristic: estimates `0` everywhere. Turns A* into Dijkstra's algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Zero;

impl<S, C: Cost> Heuristic<S, C> for Zero {
    fn estimate(&mut self, _state: &S) -> Option<C> {
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
