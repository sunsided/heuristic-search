//! # Heuristic search
//!
//! Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal, bidirectional and incremental search.
//!
//! ## Planned scope
//!
//! - Foundational: BFS, DFS, Dijkstra, GBFS, A*, IDA*
//! - Weighted and anytime: Weighted A*, ARA*, ANA*
//! - Bounded-suboptimal: EES, DPS
//! - Multiple heuristics: MHA*
//! - Bidirectional: NBS, BAE*, DIBBS
//! - Expensive edge evaluation: `LazySP`
//! - Dynamic graphs: LPA*, D* Lite, AD*
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/heuristic-search>

#![no_std]

/// A graph that can enumerate the successors of a state.
///
/// Successors are reported through a callback, so implementations need no
/// allocation.
pub trait Successors {
    /// A node of the search graph.
    type State;
    /// The label of an edge.
    type Action;
    /// The cost of an edge.
    type Cost;

    /// Calls `emit` once for every outgoing edge of `state`.
    fn successors(&self, state: &Self::State, emit: impl FnMut(Self::Action, Self::State, Self::Cost));
}
