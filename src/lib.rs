//! # Heuristic search
//!
//! Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal,
//! bidirectional and incremental search.
//!
//! A problem is described by small capability traits instead of one universal problem
//! trait: a graph implements [`Successors`], the goal test is a [`Goal`] and the estimate a
//! [`Heuristic`]. Both are implemented for closures. Every search returns a
//! [`SearchResult`] with an [`Outcome`] (a [`Solution`] path, unsolvable, or limit reached)
//! and [`Stats`]. A [`Budget`] (such as [`Limits`], or a closure that checks a clock) stops a
//! search early.
//!
//! The crate is `no_std` and needs only `alloc`.
//!
//! ## Algorithms
//!
//! - [`bfs()`]: breadth-first search, fewest edges.
//! - [`dijkstra`]: uniform-cost search, cheapest path.
//! - [`astar`]: A*, cheapest path with an admissible heuristic.
//! - [`weighted_astar`]: Weighted A*, at most `w` times the cheapest cost.
//! - [`gbfs`]: greedy best-first search, fast and unbounded.
//!
//! ## Example
//!
//! ```
//! use heuristic_search::{Limits, Successors, dijkstra};
//!
//! /// A directed graph as adjacency lists of `(target, cost)` pairs.
//! struct Graph(Vec<Vec<(usize, u32)>>);
//!
//! impl Successors for Graph {
//!     type State = usize;
//!     type Action = usize;
//!     type Cost = u32;
//!
//!     fn successors(&self, state: &usize, mut emit: impl FnMut(usize, usize, u32)) {
//!         for &(target, cost) in &self.0[*state] {
//!             emit(target, target, cost);
//!         }
//!     }
//! }
//!
//! let graph = Graph(vec![vec![(1, 5), (2, 1)], vec![(3, 1)], vec![(1, 1)], vec![]]);
//! let result = dijkstra(&graph, 0, &|state: &usize| *state == 3, Limits::NONE);
//! let solution = result.solution().expect("3 is reachable");
//!
//! assert_eq!(solution.states, [0, 2, 1, 3]);
//! assert_eq!(solution.cost, 3);
//! ```
//!
//! ## Planned scope
//!
//! - Foundational: DFS, IDA*
//! - Weighted and anytime: ARA*, ANA*
//! - Bounded-suboptimal: EES, DPS
//! - Multiple heuristics: MHA*
//! - Bidirectional: NBS, BAE*, DIBBS
//! - Expensive edge evaluation: `LazySP`
//! - Dynamic graphs: LPA*, D* Lite, AD*
//!
//! See `docs/design.md` in the repository for the design and milestones.
//!
//! ## Status
//!
//! Pre-release. The API will change.
//!
//! Repository: <https://github.com/sunsided/heuristic-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod best_first;
mod bfs;
mod cost;
mod outcome;
mod problem;
mod space;

pub use best_first::{GraphResult, astar, dijkstra, gbfs, weighted_astar};
pub use bfs::bfs;
pub use cost::{Cost, Weight};
pub use outcome::{Budget, Limits, Outcome, SearchResult, Solution, Stats};
pub use problem::{Goal, Heuristic, Successors, Zero};
