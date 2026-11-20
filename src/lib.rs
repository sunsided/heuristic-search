//! # Heuristic search
//!
//! Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal,
//! bidirectional and incremental search.
//!
//! A problem is described by small capability traits instead of one universal problem
//! trait: a graph implements [`Successors`], the goal test is a [`Goal`] and the estimate a
//! [`Heuristic`]. Goal tests can be closures; [`heuristic_fn`] wraps a closure as a
//! heuristic. Searches own their heuristic; pass `&mut h` to lend one and keep its
//! workspace. Every search returns a [`SearchResult`] with an [`Outcome`] (a [`Solution`]
//! path, unsolvable, or limit reached) and [`Stats`]. A [`Budget`] (such as [`Limits`], or a
//! closure that checks a clock) stops a search early.
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
//! Each has a `*_multi` variant ([`astar_multi`] and so on) that searches from several
//! start states at once.
//!
//! Anytime searches report each improved solution and return an [`AnytimeResult`]:
//!
//! - [`ara_star`]: ARA*, Weighted A* with a falling weight schedule.
//! - [`ana_star`]: ANA*, anytime without parameters.
//!
//! Costs are generic over [`Cost`]. For planning, `u32` is the intended default: action
//! costs are small integers and heuristics over them stay exact.
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

mod anytime;
mod best_first;
mod bfs;
mod cost;
mod open_list;
mod outcome;
mod problem;
mod space;

pub use anytime::{AnytimeGraphResult, AnytimeResult, ana_star, ara_star};
pub use best_first::{
    GraphResult, astar, astar_multi, dijkstra, dijkstra_multi, gbfs, gbfs_multi, weighted_astar,
    weighted_astar_multi,
};
pub use bfs::{bfs, bfs_multi};
pub use cost::{Cost, Weight};
pub use outcome::{Budget, Limits, Outcome, SearchResult, Solution, Stats};
pub use problem::{Goal, Heuristic, HeuristicFn, Successors, Zero, heuristic_fn};
