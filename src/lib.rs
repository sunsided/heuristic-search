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
//! Bounded-suboptimal searches return a solution within `w` times the optimum:
//!
//! - [`focal_search`]: expands by a free [`FocalKey`] among `f <= w * f_min`. Its open list,
//!   [`FocalList`], is public for searches with their own loop.
//! - [`ees`]: Explicit Estimation Search, guided by inadmissible estimates.
//! - [`dps`]: Dynamic Potential Search, needs only `h`.
//!
//! Bidirectional searches need [`Predecessors`] as well (a [`BidirectionalGraph`]) and
//! explicit goal states:
//!
//! - [`bidirectional_bfs`] and [`bidirectional_dijkstra`]: blind baselines.
//! - [`nbs`]: Near-Optimal Bidirectional Search, pairs of expansions.
//! - [`dibbs`] and [`bae_star`]: DIBBS and BAE*, ordered by `2g + h - h_opposite`.
//!
//! [`mha_star`] (shared multi-heuristic A*) combines one consistent anchor heuristic with
//! any number of inadmissible ones, within `w1 * w2` times the optimum.
//!
//! [`lazy_sp`] (`LazySP`) finds cheapest paths when true edge costs are expensive, evaluating
//! them through an [`EdgeEvaluator`] only on candidate paths.
//!
//! Incremental planners keep their search tree while edge costs of a [`DynamicGraph`]
//! change, and repair only what the changes affect:
//!
//! - [`LpaStar`]: Lifelong Planning A*, fixed start and goal.
//! - [`DStarLite`]: D* Lite, fixed goal and a moving start.
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
//! - Dynamic graphs: AD*
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
mod bidirectional;
mod bounded;
mod cost;
mod dynamic;
mod focal;
mod keyed_set;
mod lazy;
mod mha;
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
pub use bidirectional::{
    BidirectionalGraph, bae_star, bidirectional_bfs, bidirectional_dijkstra, dibbs, nbs,
};
pub use bounded::{EesEstimates, FocalFn, FocalKey, dps, ees, focal_fn, focal_search};
pub use cost::{Cost, Weight};
pub use dynamic::{DStarLite, DynamicGraph, DynamicResult, LpaStar};
pub use focal::{FocalList, TieBreak};
pub use lazy::{EdgeEvaluator, EdgeSelector, lazy_sp};
pub use mha::{MhaWeights, mha_star};
pub use outcome::{Budget, Limits, Outcome, SearchResult, Solution, Stats};
pub use problem::{Goal, Heuristic, HeuristicFn, Predecessors, Successors, Zero, heuristic_fn};
