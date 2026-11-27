//! # Heuristic search
//!
//! Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal,
//! bidirectional, incremental and real-time search.
//!
//! A problem is described by small capability traits instead of one universal problem
//! trait: a graph implements [`Successors`], the goal test is a [`Goal`] and the estimate a
//! [`Heuristic`]. Goal tests can be closures; [`heuristic_fn`] wraps a closure as a
//! heuristic. Searches own their heuristic; pass `&mut h` to lend one and keep its
//! workspace. Every search returns a [`SearchResult`] with an [`Outcome`] (a [`Solution`]
//! path, unsolvable, or limit reached) and [`Stats`]. A [`Budget`] (such as [`Limits`], or a
//! closure that checks a clock) stops a search early.
//!
//! The crate is `no_std` and needs only `alloc` (feature `parallel` adds `std`).
//!
//! ## Algorithms
//!
//! - [`bfs()`]: breadth-first search, fewest edges.
//! - [`dijkstra`]: uniform-cost search, cheapest path.
//! - [`a_star`]: A*, cheapest path with an admissible heuristic.
//! - [`weighted_a_star`]: Weighted A*, at most `w` times the cheapest cost.
//! - [`gbfs`]: greedy best-first search, fast and unbounded.
//! - [`fringe_search`]: fringe search, cheapest path with an admissible heuristic through
//!   `f` threshold iterations over a cache of every state, without a priority queue.
//!
//! Each has a `*_multi` variant ([`a_star_multi`] and so on) that searches from several
//! start states at once. So does every other search below; the incremental planners have
//! multi-start types ([`LpaStarMulti`] and so on) with fixed starts.
//!
//! Depth-first searches need memory only for the current path (except [`dfs`]):
//!
//! - [`dfs`]: graph depth-first search, some path.
//! - [`depth_limited_dfs`] and [`iddfs`]: depth-limited and iterative deepening, fewest
//!   edges.
//! - [`ida_star`]: IDA*, cheapest path with an admissible heuristic; [`ida_star_tt`] adds
//!   a bounded transposition table for graphs with many paths to each state.
//! - [`rbfs()`]: recursive best-first search (RBFS), cheapest path with an admissible
//!   heuristic, in best-first order with memory linear in the depth.
//!
//! [`sma_star`] (SMA*) is A* within a fixed number of nodes: it forgets its worst leaves
//! and regenerates them when needed, and returns the cheapest path among those that fit in
//! memory, or [`Outcome::Cutoff`] when none does.
//!
//! Anytime searches report each improved solution and return an [`AnytimeResult`]:
//!
//! - [`ara_star`]: ARA*, Weighted A* with a falling weight schedule.
//! - [`ana_star`]: ANA*, anytime without parameters.
//! - [`beam_stack_search`]: beam-stack search, a beam of fixed width that backtracks over
//!   what it pruned; optimal once complete, in memory bounded by the width per layer.
//!
//! [`beam_search`] is breadth-first beam search: a fixed number of states per layer, fast
//! but incomplete ([`Outcome::Cutoff`] when the width may have pruned the goal away).
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
//! The frontiers meet by state equality. The `*_meet` variants ([`nbs_meet`],
//! [`bidirectional_bfs_meet`] and so on) meet by a [`MeetRelation`] instead, such as a
//! state satisfying a subgoal for regression over partial states of their own type (a
//! [`RegressionGraph`]), and replay the backward actions to return a path of real states
//! or a [`ReplayError`].
//!
//! Grid searches work on a [`UniformGrid`], 8-connected without corner cutting (see
//! [`UniformGrid`]), with the exact octile cost [`Octile`]:
//!
//! - [`jps`]: Jump Point Search, cheapest path, expanding only jump points.
//! - `theta_star` and `lazy_theta_star` (feature `float`): Theta* and Lazy Theta*,
//!   any-angle paths between cell centres on a [`LineOfSight`] grid, with Euclidean
//!   float costs; never longer than the cheapest 8-connected path (Theta*).
//!
//! [`EightConnected`] makes a grid a plain graph for the other searches.
//!
//! [`mha_star`] (shared multi-heuristic A*) combines one consistent anchor heuristic with
//! any number of inadmissible ones, within `w1 * w2` times the optimum.
//!
//! `hda_star` (feature `parallel`) is hash-distributed A* (HDA*): one A* search over
//! several threads, each owning the states that hash to it; the same cost as [`a_star`].
//!
//! [`lazy_sp`] (`LazySP`) finds cheapest paths when true edge costs are expensive, evaluating
//! them through an [`EdgeEvaluator`] only on candidate paths.
//!
//! Incremental planners keep their search tree while edge costs of a [`DynamicGraph`]
//! change, and repair only what the changes affect:
//!
//! - [`LpaStar`]: Lifelong Planning A*, fixed start and goal.
//! - [`DStarLite`]: D* Lite, fixed goal and a moving start.
//! - [`AdStar`]: Anytime D*, D* Lite with an inflated heuristic the caller lowers.
//! - `FieldDStar` (feature `float`): Field D*, D* Lite over the corners of a `CostGrid`
//!   with per-cell costs, whose paths cross cells at any angle through points on cell
//!   edges.
//!
//! Real-time searches plan a bounded amount, move, and learn values in a
//! [`LearnedHeuristic`] the caller keeps across trials; each call runs one trial and
//! returns a [`TrialResult`]:
//!
//! - [`lrta_star`] and [`rta_star`]: LRTA* and RTA*, a lookahead of one expansion.
//! - [`lss_lrta_star`]: LSS-LRTA*, an A* lookahead of `n` expansions and a Dijkstra update.
//! - [`lhb_lrta_star`]: the same agent with the limited-horizon Bellman update of Hadar et
//!   al.; [`lhb_update`] is that update on its own, for offline learning.
//! - [`rtaa_star`]: RTAA*, the A* lookahead with a cheaper update.
//!
//! Costs are generic over [`Cost`]. For planning, `u32` is the intended default: action
//! costs are small integers and heuristics over them stay exact. [`Cost`], [`Heuristic`],
//! [`Goal`] and [`Budget`] come from `search-core` and are re-exported here; [`CostExt`]
//! adds the subtraction and float conversion a few searches need.
//!
//! ## Features
//!
//! - `float`: re-exports the checked float cost `NonNegF64` from `search-core`, and enables
//!   the any-angle grid searches `theta_star` and `lazy_theta_star` and the Field D*
//!   planner `FieldDStar` with its `CostGrid`, whose Euclidean costs are floats.
//! - `parallel`: multi-threaded searches; needs `std`. Enables hash-distributed A*,
//!   `hda_star`.
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
//! ## Naming
//!
//! Searches are snake-case functions; every A* variant ends in `_star` ([`a_star`],
//! [`weighted_a_star`], [`ida_star`], [`ara_star`], [`ana_star`], [`mha_star`],
//! [`bae_star`]). Suffixes compose in this order: `_tt` (a variant of the algorithm),
//! `_multi` (several starts), `_meet` (a meet relation, always with several starts).
//! Incremental planners keep state between calls, so they are types ([`LpaStar`],
//! [`DStarLite`], [`AdStar`], `FieldDStar`).
//!
//! See `docs/design.md` in the repository for the design.
//!
//! Repository: <https://github.com/sunsided/heuristic-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
#[cfg(feature = "parallel")]
extern crate std;

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

mod anytime;
mod beam;
mod best_first;
mod bfs;
mod bidirectional;
mod bounded;
mod cost;
mod depth_first;
mod dynamic;
mod flow;
mod focal;
mod fringe;
mod grid;
#[cfg(feature = "parallel")]
mod hda;
mod indexed_heap;
mod keyed_set;
mod lazy;
mod mha;
mod open_list;
mod outcome;
mod problem;
mod rbfs;
mod real_time;
mod sma;
mod space;

pub use anytime::{
    AnytimeGraphResult, AnytimeResult, ana_star, ana_star_multi, ara_star, ara_star_multi,
};
pub use beam::{beam_search, beam_search_multi, beam_stack_search, beam_stack_search_multi};
pub use best_first::{
    AStarSearch, GraphResult, Workspace, a_star, a_star_in, a_star_multi, a_star_multi_in,
    dijkstra, dijkstra_in, dijkstra_multi, dijkstra_multi_in, gbfs, gbfs_in, gbfs_multi,
    gbfs_multi_in, weighted_a_star, weighted_a_star_in, weighted_a_star_multi,
    weighted_a_star_multi_in,
};
pub use bfs::{bfs, bfs_multi};
pub use bidirectional::{
    BidirectionalGraph, MeetFn, MeetRelation, MeetResult, RegressionGraph, ReplayError, bae_star,
    bae_star_meet, bae_star_multi, bidirectional_bfs, bidirectional_bfs_meet,
    bidirectional_bfs_multi, bidirectional_dijkstra, bidirectional_dijkstra_meet,
    bidirectional_dijkstra_multi, dibbs, dibbs_meet, dibbs_multi, meet_fn, nbs, nbs_meet,
    nbs_multi,
};
pub use bounded::{
    EesEstimates, FocalFn, FocalKey, dps, dps_multi, ees, ees_multi, focal_fn, focal_search,
    focal_search_multi,
};
pub use cost::{CostExt, Weight};
pub use depth_first::{
    depth_limited_dfs, depth_limited_dfs_multi, dfs, dfs_multi, ida_star, ida_star_multi,
    ida_star_tt, ida_star_tt_multi, iddfs, iddfs_multi,
};
pub use dynamic::{
    AdStar, AdStarMulti, DStarLite, DStarLiteMulti, DynamicGraph, DynamicResult, LpaStar,
    LpaStarMulti,
};
#[cfg(feature = "float")]
pub use dynamic::{FieldDStar, FieldResult};
pub use flow::FlowField;
pub use focal::{FocalList, TieBreak};
pub use fringe::{fringe_search, fringe_search_multi};
#[cfg(feature = "float")]
pub use grid::{
    AnyAngleResult, CostGrid, lazy_theta_star, lazy_theta_star_multi, theta_star, theta_star_multi,
};
pub use grid::{
    EightConnected, GridCell, GridStep, LineOfSight, Octile, OctileDistance, UniformGrid, jps,
    jps_multi,
};
#[cfg(feature = "parallel")]
pub use hda::{hda_star, hda_star_multi};
pub use lazy::{EdgeEvaluator, EdgeSelector, lazy_sp, lazy_sp_multi};
pub use mha::{MhaWeights, mha_star, mha_star_multi};
pub use outcome::{Limits, Outcome, SearchResult, Solution, Stats};
pub use problem::{Predecessors, Successors};
pub use rbfs::{rbfs, rbfs_multi};
pub use real_time::{
    LearnedHeuristic, TrialResult, TrialStats, lhb_lrta_star, lhb_update, lrta_star, lss_lrta_star,
    rta_star, rtaa_star,
};
#[cfg(feature = "float")]
pub use search_core::NonNegF64;
pub use search_core::{Budget, Cost, Decimal, Goal, Heuristic, HeuristicFn, Zero, heuristic_fn};
pub use sma::{sma_star, sma_star_multi};
