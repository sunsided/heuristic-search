# heuristic-search

[![crates.io](https://img.shields.io/crates/v/heuristic-search.svg)](https://crates.io/crates/heuristic-search)
[![docs.rs](https://img.shields.io/docsrs/heuristic-search)](https://docs.rs/heuristic-search)
[![CI](https://github.com/sunsided/heuristic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/heuristic-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/heuristic-search/graph/badge.svg)](https://codecov.io/gh/sunsided/heuristic-search)
[![MSRV](https://img.shields.io/crates/msrv/heuristic-search)](https://crates.io/crates/heuristic-search)
[![License](https://img.shields.io/crates/l/heuristic-search)](https://github.com/sunsided/heuristic-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal, bidirectional, incremental and real-time search.

Part of the [planning and search crates](https://sunsided.github.io/planning-and-search/); see also this crate's [overview page](https://sunsided.github.io/planning-and-search/crates/heuristic-search/).

## Who this is for

Anyone who has a graph, or a problem that can be phrased as one, and needs a path through
it: a planner that generates its states on the fly, a robot or a game character on a grid.
The graph stays in your code; you describe it through small traits (`Successors`, and
`Predecessors` for the bidirectional searches), pass the goal test and the heuristic as
separate values, and get back a path, an outcome and the expansion counters. The crate
knows nothing about PDDL, SAS+ or any other model, so domain-specific representations sit
on top of it rather than in it.

It is also meant for comparing algorithms on the same problem: every search uses the same
traits, budgets and statistics, so trying IDA* or Weighted A* instead of A* changes the
call, not the problem description. Symbolic (BDD), multi-objective, stochastic and
width-based search are out of scope.

## What's included

See [docs/design.md](https://github.com/sunsided/heuristic-search/blob/main/docs/design.md)
for the design behind it.

- Foundational: BFS, Dijkstra, A*, greedy best-first search (GBFS); DFS, depth-limited DFS,
  iterative deepening DFS, IDA*, IDA* with a bounded transposition table; depth cutoffs
  are reported apart from budget stops (`Outcome::Cutoff`)
- Fringe search (`fringe_search`): optimal with an admissible heuristic, threshold
  iterations over a cache of `g` per state, no priority queue
- RBFS (`rbfs`): recursive best-first search, optimal with an admissible heuristic in
  memory linear in the depth, iterative so deep paths cannot overflow the stack
- SMA* (`sma_star`): A* within a node budget, forgetting and regenerating leaves; the
  cheapest path that fits in memory, `Cutoff` when none does
- Beam search (`beam_search`, incomplete; `Cutoff` when the width pruned) and beam-stack
  search (`beam_stack_search`): complete and anytime, optimal once it completes, with a
  fixed beam width
- Grids: Jump Point Search (`jps`) on any `UniformGrid`, 8-connected without corner
  cutting, with the exact octile cost `Octile` (`a + b√2`, compared without rounding);
  `EightConnected` makes a grid a graph for the other searches; any-angle Theta* and
  Lazy Theta* (`theta_star`, `lazy_theta_star`, feature `float`) on a `LineOfSight` grid
- Every search also from several start states (`*_multi`; the incremental planners as
  `LpaStarMulti`, `DStarLiteMulti`, `AdStarMulti` with fixed starts)
- Many queries per frame: reusable workspaces for the best-first searches (`Workspace`
  with `a_star_in`, `dijkstra_in`, ...), resumable A* under a per-call budget
  (`AStarSearch`), flow fields from a goal set shared by every agent heading there
  (`FlowField`, also on `EightConnected` grids), and batches of queries on several
  threads (`solve_batch`, feature `parallel`); graphs, heuristics, planners and results
  are `Send + Sync` when their parameters are
- Weighted and anytime: Weighted A*, ARA*, ANA*
- Bounded-suboptimal: focal search (with a public focal list and tie-breaking), EES, DPS
- Bidirectional: bidirectional BFS and Dijkstra, NBS, DIBBS, BAE*, MM (and MM0); meets by a user
  relation (`MeetRelation`) between states and backward nodes of their own type (such
  as subgoals), with path replay or a typed `ReplayError`
- Multiple heuristics: shared MHA*
- Landmark heuristics: ALT lower bounds (`Landmarks`) from forward and backward
  distance tables, farthest-first or avoidance selection, active landmarks per query;
  for A*, the anytime planners and (towards the target and from the source) the
  bidirectional searches
- Expensive edge evaluation: LazySP
- Preprocessed queries: contraction hierarchies (`ContractionHierarchy`) for many exact
  queries on a finite graph with static costs; shortcuts unpack into the graph's own
  actions, and ranks and upward search spaces are readable for visualisation
- Dynamic graphs: LPA*, D* Lite, AD* (start moves keyed by D* Lite's `km`); Field D*
  (`FieldDStar`, feature `float`) over the corners of a `CostGrid` with per-cell costs,
  interpolated any-angle paths through cell edges, cell-change repair and a moving start;
  Real-time D* (`RealTimeDStar`), D* Lite and LSS-LRTA* under a per-move expansion limit
- Real-time search with learned tables that persist across trials (`LearnedHeuristic`):
  LRTA*, RTA*, LSS-LRTA* (A* lookahead, Dijkstra update), RTAA*, and the limited-horizon
  Bellman update of Hadar et al. 2025 (`lhb_update` offline, `lhb_lrta_star` inside the
  LSS-LRTA* agent); learning curves in `benches/learning.rs` and the design notes
- Parallel (feature `parallel`): hash-distributed A* (`hda_star`), one A* search over
  several threads that own the states by hash, with the same optimal cost as A*
- Naming: every A* variant is `*_star` (`a_star`, `weighted_a_star`, ...)
- Shared traits: `Cost`, `Heuristic`, `Goal` and `Budget<Stats>` come from `search-core`
  and are re-exported from the crate root. Costs are the unsigned integers, `Decimal<K>`
  and (feature `float`) `NonNegF64`.

## Usage

Add the crate with `cargo add heuristic-search`, or in `Cargo.toml`:

```toml
[dependencies]
heuristic-search = "0.1"
```

Then describe the graph and run a search:

```rust
use heuristic_search::{Limits, Successors, dijkstra};

/// A directed graph as adjacency lists of `(target, cost)` pairs.
struct Graph(Vec<Vec<(usize, u32)>>);

impl Successors for Graph {
    type State = usize;
    type Action = usize;
    type Cost = u32;

    fn successors(&self, state: &usize, mut emit: impl FnMut(usize, usize, u32)) {
        for &(target, cost) in &self.0[*state] {
            emit(target, target, cost);
        }
    }
}

let graph = Graph(vec![vec![(1, 5), (2, 1)], vec![(3, 1)], vec![(1, 1)], vec![]]);
let result = dijkstra(&graph, 0, &|state: &usize| *state == 3, Limits::NONE);
let solution = result.solution().expect("3 is reachable");

assert_eq!(solution.states, [0, 2, 1, 3]);
assert_eq!(solution.cost, 3);
```

The crate is `no_std` and needs only `alloc`. The feature `float` adds the float cost
`NonNegF64`, the any-angle grid searches Theta* and Lazy Theta*, and Field D*. The
feature `parallel` adds hash-distributed A* and `solve_batch` and needs `std`.

## MSRV

Rust 1.85 (`rust-version` in `Cargo.toml`). The MSRV is a signal, not a contract: it may
rise in a minor release.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/heuristic-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/heuristic-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions. See
[CONTRIBUTING.md](https://github.com/sunsided/heuristic-search/blob/main/CONTRIBUTING.md).
