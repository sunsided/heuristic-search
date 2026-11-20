# heuristic-search

Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal, bidirectional and incremental search.

## Status

Pre-release; the API will change. See [docs/design.md](docs/design.md).

Implemented on the `draft/initial` branch:

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
- Weighted and anytime: Weighted A*, ARA*, ANA*
- Bounded-suboptimal: focal search (with a public focal list and tie-breaking), EES, DPS
- Bidirectional: bidirectional BFS and Dijkstra, NBS, DIBBS, BAE*; meets by a user
  relation (`MeetRelation`) between states and backward nodes of their own type (such
  as subgoals), with path replay or a typed `ReplayError`
- Multiple heuristics: shared MHA*
- Expensive edge evaluation: LazySP
- Dynamic graphs: LPA*, D* Lite, AD* (start moves keyed by D* Lite's `km`); Field D*
  (`FieldDStar`, feature `float`) over the corners of a `CostGrid` with per-cell costs,
  interpolated any-angle paths through cell edges, cell-change repair and a moving start
- Naming: every A* variant is `*_star` (`a_star`, `weighted_a_star`, ...); `astar` and
  `weighted_astar` remain as aliases
- Shared traits: `Cost`, `Heuristic`, `Goal` and `Budget<Stats>` come from `search-core`
  and are re-exported from the crate root. Costs are the unsigned integers, `Decimal<K>`
  and (feature `float`) `NonNegF64`; signed and raw `f64` costs were dropped. The design
  document lists the breaking changes under "Cross-crate needs".

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
