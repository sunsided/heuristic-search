# heuristic-search

Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal, bidirectional and incremental search.

## Status

Pre-release; the API will change. See [docs/design.md](docs/design.md).

Implemented on the `draft/initial` branch:

- Foundational: BFS, Dijkstra, A*, greedy best-first search (GBFS); DFS, depth-limited DFS,
  iterative deepening DFS, IDA*
- Every search except the incremental planners also from several start states
  (`*_multi`)
- Weighted and anytime: Weighted A*, ARA*, ANA*
- Bounded-suboptimal: focal search (with a public focal list and tie-breaking), EES, DPS
- Bidirectional: bidirectional BFS and Dijkstra, NBS, DIBBS, BAE*
- Multiple heuristics: shared MHA*
- Expensive edge evaluation: LazySP
- Dynamic graphs: LPA*, D* Lite, AD*
- Shared traits: `Cost`, `Heuristic`, `Goal` and `Budget<Stats>` come from `search-core`
  and are re-exported from the crate root. Costs are the unsigned integers, `Decimal<K>`
  and (feature `float`) `NonNegF64`; signed and raw `f64` costs were dropped. The design document lists
  the breaking changes under "Cross-crate needs".

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
