# heuristic-search

Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal, bidirectional and incremental search.

## Status

Pre-release; the API will change. See [docs/design.md](docs/design.md).

Implemented on the `draft/initial` branch:

- Foundational: BFS, Dijkstra, A*, greedy best-first search (GBFS), each also from several
  start states
- Weighted and anytime: Weighted A*, ARA*, ANA*
- Bounded-suboptimal: focal search (with a public focal list and tie-breaking), EES, DPS
- Bidirectional: bidirectional BFS and Dijkstra, NBS, DIBBS, BAE*

Planned:

- Foundational: DFS, IDA*
- Multiple heuristics: MHA*
- Expensive edge evaluation: LazySP
- Dynamic graphs: LPA*, D* Lite, AD*

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
