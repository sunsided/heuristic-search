# heuristic-search

Explicit-state heuristic graph search: A*, weighted and anytime, bounded-suboptimal, bidirectional and incremental search.

## Planned scope

- Foundational: BFS, DFS, Dijkstra, GBFS, A*, IDA*
- Weighted and anytime: Weighted A*, ARA*, ANA*
- Bounded-suboptimal: EES, DPS
- Multiple heuristics: MHA*
- Bidirectional: NBS, BAE*, DIBBS
- Expensive edge evaluation: LazySP
- Dynamic graphs: LPA*, D* Lite, AD*

## Status

Pre-release. Breadth-first search, Dijkstra, A*, Weighted A* and greedy best-first search
work on the `draft/initial` branch; the API will change. See [docs/design.md](docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
