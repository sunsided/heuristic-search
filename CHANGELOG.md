# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2](https://github.com/sunsided/heuristic-search/compare/v0.2.1...v0.2.2) - 2026-11-22

### Changed

- Link the crate to its page on the planning and search site

## [0.2.1](https://github.com/sunsided/heuristic-search/compare/v0.2.0...v0.2.1) - 2026-11-21

### Changed

- Describe the third exhaustive 3 x 3 corner pair correctly
- Justify why a finite Field D* g(start) implies a lattice path
- Format the Field D* lattice fallback and its tests
- Never return a Field D* path above the 8-connected optimum

## [0.2.0](https://github.com/sunsided/heuristic-search/compare/v0.1.0...v0.2.0) - 2026-11-21

### Changed

- Describe dependents as crates.io users in the release instructions
- Require search-core 0.3.0

### Removed

- Remove the astar and weighted_astar aliases

## [0.1.0](https://github.com/sunsided/heuristic-search/releases/tag/v0.1.0) - 2026-11-21

### Added

- Add CONTRIBUTING.md
- Add README badges, a usage example and absolute links
- Add cargo-fuzz targets for the graph and grid searches
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos configuration
- Add the codecov configuration
- Add a Taskfile and CI workflows that mirror each other
- Add real-time search with learned heuristic tables
- Add Field D* over cell corners with interpolated costs
- Add simplified memory-bounded A* (SMA*)
- Add Theta* and Lazy Theta* with a line-of-sight grid trait
- Add Jump Point Search with an exact octile cost
- Add beam search and beam-stack search
- Add recursive best-first search (RBFS)
- Add fringe search
- Add multi-start LPA*, D* Lite and AD* through a virtual super-start
- Add IDA* with a bounded transposition table and Outcome::Cutoff
- Add multi-start variants of every non-incremental search
- Add DFS, depth-limited DFS, IDDFS and IDA*
- Add benchmarks for the hot paths
- Add AD* on the incremental core
- Add LPA* and D* Lite over a DynamicGraph
- Add LazySP with an edge evaluator
- Add shared multi-heuristic A*
- Add bidirectional BFS, Dijkstra, NBS, DIBBS and BAE*
- Add focal search, EES and DPS
- Add ARA* and ANA* anytime searches
- Add examples to every search function
- Add design document
- Add best-first search over capability traits

### Changed

- Set the first release version to 0.1.0
- Write the documentation as it reads after the release
- Update the README status and add a "Who this is for" section
- Stop tracking Cargo.lock
- Record search-core from the crates.io registry in the locks
- Depend on search-core from crates.io
- Adopt the full lint table, clippy.toml and rustfmt.toml
- Render mermaid diagrams in the rustdoc output
- Test octile distance consistency and saturating octile arithmetic
- Track Cargo.lock
- Require search-core 0.3
- Require search-core 0.2
- Require search-core 0.1
- Accept search-core 0.0 and 0.1 during the 0.1.0 release
- Take CostExt from search-core
- Test the reopened-state difference and correct the learning notes
- Document real-time search and compare LHB with LSS-LRTA*
- Benchmark real-time search and print its learning curves
- Keep no-op actions as edges in the subgoal meet test model
- Widen Field D*'s tested path bound and batch the repair test
- Document Field D* in the crate docs, README and design notes
- Benchmark Field D* and look up each neighbour once per rhs
- Share a no_std f64 square root between the float grid searches
- Let incremental search models compute rhs themselves
- Document the new searches on shared types and test SMA* multi-start
- Point the status lines at draft/algorithms
- Meet over backward nodes of their own type, with typed replay errors
- Keep the proptest seed that caught stale AD* keys
- Document the meet replay fallback and heuristics on subgoals
- Re-key AD* after any weight change that can shrink keys
- Update the README status
- Name every A* variant *_star
- Benchmark the remaining searches
- Document when AD* re-keys its queue
- Describe AD*'s km keys in the design document
- Cost incremental paths by the cheapest parallel edge
- Key AD* with D* Lite's km instead of re-keying every iteration
- Meet bidirectional frontiers by a user relation
- Bucket the focal list by f and keep focal in an indexed heap
- Benchmark the focal list under decrease-key churn
- Update the design document for the search-core traits
- Adopt search-core for Cost, Heuristic, Goal and Budget
- Update the design document for milestones 2 to 6
- Diagram the NBS loop and the incremental core
- Tighten budgets and tests of the new searches
- Document heuristic ownership and search-core candidates
- Let heuristics keep a mutable workspace
- Document u32 costs, ceil rounding and multiple starts
- Search from several start states
- Saturate the best-first key to avoid overflow
- Use an independent oracle in property tests
- Test searches on small graphs, grids and random graphs
- Reserve the crate name with a 0.0.0 placeholder

### Dependencies

- Bump the cargo-major group across 1 directory with 2 updates

### Fixed

- Fix redundant and ambiguous rustdoc links
- Correct claims in the design document

### Removed

- Drop the Ordered cost wrapper
