# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/sunsided/heuristic-search/compare/v0.3.0...v0.4.0) - 2026-11-27

### Breaking

- `hierarchical_a_star_par` and `hierarchical_a_star_par_multi` take `budget: impl Budget<Stats> + Send`: the refining threads share the budget ([#41](https://github.com/sunsided/heuristic-search/pull/41))

### Changed

- Budget the refinement of hierarchical A* queries ([#41](https://github.com/sunsided/heuristic-search/pull/41))
- Resume Weighted A*, GBFS and Dijkstra like A* ([#40](https://github.com/sunsided/heuristic-search/pull/40))
- Make the refinement budget tests stop inside refinement
- Budget the refinement of hierarchical A* queries
- Resume Weighted A*, GBFS and Dijkstra like A*
- Rebuild a flow field in place for a moving goal

## [0.3.0](https://github.com/sunsided/heuristic-search/compare/v0.2.2...v0.3.0) - 2026-11-27

### Added

- Add a separate focal value to FocalList ([#36](https://github.com/sunsided/heuristic-search/pull/36))
- Add runnable examples on realistic generated inputs ([#32](https://github.com/sunsided/heuristic-search/pull/32))
- Add a separate focal value to FocalList
- Add a learning heuristics example for the real-time agents
- Add a rover replanning example for Real-time D*
- Add an indoor navigation example for hierarchical A*
- Add road network examples for CH, ALT and MM
- Add crowd pathfinding and parallel HDA* examples on a game map
- Add hierarchical A* over a region graph with parallel refinement
- Add GBFS and BWAS search phases to the LHB real-time search
- Add Real-time D* (RealTimeDStar)
- Add contraction hierarchies for many queries on static graphs
- Add MM bidirectional search that meets in the middle
- Add ALT landmark heuristics for repeated queries
- Add solve_batch for independent queries on several threads
- Add flow fields from a goal set
- Add hash-distributed A* (HDA*) behind the parallel feature
- Add the parallel feature for multi-threaded searches

### Changed

- Bring three examples in line with the merged library fixes
- Spawn refinement threads only when the work pays for them ([#37](https://github.com/sunsided/heuristic-search/pull/37))
- Keep MM's f and g minima in lazy heaps filled at the first meet ([#35](https://github.com/sunsided/heuristic-search/pull/35))
- Batch HDA* work counting and cache the incumbent bound ([#34](https://github.com/sunsided/heuristic-search/pull/34))
- Raise the default witness search limit to 256 settled states ([#33](https://github.com/sunsided/heuristic-search/pull/33))
- Name both limits of the saturation figure in the CH notes
- Keep plain FocalList pushes as cheap as before the focal value
- Spawn refinement threads only when the work pays for them
- Keep MM's f and g minima in lazy heaps filled at the first meet
- Batch HDA* work counting and cache the incumbent bound
- Raise the default witness search limit to 256 settled states
- Make the ida_star_tt example independent of the hash seed ([#31](https://github.com/sunsided/heuristic-search/pull/31))
- Share one generator across the examples and list them in the README
- Show the batch detour and keep the example prose to what is measured
- Trim the MM example to 300 queries and contrast it with A*
- Make the ida_star_tt example independent of the hash seed
- Update the suffix order in the Copilot instructions
- Test hierarchical A* on directed graphs and bench it on the wall grid
- Keep the lookahead's batch buffer when a search ends mid-batch
- Forget learned dead ends after edge changes in Real-time D*
- Shrink the contraction hierarchy property tests
- Spell out MM's C in its stopping rule
- Check the f half of MM's expansion bound
- Document workspaces, resumable A*, flow fields and batches
- Test multi-start workspace searches that expand
- Assert that graphs, planners and results are Send and Sync
- Reuse best-first workspaces and resume A* across calls
- Describe the bounded lattice search in the Field D* design notes
- Stop Field D*'s lattice A* once it cannot beat the interpolated path

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
