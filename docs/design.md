# heuristic-search: design

Status: draft, written with milestone 1 on branch `draft/initial`.

## Goal and scope

`heuristic-search` is a domain-independent library of explicit-state deterministic graph
search. It knows nothing about PDDL, SAS+ or any other model: a caller describes a graph
through small capability traits and gets back a path and counters.

In scope, in milestone order (see [Milestones](#milestones)):

- foundational: BFS, DFS, Dijkstra, greedy best-first (GBFS), A*, IDA*;
- weighted and anytime: Weighted A* (WA*), ARA*, ANA*;
- bounded-suboptimal: EES, DPS;
- multiple heuristics: MHA*;
- bidirectional: NBS, BAE*, DIBBS (plus bidirectional BFS and Dijkstra as baselines);
- expensive edge evaluation: LazySP;
- dynamic graphs: LPA*, D* Lite, AD*.

Out of scope: symbolic (BDD) search, multi-objective search, width/novelty search,
stochastic search, factored (decoupled) search. Those have their own crates in the
ecosystem (see `plan/architecture.md`).

Constraints:

- `no_std` + `alloc`. Time limits come from the caller (see [Budgets](#budgets)).
- No dependency on `sas-plus` or other ecosystem crates.
- `forbid(unsafe_code)`, pedantic clippy, docs on every item.

## Capability traits

There is no universal `SearchProblem` trait. Each algorithm names the capabilities it
needs, and the goal test and heuristic are separate values rather than methods on the
graph. One graph can then be searched towards many goals with many heuristics, and closures
work for quick uses.

### Implemented

```rust
pub trait Successors {
    type State;          // Clone + Hash + Eq where a search needs it
    type Action;         // edge label, returned in solutions
    type Cost: Cost;     // non-negative
    fn successors(&self, state: &Self::State,
                  emit: impl FnMut(Self::Action, Self::State, Self::Cost));
}

pub trait Goal<S> { fn is_goal(&self, state: &S) -> bool; }            // + impl for Fn(&S) -> bool
pub trait Heuristic<S, C> { fn estimate(&self, state: &S) -> Option<C>; } // + impl for Fn(&S) -> Option<C>
pub struct Zero; // blind heuristic
```

- `Successors` keeps the callback form of the 0.0.0 placeholder: an implementation needs no
  allocation and no iterator type. Searches copy the emitted edges into a reused buffer.
- `Heuristic::estimate` returns `None` for a proven dead end (miniplan's `HValue::INFINITY`).
  Searches keep a dead-end node for duplicate detection but never open it.
- `Goal` is a predicate, not a goal state, because planning goals are partial states.

### Cost

```rust
pub trait Cost: Copy + Add<Output = Self> {
    const ZERO: Self;
    fn cmp_cost(&self, other: &Self) -> Ordering; // total order
    fn times(self, factor: u32) -> Self;          // for weights
}
```

Implemented for `u32`, `u64`, `usize`, `i64` and `f64`. Floats compare with `total_cmp`,
so a NaN cost does not panic the heap. Weights are rational (`Weight::new(num, den)`), and
Weighted A* orders by `den * g + num * h`, which sorts like `g + w * h` without division, so
integer costs stay exact. `times` saturates for integers.

### Planned

- `Predecessors` for bidirectional search. miniplan's backward search does not walk over
  states but over *subgoals* (partial states), and the two frontiers meet when a forward
  state satisfies a backward subgoal, not when two states are equal. So the trait gets its
  own node type:

  ```rust
  pub trait Predecessors {
      type Node;    // backward node; may differ from Successors::State
      type Action;
      type Cost: Cost;
      fn predecessors(&self, node: &Self::Node,
                      emit: impl FnMut(Self::Action, Self::Node, Self::Cost));
  }
  ```

  plus a meet capability that indexes one side and answers "which open/closed nodes of the
  other side does this node meet?". Equality on `State == Node` is the default
  implementation; miniplan supplies its per-fact bucket index (`search/meet.rs`).
  Backward heuristics are plain `Heuristic<Node, C>` values.
- `LazySP` gets an `EdgeEvaluator` trait (cheap lazy weight, expensive true weight,
  per-edge identity) on top of an explicit graph trait with stable edge ids.
- D* Lite / LPA* get a `DynamicGraph` trait: predecessors and successors with stable vertex
  ids plus a change feed of edge-cost updates. These do not reuse `Successors`.
- Depth-first and IDA* need nothing beyond `Successors`, `Goal` and `Heuristic`.

## Results and statistics

```rust
pub struct SearchResult<S, A, C> { pub outcome: Outcome<S, A, C>, pub stats: Stats }
pub enum Outcome<S, A, C> { Found(Solution<S, A, C>), Unsolvable, LimitReached }
pub struct Solution<S, A, C> { pub states: Vec<S>, pub actions: Vec<A>, pub cost: C }
pub struct Stats { pub expanded: u64, pub generated: u64, pub reopened: u64 }
```

- `Solution` holds the full state sequence and the action sequence; `states.len() ==
  actions.len() + 1`. The cost is summed along the parent links, so it is the true cost of
  the returned path even if an ancestor was improved after the goal was reached.
- `expanded` counts states whose successors were generated. The goal pop is not counted.
- `generated` counts every edge reported by `successors`, duplicates included.
- `reopened` counts closed states put back on the open list after a cheaper path showed up.
- Wall-clock time is not measured (no clock in `no_std`); callers time the call.

Later algorithms add fields as needed (heuristic evaluations, anytime solution history for
ARA*/ANA*, meet count for bidirectional search). `Stats` will become `#[non_exhaustive]`
before 0.1.0.

## Budgets

```rust
pub trait Budget { fn exhausted(&mut self, stats: &Stats) -> bool; } // + impl for FnMut(&Stats) -> bool
pub struct Limits { pub max_expansions: Option<u64> }
```

Asked before every expansion. `Limits::expansions(n)` stops after exactly `n` expansions.
A closure gives `std` callers a time limit or cancellation flag without the crate needing a
clock.

## Engine internals

### Search space

`SearchSpace` is an arena `Vec<Node { state, g, parent: Option<(NodeId, Action, Cost)> }>`
plus a `hashbrown::HashTable<NodeId>` that hashes the state each id points to. Each state is
stored once; there is no second copy as a map key. Parent links make solution extraction a
walk to the root. Node ids are dense, so per-node side data (heuristic value, closed flag)
lives in parallel `Vec`s owned by the algorithm.

### Open list

A `BinaryHeap` min-heap of `(key, g, seq, node)` entries:

- order: smallest key, then largest `g`, then first in, first out. Larger `g` first is the
  usual A* tie-break and finishes plateaus sooner; the push counter makes runs
  deterministic.
- decrease-key is lazy: an improved node is pushed again, and on pop an entry whose `g`
  differs from the node's current `g` is dropped. Each `(node, g)` pair is pushed at most
  once, so no closed set is needed to avoid double expansion.
- reopening falls out of this: Dijkstra, A* and WA* accept any strictly cheaper path,
  including to closed states, so A* is optimal with admissible but inconsistent
  heuristics. GBFS ignores rediscoveries.

The four best-first algorithms share one engine and differ only in the key (`g`, `g + h`,
`den * g + num * h`, `h`) and the reopen flag. BFS uses the same search space with a FIFO
queue and goal tests on generation.

Later needs, kept out of milestone 1 on purpose:

- bucket (Dial) queues for small integer costs, as a second open-list type behind the same
  engine;
- several keyed heaps over one node set (NBS, BAE*, EES, MHA*), like miniplan's
  `Frontier<T, N>` with per-node epochs;
- a focal list (EES, DPS) keyed by a second value among nodes within a bound.

These become a small internal `OpenList` trait once a second implementation exists.

## Dependencies

- `hashbrown` (runtime): `alloc` has no hash map. `hashbrown` is the implementation behind
  `std::collections::HashMap`, works under `no_std`, and its `HashTable` API lets the arena
  store each state once. Feature `default-hasher` (foldhash) only; no `std`, no
  `allocator-api2`. A `BTreeMap` would avoid the dependency but force `Ord` states and
  `O(log n)` lookups.
- `proptest` (dev only).

## How miniplan adopts it

What miniplan has today (`crates/miniplan/src/search/`): BFS, A*, GBFS, bidirectional BFS
(`bibfs_uc`), bidirectional Dijkstra (`bidij`), NBS, BAE*, regression over subgoals, a
multi-heap `Frontier` and meet detection with per-fact buckets. Every planner implements
`Planner::solve(&Task, &SearchLimits) -> Result<SearchOutcome, MiniplanError>`.

Adoption, step by step:

1. A forward adapter `struct Forward<'t>(&'t Task)` implementing `Successors` with
   `State = State`, `Action = OpId`, `Cost = u32` (operator costs are `u32`). Its
   `successors` is the applicable-operator loop that `Astar::successors` and
   `Gbfs::successors` contain today.
2. A heuristic adapter from `&dyn miniplan::Heuristic` to `Heuristic<State, u32>`:
   `HValue::INFINITY` maps to `None`, finite values are rounded *down* to stay admissible.
   (Or miniplan uses `Cost = f64` to keep its float heuristics unchanged; see open
   questions.)
3. `Bfs`, `Astar` and `Gbfs` keep their `Planner` impls and call `bfs`, `astar`, `gbfs`.
   `SearchLimits` becomes a closure `Budget` that checks `node_budget` and an `Instant`.
   `SearchStats::{plan_cost, plan_length, elapsed}` are filled from the `Solution` and the
   caller's clock, as `outcome::Budget` does today; `plan_from_ops` turns
   `Solution::actions` into a `Plan`.
4. With milestone 3, the bidirectional planners move: miniplan implements `Predecessors`
   over `Subgoal` (its regression) and the meet capability over its fact buckets.

Behaviour changes miniplan will see:

- A* reopens closed states (miniplan's A* never does), so with inconsistent heuristics such
  as `hadd`/`hff` it returns cheaper plans and may expand more.
- Ties break by larger `g` then FIFO instead of heap order, so expansion counts shift.
- `expanded` excludes the goal pop; `generated` counts all edges, not only opened ones.
- Paths are rebuilt from parent links instead of cloning a `Vec<OpId>` per node, which
  lowers memory per node.

### Cross-crate needs

Things other repos would need to change or provide. None of these are done here.

- miniplan: forward `Successors` adapter and heuristic adapter (above); decide the cost type
  (`u32` vs `f64`).
- miniplan: preferred operators (`Heuristic::preferred_ops`) are unused by its planners
  today. If GBFS with preferred-operator queues moves here, `heuristic-search` needs an
  optional capability like `PreferredActions` (emit preferred actions per state).
- miniplan: meet detection by per-fact buckets must be expressible through the planned meet
  capability without copying states into this crate.
- miniplan: its tests that assert exact `nodes_expanded`/`nodes_generated` will need new
  values after adoption.
- `mapf-search` (planned): low-level path search needs time-expanded states and constraint
  checks; both fit `Successors` and `Goal` as they are.
- `planning-heuristics` (planned): heuristics there should implement this crate's
  `Heuristic` trait behind an optional feature, not the other way round.
- `search-core` (maybe): if `width-search` or `multiobjective-search` want the same
  `Successors` trait, the trait moves there. Until then it lives here.

## Testing strategy

- Small hand-made graphs with known answers: cheapest versus fewest edges, start is goal,
  unreachable goals, dead ends, budgets, float costs, reopening under an inconsistent
  heuristic, GBFS misled by its heuristic.
- Grids: all optimal algorithms agree on the cost, A* expands fewer states than Dijkstra
  and never reopens with a consistent heuristic, WA* and GBFS stay within bounds.
- Property tests (proptest) on random weighted graphs with zero-cost edges and cycles:
  - A* cost equals Dijkstra cost equals the exact distance, with heuristics that are
    admissible but usually inconsistent (random fractions of the true distance);
  - WA* finds a solution iff one exists and costs at most `w` times the optimum;
  - GBFS is complete on finite graphs;
  - BFS edge count equals Dijkstra on the unit-cost graph;
  - every returned path is valid and its cost is the sum of its edges.
- Every public entry point has a doctest.
- `no_std` is checked by building for `thumbv7em-none-eabihf`.
- Later: differential tests against miniplan's planners on its PDDL fixtures, run from
  miniplan once it depends on this crate; benchmarks (criterion) on grids and random graphs
  before optimising the open list.

## Milestones

1. **Foundations** (done on `draft/initial`): capability traits, `Cost`, search space,
   open list, BFS, Dijkstra, A*, WA*, GBFS, statistics, budgets, path extraction, tests.
2. **Rest of foundational + anytime**: DFS (with depth limit), IDA*, ARA*, ANA*.
   `Stats` gets `#[non_exhaustive]`; anytime algorithms report each improved solution
   through a callback.
3. **Bidirectional**: `Predecessors`, meet capability, bidirectional BFS and Dijkstra, NBS,
   BAE*, DIBBS. miniplan's forward and bidirectional planners move here.
4. **Bounded-suboptimal and multi-heuristic**: focal lists; EES, DPS, MHA*.
5. **Expensive edges**: `EdgeEvaluator`, LazySP.
6. **Dynamic graphs**: `DynamicGraph`, LPA*, D* Lite, AD*.
7. **0.1.0**: API review, benchmarks, release.

## Open questions

- miniplan cost type: `u32` (exact, needs heuristic rounding) or `f64` (no change to
  heuristics, float ties)?
- Should searches take the start by value (as now) or allow several start states (useful
  for backward search from a set of goal states)?
- Goal test on generation for best-first search is a common speedup for GBFS. Add it as an
  option, or keep expansion-time tests everywhere except BFS?
- Should `Stats` count heuristic evaluations, which dominate run time in planning?
- Is a stepping API (resume a search after a budget stop) needed, for example for
  portfolios that interleave planners?

## References

- E. W. Dijkstra. A note on two problems in connexion with graphs. Numerische
  Mathematik 1, 1959.
- P. E. Hart, N. J. Nilsson, B. Raphael. A formal basis for the heuristic determination of
  minimum cost paths. IEEE Trans. SSC 4(2), 1968. (A*)
- I. Pohl. Heuristic search viewed as path finding in a graph. Artificial Intelligence
  1(3), 1970. (Weighted A*)
- R. E. Korf. Depth-first iterative-deepening: an optimal admissible tree search.
  Artificial Intelligence 27(1), 1985. (IDA*)
- M. Likhachev, G. Gordon, S. Thrun. ARA*: Anytime A* with provable bounds on
  sub-optimality. NIPS 2003.
- J. van den Berg, R. Shah, A. Huang, K. Goldberg. ANA*: Anytime nonparametric A*.
  AAAI 2011.
- J. T. Thayer, W. Ruml. Bounded suboptimal search: a direct approach using inadmissible
  estimates. IJCAI 2011. (EES)
- J. Gilon, A. Felner, R. Stern. Dynamic potential search. SoCS 2016. (DPS)
- S. Aine, S. Swaminathan, V. Narayanan, V. Hwang, M. Likhachev. Multi-heuristic A*.
  IJRR 35(1-3), 2016.
- J. Chen, R. C. Holte, S. Zilles, N. R. Sturtevant. Front-to-end bidirectional heuristic
  search with near-optimal node expansions. IJCAI 2017. (NBS)
- S. K. Sadhukhan. A new approach to bidirectional heuristic search using error functions.
  2013. (BAE*)
- V. Sewell, S. Jacobson. Dynamically improved bounds bidirectional search. Artificial
  Intelligence 291, 2021. (DIBBS)
- C. M. Dellin, S. S. Srinivasa. A unifying formalism for shortest path problems with
  expensive edge evaluations via lazy best-first search over paths with edge selectors.
  ICAPS 2016. (LazySP)
- S. Koenig, M. Likhachev, D. Furcy. Lifelong planning A*. Artificial Intelligence
  155(1-2), 2004.
- S. Koenig, M. Likhachev. D* Lite. AAAI 2002.
- M. Likhachev, D. Ferguson, G. Gordon, A. Stentz, S. Thrun. Anytime dynamic A*: an
  anytime, replanning algorithm. ICAPS 2005. (AD*)
