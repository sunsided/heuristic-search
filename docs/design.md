# heuristic-search: design

Status: draft on branch `draft/initial`. Every algorithm of
milestones 1 to 7 is implemented; see [Milestones](#milestones).

## Goal and scope

`heuristic-search` is a domain-independent library of explicit-state deterministic graph
search. It knows nothing about PDDL, SAS+ or any other model: a caller describes a graph
through small capability traits and gets back a path and counters.

In scope, in milestone order (see [Milestones](#milestones)):

- foundational: BFS, DFS, Dijkstra, greedy best-first (GBFS), A*, IDA*;
- linear- and bounded-memory: fringe search, RBFS, SMA*, beam search, beam-stack search;
- uniform-cost grids: Jump Point Search (JPS); any-angle Theta* and Lazy Theta*;
- weighted and anytime: Weighted A* (WA*), ARA*, ANA*;
- bounded-suboptimal: EES, DPS;
- multiple heuristics: MHA*;
- bidirectional: NBS, BAE*, DIBBS (plus bidirectional BFS and Dijkstra as baselines);
- expensive edge evaluation: LazySP;
- dynamic graphs: LPA*, D* Lite, AD*; Field D* on grids with per-cell costs.

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
work for quick uses (goal tests directly, heuristics through `heuristic_fn`).

### Implemented

```rust
pub trait Successors {
    type State;          // Clone + Hash + Eq where a search needs it
    type Action;         // edge label, returned in solutions
    type Cost: Cost;     // non-negative
    fn successors(&self, state: &Self::State,
                  emit: impl FnMut(Self::Action, Self::State, Self::Cost));
}

// From search-core, re-exported:
pub trait Goal<S> { fn is_goal(&self, state: &S) -> bool; }  // + impl for Fn(&S) -> bool

pub trait Heuristic<S, C = u32> { fn estimate(&mut self, state: &S) -> Option<C>; }
impl<S, C, H: Heuristic<S, C> + ?Sized> Heuristic<S, C> for &mut H { /* forwards */ }
impl<S, C, H: Heuristic<S, C> + ?Sized> Heuristic<S, C> for Box<H> { /* forwards */ }
pub fn heuristic_fn<S, C, F: FnMut(&S) -> Option<C>>(f: F) -> HeuristicFn<F>;
pub struct Zero; // blind heuristic
```

- `Successors` keeps the callback form of the 0.0.0 placeholder: an implementation needs no
  allocation and no iterator type. Searches copy the emitted edges into a reused buffer.
- `Heuristic::estimate` returns `None` for a proven dead end (miniplan's `HValue::INFINITY`).
  Searches keep a dead-end node for duplicate detection but never open it.
- `Goal` is a predicate, not a goal state, because planning goals are partial states.

#### Heuristic ownership and workspaces

`Heuristic::estimate` takes `&mut self`. Real planning heuristics (h_add, h_FF, landmark
counting, abstractions) keep scratch buffers that are reused on every call; `&mut self`
lets them do so without `RefCell` and without runtime borrow checks. Every search takes the
heuristic by value (`heuristic: H`) and owns it for the run. A caller that wants the
heuristic and its workspace back afterwards passes `&mut h`, which is a heuristic through
the `&mut H` blanket impl. `Box<dyn Heuristic<S, C>>` works for run-time choice.

Closures need `heuristic_fn(|s| ...)`: a blanket impl for `F: FnMut(&S) -> Option<C>`
cannot coexist with the one for `&mut H`, because `&mut F` is itself `FnMut` and the impls
overlap (E0119, checked; `Fn` instead of `FnMut` overlaps too). Of the two, the `&mut H`
impl wins: lending struct heuristics with workspaces is the reason for `&mut self`, and a
one-word wrapper on closures is the cheaper cost.

`Goal` stays `&self`, taken by reference, with the closure blanket impl. Goal tests are
pure predicates over a state and need no workspace; switching to `&mut self` would force
the same wrapper on every goal closure for no gain.

### Cost

`Cost`, `Heuristic`, `Goal`, `Budget<Stats>`, `heuristic_fn` and `Zero` come from
`search-core` and are re-exported from the crate root, so `use heuristic_search::Cost` keeps
working. The shared trait:

```rust
pub trait Cost: Copy + Ord + Add<Output = Self> {
    const ZERO: Self;
    fn saturating_add(self, other: Self) -> Self;      // for priority keys
    fn saturating_times(self, factor: u32) -> Self;    // for weights
}
```

Implemented by `search-core` for `u8`, `u16`, `u32`, `u64`, `u128`, `usize`, the fixed-point
`Decimal<K>` and, behind the `float` feature, the checked float `NonNegF64` (never NaN,
never negative). Signed and raw float costs are not costs any more: a negative edge voids
the optimality guarantees, and raw `f64` is not `Ord`.

Two operations that only some searches need stay local, in the extension trait `CostExt`:

```rust
pub trait CostExt: Cost {
    fn saturating_sub(self, other: Self) -> Self; // DIBBS/BAE* keys, ANA* slack
    fn to_f64(self) -> f64;                       // ANA* and DPS ratios, priorities only
}
```

It is implemented for every `search-core` cost type and bounds only `ana_star`, `dps`,
`dibbs`, `bae_star` and `bidirectional_dijkstra` (which shares their loop). `NonNegF64`
subtraction clamps at zero (`inf - inf` is zero). `Decimal<K>::to_f64` returns the raw unit
count; the common factor `10^K` cancels in the ratios. ANA* keys its open list by a private
`f64` wrapper ordered with `total_cmp`, since its keys can be `-inf`.

Weights are rational (`Weight::new(num, den)`), and
Weighted A* orders by `den * g + num * h`, which sorts like `g + w * h` without division, so
integer costs stay exact.

The key is computed with `saturating_times` and `saturating_add`, both saturating for integers, so
heuristic values near `u32::MAX` cannot panic. Keys that saturate tie, which only changes
the expansion order among them. Path costs `g` themselves use plain `+`: a path whose true
cost overflows the cost type is a caller error.

**`u32` is the default cost type for planning.** The crate stays generic over `Cost`, but
planning action costs are small non-negative integers (miniplan's are `u32`), and integer
costs give exact sums and ties.

**Rounding float heuristics over integer costs: round up.** When a heuristic is computed in
floating point but all path costs are integers, convert it with `ceil`, not `floor`. If
`h <= h*` and `h*` is an integer, then `ceil(h) <= h*`, so `ceil` keeps admissibility, and
it keeps consistency (`h(s) <= c(s, t) + h(t)` with integer `c(s, t) + ceil(h(t))` gives
`ceil(h(s)) <= c(s, t) + ceil(h(t))`). It is also the stronger (larger) of the two.
Rounding down is safe but throws information away.

### Added for later milestones

- `Predecessors` has its own `Node` type, as planned for miniplan's regression over
  subgoals:

  ```rust
  pub trait Predecessors {
      type Node; type Action; type Cost: Cost;
      fn predecessors(&self, node: &Self::Node,
                      emit: impl FnMut(Self::Action, Self::Node, Self::Cost));
  }
  pub trait BidirectionalGraph: Successors + Predecessors<Node = State, Action = Action,
                                                         Cost = Cost> {} // blanket impl
  ```

  The bidirectional searches need `Node = State` (the `BidirectionalGraph` bound) and detect
  meets by state equality through the forward and backward search spaces. The planned
  meet capability (miniplan's per-fact buckets, a forward state satisfying a backward
  subgoal) is **not** built yet; see [Open questions](#open-questions). Goals are explicit
  states (`goals: impl IntoIterator<Item = State>`), because the backward side starts from
  them. Callers generic over `BidirectionalGraph` must write `<G as Successors>::Cost`:
  `G::Cost` and `G::Action` are ambiguous between the two traits.
- `FocalKey<S, A, C>` for focal search: `start(state)` and `extend(parent_key, parent,
  action, cost, state)`, so a key can accumulate along the path (ECBS conflict counts in
  `mapf-search`). `focal_fn` wraps a state-only closure.
- `EdgeEvaluator<S, A, C>`: `evaluate(&mut self, from, action, to) -> Option<C>`, `None` for
  an infeasible edge; implemented for closures. Unlike the plan there is no separate
  explicit-graph trait with edge ids: the graph is an ordinary `Successors` that reports
  *lazy* costs, and an edge is identified by `(source state, action)`.
- `DynamicGraph`: stable `Vertex` (`Clone + Hash + Eq`), `successors` and `predecessors`
  with current costs, no actions. Instead of a change feed inside the trait, the caller
  mutates its graph and reports each changed edge with `planner.edge_changed(&graph, &from,
  &to)`; planners do not borrow the graph between calls. Costs must be **strictly
  positive**: with a zero-cost cycle, removing the edge that fed it leaves the cycle
  supporting its own stale distance (a property test found this), and a dependency can tie
  with the target's key. Heuristics are closures `FnMut(&V, &V) -> C` estimating the cost
  from the first vertex to the second.
- Depth-first search and IDA* need nothing beyond `Successors`, `Goal` and `Heuristic`.
  `dfs` is a graph search with a closed set; `depth_limited_dfs`, `iddfs` and `ida_star`
  are iterative tree searches that keep only the current path, skip successors already on
  it (so states need only `Eq`), and test goals on generation. `depth_limited_dfs` reports
  `LimitReached` when the depth limit cut something off and no goal was found.

## Results and statistics

```rust
pub struct SearchResult<S, A, C> { pub outcome: Outcome<S, A, C>, pub stats: Stats }
pub enum Outcome<S, A, C> { Found(Solution<S, A, C>), Unsolvable, LimitReached }
pub struct Solution<S, A, C> { pub states: Vec<S>, pub actions: Vec<A>, pub cost: C }
#[non_exhaustive]
pub struct Stats { pub expanded: u64, pub generated: u64, pub reopened: u64,
                   pub evaluations: u64 }                       // LazySP edge evaluations
pub struct AnytimeResult<S, A, C> { pub solution: Option<Solution<S, A, C>>,
                                    pub complete: bool, pub stats: Stats }
pub type DynamicResult<G> = SearchResult<Vertex, (), Cost>;   // LPA*, D* Lite, AD*
```

- Anytime searches (ARA*, ANA*) call `on_solution(&Solution, lower_bound)` for every
  improvement and return the incumbent with a `complete` flag, so a budget stop keeps the
  best solution instead of collapsing to `LimitReached`.
- Incremental planners are structs (`LpaStar`, `DStarLite`, `AdStar`) whose `compute`
  returns a path of vertices with `()` actions. The budget and stats of `compute` cover
  that call; `stats()` gives the totals. A budget stop resumes on the next call.

### Start states

Every search has a single-start function (`a_star(graph, start, ...)`) and a multi-start
one (`a_star_multi(graph, starts, ...)`, taking any `IntoIterator` of states). The
single-start form calls the multi-start one with `[start]`. Multi-start semantics:

- every start is seeded with `g = 0` and no parent, so a solution may begin at any start;
- repeated starts are stored once;
- an empty set of starts returns `Unsolvable` without expanding anything;
- a start that satisfies the goal returns a zero-cost solution right away, before any
  expansion, whatever the heuristic says.

Multi-start search is what backward search from a set of goal states needs, and what a
"nearest of several sources" query is.

- `Solution` holds the full state sequence and the action sequence; `states.len() ==
  actions.len() + 1`. The cost is summed along the parent links, so it is the true cost of
  the returned path even if an ancestor was improved after the goal was reached.
- `expanded` counts states whose successors were generated. The goal pop is not counted.
- `generated` counts every edge reported by `successors`, duplicates included.
- `reopened` counts closed states put back on the open list after a cheaper path showed up.
- Wall-clock time is not measured (no clock in `no_std`); callers time the call.

`Stats` is `#[non_exhaustive]`; later algorithms may add counters (heuristic evaluations,
meet count for bidirectional search).

Every search has a `*_multi` variant (the incremental planners as types) with these
semantics. Per family:

- best-first, anytime, bounded and MHA*: all starts seed one search space;
- `lazy_sp_multi`: each round runs `a_star_multi` from all starts;
- bidirectional: all starts seed the forward side, as the goals seed the backward side.
  The backward heuristic must then estimate the cost back to the nearest start;
- `dfs_multi`: starts are searched in order with one shared closed set;
- depth-limited, IDDFS and IDA*: each iteration probes the starts in order under one
  bound, as if they hung below a virtual root; IDA*'s first threshold is the smallest
  start `h`. Starts are deduplicated by equality (quadratic, no `Hash` needed).

The incremental planners have separate multi-start types (`LpaStarMulti`,
`DStarLiteMulti`, `AdStarMulti`): they search from a virtual super-start joined to every
start by a zero-cost edge, whose heuristic is the minimum over the starts, and strip it
from the path. Because the zero-cost edges let the super-start's key tie with a start's,
these planners process tied vertices before the target counts as settled. Their starts
are fixed: changing them would change the super-start's heuristic under queued keys.

## Budgets

```rust
pub trait Budget<T> { fn exhausted(&mut self, stats: &T) -> bool; } // search-core; + impl for FnMut(&T) -> bool
pub struct Limits { pub max_expansions: Option<u64> }
```

Asked before every expansion. `Limits::expansions(n)` stops after exactly `n` expansions.
A closure gives `std` callers a time limit or cancellation flag without the crate needing a
clock; annotate its argument (`|stats: &Stats| ...`). Searches take `impl Budget<Stats>`.

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

The lazy heap is generic over the key type (`OpenList<K, C>`, `total_cmp`-ordered `f64` keys for ANA*) and has
`peek_where`/`pop_where` with a custom liveness predicate (ARA*, ANA* and MHA* also check
an open flag). It serves A*, WA*, GBFS, Dijkstra, ARA*, ANA* and MHA*.

Searches that need exact removal or several orders over one node set use `KeyedSet<K>`:
a `BTreeSet<(K, id)>` plus each id's current key, with insert-or-replace, remove, first,
and neighbours of an entry. EES (cleanup order), DPS (order by `f`), the bidirectional
searches (one or two per side) and the incremental core (inconsistency queue) use it.
Costs are `Ord`, so they serve as keys directly.

`FocalList<C, K>` is **public**, so `mapf-search` can drive its own low-level loop. Items
are dense ids, each at most once; `push` replaces. It keeps an open order by `f` and an
exact focal sublist `f <= w * f_min` ordered by the focal key, also when `f_min` falls.
Focal membership depends on `f` alone, so items sit in `f` buckets (a `BTreeMap<C,
Vec<id>>` with swap-remove slots) and a boundary `f` moves whole buckets in or out; the
focal sublist is a 4-ary indexed heap (`IndexedHeap`) with exact removal. A push or
remove is `O(log n)` plus the items of buckets that cross the boundary; `peek_min_f`
scans the bucket of minimal `f`. Ties go by `f`,
then `TieBreak::{LargerG, SmallerG}`, then FIFO; with weight 1 it is an A* open list with
the focal key as tie-breaker. It is property-tested against a naive model.

Against the earlier version (two `BTreeSet`s and a boundary entry in the open order), on
the same machine: `FocalList push+pop 10k` 9.8 ms to 3.5 ms, `FocalList churn 10k`
(decrease-key and interleaved pops) 44 ms to 6.7 ms, `focal_search grid w=2` 1.25 ms to
0.40 ms. Still open: bucket (Dial) queues for small integer costs.

Per-algorithm notes:

- SMA* is a tree search over a slot arena (freed slots are reused) with two `KeyedSet`s:
  the open list (leaves by backed-up `F`, and expanded nodes by their smallest forgotten
  child's value; smallest first, deepest first) and the prunable leaves (largest `F`,
  shallowest, then oldest; starts are never pruned). The first expansion generates every
  successor; afterwards a node regenerates only its best forgotten child. Each node keeps
  the position and backed-up value of every forgotten child, so a regenerated child gets
  its value back: an earlier version kept only the minimum (as in the paper's
  description) and regenerated all forgotten children valued at that minimum, which lost
  their larger values and thrashed forever on a wide node (a hand test caught it). Values
  are backed up before pruning, which then leaves them unchanged. `memory` counts nodes;
  a child at depth `d` is generated only if `k + d <= memory` (`k` starts), and a non-goal
  one only if `k + d + 1 <= memory`, so the result is the cheapest path of at most
  `memory - k` edges. When that cap drops a child, a failed search reports `Cutoff`
  (memory too small) rather than `Unsolvable`. Tested against a bounded Bellman-Ford
  oracle (cheapest path within an edge count) for memory 1 to 14, plus a one-off
  100 000-case stress run on 16-node graphs, all terminating within the budget. Successors
  must come in a stable order, since forgotten children are found again by position.
- Theta* and Lazy Theta* need the `float` feature: path lengths are sums of square roots,
  which no exact cost type here represents, so costs are `NonNegF64`. `f64::sqrt` is not in
  `core` at the MSRV, so a private Newton iteration (exponent-halving guess, iterated until
  it stops falling, shared with Field D*) computes it; tests agree with `std` to 4 ulp and
  check exact squares and subnormal inputs. Line of sight (`LineOfSight`, with a
  default method) means the closed segment between cell centres touches no blocked closed
  cell square, which for a unit diagonal is the no-corner-cutting rule, so every
  8-connected move has line of sight and "Theta* <= 8-connected optimum" holds. The
  default is an integer supercover walk (a crossing through a grid corner checks both side
  cells), tested against a brute-force segment-versus-square oracle. The searches use
  dense per-cell arrays (`width * height`, allocated up front) and the lazy open list.
  Properties tested on random maps: solvable exactly when the 8-connected oracle is, every
  segment has line of sight, cost is the sum of the segments and at least the straight
  line; Theta* also at most the 8-connected optimum (with a consistent Euclidean `h` and
  no reopening, the closed set never misses a cheaper grid path). Lazy Theta* breaks that
  bound on some maps (found by the property test), so its docs do not claim it. On the
  benchmark grid: Theta* 1.8 ms, Lazy Theta* 0.71 ms.
- Field D* (`FieldDStar`, feature `float`) is D* Lite's core over a different model:
  nodes are cell corners, `CostGrid` (a `UniformGrid` plus `cell_cost`, default `1`) gives
  per-cell costs, and `rhs(s)` is the minimum over the eight pairs of consecutive
  neighbours `(s1, s2)` of the papers' closed form for the cheapest path through the edge
  `s1 s2` with `g` interpolated along it (straight along `s s1` at the cheaper bordering
  cell, across the cell to a point on `s1 s2`, or along `s s1` and then across to `s2`).
  The core got a crate-private model trait, `Searched`, implemented for every
  `DynamicGraph`; its `LOOKAHEAD` flag makes `update` call the model's `rhs` with a
  non-interning `g` lookup, and `compute` is split into `repair` and path extraction. The
  edge branch is unchanged, so interning order and LPA*/D* Lite/AD* results are too. Keys
  use `min_cost / √2` times the Euclidean distance (`min_cost` a lower bound on every cell
  cost, for the planner's lifetime): every case of the closed form exceeds each `g` it
  depends on by at least `min_cost / √2` per unit of corner distance (the tightest case is
  the unclamped crossing, `rhs - g(s1) = √(c² - f²) >= c / √2`), so keys stay ordered
  along dependencies. A changed cell updates its four corners: every pair a corner uses
  lies in the four cells around it. There is no corner rule: paths follow obstacle
  borders and pass corners where blocked cells touch diagonally.
  Paths are walked from the start: at a corner the candidates are each pair's closed-form
  choice plus the plain straight and diagonal moves; at an edge point, the moves along
  the edge and the best point (closed-form convex minimum) on each other edge of both
  bordering cells. Each step must strictly lower the interpolated value (corners then
  have falling `g`, so the walk terminates; an edge point without such a step falls back
  to the lower end of its edge), and steps are ranked by cost plus remainder, the
  remainder at an edge point being its own best step's total. Ranking by the plain
  interpolated value instead let paths run onto edges whose far side is blocked, up to
  21 % above the 8-connected optimum on random grids; the lookahead brought that to
  about 2 %. The reported cost is the path's true cost (segments times their cell's
  cost, or the cheaper cell along an edge); `interpolated_cost()` returns `g(start)`.
  Properties tested: `g(start)` equals an independent fixed-point sweep of the closed
  form and is at most Dijkstra over the 8-connected corner lattice; the path's cost equals
  its recomputation from geometry, is at least the straight line times the cheapest cost,
  and is within 1.05 of the 8-connected optimum (observed 1.023 over 20 000 grids); on
  uniform grids it is within 1.02 of Theta* run over the corners (observed 1.011; Theta*
  there may only use corners whose four cells are free, so Field D* is often shorter);
  incremental repair after random cell changes and start moves matches fresh runs in
  `g(start)`. The path's cost is not guaranteed to be at most the 8-connected optimum;
  only `g` is. On the benchmark grid (65 x 65 corners), a plan plus a repair after the
  gap moves: 9.1 ms, against 1.8 ms for 4-connected D* Lite; each expansion updates eight
  neighbours whose `rhs` reads eight `g` values from the hash index.
- Grids: `UniformGrid` (width, height, `is_passable(x, y)`), cells `(x, y)`, 8-connected
  **without corner cutting** (a diagonal move needs both orthogonal neighbours open).
  Costs are `Octile { straight, diagonal }`, the exact value `straight + diagonal * √2`:
  `Ord` compares `a + b√2` with `c + d√2` by the signs of the differences and, when they
  differ, `(a - c)²` against `2 (d - b)²` in `i128`, so there is no rounding and no
  overflow, and `Eq` agrees with `Ord` because `√2` is irrational. Scaled integers (10 and
  14) were rejected: they are approximate, so ties and optimality would be off. `Octile`
  implements `Cost` (saturating per count) but not `CostExt` (subtraction leaves the
  non-negative counts). `EightConnected(grid)` is the plain graph for every other search.
- JPS runs `a_star` on a private jump graph whose nodes are `(cell, arrival direction)`
  pairs, so it reuses the engine and a cell is opened at most once per direction (eight
  times) instead of once; a cell-keyed variant would need parent directions inside the
  engine. Pruning and forced neighbours follow the no-corner-cutting rules (straight:
  natural, both sides and both forward diagonals where legal, stop where a side cell is
  open but the one behind it blocked; diagonal: both components and the diagonal, stop
  where a straight jump stops). Jumps are loops, not recursion. Solutions are sparse (jump
  points, with one unit step per segment). Property tests compare against Dijkstra on an
  independently written 8-connected graph over random maps with up to 45 % obstacles,
  replaying every segment move by move; a 200 000-case stress run on maps up to 30 x 30
  found no difference. On the 64 x 64 benchmark grid JPS takes 33 µs against 0.97 ms for
  A* on `EightConnected`.
- Beam search keeps every generated state (kept or pruned) in a `SearchSpace` and never
  generates it again from a later layer; within a layer the cheapest path wins. It reports
  `Cutoff` instead of `Unsolvable` once the width has pruned anything.
- Beam-stack search keeps its layers as `Vec`s with parent positions into the layer
  above (stable, because a layer only changes after all deeper layers are dropped) and a
  `HashTable` over every node in memory. Three rules decide correctness and termination:
  - **Ties at the width.** A layer keeps every state tied with the `width`-th smallest
    `f`, and `high` becomes the smallest `f` *above* that. Cutting inside a tie would set
    `high` equal to `low` on a plateau, and the next interval `[high, U)` would repeat
    the same one forever.
  - **Duplicates.** A successor is dropped if a copy with no larger `g` is in memory in
    any layer up to the one being filled; a dearer copy in the same layer is replaced. So
    a state's copies along the layers get strictly cheaper, which bounds the depth on
    cyclic graphs even before the first solution (when `U` is infinite) and with zero-cost
    cycles. Dropping is safe: the cheaper copy's successors are admitted, now or by a later
    interval of its own layer's item, with `f` no larger than through the dropped copy.
    Property tests with widths 1 to 3 (and a 100 000-case stress run on 30-node graphs,
    widths up to 8 and multi-start) found no counterexample.
  - **The first layer** (the starts) is never pruned: no item could remember it.
  Layers are pruned once complete rather than whenever they overflow, which gives the same
  layer, at the price of up to `width * branching` states while a layer fills. Narrow
  beams backtrack a lot: on the 64 x 64 benchmark grid width 64 proves the optimum in
  under a millisecond, width 16 in about a minute. `on_solution` gets the solution only;
  unlike ARA*/ANA* there is no lower bound to report, since the intervals left on the
  stack do not give one cheaply.
- RBFS is iterative: a stack of frames, one per node on the current path, each with its
  bound and its children's backed-up values `F` (an `Extended` cost, whose `Infinite` lies
  above every cost; a saturated sum would tie with real costs). A frame whose best child
  exceeds its bound pops and writes that `F` into its parent's entry; a descent applies
  Korf's inheritance rule (`F(child) = max(F(parent), f(child))` when `f(parent) <
  F(parent)`). Several starts are the children of a virtual root with an infinite bound.
  A returning subtree's value strictly exceeds the value it was entered with, so the search
  terminates on finite graphs; like IDA* it is a tree search that skips states on the
  current path.
- Fringe search keeps the cache in a `SearchSpace` (best `g`, parent, heuristic value per
  node) and its *now* and *later* lists as two `Vec`s of `(node, ticket)`. *Now* is a
  stack whose head is its last element, so successors inserted "after the current node"
  are pushed in reverse. Removal is lazy: listing a node again hands it a fresh ticket,
  and entries with an old ticket are skipped, so a cheaper path moves a node from *later*
  (or deeper in *now*) to the head of *now* in `O(1)`. Goals are tested when a node is
  taken within the threshold, which with an admissible `h` never exceeds the optimum.
- IDA* with a transposition table (`ida_star_tt`) skips a state reached again in the same
  iteration from no smaller `g`: the earlier visit had at least as much threshold left, and
  any path through one of its then-ancestors has a cheaper equivalent. The table is
  direct-mapped (`capacity` slots, overwrite on collision), stamped per iteration so a new
  iteration costs nothing, and a lost entry only repeats work. Plain `ida_star` runs the
  same loop with a no-op table and needs no `Hash`.
- ARA* parks states improved after their expansion (`INCONS`) except in the last
  iteration of the schedule, which reopens, so the final bound holds with inconsistent
  heuristics. Goals are recognised on generation (also in MHA*).
- ANA* orders by `h` until the first solution, then by the largest `(G - g) / h`, prunes
  `g + h >= G` and reopens; its `G` is stored apart from the incumbent's `g`, which drops as
  soon as a cheaper path to the same goal node is generated.
- Focal search, EES and DPS reopen improved closed states and test goals on expansion, so
  `f_min` stays a lower bound and the `w` bound holds for any admissible heuristic. DPS
  groups open states by `(g, h)` and scans the groups per expansion.
- Bidirectional searches share a core: two search spaces, both heuristics per node, the
  best meet `mu`, path joining (forward links, then backward links in walking order).
  NBS follows Chen et al.'s waiting/ready pair rule (miniplan's `Nbs` alternates sides
  instead). DIBBS and BAE* both order by `2g + h - h_opposite` and stop at
  `b_F + b_B >= 2 mu`; DIBBS expands the side with the smaller `b`, BAE* the side with fewer
  open states. Bidirectional BFS runs the core on a unit-cost view and restores costs.
- Meets: by default a node meets the opposite side's node with an equal state, found
  through that side's hash index. The `_meet` searches (`bidirectional_bfs_meet`,
  `bidirectional_dijkstra_meet`, `nbs_meet`, `dibbs_meet`, `bae_star_meet`, all with a set
  of starts) take a `MeetRelation<S, N>` between forward states `S` and backward nodes `N`,
  the graph's own `Predecessors::Node` type (a `RegressionGraph`), so regression can run
  over subgoals of another type. The relation is told of every new node (dense id per
  side) and asked which opposite-side ids meet a new or improved node; it keeps its own
  index (per-fact buckets for miniplan, or `meet_fn`'s linear scan). Internally the core,
  which has one node type, runs over a `Mixed { State, Node }` enum with adapters for the
  graph, the heuristics (which implement `Heuristic` for both `S` and `N`) and the
  relation. The solution replays the backward actions forwards from the meeting state
  (first successor edge with an equal label, hence `Action: PartialEq`) to return real
  states and true costs; a step without such an edge is a `ReplayError { state, action }`
  in the `MeetResult`.
- LPA*, D* Lite and AD* share one core: LPA* is D* Lite run from the start. AD* adds
  inflation for overconsistent keys and `CLOSED`/`INCONS` (reset from lists, not scans).
  Target moves raise `km` by `numerator * h(old, new)` in scaled key units, which keeps
  plain and inflated keys lower bounds, so AD* re-keys everything only when the weight
  drops below one a queued key used. Paths are read off tight edges by BFS from the
  target, each step charged its cheapest parallel edge, with a greedy fallback for AD*.

## Naming

Every A* variant is a snake-case function ending in `_star`: `a_star`, `weighted_a_star`,
`ida_star`, `ara_star`, `ana_star`, `mha_star`, `bae_star` (`astar` and `weighted_astar`
were renamed and stay as aliases). Non-A* searches keep their usual names (`bfs`,
`dijkstra`, `gbfs`, `dfs`, `iddfs`, `nbs`, `dibbs`, `ees`, `dps`, `lazy_sp`,
`focal_search`). Suffixes compose in a fixed order: `_tt` for an algorithm variant,
`_multi` for several starts, `_meet` for a meet relation (which always takes several
starts, so there is no `_meet_multi`). Incremental planners keep state between calls and
are types: `LpaStar`, `DStarLite`, `AdStar`, `FieldDStar`. Renaming them to `*_star`-style functions was
rejected: they are not one-shot searches.

## Dependencies

- `hashbrown` (runtime): `alloc` has no hash map. `hashbrown` is the implementation behind
  `std::collections::HashMap`, works under `no_std`, and its `HashTable` API lets the arena
  store each state once. Feature `default-hasher` (foldhash) only; no `std`, no
  `allocator-api2`. A `BTreeMap` would avoid the dependency but force `Ord` states and
  `O(log n)` lookups.
- `search-core` (runtime, default features, path dependency): the shared `Cost`,
  `Heuristic`, `Goal` and `Budget` traits. Feature `float` forwards to its `float`.
- `proptest` (dev only).
- `criterion` (dev only, default features off) for `benches/search.rs`.

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
   `HValue::INFINITY` maps to `None`. miniplan's `h_add` and `h_FF` are sums of `u32`
   action costs, so their values are integers and the conversion is exact. Any other float
   heuristic is rounded *up* (`ceil`), which keeps admissibility and consistency (see
   [Cost](#cost)).
3. `Bfs`, `Astar` and `Gbfs` keep their `Planner` impls and call `bfs`, `a_star`, `gbfs`.
   `SearchLimits` becomes a closure `Budget` that checks `node_budget` and an `Instant`.
   `SearchStats::{plan_cost, plan_length, elapsed}` are filled from the `Solution` and the
   caller's clock, as `outcome::Budget` does today; `plan_from_ops` turns
   `Solution::actions` into a `Plan`.
4. With milestone 3, the bidirectional planners move: miniplan implements `Predecessors`
   over `Subgoal` (its regression) and the meet capability over its fact buckets.

Behaviour changes miniplan will see:

- A* reopens closed states (miniplan's A* never does), so with inconsistent heuristics such
  as `hadd`/`hff` it may return cheaper plans and may expand more.
- Ties break by larger `g` then FIFO instead of heap order, so expansion counts shift.
- `expanded` excludes the goal pop; `generated` counts all edges, not only opened ones.
- Paths are rebuilt from parent links instead of cloning a `Vec<OpId>` per node, which
  lowers memory per node.

### Cross-crate needs

Things other repos would need to change or provide. None of these are done here, except
the move to `search-core`.

- miniplan: forward `Successors` adapter and `u32` heuristic adapter with `ceil` rounding
  (above).
- miniplan: preferred operators (`Heuristic::preferred_ops`) exist in its trait but no
  planner calls them today. If GBFS with preferred-operator queues moves here, `heuristic-search` needs an
  optional capability like `PreferredActions` (emit preferred actions per state).
- miniplan: meet detection by per-fact buckets must be expressible through the planned meet
  capability without copying states into this crate.
- miniplan: expansion and generation counts change after adoption (see above). Its tests
  do not assert exact counts today, but benchmarks and logs comparing runs will shift.
- `mapf-search` (planned): low-level path search needs time-expanded states and constraint
  checks; both fit `Successors` and `Goal` as they are.
- `planning-heuristics` (planned): adopts the exact same signature,
  `fn estimate(&mut self, state: &S) -> Option<C>`, with `None` for proven dead ends, the
  same `&mut H` and `Box<H>` blanket impls, and the same rule of rounding float values up
  over integer costs. Its heuristics implement this crate's `Heuristic` behind an optional
  feature, not the other way round.
- `Stats` became `#[non_exhaustive]` with an `evaluations` counter. `saturating_sub` and
  `to_f64` live in the local `CostExt`, since `search-core`'s `Cost` lacks them; they could
  move there if another crate needs them.
- `mapf-search` (planned): can reuse `FocalList`, `TieBreak` and `focal_search` with a
  path-dependent `FocalKey` for ECBS-style conflict counts.
- `search-core` (adopted): `Heuristic`, `Cost`, `Goal` and `Budget` moved into
  `search-core`, as `MIGRATION.md` there describes. Breaking changes for users:
  - `Cost for i64` and `Cost for f64` are gone; use `u64`, or `NonNegF64` (feature
    `float`, edge costs wrapped with `NonNegF64::new`).
  - `Cost::cmp_cost` is `Ord::cmp`; `Cost::times` is `Cost::saturating_times`.
  - `Cost::saturating_sub` and `Cost::to_f64` moved to `CostExt`, which `ana_star`, `dps`,
    `dibbs`, `bae_star` and `bidirectional_dijkstra` now require of the cost type.
  - `Budget` is `Budget<Stats>`: write `impl Budget<Stats>` in bounds and impls.
  - `Heuristic<S, C>` defaults `C = u32`; `Zero` derives `Hash`.
- `Outcome::Cutoff` (new variant, breaking for exhaustive matches): `depth_limited_dfs`
  returns it when the depth limit cut states off and no goal was found, and keeps
  `LimitReached` for budget stops.
  `Successors` follows if `width-search` or `multiobjective-search` need it.

## Testing strategy

- Small hand-made graphs with known answers: cheapest versus fewest edges, start is goal,
  unreachable goals, dead ends, budgets, float costs, reopening under an inconsistent
  heuristic, GBFS misled by its heuristic.
- Grids: all optimal algorithms agree on the cost, A* expands fewer states than Dijkstra
  and never reopens with a consistent heuristic, WA* and GBFS stay within bounds, and
  heuristic values near `u32::MAX` do not overflow the A* or WA* key.
- Multi-start: empty start sets, repeated starts, and a goal among the starts (zero cost,
  no expansion, even against a misleading heuristic).
- Property tests (proptest) on random weighted graphs with zero-cost edges and cycles:
  - A* cost equals Dijkstra cost equals the exact distance, with heuristics that are
    admissible but usually inconsistent (random fractions of the true distance);
  - WA* finds a solution iff one exists and costs at most `w` times the optimum;
  - GBFS is complete on finite graphs;
  - BFS edge count equals Dijkstra on the unit-cost graph;
  - every returned path is valid and its cost is the sum of its edges;
  - multi-start search equals the minimum over single-start searches (Dijkstra and A* cost,
    BFS edge count).
- Every search function has a doctest.
- `no_std` is checked by building for `thumbv7em-none-eabihf`.
- The property-test oracle (Bellman-Ford in the test fixtures) is independent of the
  crate's own Dijkstra.
- Later: differential tests against miniplan's planners on its PDDL fixtures, run from
  miniplan once it depends on this crate; benchmarks (criterion) on grids and random graphs
  before optimising the open list.

## Milestones

1. **Foundations** (done on `draft/initial`): capability traits, `Cost`, search space,
   open list, BFS, Dijkstra, A*, WA*, GBFS, single- and multi-start, statistics, budgets,
   path extraction, saturating keys, tests.
2. **Rest of foundational + anytime**: done: DFS, depth-limited DFS, IDDFS, IDA*, ARA*,
   ANA* (`Stats` is `#[non_exhaustive]`, improvements reported through a callback).
3. **Bidirectional**: done, including the meet capability: `Predecessors`, bidirectional BFS
   and Dijkstra, NBS, BAE*, DIBBS. miniplan's planners have not moved yet.
4. **Bounded-suboptimal and multi-heuristic**: done: focal list, focal search, EES, DPS,
   shared MHA*.
5. **Expensive edges**: done: `EdgeEvaluator`, LazySP.
6. **Dynamic graphs**: done: `DynamicGraph`, LPA*, D* Lite, AD*; Field D* (`FieldDStar`,
   feature `float`) over grid corners.
7. **More algorithms**: fringe search, RBFS, beam search,
   beam-stack search, JPS (with `UniformGrid`, `Octile`, `EightConnected`), Theta* and
   Lazy Theta* (with `LineOfSight`, feature `float`), SMA*.
8. **0.1.0**: API review, benchmarks, release.

## Open questions

- Goal test on generation for best-first search is a common speedup for GBFS. Add it as an
  option, or keep expansion-time tests everywhere except BFS?
- Should `Stats` count heuristic evaluations, which dominate run time in planning?
- Is a stepping API (resume a search after a budget stop) needed, for example for
  portfolios that interleave planners? (The incremental planners already resume.)
- BAE*: the citation (Sadhukhan 2013) is unverified, and the implementation differs from
  DIBBS only in side selection (cardinality). Confirm against the paper.
- Deprecating the `astar`/`weighted_astar` aliases: they are plain re-exports of
  `a_star`/`weighted_a_star` for now, because `#[deprecated]` warns in every dependent
  (`mapf-search` builds with `-D warnings` and calls `astar`). Deprecate them before 0.1,
  once the dependents have moved.

## References

- E. W. Dijkstra. A note on two problems in connexion with graphs. Numerische
  Mathematik 1, 1959.
- S. Russell. Efficient memory-bounded search methods. ECAI 1992. (SMA*)
- R. Zhou, E. A. Hansen. Beam-stack search: integrating backtracking with beam search.
  ICAPS 2005.
- D. Harabor, A. Grastien. Online graph pruning for pathfinding on grid maps. AAAI 2011.
  (JPS)
- A. Nash, K. Daniel, S. Koenig, A. Felner. Theta*: any-angle path planning on grids.
  AAAI 2007.
- A. Nash, S. Koenig, C. Tovey. Lazy Theta*: any-angle path planning and path length
  analysis in 3D. AAAI 2010.
- Y. Björnsson, M. Enzenberger, R. C. Holte, J. Schaeffer. Fringe search: beating A* at
  pathfinding on game maps. IEEE CIG 2005.
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
- R. E. Korf. Linear-space best-first search. Artificial Intelligence 62(1), 1993.
  (RBFS)
- J. Pearl, J. H. Kim. Studies in semi-admissible heuristics. IEEE Trans. PAMI 4(4),
  1982. (focal search, A*<sub>ε</sub>)
- J. T. Thayer, W. Ruml. Bounded suboptimal search: a direct approach using inadmissible
  estimates. IJCAI 2011. (EES)
- J. Gilon, A. Felner, R. Stern. Dynamic potential search. SoCS 2016. (DPS)
- S. Aine, S. Swaminathan, V. Narayanan, V. Hwang, M. Likhachev. Multi-heuristic A*.
  IJRR 35(1-3), 2016.
- J. Chen, R. C. Holte, S. Zilles, N. R. Sturtevant. Front-to-end bidirectional heuristic
  search with near-optimal node expansions. IJCAI 2017. (NBS)
- S. K. Sadhukhan. Bidirectional A* with error (BAE*). 2013. Cited as in miniplan;
  exact title and venue not verified.
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
- D. Ferguson, A. Stentz. The Field D* algorithm for improved path planning and
  replanning in uniform and non-uniform cost environments. Technical report
  CMU-RI-TR-05-19, Carnegie Mellon University, 2005. (Field D*)
- D. Ferguson, A. Stentz. Using interpolation to improve path planning: the Field D*
  algorithm. Journal of Field Robotics 23(2), 2006.
