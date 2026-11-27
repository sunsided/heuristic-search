# heuristic-search: design

## Goal and scope

`heuristic-search` is a domain-independent library of explicit-state deterministic graph
search. It knows nothing about PDDL, SAS+ or any other model: a caller describes a graph
through small capability traits and gets back a path and counters.

In scope:

- foundational: BFS, DFS, Dijkstra, greedy best-first (GBFS), A*, IDA*;
- linear- and bounded-memory: fringe search, RBFS, SMA*, beam search, beam-stack search;
- uniform-cost grids: Jump Point Search (JPS); any-angle Theta* and Lazy Theta*;
- hierarchical: HPA*-style A* over a region graph, with refinement across threads;
- weighted and anytime: Weighted A* (WA*), ARA*, ANA*;
- bounded-suboptimal: EES, DPS;
- multiple heuristics: MHA*;
- bidirectional: NBS, BAE*, DIBBS, MM (plus bidirectional BFS and Dijkstra as baselines);
- expensive edge evaluation: LazySP;
- preprocessed queries on static graphs: contraction hierarchies;
- dynamic graphs: LPA*, D* Lite, AD*; Field D* on grids with per-cell costs;
- real-time search with learned tables: LRTA*, RTA*, LSS-LRTA*, RTAA*, and the
  limited-horizon Bellman update (LHB) after A*, GBFS or BWAS search phases; Real-time D*
  (RTD*) for changing graphs;
- parallel (feature `parallel`): hash-distributed A* (HDA*);
- many queries: reusable workspaces, resumable A*, flow fields (backward Dijkstra from a
  goal set) and parallel batches.

Out of scope: symbolic (BDD) search, multi-objective search, width/novelty search,
stochastic search, factored (decoupled) search. Those have their own crates in the
ecosystem (see `plan/architecture.md`).

Constraints:

- `no_std` + `alloc`. Time limits come from the caller (see [Budgets](#budgets)). The
  feature `parallel` links `std` for its threads, channels and locks; the default build
  stays `no_std`.
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

- `Successors` takes an emit callback: an implementation needs no
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
`dibbs`, `bae_star`, and `bidirectional_dijkstra` and `mm` (which share their loop). `NonNegF64`
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

### Further capabilities

- `Predecessors` has its own `Node` type, for regression over subgoals (as in
  miniplan):

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
meet count for bidirectional search). It holds counters of work done, not measures of
solution quality: the suboptimality of hierarchical A* would need a plain A* run to
compare against, so it is measured in its tests and benchmark instead (see
[Hierarchical search](#hierarchical-search)).

Every search has a `*_multi` variant (the incremental planners as types) with these
semantics. Per family:

- best-first, anytime, bounded and MHA*: all starts seed one search space;
- `lazy_sp_multi`: each round runs `a_star_multi` from all starts;
- `hierarchical_a_star_multi`: every start becomes its own abstract node, connected to the
  entrances of its region, and the abstract A* seeds all of them; a repeated start gives
  two nodes with the same edges, which changes nothing but the work;
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
lives in parallel `Vec`s owned by the algorithm (for the best-first searches, by their
`Workspace`).

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

`FocalList<C, K>` is **public**, so `mapf-search` can drive its own low-level loop and
ECBS high level. Items are dense ids, each at most once; `push` replaces. It keeps an open
order by `f` (a `BTreeSet` of open keys, so `min_f` and `peek_min_f` are `O(log n)`) and
an exact focal sublist `value <= w * f_min` ordered by the focal key, also when `f_min`
falls. The value is `f` itself for `push`; `push_with_value` sets it apart, as ECBS's high
level needs (lower bound as `f`, cost as value). Focal membership depends on the value
alone, so items sit in value buckets (a `BTreeMap<C, Vec<id>>` with swap-remove slots) and
a boundary value moves whole buckets in or out; the focal sublist is a 4-ary indexed heap
(`IndexedHeap`) with exact removal. A push or remove is `O(log n)` plus the items of
buckets that cross the boundary. Ties go by `f`,
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
- ALT (`Landmarks`) stores, per state and landmark `l`, `d(l, v)` and `d(v, l)` as
  `Option<C>` in one row-major table over dense state ids (a `hashbrown` index maps states
  to ids). The bound on `d(u, w)` is the largest of `d(l, w) - d(l, u)` and
  `d(u, l) - d(w, l)` over the landmarks, with `CostExt::saturating_sub`; a term with a
  missing distance is skipped, so nothing infinite is subtracted and unsigned costs cannot
  underflow. Each term is consistent and so is their maximum with `0`. When `l` reaches
  `u` but not `w` (or `w` reaches `l` but `u` does not), `w` is unreachable from `u` and
  the estimate is `None`, which every search treats as a dead end. The tables come from
  exhaustive Dijkstra runs over the crate's `SearchSpace` and lazy open list, forward over
  `Successors` and backward over `Predecessors` (`BidirectionalGraph`); the best-first
  engine itself keeps its search space private and stops at a goal. The graph traits
  cannot list states, so `build` takes a start state: farthest-first picks the state with
  the largest round trip `d(l, v) + d(v, l)` to the start, then to the nearest landmark
  (no round trip counts as infinite). Avoidance
  (Goldberg and Werneck) grows the shortest-path tree from the start, weighs each state
  by `d(r, v)` minus its current bound, sums the weights per subtree (zero for subtrees
  holding a landmark), and follows the heaviest child from the heaviest vertex down to a
  leaf. The papers root the tree at a random vertex; without a random source the root is
  the start, every round, and the choice falls back to farthest-first once every weight
  is zero. Active landmarks rank the landmarks once per query by the bound each gives
  between the query's start and target and keep the best `k`; a fixed subset keeps the
  heuristic consistent. Tested on random directed graphs against the Bellman-Ford oracle:
  every bound is admissible, `None` only without a path, consistent along every edge in
  both directions (also with active subsets); A* with ALT matches Dijkstra's cost, never
  reopens and expands only states within the optimal cost; NBS, DIBBS and BAE* with both
  adapters match the optimum.
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
  21 % above the 8-connected optimum on 20 000 random grids; the lookahead brought that
  to about 2 % there. The reported cost is the path's true cost (segments times their
  cell's cost, or the cheaper cell along an edge); `interpolated_cost()` returns
  `g(start)`. The walk alone can still end above the 8-connected optimum (figures below),
  so `compute` also runs `a_star` over the 8-connected corner lattice (straight moves at
  the cheaper bordering cell's cost, diagonal ones at `√2` times the crossed cell's) with
  `min_cost` times the Euclidean distance, a consistent heuristic, and returns that path
  where it is cheaper. `g` itself is not used as the heuristic: corners beyond the
  repair's frontier keep stale or no values. An infinite `g(start)` rules out a lattice
  path, since `g(start)` is at most the lattice optimum. A finite one implies a lattice
  path: every finite case of the closed form takes its value from a neighbour with a
  finite, strictly smaller value that a finite-cost lattice move reaches (straight to
  `s1`, or diagonally across `c` to `s2`), so those moves lead to the goal. A path is
  therefore returned exactly when `interpolated_cost()` is finite. The A* runs outside the caller's budget.
  Properties tested: `g(start)` equals an independent fixed-point sweep of the closed
  form and is at most Dijkstra over the 8-connected corner lattice (written separately in
  the tests); the path's cost equals its recomputation from geometry, is at least the
  straight line times the cheapest cost, and is at most the 8-connected optimum. The
  walk alone, measured over a one-off run of about a million grids, came within 1.035 of
  that optimum on grids up to 7 x 7 with costs in `[1, 5)`, 1.043 with costs in
  `[1, 100)`, 1.027 on grids up to 20 x 20; the gap grows a little with cost contrast. On
  uniform grids the path is within 1.02 of Theta* run over the corners (observed 1.011;
  Theta* there may only use corners whose four cells are free, so Field D* is often
  shorter); incremental repair after random batches of cell changes and start moves
  matches fresh runs in `g(start)`, and returns a path whenever `g(start)` is finite.
  Exhaustive tests check solvability and both bounds on every 3 x 3 grid with cells of
  cost 1, 2 or blocked (by default between three corner pairs; ignored variants: every
  corner pair, costs 1, 3, 10 or blocked, and every 4 x 4 grid of costs 1, 2 or
  blocked between opposite corners). On the benchmark grid (65 x 65 corners), a plan plus a repair after the
  gap moves: about 10 ms, against 1.8 ms for 4-connected D* Lite; each expansion updates
  eight neighbours whose `rhs` reads eight `g` values from the hash index. The lattice A*
  gets the interpolated path's cost as a bound and stops once its smallest `f` reaches it
  (no lattice path can be cheaper from then on; ties keep the interpolated path). On the
  benchmark the interpolated path wins both times, and the lattice A* expands 96 and
  1 928 corners instead of 436 and 3 267.
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
  open states. MM orders by `pr = max(f, 2g)`, expands the side with the smaller `pr`, and
  keeps each side's open states also keyed by `f` and by `g` for its stopping rule
  `mu <= max(pr_min, f_min_F, f_min_B, g_min_F + g_min_B + epsilon)`. `epsilon`, the
  smallest edge cost, is zero: `Cost` cannot know it, zero is always correct, and a caller
  parameter would break the shared signature of the bidirectional searches. MM0 is `mm` with
  `Zero` heuristics, not a function of its own. Bidirectional BFS runs the core on a
  unit-cost view and restores costs.
- Meets: by default a node meets the opposite side's node with an equal state, found
  through that side's hash index. The `_meet` searches (`bidirectional_bfs_meet`,
  `bidirectional_dijkstra_meet`, `nbs_meet`, `dibbs_meet`, `bae_star_meet`, `mm_meet`, all with a set
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
- Contraction hierarchies (`ContractionHierarchy`, module `contraction`) are a
  preprocessed value, not a search function: `build(graph, states)` copies the graph,
  `query(start, goal)` and `query_multi(starts, goals)` answer from the copy alone. Costs
  are static; a changed edge cost needs a new build.
  - **States** are enumerated from the given seeds by exploring `Successors`, so only
    `Successors` and `State: Clone + Hash + Eq` are needed: passing every state covers
    the whole graph, and one seed covers a strongly connected one. No `Predecessors`
    bound: the build keeps both edge directions itself. States get dense indices in
    discovery order (a hash index like `SearchSpace`'s). A query with a state outside the
    hierarchy is `Unsolvable`, unless a start is a goal.
  - **Edges** live in one arena: an original edge keeps its action; a shortcut `u -> w`
    keeps the ids of the two edges `u -> v` and `v -> w` it replaces, which name the
    middle state `v` and also the right action among parallel edges (a map from `(u, w)`
    to `v` could not tell those apart). Self loops are dropped; of parallel edges the
    overlay keeps the cheapest, and a cheaper shortcut replaces an overlay edge in place
    (overlay edges are not referred to by anything yet).
  - **Order**: importance is edge difference (shortcuts minus removed edges) plus
    contracted neighbours plus search space depth (one more than the deepest contracted
    neighbour), recomputed when a state reaches the front of the queue and pushed back if
    it grew past the next one (lazy updates). Ties go to the smaller index, so builds are
    deterministic; all adjacency is in `Vec`s, never hash-map iteration order.
  - **Witness searches** are Dijkstra from `u` on the remaining states without `v`,
    bounded by the dearest `u -> v -> w`, by `WitnessLimits` (default 16 hops and 64
    settled states) and by a hop count per label. Any path found at most as cheap is a
    witness, settled or not; a search cut off by a limit adds the shortcut, which costs
    space but never correctness. With limits of zero every bypassed pair gets a
    shortcut; the property tests run those too.
  - **Query**: bidirectional Dijkstra relaxing only edges towards higher ranks, labels in
    per-query hash maps so a query costs what it touches, a meeting checked on every
    label improvement (roots included), and each side stops when its minimum reaches the
    best meeting. The packed path is unpacked iteratively with a stack.
  - `rank`, `forward_search_space` and `backward_search_space` expose the order and the
    full upward search spaces (in settling order) for visualisation.
- LPA*, D* Lite and AD* share one core: LPA* is D* Lite run from the start. AD* adds
  inflation for overconsistent keys and `CLOSED`/`INCONS` (reset from lists, not scans).
  Target moves raise `km` by `numerator * h(old, new)` in scaled key units, which keeps
  plain and inflated keys lower bounds, so AD* re-keys everything only when the weight
  drops below one a queued key used. Paths are read off tight edges by BFS from the
  target, each step charged its cheapest parallel edge, with a greedy fallback for AD*.

## Parallel search

Module `hda`, feature `parallel`. `hda_star` (and `hda_star_multi`) is hash-distributed
A* (Kishimoto, Fukunaga and Botea): one A* search over `threads` scoped threads
(`std::thread::scope`), with only `std::sync` for coordination.

- Ownership: a state belongs to thread `hash(state) % threads`, hashed by its own
  `DefaultHashBuilder` instance, separate from the per-thread tables (whose bucket bits
  would otherwise all share the same residue). Each thread keeps an arena of the states it
  owns, a `HashTable` index, a lazy `OpenList` by `f = g + h`, a heuristic value and
  closed flag per node, and its `Stats`.
- Messages: expanding a node sends each successor `(state, g, parent link)` over the
  owner's `mpsc` channel; successors a thread owns itself are recorded at once. The
  receiver keeps the cheaper path, reopening closed nodes. A parent link is `(owner
  thread, node id in that thread's arena, action, edge cost)`, so the path is walked
  across the arenas after the threads are joined, summing the cost along the links.
- Goals are tested on pop. A popped goal updates the incumbent (a `Mutex<Option<..>>` of
  cost and link) if cheaper and is not expanded; no thread expands a node with `f >= U`.
  Among equally cheap paths, the one returned depends on timing.
- Termination: one `AtomicUsize` counts threads that are not idle plus messages sent and
  not yet recorded. A sender counts a message before sending it; a receiver uncounts it
  after recording it; an idle thread woken by a message counts itself back in first. A
  thread goes idle when its inbox is empty and its open list has no `f < U`, and
  uncounts itself. Only that can bring the counter to zero, and zero is absorbing, so
  the thread that reaches it knows no work remains anywhere and sends `Stop` to every
  channel. Idle threads block in `recv`; no sleeps or timeouts. With an admissible `h`
  every cheaper solution would leave a node with `f < U` open or in flight, so `U` is
  optimal.
- Budget: the budget and the summed counters sit in a `Mutex`. A thread adds its
  unreported counters and asks the budget before its first expansion and then every 64
  of its expansions, so `Limits::expansions(n)` overshoots by at most about 64 per thread.
  The returned `Stats` are the exact sums over the threads, and include re-expansions of
  states reached first by a costlier path, which a single thread would not make.
- The heuristic is cloned per thread (`H: Clone + Send`), since `estimate` takes `&mut
  self`; the graph and goal are shared by reference (`Sync`), the budget moves into the
  mutex (`Send`). A panic on one thread stops the others through a drop guard, and the
  scope then propagates it.

## Many queries

Games and simulations ask for paths for many agents per frame. Four tools share work
between queries; none changes the searches themselves.

### Workspaces

```rust
pub struct Workspace<S, A, C> { /* space, open list, h, closed, successor buffer */ }
pub fn a_star_in<G, T, H>(workspace: &mut Workspace<G::State, G::Action, G::Cost>,
                          graph: &G, start: G::State, goal: &T, heuristic: H,
                          budget: impl Budget<Stats>) -> GraphResult<G>;
```

The best-first engine runs inside a `Workspace` that holds everything it allocates: the
search space (arena and hash index), the open list, the per-node `h` and closed flags and
the successor buffer. The one-shot searches create a fresh workspace and call the `_in`
variant, so there is one code path and the results are identical by construction (a
property test also compares whole results, counters included, across a reused
workspace). A search clears the workspace when it starts, not when it ends, because a
start-is-goal return or a budget stop leaves it in use. The heap's push counter is not
reset: tie-breaking only compares counters within one search. A workspace keeps the
capacity of the largest search it served; drop it to free that memory. All eight
best-first functions have an `_in` variant (`a_star_in`, `dijkstra_in`,
`weighted_a_star_in`, `gbfs_in` and their `_multi_in` forms). The other families keep their
own state types and have none.

### Resumable A*

`AStarSearch` owns a workspace and its heuristic and is seeded at construction. Each
`compute(graph, goal, budget)` continues the same run and returns `LimitReached` while it
is paused. As for the incremental planners, the graph and goal are passed per call
(storing `&goal` would not work with a temporary `&|s| ...`), and the budget and the
returned stats cover that call, so `Limits::expansions(n)` means `n` expansions per frame;
`stats()` gives the totals. To make pausing lossless the engine loop peeks the next live
entry, tests it for the goal, asks the budget, and pops only to expand; re-pushing a popped
entry would change its tie-break counter. The one-shot searches run the same loop, so
the expansion order, outcome and total counters of a resumed search equal one `a_star`
call. A finished search returns its outcome again on every further call without
expanding: a found goal stays at the top of the open list, and a start that is a goal is
remembered. This needs nothing from `search-core`: `Budget<Stats>` is asked as before.

### Flow fields

```rust
impl<S, A, C> FlowField<S, A, C> {
    pub fn build<G: Predecessors<Node = S, Action = A, Cost = C>>(
        graph: &G, goals: impl IntoIterator<Item = S>, budget: impl Budget<Stats>) -> Self;
    pub fn cost(&self, state: &S) -> Option<C>;
    pub fn next(&self, state: &S) -> Option<(&A, &S, C)>;
    pub fn path(&self, start: &S) -> Option<Solution<S, A, C>>;
}
```

A flow field is `dijkstra_multi_in` over the graph's predecessors (an adapter that turns
`Predecessors` into `Successors`) from every goal, with a goal test that never holds, so
it runs until the open list is empty or the budget stops it. The finished workspace is
the field: a state's `g` is its cost to the nearest goal, and its parent link is a forward
edge (`Predecessors` reports the forward action and cost), so `next` is one hop and
`path` is the existing walk to the root. Only expanded states are answered: with
non-negative costs their `g` is final, and so is every ancestor's; states generated but
not expanded before a budget stop have tentative costs and are reported as unknown.
`is_complete` tells "no goal reachable" apart from "not settled yet". The name avoids
`FieldDStar` and `FieldResult`, which are Field D*.

`EightConnected` implements `Predecessors` so fields run on grids: the corner rule needs
the same cells for a move and its reverse, so the predecessors are the successors with
the step negated.

### Batches

`solve_batch(queries, threads, init, solve)` (feature `parallel`, `std`) runs scoped
threads (`std::thread::scope`). Each thread builds its own state with `init`, typically a
`Workspace`, and draws query indices from an atomic counter, so uneven queries balance
across threads; answers are written back by index, so they come back in input order.
`threads` is clamped to `1..=queries.len()`, and one thread runs on the caller. A panic in
a worker is resumed on the caller after the scope has joined the others. No pool and no
work stealing: no dependency beyond `std`.

### Send and Sync

Nothing in the crate uses `Rc`, `Cell` or raw pointers, so every graph adapter, result,
learned table, planner, workspace and closure-built heuristic, goal, meet relation and
focal key is `Send + Sync` exactly when its type parameters are. `tests/it/send_sync.rs`
asserts this at compile time, generically over the parameters.

## Hierarchical search

Module `hierarchical`, after Botea, Müller and Schaeffer's HPA* (2004) and the two-level
room-and-door scheme of FloorPlan2Nav (ISARC 2026). A `Hierarchy<S, C, R>` is built once
and answers many queries:

```rust
pub trait Regions<S> { fn region(&self, state: &S) -> usize; }   // + impl for Fn(&S) -> usize
impl Hierarchy<S, C, R> {
    pub fn new(graph, regions: R, states, distance) -> Self;    // every border state an entrance
}
impl Hierarchy<GridCell, Octile, GridClusters> {
    pub fn grid(grid: &impl UniformGrid, size: usize) -> Self;  // HPA* clusters and transitions
}
pub fn hierarchical_a_star(graph, &hierarchy, start, goal: S, distance, budget) -> GraphResult<G>;
pub fn hierarchical_a_star_par(...)                              // feature `parallel`
```

- **Abstract graph.** Nodes are entrances: states with an edge into another region, and
  the targets of those edges. Edges are the graph edges between entrances of different
  regions, and between every two entrances of one region the cost of A* restricted to the
  region (a `Successors` wrapper that drops edges leaving it). Only costs are stored; a
  query recomputes the paths it uses.
- **Grid builder.** Square clusters of `size` cells. Along each border, every maximal run
  of cell pairs passable on both sides becomes one transition in its middle, or two at its
  ends when the run has six or more pairs, as in the paper. Without corner cutting, a
  diagonal move across a border has a passable straight pair beside it, and a run's cells
  are straight-connected on both sides, so the transitions keep every connection between
  clusters: the query is complete (it finds a path whenever A* does), though not optimal.
- **User regions.** `Hierarchy::new` keeps every border crossing, so a cheapest path of
  the graph decomposes into abstract edges and the query is optimal. Building runs one
  restricted A* per pair of entrances of a region, quadratic in the entrances, so it suits
  regions that meet through doors rather than open borders.
- **Query.** The start and goal are inserted temporarily: restricted A* from the start to
  each entrance of its region, from each entrance of the goal's region to the goal, and
  from the start to the goal when they share a region. A* on the abstract graph (node
  ids, heuristic `distance(node, goal)`) picks the entrances; every abstract edge is then
  refined, by restricted A* inside its region or as the cheapest graph edge between two
  regions. The solution concatenates the segments; its cost is their sum.
- **Goal and heuristic.** The goal is a state, not a `Goal` predicate, since it becomes an
  abstract node. Searches towards different targets need different heuristics, so the
  caller passes a pairwise estimate `distance(a, b)` instead of a `Heuristic`.
- **Budget and stats.** `Stats` add up the connecting, abstract and refining searches; the
  build is not counted. The budget sees the running totals while connecting and on the
  abstract graph. Refinement is not budgeted: each search stays in one region and is known
  to succeed, and an unbudgeted refinement keeps the parallel and sequential results equal.
- **Parallel refinement.** The refinements are independent. `hierarchical_a_star_par`
  splits them into one contiguous share per available core under `std::thread::scope` and
  joins the shares in order, so its result, stats included, equals the sequential one. It
  is a separate function rather than a feature-switched body because it needs `Send` and
  `Sync` bounds; enabling `parallel` anywhere in a dependency graph must not add bounds to
  `hierarchical_a_star`.
- **Smoothing.** HPA*'s post-processing pass that shortcuts the path near cluster borders
  is not implemented: it needs line-of-sight or repeated local searches over the whole
  path, which costs about as much as the refinement it would improve.
- **Quality.** On generated floor plans (square rooms, one-cell walls, doors of one or two
  cells, some walls without a door) the cluster paths cost on average about 2 % more than
  the cheapest path; the test asserts at most 10 %. `benches/search.rs` prints the cost
  ratio next to the timings. FloorPlan2Nav evaluates on ResPlan floor plans; those are not
  part of this repository (their licence is not verified), so tests and benchmarks use
  generated plans.

## Real-time search

Module `real_time`. An agent plans a bounded amount, moves, and learns: every function runs
one trial from `start` until the agent stands on a goal (`Outcome::Found` with the
travelled path, loops included), proves that no goal is reachable from where it stands
(`Unsolvable`), or the budget stops it (`LimitReached`, asked before every expansion).
Learned values live in a `LearnedHeuristic<S, C, H>`, a hash table in front of a base
heuristic that the caller keeps across trials. It implements `Heuristic`, so a learned
table can guide `a_star` afterwards. `None` means a proven dead end there as everywhere.
`TrialResult` adds `TrialStats` (moves, planning steps, changed values, and the most
expansions and changed values of any single step) to the usual `Stats`; `updates == 0`
means the trial learned nothing, which is how callers detect convergence.

| Function | Lookahead | Update of the expanded states | Keeps |
| --- | --- | --- | --- |
| `lrta_star` | the successors | `h(s) = min c(s, s') + h(s')` (Korf's min, no max with the old value) | admissible, consistent |
| `rta_star` | the successors | second-best `c + h`; the best with fewer than two live successors | nothing (inadmissible by design) |
| `lss_lrta_star` | A*, `n` expansions | Dijkstra from OPEN into CLOSED, closed values start at infinity | admissible, consistent |
| `lhb_lrta_star` | A*, `n` expansions | the paper's Eq. 5 by its Algorithm 1 | admissible, consistent |
| `lhb_gbfs` | GBFS, `n` expansions | Eq. 5 | admissible, consistent |
| `lhb_bwa_star` | BWAS (weight `w`, batch `B`), `n` expansions | Eq. 5 | admissible, consistent |
| `rtaa_star` | A*, `n` expansions | `h(s) = f(best open) - g(s)` | admissible, consistent |
| `lhb_update` | A*, `n` expansions, no move | Eq. 5, offline | admissible (consistent through the agent; tested there) |

Engine: the lookahead agents share one bounded best-first search (`Lookahead`), A* unless
a search phase says otherwise, that reopens on cheaper paths and records *every* generated
edge, not only parent links: a shortest-path
update over parent links alone overestimates on cycles. A goal that A* selects ends the
lookahead and stays a seed of the update (peeked, not popped), otherwise closed states
whose only exit is the goal would be marked dead. The agent then moves along the A* path
to the selected goal or to the open state with the smallest `f`, as Koenig and Sun's
LSS-LRTA* does, and plans again. A lookahead of zero panics: the agent could never move.
Work per step is at most `n` expansions plus one Dijkstra over at most `n` expanded states
and their edges; the tests check `max_step_expansions <= n` and `max_step_updates <= n`
(`n + 1` for LHB, whose goal clamp can change one more value).

Completeness and convergence need what the classic results need: a finite graph from
which the goal stays reachable (safely explorable) and positive edge costs. On infinite
graphs with an uninformed table the agent can wander away forever; the budget stops it.

### Paper comparison: limited-horizon Bellman update (Hadar, Agostinelli, Shperberg 2025)

| Paper (arXiv 2511.10264v1) | Here | Status |
| --- | --- | --- |
| LHBL search phase: run any search (A*, GBFS) from a sampled state for `N` expansions, guided by the target network `h_θ⁻` | `lhb_update` runs A* with the table itself as the heuristic (the tabular case has no separate target network); the agents `lhb_lrta_star`, `lhb_gbfs` and `lhb_bwa_star` run A*, GBFS and BWAS | same; GBFS and BWAS only inside the agent |
| BWAS (background section, from DeepCubeA): order `λ g + h` with `λ ∈ [0, 1]`, expand the `B` lowest-cost nodes in parallel; `B = 1`, `λ = 1` is A*, `λ = 0` GBFS | `lhb_bwa_star` orders by `g + w h` with the crate's rational `Weight` (`w = 1 / λ`), takes `B` open states, then expands them; `λ = 0` is `lhb_gbfs` | same order; `λ = 0` as its own function |
| DeepCubeA's BWAS (its code) records goals when popped and stops after the iteration that popped one | a goal among the selected batch ends the step before the batch is expanded, so a goal is never interior to the search graph; the last batch is cut at `N` expansions | deliberate deviation, as for A* |
| Search graph `G(V, E)`: generated states and edges, cycles allowed | `Lookahead` records every generated edge, duplicates included | same |
| Leaves `L(n)`: descendants of `n` without children in `G` | nodes never expanded, or expanded without successors (`children == false`) | same |
| Eq. 5: `h(s) = 0` if `s` is a goal, else `min over l in L(n) of C(n, l) + h_θ⁻(l)` | interior nodes take the minimum over reachable leaves with the old leaf values; goal nodes get 0 afterwards, so a goal leaf with `h ≠ 0` still contributes its old value to its ancestors | same; goal case read literally |
| Algorithm 1: auxiliary node `z`, edge `l -> z` of cost `h(l)`, reverse, Dijkstra from `z`, assign every `v ∈ V` | multi-source Dijkstra over reversed edges seeded with the leaf values (equivalent to the `z` construction); interior nodes and goals are written, leaves keep their value (Algorithm 1 assigns them `h(l)`, the same value) | same |
| An interior node that reaches no leaf: not discussed (`min` over an empty set) | becomes a proven dead end (`None`) | our reading |
| Algorithm 1 searches for exactly `N` steps; nothing says what happens when the search expands a goal | the lookahead stops when A* selects a goal, so a goal is never interior | deliberate deviation: an interior goal under Eq. 5 takes 0 while its ancestors may route past it, and with the goal expanded admissibility could break |
| Labels train a DNN by MSE (Eq. 4); states sampled by random walks from the goal | tabular: `lhb_update` writes the label into the table; callers choose the samples | out of scope (no function approximation) |
| LHBL is offline; LSS-LRTA* is online real-time search | both: `lhb_update` offline, `lhb_lrta_star` the LHB update inside LSS-LRTA*'s agent | extension |
| "LSS-LRTA* assumes consistent h-values"; LHB needs no consistency | Koenig and Sun's Dijkstra is a multi-source shortest-path computation, correct for any seed values, and both stay admissible from an admissible table (tested with inconsistent tables). Without consistency LSS-LRTA*'s values can fall (a reopened state seeds the update with its stale value, see case 3 below), while LHB's partition by children avoids that case | partly confirmed: the difference is real only where A* reopens |
| Fig. 1: `h_LHB` exact in one update where `h_SSB` is off (edges lost in the text extraction) | own instance `limited_horizon_update_beats_single_step`: single-step gives 2, LHB the exact 3 | own example, not the figure |
| Evaluation: Rubik's Cube, 35-STP, Lights Out with DeepCubeA networks | not reproduced; tabular learning curves on mazes below | not reproduced |

**LHB against LSS-LRTA*'s update.** Both compute, for the expanded states, the cheapest
cost to the frontier plus the frontier value, by Dijkstra over the reversed explored
graph. They write different values in exactly three cases:

1. An expanded state without successors: a leaf for LHB, which keeps its value;
   LSS-LRTA* proves it a dead end. Test: `updates_differ_on_states_without_successors`.
2. A goal with a nonzero value: LHB sets it to 0 (Eq. 5); LSS-LRTA* leaves open states
   alone. Test: `updates_differ_on_goals_with_nonzero_values`.
3. A state expanded and then reopened by a cheaper path under an inconsistent table: it
   has children, so LHB treats it as interior; LSS-LRTA* treats it as open (a seed), with
   its old value. Not reachable with a consistent table, where A* never reopens. Test:
   `updates_differ_on_reopened_states`, where LSS-LRTA* lowers a value from 5 to 1 (still
   admissible) and LHB learns the exact costs.

Elsewhere they agree value for value: the property test
`lss_and_limited_horizon_agree_without_dead_ends` runs both agents trial for trial on
strongly connected graphs with consistent tables and finds identical tables, paths and
update counts. With a lookahead of one, both are LRTA*'s update.

### Learning curves

`cargo bench --bench learning` (harness off) prints these. Eight random 24×24
4-connected mazes, about 30 % walls, start top left, goal bottom right, unit moves,
Manhattan table, means over the mazes; mean optimum 46.8. A trial "converges" when it
changes no value; every converged path was checked against Dijkstra.

| lookahead | LSS-LRTA* trials | LHB trials | RTAA* trials | LSS first-trial cost | RTAA* first-trial cost | LSS expansions | RTAA* expansions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LRTA* | 8.2 | | | 90 | | 744 | |
| 1 | 8.2 | 8.2 | 8.2 | 90 | 90 | 744 | 744 |
| 2 | 12.2 | 12.2 | 14.5 | 109 | 121 | 901 | 1121 |
| 4 | 10.2 | 10.2 | 12.9 | 95 | 109 | 749 | 1060 |
| 8 | 8.0 | 8.0 | 9.1 | 80 | 74 | 582 | 815 |
| 16 | 5.4 | 5.4 | 6.1 | 61 | 62 | 412 | 564 |
| 32 | 4.0 | 4.0 | 4.5 | 54 | 57 | 318 | 448 |
| 64 | 3.1 | 3.1 | 2.4 | 52 | 54 | 249 | 292 |
| 128 | 2.4 | 2.4 | 2.4 | 47 | 47 | 210 | 353 |

- LSS-LRTA* and LHB are identical on these mazes, as predicted: every cell has a
  successor, the goal's value is 0, and the Manhattan table is consistent.
- Larger lookaheads converge in fewer trials, with cheaper first trials and fewer total
  expansions. Lookaheads 2 and 4 are worse than 1 on these mazes; our unmeasured guess is
  that the agent commits to longer moves on little information.
- RTAA* needs a few more trials and expansions than LSS-LRTA* at most lookaheads; its
  values are lower or equal, but its update needs no second search.

Offline, as in the paper: sweeps of `lhb_update` over every cell that reaches the goal,
from a zero table, until nothing changes (the result was checked to be exact against
Dijkstra). Horizon 1 is the single-step Bellman update. The expansion column is the bound
`sweeps × cells × horizon`; lookaheads that select the goal stop early.

| horizon | sweeps | expansion bound |
| --- | --- | --- |
| 1 | 47.4 | 18816 |
| 2 | 25.4 | 20168 |
| 4 | 14.1 | 22458 |
| 10 | 6.8 | 26846 |
| 50 | 3.4 | 67200 |
| 100 | 3.0 | 119138 |

The sweep count falls roughly as `1 / horizon` down to a floor of a few sweeps, while the
work per sweep grows with the horizon, so the expansion bound is smallest at horizon 1.
Fewer sweeps for longer horizons is the tabular analogue of the paper's finding that LHB
labels need fewer sampled states than single-step labels; per expansion, the single-step
update is not worse here.

### Real-time D*

`RealTimeDStar` (Bond, Widger, Ruml and Sun 2010) is D* Lite made real-time by a per-move
expansion limit, with LSS-LRTA* taking the moves until D* Lite's search from the goal
reaches the agent. It is a type, like the incremental planners: the caller calls
`step(graph, expansions)` once per move and reports changed edges with `edge_changed` in
between, as with `DStarLite`. `into_trial` returns the run as a `TrialResult` (travelled
path, `Stats` of both searches, `TrialStats`), and `step` returns an `AgentStep`
(`Arrived`, `Informed`, `Partial`, `Waiting`).

| Paper (SoCS 2010, Fig. 1) | Here |
| --- | --- |
| `LocalLimit = LocalRatio × ExpandLimit`, `GlobalLimit = ExpandLimit − LocalLimit` | `local_share` (a `Weight` at most `1`) given at construction, `expansions` per `step`; the local share is rounded down and raised to at least one expansion, so `max_step_expansions <= expansions` |
| `ComputeShortestPath` stops at `GlobalLimit` pops (re-keys count) | `Incremental::repair` under `Limits::expansions(global)`; re-keys count as expansions there too |
| `ChooseStep`, complete path: the successor minimising `c(s, s') + g(s')` | the same, first on ties |
| `ChooseStep`, limit reached: the local search's step (LRTA* or LSS-LRTA*) | LSS-LRTA*: A* lookahead of the local share and the Dijkstra update, then the first edge of the A* path to the frontier state LSS-LRTA* would travel to; its learned table lives in the planner |
| `ChooseStep`, no path: stay, the world may change | `AgentStep::Waiting`; also when the lookahead proves no goal reachable |
| `km += h(s_last, s_start)` after each move, `ComputeShortestPath` called every iteration | `move_target` after each move; `repair` every step (the paper's footnote notes its own implementation only updated `km` on edge changes, a bug) |
| AD* as the global search | not offered; the global search is D* Lite |

The local search's heuristic is the planner's `heuristic(v, goal)`, in front of the
learned values. Both share one closure: the table keeps only learned values and gets the
heuristic as its base for the duration of each lookahead (`LearnedHeuristic::rebase`),
so `RealTimeDStar` needs no `Clone` or shared ownership of the heuristic. Learned values
can overestimate after costs fall (the paper's point that LSS-LRTA* does not handle cost
decreases); fully informed steps ignore them. Learned dead ends are dropped after any
reported edge change: an appearing edge can connect them to the goal, and with a local
share of `1` (no global search) a stale dead end would keep the agent waiting forever.
This is not in the paper, whose local search is not described as learning dead ends.

Completeness (the paper's Theorem 2): in a static finite graph from which the goal is
reachable, if the global share exceeds the re-key expansions the start's moves cause,
D* Lite eventually reaches the agent. From then on every step is fully informed: with a
consistent heuristic the moved-to vertex is settled at once, since its key is at most the
old start's. The tests find arrival for every limit from 1 and every share on random
graphs, with zero and with exact-distance heuristics; that is tested, not proven, for
the small shares. With a local share of `1` the global search gets nothing and the agent
is LSS-LRTA* moving one edge per lookahead. In changing graphs there is no guarantee (the
paper's example of doors that close as the agent approaches).

## Naming

Every A* variant is a snake-case function ending in `_star`: `a_star`, `weighted_a_star`,
`ida_star`, `ara_star`, `ana_star`, `mha_star`, `bae_star`. Non-A* searches keep their usual names (`bfs`,
`dijkstra`, `gbfs`, `dfs`, `iddfs`, `nbs`, `dibbs`, `mm`, `ees`, `dps`, `lazy_sp`,
`focal_search`). Suffixes compose in a fixed order: `_tt` for an algorithm variant,
`_par` for work split across threads (feature `parallel`), `_multi` for several starts, `_meet` for a meet relation (which always takes several
starts, so there is no `_meet_multi`), `_in` for a search in a caller's `Workspace`
(`a_star_multi_in`). Incremental planners keep state between calls and
are types: `LpaStar`, `DStarLite`, `AdStar`, `FieldDStar`, `RealTimeDStar`. Renaming them to `*_star`-style functions was
rejected: they are not one-shot searches.

## Dependencies

- `hashbrown` (runtime): `alloc` has no hash map. `hashbrown` is the implementation behind
  `std::collections::HashMap`, works under `no_std`, and its `HashTable` API lets the arena
  store each state once. Feature `default-hasher` (foldhash) only; no `std`, no
  `allocator-api2`. A `BTreeMap` would avoid the dependency but force `Ord` states and
  `O(log n)` lookups.
- `search-core` (runtime, default features, from crates.io): the shared `Cost`,
  `Heuristic`, `Goal` and `Budget` traits. Feature `float` forwards to its `float`.
- `proptest` (dev only).
- `criterion` (dev only, default features off) for `benches/search.rs`.

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
- Real-time search (`tests/it/real_time.rs`): admissibility after every trial from
  admissible, usually inconsistent tables (with zero costs, dead ends and unreachable
  goals), consistency after every trial from consistent tables, completeness and work per
  step on random strongly connected graphs with positive costs (RTA* included, from
  arbitrary tables), convergence to the Dijkstra cost from consistent and from admissible
  inconsistent tables (the latter beyond the classic theorems for the lookahead searches;
  clean over 5000 random cases per property in one run), LSS and LHB agreement where their updates cannot differ, and offline
  LHB sweeps reaching exact costs. The GBFS and BWAS phases join the admissibility,
  consistency and completeness properties; converged trials are checked against the
  optimum only for BWAS with a batch of one (within `w`), and BWAS with weight one and
  a batch of one must equal `lhb_lrta_star` trial for trial.
- Many queries (`tests/it/workspace.rs`, `flow.rs`, `batch.rs`, `send_sync.rs`): one
  workspace reused across random queries, budget stops and starts that are goals returns
  whole results equal to fresh searches; resumable A* in calls of 1 to 3 expansions
  equals one `a_star` call in outcome and total counters; flow field costs equal the
  Bellman-Ford distance to the nearest goal, `next` is a forward edge on a cheapest path,
  and a budget-cut field answers exactly its expanded states, which are the closest ones;
  `EightConnected` predecessors equal its reversed successors, and grid fields agree with
  A*; batches equal the sequential searches at 0 to 5 threads, in order; panics reach the
  caller; Send and Sync are asserted at compile time.
- Contraction hierarchies: every pair on random graphs of up to 25 nodes (zero costs,
  self loops, parallel edges, unreachable pairs) costs what `dijkstra` costs, with
  witness limits of zero, tiny and default; every path is replayed on the graph. The
  upward search spaces of start and goal meet at the query cost, ranks are a
  permutation, multi-queries cost the cheapest pair, and sampled queries on the
  benchmark grid match `dijkstra` and replay move by move.
- Real-time D* (`tests/it/dynamic.rs`): on random strongly connected graphs it reaches the
  goal for every expansion limit and local share (including `0` and `1`), along real
  edges, one per step, within the limit per step; with unlimited expansions every step is
  fully informed and the travelled cost equals Dijkstra's (or the agent waits where no
  path exists); with random cost changes, additions and removals between steps it still
  arrives; with exact distances as the heuristic (so start moves raise `km`) it arrives
  and, unlimited, travels Dijkstra's cost; it leaves a pocket it had learned as a dead end
  once an exit appears; and on a grid with toggled walls it waits while the goal is
  enclosed.
- Hierarchical A* (`tests/it/hierarchical.rs`): on random grids the cluster hierarchy finds
  a path exactly when the oracle does and never a cheaper one, the hierarchy with every
  border state as an entrance matches the oracle cost, multi-start equals the best single
  start, every path replays edge by edge on the grid, and with feature `parallel` the
  threaded refinement returns the same result as the sequential one.
- Every search function has a doctest.
- `no_std` is checked by building for `thumbv7em-none-eabihf`.
- The property-test oracle (Bellman-Ford in the test fixtures) is independent of the
  crate's own Dijkstra.

## Open questions

- Goal test on generation for best-first search is a common speedup for GBFS. Add it as an
  option, or keep expansion-time tests everywhere except BFS?
- Should `Stats` count heuristic evaluations, which dominate run time in planning?
- Resumable searches beyond A* (`AStarSearch`): Weighted A*, GBFS and Dijkstra run the same
  engine and could resume the same way. Default: A* only until a caller needs more.
- BAE*: the citation (Sadhukhan 2013) is unverified, and the implementation differs from
  DIBBS only in side selection (cardinality). Confirm against the paper.
- Real-time search: should a trial report where the agent stopped on `Unsolvable` (the
  dead end may not include `start` on graphs with irreversible edges)? Default: no, the
  outcome has no payload; the table records the dead ends.
- Real-time search: a step-level API (one plan-and-move step at a time, for agents that
  act between steps) instead of whole trials? Default: trials only; `Limits` bounds them.
- LHB: Eq. 5 and Algorithm 1 disagree for a goal leaf with a nonzero value; we use the old
  value for its ancestors and set the goal itself to 0. Algorithm 1 alone would keep the
  goal's value. Default: Eq. 5 as written.
- LHB with GBFS or BWAS: the agent moves to the open state its search would expand next.
  With GBFS a trial that learns nothing can still take a detour (test
  `gbfs_phase_follows_the_values`), and so can BWAS with a batch above one, even at
  weight one (found by the convergence property test). With a batch of one and weight
  `w` the converged trials stayed within `w` times the cheapest cost on the random test
  graphs, which is not proven.
  Moving along the cheapest path in the search graph instead would make the phases learn
  differently but move like A*. Default: each phase moves by its own order.

## References

- E. W. Dijkstra. A note on two problems in connexion with graphs. Numerische
  Mathematik 1, 1959.
- A. Treuille, S. Cooper, Z. Popović. Continuum crowds. ACM Transactions on Graphics
  25(3), 2006 (SIGGRAPH 2006). (a potential field from the goals that steers every agent)
- E. Emerson. Crowd pathfinding and steering using flow field tiles. In S. Rabin (ed.),
  Game AI Pro, chapter 23, CRC Press, 2013. (flow fields in games)
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
- R. C. Holte, A. Felner, G. Sharon, N. R. Sturtevant. Bidirectional search that is
  guaranteed to meet in the middle. AAAI 2016. (MM)
- R. Geisberger, P. Sanders, D. Schultes, D. Delling. Contraction hierarchies: faster
  and simpler hierarchical routing in road networks. WEA 2008, LNCS 5038.
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
- A. Kishimoto, A. Fukunaga, A. Botea. Scalable, parallel best-first search for optimal
  sequential planning. ICAPS 2009. (HDA*)
- A. Kishimoto, A. Fukunaga, A. Botea. Evaluation of a simple, scalable, parallel
  best-first search strategy. Artificial Intelligence 195, 2013. (HDA*)
- R. E. Korf. Real-time heuristic search. Artificial Intelligence 42(2-3), 1990. (LRTA*,
  RTA*)
- S. Koenig, X. Sun. Comparing real-time and incremental heuristic search for real-time
  situated agents. Autonomous Agents and Multi-Agent Systems 18(3), 2009. (LSS-LRTA*)
- S. Koenig, M. Likhachev. Real-time adaptive A*. AAMAS 2006. (RTAA*)
- A. V. Goldberg, C. Harrelson. Computing the shortest path: A* search meets graph
  theory. SODA 2005. (ALT, farthest landmark selection)
- A. V. Goldberg, R. F. Werneck. Computing point-to-point shortest paths from external
  memory. ALENEX 2005. (avoidance landmark selection)
- A. Botea, M. Müller, J. Schaeffer. Near optimal hierarchical path-finding. Journal of
  Game Development 1(1), 2004. (HPA*)
- FloorPlan2Nav. Proceedings of the International Symposium on Automation and Robotics in
  Construction (ISARC) 2026, part 2, PDF pages 474–480. Authors and printed page numbers
  not verified. (Room-and-door graph with local A* per segment, in parallel.)
- G. Hadar, F. Agostinelli, S. S. Shperberg. Beyond single-step updates: reinforcement
  learning of heuristics with limited-horizon search. arXiv:2511.10264, 2025. (LHB)
- D. M. Bond, N. A. Widger, W. Ruml, X. Sun. Real-time search in dynamic worlds. SoCS
  2010. (Real-time D*)
- F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi. Solving the Rubik's cube with deep
  reinforcement learning and search. Nature Machine Intelligence 1, 2019. (DeepCubeA,
  BWAS)
