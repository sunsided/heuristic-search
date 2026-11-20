//! Beam-stack search (Zhou and Hansen 2005): [`beam_stack_search`] and
//! [`beam_stack_search_multi`].
//!
//! Beam-stack search is beam search that remembers what it pruned. The *beam stack* holds
//! one item per layer, an interval `[low, high)` of `f` values: layer `d + 1` admits only
//! successors whose `f` lies in item `d`'s interval. When a layer overflows the width, the
//! worst states are pruned and `high` drops to the smallest pruned `f`. Once the search
//! runs dry, it backtracks to the deepest item that still has `f` values below the best
//! solution cost `U` left, shifts its interval to `[high, U)` and regenerates the layer
//! below from the same parents. When no item has anything left, every path cheaper than
//! `U` has been searched, so the incumbent is optimal.
//!
//! ```mermaid
//! flowchart TD
//!     expand[expand layer d in f order, admitting successors by item d] --> goal{goal below U?}
//!     goal -->|yes| better[U = g, report the solution]
//!     better --> expand
//!     goal -->|no| expand
//!     expand -->|layer done| prune[prune layer d + 1 to width; high = smallest pruned f]
//!     prune -->|layer d + 1 not empty| deeper[push item d + 1 = 0, U]
//!     deeper --> expand
//!     prune -->|empty| pop[pop items with high >= U]
//!     pop -->|stack empty| optimal[complete: incumbent is optimal]
//!     pop --> shift[top item: low = high, high = U; back to its layer]
//!     shift --> expand
//! ```

use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};
use search_core::{Budget, Cost, Goal, Heuristic};

use crate::anytime::{AnytimeGraphResult, AnytimeResult};
use crate::cost::Extended;
use crate::outcome::{Solution, Stats};
use crate::problem::{Successors, collect_successors};

/// Beam-stack search with beam width `width`: complete, and optimal for an admissible
/// heuristic once it runs to completion, in memory of about `width` states per layer.
///
/// Searches like breadth-first [`beam_search`](crate::beam_search), but backtracks over
/// the states the width pruned, so it finds a first solution as quickly as beam search
/// and then improves it, reporting each improvement to `on_solution`. When it finishes on
/// its own ([`AnytimeResult::complete`]), the solution is optimal for an admissible
/// `heuristic` (consistency is not needed), and `None` means no goal is reachable. `None`
/// from the heuristic marks a dead end.
///
/// A layer keeps every state whose `f` ties with the `width`-th smallest, so a layer can
/// exceed the width on plateaus; that keeps each backtracking step from repeating an empty
/// interval. A state is not added to a layer if a copy with no larger `g` is in memory (in
/// any layer from the start to the one being filled), which bounds the depth on graphs
/// with cycles. Goals are tested when a state is expanded. [`Stats::expanded`] counts
/// every expansion, also the repeated ones after backtracking. With width `1` the search
/// is a depth-first branch and bound; very wide beams approach breadth-first branch and
/// bound.
///
/// # Panics
///
/// Panics if `width` is zero.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Zero, beam_stack_search};
///
/// /// `0 -> 1 -> 3` costs 1 + 9 and looks best first; `0 -> 2 -> 3` costs 2 + 2.
/// struct Diamond;
///
/// impl Successors for Diamond {
///     type State = u8;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u8, mut emit: impl FnMut((), u8, u32)) {
///         match n {
///             0 => {
///                 emit((), 1, 1);
///                 emit((), 2, 2);
///             }
///             1 => emit((), 3, 9),
///             2 => emit((), 3, 2),
///             _ => {}
///         }
///     }
/// }
///
/// let mut costs = Vec::new();
/// let result = beam_stack_search(
///     &Diamond,
///     0,
///     &|n: &u8| *n == 3,
///     Zero,
///     1,
///     |solution| costs.push(solution.cost),
///     Limits::NONE,
/// );
///
/// assert!(result.complete);
/// assert_eq!(result.solution.map(|s| s.states), Some(vec![0, 2, 3]));
/// assert_eq!(costs, [10, 4]);
/// ```
pub fn beam_stack_search<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    width: usize,
    on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>),
    budget: impl Budget<Stats>,
) -> AnytimeGraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    beam_stack_search_multi(graph, [start], goal, heuristic, width, on_solution, budget)
}

/// [`beam_stack_search`] from several start states at once: the starts form the first
/// layer, to the cheapest goal from any of them.
///
/// The first layer is never pruned (there is no beam-stack item to remember what the
/// width would drop there), so it holds every distinct start that is not a dead end.
///
/// Repeated starts count once, an empty set of starts finds nothing (and is complete), and
/// a start that is a goal is returned as a zero-cost path without expanding anything.
///
/// # Panics
///
/// Panics if `width` is zero.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, beam_stack_search_multi, heuristic_fn};
///
/// /// Counting up by one, at cost 1.
/// struct Up;
///
/// impl Successors for Up {
///     type State = u32;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let distance = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let is_goal = |n: &u32| *n == 8;
/// let result = beam_stack_search_multi(&Up, [0, 6, 3], &is_goal, distance, 2, |_| {}, Limits::NONE);
///
/// assert!(result.complete);
/// assert_eq!(result.solution.map(|s| s.states), Some(vec![6, 7, 8]));
/// ```
pub fn beam_stack_search_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    width: usize,
    on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>),
    budget: impl Budget<Stats>,
) -> AnytimeGraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    assert!(width > 0, "the beam width must be at least 1");

    let mut search = BeamStack {
        graph,
        goal,
        heuristic,
        width,
        layers: Vec::new(),
        items: Vec::new(),
        index: HashTable::new(),
        hasher: DefaultHashBuilder::default(),
        upper: Extended::Infinite,
        best: None,
        stats: Stats::default(),
        buffer: Vec::new(),
    };
    let complete = search.run(starts, on_solution, budget);

    AnytimeResult {
        solution: search.best,
        complete,
        stats: search.stats,
    }
}

/// A state in a layer.
struct Node<S, A, C> {
    /// The state.
    state: S,
    /// The path cost from the start.
    g: C,
    /// `g + h`, saturating. Dead ends never enter a layer.
    f: C,
    /// The parent's position in the layer above, the edge label and the edge cost; `None`
    /// in the first layer. Stable: a layer changes only after every layer below it is
    /// dropped.
    parent: Option<(usize, A, C)>,
}

/// One layer of the search.
struct Layer<S, A, C> {
    /// The states, sorted by `f` once the layer is complete.
    nodes: Vec<Node<S, A, C>>,
    /// How many states, from the front, have been expanded (or skipped) so far.
    expanded: usize,
}

impl<S, A, C> Layer<S, A, C> {
    /// An empty layer.
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            expanded: 0,
        }
    }
}

/// An item of the beam stack: the interval `[low, high)` of `f` values the next layer
/// admits.
struct Item<C> {
    /// The smallest admitted `f`.
    low: C,
    /// Every admitted `f` is below this; [`Extended::Infinite`] until the width prunes.
    high: Extended<C>,
}

impl<C: Cost> Item<C> {
    /// The full interval `[0, U)`.
    fn full() -> Self {
        Self {
            low: C::ZERO,
            high: Extended::Infinite,
        }
    }
}

/// Where a node is in memory: its layer and its position there.
type Slot = (usize, usize);

/// One beam-stack search run.
struct BeamStack<'a, G: Successors, T: ?Sized, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic, owned for the run.
    heuristic: H,
    /// The beam width. Never zero.
    width: usize,
    /// The layers in memory, from the first; the last one is being filled.
    layers: Vec<Layer<G::State, G::Action, G::Cost>>,
    /// The beam stack: `items[d]` admits states into `layers[d + 1]`.
    items: Vec<Item<G::Cost>>,
    /// Every node in memory, by the hash of its state, for duplicate detection.
    index: HashTable<Slot>,
    /// Hashes states for [`BeamStack::index`].
    hasher: DefaultHashBuilder,
    /// `U`: the cost of the incumbent, or infinite without one.
    upper: Extended<G::Cost>,
    /// The incumbent: the cheapest solution found so far.
    best: Option<Solution<G::State, G::Action, G::Cost>>,
    /// Counters for the result.
    stats: Stats,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, T, H> BeamStack<'_, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Runs the search from `starts`; returns `true` if it completed, `false` if `budget`
    /// stopped it.
    fn run(
        &mut self,
        starts: impl IntoIterator<Item = G::State>,
        mut on_solution: impl FnMut(&Solution<G::State, G::Action, G::Cost>),
        mut budget: impl Budget<Stats>,
    ) -> bool {
        if self.seed(starts) {
            return true;
        }

        loop {
            if !self.sweep(&mut on_solution, &mut budget) {
                return false;
            }

            if !self.backtrack() {
                return true;
            }
        }
    }

    /// Builds the first layer from the distinct, live starts. Returns `true` if a start is
    /// a goal, which becomes the (optimal) zero-cost incumbent.
    fn seed(&mut self, starts: impl IntoIterator<Item = G::State>) -> bool {
        self.layers.push(Layer::new());
        for start in starts {
            if self.find(&start, 0).is_some() {
                continue;
            }

            if self.goal.is_goal(&start) {
                self.best = Some(Solution {
                    states: alloc::vec![start],
                    actions: Vec::new(),
                    cost: G::Cost::ZERO,
                });

                return true;
            }

            if let Some(h) = self.heuristic.estimate(&start) {
                self.insert(Node {
                    state: start,
                    g: G::Cost::ZERO,
                    f: h,
                    parent: None,
                });
            }
        }

        self.complete_layer(0);
        self.layers.push(Layer::new());
        self.items.push(Item::full());

        false
    }

    /// Expands layer after layer until a layer comes out empty. Returns `false` if `budget`
    /// stopped the search.
    fn sweep(
        &mut self,
        on_solution: &mut impl FnMut(&Solution<G::State, G::Action, G::Cost>),
        budget: &mut impl Budget<Stats>,
    ) -> bool {
        loop {
            let depth = self.layers.len() - 2;
            let Some(position) = self.next_open(depth) else {
                self.complete_layer(depth + 1);
                if self.layers[depth + 1].nodes.is_empty() {
                    return true;
                }

                self.layers.push(Layer::new());
                self.items.push(Item::full());
                continue;
            };

            let node = &self.layers[depth].nodes[position];
            if self.goal.is_goal(&node.state) {
                self.upper = Extended::Finite(node.g);
                let solution = self.solution(depth, position);
                on_solution(&solution);
                self.best = Some(solution);
                continue;
            }

            if budget.exhausted(&self.stats) {
                return false;
            }

            self.expand(depth, position);
        }
    }

    /// Moves past the states of layer `depth` that cannot beat the incumbent and returns
    /// the position of the next one to expand, if any.
    fn next_open(&mut self, depth: usize) -> Option<usize> {
        let upper = self.upper;
        let layer = &mut self.layers[depth];
        while let Some(node) = layer.nodes.get(layer.expanded) {
            layer.expanded += 1;
            if Extended::Finite(node.f) < upper {
                return Some(layer.expanded - 1);
            }
        }

        None
    }

    /// Generates the successors of the state at `position` in layer `depth` and admits
    /// those that fit item `depth` into layer `depth + 1`.
    fn expand(&mut self, depth: usize, position: usize) {
        let node = &self.layers[depth].nodes[position];
        let g = node.g;
        self.stats.expanded += 1;
        self.stats.generated += collect_successors(self.graph, &node.state, &mut self.buffer);

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            self.admit(depth + 1, state, g + cost, (position, action, cost));
        }

        self.buffer = buffer;
    }

    /// Adds `state`, reached at path cost `g` through `parent`, to layer `depth` if its `f`
    /// lies in the interval of item `depth - 1`, is below the incumbent's cost, and no copy
    /// in memory is as cheap. A dearer copy in the same layer is replaced.
    fn admit(
        &mut self,
        depth: usize,
        state: G::State,
        g: G::Cost,
        parent: (usize, G::Action, G::Cost),
    ) {
        let mut same_layer = None;
        for &(layer, position) in self.index.iter_hash(self.hasher.hash_one(&state)) {
            let node = &self.layers[layer].nodes[position];
            if node.state != state {
                continue;
            }

            if node.g <= g {
                return;
            }

            if layer == depth {
                same_layer = Some(position);
            }
        }

        let Some(h) = self.heuristic.estimate(&state) else {
            return;
        };

        let f = g.saturating_add(h);
        let item = &self.items[depth - 1];
        if f < item.low || Extended::Finite(f) >= item.high.min(self.upper) {
            return;
        }

        let node = Node {
            state,
            g,
            f,
            parent: Some(parent),
        };
        match same_layer {
            Some(position) => self.layers[depth].nodes[position] = node,
            None => self.insert(node),
        }
    }

    /// Sorts the finished layer `depth` by `f`, drops the states that cannot beat the
    /// incumbent, and prunes it to the width, keeping ties with the `width`-th state. The
    /// smallest pruned `f` becomes the upper end of the interval that admitted the layer.
    fn complete_layer(&mut self, depth: usize) {
        self.unindex(depth);

        let upper = self.upper;
        let width = self.width;
        let nodes = &mut self.layers[depth].nodes;
        nodes.retain(|node| Extended::Finite(node.f) < upper);
        nodes.sort_by_key(|node| node.f);
        // The first layer has no item to remember pruned starts, so it keeps them all.
        if depth > 0 && nodes.len() > width {
            let last = nodes[width - 1].f;
            let keep = nodes.partition_point(|node| node.f <= last);
            if keep < nodes.len() {
                let pruned = nodes[keep].f;
                nodes.truncate(keep);
                self.items[depth - 1].high = Extended::Finite(pruned);
            }
        }

        self.reindex(depth);
    }

    /// Pops the items with nothing left below the incumbent's cost, shifts the interval of
    /// the deepest remaining one to `[high, U)` and prepares its layer for re-expansion.
    /// Returns `false` if the beam stack ran empty: the search is complete.
    fn backtrack(&mut self) -> bool {
        while self
            .items
            .last()
            .is_some_and(|item| item.high >= self.upper)
        {
            self.items.pop();
        }

        let Some(item) = self.items.last_mut() else {
            return false;
        };

        let Extended::Finite(high) = item.high else {
            unreachable!("an infinite interval end is popped");
        };
        item.low = high;
        item.high = Extended::Infinite;

        let depth = self.items.len() - 1;
        while self.layers.len() > depth + 1 {
            let last = self.layers.len() - 1;
            self.unindex(last);
            self.layers.pop();
        }

        self.layers[depth].expanded = 0;
        self.layers.push(Layer::new());

        true
    }

    /// The path to the state at `position` in layer `depth`, along the parent positions.
    fn solution(&self, depth: usize, position: usize) -> Solution<G::State, G::Action, G::Cost> {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut cost = G::Cost::ZERO;
        let mut at = (depth, position);
        loop {
            let node = &self.layers[at.0].nodes[at.1];
            states.push(node.state.clone());
            let Some((parent, action, edge_cost)) = &node.parent else {
                break;
            };

            actions.push(action.clone());
            cost = cost + *edge_cost;
            at = (at.0 - 1, *parent);
        }

        states.reverse();
        actions.reverse();

        Solution {
            states,
            actions,
            cost,
        }
    }

    /// The position of `state` in layer `depth`, if it is there.
    fn find(&self, state: &G::State, depth: usize) -> Option<usize> {
        self.index
            .find(self.hasher.hash_one(state), |&(layer, position)| {
                layer == depth && self.layers[layer].nodes[position].state == *state
            })
            .map(|&(_, position)| position)
    }

    /// Appends `node` to the last layer and indexes it.
    fn insert(&mut self, node: Node<G::State, G::Action, G::Cost>) {
        let depth = self.layers.len() - 1;
        let position = self.layers[depth].nodes.len();
        let hash = self.hasher.hash_one(&node.state);
        self.layers[depth].nodes.push(node);

        let (layers, hasher) = (&self.layers, &self.hasher);
        self.index
            .insert_unique(hash, (depth, position), |&(layer, position)| {
                hasher.hash_one(&layers[layer].nodes[position].state)
            });
    }

    /// Removes the index entries of every node in layer `depth`.
    fn unindex(&mut self, depth: usize) {
        let layers = &self.layers;
        for (position, node) in layers[depth].nodes.iter().enumerate() {
            let hash = self.hasher.hash_one(&node.state);
            if let Ok(entry) = self
                .index
                .find_entry(hash, |&slot| slot == (depth, position))
            {
                entry.remove();
            }
        }
    }

    /// Indexes every node in layer `depth`.
    fn reindex(&mut self, depth: usize) {
        let (layers, hasher) = (&self.layers, &self.hasher);
        for (position, node) in layers[depth].nodes.iter().enumerate() {
            self.index.insert_unique(
                hasher.hash_one(&node.state),
                (depth, position),
                |&(layer, position)| hasher.hash_one(&layers[layer].nodes[position].state),
            );
        }
    }
}
