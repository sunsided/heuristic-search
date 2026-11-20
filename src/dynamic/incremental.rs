//! The incremental search shared by LPA* and D* Lite ([`Incremental`]): `g` and `rhs`
//! values, the inconsistency queue and tree repair.
//!
//! LPA* and D* Lite are the same algorithm in opposite directions. The *source* is where
//! distances are measured from (`rhs(source) = 0`) and the *target* is the vertex whose
//! distance is wanted:
//!
//! | Planner   | Source | Target | `rhs` from     | Changes propagate to |
//! | --------- | ------ | ------ | -------------- | -------------------- |
//! | LPA*      | start  | goal   | predecessors   | successors           |
//! | D* Lite   | goal   | start  | successors     | predecessors         |
//!
//! One call of [`Incremental::compute`]:
//!
//! ```mermaid
//! flowchart TD
//!     begin["AD*: clear CLOSED, INCONS into queue, re-key if the weight dropped"] --> top{"queue empty, or target settled?"}
//!     top -->|yes| path[path over tight edges from the target]
//!     top -->|no| key{"key of top vertex grew?"}
//!     key -->|yes| rekey[re-queue with new key] --> top
//!     key -->|no| over{"g > rhs?"}
//!     over -->|yes| settle["g = rhs, update neighbours away from source"] --> top
//!     over -->|no| reset["g = inf, update it and neighbours away from source"] --> top
//! ```
//!
//! AD* is D* Lite with an inflated heuristic for overconsistent vertices, a `CLOSED` set
//! per iteration and an `INCONS` list of vertices that became inconsistent after their
//! expansion ([`Anytime`]).

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};

use search_core::{Budget, Cost};

use crate::cost::Weight;
use crate::dynamic::{DynamicGraph, DynamicResult};
use crate::keyed_set::KeyedSet;
use crate::outcome::{Outcome, SearchResult, Solution, Stats};

/// A queue key `[min(g, rhs) + h + km, min(g, rhs)]`, compared lexicographically.
type Key<C> = (C, C);

/// Dense id of an interned vertex.
type VertexId = usize;

/// AD*'s per-iteration state.
struct Anytime {
    /// The inflation `epsilon` of the heuristic in overconsistent keys; at least `1`.
    weight: Weight,
    /// The largest weight any queued key may have been computed with. Keys only need
    /// re-keying when the weight drops below it: a larger weight makes keys grow, which
    /// the queue already handles lazily.
    keyed: Weight,
    /// Whether each vertex was made consistent in the current iteration (`CLOSED`).
    closed: Vec<bool>,
    /// The vertices closed in the current iteration, to reset `closed` without a scan.
    closed_list: Vec<VertexId>,
    /// Whether each vertex became inconsistent after it was closed (`INCONS`); such
    /// vertices wait for the next iteration instead of being queued.
    incons: Vec<bool>,
    /// The vertices ever flagged in `incons` this iteration; a superset of `INCONS`.
    incons_list: Vec<VertexId>,
}

/// Incremental search state, kept between calls.
pub(super) struct Incremental<V, C, D> {
    /// `true` for LPA* (distances from the start along successors), `false` for D* Lite
    /// (distances to the goal along predecessors).
    forward: bool,
    /// The vertex distances are measured from.
    source: VertexId,
    /// The vertex whose distance is wanted.
    target: VertexId,
    /// `heuristic(a, b)` estimates the cost from `a` to `b`.
    heuristic: D,
    /// Interned vertices by id.
    vertices: Vec<V>,
    /// Hash index from vertex to id.
    index: HashTable<VertexId>,
    /// Hashes vertices for [`Incremental::index`].
    hasher: DefaultHashBuilder,
    /// Current distance estimate per vertex; `None` is infinite.
    g: Vec<Option<C>>,
    /// One-step lookahead `min(c + g)` over the neighbours towards the source; `None` is
    /// infinite.
    rhs: Vec<Option<C>>,
    /// Locally inconsistent vertices (`g != rhs`) by key.
    queue: KeyedSet<Key<C>>,
    /// D* Lite's key modifier, in key units (scaled like the first key component): the
    /// heuristic distance the target has moved in total, inflated by AD*'s weight at the
    /// time of each move. Stays `0` for LPA*.
    km: C,
    /// AD*'s state; `None` for LPA* and D* Lite.
    anytime: Option<Anytime>,
    /// Counters over every call.
    pub(super) total: Stats,
    /// Neighbours of the vertex being processed; reused.
    buffer: Vec<(V, C)>,
}

impl<V, C, D> Incremental<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Creates the search with only the source inconsistent (`rhs = 0`, `g = inf`).
    ///
    /// `weight` makes it AD* with that initial inflation.
    pub(super) fn new(
        forward: bool,
        source: V,
        target: V,
        heuristic: D,
        weight: Option<Weight>,
    ) -> Self {
        let mut search = Self {
            forward,
            source: 0,
            target: 0,
            heuristic,
            vertices: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
            g: Vec::new(),
            rhs: Vec::new(),
            queue: KeyedSet::new(),
            km: C::ZERO,
            anytime: weight.map(|weight| Anytime {
                weight,
                keyed: weight,
                closed: Vec::new(),
                closed_list: Vec::new(),
                incons: Vec::new(),
                incons_list: Vec::new(),
            }),
            total: Stats::default(),
            buffer: Vec::new(),
        };

        search.source = search.intern(source);
        search.target = search.intern(target);
        search.rhs[search.source] = Some(C::ZERO);
        search.requeue(search.source);

        search
    }

    /// Processes inconsistent vertices until the target is consistent and no queued key
    /// is below its key, then returns the path from the target's tree.
    pub(super) fn compute<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        mut budget: impl Budget<Stats>,
    ) -> DynamicResult<G> {
        let mut stats = Stats::default();
        self.begin_iteration();
        while let Some((&key, vertex)) = self.queue.first() {
            if self.target_settled(key) {
                break;
            }

            if budget.exhausted(&stats) {
                self.total = self.total.plus(stats);
                return SearchResult {
                    outcome: Outcome::LimitReached,
                    stats,
                };
            }

            stats.expanded += 1;
            stats.generated += self.process(graph, vertex, key);
        }

        self.total = self.total.plus(stats);
        let outcome = match self.path(graph) {
            Some(solution) => Outcome::Found(solution),
            None => Outcome::Unsolvable,
        };

        SearchResult { outcome, stats }
    }

    /// Updates the vertex whose `rhs` depends on the changed edge `from -> to`.
    pub(super) fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        from: &V,
        to: &V,
    ) {
        let dependent = if self.forward { to } else { from };
        let id = self.intern(dependent.clone());
        self.update(graph, id);
    }

    /// Moves the target to `target` and raises `km` so every queued key stays a lower
    /// bound on its vertex's new key.
    ///
    /// By the triangle inequality of a consistent heuristic, a vertex's `h` towards the new
    /// target is at least its old `h` minus `h(old, new)`. Scaled keys weigh `h` by at most
    /// the weight's numerator, so adding `numerator * h(old, new)` covers inflated and plain
    /// keys alike (D* Lite's weight is `1`).
    pub(super) fn move_target(&mut self, target: V) {
        let old = self.vertices[self.target].clone();
        let moved = (self.heuristic)(&old, &target);
        self.km = self
            .km
            .saturating_add(moved.saturating_times(self.weight().numerator()));
        self.target = self.intern(target);
    }

    /// AD*: sets the inflation for the next iteration.
    pub(super) fn set_weight(&mut self, weight: Weight) {
        if let Some(anytime) = &mut self.anytime {
            anytime.weight = weight;
        }
    }

    /// AD*: the current inflation.
    pub(super) fn weight(&self) -> Weight {
        self.anytime
            .as_ref()
            .map_or(Weight::ONE, |anytime| anytime.weight)
    }

    /// AD*: starts an iteration. Empties `CLOSED` and moves `INCONS` into the queue, in
    /// time linear in their sizes. Re-keys every vertex only if the weight dropped below
    /// one some queued key was computed with; target moves are covered by `km`. Does
    /// nothing for LPA* and D* Lite.
    fn begin_iteration(&mut self) {
        let Some(anytime) = &mut self.anytime else {
            return;
        };

        for vertex in anytime.closed_list.drain(..) {
            anytime.closed[vertex] = false;
        }

        let incons = core::mem::take(&mut anytime.incons_list);
        for &vertex in &incons {
            anytime.incons[vertex] = false;
        }

        let (weight, keyed) = (anytime.weight, anytime.keyed);
        if is_below(weight, keyed) {
            anytime.keyed = weight;
            for vertex in 0..self.vertices.len() {
                self.requeue(vertex);
            }
        } else {
            anytime.keyed = if is_below(keyed, weight) {
                weight
            } else {
                keyed
            };
            for vertex in incons {
                self.requeue(vertex);
            }
        }
    }

    /// Returns `true` if the target is consistent and its key is at most `top`, the
    /// smallest queued key: the target's distance is final.
    ///
    /// Relies on positive edge costs, as in the papers: with a zero-cost edge a vertex the
    /// target depends on can tie with the target's key and be left unprocessed.
    fn target_settled(&mut self, top: Key<C>) -> bool {
        let consistent = self.g[self.target] == self.rhs[self.target];

        consistent && self.key(self.target).is_some_and(|key| top >= key)
    }

    /// Processes the queued vertex `vertex` with queue key `old`: re-keys it if its key
    /// grew, settles it if overconsistent, or resets it if underconsistent. Returns the
    /// number of neighbours generated.
    fn process<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        vertex: VertexId,
        old: Key<C>,
    ) -> u64 {
        let new = self
            .key(vertex)
            .expect("queued vertices are inconsistent, so finite");
        if old < new {
            self.queue.insert(vertex, new);
            return 0;
        }

        let overconsistent = match (self.g[vertex], self.rhs[vertex]) {
            (None, _) => true,
            (Some(g), Some(rhs)) => g.cmp(&rhs).is_gt(),
            (Some(_), None) => false,
        };

        let dependents = self.away_from_source(graph, vertex);
        if overconsistent {
            self.g[vertex] = self.rhs[vertex];
            self.queue.remove(vertex);
            if let Some(anytime) = &mut self.anytime {
                anytime.closed[vertex] = true;
                anytime.closed_list.push(vertex);
            }
        } else {
            self.g[vertex] = None;
            self.update(graph, vertex);
        }

        let generated = dependents.len() as u64;
        for id in dependents {
            self.update(graph, id);
        }

        generated
    }

    /// Recomputes `rhs(vertex)` from its neighbours towards the source and queues the
    /// vertex if it is inconsistent.
    fn update<G: DynamicGraph<Vertex = V, Cost = C>>(&mut self, graph: &G, vertex: VertexId) {
        if vertex != self.source {
            self.rhs[vertex] = self
                .towards_source(graph, vertex)
                .into_iter()
                .filter_map(|(id, cost)| self.g[id].map(|g| g + cost))
                .reduce(|a, b| if b.cmp(&a).is_lt() { b } else { a });
        }

        self.requeue(vertex);
    }

    /// Puts `vertex` into the queue with its current key if it is inconsistent, and
    /// takes it out otherwise. AD* parks inconsistent vertices closed in this iteration in
    /// `INCONS` instead.
    fn requeue(&mut self, vertex: VertexId) {
        let consistent = self.g[vertex] == self.rhs[vertex];
        if let Some(anytime) = self.anytime.as_mut().filter(|a| a.closed[vertex]) {
            if !consistent && !anytime.incons[vertex] {
                anytime.incons_list.push(vertex);
            }

            anytime.incons[vertex] = !consistent;
            self.queue.remove(vertex);
            return;
        }

        match self.key(vertex) {
            Some(key) if !consistent => self.queue.insert(vertex, key),
            _ => {
                self.queue.remove(vertex);
            }
        }
    }

    /// The key of `vertex`, or `None` if both `g` and `rhs` are infinite.
    ///
    /// `[min(g, rhs) + h + km, min(g, rhs)]`. AD* inflates `h` by its weight for
    /// overconsistent vertices (`g > rhs`); to stay exact with a rational weight, `g` and
    /// `rhs` in the first component are scaled by the weight's denominator, and `km` is
    /// kept in those scaled units.
    fn key(&mut self, vertex: VertexId) -> Option<Key<C>> {
        let (g, rhs) = (self.g[vertex], self.rhs[vertex]);
        let overconsistent = match (g, rhs) {
            (Some(g), Some(rhs)) => rhs.cmp(&g).is_lt(),
            (None, rhs) => rhs.is_some(),
            (Some(_), None) => false,
        };
        let best = if overconsistent { rhs } else { g }?;

        let h = self.h(vertex);
        let weight = self.weight();
        let inflation = if overconsistent {
            weight.numerator()
        } else {
            weight.denominator()
        };
        let first = best
            .saturating_times(weight.denominator())
            .saturating_add(h.saturating_times(inflation))
            .saturating_add(self.km);

        Some((first, best))
    }

    /// The heuristic between the target and `vertex`, in travel direction.
    fn h(&mut self, vertex: VertexId) -> C {
        let (vertex, target) = (&self.vertices[vertex], &self.vertices[self.target]);
        if self.forward {
            (self.heuristic)(vertex, target)
        } else {
            (self.heuristic)(target, vertex)
        }
    }

    /// The neighbours of `vertex` one step closer to the source, with edge costs:
    /// predecessors for LPA*, successors for D* Lite.
    fn towards_source<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        vertex: VertexId,
    ) -> Vec<(VertexId, C)> {
        self.neighbours(graph, vertex, !self.forward)
    }

    /// The neighbours of `vertex` one step further from the source: successors for LPA*,
    /// predecessors for D* Lite.
    fn away_from_source<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        vertex: VertexId,
    ) -> Vec<VertexId> {
        let forward = self.forward;

        self.neighbours(graph, vertex, forward)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// The successors (if `successors`) or predecessors of `vertex`, interned, with costs.
    fn neighbours<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        vertex: VertexId,
        successors: bool,
    ) -> Vec<(VertexId, C)> {
        let mut buffer = core::mem::take(&mut self.buffer);
        let state = &self.vertices[vertex];
        let mut emit = |other: V, cost: C| buffer.push((other, cost));
        if successors {
            graph.successors(state, &mut emit);
        } else {
            graph.predecessors(state, &mut emit);
        }

        let neighbours = buffer
            .drain(..)
            .map(|(other, cost)| (self.intern(other), cost))
            .collect();
        self.buffer = buffer;

        neighbours
    }

    /// A cheapest path between source and target over tight edges (`g(u) = c + g(v)`),
    /// found by breadth-first search from the target, in travel direction.
    fn path<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
    ) -> Option<Solution<V, (), C>> {
        self.g[self.target]?;

        let mut parent: Vec<Option<(VertexId, C)>> = vec![None; self.vertices.len()];
        let mut seen = vec![false; self.vertices.len()];
        let mut frontier = VecDeque::from([self.target]);
        seen[self.target] = true;

        while let Some(vertex) = frontier.pop_front() {
            if vertex == self.source {
                return Some(self.trace(&parent));
            }

            let g = self.g[vertex]?;
            let neighbours = self.towards_source(graph, vertex);
            for &(next, cost) in &neighbours {
                let tight = self
                    .g
                    .get(next)
                    .copied()
                    .flatten()
                    .is_some_and(|g_next| g_next + cost == g);
                if !tight || seen.get(next).copied().unwrap_or(false) {
                    continue;
                }

                if parent.len() <= next {
                    parent.resize(next + 1, None);
                    seen.resize(next + 1, false);
                }

                seen[next] = true;
                parent[next] = Some((vertex, cheapest_edge(&neighbours, next)));
                frontier.push_back(next);
            }
        }

        debug_assert!(
            self.anytime.is_some(),
            "without inflation, tight edges always lead from the target to the source"
        );

        self.greedy_path(graph)
    }

    /// AD*'s path when inconsistent vertices break every tight chain: from the target,
    /// repeatedly step to the unvisited neighbour towards the source with the smallest
    /// `c + g`. `None` if that gets stuck.
    fn greedy_path<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
    ) -> Option<Solution<V, (), C>> {
        let mut seen = vec![false; self.vertices.len()];
        let mut steps = vec![(self.target, C::ZERO)];
        let mut current = self.target;
        seen[current] = true;

        while current != self.source {
            let (next, cost) = self
                .towards_source(graph, current)
                .into_iter()
                .filter(|&(id, _)| !seen.get(id).copied().unwrap_or(false))
                .filter_map(|(id, cost)| self.g[id].map(|g| (id, cost, g + cost)))
                .reduce(|a, b| if b.2.cmp(&a.2).is_lt() { b } else { a })
                .map(|(id, cost, _)| (id, cost))?;

            if seen.len() <= next {
                seen.resize(next + 1, false);
            }

            seen[next] = true;
            steps.push((next, cost));
            current = next;
        }

        let mut states: Vec<V> = steps
            .iter()
            .map(|&(id, _)| self.vertices[id].clone())
            .collect();
        let cost = steps.iter().fold(C::ZERO, |sum, &(_, cost)| sum + cost);
        if self.forward {
            states.reverse();
        }

        Some(Solution {
            actions: vec![(); states.len() - 1],
            states,
            cost,
        })
    }

    /// Builds the solution from the BFS parent links, which lead from the source back to
    /// the target, in travel direction (start to goal).
    fn trace(&self, parent: &[Option<(VertexId, C)>]) -> Solution<V, (), C> {
        let mut states = vec![self.vertices[self.source].clone()];
        let mut cost = C::ZERO;
        let mut current = self.source;
        while let Some((next, edge)) = parent[current] {
            states.push(self.vertices[next].clone());
            cost = cost + edge;
            current = next;
        }

        if !self.forward {
            states.reverse();
        }

        Solution {
            actions: vec![(); states.len() - 1],
            states,
            cost,
        }
    }

    /// The id of `vertex`, interning it with infinite `g` and `rhs` if it is new.
    fn intern(&mut self, vertex: V) -> VertexId {
        let hash = self.hasher.hash_one(&vertex);
        let vertices = &self.vertices;
        if let Some(&id) = self.index.find(hash, |&id| vertices[id] == vertex) {
            return id;
        }

        let id = self.vertices.len();
        let hasher = &self.hasher;
        self.index
            .insert_unique(hash, id, |&other| hasher.hash_one(&vertices[other]));
        self.vertices.push(vertex);
        self.g.push(None);
        self.rhs.push(None);
        if let Some(anytime) = &mut self.anytime {
            anytime.closed.push(false);
            anytime.incons.push(false);
        }

        id
    }
}

/// Returns `true` if weight `a` is below weight `b`, compared exactly.
fn is_below(a: Weight, b: Weight) -> bool {
    u64::from(a.numerator()) * u64::from(b.denominator())
        < u64::from(b.numerator()) * u64::from(a.denominator())
}

/// The cheapest of the parallel edges to `target` among `neighbours`.
///
/// A stale `g` under AD* inflation can make a dearer parallel edge the tight one; the
/// path's states are the same, and its cost is that of the cheapest edges between them.
fn cheapest_edge<C: Cost>(neighbours: &[(VertexId, C)], target: VertexId) -> C {
    neighbours
        .iter()
        .filter(|&&(id, _)| id == target)
        .map(|&(_, cost)| cost)
        .min()
        .expect("`target` is one of the neighbours")
}
