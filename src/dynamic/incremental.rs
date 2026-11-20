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

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};

use crate::cost::{Cost, Ordered};
use crate::dynamic::{DynamicGraph, DynamicResult};
use crate::keyed_set::KeyedSet;
use crate::outcome::{Budget, Outcome, SearchResult, Solution, Stats};

/// A queue key `[min(g, rhs) + h + km, min(g, rhs)]`, compared lexicographically.
type Key<C> = (Ordered<C>, Ordered<C>);

/// Dense id of an interned vertex.
type VertexId = usize;

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
    /// D* Lite's key modifier: the heuristic distance the target has moved in total.
    km: C,
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
    pub(super) fn new(forward: bool, source: V, target: V, heuristic: D) -> Self {
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
        mut budget: impl Budget,
    ) -> DynamicResult<G> {
        let mut stats = Stats::default();
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

    /// Moves the target to `target`, raising `km` by the heuristic distance moved so old
    /// keys stay lower bounds.
    pub(super) fn move_target(&mut self, target: V) {
        let old = self.vertices[self.target].clone();
        self.km = self.km + (self.heuristic)(&old, &target);
        self.target = self.intern(target);
    }

    /// Returns `true` if the target is consistent and its key is at most `top`, the
    /// smallest queued key: the target's distance is final.
    ///
    /// Relies on positive edge costs, as in the papers: with a zero-cost edge a vertex the
    /// target depends on can tie with the target's key and be left unprocessed.
    fn target_settled(&mut self, top: Key<C>) -> bool {
        let consistent = self.g[self.target].map(Ordered) == self.rhs[self.target].map(Ordered);

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
            (Some(g), Some(rhs)) => g.cmp_cost(&rhs).is_gt(),
            (Some(_), None) => false,
        };

        let dependents = self.away_from_source(graph, vertex);
        if overconsistent {
            self.g[vertex] = self.rhs[vertex];
            self.queue.remove(vertex);
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
                .reduce(|a, b| if b.cmp_cost(&a).is_lt() { b } else { a });
        }

        self.requeue(vertex);
    }

    /// Puts `vertex` into the queue with its current key if it is inconsistent, and
    /// takes it out otherwise.
    fn requeue(&mut self, vertex: VertexId) {
        let consistent = self.g[vertex].map(Ordered) == self.rhs[vertex].map(Ordered);
        match self.key(vertex) {
            Some(key) if !consistent => self.queue.insert(vertex, key),
            _ => {
                self.queue.remove(vertex);
            }
        }
    }

    /// The key of `vertex`, or `None` if both `g` and `rhs` are infinite.
    fn key(&mut self, vertex: VertexId) -> Option<Key<C>> {
        let best = match (self.g[vertex], self.rhs[vertex]) {
            (Some(g), Some(rhs)) => Some(if rhs.cmp_cost(&g).is_lt() { rhs } else { g }),
            (g, rhs) => g.or(rhs),
        }?;

        let h = self.h(vertex);

        Some((Ordered(best + h + self.km), Ordered(best)))
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
            for (next, cost) in self.towards_source(graph, vertex) {
                let tight = self
                    .g
                    .get(next)
                    .copied()
                    .flatten()
                    .is_some_and(|g_next| (g_next + cost).cmp_cost(&g).is_eq());
                if !tight || seen.get(next).copied().unwrap_or(false) {
                    continue;
                }

                if parent.len() <= next {
                    parent.resize(next + 1, None);
                    seen.resize(next + 1, false);
                }

                seen[next] = true;
                parent[next] = Some((vertex, cost));
                frontier.push_back(next);
            }
        }

        debug_assert!(
            false,
            "tight edges always lead from the target to the source"
        );
        None
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

        id
    }
}
