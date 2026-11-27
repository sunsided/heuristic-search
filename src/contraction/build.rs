//! Preprocessing for [`ContractionHierarchy`]: state discovery, node ordering with lazy
//! updates, witness searches and shortcuts.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::hash::Hash;

use search_core::Cost;

use super::{BACKWARD, ContractionHierarchy, Edge, EdgeId, FORWARD, Via, WitnessLimits};
use crate::problem::{Successors, collect_successors};

/// A shortcut to add when a state is contracted: the edges `u -> v` and `v -> w` it stands
/// for, and its cost.
type Shortcut<C> = (EdgeId, EdgeId, C);

impl<S, A, C> ContractionHierarchy<S, A, C>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
{
    /// Indexes `seeds` and every state reachable from them, and returns every edge between
    /// them except self loops, in discovery order.
    pub(super) fn discover<G>(
        &mut self,
        graph: &G,
        seeds: impl IntoIterator<Item = S>,
    ) -> Vec<Edge<A, C>>
    where
        G: Successors<State = S, Action = A, Cost = C>,
    {
        for seed in seeds {
            self.intern(seed);
        }

        let mut edges = Vec::new();
        let mut buffer = Vec::new();
        let mut from = 0;
        while from < self.states.len() {
            collect_successors(graph, &self.states[from], &mut buffer);
            for (action, state, cost) in buffer.drain(..) {
                let to = self.intern(state);
                if to != from {
                    edges.push(Edge {
                        from,
                        to,
                        cost,
                        via: Via::Original(action),
                    });
                }
            }

            from += 1;
        }

        edges
    }

    /// Contracts every state, least important first, and records the ranks, the upward
    /// edges and the shortcuts.
    ///
    /// The importance is recomputed when a state reaches the front of the queue (lazy
    /// updates): if it grew beyond the next state's, the state goes back into the queue.
    pub(super) fn contract(&mut self, original: Vec<Edge<A, C>>, limits: WitnessLimits) {
        let count = self.states.len();
        let mut overlay = Overlay::new(count, limits);
        for edge in original {
            overlay.add(edge);
        }

        let mut queue: BinaryHeap<_> = (0..count)
            .map(|v| {
                let shortcuts = overlay.shortcuts(v);
                Reverse((overlay.priority(v, shortcuts.len()), v))
            })
            .collect();

        self.rank = vec![0; count];
        self.up = [vec![Vec::new(); count], vec![Vec::new(); count]];
        let mut next_rank = 0;
        while let Some(Reverse((_, v))) = queue.pop() {
            let shortcuts = overlay.shortcuts(v);
            let priority = overlay.priority(v, shortcuts.len());
            if queue
                .peek()
                .is_some_and(|Reverse((next, _))| priority > *next)
            {
                queue.push(Reverse((priority, v)));
                continue;
            }

            self.rank[v] = next_rank;
            next_rank += 1;
            let [forward, backward] = overlay.remove(v);
            self.up[FORWARD][v] = forward;
            self.up[BACKWARD][v] = backward;
            for (first, second, cost) in shortcuts {
                let edge = Edge {
                    from: overlay.edges[first].from,
                    to: overlay.edges[second].to,
                    cost,
                    via: Via::Shortcut(first, second),
                };
                overlay.add(edge);
            }
        }

        self.edges = overlay.edges;
    }
}

/// The graph of the states not contracted yet, with the shortcuts added so far: at most
/// one edge, the cheapest, from each state to each other.
struct Overlay<A, C> {
    /// Every edge created so far; becomes [`ContractionHierarchy::edges`].
    edges: Vec<Edge<A, C>>,
    /// Per state, the overlay edges leaving it.
    out: Vec<Vec<EdgeId>>,
    /// Per state, the overlay edges entering it.
    into: Vec<Vec<EdgeId>>,
    /// Per state, how many of its neighbours have been contracted.
    contracted_neighbours: Vec<usize>,
    /// Per state, an upper bound on the depth of the upward search spaces below it: one
    /// more than the deepest contracted neighbour.
    depth: Vec<usize>,
    /// The witness search and its workspace.
    witness: Witness<C>,
}

impl<A, C: Cost> Overlay<A, C> {
    /// An overlay of `count` states without edges.
    fn new(count: usize, limits: WitnessLimits) -> Self {
        Self {
            edges: Vec::new(),
            out: vec![Vec::new(); count],
            into: vec![Vec::new(); count],
            contracted_neighbours: vec![0; count],
            depth: vec![0; count],
            witness: Witness::new(count, limits),
        }
    }

    /// Adds `edge`, unless an edge between the same states is at most as cheap; a dearer
    /// one is replaced in place.
    ///
    /// Overlay edges are not referred to by shortcuts or upward edges yet (those take
    /// edges out of the overlay), so replacing one changes nothing else.
    fn add(&mut self, edge: Edge<A, C>) {
        let edges = &self.edges;
        let existing = self.out[edge.from]
            .iter()
            .copied()
            .find(|&id| edges[id].to == edge.to);

        if let Some(id) = existing {
            if edge.cost < self.edges[id].cost {
                self.edges[id] = edge;
            }

            return;
        }

        let id = self.edges.len();
        self.out[edge.from].push(id);
        self.into[edge.to].push(id);
        self.edges.push(edge);
    }

    /// The shortcuts that contracting `v` needs: one for each pair `u -> v -> w` (with
    /// `u != w`) that the witness search finds no path at most as cheap for without `v`.
    fn shortcuts(&mut self, v: usize) -> Vec<Shortcut<C>> {
        let mut shortcuts = Vec::new();
        for &first in &self.into[v] {
            let (u, to_v) = (self.edges[first].from, self.edges[first].cost);
            let bypassed = || {
                self.out[v]
                    .iter()
                    .copied()
                    .filter(|&second| self.edges[second].to != u)
                    .map(|second| (second, to_v + self.edges[second].cost))
            };
            let Some(limit) = bypassed().map(|(_, cost)| cost).max() else {
                continue;
            };

            self.witness.search(&self.edges, &self.out, u, v, limit);
            for (second, cost) in bypassed() {
                if !self.witness.reaches(self.edges[second].to, cost) {
                    shortcuts.push((first, second, cost));
                }
            }
        }

        shortcuts
    }

    /// The importance of `v` if contracting it adds `shortcuts` shortcuts: the edge
    /// difference plus the contracted neighbours plus the search space depth. Smaller is
    /// contracted first.
    fn priority(&self, v: usize, shortcuts: usize) -> i64 {
        let removed = self.out[v].len() + self.into[v].len();

        signed(shortcuts) - signed(removed)
            + signed(self.contracted_neighbours[v])
            + signed(self.depth[v])
    }

    /// Takes `v` out of the overlay and returns its edges, which all lead to states
    /// contracted later: `[leaving, entering]`.
    fn remove(&mut self, v: usize) -> [Vec<EdgeId>; 2] {
        let leaving = core::mem::take(&mut self.out[v]);
        let entering = core::mem::take(&mut self.into[v]);

        for &id in &leaving {
            let w = self.edges[id].to;
            self.into[w].retain(|&other| other != id);
            self.neighbour_contracted(w, v);
        }

        for &id in &entering {
            let u = self.edges[id].from;
            self.out[u].retain(|&other| other != id);
            self.neighbour_contracted(u, v);
        }

        [leaving, entering]
    }

    /// Records that `neighbour` lost its neighbour `v` to contraction.
    fn neighbour_contracted(&mut self, neighbour: usize, v: usize) {
        self.contracted_neighbours[neighbour] += 1;
        self.depth[neighbour] = self.depth[neighbour].max(self.depth[v] + 1);
    }
}

/// A priority term as a signed number; the counts are far below `i64::MAX`.
fn signed(count: usize) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}

/// A Dijkstra search on the overlay that avoids the state being contracted, bounded by
/// [`WitnessLimits`] and by the cost of the paths it tries to replace.
struct Witness<C> {
    /// The limits on hops and settled states.
    limits: WitnessLimits,
    /// Per state, the cheapest cost found and the edges on that path; `None` if not reached.
    labels: Vec<Option<(C, u32)>>,
    /// The states with a label, for resetting them.
    touched: Vec<usize>,
    /// The open list, with lazy deletion.
    heap: BinaryHeap<Reverse<(C, usize)>>,
}

impl<C: Cost> Witness<C> {
    /// A witness search over `count` states.
    fn new(count: usize, limits: WitnessLimits) -> Self {
        Self {
            limits,
            labels: vec![None; count],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Searches from `source` on the overlay given by `edges` and `out` without passing
    /// `skip`, until the next state costs more than `limit` or a limit is hit.
    fn search<A>(
        &mut self,
        edges: &[Edge<A, C>],
        out: &[Vec<EdgeId>],
        source: usize,
        skip: usize,
        limit: C,
    ) {
        for state in self.touched.drain(..) {
            self.labels[state] = None;
        }

        self.heap.clear();
        self.label(source, C::ZERO, 0);
        let mut settled = 0;
        while let Some(Reverse((cost, state))) = self.heap.pop() {
            let Some((best, hops)) = self.labels[state] else {
                continue;
            };
            if cost > best {
                continue;
            }
            if cost > limit || settled >= self.limits.max_settled {
                break;
            }

            settled += 1;
            if hops >= self.limits.max_hops {
                continue;
            }

            for &id in &out[state] {
                let edge = &edges[id];
                let next = cost + edge.cost;
                let better = self.labels[edge.to].is_none_or(|(old, _)| next < old);
                if edge.to != skip && better {
                    self.label(edge.to, next, hops + 1);
                }
            }
        }
    }

    /// Gives `state` the label `cost` over `hops` edges and queues it.
    fn label(&mut self, state: usize, cost: C, hops: u32) {
        if self.labels[state].is_none() {
            self.touched.push(state);
        }

        self.labels[state] = Some((cost, hops));
        self.heap.push(Reverse((cost, state)));
    }

    /// Returns `true` if the last search found a path to `state` costing at most `cost`.
    fn reaches(&self, state: usize, cost: C) -> bool {
        self.labels[state].is_some_and(|(found, _)| found <= cost)
    }
}
