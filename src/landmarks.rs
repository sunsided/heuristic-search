//! ALT landmark heuristics ([`Landmarks`]): lower bounds from precomputed distances to and
//! from a few landmark states, for many queries on one graph.
//!
//! For a landmark `l`, the triangle inequality gives two lower bounds on the distance
//! `d(v, t)`: `d(l, t) - d(l, v)` and `d(v, l) - d(t, l)`. The ALT heuristic (A*,
//! landmarks, triangle inequality; Goldberg and Harrelson) is the largest of them over all
//! landmarks. It is admissible and consistent on directed graphs, because every term is.
//!
//! [`Landmarks::build`] runs one forward and one backward exhaustive Dijkstra per landmark
//! and keeps both distances for every state it reaches. [`Landmarks::heuristic_to`] fixes a
//! target and returns a [`LandmarkHeuristic`] for [`a_star`](crate::a_star) and every
//! other search that takes a [`Heuristic`]; [`Landmarks::heuristic_from`] fixes a source,
//! for the backward side of the bidirectional searches. [`LandmarkHeuristic::active`]
//! keeps only the landmarks that bound one query best.
//!
//! A landmark that does not reach both states (or is not reached by both) gives no bound
//! for that pair and is skipped, so no infinite distance is ever subtracted. When a
//! landmark reaches `v` but not `t`, or `t` reaches it but `v` does not, `v` cannot reach
//! `t`, and the estimate is `None` (a proven dead end).
//!
//! With float costs (`NonNegF64`) the differences of rounded sums are exact only up to
//! rounding, so the guarantees hold up to that rounding as well.
//!
//! Landmarks are picked by [`LandmarkSelection`]: farthest-first (Goldberg and Harrelson)
//! or avoidance (Goldberg and Werneck).

use alloc::vec;
use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};

use search_core::{Cost, CostExt, Heuristic};

use crate::bidirectional::BidirectionalGraph;
use crate::cost::Extended;
use crate::open_list::OpenList;
use crate::problem::{Successors, collect_predecessors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// How [`Landmarks::build`] picks its landmarks.
///
/// Neither choice affects admissibility or consistency, only how tight the bounds are.
/// Both are deterministic: where the papers pick a vertex at random, this picks the start
/// state passed to [`Landmarks::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LandmarkSelection {
    /// Farthest-first: the first landmark is the state farthest from the start, and each
    /// further one the state farthest from every landmark so far. Distance is the round
    /// trip `d(l, v) + d(v, l)`; a state without a round trip to some landmark counts as
    /// infinitely far, so landmarks spread over every part of the graph that was found.
    #[default]
    Farthest,
    /// Avoidance: grows the shortest-path tree from the start and weighs each state by how
    /// far its distance exceeds the current lower bound. Starting at the state whose
    /// subtree weighs most and holds no landmark, it follows the heaviest child down to a
    /// leaf, which becomes the next landmark. Falls back to [`LandmarkSelection::Farthest`]
    /// once every bound in the tree is exact.
    Avoidance,
}

/// Exact distances to and from landmark states, for ALT lower bounds.
///
/// Built once per graph by [`Landmarks::build`]; the heuristics it hands out borrow it, so
/// one table serves any number of queries.
///
/// # Examples
///
/// ```
/// use heuristic_search::{
///     LandmarkSelection, Landmarks, Limits, Predecessors, Successors, a_star, dijkstra,
/// };
///
/// /// A directed graph as adjacency lists of `(target, cost)` pairs.
/// struct Graph(Vec<Vec<(usize, u32)>>);
///
/// impl Successors for Graph {
///     type State = usize;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, state: &usize, mut emit: impl FnMut((), usize, u32)) {
///         for &(target, cost) in &self.0[*state] {
///             emit((), target, cost);
///         }
///     }
/// }
///
/// impl Predecessors for Graph {
///     type Node = usize;
///     type Action = ();
///     type Cost = u32;
///
///     fn predecessors(&self, node: &usize, mut emit: impl FnMut((), usize, u32)) {
///         for (source, edges) in self.0.iter().enumerate() {
///             for &(target, cost) in edges {
///                 if target == *node {
///                     emit((), source, cost);
///                 }
///             }
///         }
///     }
/// }
///
/// // A ring 0 -> 1 -> ... -> 5 -> 0 with a shortcut 0 -> 3.
/// let mut edges: Vec<_> = (0..6).map(|v| vec![((v + 1) % 6, 2)]).collect();
/// edges[0].push((3, 5));
/// let graph = Graph(edges);
///
/// let landmarks = Landmarks::build(&graph, 0, 2, LandmarkSelection::Farthest);
/// let goal = |state: &usize| *state == 4;
/// let alt = a_star(&graph, 1, &goal, landmarks.heuristic_to(&4), Limits::NONE);
/// let plain = dijkstra(&graph, 1, &goal, Limits::NONE);
///
/// assert_eq!(alt.solution().map(|s| s.cost), Some(6));
/// assert!(alt.stats.expanded <= plain.stats.expanded);
/// assert_eq!(landmarks.lower_bound(&4, &1), Some(6));
/// ```
#[derive(Debug, Clone)]
pub struct Landmarks<S, C> {
    /// Every state found by the build, by id.
    table: StateTable<S>,
    /// The landmark states, by id, in the order they were picked.
    picked: Vec<usize>,
    /// Row-major: the distances of state `id` to and from landmark `l` are at
    /// `id * picked.len() + l`.
    rows: Vec<Distances<C>>,
}

/// The distances between one state and one landmark; `None` where there is no path.
#[derive(Debug, Clone, Copy)]
struct Distances<C> {
    /// `d(l, v)`: from the landmark to the state.
    from_landmark: Option<C>,
    /// `d(v, l)`: from the state to the landmark.
    to_landmark: Option<C>,
}

impl<C> Distances<C> {
    /// No path in either direction.
    const NONE: Self = Self {
        from_landmark: None,
        to_landmark: None,
    };
}

impl<S, C> Landmarks<S, C>
where
    S: Clone + Hash + Eq,
    C: CostExt,
{
    /// Picks up to `count` landmarks by `selection` and computes the distances between
    /// them and every state they reach or are reached from.
    ///
    /// `start` anchors the selection: the graph traits cannot list states, so the first
    /// landmark is found by searching from `start`. States that neither reach a landmark
    /// nor are reached from one get no bounds (their estimate is `0`, or `None` when the
    /// other end proves them disconnected). Each landmark costs one forward and one
    /// backward Dijkstra over everything it connects to, so the graph must be finite there.
    /// Fewer than `count` landmarks are picked when fewer states were found.
    pub fn build<G>(graph: &G, start: S, count: usize, selection: LandmarkSelection) -> Self
    where
        G: BidirectionalGraph + Successors<State = S, Cost = C>,
    {
        let mut builder = Builder::new(graph, start);
        while builder.landmarks.len() < count {
            let next = match selection {
                LandmarkSelection::Farthest => builder.farthest(),
                LandmarkSelection::Avoidance => builder.avoidance().or_else(|| builder.farthest()),
            };
            let Some(next) = next else {
                break;
            };

            builder.add_landmark(next);
        }

        builder.finish()
    }

    /// The landmark states, in the order they were picked.
    pub fn landmarks(&self) -> impl ExactSizeIterator<Item = &S> {
        self.picked.iter().map(|&id| &self.table.states[id])
    }

    /// A lower bound on the cost of the cheapest path from `from` to `to`, or `None` if
    /// the landmarks prove that there is no path.
    #[must_use]
    pub fn lower_bound(&self, from: &S, to: &S) -> Option<C> {
        self.bound(
            self.table.find(from),
            self.table.find(to),
            0..self.picked.len(),
        )
    }

    /// The ALT heuristic towards `target`: a lower bound on the cost from a state to
    /// `target`, for the forward side of a search.
    #[must_use]
    pub fn heuristic_to(&self, target: &S) -> LandmarkHeuristic<'_, S, C> {
        LandmarkHeuristic::new(self, self.table.find(target), Toward::Endpoint)
    }

    /// The ALT heuristic away from `source`: a lower bound on the cost from `source` to a
    /// state, for the backward side of the bidirectional searches.
    #[must_use]
    pub fn heuristic_from(&self, source: &S) -> LandmarkHeuristic<'_, S, C> {
        LandmarkHeuristic::new(self, self.table.find(source), Toward::State)
    }

    /// The distances between state `id` (`None` if it was never found) and landmark `l`.
    fn distances(&self, id: Option<usize>, l: usize) -> Distances<C> {
        id.map_or(Distances::NONE, |id| self.rows[id * self.picked.len() + l])
    }

    /// The bound from state `from` to state `to` over the landmarks `active`.
    fn bound(
        &self,
        from: Option<usize>,
        to: Option<usize>,
        active: impl IntoIterator<Item = usize>,
    ) -> Option<C> {
        bound(
            active
                .into_iter()
                .map(|l| (self.distances(from, l), self.distances(to, l))),
        )
    }
}

/// Which end of a path a [`LandmarkHeuristic`] keeps fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Toward {
    /// Estimates the cost from the evaluated state to the fixed endpoint.
    Endpoint,
    /// Estimates the cost from the fixed endpoint to the evaluated state.
    State,
}

/// The ALT heuristic for one fixed endpoint, from [`Landmarks::heuristic_to`] or
/// [`Landmarks::heuristic_from`].
///
/// Admissible and consistent. Estimates `None` only for states the landmarks prove
/// disconnected from the endpoint.
#[derive(Debug, Clone)]
pub struct LandmarkHeuristic<'a, S, C> {
    /// The distance tables.
    landmarks: &'a Landmarks<S, C>,
    /// The id of the fixed endpoint, or `None` if the build never found it.
    endpoint: Option<usize>,
    /// Whether the endpoint is the target or the source of the estimated paths.
    toward: Toward,
    /// The landmarks used, by index into [`Landmarks::landmarks`].
    active: Vec<usize>,
}

impl<'a, S, C> LandmarkHeuristic<'a, S, C>
where
    S: Clone + Hash + Eq,
    C: CostExt,
{
    /// A heuristic over every landmark.
    fn new(landmarks: &'a Landmarks<S, C>, endpoint: Option<usize>, toward: Toward) -> Self {
        Self {
            landmarks,
            endpoint,
            toward,
            active: (0..landmarks.picked.len()).collect(),
        }
    }

    /// Keeps only the `count` landmarks that give the best bound between `other` and the
    /// fixed endpoint (active landmarks).
    ///
    /// Pass the start of the query to a heuristic from [`Landmarks::heuristic_to`], or its
    /// target to one from [`Landmarks::heuristic_from`]. Fewer landmarks make each estimate
    /// cheaper. The heuristic stays admissible and consistent, because the set is fixed
    /// for the whole search.
    #[must_use]
    pub fn active(mut self, other: &S, count: usize) -> Self {
        let other = self.landmarks.table.find(other);
        let (from, to) = self.ordered(other);
        let bound_of = |l: usize| {
            landmark_bound(
                self.landmarks.distances(from, l),
                self.landmarks.distances(to, l),
            )
        };

        // A landmark that proves the pair disconnected (`None`) ranks first.
        let mut ranked: Vec<(Extended<C>, usize)> = self
            .active
            .iter()
            .map(|&l| (bound_of(l).map_or(Extended::Infinite, Extended::Finite), l))
            .collect();
        ranked.sort_by_key(|&(bound, _)| core::cmp::Reverse(bound));
        self.active = ranked.into_iter().take(count).map(|(_, l)| l).collect();

        self
    }

    /// `(from, to)` for a path between `state` and the endpoint, in this heuristic's
    /// direction.
    fn ordered(&self, state: Option<usize>) -> (Option<usize>, Option<usize>) {
        match self.toward {
            Toward::Endpoint => (state, self.endpoint),
            Toward::State => (self.endpoint, state),
        }
    }
}

impl<S, C> Heuristic<S, C> for LandmarkHeuristic<'_, S, C>
where
    S: Clone + Hash + Eq,
    C: CostExt,
{
    fn estimate(&mut self, state: &S) -> Option<C> {
        let (from, to) = self.ordered(self.landmarks.table.find(state));

        self.landmarks.bound(from, to, self.active.iter().copied())
    }
}

/// The largest bound over the landmarks' distance pairs `(from, to)`, or `None` if one of
/// them proves `to` unreachable from `from`. `0` without landmarks.
fn bound<C: CostExt>(pairs: impl IntoIterator<Item = (Distances<C>, Distances<C>)>) -> Option<C> {
    pairs.into_iter().try_fold(C::ZERO, |best, (from, to)| {
        landmark_bound(from, to).map(|bound| best.max(bound))
    })
}

/// The bound one landmark `l` gives on `d(u, w)`, from its distances to and from `u` and
/// `w`: the larger of `d(l, w) - d(l, u)` and `d(u, l) - d(w, l)`, skipping a term with a
/// missing distance. `None` if `l` reaches `u` but not `w`, or `w` reaches `l` but `u` does
/// not: then `u` cannot reach `w`.
fn landmark_bound<C: CostExt>(u: Distances<C>, w: Distances<C>) -> Option<C> {
    let mut best = C::ZERO;
    match (u.from_landmark, w.from_landmark) {
        (Some(lu), Some(lw)) => best = best.max(lw.saturating_sub(lu)),
        (Some(_), None) => return None,
        (None, _) => {}
    }
    match (u.to_landmark, w.to_landmark) {
        (Some(ul), Some(wl)) => best = best.max(ul.saturating_sub(wl)),
        (None, Some(_)) => return None,
        (_, None) => {}
    }

    Some(best)
}

/// States by dense id, with a hash index for lookups.
#[derive(Debug, Clone)]
struct StateTable<S> {
    /// The states, by id.
    states: Vec<S>,
    /// Hash index from state to id.
    index: HashTable<usize>,
    /// Hashes states for [`StateTable::index`].
    hasher: DefaultHashBuilder,
}

impl<S: Hash + Eq> StateTable<S> {
    /// An empty table.
    fn new() -> Self {
        Self {
            states: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
        }
    }

    /// The id of `state`, if it is known.
    fn find(&self, state: &S) -> Option<usize> {
        let hash = self.hasher.hash_one(state);

        self.index
            .find(hash, |&id| self.states[id] == *state)
            .copied()
    }

    /// The id of `state`, adding it if it is new.
    fn intern(&mut self, state: &S) -> usize
    where
        S: Clone,
    {
        if let Some(id) = self.find(state) {
            return id;
        }

        let id = self.states.len();
        let hash = self.hasher.hash_one(state);
        let (states, hasher) = (&self.states, &self.hasher);
        self.index
            .insert_unique(hash, id, |&other| hasher.hash_one(&states[other]));
        self.states.push(state.clone());

        id
    }
}

/// One exhaustive Dijkstra run: every state connected to the source, with its distance
/// and shortest-path parent.
struct Tree<S, C> {
    /// The states found, with distance and parent. The source is node `0`.
    space: SearchSpace<S, (), C>,
    /// The nodes in the order they were settled; parents come before their children.
    order: Vec<NodeId>,
    /// The [`StateTable`] id of each node.
    ids: Vec<usize>,
}

/// Runs Dijkstra from `source` until every connected state is settled: along the edges, or
/// against them if `backward` is set (then distances are to `source`).
fn settle<G, S, C>(graph: &G, source: S, backward: bool) -> Tree<S, C>
where
    G: BidirectionalGraph + Successors<State = S, Cost = C>,
    S: Clone + Hash + Eq,
    C: Cost,
{
    let (mut space, _) = SearchSpace::seed([source], &|_: &S| false);
    let mut open: OpenList<C> = OpenList::new();
    open.push(0, space.g(0), space.g(0));

    let mut order = Vec::new();
    let mut buffer = Vec::new();
    while let Some((node, _)) = open.pop_where(|id, g| g.cmp(&space.g(id)).is_eq()) {
        order.push(node);
        if backward {
            collect_predecessors(graph, space.state(node), &mut buffer);
        } else {
            collect_successors(graph, space.state(node), &mut buffer);
        }

        for (_, state, cost) in buffer.drain(..) {
            match space.reach(state, node, (), cost, true) {
                Reach::New(id) | Reach::Improved(id) => open.push(id, space.g(id), space.g(id)),
                Reach::Stale => {}
            }
        }
    }

    Tree {
        space,
        order,
        ids: Vec::new(),
    }
}

/// The landmark tables while [`Landmarks::build`] picks landmarks.
struct Builder<'a, G, S, C> {
    /// The graph.
    graph: &'a G,
    /// Every state found so far.
    table: StateTable<S>,
    /// The landmarks picked so far, by id.
    landmarks: Vec<usize>,
    /// Per landmark, the distances by state id; ids beyond the end have no path.
    columns: Vec<Vec<Distances<C>>>,
    /// Per state id, the round-trip distance to the nearest landmark (to the start before
    /// the first landmark).
    nearest: Vec<Extended<C>>,
    /// Per state id, whether it is a landmark.
    is_landmark: Vec<bool>,
    /// The forward shortest-path tree from the start, for avoidance. Set by
    /// [`Builder::new`].
    start_tree: Option<Tree<S, C>>,
}

impl<'a, G, S, C> Builder<'a, G, S, C>
where
    G: BidirectionalGraph + Successors<State = S, Cost = C>,
    S: Clone + Hash + Eq,
    C: CostExt,
{
    /// Searches both ways from `start` to find the first states and their distance to it.
    fn new(graph: &'a G, start: S) -> Self {
        let mut builder = Self {
            graph,
            table: StateTable::new(),
            landmarks: Vec::new(),
            columns: Vec::new(),
            nearest: Vec::new(),
            is_landmark: Vec::new(),
            start_tree: None,
        };

        let (forward, backward) = builder.explore(start);
        let column = builder.column(&forward, &backward);
        builder.nearest = column.iter().map(round_trip).collect();
        builder.start_tree = Some(forward);

        builder
    }

    /// Runs both searches from `source` and interns every state they find.
    fn explore(&mut self, source: S) -> (Tree<S, C>, Tree<S, C>) {
        let mut forward = settle(self.graph, source.clone(), false);
        let mut backward = settle(self.graph, source, true);
        for tree in [&mut forward, &mut backward] {
            tree.ids = (0..tree.space.len())
                .map(|node| self.table.intern(tree.space.state(node)))
                .collect();
        }

        let len = self.table.states.len();
        self.nearest.resize(len, Extended::Infinite);
        self.is_landmark.resize(len, false);

        (forward, backward)
    }

    /// The distances to and from the source of `forward` and `backward`, by state id.
    fn column(&self, forward: &Tree<S, C>, backward: &Tree<S, C>) -> Vec<Distances<C>> {
        let mut column = vec![Distances::NONE; self.table.states.len()];
        for node in 0..forward.space.len() {
            column[forward.ids[node]].from_landmark = Some(forward.space.g(node));
        }
        for node in 0..backward.space.len() {
            column[backward.ids[node]].to_landmark = Some(backward.space.g(node));
        }

        column
    }

    /// Makes state `id` the next landmark and records its distances.
    fn add_landmark(&mut self, id: usize) {
        let (forward, backward) = self.explore(self.table.states[id].clone());
        let column = self.column(&forward, &backward);

        let first = self.landmarks.is_empty();
        for (nearest, distances) in self.nearest.iter_mut().zip(&column) {
            let trip = round_trip(distances);
            *nearest = if first { trip } else { (*nearest).min(trip) };
        }

        self.is_landmark[id] = true;
        self.landmarks.push(id);
        self.columns.push(column);
    }

    /// The state farthest from the landmarks (or the start) that is not a landmark yet;
    /// the first one among equals. `None` if every state is a landmark.
    fn farthest(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for id in (0..self.nearest.len()).filter(|&id| !self.is_landmark[id]) {
            if best.is_none_or(|best| self.nearest[id] > self.nearest[best]) {
                best = Some(id);
            }
        }

        best
    }

    /// The avoidance choice in the start's shortest-path tree, or `None` if no subtree
    /// without a landmark has a bound below its distance.
    fn avoidance(&self) -> Option<usize> {
        let tree = self.start_tree.as_ref()?;
        let root = tree.ids[0];
        let len = tree.space.len();
        let mut size = vec![C::ZERO; len];
        let mut has_landmark = vec![false; len];
        let mut heaviest_child: Vec<Option<NodeId>> = vec![None; len];

        for &node in tree.order.iter().rev() {
            let id = tree.ids[node];
            has_landmark[node] |= self.is_landmark[id];
            if has_landmark[node] {
                size[node] = C::ZERO;
            } else {
                let bound = self.bound(root, id).unwrap_or(C::ZERO);
                let weight = tree.space.g(node).saturating_sub(bound);
                size[node] = size[node].saturating_add(weight);
            }

            if let Some((parent, (), _)) = tree.space.parent(node) {
                has_landmark[parent] |= has_landmark[node];
                size[parent] = size[parent].saturating_add(size[node]);
                if heaviest_child[parent].is_none_or(|child| size[node] > size[child]) {
                    heaviest_child[parent] = Some(node);
                }
            }
        }

        let mut node = (0..len).fold(None, |best: Option<NodeId>, node| match best {
            Some(best) if size[best] >= size[node] => Some(best),
            _ => Some(node),
        })?;
        if size[node] == C::ZERO {
            return None;
        }

        while let Some(child) = heaviest_child[node] {
            node = child;
        }

        Some(tree.ids[node])
    }

    /// The bound from state `from` to state `to` over the landmarks so far.
    fn bound(&self, from: usize, to: usize) -> Option<C> {
        let at =
            |column: &[Distances<C>], id: usize| column.get(id).copied().unwrap_or(Distances::NONE);

        bound(
            self.columns
                .iter()
                .map(|column| (at(column, from), at(column, to))),
        )
    }

    /// The finished tables, with the columns transposed into rows.
    fn finish(self) -> Landmarks<S, C> {
        let mut rows = Vec::with_capacity(self.table.states.len() * self.landmarks.len());
        for id in 0..self.table.states.len() {
            for column in &self.columns {
                rows.push(column.get(id).copied().unwrap_or(Distances::NONE));
            }
        }

        Landmarks {
            table: self.table,
            picked: self.landmarks,
            rows,
        }
    }
}

/// `d(l, v) + d(v, l)`, or infinite without a path in both directions.
fn round_trip<C: Cost>(distances: &Distances<C>) -> Extended<C> {
    match (distances.from_landmark, distances.to_landmark) {
        (Some(from), Some(to)) => Extended::Finite(from.saturating_add(to)),
        _ => Extended::Infinite,
    }
}
