//! Hierarchical A* over a region graph (HPA*, Botea, Müller and Schaeffer 2004): a
//! [`Hierarchy`] built once over a graph split into [`Regions`], and the query
//! [`hierarchical_a_star`], which plans on the small abstract graph and refines each
//! abstract edge with A* inside one region.
//!
//! The abstract graph has one node per *entrance*, a state with an edge into another
//! region (or the target of one). Its edges are the edges between entrances of different
//! regions, at their cost, and for every two entrances of one region the cost of the
//! cheapest path between them that stays inside the region. A query inserts the start and
//! goal states, connects them to the entrances of their regions (and to each other when
//! they share one), runs A* on the result and replaces every abstract edge by its path in
//! the graph.
//!
//! ```mermaid
//! flowchart LR
//!     build[Hierarchy: entrances and intra-region costs] --> insert[insert start and goal]
//!     insert --> abstract[A* on the abstract graph]
//!     abstract --> refine[A* per abstract edge, inside its region]
//!     refine --> path[concatenated path]
//! ```

mod query;

use alloc::vec::Vec;
use core::hash::Hash;

use hashbrown::HashMap;
use search_core::{Budget, Cost, heuristic_fn};

pub use query::{hierarchical_a_star, hierarchical_a_star_multi};
#[cfg(feature = "parallel")]
pub use query::{hierarchical_a_star_par, hierarchical_a_star_par_multi};

use crate::best_first::{GraphResult, a_star};
use crate::grid::{EightConnected, GridCell, Octile, UniformGrid, is_open};
use crate::outcome::Stats;
use crate::problem::Successors;

/// Assigns every state of a graph to a region, such as a room of a floor plan or a square
/// cluster of a grid.
///
/// Closures `Fn(&S) -> usize` are region maps. The assignment must not change while a
/// [`Hierarchy`] built from it is in use.
pub trait Regions<S> {
    /// The id of the region `state` belongs to.
    fn region(&self, state: &S) -> usize;
}

impl<S, F: Fn(&S) -> usize> Regions<S> for F {
    fn region(&self, state: &S) -> usize {
        self(state)
    }
}

/// The square clusters of a grid that [`Hierarchy::grid`] splits it into: cell `(x, y)`
/// lies in cluster `(x / size, y / size)`, numbered row by row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridClusters {
    /// The side length of a cluster in cells.
    size: usize,
    /// The number of clusters per row of the grid.
    columns: usize,
}

impl GridClusters {
    /// The side length of a cluster in cells.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl Regions<GridCell> for GridClusters {
    fn region(&self, &(x, y): &GridCell) -> usize {
        (y / self.size) * self.columns + x / self.size
    }
}

/// The abstract graph of a graph split into regions, built once and queried with
/// [`hierarchical_a_star`] for many start and goal states.
///
/// [`Hierarchy::new`] takes a region map and makes every state with an edge into another
/// region an entrance; paths are then optimal. [`Hierarchy::grid`] builds the HPA*
/// abstraction of a [`UniformGrid`]: square clusters with a few entrance cells per
/// border, which keeps the abstract graph small at a small cost in path quality.
///
/// # Examples
///
/// ```
/// use heuristic_search::{EightConnected, Hierarchy, Limits, Octile, UniformGrid, hierarchical_a_star};
///
/// /// Two 3 x 3 rooms side by side, joined by a door at `(3, 1)`.
/// struct Rooms;
///
/// impl UniformGrid for Rooms {
///     fn width(&self) -> usize {
///         7
///     }
///
///     fn height(&self) -> usize {
///         3
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         x != 3 || y == 1
///     }
/// }
///
/// let distance = |a: &(usize, usize), b: &(usize, usize)| Octile::between(*a, *b);
/// let room = |&(x, _): &(usize, usize)| usize::from(x >= 3);
/// let cells = (0..7).flat_map(|x| (0..3).map(move |y| (x, y)));
/// let hierarchy = Hierarchy::new(&EightConnected(Rooms), room, cells, distance);
///
/// let result = hierarchical_a_star(&EightConnected(Rooms), &hierarchy, (0, 0), (6, 2), distance, Limits::NONE);
/// let solution = result.solution().expect("the door joins the rooms");
///
/// assert_eq!(solution.cost, Octile::new(4, 2));
/// assert_eq!(solution.states[3], (3, 1));
/// ```
#[derive(Debug, Clone)]
pub struct Hierarchy<S, C, R> {
    /// The region map.
    regions: R,
    /// The entrance states; an entrance's node id indexes this.
    nodes: Vec<S>,
    /// The node id of every entrance state.
    index: HashMap<S, usize>,
    /// The entrances of every region, in node id order.
    members: HashMap<usize, Vec<usize>>,
    /// The abstract edges `(target, cost)` out of every node.
    edges: Vec<Vec<(usize, C)>>,
}

impl<S, C, R> Hierarchy<S, C, R>
where
    S: Clone + Hash + Eq,
    C: Cost,
    R: Regions<S>,
{
    /// Builds the abstract graph of `graph` over the region map `regions`.
    ///
    /// `states` must include every state with an edge into another region; the targets of
    /// those edges become entrances too. Passing every state of the graph is the simple
    /// choice. Abstract edges between two entrances of one region come from A* inside the
    /// region, guided by `distance`, an admissible estimate of the cost between two states
    /// (`|_, _| 0` gives Dijkstra's algorithm).
    ///
    /// Every edge between regions is kept, so [`hierarchical_a_star`] finds a cheapest path
    /// whenever the graph has a path. Building runs one A* per pair of entrances of a
    /// region, so regions should meet through few states (doors rather than open borders).
    pub fn new<G, D>(
        graph: &G,
        regions: R,
        states: impl IntoIterator<Item = S>,
        distance: D,
    ) -> Self
    where
        G: Successors<State = S, Cost = C>,
        G::Action: Clone,
        D: Fn(&S, &S) -> C,
    {
        let mut entrances = Vec::new();
        for state in states {
            let region = regions.region(&state);
            let before = entrances.len();
            graph.successors(&state, |_, next, _| {
                if regions.region(&next) != region {
                    entrances.push(next);
                }
            });
            if entrances.len() > before {
                entrances.push(state);
            }
        }

        Self::from_entrances(graph, regions, entrances, &distance)
    }

    /// Builds the hierarchy with the given entrances: abstract edges between entrances of
    /// different regions from `graph`, and within a region from A* inside it.
    fn from_entrances<G, D>(graph: &G, regions: R, entrances: Vec<S>, distance: &D) -> Self
    where
        G: Successors<State = S, Cost = C>,
        G::Action: Clone,
        D: Fn(&S, &S) -> C,
    {
        let mut hierarchy = Self {
            regions,
            nodes: Vec::new(),
            index: HashMap::new(),
            members: HashMap::new(),
            edges: Vec::new(),
        };
        for state in entrances {
            if !hierarchy.index.contains_key(&state) {
                let id = hierarchy.nodes.len();
                let region = hierarchy.regions.region(&state);
                hierarchy.members.entry(region).or_default().push(id);
                hierarchy.index.insert(state.clone(), id);
                hierarchy.nodes.push(state);
            }
        }

        for from in 0..hierarchy.nodes.len() {
            let edges = hierarchy.edges_from(graph, from, distance);
            hierarchy.edges.push(edges);
        }

        hierarchy
    }

    /// The abstract edges out of node `from`: to entrances of other regions along graph
    /// edges, and to the other entrances of its region along the cheapest path inside it.
    fn edges_from<G, D>(&self, graph: &G, from: usize, distance: &D) -> Vec<(usize, C)>
    where
        G: Successors<State = S, Cost = C>,
        G::Action: Clone,
        D: Fn(&S, &S) -> C,
    {
        let state = &self.nodes[from];
        let region = self.regions.region(state);
        let mut edges = Vec::new();
        graph.successors(state, |_, next, cost| {
            if self.regions.region(&next) != region {
                if let Some(&to) = self.index.get(&next) {
                    edges.push((to, cost));
                }
            }
        });

        for &to in &self.members[&region] {
            if to == from {
                continue;
            }

            let local = self.local(graph, state, &self.nodes[to], distance, |_: &Stats| false);
            if let Some(solution) = local.into_solution() {
                edges.push((to, solution.cost));
            }
        }

        edges
    }

    /// A* from `from` to `to` inside the region of `from`.
    pub(crate) fn local<G, D>(
        &self,
        graph: &G,
        from: &S,
        to: &S,
        distance: &D,
        budget: impl Budget<Stats>,
    ) -> GraphResult<G>
    where
        G: Successors<State = S, Cost = C>,
        G::Action: Clone,
        D: Fn(&S, &S) -> C,
    {
        let within = Within {
            graph,
            regions: &self.regions,
            region: self.regions.region(from),
        };
        let h = heuristic_fn(|state: &S| Some(distance(state, to)));

        a_star(&within, from.clone(), &|state: &S| state == to, h, budget)
    }
}

impl Hierarchy<GridCell, Octile, GridClusters> {
    /// Builds the HPA* abstraction of `grid` for searches over [`EightConnected`]: square
    /// clusters of `size` by `size` cells (smaller at the right and bottom edges).
    ///
    /// Along each border between two clusters, every maximal run of cell pairs that are
    /// passable on both sides is an *entrance*. A run shorter than six pairs gets one
    /// transition, a pair of entrance cells, in its middle; a longer one gets two, at its
    /// ends. Since a run is a straight line of passable cells on either side, and every
    /// diagonal move across a border has a straight pair beside it, the transitions keep
    /// every connection between clusters: [`hierarchical_a_star`] finds a path whenever
    /// [`a_star`] does. Paths through the transitions may cost more than the
    /// cheapest one.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    #[must_use]
    pub fn grid<G: UniformGrid>(grid: &G, size: usize) -> Self {
        assert!(size > 0, "clusters need at least one cell");
        let clusters = GridClusters {
            size,
            columns: grid.width().div_ceil(size),
        };
        let mut entrances = Vec::new();
        for x in (size..grid.width()).step_by(size) {
            transitions(
                grid,
                size,
                |y| ((x - 1, y), (x, y)),
                grid.height(),
                &mut entrances,
            );
        }
        for y in (size..grid.height()).step_by(size) {
            transitions(
                grid,
                size,
                |x| ((x, y - 1), (x, y)),
                grid.width(),
                &mut entrances,
            );
        }

        let distance = |a: &GridCell, b: &GridCell| Octile::between(*a, *b);

        Self::from_entrances(&EightConnected(grid), clusters, entrances, &distance)
    }
}

/// Pushes the transitions of one line of cluster borders onto `entrances`: `pair(i)` is the
/// pair of cells facing each other across the border at position `i < length`, and runs
/// are split where a cluster ends every `size` positions.
fn transitions<G: UniformGrid>(
    grid: &G,
    size: usize,
    pair: impl Fn(usize) -> (GridCell, GridCell),
    length: usize,
    entrances: &mut Vec<GridCell>,
) {
    /// Runs at least this long get a transition at each end instead of one in the middle.
    const WIDE: usize = 6;

    let open = |i: usize| {
        let (a, b) = pair(i);
        is_open(grid, a) && is_open(grid, b)
    };
    let mut push = |i: usize| {
        let (a, b) = pair(i);
        entrances.push(a);
        entrances.push(b);
    };

    let mut i = 0;
    while i < length {
        if !open(i) {
            i += 1;
            continue;
        }

        let cluster_end = (i / size + 1) * size;
        let start = i;
        while i < length.min(cluster_end) && open(i) {
            i += 1;
        }

        if i - start >= WIDE {
            push(start);
            push(i - 1);
        } else {
            push(start + (i - start) / 2);
        }
    }
}

/// A graph restricted to one region: only the edges into states of that region.
struct Within<'a, G, R> {
    /// The full graph.
    graph: &'a G,
    /// The region map.
    regions: &'a R,
    /// The region the search stays in.
    region: usize,
}

impl<G: Successors, R: Regions<G::State>> Successors for Within<'_, G, R> {
    type State = G::State;
    type Action = G::Action;
    type Cost = G::Cost;

    fn successors(&self, state: &G::State, mut emit: impl FnMut(G::Action, G::State, G::Cost)) {
        self.graph.successors(state, |action, next, cost| {
            if self.regions.region(&next) == self.region {
                emit(action, next, cost);
            }
        });
    }
}
