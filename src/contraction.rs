//! Contraction hierarchies (Geisberger, Sanders, Schultes and Delling 2008): a
//! [`ContractionHierarchy`] preprocesses a finite graph once and then answers exact
//! cheapest-path queries with a small bidirectional search.
//!
//! Preprocessing contracts the states one at a time, least important first, and adds a
//! shortcut `u -> w` whenever removing `v` would destroy the only cheapest `u -> v -> w`
//! path; a bounded witness search decides that. The position of a state in that order is
//! its rank. A query runs Dijkstra forwards from the start and backwards from the goal,
//! both relaxing only edges towards higher ranks, and unpacks the shortcuts of the
//! cheapest meeting into the graph's own edges.
//!
//! ```mermaid
//! flowchart LR
//!     build[build: discover states, contract by importance, add shortcuts]
//!     build --> query[query: upward Dijkstra from both ends]
//!     query --> meet[cheapest meeting]
//!     meet --> unpack[unpack shortcuts into original actions]
//! ```

mod build;
mod query;

use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};

use search_core::Cost;

use crate::outcome::SearchResult;
use crate::problem::Successors;

/// Index of an edge in [`ContractionHierarchy::edges`].
type EdgeId = usize;

/// The forward side of a query, and the index of the upward edges out of a state.
const FORWARD: usize = 0;

/// The backward side of a query, and the index of the upward edges into a state.
const BACKWARD: usize = 1;

/// How hard the witness searches of [`ContractionHierarchy::build_with_limits`] look for a
/// path that makes a shortcut unnecessary.
///
/// A witness search that hits a limit gives up and the shortcut is added, so smaller
/// limits build faster but add more (harmless) shortcuts; queries stay exact either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessLimits {
    /// Witness paths have at most this many edges.
    pub max_hops: u32,
    /// A witness search settles at most this many states.
    pub max_settled: usize,
}

impl WitnessLimits {
    /// The limits [`ContractionHierarchy::build`] uses: 16 hops and 64 settled states.
    pub const DEFAULT: Self = Self {
        max_hops: 16,
        max_settled: 64,
    };
}

impl Default for WitnessLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// How an edge of the hierarchy came about.
#[derive(Debug, Clone)]
enum Via<A> {
    /// An edge of the graph, with its action.
    Original(A),
    /// A shortcut for the two edges `u -> v` and `v -> w`; `v`, their shared endpoint, is
    /// the contracted middle state. Storing the edges rather than only `v` keeps the right
    /// action when the graph has parallel edges.
    Shortcut(EdgeId, EdgeId),
}

/// An edge of the hierarchy: an original edge or a shortcut.
#[derive(Debug, Clone)]
struct Edge<A, C> {
    /// The source state's index.
    from: usize,
    /// The target state's index.
    to: usize,
    /// The edge cost; for a shortcut the sum of the edges it stands for.
    cost: C,
    /// The original action or the two edges of a shortcut.
    via: Via<A>,
}

/// A contraction hierarchy over a finite graph with static edge costs, for many exact
/// cheapest-path queries between states of that graph.
///
/// [`build`](Self::build) copies the graph and preprocesses it once; queries no longer
/// touch the graph. Edge costs are fixed at build time: a changed cost needs a new
/// hierarchy. Self loops are dropped, and of parallel edges only the cheapest is kept
/// (the first one among equally cheap edges).
///
/// States get dense indices in discovery order. The hierarchy owns a copy of every state
/// and of the actions on the edges it keeps.
///
/// # Examples
///
/// ```
/// use heuristic_search::{ContractionHierarchy, Successors};
///
/// /// A directed cycle `0 -> 1 -> ... -> 5 -> 0`, every edge costing 1.
/// struct Cycle;
///
/// impl Successors for Cycle {
///     type State = u32;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
///         emit((), (n + 1) % 6, 1);
///     }
/// }
///
/// // Every state is reachable from 0, so one seed finds them all.
/// let hierarchy = ContractionHierarchy::build(&Cycle, [0]);
/// assert_eq!(hierarchy.len(), 6);
///
/// let result = hierarchy.query(4, 2);
/// let solution = result.solution().expect("the cycle connects every pair");
/// assert_eq!(solution.states, [4, 5, 0, 1, 2]);
/// assert_eq!(solution.cost, 4);
/// ```
#[derive(Debug, Clone)]
pub struct ContractionHierarchy<S, A, C> {
    /// Every state, by index.
    states: Vec<S>,
    /// Hash index from state to its index in [`ContractionHierarchy::states`].
    index: HashTable<usize>,
    /// Hashes states for [`ContractionHierarchy::index`].
    hasher: DefaultHashBuilder,
    /// The rank of each state: its position in the contraction order.
    rank: Vec<usize>,
    /// Every original edge kept and every shortcut. Shortcuts refer to earlier edges.
    edges: Vec<Edge<A, C>>,
    /// Per state, the upward edges: `up[FORWARD][v]` leave `v`, `up[BACKWARD][v]` enter it,
    /// and the other end has the higher rank.
    up: [Vec<Vec<EdgeId>>; 2],
}

impl<S, A, C> ContractionHierarchy<S, A, C>
where
    S: Clone + Hash + Eq,
    A: Clone,
    C: Cost,
{
    /// Builds the hierarchy of the part of `graph` reachable from `states`, with the
    /// default [`WitnessLimits`].
    ///
    /// The states are seeds: every state reachable from one of them through
    /// [`Successors`] is included, so passing every state of the graph covers it all, and
    /// a single seed suffices for a strongly connected graph. The graph must be finite
    /// from the seeds.
    ///
    /// See the [type documentation](Self) for an example.
    pub fn build<G>(graph: &G, states: impl IntoIterator<Item = S>) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
    {
        Self::build_with_limits(graph, states, WitnessLimits::DEFAULT)
    }

    /// [`build`](Self::build) with witness searches bounded by `limits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{ContractionHierarchy, Successors, WitnessLimits};
    ///
    /// /// A line `0 - 1 - 2 - 3` with both directions costing 1.
    /// struct Line;
    ///
    /// impl Successors for Line {
    ///     type State = u8;
    ///     type Action = i8;
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, n: &u8, mut emit: impl FnMut(i8, u8, u32)) {
    ///         if *n > 0 {
    ///             emit(-1, n - 1, 1);
    ///         }
    ///         if *n < 3 {
    ///             emit(1, n + 1, 1);
    ///         }
    ///     }
    /// }
    ///
    /// // Without any witness search every bypassed pair gets a shortcut.
    /// let none = WitnessLimits { max_hops: 0, max_settled: 0 };
    /// let hierarchy = ContractionHierarchy::build_with_limits(&Line, [0], none);
    /// let solution = hierarchy.query(0, 3).into_solution().expect("connected");
    ///
    /// assert_eq!(solution.actions, [1, 1, 1]);
    /// assert_eq!(solution.cost, 3);
    /// ```
    pub fn build_with_limits<G>(
        graph: &G,
        states: impl IntoIterator<Item = S>,
        limits: WitnessLimits,
    ) -> Self
    where
        G: Successors<State = S, Action = A, Cost = C>,
    {
        let mut hierarchy = Self {
            states: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
            rank: Vec::new(),
            edges: Vec::new(),
            up: [Vec::new(), Vec::new()],
        };
        let original = hierarchy.discover(graph, states);
        hierarchy.contract(original, limits);

        hierarchy
    }

    /// The cheapest path from `start` to `goal`.
    ///
    /// The result is [`Outcome::Found`](crate::Outcome::Found) with the graph's own
    /// actions, or [`Outcome::Unsolvable`](crate::Outcome::Unsolvable) if no path exists or
    /// a state other than a start that is the goal is not in the hierarchy. [`Stats`]
    /// count the states settled and the upward edges scanned by both sides.
    ///
    /// [`Stats`]: crate::Stats
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{ContractionHierarchy, Successors};
    ///
    /// /// `0 -> 1 -> 3` costs 2, `0 -> 2 -> 3` costs 5, and nothing leaves 3.
    /// struct Diamond;
    ///
    /// impl Successors for Diamond {
    ///     type State = u8;
    ///     type Action = char;
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, n: &u8, mut emit: impl FnMut(char, u8, u32)) {
    ///         match n {
    ///             0 => {
    ///                 emit('a', 1, 1);
    ///                 emit('b', 2, 1);
    ///             }
    ///             1 => emit('c', 3, 1),
    ///             2 => emit('d', 3, 4),
    ///             _ => {}
    ///         }
    ///     }
    /// }
    ///
    /// let hierarchy = ContractionHierarchy::build(&Diamond, [0]);
    /// let solution = hierarchy.query(0, 3).into_solution().expect("3 is reachable");
    /// assert_eq!(solution.actions, ['a', 'c']);
    /// assert_eq!(solution.cost, 2);
    ///
    /// assert!(hierarchy.query(3, 0).solution().is_none());
    /// ```
    pub fn query(&self, start: S, goal: S) -> SearchResult<S, A, C> {
        self.query_multi([start], [goal])
    }

    /// [`query`](Self::query) from several start states to several goal states at once:
    /// the cheapest path from any start to any goal.
    ///
    /// Every start and goal has path cost zero. States not in the hierarchy are ignored,
    /// except that a start which is also a goal is returned as a path without edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{ContractionHierarchy, Successors};
    ///
    /// /// Counting up by one at cost 1, from 0 to 9.
    /// struct Up;
    ///
    /// impl Successors for Up {
    ///     type State = u8;
    ///     type Action = ();
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, n: &u8, mut emit: impl FnMut((), u8, u32)) {
    ///         if *n < 9 {
    ///             emit((), n + 1, 1);
    ///         }
    ///     }
    /// }
    ///
    /// let hierarchy = ContractionHierarchy::build(&Up, [0]);
    /// let solution = hierarchy.query_multi([1, 4], [6, 8]).into_solution().expect("reachable");
    /// assert_eq!(solution.states, [4, 5, 6]);
    /// ```
    pub fn query_multi(
        &self,
        starts: impl IntoIterator<Item = S>,
        goals: impl IntoIterator<Item = S>,
    ) -> SearchResult<S, A, C> {
        query::run(self, starts, goals)
    }

    /// The number of states in the hierarchy.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` if the hierarchy has no states.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// The number of shortcuts preprocessing added.
    #[must_use]
    pub fn shortcut_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| matches!(edge.via, Via::Shortcut(..)))
            .count()
    }

    /// The rank of `state`, its position in the contraction order: `0` for the state
    /// contracted first (the least important), `len() - 1` for the last. `None` if the
    /// state is not in the hierarchy.
    #[must_use]
    pub fn rank(&self, state: &S) -> Option<usize> {
        self.find(state).map(|id| self.rank[id])
    }

    /// The forward upward search space of `state`: every state reachable from it through
    /// edges towards higher ranks, shortcuts included, with the cheapest cost of such an
    /// upward path, in the order a forward query from `state` would settle them (`state`
    /// first). Empty if the state is not in the hierarchy.
    ///
    /// A query from `state` settles a subset of these.
    ///
    /// # Examples
    ///
    /// ```
    /// use heuristic_search::{ContractionHierarchy, Successors};
    ///
    /// /// A directed cycle `0 -> 1 -> 2 -> 0`, every edge costing 1.
    /// struct Cycle;
    ///
    /// impl Successors for Cycle {
    ///     type State = u8;
    ///     type Action = ();
    ///     type Cost = u32;
    ///
    ///     fn successors(&self, n: &u8, mut emit: impl FnMut((), u8, u32)) {
    ///         emit((), (n + 1) % 3, 1);
    ///     }
    /// }
    ///
    /// let hierarchy = ContractionHierarchy::build(&Cycle, [0]);
    /// let top = (0..3).find(|n| hierarchy.rank(n) == Some(2)).expect("three ranks");
    ///
    /// // The most important state has no upward edges.
    /// assert_eq!(hierarchy.forward_search_space(&top), [(&top, 0)]);
    /// assert_eq!(hierarchy.backward_search_space(&top), [(&top, 0)]);
    /// ```
    #[must_use]
    pub fn forward_search_space(&self, state: &S) -> Vec<(&S, C)> {
        query::search_space(self, state, FORWARD)
    }

    /// The backward upward search space of `state`: every state from which `state` is
    /// reachable through edges coming down from higher ranks, with the cheapest cost of
    /// such a path, in the order a backward query from `state` would settle them (`state`
    /// first). Empty if the state is not in the hierarchy.
    ///
    /// See [`forward_search_space`](Self::forward_search_space) for an example.
    #[must_use]
    pub fn backward_search_space(&self, state: &S) -> Vec<(&S, C)> {
        query::search_space(self, state, BACKWARD)
    }

    /// The index of `state`, if it is in the hierarchy.
    fn find(&self, state: &S) -> Option<usize> {
        let hash = self.hasher.hash_one(state);
        let states = &self.states;

        self.index.find(hash, |&id| states[id] == *state).copied()
    }

    /// The index of `state`, adding it if it is new.
    fn intern(&mut self, state: S) -> usize {
        if let Some(id) = self.find(&state) {
            return id;
        }

        let id = self.states.len();
        let hash = self.hasher.hash_one(&state);
        let (states, hasher) = (&self.states, &self.hasher);
        self.index
            .insert_unique(hash, id, |&other| hasher.hash_one(&states[other]));
        self.states.push(state);

        id
    }
}
