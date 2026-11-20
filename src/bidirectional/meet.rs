//! When the two frontiers of a bidirectional search meet: the [`MeetRelation`] capability,
//! the scanning relation [`MeetFn`], and the replay that turns a meet into a real path.
//!
//! By default a forward state meets a backward node when they are equal, found through the
//! hash index both sides already keep. A [`MeetRelation`] replaces that with any relation,
//! such as "the state satisfies the subgoal" for regression over partial states, and owns
//! whatever index makes the relation fast (per-fact buckets, a trie, a plain list).

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::problem::Successors;

/// A direction of a bidirectional search, and so a side of its frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// From the starts along successor edges.
    Forward = 0,
    /// From the goals along predecessor edges.
    Backward = 1,
}

impl Direction {
    /// The other direction.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Forward => Self::Backward,
            Self::Backward => Self::Forward,
        }
    }

    /// The index of this direction: `0` forward, `1` backward.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Which nodes of the two sides of a bidirectional search meet.
///
/// A forward state `s` *meets* a backward node `n` when a path from `s` to a goal is known
/// through `n`: the backward search reached `n` from a goal, so the actions along `n`'s
/// backward links lead from `s` to a goal. With plain states that is `s == n`, the default.
/// For regression over subgoals (partial states) it is "`s` satisfies `n`".
///
/// The search reports every node it creates through [`MeetRelation::insert`], with a
/// dense id per side (`0, 1, 2, ...` in creation order), and asks
/// [`MeetRelation::meets`] for the opposite-side nodes that meet a new or improved node.
/// The relation keeps its own index of the inserted nodes.
///
/// # Contract
///
/// The stopping rules of the searches assume every start-goal path has a point where a
/// forward state meets a backward node: for regression, the subgoal obtained by regressing
/// a goal along the rest of the path is satisfied by the state on the path. Then the
/// returned solutions keep the guarantees of plain bidirectional search.
///
/// # Examples
///
/// A relation that buckets backward nodes by their lowest bit and meets by equality:
///
/// ```
/// use heuristic_search::{Direction, MeetRelation};
///
/// /// Equality, indexed per side by the state's lowest bit.
/// #[derive(Default)]
/// struct Buckets {
///     /// `[side][bit]` lists `(id, state)`.
///     nodes: [[Vec<(usize, u32)>; 2]; 2],
/// }
///
/// impl MeetRelation<u32> for Buckets {
///     fn insert(&mut self, direction: Direction, id: usize, state: &u32) {
///         self.nodes[direction.index()][(*state & 1) as usize].push((id, *state));
///     }
///
///     fn meets(&self, direction: Direction, state: &u32, mut emit: impl FnMut(usize)) {
///         let bucket = &self.nodes[direction.opposite().index()][(*state & 1) as usize];
///         for &(id, other) in bucket {
///             if other == *state {
///                 emit(id);
///             }
///         }
///     }
/// }
///
/// let mut buckets = Buckets::default();
/// buckets.insert(Direction::Backward, 0, &6);
/// let mut met = Vec::new();
/// buckets.meets(Direction::Forward, &6, |id| met.push(id));
/// assert_eq!(met, [0]);
/// ```
pub trait MeetRelation<S> {
    /// Records the new node `id` with `state` on `direction`'s side.
    fn insert(&mut self, direction: Direction, id: usize, state: &S);

    /// Calls `emit(id)` for every node on the opposite side that meets `state`, a node on
    /// `direction`'s side.
    fn meets(&self, direction: Direction, state: &S, emit: impl FnMut(usize));
}

impl<S, M: MeetRelation<S> + ?Sized> MeetRelation<S> for &mut M {
    fn insert(&mut self, direction: Direction, id: usize, state: &S) {
        (**self).insert(direction, id, state);
    }

    fn meets(&self, direction: Direction, state: &S, emit: impl FnMut(usize)) {
        (**self).meets(direction, state, emit);
    }
}

/// A [`MeetRelation`] from a closure `meets(forward_state, backward_node) -> bool`, checked
/// against every node of the opposite side; created by [`meet_fn`].
///
/// Each check scans the whole opposite side, so a search costs time quadratic in the number
/// of nodes. Fine for small problems and tests; implement [`MeetRelation`] with an index
/// for large ones.
#[derive(Debug, Clone)]
pub struct MeetFn<F, S> {
    /// The relation, forward state first.
    relation: F,
    /// The nodes of each side as `(id, state)`, indexed by [`Direction::index`].
    nodes: [Vec<(usize, S)>; 2],
}

/// Turns a closure `Fn(&S, &S) -> bool` (forward state, backward node) into a scanning
/// [`MeetRelation`].
///
/// # Examples
///
/// ```
/// use heuristic_search::{Direction, MeetRelation, meet_fn};
///
/// // A state (a bit set of facts) meets a subgoal it contains.
/// let mut satisfies = meet_fn(|state: &u8, subgoal: &u8| state & subgoal == *subgoal);
/// satisfies.insert(Direction::Backward, 0, &0b011);
/// satisfies.insert(Direction::Backward, 1, &0b100);
///
/// let mut met = Vec::new();
/// satisfies.meets(Direction::Forward, &0b111, |id| met.push(id));
/// assert_eq!(met, [0, 1]);
/// ```
pub fn meet_fn<S, F: Fn(&S, &S) -> bool>(relation: F) -> MeetFn<F, S> {
    MeetFn {
        relation,
        nodes: [Vec::new(), Vec::new()],
    }
}

impl<S: Clone, F: Fn(&S, &S) -> bool> MeetRelation<S> for MeetFn<F, S> {
    fn insert(&mut self, direction: Direction, id: usize, state: &S) {
        self.nodes[direction.index()].push((id, state.clone()));
    }

    fn meets(&self, direction: Direction, state: &S, mut emit: impl FnMut(usize)) {
        for (id, other) in &self.nodes[direction.opposite().index()] {
            let met = match direction {
                Direction::Forward => (self.relation)(state, other),
                Direction::Backward => (self.relation)(other, state),
            };

            if met {
                emit(*id);
            }
        }
    }
}

/// The default meet: equality through the sides' own hash index. Never asked.
pub(crate) struct Equality;

impl<S> MeetRelation<S> for Equality {
    fn insert(&mut self, _: Direction, _: usize, _: &S) {}

    fn meets(&self, _: Direction, _: &S, _: impl FnMut(usize)) {}
}

/// Applies `action` to `state` forwards: the successor reached by an edge with that label
/// and its cost, or `None` if `state` has no such edge.
pub(crate) type ReplayFn<G> = fn(
    &G,
    &<G as Successors>::State,
    &<G as Successors>::Action,
) -> Option<(<G as Successors>::State, <G as Successors>::Cost)>;

/// A relation-based meet: the relation and the replay that rebuilds real states past the
/// meeting point.
pub(crate) struct Meeting<G: Successors, M> {
    /// Decides which nodes meet.
    pub(crate) relation: M,
    /// Replays backward actions forwards from the meeting state.
    pub(crate) replay: ReplayFn<G>,
    /// Ties the replay's graph type to the meeting.
    graph: PhantomData<fn(&G)>,
}

impl<G: Successors, M> Meeting<G, M> {
    /// A meeting by `relation`, replaying actions through the first successor edge with an
    /// equal label.
    pub(crate) fn new(relation: M) -> Self
    where
        G::Action: PartialEq,
        G::State: Clone,
    {
        Self {
            relation,
            replay: replay_step::<G>,
            graph: PhantomData,
        }
    }
}

/// The successor of `state` along the first edge labelled `action`, and its cost.
fn replay_step<G>(graph: &G, state: &G::State, action: &G::Action) -> Option<(G::State, G::Cost)>
where
    G: Successors,
    G::Action: PartialEq,
    G::State: Clone,
{
    let mut found = None;
    graph.successors(state, |label, next, cost| {
        if found.is_none() && label == *action {
            found = Some((next, cost));
        }
    });

    found
}
