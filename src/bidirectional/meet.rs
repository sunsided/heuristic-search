//! When the two frontiers of a bidirectional search meet: the [`MeetRelation`] capability,
//! the scanning relation [`MeetFn`], and the replay that turns a meet into a real path or
//! a [`ReplayError`].
//!
//! By default a forward state meets a backward node when they are equal, found through the
//! hash index both sides already keep. A [`MeetRelation`] replaces that with any relation,
//! such as "the state satisfies the subgoal" for regression over partial states, and owns
//! whatever index makes the relation fast (per-fact buckets, a trie, a plain list).

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use crate::best_first::GraphResult;
use crate::problem::Successors;

/// A direction of a bidirectional search, and so a side of its frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Direction {
    /// From the starts along successor edges.
    Forward = 0,
    /// From the goals along predecessor edges.
    Backward = 1,
}

impl Direction {
    /// The other direction.
    pub(crate) const fn opposite(self) -> Self {
        match self {
            Self::Forward => Self::Backward,
            Self::Backward => Self::Forward,
        }
    }

    /// The index of this direction: `0` forward, `1` backward.
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

/// Which forward states (`S`) meet which backward nodes (`N`) in a bidirectional search.
///
/// A forward state `s` *meets* a backward node `n` when a path from `s` to a goal is known
/// through `n`: the backward search reached `n` from a goal, so the actions along `n`'s
/// backward links lead from `s` to a goal. With plain states that is `s == n`, the default
/// of the searches without `_meet`. For regression over subgoals (partial states, possibly
/// of another type than states) it is "`s` satisfies `n`".
///
/// The search reports every node it creates, with a dense id per side (`0, 1, 2, ...` in
/// creation order), and asks for the opposite-side ids that meet a new or improved node.
/// The relation keeps its own index of the inserted nodes (per-fact buckets, a trie, a
/// list).
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
/// Equality, indexed by the lowest bit:
///
/// ```
/// use heuristic_search::MeetRelation;
///
/// /// `[side][bit]` lists `(id, state)`; side 0 is forward.
/// #[derive(Default)]
/// struct Buckets([[Vec<(usize, u32)>; 2]; 2]);
///
/// impl Buckets {
///     /// The ids in `side`'s bucket of `state` that equal it.
///     fn equal(&self, side: usize, state: u32, mut emit: impl FnMut(usize)) {
///         for &(id, other) in &self.0[side][(state & 1) as usize] {
///             if other == state {
///                 emit(id);
///             }
///         }
///     }
/// }
///
/// impl MeetRelation<u32> for Buckets {
///     fn insert_forward(&mut self, id: usize, state: &u32) {
///         self.0[0][(*state & 1) as usize].push((id, *state));
///     }
///
///     fn insert_backward(&mut self, id: usize, node: &u32) {
///         self.0[1][(*node & 1) as usize].push((id, *node));
///     }
///
///     fn meets_forward(&self, state: &u32, emit: impl FnMut(usize)) {
///         self.equal(1, *state, emit);
///     }
///
///     fn meets_backward(&self, node: &u32, emit: impl FnMut(usize)) {
///         self.equal(0, *node, emit);
///     }
/// }
///
/// let mut buckets = Buckets::default();
/// buckets.insert_backward(0, &6);
/// let mut met = Vec::new();
/// buckets.meets_forward(&6, |id| met.push(id));
/// assert_eq!(met, [0]);
/// ```
pub trait MeetRelation<S, N = S> {
    /// Records the new forward node `id` with `state`.
    fn insert_forward(&mut self, id: usize, state: &S);

    /// Records the new backward node `id` with `node`.
    fn insert_backward(&mut self, id: usize, node: &N);

    /// Calls `emit(id)` for every backward node that `state` meets.
    fn meets_forward(&self, state: &S, emit: impl FnMut(usize));

    /// Calls `emit(id)` for every forward state that meets `node`.
    fn meets_backward(&self, node: &N, emit: impl FnMut(usize));
}

impl<S, N, M: MeetRelation<S, N> + ?Sized> MeetRelation<S, N> for &mut M {
    fn insert_forward(&mut self, id: usize, state: &S) {
        (**self).insert_forward(id, state);
    }

    fn insert_backward(&mut self, id: usize, node: &N) {
        (**self).insert_backward(id, node);
    }

    fn meets_forward(&self, state: &S, emit: impl FnMut(usize)) {
        (**self).meets_forward(state, emit);
    }

    fn meets_backward(&self, node: &N, emit: impl FnMut(usize)) {
        (**self).meets_backward(node, emit);
    }
}

/// A [`MeetRelation`] from a closure `meets(forward_state, backward_node) -> bool`, checked
/// against every node of the opposite side; created by [`meet_fn`].
///
/// Each check scans the whole opposite side, so a search costs time quadratic in the number
/// of nodes. Fine for small problems and tests; implement [`MeetRelation`] with an index
/// for large ones.
#[derive(Debug, Clone)]
pub struct MeetFn<F, S, N> {
    /// The relation, forward state first.
    relation: F,
    /// The forward states as `(id, state)`.
    forward: Vec<(usize, S)>,
    /// The backward nodes as `(id, node)`.
    backward: Vec<(usize, N)>,
}

/// Turns a closure `Fn(&S, &N) -> bool` (forward state, backward node) into a scanning
/// [`MeetRelation`].
///
/// # Examples
///
/// ```
/// use heuristic_search::{MeetRelation, meet_fn};
///
/// // A state (a bit set of facts) meets a subgoal it contains.
/// let mut satisfies = meet_fn(|state: &u8, subgoal: &u8| state & subgoal == *subgoal);
/// satisfies.insert_backward(0, &0b011);
/// satisfies.insert_backward(1, &0b100);
///
/// let mut met = Vec::new();
/// satisfies.meets_forward(&0b111, |id| met.push(id));
/// assert_eq!(met, [0, 1]);
/// ```
pub fn meet_fn<S, N, F: Fn(&S, &N) -> bool>(relation: F) -> MeetFn<F, S, N> {
    MeetFn {
        relation,
        forward: Vec::new(),
        backward: Vec::new(),
    }
}

impl<S: Clone, N: Clone, F: Fn(&S, &N) -> bool> MeetRelation<S, N> for MeetFn<F, S, N> {
    fn insert_forward(&mut self, id: usize, state: &S) {
        self.forward.push((id, state.clone()));
    }

    fn insert_backward(&mut self, id: usize, node: &N) {
        self.backward.push((id, node.clone()));
    }

    fn meets_forward(&self, state: &S, mut emit: impl FnMut(usize)) {
        for (id, node) in &self.backward {
            if (self.relation)(state, node) {
                emit(*id);
            }
        }
    }

    fn meets_backward(&self, node: &N, mut emit: impl FnMut(usize)) {
        for (id, state) in &self.forward {
            if (self.relation)(state, node) {
                emit(*id);
            }
        }
    }
}

/// A `_meet` search could not turn its meet into a path of real states: `state` has no
/// successor edge labelled `action`, although the backward search used that action.
///
/// The meet relation or the [`Predecessors`](crate::Predecessors) disagree with the
/// [`Successors`]: regression produced a subgoal that the forward model cannot follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError<S, A> {
    /// The real state the replay had reached.
    pub state: S,
    /// The backward action it could not apply to `state`.
    pub action: A,
}

impl<S, A> fmt::Display for ReplayError<S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a backward action has no matching successor edge from the meeting path")
    }
}

impl<S: fmt::Debug, A: fmt::Debug> core::error::Error for ReplayError<S, A> {}

/// What a `_meet` search returns: the search result, or the [`ReplayError`] if its
/// solution could not be replayed as real states.
pub type MeetResult<G> =
    Result<GraphResult<G>, ReplayError<<G as Successors>::State, <G as Successors>::Action>>;

/// The internal, direction-based meet the bidirectional core asks; one node type for both
/// sides. [`MeetRelation`]s reach it through `Split` over mixed nodes.
pub(crate) trait Meets<T> {
    /// Records the new node `id` with `node` on `direction`'s side.
    fn insert(&mut self, direction: Direction, id: usize, node: &T);

    /// Calls `emit(id)` for every node on the opposite side that meets `node`.
    fn meets(&self, direction: Direction, node: &T, emit: impl FnMut(usize));
}

/// The default meet: equality through the sides' own hash index. Never asked.
pub(crate) struct Equality;

impl<S> Meets<S> for Equality {
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
