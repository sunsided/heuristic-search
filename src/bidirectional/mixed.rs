//! Backward nodes of their own type: [`RegressionGraph`], and the adapters that let the
//! bidirectional core, which has one node type, search states forwards and nodes (such as
//! subgoals) backwards.
//!
//! The core runs over [`Mixed`] nodes: forward nodes are always [`Mixed::State`], backward
//! nodes always [`Mixed::Node`]. [`MixedGraph`] expands each variant in its own direction,
//! [`Both`] evaluates a heuristic on either variant, and [`Split`] hands each variant to the
//! matching half of a [`MeetRelation`]. [`unmix`] turns the result back into states.

use alloc::vec::Vec;

use search_core::Heuristic;

use crate::bidirectional::meet::{Direction, MeetRelation, Meets, ReplayError};
use crate::outcome::{Outcome, SearchResult, Solution};
use crate::problem::{Predecessors, Successors};

/// A graph searched forwards over its states and backwards over its own node type.
///
/// The backward [`Predecessors::Node`] may differ from the forward
/// [`Successors::State`]: planning regresses over partial states (subgoals). Actions and
/// costs are shared. Implemented for every graph with matching action and cost types; the
/// `_meet` searches need it together with a [`MeetRelation`] between states and nodes.
pub trait RegressionGraph:
    Successors + Predecessors<Action = <Self as Successors>::Action, Cost = <Self as Successors>::Cost>
{
}

impl<G> RegressionGraph for G where
    G: Successors
        + Predecessors<Action = <G as Successors>::Action, Cost = <G as Successors>::Cost>
{
}

/// A node of either side: a forward state or a backward node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Mixed<S, N> {
    /// A forward state.
    State(S),
    /// A backward node, such as a subgoal.
    Node(N),
}

/// `G` with forward states and backward nodes as one node type: states expand only along
/// successors, nodes only along predecessors.
pub(crate) struct MixedGraph<'g, G>(pub(crate) &'g G);

impl<G: RegressionGraph> Successors for MixedGraph<'_, G> {
    type State = Mixed<<G as Successors>::State, <G as Predecessors>::Node>;
    type Action = <G as Successors>::Action;
    type Cost = <G as Successors>::Cost;

    fn successors(
        &self,
        state: &Self::State,
        mut emit: impl FnMut(Self::Action, Self::State, Self::Cost),
    ) {
        if let Mixed::State(state) = state {
            self.0.successors(state, |action, next, cost| {
                emit(action, Mixed::State(next), cost);
            });
        }
    }
}

impl<G: RegressionGraph> Predecessors for MixedGraph<'_, G> {
    type Node = Mixed<<G as Successors>::State, <G as Predecessors>::Node>;
    type Action = <G as Successors>::Action;
    type Cost = <G as Successors>::Cost;

    fn predecessors(
        &self,
        node: &Self::Node,
        mut emit: impl FnMut(Self::Action, Self::Node, Self::Cost),
    ) {
        if let Mixed::Node(node) = node {
            self.0.predecessors(node, |action, previous, cost| {
                emit(action, Mixed::Node(previous), cost);
            });
        }
    }
}

/// A heuristic over states and over nodes, as one heuristic over [`Mixed`] nodes.
pub(crate) struct Both<H>(pub(crate) H);

impl<S, N, C, H> Heuristic<Mixed<S, N>, C> for Both<H>
where
    H: Heuristic<S, C> + Heuristic<N, C>,
{
    fn estimate(&mut self, node: &Mixed<S, N>) -> Option<C> {
        match node {
            Mixed::State(state) => Heuristic::<S, C>::estimate(&mut self.0, state),
            Mixed::Node(node) => Heuristic::<N, C>::estimate(&mut self.0, node),
        }
    }
}

/// A [`MeetRelation`] between states and nodes, as the core's meet over [`Mixed`] nodes.
pub(crate) struct Split<M>(pub(crate) M);

impl<S, N, M: MeetRelation<S, N>> Meets<Mixed<S, N>> for Split<M> {
    fn insert(&mut self, _: Direction, id: usize, node: &Mixed<S, N>) {
        match node {
            Mixed::State(state) => self.0.insert_forward(id, state),
            Mixed::Node(node) => self.0.insert_backward(id, node),
        }
    }

    fn meets(&self, _: Direction, node: &Mixed<S, N>, emit: impl FnMut(usize)) {
        match node {
            Mixed::State(state) => self.0.meets_forward(state, emit),
            Mixed::Node(node) => self.0.meets_backward(node, emit),
        }
    }
}

/// The result over states, or the [`ReplayError`] if the solution still holds a backward
/// node, which means a replay step found no edge.
pub(crate) fn unmix<S: Clone, N, A: Clone, C>(
    result: SearchResult<Mixed<S, N>, A, C>,
) -> Result<SearchResult<S, A, C>, ReplayError<S, A>> {
    let outcome = match result.outcome {
        Outcome::Found(solution) => Outcome::Found(states_only(solution)?),
        Outcome::Unsolvable => Outcome::Unsolvable,
        Outcome::LimitReached => Outcome::LimitReached,
        Outcome::Cutoff => Outcome::Cutoff,
    };

    Ok(SearchResult {
        outcome,
        stats: result.stats,
    })
}

/// The solution's states, or the error at the first backward node in it.
fn states_only<S: Clone, N, A: Clone, C>(
    solution: Solution<Mixed<S, N>, A, C>,
) -> Result<Solution<S, A, C>, ReplayError<S, A>> {
    let mut states = Vec::with_capacity(solution.states.len());
    for node in solution.states {
        match node {
            Mixed::State(state) => states.push(state),
            Mixed::Node(_) => {
                let state = states
                    .last()
                    .cloned()
                    .expect("paths start at a forward state");
                let action = solution.actions[states.len() - 1].clone();

                return Err(ReplayError { state, action });
            }
        }
    }

    Ok(Solution {
        states,
        actions: solution.actions,
        cost: solution.cost,
    })
}
