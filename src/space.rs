//! The explicit search space ([`SearchSpace`]): every state seen so far, its best known path
//! cost and its parent link, for duplicate detection and solution extraction.

use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};

use search_core::{Cost, Goal};

use crate::outcome::Solution;

/// Index of a node in a [`SearchSpace`]. Stable for the lifetime of the space.
pub(crate) type NodeId = usize;

/// One state of the search space.
struct Node<S, A, C> {
    /// The state. Stored once; the hash index refers to it by [`NodeId`].
    state: S,
    /// Best known path cost from the start state.
    g: C,
    /// The node this one was reached from with its best known path, the edge label and the
    /// edge cost. `None` only for the start node.
    parent: Option<Parent<A, C>>,
}

/// A parent link: the parent node, the edge label and the edge cost.
type Parent<A, C> = (NodeId, A, C);

/// What [`SearchSpace::reach`] did with a newly generated edge target.
pub(crate) enum Reach {
    /// The state was not known before and got a fresh node.
    New(NodeId),
    /// The state was known, and the new path is cheaper; its cost and parent were updated.
    Improved(NodeId),
    /// The state was known, and the new path is not cheaper (or improvements are ignored).
    Stale,
}

/// All states seen by one search, stored in an arena and indexed by hash.
///
/// States are stored once. The hash table holds [`NodeId`]s and hashes the state they point
/// to, so lookups need no second copy of a state.
pub(crate) struct SearchSpace<S, A, C> {
    /// The nodes in creation order. The start node is `0`.
    nodes: Vec<Node<S, A, C>>,
    /// Hash index from state to node.
    index: HashTable<NodeId>,
    /// Hashes states for [`SearchSpace::index`].
    hasher: DefaultHashBuilder,
}

impl<S: Hash + Eq, A, C: Cost> SearchSpace<S, A, C> {
    /// Creates a space from the start states, each at cost zero.
    ///
    /// Repeated starts are stored once, so the starts are nodes `0..space.len()` in order
    /// of first appearance. Stops early and returns the node of the first start that
    /// satisfies `goal`, if any.
    pub(crate) fn seed<T: Goal<S> + ?Sized>(
        starts: impl IntoIterator<Item = S>,
        goal: &T,
    ) -> (Self, Option<NodeId>) {
        let mut space = Self {
            nodes: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
        };

        for start in starts {
            if space.find(&start).is_some() {
                continue;
            }

            let id = space.insert(start, C::ZERO, None);
            if goal.is_goal(space.state(id)) {
                return (space, Some(id));
            }
        }

        (space, None)
    }

    /// The number of nodes in the space.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The state of node `id`.
    pub(crate) fn state(&self, id: NodeId) -> &S {
        &self.nodes[id].state
    }

    /// The best known path cost of node `id`.
    pub(crate) fn g(&self, id: NodeId) -> C {
        self.nodes[id].g
    }

    /// The parent link of node `id`: the parent node, the edge label and the edge cost.
    /// `None` for start nodes.
    pub(crate) fn parent(&self, id: NodeId) -> Option<(NodeId, &A, C)> {
        self.nodes[id]
            .parent
            .as_ref()
            .map(|(parent, action, cost)| (*parent, action, *cost))
    }

    /// Records that `state` is reachable from node `parent` through `action` at edge cost
    /// `cost`.
    ///
    /// A known state is updated only if `improve` is set and the new path is strictly
    /// cheaper.
    pub(crate) fn reach(
        &mut self,
        state: S,
        parent: NodeId,
        action: A,
        cost: C,
        improve: bool,
    ) -> Reach {
        let g = self.nodes[parent].g + cost;
        let Some(id) = self.find(&state) else {
            return Reach::New(self.insert(state, g, Some((parent, action, cost))));
        };

        let node = &mut self.nodes[id];
        if !improve || g.cmp(&node.g).is_ge() {
            return Reach::Stale;
        }

        node.g = g;
        node.parent = Some((parent, action, cost));

        Reach::Improved(id)
    }

    /// Looks up the node of `state`.
    pub(crate) fn find(&self, state: &S) -> Option<NodeId> {
        let hash = self.hasher.hash_one(state);
        let nodes = &self.nodes;

        self.index
            .find(hash, |&id| nodes[id].state == *state)
            .copied()
    }

    /// Adds a node for `state`, which must not be known yet, and returns its id.
    fn insert(&mut self, state: S, g: C, parent: Option<Parent<A, C>>) -> NodeId {
        let id = self.nodes.len();
        let hash = self.hasher.hash_one(&state);
        let (nodes, hasher) = (&self.nodes, &self.hasher);

        self.index
            .insert_unique(hash, id, |&other| hasher.hash_one(&nodes[other].state));
        self.nodes.push(Node { state, g, parent });

        id
    }
}

impl<S: Clone, A: Clone, C: Cost> SearchSpace<S, A, C> {
    /// Follows the parent links from node `goal` back to the start and returns the path.
    ///
    /// The cost is summed along the links rather than read from the goal's `g`: with
    /// reopening, an ancestor may have been improved after the goal was reached.
    pub(crate) fn solution(&self, goal: NodeId) -> Solution<S, A, C> {
        let mut path = self.walk_to_root(goal);
        path.states.reverse();
        path.actions.reverse();

        path
    }

    /// The parent links from node `from` to its root, in walking order: `states[0]` is
    /// `from`, and `actions[i]` is the label of the link from `states[i]` to
    /// `states[i + 1]`. The cost is summed along the links.
    ///
    /// In a backward search, whose links point towards the goal, this is the forward path
    /// from `from` to the goal.
    pub(crate) fn walk_to_root(&self, from: NodeId) -> Solution<S, A, C> {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut cost = C::ZERO;
        let mut current = from;

        states.push(self.nodes[current].state.clone());
        while let Some((parent, action, edge_cost)) = &self.nodes[current].parent {
            actions.push(action.clone());
            cost = cost + *edge_cost;
            states.push(self.nodes[*parent].state.clone());
            current = *parent;
        }

        Solution {
            states,
            actions,
            cost,
        }
    }
}
