//! The open list of best-first search ([`OpenList`]): a binary min-heap with lazy deletion.

use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use crate::cost::Cost;
use crate::space::NodeId;

/// One open-list entry.
///
/// Ordered by smallest [`Entry::key`], then largest [`Entry::g`] (deeper nodes first, which
/// usually reaches the goal sooner among equal keys), then smallest [`Entry::seq`] (first
/// in, first out), so the expansion order is deterministic.
struct Entry<C> {
    /// The priority, such as `f = g + h` for A*.
    key: C,
    /// The path cost of the node when this entry was pushed. An entry whose `g` no longer
    /// matches the node's best `g` is outdated and skipped.
    g: C,
    /// Push counter; breaks the remaining ties.
    seq: u64,
    /// The node this entry stands for.
    node: NodeId,
}

impl<C: Cost> Entry<C> {
    /// Ordering where the entry to pop first is the *greatest*, as [`BinaryHeap`] wants.
    fn priority_cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp_cost(&self.key)
            .then_with(|| self.g.cmp_cost(&other.g))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl<C: Cost> PartialEq for Entry<C> {
    fn eq(&self, other: &Self) -> bool {
        self.priority_cmp(other).is_eq()
    }
}

impl<C: Cost> Eq for Entry<C> {}

impl<C: Cost> PartialOrd for Entry<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Cost> Ord for Entry<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_cmp(other)
    }
}

/// A min-heap of nodes keyed by a cost.
///
/// Decrease-key is done by pushing a new entry; the outdated one stays in the heap and is
/// recognised on pop because its `g` differs from the node's current `g`.
pub(crate) struct OpenList<C> {
    /// The entries.
    heap: BinaryHeap<Entry<C>>,
    /// Number of pushes so far; the next entry's [`Entry::seq`].
    pushes: u64,
}

impl<C: Cost> OpenList<C> {
    /// Creates an empty open list.
    pub(crate) fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            pushes: 0,
        }
    }

    /// Opens `node` with priority `key` and current path cost `g`.
    pub(crate) fn push(&mut self, node: NodeId, key: C, g: C) {
        self.heap.push(Entry {
            key,
            g,
            seq: self.pushes,
            node,
        });
        self.pushes += 1;
    }

    /// Pops the entry with the smallest key whose `g` still equals `current_g(node)`.
    ///
    /// Outdated entries are dropped on the way. Returns `None` once the heap is empty.
    pub(crate) fn pop_live(&mut self, current_g: impl Fn(NodeId) -> C) -> Option<NodeId> {
        while let Some(entry) = self.heap.pop() {
            if entry.g.cmp_cost(&current_g(entry.node)).is_eq() {
                return Some(entry.node);
            }
        }

        None
    }
}
