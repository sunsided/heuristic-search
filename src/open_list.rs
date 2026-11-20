//! The lazy open list of best-first search ([`OpenList`]): a binary min-heap with lazy
//! deletion.

use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use crate::space::NodeId;

/// One open-list entry.
///
/// Ordered by smallest [`Entry::key`], then largest [`Entry::g`] (deeper nodes first, which
/// usually reaches the goal sooner among equal keys), then smallest [`Entry::seq`] (first
/// in, first out), so the expansion order is deterministic.
struct Entry<K, C> {
    /// The priority, such as `f = g + h` for A*.
    key: K,
    /// The path cost of the node when this entry was pushed. An entry whose `g` no longer
    /// matches the node's best `g` is outdated and skipped.
    g: C,
    /// Push counter; breaks the remaining ties.
    seq: u64,
    /// The node this entry stands for.
    node: NodeId,
}

impl<K: Ord, C: Ord> Entry<K, C> {
    /// Ordering where the entry to pop first is the *greatest*, as [`BinaryHeap`] wants.
    fn priority_cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| self.g.cmp(&other.g))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl<K: Ord, C: Ord> PartialEq for Entry<K, C> {
    fn eq(&self, other: &Self) -> bool {
        self.priority_cmp(other).is_eq()
    }
}

impl<K: Ord, C: Ord> Eq for Entry<K, C> {}

impl<K: Ord, C: Ord> PartialOrd for Entry<K, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, C: Ord> Ord for Entry<K, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_cmp(other)
    }
}

/// A min-heap of nodes keyed by `K`, with path costs of type `C` for liveness checks.
///
/// Decrease-key is done by pushing a new entry; the outdated one stays in the heap and is
/// recognised on pop because its `g` differs from the node's current `g`. Callers that
/// need stricter liveness (for example "not closed yet") pass it in the `live` predicate.
pub(crate) struct OpenList<K, C = K> {
    /// The entries.
    heap: BinaryHeap<Entry<K, C>>,
    /// Number of pushes so far; the next entry's [`Entry::seq`].
    pushes: u64,
}

impl<K: Ord + Copy, C: Ord + Copy> OpenList<K, C> {
    /// Creates an empty open list.
    pub(crate) fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            pushes: 0,
        }
    }

    /// Opens `node` with priority `key` and current path cost `g`.
    pub(crate) fn push(&mut self, node: NodeId, key: K, g: C) {
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
        self.pop_where(|node, g| g.cmp(&current_g(node)).is_eq())
            .map(|(node, _)| node)
    }

    /// Pops the smallest entry for which `live(node, g)` holds, dropping the others on the
    /// way, and returns its node and key.
    pub(crate) fn pop_where(&mut self, live: impl Fn(NodeId, C) -> bool) -> Option<(NodeId, K)> {
        self.drop_dead(live);

        self.heap.pop().map(|entry| (entry.node, entry.key))
    }

    /// Returns the node and key of the smallest entry for which `live(node, g)` holds,
    /// without removing it. Dead entries above it are dropped.
    pub(crate) fn peek_where(&mut self, live: impl Fn(NodeId, C) -> bool) -> Option<(NodeId, K)> {
        self.drop_dead(live);

        self.heap.peek().map(|entry| (entry.node, entry.key))
    }

    /// Removes every entry.
    pub(crate) fn clear(&mut self) {
        self.heap.clear();
    }

    /// Pops entries off the top until the top entry is live or the heap is empty.
    fn drop_dead(&mut self, live: impl Fn(NodeId, C) -> bool) {
        while self
            .heap
            .peek()
            .is_some_and(|entry| !live(entry.node, entry.g))
        {
            self.heap.pop();
        }
    }
}
