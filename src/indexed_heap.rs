//! A min-heap over dense item ids with exact removal and key change ([`IndexedHeap`]).

use alloc::vec::Vec;

/// Marks an item that is not in the heap in [`IndexedHeap::position`].
const ABSENT: usize = usize::MAX;

/// Number of children per heap node. Four halves the depth of a binary heap and keeps the
/// children of a node in one or two cache lines, which pays off for removals by item.
const ARITY: usize = 4;

/// A 4-ary min-heap of dense item ids (such as node ids), at most one key per item.
///
/// Unlike [`BinaryHeap`](alloc::collections::BinaryHeap) it knows where each item sits, so
/// [`IndexedHeap::remove`] and replacing a key are exact `O(log n)` operations without
/// stale entries. Keys must be distinct for a deterministic order; equal keys pop in an
/// unspecified order.
#[derive(Debug, Clone)]
pub(crate) struct IndexedHeap<K> {
    /// `(key, item)` pairs in heap order: every key is `<=` the keys of its children.
    heap: Vec<(K, usize)>,
    /// The index in `heap` of each item, indexed by item id; [`ABSENT`] if not in the heap.
    position: Vec<usize>,
}

impl<K: Ord> IndexedHeap<K> {
    /// Creates an empty heap.
    pub(crate) const fn new() -> Self {
        Self {
            heap: Vec::new(),
            position: Vec::new(),
        }
    }

    /// The number of items in the heap.
    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    /// The item with the smallest key, and the key.
    pub(crate) fn peek(&self) -> Option<(&K, usize)> {
        self.heap.first().map(|(key, item)| (key, *item))
    }

    /// Inserts `item` with `key`, replacing its previous key if it had one.
    pub(crate) fn insert(&mut self, item: usize, key: K) {
        if self.position.len() <= item {
            self.position.resize(item + 1, ABSENT);
        }

        let index = self.position[item];
        if index == ABSENT {
            let index = self.heap.len();
            self.heap.push((key, item));
            self.position[item] = index;
            self.sift_up(index);
        } else {
            self.heap[index].0 = key;
            self.restore(index);
        }
    }

    /// Removes `item` and returns its key, if it was in the heap.
    pub(crate) fn remove(&mut self, item: usize) -> Option<K> {
        let index = *self.position.get(item)?;
        if index == ABSENT {
            return None;
        }

        let last = self.heap.len() - 1;
        self.heap.swap(index, last);
        let (key, _) = self.heap.pop().expect("the heap holds `item`");
        self.position[item] = ABSENT;
        if index < last {
            self.position[self.heap[index].1] = index;
            self.restore(index);
        }

        Some(key)
    }

    /// Moves the entry at `index` up or down until the heap order holds again.
    fn restore(&mut self, index: usize) {
        if index > 0 && self.heap[index].0 < self.heap[(index - 1) / ARITY].0 {
            self.sift_up(index);
        } else {
            self.sift_down(index);
        }
    }

    /// Moves the entry at `index` towards the root while it is smaller than its parent.
    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / ARITY;
            if self.heap[index].0 >= self.heap[parent].0 {
                break;
            }

            self.swap(index, parent);
            index = parent;
        }
    }

    /// Moves the entry at `index` towards the leaves while a child is smaller.
    fn sift_down(&mut self, mut index: usize) {
        loop {
            let first = index * ARITY + 1;
            let end = (first + ARITY).min(self.heap.len());
            let Some(child) = (first..end).min_by(|&a, &b| self.heap[a].0.cmp(&self.heap[b].0))
            else {
                break;
            };

            if self.heap[child].0 >= self.heap[index].0 {
                break;
            }

            self.swap(index, child);
            index = child;
        }
    }

    /// Swaps two heap slots and updates both positions.
    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.position[self.heap[a].1] = a;
        self.position[self.heap[b].1] = b;
    }
}

#[cfg(test)]
mod tests {
    //! The indexed heap against a naive model that scans for the minimum.

    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::IndexedHeap;

    proptest! {
        /// Random inserts, replacements and removals keep the minimum and the length exact.
        #[test]
        fn matches_naive_model(ops in prop::collection::vec((0_usize..12, 0_u32..40, any::<bool>()), 0..120)) {
            let mut heap = IndexedHeap::new();
            let mut model: Vec<Option<u32>> = alloc::vec![None; 12];

            for (seq, (item, key, insert)) in ops.into_iter().enumerate() {
                // Distinct keys: `(key, seq)`.
                if insert {
                    heap.insert(item, (key, seq));
                    model[item] = Some(key);
                } else {
                    prop_assert_eq!(heap.remove(item).map(|(key, _)| key), model[item].take());
                }

                let min = model.iter().flatten().min().copied();
                prop_assert_eq!(heap.peek().map(|(key, _)| key.0), min);
                prop_assert_eq!(heap.len(), model.iter().flatten().count());
            }
        }
    }
}
