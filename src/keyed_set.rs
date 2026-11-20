//! An ordered set of items with one key each ([`KeyedSet`]), supporting replace and remove
//! by item.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ops::Bound;

/// Dense item ids (such as node ids) ordered by a key, at most one key per item.
///
/// Unlike a lazy heap it supports exact removal and replacement, so several sets can hold
/// the same items under different orders and stay in step.
#[derive(Debug, Clone)]
pub(crate) struct KeyedSet<K> {
    /// `(key, item)` pairs in key order; the item breaks ties between equal keys.
    set: BTreeSet<(K, usize)>,
    /// The current key of each item, indexed by item id. `None` if the item is absent.
    keys: Vec<Option<K>>,
}

impl<K: Ord + Clone> KeyedSet<K> {
    /// Creates an empty set.
    pub(crate) const fn new() -> Self {
        Self {
            set: BTreeSet::new(),
            keys: Vec::new(),
        }
    }

    /// The number of items in the set.
    pub(crate) fn len(&self) -> usize {
        self.set.len()
    }

    /// Returns `true` if the set holds no item.
    pub(crate) fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// The current key of `item`, if it is in the set.
    pub(crate) fn key(&self, item: usize) -> Option<&K> {
        self.keys.get(item).and_then(Option::as_ref)
    }

    /// Inserts `item` with `key`, replacing its previous key if it had one.
    pub(crate) fn insert(&mut self, item: usize, key: K) {
        self.remove(item);
        if self.keys.len() <= item {
            self.keys.resize(item + 1, None);
        }

        self.set.insert((key.clone(), item));
        self.keys[item] = Some(key);
    }

    /// Removes `item` and returns its key, if it was in the set.
    pub(crate) fn remove(&mut self, item: usize) -> Option<K> {
        let key = self.keys.get_mut(item)?.take()?;
        self.set.remove(&(key.clone(), item));

        Some(key)
    }

    /// The item with the smallest key, and the key.
    pub(crate) fn first(&self) -> Option<(&K, usize)> {
        self.set.first().map(|(key, item)| (key, *item))
    }

    /// The item with the smallest key strictly after `(key, item)`.
    pub(crate) fn after(&self, key: &K, item: usize) -> Option<(&K, usize)> {
        let start = (key.clone(), item);

        self.set
            .range((Bound::Excluded(start), Bound::Unbounded))
            .next()
            .map(|(key, item)| (key, *item))
    }

    /// The item with the largest key strictly before `(key, item)`.
    pub(crate) fn before(&self, key: &K, item: usize) -> Option<(&K, usize)> {
        let end = (key.clone(), item);

        self.set
            .range((Bound::Unbounded, Bound::Excluded(end)))
            .next_back()
            .map(|(key, item)| (key, *item))
    }
}
