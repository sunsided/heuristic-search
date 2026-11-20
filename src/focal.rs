//! The focal list ([`FocalList`]) of focal search, and how it breaks ties ([`TieBreak`]).
//!
//! A focal list holds open items under two orders. The *open* order is by an admissible
//! value `f`. The *focal* sublist is every item with `f <= w * f_min`, ordered by a second,
//! free key. Popping from the focal sublist instead of the open order is what bounded
//! suboptimal searches ([`focal_search`](crate::focal_search),
//! [`ees`](crate::ees)) are built on. With weight `1` the focal sublist holds exactly the
//! items of minimal `f`, so the list is an A* open list with the second key as tie-breaker.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Bound::{Excluded, Unbounded};

use search_core::Cost;

use crate::cost::{Weight, within};
use crate::indexed_heap::IndexedHeap;

/// How equal keys are ordered, after the primary and (for the focal sublist) the focal key.
///
/// Remaining ties after the path cost `g` go first in, first out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TieBreak {
    /// Larger path cost `g` first. Prefers deeper items, which usually finishes a plateau
    /// of equal `f` sooner. The default, and what [`a_star`](crate::a_star) does.
    #[default]
    LargerG,
    /// Smaller path cost `g` first.
    SmallerG,
}

impl TieBreak {
    /// Compares two path costs; [`Ordering::Less`] means `a` goes first.
    fn cmp_g<C: Cost>(self, a: C, b: C) -> Ordering {
        match self {
            Self::LargerG => b.cmp(&a),
            Self::SmallerG => a.cmp(&b),
        }
    }
}

/// An item's position in the open order: by `f`, then `g` per [`TieBreak`], then push
/// order.
#[derive(Debug, Clone, Copy)]
struct OpenKey<C> {
    /// The admissible value the open order and the focal bound use.
    f: C,
    /// The path cost, for tie-breaking.
    g: C,
    /// The tie-break rule; the same for every key of one list.
    tie: TieBreak,
    /// Push counter; the last tie-breaker.
    seq: u64,
}

impl<C: Cost> PartialEq for OpenKey<C> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<C: Cost> Eq for OpenKey<C> {}

impl<C: Cost> PartialOrd for OpenKey<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Cost> Ord for OpenKey<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f
            .cmp(&other.f)
            .then_with(|| self.tie.cmp_g(self.g, other.g))
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

/// An item's position in the focal order: by focal key, then as in the open order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct FocalOrder<K, O> {
    /// The focal key.
    key: K,
    /// The open-order key, for the remaining ties.
    open: O,
}

/// Open items under an admissible order `f` with a focal sublist ordered by a second key.
///
/// Items are dense ids, such as node ids, each at most once in the list: pushing an item
/// again replaces its values (decrease-key and increase-key both work), and
/// [`FocalList::remove`] takes it out. The focal sublist is kept exact:
/// it is every item with `f <= w * f_min` for the current minimum `f`, also when `f_min`
/// falls because a smaller `f` was pushed.
///
/// The focal key `K` is any [`Ord`] value, for example a conflict count in multi-agent path
/// finding. Ties go by smaller `f`, then [`TieBreak`] on `g`, then first in, first out.
///
/// # Examples
///
/// ```
/// use heuristic_search::{FocalList, Weight};
///
/// // Items 0, 1, 2 with f = 10, 12, 20 and focal keys 5, 1, 0.
/// let mut list = FocalList::new(Weight::new(3, 2));
/// list.push(0, 10_u32, 0, 5_u32);
/// list.push(1, 12, 0, 1);
/// list.push(2, 20, 0, 0);
///
/// // Focal holds f <= 15: items 0 and 1. Item 1 has the smaller focal key.
/// assert_eq!(list.min_f(), Some(10));
/// assert_eq!(list.pop(), Some(1));
/// assert_eq!(list.pop(), Some(0));
/// // Now f_min = 20, so item 2 is focal.
/// assert_eq!(list.pop(), Some(2));
/// assert_eq!(list.pop(), None);
/// ```
#[derive(Debug, Clone)]
pub struct FocalList<C, K> {
    /// The suboptimality weight `w`; at least `1`.
    weight: Weight,
    /// The tie-break rule for both orders.
    tie: TieBreak,
    /// Every item, bucketed by `f` in ascending order. Never holds an empty bucket.
    ///
    /// Focal membership depends on `f` alone, so a bucket is focal or not as a whole, and
    /// moving the focal boundary moves whole buckets.
    buckets: BTreeMap<C, Vec<usize>>,
    /// The largest `f` of a focal bucket: buckets up to and including it are focal, none
    /// after it. `None` if the sublist is empty. Always a key of `buckets` when set.
    boundary: Option<C>,
    /// The items with `f <= w * f_min`, in focal order.
    focal: IndexedHeap<FocalOrder<K, OpenKey<C>>>,
    /// What the list knows about each item, indexed by item id; `None` if absent.
    items: Vec<Option<Item<C, K>>>,
    /// The number of items in the list.
    len: usize,
    /// Number of pushes so far; the next entry's sequence number.
    pushes: u64,
}

/// One item of a [`FocalList`].
#[derive(Debug, Clone)]
struct Item<C, K> {
    /// The item's position in the open order.
    open: OpenKey<C>,
    /// The focal key, kept for when the item enters the focal sublist.
    key: K,
    /// The item's index in its `f` bucket.
    slot: usize,
}

impl<C: Cost, K: Ord + Clone> FocalList<C, K> {
    /// Creates an empty list with suboptimality weight `weight` and [`TieBreak::LargerG`].
    ///
    /// # Panics
    ///
    /// Panics if `weight` is below `1`: the focal sublist would then miss the item of
    /// minimal `f`.
    #[must_use]
    pub fn new(weight: Weight) -> Self {
        Self::with_tie_break(weight, TieBreak::LargerG)
    }

    /// Creates an empty list with suboptimality weight `weight` and tie-break rule `tie`.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is below `1`.
    #[must_use]
    pub fn with_tie_break(weight: Weight, tie: TieBreak) -> Self {
        assert!(
            weight.numerator() >= weight.denominator(),
            "focal weight must be at least 1"
        );

        Self {
            weight,
            tie,
            buckets: BTreeMap::new(),
            boundary: None,
            focal: IndexedHeap::new(),
            items: Vec::new(),
            len: 0,
            pushes: 0,
        }
    }

    /// The number of items in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list holds no item.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of items in the focal sublist. At least `1` unless the list is empty.
    #[must_use]
    pub fn focal_len(&self) -> usize {
        self.focal.len()
    }

    /// Returns `true` if `item` is in the list.
    #[must_use]
    pub fn contains(&self, item: usize) -> bool {
        self.item(item).is_some()
    }

    /// The `f` value of `item`, if it is in the list.
    #[must_use]
    pub fn f(&self, item: usize) -> Option<C> {
        self.item(item).map(|entry| entry.open.f)
    }

    /// The smallest `f` in the list.
    #[must_use]
    pub fn min_f(&self) -> Option<C> {
        self.buckets.first_key_value().map(|(&f, _)| f)
    }

    /// The item with the smallest `f` (first in open order: ties by [`TieBreak`] on `g`,
    /// then first in, first out).
    ///
    /// Scans the items of minimal `f`, so it costs time linear in their number.
    #[must_use]
    pub fn peek_min_f(&self) -> Option<usize> {
        let (_, bucket) = self.buckets.first_key_value()?;

        bucket
            .iter()
            .filter_map(|&item| self.item(item).map(|entry| (entry.open, item)))
            .min()
            .map(|(_, item)| item)
    }

    /// The best item of the focal sublist: smallest focal key among `f <= w * f_min`.
    #[must_use]
    pub fn peek(&self) -> Option<usize> {
        self.focal.peek().map(|(_, item)| item)
    }

    /// Removes and returns the best item of the focal sublist.
    pub fn pop(&mut self) -> Option<usize> {
        let item = self.peek()?;
        self.remove(item);

        Some(item)
    }

    /// Inserts `item` with admissible value `f`, path cost `g` and focal key `key`, or
    /// replaces its values if it is already in the list.
    pub fn push(&mut self, item: usize, f: C, g: C, key: K) {
        self.detach(item);

        let open = OpenKey {
            f,
            g,
            tie: self.tie,
            seq: self.pushes,
        };
        self.pushes += 1;
        self.len += 1;

        let bucket = self.buckets.entry(f).or_default();
        let slot = bucket.len();
        bucket.push(item);
        if self.is_focal(f) {
            self.focal.insert(
                item,
                FocalOrder {
                    key: key.clone(),
                    open,
                },
            );
        }

        if self.items.len() <= item {
            self.items.resize_with(item + 1, || None);
        }

        self.items[item] = Some(Item { open, key, slot });
        self.rebalance();
    }

    /// Removes `item`. Returns `true` if it was in the list.
    pub fn remove(&mut self, item: usize) -> bool {
        let present = self.detach(item);
        if present {
            self.rebalance();
        }

        present
    }

    /// The entry of `item`, if it is in the list.
    fn item(&self, item: usize) -> Option<&Item<C, K>> {
        self.items.get(item).and_then(Option::as_ref)
    }

    /// Returns `true` if items with value `f` belong to the focal sublist as it stands.
    fn is_focal(&self, f: C) -> bool {
        self.boundary.is_some_and(|boundary| f <= boundary)
    }

    /// Takes `item` out of every structure without restoring the focal invariant. Returns
    /// `true` if it was in the list.
    fn detach(&mut self, item: usize) -> bool {
        let Some(entry) = self.items.get_mut(item).and_then(Option::take) else {
            return false;
        };

        let f = entry.open.f;
        if self.is_focal(f) {
            self.focal.remove(item);
        }

        self.len -= 1;
        self.unbucket(f, entry.slot);

        true
    }

    /// Removes the item at `slot` of the bucket for `f`, dropping the bucket if it empties.
    fn unbucket(&mut self, f: C, slot: usize) {
        let bucket = self.buckets.get_mut(&f).expect("every item has a bucket");
        bucket.swap_remove(slot);
        if let Some(&moved) = bucket.get(slot) {
            self.items[moved]
                .as_mut()
                .expect("every bucketed item is in the list")
                .slot = slot;
        }

        if bucket.is_empty() {
            self.buckets.remove(&f);
            if self.boundary == Some(f) {
                self.boundary = self.buckets.range(..f).next_back().map(|(&f, _)| f);
            }
        }
    }

    /// Restores the focal invariant after `f_min` may have changed: shrinks the sublist
    /// by whole buckets while the boundary is out of bound, then grows it while the next
    /// bucket is within bound.
    fn rebalance(&mut self) {
        let Some(min_f) = self.min_f() else {
            self.boundary = None;
            return;
        };

        while let Some(boundary) = self.boundary {
            if within(boundary, self.weight, min_f) {
                break;
            }

            self.leave_focal(boundary);
            self.boundary = self.buckets.range(..boundary).next_back().map(|(&f, _)| f);
        }

        loop {
            let next = match self.boundary {
                Some(boundary) => self.buckets.range((Excluded(boundary), Unbounded)).next(),
                None => self.buckets.first_key_value(),
            };
            let Some(f) = next
                .map(|(&f, _)| f)
                .filter(|&f| within(f, self.weight, min_f))
            else {
                break;
            };

            self.enter_focal(f);
            self.boundary = Some(f);
        }
    }

    /// Moves every item of the bucket for `f` into the focal sublist.
    fn enter_focal(&mut self, f: C) {
        for &item in &self.buckets[&f] {
            let entry = self.items[item]
                .as_ref()
                .expect("every bucketed item is in the list");
            let order = FocalOrder {
                key: entry.key.clone(),
                open: entry.open,
            };
            self.focal.insert(item, order);
        }
    }

    /// Takes every item of the bucket for `f` out of the focal sublist.
    fn leave_focal(&mut self, f: C) {
        for &item in &self.buckets[&f] {
            self.focal.remove(item);
        }
    }
}

#[cfg(test)]
mod tests {
    //! The focal list against a naive model that recomputes the focal sublist on every query.

    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::{FocalList, TieBreak};
    use crate::cost::{Weight, within};

    /// One operation on the list: push `(item, f, g, key)`, remove an item, or pop.
    #[derive(Debug, Clone)]
    enum Op {
        /// Push or replace an item.
        Push(usize, u32, u32, u8),
        /// Remove an item.
        Remove(usize),
        /// Pop the best focal item.
        Pop,
    }

    /// Random operations over items `0..8`.
    fn ops() -> impl Strategy<Value = Vec<Op>> {
        let op = prop_oneof![
            (0_usize..8, 0_u32..30, 0_u32..10, 0_u8..4)
                .prop_map(|(i, f, g, k)| Op::Push(i, f, g, k)),
            (0_usize..8).prop_map(Op::Remove),
            Just(Op::Pop),
        ];

        prop::collection::vec(op, 0..60)
    }

    proptest! {
        /// After every operation, the popped item and the focal size match the model:
        /// smallest key among items with `f <= w * f_min`, ties by `f`, then larger or
        /// smaller `g` per tie-break rule, then push order.
        #[test]
        fn matches_naive_model(ops in ops(), numerator in 2_u32..6, larger_g in any::<bool>()) {
            let weight = Weight::new(numerator, 2);
            let tie = if larger_g { TieBreak::LargerG } else { TieBreak::SmallerG };
            let mut list = FocalList::with_tie_break(weight, tie);
            // (item, f, g, key, seq)
            let mut model: Vec<(usize, u32, u32, u8, usize)> = Vec::new();

            for (seq, op) in ops.into_iter().enumerate() {
                let min_f = model.iter().map(|e| e.1).min();
                let best = min_f.and_then(|min_f| {
                    model
                        .iter()
                        .filter(|e| within(e.1, weight, min_f))
                        .min_by_key(|e| (e.3, e.1, if larger_g { u32::MAX - e.2 } else { e.2 }, e.4))
                        .map(|e| e.0)
                });

                match op {
                    Op::Push(item, f, g, key) => {
                        model.retain(|e| e.0 != item);
                        model.push((item, f, g, key, seq));
                        list.push(item, f, g, key);
                    }
                    Op::Remove(item) => {
                        let present = model.iter().any(|e| e.0 == item);
                        model.retain(|e| e.0 != item);
                        prop_assert_eq!(list.remove(item), present);
                    }
                    Op::Pop => {
                        prop_assert_eq!(list.pop(), best);
                        model.retain(|e| Some(e.0) != best);
                    }
                }

                let min_f = model.iter().map(|e| e.1).min();
                let focal = min_f.map_or(0, |min_f| model.iter().filter(|e| within(e.1, weight, min_f)).count());
                prop_assert_eq!(list.len(), model.len());
                prop_assert_eq!(list.focal_len(), focal);
                prop_assert_eq!(list.min_f(), min_f);
                let first = model
                    .iter()
                    .min_by_key(|e| (e.1, if larger_g { u32::MAX - e.2 } else { e.2 }, e.4))
                    .map(|e| e.0);
                prop_assert_eq!(list.peek_min_f(), first);
            }
        }
    }
}
