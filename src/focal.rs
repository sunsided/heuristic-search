//! The focal list ([`FocalList`]) of focal search, and how it breaks ties ([`TieBreak`]).
//!
//! A focal list holds open items under two orders. The *open* order is by an admissible
//! value `f`. The *focal* sublist is every item with `f <= w * f_min`, ordered by a second,
//! free key. Popping from the focal sublist instead of the open order is what bounded
//! suboptimal searches ([`focal_search`](crate::focal_search),
//! [`ees`](crate::ees)) are built on. With weight `1` the focal sublist holds exactly the
//! items of minimal `f`, so the list is an A* open list with the second key as tie-breaker.
//!
//! An item can also carry a separate focal *value*
//! ([`FocalList::push_with_value`]): it is admitted to the focal sublist when its value is
//! at most `w * f_min`, while `f` alone still orders the open list and defines `f_min`. The
//! high level of ECBS needs this: a node's lower bound orders the list, but its cost, which
//! may be larger, must be within `w` of the smallest lower bound.

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
/// falls because a smaller `f` was pushed. Items pushed with
/// [`FocalList::push_with_value`] are admitted by their value instead of their `f`.
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
    /// The items pushed with [`FocalList::push`], whose value is their `f`, bucketed by `f`
    /// in ascending order. Never holds an empty bucket.
    ///
    /// Focal membership depends on the value alone, so a bucket is focal or not as a whole,
    /// and moving the focal boundary moves whole buckets.
    buckets: BTreeMap<C, Vec<usize>>,
    /// The items with a value apart from their `f` ([`FocalList::push_with_value`]),
    /// bucketed by value. Never holds an empty bucket; empty unless that method is used, so
    /// [`FocalList::push`] alone pays nothing for it.
    valued: BTreeMap<C, Vec<usize>>,
    /// The items of [`FocalList::valued`] again, bucketed by `f`, for [`FocalList::min_f`].
    lower: BTreeMap<C, Vec<usize>>,
    /// The largest value of a focal bucket: buckets of `buckets` and `valued` up to and
    /// including it are focal, none after it. `None` if the sublist is empty. Always a key
    /// of `buckets` or `valued` when set.
    boundary: Option<C>,
    /// The items with value `<= w * f_min`, in focal order.
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
    /// The value compared with `w * f_min` for focal membership if it differs from `f`
    /// ([`FocalList::push_with_value`]); `None` if it is `f`.
    value: Option<C>,
    /// The focal key, kept for when the item enters the focal sublist.
    key: K,
    /// The item's index in its value bucket (of `buckets` or `valued`).
    slot: usize,
    /// The item's index in its `f` bucket of `lower`; unused unless `value` is set.
    lower_slot: usize,
}

impl<C: Copy, K> Item<C, K> {
    /// The value compared with `w * f_min` for focal membership.
    fn value(&self) -> C {
        self.value.unwrap_or(self.open.f)
    }
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
            valued: BTreeMap::new(),
            lower: BTreeMap::new(),
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

    /// The number of items in the focal sublist. At least `1` unless the list is empty, as
    /// long as no item's value exceeds `w` times its `f` (always true without
    /// [`FocalList::push_with_value`]).
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
        let plain = self.buckets.first_key_value().map(|(&f, _)| f);
        let valued = self.lower.first_key_value().map(|(&f, _)| f);

        match (plain, valued) {
            (Some(plain), Some(valued)) => Some(plain.min(valued)),
            (plain, valued) => plain.or(valued),
        }
    }

    /// The item with the smallest `f` (first in open order: ties by [`TieBreak`] on `g`,
    /// then first in, first out).
    ///
    /// Scans the items of minimal `f`, so it costs time linear in their number.
    #[must_use]
    pub fn peek_min_f(&self) -> Option<usize> {
        let min_f = self.min_f()?;
        let plain = self.buckets.get(&min_f).into_iter().flatten();
        let valued = self.lower.get(&min_f).into_iter().flatten();

        plain
            .chain(valued)
            .filter_map(|&item| self.item(item).map(|entry| (entry.open, item)))
            .min()
            .map(|(_, item)| item)
    }

    /// The best item of the focal sublist: smallest focal key among `f <= w * f_min` (the
    /// value instead of `f` for items pushed with [`FocalList::push_with_value`]).
    ///
    /// `None` if the list is empty, or if no item's value is within `w * f_min`; the latter
    /// cannot happen unless some item's value exceeds `w` times its `f`.
    #[must_use]
    pub fn peek(&self) -> Option<usize> {
        self.focal.peek().map(|(_, item)| item)
    }

    /// Removes and returns the best item of the focal sublist; see [`FocalList::peek`].
    pub fn pop(&mut self) -> Option<usize> {
        let item = self.peek()?;
        self.remove(item);

        Some(item)
    }

    /// Inserts `item` with admissible value `f`, path cost `g` and focal key `key`, or
    /// replaces its values if it is already in the list.
    pub fn push(&mut self, item: usize, f: C, g: C, key: K) {
        self.insert(item, f, g, None, key);
    }

    /// Like [`FocalList::push`], but admits `item` to the focal sublist by `value` instead
    /// of `f`: it is focal while `value <= w * f_min`. `f` still orders the open list and
    /// defines `f_min`.
    ///
    /// A bounded-suboptimal search pushes the item's cost as `value` and its lower bound as
    /// `f`, so every pop costs at most `w` times the smallest lower bound. Keep `value`
    /// at most `w` times `f`, or the focal sublist may run empty while the list is not.
    pub fn push_with_value(&mut self, item: usize, f: C, g: C, value: C, key: K) {
        self.insert(item, f, g, (value != f).then_some(value), key);
    }

    /// Removes `item`. Returns `true` if it was in the list.
    pub fn remove(&mut self, item: usize) -> bool {
        let present = self.detach(item);
        if present {
            self.rebalance();
        }

        present
    }

    /// Inserts or replaces `item`; `value` is `None` if it is `f`.
    fn insert(&mut self, item: usize, f: C, g: C, value: Option<C>, key: K) {
        self.detach(item);

        let open = OpenKey {
            f,
            g,
            tie: self.tie,
            seq: self.pushes,
        };
        self.pushes += 1;
        self.len += 1;

        let (slot, lower_slot) = match value {
            None => (bucket(&mut self.buckets, f, item), 0),
            Some(value) => (
                bucket(&mut self.valued, value, item),
                bucket(&mut self.lower, f, item),
            ),
        };
        if self.is_focal(value.unwrap_or(f)) {
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

        self.items[item] = Some(Item {
            open,
            value,
            key,
            slot,
            lower_slot,
        });
        self.rebalance();
    }

    /// The entry of `item`, if it is in the list.
    fn item(&self, item: usize) -> Option<&Item<C, K>> {
        self.items.get(item).and_then(Option::as_ref)
    }

    /// Returns `true` if items with focal value `value` belong to the focal sublist as it
    /// stands.
    fn is_focal(&self, value: C) -> bool {
        self.boundary.is_some_and(|boundary| value <= boundary)
    }

    /// Takes `item` out of every structure without restoring the focal invariant. Returns
    /// `true` if it was in the list.
    fn detach(&mut self, item: usize) -> bool {
        let Some(entry) = self.items.get_mut(item).and_then(Option::take) else {
            return false;
        };

        let value = entry.value();
        if self.is_focal(value) {
            self.focal.remove(item);
        }

        self.len -= 1;
        if entry.value.is_none() {
            let moved = unbucket(&mut self.buckets, value, entry.slot);
            self.move_slot(moved, entry.slot);
        } else {
            let moved = unbucket(&mut self.valued, value, entry.slot);
            self.move_slot(moved, entry.slot);
            let moved = unbucket(&mut self.lower, entry.open.f, entry.lower_slot);
            if let Some(moved) = moved {
                self.entry_mut(moved).lower_slot = entry.lower_slot;
            }
        }

        let emptied = !self.buckets.contains_key(&value) && !self.valued.contains_key(&value);
        if emptied && self.boundary == Some(value) {
            self.boundary = self.value_before(value);
        }

        true
    }

    /// Records that `moved`, if any, now sits at `slot` of its value bucket.
    fn move_slot(&mut self, moved: Option<usize>, slot: usize) {
        if let Some(moved) = moved {
            self.entry_mut(moved).slot = slot;
        }
    }

    /// The entry of `item`, which is bucketed and so in the list.
    fn entry_mut(&mut self, item: usize) -> &mut Item<C, K> {
        self.items[item]
            .as_mut()
            .expect("every bucketed item is in the list")
    }

    /// The largest value of a bucket below `value`, in `buckets` or `valued`.
    fn value_before(&self, value: C) -> Option<C> {
        let plain = self.buckets.range(..value).next_back().map(|(&v, _)| v);
        let valued = self.valued.range(..value).next_back().map(|(&v, _)| v);

        plain.max(valued)
    }

    /// The smallest value of a bucket above `value` (of any bucket if `None`), in `buckets`
    /// or `valued`.
    fn value_after(&self, value: Option<C>) -> Option<C> {
        let first = |map: &BTreeMap<C, Vec<usize>>| match value {
            Some(value) => map
                .range((Excluded(value), Unbounded))
                .next()
                .map(|(&v, _)| v),
            None => map.first_key_value().map(|(&v, _)| v),
        };

        match (first(&self.buckets), first(&self.valued)) {
            (Some(plain), Some(valued)) => Some(plain.min(valued)),
            (plain, valued) => plain.or(valued),
        }
    }

    /// Restores the focal invariant after `f_min` may have changed: shrinks the sublist
    /// by whole value buckets while the boundary is out of bound, then grows it while the
    /// next bucket is within bound.
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
            self.boundary = self.value_before(boundary);
        }

        while let Some(value) = self
            .value_after(self.boundary)
            .filter(|&value| within(value, self.weight, min_f))
        {
            self.enter_focal(value);
            self.boundary = Some(value);
        }
    }

    /// Moves every item of the buckets for `value` into the focal sublist.
    fn enter_focal(&mut self, value: C) {
        let Self {
            buckets,
            valued,
            items,
            focal,
            ..
        } = self;
        for &item in buckets
            .get(&value)
            .into_iter()
            .chain(valued.get(&value))
            .flatten()
        {
            let entry = items[item]
                .as_ref()
                .expect("every bucketed item is in the list");
            let order = FocalOrder {
                key: entry.key.clone(),
                open: entry.open,
            };
            focal.insert(item, order);
        }
    }

    /// Takes every item of the buckets for `value` out of the focal sublist.
    fn leave_focal(&mut self, value: C) {
        let Self {
            buckets,
            valued,
            focal,
            ..
        } = self;
        for &item in buckets
            .get(&value)
            .into_iter()
            .chain(valued.get(&value))
            .flatten()
        {
            focal.remove(item);
        }
    }
}

/// Appends `item` to the bucket for `key` of `map`; returns its slot there.
fn bucket<C: Ord>(map: &mut BTreeMap<C, Vec<usize>>, key: C, item: usize) -> usize {
    let bucket = map.entry(key).or_default();
    bucket.push(item);

    bucket.len() - 1
}

/// Removes the item at `slot` of the bucket for `key` of `map`, dropping the bucket if it
/// empties. Returns the item that moved into `slot`, if any.
fn unbucket<C: Copy + Ord>(
    map: &mut BTreeMap<C, Vec<usize>>,
    key: C,
    slot: usize,
) -> Option<usize> {
    let bucket = map.get_mut(&key).expect("every item has a bucket");
    bucket.swap_remove(slot);
    let moved = bucket.get(slot).copied();
    if bucket.is_empty() {
        map.remove(&key);
    }

    moved
}

#[cfg(test)]
mod tests {
    //! The focal list against a naive model that recomputes the focal sublist on every query.

    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::{FocalList, TieBreak};
    use crate::cost::{Weight, within};

    /// One operation on the list: push `(item, f, g, extra, key)` with value `f + extra`
    /// (plain [`FocalList::push`] when `extra` is `None`), remove an item, or pop.
    #[derive(Debug, Clone)]
    enum Op {
        /// Push or replace an item.
        Push(usize, u32, u32, Option<u32>, u8),
        /// Remove an item.
        Remove(usize),
        /// Pop the best focal item.
        Pop,
    }

    /// Random operations over items `0..8`.
    fn ops() -> impl Strategy<Value = Vec<Op>> {
        let op = prop_oneof![
            (
                0_usize..8,
                0_u32..30,
                0_u32..10,
                prop::option::of(0_u32..15),
                0_u8..4
            )
                .prop_map(|(i, f, g, extra, k)| Op::Push(i, f, g, extra, k)),
            (0_usize..8).prop_map(Op::Remove),
            Just(Op::Pop),
        ];

        prop::collection::vec(op, 0..60)
    }

    /// One item of the naive model: `item`, `f`, `g`, value, key and push order.
    type Entry = (usize, u32, u32, u32, u8, usize);

    /// The model's best focal item: smallest key among items with value `<= w * f_min`,
    /// ties by `f`, then larger or smaller `g` per tie-break rule, then push order.
    fn model_best(model: &[Entry], weight: Weight, larger_g: bool) -> Option<usize> {
        let min_f = model.iter().map(|e| e.1).min()?;
        model
            .iter()
            .filter(|e| within(e.3, weight, min_f))
            .min_by_key(|e| (e.4, e.1, if larger_g { u32::MAX - e.2 } else { e.2 }, e.5))
            .map(|e| e.0)
    }

    proptest! {
        /// After every operation, the popped item and the focal size match the model
        /// ([`model_best`]), with and without separate focal values.
        #[test]
        fn matches_naive_model(ops in ops(), numerator in 2_u32..6, larger_g in any::<bool>()) {
            let weight = Weight::new(numerator, 2);
            let tie = if larger_g { TieBreak::LargerG } else { TieBreak::SmallerG };
            let mut list = FocalList::with_tie_break(weight, tie);
            let mut model: Vec<Entry> = Vec::new();

            for (seq, op) in ops.into_iter().enumerate() {
                let best = model_best(&model, weight, larger_g);

                match op {
                    Op::Push(item, f, g, extra, key) => {
                        let value = f + extra.unwrap_or(0);
                        model.retain(|e| e.0 != item);
                        model.push((item, f, g, value, key, seq));
                        match extra {
                            Some(_) => list.push_with_value(item, f, g, value, key),
                            None => list.push(item, f, g, key),
                        }
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
                let focal = min_f.map_or(0, |min_f| model.iter().filter(|e| within(e.3, weight, min_f)).count());
                prop_assert_eq!(list.len(), model.len());
                prop_assert_eq!(list.focal_len(), focal);
                prop_assert_eq!(list.min_f(), min_f);
                let first = model
                    .iter()
                    .min_by_key(|e| (e.1, if larger_g { u32::MAX - e.2 } else { e.2 }, e.5))
                    .map(|e| e.0);
                prop_assert_eq!(list.peek_min_f(), first);
            }
        }
    }
}
