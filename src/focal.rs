//! The focal list ([`FocalList`]) of focal search, and how it breaks ties ([`TieBreak`]).
//!
//! A focal list holds open items under two orders. The *open* order is by an admissible
//! value `f`. The *focal* sublist is every item with `f <= w * f_min`, ordered by a second,
//! free key. Popping from the focal sublist instead of the open order is what bounded
//! suboptimal searches ([`focal_search`](crate::focal_search),
//! [`ees`](crate::ees)) are built on. With weight `1` the focal sublist holds exactly the
//! items of minimal `f`, so the list is an A* open list with the second key as tie-breaker.

use alloc::vec::Vec;
use core::cmp::Ordering;

use search_core::Cost;

use crate::cost::{Ordered, Weight, within};
use crate::keyed_set::KeyedSet;

/// How equal keys are ordered, after the primary and (for the focal sublist) the focal key.
///
/// Remaining ties after the path cost `g` go first in, first out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TieBreak {
    /// Larger path cost `g` first. Prefers deeper items, which usually finishes a plateau
    /// of equal `f` sooner. The default, and what [`astar`](crate::astar) does.
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
    f: Ordered<C>,
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
    /// Every item, in open order.
    open: KeyedSet<OpenKey<C>>,
    /// The items with `f <= w * f_min`, in focal order.
    focal: KeyedSet<FocalOrder<K, OpenKey<C>>>,
    /// The last open-order entry in the focal sublist: every open entry up to and
    /// including it is focal, none after it. `None` if the sublist is empty.
    boundary: Option<(OpenKey<C>, usize)>,
    /// The focal key of each item in the list, indexed by item id; needed to move items
    /// into the focal sublist when `f_min` grows.
    keys: Vec<Option<K>>,
    /// Number of pushes so far; the next entry's sequence number.
    pushes: u64,
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
            open: KeyedSet::new(),
            focal: KeyedSet::new(),
            boundary: None,
            keys: Vec::new(),
            pushes: 0,
        }
    }

    /// The number of items in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.open.len()
    }

    /// Returns `true` if the list holds no item.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.open.is_empty()
    }

    /// The number of items in the focal sublist. At least `1` unless the list is empty.
    #[must_use]
    pub fn focal_len(&self) -> usize {
        self.focal.len()
    }

    /// Returns `true` if `item` is in the list.
    #[must_use]
    pub fn contains(&self, item: usize) -> bool {
        self.open.key(item).is_some()
    }

    /// The `f` value of `item`, if it is in the list.
    #[must_use]
    pub fn f(&self, item: usize) -> Option<C> {
        self.open.key(item).map(|key| key.f.0)
    }

    /// The smallest `f` in the list.
    #[must_use]
    pub fn min_f(&self) -> Option<C> {
        self.open.first().map(|(key, _)| key.f.0)
    }

    /// The item with the smallest `f` (first in open order).
    #[must_use]
    pub fn peek_min_f(&self) -> Option<usize> {
        self.open.first().map(|(_, item)| item)
    }

    /// The best item of the focal sublist: smallest focal key among `f <= w * f_min`.
    #[must_use]
    pub fn peek(&self) -> Option<usize> {
        self.focal.first().map(|(_, item)| item)
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
        self.remove(item);

        let open = OpenKey {
            f: Ordered(f),
            g,
            tie: self.tie,
            seq: self.pushes,
        };
        self.pushes += 1;
        self.open.insert(item, open);
        if self.keys.len() <= item {
            self.keys.resize(item + 1, None);
        }

        if self
            .boundary
            .is_some_and(|(last, last_item)| (open, item) < (last, last_item))
        {
            self.focal.insert(
                item,
                FocalOrder {
                    key: key.clone(),
                    open,
                },
            );
        }

        self.keys[item] = Some(key);
        self.rebalance();
    }

    /// Removes `item`. Returns `true` if it was in the list.
    pub fn remove(&mut self, item: usize) -> bool {
        let Some(open) = self.open.key(item).copied() else {
            return false;
        };

        if self.boundary == Some((open, item)) {
            self.boundary = self.open.before(&open, item).map(|(key, id)| (*key, id));
        }

        self.focal.remove(item);
        self.open.remove(item);
        self.keys[item] = None;
        self.rebalance();

        true
    }

    /// Restores the focal invariant after `f_min` may have changed: shrinks the sublist
    /// from its boundary while the boundary is out of bound, then grows it while the next
    /// open entry is within bound.
    fn rebalance(&mut self) {
        let Some(min_f) = self.min_f() else {
            self.boundary = None;
            return;
        };

        while let Some((last, item)) = self.boundary {
            if within(last.f.0, self.weight, min_f) {
                break;
            }

            self.focal.remove(item);
            self.boundary = self.open.before(&last, item).map(|(key, id)| (*key, id));
        }

        loop {
            let next = match &self.boundary {
                Some((last, item)) => self.open.after(last, *item),
                None => self.open.first(),
            };
            let Some((&open, item)) = next.filter(|(key, _)| within(key.f.0, self.weight, min_f))
            else {
                break;
            };

            let key = self.keys[item]
                .clone()
                .expect("every open item has a focal key");
            self.focal.insert(item, FocalOrder { key, open });
            self.boundary = Some((open, item));
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
            }
        }
    }
}
