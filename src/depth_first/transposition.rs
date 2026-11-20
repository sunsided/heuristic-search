//! A bounded transposition table for IDA* ([`Table`]), and the [`Transpositions`] seam that
//! lets [`ida_star`](crate::ida_star) run without one.

use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::DefaultHashBuilder;
use search_core::Cost;

/// Remembers which states an IDA* iteration has searched below, and from which `g`.
///
/// Within one iteration, a state searched from path cost `g'` need not be searched again
/// from `g >= g'`: the earlier visit had at least as much of the threshold left.
pub(crate) trait Transpositions<S, C> {
    /// Starts a new iteration: forgets every entry of the previous ones.
    fn next_iteration(&mut self);

    /// Returns `true` if `state` was already searched below in this iteration from a path
    /// cost of at most `g`.
    fn covers(&self, state: &S, g: C) -> bool;

    /// Records that `state` is about to be searched below from path cost `g`.
    fn record(&mut self, state: &S, g: C);
}

/// No table: plain IDA*, which needs neither `Hash` nor memory per state.
pub(crate) struct NoTable;

impl<S, C> Transpositions<S, C> for NoTable {
    fn next_iteration(&mut self) {}

    fn covers(&self, _: &S, _: C) -> bool {
        false
    }

    fn record(&mut self, _: &S, _: C) {}
}

/// One slot of a [`Table`].
struct Slot<S, C> {
    /// The state searched below.
    state: S,
    /// The path cost it was searched from.
    g: C,
    /// The iteration that wrote the slot; entries of older iterations are stale.
    iteration: u32,
}

/// A direct-mapped transposition table with a fixed number of slots.
///
/// Each state hashes to one slot, and recording always overwrites it, so memory stays at
/// `capacity` states however large the graph. A lost entry only costs a repeated search,
/// never correctness. Iterations are told apart by a counter, so starting one costs
/// nothing; the slots are cleared only when the counter wraps.
pub(crate) struct Table<S, C> {
    /// The slots; empty for a capacity of zero, which disables the table.
    slots: Vec<Option<Slot<S, C>>>,
    /// The current iteration.
    iteration: u32,
    /// Hashes states to slots.
    hasher: DefaultHashBuilder,
}

impl<S, C> Table<S, C> {
    /// Creates a table with `capacity` slots, allocated up front.
    pub(crate) fn new(capacity: usize) -> Self {
        let mut slots = Vec::new();
        slots.resize_with(capacity, || None);

        Self {
            slots,
            iteration: 0,
            hasher: DefaultHashBuilder::default(),
        }
    }
}

impl<S: Hash + Eq + Clone, C: Cost> Table<S, C> {
    /// The slot index of `state`, or `None` for a table without slots.
    fn index(&self, state: &S) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }

        // Multiply-shift range reduction: unbiased enough and cheaper than `%`.
        let hash = u128::from(self.hasher.hash_one(state));
        let len = self.slots.len() as u128;

        usize::try_from((hash * len) >> 64).ok()
    }
}

impl<S: Hash + Eq + Clone, C: Cost> Transpositions<S, C> for Table<S, C> {
    fn next_iteration(&mut self) {
        self.iteration = self.iteration.wrapping_add(1);
        if self.iteration == 0 {
            self.slots.iter_mut().for_each(|slot| *slot = None);
        }
    }

    fn covers(&self, state: &S, g: C) -> bool {
        let Some(index) = self.index(state) else {
            return false;
        };

        self.slots[index].as_ref().is_some_and(|slot| {
            slot.iteration == self.iteration && slot.g <= g && slot.state == *state
        })
    }

    fn record(&mut self, state: &S, g: C) {
        if let Some(index) = self.index(state) {
            self.slots[index] = Some(Slot {
                state: state.clone(),
                g,
                iteration: self.iteration,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the transposition [`Table`].

    use super::{Table, Transpositions};

    /// An entry covers equal and larger `g` in its own iteration only.
    #[test]
    fn covers_within_iteration() {
        let mut table = Table::<u32, u32>::new(16);
        table.next_iteration();
        table.record(&7, 3);

        assert!(table.covers(&7, 3));
        assert!(table.covers(&7, 5));
        assert!(!table.covers(&7, 2));
        assert!(!table.covers(&8, 9));

        table.next_iteration();
        assert!(!table.covers(&7, 5));
    }

    /// A table without slots never covers anything.
    #[test]
    fn zero_capacity_is_disabled() {
        let mut table = Table::<u32, u32>::new(0);
        table.record(&1, 0);

        assert!(!table.covers(&1, 0));
    }

    /// One slot keeps only the last state recorded.
    #[test]
    fn one_slot_overwrites() {
        let mut table = Table::<u32, u32>::new(1);
        table.record(&1, 0);
        table.record(&2, 0);

        assert!(!table.covers(&1, 0));
        assert!(table.covers(&2, 0));
    }
}
