//! Real-time heuristic search: an agent that plans a bounded amount, moves, and learns a
//! heuristic table that persists across trials.
//!
//! Every search here runs one *trial*: the agent starts at `start` and acts until it stands
//! on a goal, proves that no goal is reachable, or the [`Budget`] stops it. Before each move
//! it spends at most a fixed amount of work (a lookahead of `n` expansions) and updates the
//! values of the states it looked at in a [`LearnedHeuristic`], which the caller keeps. Run
//! the next trial with the same table and the agent does better; with an admissible initial
//! heuristic on a finite graph from which the goal stays reachable, repeated trials converge
//! to cheapest paths and the table to exact costs along them.
//!
//! | Search            | Lookahead               | Update of the expanded states                  |
//! | ----------------- | ----------------------- | ---------------------------------------------- |
//! | [`lrta_star`]     | the successors          | `h(s) = min c(s, s') + h(s')`                  |
//! | [`rta_star`]      | the successors          | second-best `c(s, s') + h(s')`, not admissible |
//! | [`lss_lrta_star`] | A*, `n` expansions      | Dijkstra from the open list (Koenig and Sun)   |
//! | [`lhb_lrta_star`] | A*, `n` expansions      | shortest paths to the frontier leaves (LHB)    |
//! | [`rtaa_star`]     | A*, `n` expansions      | `h(s) = f(best open) - g(s)`                   |
//!
//! [`lhb_update`] is the limited-horizon Bellman update of Hadi et al. on its own: a bounded
//! A* from one state and the update, without moving, as the paper uses it to label sampled
//! states offline.
//!
//! The lookahead searches move along the A* path to the open state with the smallest `f`
//! (or to the goal, if A* selected one), then plan again from there.
//!
//! ```mermaid
//! flowchart TD
//!     goal{at a goal?} -->|yes| found[Found: the travelled path]
//!     goal -->|no| plan[bounded lookahead from the current state]
//!     plan -->|budget| limit[LimitReached]
//!     plan --> learn[update the table]
//!     learn -->|no goal reachable| unsolvable[Unsolvable]
//!     learn --> act[move along the planned path]
//!     act --> goal
//! ```
//!
//! [`Outcome::Found`] holds the path the agent travelled, loops included; its cost is what
//! the trial cost. [`Outcome::Unsolvable`] means the agent proved that no goal is reachable
//! from where it stands, which on a graph with irreversible edges need not hold for `start`.

mod lookahead;
mod single;
mod table;

pub use lookahead::{lhb_lrta_star, lhb_update, lss_lrta_star, rtaa_star};
pub use single::{lrta_star, rta_star};
pub use table::LearnedHeuristic;

use alloc::vec::Vec;

use search_core::Cost;

#[cfg(doc)]
use search_core::Budget;

use crate::outcome::{Outcome, Solution, Stats};

/// Counters of one real-time trial, beyond the search [`Stats`].
///
/// Non-exhaustive: later algorithms may add counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TrialStats {
    /// Edges the agent travelled.
    pub moves: u64,
    /// Planning steps: lookahead searches, each followed by an update and at least one move
    /// (except the last, if it proved the goal unreachable).
    pub steps: u64,
    /// Table entries whose value changed, summed over the trial. `0` means the trial
    /// learned nothing: repeated trials from the same start have converged.
    pub updates: u64,
    /// The most expansions any single planning step made. Bounded by the lookahead.
    pub max_step_expansions: u64,
    /// The most table entries any single planning step changed.
    pub max_step_updates: u64,
}

impl TrialStats {
    /// Records one planning step that expanded `expansions` states and changed `updates`
    /// table entries.
    fn record_step(&mut self, expansions: u64, updates: u64) {
        self.steps += 1;
        self.updates += updates;
        self.max_step_expansions = self.max_step_expansions.max(expansions);
        self.max_step_updates = self.max_step_updates.max(updates);
    }
}

/// The result of one real-time trial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrialResult<S, A, C> {
    /// How the trial ended. A [`Solution`] is the travelled path, loops included.
    pub outcome: Outcome<S, A, C>,
    /// Search counters summed over every planning step of the trial.
    pub stats: Stats,
    /// Real-time counters: moves, learning and the work per step.
    pub trial: TrialStats,
}

impl<S, A, C> TrialResult<S, A, C> {
    /// The travelled path, if the agent reached a goal.
    #[must_use]
    pub fn solution(&self) -> Option<&Solution<S, A, C>> {
        match &self.outcome {
            Outcome::Found(solution) => Some(solution),
            Outcome::Unsolvable | Outcome::LimitReached | Outcome::Cutoff => None,
        }
    }
}

/// The path the agent has travelled so far in a trial.
struct Trajectory<S, A, C> {
    /// Visited states in order; starts with the start state.
    states: Vec<S>,
    /// The edge labels between consecutive states.
    actions: Vec<A>,
    /// The summed edge costs.
    cost: C,
}

impl<S: Clone, A, C: Cost> Trajectory<S, A, C> {
    /// A trajectory standing at `start`.
    fn new(start: S) -> Self {
        Self {
            states: alloc::vec![start],
            actions: Vec::new(),
            cost: C::ZERO,
        }
    }

    /// The state the agent stands on.
    fn current(&self) -> &S {
        self.states.last().expect("a trajectory is never empty")
    }

    /// Travels the edge `action` to `next` at `cost`.
    fn travel(&mut self, action: A, next: S, cost: C) {
        self.actions.push(action);
        self.states.push(next);
        self.cost = self.cost + cost;
    }

    /// Ends the trial as `end` says; a goal yields the travelled path.
    fn finish(self, end: End, stats: Stats, trial: TrialStats) -> TrialResult<S, A, C> {
        let outcome = match end {
            End::Goal => Outcome::Found(Solution {
                states: self.states,
                actions: self.actions,
                cost: self.cost,
            }),
            End::Unsolvable => Outcome::Unsolvable,
            End::Limit => Outcome::LimitReached,
        };

        TrialResult {
            outcome,
            stats,
            trial,
        }
    }
}

/// Why a trial ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// The agent stands on a goal.
    Goal,
    /// No goal is reachable from where the agent stands.
    Unsolvable,
    /// The budget stopped the trial.
    Limit,
}
