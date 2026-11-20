//! Anytime search: [`ara_star`] and [`ana_star`] find a first solution quickly and then
//! improve it until it is provably optimal or the [`Budget`](crate::Budget) runs out.
//!
//! Both report every improved solution through a callback, together with a lower bound on
//! the optimal cost, and return an [`AnytimeResult`] with the best solution found.
//!
//! | Algorithm    | Order                         | Needs                 | Parameters      |
//! | ------------ | ----------------------------- | --------------------- | --------------- |
//! | [`ara_star`] | `g + w * h`, `w` decreasing   | consistent `h`        | weight schedule |
//! | [`ana_star`] | largest `(G - g) / h`         | admissible `h`        | none            |

mod ana;
mod ara;

pub use ana::ana_star;
pub use ara::ara_star;

use crate::outcome::{Solution, Stats};
use crate::problem::Successors;

/// What an anytime search returns: the best solution found and whether the search ran to
/// completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnytimeResult<S, A, C> {
    /// The cheapest solution found, if any.
    pub solution: Option<Solution<S, A, C>>,
    /// `true` if the search finished on its own rather than being stopped by its budget.
    ///
    /// Then [`AnytimeResult::solution`] carries the algorithm's final guarantee (optimal for
    /// [`ana_star`], within the last weight of the schedule for [`ara_star`]), and `None`
    /// means no goal is reachable.
    pub complete: bool,
    /// Counters collected over all iterations.
    pub stats: Stats,
}

/// The result type of anytime searches over graph `G`.
pub type AnytimeGraphResult<G> =
    AnytimeResult<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;
