//! Jump Point Search (Harabor and Grastien 2011) for 8-connected grids without corner
//! cutting: [`jps`] and [`jps_multi`].
//!
//! JPS is A* over *jump points*. From a node reached in direction `d`, only the natural
//! and forced neighbours are tried, and each try *jumps*: it walks on in that direction
//! without opening the cells in between, until it meets the goal, a cell with a forced
//! neighbour, or (on a diagonal) a cell from which a straight jump finds such a cell.
//! The cells skipped over have an equally cheap path that the jump points cover, so the
//! search stays optimal while touching far fewer nodes.
//!
//! With no corner cutting, the pruning rules are:
//!
//! | arrived by          | neighbours tried                                   | jump stops at a cell with |
//! | ------------------- | -------------------------------------------------- | ------------------------- |
//! | straight `(dx, 0)`  | `(dx, 0)`, `(0, ±1)`, `(dx, ±1)`, where legal      | `(0, s)` open but `(-dx, s)` blocked, `s = ±1` |
//! | diagonal `(dx, dy)` | `(dx, 0)`, `(0, dy)`, `(dx, dy)`, where legal      | a straight jump `(dx, 0)` or `(0, dy)` that stops |
//!
//! (and the transposed rules for `(0, dy)`). The search runs [`a_star`](crate::a_star)
//! on a private jump graph whose nodes are `(cell, direction)` pairs, so one cell may be
//! opened once per arrival direction (at most eight times).

use alloc::vec::Vec;

use search_core::{Budget, Heuristic};

use crate::best_first::a_star_multi;
use crate::grid::{
    GridCell, GridStep, Octile, OctileDistance, UniformGrid, can_move, is_open, offset, passable,
};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::Successors;

/// Jump Point Search from `start` to `goal` on an 8-connected grid: a cheapest path,
/// usually after far fewer expansions than A*.
///
/// Moves and the corner rule are those of [`EightConnected`](crate::EightConnected) (see
/// [`UniformGrid`]): the cost equals the cheapest path that
/// [`a_star`](crate::a_star) finds there with [`OctileDistance`], exactly, in [`Octile`]
/// units. The returned [`Solution`] is *sparse*: its states are the start, the jump points
/// and the goal, and `actions[i]` is the unit [`GridStep`] repeated along the straight or
/// diagonal segment from `states[i]` to `states[i + 1]`. A blocked or off-grid start is
/// skipped and a blocked or off-grid goal is unreachable ([`Outcome::Unsolvable`]).
/// [`Stats`] count jump-point expansions.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Octile, UniformGrid, jps};
///
/// /// A 7 x 5 grid with a wall at `x = 3` that leaves a gap at the bottom.
/// struct Walled;
///
/// impl UniformGrid for Walled {
///     fn width(&self) -> usize {
///         7
///     }
///
///     fn height(&self) -> usize {
///         5
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         x != 3 || y == 4
///     }
/// }
///
/// let result = jps(&Walled, (0, 0), (6, 0), Limits::NONE);
/// let solution = result.solution().expect("the gap connects both sides");
///
/// // Down and diagonally to (2, 4), straight through the gap to (4, 4), and back up: the
/// // corner rule forbids cutting past the wall's end at (3, 3).
/// assert_eq!(solution.cost, Octile::new(6, 4));
/// assert_eq!(solution.states.first(), Some(&(0, 0)));
/// assert_eq!(solution.goal(), &(6, 0));
/// ```
pub fn jps<G: UniformGrid>(
    grid: &G,
    start: GridCell,
    goal: GridCell,
    budget: impl Budget<Stats>,
) -> SearchResult<GridCell, GridStep, Octile> {
    jps_multi(grid, [start], goal, budget)
}

/// [`jps`] from several start cells at once, to the cheapest path from any of them.
///
/// Blocked and off-grid starts are skipped, repeated starts count once, an empty set of
/// starts is [`Outcome::Unsolvable`], and a start that is the goal is returned as a
/// zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Octile, UniformGrid, jps_multi};
///
/// /// An open 9 x 9 grid.
/// struct Open;
///
/// impl UniformGrid for Open {
///     fn width(&self) -> usize {
///         9
///     }
///
///     fn height(&self) -> usize {
///         9
///     }
///
///     fn is_passable(&self, _: usize, _: usize) -> bool {
///         true
///     }
/// }
///
/// let result = jps_multi(&Open, [(0, 0), (8, 0)], (8, 6), Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(Octile::new(6, 0)));
/// ```
pub fn jps_multi<G: UniformGrid>(
    grid: &G,
    starts: impl IntoIterator<Item = GridCell>,
    goal: GridCell,
    budget: impl Budget<Stats>,
) -> SearchResult<GridCell, GridStep, Octile> {
    let graph = JumpGraph { grid, goal };
    let starts = starts
        .into_iter()
        .filter(|&cell| is_open(grid, cell))
        .map(|cell| (cell, (0, 0)));
    let mut distance = OctileDistance(goal);
    let heuristic =
        crate::heuristic_fn(|&(cell, _): &(GridCell, GridStep)| distance.estimate(&cell));
    let result = a_star_multi(
        &graph,
        starts,
        &|&(cell, _): &(GridCell, GridStep)| cell == goal,
        heuristic,
        budget,
    );

    let outcome = match result.outcome {
        Outcome::Found(solution) => Outcome::Found(Solution {
            states: solution
                .states
                .into_iter()
                .map(|(cell, _)| cell)
                .collect::<Vec<_>>(),
            actions: solution.actions,
            cost: solution.cost,
        }),
        Outcome::Unsolvable => Outcome::Unsolvable,
        Outcome::LimitReached => Outcome::LimitReached,
        Outcome::Cutoff => Outcome::Cutoff,
    };

    SearchResult {
        outcome,
        stats: result.stats,
    }
}

/// The jump graph of a grid: nodes are `(cell, arrival direction)` pairs, `(0, 0)` for a
/// start, and edges lead to the jump points of the pruned neighbours.
struct JumpGraph<'a, G> {
    /// The grid.
    grid: &'a G,
    /// The goal cell; every jump stops there.
    goal: GridCell,
}

impl<G: UniformGrid> Successors for JumpGraph<'_, G> {
    type State = (GridCell, GridStep);
    type Action = GridStep;
    type Cost = Octile;

    fn successors(
        &self,
        &(cell, arrival): &(GridCell, GridStep),
        mut emit: impl FnMut(GridStep, (GridCell, GridStep), Octile),
    ) {
        for step in self.pruned_steps(cell, arrival) {
            if let Some((point, moves)) = self.jump(cell, step) {
                let cost = if step.0 == 0 || step.1 == 0 {
                    Octile::new(moves, 0)
                } else {
                    Octile::new(0, moves)
                };
                emit(step, (point, step), cost);
            }
        }
    }
}

impl<G: UniformGrid> JumpGraph<'_, G> {
    /// The legal steps from `cell` worth trying after arriving in direction `arrival`.
    fn pruned_steps(&self, cell: GridCell, (dx, dy): GridStep) -> impl Iterator<Item = GridStep> {
        // The first `count` entries are the candidates; the rest is padding.
        let pad = (0, 0);
        let (candidates, count) = match (dx, dy) {
            (0, 0) => (crate::grid::STEPS, 8),
            (_, 0) => (
                [(dx, 0), (0, 1), (0, -1), (dx, 1), (dx, -1), pad, pad, pad],
                5,
            ),
            (0, _) => (
                [(0, dy), (1, 0), (-1, 0), (1, dy), (-1, dy), pad, pad, pad],
                5,
            ),
            _ => ([(dx, 0), (0, dy), (dx, dy), pad, pad, pad, pad, pad], 3),
        };

        candidates
            .into_iter()
            .take(count)
            .filter(move |&step| can_move(self.grid, cell, step))
    }

    /// Jumps from `cell` in direction `step`: the next jump point and the number of moves
    /// to it, or `None` if the jump runs into a wall or off the grid first.
    fn jump(&self, cell: GridCell, step: GridStep) -> Option<(GridCell, u32)> {
        if step.0 == 0 || step.1 == 0 {
            self.jump_straight(cell, step)
        } else {
            self.jump_diagonal(cell, step)
        }
    }

    /// A straight jump: walks until the goal or a cell with a forced neighbour.
    fn jump_straight(&self, mut cell: GridCell, step: GridStep) -> Option<(GridCell, u32)> {
        let mut moves = 0;
        loop {
            if !passable(self.grid, cell, step) {
                return None;
            }

            cell = offset(self.grid, cell, step).expect("a passable cell is on the grid");
            moves += 1;
            if cell == self.goal || self.has_forced_neighbour(cell, step) {
                return Some((cell, moves));
            }
        }
    }

    /// A diagonal jump: walks until the goal or a cell from which a straight jump along
    /// either component stops.
    fn jump_diagonal(&self, mut cell: GridCell, step: GridStep) -> Option<(GridCell, u32)> {
        let mut moves = 0;
        loop {
            if !can_move(self.grid, cell, step) {
                return None;
            }

            cell = offset(self.grid, cell, step).expect("a passable cell is on the grid");
            moves += 1;
            if cell == self.goal
                || self.jump_straight(cell, (step.0, 0)).is_some()
                || self.jump_straight(cell, (0, step.1)).is_some()
            {
                return Some((cell, moves));
            }
        }
    }

    /// Returns `true` if `cell`, reached by the straight `step`, has a forced neighbour: a
    /// side cell that is open while the side cell behind it is blocked.
    fn has_forced_neighbour(&self, cell: GridCell, (dx, dy): GridStep) -> bool {
        let (side, back) = ((dy, dx), (-dx, -dy));
        [1, -1].into_iter().any(|sign| {
            let lateral = (side.0 * sign, side.1 * sign);
            passable(self.grid, cell, lateral)
                && !passable(self.grid, cell, (lateral.0 + back.0, lateral.1 + back.1))
        })
    }
}
