//! The cheapest 8-connected path over the corners, for Field D*'s guarantee that its path
//! never costs more than that ([`cheapest`]).
//!
//! The corner lattice has the moves Field D*'s closed form contains as special cases:
//! straight to an adjacent corner at the cheaper bordering cell's cost, and diagonally
//! across a cell at `√2` times its cost. [`a_star`] searches it with `min_cost` times the
//! Euclidean distance as the heuristic, which every move's cost bounds from above, so the
//! heuristic is consistent and the result optimal. Given the cost of a path already in
//! hand, it stops once its smallest `f` reaches that cost: no lattice path is cheaper then.

use alloc::vec;

use search_core::NonNegF64;

use super::{Corners, RING, coordinate};
use crate::best_first::{a_star, a_star_below};
use crate::grid::{CostGrid, GridCell, sqrt};
use crate::outcome::{Limits, Solution};
use crate::problem::Successors;

/// The cheapest 8-connected path from `start` to `goal` over the corners, as points, or
/// `None` if there is none or, given a `bound`, none cheaper than `bound`. `min_cost` must
/// bound every cell cost from below.
pub(super) fn cheapest<G: CostGrid>(
    grid: &G,
    start: GridCell,
    goal: GridCell,
    min_cost: f64,
    bound: Option<NonNegF64>,
) -> Option<Solution<(f64, f64), (), NonNegF64>> {
    let heuristic = crate::heuristic_fn(|&corner: &GridCell| {
        let (dx, dy) = (
            coordinate(corner.0.abs_diff(goal.0)),
            coordinate(corner.1.abs_diff(goal.1)),
        );
        NonNegF64::new(min_cost * sqrt(dx * dx + dy * dy))
    });
    let (lattice, is_goal) = (Lattice(Corners(grid)), |&corner: &GridCell| corner == goal);
    let result = match bound {
        Some(bound) => a_star_below(&lattice, start, &is_goal, heuristic, bound),
        None => a_star(&lattice, start, &is_goal, heuristic, Limits::NONE),
    };
    let solution = result.into_solution()?;

    Some(Solution {
        actions: vec![(); solution.actions.len()],
        states: solution
            .states
            .into_iter()
            .map(|(x, y)| (coordinate(x), coordinate(y)))
            .collect(),
        cost: solution.cost,
    })
}

/// The 8-connected corner lattice of a grid.
struct Lattice<'g, G>(Corners<'g, G>);

impl<G: CostGrid> Successors for Lattice<'_, G> {
    type State = GridCell;
    type Action = ();
    type Cost = NonNegF64;

    /// The adjacent corners: straight at the cheaper of the two cells along the edge,
    /// diagonally at `√2` times the crossed cell's cost; none through blocked cells.
    fn successors(&self, &corner: &GridCell, mut emit: impl FnMut((), GridCell, NonNegF64)) {
        let (neighbours, cells) = self.0.around(corner);
        for (k, neighbour) in neighbours.into_iter().enumerate() {
            let Some(neighbour) = neighbour else {
                continue;
            };

            // Diagonal step `k` crosses cell `k / 2`; straight step `k` runs between the
            // cells of the diagonals on either side of it.
            let cost = if k % 2 == 1 {
                cells[k / 2] * core::f64::consts::SQRT_2
            } else {
                let (previous, next) = ((k + RING.len() - 1) % RING.len(), k + 1);
                cells[previous / 2].min(cells[next / 2])
            };
            if cost.is_finite() {
                emit(
                    (),
                    neighbour,
                    NonNegF64::new(cost).expect("cell costs are positive"),
                );
            }
        }
    }
}
