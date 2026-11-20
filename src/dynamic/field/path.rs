//! Path extraction for Field D*: a walk from the start that, at every point, takes the
//! step with the smallest cost plus interpolated `g` at its end ([`extract`]).
//!
//! From a corner, the candidates are those of the eight neighbour pairs ([`interpolate`]),
//! plus the plain straight and diagonal moves. From a point inside a cell edge, they are
//! the moves along that edge to either end and, for each of the two cells bordering it,
//! the best point on each of the cell's other three edges (a convex one-dimensional
//! minimisation with a closed form, [`best_on_edge`]).
//!
//! Every step must strictly lower the interpolated value. A corner's value is its `g`,
//! and a settled corner always has such a step, since its `g` is the value of one of its
//! pairs. An edge point without one (interpolation is not exact) moves along its edge to
//! the end with the smaller `g`, which is at most the point's value. So the corners on the
//! walk have strictly falling `g` and each is visited at most once, and between two
//! corners the walk takes at most a fixed number of edge-to-edge steps.

use alloc::vec;

use search_core::NonNegF64;

use super::{Corners, Pair, Via, cell_cost, coordinate, interpolate};
use crate::grid::{CostGrid, GridCell, sqrt};
use crate::outcome::Solution;

/// Parameters this close to an edge's end snap to that corner.
const SNAP: f64 = 1e-9;

/// A point of the walk: a corner, or a point strictly inside a unit cell edge.
#[derive(Debug, Clone, Copy, PartialEq)]
enum At {
    /// A cell corner.
    Corner(GridCell),
    /// The point `t` (in `(0, 1)`) of the way along the unit edge from `from` to
    /// `from + (1, 0)` (if `horizontal`) or `from + (0, 1)`.
    Edge {
        /// The edge's lower-left end.
        from: GridCell,
        /// Whether the edge runs along `x`.
        horizontal: bool,
        /// The position along the edge, strictly between `0` and `1`.
        t: f64,
    },
}

impl At {
    /// The point `t` of the way from corner `a` to the adjacent corner `b` (one straight
    /// unit step apart), snapped to `a` or `b` within [`SNAP`].
    fn between(a: GridCell, b: GridCell, t: f64) -> Self {
        if t <= SNAP {
            return Self::Corner(a);
        }

        if t >= 1.0 - SNAP {
            return Self::Corner(b);
        }

        let horizontal = a.1 == b.1;
        if a <= b {
            Self::Edge {
                from: a,
                horizontal,
                t,
            }
        } else {
            Self::Edge {
                from: b,
                horizontal,
                t: 1.0 - t,
            }
        }
    }

    /// The point's coordinates.
    fn point(self) -> (f64, f64) {
        match self {
            Self::Corner((x, y)) => (coordinate(x), coordinate(y)),
            Self::Edge {
                from: (x, y),
                horizontal,
                t,
            } => {
                let (x, y) = (coordinate(x), coordinate(y));
                if horizontal { (x + t, y) } else { (x, y + t) }
            }
        }
    }

    /// The interpolated `g` at the point: `g` at a corner, the linear interpolation of the
    /// ends' `g` inside an edge (infinite if either end is).
    fn value(self, g: &impl Fn(GridCell) -> f64) -> f64 {
        match self {
            Self::Corner(corner) => g(corner),
            Self::Edge {
                from,
                horizontal,
                t,
            } => {
                let (g0, g1) = (g(from), g(edge_end(from, horizontal)));
                if g0.is_finite() && g1.is_finite() {
                    (1.0 - t) * g0 + t * g1
                } else {
                    f64::INFINITY
                }
            }
        }
    }
}

/// A candidate step of the walk.
#[derive(Debug, Clone, Copy)]
struct Step {
    /// A point the step passes on the way (Field D*'s bend along an edge), if any.
    bend: Option<At>,
    /// Where the step ends.
    to: At,
    /// The step's true cost: each segment's length times the cost of the cell it crosses,
    /// or of the cheaper cell along the edge it follows.
    cost: f64,
    /// The interpolated `g` at [`Step::to`].
    value: f64,
}

impl Step {
    /// The step's estimated total: its cost plus the value where it ends.
    fn total(&self) -> f64 {
        self.cost + self.value
    }
}

/// The walk from `start` to `goal` over the interpolated values `g` (infinite where
/// unknown), or `None` if `start` has no finite value.
///
/// Returns the points from `start` to `goal` and the sum of the segment costs.
pub(super) fn extract<G: CostGrid>(
    grid: &G,
    g: impl Fn(GridCell) -> f64,
    start: GridCell,
    goal: GridCell,
) -> Option<Solution<(f64, f64), (), NonNegF64>> {
    let corners = Corners(grid);
    let mut at = At::Corner(start);
    let mut value = g(start);
    if !corners.on_lattice(start) || !value.is_finite() {
        return None;
    }

    let mut states = vec![at.point()];
    let mut cost = 0.0;
    let mut edge_steps = 0;
    // A straight line crosses at most `width + height` cells between two corners.
    let cap = 2 * (grid.width() + grid.height()) + 8;
    while at != At::Corner(goal) {
        let step = match at {
            At::Corner(corner) => from_corner(&corners, &g, corner, value),
            At::Edge { .. } if edge_steps < cap => {
                from_edge(&corners, &g, at, value).or_else(|| to_lower_end(&corners, &g, at))
            }
            At::Edge { .. } => {
                debug_assert!(false, "Field D* path extraction kept crossing cells");
                to_lower_end(&corners, &g, at)
            }
        }?;

        edge_steps = if matches!(step.to, At::Edge { .. }) {
            edge_steps + 1
        } else {
            0
        };
        states.extend(step.bend.map(At::point));
        states.push(step.to.point());
        cost += step.cost;
        at = step.to;
        value = step.value;
    }

    Some(Solution {
        actions: vec![(); states.len() - 1],
        states,
        cost: NonNegF64::new(cost).expect("segment costs are non-negative"),
    })
}

/// The best step from `corner` (value `value`) that strictly lowers the value: over each
/// neighbour pair, Field D*'s choice and the plain straight and diagonal moves.
fn from_corner<G: CostGrid>(
    corners: &Corners<'_, G>,
    g: &impl Fn(GridCell) -> f64,
    corner: GridCell,
    value: f64,
) -> Option<Step> {
    let steps = corners
        .pairs(corner)
        .flat_map(|pair| pair_steps(corner, pair, g))
        .flatten();
    let best = best(corners, g, steps, value);

    debug_assert!(
        best.is_some(),
        "a settled corner has a step that lowers its value"
    );

    best
}

/// The steps from `corner` through one neighbour pair: straight to `s1`, diagonally to
/// `s2` (if on the lattice), and Field D*'s interpolated choice if it is a crossing or a
/// bend.
fn pair_steps(corner: GridCell, pair: Pair, g: &impl Fn(GridCell) -> f64) -> [Option<Step>; 3] {
    let (g1, g2) = (g(pair.straight), pair.diagonal.map_or(f64::INFINITY, g));
    let straight = Step {
        bend: None,
        to: At::Corner(pair.straight),
        cost: pair.c.min(pair.b),
        value: g1,
    };
    let diagonal = pair.diagonal.map(|s2| Step {
        bend: None,
        to: At::Corner(s2),
        cost: pair.c * core::f64::consts::SQRT_2,
        value: g2,
    });
    let interpolated = match (interpolate(g1, g2, pair.c, pair.b).1, pair.diagonal) {
        (Via::Across(y), Some(s2)) => Some(across(pair.straight, s2, y, pair, g)),
        (Via::Bend(x), Some(s2)) => Some(bend(corner, s2, x, pair, g)),
        _ => None,
    };

    [Some(straight), diagonal, interpolated]
}

/// Field D*'s step across cell `c` from a corner to the point `y` of the way from its
/// straight neighbour `s1` to its diagonal neighbour `s2`.
fn across(s1: GridCell, s2: GridCell, y: f64, pair: Pair, g: &impl Fn(GridCell) -> f64) -> Step {
    let to = At::between(s1, s2, y);
    let cost = match to {
        At::Corner(corner) if corner == s1 => pair.c.min(pair.b),
        At::Corner(_) => pair.c * core::f64::consts::SQRT_2,
        At::Edge { .. } => pair.c * sqrt(1.0 + y * y),
    };

    Step {
        bend: None,
        to,
        cost,
        value: to.value(g),
    }
}

/// Field D*'s step from `corner` along the edge to its straight neighbour for a distance
/// `x`, then across cell `c` to the diagonal neighbour `s2`.
fn bend(corner: GridCell, s2: GridCell, x: f64, pair: Pair, g: &impl Fn(GridCell) -> f64) -> Step {
    let via = At::between(corner, pair.straight, x);
    let rest = 1.0 - x;
    let (bend, cost, to) = match via {
        At::Corner(at) if at == corner => (None, pair.c * core::f64::consts::SQRT_2, s2),
        At::Corner(_) => (None, pair.c.min(pair.b), pair.straight),
        At::Edge { .. } => (
            Some(via),
            pair.b.min(pair.c) * x + pair.c * sqrt(1.0 + rest * rest),
            s2,
        ),
    };

    Step {
        bend,
        to: At::Corner(to),
        cost,
        value: g(to),
    }
}

/// The best step from the edge point `at` (value `value`) that strictly lowers the value,
/// among [`edge_steps`]. `None` if no step lowers the value.
fn from_edge<G: CostGrid>(
    corners: &Corners<'_, G>,
    g: &impl Fn(GridCell) -> f64,
    at: At,
    value: f64,
) -> Option<Step> {
    best(corners, g, edge_steps(corners, g, at), value)
}

/// The steps from the edge point `at`: along the edge to either end, and across each of
/// the two bordering cells to the best point on each of its other three edges.
fn edge_steps<'a, G: CostGrid>(
    corners: &'a Corners<'_, G>,
    g: &'a impl Fn(GridCell) -> f64,
    at: At,
) -> impl Iterator<Item = Step> + 'a {
    let At::Edge {
        from, horizontal, ..
    } = at
    else {
        unreachable!("only edge points have edge steps");
    };

    let end = edge_end(from, horizontal);
    let crossings = bordering_cells(from, horizontal)
        .into_iter()
        .flatten()
        .flat_map(move |cell| {
            let cost = cell_cost(corners.0, cell.0, cell.1);
            cell_edges(cell)
                .into_iter()
                .filter(move |&edge| edge != (from, horizontal) && cost.is_finite())
                .filter_map(move |edge| cross(at, edge, cost, g))
                .filter(move |step| step.to != At::Corner(from) && step.to != At::Corner(end))
        });

    along(corners, g, at).into_iter().chain(crossings)
}

/// The step among `steps` that strictly lowers the value below `value` with the smallest
/// cost plus [`ahead`] estimate.
fn best<G: CostGrid>(
    corners: &Corners<'_, G>,
    g: &impl Fn(GridCell) -> f64,
    steps: impl Iterator<Item = Step>,
    value: f64,
) -> Option<Step> {
    steps
        .filter(|step| step.value < value && step.total().is_finite())
        .map(|step| (step.cost + ahead(corners, g, step.to), step))
        .filter(|(rank, _)| rank.is_finite())
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, step)| step)
}

/// The cost to the goal estimated from `at`: `g` at a corner; at an edge point, the best
/// cost plus value over its own [`edge_steps`]. The interpolated value alone is too
/// optimistic on an edge whose other side is blocked, where a path can only follow the
/// edge or turn back.
fn ahead<G: CostGrid>(corners: &Corners<'_, G>, g: &impl Fn(GridCell) -> f64, at: At) -> f64 {
    match at {
        At::Corner(corner) => g(corner),
        At::Edge { .. } => edge_steps(corners, g, at)
            .map(|step| step.total())
            .fold(f64::INFINITY, f64::min),
    }
}

/// The fallback from the edge point `at`: along the edge to the end with the smaller `g`,
/// which is at most the point's value. `None` if both ends are unreached or the edge
/// borders only blocked cells.
fn to_lower_end<G: CostGrid>(
    corners: &Corners<'_, G>,
    g: &impl Fn(GridCell) -> f64,
    at: At,
) -> Option<Step> {
    along(corners, g, at)
        .into_iter()
        .filter(|step| step.total().is_finite())
        .min_by(|a, b| a.value.total_cmp(&b.value))
}

/// The two steps from the edge point `at` along its edge to either end, at the cost of
/// the cheaper bordering cell.
fn along<G: CostGrid>(corners: &Corners<'_, G>, g: &impl Fn(GridCell) -> f64, at: At) -> [Step; 2] {
    let At::Edge {
        from,
        horizontal,
        t,
    } = at
    else {
        unreachable!("only edge points move along their edge");
    };

    let rate = bordering_cells(from, horizontal)
        .into_iter()
        .flatten()
        .map(|cell| cell_cost(corners.0, cell.0, cell.1))
        .fold(f64::INFINITY, f64::min);
    let end = edge_end(from, horizontal);
    let step = |corner: GridCell, distance: f64| Step {
        bend: None,
        to: At::Corner(corner),
        cost: rate * distance,
        value: g(corner),
    };

    [step(from, t), step(end, 1.0 - t)]
}

/// The step from the edge point `at` straight across a cell of cost `cost` to the best
/// point of the cell edge `(from, horizontal)`, or `None` if neither end of that edge has
/// a finite `g`.
fn cross(
    at: At,
    (from, horizontal): (GridCell, bool),
    cost: f64,
    g: &impl Fn(GridCell) -> f64,
) -> Option<Step> {
    let end = edge_end(from, horizontal);
    let (g0, g1) = (g(from), g(end));
    let (px, py) = at.point();
    let (ex, ey) = At::Corner(from).point();
    let (along, distance) = if horizontal {
        (px - ex, (py - ey).abs())
    } else {
        (py - ey, (px - ex).abs())
    };

    let t = match (g0.is_finite(), g1.is_finite()) {
        (true, true) => best_on_edge(along, distance, g1 - g0, cost),
        (true, false) => 0.0,
        (false, true) => 1.0,
        (false, false) => return None,
    };

    let to = At::between(from, end, t);
    let (qx, qy) = to.point();
    let (dx, dy) = (qx - px, qy - py);

    Some(Step {
        bend: None,
        to,
        cost: cost * sqrt(dx * dx + dy * dy),
        value: to.value(g),
    })
}

/// The parameter `t` in `[0, 1]` that minimises `cost * √(distance² + (t - along)²) +
/// slope * t`: the cost of crossing a cell from a point at perpendicular `distance` from
/// an edge (and at position `along` in the edge's direction) to the edge's point `t`,
/// plus the interpolated `g` there (up to a constant).
///
/// The function is convex. Where `|slope| < cost`, its stationary point is
/// `along - slope * distance / √(cost² - slope²)`, clamped to the edge; otherwise it is
/// monotone and the minimum is at an end.
fn best_on_edge(along: f64, distance: f64, slope: f64, cost: f64) -> f64 {
    if slope >= cost {
        return 0.0;
    }

    if slope <= -cost {
        return 1.0;
    }

    (along - slope * distance / sqrt(cost * cost - slope * slope)).clamp(0.0, 1.0)
}

/// The far end of the unit edge from `from`: `from + (1, 0)` if `horizontal`, else
/// `from + (0, 1)`.
fn edge_end((x, y): GridCell, horizontal: bool) -> GridCell {
    if horizontal { (x + 1, y) } else { (x, y + 1) }
}

/// The cells on either side of the unit edge from `from`: above and below a horizontal
/// edge, right and left of a vertical one. `None` for a side off the grid's lower or left
/// border; cells past the other borders are returned and cost infinity.
fn bordering_cells((x, y): GridCell, horizontal: bool) -> [Option<GridCell>; 2] {
    if horizontal {
        [Some((x, y)), y.checked_sub(1).map(|below| (x, below))]
    } else {
        [Some((x, y)), x.checked_sub(1).map(|left| (left, y))]
    }
}

/// The four edges of cell `(x, y)` as `(from, horizontal)`: bottom, top, left, right.
fn cell_edges((x, y): GridCell) -> [(GridCell, bool); 4] {
    [
        ((x, y), true),
        ((x, y + 1), true),
        ((x, y), false),
        ((x + 1, y), false),
    ]
}

#[cfg(test)]
mod tests {
    //! The closed-form minimiser against a fine scan.

    use proptest::prelude::*;

    use super::{best_on_edge, sqrt};

    /// The function [`best_on_edge`] minimises.
    fn objective(t: f64, along: f64, distance: f64, slope: f64, cost: f64) -> f64 {
        let offset = t - along;

        cost * sqrt(distance * distance + offset * offset) + slope * t
    }

    proptest! {
        /// No point of a fine grid over `[0, 1]` beats the closed form.
        #[test]
        fn closed_form_is_the_minimum(
            along in -1.0_f64..2.0,
            distance in 0.01_f64..1.0,
            slope in -5.0_f64..5.0,
            cost in 0.1_f64..4.0,
        ) {
            let t = best_on_edge(along, distance, slope, cost);
            let best = objective(t, along, distance, slope, cost);
            for i in 0..=1000_u32 {
                let other = objective(f64::from(i) / 1000.0, along, distance, slope, cost);
                prop_assert!(best <= other + 1e-9);
            }
        }
    }
}
