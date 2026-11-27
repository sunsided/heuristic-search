//! Field D* (Ferguson and Stentz 2005, 2006): D* Lite on the corners of a grid with
//! per-cell costs, where a corner's cost is interpolated along the cell edges around it
//! ([`FieldDStar`]).
//!
//! The nodes are cell corners `(x, y)` with `x <= width` and `y <= height`; cell `(x, y)`
//! is the unit square between corners `(x, y)` and `(x + 1, y + 1)`. A path may leave a
//! corner through any point on the far edges of the four cells around it, so its value
//! `rhs(s)` is the minimum over the eight pairs of consecutive neighbours `(s1, s2)` of the
//! cost of the best path through the edge between them, with `g` interpolated linearly
//! along that edge ([`interpolate`]).
//!
//! The interpolation needs a closed form per pair, so `rhs` comes from the incremental
//! core's model hook ([`Searched::lookahead`]) rather than from single edges. The path is
//! extracted by the same interpolation from the start ([`path`]), and replaced by the
//! cheapest 8-connected corner path ([`lattice`]) where that is cheaper.

mod lattice;
mod path;

use search_core::{Budget, Cost, NonNegF64};

use crate::dynamic::incremental::{Estimate, Incremental, Searched};
use crate::grid::{CostGrid, GridCell, sqrt};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};

/// The result of [`FieldDStar::compute`]: the path as points `(x, y)` in grid
/// coordinates, joined by straight segments, with its cost.
pub type FieldResult = SearchResult<(f64, f64), (), NonNegF64>;

/// Field D* (Ferguson and Stentz 2005, 2006): D* Lite over the corners of a [`CostGrid`],
/// with costs interpolated along cell edges, so paths cross cells at any angle instead of
/// in 45 degree steps.
///
/// **Nodes and paths.** Nodes are cell corners `(x, y)` ([`GridCell`]) with `x <= width`
/// and `y <= height`; cell `(x, y)` spans corners `(x, y)` to `(x + 1, y + 1)`. A path is a
/// chain of points: corners and points on cell edges. Each segment lies in one closed
/// cell and costs its length times that cell's cost, or, along an edge between two cells,
/// times the cheaper of them. Blocked cells cannot be crossed, but a path may run along
/// their border (at the free side's cost) and pass a corner where two blocked cells touch
/// diagonally: unlike the searches over cells, Field D* has no corner rule.
///
/// **Costs.** The search computes `g`, a corner's interpolated cost to the goal: for each
/// of the eight pairs of consecutive neighbours `(s1, s2)` of a corner `s` (one straight,
/// one diagonal), the cheapest path from `s` straight to a point on the edge `s1 s2`, or
/// along the edge `s s1` and then across to `s2`, using the costs of the two cells that
/// border `s s1` and the linear interpolation of `g(s1)` and `g(s2)` along `s1 s2` (the
/// closed form of the papers). `g` never exceeds the cheapest 8-connected path over the
/// corners (straight moves at the cheaper bordering cell's cost, diagonal moves at `√2`
/// times the crossed cell's cost). [`FieldDStar::interpolated_cost`] returns `g` of the
/// start.
///
/// The returned path is extracted from `g` by the same interpolation, step by step from
/// the start; its [`Solution::cost`](crate::Solution::cost) is the true cost of that path,
/// the sum over its segments. Interpolation is not exact, so the path's true cost may
/// differ a little from `g`, in either direction; it is never below the straight-line
/// distance times the cheapest cell cost. The extracted path can cost a little more than
/// the cheapest 8-connected path (on about a million random grids up to 20 x 20 with a
/// fifth of the cells blocked, at most 3.5 % with costs in `[1, 5)` and 4.3 % with costs
/// in `[1, 100)`), so [`FieldDStar::compute`] also runs A* over the 8-connected corner
/// lattice and returns that path where it is cheaper. **The returned path therefore never
/// costs more than the cheapest 8-connected path**, and a path is returned whenever
/// [`FieldDStar::interpolated_cost`] is finite. On uniform grids it was at most 1.1 %
/// longer than Theta*'s path over the corners; the property tests allow 2 %.
///
/// Extraction walks from the start; at each point it takes the step with the smallest
/// cost plus estimated remainder that strictly lowers the interpolated value. From a
/// corner, the steps are those of its eight pairs; from a point on an edge, the moves
/// along the edge and across either bordering cell to the best point on another of its
/// edges. A step to an edge point is judged by the best step from there, which keeps the
/// walk off edges whose other side is blocked.
///
/// **Incremental use.** As with [`DStarLite`](crate::DStarLite), the search runs from the
/// goal, so the start can move ([`FieldDStar::move_start`]); after changing cell costs or
/// blocking cells in the grid, report each changed cell ([`FieldDStar::cell_changed`]) and
/// [`FieldDStar::compute`] again. Only the affected corners are repaired.
///
/// **Heuristic.** Keys use `min_cost / √2` times the Euclidean distance to the start,
/// where `min_cost` is a lower bound on every cell cost, now and after every change. The
/// `1 / √2` keeps keys ordered along the interpolation's dependencies; `min_cost = 0`
/// turns the heuristic off. Square roots are computed without `std`.
///
/// # Panics
///
/// [`FieldDStar::new`] panics if `min_cost` is infinite.
///
/// # Examples
///
/// ```
/// use heuristic_search::{CostGrid, FieldDStar, Limits, NonNegF64, UniformGrid};
///
/// /// A 4 x 3 grid of unit cost, with cells listed in `blocked` blocked.
/// struct Map {
///     blocked: Vec<(usize, usize)>,
/// }
///
/// impl UniformGrid for Map {
///     fn width(&self) -> usize {
///         4
///     }
///
///     fn height(&self) -> usize {
///         3
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         !self.blocked.contains(&(x, y))
///     }
/// }
///
/// impl CostGrid for Map {}
///
/// let mut map = Map { blocked: vec![] };
/// let one = NonNegF64::new(1.0).unwrap();
/// let mut planner = FieldDStar::new(&map, (0, 0), (4, 1), one);
///
/// // Any angle: close to the straight line of length √17, not 3 + √2.
/// let solution = planner.compute(&map, Limits::NONE).into_solution().expect("open grid");
/// assert!(solution.cost.get() < 3.0 + 2_f64.sqrt());
/// assert!(solution.cost.get() >= 17_f64.sqrt() - 1e-9);
/// assert_eq!(solution.states.first(), Some(&(0.0, 0.0)));
/// assert_eq!(solution.states.last(), Some(&(4.0, 1.0)));
///
/// // Block the bottom row's middle cells and repair.
/// map.blocked = vec![(1, 0), (2, 0)];
/// planner.cell_changed(&map, 1, 0);
/// planner.cell_changed(&map, 2, 0);
/// let detour = planner.compute(&map, Limits::NONE).into_solution().expect("still connected");
/// assert!(detour.cost > solution.cost);
/// ```
pub struct FieldDStar {
    /// The incremental search from the goal over corners.
    core: Incremental<GridCell, NonNegF64, Scaled>,
    /// The current start corner.
    start: GridCell,
    /// The goal corner.
    goal: GridCell,
    /// The lower bound on every cell cost.
    min_cost: f64,
}

impl FieldDStar {
    /// Prepares a search from corner `start` to corner `goal`, with `min_cost` a lower bound
    /// on every cell cost for the planner's lifetime (see [`FieldDStar`]). Nothing is
    /// expanded until [`FieldDStar::compute`]. A corner off the lattice (`x > width` or
    /// `y > height`) is never reached.
    ///
    /// # Panics
    ///
    /// Panics if `min_cost` is infinite.
    pub fn new<G: CostGrid>(
        _grid: &G,
        start: GridCell,
        goal: GridCell,
        min_cost: NonNegF64,
    ) -> Self {
        assert!(
            min_cost.get().is_finite(),
            "Field D* needs a finite lower bound on cell costs"
        );

        let heuristic = Scaled(min_cost.get() / core::f64::consts::SQRT_2);

        Self {
            core: Incremental::new(false, goal, start, heuristic, None),
            start,
            goal,
            min_cost: min_cost.get(),
        }
    }

    /// Repairs the search after the latest changes and returns the interpolated path from
    /// the current start, or [`Outcome::Unsolvable`] if the goal cannot be reached.
    ///
    /// The budget and the returned stats cover this call's repair only; see
    /// [`FieldDStar::stats`] for the totals. A search stopped by its budget continues
    /// where it stopped on the next call. Once the repair has settled, the path is
    /// extracted and compared with the 8-connected A* (see [`FieldDStar`]) outside the
    /// budget. That A* starts afresh on every call and stops once it cannot beat the
    /// interpolated path, but may still expand every corner once, so after a small change
    /// it can cost more than the repair.
    pub fn compute<G: CostGrid>(&mut self, grid: &G, budget: impl Budget<Stats>) -> FieldResult {
        let (settled, stats) = self.core.repair(&Corners(grid), budget);
        let outcome = if settled {
            let core = &self.core;
            let g = |corner: GridCell| core.consistent_g(&corner).map_or(f64::INFINITY, f64::from);
            match self.path(grid, g) {
                Some(solution) => Outcome::Found(solution),
                None => Outcome::Unsolvable,
            }
        } else {
            Outcome::LimitReached
        };

        SearchResult { outcome, stats }
    }

    /// The cheaper of the interpolated path from the settled values `g` and the cheapest
    /// 8-connected corner path, or `None` if `g(start)` is infinite.
    ///
    /// `g(start)` is at most the 8-connected optimum, so if it is infinite there is no
    /// 8-connected path either. If it is finite, there is one: `g(start)` is the closed
    /// form's fixed point, and every finite case of [`interpolate`] uses a neighbour with a
    /// finite, strictly smaller value that a lattice move of finite cost reaches (straight
    /// to `s1` at `min(c, b)`, or diagonally across `c` to `s2`), so following those moves
    /// reaches the goal. The result is never dearer than that path. The lattice search
    /// stops once it cannot beat the interpolated path; ties keep the interpolated one.
    fn path<G: CostGrid>(
        &self,
        grid: &G,
        g: impl Fn(GridCell) -> f64,
    ) -> Option<Solution<(f64, f64), (), NonNegF64>> {
        if !g(self.start).is_finite() {
            return None;
        }

        let interpolated = path::extract(grid, g, self.start, self.goal);
        let bound = interpolated.as_ref().map(|interpolated| interpolated.cost);
        let lattice = lattice::cheapest(grid, self.start, self.goal, self.min_cost, bound);
        debug_assert!(
            bound.is_some() || lattice.is_some(),
            "a finite g(start) implies an 8-connected path"
        );

        // Bounded by the interpolated cost, the lattice search only returns a cheaper path.
        lattice.or(interpolated)
    }

    /// The interpolated cost `g` from the current start to the goal, or `None` if the goal
    /// is unreachable. Never above the cheapest 8-connected path over the corners.
    ///
    /// Meaningful only right after a [`FieldDStar::compute`] that did not stop at its
    /// budget, and before the next [`FieldDStar::move_start`] or
    /// [`FieldDStar::cell_changed`]; otherwise it may be a stale value or `None`.
    #[must_use]
    pub fn interpolated_cost(&self) -> Option<NonNegF64> {
        self.core.consistent_g(&self.start)
    }

    /// Moves the start to corner `start`, typically to a point along the last path.
    pub fn move_start(&mut self, start: GridCell) {
        self.core.move_target(start);
        self.start = start;
    }

    /// Reports that cell `(x, y)` changed its cost or became blocked or passable in
    /// `grid`, which must already show the new state.
    ///
    /// Every pair of neighbours a corner interpolates over lies in the four cells around
    /// that corner, so exactly the cell's four corners depend on its cost; they are updated
    /// now, and the change spreads from them in the next [`FieldDStar::compute`].
    pub fn cell_changed<G: CostGrid>(&mut self, grid: &G, x: usize, y: usize) {
        let corners = Corners(grid);
        for corner in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
            if corners.on_lattice(corner) {
                self.core.edge_changed(&corners, &corner, &corner);
            }
        }
    }

    /// The goal corner.
    #[must_use]
    pub fn goal(&self) -> GridCell {
        self.goal
    }

    /// The current start corner.
    #[must_use]
    pub fn start(&self) -> GridCell {
        self.start
    }

    /// Counters over every [`FieldDStar::compute`] so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.core.total
    }
}

/// The heuristic: a factor times the Euclidean distance between two corners.
struct Scaled(f64);

impl Estimate<GridCell, NonNegF64> for Scaled {
    /// The factor times the Euclidean distance; exactly zero for a zero factor.
    fn estimate(&mut self, from: &GridCell, to: &GridCell) -> NonNegF64 {
        if self.0 == 0.0 {
            return NonNegF64::ZERO;
        }

        let (dx, dy) = (
            coordinate(from.0.abs_diff(to.0)),
            coordinate(from.1.abs_diff(to.1)),
        );

        NonNegF64::new(self.0 * sqrt(dx * dx + dy * dy)).expect("distances are non-negative")
    }
}

/// The corners of a [`CostGrid`] as the incremental core's model: each corner's
/// neighbours are the up to eight adjacent corners, and `rhs` is interpolated.
struct Corners<'g, G>(&'g G);

impl<G: CostGrid> Corners<'_, G> {
    /// Returns `true` if `corner` lies on the lattice: `x <= width` and `y <= height`.
    fn on_lattice(&self, (x, y): GridCell) -> bool {
        x <= self.0.width() && y <= self.0.height()
    }

    /// `corner + step` if it lies on the lattice.
    fn offset(&self, (x, y): GridCell, (dx, dy): (i8, i8)) -> Option<GridCell> {
        let corner = (
            x.checked_add_signed(isize::from(dx))?,
            y.checked_add_signed(isize::from(dy))?,
        );

        self.on_lattice(corner).then_some(corner)
    }

    /// The cost of the cell whose corners include `corner` and `corner + diagonal`, for a
    /// diagonal unit step; infinite if it is blocked or off the grid.
    fn cell_towards(&self, (x, y): GridCell, (dx, dy): (i8, i8)) -> f64 {
        let x = if dx < 0 { x.checked_sub(1) } else { Some(x) };
        let y = if dy < 0 { y.checked_sub(1) } else { Some(y) };

        match (x, y) {
            (Some(x), Some(y)) => cell_cost(self.0, x, y),
            _ => f64::INFINITY,
        }
    }

    /// The surroundings of `corner`: its neighbours in [`RING`] order (`None` off the
    /// lattice), and the costs of the four cells around it, indexed by `k / 2` for the
    /// diagonal step `RING[k]`.
    fn around(&self, corner: GridCell) -> ([Option<GridCell>; 8], [f64; 4]) {
        let neighbours = RING.map(|step| self.offset(corner, step));
        let cells = [1, 3, 5, 7].map(|k| self.cell_towards(corner, RING[k]));

        (neighbours, cells)
    }

    /// The eight pairs of consecutive neighbours of `corner` whose straight neighbour is
    /// on the lattice, with the costs of the two cells along the straight edge.
    fn pairs(&self, corner: GridCell) -> impl Iterator<Item = Pair> {
        let (neighbours, cells) = self.around(corner);

        (0..RING.len()).filter_map(move |i| {
            let (straight, diagonal, other) = pair_indices(i);

            Some(Pair {
                straight: neighbours[straight]?,
                diagonal: neighbours[diagonal],
                c: cells[diagonal / 2],
                b: cells[other / 2],
            })
        })
    }

    /// The interpolated `rhs` of `corner` from the `g` values of its neighbours (infinite
    /// where unknown): the minimum of [`interpolate`] over its eight pairs. Looks up each
    /// neighbour's `g` and each cell's cost once.
    fn rhs(&self, corner: GridCell, g: impl Fn(GridCell) -> f64) -> f64 {
        let (neighbours, cells) = self.around(corner);
        let values = neighbours.map(|neighbour| neighbour.map_or(f64::INFINITY, &g));

        (0..RING.len())
            .map(|i| {
                let (straight, diagonal, other) = pair_indices(i);
                let (c, b) = (cells[diagonal / 2], cells[other / 2]);
                interpolate(values[straight], values[diagonal], c, b).0
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// The [`RING`] indices of pair `i` (steps `i` and `i + 1`): its straight step, its
/// diagonal step, and the other diagonal next to the straight step, whose cell is the
/// second one along the straight edge.
fn pair_indices(i: usize) -> (usize, usize, usize) {
    let next = (i + 1) % RING.len();
    let (straight, diagonal) = if i % 2 == 0 { (i, next) } else { (next, i) };
    // The diagonal on the straight step's other side: straight + (straight - diagonal).
    let other = (2 * straight + RING.len() - diagonal) % RING.len();

    (straight, diagonal, other)
}

impl<G: CostGrid> Searched for Corners<'_, G> {
    type Vertex = GridCell;
    type Cost = NonNegF64;

    const LOOKAHEAD: bool = true;

    /// The adjacent corners. Their costs are unused, since `rhs` is interpolated, and
    /// reported as zero.
    fn successors(&self, corner: &GridCell, mut emit: impl FnMut(GridCell, NonNegF64)) {
        for step in RING {
            if let Some(next) = self.offset(*corner, step) {
                emit(next, NonNegF64::ZERO);
            }
        }
    }

    /// The adjacent corners, as for [`Searched::successors`]: the corners whose `rhs`
    /// interpolates over this one.
    fn predecessors(&self, corner: &GridCell, emit: impl FnMut(GridCell, NonNegF64)) {
        self.successors(corner, emit);
    }

    /// The interpolated `rhs` ([`Corners::rhs`]) as a cost; `None` if infinite.
    fn lookahead(
        &self,
        corner: &GridCell,
        g: impl Fn(&GridCell) -> Option<NonNegF64>,
    ) -> Option<NonNegF64> {
        let rhs = self.rhs(*corner, |other| g(&other).map_or(f64::INFINITY, f64::from));

        rhs.is_finite()
            .then(|| NonNegF64::new(rhs).expect("interpolated costs are non-negative"))
    }
}

/// The eight unit steps around a corner in counterclockwise order from east; straight
/// and diagonal steps alternate, so consecutive steps form the eight pairs.
const RING: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// A pair of consecutive neighbours of a corner `s`: the straight neighbour `s1`, the
/// diagonal neighbour `s2`, and the cells along the edge `s s1`.
#[derive(Debug, Clone, Copy)]
struct Pair {
    /// `s1`, the straight neighbour.
    straight: GridCell,
    /// `s2`, the diagonal neighbour; `None` if it is off the lattice.
    diagonal: Option<GridCell>,
    /// The cost of the cell with corners `s`, `s1` and `s2`; infinite if blocked or off
    /// the grid.
    c: f64,
    /// The cost of the other cell along the edge `s s1`; infinite if blocked or off the
    /// grid.
    b: f64,
}

/// How the cheapest path from a corner `s` leaves through a pair `(s1, s2)`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Via {
    /// Along the edge `s s1` to `s1`, at the cheaper of the two cells' costs.
    Straight,
    /// Across cell `c` to `s2`.
    Diagonal,
    /// Across cell `c` to the point at distance `y` (in `(0, 1)`) from `s1` towards `s2`.
    Across(f64),
    /// Along the edge `s s1` for a distance `x` (in `(0, 1)`) at cost `b`, then across
    /// cell `c` to `s2`.
    Bend(f64),
}

/// Field D*'s closed form for one pair: the cheapest cost from a corner `s` through the
/// edge between its straight neighbour `s1` (value `g1`) and diagonal neighbour `s2`
/// (value `g2`), with `g` interpolated linearly along `s1 s2`, `c` the cost of the cell
/// with corners `s`, `s1`, `s2` and `b` that of the other cell along `s s1`. Infinite
/// values (unreached corners, blocked cells) are allowed; the result is infinite when no
/// path leaves through the pair.
///
/// With `f = g1 - g2`: if `g1 <= g2`, go along `s s1`; if `f <= b`, cross `c` to the point
/// at `y = min(f / √(c² - f²), 1)` from `s1` (straight to `s2` if `c <= f`); otherwise go
/// along `s s1` for `x = 1 - min(b / √(c² - b²), 1)` and then across to `s2` (straight to
/// `s2` if `c <= b`). Each choice minimises the path's cost over its family, a convex
/// function of `y` or `x`.
fn interpolate(g1: f64, g2: f64, c: f64, b: f64) -> (f64, Via) {
    let edge = c.min(b);
    if edge == f64::INFINITY {
        return (f64::INFINITY, Via::Straight);
    }

    // Covers both infinite: inf <= inf.
    if g1 <= g2 || c == f64::INFINITY {
        return (edge + g1, Via::Straight);
    }

    // `g2` is finite here; `f` may be infinite.
    let f = g1 - g2;
    let diagonal = (c * core::f64::consts::SQRT_2 + g2, Via::Diagonal);
    if f <= b {
        if c <= f {
            return diagonal;
        }

        let y = (f / sqrt(c * c - f * f)).min(1.0);
        if y >= 1.0 {
            return diagonal;
        }

        return (c * sqrt(1.0 + y * y) + f * (1.0 - y) + g2, Via::Across(y));
    }

    if c <= b {
        return diagonal;
    }

    let x = 1.0 - (b / sqrt(c * c - b * b)).min(1.0);
    if x <= 0.0 {
        return diagonal;
    }

    let rest = 1.0 - x;

    (c * sqrt(1.0 + rest * rest) + b * x + g2, Via::Bend(x))
}

/// The cost of cell `(x, y)` as a float: infinite if blocked or off the grid.
fn cell_cost<G: CostGrid>(grid: &G, x: usize, y: usize) -> f64 {
    if x >= grid.width() || y >= grid.height() || !grid.is_passable(x, y) {
        return f64::INFINITY;
    }

    let cost = grid.cell_cost(x, y).get();
    debug_assert!(
        cost > 0.0 && cost.is_finite(),
        "cell costs must be finite and positive, got {cost} at ({x}, {y})"
    );

    cost
}

/// A grid coordinate as a float.
#[expect(
    clippy::cast_precision_loss,
    reason = "grid coordinates are far below 2^52"
)]
fn coordinate(value: usize) -> f64 {
    value as f64
}
