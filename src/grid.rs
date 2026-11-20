//! Searches specialised to 8-connected grids with uniform costs: the [`UniformGrid`]
//! trait, the exact octile cost [`Octile`], the plain grid graph [`EightConnected`], and
//! Jump Point Search ([`jps`]).
//!
//! A cell is `(x, y)` with `x < width` and `y < height`. A move goes to one of the eight
//! neighbours: a straight move costs `1`, a diagonal move `√2`.
//!
//! **Corner rule: no corner cutting.** A diagonal move from `(x, y)` to `(x + dx, y + dy)`
//! needs the target *and both* orthogonal neighbours `(x + dx, y)` and `(x, y + dy)` to be
//! passable, so a path never squeezes between two blocked cells that touch at a corner and
//! never clips the corner of a single blocked cell. Every search here over cells follows
//! this rule; `CostGrid` (feature `float`) adds per-cell costs for Field D*, which plans
//! over cell corners instead.

mod jps;
mod sight;
#[cfg(feature = "float")]
mod theta;

use core::cmp::Ordering;
use core::fmt;
use core::ops::Add;

use search_core::{Cost, Heuristic};

pub use jps::{jps, jps_multi};
pub use sight::LineOfSight;
#[cfg(feature = "float")]
pub use theta::{
    AnyAngleResult, lazy_theta_star, lazy_theta_star_multi, theta_star, theta_star_multi,
};

use crate::problem::Successors;
#[cfg(feature = "float")]
use search_core::NonNegF64;

/// A cell of a grid: `(x, y)`, column first.
pub type GridCell = (usize, usize);

/// A unit step `(dx, dy)` between neighbouring cells, each component `-1`, `0` or `1`.
pub type GridStep = (i8, i8);

/// A rectangular grid of cells that are passable or blocked, for 8-connected searches with
/// uniform costs.
///
/// A cell is `(x, y)` ([`GridCell`]) with `x < width` and `y < height`. A move goes to one
/// of the eight neighbours: a straight move costs `1`, a diagonal move `√2` (see
/// [`Octile`]).
///
/// **Corner rule: no corner cutting.** A diagonal move from `(x, y)` to `(x + dx, y + dy)`
/// needs the target *and both* orthogonal neighbours `(x + dx, y)` and `(x, y + dy)` to be
/// passable, so a path never squeezes between two blocked cells that touch at a corner and
/// never clips the corner of a single blocked cell. Every search over grid cells in this
/// crate follows this rule; Field D* (`FieldDStar`), which plans over cell corners, does
/// not.
///
/// # Examples
///
/// ```
/// use heuristic_search::UniformGrid;
///
/// /// A grid parsed from text rows, where `#` is blocked.
/// struct Map(Vec<Vec<bool>>);
///
/// impl UniformGrid for Map {
///     fn width(&self) -> usize {
///         self.0[0].len()
///     }
///
///     fn height(&self) -> usize {
///         self.0.len()
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         self.0[y][x]
///     }
/// }
///
/// let map = Map(["..#", "..."].iter().map(|row| row.chars().map(|c| c != '#').collect()).collect());
///
/// assert!(map.is_passable(1, 0));
/// assert!(!map.is_passable(2, 0));
/// ```
pub trait UniformGrid {
    /// The number of columns. Cells have `x < width()`.
    fn width(&self) -> usize;

    /// The number of rows. Cells have `y < height()`.
    fn height(&self) -> usize;

    /// Returns `true` if the cell `(x, y)` can be entered. Only called with `x < width()`
    /// and `y < height()`.
    fn is_passable(&self, x: usize, y: usize) -> bool;
}

impl<G: UniformGrid + ?Sized> UniformGrid for &G {
    fn width(&self) -> usize {
        (**self).width()
    }

    fn height(&self) -> usize {
        (**self).height()
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        (**self).is_passable(x, y)
    }
}

/// A [`UniformGrid`] whose passable cells have traversal costs, for Field D*
/// (`FieldDStar`, feature `float`).
///
/// A cell's cost is the cost per unit of length of a path through its interior. A path
/// along the border between two cells pays the cheaper of the two. Blocked cells
/// ([`UniformGrid::is_passable`] is `false`) and cells off the grid cost infinity.
///
/// The default cost is `1` for every passable cell, so `impl CostGrid for Map {}` makes a
/// uniform grid; the costs then are Euclidean lengths.
///
/// # Examples
///
/// ```
/// use heuristic_search::{CostGrid, NonNegF64, UniformGrid};
///
/// /// A 4 x 1 strip of mud (cost 3) at x = 2, otherwise grass (cost 1).
/// struct Strip;
///
/// impl UniformGrid for Strip {
///     fn width(&self) -> usize {
///         4
///     }
///
///     fn height(&self) -> usize {
///         1
///     }
///
///     fn is_passable(&self, _: usize, _: usize) -> bool {
///         true
///     }
/// }
///
/// impl CostGrid for Strip {
///     fn cell_cost(&self, x: usize, _: usize) -> NonNegF64 {
///         NonNegF64::new(if x == 2 { 3.0 } else { 1.0 }).expect("positive")
///     }
/// }
///
/// assert_eq!(Strip.cell_cost(2, 0).get(), 3.0);
/// ```
#[cfg(feature = "float")]
pub trait CostGrid: UniformGrid {
    /// The cost per unit of length of crossing the passable cell `(x, y)`: finite and
    /// strictly positive. Only called for passable cells on the grid.
    fn cell_cost(&self, x: usize, y: usize) -> NonNegF64 {
        let _ = (x, y);

        NonNegF64::new(1.0).expect("one is a valid cost")
    }
}

#[cfg(feature = "float")]
impl<G: CostGrid + ?Sized> CostGrid for &G {
    fn cell_cost(&self, x: usize, y: usize) -> NonNegF64 {
        (**self).cell_cost(x, y)
    }
}

/// Returns `true` if `cell + (dx, dy)` lies on the grid and is passable.
pub(crate) fn passable<G: UniformGrid + ?Sized>(
    grid: &G,
    cell: GridCell,
    (dx, dy): GridStep,
) -> bool {
    offset(grid, cell, (dx, dy)).is_some_and(|(x, y)| grid.is_passable(x, y))
}

/// `cell + (dx, dy)` if it lies on the grid.
pub(crate) fn offset<G: UniformGrid + ?Sized>(
    grid: &G,
    (x, y): GridCell,
    (dx, dy): GridStep,
) -> Option<GridCell> {
    let x = x.checked_add_signed(isize::from(dx))?;
    let y = y.checked_add_signed(isize::from(dy))?;

    (x < grid.width() && y < grid.height()).then_some((x, y))
}

/// Returns `true` if `cell` lies on the grid and is passable.
pub(crate) fn is_open<G: UniformGrid + ?Sized>(grid: &G, cell: GridCell) -> bool {
    passable(grid, cell, (0, 0))
}

/// Returns `true` if the move `step` from the passable `cell` is allowed: the target is
/// passable, and for a diagonal step both orthogonal neighbours are too.
pub(crate) fn can_move<G: UniformGrid + ?Sized>(
    grid: &G,
    cell: GridCell,
    (dx, dy): GridStep,
) -> bool {
    passable(grid, cell, (dx, dy))
        && (dx == 0 || dy == 0 || (passable(grid, cell, (dx, 0)) && passable(grid, cell, (0, dy))))
}

/// The eight unit steps, straight ones first.
pub(crate) const STEPS: [GridStep; 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

/// An exact octile cost `straight + diagonal * √2`: a number of straight and of diagonal
/// unit moves.
///
/// Integer approximations of `√2` (such as 10 and 14) make ties and comparisons wrong, so
/// this type keeps both counts and compares the real values exactly: `a + b√2 < c + d√2`
/// is decided by the signs of `a - c` and `d - b` and, when they differ, by comparing
/// `(a - c)²` with `2 (d - b)²` in integers. Since `√2` is irrational, two costs are equal
/// only if both counts are, so [`Ord`] agrees with [`Eq`].
///
/// [`Cost::saturating_add`] and [`Cost::saturating_times`] saturate each count at
/// `u32::MAX`; `+` panics on overflow in debug builds like `u32`.
///
/// # Examples
///
/// ```
/// use heuristic_search::Octile;
///
/// let two_straight = Octile::new(2, 0);
/// let one_diagonal = Octile::new(0, 1);
///
/// assert!(one_diagonal < two_straight);
/// assert!(Octile::new(3, 0) > Octile::new(0, 2));
/// assert_eq!(one_diagonal + one_diagonal, Octile::new(0, 2));
/// assert!((Octile::new(1, 1).to_f64() - (1.0 + core::f64::consts::SQRT_2)).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Octile {
    /// The number of straight unit moves.
    straight: u32,
    /// The number of diagonal unit moves.
    diagonal: u32,
}

impl Octile {
    /// One straight move: `1`.
    pub const STRAIGHT: Self = Self::new(1, 0);

    /// One diagonal move: `√2`.
    pub const DIAGONAL: Self = Self::new(0, 1);

    /// The cost of `straight` straight and `diagonal` diagonal moves.
    #[must_use]
    pub const fn new(straight: u32, diagonal: u32) -> Self {
        Self { straight, diagonal }
    }

    /// The number of straight moves.
    #[must_use]
    pub const fn straight(self) -> u32 {
        self.straight
    }

    /// The number of diagonal moves.
    #[must_use]
    pub const fn diagonal(self) -> u32 {
        self.diagonal
    }

    /// The octile distance between two cells: the cost of the cheapest path on an
    /// obstacle-free grid, `min(|dx|, |dy|)` diagonal moves and `||dx| - |dy||` straight
    /// ones. An admissible and consistent heuristic for 8-connected grids.
    ///
    /// # Panics
    ///
    /// Panics if a count exceeds `u32::MAX`.
    #[must_use]
    pub fn between((x0, y0): GridCell, (x1, y1): GridCell) -> Self {
        let (dx, dy) = (x0.abs_diff(x1), y0.abs_diff(y1));
        let count = |n: usize| u32::try_from(n).expect("grid distances fit in u32");

        Self::new(count(dx.max(dy) - dx.min(dy)), count(dx.min(dy)))
    }

    /// The cost as a float, `straight + diagonal * √2`, rounded.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.straight) + f64::from(self.diagonal) * core::f64::consts::SQRT_2
    }
}

impl Ord for Octile {
    fn cmp(&self, other: &Self) -> Ordering {
        // The sign of x + y√2 with integer x, y.
        let x = i128::from(self.straight) - i128::from(other.straight);
        let y = i128::from(self.diagonal) - i128::from(other.diagonal);
        match (x.cmp(&0), y.cmp(&0)) {
            (Ordering::Equal, sign) | (sign, Ordering::Equal) => sign,
            (Ordering::Greater, Ordering::Greater) => Ordering::Greater,
            (Ordering::Less, Ordering::Less) => Ordering::Less,
            (Ordering::Greater, Ordering::Less) => (x * x).cmp(&(2 * y * y)),
            (Ordering::Less, Ordering::Greater) => (2 * y * y).cmp(&(x * x)),
        }
    }
}

impl PartialOrd for Octile {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Add for Octile {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(
            self.straight + other.straight,
            self.diagonal + other.diagonal,
        )
    }
}

impl Cost for Octile {
    const ZERO: Self = Self::new(0, 0);

    fn saturating_add(self, other: Self) -> Self {
        Self::new(
            self.straight.saturating_add(other.straight),
            self.diagonal.saturating_add(other.diagonal),
        )
    }

    fn saturating_times(self, factor: u32) -> Self {
        Self::new(
            self.straight.saturating_mul(factor),
            self.diagonal.saturating_mul(factor),
        )
    }
}

impl fmt::Display for Octile {
    /// Formats as `a + b√2`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}√2", self.straight, self.diagonal)
    }
}

/// The octile distance to a fixed goal cell, as a [`Heuristic`] over cells: admissible and
/// consistent on every 8-connected grid.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Heuristic, Octile, OctileDistance};
///
/// let mut h = OctileDistance((4, 1));
///
/// assert_eq!(h.estimate(&(0, 0)), Some(Octile::new(3, 1)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OctileDistance(pub GridCell);

impl Heuristic<GridCell, Octile> for OctileDistance {
    fn estimate(&mut self, cell: &GridCell) -> Option<Octile> {
        Some(Octile::between(*cell, self.0))
    }
}

/// A [`UniformGrid`] as an 8-connected graph with [`Octile`] costs, for the general searches
/// ([`a_star`](crate::a_star) and the others) on the same moves that [`jps`] uses.
///
/// It implements [`Successors`] only. The searches that also need
/// [`CostExt`](crate::CostExt) on the cost ([`ana_star`](crate::ana_star),
/// [`dps`](crate::dps), [`dibbs`](crate::dibbs), [`bae_star`](crate::bae_star),
/// [`bidirectional_dijkstra`](crate::bidirectional_dijkstra)) or
/// [`Predecessors`](crate::Predecessors) do not run on it: `Octile` has no subtraction.
///
/// Edges follow the corner rule of [`UniformGrid`]; an edge's action is
/// its [`GridStep`]. Blocked cells and cells off the grid have no successors.
///
/// # Examples
///
/// ```
/// use heuristic_search::{EightConnected, Limits, Octile, OctileDistance, UniformGrid, a_star};
///
/// /// An open 5 x 3 grid.
/// struct Open;
///
/// impl UniformGrid for Open {
///     fn width(&self) -> usize {
///         5
///     }
///
///     fn height(&self) -> usize {
///         3
///     }
///
///     fn is_passable(&self, _: usize, _: usize) -> bool {
///         true
///     }
/// }
///
/// let goal = (4, 2);
/// let result = a_star(&EightConnected(Open), (0, 0), &|c: &(usize, usize)| *c == goal, OctileDistance(goal), Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(Octile::new(2, 2)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EightConnected<G>(pub G);

impl<G: UniformGrid> Successors for EightConnected<G> {
    type State = GridCell;
    type Action = GridStep;
    type Cost = Octile;

    fn successors(&self, cell: &GridCell, mut emit: impl FnMut(GridStep, GridCell, Octile)) {
        let grid = &self.0;
        if !is_open(grid, *cell) {
            return;
        }

        for step in STEPS {
            if !can_move(grid, *cell, step) {
                continue;
            }

            let target = offset(grid, *cell, step).expect("a passable target is on the grid");
            let cost = if step.0 == 0 || step.1 == 0 {
                Octile::STRAIGHT
            } else {
                Octile::DIAGONAL
            };
            emit(step, target, cost);
        }
    }
}

/// The square root of `x`, without `std`: Newton's method from an exponent-halving guess,
/// iterated until it stops decreasing, accurate to about one unit in the last place.
/// `0`, `+∞` and NaN are returned unchanged; negative inputs give NaN.
#[cfg(feature = "float")]
pub(crate) fn sqrt(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 || x == f64::INFINITY {
        return x;
    }

    if x < 0.0 {
        return f64::NAN;
    }

    // Halving the biased exponent gives a guess within a factor of two for normal inputs;
    // subnormal ones take a few more halving steps.
    let guess = f64::from_bits((x.to_bits() >> 1) + (1023_u64 << 51));
    // From any positive guess, one step lands at or above the root; from there the
    // iterates fall until rounding stops them.
    let mut root = guess.midpoint(x / guess);
    loop {
        let next = root.midpoint(x / root);
        if next >= root {
            return root;
        }

        root = next;
    }
}

#[cfg(all(test, feature = "float"))]
mod tests {
    //! The `no_std` square root against exact squares and the defining relation.

    use proptest::prelude::*;

    use super::sqrt;

    proptest! {
        /// Squares of integers come back exactly.
        #[test]
        fn exact_on_squares(n in 0_u32..1_000_000) {
            let n = f64::from(n);
            prop_assert_eq!(sqrt(n * n).to_bits(), n.to_bits());
        }

        /// Over many magnitudes, including subnormals, the root squared is within a few
        /// units in the last place of the input.
        #[test]
        fn root_squared_is_the_input(bits in 1_u64..0x7FEF_FFFF_FFFF_FFFF) {
            let x = f64::from_bits(bits);
            let root = sqrt(x);
            prop_assert!(root > 0.0);
            let relative = (root * root - x).abs() / x;
            let tolerance = if root * root < f64::MIN_POSITIVE { 1e-3 } else { 1e-15 };
            prop_assert!(relative <= tolerance, "sqrt({x}) = {root}");
        }
    }

    /// Zero, infinity and NaN pass through; negative inputs give NaN.
    #[test]
    fn special_values() {
        assert_eq!(sqrt(0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(sqrt(f64::INFINITY).to_bits(), f64::INFINITY.to_bits());
        assert!(sqrt(f64::NAN).is_nan());
        assert!(sqrt(-1.0).is_nan());
    }
}
