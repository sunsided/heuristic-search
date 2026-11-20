//! Line of sight between cell centres ([`LineOfSight`]), for any-angle search.

use core::cmp::Ordering;

use crate::grid::{GridCell, UniformGrid, is_open};

/// A [`UniformGrid`] that can tell whether the straight segment between two cell centres
/// is free, for any-angle searches such as `theta_star` (feature `float`).
///
/// **Rule:** the closed segment from the centre of `from` to the centre of `to` must not
/// touch any blocked cell, counting each cell as a closed unit square. A segment that
/// passes exactly through a grid corner therefore needs all four cells around it open,
/// which for a single diagonal move is exactly the no-corner-cutting rule of
/// [`UniformGrid`]. Both end cells must be passable.
///
/// The default method walks the cells the segment crosses in exact integer arithmetic
/// (a supercover Bresenham walk), in time linear in `|dx| + |dy|`. Implement it only to
/// make it faster (for example with precomputed clearance); a different rule voids the
/// guarantees of the searches that use it.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LineOfSight, UniformGrid};
///
/// /// A 5 x 5 grid with one blocked cell at (2, 2).
/// struct Pillar;
///
/// impl UniformGrid for Pillar {
///     fn width(&self) -> usize {
///         5
///     }
///
///     fn height(&self) -> usize {
///         5
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         (x, y) != (2, 2)
///     }
/// }
///
/// impl LineOfSight for Pillar {}
///
/// assert!(!Pillar.line_of_sight((0, 0), (4, 4)));
/// assert!(Pillar.line_of_sight((0, 0), (4, 1)));
/// // Passes exactly through the pillar's corner (2, 2).
/// assert!(!Pillar.line_of_sight((0, 3), (3, 0)));
/// ```
pub trait LineOfSight: UniformGrid {
    /// Returns `true` if the segment between the centres of `from` and `to` touches no
    /// blocked cell and both cells are passable (and on the grid).
    fn line_of_sight(&self, from: GridCell, to: GridCell) -> bool {
        supercover(self, from, to)
    }
}

impl<G: LineOfSight + ?Sized> LineOfSight for &G {
    fn line_of_sight(&self, from: GridCell, to: GridCell) -> bool {
        (**self).line_of_sight(from, to)
    }
}

/// The default [`LineOfSight::line_of_sight`]: walks every cell whose closed square the
/// segment between the centres touches.
///
/// With `width = |dx|` and `height = |dy|`, after `columns` horizontal and `rows` vertical
/// steps the segment next crosses a vertical cell border at parameter
/// `(2 columns + 1) / 2 width` and a horizontal one at `(2 rows + 1) / 2 height`. Comparing
/// `(2 columns + 1) height` with `(2 rows + 1) width` says which comes first; equality means
/// both at once, through a grid corner, where the two side cells are touched as well.
fn supercover<G: UniformGrid + ?Sized>(grid: &G, from: GridCell, to: GridCell) -> bool {
    if !is_open(grid, from) || !is_open(grid, to) {
        return false;
    }

    let signed = |value: usize| i128::try_from(value).expect("grid coordinates fit in i128");
    let (width, height) = (signed(from.0.abs_diff(to.0)), signed(from.1.abs_diff(to.1)));
    let step_x: i128 = if to.0 >= from.0 { 1 } else { -1 };
    let step_y: i128 = if to.1 >= from.1 { 1 } else { -1 };
    let open = |x: i128, y: i128| {
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return false;
        };

        is_open(grid, (x, y))
    };

    let (mut x, mut y) = (signed(from.0), signed(from.1));
    let (mut columns, mut rows) = (0, 0);
    while columns < width || rows < height {
        match ((2 * columns + 1) * height).cmp(&((2 * rows + 1) * width)) {
            Ordering::Equal => {
                if !open(x + step_x, y) || !open(x, y + step_y) {
                    return false;
                }

                x += step_x;
                y += step_y;
                columns += 1;
                rows += 1;
            }
            Ordering::Less => {
                x += step_x;
                columns += 1;
            }
            Ordering::Greater => {
                y += step_y;
                rows += 1;
            }
        }

        if !open(x, y) {
            return false;
        }
    }

    true
}
