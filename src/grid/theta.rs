//! Any-angle search on grids: Theta* (Nash, Daniel, Koenig, Felner 2007) and Lazy Theta*
//! (Nash, Koenig, Tovey 2010), as [`theta_star`], [`lazy_theta_star`] and their
//! `*_multi` forms.
//!
//! Both are A* over the 8-connected moves of a [`LineOfSight`] grid that lets a path leave
//! the grid edges: when a cell `s'` is reached from `s`, it may take `s`'s *parent* as its
//! own parent if the two see each other, so paths become chains of straight segments
//! between cell centres at any angle. Costs are Euclidean lengths, so they are floats
//! ([`NonNegF64`], hence the `float` feature).
//!
//! | Algorithm           | Line-of-sight checks                          |
//! | ------------------- | --------------------------------------------- |
//! | [`theta_star`]      | for every generated neighbour                 |
//! | [`lazy_theta_star`] | once per expanded cell, repaired from neighbours if it fails |

use alloc::vec::Vec;

use search_core::{Budget, Cost, NonNegF64};

use crate::grid::{GridCell, LineOfSight, STEPS, can_move, is_open, offset};
use crate::open_list::OpenList;
use crate::outcome::{Outcome, SearchResult, Solution, Stats};

/// The result type of the any-angle searches: a path of cell centres joined by straight
/// segments, with its Euclidean length.
pub type AnyAngleResult = SearchResult<GridCell, (), NonNegF64>;

/// Theta* from `start` to `goal`: a short any-angle path between cell centres.
///
/// Searches the 8-connected moves of `grid` (no corner cutting) in A* order with the
/// Euclidean distance as heuristic, but a cell reached from `s` takes `s`'s parent as its
/// own parent whenever the two have [line of sight](LineOfSight). The path is not
/// guaranteed to be the shortest any-angle path, but it is never longer than the cheapest
/// 8-connected path (up to float rounding), never shorter than the straight line, and
/// every segment has line of sight.
///
/// The [`Solution`]'s states are the path's corners, from `start` to `goal`; its actions are
/// `()`, one per segment; its cost is the sum of the segment lengths. A blocked or off-grid
/// start is skipped and a blocked or off-grid goal is [`Outcome::Unsolvable`]. Memory is
/// `O(width * height)`, allocated up front. Square roots are computed without `std`, to
/// within a few units in the last place.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LineOfSight, Limits, UniformGrid, theta_star};
///
/// /// A 6 x 6 grid with a blocked cell at (2, 2).
/// struct Pillar;
///
/// impl UniformGrid for Pillar {
///     fn width(&self) -> usize {
///         6
///     }
///
///     fn height(&self) -> usize {
///         6
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         (x, y) != (2, 2)
///     }
/// }
///
/// impl LineOfSight for Pillar {}
///
/// let result = theta_star(&Pillar, (0, 0), (5, 1), Limits::NONE);
/// let solution = result.solution().expect("the grid is connected");
///
/// // One straight segment.
/// assert_eq!(solution.states, [(0, 0), (5, 1)]);
/// assert!((solution.cost.get() - 26_f64.sqrt()).abs() < 1e-12);
/// ```
pub fn theta_star<G: LineOfSight>(
    grid: &G,
    start: GridCell,
    goal: GridCell,
    budget: impl Budget<Stats>,
) -> AnyAngleResult {
    theta_star_multi(grid, [start], goal, budget)
}

/// [`theta_star`] from several start cells at once.
///
/// Blocked and off-grid starts are skipped, repeated starts count once, an empty set of
/// starts is [`Outcome::Unsolvable`], and a start that is the goal is returned as a
/// zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LineOfSight, Limits, UniformGrid, theta_star_multi};
///
/// /// An open 8 x 8 grid.
/// struct Open;
///
/// impl UniformGrid for Open {
///     fn width(&self) -> usize {
///         8
///     }
///
///     fn height(&self) -> usize {
///         8
///     }
///
///     fn is_passable(&self, _: usize, _: usize) -> bool {
///         true
///     }
/// }
///
/// impl LineOfSight for Open {}
///
/// let result = theta_star_multi(&Open, [(0, 0), (7, 7)], (7, 3), Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![(7, 7), (7, 3)]));
/// ```
pub fn theta_star_multi<G: LineOfSight>(
    grid: &G,
    starts: impl IntoIterator<Item = GridCell>,
    goal: GridCell,
    budget: impl Budget<Stats>,
) -> AnyAngleResult {
    AnyAngle::new(grid, goal, false).run(starts, budget)
}

/// Lazy Theta*: [`theta_star`] with far fewer line-of-sight checks.
///
/// A generated cell optimistically takes its generator's parent as parent, without
/// checking line of sight. The check happens once, when the cell is expanded; if it fails,
/// the cell's parent becomes its cheapest expanded neighbour. Paths are of similar length
/// as Theta*'s; every segment has line of sight and the path is never shorter than the
/// straight line. Unlike Theta*, the bound by the cheapest 8-connected path is not
/// guaranteed. Results, memory and arithmetic as for [`theta_star`].
///
/// # Examples
///
/// ```
/// use heuristic_search::{LineOfSight, Limits, UniformGrid, lazy_theta_star};
///
/// /// A 6 x 6 grid with a wall at x = 2 that leaves a gap at the bottom.
/// struct Wall;
///
/// impl UniformGrid for Wall {
///     fn width(&self) -> usize {
///         6
///     }
///
///     fn height(&self) -> usize {
///         6
///     }
///
///     fn is_passable(&self, x: usize, y: usize) -> bool {
///         x != 2 || y == 5
///     }
/// }
///
/// impl LineOfSight for Wall {}
///
/// let result = lazy_theta_star(&Wall, (0, 0), (5, 0), Limits::NONE);
/// let solution = result.solution().expect("the gap connects both sides");
///
/// assert!(solution.states.contains(&(1, 5)) || solution.states.contains(&(2, 5)));
/// ```
pub fn lazy_theta_star<G: LineOfSight>(
    grid: &G,
    start: GridCell,
    goal: GridCell,
    budget: impl Budget<Stats>,
) -> AnyAngleResult {
    lazy_theta_star_multi(grid, [start], goal, budget)
}

/// [`lazy_theta_star`] from several start cells at once, with the start handling of
/// [`theta_star_multi`].
///
/// # Examples
///
/// ```
/// use heuristic_search::{LineOfSight, Limits, UniformGrid, lazy_theta_star_multi};
///
/// /// An open 8 x 8 grid.
/// struct Open;
///
/// impl UniformGrid for Open {
///     fn width(&self) -> usize {
///         8
///     }
///
///     fn height(&self) -> usize {
///         8
///     }
///
///     fn is_passable(&self, _: usize, _: usize) -> bool {
///         true
///     }
/// }
///
/// impl LineOfSight for Open {}
///
/// let result = lazy_theta_star_multi(&Open, [(0, 0), (7, 7)], (7, 3), Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![(7, 7), (7, 3)]));
/// ```
pub fn lazy_theta_star_multi<G: LineOfSight>(
    grid: &G,
    starts: impl IntoIterator<Item = GridCell>,
    goal: GridCell,
    budget: impl Budget<Stats>,
) -> AnyAngleResult {
    AnyAngle::new(grid, goal, true).run(starts, budget)
}

/// The parent of a cell that has none yet.
const NO_PARENT: usize = usize::MAX;

/// One Theta* or Lazy Theta* run, over dense per-cell arrays indexed `y * width + x`.
struct AnyAngle<'a, G> {
    /// The grid.
    grid: &'a G,
    /// The goal cell.
    goal: GridCell,
    /// Whether this is Lazy Theta* (check line of sight on expansion).
    lazy: bool,
    /// Best known path length per cell; infinite if not reached.
    g: Vec<NonNegF64>,
    /// Parent index per cell; [`NO_PARENT`] if not reached, the cell itself for a start.
    parent: Vec<usize>,
    /// Whether each cell has been expanded.
    closed: Vec<bool>,
    /// Open cells by `f = g + h`.
    open: OpenList<NonNegF64>,
    /// Counters for the result.
    stats: Stats,
}

impl<'a, G: LineOfSight> AnyAngle<'a, G> {
    /// Prepares a run towards `goal`.
    fn new(grid: &'a G, goal: GridCell, lazy: bool) -> Self {
        let cells = grid.width() * grid.height();

        Self {
            grid,
            goal,
            lazy,
            g: alloc::vec![NonNegF64::INFINITY; cells],
            parent: alloc::vec![NO_PARENT; cells],
            closed: alloc::vec![false; cells],
            open: OpenList::new(),
            stats: Stats::default(),
        }
    }

    /// Searches from `starts` until the goal is expanded, the open list runs empty or
    /// `budget` is used up.
    fn run(
        mut self,
        starts: impl IntoIterator<Item = GridCell>,
        mut budget: impl Budget<Stats>,
    ) -> AnyAngleResult {
        if !is_open(self.grid, self.goal) {
            return self.finish(Outcome::Unsolvable);
        }

        for start in starts.into_iter().filter(|&cell| is_open(self.grid, cell)) {
            if start == self.goal {
                let solution = Solution {
                    states: alloc::vec![start],
                    actions: Vec::new(),
                    cost: NonNegF64::ZERO,
                };

                return self.finish(Outcome::Found(solution));
            }

            let id = self.id(start);
            if self.parent[id] == NO_PARENT {
                self.parent[id] = id;
                self.relax(id, NonNegF64::ZERO);
            }
        }

        loop {
            let (closed, g) = (&self.closed, &self.g);
            let Some(id) = self
                .open
                .pop_where(|id, entry| !closed[id] && entry == g[id])
                .map(|(id, _)| id)
            else {
                return self.finish(Outcome::Unsolvable);
            };

            if self.lazy {
                self.repair_parent(id);
            }

            if id == self.id(self.goal) {
                let solution = self.solution(id);
                return self.finish(Outcome::Found(solution));
            }

            if budget.exhausted(&self.stats) {
                return self.finish(Outcome::LimitReached);
            }

            self.expand(id);
        }
    }

    /// Closes `id` and updates its open or unreached neighbours.
    fn expand(&mut self, id: usize) {
        self.stats.expanded += 1;
        self.closed[id] = true;

        let cell = self.cell(id);
        for step in STEPS {
            if !can_move(self.grid, cell, step) {
                continue;
            }

            self.stats.generated += 1;
            let next =
                self.id(offset(self.grid, cell, step).expect("a legal move stays on the grid"));
            if !self.closed[next] {
                self.update(id, next);
            }
        }
    }

    /// Updates `next`, a neighbour of the expanded cell `id`: through `id`'s parent if
    /// that is allowed (always for Lazy Theta*, with line of sight for Theta*), else
    /// through `id`.
    fn update(&mut self, id: usize, next: usize) {
        let grandparent = self.parent[id];
        let shortcut = self.lazy
            || self
                .grid
                .line_of_sight(self.cell(grandparent), self.cell(next));
        let from = if shortcut { grandparent } else { id };
        let g = self.g[from] + self.distance(from, next);
        if g < self.g[next] {
            self.parent[next] = from;
            self.relax(next, g);
        }
    }

    /// Lazy Theta*'s check on expansion: if `id` cannot see its parent, takes the cheapest
    /// expanded neighbour as parent instead.
    fn repair_parent(&mut self, id: usize) {
        let parent = self.parent[id];
        if parent == id || self.grid.line_of_sight(self.cell(parent), self.cell(id)) {
            return;
        }

        let cell = self.cell(id);
        let best = STEPS
            .iter()
            .filter(|&&step| can_move(self.grid, cell, step))
            .filter_map(|&step| offset(self.grid, cell, step).map(|neighbour| self.id(neighbour)))
            .filter(|&neighbour| self.closed[neighbour])
            .map(|neighbour| (self.g[neighbour] + self.distance(neighbour, id), neighbour))
            .min_by(|a, b| a.0.cmp(&b.0))
            .expect("the cell was generated by an expanded neighbour");

        self.g[id] = best.0;
        self.parent[id] = best.1;
    }

    /// Sets the path length of `id` to `g` and opens it by `f = g + h`.
    fn relax(&mut self, id: usize, g: NonNegF64) {
        self.g[id] = g;
        let f = g + euclidean(self.cell(id), self.goal);
        self.open.push(id, f, g);
    }

    /// The path from a start to `id` along the parent links.
    fn solution(&self, id: usize) -> Solution<GridCell, (), NonNegF64> {
        let mut states = alloc::vec![self.cell(id)];
        let mut at = id;
        while self.parent[at] != at {
            at = self.parent[at];
            states.push(self.cell(at));
        }

        states.reverse();

        Solution {
            actions: alloc::vec![(); states.len() - 1],
            states,
            cost: self.g[id],
        }
    }

    /// The Euclidean distance between the centres of cells `a` and `b`.
    fn distance(&self, a: usize, b: usize) -> NonNegF64 {
        euclidean(self.cell(a), self.cell(b))
    }

    /// The dense index of `cell`.
    fn id(&self, (x, y): GridCell) -> usize {
        y * self.grid.width() + x
    }

    /// The cell of dense index `id`.
    fn cell(&self, id: usize) -> GridCell {
        (id % self.grid.width(), id / self.grid.width())
    }

    /// Wraps up the run with `outcome`.
    fn finish(self, outcome: Outcome<GridCell, (), NonNegF64>) -> AnyAngleResult {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}

/// The Euclidean distance between the centres of two cells.
fn euclidean(a: GridCell, b: GridCell) -> NonNegF64 {
    let (dx, dy) = (a.0.abs_diff(b.0) as u128, a.1.abs_diff(b.1) as u128);

    NonNegF64::new(sqrt(dx * dx + dy * dy)).expect("square roots are non-negative")
}

/// The square root of `n`, without `std`: Newton's method from an exponent-halving guess,
/// accurate to a few units in the last place.
#[expect(
    clippy::cast_precision_loss,
    reason = "squared grid distances; a rounded input is fine"
)]
fn sqrt(n: u128) -> f64 {
    if n == 0 {
        return 0.0;
    }

    let x = n as f64;
    // Halving the biased exponent gives a guess within a factor of two.
    let mut root = f64::from_bits((x.to_bits() >> 1) + (1023_u64 << 51));
    for _ in 0..6 {
        root = root.midpoint(x / root);
    }

    root
}
