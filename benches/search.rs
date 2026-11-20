//! Benchmarks for the hot paths: A*, the anytime, bounded-suboptimal and multi-heuristic
//! searches, `LazySP` and focal search on a grid with a wall; the focal list itself; NBS;
//! D* Lite and AD* repairs after the wall changes; and the depth-first searches on a
//! small open grid, where tree searches meet many transpositions; Jump Point Search
//! against A* on the 8-connected version of the wall grid.
//!
//! Run with `cargo bench`.

use std::hint::black_box;

use criterion::Criterion;
use heuristic_search::{
    AdStar, DStarLite, DynamicGraph, EdgeSelector, EesEstimates, EightConnected, FocalList, Limits,
    MhaWeights, OctileDistance, Predecessors, Successors, UniformGrid, Weight, a_star, ana_star,
    ara_star, beam_search, beam_stack_search, dfs, dps, ees, focal_fn, focal_search, fringe_search,
    heuristic_fn, ida_star, ida_star_tt, iddfs, jps, lazy_sp, mha_star, nbs, rbfs,
};

/// Side length of the benchmark grid.
const SIZE: usize = 64;

/// A grid cell as `(row, column)`.
type Cell = (usize, usize);

/// A square 4-connected grid with unit moves and a wall across most of the middle row.
struct Grid {
    /// `true` marks a wall, row-major.
    walls: Vec<bool>,
}

impl Grid {
    /// The benchmark grid: a horizontal wall at row `SIZE / 2` with a gap at the right end.
    fn new() -> Self {
        let mut walls = vec![false; SIZE * SIZE];
        for column in 0..SIZE - 2 {
            walls[(SIZE / 2) * SIZE + column] = true;
        }

        Self { walls }
    }

    /// Calls `emit` for every free neighbour of a free `cell`.
    fn neighbours(&self, &(row, column): &Cell, mut emit: impl FnMut(Cell)) {
        if self.walls[row * SIZE + column] {
            return;
        }

        let candidates = [
            row.checked_sub(1).map(|r| (r, column)),
            (row + 1 < SIZE).then_some((row + 1, column)),
            column.checked_sub(1).map(|c| (row, c)),
            (column + 1 < SIZE).then_some((row, column + 1)),
        ];
        for cell in candidates.into_iter().flatten() {
            if !self.walls[cell.0 * SIZE + cell.1] {
                emit(cell);
            }
        }
    }
}

impl Successors for Grid {
    type State = Cell;
    type Action = ();
    type Cost = u32;

    fn successors(&self, cell: &Cell, mut emit: impl FnMut((), Cell, u32)) {
        self.neighbours(cell, |next| emit((), next, 1));
    }
}

impl Predecessors for Grid {
    type Node = Cell;
    type Action = ();
    type Cost = u32;

    fn predecessors(&self, cell: &Cell, mut emit: impl FnMut((), Cell, u32)) {
        self.neighbours(cell, |previous| emit((), previous, 1));
    }
}

impl DynamicGraph for Grid {
    type Vertex = Cell;
    type Cost = u32;

    fn successors(&self, cell: &Cell, mut emit: impl FnMut(Cell, u32)) {
        self.neighbours(cell, |next| emit(next, 1));
    }

    fn predecessors(&self, cell: &Cell, mut emit: impl FnMut(Cell, u32)) {
        self.neighbours(cell, |previous| emit(previous, 1));
    }
}

/// The same grid for the 8-connected searches; cells are `(x, y) = (column, row)`.
impl UniformGrid for Grid {
    fn width(&self) -> usize {
        SIZE
    }

    fn height(&self) -> usize {
        SIZE
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        !self.walls[y * SIZE + x]
    }
}

/// The Manhattan distance between two cells.
fn manhattan(a: &Cell, b: &Cell) -> u32 {
    u32::try_from(a.0.abs_diff(b.0) + a.1.abs_diff(b.1)).expect("the grid is small")
}

/// The start cell, top left.
const START: Cell = (0, 0);

/// The goal cell, bottom left, behind the wall.
const GOAL: Cell = (SIZE - 1, 0);

/// A* with the Manhattan heuristic around the wall.
fn bench_a_star(c: &mut Criterion, grid: &Grid) {
    c.bench_function("a_star grid", |b| {
        b.iter(|| {
            let h = heuristic_fn(|cell: &Cell| Some(manhattan(cell, &GOAL)));
            black_box(a_star(
                grid,
                START,
                &|cell: &Cell| *cell == GOAL,
                h,
                Limits::NONE,
            ))
        });
    });
}

/// Fringe search with the Manhattan heuristic around the wall.
fn bench_fringe_search(c: &mut Criterion, grid: &Grid) {
    c.bench_function("fringe_search grid", |b| {
        b.iter(|| {
            black_box(fringe_search(
                grid,
                START,
                &at_goal,
                to_goal(),
                Limits::NONE,
            ))
        });
    });
}

/// Beam search with width 16, and beam-stack search with width 64 run to completion.
///
/// Narrow beams make beam-stack search backtrack a lot on this grid: width 16 takes about
/// a minute to prove the optimum.
fn bench_beam(c: &mut Criterion, grid: &Grid) {
    c.bench_function("beam_search grid w=16", |b| {
        b.iter(|| {
            black_box(beam_search(
                grid,
                START,
                &at_goal,
                to_goal(),
                16,
                Limits::NONE,
            ))
        });
    });
    c.bench_function("beam_stack_search grid w=64", |b| {
        b.iter(|| {
            black_box(beam_stack_search(
                grid,
                START,
                &at_goal,
                to_goal(),
                64,
                |_| {},
                Limits::NONE,
            ))
        });
    });
}

/// Jump Point Search against A* on the same 8-connected grid, around the wall.
fn bench_jps(c: &mut Criterion, grid: &Grid) {
    // `(x, y)`: from the top left to the bottom left, behind the wall.
    let (start, goal) = ((0, 0), (0, SIZE - 1));
    c.bench_function("jps grid", |b| {
        b.iter(|| black_box(jps(grid, start, goal, Limits::NONE)));
    });
    c.bench_function("a_star eight-connected grid", |b| {
        b.iter(|| {
            black_box(a_star(
                &EightConnected(grid),
                start,
                &|cell: &Cell| *cell == goal,
                OctileDistance(goal),
                Limits::NONE,
            ))
        });
    });
}

/// Focal search with weight 2 and the Manhattan distance as focal key.
fn bench_focal_search(c: &mut Criterion, grid: &Grid) {
    c.bench_function("focal_search grid w=2", |b| {
        b.iter(|| {
            let h = heuristic_fn(|cell: &Cell| Some(manhattan(cell, &GOAL)));
            let key = focal_fn(|cell: &Cell| manhattan(cell, &GOAL));
            black_box(focal_search(
                grid,
                START,
                &|cell: &Cell| *cell == GOAL,
                h,
                key,
                Weight::integer(2),
                Limits::NONE,
            ))
        });
    });
}

/// Pushes 10 000 items with pseudo-random `f` and keys, then pops them all.
fn bench_focal_list(c: &mut Criterion) {
    c.bench_function("FocalList push+pop 10k", |b| {
        b.iter(|| {
            let mut list = FocalList::new(Weight::new(3, 2));
            let mut x: u32 = 12_345;
            for item in 0..10_000 {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                list.push(item, 100 + (x >> 20) % 100, 0, (x >> 8) % 16);
            }

            while let Some(item) = list.pop() {
                black_box(item);
            }
        });
    });
}

/// Pushes 10 000 items, re-pushes every item with a lower `f` (decrease-key), and pops
/// every second step, as a best-first search with reopening does.
fn bench_focal_list_churn(c: &mut Criterion) {
    c.bench_function("FocalList churn 10k", |b| {
        b.iter(|| {
            let mut list = FocalList::new(Weight::new(3, 2));
            let mut x: u32 = 54_321;
            for item in 0..10_000 {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                list.push(item, 200 + (x >> 20) % 100, 0, (x >> 8) % 16);
            }

            for item in 0..10_000 {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                list.push(item, 100 + (x >> 20) % 150, 1, (x >> 8) % 16);
                if item % 2 == 0 {
                    black_box(list.pop());
                }
            }

            while let Some(item) = list.pop() {
                black_box(item);
            }
        });
    });
}

/// The goal test of the wall grid.
fn at_goal(cell: &Cell) -> bool {
    *cell == GOAL
}

/// The admissible Manhattan heuristic towards [`GOAL`].
fn to_goal() -> impl heuristic_search::Heuristic<Cell> + Copy {
    heuristic_fn(|cell: &Cell| Some(manhattan(cell, &GOAL)))
}

/// ARA* with the weight schedule 3, 2, 1.
fn bench_ara_star(c: &mut Criterion, grid: &Grid) {
    c.bench_function("ara_star grid 3,2,1", |b| {
        b.iter(|| {
            let weights = [Weight::integer(3), Weight::integer(2), Weight::ONE];
            black_box(ara_star(
                grid,
                START,
                &at_goal,
                to_goal(),
                weights,
                |_, _| {},
                Limits::NONE,
            ))
        });
    });
}

/// ANA* until it proves its last solution optimal.
fn bench_ana_star(c: &mut Criterion, grid: &Grid) {
    c.bench_function("ana_star grid", |b| {
        b.iter(|| {
            black_box(ana_star(
                grid,
                START,
                &at_goal,
                to_goal(),
                |_, _| {},
                Limits::NONE,
            ))
        });
    });
}

/// EES with weight 2, the Manhattan distance for all three estimates.
fn bench_ees(c: &mut Criterion, grid: &Grid) {
    c.bench_function("ees grid w=2", |b| {
        b.iter(|| {
            let estimates = EesEstimates {
                admissible: to_goal(),
                cost_to_go: to_goal(),
                distance_to_go: to_goal(),
            };
            black_box(ees(
                grid,
                START,
                &at_goal,
                estimates,
                Weight::integer(2),
                Limits::NONE,
            ))
        });
    });
}

/// DPS with weight 2.
fn bench_dps(c: &mut Criterion, grid: &Grid) {
    c.bench_function("dps grid w=2", |b| {
        b.iter(|| {
            black_box(dps(
                grid,
                START,
                &at_goal,
                to_goal(),
                Weight::integer(2),
                Limits::NONE,
            ))
        });
    });
}

/// Shared MHA* with the Manhattan anchor and one inadmissible guide (twice Manhattan).
fn bench_mha_star(c: &mut Criterion, grid: &Grid) {
    c.bench_function("mha_star grid w1=w2=2", |b| {
        b.iter(|| {
            let guide = heuristic_fn(|cell: &Cell| Some(2 * manhattan(cell, &GOAL)));
            let weights = MhaWeights {
                heuristic: Weight::integer(2),
                anchor: Weight::integer(2),
            };
            black_box(mha_star(
                grid,
                START,
                &at_goal,
                to_goal(),
                [guide],
                weights,
                Limits::NONE,
            ))
        });
    });
}

/// `LazySP` where cells in the right half truly cost 2 to leave, evaluating whole paths.
fn bench_lazy_sp(c: &mut Criterion, grid: &Grid) {
    c.bench_function("lazy_sp grid all", |b| {
        b.iter(|| {
            let evaluate =
                |from: &Cell, (): &(), _: &Cell| Some(if from.1 >= SIZE / 2 { 2 } else { 1 });
            black_box(lazy_sp(
                grid,
                START,
                &at_goal,
                to_goal(),
                evaluate,
                EdgeSelector::All,
                Limits::NONE,
            ))
        });
    });
}

/// Moves the wall's gap: closes the right end and opens a cell at column 5, reporting
/// every affected edge to `changed(from, to)`.
fn move_gap(grid: &mut Grid, mut changed: impl FnMut(&Grid, &Cell, &Cell)) {
    let (closed, opened) = ((SIZE / 2, SIZE - 1), (SIZE / 2, 5));
    grid.walls[closed.0 * SIZE + closed.1] = true;
    grid.walls[opened.0 * SIZE + opened.1] = false;
    for cell in [closed, opened] {
        let around = [
            (cell.0 - 1, cell.1),
            (cell.0 + 1, cell.1),
            (cell.0, cell.1 - 1),
            (cell.0, (cell.1 + 1).min(SIZE - 1)),
        ];
        for neighbour in around {
            changed(grid, &neighbour, &cell);
            changed(grid, &cell, &neighbour);
        }
    }
}

/// AD*: a plan at weight 3, the wall's gap moves, then plans at weights 2 and 1.
fn bench_ad_star(c: &mut Criterion) {
    c.bench_function("ad_star replan grid 3,2,1", |b| {
        b.iter(|| {
            let mut grid = Grid::new();
            let mut planner = AdStar::new(&grid, START, GOAL, manhattan, Weight::integer(3));
            black_box(planner.compute(&grid, Limits::NONE));

            move_gap(&mut grid, |grid, from, to| {
                planner.edge_changed(grid, from, to);
            });
            for weight in [Weight::integer(2), Weight::ONE] {
                planner.set_weight(weight);
                black_box(planner.compute(&grid, Limits::NONE));
            }
        });
    });
}

/// Side length of the open grid for the depth-first searches.
const OPEN: usize = 6;

/// An open `OPEN x OPEN` grid with unit moves: many paths reach each cell.
struct Open;

impl Successors for Open {
    type State = Cell;
    type Action = ();
    type Cost = u32;

    fn successors(&self, &(row, column): &Cell, mut emit: impl FnMut((), Cell, u32)) {
        let candidates = [
            row.checked_sub(1).map(|r| (r, column)),
            (row + 1 < OPEN).then_some((row + 1, column)),
            column.checked_sub(1).map(|c| (row, c)),
            (column + 1 < OPEN).then_some((row, column + 1)),
        ];
        for cell in candidates.into_iter().flatten() {
            emit((), cell, 1);
        }
    }
}

/// The far corner of [`Open`].
const CORNER: Cell = (OPEN - 1, OPEN - 1);

/// DFS on the wall grid, and IDDFS, IDA* and IDA* with a transposition table on the open
/// grid from one corner to the other.
fn bench_depth_first(c: &mut Criterion, grid: &Grid) {
    c.bench_function("dfs grid", |b| {
        b.iter(|| black_box(dfs(grid, START, &at_goal, Limits::NONE)));
    });

    let corner = |cell: &Cell| *cell == CORNER;
    // Half the Manhattan distance: admissible but weak, so IDA* needs several iterations
    // and meets many transpositions.
    let h = || heuristic_fn(|cell: &Cell| Some(manhattan(cell, &CORNER) / 2));
    c.bench_function("iddfs open 6x6", |b| {
        b.iter(|| black_box(iddfs(&Open, START, &corner, Limits::NONE)));
    });
    c.bench_function("ida_star open 6x6", |b| {
        b.iter(|| black_box(ida_star(&Open, START, &corner, h(), Limits::NONE)));
    });
    c.bench_function("ida_star_tt open 6x6 cap=256", |b| {
        b.iter(|| black_box(ida_star_tt(&Open, START, &corner, h(), 256, Limits::NONE)));
    });
    c.bench_function("rbfs open 6x6", |b| {
        b.iter(|| black_box(rbfs(&Open, START, &corner, h(), Limits::NONE)));
    });
}

/// NBS with the Manhattan heuristic in both directions.
fn bench_nbs(c: &mut Criterion, grid: &Grid) {
    c.bench_function("nbs grid", |b| {
        b.iter(|| {
            let forward = heuristic_fn(|cell: &Cell| Some(manhattan(cell, &GOAL)));
            let backward = heuristic_fn(|cell: &Cell| Some(manhattan(&START, cell)));
            black_box(nbs(grid, START, [GOAL], forward, backward, Limits::NONE))
        });
    });
}

/// D* Lite: one full plan, then the repair after the wall's gap closes and another opens.
fn bench_d_star_lite(c: &mut Criterion) {
    c.bench_function("d_star_lite replan grid", |b| {
        b.iter(|| {
            let mut grid = Grid::new();
            let mut planner = DStarLite::new(&grid, START, GOAL, manhattan);
            black_box(planner.compute(&grid, Limits::NONE));

            move_gap(&mut grid, |grid, from, to| {
                planner.edge_changed(grid, from, to);
            });

            black_box(planner.compute(&grid, Limits::NONE))
        });
    });
}

/// Runs every benchmark.
fn main() {
    let grid = Grid::new();
    let mut c = Criterion::default().configure_from_args();

    bench_a_star(&mut c, &grid);
    bench_fringe_search(&mut c, &grid);
    bench_beam(&mut c, &grid);
    bench_jps(&mut c, &grid);
    bench_focal_search(&mut c, &grid);
    bench_focal_list(&mut c);
    bench_focal_list_churn(&mut c);
    bench_nbs(&mut c, &grid);
    bench_d_star_lite(&mut c);
    bench_ara_star(&mut c, &grid);
    bench_ana_star(&mut c, &grid);
    bench_ees(&mut c, &grid);
    bench_dps(&mut c, &grid);
    bench_mha_star(&mut c, &grid);
    bench_lazy_sp(&mut c, &grid);
    bench_ad_star(&mut c);
    bench_depth_first(&mut c, &grid);

    c.final_summary();
}
