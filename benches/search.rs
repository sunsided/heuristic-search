//! Benchmarks for the hot paths: A* and focal search on an open grid with a wall, the
//! focal list itself, NBS, and a D* Lite repair after a wall appears.
//!
//! Run with `cargo bench`.

use std::hint::black_box;

use criterion::Criterion;
use heuristic_search::{
    DStarLite, DynamicGraph, FocalList, Limits, Predecessors, Successors, Weight, astar, focal_fn,
    focal_search, heuristic_fn, nbs,
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

/// The Manhattan distance between two cells.
fn manhattan(a: &Cell, b: &Cell) -> u32 {
    u32::try_from(a.0.abs_diff(b.0) + a.1.abs_diff(b.1)).expect("the grid is small")
}

/// The start cell, top left.
const START: Cell = (0, 0);

/// The goal cell, bottom left, behind the wall.
const GOAL: Cell = (SIZE - 1, 0);

/// A* with the Manhattan heuristic around the wall.
fn bench_astar(c: &mut Criterion, grid: &Grid) {
    c.bench_function("astar grid", |b| {
        b.iter(|| {
            let h = heuristic_fn(|cell: &Cell| Some(manhattan(cell, &GOAL)));
            black_box(astar(
                grid,
                START,
                &|cell: &Cell| *cell == GOAL,
                h,
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
                    planner.edge_changed(&grid, &neighbour, &cell);
                    planner.edge_changed(&grid, &cell, &neighbour);
                }
            }

            black_box(planner.compute(&grid, Limits::NONE))
        });
    });
}

/// Runs every benchmark.
fn main() {
    let grid = Grid::new();
    let mut c = Criterion::default().configure_from_args();

    bench_astar(&mut c, &grid);
    bench_focal_search(&mut c, &grid);
    bench_focal_list(&mut c);
    bench_nbs(&mut c, &grid);
    bench_d_star_lite(&mut c);

    c.final_summary();
}
