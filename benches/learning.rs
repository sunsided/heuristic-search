//! Learning curves of the real-time searches: trials to convergence against the lookahead,
//! on random 4-connected mazes with the Manhattan heuristic, and the offline sweeps of the
//! limited-horizon update against its horizon.
//!
//! Not a timing benchmark: it prints tables (the numbers in `docs/design.md`). Run with
//! `cargo bench --bench learning`.

use heuristic_search::{
    Heuristic, LearnedHeuristic, Limits, Successors, TrialResult, Zero, dijkstra, heuristic_fn,
    lhb_lrta_star, lhb_update, lrta_star, lss_lrta_star, rtaa_star,
};

/// Side length of the mazes.
const SIZE: usize = 24;

/// Number of random mazes each number is averaged over.
const MAZES: u64 = 8;

/// Trials after which a learner counts as not converged.
const MAX_TRIALS: u64 = 10_000;

/// The lookaheads of the curves.
const LOOKAHEADS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// A grid cell as `(row, column)`.
type Cell = (usize, usize);

/// A square 4-connected maze with unit moves, start top left and goal bottom right.
struct Maze {
    /// `true` marks a wall, row-major.
    walls: Vec<bool>,
}

impl Maze {
    /// A maze with about 30 % walls from `seed`, regenerated until the goal is reachable.
    fn random(mut seed: u64) -> Self {
        loop {
            let walls = (0..SIZE * SIZE)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (seed >> 33) % 10 < 3
                })
                .collect::<Vec<_>>();
            let mut maze = Self { walls };
            maze.walls[0] = false;
            maze.walls[SIZE * SIZE - 1] = false;

            if maze.optimum().is_some() {
                return maze;
            }
        }
    }

    /// The cheapest cost from [`START`] to [`GOAL`], by Dijkstra.
    fn optimum(&self) -> Option<u32> {
        dijkstra(self, START, &at_goal, Limits::NONE)
            .into_solution()
            .map(|solution| solution.cost)
    }

    /// Returns `true` if the goal is reachable from `cell`. Cells in walled-off pockets are
    /// left out of the offline sweeps, whose values there would grow forever.
    fn reaches_goal(&self, cell: Cell) -> bool {
        self.is_free(cell)
            && dijkstra(self, cell, &at_goal, Limits::NONE)
                .solution()
                .is_some()
    }

    /// Returns `true` if `cell` is free.
    fn is_free(&self, (row, column): Cell) -> bool {
        row < SIZE && column < SIZE && !self.walls[row * SIZE + column]
    }
}

impl Successors for Maze {
    type State = Cell;
    type Action = ();
    type Cost = u32;

    fn successors(&self, &(row, column): &Cell, mut emit: impl FnMut((), Cell, u32)) {
        let candidates = [
            row.checked_sub(1).map(|r| (r, column)),
            Some((row + 1, column)),
            column.checked_sub(1).map(|c| (row, c)),
            Some((row, column + 1)),
        ];
        for cell in candidates.into_iter().flatten() {
            if self.is_free(cell) {
                emit((), cell, 1);
            }
        }
    }
}

/// The start cell, top left.
const START: Cell = (0, 0);

/// The goal cell, bottom right.
const GOAL: Cell = (SIZE - 1, SIZE - 1);

/// The goal test.
fn at_goal(cell: &Cell) -> bool {
    *cell == GOAL
}

/// The consistent Manhattan heuristic towards [`GOAL`].
fn manhattan() -> impl Heuristic<Cell> + Copy {
    heuristic_fn(|&(row, column): &Cell| {
        u32::try_from(GOAL.0.abs_diff(row) + GOAL.1.abs_diff(column)).ok()
    })
}

/// A learner under measurement.
#[derive(Debug, Clone, Copy)]
enum Learner {
    /// [`lrta_star`].
    Lrta,
    /// [`lss_lrta_star`].
    Lss,
    /// [`lhb_lrta_star`].
    Lhb,
    /// [`rtaa_star`].
    Rtaa,
}

impl Learner {
    /// One trial with `lookahead` (ignored by LRTA*).
    fn trial<H: Heuristic<Cell>>(
        self,
        maze: &Maze,
        table: &mut LearnedHeuristic<Cell, u32, H>,
        lookahead: usize,
    ) -> TrialResult<Cell, (), u32> {
        match self {
            Self::Lrta => lrta_star(maze, START, &at_goal, table, Limits::NONE),
            Self::Lss => lss_lrta_star(maze, START, &at_goal, table, lookahead, Limits::NONE),
            Self::Lhb => lhb_lrta_star(maze, START, &at_goal, table, lookahead, Limits::NONE),
            Self::Rtaa => rtaa_star(maze, START, &at_goal, table, lookahead, Limits::NONE),
        }
    }
}

/// What one learner did until it converged on one maze.
#[derive(Debug, Clone, Copy, Default)]
struct Curve {
    /// Trials until one learned nothing, that one included.
    trials: u64,
    /// Cost of the first trial's path.
    first_cost: u64,
    /// Expansions over all trials.
    expansions: u64,
}

impl Curve {
    /// The sum of two curves, for averaging.
    fn plus(self, other: Self) -> Self {
        Self {
            trials: self.trials + other.trials,
            first_cost: self.first_cost + other.first_cost,
            expansions: self.expansions + other.expansions,
        }
    }
}

/// Runs trials of `learner` on `maze` until one learns nothing, and checks that its path is
/// a cheapest one.
fn converge(learner: Learner, maze: &Maze, lookahead: usize) -> Curve {
    let mut table = LearnedHeuristic::new(manhattan());
    let mut curve = Curve::default();

    for trial in 1..=MAX_TRIALS {
        let result = learner.trial(maze, &mut table, lookahead);
        let cost = result.solution().expect("the goal is reachable").cost;
        curve.expansions += result.stats.expanded;
        if trial == 1 {
            curve.first_cost = u64::from(cost);
        }

        if result.trial.updates == 0 {
            assert_eq!(
                Some(cost),
                maze.optimum(),
                "{learner:?} converged off-optimum"
            );
            curve.trials = trial;
            return curve;
        }
    }

    panic!("{learner:?} with lookahead {lookahead} did not converge");
}

/// Sweeps of the limited-horizon update over every free cell, from a zero table, until
/// nothing changes; returns the sweeps (the last included) and the expansions.
fn offline_sweeps(maze: &Maze, horizon: usize) -> (u64, u64) {
    let cells: Vec<Cell> = (0..SIZE)
        .flat_map(|row| (0..SIZE).map(move |column| (row, column)))
        .filter(|&cell| maze.reaches_goal(cell))
        .collect();
    let mut table = LearnedHeuristic::<Cell, u32, _>::new(Zero);

    for sweep in 1.. {
        let changed: u64 = cells
            .iter()
            .map(|&cell| lhb_update(maze, cell, &at_goal, &mut table, horizon))
            .sum();

        if changed == 0 {
            for cell in &cells {
                let exact = dijkstra(maze, *cell, &at_goal, Limits::NONE).into_solution();
                assert_eq!(table.get(cell), exact.map(|path| path.cost), "not exact");
            }

            return (sweep, sweep * cells.len() as u64 * horizon as u64);
        }
    }

    unreachable!("the loop only ends by returning")
}

/// Prints the averaged curve of every lookahead learner, and LRTA* once.
fn print_online(mazes: &[Maze]) {
    println!("Online: trials to convergence (mean over {MAZES} mazes {SIZE}x{SIZE}, Manhattan)");
    println!(
        "| lookahead | LSS-LRTA* trials | LHB trials | RTAA* trials | LSS first cost | RTAA* first cost | LSS expansions | RTAA* expansions |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");

    let lrta = mean(mazes, Learner::Lrta, 1);
    println!(
        "| LRTA* | {} | | | {} | | {} | |",
        lrta.trials, lrta.first_cost, lrta.expansions
    );

    for lookahead in LOOKAHEADS {
        let lss = mean(mazes, Learner::Lss, lookahead);
        let lhb = mean(mazes, Learner::Lhb, lookahead);
        let rtaa = mean(mazes, Learner::Rtaa, lookahead);
        println!(
            "| {lookahead} | {} | {} | {} | {} | {} | {} | {} |",
            lss.trials,
            lhb.trials,
            rtaa.trials,
            lss.first_cost,
            rtaa.first_cost,
            lss.expansions,
            rtaa.expansions
        );
    }
}

/// The curve of `learner` with `lookahead`, averaged over `mazes`.
fn mean(mazes: &[Maze], learner: Learner, lookahead: usize) -> Curve {
    let sum = mazes
        .iter()
        .map(|maze| converge(learner, maze, lookahead))
        .fold(Curve::default(), Curve::plus);

    Curve {
        trials: sum.trials / MAZES,
        first_cost: sum.first_cost / MAZES,
        expansions: sum.expansions / MAZES,
    }
}

/// Prints the offline sweeps of the limited-horizon update per horizon; horizon 1 is the
/// single-step Bellman update.
fn print_offline(mazes: &[Maze]) {
    println!();
    println!("Offline: sweeps of lhb_update over every cell until exact (mean, zero table)");
    println!("| horizon | sweeps | expansion bound |");
    println!("| --- | --- | --- |");

    for horizon in [1, 2, 4, 10, 50, 100] {
        let (sweeps, expansions) = mazes
            .iter()
            .map(|maze| offline_sweeps(maze, horizon))
            .fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
        println!(
            "| {horizon} | {} | {} |",
            sweeps / MAZES,
            expansions / MAZES
        );
    }
}

/// Prints both tables.
fn main() {
    let mazes: Vec<Maze> = (0..MAZES).map(|seed| Maze::random(seed + 1)).collect();
    let optimum: u32 = mazes.iter().filter_map(Maze::optimum).sum();
    println!("mean optimum {}", u64::from(optimum) / MAZES);

    print_online(&mazes);
    print_offline(&mazes);
}
