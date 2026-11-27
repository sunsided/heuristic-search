//! A planetary rover crossing unknown terrain: it only sees boulders and crater rims
//! within its sensor radius, replans as it learns, and must pick a move every control
//! cycle.
//!
//! The terrain is a generated 120 x 120 grid, 8-connected without corner cutting, with
//! boulders and long rims that each leave one gap. The rover starts with the freespace
//! assumption (every unseen cell is open) and drives from one corner to the other. Every
//! cycle it senses the cells within [`SENSOR`] cells, marks newly seen obstacles in its
//! map, reports every edge around them as changed, plans and moves one cell. The planners:
//!
//! - `DStarLite` with no limit: every cycle repairs its search until it has a cheapest
//!   path for the current map. Robots drive like this, but one cycle can take thousands of
//!   expansions when a rim turns out to be closed.
//! - `RealTimeDStar` with a hard limit of expansions per cycle, split by `local_share`
//!   between the D* Lite repair from the goal and an LSS-LRTA* lookahead around the rover.
//!   While the repair has not reached the rover, it moves on the lookahead's advice
//!   (a *partial* step) instead of stopping to think.
//!
//! The table shows the cycles until arrival, the share of fully informed cycles, the cost
//! driven against the hindsight optimum (the cheapest path on the true map, known only
//! afterwards), the most expansions in any cycle (the real-time guarantee) and the
//! expansions over the whole drive. The example asserts that every run arrives, that no
//! `RealTimeDStar` cycle exceeds its limit and that no run beats the hindsight optimum.
//!
//! Field D* (feature `float`) would replan any-angle paths over per-cell costs on the same
//! kind of map; it is left out to keep the comparison to one graph.
//!
//! Run with `cargo run --release --example rover_replanning`.

#[path = "support/prng.rs"]
pub mod prng;

use heuristic_search::{
    AgentStep, DStarLite, DynamicGraph, GridCell, Limits, Octile, RealTimeDStar, Weight,
};
use prng::SplitMix64;

/// Side length of the terrain in cells.
const SIZE: usize = 120;

/// Sensor range in cells (Euclidean).
const SENSOR: usize = 4;

/// Where the rover lands.
const START: GridCell = (2, 2);

/// Where it must go.
const GOAL: GridCell = (SIZE - 3, SIZE - 3);

/// Cycles after which a run counts as stuck, which fails the example.
const MAX_CYCLES: usize = 100_000;

/// The eight unit moves.
const MOVES: [(isize, isize); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

/// A map of blocked cells, either the true terrain or the rover's belief.
#[derive(Clone)]
struct Terrain {
    /// `true` for blocked cells, row by row.
    blocked: Vec<bool>,
}

impl Terrain {
    /// A map with no obstacles: the rover's belief before it sees anything.
    fn open() -> Self {
        Self {
            blocked: vec![false; SIZE * SIZE],
        }
    }

    /// Generates true terrain from `seed`: boulders of radius one to three and four long
    /// rims with one gap each. Retries with the next seed until the goal is reachable.
    fn generate(mut seed: u64) -> Self {
        loop {
            let mut rng = SplitMix64::new(seed);
            let mut terrain = Self::open();
            for _ in 0..140 {
                let (cx, cy, r) = (rng.below(SIZE), rng.below(SIZE), 1 + rng.below(3));
                for y in cy.saturating_sub(r)..(cy + r + 1).min(SIZE) {
                    for x in cx.saturating_sub(r)..(cx + r + 1).min(SIZE) {
                        if x.abs_diff(cx).pow(2) + y.abs_diff(cy).pow(2) <= r * r {
                            terrain.blocked[y * SIZE + x] = true;
                        }
                    }
                }
            }
            for rim in 0..4 {
                let line = 20 + rim * 25;
                let gap = rng.below(SIZE - 10) + 5;
                for along in 0..SIZE {
                    if along.abs_diff(gap) > 1 {
                        // Rims alternate between rows and columns.
                        let (x, y) = if rim % 2 == 0 {
                            (along, line)
                        } else {
                            (line, along)
                        };
                        terrain.blocked[y * SIZE + x] = true;
                    }
                }
            }
            for (x, y) in [START, GOAL] {
                terrain.blocked[y * SIZE + x] = false;
            }

            if hindsight_optimum(&terrain).is_some() {
                return terrain;
            }
            seed += 1;
        }
    }

    /// Returns `true` if `(x, y)` is on the map and not blocked.
    fn is_free(&self, x: usize, y: usize) -> bool {
        x < SIZE && y < SIZE && !self.blocked[y * SIZE + x]
    }

    /// The cell one `step` away from `cell`, if it is on the map.
    fn offset((x, y): GridCell, (dx, dy): (isize, isize)) -> Option<GridCell> {
        let x = x.checked_add_signed(dx)?;
        let y = y.checked_add_signed(dy)?;

        (x < SIZE && y < SIZE).then_some((x, y))
    }
}

impl DynamicGraph for Terrain {
    type Vertex = GridCell;
    type Cost = Octile;

    fn successors(&self, &cell: &GridCell, mut emit: impl FnMut(GridCell, Octile)) {
        if !self.is_free(cell.0, cell.1) {
            return;
        }

        for (dx, dy) in MOVES {
            let Some(next) = Self::offset(cell, (dx, dy)) else {
                continue;
            };
            let free = |step| Self::offset(cell, step).is_some_and(|(x, y)| self.is_free(x, y));
            if !free((dx, dy)) {
                continue;
            }

            if dx == 0 || dy == 0 {
                emit(next, Octile::STRAIGHT);
            } else if free((dx, 0)) && free((0, dy)) {
                // No corner cutting: a diagonal needs both cells beside it.
                emit(next, Octile::DIAGONAL);
            }
        }
    }

    fn predecessors(&self, cell: &GridCell, emit: impl FnMut(GridCell, Octile)) {
        // Every move needs the same cells in both directions.
        self.successors(cell, emit);
    }
}

/// The octile distance, consistent on 8-connected grids whatever the obstacles.
fn distance(a: &GridCell, b: &GridCell) -> Octile {
    Octile::between(*a, *b)
}

/// The cheapest cost from [`START`] to [`GOAL`] on `terrain`, if any.
fn hindsight_optimum(terrain: &Terrain) -> Option<Octile> {
    DStarLite::new(terrain, START, GOAL, distance)
        .compute(terrain, Limits::NONE)
        .into_solution()
        .map(|solution| solution.cost)
}

/// Copies the obstacles within [`SENSOR`] of `at` from `truth` into `belief` and returns
/// both directions of every edge that touches a cell around a newly seen obstacle (a
/// blocked cell also removes the diagonals past its corners).
fn sense(belief: &mut Terrain, truth: &Terrain, (ax, ay): GridCell) -> Vec<(GridCell, GridCell)> {
    let mut edges = Vec::new();
    for y in ay.saturating_sub(SENSOR)..(ay + SENSOR + 1).min(SIZE) {
        for x in ax.saturating_sub(SENSOR)..(ax + SENSOR + 1).min(SIZE) {
            let seen = x.abs_diff(ax).pow(2) + y.abs_diff(ay).pow(2) <= SENSOR * SENSOR;
            if !seen || !truth.blocked[y * SIZE + x] || belief.blocked[y * SIZE + x] {
                continue;
            }

            belief.blocked[y * SIZE + x] = true;
            let around = MOVES
                .iter()
                .filter_map(|&step| Terrain::offset((x, y), step));
            for a in around.chain([(x, y)]) {
                for b in MOVES.iter().filter_map(|&step| Terrain::offset(a, step)) {
                    edges.push((a, b));
                    edges.push((b, a));
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();

    edges
}

/// What one drive did.
struct Drive {
    /// Control cycles until the rover stood on the goal.
    cycles: usize,
    /// Cycles in which the planner knew a cheapest path for the current map.
    informed: usize,
    /// The cost of the path driven.
    cost: Octile,
    /// The most expansions in one cycle.
    max_expansions: u64,
    /// Expansions over the whole drive.
    expansions: u64,
}

/// Drives with D* Lite, repairing to a cheapest path every cycle.
fn drive_d_star_lite(truth: &Terrain) -> Drive {
    let mut belief = Terrain::open();
    let mut planner = DStarLite::new(&belief, START, GOAL, distance);
    let mut drive = Drive {
        cycles: 0,
        informed: 0,
        cost: Octile::new(0, 0),
        max_expansions: 0,
        expansions: 0,
    };
    let mut at = START;
    while at != GOAL {
        assert!(drive.cycles < MAX_CYCLES, "D* Lite is stuck at {at:?}");
        for (from, to) in sense(&mut belief, truth, at) {
            planner.edge_changed(&belief, &from, &to);
        }

        let result = planner.compute(&belief, Limits::NONE);
        drive.max_expansions = drive.max_expansions.max(result.stats.expanded);
        drive.expansions += result.stats.expanded;
        let path = result.into_solution().expect("the goal stays reachable");
        let next = path.states[1];
        assert!(
            truth.is_free(next.0, next.1),
            "the rover drove into an obstacle"
        );
        drive.cost = drive.cost + distance(&at, &next);
        drive.cycles += 1;
        drive.informed += 1;
        at = next;
        planner.move_start(at);
    }

    drive
}

/// Drives with Real-time D* under `limit` expansions per cycle.
fn drive_real_time(truth: &Terrain, limit: usize, local_share: Weight) -> Drive {
    let mut belief = Terrain::open();
    let mut agent = RealTimeDStar::new(&belief, START, GOAL, distance, local_share);
    let mut cycles = 0;
    let mut informed = 0;
    loop {
        assert!(
            cycles < MAX_CYCLES,
            "RTD* is stuck at {:?}",
            agent.position()
        );
        let at = *agent.position();
        assert!(
            truth.is_free(at.0, at.1),
            "the rover drove into an obstacle"
        );
        for (from, to) in sense(&mut belief, truth, at) {
            agent.edge_changed(&belief, &from, &to);
        }

        match agent.step(&belief, limit) {
            AgentStep::Arrived => break,
            AgentStep::Informed => informed += 1,
            AgentStep::Partial => {}
            AgentStep::Waiting => panic!("the goal stays reachable"),
        }
        cycles += 1;
    }

    let stats = agent.stats();
    let trial = agent.into_trial();
    let cost = trial.solution().expect("the rover arrived").cost;
    assert!(
        trial.trial.max_step_expansions <= u64::try_from(limit).expect("small limit"),
        "a cycle broke the real-time limit"
    );

    Drive {
        cycles,
        informed,
        cost,
        max_expansions: trial.trial.max_step_expansions,
        expansions: stats.expanded,
    }
}

/// Prints one table row.
fn print_row(name: &str, limit: &str, share: &str, drive: &Drive, optimum: Octile) {
    let informed = 100.0 * f64::from(u32::try_from(drive.informed).expect("few cycles"))
        / f64::from(u32::try_from(drive.cycles).expect("few cycles"));
    println!(
        "{name:<14} {limit:>6} {share:>6} {:>7} {informed:>8.1}% {:>8.1} {:>7.3} {:>9} {:>10}",
        drive.cycles,
        drive.cost.to_f64(),
        drive.cost.to_f64() / optimum.to_f64(),
        drive.max_expansions,
        drive.expansions,
    );
}

/// Generates the terrain, drives it with every planner and prints the table.
fn main() {
    let truth = Terrain::generate(42);
    let optimum = hindsight_optimum(&truth).expect("the generator checks reachability");
    println!(
        "terrain {SIZE} x {SIZE}, sensor radius {SENSOR}, {START:?} to {GOAL:?}, \
         hindsight optimum {:.1}\n",
        optimum.to_f64()
    );
    println!(
        "{:<14} {:>6} {:>6} {:>7} {:>9} {:>8} {:>7} {:>9} {:>10}",
        "planner", "limit", "local", "cycles", "informed", "cost", "/opt", "max/cycle", "expanded"
    );

    let drive = drive_d_star_lite(&truth);
    assert!(drive.cost >= optimum);
    print_row("DStarLite", "none", "-", &drive, optimum);

    for limit in [16, 64, 256] {
        for (share, label) in [
            (Weight::new(1, 4), "1/4"),
            (Weight::new(1, 2), "1/2"),
            (Weight::new(3, 4), "3/4"),
        ] {
            let drive = drive_real_time(&truth, limit, share);
            assert!(drive.cost >= optimum);
            print_row("RealTimeDStar", &limit.to_string(), label, &drive, optimum);
        }
    }

    println!(
        "\nD* Lite drives the shortest route its map allows but may stall a cycle for \
         thousands of expansions; Real-time D* never exceeds its limit and keeps moving."
    );
}
