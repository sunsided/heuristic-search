//! One large query on several threads: hash-distributed A* ([`hda_star`]) against
//! [`a_star`].
//!
//! HDA* splits one search over threads by hashing each state to an owner thread. Every
//! generated successor is sent to its owner over a channel, so each expansion pays for
//! messages, and threads may expand a state again when a cheaper path to it arrives late.
//! That overhead is fixed per state, so HDA* only wins when an expansion costs much more
//! than a message: a successor function that does real work (simulation, collision or
//! terrain checks, a planning domain's precondition tests), or a search large enough that
//! the threads stay busy. On a plain grid, where an expansion is a handful of array reads,
//! single-threaded A* is hard to beat, and this example shows that honestly.
//!
//! Both runs search a generated 512 x 512 RTS map (walled bases, forests and a river with
//! fords), 8-connected with exact [`Octile`] costs and the octile heuristic:
//!
//! 1. **Plain grid**, corner to corner: successors are cheap.
//! 2. **Expensive successors**, corner to centre (a shorter query, to keep the run short):
//!    the same grid, but every expansion first runs a fixed amount of CPU work (standing
//!    in for a terrain or line-of-fire evaluation). The edge costs are unchanged.
//!
//! More threads are not always faster: each extra thread adds messages, and states
//! expanded before their cheapest path arrived are reopened (the `reopened` column), so
//! past some count the speed-up falls again, sooner on a busy machine.
//!
//! Each table row gives the wall time, the states expanded (summed over threads) and the
//! speed-up against `a_star` (below 1 means slower). Every run must find the cost
//! `a_star` finds; the example asserts it. HDA*'s expansion counts depend on thread
//! timing and differ from run to run; `a_star`'s and the costs do not.
//!
//! Run with `cargo run --release --features parallel --example parallel_hda_star`.

#[path = "support/game_map.rs"]
#[expect(
    dead_code,
    reason = "this example picks its cells by position, not at random"
)]
mod game_map;

use std::hint::black_box;
use std::time::{Duration, Instant};

use game_map::{GameMap, Rng};
use heuristic_search::{
    EightConnected, GridCell, GridStep, Limits, Octile, OctileDistance, Stats, Successors, a_star,
    hda_star,
};

/// Side length of the map in cells.
const SIZE: usize = 512;

/// Seed of the map.
const SEED: u64 = 0x0DA5_7A12;

/// Thread counts to compare.
const THREADS: [usize; 4] = [1, 2, 4, 8];

/// Rounds of busy work per expansion in the expensive variant.
const WORK_ROUNDS: u32 = 20_000;

/// The map as a graph whose every expansion first does [`WORK_ROUNDS`] rounds of CPU work.
struct Expensive<'a>(EightConnected<&'a GameMap>);

impl Successors for Expensive<'_> {
    type State = GridCell;
    type Action = GridStep;
    type Cost = Octile;

    fn successors(&self, cell: &GridCell, emit: impl FnMut(GridStep, GridCell, Octile)) {
        let seed = u64::try_from(cell.0 * SIZE + cell.1).expect("cell index fits in u64");
        let mut rng = Rng::new(seed);
        for _ in 0..WORK_ROUNDS {
            black_box(rng.next_u64());
        }
        self.0.successors(cell, emit);
    }
}

/// One timed search.
struct Run {
    /// Wall time.
    time: Duration,
    /// Counters, summed over threads.
    stats: Stats,
    /// The cost found.
    cost: Octile,
}

/// Generates the map and compares `a_star` with `hda_star` on both graphs.
fn main() {
    let map = GameMap::generate(SIZE, SIZE, SEED);
    let start = map.nearest_open((0, 0));
    let target = map.nearest_open((SIZE - 1, SIZE - 1));
    println!(
        "map {SIZE} x {SIZE}, {} open cells; query {start:?} to {target:?}",
        map.open_cells()
    );

    println!("\nplain grid (cheap successors):");
    compare(&EightConnected(&map), start, target);
    println!("\nexpensive successors ({WORK_ROUNDS} rounds of work per expansion):");
    let midpoint = map.nearest_open((SIZE / 2, SIZE / 2));
    println!("query {start:?} to {midpoint:?}");
    compare(&Expensive(EightConnected(&map)), start, midpoint);
}

/// Runs `a_star` and then `hda_star` at every thread count of [`THREADS`] on `graph`,
/// prints a table and asserts that all costs agree.
fn compare<G>(graph: &G, start: GridCell, target: GridCell)
where
    G: Successors<State = GridCell, Action = GridStep, Cost = Octile> + Sync,
{
    let goal = |cell: &GridCell| *cell == target;
    let baseline = timed(|| a_star(graph, start, &goal, OctileDistance(target), Limits::NONE));
    println!(
        "  {:<16} {:>10} {:>10} {:>9} {:>9}",
        "search", "time", "expanded", "reopened", "speed-up"
    );
    print_row("a_star", &baseline, &baseline);
    for threads in THREADS {
        let run = timed(|| {
            hda_star(
                graph,
                start,
                &goal,
                OctileDistance(target),
                threads,
                Limits::NONE,
            )
        });
        print_row(&format!("hda_star x{threads}"), &run, &baseline);
        assert_eq!(
            run.cost, baseline.cost,
            "hda_star on {threads} threads changed the cost"
        );
    }
}

/// Times one search and keeps its counters and cost.
fn timed<A>(search: impl FnOnce() -> heuristic_search::SearchResult<GridCell, A, Octile>) -> Run {
    let started = Instant::now();
    let result = search();
    let time = started.elapsed();
    let cost = result.solution().expect("the map is connected").cost;
    Run {
        time,
        stats: result.stats,
        cost,
    }
}

/// Prints one table row with the speed-up of `run` over `baseline`.
fn print_row(name: &str, run: &Run, baseline: &Run) {
    println!(
        "  {name:<16} {:>7.1} ms {:>10} {:>9} {:>8.2}x",
        run.time.as_secs_f64() * 1000.0,
        run.stats.expanded,
        run.stats.reopened,
        baseline.time.as_secs_f64() / run.time.as_secs_f64()
    );
}
