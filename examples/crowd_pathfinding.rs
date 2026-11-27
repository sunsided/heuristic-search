//! Crowd pathfinding on an RTS map: hundreds of units ordered to one rally point.
//!
//! The map is a generated 256 x 256 grid with walled bases, forests and a river with fords,
//! searched 8-connected without corner cutting through [`EightConnected`], with exact
//! [`Octile`] costs. 300 units at random open cells all head to one rally point. Four
//! ways to answer them are compared, each with its wall time and the states it expanded:
//!
//! - **`a_star` per unit**: the obvious baseline, one fresh search (and fresh allocations)
//!   per unit.
//! - **`a_star_in` + `Workspace`**: the same searches in one reused [`Workspace`]. Same
//!   paths and the same expansions per unit; only the allocations are saved, which
//!   helps little here, where each search expands thousands of states.
//! - **`FlowField`**: one backward Dijkstra search from the rally point settles every cell
//!   once; then [`FlowField::path`] (or [`FlowField::next`], one step per frame) answers
//!   every unit without searching. The expansions are paid once, so the more units share a
//!   goal, the more it wins.
//! - **`solve_batch`**: when units have *different* goals no field is shared, but the
//!   queries are independent, so [`solve_batch`] spreads them over threads with one
//!   workspace per thread. It is compared with the same batch on one thread.
//!
//! Every strategy must give each unit the same cheapest cost as `a_star`; the example
//! asserts this, so a broken library makes it panic.
//!
//! Last, a frame budget: one long query (corner to corner) runs as an [`AStarSearch`] that
//! may expand only a fixed number of states per game frame and resumes on the next frame.
//! The output shows how many frames the query took; the total expansions and the cost
//! equal those of one uninterrupted `a_star`.
//!
//! Run with `cargo run --release --features parallel --example crowd_pathfinding`.
//! Expansion counts and costs are the same on every run; timings vary with the machine.

#[path = "support/game_map.rs"]
mod game_map;
#[path = "support/prng.rs"]
pub mod prng;

use std::time::{Duration, Instant};

use game_map::GameMap;
use heuristic_search::{
    AStarSearch, EightConnected, FlowField, GridCell, Limits, Octile, OctileDistance, Outcome,
    Workspace, a_star, a_star_in, solve_batch,
};
use prng::SplitMix64;

/// Side length of the map in cells.
const SIZE: usize = 256;

/// Seed of the map and of the unit positions.
const SEED: u64 = 0x5EED_CAFE;

/// The number of units.
const UNITS: usize = 300;

/// Threads for the parallel batch.
const THREADS: usize = 4;

/// Expansions an [`AStarSearch`] may make per game frame.
const FRAME_BUDGET: u64 = 500;

/// The map as a graph.
type Graph<'a> = EightConnected<&'a GameMap>;

/// What one strategy cost and what it answered.
struct Run {
    /// Wall time of the whole strategy.
    time: Duration,
    /// States expanded, summed over all its searches.
    expanded: u64,
    /// The cost of each unit's path, in unit order.
    costs: Vec<Octile>,
}

/// Generates the map and the units, runs every strategy and prints the comparison.
fn main() {
    let map = GameMap::generate(SIZE, SIZE, SEED);
    let graph = EightConnected(&map);
    let mut rng = SplitMix64::new(SEED ^ 1);
    let units: Vec<GridCell> = (0..UNITS).map(|_| map.random_open(&mut rng)).collect();
    let rally = map.nearest_open((SIZE / 2, SIZE / 3));

    println!(
        "map {SIZE} x {SIZE}, {} open cells; {UNITS} units to rally point {rally:?}\n",
        map.open_cells()
    );
    println!("{:<34} {:>10} {:>12}", "strategy", "time", "expanded");

    let baseline = per_unit_a_star(graph, &units, rally);
    print_run("a_star per unit", &baseline);
    let reused = reused_workspace(graph, &units, rally);
    print_run("a_star_in, one Workspace", &reused);
    let field = flow_field(graph, &units, rally);
    print_run("FlowField once + path per unit", &field);
    assert_eq!(reused.costs, baseline.costs, "a_star_in changed a cost");
    assert_eq!(
        reused.expanded, baseline.expanded,
        "a_star_in expanded differently"
    );
    assert_eq!(field.costs, baseline.costs, "the flow field changed a cost");

    let mut targets = SplitMix64::new(SEED ^ 2);
    let queries: Vec<(GridCell, GridCell)> = units
        .iter()
        .map(|&unit| (unit, map.random_open(&mut targets)))
        .collect();
    println!("\n{UNITS} units, each to its own random goal (no shared field):");
    let sequential = batch(graph, &queries, 1);
    print_run("solve_batch, 1 thread", &sequential);
    let parallel = batch(graph, &queries, THREADS);
    print_run(&format!("solve_batch, {THREADS} threads"), &parallel);
    assert_eq!(parallel.costs, sequential.costs, "threads changed a cost");
    assert_eq!(
        parallel.expanded, sequential.expanded,
        "threads changed the work"
    );

    frame_budget(graph, &map);
}

/// One fresh [`a_star`] per unit.
fn per_unit_a_star(graph: Graph, units: &[GridCell], rally: GridCell) -> Run {
    let goal = |cell: &GridCell| *cell == rally;
    let started = Instant::now();
    let mut expanded = 0;
    let costs = units
        .iter()
        .map(|&unit| {
            let result = a_star(&graph, unit, &goal, OctileDistance(rally), Limits::NONE);
            expanded += result.stats.expanded;
            result.solution().expect("the map is connected").cost
        })
        .collect();
    Run {
        time: started.elapsed(),
        expanded,
        costs,
    }
}

/// One [`a_star_in`] per unit, all in one [`Workspace`].
fn reused_workspace(graph: Graph, units: &[GridCell], rally: GridCell) -> Run {
    let goal = |cell: &GridCell| *cell == rally;
    let started = Instant::now();
    let mut workspace = Workspace::new();
    let mut expanded = 0;
    let costs = units
        .iter()
        .map(|&unit| {
            let result = a_star_in(
                &mut workspace,
                &graph,
                unit,
                &goal,
                OctileDistance(rally),
                Limits::NONE,
            );
            expanded += result.stats.expanded;
            result.solution().expect("the map is connected").cost
        })
        .collect();
    Run {
        time: started.elapsed(),
        expanded,
        costs,
    }
}

/// One [`FlowField`] from the rally point, then a path for every unit; afterwards checks
/// that walking [`FlowField::next`] step by step agrees with the path.
fn flow_field(graph: Graph, units: &[GridCell], rally: GridCell) -> Run {
    let started = Instant::now();
    let field = FlowField::build(&graph, [rally], Limits::NONE);
    let costs: Vec<Octile> = units
        .iter()
        .map(|unit| field.path(unit).expect("the map is connected").cost)
        .collect();
    let time = started.elapsed();

    assert!(field.is_complete());
    for (unit, &cost) in units.iter().zip(&costs) {
        assert_eq!(field.cost(unit), Some(cost));
        assert_eq!(walk(&field, *unit, rally), cost, "next disagrees with path");
    }
    Run {
        time,
        expanded: field.stats().expanded,
        costs,
    }
}

/// Follows [`FlowField::next`] from `unit` to `rally`, as a unit would one step per frame,
/// and returns the summed step costs.
fn walk(field: &FlowField<GridCell, (i8, i8), Octile>, unit: GridCell, rally: GridCell) -> Octile {
    let mut cell = unit;
    let mut cost = Octile::new(0, 0);
    while let Some((_, &next, step)) = field.next(&cell) {
        cell = next;
        cost = cost + step;
    }
    assert_eq!(cell, rally, "the field leads to the rally point");
    cost
}

/// Solves every `(start, goal)` query with [`a_star_in`] through [`solve_batch`] on
/// `threads` threads, one workspace per thread.
fn batch(graph: Graph, queries: &[(GridCell, GridCell)], threads: usize) -> Run {
    let started = Instant::now();
    let answers = solve_batch(
        queries,
        threads,
        Workspace::new,
        |workspace, &(start, target)| {
            let goal = |cell: &GridCell| *cell == target;
            let result = a_star_in(
                workspace,
                &graph,
                start,
                &goal,
                OctileDistance(target),
                Limits::NONE,
            );
            let cost = result.solution().expect("the map is connected").cost;
            (cost, result.stats.expanded)
        },
    );
    Run {
        time: started.elapsed(),
        expanded: answers.iter().map(|&(_, expanded)| expanded).sum(),
        costs: answers.into_iter().map(|(cost, _)| cost).collect(),
    }
}

/// Runs one corner-to-corner query as an [`AStarSearch`] with [`FRAME_BUDGET`] expansions
/// per frame, and checks it against one uninterrupted [`a_star`].
fn frame_budget(graph: Graph, map: &GameMap) {
    let start = map.nearest_open((0, 0));
    let target = map.nearest_open((SIZE - 1, SIZE - 1));
    let goal = |cell: &GridCell| *cell == target;
    let whole = a_star(&graph, start, &goal, OctileDistance(target), Limits::NONE);
    let whole_cost = whole.solution().expect("the map is connected").cost;

    let mut search = AStarSearch::new(&graph, start, &goal, OctileDistance(target));
    let mut frames = 0;
    let mut slowest = Duration::ZERO;
    let solution = loop {
        frames += 1;
        let frame = Instant::now();
        let result = search.compute(&graph, &goal, Limits::expansions(FRAME_BUDGET));
        slowest = slowest.max(frame.elapsed());
        match result.outcome {
            Outcome::LimitReached => {}
            Outcome::Found(solution) => break solution,
            other => panic!("the map is connected, got {other:?}"),
        }
    };

    println!(
        "\nframe budget: {start:?} to {target:?} with {FRAME_BUDGET} expansions per frame\n  \
         {frames} frames, {} expansions in all (a_star: {}), slowest frame {}",
        search.stats().expanded,
        whole.stats.expanded,
        millis(slowest)
    );
    assert_eq!(solution.cost, whole_cost, "resuming changed the cost");
    assert_eq!(
        search.stats().expanded,
        whole.stats.expanded,
        "resuming changed the work"
    );
}

/// Prints one table row.
fn print_run(name: &str, run: &Run) {
    println!("{name:<34} {:>10} {:>12}", millis(run.time), run.expanded);
}

/// A duration as milliseconds with one decimal.
fn millis(time: Duration) -> String {
    format!("{:.1} ms", time.as_secs_f64() * 1000.0)
}
