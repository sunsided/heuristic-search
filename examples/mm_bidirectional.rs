//! MM, the bidirectional search that meets in the middle, on a city road network, next to
//! bidirectional Dijkstra and A*.
//!
//! MM orders each side by `max(f, 2g)`, which guarantees that neither side expands a
//! state more than half the optimal cost `C*` away from its own root: the two frontiers
//! meet in the middle. MM0 is the same search with zero heuristics, a blind bidirectional
//! search with that guarantee.
//!
//! The example answers the same 1000 random queries on the city (see
//! `support/road_network.rs`) with `dijkstra`, `bidirectional_dijkstra`, `a_star`, `mm` and
//! MM0, the heuristic searches using the straight-line travel time (distance over top
//! speed) towards the goal and, for the backward side, from the start. The first table
//! shows expansions per query, split by side, and time; every cost is asserted equal.
//!
//! The second table checks the meet-in-the-middle property on a sample of the queries.
//! Exact distance fields (a [`FlowField`] from the goal, and one from the start over the
//! reversed network) give every expanded state its distance from its own side's root. The
//! table shows the largest such distance on each side as a fraction of `C*`: at most 0.5
//! for MM and MM0, which the example asserts (a state's `g` at expansion is at least its
//! distance, so this follows from MM's bound on `g`).
//!
//! The honest lesson of this network: MM is not worth it here. With the straight-line
//! heuristic it expands only a few percent fewer states than A* with the same heuristic,
//! and each expansion costs more (MM keeps its frontiers ordered by `pr`, `f` and `g`), so
//! it is slower in wall time. MM0 expands exactly as many states as bidirectional
//! Dijkstra, which on this network also stops at half of `C*` on each side, at a higher
//! cost per expansion. The straight-line heuristic is weak where most roads are four times
//! slower than the motorway it assumes, so `max(f, 2g)` is mostly `2g` and MM behaves
//! much like MM0. And two half-depth searches save only about half the work on a road
//! grid, whose search space grows quadratically with the cost. MM's guarantee pays off
//! where heuristic bidirectional searches would otherwise run past the middle and the
//! space grows fast with depth, as in combinatorial puzzles. For route planning, a strong
//! heuristic (ALT, see the `alt_landmarks` example) or preprocessing (the
//! `contraction_hierarchies` example) does far better.
//!
//! Run with `cargo run --release --example mm_bidirectional`. Timings vary between
//! machines and runs; the counts do not.

#[path = "support/race.rs"]
mod race;
#[path = "support/road_network.rs"]
mod road_network;

use heuristic_search::{
    FlowField, Limits, Predecessors, SearchResult, Successors, Zero, a_star,
    bidirectional_dijkstra, dijkstra, heuristic_fn, mm,
};

use race::Method;
use road_network::RoadNetwork;

/// Queries (in release builds) whose expanded states are checked against `C* / 2`.
const SAMPLED: usize = 100;

/// The network searched backwards: the predecessors of a node are its successors in the
/// network, so a [`FlowField`] over it gives the distance from its goal set.
struct Reversed<'a>(&'a RoadNetwork);

impl Predecessors for Reversed<'_> {
    type Node = u32;
    type Action = u32;
    type Cost = u32;

    fn predecessors(&self, node: &u32, emit: impl FnMut(u32, u32, u32)) {
        self.0.successors(node, emit);
    }
}

/// Builds the network, races the searches and checks where the bidirectional ones expand.
fn main() {
    let network = RoadNetwork::generate(7);
    println!("{}\n", network.summary());

    let methods = methods(&network);
    let count = race::query_count(1_000);
    let queries = network.random_queries(count, 42);
    let totals = race::race(&network, &queries, &methods);
    race::print_table(&totals, count);

    println!(
        "\nlargest distance from its own root of a state expanded on each side, as a \
         fraction of C*\n(mean and worst over {} queries)",
        race::query_count(SAMPLED)
    );
    println!(
        "{:<30} {:>18} {:>18}",
        "method", "forward mean/max", "backward mean/max"
    );
    let sample = &queries[..race::query_count(SAMPLED)];
    for (name, method) in &methods {
        if name.starts_with("bidirectional") || name.starts_with("mm") {
            meet_in_the_middle(&network, name, method, sample);
        }
    }
}

/// The searches to compare, Dijkstra first as the reference.
fn methods(network: &RoadNetwork) -> Vec<Method<'_>> {
    vec![
        (
            "dijkstra",
            Box::new(|start, goal| dijkstra(network, start, &|v: &u32| *v == goal, Limits::NONE)),
        ),
        (
            "bidirectional_dijkstra",
            Box::new(|start, goal| bidirectional_dijkstra(network, start, [goal], Limits::NONE)),
        ),
        (
            "a_star, straight line",
            Box::new(|start, goal| {
                let h = network.straight_line_to(goal);
                a_star(network, start, &|v: &u32| *v == goal, h, Limits::NONE)
            }),
        ),
        (
            "mm, straight line",
            Box::new(|start, goal| {
                let to_goal = network.straight_line_to(goal);
                let from_start = heuristic_fn(|v: &u32| Some(network.straight_line(start, *v)));
                mm(network, start, [goal], to_goal, from_start, Limits::NONE)
            }),
        ),
        (
            "mm0 (Zero heuristics)",
            Box::new(|start, goal| mm(network, start, [goal], Zero, Zero, Limits::NONE)),
        ),
    ]
}

/// Runs `method` on every query of `sample` and prints how far from its own root each side
/// expanded, relative to the optimal cost. Asserts the `C* / 2` bound for MM.
#[expect(
    clippy::cast_precision_loss,
    reason = "ratios of travel times for display"
)]
fn meet_in_the_middle(
    network: &RoadNetwork,
    name: &str,
    method: &dyn Fn(u32, u32) -> SearchResult<u32, u32, u32>,
    sample: &[(u32, u32)],
) {
    let mut sums = [0.0; 2];
    let mut worst = [0.0_f64; 2];
    for &(start, goal) in sample {
        let from_start = FlowField::build(&Reversed(network), [start], Limits::NONE);
        let to_goal = FlowField::build(network, [goal], Limits::NONE);
        network.take_expansions();

        let optimal = method(start, goal)
            .solution()
            .expect("the city is strongly connected")
            .cost;
        let (forward, backward) = network.take_expansions();
        let farthest = |expanded: &[u32], field: &FlowField<u32, u32, u32>| {
            expanded
                .iter()
                .map(|v| field.cost(v).expect("every state reaches every other"))
                .max()
                .unwrap_or(0)
        };
        let reached = [
            farthest(&forward, &from_start),
            farthest(&backward, &to_goal),
        ];
        if name.starts_with("mm") {
            for distance in reached {
                assert!(
                    2 * distance <= optimal,
                    "{name} expanded past C*/2 on {start} -> {goal}"
                );
            }
        }
        for side in 0..2 {
            let fraction = f64::from(reached[side]) / f64::from(optimal);
            sums[side] += fraction;
            worst[side] = worst[side].max(fraction);
        }
    }

    let mean = |side: usize| sums[side] / sample.len() as f64;
    println!(
        "{name:<30} {:>11.3} / {:.3} {:>11.3} / {:.3}",
        mean(0),
        worst[0],
        mean(1),
        worst[1],
    );
}
