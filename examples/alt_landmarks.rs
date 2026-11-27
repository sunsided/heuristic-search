//! ALT landmark heuristics on a city road network: a few exact distance tables turn A*
//! from a weakly guided search into a sharply guided one.
//!
//! On a road network with travel times, the straight-line heuristic (distance over the top
//! speed) is admissible but weak: it assumes the motorway goes everywhere, while most
//! trips run on 30 km/h streets. [`Landmarks::build`] precomputes exact travel times to
//! and from 16 landmark intersections; the triangle inequality turns them into lower
//! bounds that follow the actual roads, one-way streets included.
//!
//! The example builds the landmarks twice, picked farthest-first and by avoidance
//! ([`LandmarkSelection`]), and reports the build times. It then answers the same 1000
//! random queries with `dijkstra`, `a_star` with the straight-line heuristic, and `a_star`
//! with each ALT heuristic, once over all 16 landmarks and once over the 4 that bound the
//! query best (`active`), which makes each estimate cheaper. All costs are asserted equal.
//! Finally it runs BAE*, a bidirectional search, with straight-line and with ALT
//! estimates on both sides: `heuristic_to` for the forward side and `heuristic_from` for
//! the backward side.
//!
//! The table shows expansions per query (from `Stats`), the expansions the network saw
//! on each side, and the wall time per query.
//!
//! What to expect: ALT cuts A*'s expansions by an order of magnitude for a build that
//! costs about as much as a hundred Dijkstra queries. On this city farthest-first
//! landmarks bound better than avoidance, and keeping 4 active landmarks expands more
//! states than using all 16 but evaluates each estimate faster, which roughly evens out
//! in wall time. BAE* with ALT on both sides expands fewer states than ALT A*, yet a
//! bidirectional expansion costs more, so it is not faster.
//!
//! Run with `cargo run --release --example alt_landmarks`. Timings vary between machines
//! and runs; the counts do not.

#[path = "support/race.rs"]
mod race;
#[path = "support/road_network.rs"]
mod road_network;

use std::time::Instant;

use heuristic_search::{
    LandmarkSelection, Landmarks, Limits, a_star, bae_star, dijkstra, heuristic_fn,
};

use race::Method;
use road_network::RoadNetwork;

/// Landmarks per table.
const LANDMARKS: usize = 16;
/// Landmarks an `active` heuristic keeps per query.
const ACTIVE: usize = 4;

/// Builds the network and both landmark tables, then races the searches.
fn main() {
    let network = RoadNetwork::generate(7);
    println!("{}", network.summary());

    let farthest = build(&network, LandmarkSelection::Farthest);
    let avoidance = build(&network, LandmarkSelection::Avoidance);
    println!();

    let count = race::query_count(1_000);
    let queries = network.random_queries(count, 42);
    let unidirectional: Vec<Method> = vec![
        (
            "dijkstra",
            Box::new(|start, goal| dijkstra(&network, start, &|v: &u32| *v == goal, Limits::NONE)),
        ),
        (
            "a_star, straight line",
            Box::new(|start, goal| {
                let h = network.straight_line_to(goal);
                a_star(&network, start, &|v: &u32| *v == goal, h, Limits::NONE)
            }),
        ),
        alt("a_star, ALT farthest", &network, &farthest, None),
        alt(
            "a_star, ALT farthest, 4 active",
            &network,
            &farthest,
            Some(ACTIVE),
        ),
        alt("a_star, ALT avoidance", &network, &avoidance, None),
        alt(
            "a_star, ALT avoidance, 4 active",
            &network,
            &avoidance,
            Some(ACTIVE),
        ),
    ];
    let totals = race::race(&network, &queries, &unidirectional);
    race::print_table(&totals, count);

    let bidirectional: Vec<Method> = vec![
        (
            "bae_star, straight line",
            Box::new(|start, goal| {
                let to_goal = network.straight_line_to(goal);
                let from_start = heuristic_fn(|v: &u32| Some(network.straight_line(start, *v)));
                bae_star(&network, start, [goal], to_goal, from_start, Limits::NONE)
            }),
        ),
        (
            "bae_star, ALT farthest",
            Box::new(|start, goal| {
                let to_goal = farthest.heuristic_to(&goal);
                let from_start = farthest.heuristic_from(&start);
                bae_star(&network, start, [goal], to_goal, from_start, Limits::NONE)
            }),
        ),
    ];
    println!();
    let totals = race::race(&network, &queries, &bidirectional);
    race::print_table(&totals, count);
}

/// Builds a landmark table by `selection` and prints how long that took.
fn build(network: &RoadNetwork, selection: LandmarkSelection) -> Landmarks<u32, u32> {
    let started = Instant::now();
    let landmarks = Landmarks::build(network, 0, LANDMARKS, selection);
    println!(
        "{LANDMARKS} landmarks by {selection:?}: built in {:.0} ms",
        started.elapsed().as_secs_f64() * 1e3,
    );
    assert_eq!(landmarks.landmarks().len(), LANDMARKS);

    landmarks
}

/// A* with the ALT heuristic of `landmarks`, over every landmark or the `active` best
/// ones for each query.
fn alt<'a>(
    name: &'a str,
    network: &'a RoadNetwork,
    landmarks: &'a Landmarks<u32, u32>,
    active: Option<usize>,
) -> Method<'a> {
    let search = move |start, goal| {
        let h = landmarks.heuristic_to(&goal);
        let h = match active {
            Some(count) => h.active(&start, count),
            None => h,
        };
        a_star(network, start, &|v: &u32| *v == goal, h, Limits::NONE)
    };

    (name, Box::new(search))
}
