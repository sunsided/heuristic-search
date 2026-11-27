//! Contraction hierarchies on a city road network: pay for preprocessing once, then answer
//! point-to-point route queries far faster than any search over the raw graph.
//!
//! The city (see `support/road_network.rs`) has about 22 600 intersections, one-way
//! streets, arterials and a motorway, with travel times as edge costs. The example builds
//! a [`ContractionHierarchy`] once and reports how long that took and how many shortcuts
//! it added. Then it answers the same 1000 random queries three ways:
//!
//! - `dijkstra`: the blind baseline, settling everything closer than the goal;
//! - `a_star` with the straight-line heuristic (distance over top speed);
//! - the hierarchy's query, a bidirectional Dijkstra that only climbs the hierarchy.
//!
//! The table shows the states each method settled per query (the `expanded` counter of
//! its `Stats`), the expansions the network itself saw (the hierarchy's query never
//! touches the graph), and the wall time per query. The break-even lines give the number
//! of queries after which the build time is paid back by faster queries. Every cost is
//! asserted to equal Dijkstra's, and one route is unpacked: the hierarchy's solution
//! consists of the network's own road segments, not of shortcuts, so it is replayed edge
//! by edge.
//!
//! The witness searches are given more room than the default [`WitnessLimits`] (16 hops,
//! 64 settled states): a street grid has many equally fast detours, and a witness search
//! that gives up too early adds shortcuts that a longer search would have found
//! unnecessary. Here the wider limits build no slower and give fewer shortcuts and faster
//! queries.
//!
//! Run with `cargo run --release --example contraction_hierarchies`. Timings vary between
//! machines and runs; the counts do not.

#[path = "support/race.rs"]
mod race;
#[path = "support/road_network.rs"]
mod road_network;

use std::time::{Duration, Instant};

use heuristic_search::{ContractionHierarchy, Limits, Solution, WitnessLimits, a_star, dijkstra};

use race::{Method, Totals};
use road_network::RoadNetwork;

/// Witness search limits for the street grid.
const LIMITS: WitnessLimits = WitnessLimits {
    max_hops: 64,
    max_settled: 1_000,
};

/// Builds the network and the hierarchy, races the three methods and unpacks one route.
fn main() {
    let network = RoadNetwork::generate(7);
    println!("{}", network.summary());

    let started = Instant::now();
    let hierarchy = ContractionHierarchy::build_with_limits(&network, [0], LIMITS);
    let build = started.elapsed();
    assert_eq!(
        hierarchy.len(),
        network.len(),
        "one seed reaches the whole city"
    );
    println!(
        "contraction hierarchy: built in {:.2} s, {} shortcuts on top of {} roads\n",
        build.as_secs_f64(),
        hierarchy.shortcut_count(),
        network.roads.len(),
    );

    let count = race::query_count(1_000);
    let queries = network.random_queries(count, 42);
    let methods: [Method; 3] = [
        (
            "dijkstra",
            Box::new(|start, goal| dijkstra(&network, start, &|v: &u32| *v == goal, Limits::NONE)),
        ),
        (
            "a_star (straight line)",
            Box::new(|start, goal| {
                let h = network.straight_line_to(goal);
                a_star(&network, start, &|v: &u32| *v == goal, h, Limits::NONE)
            }),
        ),
        (
            "contraction hierarchy",
            Box::new(|start, goal| hierarchy.query(start, goal)),
        ),
    ];
    let totals = race::race(&network, &queries, &methods);
    race::print_table(&totals, count);

    println!();
    for baseline in &totals[..2] {
        break_even(build, baseline, &totals[2], count);
    }

    let (start, goal) = queries[0];
    let route = hierarchy
        .query(start, goal)
        .into_solution()
        .expect("the city is strongly connected");
    replay(&network, &route);
}

/// Prints after how many queries the build time of the hierarchy is paid back against
/// `baseline`.
#[expect(
    clippy::cast_precision_loss,
    reason = "query counts are small; the ratio is for display"
)]
fn break_even(build: Duration, baseline: &Totals, ch: &Totals, queries: usize) {
    let saved = (baseline.time.as_secs_f64() - ch.time.as_secs_f64()) / queries as f64;
    if saved > 0.0 {
        println!(
            "break-even against {}: after {:.0} queries ({:.3} ms saved per query)",
            baseline.name,
            (build.as_secs_f64() / saved).ceil(),
            saved * 1e3,
        );
    } else {
        println!(
            "break-even against {}: never, the hierarchy is not faster here",
            baseline.name
        );
    }
}

/// Replays a hierarchy route on the network: every action is an original road segment
/// that leaves the previous state and enters the next one, and the travel times add up to
/// the route's cost.
fn replay(network: &RoadNetwork, route: &Solution<u32, u32, u32>) {
    let mut time = 0;
    for (step, &action) in route.actions.iter().enumerate() {
        let road = network.roads[action as usize];
        assert_eq!(
            road.from, route.states[step],
            "road {action} leaves the route"
        );
        assert_eq!(
            road.to,
            route.states[step + 1],
            "road {action} leaves the route"
        );
        time += road.time;
    }
    assert_eq!(
        time, route.cost,
        "the road times add up to the route's cost"
    );

    let first: Vec<String> = route.actions.iter().take(5).map(u32::to_string).collect();
    println!(
        "\nroute {} -> {}: {:.1} min over {} original road segments [{}, ...], replayed \
         one by one; their times add up to the route's cost",
        route.states[0],
        route.goal(),
        f64::from(route.cost) / 600.0,
        route.len(),
        first.join(", "),
    );
}
