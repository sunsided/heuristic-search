//! Runs several route-finding methods over the same queries on a [`RoadNetwork`],
//! asserts that they agree on every cost, and prints a table of their work and time.

use std::time::{Duration, Instant};

use heuristic_search::SearchResult;

use crate::road_network::RoadNetwork;

/// A named way to answer a query `(start, goal)` on the network.
pub type Method<'a> = (
    &'a str,
    Box<dyn Fn(u32, u32) -> SearchResult<u32, u32, u32> + 'a>,
);

/// What one method did over all queries.
#[derive(Debug)]
pub struct Totals {
    /// The method's name.
    pub name: String,
    /// States expanded (settled), summed over the queries, from the search's `Stats`.
    pub expanded: u64,
    /// Forward expansions the network saw (calls of `successors`), summed.
    pub forward: u64,
    /// Backward expansions the network saw (calls of `predecessors`), summed.
    pub backward: u64,
    /// Wall time, summed over the queries.
    pub time: Duration,
}

/// Answers every query with every method and returns their totals, in order.
///
/// # Panics
///
/// Panics if a method finds no route, or a different cost than the first method.
pub fn race(network: &RoadNetwork, queries: &[(u32, u32)], methods: &[Method]) -> Vec<Totals> {
    let mut totals: Vec<Totals> = methods
        .iter()
        .map(|(name, _)| Totals {
            name: (*name).to_owned(),
            expanded: 0,
            forward: 0,
            backward: 0,
            time: Duration::ZERO,
        })
        .collect();
    network.take_expansions();

    for &(start, goal) in queries {
        let mut reference = None;
        for ((name, method), total) in methods.iter().zip(&mut totals) {
            let started = Instant::now();
            let result = method(start, goal);
            total.time += started.elapsed();

            let (forward, backward) = network.take_expansions();
            total.forward += forward.len() as u64;
            total.backward += backward.len() as u64;
            total.expanded += result.stats.expanded;

            let cost = result.solution().map(|solution| solution.cost);
            assert!(cost.is_some(), "{name} found no route {start} -> {goal}");
            let reference = *reference.get_or_insert(cost);
            assert_eq!(cost, reference, "{name} disagrees on {start} -> {goal}");
        }
    }

    totals
}

/// Prints the totals as a table of per-query averages over `queries` queries.
#[expect(
    clippy::cast_precision_loss,
    reason = "averages of small counts for display"
)]
pub fn print_table(totals: &[Totals], queries: usize) {
    let per_query = |count: u64| count as f64 / queries as f64;
    println!(
        "{:<34} {:>12} {:>21} {:>9}",
        "method", "expanded", "forward / backward", "ms"
    );
    for total in totals {
        let sides = if total.forward + total.backward == 0 {
            "(graph untouched)".to_owned()
        } else {
            format!(
                "{:.0} / {:.0}",
                per_query(total.forward),
                per_query(total.backward)
            )
        };
        println!(
            "{:<34} {:>12.0} {:>21} {:>9.3}",
            total.name,
            per_query(total.expanded),
            sides,
            total.time.as_secs_f64() * 1e3 / queries as f64,
        );
    }
    println!("(averages per query; every method found the same cost on every query)");
}

/// The number of queries to run: `release` in release builds, a tenth in debug builds so
/// that they finish quickly too.
pub fn query_count(release: usize) -> usize {
    if cfg!(debug_assertions) {
        release / 10
    } else {
        release
    }
}
