//! `LazySP`: optimal against Dijkstra on the true graph for every edge selector, with each
//! edge evaluated at most once.

use std::collections::HashSet;

use heuristic_search::{EdgeSelector, Limits, Outcome, Zero, dijkstra, lazy_sp};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, graph_and_goal};

/// Every selector.
const SELECTORS: [EdgeSelector; 5] = [
    EdgeSelector::Forward,
    EdgeSelector::Reverse,
    EdgeSelector::Alternate,
    EdgeSelector::Bisection,
    EdgeSelector::All,
];

/// A zero budget stops `LazySP` in its first inner search.
#[test]
fn budget_stops_search() {
    let graph = Graph::new(2, &[(0, 1, 1)]);
    let evaluate = |_: &usize, _: &usize, _: &usize| Some(1_u64);

    let result = lazy_sp(
        &graph,
        0,
        &|s: &usize| *s == 1,
        Zero,
        evaluate,
        EdgeSelector::Forward,
        Limits::expansions(0),
    );

    assert_eq!(result.outcome, Outcome::LimitReached);
}

proptest! {
    /// With lazy costs at most the true costs and random infeasible edges, `LazySP` finds
    /// the cost Dijkstra finds on the true graph, and evaluates no edge twice.
    #[test]
    fn lazy_sp_matches_dijkstra(
        (graph, goal) in graph_and_goal(),
        fractions in prop::collection::vec(0_u64..=4, 36),
        blocked in prop::collection::vec(prop::bool::weighted(0.2), 36),
    ) {
        let is_goal = |s: &usize| *s == goal;
        let edge_number = |from: usize, index: usize| graph.edges[..from].iter().map(Vec::len).sum::<usize>() + index;

        let lazy = Graph {
            edges: graph.edges.iter().enumerate().map(|(from, edges)| {
                edges.iter().enumerate().map(|(index, &(to, cost))| (to, cost * fractions[edge_number(from, index) % 36] / 4)).collect()
            }).collect(),
        };
        let truth = Graph {
            edges: graph.edges.iter().enumerate().map(|(from, edges)| {
                edges.iter().enumerate().map(|(index, &(to, cost))| {
                    if blocked[edge_number(from, index) % 36] { (to, u64::MAX / 64) } else { (to, cost) }
                }).collect()
            }).collect(),
        };
        let feasible = Graph {
            edges: graph.edges.iter().enumerate().map(|(from, edges)| {
                edges.iter().enumerate().filter(|(index, _)| !blocked[edge_number(from, *index) % 36]).map(|(_, &edge)| edge).collect()
            }).collect(),
        };
        let optimum = dijkstra(&feasible, 0, &is_goal, Limits::NONE).into_solution().map(|s| s.cost);

        for selector in SELECTORS {
            let mut seen = HashSet::new();
            let mut twice = false;
            let evaluate = |from: &usize, action: &usize, _: &usize| {
                twice |= !seen.insert((*from, *action));
                (!blocked[edge_number(*from, *action) % 36]).then(|| graph.edges[*from][*action].1)
            };

            let result = lazy_sp(&lazy, 0, &is_goal, Zero, evaluate, selector, Limits::NONE);

            prop_assert!(!twice, "{:?} evaluated an edge twice", selector);
            prop_assert_eq!(result.solution().map(|s| s.cost), optimum, "{:?}", selector);
            if let Some(solution) = result.solution() {
                assert_valid_path(&truth, 0, solution);
            }
        }
    }
}
