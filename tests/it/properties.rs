//! Property tests on random weighted graphs: A* with admissible heuristics matches
//! Dijkstra, Weighted A* stays within its bound, BFS finds the fewest edges, and every
//! returned path is valid. Searches from several starts match the best single start.

use heuristic_search::{
    Limits, Weight, astar, astar_multi, bfs, bfs_multi, dijkstra, dijkstra_multi, gbfs,
    weighted_astar,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path};

/// Exact cost from every node to `goal`, or `None` if `goal` is unreachable.
///
/// Bellman-Ford relaxation, independent of the crate's own searches so it can serve as the
/// oracle for them.
fn distances_to(graph: &Graph, goal: usize) -> Vec<Option<u64>> {
    let mut distance = vec![None; graph.edges.len()];
    distance[goal] = Some(0);

    for _ in 0..graph.edges.len() {
        for (from, edges) in graph.edges.iter().enumerate() {
            for &(to, cost) in edges {
                let Some(via) = distance[to].map(|d: u64| d + cost) else {
                    continue;
                };

                if distance[from].is_none_or(|d| via < d) {
                    distance[from] = Some(via);
                }
            }
        }
    }

    distance
}

/// A random graph with up to 12 nodes and edge costs `0..=9`, plus a goal node.
fn graph_and_goal() -> impl Strategy<Value = (Graph, usize)> {
    (2_usize..12).prop_flat_map(|nodes| {
        let edges = prop::collection::vec((0..nodes, 0..nodes, 0_u64..10), 0..nodes * 3);

        (edges, 0..nodes).prop_map(move |(edges, goal)| (Graph::new(nodes, &edges), goal))
    })
}

proptest! {
    /// With `h = scale(d*)` for random per-node scales in `[0, 1]`, the heuristic is
    /// admissible but usually inconsistent, which exercises reopening.
    #[test]
    fn astar_matches_dijkstra(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
    ) {
        let exact = distances_to(&graph, goal);
        let h = |state: &usize| Some(exact[*state].map_or(0, |d| d * scales[*state] / 4));
        let is_goal = |state: &usize| *state == goal;

        let optimal = dijkstra(&graph, 0, &is_goal, Limits::NONE).into_solution();
        let informed = astar(&graph, 0, &is_goal, &h, Limits::NONE).into_solution();

        prop_assert_eq!(optimal.as_ref().map(|s| s.cost), exact[0]);
        prop_assert_eq!(informed.as_ref().map(|s| s.cost), exact[0]);
        if let Some(solution) = &informed {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// Weighted A* finds a solution whenever one exists, within `w` times the optimum.
    #[test]
    fn weighted_astar_is_bounded(
        (graph, goal) in graph_and_goal(),
        numerator in 1_u32..8,
    ) {
        let weight = Weight::new(numerator + 2, 2);
        let exact = distances_to(&graph, goal);
        let h = |state: &usize| Some(exact[*state].unwrap_or(0));
        let is_goal = |state: &usize| *state == goal;

        let solution = weighted_astar(&graph, 0, &is_goal, &h, weight, Limits::NONE).into_solution();

        prop_assert_eq!(solution.is_some(), exact[0].is_some());
        if let (Some(solution), Some(optimum)) = (&solution, exact[0]) {
            assert_valid_path(&graph, 0, solution);
            prop_assert!(2 * solution.cost <= u64::from(numerator + 2) * optimum);
        }
    }

    /// GBFS is complete on finite graphs and returns valid paths.
    #[test]
    fn gbfs_is_complete((graph, goal) in graph_and_goal()) {
        let exact = distances_to(&graph, goal);
        let is_goal = |state: &usize| *state == goal;
        let h = |state: &usize| Some(*state as u64);

        let solution = gbfs(&graph, 0, &is_goal, &h, Limits::NONE).into_solution();

        prop_assert_eq!(solution.is_some(), exact[0].is_some());
        if let Some(solution) = &solution {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// BFS finds as few edges as Dijkstra on the unit-cost version of the graph.
    #[test]
    fn bfs_finds_fewest_edges((graph, goal) in graph_and_goal()) {
        let unit = graph.unit();
        let is_goal = |state: &usize| *state == goal;

        let shortest = bfs(&graph, 0, &is_goal, Limits::NONE).into_solution();
        let unit_optimal = dijkstra(&unit, 0, &is_goal, Limits::NONE).into_solution();

        prop_assert_eq!(shortest.as_ref().map(|s| s.len() as u64), unit_optimal.map(|s| s.cost));
        if let Some(solution) = &shortest {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// Searching from a set of starts finds the minimum over searching from each alone:
    /// cheapest cost for Dijkstra and A*, fewest edges for BFS.
    #[test]
    fn multi_start_matches_best_single_start(
        (graph, goal) in graph_and_goal(),
        raw_starts in prop::collection::vec(0_usize..12, 0..5),
    ) {
        let nodes = graph.edges.len();
        let starts: Vec<usize> = raw_starts.into_iter().map(|s| s % nodes).collect();
        let exact = distances_to(&graph, goal);
        let h = |state: &usize| Some(exact[*state].unwrap_or(0));
        let is_goal = |state: &usize| *state == goal;

        let best_cost = starts.iter().filter_map(|&s| exact[s]).min();
        let best_len = starts
            .iter()
            .filter_map(|&s| bfs(&graph, s, &is_goal, Limits::NONE).into_solution())
            .map(|solution| solution.len())
            .min();

        let cheapest = dijkstra_multi(&graph, starts.iter().copied(), &is_goal, Limits::NONE);
        let informed = astar_multi(&graph, starts.iter().copied(), &is_goal, &h, Limits::NONE);
        let shortest = bfs_multi(&graph, starts.iter().copied(), &is_goal, Limits::NONE);

        prop_assert_eq!(cheapest.solution().map(|s| s.cost), best_cost);
        prop_assert_eq!(informed.solution().map(|s| s.cost), best_cost);
        prop_assert_eq!(shortest.solution().map(heuristic_search::Solution::len), best_len);
        for solution in [&cheapest, &informed, &shortest].into_iter().filter_map(|r| r.solution()) {
            prop_assert!(starts.contains(&solution.states[0]));
            assert_valid_path(&graph, solution.states[0], solution);
        }
    }
}
