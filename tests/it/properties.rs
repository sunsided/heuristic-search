//! Property tests on random weighted graphs: A* with admissible heuristics matches
//! Dijkstra, Weighted A* stays within its bound, BFS finds the fewest edges, and every
//! returned path is valid.

use heuristic_search::{Limits, Weight, astar, bfs, dijkstra, gbfs, weighted_astar};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path};

/// Exact cost from every node to `goal`, or `None` if `goal` is unreachable, computed by
/// running Dijkstra backwards from `goal` once per node.
fn distances_to(graph: &Graph, goal: usize) -> Vec<Option<u64>> {
    let reversed = graph.reversed();

    (0..graph.edges.len())
        .map(|node| {
            dijkstra(
                &reversed,
                goal,
                &|state: &usize| *state == node,
                Limits::NONE,
            )
            .into_solution()
            .map(|solution| solution.cost)
        })
        .collect()
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
}
