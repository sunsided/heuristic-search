//! Every `*_multi` search against the best single start on random graphs: optimal searches
//! match the cheapest start, bounded ones stay within their weight, fewest-edge searches
//! match the shortest start, and every path is valid and begins at one of the starts.

use heuristic_search::{
    EdgeSelector, EesEstimates, Limits, MhaWeights, SearchResult, Weight, Zero, ana_star_multi,
    ara_star_multi, bae_star_multi, bfs, bidirectional_bfs_multi, bidirectional_dijkstra_multi,
    depth_limited_dfs_multi, dfs_multi, dibbs_multi, dps_multi, ees_multi, focal_fn,
    focal_search_multi, heuristic_fn, ida_star_multi, iddfs_multi, lazy_sp_multi, mha_star_multi,
    nbs_multi,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// The result type of every search over a test [`Graph`].
type Result = SearchResult<usize, usize, u64>;

/// A random graph and goal with up to four random start nodes (possibly repeated or none).
fn instance() -> impl Strategy<Value = (Graph, usize, Vec<usize>)> {
    (graph_and_goal(), prop::collection::vec(0_usize..12, 0..5)).prop_map(|((graph, goal), raw)| {
        let nodes = graph.edges.len();
        let starts = raw.into_iter().map(|s| s % nodes).collect();

        (graph, goal, starts)
    })
}

/// Checks that `result` found a valid path from one of `starts` exactly when `expected` is
/// `Some`, and returns its cost.
fn checked(graph: &Graph, starts: &[usize], result: &Result, expected: Option<u64>) -> Option<u64> {
    let solution = result.solution();
    assert_eq!(solution.is_some(), expected.is_some(), "solvability");
    let solution = solution?;
    assert!(
        starts.contains(&solution.states[0]),
        "path begins at a start"
    );
    assert_valid_path(graph, solution.states[0], solution);

    Some(solution.cost)
}

proptest! {
    /// Optimal searches find the cheapest path over all starts.
    #[test]
    fn optimal_searches_match_best_start((graph, goal, starts) in instance()) {
        let exact = distances_to(&graph, goal);
        let best = starts.iter().filter_map(|&s| exact[s]).min();
        let h = |state: &usize| Some(exact[*state].unwrap_or(0));
        let is_goal = |state: &usize| *state == goal;
        let s = || starts.iter().copied();
        let evaluate = |from: &usize, action: &usize, _: &usize| Some(graph.edges[*from][*action].1);

        let results: Vec<(&str, Result)> = vec![
            ("ida_star", ida_star_multi(&graph, s(), &is_goal, heuristic_fn(h), Limits::NONE)),
            ("mha_star", mha_star_multi(&graph, s(), &is_goal, heuristic_fn(h), [heuristic_fn(h)], MhaWeights::ONE, Limits::NONE)),
            ("lazy_sp", lazy_sp_multi(&graph, s(), &is_goal, heuristic_fn(h), evaluate, EdgeSelector::Forward, Limits::NONE)),
            ("bidirectional_dijkstra", bidirectional_dijkstra_multi(&graph, s(), [goal], Limits::NONE)),
            ("nbs", nbs_multi(&graph, s(), [goal], heuristic_fn(h), Zero, Limits::NONE)),
            ("dibbs", dibbs_multi(&graph, s(), [goal], heuristic_fn(h), Zero, Limits::NONE)),
            ("bae_star", bae_star_multi(&graph, s(), [goal], heuristic_fn(h), Zero, Limits::NONE)),
        ];
        for (name, result) in &results {
            prop_assert_eq!(checked(&graph, &starts, result, best), best, "{}", name);
        }

        for anytime in [
            ana_star_multi(&graph, s(), &is_goal, heuristic_fn(h), |_, _| {}, Limits::NONE),
            ara_star_multi(&graph, s(), &is_goal, heuristic_fn(h), [Weight::integer(2), Weight::ONE], |_, _| {}, Limits::NONE),
        ] {
            prop_assert!(anytime.complete);
            prop_assert_eq!(anytime.solution.map(|solution| solution.cost), best);
        }
    }

    /// Bounded-suboptimal searches stay within their weight of the best start.
    #[test]
    fn bounded_searches_stay_within_weight((graph, goal, starts) in instance()) {
        let exact = distances_to(&graph, goal);
        let best = starts.iter().filter_map(|&s| exact[s]).min();
        let h = |state: &usize| Some(exact[*state].unwrap_or(0));
        let is_goal = |state: &usize| *state == goal;
        let s = || starts.iter().copied();
        let weight = Weight::new(3, 2);
        let estimates = EesEstimates {
            admissible: heuristic_fn(h),
            cost_to_go: heuristic_fn(|state: &usize| h(state).map(|v| 2 * v)),
            distance_to_go: heuristic_fn(|state: &usize| Some(*state as u64)),
        };

        for result in [
            focal_search_multi(&graph, s(), &is_goal, heuristic_fn(h), focal_fn(|state: &usize| *state), weight, Limits::NONE),
            ees_multi(&graph, s(), &is_goal, estimates, weight, Limits::NONE),
            dps_multi(&graph, s(), &is_goal, heuristic_fn(h), weight, Limits::NONE),
        ] {
            if let (Some(cost), Some(best)) = (checked(&graph, &starts, &result, best), best) {
                prop_assert!(2 * cost <= 3 * best);
            }
        }
    }

    /// Depth-first and fewest-edge searches decide reachability from any start, and the
    /// fewest-edge ones find the shortest path over all starts.
    #[test]
    fn uninformed_searches_match_best_start((graph, goal, starts) in instance()) {
        let exact = distances_to(&graph, goal);
        let best = starts.iter().filter_map(|&s| exact[s]).min();
        let is_goal = |state: &usize| *state == goal;
        let s = || starts.iter().copied();
        let fewest = starts
            .iter()
            .filter_map(|&start| bfs(&graph, start, &is_goal, Limits::NONE).into_solution())
            .map(|solution| solution.len())
            .min();
        let nodes = graph.edges.len();

        checked(&graph, &starts, &dfs_multi(&graph, s(), &is_goal, Limits::NONE), best);
        checked(&graph, &starts, &depth_limited_dfs_multi(&graph, s(), &is_goal, nodes, Limits::NONE), best);
        for result in [
            iddfs_multi(&graph, s(), &is_goal, Limits::NONE),
            bidirectional_bfs_multi(&graph, s(), [goal], Limits::NONE),
        ] {
            checked(&graph, &starts, &result, best);
            prop_assert_eq!(result.solution().map(heuristic_search::Solution::len), fewest);
        }
    }
}
