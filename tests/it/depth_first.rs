//! Depth-first searches: DFS is complete, depth-limited DFS and IDDFS find the fewest
//! edges, IDA* (with or without a transposition table) the cheapest path; all
//! cross-checked against the Bellman-Ford oracle.

use heuristic_search::{
    Limits, Outcome, depth_limited_dfs, dfs, heuristic_fn, ida_star, ida_star_tt, iddfs,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// A zero budget stops every search before its first expansion.
#[test]
fn budget_stops_search() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 2, 1)]);
    let is_two = |s: &usize| *s == 2;
    let zero = heuristic_fn(|_: &usize| Some(0_u64));
    let none = Limits::expansions(0);

    assert_eq!(dfs(&graph, 0, &is_two, none).outcome, Outcome::LimitReached);
    assert_eq!(
        depth_limited_dfs(&graph, 0, &is_two, 5, none).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        iddfs(&graph, 0, &is_two, none).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        ida_star(&graph, 0, &is_two, zero, none).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        ida_star_tt(&graph, 0, &is_two, zero, 8, none).outcome,
        Outcome::LimitReached
    );
}

/// A depth cutoff and a budget stop are different outcomes.
#[test]
fn cutoff_differs_from_budget_stop() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 2, 1)]);
    let is_two = |s: &usize| *s == 2;

    assert_eq!(
        depth_limited_dfs(&graph, 0, &is_two, 1, Limits::NONE).outcome,
        Outcome::Cutoff
    );
    assert_eq!(
        depth_limited_dfs(&graph, 0, &is_two, 1, Limits::expansions(1)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        depth_limited_dfs(&graph, 0, &|_: &usize| false, 5, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

/// A start that is a goal costs nothing, even when the heuristic calls it a dead end.
#[test]
fn start_goal_is_free() {
    let graph = Graph::new(1, &[(0, 0, 0)]);
    let is_zero = |s: &usize| *s == 0;

    assert_eq!(
        ida_star(
            &graph,
            0,
            &is_zero,
            heuristic_fn(|_: &usize| None::<u64>),
            Limits::NONE
        )
        .solution()
        .map(|s| s.cost),
        Some(0)
    );
    assert_eq!(
        iddfs(&graph, 0, &is_zero, Limits::NONE)
            .solution()
            .map(heuristic_search::Solution::len),
        Some(0)
    );
    assert_eq!(
        dfs(&graph, 0, &is_zero, Limits::NONE)
            .solution()
            .map(heuristic_search::Solution::len),
        Some(0)
    );
}

proptest! {
    /// IDA* with an admissible, usually inconsistent heuristic (with `None` for real dead
    /// ends) finds the oracle's cost, on graphs with zero-cost cycles.
    #[test]
    fn ida_star_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s].map(|d| d * scales[*s] / 4));
        let is_goal = |s: &usize| *s == goal;

        let result = ida_star(&graph, 0, &is_goal, h, Limits::NONE);

        prop_assert_eq!(result.solution().map(|s| s.cost), exact[0]);
        if let Some(solution) = result.solution() {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// IDA* with a transposition table of any capacity, down to a single slot or none,
    /// finds the oracle's cost, also on graphs with zero-cost cycles.
    #[test]
    fn ida_star_tt_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        capacity in prop_oneof![Just(0_usize), Just(1), Just(3), Just(64)],
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s].map(|d| d * scales[*s] / 4));
        let is_goal = |s: &usize| *s == goal;

        let result = ida_star_tt(&graph, 0, &is_goal, h, capacity, Limits::NONE);

        prop_assert_eq!(result.solution().map(|s| s.cost), exact[0]);
        if let Some(solution) = result.solution() {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// IDDFS finds as few edges as the oracle on the unit-cost graph; depth-limited DFS
    /// finds a path at exactly that limit and none below it; DFS is complete.
    #[test]
    fn depth_first_matches_oracle((graph, goal) in graph_and_goal()) {
        let fewest = distances_to(&graph.unit(), goal)[0];
        let is_goal = |s: &usize| *s == goal;

        let deepening = iddfs(&graph, 0, &is_goal, Limits::NONE);
        let any = dfs(&graph, 0, &is_goal, Limits::NONE);

        prop_assert_eq!(deepening.solution().map(|s| s.len() as u64), fewest);
        prop_assert_eq!(any.solution().is_some(), fewest.is_some());
        for solution in [deepening.solution(), any.solution()].into_iter().flatten() {
            assert_valid_path(&graph, 0, solution);
        }

        match fewest {
            None => prop_assert_eq!(deepening.outcome, Outcome::Unsolvable),
            Some(edges) => {
                let limit = usize::try_from(edges).unwrap();
                let at = depth_limited_dfs(&graph, 0, &is_goal, limit, Limits::NONE);
                prop_assert!(at.solution().is_some_and(|s| s.len() <= limit));
                if let Some(below) = limit.checked_sub(1) {
                    let short = depth_limited_dfs(&graph, 0, &is_goal, below, Limits::NONE);
                    prop_assert_eq!(short.outcome, Outcome::Cutoff);
                }
            }
        }
    }
}
