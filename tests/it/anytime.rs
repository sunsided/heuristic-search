//! Anytime searches: ARA* and ANA* on hand-made graphs and cross-checked against the
//! distance oracle on random graphs.

use heuristic_search::{Limits, Weight, ana_star, ara_star, heuristic_fn};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// A graph where the greedy route is expensive: `0 -> 1 -> 4` costs 20, while
/// `0 -> 2 -> 3 -> 4` costs 6. A misleading heuristic makes `1` look closest.
fn trap() -> Graph {
    Graph::new(
        5,
        &[(0, 1, 10), (1, 4, 10), (0, 2, 2), (2, 3, 2), (3, 4, 2)],
    )
}

/// Heuristic for [`trap`]: admissible, but ranks `1` best.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the heuristic closure signature takes a reference"
)]
fn trap_h(state: &usize) -> u64 {
    [4, 1, 4, 2, 0][*state]
}

/// The weight schedule `3, 2, 3/2, 1` used throughout.
fn schedule() -> [Weight; 4] {
    [
        Weight::integer(3),
        Weight::integer(2),
        Weight::new(3, 2),
        Weight::ONE,
    ]
}

/// Goal test for node 4.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the `Goal` closure signature takes a reference"
)]
fn is_four(state: &usize) -> bool {
    *state == 4
}

/// ANA* first follows the heuristic into the trap, then improves to the optimum.
#[test]
fn ana_star_improves_until_optimal() {
    let graph = trap();
    let mut reports = Vec::new();
    let result = ana_star(
        &graph,
        0,
        &is_four,
        heuristic_fn(|s: &usize| Some(trap_h(s))),
        |solution, bound| reports.push((solution.cost, bound)),
        Limits::NONE,
    );

    assert!(result.complete);
    assert_eq!(result.solution.map(|s| s.cost), Some(6));
    assert_eq!(reports.first().map(|r| r.0), Some(20));
    assert_eq!(reports.last(), Some(&(6, 6)));
}

/// ARA* reports a first solution within the first weight and ends optimal.
#[test]
fn ara_star_ends_optimal() {
    let graph = trap();
    let mut reports = Vec::new();
    let result = ara_star(
        &graph,
        0,
        &is_four,
        heuristic_fn(|s: &usize| Some(trap_h(s))),
        schedule(),
        |solution, bound| reports.push((solution.cost, bound)),
        Limits::NONE,
    );

    assert!(result.complete);
    assert_eq!(result.solution.map(|s| s.cost), Some(6));
    assert!(reports.windows(2).all(|pair| pair[1].0 < pair[0].0));
}

/// A budget that stops the search after the first solution keeps that solution.
#[test]
fn budget_keeps_the_incumbent() {
    let graph = trap();
    let result = ana_star(
        &graph,
        0,
        &is_four,
        heuristic_fn(|s: &usize| Some(trap_h(s))),
        |_, _| {},
        Limits::expansions(2),
    );

    assert!(!result.complete);
    assert_eq!(result.solution.map(|s| s.cost), Some(20));
}

/// Unreachable goals end complete without a solution; a start goal costs nothing.
#[test]
fn unsolvable_and_trivial() {
    let graph = Graph::new(3, &[(0, 1, 1)]);
    let never = |state: &usize| *state == 2;
    let start = |state: &usize| *state == 0;
    let zero = || heuristic_fn(|_: &usize| Some(0_u64));

    let ara = ara_star(
        &graph,
        0,
        &never,
        zero(),
        schedule(),
        |_, _| {},
        Limits::NONE,
    );
    let ana = ana_star(&graph, 0, &never, zero(), |_, _| {}, Limits::NONE);
    assert!(ara.complete && ara.solution.is_none());
    assert!(ana.complete && ana.solution.is_none());

    let ara = ara_star(
        &graph,
        0,
        &start,
        zero(),
        schedule(),
        |_, _| {},
        Limits::NONE,
    );
    let ana = ana_star(&graph, 0, &start, zero(), |_, _| {}, Limits::NONE);
    assert_eq!(ara.solution.map(|s| s.cost), Some(0));
    assert_eq!(ana.solution.map(|s| s.cost), Some(0));
}

proptest! {
    /// ARA* with a consistent heuristic (the exact distance scaled by a common factor)
    /// ends optimal, and every report is a valid bound pair.
    #[test]
    fn ara_star_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scale in 0_u64..=4,
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|state: &usize| Some(exact[*state].map_or(0, |d| d * scale / 4)));
        let is_goal = |state: &usize| *state == goal;
        let mut reports = Vec::new();

        let result = ara_star(&graph, 0, &is_goal, h, schedule(), |s, b| reports.push((s.cost, b)), Limits::NONE);

        prop_assert!(result.complete);
        prop_assert_eq!(result.solution.as_ref().map(|s| s.cost), exact[0]);
        if let Some(solution) = &result.solution {
            assert_valid_path(&graph, 0, solution);
        }
        for &(cost, bound) in &reports {
            prop_assert!(bound <= exact[0].unwrap() && exact[0].unwrap() <= cost);
        }
    }

    /// ANA* with an admissible but inconsistent heuristic ends optimal.
    #[test]
    fn ana_star_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|state: &usize| Some(exact[*state].map_or(0, |d| d * scales[*state] / 4)));
        let is_goal = |state: &usize| *state == goal;
        let mut reports = Vec::new();

        let result = ana_star(&graph, 0, &is_goal, h, |s, b| reports.push((s.cost, b)), Limits::NONE);

        prop_assert!(result.complete);
        prop_assert_eq!(result.solution.as_ref().map(|s| s.cost), exact[0]);
        if let Some(solution) = &result.solution {
            assert_valid_path(&graph, 0, solution);
        }
        prop_assert!(reports.windows(2).all(|pair| pair[1].0 < pair[0].0));
        for &(cost, bound) in &reports {
            prop_assert!(bound <= exact[0].unwrap() && exact[0].unwrap() <= cost);
        }
    }
}
