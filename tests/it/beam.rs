//! Beam search and beam-stack search: hand-checked graphs; on random graphs, beam search
//! returns valid paths and tells cutoffs from unsolvable instances, and beam-stack search
//! with tiny widths matches the Bellman-Ford oracle under admissible, usually
//! inconsistent heuristics.

use heuristic_search::{Limits, Outcome, Zero, beam_search, beam_stack_search, heuristic_fn};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// A graph where the cheap-looking branch `0 -> 1` is a dead end and the goal `3` lies
/// behind the dearer branch `0 -> 2`.
fn trap() -> Graph {
    Graph::new(4, &[(0, 1, 1), (0, 2, 5), (2, 3, 1)])
}

/// Width 1 follows the trap and reports a cutoff; width 2 finds the goal.
#[test]
fn beam_search_cutoff_depends_on_width() {
    let is_three = |s: &usize| *s == 3;

    assert_eq!(
        beam_search(&trap(), 0, &is_three, Zero, 1, Limits::NONE).outcome,
        Outcome::Cutoff
    );
    assert_eq!(
        beam_search(&trap(), 0, &is_three, Zero, 2, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(6)
    );
}

/// Without pruning, an unreachable goal is unsolvable; a zero budget stops at once; a
/// start that is a goal is free.
#[test]
fn beam_search_outcomes() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 0, 1)]);
    let is_two = |s: &usize| *s == 2;

    assert_eq!(
        beam_search(&graph, 0, &is_two, Zero, 1, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        beam_search(&graph, 0, &is_two, Zero, 1, Limits::expansions(0)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        beam_search(&graph, 2, &is_two, Zero, 1, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(0)
    );
}

/// Beam-stack search with width 1 backtracks out of the trap.
#[test]
fn beam_stack_search_backtracks_out_of_trap() {
    let mut reported = Vec::new();
    let result = beam_stack_search(
        &trap(),
        0,
        &|s: &usize| *s == 3,
        Zero,
        1,
        |solution| reported.push(solution.cost),
        Limits::NONE,
    );

    assert!(result.complete);
    assert_eq!(result.solution.map(|s| s.states), Some(vec![0, 2, 3]));
    assert_eq!(reported, [6]);
}

/// A budget stop is incomplete, an unreachable goal is complete without a solution, and
/// a start that is a goal is free.
#[test]
fn beam_stack_search_outcomes() {
    let graph = Graph::new(3, &[(0, 1, 0), (1, 0, 0)]);
    let is_two = |s: &usize| *s == 2;

    let stopped = beam_stack_search(&graph, 0, &is_two, Zero, 1, |_| {}, Limits::expansions(0));
    assert!(!stopped.complete);
    assert!(stopped.solution.is_none());

    let unsolvable = beam_stack_search(&graph, 0, &is_two, Zero, 1, |_| {}, Limits::NONE);
    assert!(unsolvable.complete);
    assert!(unsolvable.solution.is_none());

    let free = beam_stack_search(&graph, 2, &is_two, Zero, 1, |_| {}, Limits::NONE);
    assert!(free.complete);
    assert_eq!(free.solution.map(|s| s.cost), Some(0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// Beam-stack search with widths 1 to 3 and an admissible, usually inconsistent
    /// heuristic completes with the oracle's cost, on graphs with zero-cost cycles and
    /// unreachable goals; the reported solutions get strictly cheaper and end in the
    /// returned one.
    #[test]
    fn beam_stack_search_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        width in 1_usize..=3,
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s].map(|d| d * scales[*s] / 4));
        let is_goal = |s: &usize| *s == goal;
        let mut reported = Vec::new();

        let result = beam_stack_search(
            &graph, 0, &is_goal, h, width, |solution| reported.push(solution.cost), Limits::NONE,
        );

        prop_assert!(result.complete);
        prop_assert_eq!(result.solution.as_ref().map(|s| s.cost), exact[0]);
        prop_assert!(reported.windows(2).all(|pair| pair[1] < pair[0]));
        // A start that is a goal is returned without being reported.
        prop_assert_eq!(reported.last().copied(), exact[0].filter(|_| goal != 0));
        if let Some(solution) = &result.solution {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// Beam search returns valid paths no cheaper than the optimum, and says
    /// `Unsolvable` only when no goal is reachable.
    #[test]
    fn beam_search_is_sound(
        (graph, goal) in graph_and_goal(),
        width in 1_usize..=3,
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s]);
        let result = beam_search(&graph, 0, &|s: &usize| *s == goal, h, width, Limits::NONE);

        match &result.outcome {
            Outcome::Found(solution) => {
                assert_valid_path(&graph, 0, solution);
                prop_assert!(exact[0].is_some_and(|best| solution.cost >= best));
            }
            Outcome::Unsolvable => prop_assert_eq!(exact[0], None),
            Outcome::Cutoff => {}
            Outcome::LimitReached => prop_assert!(false, "no budget was set"),
        }
    }
}
