//! Fringe search: hand-checked graphs, and the cost of the Bellman-Ford oracle on random
//! graphs with admissible, usually inconsistent heuristics.

use heuristic_search::{Limits, Outcome, Zero, fringe_search, heuristic_fn};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// The cheapest path wins over the one with fewer edges, and the threshold rises twice.
#[test]
fn finds_cheapest_path() {
    // 0 -> 3 costs 10 directly, 3 along 0 -> 1 -> 2 -> 3.
    let graph = Graph::new(4, &[(0, 3, 10), (0, 1, 1), (1, 2, 1), (2, 3, 1)]);
    let result = fringe_search(&graph, 0, &|s: &usize| *s == 3, Zero, Limits::NONE);
    let solution = result.solution().expect("3 is reachable");

    assert_eq!(solution.states, [0, 1, 2, 3]);
    assert_eq!(solution.cost, 3);
}

/// An inconsistent heuristic makes a state's first path too expensive; fringe search
/// reopens it and still returns the optimum.
#[test]
fn reopens_states_under_inconsistent_heuristic() {
    // 0 -> 1 (cost 4) and 0 -> 2 -> 1 (cost 1 + 1), then 1 -> 3 (cost 4).
    let graph = Graph::new(4, &[(0, 1, 4), (0, 2, 1), (2, 1, 1), (1, 3, 4)]);
    // Admissible (exact distances are 6, 4, 5, 0) but inconsistent at 2, so 1 is expanded
    // through the expensive edge first.
    let h = heuristic_fn(|s: &usize| Some([6_u64, 0, 5, 0][*s]));
    let result = fringe_search(&graph, 0, &|s: &usize| *s == 3, h, Limits::NONE);

    assert_eq!(result.solution().map(|s| s.cost), Some(6));
    assert_eq!(result.stats.reopened, 1);
}

/// A zero budget stops before the first expansion, an unreachable goal is unsolvable, a
/// start that is a goal is free even when called a dead end, and a dead-end start is
/// unsolvable.
#[test]
fn outcomes() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 0, 1)]);
    let is_two = |s: &usize| *s == 2;
    let dead = heuristic_fn(|_: &usize| None::<u64>);

    assert_eq!(
        fringe_search(&graph, 0, &is_two, Zero, Limits::expansions(0)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        fringe_search(&graph, 0, &is_two, Zero, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        fringe_search(&graph, 2, &is_two, dead, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(0)
    );
    assert_eq!(
        fringe_search(&graph, 0, &is_two, dead, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

proptest! {
    /// Fringe search with an admissible, usually inconsistent heuristic (with `None` for
    /// real dead ends) finds the oracle's cost, on graphs with zero-cost cycles.
    #[test]
    fn fringe_search_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s].map(|d| d * scales[*s] / 4));
        let is_goal = |s: &usize| *s == goal;

        let result = fringe_search(&graph, 0, &is_goal, h, Limits::NONE);

        prop_assert_eq!(result.solution().map(|s| s.cost), exact[0]);
        if let Some(solution) = result.solution() {
            assert_valid_path(&graph, 0, solution);
        }
    }
}
