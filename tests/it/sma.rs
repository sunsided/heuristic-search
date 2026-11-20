//! SMA*: hand-checked memory limits, and on random graphs the cheapest path with at most
//! `memory - 1` edges (a bounded Bellman-Ford oracle), `Cutoff` when memory is too small
//! and `Unsolvable` only when no goal is reachable.

use heuristic_search::{Limits, Outcome, Zero, heuristic_fn, sma_star, sma_star_multi};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// The cheapest cost from `start` to `goal` over paths of at most `edges` edges.
fn cheapest_within(graph: &Graph, start: usize, goal: usize, edges: usize) -> Option<u64> {
    let mut best = vec![None; graph.edges.len()];
    best[start] = Some(0_u64);
    for _ in 0..edges {
        let mut next = best.clone();
        for (from, targets) in graph.edges.iter().enumerate() {
            let Some(cost) = best[from] else {
                continue;
            };

            for &(to, edge) in targets {
                if next[to].is_none_or(|old| cost + edge < old) {
                    next[to] = Some(cost + edge);
                }
            }
        }

        best = next;
    }

    best[goal]
}

/// A chain `0 -> 1 -> ... -> 5` with a dear shortcut `0 -> 5`: enough memory finds the
/// chain, less memory the shortcut, too little memory nothing.
#[test]
fn memory_decides_the_path() {
    let graph = Graph::new(
        6,
        &[
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (0, 5, 9),
        ],
    );
    let is_five = |s: &usize| *s == 5;
    let cost = |memory| {
        sma_star(&graph, 0, &is_five, Zero, memory, Limits::NONE)
            .solution()
            .map(|s| s.cost)
    };

    assert_eq!(cost(6), Some(5));
    assert_eq!(cost(5), Some(9));
    assert_eq!(cost(2), Some(9));
    assert_eq!(
        sma_star(&graph, 0, &is_five, Zero, 1, Limits::NONE).outcome,
        Outcome::Cutoff
    );
}

/// Forgetting and regenerating: with room for the path and one sibling, a wide tree still
/// yields the optimum.
#[test]
fn regenerates_forgotten_children() {
    // Five dear children of 0, one cheap path 0 -> 6 -> 7.
    let graph = Graph::new(
        8,
        &[
            (0, 1, 2),
            (0, 2, 2),
            (0, 3, 2),
            (0, 4, 2),
            (0, 5, 2),
            (0, 6, 3),
            (6, 7, 3),
            (1, 7, 9),
        ],
    );
    let result = sma_star(&graph, 0, &|s: &usize| *s == 7, Zero, 3, Limits::NONE);

    assert_eq!(
        result.solution().map(|s| s.states.clone()),
        Some(vec![0, 6, 7])
    );
}

/// A zero budget stops at once, an unreachable goal is unsolvable with enough memory, and
/// a start that is a goal is free.
#[test]
fn outcomes() {
    let graph = Graph::new(3, &[(0, 1, 0), (1, 0, 0)]);
    let is_two = |s: &usize| *s == 2;

    assert_eq!(
        sma_star(&graph, 0, &is_two, Zero, 4, Limits::expansions(0)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        sma_star(&graph, 0, &is_two, Zero, 4, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        sma_star(&graph, 2, &is_two, Zero, 1, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(0)
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// With an admissible, usually inconsistent heuristic, SMA* returns the cheapest path
    /// of at most `memory - 1` edges; without one it reports `Cutoff`, or `Unsolvable`
    /// when no goal is reachable at all (always with memory for every simple path).
    #[test]
    fn sma_star_matches_bounded_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        memory in 1_usize..=14,
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s].map(|d| d * scales[*s] / 4));
        let result = sma_star(&graph, 0, &|s: &usize| *s == goal, h, memory, Limits::expansions(1_000_000));
        let within = cheapest_within(&graph, 0, goal, memory - 1);

        match &result.outcome {
            Outcome::Found(solution) => {
                assert_valid_path(&graph, 0, solution);
                prop_assert_eq!(Some(solution.cost), within);
                prop_assert!(solution.len() < memory);
            }
            Outcome::Cutoff => prop_assert_eq!(within, None),
            Outcome::Unsolvable => prop_assert_eq!(exact[0], None),
            Outcome::LimitReached => prop_assert!(false, "SMA* did not terminate"),
        }

        if memory > graph.edges.len() {
            prop_assert_eq!(result.solution().map(|s| s.cost), exact[0]);
            prop_assert_ne!(&result.outcome, &Outcome::Cutoff);
        }
    }

    /// From several starts under tight memory, SMA* returns the cheapest path of at most
    /// `memory - k` edges over all `k` distinct starts.
    #[test]
    fn sma_star_multi_matches_bounded_oracle(
        (graph, goal) in graph_and_goal(),
        raw in prop::collection::vec(0_usize..12, 1..4),
        memory in 1_usize..=10,
    ) {
        let nodes = graph.edges.len();
        let mut starts: Vec<usize> = raw.iter().map(|s| s % nodes).collect();
        starts.dedup();
        let mut distinct = starts.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let k = distinct.len();
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s]);
        let result = sma_star_multi(&graph, starts.clone(), &|s: &usize| *s == goal, h, memory, Limits::expansions(1_000_000));

        let within = if distinct.contains(&goal) {
            Some(0)
        } else if k > memory {
            None
        } else {
            distinct.iter().filter_map(|&s| cheapest_within(&graph, s, goal, memory - k)).min()
        };

        match &result.outcome {
            Outcome::Found(solution) => {
                prop_assert!(distinct.contains(&solution.states[0]));
                assert_valid_path(&graph, solution.states[0], solution);
                prop_assert_eq!(Some(solution.cost), within);
            }
            Outcome::Cutoff => prop_assert_eq!(within, None),
            Outcome::Unsolvable => prop_assert!(distinct.iter().all(|&s| exact[s].is_none())),
            Outcome::LimitReached => prop_assert!(false, "SMA* did not terminate"),
        }
    }
}
