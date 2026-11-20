//! Recursive best-first search: hand-checked graphs, and the cost of the Bellman-Ford
//! oracle on random graphs with admissible, usually inconsistent heuristics.

use heuristic_search::{Limits, Outcome, Successors, Zero, heuristic_fn, rbfs};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// RBFS backs up the value of the cheap-looking branch and switches to the branch that
/// is cheaper in the end.
#[test]
fn switches_to_cheaper_branch() {
    // 0 -> 1 -> 2 -> 5 costs 1 + 1 + 10, 0 -> 3 -> 4 -> 5 costs 2 + 2 + 2.
    let graph = Graph::new(
        6,
        &[
            (0, 1, 1),
            (1, 2, 1),
            (2, 5, 10),
            (0, 3, 2),
            (3, 4, 2),
            (4, 5, 2),
        ],
    );
    let result = rbfs(&graph, 0, &|s: &usize| *s == 5, Zero, Limits::NONE);
    let solution = result.solution().expect("5 is reachable");

    assert_eq!(solution.states, [0, 3, 4, 5]);
    assert_eq!(solution.cost, 6);
}

/// A zero budget stops before the first expansion, an unreachable goal is unsolvable
/// even with a cycle, a start that is a goal is free even when called a dead end, and a
/// dead-end start is unsolvable.
#[test]
fn outcomes() {
    let graph = Graph::new(3, &[(0, 1, 0), (1, 0, 0)]);
    let is_two = |s: &usize| *s == 2;
    let dead = heuristic_fn(|_: &usize| None::<u64>);

    assert_eq!(
        rbfs(&graph, 0, &is_two, Zero, Limits::expansions(0)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        rbfs(&graph, 0, &is_two, Zero, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        rbfs(&graph, 2, &is_two, dead, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(0)
    );
    assert_eq!(
        rbfs(&graph, 0, &is_two, dead, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

/// Counting up by one, at cost 1, without end.
struct Up;

impl Successors for Up {
    type State = u32;
    type Action = ();
    type Cost = u32;

    fn successors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
        emit((), n + 1, 1);
    }
}

/// A path of 20 000 edges does not overflow the call stack.
#[test]
fn deep_paths_do_not_overflow() {
    let h = heuristic_fn(|n: &u32| 20_000_u32.checked_sub(*n));
    let result = rbfs(&Up, 0, &|n: &u32| *n == 20_000, h, Limits::NONE);

    assert_eq!(result.solution().map(|s| s.cost), Some(20_000));
}

proptest! {
    /// RBFS with an admissible, usually inconsistent heuristic (with `None` for real dead
    /// ends) finds the oracle's cost, on graphs with zero-cost cycles.
    #[test]
    fn rbfs_matches_oracle(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|s: &usize| exact[*s].map(|d| d * scales[*s] / 4));
        let is_goal = |s: &usize| *s == goal;

        let result = rbfs(&graph, 0, &is_goal, h, Limits::NONE);

        prop_assert_eq!(result.solution().map(|s| s.cost), exact[0]);
        if let Some(solution) = result.solution() {
            assert_valid_path(&graph, 0, solution);
        }
    }
}
