//! Hash-distributed A*: the same cost as A* on random graphs with admissible, usually
//! inconsistent heuristics, on one to four threads; the start and budget edge cases; and
//! many repeated runs on a grid to catch races in the termination check.

use heuristic_search::{
    Limits, Outcome, Successors, Zero, a_star, a_star_multi, hda_star, hda_star_multi, heuristic_fn,
};
use proptest::prelude::*;

use crate::fixtures::{Cell, Graph, Grid, assert_valid_path, distances_to, graph_and_goal};

/// Counting up by one at cost 1, without end: only a budget stops a search for a goal that
/// is never reached.
struct Up;

impl Successors for Up {
    type State = u64;
    type Action = ();
    type Cost = u64;

    fn successors(&self, n: &u64, mut emit: impl FnMut((), u64, u64)) {
        emit((), n + 1, 1);
    }
}

/// A 16 by 16 grid with a wall across row 8 that leaves a gap at the right end, or no gap
/// if `open` is `false`.
fn walled_grid(open: bool) -> Grid {
    let mut rows = vec![".".repeat(16); 16];
    rows[8] = if open {
        format!("{}..", "#".repeat(14))
    } else {
        "#".repeat(16)
    };
    let rows: Vec<&str> = rows.iter().map(String::as_str).collect();

    Grid::parse(&rows)
}

proptest! {
    /// With `h = scale(d*)` for random per-node scales in `[0, 1]`, the heuristic is
    /// admissible but usually inconsistent, which exercises reopening across threads.
    #[test]
    fn hda_star_matches_a_star(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        threads in 1_usize..=4,
    ) {
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|state: &usize| Some(exact[*state].map_or(0, |d| d * scales[*state] / 4)));
        let is_goal = |state: &usize| *state == goal;

        let sequential = a_star(&graph, 0, &is_goal, h, Limits::NONE).into_solution();
        let parallel = hda_star(&graph, 0, &is_goal, h, threads, Limits::NONE).into_solution();

        prop_assert_eq!(parallel.as_ref().map(|s| s.cost), sequential.map(|s| s.cost));
        prop_assert_eq!(parallel.as_ref().map(|s| s.cost), exact[0]);
        if let Some(solution) = &parallel {
            assert_valid_path(&graph, 0, solution);
        }
    }

    /// From several starts, the cost matches A* from the same starts, and the path begins
    /// at one of them.
    #[test]
    fn hda_star_multi_matches_a_star_multi(
        (graph, goal) in graph_and_goal(),
        raw in prop::collection::vec(0_usize..12, 0..5),
        threads in 1_usize..=4,
    ) {
        let starts: Vec<usize> = raw.into_iter().map(|s| s % graph.edges.len()).collect();
        let exact = distances_to(&graph, goal);
        let h = heuristic_fn(|state: &usize| Some(exact[*state].unwrap_or(0)));
        let is_goal = |state: &usize| *state == goal;
        let s = || starts.iter().copied();

        let sequential = a_star_multi(&graph, s(), &is_goal, h, Limits::NONE).into_solution();
        let parallel = hda_star_multi(&graph, s(), &is_goal, h, threads, Limits::NONE).into_solution();

        prop_assert_eq!(parallel.as_ref().map(|s| s.cost), sequential.map(|s| s.cost));
        if let Some(solution) = &parallel {
            prop_assert!(starts.contains(&solution.states[0]));
            assert_valid_path(&graph, solution.states[0], solution);
        }
    }
}

/// An unreachable goal is unsolvable after the whole reachable space is searched. Several
/// threads may expand a state again when a cheaper path to it arrives late.
#[test]
fn unreachable_goal_is_unsolvable() {
    let graph = Graph::new(4, &[(0, 1, 1), (1, 0, 1), (1, 2, 3), (3, 0, 1)]);

    for threads in 1..=4 {
        let result = hda_star(&graph, 0, &|s: &usize| *s == 3, Zero, threads, Limits::NONE);

        assert_eq!(result.outcome, Outcome::Unsolvable);
        assert!(result.stats.expanded >= 3);
        if threads == 1 {
            assert_eq!(result.stats.expanded, 3);
        }
    }
}

/// A start that is a goal is a zero-cost path without any expansion, whatever the
/// heuristic says; an empty set of starts is unsolvable.
#[test]
fn start_cases() {
    let graph = Graph::new(2, &[(0, 1, 1)]);
    let misleading = heuristic_fn(|_: &usize| Some(100));

    let result = hda_star(&graph, 1, &|s: &usize| *s == 1, misleading, 4, Limits::NONE);
    let solution = result.solution().expect("the start is the goal");
    assert_eq!(solution.states, [1]);
    assert_eq!(solution.cost, 0);
    assert_eq!(result.stats.expanded, 0);

    let none = hda_star_multi(&graph, [], &|s: &usize| *s == 1, Zero, 4, Limits::NONE);
    assert_eq!(none.outcome, Outcome::Unsolvable);
}

/// No thread count is `0`: it runs on one thread.
#[test]
fn zero_threads_runs_on_one() {
    let graph = Graph::new(3, &[(0, 1, 2), (1, 2, 2), (0, 2, 5)]);
    let result = hda_star(&graph, 0, &|s: &usize| *s == 2, Zero, 0, Limits::NONE);

    assert_eq!(result.solution().map(|s| s.cost), Some(4));
}

/// A zero budget stops before any expansion; a budget of 100 expansions stops a search
/// that never ends otherwise, within 64 expansions per thread of the limit.
#[test]
fn budget_stops_the_search() {
    let never = |_: &u64| false;

    for threads in 1..=4 {
        let result = hda_star(&Up, 0, &never, Zero, threads, Limits::expansions(0));
        assert_eq!(result.outcome, Outcome::LimitReached);
        assert_eq!(result.stats.expanded, 0);

        let result = hda_star(&Up, 0, &never, Zero, threads, Limits::expansions(100));
        assert_eq!(result.outcome, Outcome::LimitReached);
        assert!(result.stats.expanded >= 100);
        assert!(result.stats.expanded <= 100 + 65 * (threads as u64 + 1));
    }
}

/// Many runs on a grid, solvable and not, all end with the optimal cost or unsolvable.
///
/// The searches end only through the termination check (the unsolvable grid always, the
/// solvable one after its goal), so a race in it shows up as a hang or a wrong answer.
#[test]
fn repeated_runs_terminate_correctly() {
    let open = walled_grid(true);
    let closed = walled_grid(false);
    let goal: Cell = (15, 0);
    let is_goal = |cell: &Cell| *cell == goal;
    let h = heuristic_fn(|cell: &Cell| {
        u32::try_from(cell.0.abs_diff(goal.0) + cell.1.abs_diff(goal.1)).ok()
    });
    let optimal = a_star(&open, (0, 0), &is_goal, h, Limits::NONE)
        .solution()
        .map(|s| s.cost);
    assert_eq!(optimal, Some(15 + 2 * 14));

    for run in 0..200 {
        let threads = 1 + run % 4;
        let found = hda_star(&open, (0, 0), &is_goal, h, threads, Limits::NONE);
        assert_eq!(found.solution().map(|s| s.cost), optimal, "run {run}");

        let blocked = hda_star(&closed, (0, 0), &is_goal, Zero, threads, Limits::NONE);
        assert_eq!(blocked.outcome, Outcome::Unsolvable, "run {run}");
        assert!(blocked.stats.expanded >= 8 * 16, "run {run}");
    }
}

/// A panic on one thread ends the whole search instead of leaving the others waiting.
#[test]
#[should_panic(expected = "goal test failed")]
fn panic_on_one_thread_propagates() {
    let grid = walled_grid(true);
    let goal = |cell: &Cell| {
        assert!(*cell != (4, 4), "goal test failed");
        false
    };

    let _ = hda_star(&grid, (0, 0), &goal, Zero, 4, Limits::NONE);
}
