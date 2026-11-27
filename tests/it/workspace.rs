//! Reusable workspaces and resumable searches: searches in one reused [`Workspace`] return
//! exactly what fresh searches return, and [`AStarSearch`] spread over many small budgets
//! returns what one call of the matching search ([`a_star`], [`dijkstra`],
//! [`weighted_a_star`], [`gbfs`]) does, with the same total counters.

use heuristic_search::{
    AStarSearch, Heuristic, Limits, Outcome, Stats, Weight, Workspace, Zero, a_star, a_star_in,
    a_star_multi, a_star_multi_in, dijkstra, dijkstra_in, dijkstra_multi, dijkstra_multi_in, gbfs,
    gbfs_in, gbfs_multi, gbfs_multi_in, heuristic_fn, weighted_a_star, weighted_a_star_in,
    weighted_a_star_multi, weighted_a_star_multi_in,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, distances_to, graph_and_goal};

proptest! {
    /// One workspace serves a sequence of queries with different starts, goals and budgets,
    /// including budget-cut searches and starts that are goals; every result, counters
    /// included, equals the search without a workspace.
    #[test]
    fn workspace_searches_match_fresh_searches(
        (graph, _) in graph_and_goal(),
        queries in prop::collection::vec((0_usize..12, 0_usize..12, prop::option::of(0_u64..6)), 1..8),
    ) {
        let nodes = graph.edges.len();
        let mut workspace = Workspace::new();
        let weight = Weight::new(3, 2);

        for (start, goal, limit) in queries {
            let (start, goal) = (start % nodes, goal % nodes);
            let exact = distances_to(&graph, goal);
            let h = |state: &usize| Some(exact[*state].map_or(0, |d| d / 2));
            let is_goal = |state: &usize| *state == goal;
            let budget = limit.map_or(Limits::NONE, Limits::expansions);
            let next = (start + 1) % nodes;

            prop_assert_eq!(
                a_star_in(&mut workspace, &graph, start, &is_goal, heuristic_fn(h), budget),
                a_star(&graph, start, &is_goal, heuristic_fn(h), budget)
            );
            prop_assert_eq!(
                dijkstra_in(&mut workspace, &graph, start, &is_goal, budget),
                dijkstra(&graph, start, &is_goal, budget)
            );
            prop_assert_eq!(
                weighted_a_star_in(&mut workspace, &graph, start, &is_goal, heuristic_fn(h), weight, budget),
                weighted_a_star(&graph, start, &is_goal, heuristic_fn(h), weight, budget)
            );
            prop_assert_eq!(
                gbfs_in(&mut workspace, &graph, start, &is_goal, heuristic_fn(h), budget),
                gbfs(&graph, start, &is_goal, heuristic_fn(h), budget)
            );
            // With the goal among the starts (no expansion), and without it, with a
            // repeated start.
            for starts in [[start, next, goal], [start, next, start]] {
                prop_assert_eq!(
                    a_star_multi_in(&mut workspace, &graph, starts, &is_goal, heuristic_fn(h), budget),
                    a_star_multi(&graph, starts, &is_goal, heuristic_fn(h), budget)
                );
                prop_assert_eq!(
                    dijkstra_multi_in(&mut workspace, &graph, starts, &is_goal, budget),
                    dijkstra_multi(&graph, starts, &is_goal, budget)
                );
                prop_assert_eq!(
                    weighted_a_star_multi_in(&mut workspace, &graph, starts, &is_goal, heuristic_fn(h), weight, budget),
                    weighted_a_star_multi(&graph, starts, &is_goal, heuristic_fn(h), weight, budget)
                );
                prop_assert_eq!(
                    gbfs_multi_in(&mut workspace, &graph, starts, &is_goal, heuristic_fn(h), budget),
                    gbfs_multi(&graph, starts, &is_goal, heuristic_fn(h), budget)
                );
            }
        }
    }

    /// Resumed in calls of at most `per_call` expansions, A*, Dijkstra (A* with `Zero`),
    /// Weighted A* and GBFS each expand the same states as their one-shot function, from
    /// one start or two: the same outcome and the same total counters. Every call stays
    /// within its own budget, and a finished search returns its outcome again without
    /// expanding.
    #[test]
    fn resumable_search_matches_one_shot(
        (graph, goal) in graph_and_goal(),
        start in 0_usize..12,
        second in prop::option::of(0_usize..12),
        scales in prop::collection::vec(0_u64..=4, 12),
        per_call in 1_u64..4,
    ) {
        let nodes = graph.edges.len();
        let starts: Vec<usize> =
            core::iter::once(start).chain(second).map(|s| s % nodes).collect();
        let exact = distances_to(&graph, goal);
        // Admissible, usually inconsistent: exercises reopening across calls.
        let h = heuristic_fn(|state: &usize| Some(exact[*state].map_or(0, |d| d * scales[*state] / 4)));
        let is_goal = |state: &usize| *state == goal;
        let weight = Weight::new(3, 2);
        let s = starts.iter().copied();

        let cases = [
            (
                AStarSearch::new_multi(&graph, s.clone(), &is_goal, h),
                a_star_multi(&graph, s.clone(), &is_goal, h, Limits::NONE),
            ),
            (
                AStarSearch::new_weighted_multi(&graph, s.clone(), &is_goal, h, weight),
                weighted_a_star_multi(&graph, s.clone(), &is_goal, h, weight, Limits::NONE),
            ),
            (
                AStarSearch::new_gbfs_multi(&graph, s.clone(), &is_goal, h),
                gbfs_multi(&graph, s.clone(), &is_goal, h, Limits::NONE),
            ),
        ];
        let dijkstra_case = (
            AStarSearch::new_multi(&graph, s.clone(), &is_goal, Zero),
            dijkstra_multi(&graph, s.clone(), &is_goal, Limits::NONE),
        );
        if let [single] = starts[..] {
            let one = AStarSearch::new_weighted(&graph, single, &is_goal, h, weight);
            prop_assert_eq!(resume(one, &graph, &is_goal, per_call)?, resume(
                AStarSearch::new_weighted_multi(&graph, [single], &is_goal, h, weight),
                &graph, &is_goal, per_call,
            )?);
            let one = AStarSearch::new_gbfs(&graph, single, &is_goal, h);
            prop_assert_eq!(resume(one, &graph, &is_goal, per_call)?, resume(
                AStarSearch::new_gbfs_multi(&graph, [single], &is_goal, h),
                &graph, &is_goal, per_call,
            )?);
        }

        for (search, one_shot) in cases {
            prop_assert_eq!(resume(search, &graph, &is_goal, per_call)?, (one_shot.outcome, one_shot.stats));
        }
        let (search, one_shot) = dijkstra_case;
        prop_assert_eq!(resume(search, &graph, &is_goal, per_call)?, (one_shot.outcome, one_shot.stats));
    }
}

/// The outcome and total counters of `search` run to the end in calls of at most
/// `per_call` expansions; checks every call's budget and that a finished search repeats
/// its outcome without expanding.
fn resume<H: Heuristic<usize, u64>>(
    mut search: AStarSearch<usize, usize, u64, H>,
    graph: &Graph,
    is_goal: &impl Fn(&usize) -> bool,
    per_call: u64,
) -> Result<(Outcome<usize, usize, u64>, Stats), TestCaseError> {
    let finished = loop {
        let result = search.compute(graph, is_goal, Limits::expansions(per_call));
        prop_assert!(result.stats.expanded <= per_call);
        if !matches!(result.outcome, Outcome::LimitReached) {
            break result;
        }
    };

    let again = search.compute(graph, is_goal, Limits::NONE);
    prop_assert_eq!(&again.outcome, &finished.outcome);
    prop_assert_eq!(again.stats, Stats::default());

    Ok((finished.outcome, search.stats()))
}

/// A start that is a goal is found without expanding, on every call.
#[test]
fn resumable_start_is_goal() {
    let graph = crate::fixtures::Graph::new(2, &[(0, 1, 1), (1, 0, 1)]);
    let is_goal = |state: &usize| *state == 1;
    let mut search =
        AStarSearch::new_multi(&graph, [0, 1], &is_goal, heuristic_fn(|_: &usize| Some(0)));

    for _ in 0..2 {
        let result = search.compute(&graph, &is_goal, Limits::expansions(0));
        assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![1]));
        assert_eq!(search.stats(), Stats::default());
    }
}

/// A zero budget makes no progress, and the search continues once the budget allows.
#[test]
fn resumable_zero_budget_pauses() {
    let graph = crate::fixtures::Graph::new(3, &[(0, 1, 1), (1, 2, 1)]);
    let is_goal = |state: &usize| *state == 2;
    let mut search = AStarSearch::new(&graph, 0, &is_goal, heuristic_fn(|_: &usize| Some(0)));

    let paused = search.compute(&graph, &is_goal, Limits::expansions(0));
    assert_eq!(paused.outcome, Outcome::LimitReached);
    assert_eq!(search.stats().expanded, 0);

    let result = search.compute(&graph, &is_goal, Limits::NONE);
    assert_eq!(result.solution().map(|s| s.cost), Some(2));
    assert_eq!(result.stats.expanded, 2);
}
