//! Bounded-suboptimal searches: focal search, EES and DPS on hand-made graphs and within
//! their bound on random graphs.

use heuristic_search::{
    EesEstimates, FocalKey, Limits, Outcome, Weight, dps, ees, focal_fn, focal_search, heuristic_fn,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// Two routes to node 3: `0 -> 1 -> 3` costs 4, `0 -> 2 -> 3` costs 5 and is preferred
/// by the focal key [`prefer_two`].
fn two_routes() -> Graph {
    Graph::new(4, &[(0, 1, 2), (1, 3, 2), (0, 2, 2), (2, 3, 3)])
}

/// Goal test for node 3.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the `Goal` closure signature takes a reference"
)]
fn is_three(state: &usize) -> bool {
    *state == 3
}

/// Focal key for [`two_routes`]: prefers nodes 2 and 3.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the focal closure signature takes a reference"
)]
fn prefer_two(state: &usize) -> u8 {
    u8::from(*state < 2)
}

/// Within the bound, focal search takes the preferred, more expensive route; with weight 1
/// it must take the cheapest.
#[test]
fn focal_search_uses_the_slack() {
    let graph = two_routes();
    let exact = || heuristic_fn(|state: &usize| Some([4, 2, 3, 0][*state]));

    let loose = focal_search(
        &graph,
        0,
        &is_three,
        exact(),
        focal_fn(prefer_two),
        Weight::integer(2),
        Limits::NONE,
    );
    let tight = focal_search(
        &graph,
        0,
        &is_three,
        exact(),
        focal_fn(prefer_two),
        Weight::ONE,
        Limits::NONE,
    );

    assert_eq!(
        loose.solution().map(|s| s.states.clone()),
        Some(vec![0, 2, 3])
    );
    assert_eq!(tight.solution().map(|s| s.cost), Some(4));
}

/// A path-dependent focal key: the number of edges so far, built with `extend`.
struct Depth;

impl FocalKey<usize, usize, u64> for Depth {
    type Key = u32;

    fn start(&mut self, _: &usize) -> u32 {
        0
    }

    fn extend(&mut self, parent_key: &u32, _: &usize, _: &usize, _: u64, _: &usize) -> u32 {
        parent_key + 1
    }
}

/// With a large weight and the depth as focal key, focal search finds the fewest edges.
#[test]
fn path_dependent_focal_key() {
    let graph = Graph::new(4, &[(0, 3, 10), (0, 1, 1), (1, 2, 1), (2, 3, 1)]);
    let zero = heuristic_fn(|_: &usize| Some(0_u64));

    let result = focal_search(
        &graph,
        0,
        &is_three,
        zero,
        Depth,
        Weight::integer(10),
        Limits::NONE,
    );

    assert_eq!(
        result.solution().map(heuristic_search::Solution::len),
        Some(1)
    );
}

/// Budgets and unreachable goals behave as in the other searches.
#[test]
fn budget_and_unsolvable() {
    let graph = two_routes();
    let zero = || heuristic_fn(|_: &usize| Some(0_u64));
    let never = |state: &usize| *state == 9;
    let estimates = || EesEstimates {
        admissible: zero(),
        cost_to_go: zero(),
        distance_to_go: zero(),
    };

    let w = Weight::integer(2);
    let limited = focal_search(
        &graph,
        0,
        &is_three,
        zero(),
        focal_fn(prefer_two),
        w,
        Limits::expansions(0),
    );
    assert_eq!(limited.outcome, Outcome::LimitReached);

    assert_eq!(
        focal_search(
            &graph,
            0,
            &never,
            zero(),
            focal_fn(prefer_two),
            w,
            Limits::NONE
        )
        .outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        ees(&graph, 0, &never, estimates(), w, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        dps(&graph, 0, &never, zero(), w, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        ees(&graph, 0, &is_three, estimates(), w, Limits::expansions(1)).outcome,
        Outcome::LimitReached
    );
}

/// Asserts that `found` is a valid path within `numerator / 2` times `optimum`, and that
/// it exists exactly when the optimum does.
fn assert_bounded(
    graph: &Graph,
    found: Option<&heuristic_search::Solution<usize, usize, u64>>,
    optimum: Option<u64>,
    numerator: u32,
) {
    assert_eq!(found.is_some(), optimum.is_some());
    if let (Some(solution), Some(optimum)) = (found, optimum) {
        assert_valid_path(graph, 0, solution);
        assert!(
            2 * solution.cost <= u64::from(numerator) * optimum,
            "{} > {numerator}/2 * {optimum}",
            solution.cost
        );
    }
}

proptest! {
    /// Focal search, EES and DPS stay within `w` times the optimum with an admissible but
    /// inconsistent heuristic and arbitrary secondary estimates.
    #[test]
    fn bounded_searches_stay_within_bound(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        hat_scales in prop::collection::vec(0_u64..=8, 12),
        keys in prop::collection::vec(0_u8..4, 12),
        numerator in 2_u32..7,
    ) {
        let weight = Weight::new(numerator, 2);
        let exact = distances_to(&graph, goal);
        // `None` where the goal is unreachable: a true dead end.
        let h = |state: &usize| exact[*state].map(|d| d * scales[*state] / 4);
        let h_hat = |state: &usize| Some(exact[*state].map_or(0, |d| d * hat_scales[*state] / 4));
        let is_goal = |state: &usize| *state == goal;

        let focal = focal_search(&graph, 0, &is_goal, heuristic_fn(h), focal_fn(|s: &usize| keys[*s]), weight, Limits::NONE);
        let estimates = EesEstimates {
            admissible: heuristic_fn(h),
            cost_to_go: heuristic_fn(h_hat),
            distance_to_go: heuristic_fn(|s: &usize| Some(u32::from(keys[*s]))),
        };
        let explicit = ees(&graph, 0, &is_goal, estimates, weight, Limits::NONE);
        let potential = dps(&graph, 0, &is_goal, heuristic_fn(h), weight, Limits::NONE);

        assert_bounded(&graph, focal.solution(), exact[0], numerator);
        assert_bounded(&graph, explicit.solution(), exact[0], numerator);
        assert_bounded(&graph, potential.solution(), exact[0], numerator);
    }

    /// With weight 1 all three are optimal.
    #[test]
    fn weight_one_is_optimal(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        keys in prop::collection::vec(0_u8..4, 12),
    ) {
        let exact = distances_to(&graph, goal);
        let h = |state: &usize| Some(exact[*state].map_or(0, |d| d * scales[*state] / 4));
        let is_goal = |state: &usize| *state == goal;
        let estimates = EesEstimates {
            admissible: heuristic_fn(h),
            cost_to_go: heuristic_fn(h),
            distance_to_go: heuristic_fn(|s: &usize| Some(u32::from(keys[*s]))),
        };

        let focal = focal_search(&graph, 0, &is_goal, heuristic_fn(h), focal_fn(|s: &usize| keys[*s]), Weight::ONE, Limits::NONE);
        let explicit = ees(&graph, 0, &is_goal, estimates, Weight::ONE, Limits::NONE);
        let potential = dps(&graph, 0, &is_goal, heuristic_fn(h), Weight::ONE, Limits::NONE);

        prop_assert_eq!(focal.solution().map(|s| s.cost), exact[0]);
        prop_assert_eq!(explicit.solution().map(|s| s.cost), exact[0]);
        prop_assert_eq!(potential.solution().map(|s| s.cost), exact[0]);
    }
}
