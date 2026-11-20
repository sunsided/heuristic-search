//! Multi-heuristic A*: bounded on random graphs with misleading inadmissible heuristics,
//! optimal with both weights `1`.

use heuristic_search::{Heuristic, Limits, MhaWeights, Outcome, Weight, heuristic_fn, mha_star};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_to, graph_and_goal};

/// A boxed heuristic over the test graph, so heuristics of different closure types share
/// one list.
type Boxed<'a> = Box<dyn Heuristic<usize, u64> + 'a>;

/// Unreachable goals are unsolvable; a zero budget stops the search.
#[test]
fn unsolvable_and_budget() {
    let graph = Graph::new(3, &[(0, 1, 1)]);
    let zero = || heuristic_fn(|_: &usize| Some(0_u64));
    let weights = MhaWeights {
        heuristic: Weight::integer(2),
        anchor: Weight::integer(2),
    };

    let never = mha_star(
        &graph,
        0,
        &|s: &usize| *s == 2,
        zero(),
        [zero()],
        weights,
        Limits::NONE,
    );
    let limited = mha_star(
        &graph,
        0,
        &|s: &usize| *s == 1,
        zero(),
        [zero()],
        weights,
        Limits::expansions(0),
    );
    let start = mha_star(
        &graph,
        0,
        &|s: &usize| *s == 0,
        zero(),
        [zero()],
        weights,
        Limits::NONE,
    );

    assert_eq!(never.outcome, Outcome::Unsolvable);
    assert_eq!(limited.outcome, Outcome::LimitReached);
    assert_eq!(start.solution().map(|s| s.cost), Some(0));
}

proptest! {
    /// With a consistent anchor (scaled exact distance) and two random inadmissible
    /// heuristics, the solution is within `w1 * w2` of the optimum; with both weights `1`
    /// it is optimal.
    #[test]
    fn mha_star_is_bounded(
        (graph, goal) in graph_and_goal(),
        scale in 0_u64..=4,
        noise in prop::collection::vec(0_u64..40, 24),
        w1 in 2_u32..6,
        w2 in 2_u32..6,
    ) {
        let exact = distances_to(&graph, goal);
        let anchor = |s: &usize| Some(exact[*s].map_or(0, |d| d * scale / 4));
        let guides = || -> Vec<Boxed<'_>> {
            vec![
                Box::new(heuristic_fn(|s: &usize| Some(noise[*s]))),
                Box::new(heuristic_fn(|s: &usize| (noise[*s + 12] % 5 != 0).then_some(noise[*s + 12]))),
            ]
        };
        let is_goal = |s: &usize| *s == goal;
        let weights = MhaWeights { heuristic: Weight::new(w1, 2), anchor: Weight::new(w2, 2) };

        let bounded = mha_star(&graph, 0, &is_goal, heuristic_fn(anchor), guides(), weights, Limits::NONE);
        let optimal = mha_star(&graph, 0, &is_goal, heuristic_fn(anchor), guides(), MhaWeights::ONE, Limits::NONE);

        prop_assert_eq!(optimal.solution().map(|s| s.cost), exact[0]);
        prop_assert_eq!(bounded.solution().is_some(), exact[0].is_some());
        if let (Some(solution), Some(optimum)) = (bounded.solution(), exact[0]) {
            assert_valid_path(&graph, 0, solution);
            prop_assert!(4 * solution.cost <= u64::from(w1 * w2) * optimum);
        }
    }
}
