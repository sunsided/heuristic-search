//! Hand-made graphs with known answers.

use heuristic_search::{
    Limits, Outcome, Stats, Weight, astar, bfs, dijkstra, gbfs, weighted_astar,
};

use crate::fixtures::{Graph, assert_valid_path};

/// A diamond where the path with fewer edges is the more expensive one:
/// `0 -> 3` costs 10 directly, `0 -> 1 -> 2 -> 3` costs 3.
fn diamond() -> Graph {
    Graph::new(4, &[(0, 3, 10), (0, 1, 1), (1, 2, 1), (2, 3, 1)])
}

/// Goal test for node 3.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the `Goal` closure signature takes a reference"
)]
fn is_three(state: &usize) -> bool {
    *state == 3
}

/// Dijkstra prefers three cheap edges over one expensive edge.
#[test]
fn dijkstra_finds_cheapest_path() {
    let graph = diamond();
    let solution = dijkstra(&graph, 0, &is_three, Limits::NONE)
        .into_solution()
        .expect("3 is reachable");

    assert_eq!(solution.states, [0, 1, 2, 3]);
    assert_eq!(solution.cost, 3);
    assert_valid_path(&graph, 0, &solution);
}

/// BFS takes the single expensive edge and reports its true cost.
#[test]
fn bfs_finds_fewest_edges_and_sums_costs() {
    let graph = diamond();
    let solution = bfs(&graph, 0, &is_three, Limits::NONE)
        .into_solution()
        .expect("3 is reachable");

    assert_eq!(solution.states, [0, 3]);
    assert_eq!(solution.cost, 10);
}

/// A* with the exact distance as heuristic finds the cheapest path.
#[test]
fn astar_with_admissible_heuristic_is_optimal() {
    let graph = diamond();
    let exact = [3_u64, 2, 1, 0];
    let solution = astar(
        &graph,
        0,
        &is_three,
        &|s: &usize| Some(exact[*s]),
        Limits::NONE,
    )
    .into_solution()
    .expect("3 is reachable");

    assert_eq!(solution.cost, 3);
}

/// A perfect heuristic leads A* straight along the cheapest path.
#[test]
fn astar_with_perfect_heuristic_expands_only_the_path() {
    let graph = diamond();
    let exact = [3_u64, 2, 1, 0];
    let result = astar(
        &graph,
        0,
        &is_three,
        &|s: &usize| Some(exact[*s]),
        Limits::NONE,
    );

    assert_eq!(result.stats.expanded, 3);
    assert_eq!(result.stats.generated, 4);
}

/// A* must reopen a closed state when an inconsistent but admissible heuristic leads to it
/// through an expensive path first.
#[test]
fn astar_reopens_with_inconsistent_heuristic() {
    // 0 -> 1 (cost 1) -> 2 (cost 1) and 0 -> 2 (cost 3); 2 -> 3 (cost 3).
    // h(1) = 3 hides the cheap path until 2 has been expanded through the expensive one.
    let graph = Graph::new(4, &[(0, 1, 1), (0, 2, 3), (1, 2, 1), (2, 3, 3)]);
    let h = [0_u64, 3, 0, 0];
    let result = astar(&graph, 0, &is_three, &|s: &usize| Some(h[*s]), Limits::NONE);

    assert_eq!(result.stats.reopened, 1);
    let solution = result.into_solution().expect("3 is reachable");
    assert_eq!(solution.states, [0, 1, 2, 3]);
    assert_eq!(solution.cost, 5);
    assert_valid_path(&graph, 0, &solution);
}

/// A start state that is a goal yields a path without edges.
#[test]
fn start_goal_yields_empty_path() {
    let graph = diamond();
    for result in [
        bfs(&graph, 3, &is_three, Limits::NONE),
        dijkstra(&graph, 3, &is_three, Limits::NONE),
    ] {
        let solution = result.into_solution().expect("start is the goal");
        assert!(solution.is_empty());
        assert_eq!(*solution.goal(), 3);
        assert_eq!(solution.cost, 0);
    }
}

/// Every algorithm reports an unreachable goal as unsolvable.
#[test]
fn unreachable_goal_is_unsolvable() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 0, 1)]);
    let goal = |state: &usize| *state == 2;
    let zero = |_: &usize| Some(0_u64);

    assert_eq!(
        bfs(&graph, 0, &goal, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        dijkstra(&graph, 0, &goal, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        astar(&graph, 0, &goal, &zero, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        gbfs(&graph, 0, &goal, &zero, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

/// States the heuristic declares dead ends are never expanded.
#[test]
fn dead_end_heuristic_prunes_states() {
    let graph = diamond();
    // Declaring 1 a dead end leaves only the direct edge.
    let h = |state: &usize| (*state != 1).then_some(0_u64);
    let solution = astar(&graph, 0, &is_three, &h, Limits::NONE)
        .into_solution()
        .expect("the direct edge remains");

    assert_eq!(solution.states, [0, 3]);
}

/// A dead end reached again through a cheaper path must stay closed, not be reopened.
#[test]
fn dead_end_is_not_reopened() {
    // 0 -> 1 costs 5, 0 -> 2 -> 1 costs 2; 1 is declared a dead end.
    let graph = Graph::new(4, &[(0, 1, 5), (0, 2, 1), (2, 1, 1), (1, 3, 1)]);
    let h = |state: &usize| (*state != 1).then_some(0_u64);
    let result = astar(&graph, 0, &is_three, &h, Limits::NONE);

    assert_eq!(result.outcome, Outcome::Unsolvable);
    assert_eq!(result.stats.expanded, 2);
    assert_eq!(result.stats.reopened, 0);
}

/// An expansion limit of `n` stops the search after exactly `n` expansions.
#[test]
fn expansion_limit_stops_search() {
    let graph = diamond();
    let result = dijkstra(&graph, 0, &is_three, Limits::expansions(2));

    assert_eq!(result.outcome, Outcome::LimitReached);
    assert_eq!(result.stats.expanded, 2);
}

/// A closure budget is asked before every expansion with the current counters.
#[test]
fn closure_budget_sees_stats() {
    let graph = diamond();
    let mut seen = Vec::new();
    let budget = |stats: &Stats| {
        seen.push(stats.expanded);
        false
    };
    let result = bfs(&graph, 0, &is_three, budget);

    assert!(result.solution().is_some());
    assert_eq!(seen, [0]);
}

/// Weighted A* with weight 1 behaves exactly like A*.
#[test]
fn weighted_astar_with_weight_one_matches_astar() {
    let graph = diamond();
    let h = |s: &usize| Some([3_u64, 2, 1, 0][*s]);
    let plain = astar(&graph, 0, &is_three, &h, Limits::NONE);
    let weighted = weighted_astar(&graph, 0, &is_three, &h, Weight::ONE, Limits::NONE);

    assert_eq!(plain, weighted);
}

/// GBFS trusts a misleading heuristic and returns the expensive path.
#[test]
fn gbfs_follows_the_heuristic() {
    let graph = diamond();
    // A misleading heuristic that makes the expensive direct edge look best.
    let h = |s: &usize| Some([1_u64, 5, 5, 0][*s]);
    let solution = gbfs(&graph, 0, &is_three, &h, Limits::NONE)
        .into_solution()
        .expect("3 is reachable");

    assert_eq!(solution.states, [0, 3]);
    assert_eq!(solution.cost, 10);
}

/// `f64` costs work like integer costs.
#[test]
fn float_costs_work() {
    use heuristic_search::Successors;

    /// Two routes from 0 to 2 with fractional costs.
    struct Floats;

    impl Successors for Floats {
        type State = u8;
        type Action = u8;
        type Cost = f64;

        fn successors(&self, state: &u8, mut emit: impl FnMut(u8, u8, f64)) {
            match state {
                0 => {
                    emit(1, 1, 0.5);
                    emit(2, 2, 1.25);
                }
                1 => emit(2, 2, 0.5),
                _ => {}
            }
        }
    }

    let solution = dijkstra(&Floats, 0, &|s: &u8| *s == 2, Limits::NONE)
        .into_solution()
        .expect("2 is reachable");

    assert_eq!(solution.states, [0, 1, 2]);
    assert!((solution.cost - 1.0).abs() < f64::EPSILON);
}
