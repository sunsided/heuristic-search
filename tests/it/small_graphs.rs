//! Hand-made graphs with known answers.

use heuristic_search::{
    Limits, Outcome, Stats, Weight, a_star, a_star_multi, bfs, bfs_multi, dijkstra, dijkstra_multi,
    gbfs, gbfs_multi, heuristic_fn, weighted_a_star, weighted_a_star_multi,
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
fn a_star_with_admissible_heuristic_is_optimal() {
    let graph = diamond();
    let exact = [3_u64, 2, 1, 0];
    let solution = a_star(
        &graph,
        0,
        &is_three,
        heuristic_fn(|s: &usize| Some(exact[*s])),
        Limits::NONE,
    )
    .into_solution()
    .expect("3 is reachable");

    assert_eq!(solution.cost, 3);
}

/// A perfect heuristic leads A* straight along the cheapest path.
#[test]
fn a_star_with_perfect_heuristic_expands_only_the_path() {
    let graph = diamond();
    let exact = [3_u64, 2, 1, 0];
    let result = a_star(
        &graph,
        0,
        &is_three,
        heuristic_fn(|s: &usize| Some(exact[*s])),
        Limits::NONE,
    );

    assert_eq!(result.stats.expanded, 3);
    assert_eq!(result.stats.generated, 4);
}

/// A* must reopen a closed state when an inconsistent but admissible heuristic leads to it
/// through an expensive path first.
#[test]
fn a_star_reopens_with_inconsistent_heuristic() {
    // 0 -> 1 (cost 1) -> 2 (cost 1) and 0 -> 2 (cost 3); 2 -> 3 (cost 3).
    // h(1) = 3 hides the cheap path until 2 has been expanded through the expensive one.
    let graph = Graph::new(4, &[(0, 1, 1), (0, 2, 3), (1, 2, 1), (2, 3, 3)]);
    let h = [0_u64, 3, 0, 0];
    let result = a_star(
        &graph,
        0,
        &is_three,
        heuristic_fn(|s: &usize| Some(h[*s])),
        Limits::NONE,
    );

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
    let mut zero = heuristic_fn(|_: &usize| Some(0_u64));

    assert_eq!(
        bfs(&graph, 0, &goal, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        dijkstra(&graph, 0, &goal, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        a_star(&graph, 0, &goal, &mut zero, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        gbfs(&graph, 0, &goal, &mut zero, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

/// States the heuristic declares dead ends are never expanded.
#[test]
fn dead_end_heuristic_prunes_states() {
    let graph = diamond();
    // Declaring 1 a dead end leaves only the direct edge.
    let mut h = heuristic_fn(|state: &usize| (*state != 1).then_some(0_u64));
    let solution = a_star(&graph, 0, &is_three, &mut h, Limits::NONE)
        .into_solution()
        .expect("the direct edge remains");

    assert_eq!(solution.states, [0, 3]);
}

/// A dead end reached again through a cheaper path must stay closed, not be reopened.
#[test]
fn dead_end_is_not_reopened() {
    // 0 -> 1 costs 5, 0 -> 2 -> 1 costs 2; 1 is declared a dead end.
    let graph = Graph::new(4, &[(0, 1, 5), (0, 2, 1), (2, 1, 1), (1, 3, 1)]);
    let mut h = heuristic_fn(|state: &usize| (*state != 1).then_some(0_u64));
    let result = a_star(&graph, 0, &is_three, &mut h, Limits::NONE);

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
fn weighted_a_star_with_weight_one_matches_a_star() {
    let graph = diamond();
    let mut h = heuristic_fn(|s: &usize| Some([3_u64, 2, 1, 0][*s]));
    let plain = a_star(&graph, 0, &is_three, &mut h, Limits::NONE);
    let weighted = weighted_a_star(&graph, 0, &is_three, &mut h, Weight::ONE, Limits::NONE);

    assert_eq!(plain, weighted);
}

/// The pre-rename names `astar` and `weighted_astar` still call the same searches.
#[test]
fn old_names_are_aliases() {
    let graph = diamond();
    let mut h = heuristic_fn(|s: &usize| Some([3_u64, 2, 1, 0][*s]));
    let new = a_star(&graph, 0, &is_three, &mut h, Limits::NONE);

    assert_eq!(
        heuristic_search::astar(&graph, 0, &is_three, &mut h, Limits::NONE),
        new
    );
    assert_eq!(
        heuristic_search::weighted_astar(&graph, 0, &is_three, &mut h, Weight::ONE, Limits::NONE),
        new
    );
}

/// GBFS trusts a misleading heuristic and returns the expensive path.
#[test]
fn gbfs_follows_the_heuristic() {
    let graph = diamond();
    // A misleading heuristic that makes the expensive direct edge look best.
    let mut h = heuristic_fn(|s: &usize| Some([1_u64, 5, 5, 0][*s]));
    let solution = gbfs(&graph, 0, &is_three, &mut h, Limits::NONE)
        .into_solution()
        .expect("3 is reachable");

    assert_eq!(solution.states, [0, 3]);
    assert_eq!(solution.cost, 10);
}

/// Float costs (`NonNegF64`) work like integer costs.
#[cfg(feature = "float")]
#[test]
fn float_costs_work() {
    use heuristic_search::{NonNegF64, Successors};

    /// Two routes from 0 to 2 with fractional costs.
    struct Floats;

    /// Wraps a literal edge cost; every literal here is non-negative.
    fn cost(value: f64) -> NonNegF64 {
        NonNegF64::new(value).expect("non-negative")
    }

    impl Successors for Floats {
        type State = u8;
        type Action = u8;
        type Cost = NonNegF64;

        fn successors(&self, state: &u8, mut emit: impl FnMut(u8, u8, NonNegF64)) {
            match state {
                0 => {
                    emit(1, 1, cost(0.5));
                    emit(2, 2, cost(1.25));
                }
                1 => emit(2, 2, cost(0.5)),
                _ => {}
            }
        }
    }

    let solution = dijkstra(&Floats, 0, &|s: &u8| *s == 2, Limits::NONE)
        .into_solution()
        .expect("2 is reachable");

    assert_eq!(solution.states, [0, 1, 2]);
    assert!((solution.cost.get() - 1.0).abs() < f64::EPSILON);
}

/// An empty set of start states is unsolvable for every algorithm.
#[test]
fn no_starts_is_unsolvable() {
    let graph = diamond();
    let mut zero = heuristic_fn(|_: &usize| Some(0_u64));
    let none = core::iter::empty::<usize>;

    let results = [
        bfs_multi(&graph, none(), &is_three, Limits::NONE),
        dijkstra_multi(&graph, none(), &is_three, Limits::NONE),
        a_star_multi(&graph, none(), &is_three, &mut zero, Limits::NONE),
        weighted_a_star_multi(
            &graph,
            none(),
            &is_three,
            &mut zero,
            Weight::ONE,
            Limits::NONE,
        ),
        gbfs_multi(&graph, none(), &is_three, &mut zero, Limits::NONE),
    ];

    for result in results {
        assert_eq!(result.outcome, Outcome::Unsolvable);
        assert_eq!(result.stats, Stats::default());
    }
}

/// A start that is a goal wins at cost zero, even when the heuristic prefers another start.
#[test]
fn goal_among_starts_costs_nothing() {
    let graph = diamond();
    // Misleading: node 2 looks closer than the goal itself.
    let mut h = heuristic_fn(|s: &usize| Some([9_u64, 9, 0, 5][*s]));

    let results = [
        bfs_multi(&graph, [2, 3], &is_three, Limits::NONE),
        dijkstra_multi(&graph, [2, 3], &is_three, Limits::NONE),
        a_star_multi(&graph, [2, 3], &is_three, &mut h, Limits::NONE),
        weighted_a_star_multi(
            &graph,
            [2, 3],
            &is_three,
            &mut h,
            Weight::integer(2),
            Limits::NONE,
        ),
        gbfs_multi(&graph, [2, 3], &is_three, &mut h, Limits::NONE),
    ];

    for result in results {
        assert_eq!(result.stats.expanded, 0);
        let solution = result.into_solution().expect("a start is the goal");
        assert_eq!(solution.states, [3]);
        assert_eq!(solution.cost, 0);
    }
}

/// Repeated starts are searched once and do not change the result.
#[test]
fn repeated_starts_are_deduplicated() {
    let graph = diamond();
    let once = dijkstra_multi(&graph, [0], &is_three, Limits::NONE);
    let repeated = dijkstra_multi(&graph, [0, 0, 0], &is_three, Limits::NONE);

    assert_eq!(once, repeated);
}

/// Several starts find the cheapest goal from any of them.
#[test]
fn multi_start_picks_the_cheapest_start() {
    let graph = diamond();
    let solution = dijkstra_multi(&graph, [0, 1], &is_three, Limits::NONE)
        .into_solution()
        .expect("3 is reachable");

    assert_eq!(solution.states, [1, 2, 3]);
    assert_eq!(solution.cost, 2);
    assert_valid_path(&graph, 1, &solution);
}

/// A struct heuristic with a reusable workspace, lent to searches as `&mut`.
struct Counting {
    /// Exact distances to node 3, by node.
    exact: [u64; 4],
    /// Every state estimated so far, in order; stands in for a reusable workspace.
    seen: Vec<usize>,
}

impl heuristic_search::Heuristic<usize, u64> for Counting {
    fn estimate(&mut self, state: &usize) -> Option<u64> {
        self.seen.push(*state);
        Some(self.exact[*state])
    }
}

/// A heuristic lent as `&mut` keeps its workspace across searches, and each state is
/// estimated at most once per search.
#[test]
fn borrowed_heuristic_keeps_its_workspace() {
    let graph = diamond();
    let mut h = Counting {
        exact: [3, 2, 1, 0],
        seen: Vec::new(),
    };

    let first = a_star(&graph, 0, &is_three, &mut h, Limits::NONE);
    let estimated_once = h.seen.len();
    let second = gbfs(&graph, 0, &is_three, &mut h, Limits::NONE);

    assert_eq!(first.solution().map(|s| s.cost), Some(3));
    assert!(second.solution().is_some());
    // A* reaches 3 again through 2 without estimating it twice; GBFS stops at 3 early.
    assert_eq!(h.seen[..estimated_once], [0, 3, 1, 2]);
    assert_eq!(h.seen[estimated_once..], [0, 3, 1]);
}

/// A boxed trait object works as a heuristic, for callers that choose one at run time.
#[test]
fn boxed_dyn_heuristic_works() {
    let graph = diamond();
    let h: Box<dyn heuristic_search::Heuristic<usize, u64>> = Box::new(Counting {
        exact: [3, 2, 1, 0],
        seen: Vec::new(),
    });
    let solution = a_star(&graph, 0, &is_three, h, Limits::NONE)
        .into_solution()
        .expect("3 is reachable");

    assert_eq!(solution.cost, 3);
}
