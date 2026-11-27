//! Contraction hierarchies: every query cross-checked against Dijkstra on random graphs
//! and on the benchmark grid, solutions replayed on the original graph, and the ranks and
//! search spaces checked against the queries.

use heuristic_search::{
    ContractionHierarchy, Limits, Outcome, Successors, WitnessLimits, dijkstra,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, Grid, assert_valid_path};

/// A random directed graph with up to 40 nodes and edge costs `0..=9`: zero-cost edges,
/// self loops, parallel edges and unreachable pairs all occur.
fn random_graph() -> impl Strategy<Value = Graph> {
    (2_usize..40).prop_flat_map(|nodes| {
        prop::collection::vec((0..nodes, 0..nodes, 0_u64..10), 0..nodes * 3)
            .prop_map(move |edges| Graph::new(nodes, &edges))
    })
}

/// Witness limits from none at all to the defaults, so the limits' branches run.
fn limits() -> impl Strategy<Value = WitnessLimits> {
    prop_oneof![
        Just(WitnessLimits {
            max_hops: 0,
            max_settled: 0,
        }),
        Just(WitnessLimits {
            max_hops: 1,
            max_settled: 3,
        }),
        Just(WitnessLimits::DEFAULT),
    ]
}

/// The Dijkstra cost from `start` to `goal`, or `None` if unreachable.
fn dijkstra_cost<G>(graph: &G, start: G::State, goal: &G::State) -> Option<G::Cost>
where
    G: Successors,
    G::State: Clone + core::hash::Hash + Eq,
    G::Action: Clone,
{
    dijkstra(
        graph,
        start,
        &|state: &G::State| state == goal,
        Limits::NONE,
    )
    .into_solution()
    .map(|solution| solution.cost)
}

proptest! {
    /// Every query costs what Dijkstra costs, and every path is a real path of the graph
    /// from the start to the goal at the reported cost.
    #[test]
    fn queries_match_dijkstra(graph in random_graph(), limits in limits()) {
        let nodes = graph.edges.len();
        let hierarchy = ContractionHierarchy::build_with_limits(&graph, 0..nodes, limits);
        prop_assert_eq!(hierarchy.len(), nodes);

        for start in 0..nodes {
            for goal in 0..nodes {
                let result = hierarchy.query(start, goal);
                let expected = dijkstra_cost(&graph, start, &goal);
                prop_assert_eq!(result.solution().map(|s| s.cost), expected);

                match result.outcome {
                    Outcome::Found(solution) => {
                        assert_valid_path(&graph, start, &solution);
                        prop_assert_eq!(*solution.goal(), goal);
                    }
                    Outcome::Unsolvable => prop_assert!(expected.is_none()),
                    outcome => prop_assert!(false, "unexpected outcome {outcome:?}"),
                }
            }
        }
    }

    /// The ranks are a permutation, and the cheapest meeting of the forward search space
    /// of the start and the backward one of the goal is the query cost.
    #[test]
    fn search_spaces_meet_at_query_cost(graph in random_graph()) {
        let nodes = graph.edges.len();
        let hierarchy = ContractionHierarchy::build(&graph, 0..nodes);

        let mut ranks: Vec<_> = (0..nodes).map(|n| hierarchy.rank(&n)).collect();
        ranks.sort_unstable();
        prop_assert_eq!(ranks, (0..nodes).map(Some).collect::<Vec<_>>());

        for start in 0..nodes {
            let forward = hierarchy.forward_search_space(&start);
            prop_assert_eq!(forward[0], (&start, 0));
            for goal in 0..nodes {
                let backward = hierarchy.backward_search_space(&goal);
                let meeting = forward
                    .iter()
                    .filter_map(|&(state, f)| {
                        backward.iter().find(|&&(other, _)| other == state).map(|&(_, b)| f + b)
                    })
                    .min();
                let query = hierarchy.query(start, goal).into_solution().map(|s| s.cost);
                prop_assert_eq!(meeting, query);
            }
        }
    }

    /// A query from several starts to several goals costs the minimum over the pairs.
    #[test]
    fn multi_query_is_cheapest_pair(
        graph in random_graph(),
        starts in prop::collection::vec(0_usize..40, 0..4),
        goals in prop::collection::vec(0_usize..40, 0..4),
    ) {
        let nodes = graph.edges.len();
        let hierarchy = ContractionHierarchy::build(&graph, 0..nodes);
        let starts: Vec<_> = starts.into_iter().filter(|&s| s < nodes).collect();
        let goals: Vec<_> = goals.into_iter().filter(|&g| g < nodes).collect();

        let graph_ref = &graph;
        let expected = starts
            .iter()
            .flat_map(|&s| goals.iter().filter_map(move |g| dijkstra_cost(graph_ref, s, g)))
            .min();
        let result = hierarchy.query_multi(starts.clone(), goals.clone());
        prop_assert_eq!(result.solution().map(|s| s.cost), expected);

        if let Some(solution) = result.solution() {
            prop_assert!(starts.contains(&solution.states[0]));
            prop_assert!(goals.contains(solution.goal()));
            assert_valid_path(&graph, solution.states[0], solution);
        }
    }
}

/// Of parallel edges the cheapest is used, with its own action; self loops are ignored.
#[test]
fn parallel_edges_and_self_loops() {
    let graph = Graph::new(3, &[(0, 0, 0), (0, 1, 5), (0, 1, 2), (1, 1, 1), (1, 2, 1)]);
    let hierarchy = ContractionHierarchy::build(&graph, [0]);

    let solution = hierarchy.query(0, 2).into_solution().expect("reachable");
    assert_eq!(solution.states, [0, 1, 2]);
    assert_eq!(solution.actions, [2, 1]);
    assert_eq!(solution.cost, 3);
}

/// Only states reachable from the seeds are in the hierarchy; other states are
/// unsolvable unless the start is the goal.
#[test]
fn states_outside_the_hierarchy() {
    let graph = Graph::new(3, &[(0, 1, 1), (2, 0, 1)]);
    let hierarchy = ContractionHierarchy::build(&graph, [0]);

    assert_eq!(hierarchy.len(), 2);
    assert_eq!(hierarchy.rank(&2), None);
    assert_eq!(hierarchy.forward_search_space(&2), []);
    assert_eq!(hierarchy.query(2, 1).outcome, Outcome::Unsolvable);
    assert_eq!(hierarchy.query(7, 7).solution().map(|s| s.cost), Some(0));
    assert_eq!(hierarchy.query(1, 0).outcome, Outcome::Unsolvable);

    let empty = ContractionHierarchy::<usize, usize, u64>::build(&graph, []);
    assert!(empty.is_empty());
}

/// Side length of the benchmark grid.
const SIZE: usize = 64;

/// The benchmark grid of `benches/search.rs`: `SIZE` square, a wall across the middle row
/// with a gap of two cells at the right end.
fn bench_grid() -> Grid {
    let mut walls = vec![vec![false; SIZE]; SIZE];
    for cell in walls[SIZE / 2].iter_mut().take(SIZE - 2) {
        *cell = true;
    }

    Grid { walls }
}

/// On the benchmark grid, sampled queries match Dijkstra and replay move by move, and
/// preprocessing adds shortcuts.
#[test]
fn bench_grid_matches_dijkstra() {
    let grid = bench_grid();
    let hierarchy = ContractionHierarchy::build(&grid, [(0, 0)]);
    assert_eq!(hierarchy.len(), SIZE * SIZE - (SIZE - 2));
    assert!(hierarchy.shortcut_count() > 0);

    let cells = [
        (0, 0),
        (SIZE - 1, 0),
        (0, SIZE - 1),
        (SIZE / 2, SIZE - 1),
        (17, 40),
        (50, 3),
    ];
    for &start in &cells {
        for &goal in &cells {
            let solution = hierarchy
                .query(start, goal)
                .into_solution()
                .expect("connected");
            assert_eq!(Some(solution.cost), dijkstra_cost(&grid, start, &goal));
            assert_eq!(solution.states[0], start);
            assert_eq!(*solution.goal(), goal);

            for (step, action) in solution.actions.iter().enumerate() {
                let mut found = false;
                grid.successors(&solution.states[step], |a, next, _| {
                    found |= a == *action && next == solution.states[step + 1];
                });
                assert!(found, "step {step} of {start:?} -> {goal:?} is no move");
            }
            assert_eq!(solution.cost as usize, solution.len());
        }
    }

    // A wall cell is not in the hierarchy.
    assert_eq!(
        hierarchy.query((0, 0), (SIZE / 2, 0)).outcome,
        Outcome::Unsolvable
    );
}
