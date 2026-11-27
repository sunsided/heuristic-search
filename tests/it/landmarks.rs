//! ALT landmark heuristics: admissibility and consistency on random directed graphs, A*
//! and the bidirectional searches with them against Dijkstra, and disconnected graphs.

use heuristic_search::{
    Heuristic, LandmarkSelection, Landmarks, Limits, Outcome, Weight, a_star, ara_star, bae_star,
    dibbs, dijkstra, nbs,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_from, distances_to, graph_and_goal};

/// Both landmark selections.
fn selection() -> impl Strategy<Value = LandmarkSelection> {
    prop_oneof![
        Just(LandmarkSelection::Farthest),
        Just(LandmarkSelection::Avoidance)
    ]
}

/// An undirected `side` x `side` grid of unit edges as a [`Graph`], row-major.
fn grid(side: usize) -> Graph {
    let mut edges = Vec::new();
    for row in 0..side {
        for column in 0..side {
            let node = row * side + column;
            if column + 1 < side {
                edges.extend([(node, node + 1, 1), (node + 1, node, 1)]);
            }
            if row + 1 < side {
                edges.extend([(node, node + side, 1), (node + side, node, 1)]);
            }
        }
    }

    Graph::new(side * side, &edges)
}

proptest! {
    /// Every bound is at most the true distance, `None` only where there is no path, and
    /// the heuristics agree with `lower_bound` in both directions, also with any number of
    /// active landmarks chosen for that pair. Landmarks are distinct.
    #[test]
    fn bounds_are_admissible(
        (graph, _) in graph_and_goal(),
        count in 0_usize..5,
        selection in selection(),
    ) {
        let landmarks = Landmarks::build(&graph, 0, count, selection);
        let picked: Vec<_> = landmarks.landmarks().copied().collect();
        let mut distinct = picked.clone();
        distinct.sort_unstable();
        distinct.dedup();
        prop_assert_eq!(distinct.len(), picked.len());
        prop_assert!(picked.len() <= count);

        let nodes = graph.edges.len();
        for target in 0..nodes {
            let exact = distances_to(&graph, target);
            let mut to = landmarks.heuristic_to(&target);
            for from in 0..nodes {
                let bound = landmarks.lower_bound(&from, &target);
                match (bound, exact[from]) {
                    (Some(bound), Some(exact)) => prop_assert!(bound <= exact),
                    (None, Some(_)) => prop_assert!(false, "{from} reaches {target}"),
                    (_, None) => {}
                }
                prop_assert_eq!(to.estimate(&from), bound);
                prop_assert_eq!(landmarks.heuristic_from(&from).estimate(&target), bound);
                for k in 1..=picked.len() {
                    let mut to = landmarks.heuristic_to(&target).active(&from, k);
                    let mut back = landmarks.heuristic_from(&from).active(&target, k);
                    prop_assert_eq!(to.estimate(&from), bound);
                    prop_assert_eq!(back.estimate(&target), bound);
                }
            }
        }
    }

    /// `h(u) <= c + h(v)` along every edge towards every target, and `h(v) <= h(u) + c`
    /// from every source, with all landmarks and with an active subset.
    #[test]
    fn heuristics_are_consistent(
        (graph, _) in graph_and_goal(),
        count in 0_usize..5,
        active in 0_usize..3,
        selection in selection(),
    ) {
        let landmarks = Landmarks::build(&graph, 0, count, selection);
        let nodes = graph.edges.len();
        for endpoint in 0..nodes {
            let other = (endpoint + 1) % nodes;
            let mut to = landmarks.heuristic_to(&endpoint);
            let mut to_active = landmarks.heuristic_to(&endpoint).active(&other, active);
            let mut from = landmarks.heuristic_from(&endpoint);
            let mut from_active = landmarks.heuristic_from(&endpoint).active(&other, active);
            for (u, edges) in graph.edges.iter().enumerate() {
                for &(v, cost) in edges {
                    for h in [&mut to, &mut to_active] {
                        if let (Some(hu), Some(hv)) = (h.estimate(&u), h.estimate(&v)) {
                            prop_assert!(hu <= cost + hv);
                        }
                    }
                    for h in [&mut from, &mut from_active] {
                        if let (Some(hu), Some(hv)) = (h.estimate(&u), h.estimate(&v)) {
                            prop_assert!(hv <= hu + cost);
                        }
                    }
                }
            }
        }
    }

    /// A* with ALT finds Dijkstra's cost without reopening, expands each state at most
    /// once and only states no farther than the optimum, and no more than Dijkstra when
    /// there is no path.
    #[test]
    fn a_star_with_alt_matches_dijkstra(
        (graph, goal) in graph_and_goal(),
        count in 0_usize..5,
        active in 1_usize..4,
        selection in selection(),
    ) {
        let landmarks = Landmarks::build(&graph, goal, count, selection);
        let is_goal = |state: &usize| *state == goal;
        let optimal = dijkstra(&graph, 0, &is_goal, Limits::NONE);
        let from_start = distances_from(&graph, 0);

        for h in [
            landmarks.heuristic_to(&goal),
            landmarks.heuristic_to(&goal).active(&0, active),
        ] {
            let alt = a_star(&graph, 0, &is_goal, h, Limits::NONE);
            prop_assert_eq!(alt.solution().map(|s| s.cost), optimal.solution().map(|s| s.cost));
            prop_assert_eq!(alt.stats.reopened, 0);
            if let Some(solution) = alt.solution() {
                assert_valid_path(&graph, 0, solution);
                let within = from_start
                    .iter()
                    .filter(|d| d.is_some_and(|d| d <= solution.cost))
                    .count() as u64;
                prop_assert!(alt.stats.expanded <= within);
            } else {
                prop_assert!(alt.stats.expanded <= optimal.stats.expanded);
            }
        }
    }

    /// NBS, DIBBS and BAE* with ALT towards the goal and from the start find Dijkstra's cost.
    #[test]
    fn bidirectional_with_alt_matches_dijkstra(
        (graph, goal) in graph_and_goal(),
        count in 0_usize..5,
        selection in selection(),
    ) {
        let landmarks = Landmarks::build(&graph, 0, count, selection);
        let optimum = distances_to(&graph, goal)[0];
        let forward = || landmarks.heuristic_to(&goal);
        let backward = || landmarks.heuristic_from(&0);

        let results = [
            nbs(&graph, 0, [goal], forward(), backward(), Limits::NONE),
            dibbs(&graph, 0, [goal], forward(), backward(), Limits::NONE),
            bae_star(&graph, 0, [goal], forward(), backward(), Limits::NONE),
        ];
        for result in &results {
            prop_assert_eq!(result.solution().map(|s| s.cost), optimum);
        }
    }
}

/// On an open grid, A* with farthest and avoidance landmarks expands fewer states than
/// Dijkstra and finds the same cost; ARA* takes the heuristic too.
#[test]
fn alt_expands_fewer_than_dijkstra_on_a_grid() {
    let graph = grid(12);
    let (start, goal) = (5 * 12, 6 * 12 + 11);
    let is_goal = |state: &usize| *state == goal;
    let plain = dijkstra(&graph, start, &is_goal, Limits::NONE);

    for selection in [LandmarkSelection::Farthest, LandmarkSelection::Avoidance] {
        let landmarks = Landmarks::build(&graph, 0, 4, selection);
        assert_eq!(landmarks.landmarks().len(), 4);

        let alt = a_star(
            &graph,
            start,
            &is_goal,
            landmarks.heuristic_to(&goal),
            Limits::NONE,
        );
        assert_eq!(alt.solution().map(|s| s.cost), Some(12));
        assert!(
            alt.stats.expanded < plain.stats.expanded,
            "{selection:?}: {} vs {}",
            alt.stats.expanded,
            plain.stats.expanded
        );

        let anytime = ara_star(
            &graph,
            start,
            &is_goal,
            landmarks.heuristic_to(&goal),
            [Weight::integer(2), Weight::ONE],
            |_, _| {},
            Limits::NONE,
        );
        assert!(anytime.complete);
        assert_eq!(anytime.solution.map(|s| s.cost), Some(12));
    }
}

/// Farthest-first picks the end of the longest round trip from the start; avoidance
/// descends into the heavier subtree of the start's shortest-path tree, which is the
/// bushy one, not the deep one.
#[test]
fn selections_pick_their_first_landmark() {
    // A path 0-1-2-3-9 of unit edges, and a star around 4 (one edge from 0) whose leaf 7
    // is two away from 4, the others one. Edges go both ways.
    let one_way = [
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 9, 1),
        (0, 4, 1),
        (4, 5, 1),
        (4, 6, 1),
        (4, 7, 2),
        (4, 8, 1),
        (4, 10, 1),
    ];
    let edges: Vec<_> = one_way
        .iter()
        .flat_map(|&(a, b, cost)| [(a, b, cost), (b, a, cost)])
        .collect();
    let graph = Graph::new(11, &edges);

    let first = |selection| {
        let landmarks = Landmarks::build(&graph, 0, 1, selection);
        landmarks.landmarks().next().copied()
    };

    assert_eq!(first(LandmarkSelection::Farthest), Some(9));
    assert_eq!(first(LandmarkSelection::Avoidance), Some(7));
}

/// Between two components, the landmarks prove there is no path, so A* expands nothing;
/// states the build never found get the bound `0`; asking for more landmarks than states
/// picks every state once.
#[test]
fn disconnected_states() {
    let graph = Graph::new(4, &[(0, 1, 3), (1, 0, 3), (2, 3, 1), (3, 2, 1)]);
    let landmarks = Landmarks::build(&graph, 0, 10, LandmarkSelection::Farthest);

    let mut picked: Vec<_> = landmarks.landmarks().copied().collect();
    picked.sort_unstable();
    assert_eq!(picked, [0, 1]);

    assert_eq!(landmarks.lower_bound(&0, &2), None);
    assert_eq!(landmarks.lower_bound(&3, &1), None);
    assert_eq!(landmarks.lower_bound(&2, &3), Some(0));
    assert_eq!(landmarks.lower_bound(&1, &0), Some(3));

    let result = a_star(
        &graph,
        0,
        &|s: &usize| *s == 2,
        landmarks.heuristic_to(&2),
        Limits::NONE,
    );
    assert_eq!(result.outcome, Outcome::Unsolvable);
    assert_eq!(result.stats.expanded, 0);

    let none = Landmarks::build(&graph, 0, 0, LandmarkSelection::Avoidance);
    assert_eq!(none.landmarks().len(), 0);
    assert_eq!(none.lower_bound(&0, &2), Some(0));
}
