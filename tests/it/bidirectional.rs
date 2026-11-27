//! Bidirectional searches: hand-made cases and cross-checks against the distance oracle on
//! random graphs with several goals, and MM's meet-in-the-middle bound on its expansions.

use std::cell::RefCell;

use heuristic_search::{
    Limits, Outcome, Predecessors, Successors, Zero, bae_star, bidirectional_bfs,
    bidirectional_dijkstra, dibbs, heuristic_fn, mm, nbs,
};
use proptest::prelude::*;

use crate::fixtures::{Graph, assert_valid_path, distances_from, distances_to, graph_and_goal};

/// The smallest of `distances` over `goals`, per node: the distance to the nearest goal.
fn nearest(graph: &Graph, goals: &[usize]) -> Vec<Option<u64>> {
    let per_goal: Vec<_> = goals
        .iter()
        .map(|&goal| distances_to(graph, goal))
        .collect();

    (0..graph.edges.len())
        .map(|node| per_goal.iter().filter_map(|d| d[node]).min())
        .collect()
}

/// Unreachable goals are unsolvable; a start among the goals costs nothing.
#[test]
fn unsolvable_and_trivial() {
    let graph = Graph::new(3, &[(0, 1, 1)]);

    let results = [
        bidirectional_bfs(&graph, 0, [2], Limits::NONE),
        bidirectional_dijkstra(&graph, 0, [2], Limits::NONE),
        nbs(&graph, 0, [2], Zero, Zero, Limits::NONE),
        dibbs(&graph, 0, [2], Zero, Zero, Limits::NONE),
        bae_star(&graph, 0, [2], Zero, Zero, Limits::NONE),
        mm(&graph, 0, [2], Zero, Zero, Limits::NONE),
    ];
    assert!(results.iter().all(|r| r.outcome == Outcome::Unsolvable));

    let results = [
        bidirectional_bfs(&graph, 0, [1, 0], Limits::NONE),
        bidirectional_dijkstra(&graph, 0, [1, 0], Limits::NONE),
        nbs(&graph, 0, [1, 0], Zero, Zero, Limits::NONE),
        dibbs(&graph, 0, [1, 0], Zero, Zero, Limits::NONE),
        bae_star(&graph, 0, [1, 0], Zero, Zero, Limits::NONE),
        mm(&graph, 0, [1, 0], Zero, Zero, Limits::NONE),
    ];
    for result in results {
        assert_eq!(result.solution().map(|s| s.cost), Some(0));
        assert_eq!(result.stats.expanded, 0);
    }
}

/// A budget of zero expansions stops every search before it decides.
#[test]
fn budget_stops_search() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 2, 1)]);
    let none = Limits::expansions(0);

    assert_eq!(
        bidirectional_bfs(&graph, 0, [2], none).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        bidirectional_dijkstra(&graph, 0, [2], none).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        nbs(&graph, 0, [2], Zero, Zero, none).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        mm(&graph, 0, [2], Zero, Zero, none).outcome,
        Outcome::LimitReached
    );
}

/// On a line of three unit edges MM0 stops after three expansions through the
/// `g_min_F + g_min_B` term of its stopping rule; the other terms alone would need a
/// fourth.
#[test]
fn mm_stops_on_g_sum() {
    let graph = Graph::new(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 1)]);

    let result = mm(&graph, 0, [3], Zero, Zero, Limits::NONE);

    assert_eq!(result.solution().map(|s| s.cost), Some(3));
    assert_eq!(result.stats.expanded, 3);
}

/// A [`Graph`] that records the state of every expansion: forward ones through
/// [`Successors`], backward ones through [`Predecessors`].
struct Recorder<'a> {
    /// The graph searched.
    graph: &'a Graph,
    /// States expanded forwards, in order.
    forward: RefCell<Vec<usize>>,
    /// States expanded backwards, in order.
    backward: RefCell<Vec<usize>>,
}

impl Successors for Recorder<'_> {
    type State = usize;
    type Action = usize;
    type Cost = u64;

    fn successors(&self, state: &usize, emit: impl FnMut(usize, usize, u64)) {
        self.forward.borrow_mut().push(*state);
        self.graph.successors(state, emit);
    }
}

impl Predecessors for Recorder<'_> {
    type Node = usize;
    type Action = usize;
    type Cost = u64;

    fn predecessors(&self, node: &usize, emit: impl FnMut(usize, usize, u64)) {
        self.backward.borrow_mut().push(*node);
        self.graph.predecessors(node, emit);
    }
}

/// Bidirectional Dijkstra prefers three cheap edges over one expensive edge, and the path
/// joins both halves correctly.
#[test]
fn joins_both_halves() {
    let graph = Graph::new(5, &[(0, 4, 10), (0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);

    let solution = bidirectional_dijkstra(&graph, 0, [4], Limits::NONE)
        .into_solution()
        .expect("4 is reachable");

    assert_eq!(solution.states, [0, 1, 2, 3, 4]);
    assert_eq!(solution.cost, 4);
    assert_valid_path(&graph, 0, &solution);
}

proptest! {
    /// All bidirectional searches agree with the oracle: the cheapest cost to the nearest
    /// goal (fewest edges for BFS). NBS gets admissible, inconsistent heuristics; DIBBS and
    /// BAE* consistent ones (exact distances scaled by a common factor).
    #[test]
    fn bidirectional_matches_oracle(
        (graph, goal) in graph_and_goal(),
        extra_goals in prop::collection::vec(0_usize..12, 0..3),
        scales in prop::collection::vec(0_u64..=4, 12),
        scale in 0_u64..=4,
    ) {
        let nodes = graph.edges.len();
        let mut goals = vec![goal];
        goals.extend(extra_goals.into_iter().map(|g| g % nodes));
        let to_goal = nearest(&graph, &goals);
        let from_start = distances_from(&graph, 0);
        let unit_cost = nearest(&graph.unit(), &goals)[0];

        let scaled = |d: &Vec<Option<u64>>, s: usize, k: u64| d[s].map_or(0, |d| d * k / 4);
        // NBS gets `None` where a state cannot be on any start-goal path: true dead ends.
        let forward = |s: &usize| to_goal[*s].map(|d| d * scales[*s] / 4);
        let backward = |s: &usize| from_start[*s].map(|d| d * scales[(*s + 5) % 12] / 4);
        let forward_c = |s: &usize| Some(scaled(&to_goal, *s, scale));
        let backward_c = |s: &usize| Some(scaled(&from_start, *s, scale));
        let goals_iter = || goals.iter().copied();

        let shortest = bidirectional_bfs(&graph, 0, goals_iter(), Limits::NONE);
        prop_assert_eq!(shortest.solution().map(|s| s.len() as u64), unit_cost);

        let results = [
            bidirectional_dijkstra(&graph, 0, goals_iter(), Limits::NONE),
            nbs(&graph, 0, goals_iter(), heuristic_fn(forward), heuristic_fn(backward), Limits::NONE),
            dibbs(&graph, 0, goals_iter(), heuristic_fn(forward_c), heuristic_fn(backward_c), Limits::NONE),
            bae_star(&graph, 0, goals_iter(), heuristic_fn(forward_c), heuristic_fn(backward_c), Limits::NONE),
            mm(&graph, 0, goals_iter(), heuristic_fn(forward), heuristic_fn(backward), Limits::NONE),
            mm(&graph, 0, goals_iter(), Zero, Zero, Limits::NONE),
        ];
        for (index, result) in results.iter().enumerate() {
            prop_assert_eq!(result.solution().map(|s| s.cost), to_goal[0], "algorithm {}", index);
        }

        for result in results.iter().chain([&shortest]) {
            if let Some(solution) = result.solution() {
                assert_valid_path(&graph, 0, solution);
                prop_assert!(goals.contains(solution.goal()));
            }
        }
    }

    /// MM meets in the middle (Holte et al. 2016, Theorem 1): every state it expands has
    /// `pr <= C*`, and as `pr >= 2g`, a cost from its side's root of at most `C* / 2`. The
    /// true distance never exceeds the `g` at expansion, so `2 * distance <= C*` holds for
    /// every expansion, with scaled and halved distances and with zero heuristics.
    #[test]
    fn mm_meets_in_the_middle(
        (graph, goal) in graph_and_goal(),
        extra_goals in prop::collection::vec(0_usize..12, 0..3),
        scales in prop::collection::vec(0_u64..=4, 12),
    ) {
        let nodes = graph.edges.len();
        let mut goals = vec![goal];
        goals.extend(extra_goals.into_iter().map(|g| g % nodes));
        let to_goal = nearest(&graph, &goals);
        let from_start = distances_from(&graph, 0);
        let Some(optimal) = to_goal[0] else {
            return Ok(());
        };

        let forward = |s: &usize| to_goal[*s].map(|d| d * scales[*s] / 4);
        let backward = |s: &usize| from_start[*s].map(|d| d * scales[(*s + 5) % 12] / 4);
        let forward_c = |s: &usize| Some(to_goal[*s].map_or(0, |d| d / 2));
        let backward_c = |s: &usize| Some(from_start[*s].map_or(0, |d| d / 2));
        let goals_iter = || goals.iter().copied();

        for heuristics in 0..3 {
            let recorder = Recorder { graph: &graph, forward: RefCell::default(), backward: RefCell::default() };
            let result = match heuristics {
                0 => mm(&recorder, 0, goals_iter(), heuristic_fn(forward), heuristic_fn(backward), Limits::NONE),
                1 => mm(&recorder, 0, goals_iter(), heuristic_fn(forward_c), heuristic_fn(backward_c), Limits::NONE),
                _ => mm(&recorder, 0, goals_iter(), Zero, Zero, Limits::NONE),
            };
            prop_assert_eq!(result.solution().map(|s| s.cost), Some(optimal));

            for &state in recorder.forward.borrow().iter() {
                let distance = from_start[state].expect("expanded states are reachable");
                prop_assert!(2 * distance <= optimal, "forward {} at {} of {}", state, distance, optimal);
            }
            for &state in recorder.backward.borrow().iter() {
                let distance = to_goal[state].expect("expanded states reach a goal");
                prop_assert!(2 * distance <= optimal, "backward {} at {} of {}", state, distance, optimal);
            }
        }
    }
}
