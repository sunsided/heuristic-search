//! Real-time search: hand-checked updates, the differences between LSS-LRTA*'s update and
//! the limited-horizon Bellman update, the GBFS and BWAS search phases, and property tests
//! on random graphs for admissibility, consistency, convergence to cheapest paths (against
//! Dijkstra), completeness and bounded work per step.

use heuristic_search::{
    Heuristic, LearnedHeuristic, Limits, Outcome, TrialResult, Weight, Zero, dijkstra,
    heuristic_fn, lhb_bwa_star, lhb_gbfs, lhb_lrta_star, lhb_update, lrta_star, lss_lrta_star,
    rta_star, rtaa_star,
};
use proptest::prelude::*;

use crate::fixtures::{
    Graph, assert_valid_path, distances_to, graph_and_goal, strongly_connected_graph_and_goal,
};

/// Expansions one trial may make before the tests call it a failure instead of hanging.
const TRIAL_BUDGET: u64 = 200_000;

/// Trials a learner may need to converge on the random test graphs.
const MAX_TRIALS: usize = 1_000;

/// A real-time search under test, with its lookahead.
#[derive(Debug, Clone, Copy)]
enum Agent {
    /// [`lrta_star`].
    Lrta,
    /// [`rta_star`].
    Rta,
    /// [`lss_lrta_star`] with this lookahead.
    Lss(usize),
    /// [`lhb_lrta_star`] with this lookahead.
    Lhb(usize),
    /// [`lhb_gbfs`] with this lookahead.
    LhbGbfs(usize),
    /// [`lhb_bwa_star`] with this lookahead, weight and batch size.
    LhbBwa(usize, Weight, usize),
    /// [`rtaa_star`] with this lookahead.
    Rtaa(usize),
}

/// The searches whose learned values stay admissible, with several lookaheads.
const LEARNERS: [Agent; 17] = [
    Agent::Lrta,
    Agent::Lss(1),
    Agent::Lss(3),
    Agent::Lss(8),
    Agent::Lhb(1),
    Agent::Lhb(3),
    Agent::Lhb(8),
    Agent::LhbGbfs(1),
    Agent::LhbGbfs(3),
    Agent::LhbGbfs(8),
    Agent::LhbBwa(3, Weight::ONE, 2),
    Agent::LhbBwa(8, Weight::ONE, 3),
    Agent::LhbBwa(8, Weight::new(5, 3), 1),
    Agent::LhbBwa(8, Weight::new(5, 3), 4),
    Agent::Rtaa(1),
    Agent::Rtaa(3),
    Agent::Rtaa(8),
];

impl Agent {
    /// Runs one trial on `graph` from `start` to `goal`, learning in `table`.
    fn trial<H: Heuristic<usize, u64>>(
        self,
        graph: &Graph,
        start: usize,
        goal: usize,
        table: &mut LearnedHeuristic<usize, u64, H>,
    ) -> TrialResult<usize, usize, u64> {
        let is_goal = |state: &usize| *state == goal;
        let budget = Limits::expansions(TRIAL_BUDGET);

        match self {
            Self::Lrta => lrta_star(graph, start, &is_goal, table, budget),
            Self::Rta => rta_star(graph, start, &is_goal, table, budget),
            Self::Lss(n) => lss_lrta_star(graph, start, &is_goal, table, n, budget),
            Self::Lhb(n) => lhb_lrta_star(graph, start, &is_goal, table, n, budget),
            Self::LhbGbfs(n) => lhb_gbfs(graph, start, &is_goal, table, n, budget),
            Self::LhbBwa(n, weight, batch) => {
                lhb_bwa_star(graph, start, &is_goal, table, n, weight, batch, budget)
            }
            Self::Rtaa(n) => rtaa_star(graph, start, &is_goal, table, n, budget),
        }
    }

    /// The most expansions one planning step may make.
    fn lookahead(self) -> u64 {
        match self {
            Self::Lrta | Self::Rta => 1,
            Self::Lss(n)
            | Self::Lhb(n)
            | Self::LhbGbfs(n)
            | Self::LhbBwa(n, ..)
            | Self::Rtaa(n) => n as u64,
        }
    }

    /// The weight `w` such that a trial that learns nothing costs at most `w` times the
    /// optimum, or `None` if there is no such bound: GBFS ignores path costs, and a batch
    /// expands states out of order.
    fn converged_bound(self) -> Option<Weight> {
        match self {
            Self::LhbGbfs(_) | Self::LhbBwa(_, _, 2..) => None,
            Self::LhbBwa(_, weight, _) => Some(weight),
            _ => Some(Weight::ONE),
        }
    }

    /// The most table values one planning step may change: one per expansion, plus the
    /// goal that the limited-horizon update sets to zero.
    fn max_updates(self) -> u64 {
        match self {
            Self::Lhb(_) | Self::LhbGbfs(_) | Self::LhbBwa(..) => self.lookahead() + 1,
            _ => self.lookahead(),
        }
    }
}

/// Asserts that every value in `table` is at most the exact cost in `exact`, and that only
/// real dead ends have no value.
fn assert_admissible<H: Heuristic<usize, u64>>(
    table: &mut LearnedHeuristic<usize, u64, H>,
    exact: &[Option<u64>],
) {
    for (state, &exact) in exact.iter().enumerate() {
        match (table.get(&state), exact) {
            (Some(value), Some(exact)) => assert!(value <= exact, "h({state}) = {value} > {exact}"),
            (None, Some(exact)) => panic!("h({state}) is a dead end, but costs {exact}"),
            (_, None) => {}
        }
    }
}

/// Asserts that `table` is consistent on `graph`: `h(s) <= c + h(s')` for every edge,
/// where a dead end counts as infinite, and the goal has value zero.
fn assert_consistent<H: Heuristic<usize, u64>>(
    graph: &Graph,
    goal: usize,
    table: &mut LearnedHeuristic<usize, u64, H>,
) {
    assert_eq!(table.get(&goal), Some(0), "the goal keeps value zero");

    for (from, edges) in graph.edges.iter().enumerate() {
        for &(to, cost) in edges {
            let Some(next) = table.get(&to) else {
                continue;
            };

            let value = table.get(&from);
            assert!(
                value.is_some_and(|value| value <= cost + next),
                "h({from}) = {value:?} > {cost} + h({to}) = {}",
                cost + next
            );
        }
    }
}

/// The cheapest cost from `start` to `goal`, by the crate's Dijkstra.
fn dijkstra_cost(graph: &Graph, start: usize, goal: usize) -> Option<u64> {
    dijkstra(graph, start, &|state: &usize| *state == goal, Limits::NONE)
        .into_solution()
        .map(|solution| solution.cost)
}

/// Runs trials of `agent` from `start` until one changes no value, checking every trial's
/// path, work bound and admissibility, and returns the number of trials.
///
/// The converged trial's cost must be within [`Agent::converged_bound`] of the cheapest
/// cost.
fn converge<H: Heuristic<usize, u64>>(
    agent: Agent,
    graph: &Graph,
    start: usize,
    goal: usize,
    table: &mut LearnedHeuristic<usize, u64, H>,
) -> usize {
    let exact = distances_to(graph, goal);

    for trial in 1..=MAX_TRIALS {
        let result = agent.trial(graph, start, goal, table);
        let solution = result
            .solution()
            .expect("strongly connected: the goal is reached");
        assert_valid_path(graph, start, solution);
        assert!(result.trial.max_step_expansions <= agent.lookahead());
        assert_admissible(table, &exact);

        if result.trial.updates == 0 {
            let optimum = exact[start].expect("strongly connected");
            assert_eq!(Some(optimum), dijkstra_cost(graph, start, goal));
            if let Some(weight) = agent.converged_bound() {
                assert!(
                    solution.cost * u64::from(weight.denominator())
                        <= optimum * u64::from(weight.numerator()),
                    "{agent:?} converged at {} against the optimum {optimum}",
                    solution.cost
                );
            }

            return trial;
        }
    }

    panic!("{agent:?} did not converge in {MAX_TRIALS} trials");
}

/// The graph of the limited-horizon example: `0 -> 1 -> 3` (a leaf with a high estimate)
/// and `0 -> 2 -> 4` (the goal). All values are admissible; the true cost from `0` is 3.
fn lookahead_example() -> (Graph, [u64; 5]) {
    let graph = Graph::new(5, &[(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 4, 2)]);

    (graph, [0, 1, 2, 9, 0])
}

/// With one expansion the update is the single-step Bellman update; with three, A*
/// reaches the goal and the update learns the exact cost of the start.
#[test]
fn limited_horizon_update_beats_single_step() {
    let (graph, h) = lookahead_example();
    let is_goal = |state: &usize| *state == 4;

    let mut single = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));
    assert_eq!(lhb_update(&graph, 0, &is_goal, &mut single, 1), 1);
    assert_eq!(
        single.get(&0),
        Some(2),
        "min(1 + h(1), 1 + h(2)) underestimates"
    );

    let mut horizon = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));
    lhb_update(&graph, 0, &is_goal, &mut horizon, 3);
    assert_eq!(horizon.get(&0), Some(3), "0 -> 2 -> 4, the exact cost");
    assert_eq!(horizon.get(&1), Some(10), "1 -> 3 and h(3) = 9");
    assert_eq!(horizon.get(&2), Some(2));
    assert_eq!(horizon.get(&3), Some(9), "a leaf keeps its value");
}

/// LRTA*'s first step from the example (with edges back out of the leaves) moves to `1`,
/// whose `c + h` is the smallest, and the trial still reaches the goal.
#[test]
fn lrta_star_updates_and_moves_to_the_best_successor() {
    let (mut graph, h) = lookahead_example();
    graph.edges[1].push((0, 1));
    graph.edges[3].push((1, 1));
    let mut table = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));

    let result = lrta_star(
        &graph,
        0,
        &|state: &usize| *state == 4,
        &mut table,
        Limits::NONE,
    );
    let solution = result.solution().expect("the goal is reachable");

    assert_eq!(solution.states[..2], [0, 1]);
    assert!(solution.cost > 3, "the detour through 1 costs extra");
    assert_valid_path(&graph, 0, solution);
}

/// A state without successors: LSS-LRTA* proves it a dead end, the limited-horizon update
/// keeps its value as a leaf, so the start's values differ.
#[test]
fn updates_differ_on_states_without_successors() {
    let graph = Graph::new(3, &[(0, 1, 1), (0, 2, 5)]);
    let is_goal = |state: &usize| *state == 2;

    let mut lss = LearnedHeuristic::new(Zero);
    let result = lss_lrta_star(&graph, 0, &is_goal, &mut lss, 2, Limits::NONE);
    assert_eq!(result.solution().map(|path| path.cost), Some(5));
    assert_eq!(lss.get(&1), None, "proven dead end");
    assert_eq!(lss.get(&0), Some(5));

    let mut lhb = LearnedHeuristic::new(Zero);
    let result = lhb_lrta_star(&graph, 0, &is_goal, &mut lhb, 2, Limits::NONE);
    assert_eq!(result.solution().map(|path| path.cost), Some(5));
    assert_eq!(lhb.get(&1), Some(0), "a leaf keeps its value");
    assert_eq!(lhb.get(&0), Some(1), "through the leaf 1");
}

/// A goal with a nonzero value: the limited-horizon update sets it to zero (Eq. 5), but
/// still uses the old value for the states before it; LSS-LRTA* leaves it alone.
#[test]
fn updates_differ_on_goals_with_nonzero_values() {
    let graph = Graph::new(2, &[(0, 1, 1)]);
    let is_goal = |state: &usize| *state == 1;
    let three = |_: &usize| Some(3);

    let mut lhb = LearnedHeuristic::new(heuristic_fn(three));
    assert_eq!(lhb_update(&graph, 0, &is_goal, &mut lhb, 1), 2);
    assert_eq!((lhb.get(&0), lhb.get(&1)), (Some(4), Some(0)));

    let mut lss = LearnedHeuristic::new(heuristic_fn(three));
    lss_lrta_star(&graph, 0, &is_goal, &mut lss, 1, Limits::NONE);
    assert_eq!((lss.get(&0), lss.get(&1)), (Some(4), Some(3)));
}

/// A state reopened by the lookahead under an admissible, inconsistent table: interior for
/// the limited-horizon update, open for LSS-LRTA*. Exact costs are `[7, 5, 6, 0]`; the
/// first lookahead expands 0, 1 and 2, and 2 reopens 1. LSS-LRTA* seeds from 1's old value
/// and lowers `h(2)` from 5 to 1; LHB routes through the leaf 3 and learns exact values.
#[test]
fn updates_differ_on_reopened_states() {
    let graph = Graph::new(4, &[(0, 1, 3), (0, 2, 1), (2, 1, 1), (1, 3, 5)]);
    let is_goal = |state: &usize| *state == 3;
    let h = [0, 0, 5, 0];

    let mut lss = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));
    let result = lss_lrta_star(&graph, 0, &is_goal, &mut lss, 3, Limits::NONE);
    assert_eq!(result.stats.reopened, 1);
    assert_eq!(
        (lss.get(&0), lss.get(&1), lss.get(&2)),
        (Some(2), Some(5), Some(1))
    );

    let mut lhb = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));
    lhb_lrta_star(&graph, 0, &is_goal, &mut lhb, 3, Limits::NONE);
    assert_eq!(
        (lhb.get(&0), lhb.get(&1), lhb.get(&2)),
        (Some(7), Some(5), Some(6))
    );
}

/// A start that is a goal ends the trial at once, and the update of a goal sets zero.
#[test]
fn goal_start_ends_at_once() {
    let graph = Graph::new(1, &[]);
    let mut table = LearnedHeuristic::new(Zero);

    let result = lss_lrta_star(&graph, 0, &|_: &usize| true, &mut table, 4, Limits::NONE);
    assert_eq!(
        result.solution().map(|path| path.states.clone()),
        Some(vec![0])
    );
    assert_eq!(result.trial.steps, 0);
    assert_eq!(lhb_update(&graph, 0, &|_: &usize| true, &mut table, 4), 0);
}

/// A lookahead that covers every reachable state proves the goal unreachable.
#[test]
fn lookahead_proves_unsolvable() {
    let graph = Graph::new(4, &[(0, 1, 1), (1, 0, 1), (1, 2, 1), (3, 0, 1)]);
    let is_goal = |state: &usize| *state == 3;
    let mut table = LearnedHeuristic::new(Zero);

    let result = lss_lrta_star(&graph, 0, &is_goal, &mut table, 8, Limits::NONE);
    assert_eq!(result.outcome, Outcome::Unsolvable);
    assert_eq!(
        (table.get(&0), table.get(&1), table.get(&2)),
        (None, None, None)
    );

    let result = lrta_star(&graph, 0, &is_goal, &mut table, Limits::NONE);
    assert_eq!(
        result.outcome,
        Outcome::Unsolvable,
        "the start is now a known dead end"
    );
}

/// The budget stops a trial that cannot finish.
#[test]
fn budget_stops_a_trial() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 0, 1), (2, 0, 1)]);
    let is_goal = |state: &usize| *state == 2;
    let mut table = LearnedHeuristic::new(Zero);

    let result = lrta_star(&graph, 0, &is_goal, &mut table, Limits::expansions(50));
    assert_eq!(result.outcome, Outcome::LimitReached);
    assert_eq!(result.stats.expanded, 50);
}

/// GBFS moves towards the smallest value, not the cheapest path: from `0` it heads for `2`
/// (value 0, edge cost 4) where A* heads for `1`. Both learn the exact `h(0) = 2`. The
/// table then stops changing while GBFS keeps taking the detour: a trial that learns
/// nothing need not be cheapest when the search phase ignores path costs.
#[test]
fn gbfs_phase_follows_the_values() {
    let graph = Graph::new(4, &[(0, 1, 1), (0, 2, 4), (1, 3, 1), (2, 3, 1)]);
    let is_goal = |state: &usize| *state == 3;
    let h = [0, 1, 0, 0];

    let mut a_star = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));
    let result = lhb_lrta_star(&graph, 0, &is_goal, &mut a_star, 1, Limits::NONE);
    assert_eq!(
        result.solution().map(|path| path.states.clone()),
        Some(vec![0, 1, 3])
    );
    assert_eq!(a_star.get(&0), Some(2));

    let mut greedy = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(h[*state])));
    let result = lhb_gbfs(&graph, 0, &is_goal, &mut greedy, 1, Limits::NONE);
    assert_eq!(
        result.solution().map(|path| path.states.clone()),
        Some(vec![0, 2, 3])
    );
    assert_eq!((greedy.get(&0), greedy.get(&2)), (Some(2), Some(1)));

    let result = lhb_gbfs(&graph, 0, &is_goal, &mut greedy, 1, Limits::NONE);
    assert_eq!(result.trial.updates, 0);
    assert_eq!(result.solution().map(|path| path.cost), Some(5));
}

/// A batch takes several open states before expanding any; when a goal is among them, the
/// step stops without expanding the batch. With a batch of two, `1` and the goal `2` are
/// taken together after `0`, so only `0` is expanded; a batch of one expands `1` first.
#[test]
fn bwa_star_stops_on_a_goal_in_the_batch() {
    let graph = Graph::new(3, &[(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
    let is_goal = |state: &usize| *state == 2;

    for (batch, expanded) in [(1, 2), (2, 1)] {
        let mut table = LearnedHeuristic::new(Zero);
        let result = lhb_bwa_star(
            &graph,
            0,
            &is_goal,
            &mut table,
            8,
            Weight::ONE,
            batch,
            Limits::NONE,
        );

        assert_eq!(
            result.solution().map(|path| path.states.clone()),
            Some(vec![0, 2])
        );
        assert_eq!(result.stats.expanded, expanded, "batch {batch}");
        assert_eq!(table.get(&0), Some(1));
    }
}

/// The budget stops GBFS and BWAS trials that cannot finish, after exactly the allowed
/// expansions.
#[test]
fn budget_stops_search_phase_trials() {
    let graph = Graph::new(3, &[(0, 1, 1), (1, 0, 1), (2, 0, 1)]);
    let is_goal = |state: &usize| *state == 2;

    for agent in [
        Agent::LhbGbfs(1),
        Agent::LhbBwa(1, Weight::ONE, 3),
        Agent::LhbBwa(1, Weight::integer(2), 1),
    ] {
        let mut table = LearnedHeuristic::new(Zero);
        let budget = Limits::expansions(50);
        let result = match agent {
            Agent::LhbGbfs(n) => lhb_gbfs(&graph, 0, &is_goal, &mut table, n, budget),
            Agent::LhbBwa(n, weight, batch) => {
                lhb_bwa_star(&graph, 0, &is_goal, &mut table, n, weight, batch, budget)
            }
            _ => unreachable!("only search-phase agents"),
        };

        assert_eq!(result.outcome, Outcome::LimitReached, "{agent:?}");
        assert_eq!(result.stats.expanded, 50, "{agent:?}");
    }
}

/// A batch of zero cannot make progress.
#[test]
#[should_panic(expected = "a batch needs at least one state")]
fn bwa_star_rejects_an_empty_batch() {
    let graph = Graph::new(2, &[(0, 1, 1)]);
    let mut table = LearnedHeuristic::new(Zero);
    lhb_bwa_star(
        &graph,
        0,
        &|state: &usize| *state == 1,
        &mut table,
        4,
        Weight::ONE,
        0,
        Limits::NONE,
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// From an admissible, usually inconsistent table (random fractions of the exact cost),
    /// every learner keeps it admissible after every trial, on graphs with zero costs, dead
    /// ends and unreachable goals, and every path it reports is real.
    #[test]
    fn learning_keeps_admissibility(
        (graph, goal) in graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        start in 0_usize..12,
    ) {
        let start = start % graph.edges.len();
        let exact = distances_to(&graph, goal);

        for agent in LEARNERS {
            let base = heuristic_fn(|state: &usize| exact[*state].map(|d| d * scales[*state] / 4));
            let mut table = LearnedHeuristic::new(base);
            for _ in 0..5 {
                let result = agent.trial(&graph, start, goal, &mut table);
                if let Some(solution) = result.solution() {
                    assert_valid_path(&graph, start, solution);
                }

                assert_admissible(&mut table, &exact);
            }
        }
    }

    /// From a consistent table (the exact cost capped), every learner keeps it consistent
    /// after every trial.
    #[test]
    fn learning_keeps_consistency(
        (graph, goal) in graph_and_goal(),
        cap in 0_u64..30,
        start in 0_usize..12,
    ) {
        let start = start % graph.edges.len();
        let exact = distances_to(&graph, goal);

        for agent in LEARNERS {
            let base = heuristic_fn(|state: &usize| exact[*state].map(|d| d.min(cap)));
            let mut table = LearnedHeuristic::new(base);
            for _ in 0..5 {
                agent.trial(&graph, start, goal, &mut table);
                assert_consistent(&graph, goal, &mut table);
            }
        }
    }

    /// Every search, RTA* included and from an arbitrary (inadmissible) table, reaches the
    /// goal on a strongly connected graph, within its work bound per step.
    #[test]
    fn real_time_search_is_complete(
        (graph, goal) in strongly_connected_graph_and_goal(),
        values in prop::collection::vec(0_u64..50, 12),
        start in 0_usize..12,
    ) {
        let start = start % graph.edges.len();

        for agent in LEARNERS.into_iter().chain([Agent::Rta]) {
            let mut table = LearnedHeuristic::new(heuristic_fn(|state: &usize| Some(values[*state])));
            let result = agent.trial(&graph, start, goal, &mut table);
            let solution = result.solution();

            prop_assert!(solution.is_some(), "{:?}: {:?}", agent, result.outcome);
            assert_valid_path(&graph, start, solution.expect("checked"));
            prop_assert!(result.trial.max_step_expansions <= agent.lookahead());
            prop_assert!(result.trial.max_step_updates <= agent.max_updates());
            prop_assert!(result.trial.moves >= result.trial.steps);
        }
    }

    /// Repeated trials from a consistent table converge: some trial learns nothing, and its
    /// path is a cheapest one.
    #[test]
    fn trials_converge_from_consistent_values(
        (graph, goal) in strongly_connected_graph_and_goal(),
        cap in 0_u64..30,
        start in 0_usize..12,
    ) {
        let start = start % graph.edges.len();
        let exact = distances_to(&graph, goal);

        for agent in LEARNERS {
            let base = heuristic_fn(|state: &usize| exact[*state].map(|d| d.min(cap)));
            converge(agent, &graph, start, goal, &mut LearnedHeuristic::new(base));
        }
    }

    /// Repeated trials from an admissible but inconsistent table converge as well.
    #[test]
    fn trials_converge_from_admissible_values(
        (graph, goal) in strongly_connected_graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        start in 0_usize..12,
    ) {
        let start = start % graph.edges.len();
        let exact = distances_to(&graph, goal);

        for agent in LEARNERS {
            let base = heuristic_fn(|state: &usize| exact[*state].map(|d| d * scales[*state] / 4));
            converge(agent, &graph, start, goal, &mut LearnedHeuristic::new(base));
        }
    }

    /// Where every state has a successor, goals have value zero and the table is consistent
    /// (so the lookahead never reopens), LSS-LRTA* and the limited-horizon update learn the
    /// same values and the agents travel the same paths, trial for trial.
    #[test]
    fn lss_and_limited_horizon_agree_without_dead_ends(
        (graph, goal) in strongly_connected_graph_and_goal(),
        cap in 0_u64..30,
        start in 0_usize..12,
        lookahead in 1_usize..10,
    ) {
        let start = start % graph.edges.len();
        let exact = distances_to(&graph, goal);
        let base = |state: &usize| exact[*state].map(|d| d.min(cap));
        let mut lss = LearnedHeuristic::new(heuristic_fn(base));
        let mut lhb = LearnedHeuristic::new(heuristic_fn(base));

        for _ in 0..10 {
            let by_lss = Agent::Lss(lookahead).trial(&graph, start, goal, &mut lss);
            let by_lhb = Agent::Lhb(lookahead).trial(&graph, start, goal, &mut lhb);
            prop_assert_eq!(&by_lss.outcome, &by_lhb.outcome);
            prop_assert_eq!(by_lss.trial.updates, by_lhb.trial.updates);

            for state in 0..graph.edges.len() {
                prop_assert_eq!(lss.get(&state), lhb.get(&state));
            }
        }
    }

    /// BWAS with weight one and a batch of one is A*: its agent is `lhb_lrta_star`, trial
    /// for trial, from admissible, usually inconsistent tables.
    #[test]
    fn bwa_star_with_weight_one_and_batch_one_is_lhb_lrta_star(
        (graph, goal) in strongly_connected_graph_and_goal(),
        scales in prop::collection::vec(0_u64..=4, 12),
        start in 0_usize..12,
        lookahead in 1_usize..10,
    ) {
        let start = start % graph.edges.len();
        let exact = distances_to(&graph, goal);
        let base = |state: &usize| exact[*state].map(|d| d * scales[*state] / 4);
        let mut a_star = LearnedHeuristic::new(heuristic_fn(base));
        let mut bwas = LearnedHeuristic::new(heuristic_fn(base));

        for _ in 0..5 {
            let by_a_star = Agent::Lhb(lookahead).trial(&graph, start, goal, &mut a_star);
            let by_bwas =
                Agent::LhbBwa(lookahead, Weight::ONE, 1).trial(&graph, start, goal, &mut bwas);
            prop_assert_eq!(&by_a_star, &by_bwas);

            for state in 0..graph.edges.len() {
                prop_assert_eq!(a_star.get(&state), bwas.get(&state));
            }
        }
    }

    /// Tabular limited-horizon learning as in the paper, offline: sweeping the update over
    /// every state until nothing changes reaches the exact costs, for every horizon.
    #[test]
    fn offline_limited_horizon_learning_reaches_exact_costs(
        (graph, goal) in strongly_connected_graph_and_goal(),
        horizon in 1_usize..10,
    ) {
        let exact = distances_to(&graph, goal);
        let is_goal = |state: &usize| *state == goal;
        let mut table = LearnedHeuristic::new(Zero);

        let mut sweeps = 0;
        loop {
            let changed: u64 = (0..graph.edges.len())
                .map(|state| lhb_update(&graph, state, &is_goal, &mut table, horizon))
                .sum();
            assert_admissible(&mut table, &exact);
            sweeps += 1;
            prop_assert!(sweeps < 10_000, "no convergence");

            if changed == 0 {
                break;
            }
        }

        for (state, &exact) in exact.iter().enumerate() {
            prop_assert_eq!(table.get(&state), exact);
            prop_assert_eq!(exact, dijkstra_cost(&graph, state, goal));
        }
    }
}
