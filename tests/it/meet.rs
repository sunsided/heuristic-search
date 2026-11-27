//! Bidirectional searches with a meet relation: over subgoals of their own type
//! (regression of delete-free STRIPS actions) they match forward Dijkstra and BFS, with
//! zero and with real admissible, consistent heuristics; by equality they match Dijkstra on
//! random graphs; and a model whose regression the successors cannot follow is a
//! `ReplayError`.

use heuristic_search::{
    Heuristic, Limits, Predecessors, ReplayError, Solution, Successors, Zero, bae_star_meet, bfs,
    bidirectional_bfs_meet, bidirectional_dijkstra_meet, dibbs_meet, dijkstra, meet_fn, mm_meet,
    nbs_meet,
};
use proptest::prelude::*;

use crate::fixtures::{assert_valid_path, distances_to, graph_and_goal};

/// Delete-free STRIPS actions over four facts (bits), as `(precondition, add, cost)`.
#[derive(Debug, Clone)]
struct Strips(Vec<(u8, u8, u32)>);

/// A subgoal: the facts that must hold. A type of its own, apart from states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Subgoal(u8);

impl Successors for Strips {
    type State = u8;
    type Action = usize;
    type Cost = u32;

    fn successors(&self, state: &u8, mut emit: impl FnMut(usize, u8, u32)) {
        for (index, &(pre, add, cost)) in self.0.iter().enumerate() {
            // No-op applications stay edges (self-loops): regression may use an action
            // whose facts the state already has, and the replay must be able to follow it.
            if state & pre == pre {
                emit(index, state | add, cost);
            }
        }
    }
}

/// Regression: an action that adds part of a subgoal leads back to the subgoal without
/// those facts but with the action's precondition.
impl Predecessors for Strips {
    type Node = Subgoal;
    type Action = usize;
    type Cost = u32;

    fn predecessors(&self, subgoal: &Subgoal, mut emit: impl FnMut(usize, Subgoal, u32)) {
        for (index, &(pre, add, cost)) in self.0.iter().enumerate() {
            if subgoal.0 & add != 0 {
                emit(index, Subgoal((subgoal.0 & !add) | pre), cost);
            }
        }
    }
}

impl Strips {
    /// The cheapest action that adds `fact`, or `None` if none does.
    fn cheapest_adder(&self, fact: u8) -> Option<u32> {
        self.0
            .iter()
            .filter(|action| action.1 & fact != 0)
            .map(|action| action.2)
            .min()
    }

    /// `max` over the facts in `needed` but not in `have` of their cheapest adder: an
    /// admissible, consistent estimate of reaching `needed` from `have`, `None` if a fact
    /// can never be added.
    fn h_max(&self, have: u8, needed: u8) -> Option<u32> {
        (0..4)
            .map(|bit| 1_u8 << bit)
            .filter(|&fact| needed & fact != 0 && have & fact == 0)
            .try_fold(0, |max, fact| {
                self.cheapest_adder(fact).map(|cost| max.max(cost))
            })
    }
}

/// `h_max` towards the goal: on a state its facts, on a subgoal its own facts (the
/// cheapest state that satisfies it).
struct ToGoal<'a>(&'a Strips, u8);

impl Heuristic<u8, u32> for ToGoal<'_> {
    fn estimate(&mut self, state: &u8) -> Option<u32> {
        self.0.h_max(*state, self.1)
    }
}

impl Heuristic<Subgoal, u32> for ToGoal<'_> {
    fn estimate(&mut self, subgoal: &Subgoal) -> Option<u32> {
        self.0.h_max(subgoal.0, self.1)
    }
}

/// `h_max` from the start: the facts a state or subgoal needs beyond the start's.
struct FromStart<'a>(&'a Strips, u8);

impl Heuristic<u8, u32> for FromStart<'_> {
    fn estimate(&mut self, state: &u8) -> Option<u32> {
        self.0.h_max(self.1, *state)
    }
}

impl Heuristic<Subgoal, u32> for FromStart<'_> {
    fn estimate(&mut self, subgoal: &Subgoal) -> Option<u32> {
        self.0.h_max(self.1, subgoal.0)
    }
}

/// Random delete-free tasks: up to six actions over four facts, a start state and a goal.
fn task() -> impl Strategy<Value = (Strips, u8, u8)> {
    let action = (0_u8..16, 1_u8..16, 0_u32..5);

    (prop::collection::vec(action, 0..7), 0_u8..16, 0_u8..16)
        .prop_map(|(actions, start, goal)| (Strips(actions), start, goal))
}

/// Checks that a solution is a real forward path from `start` to a goal-satisfying state.
fn assert_real_path(task: &Strips, start: u8, goal: u8, solution: &Solution<u8, usize, u32>) {
    assert_eq!(solution.states[0], start);
    assert_eq!(
        solution.goal() & goal,
        goal,
        "the last state satisfies the goal"
    );

    let mut cost = 0;
    for (step, &action) in solution.actions.iter().enumerate() {
        let (pre, add, edge) = task.0[action];
        let state = solution.states[step];
        assert_eq!(state & pre, pre, "step {step} is applicable");
        assert_eq!(
            solution.states[step + 1],
            state | add,
            "step {step} adds its facts"
        );
        cost += edge;
    }

    assert_eq!(cost, solution.cost);
}

/// A state meets a subgoal it satisfies.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "meet relations take both sides by reference"
)]
fn satisfies(state: &u8, subgoal: &Subgoal) -> bool {
    state & subgoal.0 == subgoal.0
}

proptest! {
    /// Over subgoals, every meet search finds the cheapest plan forward Dijkstra finds,
    /// blind and with `h_max` in both directions; bidirectional BFS the fewest actions.
    #[test]
    fn subgoal_meets_match_forward_search((task, start, goal) in task()) {
        let reached = |s: &u8| s & goal == goal;
        let optimal = dijkstra(&task, start, &reached, Limits::NONE).into_solution().map(|s| s.cost);
        let fewest = bfs(&task, start, &reached, Limits::NONE).into_solution().map(|s| s.len());
        let (to, from) = (|| ToGoal(&task, goal), || FromStart(&task, start));
        let goals = [Subgoal(goal)];

        let results = [
            bidirectional_dijkstra_meet(&task, [start], goals, meet_fn(satisfies), Limits::NONE),
            nbs_meet(&task, [start], goals, Zero, Zero, meet_fn(satisfies), Limits::NONE),
            dibbs_meet(&task, [start], goals, Zero, Zero, meet_fn(satisfies), Limits::NONE),
            bae_star_meet(&task, [start], goals, Zero, Zero, meet_fn(satisfies), Limits::NONE),
            mm_meet(&task, [start], goals, Zero, Zero, meet_fn(satisfies), Limits::NONE),
            nbs_meet(&task, [start], goals, to(), from(), meet_fn(satisfies), Limits::NONE),
            dibbs_meet(&task, [start], goals, to(), from(), meet_fn(satisfies), Limits::NONE),
            bae_star_meet(&task, [start], goals, to(), from(), meet_fn(satisfies), Limits::NONE),
            mm_meet(&task, [start], goals, to(), from(), meet_fn(satisfies), Limits::NONE),
        ];
        for result in results {
            let result = result.expect("delete-free regression replays");
            prop_assert_eq!(result.solution().map(|s| s.cost), optimal);
            if let Some(solution) = result.solution() {
                assert_real_path(&task, start, goal, solution);
            }
        }

        let shortest = bidirectional_bfs_meet(&task, [start], goals, meet_fn(satisfies), Limits::NONE)
            .expect("delete-free regression replays");
        prop_assert_eq!(shortest.solution().map(Solution::len), fewest);
        if let Some(solution) = shortest.solution() {
            assert_real_path(&task, start, goal, solution);
        }
    }

    /// With the equality relation the meet searches agree with Dijkstra on random graphs.
    #[test]
    fn equality_meets_match_dijkstra((graph, goal) in graph_and_goal()) {
        let exact = distances_to(&graph, goal)[0];
        let equal = |a: &usize, b: &usize| a == b;

        let results = [
            bidirectional_dijkstra_meet(&graph, [0], [goal], meet_fn(equal), Limits::NONE),
            nbs_meet(&graph, [0], [goal], Zero, Zero, meet_fn(equal), Limits::NONE),
            dibbs_meet(&graph, [0], [goal], Zero, Zero, meet_fn(equal), Limits::NONE),
            bae_star_meet(&graph, [0], [goal], Zero, Zero, meet_fn(equal), Limits::NONE),
            mm_meet(&graph, [0], [goal], Zero, Zero, meet_fn(equal), Limits::NONE),
        ];
        for result in results {
            let result = result.expect("equality always replays");
            prop_assert_eq!(result.solution().map(|s| s.cost), exact);
            if let Some(solution) = result.solution() {
                assert_valid_path(&graph, 0, solution);
            }
        }
    }
}

/// Regression that claims an action the forward model cannot apply is a typed error.
#[test]
fn unreplayable_regression_is_an_error() {
    /// Forward: 0 leads to 2 by action 3. Backward: node 1 regresses to 0 by action 7.
    struct Broken;

    impl Successors for Broken {
        type State = u8;
        type Action = u8;
        type Cost = u32;

        fn successors(&self, state: &u8, mut emit: impl FnMut(u8, u8, u32)) {
            if *state == 0 {
                emit(3, 2, 1);
            }
        }
    }

    impl Predecessors for Broken {
        type Node = u8;
        type Action = u8;
        type Cost = u32;

        fn predecessors(&self, node: &u8, mut emit: impl FnMut(u8, u8, u32)) {
            if *node == 1 {
                emit(7, 0, 1);
            }
        }
    }

    let equal = |a: &u8, b: &u8| a == b;
    let result = bidirectional_dijkstra_meet(&Broken, [0], [1], meet_fn(equal), Limits::NONE);

    assert_eq!(
        result,
        Err(ReplayError {
            state: 0,
            action: 7
        })
    );
}
