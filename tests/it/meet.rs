//! Bidirectional searches with a meet relation: by equality they match Dijkstra on random
//! graphs, and over subgoals (regression of delete-free STRIPS actions) they match forward
//! Dijkstra to the first state that satisfies the goal.

use heuristic_search::{
    Limits, Predecessors, Successors, Zero, bae_star_meet, bidirectional_dijkstra_meet, dibbs_meet,
    dijkstra, meet_fn, nbs_meet,
};
use proptest::prelude::*;

use crate::fixtures::{assert_valid_path, distances_to, graph_and_goal};

/// Delete-free STRIPS actions over four facts (bits), as `(precondition, add, cost)`.
#[derive(Debug, Clone)]
struct Strips(Vec<(u8, u8, u32)>);

impl Successors for Strips {
    type State = u8;
    type Action = usize;
    type Cost = u32;

    fn successors(&self, state: &u8, mut emit: impl FnMut(usize, u8, u32)) {
        for (index, &(pre, add, cost)) in self.0.iter().enumerate() {
            if state & pre == pre && state | add != *state {
                emit(index, state | add, cost);
            }
        }
    }
}

/// Regression over subgoals: an action that adds part of a subgoal leads back to the
/// subgoal without those facts but with the action's precondition.
impl Predecessors for Strips {
    type Node = u8;
    type Action = usize;
    type Cost = u32;

    fn predecessors(&self, subgoal: &u8, mut emit: impl FnMut(usize, u8, u32)) {
        for (index, &(pre, add, cost)) in self.0.iter().enumerate() {
            if subgoal & add != 0 {
                emit(index, (subgoal & !add) | pre, cost);
            }
        }
    }
}

/// Random delete-free tasks: up to six actions over four facts, a start state and a goal.
fn task() -> impl Strategy<Value = (Strips, u8, u8)> {
    let action = (0_u8..16, 1_u8..16, 0_u32..5);

    (prop::collection::vec(action, 0..7), 0_u8..16, 0_u8..16)
        .prop_map(|(actions, start, goal)| (Strips(actions), start, goal))
}

/// Checks that a solution is a real forward path from `start` to a goal-satisfying state.
fn assert_real_path(
    task: &Strips,
    start: u8,
    goal: u8,
    solution: &heuristic_search::Solution<u8, usize, u32>,
) {
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

proptest! {
    /// Over subgoals, every meet search finds the cheapest plan that forward Dijkstra finds.
    #[test]
    fn subgoal_meets_match_forward_dijkstra((task, start, goal) in task()) {
        let satisfies = |state: &u8, subgoal: &u8| state & subgoal == *subgoal;
        let optimal = dijkstra(&task, start, &|s: &u8| s & goal == goal, Limits::NONE)
            .into_solution()
            .map(|s| s.cost);

        let results = [
            bidirectional_dijkstra_meet(&task, [start], [goal], meet_fn(satisfies), Limits::NONE),
            nbs_meet(&task, [start], [goal], Zero, Zero, meet_fn(satisfies), Limits::NONE),
            dibbs_meet(&task, [start], [goal], Zero, Zero, meet_fn(satisfies), Limits::NONE),
            bae_star_meet(&task, [start], [goal], Zero, Zero, meet_fn(satisfies), Limits::NONE),
        ];
        for result in &results {
            prop_assert_eq!(result.solution().map(|s| s.cost), optimal);
            if let Some(solution) = result.solution() {
                assert_real_path(&task, start, goal, solution);
            }
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
        ];
        for result in &results {
            prop_assert_eq!(result.solution().map(|s| s.cost), exact);
            if let Some(solution) = result.solution() {
                assert_valid_path(&graph, 0, solution);
            }
        }
    }
}
