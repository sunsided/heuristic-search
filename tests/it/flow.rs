//! Flow fields: on random graphs the field's costs equal the exact distance to the nearest
//! goal, its steps are forward edges along cheapest paths, and a budget-cut field answers
//! exactly for the states closest to the goals. On grids, the predecessors of
//! [`EightConnected`] mirror its successors and the field agrees with A*.

use std::collections::BTreeSet;

use heuristic_search::{
    EightConnected, FlowField, GridCell, Limits, OctileDistance, Predecessors, Successors, a_star,
};
use proptest::prelude::*;

use crate::fixtures::{assert_valid_path, distances_to, graph_and_goal};
use crate::jps::{Map, map_and_cells};

proptest! {
    /// Costs are the exact distance to the nearest goal, paths are valid and end in a goal,
    /// and `next` is an edge of the forward graph on a cheapest path.
    #[test]
    fn flow_field_matches_distances(
        (graph, goal) in graph_and_goal(),
        extra in prop::option::of(0_usize..12),
    ) {
        let nodes = graph.edges.len();
        let goals: Vec<usize> = core::iter::once(goal).chain(extra.map(|e| e % nodes)).collect();
        let to_goals: Vec<_> = goals.iter().map(|&g| distances_to(&graph, g)).collect();
        let exact: Vec<Option<u64>> =
            (0..nodes).map(|v| to_goals.iter().filter_map(|d| d[v]).min()).collect();

        let field = FlowField::build(&graph, goals.iter().copied(), Limits::NONE);
        prop_assert!(field.is_complete());
        prop_assert_eq!(field.stats().expanded, exact.iter().flatten().count() as u64);

        for v in 0..nodes {
            prop_assert_eq!(field.cost(&v), exact[v]);

            let path = field.path(&v);
            prop_assert_eq!(path.as_ref().map(|p| p.cost), exact[v]);
            if let Some(path) = &path {
                assert_valid_path(&graph, v, path);
                prop_assert!(goals.contains(path.goal()));
            }

            match field.next(&v) {
                Some((&action, &next, cost)) => {
                    prop_assert_eq!(graph.edges[v][action], (next, cost));
                    prop_assert_eq!(field.cost(&v), field.cost(&next).map(|c| c + cost));
                }
                None => prop_assert!(exact[v].is_none() || goals.contains(&v)),
            }
        }
    }

    /// Stopped after `limit` expansions, the field answers for exactly the states it
    /// expanded, with exact costs, and those are the states closest to the goal.
    #[test]
    fn budget_cut_field_answers_only_settled_states(
        (graph, goal) in graph_and_goal(),
        limit in 0_u64..6,
    ) {
        let exact = distances_to(&graph, goal);
        let field = FlowField::build(&graph, [goal], Limits::expansions(limit));

        let answered: Vec<usize> = (0..graph.edges.len()).filter(|v| field.cost(v).is_some()).collect();
        prop_assert_eq!(answered.len() as u64, field.stats().expanded);
        prop_assert!(field.stats().expanded <= limit);
        if !field.is_complete() {
            prop_assert_eq!(field.stats().expanded, limit);
        }

        let farthest = answered.iter().filter_map(|&v| exact[v]).max();
        for (v, &distance) in exact.iter().enumerate() {
            if answered.contains(&v) {
                prop_assert_eq!(field.cost(&v), distance);
                let path = field.path(&v).expect("answered states have a path");
                assert_valid_path(&graph, v, &path);
            } else {
                prop_assert_eq!(field.next(&v), None);
                prop_assert!(field.path(&v).is_none());
                if let (Some(d), Some(farthest)) = (distance, farthest) {
                    prop_assert!(d >= farthest, "{v} at {d} is closer than a settled state");
                }
            }
        }
    }

    /// Every predecessor edge of a cell is a successor edge of its source, and the counts
    /// agree, so the two traits describe the same edges.
    #[test]
    fn grid_predecessors_mirror_successors((map, _, _) in map_and_cells()) {
        let grid = EightConnected(&map);
        let mut forward = BTreeSet::new();
        let mut backward = BTreeSet::new();
        for y in 0..map.open.len() {
            for x in 0..map.open[0].len() {
                grid.successors(&(x, y), |step, to, cost| {
                    forward.insert(((x, y), step, to, cost));
                });
                grid.predecessors(&(x, y), |step, from, cost| {
                    backward.insert((from, step, (x, y), cost));
                });
            }
        }

        prop_assert_eq!(forward, backward);
    }

    /// On a grid, the field's cost from every cell equals the A* cost to the goal.
    #[test]
    fn grid_field_matches_a_star((map, _, goal) in map_and_cells()) {
        let grid = EightConnected(&map);
        let field = FlowField::build(&grid, [goal], Limits::NONE);

        for y in 0..map.open.len() {
            for x in 0..map.open[0].len() {
                let start: GridCell = (x, y);
                let expected = a_star(&grid, start, &|c: &GridCell| *c == goal, OctileDistance(goal), Limits::NONE)
                    .solution()
                    .map(|s| s.cost);
                prop_assert_eq!(field.cost(&start), expected);
            }
        }
    }
}

/// A field from a blocked goal, an empty goal set, and repeated goals.
#[test]
fn degenerate_goal_sets() {
    let map = Map::parse(&["..#", "..."]);
    let grid = EightConnected(&map);

    let blocked = FlowField::build(&grid, [(2, 0)], Limits::NONE);
    assert_eq!(
        blocked.cost(&(2, 0)),
        Some(heuristic_search::Octile::new(0, 0))
    );
    assert_eq!(
        blocked.cost(&(1, 0)),
        None,
        "a blocked goal has no predecessors"
    );

    let empty = FlowField::build(&grid, [], Limits::NONE);
    assert!(empty.is_complete());
    assert_eq!(empty.cost(&(0, 0)), None);

    let repeated = FlowField::build(&grid, [(0, 0), (0, 0)], Limits::NONE);
    assert_eq!(repeated.path(&(0, 0)).map(|p| p.states), Some(vec![(0, 0)]));
}
