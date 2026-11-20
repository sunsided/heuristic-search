//! Path finding on small grids.

use heuristic_search::{Limits, Weight, astar, bfs, dijkstra, gbfs, heuristic_fn, weighted_astar};

use crate::fixtures::{Cell, Grid, manhattan};

/// A grid with a wall that forces a detour; the shortest path from the top-left to the
/// top-right corner has 14 moves.
fn walled() -> Grid {
    Grid::parse(&[
        "....#....",
        "....#....",
        "....#....",
        ".........",
        "....#....",
    ])
}

/// The top-left corner.
const START: Cell = (0, 0);
/// The top-right corner.
const GOAL: Cell = (0, 8);

/// Goal test for [`GOAL`].
fn is_goal(cell: &Cell) -> bool {
    *cell == GOAL
}

/// Manhattan distance to [`GOAL`].
#[expect(
    clippy::unnecessary_wraps,
    reason = "the `Heuristic` closure signature returns `Option`"
)]
fn to_goal(cell: &Cell) -> Option<u32> {
    Some(manhattan(*cell, GOAL))
}

/// BFS, Dijkstra and A* agree on the shortest detour around the wall.
#[test]
fn all_optimal_algorithms_agree() {
    let grid = walled();
    let costs = [
        bfs(&grid, START, &is_goal, Limits::NONE),
        dijkstra(&grid, START, &is_goal, Limits::NONE),
        astar(&grid, START, &is_goal, heuristic_fn(to_goal), Limits::NONE),
    ]
    .map(|result| result.into_solution().expect("goal is reachable").cost);

    assert_eq!(costs, [14; 3]);
}

/// The Manhattan heuristic saves expansions and never causes reopening.
#[test]
fn astar_expands_fewer_states_than_dijkstra() {
    let grid = walled();
    let informed = astar(&grid, START, &is_goal, heuristic_fn(to_goal), Limits::NONE).stats;
    let blind = dijkstra(&grid, START, &is_goal, Limits::NONE).stats;

    assert!(
        informed.expanded < blind.expanded,
        "A* {informed:?} vs Dijkstra {blind:?}"
    );
    assert_eq!(informed.reopened, 0, "Manhattan distance is consistent");
}

/// Weighted A* stays within its bound; GBFS still reaches the goal.
#[test]
fn suboptimal_searches_stay_within_bounds() {
    let grid = walled();
    let weight = Weight::integer(3);
    let weighted = weighted_astar(
        &grid,
        START,
        &is_goal,
        heuristic_fn(to_goal),
        weight,
        Limits::NONE,
    )
    .into_solution()
    .expect("goal is reachable");
    let greedy = gbfs(&grid, START, &is_goal, heuristic_fn(to_goal), Limits::NONE)
        .into_solution()
        .expect("goal is reachable");

    assert!(weighted.cost <= 3 * 14);
    assert!(greedy.cost >= 14);
    assert_eq!(*greedy.goal(), GOAL);
}

/// A walled-off goal is unsolvable after expanding every reachable cell once.
#[test]
fn enclosed_goal_is_unsolvable() {
    let grid = Grid::parse(&["..#.", "..#.", "..#."]);
    let goal = |cell: &Cell| *cell == (0, 3);
    let result = astar(
        &grid,
        START,
        &goal,
        heuristic_fn(|c: &Cell| Some(manhattan(*c, (0, 3)))),
        Limits::NONE,
    );

    assert!(result.solution().is_none());
    assert_eq!(
        result.stats.expanded, 6,
        "every reachable cell is expanded once"
    );
}

/// Heuristic values near `u32::MAX` must not overflow the open-list key, for A* (`g + h`)
/// and for Weighted A* (`den * g + num * h`).
#[test]
fn huge_heuristic_values_do_not_overflow() {
    let grid = walled();
    let mut huge = heuristic_fn(|cell: &Cell| Some(if *cell == GOAL { 0 } else { u32::MAX - 1 }));
    let weight = Weight::new(3, 2);

    let plain = astar(&grid, START, &is_goal, &mut huge, Limits::NONE);
    let weighted = weighted_astar(&grid, START, &is_goal, &mut huge, weight, Limits::NONE);

    assert_eq!(plain.solution().map(|s| *s.goal()), Some(GOAL));
    assert_eq!(weighted.solution().map(|s| *s.goal()), Some(GOAL));
}
