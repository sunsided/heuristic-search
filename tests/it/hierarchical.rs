//! Hierarchical A*: on random grids the HPA* abstraction finds a path exactly when the
//! oracle does, never cheaper; with every border cell as an entrance the path is optimal;
//! every path replays on the grid; generated floor plans (rooms joined by doors) bound the
//! average suboptimality; the budget bounds refinement too; and the edge cases of start,
//! goal, starts and budget.

use heuristic_search::{
    EightConnected, GridCell, GridStep, Hierarchy, Limits, Octile, Outcome, Solution, Successors,
    UniformGrid, hierarchical_a_star, hierarchical_a_star_multi,
};
use proptest::prelude::*;

use crate::fixtures::{assert_valid_path, distances_to, graph_and_goal};
use crate::jps::{Map, map_and_cells, oracle};

/// The admissible octile estimate between two cells.
fn distance(a: &GridCell, b: &GridCell) -> Octile {
    Octile::between(*a, *b)
}

/// Every cell of `map`.
fn cells(map: &Map) -> impl Iterator<Item = GridCell> + use<> {
    let (width, height) = (map.width(), map.height());

    (0..width).flat_map(move |x| (0..height).map(move |y| (x, y)))
}

/// The square clusters of side `size` as a region map, numbered by `(x / size, y / size)`.
fn clusters(size: usize) -> impl Fn(&GridCell) -> usize + Copy {
    move |&(x, y): &GridCell| (y / size) * 1000 + x / size
}

/// Asserts that `solution` runs from `start` to `goal` along edges of the 8-connected
/// grid, and that its cost is the sum of those edges.
fn assert_replays(
    map: &Map,
    start: GridCell,
    goal: GridCell,
    solution: &Solution<GridCell, GridStep, Octile>,
) {
    assert_eq!(
        solution.states.first(),
        Some(&start),
        "path begins at the start"
    );
    assert_eq!(solution.states.last(), Some(&goal), "path ends at the goal");
    assert_eq!(solution.states.len(), solution.actions.len() + 1);

    let mut cost = Octile::new(0, 0);
    for (index, action) in solution.actions.iter().enumerate() {
        let mut edge = None;
        EightConnected(map).successors(&solution.states[index], |step, next, c| {
            if step == *action && next == solution.states[index + 1] {
                edge = Some(c);
            }
        });
        cost = cost + edge.unwrap_or_else(|| panic!("step {index} is not a grid edge"));
    }

    assert_eq!(cost, solution.cost, "reported cost matches the path");
}

/// A floor plan of `rooms` by `rooms` square rooms of `side` cells, one-cell walls between
/// them, and a door of width 1 or 2 in most walls between neighbouring rooms, placed by a
/// linear congruential generator seeded with `seed`. Rooms may be cut off.
fn floor_plan(rooms: usize, side: usize, seed: u64) -> Map {
    let extent = rooms * (side + 1) + 1;
    let mut state = seed;
    let mut next = |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(state >> 33).unwrap() % bound
    };
    let wall = |i: usize| i % (side + 1) == 0;
    let mut open: Vec<Vec<bool>> = (0..extent)
        .map(|y| (0..extent).map(|x| !wall(x) && !wall(y)).collect())
        .collect();

    for a in 0..rooms {
        for b in 1..rooms {
            for vertical in [true, false] {
                if next(8) == 0 {
                    continue;
                }

                let line = b * (side + 1);
                let offset = a * (side + 1) + 1 + next(side - 1);
                for along in offset..offset + 1 + next(2) {
                    let (x, y) = if vertical {
                        (line, along)
                    } else {
                        (along, line)
                    };
                    open[y][x] = true;
                }
            }
        }
    }

    Map { open }
}

proptest! {
    /// The HPA* grid abstraction is complete and never beats the oracle; paths replay.
    #[test]
    fn grid_hierarchy_is_complete((map, start, goal) in map_and_cells(), size in 1_usize..6) {
        let hierarchy = Hierarchy::grid(&map, size);
        let result = hierarchical_a_star(&EightConnected(&map), &hierarchy, start, goal, distance, Limits::NONE);
        let expected = if start == goal { Some(Octile::new(0, 0)) } else { oracle(&map, start, goal) };

        prop_assert_eq!(result.solution().is_some(), expected.is_some(), "solvability");
        if let (Some(solution), Some(optimal)) = (result.solution(), expected) {
            assert_replays(&map, start, goal, solution);
            prop_assert!(solution.cost >= optimal);
        }
    }

    /// With every cell offered to [`Hierarchy::new`], every border crossing is an abstract
    /// edge and the path is optimal.
    #[test]
    fn full_hierarchy_is_optimal((map, start, goal) in map_and_cells(), size in 1_usize..6) {
        let graph = EightConnected(&map);
        let hierarchy = Hierarchy::new(&graph, clusters(size), cells(&map), distance);
        let result = hierarchical_a_star(&graph, &hierarchy, start, goal, distance, Limits::NONE);
        let expected = if start == goal { Some(Octile::new(0, 0)) } else { oracle(&map, start, goal) };

        prop_assert_eq!(result.solution().map(|s| s.cost), expected);
        if let Some(solution) = result.solution() {
            assert_replays(&map, start, goal, solution);
        }
    }

    /// On random directed graphs with zero-cost edges and three regions, the hierarchy of
    /// [`Hierarchy::new`] finds a cheapest path from every node, or none exactly when the
    /// goal is unreachable.
    #[test]
    fn full_hierarchy_on_graphs((graph, goal) in graph_and_goal()) {
        let nodes = graph.edges.len();
        let region = |s: &usize| s % 3;
        let blind = |_: &usize, _: &usize| 0_u64;
        let hierarchy = Hierarchy::new(&graph, region, 0..nodes, blind);
        let exact = distances_to(&graph, goal);

        for (start, &expected) in exact.iter().enumerate() {
            let result = hierarchical_a_star(&graph, &hierarchy, start, goal, blind, Limits::NONE);
            prop_assert_eq!(result.solution().map(|s| s.cost), expected, "from {}", start);
            if let Some(solution) = result.solution() {
                assert_valid_path(&graph, start, solution);
            }
        }
    }

    /// Several starts: the path begins at one of them and costs no more than the best
    /// single-start query.
    #[test]
    fn multi_matches_best_start((map, start, goal) in map_and_cells(), other in (0_usize..15, 0_usize..15)) {
        let hierarchy = Hierarchy::grid(&map, 3);
        let graph = EightConnected(&map);
        let single = |s: GridCell| {
            hierarchical_a_star(&graph, &hierarchy, s, goal, distance, Limits::NONE)
                .into_solution()
                .map(|solution| solution.cost)
        };
        let best = [start, other].into_iter().filter_map(single).min();
        let result = hierarchical_a_star_multi(&graph, &hierarchy, [start, other], goal, distance, Limits::NONE);

        prop_assert_eq!(result.solution().map(|s| s.cost), best);
        if let Some(solution) = result.solution() {
            assert_replays(&map, solution.states[0], goal, solution);
            prop_assert!(solution.states[0] == start || solution.states[0] == other);
        }
    }
}

/// On generated floor plans, room regions give optimal paths, and the HPA* clusters stay
/// complete with an average cost at most 10 % above the optimum over many queries.
#[test]
fn floor_plans() {
    let mut ratio_sum = 0.0;
    let mut queries = 0_u32;
    for seed in 0..6 {
        let map = floor_plan(4, 7, seed);
        let graph = EightConnected(&map);
        let room = |&(x, y): &GridCell| (y / 8).min(3) * 4 + (x / 8).min(3);
        let rooms = Hierarchy::new(&graph, room, cells(&map), distance);
        let grid = Hierarchy::grid(&map, 8);
        let open: Vec<GridCell> = cells(&map).filter(|&(x, y)| map.open[y][x]).collect();

        for (index, &start) in open.iter().enumerate().step_by(37) {
            let goal = open[(index * 7919 + 13) % open.len()];
            let expected = if start == goal {
                Some(Octile::new(0, 0))
            } else {
                oracle(&map, start, goal)
            };

            let by_room = hierarchical_a_star(&graph, &rooms, start, goal, distance, Limits::NONE);
            assert_eq!(
                by_room.solution().map(|s| s.cost),
                expected,
                "rooms are exact"
            );

            let by_cluster =
                hierarchical_a_star(&graph, &grid, start, goal, distance, Limits::NONE);
            assert_eq!(
                by_cluster.solution().is_some(),
                expected.is_some(),
                "clusters are complete"
            );
            if let (Some(solution), Some(optimal)) = (by_cluster.solution(), expected) {
                assert_replays(&map, start, goal, solution);
                if optimal > Octile::new(0, 0) {
                    ratio_sum += solution.cost.to_f64() / optimal.to_f64();
                    queries += 1;
                }
            }
        }
    }

    assert!(queries > 50, "enough solvable queries: {queries}");
    let mean = ratio_sum / f64::from(queries);
    assert!((1.0..=1.1).contains(&mean), "mean suboptimality {mean}");
}

/// Two 4 x 8 halves joined by a gap at `(4, 2)`, with `(7, 6)` walled in.
fn halves() -> Map {
    Map::parse(&[
        "....#...", "....#...", "........", "....#...", "....#...", "....#..#", "....#.#.",
        "....#..#",
    ])
}

/// A start that is the goal is free; a start or goal on an entrance and a start and goal
/// in one cluster refine without repeated states.
#[test]
fn edge_cases() {
    let map = halves();
    let graph = EightConnected(&map);
    let hierarchy = Hierarchy::grid(&map, 4);
    let query = |start: GridCell, goal: GridCell| {
        hierarchical_a_star(&graph, &hierarchy, start, goal, distance, Limits::NONE)
    };

    let same = query((1, 1), (1, 1));
    assert_eq!(
        same.solution().map(|s| s.states.clone()),
        Some(vec![(1, 1)])
    );
    assert_eq!(same.stats.expanded, 0);

    for (start, goal) in [
        ((3, 2), (4, 2)),
        ((4, 2), (3, 2)),
        ((0, 0), (2, 3)),
        ((0, 0), (7, 0)),
    ] {
        let solution = query(start, goal).into_solution().expect("reachable");
        assert_replays(&map, start, goal, &solution);
        assert_eq!(
            Some(solution.cost),
            oracle(&map, start, goal),
            "{start:?} -> {goal:?}"
        );
        let mut seen = solution.states.clone();
        seen.dedup();
        assert_eq!(seen, solution.states, "no repeated states");
    }
}

/// Repeated starts change nothing; empty starts find nothing; an unreachable goal is
/// unsolvable; a zero budget stops.
#[test]
fn outcomes() {
    let map = halves();
    let graph = EightConnected(&map);
    let hierarchy = Hierarchy::grid(&map, 4);
    let query = |start: GridCell, goal: GridCell| {
        hierarchical_a_star(&graph, &hierarchy, start, goal, distance, Limits::NONE)
    };

    let repeated = hierarchical_a_star_multi(
        &graph,
        &hierarchy,
        [(0, 0), (0, 0)],
        (7, 0),
        distance,
        Limits::NONE,
    );
    assert_eq!(
        repeated.outcome,
        query((0, 0), (7, 0)).outcome,
        "repeated starts"
    );

    assert_eq!(query((0, 0), (7, 6)).outcome, Outcome::Unsolvable);
    assert_eq!(
        hierarchical_a_star_multi(&graph, &hierarchy, [], (0, 0), distance, Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        hierarchical_a_star(
            &graph,
            &hierarchy,
            (0, 0),
            (7, 0),
            distance,
            Limits::expansions(0)
        )
        .outcome,
        Outcome::LimitReached
    );
}

/// An expansion budget bounds the whole query, refinement included: a query that fits is
/// unchanged, and one that does not stops after exactly as many expansions as allowed.
#[test]
fn budget_covers_refinement() {
    let map = floor_plan(6, 6, 3);
    let graph = EightConnected(&map);
    let hierarchy = Hierarchy::grid(&map, 7);
    let (start, goal) = ((1, 1), (40, 40));
    let query = |budget| hierarchical_a_star(&graph, &hierarchy, start, goal, distance, budget);

    let unlimited = query(Limits::NONE);
    let total = unlimited.stats.expanded;
    assert_eq!(query(Limits::expansions(total)), unlimited);
    // The last expansions are refinement, so `total - 1` stops inside the last segment.
    for limit in [total - 1, total - 50, total / 2] {
        let stopped = query(Limits::expansions(limit));
        assert_eq!(stopped.outcome, Outcome::LimitReached, "limit {limit}");
        assert_eq!(stopped.stats.expanded, limit);
    }
}

/// A cluster size of zero is rejected.
#[test]
#[should_panic(expected = "clusters need at least one cell")]
fn zero_cluster_size_panics() {
    let _ = Hierarchy::grid(&Map::parse(&["..."]), 0);
}

/// Parallel and sequential refinement return the same result, statistics included.
#[cfg(feature = "parallel")]
mod parallel {
    use heuristic_search::{
        EightConnected, Hierarchy, Limits, hierarchical_a_star, hierarchical_a_star_multi,
        hierarchical_a_star_par, hierarchical_a_star_par_multi,
    };
    use proptest::prelude::*;

    use super::{distance, floor_plan};
    use crate::jps::map_and_cells;

    proptest! {
        /// Random grids, one start and two starts.
        #[test]
        fn parallel_matches_sequential((map, start, goal) in map_and_cells(), size in 1_usize..5) {
            let graph = EightConnected(&map);
            let hierarchy = Hierarchy::grid(&map, size);

            prop_assert_eq!(
                hierarchical_a_star_par(&graph, &hierarchy, start, goal, distance, Limits::NONE),
                hierarchical_a_star(&graph, &hierarchy, start, goal, distance, Limits::NONE)
            );
            prop_assert_eq!(
                hierarchical_a_star_par_multi(&graph, &hierarchy, [start, goal], (0, 0), distance, Limits::NONE),
                hierarchical_a_star_multi(&graph, &hierarchy, [start, goal], (0, 0), distance, Limits::NONE)
            );
        }
    }

    /// On the threaded path a budget that runs out during refinement stops the query, and
    /// one that fits leaves it unchanged.
    #[test]
    fn floor_plan_budget() {
        let map = floor_plan(6, 6, 3);
        let graph = EightConnected(&map);
        let hierarchy = Hierarchy::grid(&map, 7);
        let (start, goal) = ((1, 1), (40, 40));
        let query =
            |budget| hierarchical_a_star_par(&graph, &hierarchy, start, goal, distance, budget);

        let unlimited = query(Limits::NONE);
        let total = unlimited.stats.expanded;
        assert_eq!(query(Limits::expansions(total)), unlimited);
        let stopped = query(Limits::expansions(total / 2));
        assert_eq!(stopped.outcome, heuristic_search::Outcome::LimitReached);
        assert!(stopped.stats.expanded >= total / 2);
    }

    /// A long path across a floor plan, with many segments to share among threads.
    #[test]
    fn floor_plan_matches() {
        let map = floor_plan(6, 6, 3);
        let graph = EightConnected(&map);
        let hierarchy = Hierarchy::grid(&map, 7);
        let (start, goal) = ((1, 1), (40, 40));

        let sequential =
            hierarchical_a_star(&graph, &hierarchy, start, goal, distance, Limits::NONE);
        assert!(sequential.solution().is_some_and(|s| s.len() > 40));
        assert_eq!(
            hierarchical_a_star_par(&graph, &hierarchy, start, goal, distance, Limits::NONE),
            sequential
        );
    }
}
