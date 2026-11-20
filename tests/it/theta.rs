//! Theta* and Lazy Theta*: line of sight against an exact segment-versus-square oracle,
//! and on random maps, any-angle paths that are no longer than the cheapest 8-connected
//! path, no shorter than the straight line, and made of segments with line of sight.

use heuristic_search::{
    GridCell, Limits, LineOfSight, Outcome, UniformGrid, lazy_theta_star, theta_star,
    theta_star_multi,
};
use proptest::prelude::*;

use crate::jps::{Map, map_and_cells, oracle};

impl LineOfSight for Map {}

/// Tolerance for float comparisons of path lengths on these small maps.
const EPSILON: f64 = 1e-9;

/// Returns `true` if the closed segment between the centres of `a` and `b` touches the
/// closed unit square of cell `c`, in integers scaled by two: no corner of the square lies
/// strictly on one side of the line together with all the others, and the bounding boxes
/// overlap.
fn touches(a: GridCell, b: GridCell, c: GridCell) -> bool {
    let scaled = |v: usize| i64::try_from(v).unwrap() * 2;
    let (px, py) = (scaled(a.0) + 1, scaled(a.1) + 1);
    let (qx, qy) = (scaled(b.0) + 1, scaled(b.1) + 1);
    let (left, bottom) = (scaled(c.0), scaled(c.1));
    let (right, top) = (left + 2, bottom + 2);

    let boxes_overlap =
        px.min(qx) <= right && px.max(qx) >= left && py.min(qy) <= top && py.max(qy) >= bottom;
    let side = |x: i64, y: i64| ((qx - px) * (y - py) - (qy - py) * (x - px)).signum();
    let sides = [
        side(left, bottom),
        side(left, top),
        side(right, bottom),
        side(right, top),
    ];

    boxes_overlap && !(sides.iter().all(|&s| s > 0) || sides.iter().all(|&s| s < 0))
}

/// Line of sight by brute force: both ends open, and no blocked cell touched.
fn sees(map: &Map, a: GridCell, b: GridCell) -> bool {
    let open = |(x, y): GridCell| map.free(i64::try_from(x).unwrap(), i64::try_from(y).unwrap());
    let blocked_touched = (0..map.height())
        .flat_map(|y| (0..map.width()).map(move |x| (x, y)))
        .any(|cell| !open(cell) && touches(a, b, cell));

    open(a) && open(b) && !blocked_touched
}

/// The Euclidean distance between two cell centres, with `std`.
fn euclidean(a: GridCell, b: GridCell) -> f64 {
    let d = |p: usize, q: usize| f64::from(u32::try_from(p.abs_diff(q)).unwrap());

    d(a.0, b.0).hypot(d(a.1, b.1))
}

/// Line of sight around a single pillar: blocked through it and through its corners,
/// free past it.
#[test]
fn line_of_sight_around_a_pillar() {
    let map = Map::parse(&["....", "....", "..#.", "...."]);

    assert!(!map.line_of_sight((0, 0), (3, 3)));
    assert!(
        !map.line_of_sight((0, 3), (3, 0)),
        "touches the pillar's corner"
    );
    assert!(map.line_of_sight((0, 0), (3, 1)));
    assert!(map.line_of_sight((1, 1), (1, 1)));
    assert!(
        !map.line_of_sight((2, 2), (2, 2)),
        "a blocked cell sees nothing"
    );
}

/// On an open grid both searches return the straight segment, at the exact length.
#[test]
fn open_grid_is_one_segment() {
    let map = Map::parse(&["........"; 6]);
    for search in [theta_star::<Map>, lazy_theta_star::<Map>] {
        let solution = search(&map, (0, 5), (7, 1), Limits::NONE)
            .into_solution()
            .unwrap();

        assert_eq!(solution.states, [(0, 5), (7, 1)]);
        assert!((solution.cost.get() - 65_f64.sqrt()).abs() < EPSILON);
    }
}

/// A zero budget stops at once, blocked or off-grid ends are unsolvable, a start that is
/// the goal is free, and the nearer start wins.
#[test]
fn outcomes() {
    let map = Map::parse(&["..#", "..."]);

    assert_eq!(
        theta_star(&map, (0, 0), (1, 1), Limits::expansions(0)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        theta_star(&map, (0, 0), (2, 0), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        lazy_theta_star(&map, (2, 0), (0, 0), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        theta_star(&map, (5, 5), (0, 0), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        theta_star(&map, (1, 1), (1, 1), Limits::NONE)
            .solution()
            .map(|s| s.cost.get()),
        Some(0.0)
    );
    assert_eq!(
        theta_star_multi(&map, [(0, 0), (2, 1)], (1, 1), Limits::NONE)
            .solution()
            .map(|s| s.states.clone()),
        Some(vec![(2, 1), (1, 1)])
    );
}

/// Checks an any-angle result against the 8-connected oracle: solvable exactly when the
/// oracle is, every segment has line of sight, the cost is the sum of the segments and
/// lies between the straight line and (if `bounded`) the 8-connected optimum.
fn check(
    map: &Map,
    start: GridCell,
    goal: GridCell,
    result: &heuristic_search::AnyAngleResult,
    bounded: bool,
) {
    let grid_optimum = oracle(map, start, goal);
    let solution = result.solution();
    assert_eq!(solution.is_some(), grid_optimum.is_some(), "solvability");
    let (Some(solution), Some(grid_optimum)) = (solution, grid_optimum) else {
        return;
    };

    assert_eq!(solution.states.first(), Some(&start));
    assert_eq!(solution.goal(), &goal);
    let mut length = 0.0;
    for pair in solution.states.windows(2) {
        assert!(sees(map, pair[0], pair[1]), "segment {pair:?} is blocked");
        length += euclidean(pair[0], pair[1]);
    }

    let cost = solution.cost.get();
    assert!(
        (cost - length).abs() < EPSILON,
        "cost {cost} vs segments {length}"
    );
    assert!(cost >= euclidean(start, goal) - EPSILON);
    if bounded {
        assert!(
            cost <= grid_optimum.to_f64() + EPSILON,
            "{cost} > {grid_optimum}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The supercover walk agrees with the brute-force segment test.
    #[test]
    fn line_of_sight_matches_geometry((map, a, b) in map_and_cells()) {
        let on = |(x, y): GridCell| (x % map.width(), y % map.height());
        let (a, b) = (on(a), on(b));

        prop_assert_eq!(map.line_of_sight(a, b), sees(&map, a, b));
    }

    /// Theta* paths are valid and never longer than the cheapest 8-connected path.
    #[test]
    fn theta_star_is_bounded_by_grid_optimum((map, start, goal) in map_and_cells()) {
        check(&map, start, goal, &theta_star(&map, start, goal, Limits::NONE), true);
    }

    /// Lazy Theta* paths are valid and never shorter than the straight line.
    #[test]
    fn lazy_theta_star_is_valid((map, start, goal) in map_and_cells()) {
        check(&map, start, goal, &lazy_theta_star(&map, start, goal, Limits::NONE), false);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The `no_std` square root agrees with `std` to a few units in the last place.
    #[test]
    fn lengths_match_std(x in 0_usize..300, y in 0_usize..300) {
        let map = Map { open: vec![vec![true; 300]; 300] };
        let cost = theta_star(&map, (0, 0), (x, y), Limits::NONE).into_solution().unwrap().cost.get();
        let exact = euclidean((0, 0), (x, y));

        prop_assert!((cost - exact).abs() <= 4.0 * f64::EPSILON * exact.max(1.0));
    }
}
