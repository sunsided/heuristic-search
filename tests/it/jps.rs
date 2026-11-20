//! Jump Point Search and the octile cost: JPS against Dijkstra on an independently written
//! 8-connected grid graph with random obstacles (same corner rule), with every segment of
//! the sparse path checked move by move; and the exact octile order against floats.

use heuristic_search::{
    EightConnected, GridCell, GridStep, Limits, Octile, OctileDistance, Outcome, Solution,
    Successors, UniformGrid, a_star, dijkstra, jps, jps_multi,
};
use proptest::prelude::*;

/// A grid of passable (`true`) and blocked cells, row-major by `y`.
#[derive(Debug, Clone)]
pub struct Map {
    /// `open[y][x]`; every row has the same length.
    pub open: Vec<Vec<bool>>,
}

impl Map {
    /// Parses text rows, where `#` is blocked.
    pub fn parse(rows: &[&str]) -> Self {
        let open = rows
            .iter()
            .map(|row| row.chars().map(|c| c != '#').collect())
            .collect();

        Self { open }
    }

    /// Returns `true` if `(x, y)` (possibly off the grid) is on the grid and passable.
    pub fn free(&self, x: i64, y: i64) -> bool {
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return false;
        };

        self.open
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(false)
    }

    /// Returns `true` if the unit move from `(x, y)` by `(dx, dy)` is allowed: the target
    /// is free and, for a diagonal, both orthogonal neighbours are free.
    pub fn allowed(&self, (x, y): (i64, i64), (dx, dy): (i64, i64)) -> bool {
        self.free(x, y)
            && self.free(x + dx, y + dy)
            && (dx == 0 || dy == 0 || (self.free(x + dx, y) && self.free(x, y + dy)))
    }
}

impl UniformGrid for Map {
    fn width(&self) -> usize {
        self.open.first().map_or(0, Vec::len)
    }

    fn height(&self) -> usize {
        self.open.len()
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        self.open[y][x]
    }
}

/// The oracle graph: the 8-connected moves of a [`Map`], written without the crate's grid
/// helpers.
struct Oracle<'a>(&'a Map);

impl Successors for Oracle<'_> {
    type State = GridCell;
    type Action = ();
    type Cost = Octile;

    fn successors(&self, &(x, y): &GridCell, mut emit: impl FnMut((), GridCell, Octile)) {
        let here = (i64::try_from(x).unwrap(), i64::try_from(y).unwrap());
        for dx in -1..=1 {
            for dy in -1..=1 {
                if (dx, dy) == (0, 0) || !self.0.allowed(here, (dx, dy)) {
                    continue;
                }

                let target = (
                    usize::try_from(here.0 + dx).unwrap(),
                    usize::try_from(here.1 + dy).unwrap(),
                );
                let cost = if dx == 0 || dy == 0 {
                    Octile::STRAIGHT
                } else {
                    Octile::DIAGONAL
                };
                emit((), target, cost);
            }
        }
    }
}

/// Walks every segment of a sparse JPS path move by move, checks each move against the
/// corner rule, and returns the summed cost.
fn walk(map: &Map, solution: &Solution<GridCell, GridStep, Octile>) -> Octile {
    let signed = |(x, y): GridCell| (i64::try_from(x).unwrap(), i64::try_from(y).unwrap());
    let mut cost = Octile::new(0, 0);
    for (index, &(dx, dy)) in solution.actions.iter().enumerate() {
        let (mut at, to) = (
            signed(solution.states[index]),
            signed(solution.states[index + 1]),
        );
        let step = (i64::from(dx), i64::from(dy));
        assert!(step != (0, 0), "a segment has a direction");
        while at != to {
            assert!(
                map.allowed(at, step),
                "move from {at:?} by {step:?} breaks the corner rule"
            );
            at = (at.0 + step.0, at.1 + step.1);
            cost = cost
                + if step.0 == 0 || step.1 == 0 {
                    Octile::STRAIGHT
                } else {
                    Octile::DIAGONAL
                };
            assert!(
                at.0.abs_diff(to.0) <= 64 && at.1.abs_diff(to.1) <= 64,
                "segment overshoots"
            );
        }
    }

    cost
}

/// The cheapest cost from `start` to `goal` by Dijkstra on the oracle graph; `None` if
/// either is blocked or off the grid, or the goal is unreachable.
fn oracle(map: &Map, start: GridCell, goal: GridCell) -> Option<Octile> {
    let open = |(x, y): GridCell| map.free(i64::try_from(x).unwrap(), i64::try_from(y).unwrap());
    if !open(start) || !open(goal) {
        return None;
    }

    dijkstra(
        &Oracle(map),
        start,
        &|cell: &GridCell| *cell == goal,
        Limits::NONE,
    )
    .into_solution()
    .map(|solution| solution.cost)
}

/// The wall forces a detour through the gap; JPS, A* on [`EightConnected`] and the oracle
/// agree, and JPS expands fewer nodes than A*.
#[test]
fn detour_through_gap() {
    let map = Map::parse(&[
        "........#.......",
        "........#.......",
        "........#.......",
        "........#.......",
        "................",
        "........#.......",
    ]);
    let (start, goal) = ((0, 0), (15, 0));

    let jump = jps(&map, start, goal, Limits::NONE);
    let solution = jump.solution().expect("the gap connects both sides");
    let plain = a_star(
        &EightConnected(&map),
        start,
        &|c: &GridCell| *c == goal,
        OctileDistance(goal),
        Limits::NONE,
    );

    assert_eq!(Some(solution.cost), oracle(&map, start, goal));
    assert_eq!(plain.solution().map(|s| s.cost), Some(solution.cost));
    assert_eq!(walk(&map, solution), solution.cost);
    assert!(
        jump.stats.expanded < plain.stats.expanded,
        "{:?} vs {:?}",
        jump.stats,
        plain.stats
    );
}

/// Squeezing between two diagonally touching blocked cells is not allowed.
#[test]
fn no_corner_cutting() {
    let map = Map::parse(&[".#", "#."]);

    assert_eq!(
        jps(&map, (0, 0), (1, 1), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
}

/// A zero budget stops at once, a blocked goal or start is unsolvable, and a start that is
/// the goal is free.
#[test]
fn outcomes() {
    let map = Map::parse(&["..#", "..."]);

    assert_eq!(
        jps(&map, (0, 0), (1, 1), Limits::expansions(0)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        jps(&map, (0, 0), (2, 0), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        jps(&map, (2, 0), (0, 0), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        jps(&map, (9, 9), (0, 0), Limits::NONE).outcome,
        Outcome::Unsolvable
    );
    assert_eq!(
        jps(&map, (1, 1), (1, 1), Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(Octile::new(0, 0))
    );
}

/// Pell pairs `x² - 2y² = ±1` are the closest calls between `x` and `y√2`, closer than
/// `f64` resolves at this size; the exact order still gets them right.
#[test]
fn octile_order_resolves_pell_pairs() {
    assert!(Octile::new(768_398_401, 0) > Octile::new(0, 543_339_720));
    assert!(Octile::new(1_855_077_841, 0) < Octile::new(0, 1_311_738_121));
    assert!(Octile::new(u32::MAX, 0) < Octile::new(0, u32::MAX));
    assert!(Octile::new(u32::MAX, 1) < Octile::new(1, u32::MAX - 1_000_000_000));
}

/// A random map up to 14 x 14 with 0 to 45 percent blocked cells, and two cells that may
/// lie off the grid.
pub fn map_and_cells() -> impl Strategy<Value = (Map, GridCell, GridCell)> {
    (1_usize..=14, 1_usize..=14, 0_u32..=45).prop_flat_map(|(width, height, density)| {
        let cells = prop::collection::vec(
            prop::collection::vec((0_u32..100).prop_map(move |roll| roll >= density), width),
            height,
        );
        let cell = (0..width + 1, 0..height + 1);

        (cells, cell.clone(), cell).prop_map(|(open, start, goal)| (Map { open }, start, goal))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// JPS finds exactly the oracle's cost, and its sparse path is a chain of legal moves
    /// that adds up to that cost.
    #[test]
    fn jps_matches_dijkstra((map, start, goal) in map_and_cells()) {
        let result = jps(&map, start, goal, Limits::NONE);
        let expected = oracle(&map, start, goal);

        prop_assert_eq!(result.solution().map(|s| s.cost), expected);
        if let Some(solution) = result.solution() {
            prop_assert_eq!(solution.states[0], start);
            prop_assert_eq!(*solution.goal(), goal);
            prop_assert_eq!(walk(&map, solution), solution.cost);
        }
    }

    /// JPS from several starts matches the best single start.
    #[test]
    fn jps_multi_matches_best_start((map, start, goal) in map_and_cells(), other in (0_usize..15, 0_usize..15)) {
        let best = [start, other].into_iter().filter_map(|s| oracle(&map, s, goal)).min();
        let result = jps_multi(&map, [start, other], goal, Limits::NONE);

        prop_assert_eq!(result.solution().map(|s| s.cost), best);
        if let Some(solution) = result.solution() {
            prop_assert!([start, other].contains(&solution.states[0]));
            prop_assert_eq!(walk(&map, solution), solution.cost);
        }
    }

    /// The exact octile order agrees with floats wherever floats can tell the values
    /// apart, is antisymmetric, and is compatible with addition.
    #[test]
    fn octile_order_is_exact(
        a in (0_u32..2000, 0_u32..2000),
        b in (0_u32..2000, 0_u32..2000),
        c in (0_u32..2000, 0_u32..2000),
    ) {
        let (a, b, c) = (Octile::new(a.0, a.1), Octile::new(b.0, b.1), Octile::new(c.0, c.1));
        let difference = a.to_f64() - b.to_f64();
        if difference.abs() > 1e-9 {
            prop_assert_eq!(a.cmp(&b), difference.partial_cmp(&0.0).unwrap());
        }

        prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        prop_assert_eq!(a == b, (a.straight(), a.diagonal()) == (b.straight(), b.diagonal()));
        prop_assert_eq!((a + c).cmp(&(b + c)), a.cmp(&b));
    }

    /// Counts near `u32::MAX` compare without overflow.
    #[test]
    fn octile_order_handles_large_counts(x in any::<u32>(), y in any::<u32>()) {
        let (a, b) = (Octile::new(x, 0), Octile::new(0, y));
        let difference = f64::from(x) - f64::from(y) * core::f64::consts::SQRT_2;
        if difference.abs() > 1e-3 {
            prop_assert_eq!(a.cmp(&b), difference.partial_cmp(&0.0).unwrap());
        }
    }
}
