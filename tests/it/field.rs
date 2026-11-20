//! Field D* on random cost grids: the interpolated cost against an independent fixed-point
//! sweep of the same closed form and against the cheapest 8-connected corner path, the
//! returned path's cost against a recomputation from its segments, any-angle lengths on
//! uniform grids against the straight line and Theta*, and incremental repair against
//! fresh runs.

use std::collections::BinaryHeap;

use heuristic_search::{
    Cost, CostGrid, FieldDStar, GridCell, Limits, LineOfSight, NonNegF64, Outcome, UniformGrid,
    theta_star,
};
use proptest::prelude::*;

/// Relative tolerance for float comparisons of costs.
const EPSILON: f64 = 1e-9;

/// A grid of cell costs, row-major by `y`; `None` is blocked.
#[derive(Debug, Clone)]
struct Costs {
    /// `cells[y][x]`; every row has the same length.
    cells: Vec<Vec<Option<f64>>>,
}

impl Costs {
    /// The cost of cell `(x, y)`, possibly off the grid: infinite if blocked or off it.
    fn at(&self, x: i64, y: i64) -> f64 {
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return f64::INFINITY;
        };

        self.cells
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .flatten()
            .unwrap_or(f64::INFINITY)
    }

    /// The number of corners along `x` and `y`.
    fn lattice(&self) -> (usize, usize) {
        (self.width() + 1, self.height() + 1)
    }

    /// Every corner of the lattice.
    fn corners(&self) -> impl Iterator<Item = GridCell> {
        let (columns, rows) = self.lattice();

        (0..rows).flat_map(move |y| (0..columns).map(move |x| (x, y)))
    }

    /// The dense index of a corner.
    fn index(&self, (x, y): GridCell) -> usize {
        y * self.lattice().0 + x
    }
}

impl UniformGrid for Costs {
    fn width(&self) -> usize {
        self.cells.first().map_or(0, Vec::len)
    }

    fn height(&self) -> usize {
        self.cells.len()
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        self.cells[y][x].is_some()
    }
}

impl CostGrid for Costs {
    fn cell_cost(&self, x: usize, y: usize) -> NonNegF64 {
        NonNegF64::new(self.cells[y][x].expect("only called for passable cells")).unwrap()
    }
}

/// A corner as signed coordinates.
fn signed((x, y): GridCell) -> (i64, i64) {
    (i64::try_from(x).unwrap(), i64::try_from(y).unwrap())
}

/// The corner at signed coordinates, if it lies on the lattice of `costs`.
fn corner(costs: &Costs, x: i64, y: i64) -> Option<GridCell> {
    let (columns, rows) = costs.lattice();
    let (x, y) = (usize::try_from(x).ok()?, usize::try_from(y).ok()?);

    (x < columns && y < rows).then_some((x, y))
}

/// The eight corner steps in counterclockwise order from east.
const RING: [(i64, i64); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// The cost of the cell between corner `s` and its diagonal neighbour `s + d`.
fn diagonal_cell(costs: &Costs, (x, y): (i64, i64), (dx, dy): (i64, i64)) -> f64 {
    costs.at(x + dx.min(0), y + dy.min(0))
}

/// The cost of the cheaper cell along the straight edge from `s` by `d`.
fn edge_cost(costs: &Costs, s: (i64, i64), (dx, dy): (i64, i64)) -> f64 {
    let (left, right) = if dx == 0 {
        ((-1, dy), (1, dy))
    } else {
        ((dx, -1), (dx, 1))
    };

    diagonal_cell(costs, s, left).min(diagonal_cell(costs, s, right))
}

/// The cheapest 8-connected path over the corners from `start` to `goal` by Dijkstra:
/// straight moves at the cheaper bordering cell's cost, diagonal moves at `√2` times the
/// crossed cell's cost.
fn eight_connected(costs: &Costs, start: GridCell, goal: GridCell) -> f64 {
    /// A queue entry, ordered by smallest distance first.
    #[derive(PartialEq)]
    struct Entry(f64, GridCell);

    impl Eq for Entry {}

    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.0.total_cmp(&self.0)
        }
    }

    let mut distance = vec![f64::INFINITY; costs.corners().count()];
    let mut queue = BinaryHeap::from([Entry(0.0, start)]);
    distance[costs.index(start)] = 0.0;
    while let Some(Entry(d, at)) = queue.pop() {
        if at == goal {
            return d;
        }

        if d > distance[costs.index(at)] {
            continue;
        }

        let s = signed(at);
        for step in RING {
            let Some(next) = corner(costs, s.0 + step.0, s.1 + step.1) else {
                continue;
            };

            let cost = if step.0 == 0 || step.1 == 0 {
                edge_cost(costs, s, step)
            } else {
                diagonal_cell(costs, s, step) * std::f64::consts::SQRT_2
            };
            let candidate = d + cost;
            if candidate < distance[costs.index(next)] {
                distance[costs.index(next)] = candidate;
                queue.push(Entry(candidate, next));
            }
        }
    }

    f64::INFINITY
}

/// Field D*'s closed form for one corner pair, written from the paper: `g1` at the
/// straight neighbour, `g2` at the diagonal one, `c` the cell of the three corners, `b`
/// the other cell along the straight edge.
fn pair_cost(g1: f64, g2: f64, c: f64, b: f64) -> f64 {
    if c.min(b).is_infinite() {
        return f64::INFINITY;
    }

    if g1 <= g2 {
        return c.min(b) + g1;
    }

    if c.is_infinite() {
        return b + g1;
    }

    let f = g1 - g2;
    if f <= b {
        if c <= f {
            return c * std::f64::consts::SQRT_2 + g2;
        }

        let y = (f / (c * c - f * f).sqrt()).min(1.0);
        return c * (1.0 + y * y).sqrt() + f * (1.0 - y) + g2;
    }

    if c <= b {
        return c * std::f64::consts::SQRT_2 + g2;
    }

    let x = 1.0 - (b / (c * c - b * b).sqrt()).min(1.0);
    c * (1.0 + (1.0 - x) * (1.0 - x)).sqrt() + b * x + g2
}

/// Field D*'s values by brute force: sweeps `g(s) = min over pairs` over every corner
/// until nothing changes, from `g(goal) = 0` and infinity elsewhere.
fn fixed_point(costs: &Costs, goal: GridCell) -> Vec<f64> {
    let mut g = vec![f64::INFINITY; costs.corners().count()];
    g[costs.index(goal)] = 0.0;
    let value = |g: &[f64], x: i64, y: i64| {
        corner(costs, x, y).map_or(f64::INFINITY, |c| g[costs.index(c)])
    };

    loop {
        let mut changed = false;
        for at in costs.corners().filter(|&c| c != goal) {
            let s = signed(at);
            let best = (0..8)
                .map(|i| {
                    let (a, b) = (RING[i], RING[(i + 1) % 8]);
                    let (straight, diagonal) = if a.0 == 0 || a.1 == 0 { (a, b) } else { (b, a) };
                    let mirrored = if straight.0 == 0 {
                        (-diagonal.0, diagonal.1)
                    } else {
                        (diagonal.0, -diagonal.1)
                    };
                    pair_cost(
                        value(&g, s.0 + straight.0, s.1 + straight.1),
                        value(&g, s.0 + diagonal.0, s.1 + diagonal.1),
                        diagonal_cell(costs, s, diagonal),
                        diagonal_cell(costs, s, mirrored),
                    )
                })
                .fold(f64::INFINITY, f64::min);
            if best < g[costs.index(at)] * (1.0 - 1e-15) {
                g[costs.index(at)] = best;
                changed = true;
            }
        }

        if !changed {
            return g;
        }
    }
}

/// The cost of a path of points recomputed from its geometry: each segment must lie in
/// one closed cell; it costs its length times that cell's cost, or, along a cell edge,
/// times the cheaper cell on either side. Returns `None` if a segment does not lie in one
/// closed cell.
fn recompute(costs: &Costs, points: &[(f64, f64)]) -> Option<f64> {
    let mut total = 0.0;
    for pair in points.windows(2) {
        let ((ax, ay), (bx, by)) = (pair[0], pair[1]);
        let (mx, my) = (ax.midpoint(bx), ay.midpoint(by));
        let length = (bx - ax).hypot(by - ay);
        if length == 0.0 {
            continue;
        }

        // The cells whose closed square holds the midpoint; the segment must lie in one.
        let cells_x: Vec<i64> = cover(mx);
        let cells_y: Vec<i64> = cover(my);
        let inside = |cx: i64, cy: i64| {
            let within = |v: f64, c: i64| {
                let c = f64::from(i32::try_from(c).unwrap());
                v >= c - 1e-9 && v <= c + 1.0 + 1e-9
            };
            within(ax, cx) && within(bx, cx) && within(ay, cy) && within(by, cy)
        };
        let rate = cells_x
            .iter()
            .flat_map(|&cx| cells_y.iter().map(move |&cy| (cx, cy)))
            .map(|(cx, cy)| {
                if inside(cx, cy) {
                    Some(costs.at(cx, cy))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if rate.iter().all(Option::is_none) {
            return None;
        }

        total += length * rate.into_iter().flatten().fold(f64::INFINITY, f64::min);
    }

    Some(total)
}

/// The cells along one axis whose closed unit interval holds `v`: one inside a cell, two
/// on a cell border.
fn cover(v: f64) -> Vec<i64> {
    let floor = v.floor();
    #[expect(clippy::cast_possible_truncation, reason = "small grids")]
    let cell = floor as i64;
    if (v - floor).abs() < 1e-9 {
        vec![cell - 1, cell]
    } else if (v - floor - 1.0).abs() < 1e-9 {
        vec![cell, cell + 1]
    } else {
        vec![cell]
    }
}

/// Random cost grids up to 7 x 7 cells, a fifth of them blocked, costs in `[1, 5)`, with a
/// start and goal corner.
fn grids() -> impl Strategy<Value = (Costs, GridCell, GridCell)> {
    (1_usize..=7, 1_usize..=7).prop_flat_map(|(width, height)| {
        let cell = prop_oneof![1 => Just(None), 4 => (1.0_f64..5.0).prop_map(Some)];
        let cells = prop::collection::vec(prop::collection::vec(cell, width), height);
        let corner = (0..=width, 0..=height);

        (cells, corner.clone(), corner)
            .prop_map(|(cells, start, goal)| (Costs { cells }, start, goal))
    })
}

/// The minimum cost of `costs`, a valid lower bound for the heuristic.
fn min_cost(costs: &Costs) -> NonNegF64 {
    let min = costs
        .cells
        .iter()
        .flatten()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);

    NonNegF64::new(if min.is_finite() { min } else { 0.0 }).unwrap()
}

/// What one Field D* run found: its interpolated cost `g(start)`, the path's points and
/// the path's cost.
type Found = (f64, Vec<(f64, f64)>, f64);

/// Runs Field D* once; returns what it found, or `None` if unsolvable.
fn field(costs: &Costs, start: GridCell, goal: GridCell) -> Option<Found> {
    let mut planner = FieldDStar::new(costs, start, goal, min_cost(costs));
    match planner.compute(costs, Limits::NONE).outcome {
        Outcome::Found(solution) => Some((
            planner.interpolated_cost().unwrap().get(),
            solution.states,
            solution.cost.get(),
        )),
        Outcome::Unsolvable => {
            assert!(planner.interpolated_cost().is_none());
            None
        }
        outcome => panic!("unexpected {outcome:?}"),
    }
}

/// Asserts `a <= b` up to the relative tolerance.
fn at_most(a: f64, b: f64) -> bool {
    a <= b + EPSILON * b.max(1.0)
}

/// Asserts `a == b` up to the relative tolerance.
fn close(a: f64, b: f64) -> bool {
    at_most(a, b) && at_most(b, a)
}

/// The corners of a uniform-cost [`Costs`] as a grid of cells for Theta*: corner `(x, y)`
/// is passable if every cell around it on the grid is. Theta*'s segments then touch only
/// the free cells' closed squares, so each of its paths is a Field D* path of the same
/// length.
struct Dual<'a>(&'a Costs);

impl UniformGrid for Dual<'_> {
    fn width(&self) -> usize {
        self.0.lattice().0
    }

    fn height(&self) -> usize {
        self.0.lattice().1
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        let (x, y) = signed((x, y));
        let (width, height) = signed((self.0.width(), self.0.height()));
        [(-1, -1), (-1, 0), (0, -1), (0, 0)]
            .iter()
            .map(|&(dx, dy)| (x + dx, y + dy))
            .filter(|&(cx, cy)| (0..width).contains(&cx) && (0..height).contains(&cy))
            .all(|(cx, cy)| costs_free(self.0, cx, cy))
    }
}

impl LineOfSight for Dual<'_> {}

/// Returns `true` if cell `(x, y)` of `costs` is passable.
fn costs_free(costs: &Costs, x: i64, y: i64) -> bool {
    costs.at(x, y).is_finite()
}

/// Random uniform-cost grids up to 8 x 8 cells, a sixth of them blocked (none if
/// `open`), with a start and goal corner.
fn uniform(open: bool) -> impl Strategy<Value = (Costs, GridCell, GridCell)> {
    (1_usize..=8, 1_usize..=8).prop_flat_map(move |(width, height)| {
        let blocked = u32::from(!open);
        let cell = prop_oneof![blocked => Just(None), 5 => Just(Some(1.0))];
        let cells = prop::collection::vec(prop::collection::vec(cell, width), height);
        let corner = (0..=width, 0..=height);

        (cells, corner.clone(), corner)
            .prop_map(|(cells, start, goal)| (Costs { cells }, start, goal))
    })
}

/// The Euclidean distance between two corners.
fn euclidean(a: GridCell, b: GridCell) -> f64 {
    let d = |p: usize, q: usize| f64::from(u32::try_from(p.abs_diff(q)).unwrap());

    d(a.0, b.0).hypot(d(a.1, b.1))
}

/// The observed worst ratio of a Field D* path's cost to the cheapest 8-connected corner
/// path, with margin: about 1.035 over 780 000 random grids like [`grids`], 1.043 with
/// costs up to 100, 1.027 on grids up to 20 x 20. Interpolation is not exact, so the path
/// can be dearer than the corner lattice's optimum; `g` never is.
const EIGHT_CONNECTED_FACTOR: f64 = 1.1;

/// The observed worst ratio of a Field D* path's length to Theta*'s on uniform grids, with
/// margin: about 1.011 over 670 000 random grids, open or with obstacles.
const THETA_FACTOR: f64 = 1.02;

/// The most batches of cell changes and start moves for the repair test.
const CHANGES: usize = 8;

/// One change for the repair test: set a cell's cost (`None` blocks it), or move the
/// start to a corner; coordinates are taken modulo the lattice.
#[derive(Debug, Clone)]
enum Change {
    /// Sets cell `(x, y)` to the cost.
    Cell(usize, usize, Option<f64>),
    /// Moves the start to corner `(x, y)`.
    Start(usize, usize),
}

/// Random batches of changes for [`repair_matches_fresh_runs`], each applied before one
/// repair.
fn changes() -> impl Strategy<Value = Vec<Vec<Change>>> {
    let cost = prop_oneof![1 => Just(None), 3 => (1.0_f64..5.0).prop_map(Some)];
    let change = prop_oneof![
        3 => (0_usize..64, 0_usize..64, cost).prop_map(|(x, y, c)| Change::Cell(x, y, c)),
        1 => (0_usize..64, 0_usize..64).prop_map(|(x, y)| Change::Start(x, y)),
    ];

    prop::collection::vec(prop::collection::vec(change, 1..=3), 1..=CHANGES)
}

/// Checks a found path: it runs from `start` to `goal`, its cost is its recomputed
/// cost, and it is no cheaper than the straight line at the cheapest cell cost.
fn check_path(
    costs: &Costs,
    start: GridCell,
    goal: GridCell,
    points: &[(f64, f64)],
    cost: f64,
) -> Result<(), TestCaseError> {
    let point = |(x, y): GridCell| {
        (
            f64::from(u32::try_from(x).unwrap()),
            f64::from(u32::try_from(y).unwrap()),
        )
    };
    prop_assert_eq!(points.first().copied(), Some(point(start)));
    prop_assert_eq!(points.last().copied(), Some(point(goal)));
    let recomputed = recompute(costs, points);
    prop_assert!(
        recomputed.is_some_and(|r| close(r, cost)),
        "reported {} but recomputed {:?}",
        cost,
        recomputed
    );
    prop_assert!(at_most(
        min_cost(costs).get() * euclidean(start, goal),
        cost
    ));

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// On random weighted grids: `g(start)` is the fixed point of the closed form (an
    /// independent brute-force sweep) and at most the cheapest 8-connected corner path;
    /// the path's cost is its recomputed cost, within [`EIGHT_CONNECTED_FACTOR`] of that
    /// optimum.
    #[test]
    fn interpolated_cost_and_path((costs, start, goal) in grids()) {
        let optimum = eight_connected(&costs, start, goal);
        let sweep = fixed_point(&costs, goal)[costs.index(start)];
        let Some((g, points, cost)) = field(&costs, start, goal) else {
            prop_assert!(optimum.is_infinite());
            prop_assert!(sweep.is_infinite());
            return Ok(());
        };

        prop_assert!(close(g, sweep), "g {} but the sweep gives {}", g, sweep);
        prop_assert!(at_most(g, optimum), "g {} above the 8-connected {}", g, optimum);
        check_path(&costs, start, goal, &points, cost)?;
        prop_assert!(
            at_most(cost, EIGHT_CONNECTED_FACTOR * optimum),
            "path {} vs 8-connected {}",
            cost,
            optimum
        );
    }

    /// On uniform grids, open or with obstacles: the path is no shorter than the straight
    /// line, and within [`THETA_FACTOR`] of Theta* on the corners (whose paths are Field D*
    /// paths too, but which can only use corners all of whose cells are free). On open
    /// grids Theta* is the straight line, so the path is within that factor of it.
    #[test]
    fn uniform_grids_against_theta_star(open in any::<bool>(), (costs, start, goal) in uniform(false)) {
        let costs = if open {
            Costs { cells: costs.cells.iter().map(|row| vec![Some(1.0); row.len()]).collect() }
        } else {
            costs
        };
        let theta = theta_star(&Dual(&costs), start, goal, Limits::NONE);
        let Some((_, points, cost)) = field(&costs, start, goal) else {
            prop_assert!(!open && theta.solution().is_none());
            return Ok(());
        };

        check_path(&costs, start, goal, &points, cost)?;
        prop_assert!(at_most(euclidean(start, goal), cost));
        if let Some(solution) = theta.solution() {
            let length = solution.cost.get();
            prop_assert!(at_most(cost, THETA_FACTOR * length), "path {} vs Theta* {}", cost, length);
        }

        if open {
            prop_assert!(at_most(cost, THETA_FACTOR * euclidean(start, goal)));
        }
    }

    /// After random batches of cell changes and start moves, the repaired planner agrees
    /// with a fresh one on the current grid: same solvability, same `g(start)`, and a correct path.
    #[test]
    fn repair_matches_fresh_runs((costs, start, goal) in grids(), changes in changes()) {
        let mut costs = costs;
        let one = NonNegF64::new(1.0).unwrap();
        let mut planner = FieldDStar::new(&costs, start, goal, one);
        planner.compute(&costs, Limits::NONE);
        for batch in changes {
            for change in batch {
                let (columns, rows) = costs.lattice();
                match change {
                    Change::Cell(x, y, cost) => {
                        let (x, y) = (x % costs.width(), y % costs.height());
                        costs.cells[y][x] = cost;
                        planner.cell_changed(&costs, x, y);
                    }
                    Change::Start(x, y) => planner.move_start((x % columns, y % rows)),
                }
            }

            let repaired = planner.compute(&costs, Limits::NONE);
            let mut fresh = FieldDStar::new(&costs, planner.start(), goal, one);
            let expected = fresh.compute(&costs, Limits::NONE);
            prop_assert_eq!(repaired.solution().is_some(), expected.solution().is_some());
            let (Some(g), Some(expected_g)) = (planner.interpolated_cost(), fresh.interpolated_cost()) else {
                prop_assert!(planner.interpolated_cost().is_none() && fresh.interpolated_cost().is_none());
                continue;
            };

            prop_assert!(close(g.get(), expected_g.get()), "repaired {} vs fresh {}", g, expected_g);
            let solution = repaired.solution().unwrap();
            check_path(&costs, planner.start(), goal, &solution.states, solution.cost.get())?;
        }
    }
}

/// A corner surrounded by blocked cells is unreachable, and a start that is the goal is a
/// single point at cost zero.
#[test]
fn enclosed_and_trivial() {
    let costs = Costs {
        cells: vec![vec![Some(1.0), None, None], vec![Some(1.0), None, None]],
    };
    let one = NonNegF64::new(1.0).unwrap();

    let mut enclosed = FieldDStar::new(&costs, (0, 0), (3, 1), one);
    assert!(matches!(
        enclosed.compute(&costs, Limits::NONE).outcome,
        Outcome::Unsolvable
    ));
    assert_eq!(enclosed.interpolated_cost(), None);

    let mut trivial = FieldDStar::new(&costs, (1, 1), (1, 1), one);
    let solution = trivial
        .compute(&costs, Limits::NONE)
        .into_solution()
        .unwrap();
    assert_eq!(solution.states, [(1.0, 1.0)]);
    assert_eq!(solution.cost, NonNegF64::ZERO);
}

/// A search stopped by its budget continues where it stopped, and the totals add up.
#[test]
fn budget_resumes() {
    let costs = Costs {
        cells: vec![vec![Some(1.0); 6]; 6],
    };
    let one = NonNegF64::new(1.0).unwrap();
    let mut planner = FieldDStar::new(&costs, (0, 0), (6, 6), one);

    let first = planner.compute(&costs, Limits::expansions(3));
    assert!(matches!(first.outcome, Outcome::LimitReached));
    assert_eq!(first.stats.expanded, 3);

    let second = planner.compute(&costs, Limits::NONE);
    let solution = second.solution().unwrap();
    assert!((solution.cost.get() - 72_f64.sqrt()).abs() < 1e-6);
    assert_eq!(planner.stats().expanded, 3 + second.stats.expanded);
}

/// Along a wall: the path may follow the border of blocked cells, at the free side's cost,
/// and pass the corner where two blocked cells touch diagonally.
#[test]
fn borders_and_diagonal_gaps() {
    let costs = Costs {
        cells: vec![vec![Some(1.0), None], vec![None, Some(1.0)]],
    };
    let one = NonNegF64::new(1.0).unwrap();
    let mut planner = FieldDStar::new(&costs, (0, 0), (2, 2), one);

    let solution = planner
        .compute(&costs, Limits::NONE)
        .into_solution()
        .unwrap();
    assert!(solution.states.contains(&(1.0, 1.0)));
    assert!((solution.cost.get() - 2.0 * std::f64::consts::SQRT_2).abs() < 1e-9);

    let wall = Costs {
        cells: vec![vec![Some(2.0), Some(2.0)], vec![None, None]],
    };
    let mut along = FieldDStar::new(&wall, (0, 1), (2, 1), one);
    let solution = along.compute(&wall, Limits::NONE).into_solution().unwrap();
    assert!((solution.cost.get() - 4.0).abs() < 1e-9);
}
