//! Delivery robots on a hospital floor: many queries on one static floor plan, answered by
//! plain A* and by hierarchical A* (HPA*) over two kinds of regions.
//!
//! The floor is a generated 300 x 300 grid: a lattice of corridors four cells wide, and
//! between them blocks of wards split into rooms by thin walls with one-cell doors. Each
//! query sends a robot from a random room to another one. The searches compared:
//!
//! - `a_star` with the octile distance, from scratch for every query: the baseline.
//! - `hierarchical_a_star` over the *rooms* (`Hierarchy::new` with a room map; corridors
//!   are cut into short segments): every door is an entrance, so paths are exactly as
//!   cheap as A*'s.
//! - `hierarchical_a_star` over square *clusters* (`Hierarchy::grid`, classic HPA*): no
//!   floor-plan knowledge needed, a few entrance cells per cluster border, paths a little
//!   longer than the cheapest.
//! - `hierarchical_a_star_par` (feature `parallel`): the same plans, with the per-region
//!   refinements spread across threads.
//!
//! The table shows the one-off build time of each hierarchy, the mean wall time and the
//! mean expansions per query (the hierarchical searches count the connecting, abstract and
//! refining searches, not the build), and the path cost against A*'s (`1.0000` is
//! optimal). Room regions must match A* exactly and clusters must never beat it; the
//! example asserts both. Timings are as measured on your machine; the parallel variant
//! spawns threads for every query, which on refinements this short costs more than it
//! saves.
//!
//! Run with `cargo run --release --features parallel --example indoor_navigation`.

#[path = "support/prng.rs"]
pub mod prng;

use std::time::{Duration, Instant};

use heuristic_search::{
    EightConnected, GridCell, GridStep, Hierarchy, Limits, Octile, OctileDistance, Regions,
    SearchResult, UniformGrid, a_star, hierarchical_a_star, hierarchical_a_star_par,
};
use prng::SplitMix64;

/// Side length of the floor in cells.
const SIZE: usize = 300;

/// Distance between the starts of two parallel corridors.
const PITCH: usize = 60;

/// Width of a corridor in cells.
const CORRIDOR: usize = 4;

/// Rooms wider or taller than this (inside their walls) are split further.
const MAX_ROOM: usize = 16;

/// No room is split into parts narrower than this (inside their walls).
const MIN_ROOM: usize = 5;

/// Side length of the corridor segments that form one region each.
const SEGMENT: usize = 30;

/// Region ids of corridor segments start here, above every room id.
const CORRIDOR_REGIONS: usize = 1_000_000;

/// Region id of wall cells, which no search enters.
const WALL: usize = usize::MAX;

/// Number of delivery queries.
const QUERIES: usize = 100;

/// A rectangle of cells `x0..x1` by `y0..y1` whose outermost rows and columns are walls.
#[derive(Debug, Clone, Copy)]
struct Rect {
    /// First column, a wall.
    x0: usize,
    /// First row, a wall.
    y0: usize,
    /// One past the last column, which is a wall.
    x1: usize,
    /// One past the last row, which is a wall.
    y1: usize,
}

/// The generated floor plan: which cells are open, and which room or corridor segment
/// every cell belongs to.
struct Floor {
    /// `true` for open cells, row by row.
    open: Vec<bool>,
    /// The region of every cell, row by row.
    region: Vec<usize>,
    /// The rooms, as rectangles including their walls.
    rooms: Vec<Rect>,
}

impl UniformGrid for Floor {
    fn width(&self) -> usize {
        SIZE
    }

    fn height(&self) -> usize {
        SIZE
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        self.open[y * SIZE + x]
    }
}

/// Returns `true` if row or column `c` lies in a corridor: one every [`PITCH`] cells, and
/// one along the far edge of the floor.
fn in_corridor_line(c: usize) -> bool {
    (1..SIZE - 1).contains(&c) && ((c - 1) % PITCH < CORRIDOR || c >= SIZE - 1 - CORRIDOR)
}

/// The spans `a..b` between corridors along one axis, each with a wall at `a` and `b - 1`.
fn block_spans() -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for c in 1..SIZE - 1 {
        match (in_corridor_line(c), start) {
            (false, None) => start = Some(c),
            (true, Some(a)) => {
                spans.push((a, c));
                start = None;
            }
            _ => {}
        }
    }

    spans
}

impl Floor {
    /// Generates the floor from `seed`: corridors, blocks of rooms split recursively, one
    /// door in every inner wall, and doors from rooms onto the corridors.
    fn generate(seed: u64) -> Self {
        let mut floor = Self {
            open: vec![false; SIZE * SIZE],
            region: vec![WALL; SIZE * SIZE],
            rooms: Vec::new(),
        };
        let mut rng = SplitMix64::new(seed);
        for y in 1..SIZE - 1 {
            for x in 1..SIZE - 1 {
                if in_corridor_line(x) || in_corridor_line(y) {
                    floor.open[y * SIZE + x] = true;
                    floor.region[y * SIZE + x] =
                        CORRIDOR_REGIONS + (y / SEGMENT) * SIZE + x / SEGMENT;
                }
            }
        }

        let spans = block_spans();
        for &(y0, y1) in &spans {
            for &(x0, x1) in &spans {
                floor.block(Rect { x0, y0, x1, y1 }, &mut rng);
            }
        }

        floor
    }

    /// Splits one block between corridors into rooms and gives it doors onto the
    /// corridors around it, at least one.
    fn block(&mut self, block: Rect, rng: &mut SplitMix64) {
        let mut doors = Vec::new();
        let first_room = self.rooms.len();
        self.split(block, rng, &mut doors);
        for &(x, y) in &doors {
            // A door between two rooms belongs to the room left of or above it.
            let side = if self.open[y * SIZE + x - 1] {
                y * SIZE + x - 1
            } else {
                (y - 1) * SIZE + x
            };
            self.open[y * SIZE + x] = true;
            self.region[y * SIZE + x] = self.region[side];
        }

        let mut outside = 0;
        for id in first_room..self.rooms.len() {
            let room = self.rooms[id];
            let sides = [
                (room.y0 == block.y0)
                    .then(|| (room.x0 + 1 + rng.below(room.x1 - room.x0 - 2), room.y0)),
                (room.y1 == block.y1)
                    .then(|| (room.x0 + 1 + rng.below(room.x1 - room.x0 - 2), room.y1 - 1)),
                (room.x0 == block.x0)
                    .then(|| (room.x0, room.y0 + 1 + rng.below(room.y1 - room.y0 - 2))),
                (room.x1 == block.x1)
                    .then(|| (room.x1 - 1, room.y0 + 1 + rng.below(room.y1 - room.y0 - 2))),
            ];
            let wanted = rng.below(2) == 0 || (outside == 0 && id + 1 == self.rooms.len());
            if let Some((x, y)) = sides.into_iter().flatten().next().filter(|_| wanted) {
                self.open[y * SIZE + x] = true;
                self.region[y * SIZE + x] = id;
                outside += 1;
            }
        }
        assert!(outside > 0, "the last room of a block touches its edge");
    }

    /// Splits `rect` by a wall with one door while a part is larger than [`MAX_ROOM`], and
    /// records the leaves as rooms. A wall never meets an earlier door, so every door
    /// keeps an open cell on both sides.
    fn split(&mut self, rect: Rect, rng: &mut SplitMix64, doors: &mut Vec<GridCell>) {
        let (width, height) = (rect.x1 - rect.x0 - 2, rect.y1 - rect.y0 - 2);
        let vertical = width >= height;
        let (low, high) = if vertical {
            (rect.x0, rect.x1)
        } else {
            (rect.y0, rect.y1)
        };
        let fits = |at: usize| {
            doors.iter().all(|&(x, y)| {
                if vertical {
                    x != at || (y != rect.y0 && y != rect.y1 - 1)
                } else {
                    y != at || (x != rect.x0 && x != rect.x1 - 1)
                }
            })
        };
        let span = high - low - 2;
        let at = (span > MAX_ROOM)
            .then(|| {
                (0..20)
                    .map(|_| low + 1 + MIN_ROOM + rng.below(span - 2 * MIN_ROOM))
                    .find(|&at| fits(at))
            })
            .flatten();

        let Some(at) = at else {
            let id = self.rooms.len();
            for y in rect.y0 + 1..rect.y1 - 1 {
                for x in rect.x0 + 1..rect.x1 - 1 {
                    self.open[y * SIZE + x] = true;
                    self.region[y * SIZE + x] = id;
                }
            }
            self.rooms.push(rect);
            return;
        };

        let (first, second, door) = if vertical {
            (
                Rect { x1: at + 1, ..rect },
                Rect { x0: at, ..rect },
                (at, rect.y0 + 1 + rng.below(height)),
            )
        } else {
            (
                Rect { y1: at + 1, ..rect },
                Rect { y0: at, ..rect },
                (rect.x0 + 1 + rng.below(width), at),
            )
        };
        doors.push(door);
        self.split(first, rng, doors);
        self.split(second, rng, doors);
    }

    /// A random open cell inside a random room.
    fn random_room_cell(&self, rng: &mut SplitMix64) -> GridCell {
        let room = self.rooms[rng.below(self.rooms.len())];
        (
            room.x0 + 1 + rng.below(room.x1 - room.x0 - 2),
            room.y0 + 1 + rng.below(room.y1 - room.y0 - 2),
        )
    }
}

/// The octile distance, admissible and consistent on 8-connected grids.
fn distance(a: &GridCell, b: &GridCell) -> Octile {
    Octile::between(*a, *b)
}

/// What one method did over all queries.
struct Row {
    /// Mean wall time per query.
    time: Duration,
    /// Mean expansions per query.
    expanded: f64,
    /// Mean ratio of the path cost to A*'s.
    mean_ratio: f64,
    /// Largest ratio of the path cost to A*'s.
    max_ratio: f64,
    /// Path costs in query order.
    costs: Vec<Octile>,
}

/// Runs `search` on every query and compares its costs with `optimum` (A*'s, or the
/// method's own when `optimum` is empty).
fn measure(
    queries: &[(GridCell, GridCell)],
    optimum: &[Octile],
    mut search: impl FnMut(GridCell, GridCell) -> SearchResult<GridCell, GridStep, Octile>,
) -> Row {
    let mut costs = Vec::with_capacity(queries.len());
    let mut expanded = 0;
    let started = Instant::now();
    for &(start, goal) in queries {
        let result = search(start, goal);
        expanded += result.stats.expanded;
        costs.push(result.solution().expect("every room is reachable").cost);
    }
    let time = started.elapsed();

    let ratios: Vec<f64> = if optimum.is_empty() {
        vec![1.0]
    } else {
        costs
            .iter()
            .zip(optimum)
            .map(|(cost, best)| cost.to_f64() / best.to_f64().max(1.0))
            .collect()
    };
    let n = u32::try_from(queries.len()).expect("few queries");

    Row {
        time: time / n,
        expanded: f64::from(u32::try_from(expanded).expect("few expansions")) / f64::from(n),
        mean_ratio: ratios.iter().sum::<f64>()
            / f64::from(u32::try_from(ratios.len()).expect("few")),
        max_ratio: ratios.iter().copied().fold(1.0, f64::max),
        costs,
    }
}

/// Prints one table row; `build` is the hierarchy's build time, if it has one.
fn print_row(name: &str, build: Option<Duration>, row: &Row) {
    let build = build.map_or_else(
        || "-".to_string(),
        |build| format!("{:.1}", build.as_secs_f64() * 1e3),
    );
    println!(
        "{name:<30} {build:>9} {:>10.1} {:>10.0} {:>8.4} {:>8.4}",
        row.time.as_secs_f64() * 1e6,
        row.expanded,
        row.mean_ratio,
        row.max_ratio,
    );
}

/// Answers the queries over `hierarchy` sequentially and in parallel, prints both rows and
/// checks the costs: equal to `optimum` if the hierarchy is `exact`, never below it
/// otherwise, and the same in both variants.
fn hierarchy_rows<R: Regions<GridCell> + Sync>(
    label: &str,
    build: Duration,
    graph: EightConnected<&Floor>,
    hierarchy: &Hierarchy<GridCell, Octile, R>,
    queries: &[(GridCell, GridCell)],
    (optimum, exact): (&[Octile], bool),
) {
    let row = measure(queries, optimum, |start, goal| {
        hierarchical_a_star(&graph, hierarchy, start, goal, distance, Limits::NONE)
    });
    print_row(&format!("hierarchical, {label}"), Some(build), &row);
    if exact {
        assert_eq!(row.costs, optimum, "room regions keep every door");
    } else {
        let never_cheaper = row
            .costs
            .iter()
            .zip(optimum)
            .all(|(cost, best)| cost >= best);
        assert!(never_cheaper, "no path beats A*'s");
    }

    let par = measure(queries, optimum, |start, goal| {
        hierarchical_a_star_par(&graph, hierarchy, start, goal, distance, Limits::NONE)
    });
    print_row(&format!("hierarchical_par, {label}"), Some(build), &par);
    assert_eq!(par.costs, row.costs, "refining in parallel changes nothing");
}

/// Builds the floor, answers the queries with every method and prints the table.
fn main() {
    let floor = Floor::generate(0x00C0_FFEE);
    let graph = EightConnected(&floor);
    let mut rng = SplitMix64::new(7);
    let queries: Vec<(GridCell, GridCell)> = (0..QUERIES)
        .map(|_| {
            (
                floor.random_room_cell(&mut rng),
                floor.random_room_cell(&mut rng),
            )
        })
        .collect();
    let open = floor.open.iter().filter(|&&open| open).count();
    println!(
        "floor {SIZE} x {SIZE}: {open} open cells, {} rooms, {QUERIES} delivery queries\n",
        floor.rooms.len()
    );
    println!(
        "{:<30} {:>9} {:>10} {:>10} {:>8} {:>8}",
        "method", "build ms", "query µs", "expanded", "cost/A*", "worst"
    );

    let baseline = measure(&queries, &[], |start, goal| {
        let at_goal = |cell: &GridCell| *cell == goal;
        a_star(&graph, start, &at_goal, OctileDistance(goal), Limits::NONE)
    });
    print_row("a_star", None, &baseline);
    let optimum = (baseline.costs.as_slice(), true);

    let started = Instant::now();
    let cells = (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (x, y)));
    let room = |&(x, y): &GridCell| floor.region[y * SIZE + x];
    let rooms = Hierarchy::new(&graph, room, cells, distance);
    hierarchy_rows("rooms", started.elapsed(), graph, &rooms, &queries, optimum);

    for size in [8, 16] {
        let started = Instant::now();
        let clusters = Hierarchy::grid(&floor, size);
        let label = format!("{size}x{size} cells");
        let near_optimum = (optimum.0, false);
        hierarchy_rows(
            &label,
            started.elapsed(),
            graph,
            &clusters,
            &queries,
            near_optimum,
        );
    }

    println!(
        "\nRoom regions are exact (cost/A* = 1); square clusters need no floor plan \
         and cost a few percent more per path."
    );
}
