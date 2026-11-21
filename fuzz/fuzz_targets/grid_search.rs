//! Fuzzes Jump Point Search on small grids built from raw bytes: JPS must find the same
//! octile cost as Dijkstra and A* on the plain 8-connected graph of the same grid, and its
//! sparse path must run from the start to the goal. JPS skips a blocked or off-grid start
//! and cannot reach such a goal, while the general searches accept a start that is the goal
//! anywhere, so those endpoints expect no path.

#![no_main]
#![forbid(unsafe_code)]

use heuristic_search::{
    EightConnected, GridCell, Limits, OctileDistance, UniformGrid, a_star, dijkstra, jps,
};
use libfuzzer_sys::fuzz_target;

/// The most columns and rows a fuzzed grid has.
const MAX_SIDE: usize = 12;

/// A grid of passable (`true`) and blocked cells.
struct Map {
    /// The number of columns; at least one.
    width: usize,
    /// The number of rows; at least one.
    height: usize,
    /// `open[y * width + x]`; exactly `width * height` entries.
    open: Vec<bool>,
}

impl Map {
    /// Returns `true` if `(x, y)` lies on the grid and is passable.
    fn is_open(&self, (x, y): GridCell) -> bool {
        x < self.width && y < self.height && self.is_passable(x, y)
    }
}

impl UniformGrid for Map {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        self.open[y * self.width + x]
    }
}

/// Builds a grid, a start and a goal from bytes: the width and height, the two cells
/// (which may lie off the grid or on a blocked cell), then one bit per cell where a set bit
/// blocks it. Missing bits leave cells open. Returns `None` for too short input.
fn parse(data: &[u8]) -> Option<(Map, GridCell, GridCell)> {
    let (&[width, height, x0, y0, x1, y1], bits) = data.split_first_chunk::<6>()?;
    let width = 1 + usize::from(width) % MAX_SIDE;
    let height = 1 + usize::from(height) % MAX_SIDE;
    let cell = |x: u8, y: u8| (usize::from(x) % (width + 1), usize::from(y) % (height + 1));
    let blocked = |index: usize| {
        bits.get(index / 8)
            .is_some_and(|byte| byte >> (index % 8) & 1 == 1)
    };
    let open = (0..width * height).map(|index| !blocked(index)).collect();

    Some((
        Map {
            width,
            height,
            open,
        },
        cell(x0, y0),
        cell(x1, y1),
    ))
}

fuzz_target!(|data: &[u8]| {
    let Some((map, start, goal)) = parse(data) else {
        return;
    };

    let graph = EightConnected(&map);
    let is_goal = |cell: &GridCell| *cell == goal;
    let endpoints_open = map.is_open(start) && map.is_open(goal);
    let expected = dijkstra(&graph, start, &is_goal, Limits::NONE)
        .solution()
        .map(|s| s.cost)
        .filter(|_| endpoints_open);
    let informed = a_star(&graph, start, &is_goal, OctileDistance(goal), Limits::NONE);
    let result = jps(&map, start, goal, Limits::NONE);

    if endpoints_open {
        assert_eq!(informed.solution().map(|s| s.cost), expected, "a_star cost");
    }

    assert_eq!(result.solution().map(|s| s.cost), expected, "jps cost");

    if let Some(solution) = result.solution() {
        assert_eq!(solution.states[0], start, "jps starts elsewhere");
        assert_eq!(solution.states.last(), Some(&goal), "jps ends elsewhere");
    }
});
