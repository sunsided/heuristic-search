//! A game character that learns its way across a map with a misleading heuristic: real-time
//! search agents that plan a few expansions per move and keep what they learn across
//! repeated trials.
//!
//! The map has a cul-de-sac that opens towards the start and points at the goal, so the
//! Manhattan distance lures every agent into it. Past it, a narrow passage leads to the
//! goal, either straight through a swamp ford (stepping into it costs [`SWAMP`]) or
//! along a short bypass of floor. Each agent runs trial after trial from the same start
//! with one `LearnedHeuristic`, which starts at the Manhattan distance and is raised
//! wherever the agent finds it too optimistic. A trial that changes nothing in the table
//! means the agent has converged.
//!
//! The agents, all with a lookahead of [`LOOKAHEAD`] expansions per move except LRTA*:
//!
//! - `lrta_star`: one expansion per move, the classic.
//! - `lss_lrta_star`: an A* lookahead and a Dijkstra update over it.
//! - `lhb_lrta_star`: the same A* lookahead with the limited-horizon Bellman update.
//! - `lhb_gbfs`: that update after a greedy best-first lookahead.
//! - `lhb_bwa_star`: that update after a batch weighted A* (BWAS) lookahead, with weight
//!   `w` and batch size `b`; `w = 1, b = 1` is `lhb_lrta_star`.
//!
//! The table shows the cost of the first trials, the trial that converged, its cost
//! against the cheapest path (by Dijkstra) and the expansions over all trials. The agents
//! that move along cheapest paths in their lookahead (LRTA*, LSS-LRTA*, LHB with A* or
//! BWAS with `w = 1, b = 1`) must converge to the cheapest path, which the example
//! asserts. GBFS ignores path costs and BWAS trades cost for depth through its weight and
//! expands batches out of order, so their converged trials may keep a detour. On this map
//! GBFS and BWAS with a batch of three settle on the ford: a batch can take in the goal
//! through the ford before the bypass is expanded, and a goal in the batch ends the
//! lookahead. The other batch sizes and the
//! weight converge to the cheapest path here; whether a detour sticks depends on the
//! map. The verdict column is computed from the costs, not written in.
//!
//! Run with `cargo run --release --example learning_heuristics`.

use heuristic_search::{
    Heuristic, LearnedHeuristic, Limits, Successors, TrialResult, Weight, dijkstra, heuristic_fn,
    lhb_bwa_star, lhb_gbfs, lhb_lrta_star, lrta_star, lss_lrta_star,
};

/// The map: `#` wall, `.` floor (cost 1 to enter), `~` swamp (cost [`SWAMP`] to enter),
/// `S` start and `G` goal.
const MAP: &str = "\
######################
#....................#
#.....#######........#
#...........#........#
#.S.........#........#
#...........#........#
#.....#######........#
#....................#
#########.############
#########.~G##########
#########...##########
######################";

/// The cost of stepping into a swamp cell.
const SWAMP: u32 = 12;

/// Expansions per move of the lookahead agents.
const LOOKAHEAD: usize = 8;

/// Trials after which an agent counts as not converged, which fails the example.
const MAX_TRIALS: usize = 5_000;

/// A cell as `(x, y)`.
type Cell = (usize, usize);

/// The parsed map: the cost of entering every cell, if it can be entered.
struct Map {
    /// Row by row: `None` for walls, else the cost to enter.
    cells: Vec<Option<u32>>,
    /// Width in cells.
    width: usize,
    /// The start cell.
    start: Cell,
    /// The goal cell.
    goal: Cell,
}

impl Map {
    /// Parses [`MAP`].
    fn parse() -> Self {
        let width = MAP.lines().next().expect("the map has rows").len();
        let (mut start, mut goal) = ((0, 0), (0, 0));
        let mut cells = Vec::new();
        for (y, row) in MAP.lines().enumerate() {
            for (x, c) in row.chars().enumerate() {
                match c {
                    'S' => start = (x, y),
                    'G' => goal = (x, y),
                    _ => {}
                }
                cells.push(match c {
                    '#' => None,
                    '~' => Some(SWAMP),
                    _ => Some(1),
                });
            }
        }

        Self {
            cells,
            width,
            start,
            goal,
        }
    }

    /// The cost of the cheapest path from the start to the goal, by Dijkstra.
    fn optimum(&self) -> u32 {
        dijkstra(
            self,
            self.start,
            &|cell: &Cell| *cell == self.goal,
            Limits::NONE,
        )
        .solution()
        .expect("the goal is reachable")
        .cost
    }

    /// The Manhattan distance to the goal: consistent, since every step costs at least 1.
    fn manhattan(&self) -> impl Heuristic<Cell> + Copy {
        let goal = self.goal;
        heuristic_fn(move |&(x, y): &Cell| {
            u32::try_from(x.abs_diff(goal.0) + y.abs_diff(goal.1)).ok()
        })
    }
}

impl Successors for Map {
    type State = Cell;
    type Action = ();
    type Cost = u32;

    fn successors(&self, &(x, y): &Cell, mut emit: impl FnMut((), Cell, u32)) {
        // The border is all walls, so the neighbours of a floor cell are on the map.
        for next in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            if let Some(cost) = self.cells[next.1 * self.width + next.0] {
                emit((), next, cost);
            }
        }
    }
}

/// An agent under comparison.
#[derive(Debug, Clone, Copy)]
enum Agent {
    /// [`lrta_star`].
    Lrta,
    /// [`lss_lrta_star`].
    LssLrta,
    /// [`lhb_lrta_star`].
    LhbLrta,
    /// [`lhb_gbfs`].
    LhbGbfs,
    /// [`lhb_bwa_star`] with a weight and a batch size.
    LhbBwa(Weight, usize),
}

impl Agent {
    /// The agent's row label.
    fn name(self) -> String {
        match self {
            Self::Lrta => "lrta_star".into(),
            Self::LssLrta => "lss_lrta_star".into(),
            Self::LhbLrta => "lhb_lrta_star".into(),
            Self::LhbGbfs => "lhb_gbfs".into(),
            Self::LhbBwa(w, batch) => {
                format!(
                    "lhb_bwa_star w={}/{} b={batch}",
                    w.numerator(),
                    w.denominator()
                )
            }
        }
    }

    /// Returns `true` if the agent's moves follow cheapest paths in its lookahead, so it
    /// must converge to a cheapest path.
    fn moves_by_cost(self) -> bool {
        match self {
            Self::Lrta | Self::LssLrta | Self::LhbLrta => true,
            Self::LhbGbfs => false,
            Self::LhbBwa(w, batch) => w == Weight::ONE && batch == 1,
        }
    }

    /// One trial from the start, learning in `table`.
    fn trial<H: Heuristic<Cell>>(
        self,
        map: &Map,
        table: &mut LearnedHeuristic<Cell, u32, H>,
    ) -> TrialResult<Cell, (), u32> {
        let goal = |cell: &Cell| *cell == map.goal;
        let start = map.start;
        match self {
            Self::Lrta => lrta_star(map, start, &goal, table, Limits::NONE),
            Self::LssLrta => lss_lrta_star(map, start, &goal, table, LOOKAHEAD, Limits::NONE),
            Self::LhbLrta => lhb_lrta_star(map, start, &goal, table, LOOKAHEAD, Limits::NONE),
            Self::LhbGbfs => lhb_gbfs(map, start, &goal, table, LOOKAHEAD, Limits::NONE),
            Self::LhbBwa(weight, batch) => lhb_bwa_star(
                map,
                start,
                &goal,
                table,
                LOOKAHEAD,
                weight,
                batch,
                Limits::NONE,
            ),
        }
    }
}

/// Runs trials of `agent` with one table until a trial learns nothing; returns the cost of
/// every trial, the converged one last, and the expansions over all of them.
fn learn(agent: Agent, map: &Map) -> (Vec<u32>, u64) {
    let mut table = LearnedHeuristic::new(map.manhattan());
    let mut costs = Vec::new();
    let mut expansions = 0;
    while costs.len() < MAX_TRIALS {
        let result = agent.trial(map, &mut table);
        costs.push(result.solution().expect("the goal is reachable").cost);
        expansions += result.stats.expanded;
        if result.trial.updates == 0 {
            return (costs, expansions);
        }
    }

    panic!("{} did not converge in {MAX_TRIALS} trials", agent.name());
}

/// Runs every agent, prints the table and checks the cost-ordered agents' convergence.
fn main() {
    let map = Map::parse();
    let optimum = map.optimum();
    println!("{MAP}\n");
    println!(
        "start {:?}, goal {:?}, cheapest path {optimum}, Manhattan estimate at the start {}\n",
        map.start,
        map.goal,
        map.manhattan().estimate(&map.start).expect("finite"),
    );
    println!(
        "{:<24} {:>6} {:>6} {:>6} {:>9} {:>6} {:>6} {:>10}  verdict",
        "agent", "1st", "2nd", "5th", "converged", "cost", "/opt", "expanded"
    );

    let agents = [
        Agent::Lrta,
        Agent::LssLrta,
        Agent::LhbLrta,
        Agent::LhbGbfs,
        Agent::LhbBwa(Weight::ONE, 1),
        Agent::LhbBwa(Weight::ONE, 2),
        Agent::LhbBwa(Weight::ONE, 3),
        Agent::LhbBwa(Weight::ONE, 4),
        Agent::LhbBwa(Weight::integer(3), 4),
    ];
    for agent in agents {
        let (costs, expansions) = learn(agent, &map);
        let last = *costs.last().expect("at least one trial");
        let nth = |n: usize| costs.get(n - 1).map_or_else(|| "-".into(), u32::to_string);
        let verdict = if last == optimum {
            "optimal".to_string()
        } else {
            format!("detour, +{}", last - optimum)
        };
        println!(
            "{:<24} {:>6} {:>6} {:>6} {:>9} {:>6} {:>6.3} {:>10}  {verdict}",
            agent.name(),
            nth(1),
            nth(2),
            nth(5),
            costs.len(),
            last,
            f64::from(last) / f64::from(optimum),
            expansions,
        );

        assert!(last >= optimum, "no trial beats the cheapest path");
        if agent.moves_by_cost() {
            assert_eq!(
                last,
                optimum,
                "{} converges to a cheapest path",
                agent.name()
            );
        }
    }
}
