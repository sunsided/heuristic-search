//! Incremental planners: LPA* and D* Lite after random edge changes, start moves and wall
//! toggles, each time cross-checked against Dijkstra on the current graph; Real-time D*
//! reaching the goal under a per-step expansion limit while edges change.

use heuristic_search::{
    AdStar, AdStarMulti, AgentStep, DStarLite, DStarLiteMulti, DynamicGraph, Limits, LpaStar,
    LpaStarMulti, Outcome, RealTimeDStar, Weight, dijkstra,
};
use proptest::prelude::*;

use crate::fixtures::{Cell, Graph, Grid, manhattan};

/// A mutable directed graph as a list of `(from, to, cost)` edges over nodes `0..nodes`.
#[derive(Debug, Clone)]
struct Edges {
    /// Number of nodes.
    nodes: usize,
    /// The edges; parallel edges allowed.
    edges: Vec<(usize, usize, u64)>,
}

impl Edges {
    /// The same graph as a static [`Graph`], for Dijkstra.
    fn snapshot(&self) -> Graph {
        Graph::new(self.nodes, &self.edges)
    }
}

impl DynamicGraph for Edges {
    type Vertex = usize;
    type Cost = u64;

    fn successors(&self, v: &usize, mut emit: impl FnMut(usize, u64)) {
        self.edges
            .iter()
            .filter(|e| e.0 == *v)
            .for_each(|e| emit(e.1, e.2));
    }

    fn predecessors(&self, v: &usize, mut emit: impl FnMut(usize, u64)) {
        self.edges
            .iter()
            .filter(|e| e.1 == *v)
            .for_each(|e| emit(e.0, e.2));
    }
}

/// One change to an [`Edges`] graph.
#[derive(Debug, Clone)]
enum Change {
    /// Set the cost of edge `index % len`.
    Cost(usize, u64),
    /// Remove edge `index % len`.
    Remove(usize),
    /// Add the edge `(from % nodes, to % nodes, cost)`.
    Add(usize, usize, u64),
    /// D* Lite only: move the start one step along the current path.
    Step,
}

/// Applies `change` to `graph` and returns the endpoints of the changed edge, if any.
fn apply(graph: &mut Edges, change: &Change) -> Option<(usize, usize)> {
    let len = graph.edges.len();
    match *change {
        Change::Cost(index, cost) if len > 0 => {
            let edge = &mut graph.edges[index % len];
            edge.2 = cost;
            Some((edge.0, edge.1))
        }
        Change::Remove(index) if len > 0 => {
            let (from, to, _) = graph.edges.remove(index % len);
            Some((from, to))
        }
        Change::Add(from, to, cost) => {
            let (from, to) = (from % graph.nodes, to % graph.nodes);
            graph.edges.push((from, to, cost));
            Some((from, to))
        }
        _ => None,
    }
}

/// A random graph with up to 10 nodes and positive costs `1..10`, plus a goal and changes.
fn scenario() -> impl Strategy<Value = (Edges, usize, Vec<Change>)> {
    let change = prop_oneof![
        (0_usize..40, 1_u64..10).prop_map(|(i, c)| Change::Cost(i, c)),
        (0_usize..40).prop_map(Change::Remove),
        (0_usize..10, 0_usize..10, 1_u64..10).prop_map(|(a, b, c)| Change::Add(a, b, c)),
        Just(Change::Step),
    ];

    (2_usize..10).prop_flat_map(move |nodes| {
        let edges = prop::collection::vec((0..nodes, 0..nodes, 1_u64..10), 0..nodes * 3);

        (
            edges,
            0..nodes,
            prop::collection::vec(change.clone(), 0..12),
        )
            .prop_map(move |(edges, goal, changes)| (Edges { nodes, edges }, goal, changes))
    })
}

/// Dijkstra's cost from `start` to `goal` on the current graph.
fn optimum(graph: &Edges, start: usize, goal: usize) -> Option<u64> {
    dijkstra(
        &graph.snapshot(),
        start,
        &|s: &usize| *s == goal,
        Limits::NONE,
    )
    .into_solution()
    .map(|s| s.cost)
}

/// Asserts that `states` is a path in `graph` whose cheapest edge costs sum to `cost`.
fn assert_path(graph: &Edges, states: &[usize], cost: u64) {
    let sum: u64 = states
        .windows(2)
        .map(|pair| {
            graph
                .edges
                .iter()
                .filter(|e| e.0 == pair[0] && e.1 == pair[1])
                .map(|e| e.2)
                .min()
                .expect("consecutive states are joined by an edge")
        })
        .sum();

    assert_eq!(sum, cost);
}

proptest! {
    /// After every change, LPA* returns the current optimum, and only a valid path.
    #[test]
    fn lpa_star_tracks_changes((mut graph, goal, changes) in scenario()) {
        let mut planner = LpaStar::new(&graph, 0, goal, |_: &usize, _: &usize| 0);

        for change in std::iter::once(None).chain(changes.iter().map(Some)) {
            if let Some((from, to)) = change.and_then(|c| apply(&mut graph, c)) {
                planner.edge_changed(&graph, &from, &to);
            }

            let result = planner.compute(&graph, Limits::NONE);
            prop_assert_eq!(result.solution().map(|s| s.cost), optimum(&graph, 0, goal));
            if let Some(solution) = result.solution() {
                prop_assert_eq!(solution.states.first(), Some(&0));
                prop_assert_eq!(solution.goal(), &goal);
                assert_path(&graph, &solution.states, solution.cost);
            }
        }
    }

    /// After every change or step along the path, D* Lite returns the optimum from the
    /// current start.
    #[test]
    fn d_star_lite_tracks_changes((mut graph, goal, changes) in scenario()) {
        let mut start = 0;
        let mut planner = DStarLite::new(&graph, start, goal, |_: &usize, _: &usize| 0);
        let mut path = Vec::new();

        for change in std::iter::once(None).chain(changes.iter().map(Some)) {
            if matches!(change, Some(Change::Step)) && path.len() > 1 {
                start = path[1];
                planner.move_start(start);
            } else if let Some((from, to)) = change.and_then(|c| apply(&mut graph, c)) {
                planner.edge_changed(&graph, &from, &to);
            }

            let result = planner.compute(&graph, Limits::NONE);
            prop_assert_eq!(result.solution().map(|s| s.cost), optimum(&graph, start, goal));
            path = result.solution().map(|s| s.states.clone()).unwrap_or_default();
            if let Some(solution) = result.solution() {
                prop_assert_eq!(solution.states.first(), Some(&start));
                assert_path(&graph, &solution.states, solution.cost);
            }
        }
    }
}

proptest! {
    /// AD* with a varying weight stays within that weight of the optimum from the current
    /// start after every change, and is optimal at weight 1. The heuristic `[a != b]` is
    /// consistent for costs of at least 1.
    #[test]
    fn ad_star_tracks_changes(
        (mut graph, goal, changes) in scenario(),
        numerators in prop::collection::vec(2_u32..7, 13),
    ) {
        let unit = |a: &usize, b: &usize| u64::from(a != b);
        let mut start = 0;
        let mut planner = AdStar::new(&graph, start, goal, unit, Weight::integer(3));
        let mut path = Vec::new();

        for (step, change) in std::iter::once(None).chain(changes.iter().map(Some)).enumerate() {
            if matches!(change, Some(Change::Step)) && path.len() > 1 {
                start = path[1];
                planner.move_start(start);
            } else if let Some((from, to)) = change.and_then(|c| apply(&mut graph, c)) {
                planner.edge_changed(&graph, &from, &to);
            }

            let numerator = numerators[step];
            planner.set_weight(Weight::new(numerator, 2));
            let inflated = planner.compute(&graph, Limits::NONE);
            planner.set_weight(Weight::ONE);
            let exact = planner.compute(&graph, Limits::NONE);
            let best = optimum(&graph, start, goal);

            prop_assert_eq!(exact.solution().map(|s| s.cost), best);
            prop_assert_eq!(inflated.solution().is_some(), best.is_some());
            if let (Some(solution), Some(best)) = (inflated.solution(), best) {
                prop_assert_eq!(solution.states.first(), Some(&start));
                assert_path(&graph, &solution.states, solution.cost);
                prop_assert!(2 * solution.cost <= u64::from(numerator) * best);
            }

            path = exact.solution().map(|s| s.states.clone()).unwrap_or_default();
        }
    }
}

proptest! {
    /// AD* stays within its weight when weights with different denominators alternate and
    /// edges change or the start moves while an inflated weight is set.
    #[test]
    fn ad_star_handles_weight_changes_between_updates(
        (mut graph, goal, changes) in scenario(),
        weights in prop::collection::vec(((1_u32..4), (1_u32..9), (1_u32..9)), 13),
    ) {
        let unit = |a: &usize, b: &usize| u64::from(a != b);
        let mut start = 0;
        let mut planner = AdStar::new(&graph, start, goal, unit, Weight::integer(2));
        let mut path = Vec::new();
        let weight = |(den, extra): (u32, u32)| Weight::new(den + extra % (3 * den), den);

        for (step, change) in std::iter::once(None).chain(changes.iter().map(Some)).enumerate() {
            let (den, during, solve) = weights[step];
            planner.set_weight(weight((den, during)));
            if matches!(change, Some(Change::Step)) && path.len() > 1 {
                start = path[1];
                planner.move_start(start);
            } else if let Some((from, to)) = change.and_then(|c| apply(&mut graph, c)) {
                planner.edge_changed(&graph, &from, &to);
            }

            let bound = weight((den, solve));
            planner.set_weight(bound);
            let inflated = planner.compute(&graph, Limits::NONE);
            let best = optimum(&graph, start, goal);

            prop_assert_eq!(inflated.solution().is_some(), best.is_some());
            if let (Some(solution), Some(best)) = (inflated.solution(), best) {
                prop_assert_eq!(solution.states.first(), Some(&start));
                assert_path(&graph, &solution.states, solution.cost);
                prop_assert!(
                    u64::from(bound.denominator()) * solution.cost
                        <= u64::from(bound.numerator()) * best
                );
            }

            path = inflated.solution().map(|s| s.states.clone()).unwrap_or_default();
        }
    }
}

/// A grid whose walls can be toggled; moving costs 1.
struct Walls {
    /// The current walls, as a static grid (also used for Dijkstra).
    grid: Grid,
}

impl DynamicGraph for Walls {
    type Vertex = Cell;
    type Cost = u32;

    fn successors(&self, cell: &Cell, mut emit: impl FnMut(Cell, u32)) {
        if self.grid.walls[cell.0][cell.1] {
            return;
        }

        heuristic_search::Successors::successors(&self.grid, cell, |_, next, cost| {
            emit(next, cost);
        });
    }

    fn predecessors(&self, cell: &Cell, emit: impl FnMut(Cell, u32)) {
        // Moves are symmetric: the free neighbours of a free cell reach it.
        self.successors(cell, emit);
    }
}

/// D* Lite with the Manhattan heuristic drives across a grid while walls appear in front
/// of it (for a while enclosing the goal), and its costs match Dijkstra after every toggle.
#[test]
fn d_star_lite_drives_through_appearing_walls() {
    let mut world = Walls {
        grid: Grid::parse(&["......", "......", "......", "......", "......", "......"]),
    };
    let goal = (5, 5);
    let mut start = (0, 0);
    let mut planner = DStarLite::new(&world, start, goal, |a: &Cell, b: &Cell| manhattan(*a, *b));
    let toggles = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (2, 2),
        (5, 4),
        (4, 5),
        (5, 4),
    ];

    for toggle in toggles {
        let path = planner.compute(&world, Limits::NONE).into_solution();
        if let Some(next) = path.and_then(|solution| solution.states.get(1).copied()) {
            start = next;
            planner.move_start(start);
        }

        world.grid.walls[toggle.0][toggle.1] ^= true;
        let mut neighbours = Vec::new();
        heuristic_search::Successors::successors(
            &Grid {
                walls: vec![vec![false; 6]; 6],
            },
            &toggle,
            |_, n, _| neighbours.push(n),
        );
        for neighbour in neighbours {
            planner.edge_changed(&world, &neighbour, &toggle);
            planner.edge_changed(&world, &toggle, &neighbour);
        }

        let expected = dijkstra(&world.grid, start, &|c: &Cell| *c == goal, Limits::NONE)
            .into_solution()
            .map(|s| s.cost);
        let result = planner.compute(&world, Limits::NONE);
        assert_eq!(
            result.solution().map(|s| s.cost),
            expected,
            "after toggling {toggle:?}"
        );
    }
}

/// A budget stops the computation, and the next call picks up where it stopped.
#[test]
fn budget_resumes() {
    let graph = Edges {
        nodes: 4,
        edges: vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)],
    };
    let mut planner = LpaStar::new(&graph, 0, 3, |_: &usize, _: &usize| 0);

    assert_eq!(
        planner.compute(&graph, Limits::expansions(1)).outcome,
        Outcome::LimitReached
    );
    assert_eq!(
        planner
            .compute(&graph, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(3)
    );
    assert_eq!(planner.stats().expanded, 4);
}

/// AD* with the Manhattan heuristic lowers its weight from 3 to 1 while it drives and walls
/// toggle; every path is within the current weight of Dijkstra's cost, and ends optimal.
#[test]
fn ad_star_improves_while_driving() {
    let mut world = Walls {
        grid: Grid::parse(&[
            "........", ".######.", "......#.", ".####.#.", "......#.", "........",
        ]),
    };
    let goal = (2, 5);
    let mut start = (5, 0);
    let weights = [6, 5, 4, 3, 2, 2, 2];
    let mut planner = AdStar::new(
        &world,
        start,
        goal,
        |a: &Cell, b: &Cell| manhattan(*a, *b),
        Weight::integer(3),
    );
    let toggles = [(4, 5), (2, 2), (4, 5), (0, 3), (2, 2), (0, 3), (5, 3)];

    for (toggle, numerator) in toggles.into_iter().zip(weights) {
        world.grid.walls[toggle.0][toggle.1] ^= true;
        let mut neighbours = Vec::new();
        heuristic_search::Successors::successors(
            &Grid {
                walls: vec![vec![false; 8]; 6],
            },
            &toggle,
            |_, n, _| neighbours.push(n),
        );
        for neighbour in neighbours {
            planner.edge_changed(&world, &neighbour, &toggle);
            planner.edge_changed(&world, &toggle, &neighbour);
        }

        planner.set_weight(Weight::new(numerator, 2));
        let expected = dijkstra(&world.grid, start, &|c: &Cell| *c == goal, Limits::NONE)
            .into_solution()
            .map(|s| s.cost);
        let path = planner.compute(&world, Limits::NONE).into_solution();
        assert_eq!(path.is_some(), expected.is_some());
        if let (Some(path), Some(expected)) = (&path, expected) {
            assert!(
                2 * path.cost <= numerator * expected,
                "{} > {numerator}/2 * {expected}",
                path.cost
            );
            if let Some(&next) = path.states.get(1) {
                start = next;
                planner.move_start(start);
            }
        }
    }

    planner.set_weight(Weight::ONE);
    let expected = dijkstra(&world.grid, start, &|c: &Cell| *c == goal, Limits::NONE)
        .into_solution()
        .map(|s| s.cost);
    assert_eq!(
        planner
            .compute(&world, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        expected
    );
}

/// A cheaper parallel edge that a stale, inflated `g` hides still sets the path's cost.
#[test]
fn ad_star_costs_parallel_edges_by_the_cheapest() {
    let mut graph = Edges {
        nodes: 3,
        edges: vec![(2, 1, 4), (0, 2, 1), (2, 1, 4)],
    };
    let unit = |a: &usize, b: &usize| u64::from(a != b);
    let mut planner = AdStar::new(&graph, 0, 1, unit, Weight::integer(1));
    assert_eq!(
        planner
            .compute(&graph, Limits::NONE)
            .solution()
            .map(|s| s.cost),
        Some(5)
    );

    graph.edges[0].2 = 3;
    planner.edge_changed(&graph, &2, &1);
    planner.set_weight(Weight::new(5, 2));
    let solution = planner
        .compute(&graph, Limits::NONE)
        .into_solution()
        .expect("1 is reachable");

    assert_eq!(solution.states, [0, 2, 1]);
    assert_path(&graph, &solution.states, solution.cost);
}

proptest! {
    /// The multi-start planners track edge changes like Dijkstra from the best start:
    /// LPA* and D* Lite exactly, AD* within its weight and exactly at weight 1. Paths start
    /// at one of the starts.
    #[test]
    fn multi_start_planners_track_changes(
        (mut graph, goal, changes) in scenario(),
        raw_starts in prop::collection::vec(0_usize..10, 0..4),
        numerator in 2_u32..7,
    ) {
        let starts: Vec<usize> = raw_starts.into_iter().map(|s| s % graph.nodes).collect();
        let unit = |a: &usize, b: &usize| u64::from(a != b);
        let mut lpa = LpaStarMulti::new(&graph, starts.iter().copied(), goal, unit);
        let mut lite = DStarLiteMulti::new(&graph, starts.iter().copied(), goal, unit);
        let mut ad = AdStarMulti::new(&graph, starts.iter().copied(), goal, unit, Weight::integer(2));

        for change in std::iter::once(None).chain(changes.iter().map(Some)) {
            if let Some((from, to)) = change.and_then(|c| apply(&mut graph, c)) {
                lpa.edge_changed(&graph, &from, &to);
                lite.edge_changed(&graph, &from, &to);
                ad.edge_changed(&graph, &from, &to);
            }

            let best = starts.iter().filter_map(|&s| optimum(&graph, s, goal)).min();
            ad.set_weight(Weight::new(numerator, 2));
            let inflated = ad.compute(&graph, Limits::NONE);
            ad.set_weight(Weight::ONE);
            let results = [
                lpa.compute(&graph, Limits::NONE),
                lite.compute(&graph, Limits::NONE),
                ad.compute(&graph, Limits::NONE),
            ];

            for result in &results {
                prop_assert_eq!(result.solution().map(|s| s.cost), best);
            }

            prop_assert_eq!(inflated.solution().is_some(), best.is_some());
            for solution in results.iter().chain([&inflated]).filter_map(|r| r.solution()) {
                prop_assert!(starts.contains(&solution.states[0]));
                prop_assert_eq!(solution.states.last(), Some(&goal));
                assert_path(&graph, &solution.states, solution.cost);
            }

            if let (Some(solution), Some(best)) = (inflated.solution(), best) {
                prop_assert!(2 * solution.cost <= u64::from(numerator) * best);
            }
        }
    }
}

/// Steps Real-time D* may take before the tests call it a failure instead of hanging.
const MAX_STEPS: usize = 20_000;

/// Local shares of Real-time D* under test: D* Lite with a one-expansion lookahead, the
/// paper's three ratios, and LSS-LRTA* alone.
const LOCAL_SHARES: [Weight; 5] = [
    Weight::new(0, 1),
    Weight::new(1, 4),
    Weight::new(1, 2),
    Weight::new(3, 4),
    Weight::ONE,
];

/// A random strongly connected graph with up to 12 nodes and costs `1..10`, plus a start
/// and a goal: a cycle through every node (the first `nodes` edges), plus random edges.
fn connected_scenario() -> impl Strategy<Value = (Edges, usize, usize)> {
    (2_usize..12).prop_flat_map(|nodes| {
        let cycle = prop::collection::vec(1_u64..10, nodes);
        let extra = prop::collection::vec((0..nodes, 0..nodes, 1_u64..10), 0..nodes * 2);

        (cycle, extra, 0..nodes, 0..nodes).prop_map(move |(cycle, extra, start, goal)| {
            let mut edges: Vec<_> = cycle
                .iter()
                .enumerate()
                .map(|(from, &cost)| (from, (from + 1) % nodes, cost))
                .collect();
            edges.extend(extra);

            (Edges { nodes, edges }, start, goal)
        })
    })
}

/// One step of `agent` with `expansions`; if it moves, it must follow an edge of `graph`.
fn step_checked<D: FnMut(&usize, &usize) -> u64>(
    agent: &mut RealTimeDStar<usize, u64, D>,
    graph: &Edges,
    expansions: usize,
) -> AgentStep {
    let from = *agent.position();
    let step = agent.step(graph, expansions);
    let to = *agent.position();
    assert!(
        from == to || graph.edges.iter().any(|e| e.0 == from && e.1 == to),
        "moved along {from} -> {to}, which is no edge"
    );

    step
}

/// Steps `agent` until it arrives or waits; panics after [`MAX_STEPS`] steps.
fn drive<D: FnMut(&usize, &usize) -> u64>(
    agent: &mut RealTimeDStar<usize, u64, D>,
    graph: &Edges,
    expansions: usize,
) -> AgentStep {
    for _ in 0..MAX_STEPS {
        let step = step_checked(agent, graph, expansions);
        if matches!(step, AgentStep::Arrived | AgentStep::Waiting) {
            return step;
        }
    }

    panic!("no arrival within {MAX_STEPS} steps");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// On a static strongly connected graph Real-time D* reaches the goal for every
    /// expansion limit and local share, one edge per step, within the limit per step.
    #[test]
    fn rtd_star_reaches_the_goal_on_static_graphs(
        (graph, start, goal) in connected_scenario(),
        expansions in 1_usize..12,
    ) {
        for share in LOCAL_SHARES {
            let mut agent =
                RealTimeDStar::new(&graph, start, goal, |_: &usize, _: &usize| 0, share);
            prop_assert_eq!(drive(&mut agent, &graph, expansions), AgentStep::Arrived);

            let trial = agent.into_trial();
            let solution = trial.solution().expect("arrived");
            prop_assert_eq!(solution.states.first(), Some(&start));
            prop_assert_eq!(solution.goal(), &goal);
            assert_path(&graph, &solution.states, solution.cost);
            prop_assert!(trial.trial.max_step_expansions <= expansions as u64);
            prop_assert_eq!(trial.trial.moves, trial.trial.steps);
        }
    }

    /// With unlimited expansions every step is fully informed and the agent follows a
    /// cheapest path, or waits where no path exists.
    #[test]
    fn rtd_star_with_unlimited_expansions_is_optimal(
        (graph, goal, _changes) in scenario(),
        start in 0_usize..10,
    ) {
        let start = start % graph.nodes;
        let mut agent = RealTimeDStar::new(
            &graph,
            start,
            goal,
            |_: &usize, _: &usize| 0,
            Weight::new(1, 2),
        );
        let mut steps = Vec::new();
        let last = loop {
            let step = step_checked(&mut agent, &graph, usize::MAX);
            if matches!(step, AgentStep::Arrived | AgentStep::Waiting) {
                break step;
            }

            steps.push(step);
        };

        prop_assert!(steps.iter().all(|&step| step == AgentStep::Informed));
        let trial = agent.into_trial();
        if let Some(cost) = optimum(&graph, start, goal) {
            prop_assert_eq!(last, AgentStep::Arrived);
            let solution = trial.solution().expect("the goal is reachable");
            prop_assert_eq!(solution.cost, cost);
            assert_path(&graph, &solution.states, solution.cost);
        } else {
            prop_assert!(steps.is_empty());
            prop_assert_eq!(trial.outcome, Outcome::Unsolvable);
        }
    }

    /// Edges change their costs, appear and disappear while the agent drives, keeping the
    /// graph strongly connected; the agent still reaches the goal, within the limit per
    /// step, moving only along edges that exist when it moves.
    #[test]
    fn rtd_star_reaches_the_goal_while_edges_change(
        (mut graph, start, goal) in connected_scenario(),
        changes in prop::collection::vec(
            (0_usize..4, 0_usize..40, 0_usize..12, 1_u64..10, 0_u8..3),
            0..16,
        ),
        expansions in 1_usize..12,
        share in 0_usize..5,
    ) {
        let cycle = graph.nodes;
        let mut agent = RealTimeDStar::new(
            &graph,
            start,
            goal,
            |_: &usize, _: &usize| 0,
            LOCAL_SHARES[share],
        );

        for (steps, index, other, cost, kind) in changes {
            for _ in 0..steps {
                step_checked(&mut agent, &graph, expansions);
            }

            let extra = graph.edges.len() - cycle;
            let change = match kind {
                0 => Change::Cost(index, cost),
                1 if extra > 0 => Change::Remove(cycle + index % extra),
                _ => Change::Add(index, other, cost),
            };
            if let Some((from, to)) = apply(&mut graph, &change) {
                agent.edge_changed(&graph, &from, &to);
            }
        }

        prop_assert_eq!(drive(&mut agent, &graph, expansions), AgentStep::Arrived);
        let trial = agent.into_trial();
        prop_assert_eq!(trial.solution().map(|s| *s.goal()), Some(goal));
        prop_assert!(trial.trial.max_step_expansions <= expansions as u64);
    }
}

/// Real-time D* with the Manhattan heuristic and four expansions per move drives across a
/// grid while walls toggle around it, for a while enclosing the goal: it takes partially
/// informed steps before D* Lite reaches it, waits while the goal is enclosed, and arrives
/// once the way is open again.
#[test]
fn rtd_star_drives_through_appearing_walls() {
    let mut world = Walls {
        grid: Grid::parse(&["......", "......", "......", "......", "......", "......"]),
    };
    let goal = (5, 5);
    let distance = |a: &Cell, b: &Cell| manhattan(*a, *b);
    let mut agent = RealTimeDStar::new(&world, (0, 0), goal, distance, Weight::new(1, 2));
    let mut seen = Vec::new();
    let toggles = [(1, 1), (2, 2), (5, 4), (4, 5), (3, 3), (5, 4)];

    for toggle in toggles {
        for _ in 0..2 {
            seen.push(agent.step(&world, 4));
        }

        world.grid.walls[toggle.0][toggle.1] ^= true;
        let mut neighbours = Vec::new();
        heuristic_search::Successors::successors(
            &Grid {
                walls: vec![vec![false; 6]; 6],
            },
            &toggle,
            |_, n, _| neighbours.push(n),
        );
        for neighbour in neighbours {
            agent.edge_changed(&world, &neighbour, &toggle);
            agent.edge_changed(&world, &toggle, &neighbour);
        }

        if toggle == (4, 5) {
            // The goal is enclosed: once D* Lite has proven it, the agent stays put.
            let waits = (0..100).any(|_| agent.step(&world, 4) == AgentStep::Waiting);
            assert!(waits, "the enclosed goal is proven unreachable");
            assert_eq!(agent.step(&world, 4), AgentStep::Waiting);
        }
    }

    let arrived = (0..200).any(|_| {
        let step = agent.step(&world, 4);
        seen.push(step);
        step == AgentStep::Arrived
    });

    assert!(arrived, "the way to the goal is open again");
    assert!(seen.contains(&AgentStep::Partial));
    assert!(seen.contains(&AgentStep::Informed));
    let trial = agent.into_trial();
    assert_eq!(trial.solution().map(|s| *s.goal()), Some(goal));
    assert!(trial.trial.max_step_expansions <= 4);
}

/// A step needs at least one expansion.
#[test]
#[should_panic(expected = "at least one expansion")]
fn rtd_star_rejects_zero_expansions() {
    let graph = Edges {
        nodes: 2,
        edges: vec![(0, 1, 1)],
    };
    let mut agent = RealTimeDStar::new(&graph, 0, 1, |_: &usize, _: &usize| 0, Weight::ONE);
    agent.step(&graph, 0);
}

/// The local share is a fraction of the expansions.
#[test]
#[should_panic(expected = "at most 1")]
fn rtd_star_rejects_local_shares_above_one() {
    let graph = Edges {
        nodes: 2,
        edges: vec![(0, 1, 1)],
    };
    let _ = RealTimeDStar::new(&graph, 0, 1, |_: &usize, _: &usize| 0, Weight::new(3, 2));
}
