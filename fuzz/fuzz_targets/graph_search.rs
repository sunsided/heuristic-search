//! Fuzzes the optimal graph searches on small graphs built from raw bytes: Dijkstra, A*
//! (blind and with an admissible heuristic), fringe search and bidirectional Dijkstra must
//! all return a real path of the cheapest cost, which a Bellman-Ford oracle computes
//! independently of the crate.

#![no_main]
#![forbid(unsafe_code)]

use heuristic_search::{
    GraphResult, Limits, Predecessors, Successors, Zero, a_star, bidirectional_dijkstra, dijkstra,
    fringe_search, heuristic_fn,
};
use libfuzzer_sys::fuzz_target;

/// The most nodes a fuzzed graph has; small graphs keep each run fast.
const MAX_NODES: usize = 12;

/// The most edges read from the input.
const MAX_EDGES: usize = 48;

/// A directed graph as adjacency lists of `(target, cost)` pairs. States are node indices;
/// an edge's action is its index in the source's adjacency list.
struct Graph {
    /// `edges[v]` lists the outgoing edges of node `v`. Never empty as a whole: node `0`
    /// always exists.
    edges: Vec<Vec<(usize, u64)>>,
}

impl Successors for Graph {
    type State = usize;
    type Action = usize;
    type Cost = u64;

    fn successors(&self, state: &usize, mut emit: impl FnMut(usize, usize, u64)) {
        for (index, &(to, cost)) in self.edges[*state].iter().enumerate() {
            emit(index, to, cost);
        }
    }
}

/// Predecessors by scanning every adjacency list, with the same actions as
/// [`Successors`].
impl Predecessors for Graph {
    type Node = usize;
    type Action = usize;
    type Cost = u64;

    fn predecessors(&self, node: &usize, mut emit: impl FnMut(usize, usize, u64)) {
        for (from, edges) in self.edges.iter().enumerate() {
            for (index, &(to, cost)) in edges.iter().enumerate() {
                if to == *node {
                    emit(index, from, cost);
                }
            }
        }
    }
}

/// A fuzzed search problem: a graph, a goal node and a heuristic scale.
struct Problem {
    /// The graph; the start is node `0`.
    graph: Graph,
    /// The goal node, below the node count.
    goal: usize,
    /// The heuristic is `scale / 3` of the true distance, so `0..=3` keeps it admissible.
    scale: u64,
}

impl Problem {
    /// Builds a problem from bytes: the node count, the goal and the heuristic scale, then
    /// `(from, to, cost)` triples with costs `0..=9`. Returns `None` for too short input.
    fn parse(data: &[u8]) -> Option<Self> {
        let (&[nodes, goal, scale], rest) = data.split_first_chunk::<3>()?;
        let nodes = 1 + usize::from(nodes) % MAX_NODES;
        let mut edges = vec![Vec::new(); nodes];

        for &[from, to, cost] in rest.as_chunks::<3>().0.iter().take(MAX_EDGES) {
            let (from, to) = (usize::from(from) % nodes, usize::from(to) % nodes);
            edges[from].push((to, u64::from(cost % 10)));
        }

        Some(Self {
            graph: Graph { edges },
            goal: usize::from(goal) % nodes,
            scale: u64::from(scale % 4),
        })
    }
}

/// The exact cost from every node to `goal`, or `None` where `goal` is unreachable.
fn distances_to(graph: &Graph, goal: usize) -> Vec<Option<u64>> {
    let mut distance = vec![None; graph.edges.len()];
    distance[goal] = Some(0);

    for _ in 0..graph.edges.len() {
        for (from, edges) in graph.edges.iter().enumerate() {
            for &(to, cost) in edges {
                let Some(via) = distance[to].map(|d: u64| d + cost) else {
                    continue;
                };

                if distance[from].is_none_or(|d| via < d) {
                    distance[from] = Some(via);
                }
            }
        }
    }

    distance
}

/// Asserts that `result` found a real path from node `0` to `goal` of cost `expected`, or
/// no path when `expected` is `None`.
fn check(
    name: &str,
    graph: &Graph,
    goal: usize,
    result: &GraphResult<Graph>,
    expected: Option<u64>,
) {
    let solution = result.solution();
    assert_eq!(solution.map(|s| s.cost), expected, "{name} cost");

    let Some(solution) = solution else {
        return;
    };

    assert_eq!(solution.states[0], 0, "{name} starts elsewhere");
    assert_eq!(solution.states.last(), Some(&goal), "{name} ends elsewhere");
    assert_eq!(
        solution.states.len(),
        solution.actions.len() + 1,
        "{name} path shape"
    );

    let mut cost = 0;
    for (step, &action) in solution.actions.iter().enumerate() {
        let (to, edge_cost) = graph.edges[solution.states[step]][action];
        assert_eq!(
            to,
            solution.states[step + 1],
            "{name} action {step} leads elsewhere"
        );
        cost += edge_cost;
    }

    assert_eq!(cost, solution.cost, "{name} cost does not match its path");
}

fuzz_target!(|data: &[u8]| {
    let Some(Problem { graph, goal, scale }) = Problem::parse(data) else {
        return;
    };

    let distance = distances_to(&graph, goal);
    let expected = distance[0];
    let is_goal = |state: &usize| *state == goal;
    let admissible = heuristic_fn(|state: &usize| distance[*state].map(|d| d * scale / 3));

    let runs = [
        ("dijkstra", dijkstra(&graph, 0, &is_goal, Limits::NONE)),
        (
            "blind a_star",
            a_star(&graph, 0, &is_goal, Zero, Limits::NONE),
        ),
        (
            "a_star",
            a_star(&graph, 0, &is_goal, admissible, Limits::NONE),
        ),
        (
            "fringe_search",
            fringe_search(&graph, 0, &is_goal, Zero, Limits::NONE),
        ),
        (
            "bidirectional_dijkstra",
            bidirectional_dijkstra(&graph, 0, [goal], Limits::NONE),
        ),
    ];

    for (name, result) in &runs {
        check(name, &graph, goal, result, expected);
    }
});
