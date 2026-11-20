//! Test graphs: a weighted adjacency-list [`Graph`] and a 4-connected [`Grid`], random graphs,
//! a Bellman-Ford distance oracle, and a checker that a [`Solution`] is a real path.

use heuristic_search::{Predecessors, Solution, Successors};
use proptest::prelude::*;

/// A directed graph as adjacency lists of `(target, cost)` pairs. States are node indices;
/// an edge's action is its index in the source's adjacency list.
#[derive(Debug, Clone)]
pub struct Graph {
    /// `edges[v]` lists the outgoing edges of node `v`.
    pub edges: Vec<Vec<(usize, u64)>>,
}

impl Graph {
    /// Creates a graph with `nodes` nodes and the directed `edges` `(from, to, cost)`.
    pub fn new(nodes: usize, edges: &[(usize, usize, u64)]) -> Self {
        let mut adjacency = vec![Vec::new(); nodes];
        for &(from, to, cost) in edges {
            adjacency[from].push((to, cost));
        }

        Self { edges: adjacency }
    }

    /// The same graph with every edge reversed. An edge keeps its cost.
    pub fn reversed(&self) -> Self {
        let mut edges = vec![Vec::new(); self.edges.len()];
        for (from, targets) in self.edges.iter().enumerate() {
            for &(to, cost) in targets {
                edges[to].push((from, cost));
            }
        }

        Self { edges }
    }

    /// The same graph with every edge cost set to `1`.
    pub fn unit(&self) -> Self {
        let edges = self
            .edges
            .iter()
            .map(|edges| edges.iter().map(|&(to, _)| (to, 1)).collect())
            .collect();

        Self { edges }
    }
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

/// Exact cost from every node to `goal`, or `None` if `goal` is unreachable.
///
/// Bellman-Ford relaxation, independent of the crate's own searches so it can serve as the
/// oracle for them.
pub fn distances_to(graph: &Graph, goal: usize) -> Vec<Option<u64>> {
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

/// A random graph with up to 12 nodes and edge costs `0..=9`, plus a goal node.
pub fn graph_and_goal() -> impl Strategy<Value = (Graph, usize)> {
    (2_usize..12).prop_flat_map(|nodes| {
        let edges = prop::collection::vec((0..nodes, 0..nodes, 0_u64..10), 0..nodes * 3);

        (edges, 0..nodes).prop_map(move |(edges, goal)| (Graph::new(nodes, &edges), goal))
    })
}

/// Exact cost from `start` to every node, or `None` where a node is unreachable.
///
/// [`distances_to`] on the reversed graph.
pub fn distances_from(graph: &Graph, start: usize) -> Vec<Option<u64>> {
    distances_to(&graph.reversed(), start)
}

/// Predecessors by scanning every adjacency list; the action is the edge's index in its
/// source's list, as for [`Successors`].
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

/// Asserts that `solution` is a path in `graph` from `start` whose cost is the sum of its
/// edge costs.
pub fn assert_valid_path(graph: &Graph, start: usize, solution: &Solution<usize, usize, u64>) {
    assert_eq!(solution.states[0], start, "path must begin at the start");
    assert_eq!(solution.states.len(), solution.actions.len() + 1);

    let mut cost = 0;
    for (step, &action) in solution.actions.iter().enumerate() {
        let (to, edge_cost) = graph.edges[solution.states[step]][action];
        assert_eq!(
            to,
            solution.states[step + 1],
            "action {step} leads elsewhere"
        );
        cost += edge_cost;
    }

    assert_eq!(cost, solution.cost, "reported cost must match the path");
}

/// A 4-connected grid with blocked cells. Every move costs `1`.
pub struct Grid {
    /// Rows of the grid, top to bottom; `true` marks a wall.
    pub walls: Vec<Vec<bool>>,
}

/// A grid cell as `(row, column)`.
pub type Cell = (usize, usize);

/// A move on a [`Grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// One row up.
    Up,
    /// One row down.
    Down,
    /// One column left.
    Left,
    /// One column right.
    Right,
}

impl Grid {
    /// Parses a grid from text rows, where `#` is a wall and anything else is free.
    pub fn parse(rows: &[&str]) -> Self {
        let walls = rows
            .iter()
            .map(|row| row.chars().map(|c| c == '#').collect())
            .collect();

        Self { walls }
    }

    /// Returns `true` if `cell` lies on the grid and is not a wall.
    fn is_free(&self, (row, column): Cell) -> bool {
        self.walls
            .get(row)
            .and_then(|cells| cells.get(column))
            .is_some_and(|wall| !wall)
    }
}

impl Successors for Grid {
    type State = Cell;
    type Action = Move;
    type Cost = u32;

    fn successors(&self, &(row, column): &Cell, mut emit: impl FnMut(Move, Cell, u32)) {
        let moves = [
            (Move::Up, row.checked_sub(1).map(|row| (row, column))),
            (Move::Down, Some((row + 1, column))),
            (
                Move::Left,
                column.checked_sub(1).map(|column| (row, column)),
            ),
            (Move::Right, Some((row, column + 1))),
        ];

        for (action, target) in moves {
            if let Some(target) = target.filter(|&cell| self.is_free(cell)) {
                emit(action, target, 1);
            }
        }
    }
}

/// The Manhattan distance between two cells: admissible and consistent on a [`Grid`].
pub fn manhattan(from: Cell, to: Cell) -> u32 {
    let distance = from.0.abs_diff(to.0) + from.1.abs_diff(to.1);

    u32::try_from(distance).expect("test grids are small")
}
