//! Beam search: [`beam_search`], breadth-first with a fixed beam width and incomplete,
//! and [`beam_stack_search`], which backtracks over what the beam pruned and is complete
//! and anytime-optimal.
//!
//! | Algorithm             | Memory                 | Complete | Returns                     |
//! | --------------------- | ---------------------- | -------- | --------------------------- |
//! | [`beam_search`]       | every state generated  | no       | some path                   |
//! | [`beam_stack_search`] | `width` per layer      | yes      | cheapest (admissible), anytime |
//!
//! Both search layer by layer: layer `d + 1` holds the successors of the states in layer
//! `d`, and only the `width` states with the smallest `f = g + h` of each layer are kept.

mod stack;

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

pub use stack::{beam_stack_search, beam_stack_search_multi};

use crate::best_first::GraphResult;
use crate::outcome::{Outcome, SearchResult, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// Breadth-first beam search with beam width `width`: fast and memory-light, but
/// incomplete and not optimal.
///
/// Expands the search layer by layer and keeps only the `width` states with the smallest
/// `f = g + h` of each layer (ties go to the state generated first). Every generated state
/// is remembered, kept or pruned, and never generated again from a later layer; inside a
/// layer, a cheaper path to a state replaces the dearer one. States of a layer are expanded
/// in order of `f`, and goals are tested when a state is expanded. `None` from the
/// heuristic marks a dead end, which is never kept.
///
/// The width can prune every path to a goal. If no goal is found, the outcome is
/// [`Outcome::Unsolvable`] only if the width never pruned a state; otherwise it is
/// [`Outcome::Cutoff`], and a wider beam (or [`beam_stack_search`]) may find a goal.
///
/// # Panics
///
/// Panics if `width` is zero.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, beam_search, heuristic_fn};
///
/// /// A line of integers; each step moves one to the left or right at cost 1.
/// struct Line;
///
/// impl Successors for Line {
///     type State = i32;
///     type Action = i32;
///     type Cost = u32;
///
///     fn successors(&self, state: &i32, mut emit: impl FnMut(i32, i32, u32)) {
///         emit(-1, state - 1, 1);
///         emit(1, state + 1, 1);
///     }
/// }
///
/// let distance = heuristic_fn(|state: &i32| Some(5_i32.abs_diff(*state)));
/// let result = beam_search(&Line, 0, &|state: &i32| *state == 5, distance, 1, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(5));
/// assert_eq!(result.stats.expanded, 5);
/// ```
pub fn beam_search<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    width: usize,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    beam_search_multi(graph, [start], goal, heuristic, width, budget)
}

/// [`beam_search`] from several start states at once: the starts form the first layer,
/// pruned to `width` like every other layer.
///
/// Repeated starts count once, an empty set of starts is [`Outcome::Unsolvable`], and a
/// start that is a goal is returned as a zero-cost path without expanding anything.
///
/// # Panics
///
/// Panics if `width` is zero.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, beam_search_multi, heuristic_fn};
///
/// /// Counting up by one, at cost 1.
/// struct Up;
///
/// impl Successors for Up {
///     type State = u32;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut((), u32, u32)) {
///         emit((), n + 1, 1);
///     }
/// }
///
/// let distance = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let result = beam_search_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, distance, 1, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![6, 7, 8]));
/// ```
pub fn beam_search_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    width: usize,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    assert!(width > 0, "the beam width must be at least 1");

    let (space, start_goal) = SearchSpace::seed(starts, goal);
    let mut beam = Beam {
        graph,
        goal,
        heuristic,
        width,
        space,
        h: Vec::new(),
        depth: Vec::new(),
        pruned: false,
        stats: Stats::default(),
        buffer: Vec::new(),
    };
    let outcome = match start_goal {
        Some(found) => Outcome::Found(beam.space.solution(found)),
        None => beam.run(budget),
    };

    SearchResult {
        outcome,
        stats: beam.stats,
    }
}

/// One beam search run.
struct Beam<'a, G: Successors, T: ?Sized, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic, owned for the run.
    heuristic: H,
    /// The number of states kept per layer. Never zero.
    width: usize,
    /// Every state generated so far, kept or pruned.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// Heuristic value per node, by [`NodeId`]; `None` for dead ends.
    h: Vec<Option<G::Cost>>,
    /// The layer each node was generated in, by [`NodeId`].
    depth: Vec<usize>,
    /// Whether the width has pruned a state so far.
    pruned: bool,
    /// Counters for the result.
    stats: Stats,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, T, H> Beam<'_, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Searches layer by layer from the seeded starts, none of which is a goal.
    fn run(&mut self, mut budget: impl Budget<Stats>) -> Outcome<G::State, G::Action, G::Cost> {
        let mut layer: Vec<NodeId> = (0..self.space.len())
            .filter(|&id| self.evaluate(id, 0))
            .collect();

        for depth in 0.. {
            self.keep_best(&mut layer);
            if layer.is_empty() {
                break;
            }

            let mut next = Vec::new();
            for &id in &layer {
                if self.goal.is_goal(self.space.state(id)) {
                    return Outcome::Found(self.space.solution(id));
                }

                if budget.exhausted(&self.stats) {
                    return Outcome::LimitReached;
                }

                self.expand(id, depth + 1, &mut next);
            }

            layer = next;
        }

        if self.pruned {
            Outcome::Cutoff
        } else {
            Outcome::Unsolvable
        }
    }

    /// Generates the successors of `id` into layer `depth`; new states that are not dead
    /// ends join `next`.
    fn expand(&mut self, id: NodeId, depth: usize, next: &mut Vec<NodeId>) {
        self.stats.expanded += 1;
        self.stats.generated +=
            collect_successors(self.graph, self.space.state(id), &mut self.buffer);

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            if self
                .space
                .find(&state)
                .is_some_and(|known| self.depth[known] < depth)
            {
                continue;
            }

            if let Reach::New(child) = self.space.reach(state, id, action, cost, true) {
                if self.evaluate(child, depth) {
                    next.push(child);
                }
            }
        }

        self.buffer = buffer;
    }

    /// Records the heuristic value and layer of the new node `id`; returns `false` for a
    /// dead end.
    fn evaluate(&mut self, id: NodeId, depth: usize) -> bool {
        debug_assert_eq!(id, self.h.len(), "nodes are evaluated in creation order");

        let h = self.heuristic.estimate(self.space.state(id));
        self.h.push(h);
        self.depth.push(depth);

        h.is_some()
    }

    /// Sorts `layer` by `f` (stable, so ties keep generation order) and keeps the first
    /// `width` nodes.
    fn keep_best(&mut self, layer: &mut Vec<NodeId>) {
        layer.sort_by_key(|&id| {
            let h = self.h[id].expect("dead ends never join a layer");

            self.space.g(id).saturating_add(h)
        });

        if layer.len() > self.width {
            layer.truncate(self.width);
            self.pruned = true;
        }
    }
}
