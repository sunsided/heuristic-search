//! Dynamic Potential Search ([`dps`], Gilon, Felner and Stern 2016).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::hash::Hash;

use crate::best_first::GraphResult;
use crate::cost::{Cost, Ordered, Weight, within};
use crate::keyed_set::KeyedSet;
use crate::outcome::{Budget, Outcome, SearchResult, Stats};
use crate::problem::{Goal, Heuristic, Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// Dynamic Potential Search (DPS): bounded suboptimal search without a second estimate,
/// within `weight` times the optimum.
///
/// Expands the open state with the largest potential `(w * f_min - g) / h`, the state most
/// likely to lead to a solution within the bound, where `f_min` is the smallest
/// `f = g + h` open. States with `h = 0` have infinite potential. Only states with
/// `f <= w * f_min` are candidates, so with an admissible `heuristic` the solution costs at
/// most `w` times the optimum.
///
/// The potential changes whenever `f_min` does, so open states are grouped by their
/// `(g, h)` pair, as in the paper, and every expansion scans the groups. That is cheap when
/// costs are small integers and few distinct pairs exist, and linear in the open list in
/// the worst case. Potentials are compared in `f64` (see [`Cost::to_f64`]); the bound is
/// checked exactly. Ties go to larger `g`.
///
/// # Panics
///
/// Panics if `weight` is below `1`.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Weight, dps, heuristic_fn};
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
/// let distance = heuristic_fn(|state: &i32| Some(6_i32.abs_diff(*state)));
/// let result = dps(&Line, 0, &|s: &i32| *s == 6, distance, Weight::new(3, 2), Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.cost), Some(6));
/// ```
pub fn dps<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    weight: Weight,
    budget: impl Budget,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    assert!(
        weight.numerator() >= weight.denominator(),
        "DPS weight must be at least 1"
    );

    let (space, start_goal) = SearchSpace::seed([start], goal);
    let search = Dps {
        graph,
        heuristic,
        weight,
        space,
        by_f: KeyedSet::new(),
        groups: BTreeMap::new(),
        stats: Stats::default(),
        h: Vec::new(),
        closed: Vec::new(),
        buffer: Vec::new(),
    };

    if let Some(found) = start_goal {
        let solution = search.space.solution(found);
        return search.finish(Outcome::Found(solution));
    }

    search.run(goal, budget)
}

/// A pair of costs: a `(g, h)` group key, or an `(f, g)` order key.
type Pair<C> = (Ordered<C>, Ordered<C>);

/// The state of one DPS run.
struct Dps<'a, G: Successors, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The admissible heuristic, owned for the run.
    heuristic: H,
    /// The suboptimality bound.
    weight: Weight,
    /// Every state seen so far.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// Open states by `(f, g)`, which gives `f_min`. The key also records each open node's
    /// `g`, so the node's group can be found again.
    by_f: KeyedSet<Pair<G::Cost>>,
    /// Open states grouped by their `(g, h)` pair. Never holds an empty group.
    groups: BTreeMap<Pair<G::Cost>, BTreeSet<NodeId>>,
    /// Counters for the result.
    stats: Stats,
    /// Heuristic value per node; `None` for proven dead ends, which are never opened.
    h: Vec<Option<G::Cost>>,
    /// Whether each node has been expanded at least once.
    closed: Vec<bool>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, H> Dps<'_, G, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    H: Heuristic<G::State, G::Cost>,
{
    /// Expands until a goal is expanded, nothing is open or the budget is used up.
    fn run<T: Goal<G::State> + ?Sized>(
        mut self,
        goal: &T,
        mut budget: impl Budget,
    ) -> GraphResult<G> {
        for start in 0..self.space.len() {
            self.add_node(start);
        }

        while let Some(node) = self.select() {
            self.close(node);
            if goal.is_goal(self.space.state(node)) {
                let solution = self.space.solution(node);
                return self.finish(Outcome::Found(solution));
            }

            if budget.exhausted(&self.stats) {
                return self.finish(Outcome::LimitReached);
            }

            self.expand(node);
        }

        self.finish(Outcome::Unsolvable)
    }

    /// The open node of largest potential among those with `f <= w * f_min`.
    fn select(&self) -> Option<NodeId> {
        let (key, _) = self.by_f.first()?;
        let f_min = key.0.0;
        let budget = f_min.to_f64() * f64::from(self.weight.numerator())
            / f64::from(self.weight.denominator());

        let mut best: Option<(f64, NodeId)> = None;
        for (&(Ordered(g), Ordered(h)), nodes) in self.groups.iter().rev() {
            if !within(g + h, self.weight, f_min) {
                continue;
            }

            let potential = potential(budget, g, h);
            if best.is_none_or(|(top, _)| potential > top) {
                best = nodes.first().map(|&node| (potential, node));
            }
        }

        best.map(|(_, node)| node)
    }

    /// Takes `node` off the open list.
    fn close(&mut self, node: NodeId) {
        let Some((_, Ordered(g))) = self.by_f.remove(node) else {
            return;
        };

        let h = self.h[node].expect("open nodes are live");
        let pair = (Ordered(g), Ordered(h));
        if let Some(group) = self.groups.get_mut(&pair) {
            group.remove(&node);
            if group.is_empty() {
                self.groups.remove(&pair);
            }
        }
    }

    /// Generates the successors of `node` and opens the new and improved ones.
    fn expand(&mut self, node: NodeId) {
        self.stats.expanded += 1;
        self.closed[node] = true;
        self.stats.generated +=
            collect_successors(self.graph, self.space.state(node), &mut self.buffer);

        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match self.space.reach(state, node, action, cost, true) {
                Reach::New(id) => self.add_node(id),
                Reach::Improved(id) => self.reopen(id),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
    }

    /// Evaluates the new node `id` and opens it, unless it is a dead end.
    fn add_node(&mut self, id: NodeId) {
        let h = self.heuristic.estimate(self.space.state(id));

        self.h.push(h);
        self.closed.push(false);
        self.open(id);
    }

    /// Opens the known node `id` again after its path cost improved.
    fn reopen(&mut self, id: NodeId) {
        if self.h[id].is_some() && self.closed[id] && self.by_f.key(id).is_none() {
            self.stats.reopened += 1;
        }

        self.close(id);
        self.open(id);
    }

    /// Opens `id` with its current `g`, unless it is a dead end.
    fn open(&mut self, id: NodeId) {
        let Some(h) = self.h[id] else {
            return;
        };

        let g = self.space.g(id);
        self.by_f.insert(id, (Ordered(g + h), Ordered(g)));
        self.groups
            .entry((Ordered(g), Ordered(h)))
            .or_default()
            .insert(id);
    }

    /// Wraps up the run with `outcome`.
    fn finish(self, outcome: Outcome<G::State, G::Action, G::Cost>) -> GraphResult<G> {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}

/// The potential `(budget - g) / h` of a group; infinite for `h = 0`.
fn potential<C: Cost>(budget: f64, g: C, h: C) -> f64 {
    let h = h.to_f64();
    if h > 0.0 {
        (budget - g.to_f64()) / h
    } else {
        f64::INFINITY
    }
}
