//! Multi-heuristic A* ([`mha_star`], Aine, Swaminathan, Narayanan, Hwang and Likhachev
//! 2016): one consistent anchor heuristic for the bound, any number of inadmissible ones
//! for guidance.

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::GraphResult;
use crate::cost::{Weight, within};
use crate::open_list::OpenList;
use crate::outcome::{Outcome, SearchResult, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// The two weights of [`mha_star`]. The solution costs at most `heuristic * anchor` times
/// the optimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MhaWeights {
    /// `w1`: inflates every heuristic in its queue's key `g + w1 * h`.
    pub heuristic: Weight,
    /// `w2`: an inadmissible queue may expand while its smallest key is at most `w2` times
    /// the anchor's.
    pub anchor: Weight,
}

impl MhaWeights {
    /// Both weights `1`: optimal, with inadmissible heuristics only breaking ties.
    pub const ONE: Self = Self {
        heuristic: Weight::ONE,
        anchor: Weight::ONE,
    };
}

/// Shared multi-heuristic A* (SMHA*): Weighted A* over one anchor queue and one queue per
/// inadmissible heuristic, sharing path costs.
///
/// Each queue `i` orders by `g + w1 * h_i`. In round robin over the inadmissible queues,
/// queue `i` expands its best state if its smallest key is at most `w2` times the anchor's
/// smallest key; otherwise the anchor expands. An expanded state leaves every queue, and a
/// state is put back into the anchor queue or the inadmissible queues after an improvement
/// only if it was not yet expanded by the anchor, or by any inadmissible queue,
/// respectively. So every state is expanded at most twice.
///
/// With a consistent `anchor` heuristic the solution costs at most `w1 * w2` times the
/// optimum, whatever the inadmissible `heuristics` say; they need no guarantees, and
/// `None` from one of them only keeps a state out of that queue. `None` from the anchor is
/// a proven dead end. Goals are recognised when they are generated and are not expanded.
/// The search stops once the cheapest goal found has `w1 * g` within the smallest key of
/// the queue about to expand.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, MhaWeights, Successors, Weight, heuristic_fn, mha_star};
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
/// let anchor = heuristic_fn(|s: &i32| Some(7_i32.abs_diff(*s)));
/// let guides = [heuristic_fn(|s: &i32| Some(2 * 7_i32.abs_diff(*s)))];
/// let weights = MhaWeights { heuristic: Weight::integer(2), anchor: Weight::integer(2) };
/// let result = mha_star(&Line, 0, &|s: &i32| *s == 7, anchor, guides, weights, Limits::NONE);
///
/// assert_eq!(result.solution().map(|solution| solution.cost), Some(7));
/// ```
pub fn mha_star<G, T, H0, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    anchor: H0,
    heuristics: impl IntoIterator<Item = H>,
    weights: MhaWeights,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H0: Heuristic<G::State, G::Cost>,
    H: Heuristic<G::State, G::Cost>,
{
    let (space, start_goal) = SearchSpace::seed([start], goal);
    let heuristics: Vec<H> = heuristics.into_iter().collect();
    let queues = (0..=heuristics.len()).map(|_| OpenList::new()).collect();
    let search = Mha {
        graph,
        goal,
        anchor,
        heuristics,
        weights,
        space,
        queues,
        stats: Stats::default(),
        h: Vec::new(),
        in_anchor: Vec::new(),
        in_inadmissible: Vec::new(),
        closed_anchor: Vec::new(),
        closed_inadmissible: Vec::new(),
        incumbent: start_goal,
        buffer: Vec::new(),
    };

    if start_goal.is_some() {
        return search.finish(true);
    }

    search.run(budget)
}

/// Index of the anchor queue in [`Mha::queues`].
const ANCHOR: usize = 0;

/// What a queue's turn in [`Mha::run`] decided.
enum Turn {
    /// A node of the given queue is to be expanded.
    Expand(usize, NodeId),
    /// The incumbent is within the bound; stop with it.
    Done,
}

/// The state of one SMHA* run.
struct Mha<'a, G: Successors, T: ?Sized, H0, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The consistent anchor heuristic.
    anchor: H0,
    /// The inadmissible heuristics; heuristic `i` feeds queue `i + 1`.
    heuristics: Vec<H>,
    /// `w1` and `w2`.
    weights: MhaWeights,
    /// Every state seen so far, with the shared path costs.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// The anchor queue ([`ANCHOR`]) and one queue per inadmissible heuristic, keyed by
    /// `den1 * g + num1 * h_i`.
    queues: Vec<OpenList<G::Cost>>,
    /// Counters for the result.
    stats: Stats,
    /// Per node, the value of every heuristic, anchor first. `None` if the anchor proves the
    /// node a dead end.
    h: Vec<Option<Vec<Option<G::Cost>>>>,
    /// Whether each node is in the anchor queue.
    in_anchor: Vec<bool>,
    /// Whether each node is in the inadmissible queues (all whose heuristic accepted it).
    in_inadmissible: Vec<bool>,
    /// Whether each node was expanded from the anchor queue.
    closed_anchor: Vec<bool>,
    /// Whether each node was expanded from an inadmissible queue.
    closed_inadmissible: Vec<bool>,
    /// The cheapest goal node found so far.
    incumbent: Option<NodeId>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, T, H0, H> Mha<'_, G, T, H0, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H0: Heuristic<G::State, G::Cost>,
    H: Heuristic<G::State, G::Cost>,
{
    /// Runs rounds over the inadmissible queues until the incumbent is within the bound,
    /// the anchor queue runs empty or the budget is used up.
    fn run(mut self, mut budget: impl Budget<Stats>) -> GraphResult<G> {
        for start in 0..self.space.len() {
            self.add_node(start);
        }

        let rounds = self.heuristics.len().max(1);
        for queue in (1..=rounds).cycle() {
            let Some(Turn::Expand(queue, node)) = self.turn(queue) else {
                return self.finish(true);
            };

            if budget.exhausted(&self.stats) {
                return self.finish(false);
            }

            self.expand(queue, node);
        }

        unreachable!("the round robin never ends on its own")
    }

    /// Decides what inadmissible queue `queue` (or the anchor in its stead) does this turn.
    /// `None` if the anchor queue is empty, which ends the search like [`Turn::Done`].
    fn turn(&mut self, queue: usize) -> Option<Turn> {
        let (_, anchor_key) = self.peek(ANCHOR)?;
        let chosen = match self.peek(queue) {
            Some((node, key)) if within(key, self.weights.anchor, anchor_key) => (queue, node, key),
            _ => {
                let (node, key) = self.peek(ANCHOR).expect("the anchor is not empty");
                (ANCHOR, node, key)
            }
        };

        let (queue, node, key) = chosen;
        let done = self.incumbent.is_some_and(|goal| {
            let g = self
                .space
                .g(goal)
                .saturating_times(self.weights.heuristic.denominator());
            g.cmp(&key).is_le()
        });

        Some(if done {
            Turn::Done
        } else {
            Turn::Expand(queue, node)
        })
    }

    /// The best live entry of queue `queue`, if any.
    fn peek(&mut self, queue: usize) -> Option<(NodeId, G::Cost)> {
        let (space, flags) = if queue == ANCHOR {
            (&self.space, &self.in_anchor)
        } else {
            (&self.space, &self.in_inadmissible)
        };

        self.queues
            .get_mut(queue)?
            .peek_where(|node, g| flags[node] && g.cmp(&space.g(node)).is_eq())
    }

    /// Expands `node` from `queue`: takes it out of every queue, marks it closed for that
    /// kind of queue and (re)queues its new and improved successors.
    fn expand(&mut self, queue: usize, node: NodeId) {
        self.stats.expanded += 1;
        self.in_anchor[node] = false;
        self.in_inadmissible[node] = false;
        if queue == ANCHOR {
            self.closed_anchor[node] = true;
        } else {
            self.closed_inadmissible[node] = true;
        }

        self.stats.generated +=
            collect_successors(self.graph, self.space.state(node), &mut self.buffer);
        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match self.space.reach(state, node, action, cost, true) {
                Reach::New(id) => self.add_node(id),
                Reach::Improved(id) => self.improved(id),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
    }

    /// Evaluates the new node `id` with every heuristic and queues it.
    fn add_node(&mut self, id: NodeId) {
        let state = self.space.state(id);
        let values = self.anchor.estimate(state).map(|anchor| {
            let mut values = Vec::with_capacity(self.heuristics.len() + 1);
            values.push(Some(anchor));
            values.extend(self.heuristics.iter_mut().map(|h| h.estimate(state)));
            values
        });

        self.h.push(values);
        self.in_anchor.push(false);
        self.in_inadmissible.push(false);
        self.closed_anchor.push(false);
        self.closed_inadmissible.push(false);
        self.improved(id);
    }

    /// Handles a new or cheaper path to `id`: a better goal, or (re)queueing where the node
    /// has not been expanded yet.
    fn improved(&mut self, id: NodeId) {
        if self.goal.is_goal(self.space.state(id)) {
            let cheaper = self
                .incumbent
                .is_none_or(|goal| goal == id || self.space.g(id).cmp(&self.space.g(goal)).is_lt());
            if cheaper {
                self.incumbent = Some(id);
            }

            return;
        }

        if self.h[id].is_none() {
            return;
        }

        let was_closed = self.closed_anchor[id] || self.closed_inadmissible[id];
        let requeued = !(self.closed_anchor[id] && self.closed_inadmissible[id]);
        if was_closed && requeued {
            self.stats.reopened += 1;
        }

        if !self.closed_anchor[id] {
            self.push(ANCHOR, id);
            self.in_anchor[id] = true;
        }

        if !self.closed_inadmissible[id] {
            let anchor_key = self
                .key(ANCHOR, id)
                .expect("live nodes have an anchor value");
            for queue in 1..self.queues.len() {
                if self
                    .key(queue, id)
                    .is_some_and(|key| within(key, self.weights.anchor, anchor_key))
                {
                    self.push(queue, id);
                    self.in_inadmissible[id] = true;
                }
            }
        }
    }

    /// The key of `id` in queue `queue`, if its heuristic has a value for it.
    fn key(&self, queue: usize, id: NodeId) -> Option<G::Cost> {
        let h = self.h[id].as_ref()?[queue]?;
        let weight = self.weights.heuristic;

        Some(
            self.space
                .g(id)
                .saturating_times(weight.denominator())
                .saturating_add(h.saturating_times(weight.numerator())),
        )
    }

    /// Pushes `id` into queue `queue` with its current key.
    fn push(&mut self, queue: usize, id: NodeId) {
        if let Some(key) = self.key(queue, id) {
            self.queues[queue].push(id, key, self.space.g(id));
        }
    }

    /// Wraps up the run: the incumbent if `decided`, else [`Outcome::LimitReached`].
    fn finish(self, decided: bool) -> GraphResult<G> {
        let outcome = match (decided, self.incumbent) {
            (false, _) => Outcome::LimitReached,
            (true, Some(goal)) => Outcome::Found(self.space.solution(goal)),
            (true, None) => Outcome::Unsolvable,
        };

        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
