//! Fringe search (Björnsson, Enzenberger, Holte, Schaeffer 2005): [`fringe_search`] and
//! [`fringe_search_multi`].
//!
//! Fringe search sits between IDA* and A*. Like IDA* it runs iterations under an `f`
//! threshold, and like A* it keeps every state it has seen (the *cache*: best `g`, parent
//! link and heuristic value), so no state is searched twice from the same path cost. It
//! needs no priority queue: the fringe is two lists, *now* and *later*.
//!
//! ```mermaid
//! flowchart TD
//!     pop[pop the head of now] -->|now empty| swap{later empty?}
//!     swap -->|yes| unsolvable[Unsolvable]
//!     swap -->|no| next[threshold = smallest deferred f; now = later]
//!     next --> pop
//!     pop --> over{f > threshold?}
//!     over -->|yes| defer[move to later]
//!     defer --> pop
//!     over -->|no| goal{goal?}
//!     goal -->|yes| found[Found]
//!     goal -->|no| expand[expand: new or cheaper successors go to the head of now]
//!     expand --> pop
//! ```
//!
//! A state reached again through a cheaper path is moved to the head of *now* (its old list
//! entry is dropped lazily), so the order inside an iteration is depth-first, as in IDA*.

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::GraphResult;
use crate::outcome::{Outcome, SearchResult, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, Reach, SearchSpace};

/// Fringe search from `start` to the first state that satisfies `goal`: optimal with an
/// admissible heuristic, without a priority queue.
///
/// Runs threshold iterations like [`ida_star`](crate::ida_star), starting at `h(start)` and
/// raising the threshold to the smallest `f = g + h` deferred by the last iteration, but
/// caches the best known `g` and parent of every state like [`a_star`](crate::a_star), so a
/// state is expanded again only after a cheaper path to it turns up (counted in
/// [`Stats::reopened`]). Optimal for any admissible `heuristic`; consistency is not needed.
/// `None` marks a dead end, which is never expanded. Goals are tested when a state is taken
/// from the fringe within the threshold.
///
/// Memory grows with the number of states seen, as for A*. Each iteration walks the whole
/// fringe, so many distinct `f` values (such as float costs) mean many iterations.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, fringe_search, heuristic_fn};
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
/// let distance = heuristic_fn(|state: &i32| Some(7_i32.abs_diff(*state)));
/// let result = fringe_search(&Line, 0, &|state: &i32| *state == 7, distance, Limits::NONE);
/// let solution = result.solution().expect("7 is reachable");
///
/// assert_eq!(solution.cost, 7);
/// assert_eq!(solution.actions, [1; 7]);
/// ```
pub fn fringe_search<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    fringe_search_multi(graph, [start], goal, heuristic, budget)
}

/// [`fringe_search`] from several start states at once, to the cheapest goal from any of
/// them.
///
/// Every start has path cost zero and the first threshold is the smallest `h` of any
/// start. Repeated starts count once, an empty set of starts is [`Outcome::Unsolvable`],
/// and a start that is a goal is returned as a zero-cost path without expanding anything.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, fringe_search_multi, heuristic_fn};
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
/// let result = fringe_search_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, distance, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![6, 7, 8]));
/// ```
pub fn fringe_search_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let (space, start_goal) = SearchSpace::seed(starts, goal);
    let mut fringe = Fringe::new(graph, goal, heuristic, space);
    if let Some(found) = start_goal {
        return fringe.finish(Outcome::Found(fringe.space.solution(found)));
    }

    let outcome = fringe.run(budget);

    fringe.finish(outcome)
}

/// The outcome type of searches over graph `G`.
type GraphOutcome<G> =
    Outcome<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;

/// The ticket of a node that is in neither list: it was expanded and not reached again
/// through a cheaper path since, or it is a dead end.
const UNLISTED: u64 = u64::MAX;

/// A list entry: a node and the ticket it was listed with. The entry is live only while
/// the ticket matches the node's current ticket.
type Entry = (NodeId, u64);

/// One fringe search run.
struct Fringe<'a, G: Successors, T: ?Sized, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic, owned for the run.
    heuristic: H,
    /// The cache: every state seen, with its best known `g` and parent link.
    space: SearchSpace<G::State, G::Action, G::Cost>,
    /// Heuristic value per node, by [`NodeId`]; `None` for dead ends.
    h: Vec<Option<G::Cost>>,
    /// Current ticket per node, by [`NodeId`]; [`UNLISTED`] if the node is in no list.
    ticket: Vec<u64>,
    /// Tickets handed out so far; the next ticket.
    tickets: u64,
    /// The *now* list as a stack: the head is the last element.
    now: Vec<Entry>,
    /// The *later* list, in the order states were deferred.
    later: Vec<Entry>,
    /// Counters for the result.
    stats: Stats,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<'a, G, T, H> Fringe<'a, G, T, H>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Prepares a run over the seeded `space`; its nodes are the distinct starts.
    fn new(
        graph: &'a G,
        goal: &'a T,
        heuristic: H,
        space: SearchSpace<G::State, G::Action, G::Cost>,
    ) -> Self {
        Self {
            graph,
            goal,
            heuristic,
            space,
            h: Vec::new(),
            ticket: Vec::new(),
            tickets: 0,
            now: Vec::new(),
            later: Vec::new(),
            stats: Stats::default(),
            buffer: Vec::new(),
        }
    }

    /// Runs threshold iterations until a goal is found, the fringe runs empty or `budget`
    /// is used up.
    fn run(&mut self, mut budget: impl Budget<Stats>) -> GraphOutcome<G> {
        for start in 0..self.space.len() {
            self.evaluate(start);
        }

        self.now.reverse();
        let Some(mut threshold) = self.now.iter().filter_map(|&(id, _)| self.f(id)).min() else {
            return Outcome::Unsolvable;
        };

        loop {
            let deferred = match self.iterate(threshold, &mut budget) {
                Ok(deferred) => deferred,
                Err(outcome) => return outcome,
            };

            let Some(next) = deferred else {
                return Outcome::Unsolvable;
            };

            threshold = next;
            core::mem::swap(&mut self.now, &mut self.later);
            self.now.reverse();
        }
    }

    /// One iteration under `threshold`: works through *now* until it is empty.
    ///
    /// Returns the smallest `f` moved to *later* (`None` if nothing was), or the outcome
    /// that ends the search.
    fn iterate(
        &mut self,
        threshold: G::Cost,
        budget: &mut impl Budget<Stats>,
    ) -> Result<Option<G::Cost>, GraphOutcome<G>> {
        let mut deferred: Option<G::Cost> = None;
        while let Some((id, ticket)) = self.now.pop() {
            if ticket != self.ticket[id] {
                continue;
            }

            let f = self.f(id).expect("dead ends are never listed");
            if f > threshold {
                deferred = Some(deferred.map_or(f, |old| old.min(f)));
                self.later.push((id, ticket));
                continue;
            }

            if self.goal.is_goal(self.space.state(id)) {
                return Err(Outcome::Found(self.space.solution(id)));
            }

            if budget.exhausted(&self.stats) {
                return Err(Outcome::LimitReached);
            }

            self.expand(id);
        }

        Ok(deferred)
    }

    /// Expands `id` and lists its new and improved successors at the head of *now*, the
    /// first reported successor first.
    fn expand(&mut self, id: NodeId) {
        self.stats.expanded += 1;
        self.ticket[id] = UNLISTED;
        self.stats.generated +=
            collect_successors(self.graph, self.space.state(id), &mut self.buffer);

        let first = self.now.len();
        let mut buffer = core::mem::take(&mut self.buffer);
        for (action, state, cost) in buffer.drain(..) {
            match self.space.reach(state, id, action, cost, true) {
                Reach::New(child) => self.evaluate(child),
                Reach::Improved(child) => self.relist(child),
                Reach::Stale => {}
            }
        }

        self.buffer = buffer;
        self.now[first..].reverse();
    }

    /// Computes the heuristic value of the new node `id` and lists it on *now*, unless it
    /// is a dead end.
    fn evaluate(&mut self, id: NodeId) {
        debug_assert_eq!(id, self.h.len(), "nodes are evaluated in creation order");

        let h = self.heuristic.estimate(self.space.state(id));
        self.h.push(h);
        self.ticket.push(UNLISTED);
        if h.is_some() {
            self.list(id);
        }
    }

    /// Lists the known node `id` again on *now* after its path cost improved, dropping its
    /// old entry. Dead ends stay out.
    fn relist(&mut self, id: NodeId) {
        if self.h[id].is_none() {
            return;
        }

        if self.ticket[id] == UNLISTED {
            self.stats.reopened += 1;
        }

        self.list(id);
    }

    /// Pushes a fresh entry for `id` onto *now*; any older entry of `id` turns stale.
    fn list(&mut self, id: NodeId) {
        self.ticket[id] = self.tickets;
        self.now.push((id, self.tickets));
        self.tickets += 1;
    }

    /// `f = g + h` of node `id`, saturating; `None` for dead ends.
    fn f(&self, id: NodeId) -> Option<G::Cost> {
        self.h[id].map(|h| self.space.g(id).saturating_add(h))
    }

    /// Wraps up the run with `outcome`.
    fn finish(&self, outcome: GraphOutcome<G>) -> GraphResult<G> {
        SearchResult {
            outcome,
            stats: self.stats,
        }
    }
}
