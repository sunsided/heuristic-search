//! Hash-distributed A* ([`hda_star`]): one A* search spread over several threads.
//!
//! Every state has an owner thread, chosen by a hash of the state. Each thread keeps its
//! own open list and its own table of the states it owns, expands its cheapest open state
//! and sends every successor to the successor's owner over a channel, which records it
//! and opens it if the path is new or cheaper. There are no shared tables to lock.
//!
//! Goals are tested when a state is popped, as in [`a_star`](crate::a_star). A popped goal
//! becomes the incumbent if it is cheaper than the current one; it is not expanded, and no
//! thread expands a state whose `f = g + h` is not below the incumbent's cost `U`. With an
//! admissible heuristic the incumbent is optimal once every thread is idle (no open state
//! with `f < U`) and no message is in flight.
//!
//! That moment is detected with one shared counter, `work`: the number of threads that
//! are not idle plus the number of messages sent and not yet recorded. A thread counts a
//! message before sending it and uncounts it after recording it; an idle thread that
//! receives a message counts itself back in first. Only a thread going idle can bring the
//! counter to zero, and once it is zero nothing can raise it again, so the thread that
//! brings it to zero stops the search.
//!
//! Parent links name the owner thread and the node in that thread's table, so the path is
//! put together from the tables of all threads after the search ends.
//!
//! ```mermaid
//! flowchart TD
//!     recv[record received states] --> stop{stopped?}
//!     stop -->|yes| done[return table]
//!     stop -->|no| pop{open state with f < U?}
//!     pop -->|none| idle[go idle, wait for a message]
//!     idle --> recv
//!     pop --> goal{goal?}
//!     goal -->|yes| incumbent[update incumbent U]
//!     incumbent --> recv
//!     goal -->|no| expand[expand, send successors to their owners]
//!     expand --> recv
//! ```

use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread;

use hashbrown::{DefaultHashBuilder, HashTable};

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::GraphResult;
use crate::open_list::OpenList;
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::{Successors, collect_successors};
use crate::space::{NodeId, SearchSpace};

/// How many expansions a thread makes between two checks of the shared budget.
const BUDGET_INTERVAL: u64 = 64;

/// Hash-distributed A* (HDA*) on `threads` threads, from `start` to the cheapest state that
/// satisfies `goal`.
///
/// Finds the same cost as [`a_star`](crate::a_star): optimal with an admissible
/// `heuristic`, also an inconsistent one, because cheaper paths to closed states reopen
/// them. Among several cheapest paths, which one is returned depends on thread timing.
///
/// Each thread works on a clone of `heuristic`, so the heuristic must be [`Clone`] and
/// cannot be lent as `&mut h`; a heuristic that only reads shared data, such as a closure
/// over a reference passed to [`heuristic_fn`](crate::heuristic_fn), clones cheaply. The
/// graph and the goal test are shared between the threads by reference.
///
/// `threads` is the number of worker threads; `0` counts as `1`. The threads are scoped
/// and all have ended when the call returns.
///
/// [`Stats`] are the sums over all threads. A state may be expanded again when a cheaper
/// path to it arrives after its expansion, so [`Stats::reopened`] can be above `0` even
/// with a consistent heuristic. The `budget` is shared by the threads behind a lock, so it
/// must be [`Send`]; it is asked with the summed counters
/// before the first expansion of each thread and then after every 64 expansions of that
/// thread, so a search stopped by [`Limits::max_expansions`](crate::Limits::max_expansions)
/// `= n` has expanded at least `n` states and at most about `n + 64 * threads`. Once the
/// budget is exhausted the search ends with [`Outcome::LimitReached`], even if a solution
/// was found but not yet proven optimal.
///
/// # Panics
///
/// Panics if the graph, the goal test, the heuristic or the budget panics on any thread.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, a_star, hda_star, heuristic_fn};
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
/// let is_goal = |state: &i32| *state == 7;
/// let result = hda_star(&Line, 0, &is_goal, distance, 4, Limits::NONE);
/// let solution = result.solution().expect("7 is reachable");
///
/// assert_eq!(solution.cost, 7);
/// assert_eq!(solution.actions, [1; 7]);
/// ```
pub fn hda_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    threads: usize,
    budget: impl Budget<Stats> + Send,
) -> GraphResult<G>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send,
    G::Action: Clone + Send,
    G::Cost: Send,
    T: Goal<G::State> + Sync + ?Sized,
    H: Heuristic<G::State, G::Cost> + Clone + Send,
{
    hda_star_multi(graph, [start], goal, heuristic, threads, budget)
}

/// [`hda_star`] from several start states at once, to the cheapest goal from any of them.
///
/// Start handling as in [`a_star_multi`](crate::a_star_multi): every start has path cost
/// zero, repeated starts count once, an empty set of starts is [`Outcome::Unsolvable`],
/// and a start that is a goal is returned as a zero-cost path without starting any thread.
///
/// # Panics
///
/// Panics if the graph, the goal test, the heuristic or the budget panics on any thread.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, hda_star_multi, heuristic_fn};
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
///         if *n < 100 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let distance = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let is_goal = |n: &u32| *n == 8;
/// let result = hda_star_multi(&Up, [0, 6, 3], &is_goal, distance, 2, Limits::NONE);
/// let solution = result.into_solution().expect("8 is reachable");
///
/// assert_eq!(solution.states, [6, 7, 8]);
/// assert_eq!(solution.cost, 2);
/// ```
pub fn hda_star_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    threads: usize,
    budget: impl Budget<Stats> + Send,
) -> GraphResult<G>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send,
    G::Action: Clone + Send,
    G::Cost: Send,
    T: Goal<G::State> + Sync + ?Sized,
    H: Heuristic<G::State, G::Cost> + Clone + Send,
{
    let (space, start_goal) = SearchSpace::<G::State, G::Action, G::Cost>::seed(starts, goal);
    if let Some(found) = start_goal {
        return finish(Outcome::Found(space.solution(found)), Stats::default());
    }
    if space.len() == 0 {
        return finish(Outcome::Unsolvable, Stats::default());
    }

    let threads = threads.max(1);
    let owners = DefaultHashBuilder::default();
    let mut partitions: Vec<GraphPartition<G, H>> = (1..threads)
        .map(|index| Partition::new(index, heuristic.clone()))
        .collect();
    partitions.insert(0, Partition::new(0, heuristic));
    for start in 0..space.len() {
        let state = space.state(start).clone();
        let owner = owner_of(&owners, threads, &state);
        partitions[owner].record(state, G::Cost::ZERO, None);
    }

    let (senders, receivers): (Vec<_>, Vec<_>) = (0..threads).map(|_| mpsc::channel()).unzip();
    let shared = Shared {
        graph,
        goal,
        owners,
        senders,
        work: AtomicUsize::new(threads),
        stopped: AtomicBool::new(false),
        limit_reached: AtomicBool::new(false),
        incumbent: Mutex::new(None),
        budget: Mutex::new((Stats::default(), budget)),
    };

    let partitions = run_workers(&shared, partitions, receivers);
    let totals = partitions
        .iter()
        .fold(Stats::default(), |total, partition| {
            total.plus(partition.stats)
        });

    let outcome = if shared.limit_reached.load(Ordering::SeqCst) {
        Outcome::LimitReached
    } else {
        match *lock(&shared.incumbent) {
            Some(incumbent) => Outcome::Found(path(&partitions, incumbent.at)),
            None => Outcome::Unsolvable,
        }
    };

    finish(outcome, totals)
}

/// Runs one worker per partition on scoped threads and returns the partitions once all
/// workers have ended.
fn run_workers<G, T, H, B>(
    shared: &Shared<'_, G, T, B>,
    partitions: Vec<GraphPartition<G, H>>,
    receivers: Vec<Receiver<GraphMessage<G>>>,
) -> Vec<GraphPartition<G, H>>
where
    G: Successors + Sync,
    G::State: Clone + Hash + Eq + Send,
    G::Action: Clone + Send,
    G::Cost: Send,
    T: Goal<G::State> + Sync + ?Sized,
    H: Heuristic<G::State, G::Cost> + Send,
    B: Budget<Stats> + Send,
{
    thread::scope(|scope| {
        let handles: Vec<_> = partitions
            .into_iter()
            .zip(receivers)
            .map(|(partition, inbox)| {
                scope.spawn(move || {
                    let guard = StopOnPanic(shared);
                    let mut worker = Worker {
                        shared,
                        inbox,
                        partition,
                        buffer: Vec::new(),
                    };
                    worker.run();
                    drop(guard);

                    worker.partition
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

/// Wraps up the search with `outcome` and `stats`.
fn finish<S, A, C>(outcome: Outcome<S, A, C>, stats: Stats) -> SearchResult<S, A, C> {
    SearchResult { outcome, stats }
}

/// The thread that owns `state` among `threads` threads.
fn owner_of<S: Hash>(owners: &DefaultHashBuilder, threads: usize, state: &S) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the remainder is below `threads`, a `usize`"
    )]
    let owner = (owners.hash_one(state) % threads as u64) as usize;

    owner
}

/// Locks `mutex`, also when another thread panicked while holding it; the search is then
/// stopping anyway.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Follows the parent links across the partitions from the goal node `goal` back to a start
/// and returns the path. The cost is summed along the links.
fn path<S: Clone, A: Clone, C: Cost, H>(
    partitions: &[Partition<S, A, C, H>],
    goal: Link,
) -> Solution<S, A, C> {
    let mut states = Vec::new();
    let mut actions = Vec::new();
    let mut cost = C::ZERO;
    let mut current = goal;

    loop {
        let node = &partitions[current.thread].nodes[current.node];
        states.push(node.state.clone());
        let Some((parent, action, edge_cost)) = &node.parent else {
            break;
        };

        actions.push(action.clone());
        cost = cost + *edge_cost;
        current = *parent;
    }

    states.reverse();
    actions.reverse();

    Solution {
        states,
        actions,
        cost,
    }
}

/// A node in the table of one thread.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Link {
    /// The thread that owns the node.
    thread: usize,
    /// The node in that thread's table.
    node: NodeId,
}

/// The partition of one thread over graph `G` with heuristic `H`.
type GraphPartition<G, H> =
    Partition<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost, H>;

/// A message between the threads searching graph `G`.
type GraphMessage<G> =
    Message<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;

/// A parent link: the parent node, the edge label and the edge cost.
type Parent<A, C> = (Link, A, C);

/// The cheapest goal found so far.
#[derive(Clone, Copy)]
struct Incumbent<C> {
    /// Its path cost `g`, the bound `U`.
    cost: C,
    /// Where the goal node is stored.
    at: Link,
}

/// A message to a thread.
enum Message<S, A, C> {
    /// A state reached at path cost `g` through the parent link, for its owner to record.
    State(S, C, Parent<A, C>),
    /// The search has ended.
    Stop,
}

/// What all threads share.
struct Shared<'a, G: Successors, T: ?Sized, B> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// Hashes states to their owner thread. Separate from the hashers of the threads'
    /// tables, so a thread's states do not share low hash bits there.
    owners: DefaultHashBuilder,
    /// One channel per thread, by thread index.
    senders: Vec<Sender<GraphMessage<G>>>,
    /// Threads that are not idle plus messages sent and not yet recorded. Zero means the
    /// search has ended.
    work: AtomicUsize,
    /// Set once the search has ended, for whatever reason.
    stopped: AtomicBool,
    /// Set if the budget ended the search.
    limit_reached: AtomicBool,
    /// The cheapest goal found so far.
    incumbent: Mutex<Option<Incumbent<G::Cost>>>,
    /// The counters reported so far by all threads, and the budget asked with them.
    budget: Mutex<(Stats, B)>,
}

impl<G: Successors, T: ?Sized, B> Shared<'_, G, T, B> {
    /// Ends the search and wakes every thread.
    fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        for sender in &self.senders {
            // A thread that has already returned has dropped its receiver; nothing to wake.
            let _ = sender.send(Message::Stop);
        }
    }

    /// The incumbent's cost `U`, if a goal was found.
    fn bound(&self) -> Option<G::Cost> {
        lock(&self.incumbent).map(|incumbent| incumbent.cost)
    }
}

/// Stops the search if the thread holding it unwinds, so the other threads do not wait
/// forever for messages.
struct StopOnPanic<'s, 'a, G: Successors, T: ?Sized, B>(&'s Shared<'a, G, T, B>);

impl<G: Successors, T: ?Sized, B> Drop for StopOnPanic<'_, '_, G, T, B> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.stop();
        }
    }
}

/// A state in the table of one thread.
struct Node<S, A, C> {
    /// The state.
    state: S,
    /// Best known path cost from a start.
    g: C,
    /// Heuristic value, computed once when the node is created. `None` for proven dead
    /// ends, which are never opened.
    h: Option<C>,
    /// Whether the node has been expanded at least once.
    closed: bool,
    /// The node this one was reached from with its best known path. `None` for starts.
    parent: Option<Parent<A, C>>,
}

/// The states one thread owns: their table, its open list and its counters.
struct Partition<S, A, C, H> {
    /// The thread's index.
    index: usize,
    /// The thread's copy of the heuristic.
    heuristic: H,
    /// The nodes in creation order.
    nodes: Vec<Node<S, A, C>>,
    /// Hash index from state to node.
    table: HashTable<NodeId>,
    /// Hashes states for [`Partition::table`].
    hasher: DefaultHashBuilder,
    /// Open nodes keyed by `f = g + h`.
    open: OpenList<C>,
    /// The thread's counters.
    stats: Stats,
    /// Counters not yet reported to the shared budget.
    unreported: Stats,
}

impl<S: Hash + Eq, A, C: Cost, H: Heuristic<S, C>> Partition<S, A, C, H> {
    /// Creates the empty partition of thread `index`.
    fn new(index: usize, heuristic: H) -> Self {
        Self {
            index,
            heuristic,
            nodes: Vec::new(),
            table: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
            open: OpenList::new(),
            stats: Stats::default(),
            unreported: Stats::default(),
        }
    }

    /// Records that `state` is reachable at path cost `g` through `parent`, and opens it if
    /// it is new or the path is cheaper than the known one.
    fn record(&mut self, state: S, g: C, parent: Option<Parent<A, C>>) {
        let hash = self.hasher.hash_one(&state);
        let nodes = &self.nodes;
        let Some(&id) = self.table.find(hash, |&id| nodes[id].state == state) else {
            self.insert(hash, state, g, parent);
            return;
        };

        let node = &mut self.nodes[id];
        if g.cmp(&node.g).is_ge() {
            return;
        }

        node.g = g;
        node.parent = parent;
        if let Some(h) = node.h {
            if node.closed {
                self.stats.reopened += 1;
                self.unreported.reopened += 1;
            }
            self.open.push(id, g.saturating_add(h), g);
        }
    }

    /// Adds a node for the new `state` with hash `hash` and opens it, unless the heuristic
    /// proves it a dead end.
    fn insert(&mut self, hash: u64, state: S, g: C, parent: Option<Parent<A, C>>) {
        let id = self.nodes.len();
        let h = self.heuristic.estimate(&state);
        let (nodes, hasher) = (&self.nodes, &self.hasher);

        self.table
            .insert_unique(hash, id, |&other| hasher.hash_one(&nodes[other].state));
        self.nodes.push(Node {
            state,
            g,
            h,
            closed: false,
            parent,
        });
        if let Some(h) = h {
            self.open.push(id, g.saturating_add(h), g);
        }
    }

    /// Pops the open node with the smallest `f`, if that `f` is below `bound`.
    fn pop_below(&mut self, bound: Option<C>) -> Option<NodeId> {
        let nodes = &self.nodes;
        let (_, f) = self.open.peek_where(|id, g| g.cmp(&nodes[id].g).is_eq())?;
        if bound.is_some_and(|bound| f.cmp(&bound).is_ge()) {
            return None;
        }

        self.open
            .pop_where(|id, g| g.cmp(&nodes[id].g).is_eq())
            .map(|(id, _)| id)
    }
}

/// One thread of the search: its partition and its end of the channels.
struct Worker<'s, 'a, G: Successors, T: ?Sized, H, B> {
    /// What all threads share.
    shared: &'s Shared<'a, G, T, B>,
    /// Messages to this thread.
    inbox: Receiver<GraphMessage<G>>,
    /// The states this thread owns.
    partition: GraphPartition<G, H>,
    /// Successors of the node being expanded; reused between expansions.
    buffer: Vec<(G::Action, G::State, G::Cost)>,
}

impl<G, T, H, B> Worker<'_, '_, G, T, H, B>
where
    G: Successors,
    G::State: Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    B: Budget<Stats>,
{
    /// Searches until the search ends.
    fn run(&mut self) {
        loop {
            if !self.receive_pending() || self.shared.stopped.load(Ordering::SeqCst) {
                return;
            }

            match self.partition.pop_below(self.shared.bound()) {
                Some(id) => self.visit(id),
                None => {
                    if !self.idle() {
                        return;
                    }
                }
            }
        }
    }

    /// Records every message waiting in the inbox. Returns `false` on [`Message::Stop`].
    fn receive_pending(&mut self) -> bool {
        while let Ok(message) = self.inbox.try_recv() {
            if !self.receive(message) {
                return false;
            }
        }

        true
    }

    /// Records one message and uncounts it. Returns `false` on [`Message::Stop`].
    fn receive(&mut self, message: GraphMessage<G>) -> bool {
        let Message::State(state, g, parent) = message else {
            return false;
        };

        self.partition.record(state, g, Some(parent));
        self.shared.work.fetch_sub(1, Ordering::SeqCst);

        true
    }

    /// Goes idle and waits for a message; stops the search if every thread is idle and no
    /// message is in flight. Returns `false` once the search has ended.
    fn idle(&mut self) -> bool {
        if self.shared.work.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.shared.stop();
            return false;
        }

        match self.inbox.recv() {
            Ok(message @ Message::State(..)) => {
                // Count this thread back in before uncounting the message, so `work` cannot
                // touch zero in between.
                self.shared.work.fetch_add(1, Ordering::SeqCst);
                self.receive(message)
            }
            Ok(Message::Stop) | Err(_) => false,
        }
    }

    /// Handles the popped node `id`: a goal becomes the incumbent if it is cheaper, any
    /// other node is expanded unless the budget is exhausted.
    fn visit(&mut self, id: NodeId) {
        let node = &self.partition.nodes[id];
        if self.shared.goal.is_goal(&node.state) {
            let at = Link {
                thread: self.partition.index,
                node: id,
            };
            let mut incumbent = lock(&self.shared.incumbent);
            if incumbent.is_none_or(|incumbent| node.g.cmp(&incumbent.cost).is_lt()) {
                *incumbent = Some(Incumbent { cost: node.g, at });
            }
            return;
        }

        if self.budget_exhausted() {
            self.shared.limit_reached.store(true, Ordering::SeqCst);
            self.shared.stop();
            return;
        }

        self.expand(id);
    }

    /// Reports the counters to the shared budget and asks it, before the first expansion
    /// of this thread and then every [`BUDGET_INTERVAL`] expansions.
    fn budget_exhausted(&mut self) -> bool {
        if self.partition.stats.expanded % BUDGET_INTERVAL != 0 {
            return false;
        }

        let mut budget = lock(&self.shared.budget);
        let (total, budget) = &mut *budget;
        *total = total.plus(core::mem::take(&mut self.partition.unreported));

        budget.exhausted(total)
    }

    /// Expands node `id` and sends each successor to its owner; successors this thread
    /// owns are recorded at once.
    fn expand(&mut self, id: NodeId) {
        let partition = &mut self.partition;
        partition.nodes[id].closed = true;
        let g = partition.nodes[id].g;
        let generated = collect_successors(
            self.shared.graph,
            &partition.nodes[id].state,
            &mut self.buffer,
        );
        for stats in [&mut partition.stats, &mut partition.unreported] {
            stats.expanded += 1;
            stats.generated += generated;
        }

        let threads = self.shared.senders.len();
        let from = Link {
            thread: partition.index,
            node: id,
        };
        for (action, state, cost) in self.buffer.drain(..) {
            let parent = (from, action, cost);
            let owner = owner_of(&self.shared.owners, threads, &state);
            if owner == partition.index {
                partition.record(state, g + cost, Some(parent));
                continue;
            }

            self.shared.work.fetch_add(1, Ordering::SeqCst);
            if self.shared.senders[owner]
                .send(Message::State(state, g + cost, parent))
                .is_err()
            {
                // The owner has returned, so the search has ended; the count is moot.
                return;
            }
        }
    }
}
