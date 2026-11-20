//! Real-time search with a lookahead of one expansion: [`lrta_star`] (Korf's LRTA*) and
//! [`rta_star`] (RTA*). Both look only at the successors of the current state, update its
//! value and move to the best successor.

use alloc::vec::Vec;
use core::hash::Hash;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::outcome::Stats;
use crate::problem::{Successors, collect_successors};
use crate::real_time::{End, LearnedHeuristic, Trajectory, TrialResult, TrialStats};

/// The result type of real-time trials over graph `G`.
type GraphTrial<G> =
    TrialResult<<G as Successors>::State, <G as Successors>::Action, <G as Successors>::Cost>;

/// One trial of LRTA* (Korf 1990) from `start` to a state that satisfies `goal`, learning
/// in `table`.
///
/// In every state the agent looks at each successor `s'`, sets
/// `h(s) = min c(s, s') + h(s')` and moves to the successor that attains the minimum (the
/// first one on ties). The update is Korf's: the minimum, not its maximum with the old
/// value. From an admissible table it stays admissible, from a consistent one consistent
/// (then values only grow). A state whose successors are all dead ends becomes one.
///
/// Each step expands one state ([`Stats::expanded`]), and the [`Budget`] is asked before
/// each. On a finite graph with positive edge costs from which the goal stays reachable,
/// every trial reaches it, and repeated trials with the same table converge to a cheapest
/// path when the table starts admissible.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Zero, lrta_star};
///
/// /// A corridor `0..=5` with unit moves both ways.
/// struct Corridor;
///
/// impl Successors for Corridor {
///     type State = u8;
///     type Action = i8;
///     type Cost = u32;
///
///     fn successors(&self, cell: &u8, mut emit: impl FnMut(i8, u8, u32)) {
///         if *cell < 5 {
///             emit(1, cell + 1, 1);
///         }
///         if let Some(left) = cell.checked_sub(1) {
///             emit(-1, left, 1);
///         }
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let at_end = |cell: &u8| *cell == 5;
///
/// let first = lrta_star(&Corridor, 0, &at_end, &mut table, Limits::expansions(100));
/// assert!(first.solution().is_some());
///
/// let mut trials = 1;
/// while lrta_star(&Corridor, 0, &at_end, &mut table, Limits::expansions(100)).trial.updates > 0 {
///     trials += 1;
/// }
///
/// assert!(trials < 10);
/// assert_eq!(table.get(&0), Some(5), "converged to the exact cost");
/// ```
pub fn lrta_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    run(graph, start, goal, table, Rule::Lrta, budget)
}

/// One trial of RTA* (Korf 1990) from `start` to a state that satisfies `goal`, learning
/// in `table`.
///
/// Like [`lrta_star`], but the value stored for the state the agent leaves is the
/// *second-best* `c(s, s') + h(s')` among its successors: the cost of coming back and
/// trying the best alternative. That makes the agent avoid backtracking within a trial,
/// but the values overestimate, so the table is not admissible and is meant for the
/// current trial only (repeated trials need not converge; use [`lrta_star`] for that).
///
/// Korf stores infinity when a state has a single successor. In a directed graph that
/// would mark the only route through that state as a dead end, so a state with fewer than
/// two live successors stores the best value instead.
///
/// On a finite graph with positive edge costs from which the goal stays reachable, every
/// trial reaches it.
///
/// # Examples
///
/// ```
/// use heuristic_search::{LearnedHeuristic, Limits, Successors, Zero, rta_star};
///
/// /// A ring of 6 states; moving forward costs 1, back costs 1.
/// struct Ring;
///
/// impl Successors for Ring {
///     type State = u8;
///     type Action = ();
///     type Cost = u32;
///
///     fn successors(&self, state: &u8, mut emit: impl FnMut((), u8, u32)) {
///         emit((), (state + 1) % 6, 1);
///         emit((), (state + 5) % 6, 1);
///     }
/// }
///
/// let mut table = LearnedHeuristic::new(Zero);
/// let result = rta_star(&Ring, 0, &|state: &u8| *state == 3, &mut table, Limits::expansions(50));
///
/// assert_eq!(result.solution().map(|path| path.cost), Some(3));
/// ```
pub fn rta_star<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    run(graph, start, goal, table, Rule::Rta, budget)
}

/// Which value a one-step search stores for the state it leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rule {
    /// The best `c + h` (LRTA*).
    Lrta,
    /// The second-best `c + h`, or the best with fewer than two live successors (RTA*).
    Rta,
}

/// The successors of one state, evaluated.
struct Choice<C> {
    /// Index of the best successor in the successor buffer; the first on ties.
    best: usize,
    /// Its `c + h`.
    best_value: C,
    /// The second-smallest `c + h`, if there are two live successors.
    second_value: Option<C>,
}

impl<C: Cost> Choice<C> {
    /// The value `rule` stores for the state being left.
    fn learned(&self, rule: Rule) -> C {
        match rule {
            Rule::Lrta => self.best_value,
            Rule::Rta => self.second_value.unwrap_or(self.best_value),
        }
    }
}

/// Runs one trial of a one-step search with `rule`.
fn run<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    table: &mut LearnedHeuristic<G::State, G::Cost, H>,
    rule: Rule,
    mut budget: impl Budget<Stats>,
) -> GraphTrial<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let mut path = Trajectory::new(start);
    let mut stats = Stats::default();
    let mut trial = TrialStats::default();
    let mut buffer = Vec::new();

    let end = loop {
        if goal.is_goal(path.current()) {
            break End::Goal;
        }

        if budget.exhausted(&stats) {
            break End::Limit;
        }

        stats.expanded += 1;
        stats.generated += collect_successors(graph, path.current(), &mut buffer);

        let choice = evaluate(&buffer, table);
        let value = choice.as_ref().map(|choice| choice.learned(rule));
        let changed = table.set(path.current().clone(), value);
        trial.record_step(1, u64::from(changed));

        let Some(choice) = choice else {
            break End::Unsolvable;
        };

        let (action, next, cost) = buffer.swap_remove(choice.best);
        path.travel(action, next, cost);
        trial.moves += 1;
        buffer.clear();
    };

    path.finish(end, stats, trial)
}

/// Evaluates `c + h` for every successor in `buffer`; `None` if all are dead ends.
fn evaluate<S: Hash + Eq, A, C: Cost, H: Heuristic<S, C>>(
    buffer: &[(A, S, C)],
    table: &mut LearnedHeuristic<S, C, H>,
) -> Option<Choice<C>> {
    let mut choice: Option<Choice<C>> = None;

    for (index, (_, next, cost)) in buffer.iter().enumerate() {
        let Some(h) = table.get(next) else {
            continue;
        };

        let value = cost.saturating_add(h);
        choice = Some(match choice {
            None => Choice {
                best: index,
                best_value: value,
                second_value: None,
            },
            Some(old) if value < old.best_value => Choice {
                best: index,
                best_value: value,
                second_value: Some(old.best_value),
            },
            Some(old) => Choice {
                second_value: Some(old.second_value.map_or(value, |second| second.min(value))),
                ..old
            },
        });
    }

    choice
}
