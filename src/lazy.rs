//! Lazy shortest paths for graphs whose true edge costs are expensive to compute
//! ([`lazy_sp`], Dellin and Srinivasa 2016), with an [`EdgeEvaluator`] for the true costs.

use alloc::vec::Vec;
use core::hash::{BuildHasher, Hash};

use hashbrown::{DefaultHashBuilder, HashTable};

use search_core::{Budget, Goal, Heuristic};

use crate::best_first::{GraphResult, a_star_multi};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::Successors;

/// Computes the true cost of an edge, which is expensive (a collision check, a
/// simulation).
///
/// An edge is identified by its source state and its action, so the action must tell the
/// outgoing edges of a state apart. [`lazy_sp`] evaluates each edge at most once.
pub trait EdgeEvaluator<S, A, C> {
    /// The true cost of the edge `from --action--> to`, or `None` if the edge is not
    /// traversable after all.
    fn evaluate(&mut self, from: &S, action: &A, to: &S) -> Option<C>;
}

impl<S, A, C, F: FnMut(&S, &A, &S) -> Option<C>> EdgeEvaluator<S, A, C> for F {
    fn evaluate(&mut self, from: &S, action: &A, to: &S) -> Option<C> {
        self(from, action, to)
    }
}

/// Which unevaluated edges of a candidate path [`lazy_sp`] evaluates next.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum EdgeSelector {
    /// The first unevaluated edge from the start. Good when obstacles are near the start.
    #[default]
    Forward,
    /// The last unevaluated edge, nearest the goal.
    Reverse,
    /// Forward and reverse in turn.
    Alternate,
    /// The unevaluated edge in the middle of the unevaluated ones; splits long uncertain
    /// stretches first.
    Bisection,
    /// Every unevaluated edge of the path at once (lazy weighted A* in effect).
    All,
}

/// Lazy shortest path (LazySP): the cheapest path while evaluating as few true edge costs
/// as possible.
///
/// The graph's [`Successors`] report a cheap *lazy* cost for every edge. `LazySP` repeatedly
/// finds the cheapest path with A*, using the true cost for edges already evaluated
/// (dropping infeasible ones) and the lazy cost for the rest. If every edge on that path
/// is evaluated, it is the answer. Otherwise the `selector` picks unevaluated edges of the
/// path, `evaluator` computes their true costs, and the search repeats.
///
/// The solution is optimal if every lazy cost is at most the edge's true cost and the
/// `heuristic` is admissible for the true costs. [`Stats::evaluations`] counts the true
/// costs computed; the other counters add up all inner A* runs. The budget sees those
/// running totals.
///
/// # Examples
///
/// ```
/// use heuristic_search::{EdgeSelector, Limits, Successors, Zero, lazy_sp};
///
/// /// A diamond: `0 -> 1 -> 3` and `0 -> 2 -> 3`, every edge lazily costing 1.
/// struct Diamond;
///
/// impl Successors for Diamond {
///     type State = u8;
///     type Action = u8;
///     type Cost = u32;
///
///     fn successors(&self, n: &u8, mut emit: impl FnMut(u8, u8, u32)) {
///         let targets: &[u8] = match n {
///             0 => &[1, 2],
///             1 | 2 => &[3],
///             _ => &[],
///         };
///         for &target in targets {
///             emit(target, target, 1);
///         }
///     }
/// }
///
/// // The edge 0 -> 1 is blocked; everything else costs 1.
/// let evaluate = |from: &u8, to: &u8, _: &u8| ((*from, *to) != (0, 1)).then_some(1);
/// let result = lazy_sp(&Diamond, 0, &|n: &u8| *n == 3, Zero, evaluate, EdgeSelector::Forward, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![0, 2, 3]));
/// assert_eq!(result.stats.evaluations, 3);
/// ```
pub fn lazy_sp<G, T, H, E>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    evaluator: E,
    selector: EdgeSelector,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone + Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    E: EdgeEvaluator<G::State, G::Action, G::Cost>,
{
    lazy_sp_multi(graph, [start], goal, heuristic, evaluator, selector, budget)
}

/// [`lazy_sp`] from several start states at once.
///
/// Every start has path cost zero; each round runs [`a_star_multi`] from
/// all of them over the lazily evaluated graph. Repeated starts count once, an empty set of
/// starts finds nothing, and a start that is a goal is returned without any evaluation.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Zero, lazy_sp_multi, EdgeSelector};
///
/// /// Counting up by one or two; lazily every edge costs 1.
/// struct Up;
///
/// impl Successors for Up {
///     type State = u32;
///     type Action = u32;
///     type Cost = u32;
///
///     fn successors(&self, n: &u32, mut emit: impl FnMut(u32, u32, u32)) {
///         emit(1, n + 1, 1);
///         emit(2, n + 2, 1);
///     }
/// }
///
/// // Every edge out of 6 truly costs 10; the others cost 1. Start 6 looks closest, but
/// // the cheapest path starts at 3 and jumps over 6.
/// let evaluate = |from: &u32, _: &u32, _: &u32| Some(if *from == 6 { 10 } else { 1 });
/// let result = lazy_sp_multi(&Up, [6, 3], &|n: &u32| *n == 8, Zero, evaluate, EdgeSelector::All, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(3));
/// assert_eq!(result.solution().map(|s| s.states[0]), Some(3));
/// ```
#[expect(
    clippy::similar_names,
    reason = "`starts` is the parameter name of every `*_multi` search"
)]
pub fn lazy_sp_multi<G, T, H, E>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    mut heuristic: H,
    mut evaluator: E,
    selector: EdgeSelector,
    mut budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Hash + Eq,
    G::Action: Clone + Hash + Eq,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
    E: EdgeEvaluator<G::State, G::Action, G::Cost>,
{
    let starts: Vec<G::State> = starts.into_iter().collect();
    let mut view = Evaluated {
        graph,
        edges: HashTable::new(),
        hasher: DefaultHashBuilder::default(),
    };
    let mut stats = Stats::default();

    for round in 0_usize.. {
        let inner = a_star_multi(
            &view,
            starts.iter().cloned(),
            goal,
            &mut heuristic,
            |inner: &Stats| budget.exhausted(&stats.plus(*inner)),
        );
        stats = stats.plus(inner.stats);

        let solution = match inner.outcome {
            Outcome::Found(solution) => solution,
            outcome => return SearchResult { outcome, stats },
        };

        let pending = view.unevaluated(&solution);
        if pending.is_empty() {
            return SearchResult {
                outcome: Outcome::Found(solution),
                stats,
            };
        }

        for index in select(selector, &pending, round) {
            let (from, action, to) = (
                &solution.states[index],
                &solution.actions[index],
                &solution.states[index + 1],
            );
            let cost = evaluator.evaluate(from, action, to);
            stats.evaluations += 1;
            view.record(from.clone(), action.clone(), cost);
        }
    }

    unreachable!("the rounds never run out")
}

/// The indices into `pending` (edge indices along the path, ascending, never empty) that
/// `selector` evaluates in round `round`.
fn select(selector: EdgeSelector, pending: &[usize], round: usize) -> Vec<usize> {
    let from_start = match selector {
        EdgeSelector::Forward => true,
        EdgeSelector::Reverse => false,
        EdgeSelector::Alternate => round % 2 == 0,
        EdgeSelector::Bisection => return alloc::vec![pending[pending.len() / 2]],
        EdgeSelector::All => return pending.to_vec(),
    };

    let end = if from_start {
        pending.first()
    } else {
        pending.last()
    };

    end.copied().into_iter().collect()
}

/// An evaluated edge: source, action and true cost (`None` if infeasible).
type Evaluation<G> = (
    <G as Successors>::State,
    <G as Successors>::Action,
    Option<<G as Successors>::Cost>,
);

/// What is known about an edge's cost.
enum Known<C> {
    /// Not evaluated yet; only the lazy cost is known.
    Lazy,
    /// Evaluated with this true cost.
    True(C),
    /// Evaluated and not traversable.
    Infeasible,
}

/// The graph as `LazySP`'s inner A* sees it: true costs for evaluated edges, lazy costs for
/// the rest, and no infeasible edges.
struct Evaluated<'a, G: Successors> {
    /// The graph with lazy costs.
    graph: &'a G,
    /// Every evaluated edge as `(source, action, true cost)`; `None` marks an infeasible
    /// edge. Hashed by `(source, action)`.
    edges: HashTable<Evaluation<G>>,
    /// Hashes `(source, action)` pairs for [`Evaluated::edges`].
    hasher: DefaultHashBuilder,
}

impl<G> Evaluated<'_, G>
where
    G: Successors,
    G::State: Hash + Eq,
    G::Action: Hash + Eq,
{
    /// What is known about the cost of the edge `(from, action)`.
    fn lookup(&self, from: &G::State, action: &G::Action) -> Known<G::Cost> {
        let hash = self.hasher.hash_one((from, action));
        let found = self.edges.find(hash, |(s, a, _)| s == from && a == action);

        match found {
            None => Known::Lazy,
            Some((_, _, Some(cost))) => Known::True(*cost),
            Some((_, _, None)) => Known::Infeasible,
        }
    }

    /// Records the true cost of the edge `(from, action)`.
    fn record(&mut self, from: G::State, action: G::Action, cost: Option<G::Cost>) {
        let hash = self.hasher.hash_one((&from, &action));
        let hasher = &self.hasher;

        self.edges
            .insert_unique(hash, (from, action, cost), |(s, a, _)| {
                hasher.hash_one((s, a))
            });
    }

    /// The indices of the edges of `path` that are not evaluated yet, ascending.
    fn unevaluated(&self, path: &Solution<G::State, G::Action, G::Cost>) -> Vec<usize> {
        path.actions
            .iter()
            .enumerate()
            .filter(|(index, action)| {
                matches!(self.lookup(&path.states[*index], action), Known::Lazy)
            })
            .map(|(index, _)| index)
            .collect()
    }
}

impl<G> Successors for Evaluated<'_, G>
where
    G: Successors,
    G::State: Hash + Eq,
    G::Action: Hash + Eq,
{
    type State = G::State;
    type Action = G::Action;
    type Cost = G::Cost;

    fn successors(&self, state: &G::State, mut emit: impl FnMut(G::Action, G::State, G::Cost)) {
        self.graph.successors(state, |action, next, lazy| {
            match self.lookup(state, &action) {
                Known::Lazy => emit(action, next, lazy),
                Known::True(cost) => emit(action, next, cost),
                Known::Infeasible => {}
            }
        });
    }
}
