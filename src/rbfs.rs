//! Recursive best-first search (RBFS, Korf 1993): [`rbfs`] and [`rbfs_multi`].
//!
//! RBFS is a best-first tree search in memory linear in the search depth. It descends into
//! the child with the smallest backed-up value `F` as long as that value stays within a
//! bound, the smaller of the parent's bound and the second-best sibling's `F`. A subtree
//! that exceeds its bound returns its smallest leaf `F` to the parent, which remembers it
//! and switches to the sibling that is now best. A re-expanded node hands its remembered
//! `F` down to children whose own `f` is smaller, so a subtree that was already searched is
//! crossed again without re-sorting it leaf by leaf.
//!
//! The recursion of the paper is an explicit stack here: each frame holds one node of the
//! current path with its bound and its children's `F`, so deep paths cannot overflow the
//! call stack.
//!
//! ```mermaid
//! flowchart TD
//!     pick[pick best child of the top frame] --> within{F best <= bound and finite?}
//!     within -->|no| back[pop: store F best in the parent's entry]
//!     back -->|stack empty| unsolvable[Unsolvable]
//!     back --> pick
//!     within -->|yes| goal{goal?}
//!     goal -->|yes| found[Found]
//!     goal -->|no| push[expand; push frame with bound min of bound and F second]
//!     push --> pick
//! ```

use alloc::vec::Vec;

use search_core::{Budget, Cost, Goal, Heuristic};

use crate::best_first::GraphResult;
use crate::cost::Extended;
use crate::depth_first::{distinct, goal_root};
use crate::outcome::{Outcome, SearchResult, Solution, Stats};
use crate::problem::{Successors, collect_successors};

/// Recursive best-first search (RBFS) from `start`: cheapest path with an admissible
/// heuristic, in memory linear in the search depth.
///
/// Expands nodes in best-first order with respect to their backed-up `f` values while
/// keeping only the current path and its siblings. Optimal for any admissible `heuristic`
/// (consistency is not needed); `None` marks a dead end. Like [`ida_star`](crate::ida_star)
/// it is a tree search: it skips a successor already on the current path, so states need
/// only `Eq` and zero-cost cycles are harmless, but a state reached along several paths is
/// searched along each. It regenerates subtrees when it returns to them, counted in
/// [`Stats::expanded`] every time; with many distinct `f` values (such as float costs) it
/// switches between subtrees often. Goals are tested when the search descends into a node
/// (and on the start).
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, rbfs};
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
/// let distance = heuristic_fn(|s: &i32| Some(6_i32.abs_diff(*s)));
/// let result = rbfs(&Line, 0, &|s: &i32| *s == 6, distance, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.cost), Some(6));
/// assert_eq!(result.stats.expanded, 6);
/// ```
pub fn rbfs<G, T, H>(
    graph: &G,
    start: G::State,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    rbfs_multi(graph, [start], goal, heuristic, budget)
}

/// [`rbfs`] from several start states at once, to the cheapest goal from any of them.
///
/// The starts hang below a virtual root with an infinite bound, so the search moves
/// between them by their backed-up values like between any other siblings.
///
/// Repeated starts count once, an empty set of starts is [`Outcome::Unsolvable`], and a
/// start that is a goal is returned as a zero-cost path.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, heuristic_fn, rbfs_multi};
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
///         if *n < 8 {
///             emit((), n + 1, 1);
///         }
///     }
/// }
///
/// let h = heuristic_fn(|n: &u32| 8_u32.checked_sub(*n));
/// let result = rbfs_multi(&Up, [0, 6, 3], &|n: &u32| *n == 8, h, Limits::NONE);
///
/// assert_eq!(result.solution().map(|s| s.states.clone()), Some(vec![6, 7, 8]));
/// ```
pub fn rbfs_multi<G, T, H>(
    graph: &G,
    starts: impl IntoIterator<Item = G::State>,
    goal: &T,
    heuristic: H,
    budget: impl Budget<Stats>,
) -> GraphResult<G>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    let roots = distinct(starts);
    if let Some(solution) = goal_root(&roots, goal) {
        return SearchResult {
            outcome: Outcome::Found(solution),
            stats: Stats::default(),
        };
    }

    let mut search = Rbfs {
        graph,
        goal,
        heuristic,
        stats: Stats::default(),
    };
    let outcome = search.run(roots, budget);

    SearchResult {
        outcome,
        stats: search.stats,
    }
}

/// A node below a frame (or below the virtual root): a child of the frame's node.
struct Child<S, A, C> {
    /// The state.
    state: S,
    /// The edge label and cost that led here; `None` for a start.
    via: Option<(A, C)>,
    /// The path cost from the start.
    g: C,
    /// The static value `g + h`; infinite for a dead end.
    f: Extended<C>,
    /// The backed-up value `F`: `f` until the child's subtree was searched, then the
    /// smallest `F` of the leaves it returned with.
    value: Extended<C>,
}

/// One node of the current path, with the bound it is searched under and its children.
struct Frame<S, A, C> {
    /// The node itself, as its parent lists it (its `value` as of the descent).
    node: Child<S, A, C>,
    /// The position of [`Frame::node`] in its parent's children.
    index: usize,
    /// The bound: the subtree returns once its best `F` exceeds this.
    bound: Extended<C>,
    /// The children not on the path, in the order the graph reported them.
    children: Vec<Child<S, A, C>>,
}

/// One RBFS run over a graph, a goal and a heuristic.
struct Rbfs<'a, G, T: ?Sized, H> {
    /// The graph being searched.
    graph: &'a G,
    /// The goal test.
    goal: &'a T,
    /// The heuristic, owned for the run.
    heuristic: H,
    /// Counters for the result.
    stats: Stats,
}

impl<G, T, H> Rbfs<'_, G, T, H>
where
    G: Successors,
    G::State: Clone + Eq,
    G::Action: Clone,
    T: Goal<G::State> + ?Sized,
    H: Heuristic<G::State, G::Cost>,
{
    /// Searches below the virtual root over `roots`, none of which is a goal.
    fn run(
        &mut self,
        roots: Vec<G::State>,
        mut budget: impl Budget<Stats>,
    ) -> Outcome<G::State, G::Action, G::Cost> {
        let mut base: Vec<_> = roots
            .into_iter()
            .map(|state| self.child(state, None, G::Cost::ZERO, None))
            .collect();
        let mut stack: Vec<Frame<G::State, G::Action, G::Cost>> = Vec::new();

        loop {
            let (children, bound) = stack.last().map_or((&base, Extended::Infinite), |frame| {
                (&frame.children, frame.bound)
            });
            let (best, second) = best_two(children);

            let Some(best) = best.filter(|&best| {
                let value = children[best].value;
                value <= bound && value != Extended::Infinite
            }) else {
                let value = best.map_or(Extended::Infinite, |best| children[best].value);
                let Some(frame) = stack.pop() else {
                    return Outcome::Unsolvable;
                };

                let parent = stack
                    .last_mut()
                    .map_or(&mut base, |frame| &mut frame.children);
                parent[frame.index].value = value;
                continue;
            };

            let node = &children[best];
            if self.goal.is_goal(&node.state) {
                return Outcome::Found(path(&stack, node));
            }

            if budget.exhausted(&self.stats) {
                return Outcome::LimitReached;
            }

            let frame = self.descend(&stack, node, best, bound.min(second));
            stack.push(frame);
        }
    }

    /// Expands `node`, the child at `index` of the top frame (or of the virtual root), into
    /// a frame with `bound`; children already on the path are left out.
    fn descend(
        &mut self,
        stack: &[Frame<G::State, G::Action, G::Cost>],
        node: &Child<G::State, G::Action, G::Cost>,
        index: usize,
        bound: Extended<G::Cost>,
    ) -> Frame<G::State, G::Action, G::Cost> {
        let mut edges = Vec::new();
        self.stats.expanded += 1;
        self.stats.generated += collect_successors(self.graph, &node.state, &mut edges);

        // A node searched before hands its backed-up value down to children below it.
        let inherited = (node.f < node.value).then_some(node.value);
        let children = edges
            .into_iter()
            .filter(|(_, state, _)| {
                *state != node.state && stack.iter().all(|frame| frame.node.state != *state)
            })
            .map(|(action, state, cost)| {
                let mut child = self.child(state, Some((action, cost)), node.g + cost, inherited);
                child.value = child.value.max(child.f);
                child
            })
            .collect();

        Frame {
            node: Child {
                state: node.state.clone(),
                via: node.via.clone(),
                g: node.g,
                f: node.f,
                value: node.value,
            },
            index,
            bound,
            children,
        }
    }

    /// A child entry for `state` at path cost `g`, with its static `f` and a backed-up
    /// value of `inherited` (or `f` if `None`; the caller takes the maximum).
    fn child(
        &mut self,
        state: G::State,
        via: Option<(G::Action, G::Cost)>,
        g: G::Cost,
        inherited: Option<Extended<G::Cost>>,
    ) -> Child<G::State, G::Action, G::Cost> {
        let f = Extended::f(g, self.heuristic.estimate(&state));

        Child {
            state,
            via,
            g,
            f,
            value: inherited.unwrap_or(f),
        }
    }
}

/// The position of the child with the smallest backed-up value (the first of equals) and
/// the smallest value among the others (infinite if there are none).
fn best_two<S, A, C: Cost>(children: &[Child<S, A, C>]) -> (Option<usize>, Extended<C>) {
    let mut best: Option<usize> = None;
    let mut second = Extended::Infinite;
    for (index, child) in children.iter().enumerate() {
        match best {
            Some(current) if child.value >= children[current].value => {
                second = second.min(child.value);
            }
            Some(current) => {
                second = second.min(children[current].value);
                best = Some(index);
            }
            None => best = Some(index),
        }
    }

    (best, second)
}

/// The solution along the frames of the current path, ending in `last`.
fn path<S: Clone, A: Clone, C: Cost>(
    stack: &[Frame<S, A, C>],
    last: &Child<S, A, C>,
) -> Solution<S, A, C> {
    let nodes = || stack.iter().map(|frame| &frame.node).chain([last]);
    let states = nodes().map(|node| node.state.clone()).collect();
    let actions = nodes()
        .filter_map(|node| node.via.as_ref().map(|(action, _)| action.clone()))
        .collect();
    let cost = nodes()
        .filter_map(|node| node.via.as_ref().map(|&(_, cost)| cost))
        .fold(C::ZERO, |sum, cost| sum + cost);

    Solution {
        states,
        actions,
        cost,
    }
}
