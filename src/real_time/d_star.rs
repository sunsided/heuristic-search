//! Real-time D* ([`RealTimeDStar`], Bond, Widger, Ruml and Sun 2010): D* Lite from the
//! goal and LSS-LRTA* around the agent, under a fixed number of expansions per move.
//!
//! Each [`RealTimeDStar::step`] splits its expansions into a global share for the D* Lite
//! repair (the paper's `GlobalLimit`) and a local share for the lookahead (`LocalLimit`,
//! `LocalRatio` times the limit). Then it picks the move as the paper's `ChooseStep`:
//!
//! ```mermaid
//! flowchart TD
//!     goal{at the goal?} -->|yes| arrived[Arrived]
//!     goal -->|no| repair[D* Lite repair, global share]
//!     repair -->|reached the agent| path{"path to the goal?"}
//!     path -->|yes| informed["Informed: best c + g successor"]
//!     path -->|no| waiting[Waiting: stay]
//!     repair -->|share used up| local[LSS-LRTA* lookahead and update, local share]
//!     local -->|frontier| partial[Partial: first edge towards it]
//!     local -->|no goal reachable| waiting
//! ```
//!
//! After the move, D* Lite's `km` absorbs the start's shift, so the repair resumes where
//! it stopped on the next step, with every edge change reported in between.

use core::hash::Hash;

use search_core::{Cost, Zero, heuristic_fn};

use super::lookahead::lss_lrta_step;
use crate::cost::Weight;
use crate::dynamic::{DynamicGraph, Incremental};
use crate::outcome::{Limits, Stats};
use crate::problem::Successors;
use crate::real_time::{End, LearnedHeuristic, Trajectory, TrialResult, TrialStats};

/// What one [`RealTimeDStar::step`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStep {
    /// The agent stands on the goal; nothing was planned or moved.
    Arrived,
    /// The global search reached the agent, which moved to the successor with the smallest
    /// `c + g` along a cheapest path (the paper's fully informed step).
    Informed,
    /// The global search used up its share before it reached the agent, which took the
    /// first edge towards the state LSS-LRTA* chose (a partially informed step).
    Partial,
    /// No goal is reachable from the agent in the current graph; it stayed where it is,
    /// since later edge changes may open a way.
    Waiting,
}

/// Real-time D* (RTD*, Bond, Widger, Ruml and Sun 2010): an agent that moves towards a
/// fixed goal while edge costs change, spending at most a fixed number of expansions per
/// move.
///
/// RTD* runs D* Lite backwards from the goal (the global search) and LSS-LRTA* forwards
/// from the agent (the local search). Each [`RealTimeDStar::step`] gets `expansions`: the
/// local search keeps `local_share` of them (rounded down, at least one), the global
/// search gets the rest. If the global search reaches the agent within its share, the
/// agent takes the first edge of a cheapest path, the successor with the smallest `c + g`.
/// Otherwise LSS-LRTA* looks ahead with the local share, updates its learned values and
/// the agent takes the first edge of the A* path to the frontier state it chose. Either
/// way the agent moves one edge per step, and the global search resumes where it stopped.
/// [`TrialStats::max_step_expansions`] never exceeds `expansions`.
///
/// Report changed edges with [`RealTimeDStar::edge_changed`] between steps, as with
/// [`DStarLite`](crate::DStarLite). With unlimited expansions every step is fully
/// informed, so the agent follows D* Lite's cheapest path for the current costs. The
/// learned values of the local search last for the planner's lifetime; they assume the
/// costs at the time they were learned and can overestimate after costs fall, which the
/// global search corrects once it reaches the agent.
///
/// On a finite graph with positive edge costs from which the goal stays reachable, the
/// agent reaches the goal (the paper's Theorem 2: the global search reaches the agent
/// after finitely many steps). As for [`DStarLite`](crate::DStarLite), `heuristic(a, b)`
/// estimates the cost from `a` to `b` and must be consistent for the costs at every point
/// in time; the local search uses `heuristic(v, goal)`. [`RealTimeDStar::into_trial`]
/// reports the run as the real-time searches report a trial.
///
/// # Panics
///
/// [`RealTimeDStar::new`] panics if `local_share` exceeds `1`; [`RealTimeDStar::step`]
/// panics if `expansions` is zero.
///
/// # Examples
///
/// ```
/// use heuristic_search::{AgentStep, DynamicGraph, RealTimeDStar, Weight};
///
/// /// A corridor `0..=20` with unit moves both ways.
/// struct Corridor;
///
/// impl DynamicGraph for Corridor {
///     type Vertex = u8;
///     type Cost = u32;
///
///     fn successors(&self, v: &u8, mut emit: impl FnMut(u8, u32)) {
///         if *v > 0 {
///             emit(v - 1, 1);
///         }
///         if *v < 20 {
///             emit(v + 1, 1);
///         }
///     }
///
///     fn predecessors(&self, v: &u8, emit: impl FnMut(u8, u32)) {
///         self.successors(v, emit);
///     }
/// }
///
/// let distance = |a: &u8, b: &u8| u32::from(a.abs_diff(*b));
/// let mut agent = RealTimeDStar::new(&Corridor, 0, 20, distance, Weight::new(1, 2));
///
/// // Four expansions per move: two for D* Lite, two for the lookahead.
/// assert_eq!(agent.step(&Corridor, 4), AgentStep::Partial);
/// while agent.step(&Corridor, 4) != AgentStep::Arrived {}
///
/// let trial = agent.into_trial();
/// assert_eq!(trial.solution().map(|path| path.cost), Some(20));
/// assert!(trial.trial.max_step_expansions <= 4);
/// ```
pub struct RealTimeDStar<V, C, D> {
    /// D* Lite from the goal: the global search.
    core: Incremental<V, C, D>,
    /// The goal vertex.
    goal: V,
    /// The fraction of each step's expansions kept for the local search; at most `1`.
    local_share: Weight,
    /// The local search's learned values. The base is a placeholder: each lookahead puts
    /// the heuristic towards the goal in front of the values.
    table: LearnedHeuristic<V, C, Zero>,
    /// The path the agent has travelled.
    path: Trajectory<V, (), C>,
    /// Counters of both searches over every step.
    stats: Stats,
    /// Moves, steps, learning and the work per step.
    trial: TrialStats,
    /// Whether the last step found no way to the goal.
    waiting: bool,
}

impl<V, C, D> RealTimeDStar<V, C, D>
where
    V: Clone + Hash + Eq,
    C: Cost,
    D: FnMut(&V, &V) -> C,
{
    /// Places the agent on `start`, heading for `goal`, with `local_share` of every step's
    /// expansions kept for the local search (the paper's `LocalRatio`; it tried `1/4`,
    /// `1/2` and `3/4`). Nothing is expanded until [`RealTimeDStar::step`].
    ///
    /// # Panics
    ///
    /// Panics if `local_share` exceeds `1`.
    pub fn new<G: DynamicGraph<Vertex = V, Cost = C>>(
        _graph: &G,
        start: V,
        goal: V,
        heuristic: D,
        local_share: Weight,
    ) -> Self {
        assert!(
            local_share.numerator() <= local_share.denominator(),
            "the local share is a fraction of the expansions, at most 1"
        );

        Self {
            core: Incremental::new(false, goal.clone(), start.clone(), heuristic, None),
            goal,
            local_share,
            table: LearnedHeuristic::new(Zero),
            path: Trajectory::new(start),
            stats: Stats::default(),
            trial: TrialStats::default(),
            waiting: false,
        }
    }

    /// Plans with at most `expansions` expansions and moves the agent at most one edge;
    /// returns what it did.
    ///
    /// `graph` must show the current costs, with every change since the last step
    /// reported through [`RealTimeDStar::edge_changed`].
    ///
    /// # Panics
    ///
    /// Panics if `expansions` is zero: the agent could never move.
    pub fn step<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        expansions: usize,
    ) -> AgentStep {
        assert!(
            expansions > 0,
            "a real-time step needs at least one expansion"
        );

        let position = self.path.current().clone();
        if position == self.goal {
            return AgentStep::Arrived;
        }

        let (global, local) = self.split(expansions);
        let limit = Limits::expansions(u64::try_from(global).unwrap_or(u64::MAX));
        let (reached, mut stats) = self.core.repair(graph, limit);
        let mut updates = 0;
        let choice = if reached {
            self.informed_step(graph, &position)
                .map(|edge| (edge, AgentStep::Informed))
        } else {
            let (edge, changed) = self.local_step(graph, position, local, &mut stats);
            updates = changed;
            edge.map(|edge| (edge, AgentStep::Partial))
        };

        self.stats = self.stats.plus(stats);
        self.trial.record_step(stats.expanded, updates);
        self.waiting = choice.is_none();
        let Some(((next, cost), kind)) = choice else {
            return AgentStep::Waiting;
        };

        self.path.travel((), next.clone(), cost);
        self.trial.moves += 1;
        self.core.move_target(next);

        kind
    }

    /// Reports that the edge `from -> to` changed its cost, appeared or disappeared in
    /// `graph`, which must already show the new state.
    pub fn edge_changed<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        from: &V,
        to: &V,
    ) {
        self.core.edge_changed(graph, from, to);
    }

    /// The vertex the agent stands on.
    #[must_use]
    pub fn position(&self) -> &V {
        self.path.current()
    }

    /// Counters of both searches over every [`RealTimeDStar::step`] so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The run as a real-time trial: [`Outcome::Found`](crate::Outcome::Found) with the
    /// travelled path if the agent stands on the goal,
    /// [`Outcome::Unsolvable`](crate::Outcome::Unsolvable) if the last step found no way
    /// there, and [`Outcome::LimitReached`](crate::Outcome::LimitReached) if the caller
    /// stopped stepping before the goal.
    #[must_use]
    pub fn into_trial(self) -> TrialResult<V, (), C> {
        let end = if *self.path.current() == self.goal {
            End::Goal
        } else if self.waiting {
            End::Unsolvable
        } else {
            End::Limit
        };

        self.path.finish(end, self.stats, self.trial)
    }

    /// Splits `expansions` into the global and the local share; the local share is at
    /// least one, so a partially informed step can always look ahead.
    fn split(&self, expansions: usize) -> (usize, usize) {
        let numerator = u128::from(self.local_share.numerator());
        let denominator = u128::from(self.local_share.denominator());
        let local = expansions as u128 * numerator / denominator;
        let local = usize::try_from(local)
            .expect("the share is at most 1")
            .max(1);

        (expansions - local, local)
    }

    /// The edge to the successor of `position` with the smallest `c + g` (the first on
    /// ties), or `None` if no successor reaches the goal.
    fn informed_step<G: DynamicGraph<Vertex = V, Cost = C>>(
        &self,
        graph: &G,
        position: &V,
    ) -> Option<(V, C)> {
        let mut best: Option<(V, C, C)> = None;
        graph.successors(position, |next, cost| {
            let Some(g) = self.core.g(&next) else {
                return;
            };

            let value = cost.saturating_add(g);
            if best.as_ref().is_none_or(|&(_, _, old)| value < old) {
                best = Some((next, cost, value));
            }
        });

        best.map(|(next, cost, _)| (next, cost))
    }

    /// LSS-LRTA*'s lookahead of `local` expansions from `position` and its update; returns
    /// the first edge towards the chosen frontier state and the number of changed values.
    fn local_step<G: DynamicGraph<Vertex = V, Cost = C>>(
        &mut self,
        graph: &G,
        position: V,
        local: usize,
        stats: &mut Stats,
    ) -> (Option<(V, C)>, u64) {
        let goal = &self.goal;
        let heuristic = self.core.heuristic();
        let table = core::mem::replace(&mut self.table, LearnedHeuristic::new(Zero));
        let mut table = table.rebase(heuristic_fn(|state: &V| Some(heuristic(state, goal))));
        let is_goal = |state: &V| state == goal;

        let (edge, changed) = lss_lrta_step(
            &Forward(graph),
            position,
            &is_goal,
            &mut table,
            local,
            stats,
        );
        self.table = table.rebase(Zero);

        (edge.map(|((), next, cost)| (next, cost)), changed)
    }
}

/// A [`DynamicGraph`] seen as a [`Successors`] graph with unlabelled edges, for the local
/// search.
struct Forward<'a, G>(&'a G);

impl<G: DynamicGraph> Successors for Forward<'_, G> {
    type State = G::Vertex;
    type Action = ();
    type Cost = G::Cost;

    fn successors(
        &self,
        state: &Self::State,
        mut emit: impl FnMut(Self::Action, Self::State, Self::Cost),
    ) {
        self.0.successors(state, |to, cost| emit((), to, cost));
    }
}
