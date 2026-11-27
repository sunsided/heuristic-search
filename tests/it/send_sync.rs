//! Compile-time checks that graphs, results, heuristics, planners and workspaces are `Send`
//! and `Sync` whenever their type parameters are, so they can be shared with or moved to
//! other threads.

use heuristic_search::{
    AStarSearch, AdStar, AdStarMulti, AnytimeResult, DStarLite, DStarLiteMulti, EightConnected,
    FlowField, FocalList, LearnedHeuristic, Limits, LpaStar, LpaStarMulti, MeetFn, MhaWeights,
    Octile, OctileDistance, Outcome, ReplayError, SearchResult, Solution, Stats, TrialResult,
    TrialStats, Weight, Workspace, focal_fn, heuristic_fn, meet_fn,
};

/// Compiles only if `T` is `Send` and `Sync`.
fn assert_send_sync<T: Send + Sync>() {}

/// Compiles only if the type of the argument is `Send` and `Sync`; for closure types,
/// which cannot be named.
fn check<T: Send + Sync>(_: &T) {}

/// Every generic type is `Send + Sync` for any `Send + Sync` parameters: the bound is
/// checked when this function is compiled, for all parameters at once.
fn generic<S, A, C, H, D, G, K>()
where
    S: Send + Sync,
    A: Send + Sync,
    C: Send + Sync,
    H: Send + Sync,
    D: Send + Sync,
    G: Send + Sync,
    K: Send + Sync,
{
    assert_send_sync::<EightConnected<G>>();
    assert_send_sync::<SearchResult<S, A, C>>();
    assert_send_sync::<Outcome<S, A, C>>();
    assert_send_sync::<Solution<S, A, C>>();
    assert_send_sync::<AnytimeResult<S, A, C>>();
    assert_send_sync::<TrialResult<S, A, C>>();
    assert_send_sync::<ReplayError<S, A>>();
    assert_send_sync::<LearnedHeuristic<S, C, H>>();
    assert_send_sync::<MeetFn<D, S, A>>();
    assert_send_sync::<FocalList<C, K>>();
    assert_send_sync::<LpaStar<S, C, D>>();
    assert_send_sync::<DStarLite<S, C, D>>();
    assert_send_sync::<AdStar<S, C, D>>();
    assert_send_sync::<LpaStarMulti<S, C, D>>();
    assert_send_sync::<DStarLiteMulti<S, C, D>>();
    assert_send_sync::<AdStarMulti<S, C, D>>();
    assert_send_sync::<Workspace<S, A, C>>();
    assert_send_sync::<AStarSearch<S, A, C, H>>();
    assert_send_sync::<FlowField<S, A, C>>();
}

/// The concrete types and closure-built heuristics, goals and relations.
#[test]
fn types_are_send_and_sync() {
    generic::<u32, u8, u64, (), fn(), (), u32>();

    assert_send_sync::<Octile>();
    assert_send_sync::<OctileDistance>();
    assert_send_sync::<Stats>();
    assert_send_sync::<TrialStats>();
    assert_send_sync::<Limits>();
    assert_send_sync::<Weight>();
    assert_send_sync::<MhaWeights>();

    let target = 7_u32;
    check(&heuristic_fn(move |n: &u32| target.checked_sub(*n)));
    check(&|n: &u32| *n == target);
    check(&meet_fn(|s: &u32, n: &u32| s == n));
    check(&focal_fn(|n: &u32| *n));
    check(&LearnedHeuristic::<u32, u32, _>::new(heuristic_fn(
        |_: &u32| Some(0),
    )));
}

/// The float-cost types of Theta* and Field D*.
#[cfg(feature = "float")]
#[test]
fn float_types_are_send_and_sync() {
    use heuristic_search::{AnyAngleResult, FieldDStar, FieldResult, NonNegF64};

    assert_send_sync::<NonNegF64>();
    assert_send_sync::<AnyAngleResult>();
    assert_send_sync::<FieldResult>();
    assert_send_sync::<FieldDStar>();
}
