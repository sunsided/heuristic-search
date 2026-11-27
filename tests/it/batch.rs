//! Batches solved on several threads: answers come back in input order and equal the
//! sequential searches, for any thread count, and panics reach the caller.

use std::sync::atomic::{AtomicUsize, Ordering};

use heuristic_search::{Limits, Workspace, a_star, a_star_in, heuristic_fn, solve_batch};
use proptest::prelude::*;

use crate::fixtures::{distances_to, graph_and_goal};

proptest! {
    /// Every answer equals the sequential search for its query, at every thread count.
    #[test]
    fn batch_matches_sequential(
        (graph, _) in graph_and_goal(),
        queries in prop::collection::vec((0_usize..12, 0_usize..12), 0..40),
        threads in 0_usize..6,
    ) {
        let nodes = graph.edges.len();
        let queries: Vec<_> = queries.into_iter().map(|(s, g)| (s % nodes, g % nodes)).collect();
        let solve = |workspace: &mut Workspace<_, _, _>, &(start, goal): &(usize, usize)| {
            let exact = distances_to(&graph, goal);
            let h = heuristic_fn(|state: &usize| Some(exact[*state].unwrap_or(0)));
            a_star_in(workspace, &graph, start, &|s: &usize| *s == goal, h, Limits::NONE)
        };

        let expected: Vec<_> = queries
            .iter()
            .map(|&(start, goal)| {
                let exact = distances_to(&graph, goal);
                let h = heuristic_fn(|state: &usize| Some(exact[*state].unwrap_or(0)));
                a_star(&graph, start, &|s: &usize| *s == goal, h, Limits::NONE)
            })
            .collect();

        prop_assert_eq!(solve_batch(&queries, threads, Workspace::new, solve), expected);
    }
}

/// Each thread calls `init` once, and never more threads than queries start.
#[test]
fn one_state_per_thread() {
    let inits = AtomicUsize::new(0);
    let queries: Vec<u32> = (0..50).collect();
    let init = || inits.fetch_add(1, Ordering::Relaxed);

    let doubled = solve_batch(&queries, 4, init, |_, &q| q * 2);
    assert_eq!(doubled, (0..50).map(|q| q * 2).collect::<Vec<_>>());
    assert!((1..=4).contains(&inits.load(Ordering::Relaxed)));

    inits.store(0, Ordering::Relaxed);
    let few = solve_batch(&[1_u32, 2], 8, init, |_, &q| q);
    assert_eq!(few, [1, 2]);
    assert!(inits.load(Ordering::Relaxed) <= 2);

    assert_eq!(
        solve_batch(&[] as &[u32], 3, init, |_, &q| q),
        Vec::<u32>::new()
    );
}

/// A panic in a worker reaches the caller.
#[test]
#[should_panic(expected = "query 7")]
fn panics_propagate() {
    let queries: Vec<u32> = (0..20).collect();
    solve_batch(
        &queries,
        3,
        || (),
        |(), &q| {
            assert_ne!(q, 7, "query 7");
            q
        },
    );
}
