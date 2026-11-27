//! Batches of independent queries solved on several threads ([`solve_batch`], feature
//! `parallel`).

use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::panic::resume_unwind;
use std::thread;

/// Solves every query on up to `threads` threads and returns the answers in the order of
/// `queries`.
///
/// Each thread calls `init` once for its own state, such as a [`Workspace`](crate::Workspace)
/// reused by every query the thread solves, then takes the next unsolved query and calls
/// `solve` on it until none are left, so threads that draw cheap queries take more of
/// them. `threads` is clamped to `1..=queries.len()`; one thread solves the queries on
/// the calling thread, in order.
///
/// # Panics
///
/// Panics with the payload of a panic in `init` or `solve`, once the other threads have
/// stopped.
///
/// # Examples
///
/// ```
/// use heuristic_search::{Limits, Successors, Workspace, dijkstra_in, solve_batch};
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
/// let queries: Vec<(u32, u32)> = (0..100).map(|i| (i, i + i % 7)).collect();
/// let costs = solve_batch(&queries, 4, Workspace::new, |workspace, &(start, target)| {
///     let goal = |n: &u32| *n == target;
///     dijkstra_in(workspace, &Up, start, &goal, Limits::NONE).solution().map(|s| s.cost)
/// });
///
/// assert_eq!(costs.len(), 100);
/// assert!(costs.iter().zip(&queries).all(|(cost, (s, t))| *cost == Some(t - s)));
/// ```
pub fn solve_batch<Q, R, W>(
    queries: &[Q],
    threads: usize,
    init: impl Fn() -> W + Sync,
    solve: impl Fn(&mut W, &Q) -> R + Sync,
) -> Vec<R>
where
    Q: Sync,
    R: Send,
{
    let threads = threads.clamp(1, queries.len().max(1));
    if threads == 1 {
        let mut state = init();
        return queries
            .iter()
            .map(|query| solve(&mut state, query))
            .collect();
    }

    let next = AtomicUsize::new(0);
    let worker = || {
        let mut state = init();
        let mut answers = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(query) = queries.get(index) else {
                return answers;
            };

            answers.push((index, solve(&mut state, query)));
        }
    };

    let mut slots: Vec<Option<R>> = queries.iter().map(|_| None).collect();
    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|_| scope.spawn(worker)).collect();
        for handle in handles {
            let answers = handle.join().unwrap_or_else(|panic| resume_unwind(panic));
            for (index, answer) in answers {
                slots[index] = Some(answer);
            }
        }
    });

    slots
        .into_iter()
        .map(|answer| answer.expect("every query is solved once"))
        .collect()
}
