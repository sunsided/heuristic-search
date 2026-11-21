//! Fuzzes the exact order of [`Octile`] costs `a + b√2` over the full `u32` range: it must
//! be a total order that agrees with floats wherever they can tell two values apart, and
//! that is compatible with addition.

#![no_main]
#![forbid(unsafe_code)]

use core::cmp::Ordering;

use heuristic_search::{Cost, Octile};
use libfuzzer_sys::fuzz_target;

/// The gap above which `f64` can tell two octile costs apart without rounding doubt:
/// values reach about `2.4e9 · 2^-52 ≈ 5e-7` of rounding error each.
const FLOAT_GAP: f64 = 1e-3;

/// Asserts that `a` and `b` compare like their float values when those differ clearly.
fn assert_agrees_with_floats(a: Octile, b: Octile) {
    let difference = a.to_f64() - b.to_f64();
    if difference.abs() > FLOAT_GAP {
        let expected = difference
            .partial_cmp(&0.0)
            .expect("octile floats are finite");
        assert_eq!(a.cmp(&b), expected, "{a} against {b}");
    }
}

/// Asserts that `a <= b <= c` implies `a <= c`, for every order of the three.
fn assert_transitive(values: [Octile; 3]) {
    let mut sorted = values;
    sorted.sort();

    assert!(
        sorted[0] <= sorted[1] && sorted[1] <= sorted[2],
        "sort is inconsistent"
    );
    assert!(sorted[0] <= sorted[2], "{} > {}", sorted[0], sorted[2]);
}

/// Asserts that adding `c` to both sides keeps the order of `a` and `b`, where the sums
/// do not saturate.
fn assert_compatible_with_addition(a: Octile, b: Octile, c: Octile) {
    let fits = |x: Octile| {
        x.straight().checked_add(c.straight()).is_some()
            && x.diagonal().checked_add(c.diagonal()).is_some()
    };

    if fits(a) && fits(b) {
        assert_eq!(
            a.saturating_add(c).cmp(&b.saturating_add(c)),
            a.cmp(&b),
            "{a}, {b} + {c}"
        );
    }
}

fuzz_target!(|input: [(u32, u32); 3]| {
    let [a, b, c] = input.map(|(straight, diagonal)| Octile::new(straight, diagonal));

    assert_eq!(
        a.cmp(&b),
        b.cmp(&a).reverse(),
        "antisymmetry of {a} and {b}"
    );
    assert_eq!(
        a.cmp(&b) == Ordering::Equal,
        a == b,
        "equality of {a} and {b}"
    );
    assert_agrees_with_floats(a, b);
    assert_agrees_with_floats(b, c);
    assert_transitive([a, b, c]);
    assert_compatible_with_addition(a, b, c);
});
