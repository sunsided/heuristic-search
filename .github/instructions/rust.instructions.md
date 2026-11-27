---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` with `alloc`. Flag any `std` use. Float costs (`NonNegF64`),
  Theta*, Lazy Theta* and Field D* sit behind the `float` feature; flag a feature-gated item
  without its `#[cfg]` on the re-export in `lib.rs`.
- No universal problem trait: each search names the capability traits it needs
  (`Successors`, `Predecessors`, `UniformGrid`, `DynamicGraph`, ...), and goal tests and
  heuristics are separate values. Every A* variant is named `*_star`; suffixes compose as
  `_tt`, `_par`, `_multi`, `_meet`, `_in`.
- Every search returns a `SearchResult` (or its anytime, dynamic or trial form) whose
  `Solution` is a real path whose `cost` is the sum of its edges. Optimal searches have a
  proptest against an independent oracle in `tests/it/`; a new search gets one too.
- Public API changes go with a matching update to the README status list, the crate docs in
  `lib.rs` and `docs/design.md`.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
