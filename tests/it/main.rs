//! Integration tests: small hand-made graphs, grids and property tests that compare the
//! algorithms against each other.

mod anytime;
mod beam;
mod bidirectional;
mod bounded;
mod depth_first;
mod dynamic;
mod fixtures;
mod fringe;
mod grid;
mod jps;
mod lazy;
mod meet;
mod mha;
mod multi;
mod properties;
mod rbfs;
mod sma;
mod small_graphs;
#[cfg(feature = "float")]
mod theta;
