//! Integration tests: small hand-made graphs, grids and property tests that compare the
//! algorithms against each other.

mod anytime;
#[cfg(feature = "parallel")]
mod batch;
mod beam;
mod bidirectional;
mod bounded;
mod depth_first;
mod dynamic;
#[cfg(feature = "float")]
mod field;
mod fixtures;
mod flow;
mod fringe;
mod grid;
#[cfg(feature = "parallel")]
mod hda;
mod jps;
mod landmarks;
mod lazy;
mod meet;
mod mha;
mod multi;
mod properties;
mod rbfs;
mod real_time;
mod send_sync;
mod sma;
mod small_graphs;
#[cfg(feature = "float")]
mod theta;
mod workspace;
