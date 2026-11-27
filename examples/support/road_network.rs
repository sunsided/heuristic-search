//! A generated road network of a city, shared by the road examples.
//!
//! The city is a square street grid of 150 x 150 intersections, 100 m apart and slightly
//! jittered so that straight-line distances are not grid distances. Every tenth street
//! is a two-way arterial (50 km/h); the streets between them are local (30 km/h), and
//! every other local street is one-way, alternating in direction like a Manhattan grid.
//! The streets on the city boundary are two-way, so every intersection can reach every
//! other: the graph is directed but strongly connected. A motorway (100 km/h, the top
//! speed) runs as a ring 6 km around the centre plus four spokes from the centre to the
//! edge of the city, with interchanges to the street grid every 2 km.
//!
//! Edge costs are travel times in whole deciseconds (`u32`), rounded up. The straight-line
//! heuristic divides the Euclidean distance by the top speed and rounds down, which makes
//! it admissible and consistent.

use std::cell::RefCell;

use heuristic_search::{Heuristic, Predecessors, Successors, heuristic_fn};

/// Intersections per side of the street grid.
const GRID: u32 = 150;
/// Distance between neighbouring intersections in metres.
const BLOCK: f64 = 100.0;
/// Largest random offset of an intersection from its grid position, in metres.
const JITTER: f64 = 25.0;
/// Every `ARTERIAL_EVERY`-th street is an arterial.
const ARTERIAL_EVERY: u32 = 10;
/// Speed on local streets in metres per second (30 km/h).
const LOCAL_SPEED: f64 = 30.0 / 3.6;
/// Speed on arterials in metres per second (50 km/h).
const ARTERIAL_SPEED: f64 = 50.0 / 3.6;
/// Speed on ramps between the motorway and the street grid (40 km/h).
const RAMP_SPEED: f64 = 40.0 / 3.6;
/// Speed on the motorway in metres per second (100 km/h), the fastest road.
pub const TOP_SPEED: f64 = 100.0 / 3.6;
/// Radius of the motorway ring around the city centre, in metres.
const RING_RADIUS: f64 = 6_000.0;
/// Distance between motorway nodes in metres.
const MOTORWAY_STEP: f64 = 500.0;
/// Every `INTERCHANGE_EVERY`-th motorway node has ramps to the nearest intersection.
const INTERCHANGE_EVERY: usize = 4;

/// The kind of a road, which sets its speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadKind {
    /// A local street at 30 km/h, one-way on every other street.
    Local,
    /// An arterial at 50 km/h, always two-way.
    Arterial,
    /// A motorway segment at 100 km/h.
    Motorway,
    /// A ramp between the motorway and the street grid.
    Ramp,
}

/// A directed road segment between two intersections.
#[derive(Debug, Clone, Copy)]
pub struct Road {
    /// The intersection the segment leaves.
    pub from: u32,
    /// The intersection the segment enters.
    pub to: u32,
    /// The travel time in deciseconds.
    pub time: u32,
    /// What kind of road this is.
    pub kind: RoadKind,
}

/// A directed road network: intersections with coordinates and road segments between
/// them, searchable forwards ([`Successors`]) and backwards ([`Predecessors`]).
///
/// States are intersection indices; actions are indices into [`RoadNetwork::roads`].
#[derive(Debug)]
pub struct RoadNetwork {
    /// The position of every intersection in metres.
    coords: Vec<(f64, f64)>,
    /// Every road segment; the action of an edge is its index here.
    pub roads: Vec<Road>,
    /// Outgoing road indices per intersection, in compressed rows.
    outgoing: Adjacency,
    /// Incoming road indices per intersection, in compressed rows.
    incoming: Adjacency,
    /// The states passed to [`Successors::successors`] since the last
    /// [`RoadNetwork::take_expansions`]: the forward expansions.
    forward_expanded: RefCell<Vec<u32>>,
    /// The nodes passed to [`Predecessors::predecessors`] since the last
    /// [`RoadNetwork::take_expansions`]: the backward expansions.
    backward_expanded: RefCell<Vec<u32>>,
}

/// Road indices grouped by intersection: the roads of `v` are
/// `roads[start[v]..start[v + 1]]`.
#[derive(Debug)]
struct Adjacency {
    /// Where the roads of each intersection begin in [`Adjacency::roads`].
    start: Vec<usize>,
    /// Road indices, grouped by intersection.
    roads: Vec<u32>,
}

impl Adjacency {
    /// Groups `roads` by the end `key` picks out of each one.
    fn new(nodes: usize, roads: &[Road], key: impl Fn(&Road) -> u32) -> Self {
        let mut start = vec![0; nodes + 1];
        for road in roads {
            start[key(road) as usize + 1] += 1;
        }
        for v in 0..nodes {
            start[v + 1] += start[v];
        }

        let mut next = start.clone();
        let mut grouped = vec![0; roads.len()];
        for (id, road) in roads.iter().enumerate() {
            let slot = &mut next[key(road) as usize];
            grouped[*slot] = u32::try_from(id).expect("fewer than 2^32 roads");
            *slot += 1;
        }

        Self {
            start,
            roads: grouped,
        }
    }

    /// The road indices of intersection `v`.
    fn of(&self, v: u32) -> &[u32] {
        &self.roads[self.start[v as usize]..self.start[v as usize + 1]]
    }
}

/// A tiny seeded pseudo-random generator (`SplitMix64`), so every run sees the same city.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator with a fixed `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform float in `[-1, 1)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits fit an f64 exactly"
    )]
    fn next_signed(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }

    /// A uniform integer in `0..n`.
    pub fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next_u64() % u64::from(n)).expect("below n fits u32")
    }
}

/// The travel time in deciseconds for `metres` at `speed` metres per second, rounded up.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "travel times are small and non-negative"
)]
fn travel_time(metres: f64, speed: f64) -> u32 {
    (metres / speed * 10.0).ceil() as u32
}

/// Collects intersections and roads while the city is laid out.
struct Builder {
    /// The position of every intersection so far.
    coords: Vec<(f64, f64)>,
    /// Every road so far.
    roads: Vec<Road>,
}

impl Builder {
    /// Adds an intersection at `at` and returns its index.
    fn node(&mut self, at: (f64, f64)) -> u32 {
        self.coords.push(at);
        u32::try_from(self.coords.len() - 1).expect("fewer than 2^32 intersections")
    }

    /// Adds a road of `kind` from `from` to `to`, timed by the kind's speed, and the
    /// opposite road too if `two_way`.
    fn road(&mut self, from: u32, to: u32, kind: RoadKind, two_way: bool) {
        let (a, b) = (self.coords[from as usize], self.coords[to as usize]);
        let metres = (a.0 - b.0).hypot(a.1 - b.1);
        let time = match kind {
            RoadKind::Local => travel_time(metres, LOCAL_SPEED),
            RoadKind::Arterial => travel_time(metres, ARTERIAL_SPEED),
            RoadKind::Motorway => travel_time(metres, TOP_SPEED),
            // Merging costs 10 s on top of the drive.
            RoadKind::Ramp => travel_time(metres, RAMP_SPEED) + 100,
        };
        self.roads.push(Road {
            from,
            to,
            time,
            kind,
        });
        if two_way {
            self.roads.push(Road {
                from: to,
                to: from,
                time,
                kind,
            });
        }
    }

    /// Lays out the jittered street grid; intersection `y * GRID + x` is at column `x`,
    /// row `y`.
    fn streets(&mut self, rng: &mut SplitMix64) {
        for y in 0..GRID {
            for x in 0..GRID {
                let jitter = (JITTER * rng.next_signed(), JITTER * rng.next_signed());
                self.node((
                    f64::from(x) * BLOCK + jitter.0,
                    f64::from(y) * BLOCK + jitter.1,
                ));
            }
        }
        for line in 0..GRID {
            let (kind, direction) = street(line);
            for step in 0..GRID - 1 {
                // Along row `line` (west to east), then along column `line` (south to north).
                for (a, b) in [
                    (line * GRID + step, line * GRID + step + 1),
                    (step * GRID + line, (step + 1) * GRID + line),
                ] {
                    match direction {
                        Direction::Both => self.road(a, b, kind, true),
                        Direction::Ascending => self.road(a, b, kind, false),
                        Direction::Descending => self.road(b, a, kind, false),
                    }
                }
            }
        }
    }

    /// Lays out the motorway ring, the four spokes and their interchanges.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "motorway node counts are small and positive"
    )]
    fn motorway(&mut self) {
        let centre = f64::from(GRID - 1) * BLOCK / 2.0;
        let ring_nodes =
            (std::f64::consts::TAU * RING_RADIUS / MOTORWAY_STEP / 4.0).round() as usize * 4;
        let ring: Vec<u32> = (0..ring_nodes)
            .map(|i| {
                let angle = std::f64::consts::TAU * i as f64 / ring_nodes as f64;
                self.node((
                    centre + RING_RADIUS * angle.cos(),
                    centre + RING_RADIUS * angle.sin(),
                ))
            })
            .collect();
        let mut motorway = ring.clone();
        for i in 0..ring_nodes {
            self.road(
                ring[i],
                ring[(i + 1) % ring_nodes],
                RoadKind::Motorway,
                true,
            );
        }

        let hub = self.node((centre, centre));
        let ring_step = (RING_RADIUS / MOTORWAY_STEP).round() as usize;
        let spoke_nodes = (centre / MOTORWAY_STEP) as usize;
        for quarter in 0..4 {
            let (dx, dy) = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][quarter];
            let mut previous = hub;
            for step in 1..=spoke_nodes {
                let r = step as f64 * MOTORWAY_STEP;
                // The spoke joins the ring where it crosses it.
                let node = if step == ring_step {
                    ring[quarter * ring_nodes / 4]
                } else {
                    let node = self.node((centre + r * dx, centre + r * dy));
                    motorway.push(node);
                    node
                };
                self.road(previous, node, RoadKind::Motorway, true);
                previous = node;
            }
        }
        motorway.push(hub);

        for &node in motorway.iter().step_by(INTERCHANGE_EVERY) {
            let street = nearest_intersection(self.coords[node as usize]);
            self.road(node, street, RoadKind::Ramp, true);
        }
    }
}

/// Which way traffic may flow along a street.
#[derive(Debug, Clone, Copy)]
enum Direction {
    /// Two-way.
    Both,
    /// One-way towards higher coordinates.
    Ascending,
    /// One-way towards lower coordinates.
    Descending,
}

/// The kind and direction of street `line` (a row or a column of the grid).
fn street(line: u32) -> (RoadKind, Direction) {
    if line % ARTERIAL_EVERY == 0 || line == GRID - 1 {
        (RoadKind::Arterial, Direction::Both)
    } else if line % 2 == 0 {
        (RoadKind::Local, Direction::Both)
    } else if line % 4 == 1 {
        (RoadKind::Local, Direction::Ascending)
    } else {
        (RoadKind::Local, Direction::Descending)
    }
}

/// The grid intersection closest to the unjittered position `at`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "positions lie inside the grid"
)]
fn nearest_intersection(at: (f64, f64)) -> u32 {
    let cell = |c: f64| ((c / BLOCK).round() as u32).min(GRID - 1);
    cell(at.1) * GRID + cell(at.0)
}

impl RoadNetwork {
    /// Generates the city. The layout is fixed; `seed` only moves the intersections a
    /// little.
    ///
    /// # Panics
    ///
    /// Panics if the generated network is not strongly connected, which would be a bug in
    /// the layout.
    pub fn generate(seed: u64) -> Self {
        let mut builder = Builder {
            coords: Vec::new(),
            roads: Vec::new(),
        };
        builder.streets(&mut SplitMix64::new(seed));
        builder.motorway();

        let nodes = builder.coords.len();
        let network = Self {
            outgoing: Adjacency::new(nodes, &builder.roads, |road| road.from),
            incoming: Adjacency::new(nodes, &builder.roads, |road| road.to),
            coords: builder.coords,
            roads: builder.roads,
            forward_expanded: RefCell::default(),
            backward_expanded: RefCell::default(),
        };
        assert!(
            network.reaches_all(&network.outgoing, |road| road.to)
                && network.reaches_all(&network.incoming, |road| road.from),
            "the road network must be strongly connected"
        );

        network
    }

    /// Whether every intersection is reached from intersection 0 along `adjacency`, whose
    /// roads lead to `next`.
    fn reaches_all(&self, adjacency: &Adjacency, next: impl Fn(&Road) -> u32) -> bool {
        let mut seen = vec![false; self.len()];
        let mut stack = vec![0_u32];
        seen[0] = true;
        while let Some(v) = stack.pop() {
            for &road in adjacency.of(v) {
                let w = next(&self.roads[road as usize]);
                if !std::mem::replace(&mut seen[w as usize], true) {
                    stack.push(w);
                }
            }
        }

        seen.into_iter().all(|seen| seen)
    }

    /// The number of intersections.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// A one-line description: intersections, road segments by kind, one-way share.
    pub fn summary(&self) -> String {
        let count = |kind| self.roads.iter().filter(|road| road.kind == kind).count();
        format!(
            "{} intersections, {} directed road segments ({} local, {} arterial, {} motorway, \
             {} ramp)",
            self.len(),
            self.roads.len(),
            count(RoadKind::Local),
            count(RoadKind::Arterial),
            count(RoadKind::Motorway),
            count(RoadKind::Ramp),
        )
    }

    /// The straight-line travel time from `a` to `b` at the top speed, in deciseconds,
    /// rounded down: a lower bound on the travel time.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "travel times are small and non-negative"
    )]
    pub fn straight_line(&self, a: u32, b: u32) -> u32 {
        let (p, q) = (self.coords[a as usize], self.coords[b as usize]);
        // Shaved by a hair so float rounding never lifts it above a road's rounded-up time.
        ((p.0 - q.0).hypot(p.1 - q.1) / TOP_SPEED * 10.0 - 1e-6).max(0.0) as u32
    }

    /// The straight-line heuristic towards `target`, for a forward search.
    pub fn straight_line_to(&self, target: u32) -> impl Heuristic<u32, u32> + '_ {
        heuristic_fn(move |v: &u32| Some(self.straight_line(*v, target)))
    }

    /// `count` random pairs of distinct intersections, the same for the same `seed`.
    pub fn random_queries(&self, count: usize, seed: u64) -> Vec<(u32, u32)> {
        let mut rng = SplitMix64::new(seed);
        let nodes = u32::try_from(self.len()).expect("fewer than 2^32 intersections");
        let mut queries = Vec::with_capacity(count);
        while queries.len() < count {
            let (start, goal) = (rng.below(nodes), rng.below(nodes));
            if start != goal {
                queries.push((start, goal));
            }
        }

        queries
    }

    /// The states expanded forwards ([`Successors::successors`]) and backwards
    /// ([`Predecessors::predecessors`]) since the last call, in order; clears both lists.
    pub fn take_expansions(&self) -> (Vec<u32>, Vec<u32>) {
        (self.forward_expanded.take(), self.backward_expanded.take())
    }
}

impl Successors for RoadNetwork {
    type State = u32;
    type Action = u32;
    type Cost = u32;

    fn successors(&self, state: &u32, mut emit: impl FnMut(u32, u32, u32)) {
        self.forward_expanded.borrow_mut().push(*state);
        for &id in self.outgoing.of(*state) {
            let road = &self.roads[id as usize];
            emit(id, road.to, road.time);
        }
    }
}

impl Predecessors for RoadNetwork {
    type Node = u32;
    type Action = u32;
    type Cost = u32;

    fn predecessors(&self, node: &u32, mut emit: impl FnMut(u32, u32, u32)) {
        self.backward_expanded.borrow_mut().push(*node);
        for &id in self.incoming.of(*node) {
            let road = &self.roads[id as usize];
            emit(id, road.from, road.time);
        }
    }
}
