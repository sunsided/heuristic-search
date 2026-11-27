//! A generated RTS-style map for the game examples, and the seeded random numbers that
//! build it.
//!
//! The map is a [`UniformGrid`]: open ground with walled bases (rooms with gates), forest
//! clumps and a river with a few fords. Cells outside the largest connected region are filled
//! in, so every open cell can reach every other one and a query between open cells always
//! has a path. Searches see it as a graph through
//! [`EightConnected`](heuristic_search::EightConnected).

use heuristic_search::{GridCell, UniformGrid};

/// A tiny seeded pseudo-random generator (`SplitMix64`), so the maps and unit positions are
/// the same on every run without a dependency.
pub struct Rng(u64);

impl Rng {
    /// A generator that starts from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n`; `n` must be above zero. The modulo bias is negligible for the
    /// small `n` used here.
    pub fn below(&mut self, n: usize) -> usize {
        let n = u64::try_from(n).expect("usize fits in u64");
        usize::try_from(self.next_u64() % n).expect("below n, which came from a usize")
    }

    /// A number in `low..high`; `high` must be above `low`.
    pub fn range(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low)
    }
}

/// A rectangular map of passable and blocked cells.
pub struct GameMap {
    /// The number of columns.
    width: usize,
    /// The number of rows.
    height: usize,
    /// `true` for open ground, row-major.
    open: Vec<bool>,
}

impl UniformGrid for GameMap {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn is_passable(&self, x: usize, y: usize) -> bool {
        self.open[y * self.width + x]
    }
}

impl GameMap {
    /// Generates a `width` x `height` map from `seed`: forests, a river with fords and
    /// walled bases, then fills in every pocket cut off from the largest open region.
    pub fn generate(width: usize, height: usize, seed: u64) -> Self {
        let mut map = Self {
            width,
            height,
            open: vec![true; width * height],
        };
        let mut rng = Rng::new(seed);
        let area = width * height;
        for _ in 0..area / 1200 {
            map.forest(&mut rng);
        }
        map.river(&mut rng);
        for _ in 0..area / 2500 {
            map.base(&mut rng);
        }
        map.fill_pockets();
        map
    }

    /// The number of open cells.
    pub fn open_cells(&self) -> usize {
        self.open.iter().filter(|&&open| open).count()
    }

    /// The open cell closest to `(x, y)` (by squared Euclidean distance).
    pub fn nearest_open(&self, (x, y): GridCell) -> GridCell {
        (0..self.height)
            .flat_map(|cy| (0..self.width).map(move |cx| (cx, cy)))
            .filter(|&(cx, cy)| self.is_passable(cx, cy))
            .min_by_key(|&(cx, cy)| cx.abs_diff(x).pow(2) + cy.abs_diff(y).pow(2))
            .expect("the map has open cells")
    }

    /// A uniformly random open cell.
    pub fn random_open(&self, rng: &mut Rng) -> GridCell {
        loop {
            let cell = (rng.below(self.width), rng.below(self.height));
            if self.is_passable(cell.0, cell.1) {
                return cell;
            }
        }
    }

    /// Sets the cell `(x, y)` to `open`, ignoring cells off the map.
    fn set(&mut self, x: usize, y: usize, open: bool) {
        if x < self.width && y < self.height {
            self.open[y * self.width + x] = open;
        }
    }

    /// Blocks a disc of trees with a random centre and a radius of 2 to 9 cells.
    fn forest(&mut self, rng: &mut Rng) {
        let (cx, cy) = (rng.below(self.width), rng.below(self.height));
        let radius = rng.range(2, 10);
        for y in cy.saturating_sub(radius)..=cy + radius {
            for x in cx.saturating_sub(radius)..=cx + radius {
                if x.abs_diff(cx).pow(2) + y.abs_diff(cy).pow(2) <= radius * radius {
                    self.set(x, y, false);
                }
            }
        }
    }

    /// Blocks a river, 3 to 5 cells wide, that meanders from the top edge to the bottom,
    /// with a ford of open ground every 40 to 80 rows.
    fn river(&mut self, rng: &mut Rng) {
        let mut x = rng.range(self.width / 3, 2 * self.width / 3);
        let mut next_ford = rng.range(20, 60);
        for y in 0..self.height {
            x = match rng.below(3) {
                0 => x.saturating_sub(1),
                1 => (x + 1).min(self.width - 1),
                _ => x,
            };
            if y == next_ford {
                next_ford += rng.range(40, 80);
                continue;
            }
            let half = rng.range(1, 3);
            for dx in x.saturating_sub(half)..=x + half {
                self.set(dx, y, false);
            }
        }
    }

    /// Builds a walled base, 12 to 39 cells a side, with a gate three cells wide in two
    /// of its sides, and clears the inside.
    fn base(&mut self, rng: &mut Rng) {
        let (w, h) = (rng.range(12, 40), rng.range(12, 40));
        let (x0, y0) = (
            rng.below(self.width.saturating_sub(w)),
            rng.below(self.height.saturating_sub(h)),
        );
        let (x1, y1) = (x0 + w - 1, y0 + h - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let wall = x == x0 || x == x1 || y == y0 || y == y1;
                self.set(x, y, !wall);
            }
        }
        let (gx, gy) = (rng.range(x0 + 1, x1 - 3), rng.range(y0 + 1, y1 - 3));
        for d in 0..3 {
            self.set(gx + d, if rng.below(2) == 0 { y0 } else { y1 }, true);
            self.set(if rng.below(2) == 0 { x0 } else { x1 }, gy + d, true);
        }
    }

    /// Blocks every open cell outside the largest connected region. A 4-connected flood
    /// fill is exact here: with no corner cutting, an 8-connected move exists only where
    /// two 4-connected ones do.
    fn fill_pockets(&mut self) {
        let mut seen = vec![false; self.open.len()];
        let mut largest = Vec::new();
        for index in 0..self.open.len() {
            if self.open[index] && !seen[index] {
                let region = self.flood(index, &mut seen);
                if region.len() > largest.len() {
                    largest = region;
                }
            }
        }
        self.open = vec![false; self.open.len()];
        for index in largest {
            self.open[index] = true;
        }
    }

    /// The indices of the open cells 4-connected to the cell at `from`, marked in `seen`.
    fn flood(&self, from: usize, seen: &mut [bool]) -> Vec<usize> {
        let mut region = vec![from];
        seen[from] = true;
        let mut next = 0;
        while let Some(&index) = region.get(next) {
            next += 1;
            let (x, y) = (index % self.width, index / self.width);
            let neighbours = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbours {
                if nx >= self.width || ny >= self.height {
                    continue;
                }
                let index = ny * self.width + nx;
                if self.open[index] && !seen[index] {
                    seen[index] = true;
                    region.push(index);
                }
            }
        }
        region
    }
}
