use super::read_lines;
use std::fmt::Debug;
use std::num::ParseIntError;
use std::ops::{Index, IndexMut};
use std::path::Path;
use std::str::FromStr;
use std::{fmt, io};
use thiserror::Error;

#[derive(PartialEq, Eq, Hash)]
pub struct Grid<T> {
    pub width: usize,
    pub height: usize,
    pub g: Vec<T>,
}

#[derive(Error, Debug)]
pub enum GridError {
    #[error("empty grid")]
    EmptyGrid,
    #[error("inconsistent row lengths")]
    Inconsistent,
    #[error("{0}")]
    IOError(#[from] io::Error),
    #[error("{0}")]
    ParseError(#[from] ParseIntError),
}

impl<T> fmt::Display for Grid<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.height {
            for j in 0..self.width {
                write!(f, "{}", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<T> fmt::Debug for Grid<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        fmt::Display::fmt(self, f)
    }
}

impl<T> Grid<T>
where
    T: Clone,
{
    pub fn new(fill: T, width: usize, height: usize) -> Self {
        let g = vec![fill; width * height];
        Self { width, height, g }
    }

    pub fn from_vals(vals: Vec<T>, width: usize, height: usize) -> Self {
        assert_eq!(vals.len(), width * height);
        Self {
            width,
            height,
            g: vals,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut g = vec![];
        for j in 0..self.width {
            for i in 0..self.height {
                g.push(self[(i, j)].clone());
            }
        }
        Self {
            width: self.height,
            height: self.width,
            g,
        }
    }

    pub fn rotate_right(&self) -> Self {
        let mut g = vec![];
        for j in 0..self.width {
            for i in (0..self.height).rev() {
                g.push(self[(i, j)].clone());
            }
        }
        Self {
            width: self.height,
            height: self.width,
            g,
        }
    }

    pub fn rotate_left(&self) -> Self {
        let mut g = vec![];
        for j in (0..self.width).rev() {
            for i in 0..self.height {
                g.push(self[(i, j)].clone());
            }
        }
        Self {
            width: self.height,
            height: self.width,
            g,
        }
    }
}

impl<T> From<Vec<Vec<T>>> for Grid<T> {
    fn from(v: Vec<Vec<T>>) -> Self {
        let width = v[0].len();
        let height = v.len();
        let g = v.into_iter().flatten().collect();
        Self { width, height, g }
    }
}

impl<T> FromStr for Grid<T>
where
    T: From<char>,
{
    type Err = GridError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut g = vec![];
        let mut lines = s.lines();
        let first_line = lines.next().ok_or(GridError::EmptyGrid)?;
        let row = first_line.chars().map(|c| T::from(c)).collect::<Vec<_>>();
        let width = row.len();

        g.extend(row);

        for line in lines {
            let row = line.chars().map(|c| T::from(c)).collect::<Vec<_>>();
            if row.len() != width {
                return Err(GridError::Inconsistent);
            }
            g.extend(row);
        }
        let height = g.len() / width;
        Ok(Self { width, height, g })
    }
}

impl<T> Grid<T>
where
    T: From<char>,
{
    pub fn read_from_file<P>(filename: P) -> Result<Self, GridError>
    where
        P: AsRef<Path>,
    {
        let mut g = vec![];
        let mut lines = read_lines(filename)?;
        let first_line = lines.next().ok_or(GridError::EmptyGrid)??;
        let row = first_line.chars().map(|c| T::from(c)).collect::<Vec<_>>();
        let width = row.len();

        g.extend(row);

        for line in lines {
            let row = line?.chars().map(|c| T::from(c)).collect::<Vec<_>>();
            if row.len() != width {
                return Err(GridError::Inconsistent);
            }
            g.extend(row);
        }
        let height = g.len() / width;
        Ok(Self { width, height, g })
    }
}

impl<T> Grid<T>
where
    T: FromStr,
    GridError: From<<T as FromStr>::Err>,
{
    pub fn from_space_sep(s: &str) -> Result<Self, GridError> {
        let mut g = vec![];
        let mut lines = s.lines();
        let first_line = lines.next().ok_or(GridError::EmptyGrid)?;
        let row = first_line
            .split_ascii_whitespace()
            .map(|val| val.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        let width = row.len();

        g.extend(row);

        for line in lines {
            let row = line
                .split_ascii_whitespace()
                .map(|val| val.parse::<T>())
                .collect::<Result<Vec<_>, _>>()?;
            if row.len() != width {
                return Err(GridError::Inconsistent);
            }
            g.extend(row);
        }
        let height = g.len() / width;
        Ok(Self { width, height, g })
    }
}

impl<T> Index<(usize, usize)> for Grid<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.g[i * self.width + j]
    }
}

impl<T> IndexMut<(usize, usize)> for Grid<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.g[i * self.width + j]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pos(pub usize, pub usize);

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl<T> Index<Pos> for Grid<T> {
    type Output = T;

    fn index(&self, index: Pos) -> &Self::Output {
        let Pos(i, j) = index;
        &self.g[i * self.width + j]
    }
}

impl<T> IndexMut<Pos> for Grid<T> {
    fn index_mut(&mut self, index: Pos) -> &mut Self::Output {
        let Pos(i, j) = index;
        &mut self.g[i * self.width + j]
    }
}

pub struct Neighbors<'a, T> {
    grid: &'a Grid<T>,
    offsets: &'static [(isize, isize)],
    pos: Pos,
    i: usize,
}

impl<T> Iterator for Neighbors<'_, T> {
    type Item = Pos;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.offsets.len() {
            return None;
        }
        let (di, dj) = self.offsets[self.i];
        self.i += 1;
        let Pos(i, j) = self.pos;
        let new_i = i as isize + di;
        let new_j = j as isize + dj;
        if new_i >= 0
            && new_i < self.grid.height as isize
            && new_j >= 0
            && new_j < self.grid.width as isize
        {
            Some(Pos(new_i as usize, new_j as usize))
        } else {
            self.next()
        }
    }
}

pub struct GridRow<'a, T> {
    grid: &'a Grid<T>,
    i: usize,
    j: usize,
}

impl<'a, T> Iterator for GridRow<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.j == self.grid.width {
            return None;
        }
        let res = &self.grid[(self.i, self.j)];
        self.j += 1;
        Some(res)
    }
}

pub struct GridCol<'a, T> {
    grid: &'a Grid<T>,
    i: usize,
    j: usize,
}

impl<'a, T> Iterator for GridCol<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.grid.height {
            return None;
        }
        let res = &self.grid[(self.i, self.j)];
        self.i += 1;
        Some(res)
    }
}

impl<T> Grid<T> {
    pub fn cardinal_neighbors(&self, pos: Pos) -> Neighbors<'_, T> {
        Neighbors {
            grid: self,
            offsets: &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            pos,
            i: 0,
        }
    }

    pub fn all_neighbors(&self, pos: Pos) -> Neighbors<'_, T> {
        Neighbors {
            grid: self,
            offsets: &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
            pos,
            i: 0,
        }
    }

    pub fn position(&self, f: impl Fn(&T) -> bool) -> Option<Pos> {
        let ind = self.g.iter().position(f)?;
        Some(Pos(ind / self.width, ind % self.width))
    }

    pub fn all_positions<'a>(
        &'a self,
        f: impl Fn(&T) -> bool + 'a,
    ) -> impl Iterator<Item = Pos> + '_ {
        self.g.iter().enumerate().flat_map(move |(i, val)| {
            if f(val) {
                Some(Pos(i / self.width, i % self.width))
            } else {
                None
            }
        })
    }

    pub fn row(&self, i: usize) -> GridRow<T> {
        GridRow {
            grid: self,
            i,
            j: 0,
        }
    }

    pub fn col(&self, j: usize) -> GridCol<T> {
        GridCol {
            grid: self,
            i: 0,
            j,
        }
    }

    fn _flood_fill<F>(&mut self, start_pos: Pos, fill_value: T, is_blocked: F, cardinal: bool) -> usize
    where
        F: Fn(&T) -> bool,
        T: Clone + Eq,
    {
        assert!(!is_blocked(&self[start_pos]));

        let mut stack = vec![start_pos];
        let neighbors = if cardinal { Self::cardinal_neighbors } else { Self::all_neighbors };
        let mut changed = 0;

        while let Some(pos) = stack.pop() {
            if self[pos] != fill_value {
                self[pos] = fill_value.clone();
                changed += 1;
            }
            let new_positions = neighbors(self, pos).filter(|&np| {
                let val = &self[np];
                !is_blocked(&self[np]) && *val != fill_value
            });
            stack.extend(new_positions);
        }
        changed
    }

    pub fn flood_fill<F>(&mut self, start_pos: Pos, fill_value: T, is_blocked: F) -> usize
    where
        F: Fn(&T) -> bool,
        T: Clone + Eq,
    {
        self._flood_fill(start_pos, fill_value, is_blocked, false)
    }

    pub fn flood_fill_cardinal<F>(&mut self, start_pos: Pos, fill_value: T, is_blocked: F) -> usize
    where
        F: Fn(&T) -> bool,
        T: Clone + Eq,
    {
        self._flood_fill(start_pos, fill_value, is_blocked, true)
    }    

    pub fn map<U, F>(&self, f: F) -> Grid<U>
    where
        F: Fn(&T) -> U,
    {
        let g = self.g.iter().map(f).collect::<Vec<_>>();
        Grid {
            width: self.width,
            height: self.height,
            g,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grid() -> Grid<char> {
        Grid {
            width: 3,
            height: 3,
            g: vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
        }
    }

    #[test]
    fn test_cardinal_neighbors() {
        let g = sample_grid();
        assert_eq!(
            g.cardinal_neighbors(Pos(0, 0)).collect::<Vec<_>>(),
            vec![Pos(1, 0), Pos(0, 1)]
        );
    }

    #[test]
    fn test_all_neighbors() {
        let g = sample_grid();
        assert_eq!(
            g.all_neighbors(Pos(0, 0)).collect::<Vec<_>>(),
            vec![Pos(0, 1), Pos(1, 0), Pos(1, 1),]
        );
        assert_eq!(
            g.all_neighbors(Pos(1, 1)).collect::<Vec<_>>(),
            vec![
                Pos(0, 0),
                Pos(0, 1),
                Pos(0, 2),
                Pos(1, 0),
                Pos(1, 2),
                Pos(2, 0),
                Pos(2, 1),
                Pos(2, 2)
            ]
        );
        assert_eq!(
            g.all_neighbors(Pos(2, 2)).collect::<Vec<_>>(),
            vec![Pos(1, 1), Pos(1, 2), Pos(2, 1),]
        );
    }

    #[test]
    fn test_grid_index() {
        let g = sample_grid();
        assert_eq!(g[Pos(0, 0)], 'a');
        assert_eq!(g[Pos(1, 1)], 'e');
        assert_eq!(g[Pos(2, 2)], 'i');
    }

    #[test]
    fn test_position() {
        let g = sample_grid();
        assert_eq!(g.position(|c| *c == 'a'), Some(Pos(0, 0)));
        assert_eq!(g.position(|c| *c == 'e'), Some(Pos(1, 1)));
        assert_eq!(g.position(|c| *c == 'i'), Some(Pos(2, 2)));
        assert_eq!(g.position(|c| *c == 'j'), None);
    }

    #[test]
    fn test_all_positions() {
        let g = sample_grid();
        assert_eq!(
            g.all_positions(|c| *c == 'a').collect::<Vec<_>>(),
            vec![Pos(0, 0)]
        );
        assert_eq!(
            g.all_positions(|c| *c == 'e').collect::<Vec<_>>(),
            vec![Pos(1, 1)]
        );
        assert_eq!(
            g.all_positions(|c| *c == 'i').collect::<Vec<_>>(),
            vec![Pos(2, 2)]
        );
        assert_eq!(
            g.all_positions(|c| ['a', 'e', 'i'].contains(c))
                .collect::<Vec<_>>(),
            vec![Pos(0, 0), Pos(1, 1), Pos(2, 2),]
        );
    }

    #[test]
    fn test_transpose() {
        let g = Grid {
            width: 3,
            height: 4,
            g: vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
        };
        let g_t = g.transpose();
        let transposed = Grid {
            width: 4,
            height: 3,
            g: vec!['a', 'd', 'g', 'j', 'b', 'e', 'h', 'k', 'c', 'f', 'i', 'l'],
        };
        assert_eq!(g_t, transposed);
        assert_eq!(g_t.transpose(), g);
    }

    #[test]
    fn test_rotations() {
        let g = sample_grid();
        let g_r = g.rotate_right();
        let rotated_right = Grid {
            width: 3,
            height: 3,
            g: vec!['g', 'd', 'a', 'h', 'e', 'b', 'i', 'f', 'c'],
        };
        assert_eq!(g_r, rotated_right);
        assert_eq!(g_r.rotate_left(), g);
        let g_l = g.rotate_left();
        let rotated_left = Grid {
            width: 3,
            height: 3,
            g: vec!['c', 'f', 'i', 'b', 'e', 'h', 'a', 'd', 'g'],
        };
        assert_eq!(g_l, rotated_left);
    }

    #[test]
    fn test_row_iter() {
        let g = sample_grid();
        assert_eq!(
            g.row(0).collect::<Vec<_>>(),
            vec![&'a', &'b', &'c']
        );
        assert_eq!(
            g.row(1).collect::<Vec<_>>(),
            vec![&'d', &'e', &'f']
        );
        assert_eq!(
            g.row(2).collect::<Vec<_>>(),
            vec![&'g', &'h', &'i']
        );
    }

    #[test]
    fn test_col_iter() {
        let g = sample_grid();
        assert_eq!(
            g.col(0).collect::<Vec<_>>(),
            vec![&'a', &'d', &'g']
        );
        assert_eq!(
            g.col(1).collect::<Vec<_>>(),
            vec![&'b', &'e', &'h']
        );
        assert_eq!(
            g.col(2).collect::<Vec<_>>(),
            vec![&'c', &'f', &'i']
        );
    }

    #[test]
    fn test_flood_fill() {
        let orig = "........
.######.
.#...#..
.#.####.
.#..#...
.####...
........";
        let flooded = "........
.######.
.#xxx#..
.#x####.
.#xx#...
.####...
........";
        let mut g1 = orig.parse::<Grid<char>>().unwrap();
        g1.flood_fill(Pos(2, 3), 'x', |c| *c == '#');
        let g2 = flooded.parse::<Grid<char>>().unwrap();
        assert_eq!(g1, g2);
    }

    #[test]
    fn test_map() {
        let g = sample_grid().map(|c| *c as u8 - b'a');
        assert_eq!(
            g.row(0).copied().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_from_space_sep() {
        let g: Grid<u8> = Grid::from_space_sep(" 1 2   3 \n4 5 6\n7 8 9").unwrap();
        assert_eq!(g.width, 3);
        assert_eq!(g.height, 3);
        assert_eq!(g.g, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
