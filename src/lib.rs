pub mod grid;
pub mod search;
pub mod counter;
pub mod vecmap;
pub mod toposort;
pub mod disjointset;
pub mod segmenttree;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub fn read_lines<T>(filename: T) -> io::Result<io::Lines<io::BufReader<File>>>
where
    T: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
