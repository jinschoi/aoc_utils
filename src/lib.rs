pub mod grid;
pub mod search;

use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead};
use std::path::Path;

pub fn read_lines<T>(filename: T) -> io::Result<io::Lines<io::BufReader<File>>>
where
    T: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub struct Counter<K>(pub HashMap<K, usize>);

impl<K> FromIterator<K> for Counter<K>
where
    K: PartialEq + Eq + Hash,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = K>,
    {
        let mut counter = HashMap::new();
        for k in iter {
            *counter.entry(k).or_default() += 1;
        }
        Self(counter)
    }
}

impl<K> From<&[K]> for Counter<K>
where
    K: PartialEq + Eq + Hash + Clone,
{
    fn from(slice: &[K]) -> Self {
        slice.iter().cloned().collect::<Self>()
    }
}
