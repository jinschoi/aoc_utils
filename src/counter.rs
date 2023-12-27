use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug)]
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
