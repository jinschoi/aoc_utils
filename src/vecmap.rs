use std::mem;

pub struct VecMap<K, V> {
    pub keys: Vec<K>,
    pub values: Vec<V>,
}

impl<K, V> VecMap<K, V> {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}

// impl Default for VecMap
impl<K, V> Default for VecMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq, V> VecMap<K, V> {
    pub fn index_of_key(&self, k: &K) -> Option<usize> {
        self.keys.iter().position(|key| key == k)
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        match self.keys.iter().position(|key| key == &k) {
            Some(i) => {
                let mut v = v;
                mem::swap(&mut v, &mut self.values[i]);
                Some(v)
            }
            None => {
                self.keys.push(k);
                self.values.push(v);
                None
            }
        }
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.keys
            .iter()
            .position(|key| key == k)
            .map(|i| &self.values[i])
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.keys.iter().any(|key| key == k)
    }
}

