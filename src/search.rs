use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::rc::Rc;
use std::hash::Hash;

#[derive(PartialEq, Eq)]
struct State<T: Eq> {
    pub cost: u32,
    pub node: T,
}

impl<T: Eq> Ord for State<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: Eq> PartialOrd for State<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn dijkstra<T, G, N>(start: T, goal: G, neighbors: N) -> Option<u32>
where
    T: Hash + Eq,
    G: Fn(&T) -> bool,
    N: Fn(&T) -> Vec<(u32, T)>,
{
    let mut dist = HashMap::new();
    let mut q = BinaryHeap::new();
    let start = Rc::new(start);
    dist.insert(start.clone(), 0);
    q.push(Reverse(State {
        cost: 0,
        node: start,
    }));

    while let Some(Reverse(State { cost: p, node })) = q.pop() {
        if &p > dist.get(&node).unwrap_or(&u32::MAX) {
            continue;
        }
        if goal(&node) {
            return Some(p);
        }
        for (cost, n) in neighbors(&node) {
            let alt = p + cost;
            if &alt < dist.get(&n).unwrap_or(&u32::MAX) {
                let n = Rc::new(n);
                dist.insert(n.clone(), alt);
                q.push(Reverse(State {
                    cost: alt,
                    node: n.clone(),
                }));
            }
        }
    }
    None
}
