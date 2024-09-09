use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::hash::Hash;

pub fn bfs_full_path<T, S, G, N, NR>(start_nodes: S, goal: G, neighbors: N) -> Option<Vec<T>>
where
    T: Hash + Eq + Clone,
    S: IntoIterator<Item = T>,
    G: Fn(&T) -> bool,
    N: Fn(&T) -> NR,
    NR: IntoIterator<Item = T>,
{
    let mut prev = HashMap::<T, T>::new();
    let mut q = VecDeque::from_iter(start_nodes);
    let mut seen = HashSet::<T>::from_iter(q.iter().cloned());

    while let Some(s) = q.pop_front() {
        if goal(&s) {
            let mut path = Vec::new();
            let mut cur = &s;
            while let Some(prev) = prev.get(cur) {
                path.push(cur.clone());
                cur = prev;
            }
            path.push(cur.clone());
            path.reverse();
            return Some(path);
        }
        for n in neighbors(&s) {
            if !seen.contains(&n) {
                seen.insert(n.clone());
                prev.insert(n.clone(), s.clone());
                q.push_back(n);
            }
        }
    }
    None
}

#[derive(PartialEq, Eq)]
struct DijkstraState<T: Eq> {
    pub cost: u32,
    pub node: T,
}

impl<T: Eq> Ord for DijkstraState<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl<T: Eq> PartialOrd for DijkstraState<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn dijkstra<T, S, G, N, NR>(start_nodes: S, goal: G, neighbors: N) -> Option<u32>
where
    T: Hash + Eq + Clone,
    S: IntoIterator<Item = T>,
    G: Fn(&T) -> bool,
    N: Fn(&T) -> NR,
    NR: IntoIterator<Item = (u32, T)>,
{
    let mut dist = HashMap::new();
    let mut q = BinaryHeap::new();
    for start in start_nodes {
        dist.insert(start.clone(), 0);
        q.push(DijkstraState {
            cost: 0,
            node: start,
        });
    }

    let mut max_q: usize = 0;
    let mut steps = 0;
    while let Some(DijkstraState { cost: p, node }) = q.pop() {
        steps += 1;
        max_q = max_q.max(q.len());

        if &p > dist.get(&node).unwrap_or(&u32::MAX) {
            continue;
        }
        if goal(&node) {
            dbg!(max_q, steps);
            return Some(p);
        }
        for (cost, n) in neighbors(&node) {
            let alt = p + cost;
            if &alt < dist.get(&n).unwrap_or(&u32::MAX) {
                dist.insert(n.clone(), alt);
                q.push(DijkstraState { cost: alt, node: n });
            }
        }
    }
    None
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct AstarResult<T> {
    pub cost: u32,
    pub node: T,
    pub came_from: HashMap<T, T>,
}

#[derive(PartialEq, Eq)]
struct AstarState<T: Eq> {
    estimated_cost: u32,
    node: T,
}

impl<T: Eq> Ord for AstarState<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // order by reverse estimated cost to be used in a max heap
        other.estimated_cost.cmp(&self.estimated_cost)
    }
}

impl<T: Eq> PartialOrd for AstarState<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn astar_full_path<T, S, G, N, NR, H>(
    start_nodes: S,
    goal: G,
    neighbors: N,
    heuristic: H,
) -> Option<AstarResult<T>>
where
    T: Hash + Eq + Clone,
    S: IntoIterator<Item = T>,
    G: Fn(&T) -> bool,
    N: Fn(&T) -> NR,
    NR: IntoIterator<Item = (u32, T)>,
    H: Fn(&T) -> u32,
{
    let mut dist = HashMap::new();
    let mut came_from = HashMap::new();
    let mut q = BinaryHeap::new();

    for start in start_nodes {
        dist.insert(start.clone(), 0);
        q.push(AstarState {
            estimated_cost: heuristic(&start),
            node: start,
        });
    }

    let mut max_q = 0;
    let mut steps = 0;
    while let Some(AstarState {
        estimated_cost,
        node,
    }) = q.pop()
    {
        steps += 1;
        max_q = max_q.max(q.len());

        let node_dist = *dist.get(&node).expect("best distance should exist");
        if estimated_cost > node_dist + heuristic(&node) {
            // Don't reprocess something we have a better distance for.
            continue;
        }

        if goal(&node) {
            dbg!(max_q, steps);
            return Some(AstarResult {
                cost: estimated_cost,
                node,
                came_from,
            });
        }
        for (cost, n) in neighbors(&node) {
            let alt = node_dist + cost;
            if alt < *dist.get(&n).unwrap_or(&u32::MAX) {
                came_from.insert(n.clone(), node.clone());
                dist.insert(n.clone(), alt);
                q.push(AstarState {
                    estimated_cost: alt + heuristic(&n),
                    node: n,
                });
            }
        }
    }
    None
}

pub fn astar_dist<T, S, G, N, NR, H>(start_nodes: S, goal: G, neighbors: N, heuristic: H) -> Option<u32>
where
    T: Hash + Eq + Clone,
    S: IntoIterator<Item = T>,
    G: Fn(&T) -> bool,
    N: Fn(&T) -> NR,
    NR: IntoIterator<Item = (u32, T)>,
    H: Fn(&T) -> u32,
{
    let mut dist = HashMap::new();
    let mut q = BinaryHeap::new();

    for start in start_nodes {
        dist.insert(start.clone(), 0);
        q.push(AstarState {
            estimated_cost: heuristic(&start),
            node: start,
        });
    }

    let mut max_q = 0;
    let mut steps = 0;
    while let Some(AstarState {
        estimated_cost,
        node,
    }) = q.pop()
    {
        steps += 1;
        max_q = max_q.max(q.len());

        let node_dist = *dist.get(&node).expect("best distance should exist");
        if estimated_cost > node_dist + heuristic(&node) {
            // Don't reprocess something we have a better distance for.
            continue;
        }

        if goal(&node) {
            dbg!(max_q, steps);
            return Some(estimated_cost);
        }
        for (cost, n) in neighbors(&node) {
            let alt = node_dist + cost;
            if alt < *dist.get(&n).unwrap_or(&u32::MAX) {
                dist.insert(n.clone(), alt);
                q.push(AstarState {
                    estimated_cost: alt + heuristic(&n),
                    node: n,
                });
            }
        }
    }
    None
}
