use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

#[derive(Debug, Default)]
struct Node<T> {
    in_count: u32,
    children: Vec<T>,
}

#[derive(Default)]
pub struct TopoSort<T> {
    nodes: HashMap<T, Node<T>>,
}

impl<T> TopoSort<T>
where
    T: Default + PartialEq + Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_link(&mut self, prec: T, succ: T) {
        let p = self.nodes.entry(prec).or_default();
        p.children.push(succ.clone());
        let s = self.nodes.entry(succ).or_default();
        s.in_count += 1;
    }

    pub fn sort(mut self) -> Vec<T> {
        let mut q = self
            .nodes
            .iter()
            .filter_map(|(val, node)| if node.in_count == 0 { Some(val.clone()) } else { None })
            .collect::<VecDeque<_>>();
        let mut res = vec![];
        while let Some(val) = q.pop_front() {
            let node = self.nodes.get_mut(&val).unwrap();
            for c in std::mem::take(&mut node.children) {
                let cnode = self.nodes.get_mut(&c).unwrap();
                cnode.in_count -= 1;
                if cnode.in_count == 0 {
                    q.push_back(c);
                }
            }
            res.push(val);
        }
        res
    }
}
