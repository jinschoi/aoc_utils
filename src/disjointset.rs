#[derive(Debug)]
pub struct Node<T> {
    pub data: T,
    pub parent: usize,
    pub size: usize,
}

impl<T> Node<T> {
    fn new(data: T, ind: usize) -> Self {
        Self {
            data,
            parent: ind,
            size: 1,
        }
    }
}

#[derive(Debug, Default)]
pub struct DisjointSet<T> {
    pub nodes: Vec<Node<T>>,
    pub count: usize,
}

impl<T> std::iter::FromIterator<T> for DisjointSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let nodes = iter
            .into_iter()
            .enumerate()
            .map(|(i, data)| Node::new(data, i))
            .collect::<Vec<_>>();
        let count = nodes.len();
        Self { nodes, count }
    }
}

impl<T: std::fmt::Debug> DisjointSet<T> {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            count: 0,
        }
    }

    pub fn add_node(&mut self, data: T) -> usize {
        let i = self.nodes.len();
        self.nodes.push(Node::new(data, i));
        self.count += 1;
        i
    }

    pub fn find(&mut self, i: usize) -> usize {
        if self.nodes[i].parent != i {
            self.nodes[i].parent = self.find(self.nodes[i].parent);
            self.nodes[i].parent
        } else {
            i
        }
    }

    pub fn union(&mut self, i: usize, j: usize) {
        let (i, j) = (self.find(i), self.find(j));
        if i == j {
            return;
        }
        let (mut i, j) = (i.min(j), i.max(j));

        let (a, b) = self.nodes.split_at_mut(j);
        let (mut x, mut y) = (&mut a[i], &mut b[0]);
        if x.size < y.size {
            (x, y) = (y, x);
            i = j;
        }

        y.parent = i;
        x.size += y.size;
        self.count -= 1;
    }
}
