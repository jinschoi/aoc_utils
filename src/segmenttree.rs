use std::collections::HashMap;
use thiserror::Error;

/// Error types for segment tree operations
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SegmentError {
    #[error("Invalid range: start={start} >= end={end}")]
    InvalidRange { start: i32, end: i32 },
    #[error("Point {0} not found in the coordinate set")]
    InvalidPoint(i32),
    #[error("Width calculation resulted in arithmetic overflow")]
    WidthOverflow,
}

/// Node in the segment tree
#[derive(Debug, Default, Clone)]
struct Node {
    /// Reference count - how many segments cover this node
    ref_count: u32,
    /// Total width covered by active segments in this subtree
    covered_width: i64,
}

/// A segment tree optimized for line sweep algorithms on axis-aligned rectangles.
///
/// This data structure supports dynamic insertion and removal of segments (intervals)
/// and efficiently queries the total width covered by all active segments. It uses
/// coordinate compression to handle arbitrary coordinate values efficiently.
///
/// # Time Complexity
/// - Construction: O(n log n) where n is the number of unique coordinates
/// - Insert/Remove: O(log n)
/// - Query: O(1)
///
/// # Space Complexity
/// - O(n) where n is the number of unique coordinates
///
/// # Example: Rectangle Union Line Sweep
/// ```
/// // Set up coordinates for rectangles: [(0,0,3,2), (1,1,4,3)]
/// let x_coords = vec![0, 1, 3, 4];
/// let mut tree = SegmentTree::new(x_coords);
///
/// // Process left edges: activate segments
/// tree.activate_segment(0, 3).unwrap();
/// assert_eq!(tree.total_covered_width(), 3);
///
/// tree.activate_segment(1, 4).unwrap();
/// assert_eq!(tree.total_covered_width(), 4); // Union is [0,4)
///
/// // Process right edges: deactivate segments
/// tree.deactivate_segment(0, 3).unwrap();
/// assert_eq!(tree.total_covered_width(), 3); // Only [1,4) remains
///
/// tree.deactivate_segment(1, 4).unwrap();
/// assert_eq!(tree.total_covered_width(), 0); // No active segments
/// ```
#[derive(Debug)]
pub struct SegmentTree {
    size: usize,
    nodes: Vec<Node>,
    points: Vec<i32>,
    point_to_index: HashMap<i32, usize>,
}

impl SegmentTree {
    /// Creates a new segment tree with the given set of coordinates.
    ///
    /// The coordinates will be sorted and deduplicated automatically.
    /// The tree will manage segments between consecutive coordinate pairs.
    ///
    /// # Arguments
    /// * `coordinates` - Vector of x-coordinates that segments can start/end at
    ///
    /// # Panics
    /// Panics if the coordinates vector is empty.
    ///
    /// # Example
    /// ```
    /// let tree = SegmentTree::new(vec![0, 5, 10, 15]);
    /// // Tree can now handle segments like [0,5), [5,10), [0,15), etc.
    /// ```
    #[must_use]
    pub fn new(mut coordinates: Vec<i32>) -> Self {
        assert!(
            !coordinates.is_empty(),
            "Coordinates vector cannot be empty"
        );
        coordinates.sort_unstable();
        coordinates.dedup();
        let size = coordinates.len();
        let point_to_index = coordinates
            .iter()
            .copied()
            .enumerate()
            .map(|(i, point)| (point, i))
            .collect();
        let nodes = vec![Node::default(); 4 * size];
        Self {
            size,
            nodes,
            points: coordinates,
            point_to_index,
        }
    }

    /// Activates a segment [start, end), making it contribute to the covered width.
    ///
    /// Multiple segments can overlap - the tree correctly handles the union of all
    /// active segments. The same segment can be activated multiple times and must
    /// be deactivated the same number of times to be fully removed.
    ///
    /// # Arguments
    /// * `start` - Start coordinate of the segment (inclusive)
    /// * `end` - End coordinate of the segment (exclusive)
    ///
    /// # Returns
    /// * `Ok(())` if the segment was successfully activated
    /// * `Err(SegmentError)` if the coordinates are invalid
    ///
    /// # Errors
    /// Returns `SegmentError::InvalidRange` if start >= end.
    /// Returns `SegmentError::InvalidPoint` if start or end coordinates are not in the coordinate set.
    pub fn activate_segment(&mut self, start: i32, end: i32) -> Result<(), SegmentError> {
        self.validate_segment(start, end)?;

        let start_idx = self.point_to_index[&start];
        let end_idx = self.point_to_index[&end];

        self.update_range(0, start_idx, end_idx, 0, self.size - 1, 1);
        Ok(())
    }

    /// Deactivates a segment [start, end), removing one reference to it.
    ///
    /// If the segment was activated multiple times, it remains active until
    /// deactivated the same number of times.
    ///
    /// # Arguments
    /// * `start` - Start coordinate of the segment (inclusive)
    /// * `end` - End coordinate of the segment (exclusive)
    ///
    /// # Returns
    /// * `Ok(())` if the segment was successfully deactivated
    /// * `Err(SegmentError)` if the coordinates are invalid
    ///
    /// # Errors
    /// Returns `SegmentError::InvalidRange` if start >= end.
    /// Returns `SegmentError::InvalidPoint` if start or end coordinates are not in the coordinate set.
    pub fn deactivate_segment(&mut self, start: i32, end: i32) -> Result<(), SegmentError> {
        self.validate_segment(start, end)?;

        let start_idx = self.point_to_index[&start];
        let end_idx = self.point_to_index[&end];

        self.update_range(0, start_idx, end_idx, 0, self.size - 1, -1);
        Ok(())
    }

    /// Returns the total width covered by all currently active segments.
    ///
    /// This is the union of all active segments, so overlapping segments
    /// don't contribute multiple times to the total width.
    ///
    #[must_use]
    pub fn total_covered_width(&self) -> i64 {
        if self.nodes.is_empty() {
            0
        } else {
            self.nodes[0].covered_width
        }
    }

    /// Returns the number of unique coordinates in this tree.
    #[must_use]
    pub fn coordinate_count(&self) -> usize {
        self.size
    }

    /// Returns a reference to the sorted coordinate vector.
    #[must_use]
    pub fn coordinates(&self) -> &[i32] {
        &self.points
    }

    /// Validates that a segment has valid coordinates.
    fn validate_segment(&self, start: i32, end: i32) -> Result<(), SegmentError> {
        if start >= end {
            return Err(SegmentError::InvalidRange { start, end });
        }

        if !self.point_to_index.contains_key(&start) {
            return Err(SegmentError::InvalidPoint(start));
        }

        if !self.point_to_index.contains_key(&end) {
            return Err(SegmentError::InvalidPoint(end));
        }

        Ok(())
    }

    /// Updates a range in the segment tree.
    fn update_range(
        &mut self,
        node: usize,
        range_start: usize,
        range_end: usize,
        node_left: usize,
        node_right: usize,
        delta: i32,
    ) {
        if range_start <= node_left && node_right <= range_end {
            // Complete overlap: update this node
            if delta > 0 {
                self.nodes[node].ref_count += delta as u32;
            } else {
                self.nodes[node].ref_count =
                    self.nodes[node].ref_count.saturating_sub(-delta as u32);
            }
        } else if node_right > node_left {
            // Partial overlap: recurse to children
            let mid = usize::midpoint(node_left, node_right);
            if range_start < mid {
                self.update_range(2 * node + 1, range_start, range_end, node_left, mid, delta);
            }
            if mid < range_end {
                self.update_range(2 * node + 2, range_start, range_end, mid, node_right, delta);
            }
        }

        // Update the covered width for this node
        self.update_covered_width(node, node_left, node_right);
    }

    /// Updates the covered width for a node based on its reference count and children.
    fn update_covered_width(&mut self, node: usize, node_left: usize, node_right: usize) {
        if self.nodes[node].ref_count > 0 {
            // This entire range is covered
            self.nodes[node].covered_width = self.calculate_width(node_left, node_right);
        } else if node_right - node_left > 1 {
            // Not covered, sum children's coverage
            let left_child = 2 * node + 1;
            let right_child = 2 * node + 2;
            self.nodes[node].covered_width =
                self.nodes[left_child].covered_width + self.nodes[right_child].covered_width;
        } else {
            // Leaf node with no coverage
            self.nodes[node].covered_width = 0;
        }
    }

    /// Calculates the width between two coordinate indices with overflow protection.
    fn calculate_width(&self, start_idx: usize, end_idx: usize) -> i64 {
        debug_assert!(start_idx < end_idx, "Invalid width calculation indices");
        debug_assert!(end_idx < self.points.len(), "End index out of bounds");

        // Use i64 arithmetic to prevent overflow
        i64::from(self.points[end_idx]) - i64::from(self.points[start_idx])
    }

    /// Checks internal consistency of the tree (debug builds only).
    /// Returns true if the tree is in a valid state.
    #[cfg(debug_assertions)]
    #[must_use]
    pub fn validate_consistency(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        self.validate_node(0, 0, self.size - 1)
    }

    /// Validates a single node's consistency (debug builds only).
    #[cfg(debug_assertions)]
    fn validate_node(&self, node: usize, left: usize, right: usize) -> bool {
        let expected_width = if self.nodes[node].ref_count > 0 {
            self.calculate_width(left, right)
        } else if right - left > 1 {
            let left_child = 2 * node + 1;
            let right_child = 2 * node + 2;
            let mid = usize::midpoint(left, right);

            // Recursively validate children
            if !self.validate_node(left_child, left, mid)
                || !self.validate_node(right_child, mid, right)
            {
                return false;
            }

            self.nodes[left_child].covered_width + self.nodes[right_child].covered_width
        } else {
            0
        };

        self.nodes[node].covered_width == expected_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_segment_tree() {
        let tree = SegmentTree::new(vec![1, 3, 5, 7]);
        assert_eq!(tree.coordinate_count(), 4);
        assert_eq!(tree.coordinates(), &[1, 3, 5, 7]);
        assert_eq!(tree.total_covered_width(), 0);
    }

    #[test]
    fn test_new_with_duplicates_and_unsorted() {
        let tree = SegmentTree::new(vec![5, 1, 3, 1, 7, 3]);
        assert_eq!(tree.coordinate_count(), 4);
        assert_eq!(tree.coordinates(), &[1, 3, 5, 7]);
    }

    #[test]
    fn test_single_segment_operations() {
        let mut tree = SegmentTree::new(vec![1, 2, 4, 5]);

        tree.activate_segment(1, 2).unwrap();
        assert_eq!(tree.total_covered_width(), 1);

        tree.deactivate_segment(1, 2).unwrap();
        assert_eq!(tree.total_covered_width(), 0);
    }

    #[test]
    fn test_multiple_non_overlapping_segments() {
        let mut tree = SegmentTree::new(vec![1, 2, 4, 5]);

        tree.activate_segment(1, 2).unwrap();
        assert_eq!(tree.total_covered_width(), 1);

        tree.activate_segment(4, 5).unwrap();
        assert_eq!(tree.total_covered_width(), 2);
    }

    #[test]
    fn test_overlapping_segments() {
        let mut tree = SegmentTree::new(vec![1, 2, 3, 4]);

        tree.activate_segment(1, 3).unwrap();
        assert_eq!(tree.total_covered_width(), 2);

        tree.activate_segment(2, 4).unwrap();
        assert_eq!(tree.total_covered_width(), 3); // Union is [1,4)
    }

    #[test]
    fn test_contained_segment() {
        let mut tree = SegmentTree::new(vec![1, 2, 3, 4]);

        tree.activate_segment(1, 4).unwrap();
        assert_eq!(tree.total_covered_width(), 3);

        tree.activate_segment(2, 3).unwrap();
        assert_eq!(tree.total_covered_width(), 3); // Still [1,4)
    }

    #[test]
    fn test_multiple_activations_same_segment() {
        let mut tree = SegmentTree::new(vec![1, 2, 3]);

        tree.activate_segment(1, 2).unwrap();
        assert_eq!(tree.total_covered_width(), 1);

        tree.activate_segment(1, 2).unwrap(); // Activate again
        assert_eq!(tree.total_covered_width(), 1);

        tree.deactivate_segment(1, 2).unwrap(); // Deactivate once
        assert_eq!(tree.total_covered_width(), 1); // Still active

        tree.deactivate_segment(1, 2).unwrap(); // Deactivate again
        assert_eq!(tree.total_covered_width(), 0); // Now inactive
    }

    #[test]
    fn test_complex_scenario() {
        let mut tree = SegmentTree::new(vec![0, 1, 2, 3, 4, 5, 6]);

        tree.activate_segment(0, 2).unwrap();
        tree.activate_segment(1, 4).unwrap();
        tree.activate_segment(3, 6).unwrap();
        assert_eq!(tree.total_covered_width(), 6); // Union covers [0,6)

        tree.deactivate_segment(1, 4).unwrap();
        assert_eq!(tree.total_covered_width(), 5); // [0,2) + [3,6) = 2 + 3

        tree.deactivate_segment(0, 2).unwrap();
        assert_eq!(tree.total_covered_width(), 3); // Only [3,6) remains

        tree.deactivate_segment(3, 6).unwrap();
        assert_eq!(tree.total_covered_width(), 0); // Empty
    }

    #[test]
    fn test_single_point() {
        let tree = SegmentTree::new(vec![5]);
        assert_eq!(tree.coordinate_count(), 1);
        assert_eq!(tree.total_covered_width(), 0);
    }

    #[test]
    fn test_two_points() {
        let mut tree = SegmentTree::new(vec![1, 3]);
        tree.activate_segment(1, 3).unwrap();
        assert_eq!(tree.total_covered_width(), 2);
    }

    #[test]
    fn test_invalid_range_error() {
        let mut tree = SegmentTree::new(vec![1, 2, 3]);

        let result = tree.activate_segment(3, 2);
        assert!(matches!(
            result,
            Err(SegmentError::InvalidRange { start: 3, end: 2 })
        ));

        let result = tree.activate_segment(2, 2);
        assert!(matches!(
            result,
            Err(SegmentError::InvalidRange { start: 2, end: 2 })
        ));
    }

    #[test]
    fn test_invalid_point_error() {
        let mut tree = SegmentTree::new(vec![1, 2, 3]);

        let result = tree.activate_segment(0, 2);
        assert!(matches!(result, Err(SegmentError::InvalidPoint(0))));

        let result = tree.activate_segment(1, 4);
        assert!(matches!(result, Err(SegmentError::InvalidPoint(4))));
    }

    #[test]
    fn test_example_from_original_main() {
        let mut tree = SegmentTree::new(vec![1, 2, 4, 5]);

        tree.activate_segment(1, 2).unwrap();
        assert_eq!(tree.total_covered_width(), 1);

        tree.activate_segment(4, 5).unwrap();
        assert_eq!(tree.total_covered_width(), 2);

        tree.activate_segment(2, 4).unwrap();
        assert_eq!(tree.total_covered_width(), 4); // [1,2] + [2,4] + [4,5] = [1,5]

        tree.deactivate_segment(2, 4).unwrap();
        assert_eq!(tree.total_covered_width(), 2); // Back to [1,2] + [4,5]

        tree.activate_segment(1, 5).unwrap();
        assert_eq!(tree.total_covered_width(), 4); // [1,5] covers everything

        tree.deactivate_segment(1, 5).unwrap();
        assert_eq!(tree.total_covered_width(), 2); // Back to [1,2] + [4,5]
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_consistency_validation() {
        let mut tree = SegmentTree::new(vec![0, 1, 2, 3, 4]);
        assert!(tree.validate_consistency());

        tree.activate_segment(0, 2).unwrap();
        assert!(tree.validate_consistency());

        tree.activate_segment(1, 4).unwrap();
        assert!(tree.validate_consistency());

        tree.deactivate_segment(0, 2).unwrap();
        assert!(tree.validate_consistency());

        tree.deactivate_segment(1, 4).unwrap();
        assert!(tree.validate_consistency());
    }

    #[test]
    fn test_large_coordinates() {
        let mut tree = SegmentTree::new(vec![i32::MIN, 0, i32::MAX]);
        tree.activate_segment(i32::MIN, 0).unwrap();

        let expected_width = (0i64) - (i32::MIN as i64);
        assert_eq!(tree.total_covered_width(), expected_width);
    }

    #[test]
    #[should_panic(expected = "Coordinates vector cannot be empty")]
    fn test_empty_coordinates_panic() {
        let _ = SegmentTree::new(vec![]);
    }

    #[test]
    fn test_node_allocation_sufficiency() {
        // This test demonstrates why 4*size nodes are necessary
        // With 7 coordinates, we have 6 elementary segments
        // The segment tree over 6 segments can become unbalanced
        let coordinates = vec![1, 2, 3, 4, 5, 6, 7];
        let tree = SegmentTree::new(coordinates);

        // With 4*7 = 28 nodes allocated, we should have enough space
        assert_eq!(tree.nodes.len(), 28);

        // Verify we can perform operations without panicking
        let mut tree = tree;
        tree.activate_segment(1, 3).unwrap();
        tree.activate_segment(2, 5).unwrap();
        tree.activate_segment(4, 7).unwrap();

        // This complex pattern of overlapping segments exercises the tree structure
        assert!(tree.total_covered_width() > 0);

        // If we only had 2*7 = 14 nodes, we might get index out of bounds errors
        // in worst-case tree configurations
    }
}
