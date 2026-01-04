use num::{rational::Rational64, Signed, Zero};
use std::fmt;
use tabled::{builder::Builder, settings::Alignment};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LPOp {
    Gte,
    Lte,
    Eq,
}

fn r(n: i64) -> Rational64 {
    Rational64::from_integer(n)
}

fn fmt_rat(x: &Rational64) -> String {
    if *x.denom() == 1 {
        x.numer().to_string()
    } else {
        format!("{}/{}", x.numer(), x.denom())
    }
}

/// Builds a two-phase simplex tableau with columns:
/// x... | s... | a... | z | w | rhs
///
/// Rows:
/// constraints...
/// phase 2 objective
/// phase 1 objective
///
/// Conventions:
/// - Phase 2 objective row encodes:  -c^T x + z = 0  (maximize c^T x)
/// - Phase 1 objective row encodes:   Σ a + w = 0     (so w = -Σ a, maximize w ⇔ minimize Σ a)
/// - Entering variable: most negative coefficient in objective row (excluding `z`, `w`, `rhs`)
/// - Leaving variable: minimum ratio test among constraint rows with positive pivot column coefficient
#[derive(Default, Clone)]
pub struct LPBuilder {
    objective: Vec<i64>,
    constraints: Vec<Vec<i64>>,
    answers: Vec<i64>,
    ops: Vec<LPOp>,
}

impl LPBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_objective(&mut self, o: Vec<i64>) {
        // Maximize c^T x, with c provided directly.
        self.objective = o;
    }

    pub fn add_constraint(&mut self, mut c: Vec<i64>, mut op: LPOp, mut ans: i64) {
        // Ensure RHS is non-negative (helps preserve feasibility assumptions for ratio test).
        if ans < 0 {
            for v in &mut c {
                *v *= -1;
            }
            ans *= -1;
            op = match op {
                LPOp::Lte => LPOp::Gte,
                LPOp::Gte => LPOp::Lte,
                LPOp::Eq => LPOp::Eq,
            };
        }
        self.constraints.push(c);
        self.answers.push(ans);
        self.ops.push(op);
    }

    pub fn build(self) -> LP {
        let m = self.constraints.len();
        let n_x = self.constraints.first().map(|v| v.len()).unwrap_or(0);

        let n_slacks = self
            .ops
            .iter()
            .filter(|&&op| matches!(op, LPOp::Lte | LPOp::Gte))
            .count();

        let n_art = self
            .ops
            .iter()
            .filter(|&&op| matches!(op, LPOp::Gte | LPOp::Eq))
            .count();

        let slack_start = n_x;
        let art_start = slack_start + n_slacks;

        let z_col = art_start + n_art;
        let w_col = z_col + 1;
        let rhs_col = w_col + 1;

        let total_cols = rhs_col + 1;
        let total_rows = m + 2; // constraints + phase2 + phase1

        let p2_row = m;
        let p1_row = m + 1;

        let mut t = vec![vec![Rational64::ZERO; total_cols]; total_rows];
        let mut active = vec![usize::MAX; total_rows];

        let mut slack_j = slack_start;
        let mut art_j = art_start;

        // Constraints
        for i in 0..m {
            // x coefficients
            for j in 0..n_x {
                t[i][j] = r(self.constraints[i][j]);
            }

            // slack/surplus/artificial
            match self.ops[i] {
                LPOp::Lte => {
                    t[i][slack_j] = Rational64::ONE;
                    active[i] = slack_j;
                    slack_j += 1;
                }
                LPOp::Gte => {
                    // surplus -1 and artificial +1, basic is artificial
                    t[i][slack_j] = -Rational64::ONE;
                    slack_j += 1;

                    t[i][art_j] = Rational64::ONE;
                    active[i] = art_j;
                    art_j += 1;
                }
                LPOp::Eq => {
                    t[i][art_j] = Rational64::ONE;
                    active[i] = art_j;
                    art_j += 1;
                }
            }

            t[i][rhs_col] = r(self.answers[i]);
        }

        // Phase 2 objective: -c^T x + z = 0
        for j in 0..n_x {
            t[p2_row][j] = r(-self.objective[j]);
        }
        t[p2_row][z_col] = Rational64::ONE;
        active[p2_row] = z_col;

        // Phase 1 objective: (sum artificials) + w = 0  => w = -sum a
        // Initialize coefficients on artificials to +1, and w to +1.
        for j in art_start..z_col {
            t[p1_row][j] = Rational64::ONE;
        }
        t[p1_row][w_col] = Rational64::ONE;
        active[p1_row] = w_col;

        let mut lp = LP {
            t,
            rows: m,
            slack_start,
            art_start,
            z_col,
            rhs_col,
            active,
        };

        // "Price out" phase 1 objective with respect to the initial basis:
        // For each constraint row where an artificial is basic, eliminate it from the phase 1 row.
        for i in 0..m {
            let bc = lp.active[i];
            if bc >= lp.art_start && bc < lp.z_col {
                // Phase1 has +1 at this artificial; subtract the row to make it 0.
                lp.row_add_scaled(p1_row, i, -lp.t[p1_row][bc]);
            }
        }

        lp
    }
}

#[derive(Debug)]
pub enum SimplexResult {
    Optimal,
    Unbounded,
}

pub struct LP {
    t: Vec<Vec<Rational64>>,
    rows: usize, // number of constraint rows
    slack_start: usize,
    art_start: usize,
    z_col: usize,
    rhs_col: usize,
    active: Vec<usize>,
}

impl LP {
    fn is_basic_in_constraints(&self, col: usize) -> bool {
        self.active.iter().take(self.rows).any(|&bc| bc == col)
    }

    fn rhs(&self, row: usize) -> Rational64 {
        self.t[row][self.rhs_col]
    }

    fn row_add_scaled(&mut self, dst: usize, src: usize, scale: Rational64) {
        if scale.is_zero() {
            return;
        }
        for j in 0..self.t[dst].len() {
            let v = scale * self.t[src][j];
            self.t[dst][j] += v;
        }
    }

    /// Choose entering variable as the most negative coefficient in the objective row
    /// among allowed columns, excluding `z`, `w`, `rhs`, and columns currently basic.
    fn pivot_col(&self, obj_row: usize) -> Option<usize> {
        self.t[obj_row][0..self.art_start]
            .iter()
            .enumerate()
            .filter(|&(col, coeff)| !self.is_basic_in_constraints(col) && coeff.is_negative())
            .min_by_key(|&(_, coeff)| *coeff)
            .map(|(col, _)| col)
    }

    /// Choose leaving row by minimum ratio test among constraint rows with positive pivot column coefficient.
    fn pivot_row(&self, enter_col: usize) -> Option<usize> {
        (0..self.rows)
            .map(|i| (i, self.t[i][enter_col]))
            .filter(|(_, a)| a.is_positive())
            .min_by_key(|&(i, a)| self.rhs(i) / a)
            .map(|(i, _)| i)
    }

    /// Correct Gauss-Jordan simplex pivot:
    /// - Normalize the pivot row so pivot element becomes 1.
    /// - Eliminate the entering column from all other rows.
    fn pivot(&mut self, pr: usize, pc: usize) {
        let pivot = self.t[pr][pc];
        assert!(!pivot.is_zero(), "pivot element must be non-zero");

        // Normalize pivot row
        let ncols = self.t[pr].len();
        for j in 0..ncols {
            self.t[pr][j] /= pivot;
        }

        // Eliminate pivot column in all other rows
        let pivot_row = self.t[pr].clone();
        for i in 0..self.t.len() {
            if i == pr {
                continue;
            }
            let factor = self.t[i][pc];
            if factor.is_zero() {
                continue;
            }
            for j in 0..ncols {
                self.t[i][j] -= factor * pivot_row[j];
            }
        }

        self.active[pr] = pc;
    }

    fn remove_degenerate_artificials_from_basis(&mut self) {
        for i in 0..self.rows {
            let basic = self.active[i];

            // Artificial basic?
            if !(basic >= self.art_start && basic < self.z_col) {
                continue;
            }
            // If this artificial is non-degenerate (rhs != 0), that
            // indicates phase 1 didn't actually drive artificials to
            // 0 (infeasible or not fully optimized).
            if !self.rhs(i).is_zero() {
                continue;
            }

            // Find a non-artificial, nonbasic column with a nonzero coefficient in this row.
            let entering = (0..self.art_start)
                .find(|&col| !self.t[i][col].is_zero() && !self.is_basic_in_constraints(col));

            if let Some(col) = entering {
                self.pivot(i, col);
            }
        }
    }

    fn simplex(&mut self, obj_row: usize) -> SimplexResult {
        loop {
            let Some(enter) = self.pivot_col(obj_row) else {
                return SimplexResult::Optimal;
            };
            let Some(leave) = self.pivot_row(enter) else {
                return SimplexResult::Unbounded;
            };
            self.pivot(leave, enter);
        }
    }

    pub fn minimize(&mut self) -> Option<Rational64> {
        let p2 = self.rows;
        for v in self.t[p2][0..self.slack_start].iter_mut() {
            *v = -*v;
        }
        self.maximize().map(|n| -n)
    }

    pub fn maximize(&mut self) -> Option<Rational64> {
        let p2 = self.rows;
        let p1 = self.rows + 1;

        match self.simplex(p1) {
            SimplexResult::Optimal => {}
            SimplexResult::Unbounded => return None,
        }

        // Feasibility: w = RHS in phase1 row (since w is the objective variable with coefficient 1).
        if !self.rhs(p1).is_zero() {
            return None; // infeasible
        }
        self.remove_degenerate_artificials_from_basis();

        match self.simplex(p2) {
            SimplexResult::Optimal => Some(self.rhs(p2)),
            SimplexResult::Unbounded => None,
        }
    }

    pub fn solution_x(&self) -> Vec<Rational64> {
        let mut x = vec![Rational64::ZERO; self.slack_start];

        for row in 0..self.rows {
            let col = self.active[row];
            if col < self.slack_start {
                x[col] = self.rhs(row);
            }
        }

        x
    }
}

impl fmt::Display for LP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = Builder::default();
        for row in &self.t {
            builder.push_record(row.iter().map(fmt_rat));
        }
        let mut table = builder.build();
        table.with(Alignment::right());
        writeln!(f, "{table}")?;
        writeln!(f, "active={:?}", self.active)
    }
}
