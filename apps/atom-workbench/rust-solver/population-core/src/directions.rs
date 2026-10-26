//! Shift-invert trial directions `v = (sigma E - F)^{-1} b` by GMRES, right
//! preconditioned with P0. Directions only shape the trial space: the reduced
//! model and every error estimate use exact images of the actual columns, so
//! these inner solves are deliberately loose. Several right-hand sides run in
//! lockstep so every operator application processes a block of columns.

use super::trial::DomainOperator;

#[derive(Clone, Copy, Debug)]
pub struct InnerSolve {
    pub maximum_steps: usize,
    pub relative_residual: f64,
}

struct Column {
    beta: f64,
    basis: Vec<Vec<f64>>,
    preconditioned: Vec<Vec<f64>>,
    hessenberg: Vec<Vec<f64>>,
    cosines: Vec<f64>,
    sines: Vec<f64>,
    rotated: Vec<f64>,
    active: bool,
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Approximate solutions of `(pole_j E - F) v_j = b_j` for the `k` columns of
/// `rhs` (column-major `[rows, k]`); `pole[j]` indexes the operator's poles.
/// Also returns the number of single-column operator applications.
pub(crate) fn shift_invert(
    op: &mut DomainOperator, rhs: &[f64], k: usize, pole: &[usize], settings: InnerSolve,
) -> Result<(Vec<f64>, usize), String> {
    let rows = op.rows();
    if rhs.len() != rows * k || pole.len() != k || settings.maximum_steps == 0 {
        return Err("Invalid shift-invert right-hand sides".into());
    }
    let mut columns: Vec<Column> = (0..k)
        .map(|j| {
            let b = &rhs[j * rows..(j + 1) * rows];
            let beta = norm(b);
            let active = beta > 0. && beta.is_finite();
            let first = if active { b.iter().map(|x| x / beta).collect() } else { vec![0.; rows] };
            let mut rotated = vec![0.; settings.maximum_steps + 1];
            rotated[0] = beta;
            Column {
                beta,
                basis: vec![first],
                preconditioned: vec![],
                hessenberg: vec![],
                cosines: vec![],
                sines: vec![],
                rotated,
                active,
            }
        })
        .collect();
    let mut applications = 0;
    for step in 0..settings.maximum_steps {
        let active: Vec<usize> = (0..k).filter(|&j| columns[j].active).collect();
        if active.is_empty() {
            break;
        }
        let count = active.len();
        let mut block = Vec::with_capacity(rows * count);
        for &j in &active {
            block.extend_from_slice(&columns[j].basis[step]);
        }
        let poles: Vec<usize> = active.iter().map(|&j| pole[j]).collect();
        let mut z = vec![0.; rows * count];
        op.precondition(&block, count, &poles, &mut z)?;
        let mut w = vec![0.; rows * count];
        op.shifted(&z, count, &poles, &mut w)?;
        applications += count;
        for (slot, &j) in active.iter().enumerate() {
            let column = &mut columns[j];
            let mut vector = w[slot * rows..(slot + 1) * rows].to_vec();
            let mut h = vec![0.; step + 2];
            for _ in 0..2 {
                for (i, basis) in column.basis.iter().enumerate() {
                    let c = dot(basis, &vector);
                    h[i] += c;
                    for (x, b) in vector.iter_mut().zip(basis) {
                        *x -= c * b;
                    }
                }
            }
            let tail = norm(&vector);
            h[step + 1] = tail;
            // Apply the previous Givens rotations, then a new one.
            for i in 0..step {
                let (a, b) = (h[i], h[i + 1]);
                h[i] = column.cosines[i] * a + column.sines[i] * b;
                h[i + 1] = -column.sines[i] * a + column.cosines[i] * b;
            }
            let (a, b) = (h[step], h[step + 1]);
            let radius = a.hypot(b);
            if !(radius > 0.) || !radius.is_finite() {
                column.active = false;
                column.preconditioned.push(z[slot * rows..(slot + 1) * rows].to_vec());
                column.hessenberg.push(vec![0.; step + 1]);
                column.cosines.push(1.);
                column.sines.push(0.);
                continue;
            }
            let (c, s) = (a / radius, b / radius);
            column.cosines.push(c);
            column.sines.push(s);
            h[step] = radius;
            column.rotated[step + 1] = -s * column.rotated[step];
            column.rotated[step] *= c;
            column.hessenberg.push(h[..=step].to_vec());
            column.preconditioned.push(z[slot * rows..(slot + 1) * rows].to_vec());
            let relative = column.rotated[step + 1].abs() / column.beta;
            if relative <= settings.relative_residual || tail <= 1e-14 * column.beta {
                column.active = false;
            } else {
                for x in vector.iter_mut() {
                    *x /= tail;
                }
                column.basis.push(vector);
            }
        }
    }
    let mut out = vec![0.; rows * k];
    for (j, column) in columns.iter().enumerate() {
        let steps = column.preconditioned.len();
        if steps == 0 {
            continue;
        }
        // Back substitution on the rotated upper-triangular system.
        let mut y = column.rotated[..steps].to_vec();
        for i in (0..steps).rev() {
            for l in i + 1..steps {
                y[i] -= column.hessenberg[l][i] * y[l];
            }
            let diagonal = column.hessenberg[i][i];
            y[i] = if diagonal != 0. { y[i] / diagonal } else { 0. };
        }
        let target = &mut out[j * rows..(j + 1) * rows];
        for (i, z) in column.preconditioned.iter().enumerate() {
            for (x, v) in target.iter_mut().zip(z) {
                *x += y[i] * v;
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("Nonfinite shift-invert direction".into());
    }
    Ok((out, applications))
}
