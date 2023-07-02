use std::fmt::{Display, Formatter};
use nalgebra::{DMatrix, DVector};

pub fn run() {
    println!("\nTask 2: Solving linear systems to nearest answer with Gram-Schmidt.\n");

    // Test case 1
    const N_1: usize = 64;
    const M_1: usize = 64;
    let mut a = DMatrix::<f64>::zeros(N_1, M_1);
    for i in 0..N_1 {
        for j in 0..M_1 {
            if i == j {
                a[(i, j)] = 2.;
            } else if i + 1 == j || j + 1 == i {
                a[(i, j)] = 1.;
            }
        }
    }
    let b = DVector::<f64>::repeat(N_1, 1.);

    let (x, residue_norm) = match solve_system(&a, &b) {
        Ok((x, residue_norm)) => (x, residue_norm),
        Err(err) => {
            println!("{err}");
            return;
        }
    };
    println!("Test 1 result:");
    println!("x ={x:.3e}residue norm = {residue_norm:.3e}\n");

    // Test case 2
    const N_2: usize = 20;
    const M_2: usize = 17;
    let mut a = DMatrix::<f64>::zeros(N_2, M_2);
    for i in 0..N_2 {
        for j in 0..M_2 {
            if (i as i32 - j as i32).abs() < 5 {
                a[(i, j)] = 1. / ((i + j) as f64 + 1.);
            }
        }
    }
    let b = DVector::from_iterator(N_2, (1..N_2+1).map(|x| x as f64));

    let (x, residue_norm) = match solve_system(&a, &b) {
        Ok((x, residue_norm)) => (x, residue_norm),
        Err(err) => {
            println!("{err}");
            return;
        }
    };
    println!("Test 2 result:");
    println!("x ={x:.3e}residue norm = {residue_norm:.3e}\n");

    // Test case 3
    const N_3: usize = 64;
    const M_3: usize = 64;
    const P_3: usize = 3;
    let mut a = DMatrix::<f64>::zeros(N_3, M_3);
    for i in 0..N_3 {
        for j in 0..M_3 {
            if i == j {
                a[(i, j)] = 2.;
            } else if i + 1 == j || j + 1 == i {
                a[(i, j)] = 1.;
            }
        }
    }
    let mut b = DMatrix::<f64>::zeros(N_3, P_3);
    for i in 0..N_3 {
        b[(i, 0)] = 1.;
        b[(i, 1)] = i as f64 + 1.;
        b[(i, 2)] = (2 * i) as f64 + 1.;
    }

    let (x, residue_norm) = match solve_multiple_systems(&a, &b) {
        Ok((x, residue_norm)) => (x, residue_norm),
        Err(err) => {
            println!("{err}");
            return;
        }
    };
    println!("Test 3 result:");
    println!("x ={x:.3e}residue norm = {residue_norm:.3e}\n");

    // Test case 4
    const N_4: usize = 20;
    const M_4: usize = 17;
    const P_4: usize = 3;
    let mut a = DMatrix::<f64>::zeros(N_4, M_4);
    for i in 0..N_4 {
        for j in 0..M_4 {
            if (i as i32 - j as i32).abs() < 5 {
                a[(i, j)] = 1. / ((i + j) as f64 + 1.);
            }
        }
    }
    let mut b = DMatrix::<f64>::zeros(N_4, P_4);
    for i in 0..N_4 {
        b[(i, 0)] = 1.;
        b[(i, 1)] = i as f64 + 1.;
        b[(i, 2)] = (2 * i) as f64 + 1.;
    }

    let (x, residue_norm) = match solve_multiple_systems(&a, &b) {
        Ok((x, residue_norm)) => (x, residue_norm),
        Err(err) => {
            println!("{err}");
            return;
        }
    };
    println!("Test 4 result:");
    println!("x ={x:.3e}residue norm = {residue_norm:.3e}\n");
}

fn frobenius_norm(matrix: &DMatrix<f64>) -> f64 {
    let mut norm = 0.;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let el = matrix[(i, j)];
            norm += el * el;
        }
    }
    norm.sqrt()
}
fn qr_factorization(matrix: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>), Error> {
    let m = matrix.ncols();
    let mut r = DMatrix::<f64>::zeros(m, m);
    let mut q_set: Vec<DVector<f64>> = matrix
        .column_iter()
        .map(|column_view| column_view.into())
        .collect();
    for i in 0..m {
        for j in 0..i {
            let r_ji = q_set[j].dot(&q_set[i]);
            r[(j, i)] = r_ji;
            let local_q = q_set[j].clone();
            q_set[i] -= r_ji * local_q;
        }
        let r_ii = q_set[i].norm();
        r[(i, i)] = r_ii;
        if r_ii == 0. {
            return Err(Error::MatrixIsSingular);
        } else {
            q_set[i] /= r_ii;
        }
    }
    let q = DMatrix::from_columns(&q_set);
    Ok((q, r))
}
fn solve_multiple_systems(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<(DMatrix<f64>, f64), Error> {
    let n = a.nrows();
    let p = b.ncols();
    if n != b.nrows() {
        return Err(Error::MismatchedDimensions);
    }

    let b_set: Vec<DVector<f64>> = b
        .column_iter()
        .map(|column_view| column_view.into())
        .collect();

    let mut x_set = Vec::new();
    for i in 0..p {
        x_set.push(match solve_system(a, &b_set[i]) {
            Ok(x) => x.0,
            Err(err) => return Err(err),
        });
    }
    let x = DMatrix::from_columns(&x_set);

    Ok((x.clone(), frobenius_norm(&(b - a * x))))
}
fn solve_system(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<(DVector<f64>, f64), Error> {
    let n = a.nrows();
    let m = a.ncols();
    if n != b.len() {
        return Err(Error::MismatchedDimensions);
    }

    let (q, r) = qr_factorization(&a)?;

    let q_set: Vec<DVector<f64>> = q
        .column_iter()
        .map(|column_view| column_view.into())
        .collect();
    let mut b_set = vec![b.clone()];
    let mut z = DVector::<f64>::zeros(m);
    for k in 0..m {
        z[k] = q_set[k].dot(&b_set[k]);
        b_set.push(b_set[k].clone() - z[k] * q_set[k].clone());
    }

    let mut x = DVector::<f64>::zeros(m);
    for i in (0..m).rev() {
        for j in i+1..m {
            x[i] -= r[(i, j)] * x[j];
        }
        x[i] += z[i];
        x[i] /= r[(i, i)];
    }
    Ok((x, b_set[m].norm()))
}

enum Error {
    MatrixIsSingular,
    MismatchedDimensions,
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Error::MatrixIsSingular => "Matrix input to QR algorithm is singular!",
            Error::MismatchedDimensions => "Dimensions are mismatched!",
        })
    }
}