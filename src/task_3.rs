use nalgebra::{DMatrix, DVector};
use rand::rngs::ThreadRng;
use std::fmt::{Display, Formatter};
use rand::distributions::Uniform;
use rand::thread_rng;

pub fn run() {
    println!("\nTask 3: NMF factorization.\n");

    let mut rng = thread_rng();

    let a = DMatrix::<f64>::from_vec(3, 3, vec![
        0.3, 0.6, 0.0,
        0.5, 0.0, 1.0,
        0.4, 0.8, 0.0,
    ]);

    for i in 1..=10 {
        let (w, h, err) = match nmf_factorization(&a, 2, &mut rng) {
            Ok(nmf) => (nmf.w, nmf.h, nmf.err),
            Err(err) => {
                println!("{err}");
                return;
            }
        };

        println!("Test {i}:");
        println!("err={err:.3e}\nw={w:.3e}h={h:.3e}");
    }
}

fn nmf_factorization(matrix: &DMatrix<f64>, p: usize, rng: &mut ThreadRng) -> Result<NMF, Error> {
    const MAX_ITER: usize = 1_000;

    let n = matrix.nrows();
    let m = matrix.ncols();

    let dist = Uniform::new(0., 1.);
    let mut w = DMatrix::<f64>::from_distribution(n, p, &dist, rng);
    let mut h = DMatrix::<f64>::zeros(p, m);

    let mut err = 0.;
    let mut it = 0;
    while it < MAX_ITER {
        for j in 0..p {
            let mut norm = 0.;
            for i in 0..n {
                norm += w[(i, j)] * w[(i, j)];
            }
            norm = norm.sqrt();
            for i in 0..n {
                w[(i, j)] /= norm;
            }
        }

        h = match solve_multiple_systems(&w, matrix) {
            Ok(h) => h,
            Err(err) => return Err(err),
        };
        let _ = h.iter_mut().map(|x| *x = 0f64.max(*x));

        w = match solve_multiple_systems(&h.transpose(), &matrix.transpose()) {
            Ok(w_transpose) => w_transpose.transpose(),
            Err(err) => return Err(err),
        };
        let _ = w.iter_mut().map(|x| *x = 0f64.max(*x));

        let err_matrix = matrix - w.clone() * h.clone();
        let mut new_err = 0.;
        for i in 0..n {
            for j in 0..m {
                new_err += err_matrix[(i, j)] * err_matrix[(i, j)];
            }
        }

        if new_err == err {
            break;
        }

        err = new_err;
        it += 1;
    }

    Ok(NMF {
        w,
        h,
        err,
    })
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
fn solve_multiple_systems(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>, Error> {
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
            Ok(x) => x,
            Err(err) => return Err(err),
        });
    }

    Ok(DMatrix::from_columns(&x_set))
}
fn solve_system(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, Error> {
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
    Ok(x)
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

struct NMF {
    pub w: DMatrix<f64>,
    pub h: DMatrix<f64>,
    pub err: f64,
}