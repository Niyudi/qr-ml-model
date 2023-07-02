use nalgebra::{Const, DMatrix, Dyn, Matrix, OMatrix, SMatrix, SVector};
use rand::distributions::Uniform;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use std::fmt::{Display, Formatter};

pub fn run() {
    println!("\nTask 1: Orthogonality loss in Gram-Schmidt algorithms.\n");

    const N: usize = 50;
    const M: usize = 10;
    let mut rng = thread_rng();

    // Generates A matrix
    let u: SMatrix<f64, N, M> = random_semi_orthonormal_matrix(&mut rng);
    let d: SMatrix<f64, M, M> = (0..M).fold(SMatrix::zeros(), |mut matrix, i| {
        matrix[(i, i)] = 10f64.powi(-(i as i32));
        matrix
    });
    let v: SMatrix<f64, M, M> = random_semi_orthonormal_matrix(&mut rng);
    let a = u * d * v;

    // Extracts A columns set and applies Gram-Schmidt
    let a_set: [SVector<f64, N> ; M] = a
        .column_iter()
        .map(|column_view| column_view.into())
        .collect::<Vec<SVector<f64, N>>>()
        .try_into().unwrap();

    let q_set_classic = match gram_schmidt_classic(&a_set) {
        Ok(q_set_classic) => q_set_classic,
        Err(err) => {
            println!("{err}");
            return;
        }
    };
    let q_set_modified = match gram_schmidt_modified(&a_set) {
        Ok(q_set_modified) => q_set_modified,
        Err(err) => {
            println!("{err}");
            return;
        }
    };

    // Tests of orthogonality
    for k in 1..=M {
        let test_result = orthogonality_test(
            Matrix::from_columns(&a_set[..k]),
            Matrix::from_columns(&q_set_classic[..k]),
            Matrix::from_columns(&q_set_modified[..k]),
        );

        println!("k = {k}");
        println!("A_{k} condition number: {:7.3e}", test_result.a_condition_number);
        println!("Classic Q_{k} norm: {:7.3e}", test_result.q_classic_norm);
        println!("Modified Q_{k} norm: {:7.3e}", test_result.q_modified_norm);
    }
}

fn gram_schmidt_classic<const N: usize, const M: usize>(set: &[SVector<f64, N> ; M]) -> Result<[SVector<f64, N> ; M], Error> {
    let mut q_set = set.clone();
    for i in 0..M {
        for j in 0..i {
            let r = q_set[j].dot(&set[i]);
            q_set[i] -= r * q_set[j];
        }
        let norm = q_set[i].norm();
        if norm == 0. {
            return Err(Error::MatrixIsSingular);
        } else {
            q_set[i] /= norm;
        }
    }
    Ok(q_set)
}
fn gram_schmidt_modified<const N: usize, const M: usize>(set: &[SVector<f64, N> ; M]) -> Result<[SVector<f64, N> ; M], Error> {
    let mut q_set = set.clone();
    for i in 0..M {
        for j in 0..i {
            let r = q_set[j].dot(&q_set[i]);
            q_set[i] -= r * q_set[j];
        }
        let norm = q_set[i].norm();
        if norm == 0. {
            return Err(Error::MatrixIsSingular);
        } else {
            q_set[i] /= norm;
        }
    }
    Ok(q_set)
}
fn orthogonality_test<const N: usize>(a_matrix: OMatrix<f64, Const<N>, Dyn>, q_matrix_classic: OMatrix<f64, Const<N>, Dyn>, q_matrix_modified: OMatrix<f64, Const<N>, Dyn>) -> OrthogonalityTest {
    let a_svd = a_matrix.svd(false, false);
    let a_condition_number = a_svd.singular_values[0] / a_svd.singular_values[a_svd.singular_values.len() - 1];

    let k = q_matrix_classic.ncols();
    let identity = DMatrix::<f64>::identity(k, k);
    let q_classic_norm = (identity.clone() - q_matrix_classic.transpose() * q_matrix_classic).norm();
    assert_eq!(k, q_matrix_modified.ncols());
    let q_modified_norm = (identity - q_matrix_modified.transpose() * q_matrix_modified).norm();

    OrthogonalityTest {
        a_condition_number,
        q_classic_norm,
        q_modified_norm,
    }
}
fn random_semi_orthonormal_matrix<const N: usize, const M: usize>(rng: &mut ThreadRng) -> SMatrix<f64, N, M> {
    let dist = Uniform::new_inclusive(-1., 1.);
    if N >= M {
        let mut columns: Vec<SVector<f64, N>> = Vec::new();
        for _ in 0..M {
            columns.push(SVector::from_distribution(&dist, rng));
        }
        Matrix::orthonormalize(&mut columns[..]);
        Matrix::from_columns(&columns)
    } else {
        let mut rows: Vec<SVector<f64, M>> = Vec::new();
        for _ in 0..N {
            rows.push(SVector::from_distribution(&dist, rng));
        }
        Matrix::orthonormalize(&mut rows[..]);
        Matrix::from_columns(&rows).transpose()
    }
}

enum Error {
    MatrixIsSingular,
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Error::MatrixIsSingular => "Matrix input to QR algorithm is singular!",
        })
    }
}

struct OrthogonalityTest {
    pub a_condition_number: f64,
    pub q_classic_norm: f64,
    pub q_modified_norm: f64,
}