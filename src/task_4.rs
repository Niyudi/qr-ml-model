use nalgebra::{DMatrix, DVector};
use rand::distributions::Uniform;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use show_image::{create_window, ImageInfo, ImageView, WindowOptions, WindowProxy};
use std::fmt::{Display, Formatter};
use std::fs;
use std::fs::{File, read_dir};
use std::io::{Read, stdin, stdout, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{available_parallelism, sleep};
use std::time::{Duration, Instant};
use show_image::error::SetImageError;
use threadpool::ThreadPool;

pub fn run() {
    println!("\nTask 4: Handwritten digits classification.\n");

    let mut buffer = String::new();

    let training_mode = loop {
        print!("Run in training mode? (y/n): ");
        stdout().flush().unwrap();
        buffer.clear();
        stdin().read_line(&mut buffer).unwrap();
        match buffer.trim() {
            "y" => break true,
            "n" => break false,
            _ => {}
        }
    };

    if training_mode {
        run_training_mode();
    } else {
        run_testing_mode();
    }
}

fn nmf_training(matrix: &DMatrix<f64>, w_init: &DMatrix<f64>) -> Result<Model, Error> {
    let n = matrix.nrows();
    if n != w_init.nrows() {
        return Err(Error::MismatchedDimensions);
    }
    let p = w_init.ncols();
    let m = matrix.ncols();

    let mut w = w_init.clone();

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

    let mut h = match solve_multiple_systems(&w, matrix) {
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
    let mut err = 0.;
    for i in 0..n {
        for j in 0..m {
            err += err_matrix[(i, j)] * err_matrix[(i, j)];
        }
    }

    Ok(Model {
        w,
        err,
    })
}
fn qr_factorization(matrix: &DMatrix<f64>) -> Result<QRResult, Error> {
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
    Ok(QRResult {
        q,
        r,
    })
}
fn read_testing_files() -> Vec<Image> {
    let mut int_buf = [0 ; 4];
    let mut images_buf = Vec::new();
    let mut labels_buf = Vec::new();

    // Opens files and tests magic numbers
    let mut test_images_file = File::open(&Path::new("data\\test-images.idx3-ubyte")).unwrap();
    test_images_file.read_exact(&mut int_buf).unwrap();
    assert_eq!(int_buf, [0, 0, 8, 3], "Wrong magic number for testing images!");
    let mut test_labels_file = File::open(&Path::new("data\\test-labels.idx1-ubyte")).unwrap();
    test_labels_file.read_exact(&mut int_buf).unwrap();
    assert_eq!(int_buf, [0, 0, 8, 1], "Wrong magic number for testing labels!");

    // Test set
    test_images_file.read_exact(&mut int_buf).unwrap();
    let test_num = u32::from_be_bytes(int_buf) as usize;
    test_labels_file.read_exact(&mut int_buf).unwrap();
    assert_eq!(test_num, u32::from_be_bytes(int_buf) as usize, "Testing images and labels have mismatching sizes!");
    test_images_file.read_exact(&mut int_buf).unwrap();
    test_images_file.read_exact(&mut int_buf).unwrap();

    test_images_file.read_to_end(&mut images_buf).unwrap();
    test_labels_file.read_to_end(&mut labels_buf).unwrap();
    let mut test_images = Vec::new();
    for i in 0..test_num {
        print!("\rReading testing image {}/{test_num}", i + 1);
        let data = DVector::<f64>::from_iterator(784, images_buf[i*784..(i+1)*784].iter().map(|x| *x as f64 / 255.));
        let label = labels_buf[i] as usize;
        assert!(label < 10, "Invalid label in testing set!");

        test_images.push(Image {
            data,
            label,
        });
    }
    println!();

    test_images
}
fn read_training_files() -> Vec<Image> {
    let mut int_buf = [0 ; 4];
    let mut images_buf = Vec::new();
    let mut labels_buf = Vec::new();

    // Opens files and tests magic numbers
    let mut training_images_file = File::open(&Path::new("data\\train-images.idx3-ubyte")).unwrap();
    training_images_file.read_exact(&mut int_buf).unwrap();
    assert_eq!(int_buf, [0, 0, 8, 3], "Wrong magic number for training images!");
    let mut training_labels_file = File::open(&Path::new("data\\train-labels.idx1-ubyte")).unwrap();
    training_labels_file.read_exact(&mut int_buf).unwrap();
    assert_eq!(int_buf, [0, 0, 8, 1], "Wrong magic number for training labels!");

    // Training set
    training_images_file.read_exact(&mut int_buf).unwrap();
    let training_num = u32::from_be_bytes(int_buf) as usize;
    training_labels_file.read_exact(&mut int_buf).unwrap();
    assert_eq!(training_num, u32::from_be_bytes(int_buf) as usize, "Training images and labels have mismatching sizes!");
    training_images_file.read_exact(&mut int_buf).unwrap();
    training_images_file.read_exact(&mut int_buf).unwrap();

    training_images_file.read_to_end(&mut images_buf).unwrap();
    training_labels_file.read_to_end(&mut labels_buf).unwrap();
    let mut training_images = Vec::new();
    for i in 0..training_num {
        print!("\rReading training image {}/{training_num}", i + 1);
        let data = DVector::<f64>::from_iterator(784, images_buf[i*784..(i+1)*784].iter().map(|x| *x as f64 / 255.));
        let label = labels_buf[i] as usize;
        assert!(label < 10, "Invalid label in training set!");

        training_images.push(Image {
            data,
            label,
        });
    }
    println!();

    training_images
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
            Ok(result) => result.x,
            Err(err) => return Err(err),
        });
    }

    Ok(DMatrix::from_columns(&x_set))
}
fn solve_system(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<SolveSystemResult, Error> {
    let n = a.nrows();
    let m = a.ncols();
    if n != b.len() {
        return Err(Error::MismatchedDimensions);
    }

    let qr_result = qr_factorization(&a)?;

    let q_set: Vec<DVector<f64>> = qr_result.q
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
            x[i] -= qr_result.r[(i, j)] * x[j];
        }
        x[i] += z[i];
        x[i] /= qr_result.r[(i, i)];
    }
    Ok(SolveSystemResult {
        x,
        err: b_set[m].norm(),
    })
}
fn run_testing_mode() {
    let mut buffer = String::new();

    let model_folder_name = loop {
        print!("Model name: ");
        stdout().flush().unwrap();
        buffer.clear();
        stdin().read_line(&mut buffer).unwrap();
        match buffer.trim() {
            "" => {}
            txt => {
                let txt_path = format!("models\\{txt}");
                let path = Path::new(&txt_path);
                if path.is_dir() {
                    break txt_path;
                }
            }
        }
    };

    println!("Reading model");
    let model_set = {
        let mut model_set = Vec::new();
        for _ in 0..10 {
            model_set.push(None);
        }
        for entry in read_dir(Path::new(&model_folder_name)).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap();
            let label = match name[..name.len() - 5].parse::<usize>() {
                Ok(label) => label,
                Err(_) => continue,
            };
            if label > 9 {
                continue;
            }
            let file = File::open(path).unwrap();
            let model: Model = serde_json::from_reader(file).unwrap();
            model_set[label] = Some(model);
        }

        for label in 0..10 {
            if model_set[label].is_none() {
                println!("Input model incomplete!");
                return;
            }
        }
        Arc::new(model_set.into_iter().map(|x| x.unwrap()).collect::<Vec<Model>>())
    };

    let testing_images = read_testing_files();
    let total = testing_images.len();

    println!("Testing start!");
    let start = Instant::now();

    let num_thread = available_parallelism().unwrap().get() + 1;
    println!("Running on {num_thread} threads");
    let threadpool = ThreadPool::new(num_thread);

    let results_mutex = Arc::new(Mutex::new(TestingResult::new()));
    for image in testing_images {
        let results_mutex_clone = results_mutex.clone();
        let model_set_clone = model_set.clone();
        threadpool.execute(move || {
            let v = &image.data;

            let mut residues = Vec::new();
            for _ in 0..10 {
                residues.push(0.);
            }
            for label in 0..10 {
                let w = &model_set_clone[label].w;
                let err = match solve_system(w, v) {
                    Ok(solve_system_result) => solve_system_result.err,
                    Err(err) => {
                        println!("{err}");
                        return;
                    }
                };

                residues[label] = err;
            }

            let mut results = results_mutex_clone.lock().unwrap();
            results.push(ClassificationResult::new(residues, image));
        });
    }

    loop {
        let results_num = results_mutex.lock().unwrap().len();
        print!("\rTesting image {results_num}/{total}");
        stdout().flush().unwrap();
        if results_num == total {
            break;
        }
        sleep(Duration::from_secs(1));
    }
    println!();
    threadpool.join();
    let delta = start.elapsed();
    println!("Testing done! Time: {delta:.2?}");

    let testing_results = Arc::try_unwrap(results_mutex).unwrap().into_inner().unwrap();
    let correct_num = testing_results.correct_num();
    let incorrect_num = total - correct_num;

    let mut window = create_window("Image", WindowOptions {
        size: Some([500, 500]),
        resizable: false,
        default_controls: false,
        ..Default::default()
    }).unwrap();
    loop {
        print!("Insert command (enter \"help\" for help): ");
        stdout().flush().unwrap();
        buffer.clear();
        stdin().read_line(&mut buffer).unwrap();
        let tokens: Vec<&str> = buffer.trim().split(" ").collect();
        if tokens.len() == 0 {
            continue;
        }
        match tokens[0] {
            "correct" => {
                if tokens.len() != 2 {
                    println!("Invalid index!");
                    continue;
                }
                match tokens[1].parse::<usize>() {
                    Ok(index) => {
                        if index >= correct_num {
                            println!("Index out of bounds! Maximum of {}.", correct_num - 1);
                            continue;
                        }

                        let mut var_index = index + 1;
                        let mut true_index = 0;
                        loop {
                            if testing_results.results[true_index].is_correct() {
                                var_index -= 1;
                            }
                            if var_index == 0 {
                                break;
                            }
                            true_index += 1;
                        };

                        show_image_and_prediction(&mut window, &testing_results, true_index);
                    }
                    Err(_) => println!("Invalid index!"),
                }
            }
            "help" => {
                println!("correct i - Views correctly labeled image number i.");
                println!("incorrect i - Views incorrectly labeled image number i.");
                println!("overview - Overview of the results.");
                println!("quit - Quits testing.");
                println!("view i - Views test image number i.");
            }
            "incorrect" => {
                if tokens.len() != 2 {
                    println!("Invalid index!");
                    continue;
                }
                match tokens[1].parse::<usize>() {
                    Ok(index) => {
                        if index >= incorrect_num {
                            println!("Index out of bounds! Maximum of {}.", incorrect_num - 1);
                            continue;
                        }

                        let mut var_index = index + 1;
                        let mut true_index = 0;
                        loop {
                            if !testing_results.results[true_index].is_correct() {
                                var_index -= 1;
                            }
                            if var_index == 0 {
                                break;
                            }
                            true_index += 1;
                        };

                        show_image_and_prediction(&mut window, &testing_results, true_index);
                    }
                    Err(_) => println!("Invalid index!"),
                }
            }
            "overview" => {
                println!("Result: {correct_num} correct of {total} total.");
                println!("Accuracy: {:.2}%", 100. * (correct_num as f64 / total as f64));
                println!("Residues from NMF training: ");
                for i in 0..10 {
                    print!("{i}: {:5.03e}   ", model_set[i].err);
                }
                println!();
            }
            "quit" => break,
            "view" => {
                if tokens.len() != 2 {
                    println!("Invalid index!");
                    continue;
                }
                match tokens[1].parse::<usize>() {
                    Ok(index) => {
                        if index >= total {
                            println!("Index out of bounds! Maximum of {}.", total - 1);
                            continue;
                        }

                        show_image_and_prediction(&mut window, &testing_results, index);
                    }
                    Err(_) => println!("Invalid index!"),
                }
            }
            _ => {}
        }
    }
}
fn run_training_mode() {
    let mut buffer = String::new();
    let existing_model = loop {
        print!("Base model name (leave empty for new model): ");
        stdout().flush().unwrap();
        buffer.clear();
        stdin().read_line(&mut buffer).unwrap();
        match buffer.trim() {
            "" => break None,
            txt => {
                let txt_path = format!("models\\{txt}");
                let path = Path::new(&txt_path);
                if path.is_dir() {
                    break Some(txt_path);
                }
            }
        }
    };
    let max_iter = loop {
        print!("How many iterations should be done at max? ");
        stdout().flush().unwrap();
        buffer.clear();
        stdin().read_line(&mut buffer).unwrap();
        match buffer.trim().parse::<usize>() {
            Ok(num) => match num {
                0 => {}
                _ => break num,
            }
            Err(_) => {}
        }
    };

    let training_images = read_training_files();

    println!("Building training matrices");
    let mut sorted_training_data = Vec::new();
    for _ in 0..10 {
        sorted_training_data.push(Vec::new());
    }
    for image in training_images.into_iter() {
        sorted_training_data[image.label].push(image.data);
    }
    let mut training_matrices = Vec::new();
    for label in 0..10 {
        training_matrices.push(DMatrix::<f64>::from_columns(&sorted_training_data[label]));
    }
    let model_set = if let Some(folder_name) = existing_model {
        println!("Reading base model");
        let mut model_set = Vec::new();
        for _ in 0..10 {
            model_set.push(None);
        }
        for entry in read_dir(Path::new(&folder_name)).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap();
            let label = match name[..name.len()-5].parse::<usize>() {
                Ok(label) => label,
                Err(_) => continue,
            };
            if label > 9 {
                continue;
            }
            let file = File::open(path).unwrap();
            let model: Model = serde_json::from_reader(file).unwrap();
            model_set[label] = Some(model);
        }
        for label in 0..10 {
            if model_set[label].is_none() {
                println!("Input model incomplete!");
                return;
            }
        }
        model_set.into_iter().map(|x| x.unwrap()).collect::<Vec<Model>>()
    } else {
        let mut model_set = Vec::new();
        let dist = Uniform::new_inclusive(0., 1.);

        for i in 0..10 {
            model_set.push(Model {
                w: DMatrix::<f64>::from_distribution(training_matrices[i].clone().nrows(), 10, &dist, &mut thread_rng()),
                err: 0.,
            });
        }

        model_set
    };

    println!("Training start!");
    let start = Instant::now();

    let num_thread = available_parallelism().unwrap().get() + 1;
    println!("Running on {num_thread} threads");
    let threadpool = ThreadPool::new(num_thread);

    let training_result_mutex = Arc::new(Mutex::new(TrainingResult::new(model_set)));
    loop {
        let mut done = true;
        let mut training_result = training_result_mutex.lock().unwrap();
        for i in 0..10 {
            let model_err = training_result.get_err(i);
            let (ready, it) = training_result.get_mut_status(i);
            if *ready {
                if *it < max_iter {
                    done = false;

                    println!("Iteration {it}/{max_iter} of digit {i}. Error: {model_err:5.03e}");

                    *ready = false;
                    *it += 1;

                    let training_result_mutex_clone = training_result_mutex.clone();
                    let matrix = training_matrices[i].clone();
                    threadpool.execute(move || {
                        let w_init = training_result_mutex_clone.lock().unwrap().get_w_clone(i);

                        let model = match nmf_training(&matrix, &w_init) {
                            Ok(model) => model,
                            Err(err) => {
                                panic!("{}", err);
                            }
                        };

                        let mut training_result = training_result_mutex_clone.lock().unwrap();
                        training_result.update(i, model);
                        training_result.set_ready(i);
                    });
                } else if *it == max_iter {
                    println!("Iteration {it}/{max_iter} of digit {i}. Error: {model_err:5.03e}");

                    *it += 1;
                }
            } else {
                done = false;
            }
        }
        drop(training_result);

        if done {
            break;
        }

        sleep(Duration::from_millis(500));
    }

    let delta = start.elapsed();
    println!("Training done! Time: {delta:.2?}");

    let training_results = Arc::try_unwrap(training_result_mutex).unwrap().into_inner().unwrap();

    println!("Saving models to json");
    let mut num = 0;
    let folder_name = loop {
        let folder_name = format!("models\\model{num}");
        if Path::new(&folder_name).is_dir() {
            num += 1;
        } else {
            break folder_name;
        }
    };
    fs::create_dir_all(&folder_name).unwrap();
    for label in 0..10 {
        let mut file = File::create(&format!("{folder_name}\\{label}.json")).unwrap();
        write!(file, "{}", serde_json::to_string(training_results.get_model(label)).unwrap()).unwrap();
    }
    println!("Models saved at {folder_name}!");
}
fn show_image_and_prediction(window: &mut WindowProxy, testing_results: &TestingResult, index: usize) {
    let data: Vec<u8> = testing_results.results[index].image.data.iter().map(|x| (*x * 255.) as u8).collect();
    let image = ImageView::new(ImageInfo::mono8(28, 28), data.as_slice());

    match window.set_image(format!("image{index}"), image) {
        Ok(_) => {}
        Err(err) => match err {
            SetImageError::InvalidWindowId(_) => {
                *window = create_window("Image", WindowOptions {
                    size: Some([500, 500]),
                    resizable: false,
                    default_controls: false,
                    ..Default::default()
                }).unwrap();
                window.set_image(format!("image{index}"), image).unwrap();
            }
            SetImageError::ImageDataError(err) => panic!("{:?}", err),
        }
    }

    println!("Prediction: {}   Label: {}", testing_results.results[index].prediction(), testing_results.results[index].image.label);
    println!("Residues:");
    for label in 0..10 {
        print!("{label}: {:5.3e}   ", testing_results.results[index].residues[label]);
    }
    println!();
}

enum Error {
    MatrixIsSingular,
    MismatchedDimensions,
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Error::MatrixIsSingular => "Matrix is singular!",
            Error::MismatchedDimensions => "Dimensions are mismatched!",
        })
    }
}
#[derive(Debug)]
struct Image {
    data: DVector<f64>,
    label: usize,
}
#[derive(Debug, Deserialize, Serialize)]
struct Model {
    pub w: DMatrix<f64>,
    pub err: f64,
}
struct QRResult {
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
}
#[derive(Debug)]
struct TestingResult {
    results: Vec<ClassificationResult>,
    correct_num: usize,
}
impl TestingResult {
    pub fn new() -> Self {
        TestingResult {
            results: Vec::new(),
            correct_num: 0,
        }
    }

    pub fn correct_num(&self) -> usize {
        self.correct_num
    }
    pub fn len(&self) -> usize {
        self.results.len()
    }
    pub fn push(&mut self, test_result: ClassificationResult) {
        if test_result.is_correct() {
            self.correct_num += 1;
        }
        self.results.push(test_result);
    }
}
#[derive(Debug)]
struct TrainingResult {
    model_set: Vec<Model>,
    status: Vec<(bool, usize)>,
}
impl TrainingResult {
    pub fn new(model_set: Vec<Model>) -> Self {
        assert_eq!(model_set.len(), 10, "Wrong number of elements in model set!");

        TrainingResult {
            status: vec![(true, 0) ; 10],
            model_set,
        }
    }

    pub fn get_err(&self, label: usize) -> f64 {
        self.model_set[label].err
    }
    pub fn get_model(&self, label: usize) -> &Model {
        &self.model_set[label]
    }
    pub fn get_mut_status(&mut self, label: usize) -> (&mut bool, &mut usize) {
        let (ready, it) = &mut self.status[label];
        (ready, it)
    }
    pub fn get_w_clone(&self, label: usize) -> DMatrix<f64> {
        self.model_set[label].w.clone()
    }
    pub fn set_ready(&mut self, label: usize) {
        self.status[label].0 = true;
    }
    pub fn update(&mut self, label: usize, model: Model) {
        self.model_set[label] = model;
    }
}
struct SolveSystemResult {
    pub x: DVector<f64>,
    pub err: f64,
}
#[derive(Debug)]
struct ClassificationResult {
    residues: Vec<f64>,
    image: Image,
}
impl ClassificationResult {
    pub fn new(residues: Vec<f64>, image: Image) -> Self {
        ClassificationResult {
            residues,
            image,
        }
    }

    pub fn is_correct(&self) -> bool {
        self.prediction() == self.image.label
    }
    pub fn prediction(&self) -> usize {
        let mut prediction = 0;
        let mut prediction_residue = f64::MAX;
        for label in 0..10 {
            let residue = self.residues[label];
            if residue < prediction_residue {
                prediction = label;
                prediction_residue = residue;
            }
        }

        prediction
    }
}