//! Бонус Д3 (неделя 1): полный малый цикл Keras -> tflite -> Rust.
//!
//! Модель `dense_spike.tflite` собрана скриптом `ml/scripts/build_dense_model.py`
//! (Dense(16, relu) -> Dense(4) -> Softmax, full-int8), вход (1, 8), выход (1, 4).

use microflow::model;
use nalgebra::matrix;

#[model("models/dense_spike.tflite")]
struct DenseSpike;

fn main() {
    let input = matrix![0.5, -1.2, 0.3, 0.8, -0.4, 1.1, 0.0, 0.25];
    let output = DenseSpike::predict(input);
    println!();
    println!("Input:    [0.5, -1.2, 0.3, 0.8, -0.4, 1.1, 0.0, 0.25]");
    print!("Probabilities: [");
    for (i, p) in output.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{p:.4}");
    }
    println!("]");
    let argmax = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(usize::MAX);
    println!("Predicted class: {argmax}");
}
