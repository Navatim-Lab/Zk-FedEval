#![no_main]
sp1_zkvm::entrypoint!(main);

mod evaluator;
mod model;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;

use evaluator::Evaluator;
use model::CnnModel;
use zk_fedeval_common::{EvalBatch, EvalReport, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH};

type ProverBackend = NdArray<f32>;

pub fn main() {
    let batch: EvalBatch = sp1_zkvm::io::read();
    let num_samples = batch.num_samples();

    let device = NdArrayDevice::default();

    let images = Tensor::<ProverBackend, 1>::from_floats(batch.images.as_slice(), &device)
        .reshape([num_samples, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);

    let label_data: Vec<i32> = batch.labels.iter().map(|&label| label as i32).collect();
    let labels = Tensor::<ProverBackend, 1, Int>::from_ints(label_data.as_slice(), &device);

    let model = CnnModel::<ProverBackend>::load(&device);
    let evaluator = Evaluator::new(model);
    let correct = evaluator.count_correct(images, labels);

    let report = EvalReport {
        num_samples: num_samples as u32,
        correct,
    };

    // commit only the hash of the report and we will store the report data
    let encoded_report = parity_scale_codec::Encode::encode(&report);
    let report_hash = sp_crypto_hashing::keccak256(&encoded_report);
    sp1_zkvm::io::commit(&report_hash);
    
}
