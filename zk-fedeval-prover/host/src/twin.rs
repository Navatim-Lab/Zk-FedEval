use std::path::Path;

use anyhow::{ensure, Result};
use ndarray::Array1;
use parity_scale_codec::Encode;
use zk_fedeval_common::int8::{Int8Model, Scratch, PIXELS_PER_IMAGE};
use zk_fedeval_common::EvalReport;

use crate::dataset::RawBatch;

/// The same quantized model the int8 guest embeds (both copied from
/// `outputs/eval/partition-0/model_int8.bin` at build time).
static MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model_int8.bin"));

/// Runs the int8 CNN natively with the guest's own kernel, so every proven batch can
/// be checked against an expected report, and the kernel against the Python reference.
pub struct NativeTwin {
    model: Int8Model,
    /// `int8_predictions.npy` from `python -m zk_fedeval.quantize`, when present.
    python_predictions: Option<Array1<i64>>,
}

impl NativeTwin {
    pub fn load(eval_data_path: &Path) -> Result<Self> {
        let model = Int8Model::from_bytes(MODEL).map_err(anyhow::Error::msg)?;
        let predictions_path = eval_data_path.with_file_name("int8_predictions.npy");
        let python_predictions = ndarray_npy::read_npy(&predictions_path).ok();
        if python_predictions.is_none() {
            println!("[twin] no {} found; skipping the Python cross-check", predictions_path.display());
        }
        Ok(Self {
            model,
            python_predictions,
        })
    }

    /// The report the guest should commit for `batch`, whose first sample is `start`.
    pub fn expected_report(&self, batch: &RawBatch, start: usize) -> Result<EvalReport> {
        let mut scratch = Scratch::default();
        let predictions: Vec<usize> = batch
            .images
            .chunks_exact(PIXELS_PER_IMAGE)
            .map(|image| self.model.predict(image, &mut scratch))
            .collect();

        if let Some(python) = &self.python_predictions {
            for (i, &rust) in predictions.iter().enumerate() {
                ensure!(
                    python[start + i] == rust as i64,
                    "sample {}: Rust int8 kernel predicts {rust}, Python reference predicts {}",
                    start + i,
                    python[start + i],
                );
            }
        }

        let correct = predictions
            .iter()
            .zip(&batch.labels)
            .filter(|(p, label)| **p == **label as usize)
            .count() as u32;
        Ok(EvalReport {
            num_samples: batch.labels.len() as u32,
            correct,
        })
    }
}

/// The guest's public value for a report: keccak-256 of its SCALE encoding.
pub fn report_hash(report: &EvalReport) -> [u8; 32] {
    sp_crypto_hashing::keccak_256(&report.encode())
}
