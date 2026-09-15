//! Types shared across the SP1 guest (`program`) and host (`host`) crates.
//!
//! Keeping these in one crate guarantees both sides serialize/deserialize
//! the exact same shape over the guest's stdin/public-values boundary.

use serde::{Deserialize, Serialize};

pub const IMAGE_CHANNELS: usize = 3;
pub const IMAGE_HEIGHT: usize = 32;
pub const IMAGE_WIDTH: usize = 32;

/// The private input fed to the guest: a batch of CIFAR-100-shaped images
/// (flattened `[n, 3, 32, 32]`, row-major) and their ground-truth labels.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvalBatch {
    pub images: Vec<f32>,
    pub labels: Vec<i64>,
}

impl EvalBatch {
    pub fn num_samples(&self) -> usize {
        self.labels.len()
    }
}

/// The public output the guest commits to: how many of the batch's
/// predictions matched their label.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalReport {
    pub num_samples: u32,
    pub correct: u32,
}

impl EvalReport {
    pub fn accuracy(&self) -> f32 {
        if self.num_samples == 0 {
            return 0.0;
        }
        self.correct as f32 / self.num_samples as f32
    }
}
