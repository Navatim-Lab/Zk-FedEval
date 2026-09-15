use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{s, Array1, Array4};
use ndarray_npy::NpzReader;
use zk_fedeval_common::EvalBatch;

/// Reads the `.npz` evaluation set exported by the Python client and slices
/// off the first `batch_size` samples to prove, since proving a full CNN
/// forward pass inside the zkVM is far more expensive than native execution.
pub struct EvalDataSource {
    images: Array4<f32>,
    labels: Array1<i64>,
}

impl EvalDataSource {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening eval data at {}", path.display()))?;
        let mut npz = NpzReader::new(file)
            .with_context(|| format!("reading npz archive at {}", path.display()))?;

        let images: Array4<f32> = npz
            .by_name("images.npy")
            .context("reading 'images' array from eval data")?;
        let labels: Array1<i64> = npz
            .by_name("labels.npy")
            .context("reading 'labels' array from eval data")?;

        anyhow::ensure!(
            images.shape()[0] == labels.len(),
            "image count ({}) does not match label count ({})",
            images.shape()[0],
            labels.len(),
        );

        Ok(Self { images, labels })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Take the first `batch_size` samples as a private witness batch for the guest.
    pub fn take_batch(&self, batch_size: usize) -> EvalBatch {
        let n = batch_size.min(self.len());

        let images = self.images.slice(s![0..n, .., .., ..]).iter().copied().collect();
        let labels = self.labels.slice(s![0..n]).iter().copied().collect();

        EvalBatch { images, labels }
    }
}
