use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{s, Array1, Array4};
use ndarray_npy::NpzReader;
use zk_fedeval_common::EvalBatch;

/// Reads the `.npz` evaluation set exported by the Python client and slices
/// off batches to prove, since proving a full CNN forward pass inside the
/// zkVM is far more expensive than native execution.
pub struct EvalDataSource {
    images: Array4<f32>,
    labels: Array1<i64>,
}

/// A batch for the int8 guest: raw 0-255 pixels (`n * 3072`, CHW) and u8 labels.
pub struct RawBatch {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
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

    /// Clamp `[start, start + batch_size)` to the available samples.
    pub fn range(&self, start: usize, batch_size: usize) -> std::ops::Range<usize> {
        start.min(self.len())..(start + batch_size).min(self.len())
    }

    /// Take a slice of samples as a private `f32` witness batch for the Burn guest.
    pub fn take_batch(&self, start: usize, batch_size: usize) -> EvalBatch {
        let r = self.range(start, batch_size);

        let images = self.images.slice(s![r.clone(), .., .., ..]).iter().copied().collect();
        let labels = self.labels.slice(s![r]).iter().copied().collect();

        EvalBatch { images, labels }
    }

    /// Take a slice of samples for the int8 guest, undoing the Python client's
    /// `Normalize((0.5,) * 3, (0.5,) * 3)`: pixel = round((x + 1) * 127.5).
    pub fn take_raw_batch(&self, start: usize, batch_size: usize) -> RawBatch {
        let r = self.range(start, batch_size);

        let images = self
            .images
            .slice(s![r.clone(), .., .., ..])
            .iter()
            .map(|&x| ((x as f64 + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
            .collect();
        let labels = self.labels.slice(s![r]).iter().map(|&label| label as u8).collect();

        RawBatch { images, labels }
    }
}
