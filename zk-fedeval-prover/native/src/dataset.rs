use std::fs::File;
use std::path::Path;

use anyhow::{Context as _, Result};
use burn::prelude::*;
use ndarray::{Array1, Array4};
use ndarray_npy::NpzReader;

/// CIFAR-100-shaped evaluation set exported by the Python client (`eval_data.npz`).
pub struct EvalDataset {
    images: Array4<f32>,
    labels: Array1<i64>,
}

impl EvalDataset {
    /// Load images (`N,3,32,32` f32) and labels (`N` i64) from an `.npz` archive.
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

    /// Convert the dataset into burn tensors on the given device.
    pub fn to_tensors<B: Backend>(&self, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 1, Int>) {
        let shape = self.images.shape();
        let [n, c, h, w] = [shape[0], shape[1], shape[2], shape[3]];

        let image_data: Vec<f32> = self.images.iter().copied().collect();
        let images = Tensor::<B, 1>::from_floats(image_data.as_slice(), device)
            .reshape([n, c, h, w]);

        let label_data: Vec<i32> = self.labels.iter().map(|&label| label as i32).collect();
        let labels = Tensor::<B, 1, Int>::from_ints(label_data.as_slice(), device);

        (images, labels)
    }
}
