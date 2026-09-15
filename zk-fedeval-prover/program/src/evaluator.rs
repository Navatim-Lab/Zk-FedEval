use burn::prelude::*;

use crate::model::CnnModel;

/// Runs a [`CnnModel`] over a batch and counts how many predictions match
/// their labels.
pub struct Evaluator<B: Backend> {
    model: CnnModel<B>,
}

impl<B: Backend> Evaluator<B> {
    pub fn new(model: CnnModel<B>) -> Self {
        Self { model }
    }

    pub fn count_correct(&self, images: Tensor<B, 4>, labels: Tensor<B, 1, Int>) -> u32 {
        let logits = self.model.predict(images);
        // `squeeze::<1>()` drops *every* size-1 dim, which breaks for a
        // batch of exactly 1 (both dims are size 1, so it can't reach 1D).
        // Squeeze only the class axis explicitly instead.
        let predictions = logits.argmax(1).squeeze_dims::<1>(&[1]);

        predictions
            .equal(labels)
            .int()
            .sum()
            .into_scalar()
            .elem::<i64>() as u32
    }
}
