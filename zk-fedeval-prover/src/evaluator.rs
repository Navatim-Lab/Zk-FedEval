use std::fmt;

use burn::prelude::*;

use crate::dataset::EvalDataset;
use crate::model::CnnModel;

/// Accuracy/loss summary produced by running a model over an [`EvalDataset`].
pub struct EvaluationReport {
    pub num_samples: usize,
    pub correct: usize,
}

impl EvaluationReport {
    pub fn accuracy(&self) -> f32 {
        if self.num_samples == 0 {
            return 0.0;
        }
        self.correct as f32 / self.num_samples as f32
    }
}

impl fmt::Display for EvaluationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "accuracy={:.4} ({}/{} correct)",
            self.accuracy(),
            self.correct,
            self.num_samples,
        )
    }
}

/// Runs a [`CnnModel`] over an [`EvalDataset`] and reports the resulting accuracy.
pub struct Evaluator<B: Backend> {
    model: CnnModel<B>,
    device: B::Device,
}

impl<B: Backend> Evaluator<B> {
    pub fn new(model: CnnModel<B>, device: B::Device) -> Self {
        Self { model, device }
    }

    pub fn run(&self, dataset: &EvalDataset) -> EvaluationReport {
        let (images, labels) = dataset.to_tensors::<B>(&self.device);

        let logits = self.model.predict(images);
        let predictions = logits.argmax(1).squeeze::<1>();

        let correct = predictions
            .equal(labels)
            .int()
            .sum()
            .into_scalar()
            .elem::<i64>() as usize;

        EvaluationReport {
            num_samples: dataset.len(),
            correct,
        }
    }
}
