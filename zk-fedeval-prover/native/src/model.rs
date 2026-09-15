use burn::prelude::*;

/// Code generated at build time by `burn-import` from the exported ONNX model.
mod generated {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

/// The CNN exported from the federated learning client, ready for inference.
pub struct CnnModel<B: Backend> {
    inner: generated::Model<B>,
}

impl<B: Backend> CnnModel<B> {
    /// Instantiate the model with its embedded ONNX weights on the given device.
    ///
    /// `generated::Model::new` only builds the architecture with fresh
    /// (untrained) parameters; the real weights exported from the ONNX file
    /// live in the sibling `model.bpk` burnpack file and must be loaded
    /// explicitly.
    pub fn load(device: &B::Device) -> Self {
        let weights_path = concat!(env!("OUT_DIR"), "/model/model.bpk");
        Self {
            inner: generated::Model::from_file(weights_path, device),
        }
    }

    /// Run a forward pass, returning per-class logits of shape `[batch, classes]`.
    pub fn predict(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        self.inner.forward(images)
    }
}
