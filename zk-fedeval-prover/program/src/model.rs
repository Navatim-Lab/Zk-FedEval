use burn::prelude::*;
use burn::tensor::Bytes;

/// Code generated at build time by `burn-import` from the exported ONNX model.
mod generated {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

/// The CNN exported from the federated learning client, ready for inference
/// inside the SP1 guest.
pub struct CnnModel<B: Backend> {
    inner: generated::Model<B>,
}

/// The sibling `model.bpk` burnpack file baked into `OUT_DIR` by `build.rs`,
/// embedded directly into the guest binary at compile time: the zkVM guest
/// has no real filesystem, so weights must arrive as bytes compiled into the
/// program rather than read from a path at runtime.
static WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model/model.bpk"));

impl<B: Backend> CnnModel<B> {
    /// Instantiate the model with its embedded ONNX weights on the given device.
    pub fn load(device: &B::Device) -> Self {
        let bytes = Bytes::from_bytes_vec(WEIGHTS.to_vec());
        Self {
            inner: generated::Model::from_bytes(bytes, device),
        }
    }

    /// Run a forward pass, returning per-class logits of shape `[batch, classes]`.
    pub fn predict(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        self.inner.forward(images)
    }
}
