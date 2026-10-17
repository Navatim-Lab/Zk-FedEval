#![no_main]
sp1_zkvm::entrypoint!(main);

use zk_fedeval_common::int8::Int8Model;
use zk_fedeval_common::EvalReport;

/// The quantized weights, baked into the ELF so the verifying key identifies the model.
static MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model_int8.bin"));

pub fn main() {
    // Raw u8 pixels (`n * 3072`, CHW) and u8 labels; no float decoding in the guest.
    let images = sp1_zkvm::io::read_vec();
    let labels = sp1_zkvm::io::read_vec();

    let model = Int8Model::from_bytes(MODEL).expect("invalid int8 model");
    let correct = model.count_correct(&images, &labels);

    let report = EvalReport {
        num_samples: labels.len() as u32,
        correct,
    };

    // Same public value as the f32 guest, so `FedEval` verifies either one.
    let encoded_report = parity_scale_codec::Encode::encode(&report);
    sp1_zkvm::io::commit(&sp_crypto_hashing::keccak_256(&encoded_report));
}
