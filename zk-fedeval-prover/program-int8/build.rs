use std::path::Path;

/// Partition whose quantized model is compiled into the guest program.
const PARTITION_ID: &str = "partition-0";

/// Copy `model_int8.bin` (written by `python -m zk_fedeval.quantize`) into `OUT_DIR`.
///
/// A missing file becomes an empty placeholder so the f32-only workflow still builds;
/// the guest then fails at runtime with a message telling you to run the quantizer.
fn main() {
    let model_path = format!("../../outputs/eval/{PARTITION_ID}/model_int8.bin");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("model_int8.bin");
    let bytes = std::fs::read(&model_path).unwrap_or_default();
    std::fs::write(out, bytes).unwrap();

    println!("cargo:rerun-if-changed={model_path}");
}
