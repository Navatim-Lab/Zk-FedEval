use std::path::Path;

/// Partition whose quantized model the native int8 twin checks against.
const PARTITION_ID: &str = "partition-0";

fn main() {
    sp1_build::build_program("../program");
    sp1_build::build_program("../program-int8");

    // The native twin embeds the same `model_int8.bin` as the int8 guest; a missing
    // file becomes an empty placeholder so the f32-only workflow still builds.
    let model_path = format!("../../outputs/eval/{PARTITION_ID}/model_int8.bin");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("model_int8.bin");
    std::fs::write(out, std::fs::read(&model_path).unwrap_or_default()).unwrap();

    println!("cargo:rerun-if-changed={model_path}");
}
