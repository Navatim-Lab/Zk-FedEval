use burn_import::onnx::ModelGen;

/// Partition whose exported ONNX model is compiled into the prover.
const PARTITION_ID: &str = "partition-0";

fn main() {
    let onnx_path = format!("../../outputs/eval/{PARTITION_ID}/model.onnx");

    ModelGen::new()
        .input(&onnx_path)
        .out_dir("model/")
        .run_from_script();

    println!("cargo:rerun-if-changed={onnx_path}");
}
