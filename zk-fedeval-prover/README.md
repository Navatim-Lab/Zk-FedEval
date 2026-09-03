# zk-fedeval-prover

Rust prover for the Zk-FedEval pipeline.

This crate takes the trained ONNX model exported from the Python FedEval workflow, loads it as a runnable inference graph, runs evaluation on CIFAR-10 data, and uses the Succinct SP1 prover to generate a zero-knowledge proof of the CNN evaluation step.

## Pipeline

1. **Input** — ONNX file produced by the Python `zk_fedeval` training/export step.
2. **Model** — Convert and load the ONNX graph into a running evaluation model.
3. **Data** — Run inference on CIFAR-10 samples.
4. **Proof** — Execute the evaluation logic inside SP1 and output a proof of the result.
