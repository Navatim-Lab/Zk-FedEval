# zk-fedeval-prover

Rust/SP1 evaluation prover for the Zk-FedEval CIFAR-100 CNN. It runs CNN inference using Burn inside the SP1 guest and generates a Groth16 proof of evaluation execution. The workspace uses SP1 6.6.0 and Burn 0.21.0.

## Workspace

| Crate directory | Responsibility |
| --- | --- |
| `common` | Shared image dimensions, `EvalBatch`, and `EvalReport { num_samples, correct }`. |
| `native` | Reference CNN evaluation outside SP1. |
| `program` | Guest inference with `NdArray<f32>`, argmax predictions, and correct-prediction counting. |
| `host` | Evaluation-data loading, batch selection, program-key setup, Groth16 proving, JSON export, and local verification. |

## Evaluation proof

Images and labels are supplied as the guest's private evaluation input. The guest counts correct predictions and commits `keccak256(SCALE(EvalReport))`. The host writes the program verification key, proof, and public values to `prover-report.json`.

The current host proves one selected batch using `.groth16()`. It does not aggregate evaluation proofs across clients or rounds, and Flower does not yet use these proofs to gate model updates.

The guest currently commits a report hash while the host reads public values as an `EvalReport`; the reporting interface is inconsistent.

## Open question: N-round evaluation proving

How to make N rounds of evaluation fast to prove remains unresolved. The role, cost, and design of proof aggregation are still under investigation. No multi-round proving strategy has been selected or benchmarked.

See the [main README's open question](../README.md#open-question-proving-n-rounds-of-evaluation) for the project-level problem statement.
