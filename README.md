# Zk-FedEval

Privacy-preserving, verifiable evaluation for federated learning, targeting PolkaVM verification in the Polkadot/Kusama ecosystem.

Grant: [Kusama ZK RFP #002 — Privacy-Preserving FL](https://zk.kusama.vision/rfps/#002-Privacy-Preserving%20FL).

The goal is to let clients prove model evaluation results without publishing private evaluation samples. Flower orchestrates federated learning, and the Rust SP1 prover generates proofs of evaluation execution.

Although the current implementation is built around a CNN, the broader vision is to extend the approach to any model and integrate fully homomorphic encryption (FHE) using the HEIR compiler. The goal is to enable private learning, insight sharing, and interoperability while protecting the underlying data.

Potential applications include hospitals, agriculture, military organizations, and other settings where data sensitivity is critical. These organizations need to learn from sensitive data and share useful insights or model improvements while preserving the confidentiality of the source data.

## Current components

| Component | Role |
| --- | --- |
| Flower application | Local simulation with two clients, one round by default, a 100-output PyTorch CNN, deterministic non-IID CIFAR-100 partitions, and FedAvg. |
| Rust native evaluator | Reference CNN evaluation using Burn outside the zkVM. |
| SP1 prover | Executes CNN evaluation inside SP1, counts correct predictions, generates a Groth16 proof, and invokes local verification. |
| PolkaVM interface | Experimental proof submission to a configured verifier on a local Revive node. |

The current proof workload counts correct predictions. Flower does not yet transport evaluation proofs or verify them before accepting model updates. Proof aggregation across clients or rounds is not implemented.

`CURRENT_STATE.md` records an earlier Flower-only milestone; the Rust prover is subsequent work.

## Prover

The Rust workspace in [`zk-fedeval-prover/`](zk-fedeval-prover/README.md) contains:

- `common`: shared evaluation input and report types.
- `native`: reference evaluation using Burn's `NdArray<f32>` backend.
- `program`: SP1 guest that evaluates the CNN and commits a hash of `EvalReport { num_samples, correct }`.
- `host`: loads an evaluation batch, sets up the proving key, generates a Groth16 proof, writes `prover-report.json`, and invokes verification.

The experimental [`pvm-sp1-verifier/`](pvm-sp1-verifier/README.md) interface consumes the proof for local PolkaVM verification.

## Open question: proving N rounds of evaluation

Generating a proof for one evaluation does not establish that proving evaluations across **N federated rounds** is practical. Repeated evaluations introduce proving time and resource costs, while multiple clients can produce multiple proofs per round.

The main open question is: **how can we make N rounds of evaluation fast to prove, and what role should proof aggregation play?**

Questions still to investigate include:

- How does evaluation proving time grow with the number of rounds, clients, and evaluated samples?
- What are the costs and tradeoffs of aggregating evaluation proofs across clients and rounds?
- What strategy can meet the project's proving-time, verification-cost, and privacy requirements?

No strategy has been selected, and no N-round proving speedup is claimed. The proving and aggregation approach remains an open research and engineering question.

FedAvg combines model updates; proof aggregation concerns the cryptographic evidence for evaluations.

## Running the Flower baseline

Python requires 3.11 and uses `uv`:

```bash
uv sync
uv run flwr run . local --stream \
  --federation-config="num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0"
```

Federated client evaluation is disabled by default. The simulated clients are not separate authenticated machines.

Baseline Python checks:

```bash
uv run pytest -q
```
