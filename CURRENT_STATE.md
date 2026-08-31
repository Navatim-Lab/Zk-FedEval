# Current Project State

**Last updated:** 2026-08-31  
**Current milestone:** Flower-only federated learning baseline

## Mission

Build a reduced-scale but production-shaped proof of concept for privacy-preserving federated learning using:

1. Flower for federated orchestration.
2. PyTorch and CIFAR-100 for the CNN workload.
3. EZKL/Halo2-KZG for private evaluation proofs.
4. PolkaVM and Polkadot Hub TestNet for eventual verifier deployment.

The PoC uses two clients and one federated round to keep execution manageable. The critical Flower-to-proof-to-verification path must eventually be genuine; mocks are temporary integration scaffolding only.

## Current Scope

The project is currently implementing and validating the Flower baseline.

Currently included:

- Flower simulation.
- Two simulated clients.
- One federated round.
- A PyTorch CNN compatible with CIFAR-100.
- Deterministic configuration.
- Non-IID CIFAR-100 client partitions.
- Local client training.
- Complete candidate model arrays returned by each client.
- FedAvg aggregation.
- Centralized server evaluation for reporting.

Not implemented yet:

- Evidence envelopes in Flower responses.
- Candidate-array hashing and model binding.
- Mock acceptance filtering.
- EZKL circuits.
- Witness generation.
- Genuine proof generation or verification.
- Replay protection.
- PolkaVM verifier compilation.
- Polkadot Hub TestNet deployment.

## Flower Foundation

The repository uses the official Flower PyTorch quickstart as its permanent foundation. The generated Flower application was adapted rather than replaced with unrelated boilerplate.

The application follows Flower’s application structure:

```text
src/zk_fedeval/
├── __init__.py
├── task.py
├── client_app.py
└── server_app.py
```

The current responsibilities are:

- `task.py`: CNN, dataset loading, training, evaluation, and parameter handling.
- `client_app.py`: Flower `ClientApp` training handler.
- `server_app.py`: Flower `ServerApp`, strategy configuration, and centralized reporting.
- `tests/test_data.py`: deterministic non-IID data-partition checks.
- `tests/test_training.py`: deterministic local-training and complete-model-array checks.

The exact repository state and file list should be verified with:

```bash
git status --short
git ls-files
```

## Dataset

The baseline uses the public `uoft-cs/cifar100` dataset through `flwr-datasets`.

Dataset properties:

- Image shape: `[3, 32, 32]`.
- Number of classes: 100.
- Partition column: `fine_label`.
- Number of client partitions: 2.
- Partitioning approach: non-IID Dirichlet partitioning.
- Current Dirichlet concentration: `alpha=0.5`.
- Partition assignment: Flower’s deterministic `partition-id`.
- Each client loads a different logical partition.

CIFAR-100 is downloaded automatically on first use and normally cached under:

```text
~/.cache/huggingface/
```

The dataset is not stored in Git.

Logical partition IDs are used only for deterministic dataset selection. Flower transport node IDs in local simulation are not authenticated real-world client identities.

## Model and Training

The baseline uses a compact CNN compatible with CIFAR-100 rather than a linear toy model.

Current training shape:

- Two clients.
- One federated round.
- One local epoch per client.
- CPU-only simulation.
- One CPU allocated to each simulated client.
- No GPU allocation.
- FedAvg weighted using `num-examples`.

Each client starts with the server-provided global CNN and returns the complete trained model state as candidate model arrays.

The returned arrays are not parameter deltas. They are the exact complete arrays supplied to FedAvg for aggregation.

## Evaluation

Flower’s later federated client-evaluation phase is disabled:

```python
fraction_evaluate = 0.0
min_evaluate_nodes = 0
```

The centralized server evaluation currently runs for reporting only. It does not decide whether a client update is accepted.

The future private candidate evaluation and proof-verification gate have not yet been implemented.

## Confirmed Flower Simulation

The baseline was executed using:

```bash
uv run flwr run . local --stream \
  --federation-config="num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0"
```

Observed results from the captured run:

- Flower started two simulated SuperNodes.
- The strategy configured one federated round.
- Both nodes were selected for training.
- Two training responses were received.
- Zero training failures were reported.
- FedAvg completed aggregation.
- Final global model arrays were produced.
- Federated client evaluation was skipped.
- Centralized evaluation ran before and after aggregation.
- Flower reported that strategy execution finished.
- Total strategy runtime was approximately 60.95 seconds.
- Final global `ArrayRecord` size was approximately 0.267 MB.

Observed reporting metrics:

| Stage | Accuracy | Loss |
|---|---:|---:|
| Initial centralized evaluation | 0.01 | 4.6068253 |
| After round 1 | 0.01 | 4.7348803 |
| Aggregated client training loss | — | 4.4008590 |

The low accuracy is not considered a Flower infrastructure failure. Only one federated round and minimal local training were used on non-IID CIFAR-100 partitions.

The captured output reached Flower’s final-results stage, but it did not print the shell exit code. Future evidence captures can record it immediately after execution:

```bash
echo $?
```

## What the Current Run Demonstrates

The current run demonstrates that:

- The Flower application can start successfully.
- Two logical clients can participate in the simulation.
- The server can distribute a CNN for training.
- Both clients can return successful training responses.
- Flower can aggregate the returned arrays with FedAvg.
- The updated global arrays can be evaluated centrally.
- The federated client-evaluation phase can remain disabled.
- The application can complete the configured one-round strategy.

## What the Current Run Does Not Demonstrate

The simulation log alone does not prove that:

- Both clients received byte-identical base model arrays.
- Each client used the intended non-IID partition.
- Local training is byte-identical across repeated executions.
- Candidate arrays are correctly bound to evaluation evidence.
- Private evaluation data was genuinely used.
- Any evaluation result was computed inside a constrained graph.
- Any submission contains a genuine zero-knowledge proof.
- The server filters updates before aggregation.
- Replay, duplication, tampering, or malformed evidence is rejected.
- The system is production-ready.

Those properties require focused tests and later proof-system integration.

## Determinism Boundary

Determinism is required only under:

- The locked dependency environment.
- The same hardware configuration.
- The same runtime configuration.
- The same seeds, partitions, scenario, and round.
- CPU-only execution where configured.

The project does not claim cross-platform byte-identical execution.

Flower-generated run IDs, timestamps, process IDs, ports, and timing data must be normalized or excluded when comparing repeated simulation outputs.

## Current Warnings

The successful simulation included several non-blocking warnings:

- Missing recommended project license metadata.
- Deprecated Flower `options.*` configuration.
- Unauthenticated Hugging Face Hub access.
- A Ray environment-variable future warning.
- Ray `ClientAppActor` cleanup messages after strategy completion.

These warnings did not prevent training or aggregation. They should be cleaned up where practical but are not blockers for the current baseline.

## Security Boundary

The current Flower baseline does not provide cryptographic evidence.

At this stage:

- Flower transport demonstrates message exchange only.
- A successful training response does not prove that private data was used.
- A successful training response does not prove that evaluation used the submitted candidate model.
- Local simulation node identifiers are not authenticated identities.
- The server currently accepts successful Flower responses without proof verification.
- Centralized reporting metrics do not gate aggregation.

Cryptographic proof-to-model binding belongs to the later EZKL integration.

## Generated Files and Sensitive Material

The following must not be committed:

- CIFAR-100 dataset files.
- Hugging Face caches.
- Python virtual environments.
- Python and test caches.
- Training checkpoints.
- Candidate or global model dumps.
- Private evaluation datasets.
- Environment-secret files.
- Credentials and signing keys.
- Witnesses and proofs.
- SRS files.
- Proving keys.
- Verification keys.
- Generated circuits.
- Large build outputs.

SRS, proving keys, and verification keys are generally public rather than secret, but they should remain ignored during development because they are generated and potentially large.

`uv.lock` should be committed because it records the locked Python environment.

## Tests

The focused baseline tests are:

```bash
uv run pytest -q tests/test_data.py
uv run pytest -q tests/test_training.py
uv run pytest -q
```

The data tests should verify:

- Exactly two partitions.
- Both partitions are nonempty.
- CIFAR-100 image and label shapes are valid.
- Partition loading is deterministic.
- Partition sizes and label histograms are repeatable.
- The two clients have measurably different class distributions.

The training tests should verify:

- Both runs start from the same seeded CNN.
- The same client operation is repeatable.
- Complete state dictionaries are returned.
- Returned values are candidate model arrays rather than deltas.
- Different client partitions normally produce different candidate models.

Exact commands and outputs must be recorded in `docs/EVIDENCE.md` before calling those results reproduced.

## Next Flower Milestone

Before integrating EZKL, the Flower baseline must add a stable evidence and filtering boundary:

1. Define the Flower response schema for candidate arrays and evidence.
2. Add canonical candidate-array hashing.
3. Add deterministic round nonces for tests.
4. Add a clearly labelled mock verifier.
5. Add server-side acceptance policy before FedAvg.
6. Reject malformed, mismatched, duplicated, replayed, and failed submissions.
7. Preserve the previous global model when no submissions are accepted.
8. Test mixed accepted and rejected client responses.
9. Repeat the full simulation and record normalized results.
10. Commit the completed Flower-only milestone.

The mock verifier will later be replaced by the genuine EZKL verifier without redesigning the Flower response protocol or server aggregation boundary.

## Current Conclusion

The Flower/CIFAR-100 end-to-end smoke test is operational.

The current result is evidence of Flower orchestration, client training, and FedAvg aggregation. It is not yet evidence of private evaluation, cryptographic verification, proof-to-model binding, or secure federated aggregation.