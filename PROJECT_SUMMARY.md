# Zk-FedEval — Detailed Project Summary

**Last updated:** 2026-10-06 (generated from repo state)
**Repo:** `Zk-FedEval` / `ZKP-FedEval`

## 1. Mission / Elevator Pitch

Privacy-preserving, verifiable evaluation for Federated Learning on Kusama:

> Can a Flower client prove that a constrained evaluation produced a loss/accuracy below/above a public threshold, **without revealing private evaluation samples**?

Full intended pipeline:
`Flower federated training (PyTorch/CIFAR-100)` → `client local evaluation + export ONNX + eval_data.npz` → `SP1 zk-proof of CNN evaluation (Rust/Burn)` → `Flower proof transport + server acceptance policy + replay protection + model-version binding` → `on-chain verification on PolkaVM / Polkadot Hub TestNet`.

Explicit design rule from `CURRENT_STATE.md`: the critical Flower-to-proof-to-verification path must eventually be genuine; mocks are temporary scaffolding only.

## 2. Current Status

**Current milestone (per `CURRENT_STATE.md` 2026-08-31): Flower-only federated learning baseline — operational.**

Working:
- Flower simulation with 2 SuperNodes, 1 round, CPU-only (`num-supernodes=2, num-cpus=1, num-gpus=0`)
- Compact PyTorch CNN (100-output, ~69,656 params) on CIFAR-100
- Deterministic non-IID Dirichlet partitions (`alpha=0.5`, `seed=2026`, `fine_label`, revision-pinned `uoft-cs/cifar100@aadb3af`)
- Local training (1 epoch, SGD lr=0.1, batch=4), complete candidate `ArrayRecord` return (not deltas)
- FedAvg aggregation weighted by `num-examples` (~0.267 MB global `ArrayRecord`, ~60.95s runtime)
- Centralized server evaluation for reporting only (before: acc 0.01 / loss 4.6068, after round 1: acc 0.01 / loss 4.7348, aggregated train loss 4.40085)
- Focused pytest suite: `test_data.py`, `test_training.py`, `test_task.py`
- SP1 prover skeleton (`common`/`native`/`program`/`host`) + PolkaVM verifier skeleton (`pvm-sp1-verifier/main.ts`) exist in tree but are **not yet integrated** into Flower gating per CURRENT_STATE.

Not yet implemented (explicit):
- Evidence envelopes in Flower responses, canonical candidate-array hashing, round nonces
- Mock verifier → genuine EZKL/Halo2-KZG → SP1 Groth16 path, witness generation, genuine proof gen/verify in Flower loop
- Server-side acceptance filtering before FedAvg, reject malformed/mismatched/duplicated/replayed/failed, preserve global on all-reject, mixed accept/reject tests
- Replay protection, model-version binding, PolkaVM compile + TestNet deploy

Determinism boundary: locked env + same HW + same config/seeds/partitions/round + CPU-only. No cross-platform byte-identical claim. Run IDs, timestamps, PIDs, ports, timing must be normalized.

Security boundary today: transport only, no crypto. Successful Flower response ≠ proof of private data use, ≠ proof candidate model was evaluated, node IDs unauthenticated, server accepts without verification, centralized metrics don't gate.

## 3. Architecture

```
CIFAR-100 (HF uoft-cs/cifar100)
  → src/zk_fedeval/task.py (Net, DirichletPartitioner, load_data, train, test + ONNX/npz export)
    → client_app.py: ClientApp.train() / .evaluate() (partition-id based, deterministic, exports outputs/eval/partition-{id}/)
    → server_app.py: ServerApp + FedAvg (fraction_evaluate=0, global_evaluate central test set)
      → [FUTURE] evidence envelope + hash + nonce + verifier gate before FedAvg

outputs/eval/partition-{id}/model.onnx + eval_data.npz
  → zk-fedeval-prover/ (Rust workspace: common, native, program, host)
    common: EvalBatch {images: Vec<f32>, labels: Vec<i64>} + EvalReport {num_samples, correct} (SCALE codec)
    native: Burn NdArray reference evaluator (no zk)
    program: SP1 guest — Burn CNN (burn-import generated from ONNX, weights embedded as model.bpk via include_bytes!), Evaluator::count_correct, commit keccak256(report)
    host: loads npz, take_batch(16 default via ZK_FEDEVAL_EVAL_DATA / ZK_FEDEVAL_BATCH_SIZE), ProverClient::from_env setup → prove Groth16 → write prover-report.json → verify locally

prover-report.json (vk + SP1ProofWithPublicValues Groth16)
  → pvm-sp1-verifier/main.ts (Bun, polkadot-api, viem)
    reviveDev pallet on ws://127.0.0.1:9944, encode verifyProof(vk, public_values, proofBytesFull=selector+encoded_proof) → dry-run ReviveApi.call → signed Revive.call (Alice, 130% weight, 10 DOT deposit limit)
    contractsBytes/: verifierEZKL (FedEval.pvm) + SP1VerifierGroth16 + Groth16Verifier
```

Inconsistency note: `zk-fedeval-prover/README.md` mentions CIFAR-10 / EZKL while code + CURRENT_STATE target CIFAR-100 / SP1 — README is stale.

## 4. Component Details

### 4.1 Python / Flower (`src/zk_fedeval/`, `pyproject.toml` requires-python >=3.11,<3.12)
- `task.py:31 Net`: conv1 3→6/5, pool, conv2 6→16/5, pool, fc 16*5*5→120→84→100. `configure_determinism()` seeds random/numpy/torch + deterministic algos + 1 thread. `load_data()` 80/20 train_test_split per partition, DataLoader shuffle with `Generator(seed+partition_id)`. `train()` SGD momentum 0.9. `test()` optional export. `export_model_onnx()` opset 17, `export_eval_data()` npz.
- `client_app.py:16 train()`: loads global state_dict, trains, returns `RecordDict{arrays:ArrayRecord(state_dict), metrics:{train_loss, num-examples}}`. `evaluate()`: evaluates val split, exports, returns metrics only.
- `server_app.py:15 main()`: reads run_config, FedAvg(fraction_train=1.0, fraction_evaluate=0.0, min_train=2, min_eval=0, min_avail=2), 1 round, `global_evaluate()` central test loader batch 128.
- `pyproject.toml [tool.flwr.app]`: serverapp/clientapp entrypoints, config num-server-rounds=1, local-epochs=1, batch-size=4, lr=0.1, seed=2026, partition-alpha=0.5.

### 4.2 Tests (`tests/`)
- `test_data.py`: builds real FederatedDataset twice, asserts identical snapshots, pinned fingerprints/sizes (25321, 24679), membership/label digests, TVD 0.6012 ≥0.10, image (3,32,32) float32.
- `test_training.py`: same seeded Net → base digest `1c43da...`, train each partition twice → identical candidate digests (`afd59d...`, `f9055b...`), distinct across partitions, complete ArrayRecord (not delta), 10 arrays, 69,656 params, examples (20256,19743)/(5065,4936).
- `test_task.py`: contract tests (logits (2,100)), partitioner/dataset config capture, init repeatability.

### 4.3 Rust prover (`zk-fedeval-prover/`: workspace `common,native,program,host`)
- `common/src/lib.rs`: `IMAGE_*=3,32,32`, `EvalBatch`, `EvalReport::accuracy()`.
- `native/`: `ProverConfig::from_env`, `EvalDataset::load`, `CnnModel::load`, `Evaluator::run` — parity baseline for guest.
- `program/src/main.rs`: `sp1_zkvm::io::read()` batch → NdArray tensors → `CnnModel::load` → `count_correct` → `keccak_256(SCALE(report))` → `io::commit`. `model.rs` burn-import + `include_bytes!` bpk (no FS in zkVM). `evaluator.rs` argmax(1) + squeeze_dims [1] (batch-1 bugfix) → equal→sum.
- `host/`: `HostConfig` default `../../outputs/eval/partition-0/eval_data.npz`, batch 16; `EvalProver::setup(ELF)` prints VK bytes32; `prove()` stdin.write(batch) → `.groth16().run()` → `prover-report.json {verifying_key, proof}` → local `verify()` → return `public_values.read()` (currently `EvalReport`, guest commits hash — mismatch to reconcile).

### 4.4 PolkaVM verifier (`pvm-sp1-verifier/`, Bun + polkadot-api 2.1.0 + viem)
- `main.ts`: hardcoded `MAIN_VERIFIER 0x1249...cda5`, reads `../zk-fedeval-prover/prover-report.json`, builds `proofBytesFull = selector(4B vkey_hash)+encoded_proof`, `encodeFunctionData(verifyProof)` with EZKL ABI, Alice//Alice sr25519, dry-run then `Reveive.call` with 130% weight buffer.

## 5. How to Run (known-good)

```bash
uv run flwr run . local --stream --federation-config="num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0"
echo $?
uv run pytest -q tests/test_data.py
uv run pytest -q tests/test_training.py
uv run pytest -q
cargo run -p zk-fedeval-native  # reference eval
ZK_FEDEVAL_EVAL_DATA=outputs/eval/partition-0/eval_data.npz ZK_FEDEVAL_BATCH_SIZE=16 cargo run -p zk-fedeval-host
bun install && bun run main.ts  # needs local reviveDev node on :9944 + prover-report.json
```

## 6. Next Milestones (from CURRENT_STATE § Next Flower Milestone)

1. Flower response schema (arrays + evidence)
2. Canonical candidate hashing
3. Deterministic round nonces
4. Labelled mock verifier
5. Server acceptance before FedAvg
6. Reject malformed/mismatched/dup/replay/failed
7. Preserve global on zero-accept
8. Mixed accept/reject tests
9. Re-run simulation + normalized evidence in `docs/EVIDENCE.md`
10. Commit Flower-only milestone → then replace mock with genuine SP1/EZKL verifier without protocol redesign.

Plus: fix prover public-values shape (hash vs struct), pin CIFAR-100 vs CIFAR-10 docs, clean warnings (license metadata, deprecated `options.*`, HF auth, Ray vars), `.gitignore` secrets/proofs/keys/circuits/checkpoints.

## 7. Key Files

`CURRENT_STATE.md`, `README.md`, `PROJECT_STRUCTURE.md`, `pyproject.toml`, `src/zk_fedeval/{task,client_app,server_app}.py`, `tests/test_{data,training,task}.py`, `zk-fedeval-prover/{common,host,program,native}/src/`, `pvm-sp1-verifier/{main.ts,contractsBytes/}`.
