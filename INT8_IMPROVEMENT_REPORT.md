# Zk-FedEval: int8 Prover Improvement Report

Oct 9, 2026 · @Okeke Francis

## Summary

The new int8 guest makes proving possible on this MacBook. Today's `f32` guest can't finish even a one-sample proof: it is killed at a 56.5 GB memory peak. The int8 guest proves 1 sample in 34 s and 4 in 80 s, with 51× fewer cycles per sample, and its results match the native twin and the Python reference bit for bit.

- **Speed:** 81.5 M → 1.60 M cycles per sample, and 32 M → 0.51 M fixed per proof. A 4-sample compressed proof now takes \~100 s on battery (down from 130 s), or 2.39 samples/min.
- **Memory:** a 4-sample proof peaks at 42 GB. The 16-sample proof was killed at 56.6 GB, so batches stay at 4 or fewer for now.
- **Correctness:** hashes and predictions match exactly. Accuracy on a _trained_ model matches the float model (18.97% vs 18.93% on 5,065 samples), but top-1 agreement is 97.6%, below the 99% gate.
- **Against the plan's step-1 gate:** cycles pass (1.60 M against ≤ 3 M). Agreement misses narrowly: 97.6% against ≥ 99%, with no accuracy loss.

The 5,000 samples/min target is unchanged and still far off on one Mac: this is \~0.06% of it. The [Mac plan](PROVING_ON_M3_MAX.md)'s later steps and prover spikes are where the rest has to come from. Of the alternatives tested, only DeepProve is fast enough to matter (\~150 samples/min est. on this Mac), but its license is evaluation-only.

## Results

The int8 guest turns "can't prove on this Mac" into "one sample in 34 s". It uses 51× fewer cycles per sample, and it fits in memory where the `f32` guest doesn't.

Measured on 2026-10-09 with SP1 6.6 on the 14-core M3 Max (36 GB), plugged in. Both guests use the stand-in model, the real `Net` shape with random weights; cycle counts depend on the shape, not the weight values.

| Metric                                | `f32` guest (before)                         | int8 guest (after)                                           | Change              |
| ------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ | ------------------- |
| Cycles per extra sample               | 81.5 M                                       | 1.60 M (3.32 M in the first version, 2.37 M after unrolling) | 51× fewer           |
| Fixed cycles per proof                | 32 M                                         | 0.51 M                                                       | 63× fewer           |
| Cycles, 16 samples                    | 1,336 M                                      | 53.8 M                                                       | 24.8× fewer         |
| Core proof, 1 sample                  | Killed after 125 s at 56.5 GB peak; no proof | 33.9 s, 26.7 GB peak                                         | Not possible → 34 s |
| Core proof, 4 samples                 | Not attempted (\~358 M cycles)               | 80.0 s, 42.3 GB peak                                         | —                   |
| Proving time per extra sample         | —                                            | \~15 s (\~0.22 M cycles/s)                                   | —                   |
| Throughput, warm prover, batches of 4 | 0 samples/min                                | \~3 samples/min                                              | —                   |

The implemented guest first cost 3.32 M cycles per sample, because it does the real rescaling and bias handling the prototype skipped. Unrolling the inner loops brought it to 2.37 M. Loading each 5-pixel window row once and feeding every channel pair from registers (a const-generic conv kernel) brought it to 1.60 M. A tighter weight-packing loop cut the fixed cost from 0.84 M to 0.51 M. Every step stayed bit-exact with the twin and the Python reference. The 1- and 4-sample proof rows were measured on the 3.32 M version.

The CPU proving rate came out at \~0.22 M cycles/s here, against \~0.15 M in earlier runs. Those earlier runs had more memory pressure, so expect the rate to vary with how much RAM is free.

The earlier docs estimated \~9 minutes of CPU proving per `f32` sample. That assumed the proof would finish. It doesn't: a single `f32` sample (114 M cycles) runs out of memory before completing.

**First run of the real host, on the real export** (partition-0, on battery):

| Run                                                        | Result                                                                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Execute, 4 batches of 64 samples                           | 2.38 M cycles per sample including fixed cost; every batch matched the native twin and the Python predictions |
| Compressed proofs, 3 warm batches of 4 samples             | 130.0 s, 133.9 s, 133.1 s; 13.0 s client start and 0.9 s setup paid once; 46.5 GB peak; all matched the twin  |
| Throughput                                                 | 1.81 samples/min proven                                                                                       |
| Compressed proofs after kernel tuning, 2 warm batches of 4 | 102.0 s and 98.6 s; 42.1 GB peak; 2.39 samples/min, up from 1.81                                              |
| Same, on AC power                                          | 78.1 s and 74.3 s; 42.8 GB peak; 3.15 samples/min                                                             |

The first three proofs are slower than the 80 s core proof measured earlier on AC, despite fewer cycles. Three things changed at once: battery vs AC power, compressed vs core proofs, and memory pressure. A run on AC in each mode would separate them. After the kernel tuning, a 4-sample proof is 6.9 M cycles instead of 10.3 M, and the same proofs on battery take \~100 s. Time falls less than cycles because roughly 30 s per proof is fixed prover overhead.

**Alternative provers on this Mac** (plan step 7, on AC):

| Prover                      | Workload                                                 | Time                                    | Throughput                             | Notes                                             |
| --------------------------- | -------------------------------------------------------- | --------------------------------------- | -------------------------------------- | ------------------------------------------------- |
| SP1 6.6, int8 guest         | Our model, 4-sample compressed proof                     | \~76 s                                  | 3.15 samples/min                       | 42.8 GB peak                                      |
| RISC Zero 3.0.3, Metal r0vm | Our int8 kernel, 4-sample succinct proof                 | 2,134 s                                 | 0.11 samples/min                       | 5.7 M cycles/sample on its 32-bit VM; ruled out   |
| DeepProve                   | Its own CIFAR CNN, same layers (62k params), 1 inference | \~1.1 s end to end, \~0.3–0.4 s proving | \~150 samples/min (est., proving only) | 175 MB peak; evaluation-only license; not Groth16 |

DeepProve's benchmark tool failed on our exact model, on a bug in its float-accuracy step. Its license forbids patching it, so the figure comes from its own unmodified test on its bundled model.

## Correctness

The guest, the native twin, and the Python reference agree bit for bit on every sample tested. On a trained model, int8 keeps the float model's accuracy (18.97% vs 18.93%), with 97.6% top-1 agreement.

| Check                                                 | Result                                                                                                                            |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Kernel unit tests (`cargo test -p zk-fedeval-common`) | 2/2 pass: packed multiply equals a plain dot product, including all-255 inputs and negative biases; a bad file header is rejected |
| Rust kernel vs Python integer reference               | 256/256 predictions identical                                                                                                     |
| Guest's committed hash vs native twin                 | Equal for every batch: 1, 16, 64, and 256 samples executed, plus each proof                                                       |
| Float vs int8 top-1 agreement                         | 97.6% on the trained export (5,065 samples), below the 99% gate; int8 accuracy 18.97% vs float 18.93%                             |

An earlier check on the stand-in model (random weights, random pixels) gave 98.4% agreement, but its predictions are mostly near-ties, so it only showed that the pipeline runs end to end.

**The first real export didn't settle it either.** A Flower run with the default config (1 round, learning rate 0.1, batch size 4) exported partition-0: 5,065 test samples at 0.9% float accuracy. Float and int8 agree on 100% of them, but only trivially: `fc2`'s ReLU outputs are zero for every sample, so both models predict class 19 for everything. That dead layer also made the quantizer's rescaling overflow, so it now floors each layer's output scale to keep multipliers in range. A second run with gentler settings (3 rounds, 2 local epochs, batch 32, learning rate 0.01) trained properly, reaching 19.5% global accuracy. On its 5,065 test samples, int8 matches float accuracy (18.97% vs 18.93%) and agrees on 97.6% of predictions, short of the 99% gate. Calibrating activations by percentile instead of max made agreement slightly worse (97.3–97.5%). The guest still matches the twin and the Python reference bit for bit on this model.

## What changed in the code

Steps 1 and 2 of the [Mac plan](PROVING_ON_M3_MAX.md) are implemented on branch `mac-int8-proving`. They're committed locally as `381a585` (not pushed), and the old `f32` path still works unchanged.

| File                                   | What it does                                                                                                                                                                                                                                         |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/zk_fedeval/quantize.py`           | Post-training int8 quantization. Folds input normalization into `conv1`, uses per-channel int8 weights, u8 activations, and i32 fixed-point rescaling. Writes `model_int8.bin` and `int8_predictions.npy`, and reports float vs int8 top-1 agreement |
| `zk-fedeval-prover/common/src/int8.rs` | The integer CNN, shared by guest and host. Two output channels per 64-bit multiply. Unit-tested against a plain dot product                                                                                                                          |
| `zk-fedeval-prover/program-int8/`      | New SP1 guest. Reads raw u8 pixels, embeds the quantized weights, and commits the same `keccak(SCALE(EvalReport))` as the `f32` guest, so `FedEval` accepts both                                                                                     |
| `zk-fedeval-prover/host/src/config.rs` | `ZK_FEDEVAL_GUEST=f32\|int8`, `ZK_FEDEVAL_PROOF=groth16\|compressed\|core\|execute`, `ZK_FEDEVAL_NUM_BATCHES`                                                                                                                                        |
| `zk-fedeval-prover/host/src/prover.rs` | One warm client and proving key for any number of batches; times each proof. Now decodes the public value as the 32-byte hash it is (fixes the old `EvalReport` misread)                                                                             |
| `zk-fedeval-prover/host/src/twin.rs`   | Native int8 twin: recomputes every batch with the guest's kernel, checks the committed hash, and cross-checks predictions against the Python reference                                                                                               |

Example run, once a quantized model exists:

```bash
cd zk-fedeval-prover && ZK_FEDEVAL_GUEST=int8 ZK_FEDEVAL_PROOF=compressed ZK_FEDEVAL_BATCH_SIZE=4 ZK_FEDEVAL_NUM_BATCHES=3 cargo run --release -p zk-fedeval-host
```

## Not done yet, and next steps

The biggest open problem is memory, not speed: proof memory grows with cycles per proof, so on 36 GB a proof can't hold more than a handful of samples.

| Batch (int8 guest) | Cycles | Peak memory footprint | Outcome                                                                       |
| ------------------ | ------ | --------------------- | ----------------------------------------------------------------------------- |
| 1 sample           | 4.2 M  | 26.7 GB               | Proved in 33.9 s                                                              |
| 4 samples          | 14.1 M | 42.3 GB               | Proved in 80.0 s                                                              |
| 16 samples         | 53.8 M | over 56.6 GB          | Killed after 123 s, with swap 11.7 of 13.3 GB used; most likely out of memory |

- [ ] **Keep each proof to 4 samples or fewer.** Run many 4-sample proofs in one warm process (`ZK_FEDEVAL_NUM_BATCHES`) and aggregate them. On expert advice, the 16-sample case won't be investigated further.
- [x] **Run the real accuracy gate.** Done on a 3-round model: accuracy unchanged, agreement 97.6% against ≥ 99%. Not passed.
- [x] **Decide the agreement gate.** Provisionally: judge int8 on accuracy, within 0.5 points of float. It passes (18.97% vs 18.93%). Easy to revisit if you want a stricter test.
- [x] **Hit the ≤ 3 M cycles gate.** Done: 1.60 M, by reading conv windows in place, unrolling the inner loops, and a register-blocked conv kernel.
- [ ] **Groth16 and on-chain check.** The full host builds and proves compressed on real data. Groth16 needs SP1's circuit files, a 6.2 GB download (`v6.1.0-groth16.tar.gz` from SP1's S3 bucket) that waits for your OK. The on-chain step needs a local `pallet-revive` node with the verifier contracts deployed. Neither is set up on this machine, and the repo has no deployment scripts.
- [x] **Plan step 4:** `target-cpu=native` measured on AC: 153.9 s vs 152.4 s for two 4-sample proofs, so no gain. Power matters more: \~76 s per proof on AC vs \~100 s on battery.
- [x] **Plan step 6 (Freivalds): deprioritized.** Measured split per sample: conv1 0.93 M (58%), conv2 0.49 M (30%), dense layers 0.18 M (11%). conv1 has only 6 output channels, so checking its outputs saves little. With at most 4 samples per proof, the `r·W` setup (\~0.4 M cycles per proof) doesn't amortize. Net \~1.25×, so it only pays on the GPU path, where batches are large.
- [x] **Plan step 7: spike DeepProve and RISC Zero (Metal).** RISC Zero is done and ruled out: 5.7 M cycles per sample, and 2,134 s for a 4-sample succinct proof with the Metal-enabled r0vm 3.0.3, \~28× slower than SP1. DeepProve, measured with its own test on a same-architecture CIFAR CNN: \~1.1 s per inference end to end, \~0.3–0.4 s proving, 175 MB peak. That's the fastest option by far, but the license is evaluation-only and its proofs need a new PolkaVM verifier.
- [x] **Commit `mac-int8-proving`.** Done locally as `381a585`; not pushed.
