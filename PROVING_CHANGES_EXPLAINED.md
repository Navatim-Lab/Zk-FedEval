# How the int8 Changes Cut Proving Time

Oct 9, 2026 · branch `mac-int8-proving` (commit `381a585`)

Before these changes, the SP1 guest could not prove even one sample on a 36 GB M3 Max: the proof ran out of memory and was killed at a 56.5 GB peak. After them, a 4-sample proof finishes in about 76 s on AC power (3.15 samples a minute), because each sample now costs **1.60 M RISC-V cycles instead of 81.5 M (51× fewer)**.

All numbers below were measured on 2026-10-09 with SP1 6.6 on a 14-core M3 Max (36 GB). The companion docs have the wider context: [overview](ZK_FEDEVAL_OVERVIEW.md), [Mac plan](PROVING_ON_M3_MAX.md), [GPU plan](PROVING_SPEEDUP_PLAN.md), [improvement report](INT8_IMPROVEMENT_REPORT.md).

## Why proving was slow: cycles are the cost

SP1 proves a RISC-V program, and its proving time and memory grow with the number of instructions (cycles) the program executes. So the whole game is executing fewer cycles inside the guest.

The original guest ran the CNN in `f32` through Burn's `ndarray` backend. SP1's RISC-V has no floating-point unit, so every float operation is emulated in software, with bit-twiddling to unpack, align, add, normalize, and round. The CNN needs about 0.66 M multiply-adds per image, and the guest spent **~123 cycles on each one**.

| `f32` guest | Measured |
| --- | --- |
| Cycles, 1 sample | 113.9 M |
| Cycles, 16 samples | 1,336 M |
| Cycles per extra sample | 81.5 M |
| Fixed cycles per proof (loading Burn tensors and weights) | 32 M |
| 1-sample core proof on the M3 Max | Killed after 125 s at a 56.5 GB memory peak; no proof |

Memory matters as much as time on a laptop. Proof memory grows with cycles per proof: 26.7 GB at 4.2 M cycles, 42.3 GB at 14.1 M. A single `f32` sample is 114 M cycles, far past what 36 GB of RAM plus swap can hold.

## The changes, in the order they landed

| # | Change | Where | Cycles per sample | Fixed per proof |
| --- | --- | --- | --- | --- |
| 0 | Baseline: `f32` Burn guest | `program/` | 81.5 M | 32 M |
| 1 | int8 quantization + plain integer loops (prototype) | scratch benchmark | 4.30 M | ~0.01 M |
| 2 | Two output channels per 64-bit multiply (prototype) | scratch benchmark | 2.92 M | ~0.01 M |
| 3 | First real implementation, with true rescaling and biases | `common/src/int8.rs`, `program-int8/` | 3.32 M | 0.84 M |
| 4 | Unrolled inner loops; conv windows read in place, no copy | `common/src/int8.rs` | 2.37 M | 0.84 M |
| 5 | Register-blocked, const-generic conv kernel | `common/src/int8.rs` | **1.60 M** | 0.84 M |
| 6 | Faster weight-packing loop at model load | `common/src/int8.rs` | 1.60 M | **0.51 M** |

Each step was checked bit for bit against the previous one (see [Correctness](#correctness)), so none of the speedup comes from computing something different.

### 1. Quantize to int8 and drop floats

**File:** [`src/zk_fedeval/quantize.py`](src/zk_fedeval/quantize.py)

Integer math is native on RISC-V, so the first job is a model that never needs a float. The quantizer turns the exported ONNX model into `model_int8.bin`:

- **Weights:** int8, scaled per output channel, so each channel uses its full −127…127 range.
- **Activations:** u8 after each ReLU, scaled per layer from the activation range on the test split.
- **Rescaling between layers:** a fixed-point multiplier and shift, `(acc × M + 2^30) >> 31`, then clamp to 0…255. The clamp also does the ReLU.
- **Input normalization folded into `conv1`:** the client normalizes pixels as `x = p / 127.5 − 1`. The quantizer folds that into `conv1`'s weights and bias, so the guest reads raw 0–255 pixels and does no per-pixel arithmetic.

It also writes `int8_predictions.npy`, an exact integer reference used to check the Rust kernel.

**Effect:** moves each multiply-add from ~123 cycles of emulated float to a few integer instructions. The plain-loop prototype measured **4.30 M cycles per sample, 19× fewer**.

### 2. Two multiply-adds per instruction

**File:** [`zk-fedeval-prover/common/src/int8.rs`](zk-fedeval-prover/common/src/int8.rs) (`Layer::parse`, `Layer::unpack`)

SP1's RISC-V is 64-bit, but an int8 × u8 product needs only 16 bits. So two output channels share one 64-bit multiply:

1. Shift each weight to unsigned: `w + 128` (range 1…255).
2. Pack two channels' weights into one word: `lo | hi << 32`.
3. Multiply that word by the activation and accumulate. Each 32-bit lane collects one channel's sum.
4. Correct each lane by subtracting `128 × sum(inputs)`, then add the bias.

The largest possible lane sum is 400 inputs × 255 × 255 ≈ 26 M, well under 2^32, so one lane can never carry into the other. A unit test checks the packed result against a plain dot product, including the all-255 worst case.

**Effect:** prototype measured **2.92 M cycles per sample**. The first real implementation came in at **3.32 M**: it does the true per-channel rescaling and bias handling that the prototype skipped.

### 3. Raw bytes in, same hash out

**File:** [`zk-fedeval-prover/program-int8/src/main.rs`](zk-fedeval-prover/program-int8/src/main.rs)

The new guest reads pixels and labels with `sp1_zkvm::io::read_vec()` as raw `u8`: 3 KB per sample instead of a 12 KB bincode `Vec<f32>`, and no deserialization. The quantized weights are embedded with `include_bytes!`, so the program's verifying key identifies the model exactly, as before.

It commits the same public value as the `f32` guest, `keccak256(SCALE(EvalReport))`, so the existing on-chain `FedEval` verifier accepts proofs from either guest unchanged.

### 4. Unrolled loops and in-place windows

**File:** [`zk-fedeval-prover/common/src/int8.rs`](zk-fedeval-prover/common/src/int8.rs)

The first version copied every 5×5 conv window into a buffer before multiplying, and iterated element by element with bounds checks. The fix:

- Dense layers walk inputs in fixed chunks of 4 (`chunks_exact(4)`), so the compiler drops per-element bounds checks.
- Conv layers read each 5-pixel window row straight from the input as a `[u8; 5]` and multiply it out unrolled. No window copy.

**Effect:** 3.32 M → **2.37 M cycles per sample** (1.4×).

### 5. Register-blocked conv kernel

**File:** [`zk-fedeval-prover/common/src/int8.rs`](zk-fedeval-prover/common/src/int8.rs) (`conv_relu_pool::<PAIRS>`, `Layer::with_rows`)

A per-layer cycle split showed the convolutions dominate:

| Layer | Cycles per sample, before (2.37 M total) | After (1.60 M total) |
| --- | --- | --- |
| `conv1` (6 output channels) | 1.41 M | 0.93 M |
| `conv2` (16 output channels) | 0.79 M | 0.49 M |
| Dense layers | 0.18 M | 0.18 M |

Before, each channel pair re-read the same 75 (or 150) window pixels. Now the loop is inverted: each 5-pixel row is loaded once and multiplied against every channel pair's weights. The pair count is a compile-time constant (`::<3>` for `conv1`, `::<8>` for `conv2`), so the accumulators stay in registers. The window sum used for the lane correction comes from the same loaded pixels, for free. The weights are reordered once at load time (`by_row`) so each row's weights for all pairs sit contiguously.

**Effect:** 2.37 M → **1.60 M cycles per sample** (1.5×).

### 6. Cheaper model loading

**File:** [`zk-fedeval-prover/common/src/int8.rs`](zk-fedeval-prover/common/src/int8.rs) (`Layer::parse`)

Every proof parses the embedded model and packs ~35 k weight pairs. The original packing loop indexed through a closure with bounds checks. It now walks each pair of weight rows with `chunks_exact` and `zip`.

**Effect:** fixed cost per proof 0.84 M → **0.51 M cycles**. That matters because proofs here hold at most 4 samples, so fixed cost was ~13% of each proof.

### 7. Warm prover, selectable guest and proof mode

**Files:** [`host/src/config.rs`](zk-fedeval-prover/host/src/config.rs), [`host/src/prover.rs`](zk-fedeval-prover/host/src/prover.rs), [`host/src/main.rs`](zk-fedeval-prover/host/src/main.rs)

The host previously built the SP1 client and ran setup on every process start, then proved one batch. Now:

| Variable | Values | Purpose |
| --- | --- | --- |
| `ZK_FEDEVAL_GUEST` | `f32` (default), `int8` | Which guest to prove |
| `ZK_FEDEVAL_PROOF` | `groth16` (default), `compressed`, `core`, `execute` | How far to take each batch; `execute` gives cycle counts only |
| `ZK_FEDEVAL_NUM_BATCHES` | integer, default 1 | Prove consecutive batches with one client and one proving key |
| `ZK_FEDEVAL_BATCH_SIZE` | integer, default 16 | Samples per proof (keep at 4 or fewer on this Mac) |

**Effect:** the ~13 s client start-up and the setup are paid once per process instead of once per batch. Compressed proofs skip the Groth16 wrap while iterating. On this Mac, compressed and core proofs took the same time at 1 sample.

### 8. Correctness guards (no speed effect)

- [`host/src/twin.rs`](zk-fedeval-prover/host/src/twin.rs): a native twin runs the same `int8.rs` kernel outside the zkVM for every batch. It checks that the guest's committed hash equals `keccak256(SCALE(expected report))`, and that its predictions match `int8_predictions.npy` from Python.
- `host/src/prover.rs` now decodes the public value as the 32-byte hash the guest actually commits. The old host decoded it as an `EvalReport`, so its "x/y correct" line was read from hash bytes.
- `quantize.py` floors each layer's output scale so the fixed-point multiplier always fits in `i32`. This was needed because the default 1-round Flower model has a dead `fc2` layer, where every ReLU output is 0.

## Impact on proving time

Proof time on the CPU behaves like **~30 s of fixed prover overhead + cycles ÷ ~0.15–0.22 M cycles/s**. That's why halving cycles doesn't quite halve time once proofs get small.

| Guest version | Proof | Cycles per proof | Time | Peak memory | Power |
| --- | --- | --- | --- | --- | --- |
| `f32` (before) | Core, 1 sample | 114 M | Killed after 125 s | 56.5 GB | AC |
| int8, step 3 (3.32 M/sample) | Core, 1 sample | 4.2 M | 33.9 s | 26.7 GB | AC |
| int8, step 3 | Core, 4 samples | 14.1 M | 80.0 s | 42.3 GB | AC |
| int8, step 4 (2.37 M/sample) | Compressed, 4 samples × 3 warm batches | 10.3 M | 130.0 s, 133.9 s, 133.1 s | 46.5 GB | Battery |
| int8, steps 5–6 (1.60 M/sample) | Compressed, 4 samples × 2 warm batches | 6.9 M | 102.0 s, 98.6 s | 42.1 GB | Battery |
| int8, steps 5–6 | Compressed, 4 samples × 2 warm batches | 6.9 M | **78.1 s, 74.3 s** | 42.8 GB | **AC** |

| Throughput on this Mac | Samples proven per minute |
| --- | --- |
| `f32` guest | 0 (no proof completes) |
| int8, step 4, battery | 1.81 |
| int8, steps 5–6, battery | 2.39 |
| int8, steps 5–6, AC | **3.15** |

Three things drove the improvement, in order of size:

1. **Making proofs possible at all.** Cutting cycles per sample ~50× took a proof's peak memory from beyond 56 GB (one `f32` sample) to ~42 GB (four int8 samples), so proofs finish.
2. **Fewer cycles per sample.** Steps 4–6 cut a 4-sample proof from 10.3 M to 6.9 M cycles, and time fell ~25% on the same power source (130 s → 100 s on battery).
3. **Power source.** The same proof takes ~76 s on AC vs ~100 s on battery.

## What was tried and didn't help

| Attempt | Result |
| --- | --- |
| SP1 shard settings (`SHARD_SIZE`, `ELEMENT_THRESHOLD`, `HEIGHT_THRESHOLD`, `TRACE_CHUNK_SLOTS`) | Peak memory unchanged (45.6–46.8 GB vs 45.8 GB); time within ±4% |
| `RUSTFLAGS="-C target-cpu=native"` for the host | 153.9 s vs 152.4 s for two 4-sample proofs: no gain |
| Freivalds hint checking (analysis, not built) | ~1.25× at ≤ 4 samples per proof: `conv1` has only 6 channels, and the per-proof `r·W` setup doesn't amortize. Deprioritized for the Mac; still worth it for large GPU batches |
| RISC Zero 3.0.3 with Metal, same int8 kernel | 5.7 M cycles per sample on its 32-bit VM; a 4-sample proof took 2,134 s, ~28× slower than SP1 |
| 16-sample proofs | Killed at a 56.6 GB peak; not pursued further, on expert advice |

## Correctness

The speedups change how the arithmetic is scheduled, not what is computed:

- Kernel unit tests pass: the packed multiply equals a plain dot product, including worst-case inputs.
- The Rust kernel and the Python integer reference produced identical predictions on every sample checked: 256 stand-in samples, plus every batch on the real export.
- The guest's committed hash equalled the native twin's for every executed batch and every proof.
- On the 3-round trained model (5,065 test samples), int8 accuracy is 18.97% against 18.93% for float, with 97.6% top-1 agreement. The provisional gate is "accuracy within 0.5 points of float", which passes; the stricter 99% agreement target does not.

## Reproduce

```bash
# 1. Export a trained model (gentler settings than the default, which leaves fc2 dead)
uv run flwr run . local --stream \
  --federation-config="num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0" \
  --run-config "num-server-rounds=3 local-epochs=2 batch-size=32 learning-rate=0.01 fraction-evaluate=1.0 min-evaluate-nodes=2"
```

```bash
# 2. Quantize it (writes model_int8.bin and int8_predictions.npy)
uv run python -m zk_fedeval.quantize outputs/eval/partition-0
```

```bash
# 3. Cycle counts only
cd zk-fedeval-prover && ZK_FEDEVAL_GUEST=int8 ZK_FEDEVAL_PROOF=execute ZK_FEDEVAL_BATCH_SIZE=64 ZK_FEDEVAL_NUM_BATCHES=4 cargo run --release -p zk-fedeval-host
```

```bash
# 4. Prove: compressed 4-sample proofs, several per warm process
cd zk-fedeval-prover && ZK_FEDEVAL_GUEST=int8 ZK_FEDEVAL_PROOF=compressed ZK_FEDEVAL_BATCH_SIZE=4 ZK_FEDEVAL_NUM_BATCHES=3 cargo run --release -p zk-fedeval-host
```

## Still open

- A Groth16 proof of the int8 guest, which needs SP1's 6.2 GB circuit download, and an on-chain check against a local `pallet-revive` node.
- The 5,000 samples/min target remains ~1,600× away on one Mac with SP1. The GPU plan puts it at roughly 38 RTX 5090s with this guest, and DeepProve proved a same-architecture CNN inference in ~0.3–0.4 s here, but under an evaluation-only license.
