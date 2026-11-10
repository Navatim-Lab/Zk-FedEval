> [!WARNING]
> **Experimental software:** ZK-FedEval is an ongoing research prototype. It is not production-ready and has not undergone a complete security audit. Use at your own risk.

**Technical Design:** [ZK-FedEval: Technical Design (PDF)](https://resume.navatim.com/Zk-FedEval-Technical-Design.pdf)

# ZK-FedEval

ZK-FedEval is a proof of concept for privacy-preserving federated learning and evaluation using Flower, SP1, and PolkaVM. Its mission is to enable federated learning and model evaluation in data-sensitive sectors and businesses, where organizations need to collaborate on machine learning without exposing their underlying data. The project currently demonstrates a two-client Flower setup using CIFAR-100 and a CNN model. The CNN is an example and can be replaced with another model by adapting the training and evaluation implementation.

## Running the experiments

**1. Install dependencies** (from the repository root):

```bash
uv sync
```

**2. Run vanilla federated evaluation:**

```bash
uv run flwr run . local
```

**3. Generate a zero-knowledge proof:**

The current SP1 implementation proves the evaluation of a selected number of samples from `partition-0` (client 0), rather than aggregating proofs from both clients.

```bash
cd zk-fedeval-prover
ZK_FEDEVAL_BATCH_SIZE=16 cargo run --release -p zk-fedeval-host
```

Change `16` to select a different number of evaluation samples. The generated `prover-report.json` contains the Groth16 proof. Different sample counts affect proof generation time and proof size.

**4. Verify the proof on PolkaVM:**

A local Polkadot Revive development node is required. If you do not have one installed, follow the installation instructions on the [Polkadot SDK documentation](https://docs.polkadot.com/).

Start the development node:

```bash
./target/release/revive-dev-node --dev
```

In another terminal:

```bash
cd pvm-sp1-verifier
bun install
bun run main.ts
```

**5. Run the benchmarks:**

```bash
cd pvm-sp1-verifier
bun run benches.ts
```

The benchmarks for FHE is on work ....


The benchmark report is generated at `pvm-sp1-verifier/benchmark-report.md`. It contains machine specifications, ZK proof size, verification weights (`ref_time` and `proof_size`), and verification contract size. The `pvm-sp1-verifier/README.md` contains the full verification and benchmark results.

Repeat the proving, verification, and benchmarking workflow with different sample counts to compare proof generation time, proof size, and on-chain verification costs.
