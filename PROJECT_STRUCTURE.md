# Zk-FedEval — Project Structure

```
Zk-FedEval/
├── README.md                          # Top-level overview: Flower + EZKL PoC
├── CURRENT_STATE.md                   # Current implementation status / notes
├── LICENSE
├── pyproject.toml                     # Python package (zk_fedeval), Flower app config
├── .python-version                    # Python >=3.11,<3.12
├── uv.lock
├── .gitignore
│
├── src/
│   └── zk_fedeval/                    # Flower federated evaluation (Python)
│       ├── __init__.py
│       ├── task.py                    # Model, dataset (CIFAR-100), train/test logic
│       ├── client_app.py              # Flower ClientApp
│       └── server_app.py              # Flower ServerApp
│
├── tests/                             # Python tests (pytest)
│   ├── test_data.py
│   ├── test_task.py
│   └── test_training.py
│
├── zk-fedeval-prover/                 # SP1 zk-prover (Rust workspace)
│   ├── Cargo.toml                     # Workspace: common, native, program, host
│   ├── Cargo.lock
│   ├── README.md
│   ├── .gitignore
│   │
│   ├── common/                        # Shared types / constants
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   │
│   ├── native/                        # Native (non-zk) evaluator / reference impl
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/
│   │       ├── main.rs
│   │       ├── config.rs
│   │       ├── dataset.rs
│   │       ├── evaluator.rs
│   │       └── model.rs
│   │
│   ├── program/                       # SP1 guest program (zk circuit)
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/
│   │       ├── main.rs                # Guest entrypoint
│   │       ├── evaluator.rs
│   │       └── model.rs
│   │
│   └── host/                          # SP1 host / prover driver
│       ├── Cargo.toml
│       ├── build.rs
│       └── src/
│           ├── main.rs
│           ├── config.rs
│           ├── dataset.rs
│           └── prover.rs
│
└── pvm-sp1-verifier/                  # PolkaVM on-chain verifier (TypeScript / Bun)
    ├── README.md
    ├── package.json                   # Bun + polkadot-api + viem
    ├── tsconfig.json
    ├── bun.lock
    ├── .gitignore
    ├── main.ts                        # Verifier entrypoint
    ├── contractsBytes/                # Compiled PVM contracts + ABIs
    │   ├── verifierEZKL_ABI.json
    │   ├── verifierEZKL.sol:FedEval.pvm
    │   ├── SP1VerifierGroth16_ABI.json
    │   ├── SP1VerifierGroth16.sol:SP1Verifier.pvm
    │   └── Groth16Verifier.sol:Verifier.pvm
    └── .papi/                         # polkadot-api descriptors (generated)
        ├── polkadot-api.json
        ├── descriptors/
        └── metadata/
```

Excluded from tree: `.git/`, `*/target/`, `*/node_modules/`, `*/.papi/` contents (generated), `uv.lock`, `bun.lock`, `Cargo.lock` contents.

## Components

1. `src/zk_fedeval/` — Flower simulation: 2 clients, 1 round, 100-output CNN, deterministic non-IID CIFAR-100 partitions.
2. `zk-fedeval-prover/` — SP1 proof system: `program` (guest), `host` (prove), `native` (reference), `common` (shared).
3. `pvm-sp1-verifier/` — Experimental PolkaVM verification on Kusama via `polkadot-api`.
4. `tests/` — Data / task / training tests.
