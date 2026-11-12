use std::path::PathBuf;

use anyhow::{bail, Result};

/// Which guest program to run: the Burn `f32` CNN or the integer-only int8 CNN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guest {
    F32,
    Int8,
}

/// How far to take each batch: execute only (cycle counts), or prove to a given proof type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofMode {
    Execute,
    Core,
    Compressed,
    Groth16,
}

/// Filesystem locations, batch sizing, and guest/proof selection the host reads at runtime.
pub struct HostConfig {
    pub eval_data_path: PathBuf,
    pub batch_size: usize,
    pub num_batches: usize,
    pub guest: Guest,
    pub proof_mode: ProofMode,
}

impl HostConfig {
    const DEFAULT_BATCH_SIZE: usize = 16;
    const EVAL_DATA_ENV_VAR: &'static str = "ZK_FEDEVAL_EVAL_DATA";
    const BATCH_SIZE_ENV_VAR: &'static str = "ZK_FEDEVAL_BATCH_SIZE";
    const NUM_BATCHES_ENV_VAR: &'static str = "ZK_FEDEVAL_NUM_BATCHES";
    const GUEST_ENV_VAR: &'static str = "ZK_FEDEVAL_GUEST";
    const PROOF_ENV_VAR: &'static str = "ZK_FEDEVAL_PROOF";

    /// Read config from the environment, falling back to one 16-sample Groth16 proof
    /// of the `f32` guest over the partition-0 defaults.
    ///
    /// The default path is anchored to `CARGO_MANIFEST_DIR` (baked in at
    /// compile time), not the runtime working directory: `cargo run -p
    /// zk-fedeval-host` executes with the CWD of wherever `cargo` was
    /// invoked from, which is not necessarily this crate's own directory.
    pub fn from_env() -> Result<Self> {
        let eval_data_path = std::env::var(Self::EVAL_DATA_ENV_VAR)
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../outputs/eval/partition-0/eval_data.npz")
            });

        let batch_size = Self::parse_usize(Self::BATCH_SIZE_ENV_VAR, Self::DEFAULT_BATCH_SIZE)?;
        let num_batches = Self::parse_usize(Self::NUM_BATCHES_ENV_VAR, 1)?;

        let guest = match std::env::var(Self::GUEST_ENV_VAR).as_deref() {
            Err(_) | Ok("f32") => Guest::F32,
            Ok("int8") => Guest::Int8,
            Ok(other) => bail!("{}={other}: expected f32 or int8", Self::GUEST_ENV_VAR),
        };

        let proof_mode = match std::env::var(Self::PROOF_ENV_VAR).as_deref() {
            Err(_) | Ok("groth16") => ProofMode::Groth16,
            Ok("compressed") => ProofMode::Compressed,
            Ok("core") => ProofMode::Core,
            Ok("execute") => ProofMode::Execute,
            Ok(other) => bail!("{}={other}: expected groth16, compressed, core or execute", Self::PROOF_ENV_VAR),
        };

        Ok(Self {
            eval_data_path,
            batch_size,
            num_batches,
            guest,
            proof_mode,
        })
    }

    fn parse_usize(var: &str, default: usize) -> Result<usize> {
        match std::env::var(var) {
            Err(_) => Ok(default),
            Ok(value) => match value.parse() {
                Ok(n) if n > 0 => Ok(n),
                _ => bail!("{var}={value}: expected a positive integer"),
            },
        }
    }
}
