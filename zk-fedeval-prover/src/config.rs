use std::path::PathBuf;

/// Filesystem locations the prover reads at runtime.
///
/// The ONNX model itself is not part of this config: it is compiled in at
/// build time by `build.rs` via `burn-import`, so the prover binary is only
/// ever paired with the model it was built against.
pub struct ProverConfig {
    pub eval_data_path: PathBuf,
}

impl ProverConfig {
    const DEFAULT_EVAL_DATA_PATH: &'static str = "../outputs/eval/partition-0/eval_data.npz";
    const EVAL_DATA_ENV_VAR: &'static str = "ZK_FEDEVAL_EVAL_DATA";

    /// Read config from the environment, falling back to the partition-0 defaults.
    pub fn from_env() -> Self {
        let eval_data_path = std::env::var(Self::EVAL_DATA_ENV_VAR)
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(Self::DEFAULT_EVAL_DATA_PATH));

        Self { eval_data_path }
    }
}
