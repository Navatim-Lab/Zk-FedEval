use std::path::PathBuf;

/// Filesystem locations and batch sizing the host reads at runtime.
pub struct HostConfig {
    pub eval_data_path: PathBuf,
    pub batch_size: usize,
}

impl HostConfig {
    const DEFAULT_BATCH_SIZE: usize = 16;
    const EVAL_DATA_ENV_VAR: &'static str = "ZK_FEDEVAL_EVAL_DATA";
    const BATCH_SIZE_ENV_VAR: &'static str = "ZK_FEDEVAL_BATCH_SIZE";

    /// Read config from the environment, falling back to a 16-sample batch
    /// from the partition-0 defaults.
    ///
    /// The default path is anchored to `CARGO_MANIFEST_DIR` (baked in at
    /// compile time), not the runtime working directory: `cargo run -p
    /// zk-fedeval-host` executes with the CWD of wherever `cargo` was
    /// invoked from, which is not necessarily this crate's own directory.
    pub fn from_env() -> Self {
        let eval_data_path = std::env::var(Self::EVAL_DATA_ENV_VAR)
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../outputs/eval/partition-0/eval_data.npz")
            });

        let batch_size = std::env::var(Self::BATCH_SIZE_ENV_VAR)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(Self::DEFAULT_BATCH_SIZE);

        Self {
            eval_data_path,
            batch_size,
        }
    }
}
