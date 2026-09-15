mod config;
mod dataset;
mod prover;

use sp1_sdk::blocking::Elf;
use sp1_sdk::include_elf;

use config::HostConfig;
use dataset::EvalDataSource;
use prover::EvalProver;

const ELF: Elf = include_elf!("zk-fedeval-program");

fn main() -> anyhow::Result<()> {
    // Surfaces SP1's internal execution/proving trace (silent by default).
    // Run with `RUST_LOG=info` (or `debug`) to see it.
    sp1_sdk::utils::setup_logger();

    let config = HostConfig::from_env();

    let data = EvalDataSource::load(&config.eval_data_path)?;
    let batch = data.take_batch(config.batch_size);
    println!(
        "proving evaluation over {} of {} available samples from {}",
        batch.num_samples(),
        data.len(),
        config.eval_data_path.display(),
    );

    let prover = EvalProver::setup(ELF)?;
    let report = prover.prove(&batch)?;

    println!(
        "proved evaluation: {}/{} correct (accuracy={:.4})",
        report.correct,
        report.num_samples,
        report.accuracy(),
    );

    Ok(())
}
