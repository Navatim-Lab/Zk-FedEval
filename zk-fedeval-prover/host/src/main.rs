mod config;
mod dataset;
mod prover;
mod twin;

use sp1_sdk::blocking::prelude::*;
use sp1_sdk::include_elf;

use config::{Guest, HostConfig, ProofMode};
use dataset::EvalDataSource;
use prover::EvalProver;
use twin::NativeTwin;

const ELF_F32: Elf = include_elf!("zk-fedeval-program");
const ELF_INT8: Elf = include_elf!("zk-fedeval-program-int8");

fn main() -> anyhow::Result<()> {
    // Surfaces SP1's internal execution/proving trace (silent by default).
    // Run with `RUST_LOG=info` (or `debug`) to see it.
    sp1_sdk::utils::setup_logger();

    let config = HostConfig::from_env()?;
    let data = EvalDataSource::load(&config.eval_data_path)?;
    println!(
        "{:?} guest, {:?} mode: {} batch(es) of up to {} from {} samples in {}",
        config.guest,
        config.proof_mode,
        config.num_batches,
        config.batch_size,
        data.len(),
        config.eval_data_path.display(),
    );

    let (elf, twin) = match config.guest {
        Guest::F32 => (ELF_F32, None),
        Guest::Int8 => (ELF_INT8, Some(NativeTwin::load(&config.eval_data_path)?)),
    };
    let prover = EvalProver::setup(elf, config.proof_mode)?;

    let (mut samples, mut seconds, mut cycles) = (0usize, 0f64, 0u64);
    for index in 0..config.num_batches {
        let start = index * config.batch_size;
        let range = data.range(start, config.batch_size);
        if range.is_empty() {
            println!("[batch {index}] no samples left at {start}; stopping");
            break;
        }

        let mut stdin = SP1Stdin::new();
        let expected = match &twin {
            None => {
                stdin.write(&data.take_batch(start, config.batch_size));
                None
            }
            Some(twin) => {
                let batch = data.take_raw_batch(start, config.batch_size);
                let report = twin.expected_report(&batch, start)?;
                stdin.write_vec(batch.images);
                stdin.write_vec(batch.labels);
                Some(report)
            }
        };

        let outcome = prover.run(stdin)?;
        samples += range.len();
        seconds += outcome.seconds;
        cycles += outcome.cycles.unwrap_or(0);

        print!(
            "[batch {index}] samples {}..{}: {:.1}s, report hash 0x{}",
            range.start,
            range.end,
            outcome.seconds,
            hex::encode(outcome.report_hash),
        );
        if let Some(c) = outcome.cycles {
            print!(", {c} cycles");
        }
        println!();

        if let Some(report) = expected {
            anyhow::ensure!(
                twin::report_hash(&report) == outcome.report_hash,
                "guest committed a different report than the native twin computed",
            );
            println!(
                "[batch {index}] matches native twin: {}/{} correct (accuracy={:.4})",
                report.correct,
                report.num_samples,
                report.accuracy(),
            );
        }
    }

    let verb = if config.proof_mode == ProofMode::Execute { "executed" } else { "proved" };
    println!(
        "total: {samples} samples {verb} in {seconds:.1}s ({:.2} samples/min, excluding client start-up and setup)",
        samples as f64 * 60.0 / seconds.max(f64::EPSILON),
    );
    if config.proof_mode == ProofMode::Execute && samples > 0 {
        println!("cycles: {cycles} total, {} per sample", cycles / samples as u64);
    }

    Ok(())
}
