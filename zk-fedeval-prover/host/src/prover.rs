use std::fs::File;
use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use sp1_sdk::blocking::prelude::*;
use sp1_sdk::blocking::{EnvProver, EnvProvingKey, ProverClient};

use crate::config::ProofMode;

/// Builds the SP1 client once and, unless only executing, the proving key for the
/// guest ELF, then runs any number of batches against them: a warm prover, so the
/// client start-up and setup are paid once per process rather than once per batch.
pub struct EvalProver {
    client: EnvProver,
    elf: Elf,
    mode: ProofMode,
    proving_key: Option<EnvProvingKey>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct FullReport {
    verifying_key: String,
    proof: SP1ProofWithPublicValues,
}

/// What one batch produced.
pub struct BatchOutcome {
    /// The guest's only public value: keccak-256 of the SCALE-encoded `EvalReport`.
    pub report_hash: [u8; 32],
    /// RISC-V cycles, when the batch was only executed.
    pub cycles: Option<u64>,
    pub seconds: f64,
}

impl EvalProver {
    pub fn setup(elf: Elf, mode: ProofMode) -> Result<Self> {
        let started = Instant::now();
        let client = ProverClient::from_env();
        println!("[prover] client ready in {:.1}s", started.elapsed().as_secs_f64());

        let proving_key = if mode == ProofMode::Execute {
            None
        } else {
            let started = Instant::now();
            let proving_key = client.setup(elf.clone()).context("SP1 setup failed")?;
            println!("VK: {}", proving_key.verifying_key().bytes32());
            println!("[prover] setup complete in {:.1}s", started.elapsed().as_secs_f64());
            Some(proving_key)
        };

        Ok(Self {
            client,
            elf,
            mode,
            proving_key,
        })
    }

    pub fn run(&self, stdin: SP1Stdin) -> Result<BatchOutcome> {
        let started = Instant::now();

        let Some(proving_key) = &self.proving_key else {
            let (mut public_values, report) = self
                .client
                .execute(self.elf.clone(), stdin)
                .run()
                .context("SP1 execution failed")?;
            return Ok(BatchOutcome {
                report_hash: public_values.read(),
                cycles: Some(report.total_instruction_count()),
                seconds: started.elapsed().as_secs_f64(),
            });
        };

        let request = self.client.prove(proving_key, stdin);
        let proof = match self.mode {
            ProofMode::Core => request.core().run(),
            ProofMode::Compressed => request.compressed().run(),
            ProofMode::Groth16 => request.groth16().run(),
            ProofMode::Execute => unreachable!("execute mode has no proving key"),
        }
        .context("SP1 proving failed")?;
        let seconds = started.elapsed().as_secs_f64();

        self.client
            .verify(&proof, proving_key.verifying_key(), None)
            .context("SP1 proof verification failed")?;

        if self.mode == ProofMode::Groth16 {
            let full_report = FullReport {
                verifying_key: proving_key.verifying_key().bytes32(),
                proof: proof.clone(),
            };
            let json_data = serde_json::to_string_pretty(&full_report)?;
            File::create("prover-report.json")?.write_all(json_data.as_bytes())?;
        }

        let mut public_values = proof.public_values.clone();
        Ok(BatchOutcome {
            report_hash: public_values.read(),
            cycles: None,
            seconds,
        })
    }
}
