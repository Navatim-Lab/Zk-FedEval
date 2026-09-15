use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use sp1_sdk::blocking::prelude::*;
use sp1_sdk::blocking::{EnvProver, EnvProvingKey, ProverClient};

use zk_fedeval_common::{EvalBatch, EvalReport};

/// Sets up the SP1 proving/verifying keys once for the guest ELF and proves
/// the CNN evaluation over a given batch, verifying the proof it produces.
pub struct EvalProver {
    client: EnvProver,
    proving_key: EnvProvingKey,
}


#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct FullReport {
    verifying_key: String,
    proof: SP1ProofWithPublicValues,
}

impl EvalProver {
    pub fn setup(elf: Elf) -> Result<Self> {
        println!("[prover] setting up proving/verifying keys for the guest ELF...");
        let client = ProverClient::from_env();
        let proving_key = client.setup(elf).context("SP1 setup failed")?;
        let vk = proving_key.clone().verifying_key().bytes32();

        println!("VK: 0x{}", hex::encode(vk));       
        println!("[prover] setup complete");

        Ok(Self {
            client,
            proving_key,
        })
    }

    pub fn prove(&self, batch: &EvalBatch) -> Result<EvalReport> {
        let mut stdin = SP1Stdin::new();
        stdin.write(batch);

        println!(
            "[prover] proving guest execution over {} samples (this runs the CNN inside the zkVM)...",
            batch.num_samples()
        );
        let mut proof = self
            .client
            .prove(&self.proving_key, stdin)
            .run()
            .context("SP1 proving failed")?;
        println!("[prover] proof generated, verifying...");

        let full_report = FullReport {
            verifying_key: self.proving_key.clone().verifying_key().bytes32(),
            proof: proof.clone(),
        };

        let json_data = serde_json::to_string_pretty(&full_report)?;
        let mut file = File::create("prover-report.json")?;
        file.write_all(json_data.as_bytes())?;

        self.client
            .verify(&proof, self.proving_key.verifying_key(), None)
            .context("SP1 proof verification failed")?;
        println!("[prover] proof verified");

        Ok(proof.public_values.read())
    }
}
