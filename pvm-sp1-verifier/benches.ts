import si from "systeminformation";
import { readFile, writeFile } from "node:fs/promises";

const [cpu, mem, os] = await Promise.all([
  si.cpu(),
  si.mem(),
  si.osInfo(),
]);

const run = Bun.spawn(["bun", "run", "main.ts"], {
  stdout: "inherit",
  stderr: "inherit",
});

if (await run.exited !== 0) throw new Error("Benchmark failed");

const results = JSON.parse(await readFile("benchmark-results.json", "utf8"));
const prover = JSON.parse(
  await readFile("../zk-fedeval-prover/prover-report.json", "utf8"),
);

const report = {
  machine: {
    cpu: cpu.brand,
    cores: cpu.cores,
    ramGB: +(mem.total / 1024 ** 3).toFixed(2),
    os: `${os.distro} ${os.release}`,
  },
  zkProofSizeBytes: prover.proof.proof.Groth16.encoded_proof.length / 2,
  verificationWeights: results.verification.execution.weight,
  verificationContractSizeBytes: results.contracts.SP1Verifier.codeBytes,
};

const fmt = (n: number | string | bigint) => Number(n).toLocaleString("en-US");

const table = `| Category | Metric | Result |
|:---|:---|---:|
| **Machine** | CPU | ${report.machine.cpu} |
| **Machine** | Cores | ${report.machine.cores} |
| **Machine** | RAM | ${report.machine.ramGB} GB |
| **Machine** | OS | ${report.machine.os} |
| **ZK proof** | Proof size | ${fmt(report.zkProofSizeBytes)} bytes |
| **Verification weight (2D)** | \`ref_time\` | ${fmt(report.verificationWeights.ref_time)} |
| **Verification weight (2D)** | \`proof_size\` | ${fmt(report.verificationWeights.proof_size)} |
| **Verification contract** | Code size | ${fmt(report.verificationContractSizeBytes)} bytes |
`;
await writeFile("README.md", table);
