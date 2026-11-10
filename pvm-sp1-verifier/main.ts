
import { createClient } from "polkadot-api";
import { getPolkadotSigner } from "polkadot-api/signer";
import { getWsProvider } from "polkadot-api/ws";
import { reviveDev } from "@polkadot-api/descriptors";
import { Keyring } from "@polkadot/keyring";
import {
  encodeFunctionData,
  bytesToHex,
  hexToBytes,
} from "viem";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { readFile } from "node:fs/promises";

import abi from "./contractsBytes/verifierEZKL_ABI.json" with { type: "json" };
import proverReport from "../zk-fedeval-prover/prover-report.json" with { type: "json" };

import { deployContracts } from "./deploy";

const directory = import.meta.dir;
const NODE_URL = "ws://127.0.0.1:9944";
const WEIGHT_MARGIN_PERCENT = 30n;
const STORAGE_DEPOSIT_LIMIT = 10_000_000_000_000n;

function serialize(value: unknown): any {
  return JSON.parse(
    JSON.stringify(value, (_, item) => {
      if (typeof item === "bigint") return item.toString();

      if (item instanceof Uint8Array) {
        return `0x${Buffer.from(item).toString("hex")}`;
      }

      return item;
    }),
  );
}

function addWeightMargin(weight: {
  ref_time: bigint;
  proof_size: bigint;
}) {
  return {
    ref_time:
      (weight.ref_time * (100n + WEIGHT_MARGIN_PERCENT)) / 100n,
    proof_size:
      (weight.proof_size * (100n + WEIGHT_MARGIN_PERCENT)) / 100n,
  };
}

async function getExistingDeployment(api: any) {
  const paths = [
    path.join(directory, "benchmark-results.json"),
    path.join(directory, "deployment-result.json"),
  ];

  let saved: any;

  for (const file of paths) {
    try {
      saved = JSON.parse(await readFile(file, "utf8"));
      if (saved.contracts?.SP1Verifier && saved.contracts?.FedEval) {
        break;
      }
      saved = undefined;
    } catch {
      saved = undefined;
    }
  }

  if (!saved) return undefined;

  const sp1 = saved.contracts.SP1Verifier;
  const fedEval = saved.contracts.FedEval;

  const entries = await api.query.Revive.AccountInfoOf.getEntries();

  function isDeployed(contract: any) {
    const address = contract.address?.toLowerCase();
    const expectedHash = contract.codeHash?.toLowerCase();

    return entries.some((entry: any) => {
      const accountAddress = String(entry.keyArgs[0]).toLowerCase();
      const accountType = entry.value.account_type;

      return (
        accountAddress === address &&
        accountType.type === "Contract" &&
        String(accountType.value.code_hash).toLowerCase() === expectedHash
      );
    });
  }

  if (!isDeployed(sp1) || !isDeployed(fedEval)) {
    return undefined;
  }

  return {
    SP1Verifier: sp1,
    FedEval: fedEval,
  };
}

async function verifyProof(
  api: any,
  signer: any,
  aliceAddress: string,
  contractAddress: string,
) {
  console.log("\nPreparing proof verification...");

  const groth16VkeyHash =
    proverReport.proof.proof.Groth16.groth16_vkey_hash;

  const selector = bytesToHex(
    new Uint8Array(groth16VkeyHash.slice(0, 4)),
  );

  const encodedProofBytes =
    `0x${proverReport.proof.proof.Groth16.encoded_proof}`;

  const proofBytesFull =
    (selector + encodedProofBytes.slice(2)) as `0x${string}`;

  const encodedCallData = encodeFunctionData({
    abi,
    functionName: "verifyProof",
    args: [
      proverReport.verifying_key,
      bytesToHex(
        proverReport.proof.public_values.buffer.data as any,
      ),
      proofBytesFull,
    ],
  });

  console.log("Dry-running proof verification...");

  const dryRun = await api.apis.ReviveApi.call(
    aliceAddress,
    contractAddress,
    0n,
    undefined,
    undefined,
    hexToBytes(encodedCallData),
  );

  if (dryRun.result.value.flags !== 0) {
    throw new Error(
      `Proof verification dry-run reverted: ${JSON.stringify(
        serialize(dryRun.result),
      )}`,
    );
  }

  const dryRunWeight = {
    ref_time: BigInt(dryRun.weight_required.ref_time),
    proof_size: BigInt(dryRun.weight_required.proof_size),
  };

  const submittedWeight = addWeightMargin(dryRunWeight);

  console.log("Dry-run successful.");
  console.log("Required weight:", serialize(dryRunWeight));
  console.log("Submitted weight:", serialize(submittedWeight));
  console.log("Gas consumed:", dryRun.gas_consumed.toString());

  const transaction = api.tx.Revive.call({
    dest: contractAddress,
    value: 0n,
    weight_limit: submittedWeight,
    storage_deposit_limit: STORAGE_DEPOSIT_LIMIT,
    data: hexToBytes(encodedCallData),
  });

  console.log("Submitting proof verification...");

  const finalized = await new Promise<any>((resolve, reject) => {
    let subscription: any;

    subscription = transaction.signSubmitAndWatch(signer).subscribe({
      next: (event: any) => {
        if (event.type === "finalized") {
          subscription?.unsubscribe();

          if (!event.ok) {
            reject(
              new Error(
                `Verification transaction failed: ${JSON.stringify(
                  serialize(event),
                )}`,
              ),
            );
            return;
          }

          resolve(event);
        }

        if (event.type === "invalid" || event.type === "error") {
          subscription?.unsubscribe();
          reject(new Error(`Verification transaction failed: ${event.type}`));
        }
      },

      error: (error: unknown) => {
        subscription?.unsubscribe();
        reject(error);
      },
    });
  });

  const feeEvent = finalized.events.find(
    (event: any) =>
      event.type === "TransactionPayment" &&
      event.value.type === "TransactionFeePaid",
  );

  const successEvent = finalized.events.find(
    (event: any) =>
      event.type === "System" &&
      event.value.type === "ExtrinsicSuccess",
  );

  const actualWeight =
    successEvent?.value.value.dispatch_info.weight;

  const actualFee = feeEvent?.value.value.actual_fee;
  const tip = feeEvent?.value.value.tip;

  console.log("\nProof verification finalized.");
  console.log("Transaction hash:", finalized.txHash);

  return {
    contractAddress,
    transactionHash: finalized.txHash,
    block: finalized.block,

    dryRun: {
      weight: dryRunWeight,
      submittedWeight,
      gasConsumed: BigInt(dryRun.gas_consumed),
    },

    execution: {
      weight: actualWeight
        ? {
            ref_time: BigInt(actualWeight.ref_time),
            proof_size: BigInt(actualWeight.proof_size),
          }
        : undefined,
      actualFee: actualFee !== undefined ? BigInt(actualFee) : undefined,
      tip: tip !== undefined ? BigInt(tip) : undefined,
    },

    events: finalized.events,
  };
}

async function main() {
  const client = createClient(getWsProvider(NODE_URL));

  try {
    const api = client.getTypedApi(reviveDev);

    const keyring = new Keyring({ type: "sr25519" });
    const alicePair = keyring.addFromUri("//Alice");

    const signer = getPolkadotSigner(
      alicePair.publicKey,
      "Sr25519",
      (payload) => alicePair.sign(payload),
    );

    // 1. Deploy both contracts buut check first
    let contracts = await getExistingDeployment(api);

    if (contracts) {
      console.log("Skipping deployment.");
    } else {
      console.log("Contracts not found. Deploying...");

      contracts = await deployContracts(
        api,
        signer,
        alicePair.address,
      );
    }

    // 2. Verify the proof against the deployed FedEval.
    const verification = await verifyProof(
      api,
      signer,
      alicePair.address,
      contracts.FedEval.address,
    );

    // 3. Save all metrics.
    const result = {
      timestamp: new Date().toISOString(),
      node: NODE_URL,
      weightMarginPercent: WEIGHT_MARGIN_PERCENT.toString(),

      contracts: serialize(contracts),

      verification: serialize(verification),
    };

    const outputPath = path.join(
      directory,
      "benchmark-results.json",
    );

    await writeFile(
      outputPath,
      JSON.stringify(result, null, 2),
    );

    console.log("\nBenchmark completed.");
    console.log(JSON.stringify(result, null, 2));
    console.log(`\nSaved: ${outputPath}`);
  } finally {
    client.destroy();
  }
}

main().catch((error) => {
  console.error("Benchmark failed:", error);
  process.exit(1);
});
