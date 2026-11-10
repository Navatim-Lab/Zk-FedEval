
import { Binary } from "polkadot-api";
import {
  encodeAbiParameters,
  parseAbiParameters,
} from "viem";
import {
  readFile,
  readdir,
} from "node:fs/promises";
import path from "node:path";

const directory = import.meta.dir;
const contractsDirectory = path.join(directory, "contractsBytes");

const STORAGE_DEPOSIT_LIMIT = 10_000_000_000_000n;
const WEIGHT_MARGIN_PERCENT = 30n;

type Entry = {
  keyArgs: unknown[];
  value: any;
};

type Weight = {
  ref_time: bigint;
  proof_size: bigint;
};

function normalizeHex(value: unknown): string {
  if (typeof value === "string") return value.toLowerCase();

  if (value instanceof Uint8Array) {
    return `0x${Buffer.from(value).toString("hex")}`.toLowerCase();
  }

  if (value && typeof value === "object" && "toString" in value) {
    return String(value).toLowerCase();
  }

  throw new Error(`Cannot normalize hex value: ${String(value)}`);
}

function getCodeHash(entry: Entry): string {
  return normalizeHex(entry.keyArgs[0]);
}

function getAccountAddress(entry: Entry): string {
  return normalizeHex(entry.keyArgs[0]);
}

function getAccountCodeHash(entry: Entry): string | undefined {
  const accountType = entry.value.account_type;

  if (accountType.type !== "Contract") return undefined;

  const codeHash = accountType.value.code_hash;

  return codeHash ? normalizeHex(codeHash) : undefined;
}

function getContractAddress(
  entries: Entry[],
  expectedCodeHash: string,
  previousAddresses: Set<string>,
): string {
  const entry = entries.find((entry) => {
    const address = getAccountAddress(entry);

    return (
      !previousAddresses.has(address) &&
      getAccountCodeHash(entry) === expectedCodeHash
    );
  });

  if (!entry) {
    throw new Error(
      `Could not find contract address for code hash ${expectedCodeHash}`,
    );
  }

  return getAccountAddress(entry);
}

function addWeightMargin(weight: Weight): Weight {
  return {
    ref_time:
      (weight.ref_time * (100n + WEIGHT_MARGIN_PERCENT)) / 100n,
    proof_size:
      (weight.proof_size * (100n + WEIGHT_MARGIN_PERCENT)) / 100n,
  };
}

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

async function waitForTransaction(
  transaction: any,
  signer: any,
): Promise<any> {
  return new Promise((resolve, reject) => {
    let subscription: any;

    subscription = transaction.signSubmitAndWatch(signer).subscribe({
      next: (result: any) => {
        if (result.type === "finalized") {
          subscription?.unsubscribe();

          console.log(`Transaction finalized: ${result.txHash}`);

          if (!result.ok) {
            reject(
              new Error(
                `Transaction failed: ${JSON.stringify(
                  serialize(result.dispatchError),
                )}`,
              ),
            );
            return;
          }

          console.log("Transaction execution successful.");
          resolve(result);
        }

        if (result.type === "invalid" || result.type === "error") {
          subscription?.unsubscribe();
          reject(new Error(`Transaction failed: ${result.type}`));
        }
      },

      error: (error: unknown) => {
        subscription?.unsubscribe();
        reject(error);
      },
    });
  });
}

async function deployContract(
  api: any,
  signer: any,
  origin: string,
  contractName: string,
  code: Uint8Array,
  constructorData: Uint8Array,
) {
  console.log(`\nUploading ${contractName}...`);

  const previousCodeEntries =
    await api.query.Revive.CodeInfoOf.getEntries();

  const uploadResult = await waitForTransaction(
    api.tx.Revive.upload_code({
      code,
      storage_deposit_limit: STORAGE_DEPOSIT_LIMIT,
    }),
    signer,
  );

  const currentCodeEntries =
    await api.query.Revive.CodeInfoOf.getEntries();

  const previousCodeHashes = new Set(
    previousCodeEntries.map((entry: Entry) => getCodeHash(entry)),
  );

  const newCodeEntries = currentCodeEntries.filter(
    (entry: Entry) => !previousCodeHashes.has(getCodeHash(entry)),
  );

  if (newCodeEntries.length !== 1) {
    throw new Error(
      `Could not uniquely identify uploaded ${contractName} code. New entries: ${newCodeEntries.length}.`,
    );
  }

  const codeHash = getCodeHash(newCodeEntries[0]);

  console.log(`${contractName} code hash: ${codeHash}`);

  const previousAccountEntries =
    await api.query.Revive.AccountInfoOf.getEntries();

  const previousAddresses = new Set(
    previousAccountEntries.map((entry: Entry) =>
      getAccountAddress(entry),
    ),
  );

  console.log(`Dry-running ${contractName} instantiation...`);

  const dryRun = await api.apis.ReviveApi.instantiate(
    origin,
    0n,
    undefined,
    undefined,
    {
      type: "Existing",
      value: codeHash,
    },
    constructorData,
    undefined,
  );

  if (dryRun.result?.type === "Err") {
    throw new Error(
      `${contractName} dry-run failed: ${JSON.stringify(
        serialize(dryRun.result),
      )}`,
    );
  }

  const dryRunWeight: Weight = {
    ref_time: BigInt(dryRun.weight_required.ref_time),
    proof_size: BigInt(dryRun.weight_required.proof_size),
  };

  const submittedWeight = addWeightMargin(dryRunWeight);

  const storageDepositLimit =
    dryRun.storage_deposit.type === "Charge"
      ? BigInt(dryRun.storage_deposit.value)
      : 0n;

  console.log("Required weight:", serialize(dryRunWeight));
  console.log("Submitted weight:", serialize(submittedWeight));
  console.log("Storage deposit:", storageDepositLimit.toString());
  console.log("Gas consumed:", dryRun.gas_consumed.toString());

  console.log(`Instantiating ${contractName}...`);

  const deploymentResult = await waitForTransaction(
    api.tx.Revive.instantiate({
      value: 0n,
      weight_limit: submittedWeight,
      storage_deposit_limit: storageDepositLimit,
      code_hash: codeHash,
      data: Binary.fromHex(
        `0x${Buffer.from(constructorData).toString("hex")}`,
      ),
      salt: undefined,
    }),
    signer,
  );

  const currentAccountEntries =
    await api.query.Revive.AccountInfoOf.getEntries();

  const address = getContractAddress(
    currentAccountEntries,
    codeHash,
    previousAddresses,
  );

  console.log(`${contractName} deployed at: ${address}`);

  return {
    address,
    codeHash,
    codeBytes: code.length,
    dryRunWeight,
    submittedWeight,
    gasConsumed: BigInt(dryRun.gas_consumed),
    storageDepositLimit,
    uploadTransaction: uploadResult.txHash,
    deploymentTransaction: deploymentResult.txHash,
  };
}

export async function deployContracts(
  api: any,
  signer: any,
  origin: string,
) {
  const files = await readdir(contractsDirectory);

  const sp1File =
    process.env.SP1_CONTRACT_FILE ??
    files.find((file) => /SP1Verifier.*\.pvm$/i.test(file));

  const fedEvalFile =
    process.env.FEDEVAL_CONTRACT_FILE ??
    files.find((file) => /FedEval.*\.pvm$/i.test(file));

  if (!sp1File || !fedEvalFile) {
    throw new Error(
      `Contract artifacts not found in ${contractsDirectory}.`,
    );
  }

  const sp1Code = new Uint8Array(
    await readFile(path.join(contractsDirectory, sp1File)),
  );

  const fedEvalCode = new Uint8Array(
    await readFile(path.join(contractsDirectory, fedEvalFile)),
  );

  console.log("\nDeploying SP1Verifier...");

  const sp1 = await deployContract(
    api,
    signer,
    origin,
    "SP1Verifier",
    sp1Code,
    new Uint8Array(),
  );

  const constructorHex = encodeAbiParameters(
    parseAbiParameters("address"),
    [sp1.address as `0x${string}`],
  );

  const constructorData = new Uint8Array(
    Buffer.from(constructorHex.slice(2), "hex"),
  );

  console.log("\nDeploying FedEval...");

  const fedEval = await deployContract(
    api,
    signer,
    origin,
    "FedEval",
    fedEvalCode,
    constructorData,
  );

  return {
    SP1Verifier: sp1,
    FedEval: {
      ...fedEval,
      constructorArgument: sp1.address,
    },
  };
}
