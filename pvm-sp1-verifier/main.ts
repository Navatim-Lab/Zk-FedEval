const MAIN_VERIFIER_CONTRACT_ADDRESS = "0x124954f324ab6253b8efc74c3073a7e88338cda5"


// submit a proof from zk-fedeval-prover
import { reviveDev} from '@polkadot-api/descriptors';
import { createClient } from 'polkadot-api';
import { Keyring } from "@polkadot/keyring";
import { getPolkadotSigner } from "polkadot-api/signer";
import { getWsProvider } from 'polkadot-api/ws';
import {encodeFunctionData, bytesToHex, hexToBytes, decodeErrorResult} from "viem";
import abi from "./contractsBytes/verifierEZKL_ABI.json" with { type: 'json' };
import abiSp1Verifier from "./contractsBytes/SP1VerifierGroth16_ABI.json" with { type: 'json' };
// import the prover-report.json file
import proverReport from '../zk-fedeval-prover/prover-report.json' with { type: 'json' };


const client = createClient(getWsProvider('ws://127.0.0.1:9944'));
const reviveApi = client.getTypedApi(reviveDev);

// get the contract addresses and their code hash
const contractsObject = await reviveApi.query.Revive.AccountInfoOf.getEntries()
    .then((entries) =>
        entries.map(({ keyArgs, value }) => {
            const [accountId] = keyArgs;
            const codeHash = value.account_type.value?.code_hash;

            return {
                accountId,
                codeHash,
            };
        })
    );

// console.log(contractsObject);
// call the contract with the prover report
const groth16VkeyHash = proverReport.proof.proof.Groth16.groth16_vkey_hash;
const selector = bytesToHex(new Uint8Array(groth16VkeyHash.slice(0, 4)));
const encodedProofBytes = `0x${proverReport.proof.proof.Groth16.encoded_proof}`;

const proofBytesFull = (selector + encodedProofBytes.slice(2)) as `0x${string}`;

const encodedCallData = encodeFunctionData({
    abi: abi,
    functionName: 'verifyProof',
    args: [
        proverReport.verifying_key,
        bytesToHex(proverReport.proof.public_values.buffer.data as any),
        proofBytesFull,
    ],
})

const keyring = new Keyring({ type: "sr25519" });
const alicePair = keyring.addFromUri("//Alice");

const alice = getPolkadotSigner(
    alicePair.publicKey,
    "Sr25519",
    (data) => alicePair.sign(data),
);

const dryRun = await reviveApi.apis.ReviveApi.call(
    alicePair.address,
    MAIN_VERIFIER_CONTRACT_ADDRESS,
    0n,
    undefined,
    undefined,
    hexToBytes(encodedCallData),
);

if (dryRun.result.value.flags !== 0) {
    throw new Error("Dry run reverted, aborting before submit");
}

const { ref_time, proof_size } = dryRun.weight_required;

const tx = reviveApi.tx.Revive.call({
    dest: MAIN_VERIFIER_CONTRACT_ADDRESS,
    value: 0n,
    weight_limit: {
        ref_time: (ref_time * 130n) / 100n,  
        proof_size: (proof_size * 130n) / 100n,
    },
    storage_deposit_limit: 10_000_000_000_000n,
    data: hexToBytes(encodedCallData),
});


tx.signSubmitAndWatch(alice).subscribe({
    next: (event) => {
        console.log("Event:", event);
    },
    error: (error) => {
        console.error("Transaction error:", error);
        client.destroy();
    },
    complete: () => {
        console.log("Transaction complete");
        client.destroy();
    },
});
