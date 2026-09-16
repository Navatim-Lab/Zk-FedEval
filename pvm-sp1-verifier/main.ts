// submit a proof from zk-fedeval-prover
import { reviveDev } from '@polkadot-api/descriptors';
import { createClient } from 'polkadot-api';
import { getWsProvider } from 'polkadot-api/ws';

// import the prover-report.json file
import proverReport from '../zk-fedeval-prover/prover-report.json' with { type: 'json' };

const client = createClient(getWsProvider('ws://127.0.0.1:9944'));
const reviveApi = client.getTypedApi(reviveDev);

// get the contract addresses and their code hash
const contractObject = await reviveApi.query.Revive.AccountInfoOf.getEntries().then((entries) => {
    entries.forEach(({keyArgs, value}) => {
        const [accountId] = keyArgs;
        const codeHash = value.account_type.value?.code_hash;
        return { accountId, codeHash };
    })
});

// call the contract with the prover report



client.destroy();