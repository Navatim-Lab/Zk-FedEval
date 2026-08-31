# Zk-FedEval
ZKP-FedEval — a privacy-preserving verifiable evaluation mechanism for Federated Learning on Kusama.

# Privacy-Preserving Flower Evaluation PoC

This project tests whether Flower clients can prove that a constrained evaluation produced a loss below a public threshold, without revealing private evaluation samples.

The current Flower baseline runs locally in Flower simulation with two clients,
one round, a 100-output CNN, and deterministic non-IID CIFAR-100 fine-label
partitions. The simulated clients use Flower's real application and Message API
but are not separate authenticated machines.

The PoC covers EZKL proof generation and verification, Flower proof transport, server acceptance policy, replay protection, model-version binding, and experimental PolkaVM verification.
