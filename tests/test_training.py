"""Real-data regression tests for deterministic local CNN training."""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass

import torch
from flwr.app import ArrayRecord

from zk_fedeval import task

BATCH_SIZE = 4
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.1
EXPECTED_ARRAY_NAMES = (
    "conv1.bias",
    "conv1.weight",
    "conv2.bias",
    "conv2.weight",
    "fc1.bias",
    "fc1.weight",
    "fc2.bias",
    "fc2.weight",
    "fc3.bias",
    "fc3.weight",
)
EXPECTED_PARAMETER_COUNT = 69_656
EXPECTED_TRAINING_EXAMPLES = (20_256, 19_743)
EXPECTED_VALIDATION_EXAMPLES = (5_065, 4_936)
EXPECTED_BASE_DIGEST = (
    "1c43da4d7c7513f662796366de39230da84bc7f69e52626bd12b7d31f2a6f7c1"
)
EXPECTED_CANDIDATE_DIGESTS = (
    "afd59d5faea01c561fdc1b7939868a31338a761b300ed7e2794adf292e5a6f1f",
    "f9055bf57ba0491ad7e4a753bf464ba96263076a23b2a251a95c97e52c21dd3c",
)
TEST_DIGEST_DOMAIN = b"deterministic-training-state-sha256-v1\x00"


@dataclass(frozen=True)
class TrainingSnapshot:
    """Public reproducibility facts for one trained client candidate."""

    partition_id: int
    training_examples: int
    validation_examples: int
    array_names: tuple[str, ...]
    parameter_count: int
    candidate_digest: str


def _clone_state_dict(model: task.Net) -> dict[str, torch.Tensor]:
    """Copy a model state onto CPU for independent comparisons."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _test_state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash complete tensors for this deterministic-training regression only."""
    digest = hashlib.sha256(TEST_DIGEST_DOMAIN)
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        raw_bytes = tensor.numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(raw_bytes)))
        digest.update(raw_bytes)
    return digest.hexdigest()


def _assert_complete_array_record(
    candidate_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
) -> None:
    """Prove Flower serialization contains the candidate state, not its delta."""
    transported = ArrayRecord(candidate_state).to_torch_state_dict()

    assert set(transported) == set(candidate_state) == set(base_state)
    assert len(transported) == len(EXPECTED_ARRAY_NAMES)
    for name, candidate_tensor in candidate_state.items():
        assert transported[name].shape == base_state[name].shape
        assert transported[name].dtype == base_state[name].dtype
        assert torch.equal(transported[name], candidate_tensor)

    delta_state = {
        name: candidate_state[name] - base_state[name] for name in candidate_state
    }
    assert any(
        not torch.equal(transported[name], delta_state[name]) for name in transported
    )


def _train_once(
    partition_id: int,
    base_state: dict[str, torch.Tensor],
) -> tuple[TrainingSnapshot, dict[str, torch.Tensor], float]:
    """Train one client once from the shared base state."""
    task.configure_determinism(task.PARTITION_SEED + partition_id)
    model = task.Net()
    model.load_state_dict(base_state)

    trainloader, valloader = task.load_data(
        partition_id=partition_id,
        num_partitions=task.NUM_PARTITIONS,
        batch_size=BATCH_SIZE,
        partition_alpha=task.PARTITION_ALPHA,
        partition_seed=task.PARTITION_SEED,
    )
    train_loss = task.train(
        model,
        trainloader,
        epochs=LOCAL_EPOCHS,
        lr=LEARNING_RATE,
        device=torch.device("cpu"),
    )
    candidate_state = _clone_state_dict(model)
    _assert_complete_array_record(candidate_state, base_state)

    snapshot = TrainingSnapshot(
        partition_id=partition_id,
        training_examples=len(trainloader.dataset),
        validation_examples=len(valloader.dataset),
        array_names=tuple(sorted(candidate_state)),
        parameter_count=sum(tensor.numel() for tensor in candidate_state.values()),
        candidate_digest=_test_state_digest(candidate_state),
    )
    return snapshot, candidate_state, train_loss


def _assert_states_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_two_client_local_training_is_deterministic_and_distinct() -> None:
    """Both clients must repeat exactly while producing distinct candidates."""
    task.fds = None
    task.fds_config = None
    task.configure_determinism(task.PARTITION_SEED)
    base_model = task.Net()
    base_state = _clone_state_dict(base_model)
    base_digest = _test_state_digest(base_state)
    assert base_digest == EXPECTED_BASE_DIGEST

    client_snapshots = []
    client_states = []
    for partition_id in range(task.NUM_PARTITIONS):
        first_snapshot, first_state, first_loss = _train_once(
            partition_id,
            base_state,
        )
        second_snapshot, second_state, second_loss = _train_once(
            partition_id,
            base_state,
        )

        assert first_snapshot == second_snapshot
        assert first_loss == second_loss
        assert math.isfinite(first_loss)
        _assert_states_equal(first_state, second_state)
        assert first_snapshot.candidate_digest != base_digest
        assert any(
            not torch.equal(first_state[name], base_state[name]) for name in base_state
        )

        client_snapshots.append(first_snapshot)
        client_states.append(first_state)

    assert (
        tuple(snapshot.training_examples for snapshot in client_snapshots)
        == EXPECTED_TRAINING_EXAMPLES
    )
    assert (
        tuple(snapshot.validation_examples for snapshot in client_snapshots)
        == EXPECTED_VALIDATION_EXAMPLES
    )
    assert (
        tuple(snapshot.candidate_digest for snapshot in client_snapshots)
        == EXPECTED_CANDIDATE_DIGESTS
    )
    assert client_snapshots[0].candidate_digest != client_snapshots[1].candidate_digest
    assert any(
        not torch.equal(client_states[0][name], client_states[1][name])
        for name in base_state
    )
    assert all(
        snapshot.array_names == EXPECTED_ARRAY_NAMES for snapshot in client_snapshots
    )
    assert all(
        snapshot.parameter_count == EXPECTED_PARAMETER_COUNT
        for snapshot in client_snapshots
    )

    print(
        "DETERMINISTIC_TRAINING_SNAPSHOT="
        + json.dumps(
            {
                "base_digest": base_digest,
                "clients": [asdict(snapshot) for snapshot in client_snapshots],
            },
            sort_keys=True,
        )
    )
