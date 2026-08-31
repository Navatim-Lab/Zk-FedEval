"""Focused tests for the non-IID CIFAR-100 Flower task."""

import torch

from zk_fedeval import task


def test_cifar100_model_and_dataset_contract() -> None:
    """The retained quickstart CNN must emit one logit per CIFAR-100 class."""
    model = task.Net().eval()

    with torch.no_grad():
        output = model(torch.zeros(2, 3, 32, 32, dtype=torch.float32))

    assert task.DATASET_NAME == "uoft-cs/cifar100"
    assert task.IMAGE_COLUMN == "img"
    assert task.LABEL_COLUMN == "fine_label"
    assert task.NUM_CLASSES == 100
    assert model.fc3.out_features == 100
    assert output.shape == (2, 100)


def test_dirichlet_partitioner_configuration(monkeypatch) -> None:
    """The non-IID partitioner must use the frozen fine-label-skew settings."""
    captured = {}

    class CapturingPartitioner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(task, "DirichletPartitioner", CapturingPartitioner)

    partitioner = task.create_partitioner(
        num_partitions=2,
        alpha=task.PARTITION_ALPHA,
        seed=task.PARTITION_SEED,
    )

    assert isinstance(partitioner, CapturingPartitioner)
    assert captured == {
        "num_partitions": 2,
        "partition_by": "fine_label",
        "alpha": 0.5,
        "min_partition_size": 10,
        "self_balancing": False,
        "shuffle": True,
        "seed": 2026,
    }


def test_federated_dataset_configuration(monkeypatch) -> None:
    """The dataset loader must pin its revision and global shuffle seed."""
    captured = {}
    partitioner = object()

    class CapturingFederatedDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(task, "FederatedDataset", CapturingFederatedDataset)
    monkeypatch.setattr(task, "create_partitioner", lambda **_kwargs: partitioner)

    result = task.create_federated_dataset()

    assert isinstance(result, CapturingFederatedDataset)
    assert captured == {
        "dataset": task.DATASET_NAME,
        "revision": task.DATASET_REVISION,
        "preprocessor": task.add_sample_ids,
        "partitioners": {"train": partitioner},
        "shuffle": True,
        "seed": task.PARTITION_SEED,
    }


def test_model_initialization_is_repeatable_for_fixed_seed() -> None:
    """The same PyTorch seed must produce identical initial CNN parameters."""
    task.configure_determinism(task.PARTITION_SEED)
    first_state = {
        name: tensor.detach().clone()
        for name, tensor in task.Net().state_dict().items()
    }

    task.configure_determinism(task.PARTITION_SEED)
    second_state = task.Net().state_dict()

    assert first_state.keys() == second_state.keys()
    assert all(
        torch.equal(first_state[name], second_state[name]) for name in first_state
    )
