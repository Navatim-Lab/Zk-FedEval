"""Real-data tests for deterministic non-IID CIFAR-100 partitions."""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version

import datasets
import numpy as np
import torch

from zk_fedeval import task

MIN_TOTAL_VARIATION_DISTANCE = 0.10
EXPECTED_DATASET_FINGERPRINT = "1e792e149ee23766"
EXPECTED_DATASET_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
EXPECTED_DATASETS_VERSION = "3.1.0"
EXPECTED_FLWR_DATASETS_VERSION = "0.5.0"
EXPECTED_PARTITION_SIZES = (25_321, 24_679)
EXPECTED_PARTITION_FINGERPRINTS = ("50ff3549b696ba7b", "4ac39c335721610d")
EXPECTED_MEMBERSHIP_DIGESTS = (
    "88558b48f8eac7362d82ff1c7c9f4252da303dc12a4d627e86b1ff340fd914a4",
    "322f9ceacb7050a27dd816123b7408aae5cb2f672a4e98d05f6a0a9c80e9720c",
)
EXPECTED_LABEL_HISTOGRAM_DIGESTS = (
    "7e2c7f4527ce44238054567eb77818e60ced5991a4dd00de6e45c6d6833ee610",
    "34e8599db9c36ddd589104772acfbbc3467d88d6bb41cc03399f08be41858ef5",
)
EXPECTED_TOTAL_VARIATION_DISTANCE = 0.6012255214063219


@dataclass(frozen=True)
class PartitionSnapshot:
    """Reproducibility evidence for one complete partitioning run."""

    dataset_revision: str
    dataset_fingerprint: str
    datasets_version: str
    flwr_datasets_version: str
    partition_sizes: tuple[int, int]
    partition_fingerprints: tuple[str, str]
    membership_digests: tuple[str, str]
    label_histograms: tuple[tuple[int, ...], tuple[int, ...]]
    label_histogram_digests: tuple[str, str]
    total_variation_distance: float


def _membership_digest(sample_ids: np.ndarray) -> str:
    """Hash sorted source-row IDs using a fixed little-endian representation."""
    canonical = np.sort(sample_ids.astype("<u8", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _histogram_digest(histogram: np.ndarray) -> str:
    """Hash a 100-bin histogram using a fixed little-endian representation."""
    canonical = histogram.astype("<u8", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _load_snapshot() -> PartitionSnapshot:
    """Construct the real dataset and validate both raw client partitions."""
    federated_dataset = task.create_federated_dataset()
    partitions = tuple(
        federated_dataset.load_partition(partition_id, split="train")
        for partition_id in range(task.NUM_PARTITIONS)
    )
    full_train = federated_dataset.load_split("train")

    assert task.NUM_PARTITIONS == 2
    assert federated_dataset.partitioners["train"].num_partitions == 2
    assert len(full_train) == 50_000
    assert all(len(partition) > 0 for partition in partitions)
    assert sum(len(partition) for partition in partitions) == len(full_train)

    sample_ids = tuple(
        np.asarray(partition[task.SAMPLE_ID_COLUMN], dtype=np.uint64)
        for partition in partitions
    )
    combined_ids = np.concatenate(sample_ids)
    assert np.unique(combined_ids).size == combined_ids.size
    assert np.array_equal(
        np.sort(combined_ids),
        np.arange(len(full_train), dtype=np.uint64),
    )

    labels = tuple(
        np.asarray(partition[task.LABEL_COLUMN], dtype=np.int64)
        for partition in partitions
    )
    assert all(label_values.min() >= 0 for label_values in labels)
    assert all(label_values.max() < task.NUM_CLASSES for label_values in labels)

    histograms = tuple(
        np.bincount(label_values, minlength=task.NUM_CLASSES) for label_values in labels
    )
    distributions = tuple(
        histogram.astype(np.float64) / histogram.sum() for histogram in histograms
    )
    total_variation_distance = 0.5 * np.abs(distributions[0] - distributions[1]).sum()

    for partition in partitions:
        transformed = task.apply_transforms(partition[:2])
        images = transformed[task.IMAGE_COLUMN]
        assert len(images) == 2
        assert all(image.dtype == torch.float32 for image in images)
        assert all(image.shape == (3, 32, 32) for image in images)

    return PartitionSnapshot(
        dataset_revision=task.DATASET_REVISION,
        dataset_fingerprint=full_train._fingerprint,
        datasets_version=datasets.__version__,
        flwr_datasets_version=version("flwr-datasets"),
        partition_sizes=tuple(len(partition) for partition in partitions),
        partition_fingerprints=tuple(
            partition._fingerprint for partition in partitions
        ),
        membership_digests=tuple(
            _membership_digest(partition_ids) for partition_ids in sample_ids
        ),
        label_histograms=tuple(
            tuple(int(count) for count in histogram) for histogram in histograms
        ),
        label_histogram_digests=tuple(
            _histogram_digest(histogram) for histogram in histograms
        ),
        total_variation_distance=float(total_variation_distance),
    )


def test_real_cifar100_partitions_are_non_iid_and_repeatable() -> None:
    """Independent constructions must yield the same usable skewed partitions."""
    first = _load_snapshot()
    second = _load_snapshot()

    print("CIFAR100_PARTITION_SNAPSHOT=" + json.dumps(asdict(first), sort_keys=True))

    assert first == second
    assert first.dataset_fingerprint == EXPECTED_DATASET_FINGERPRINT
    assert first.datasets_version == EXPECTED_DATASETS_VERSION
    assert first.dataset_revision == EXPECTED_DATASET_REVISION
    assert first.flwr_datasets_version == EXPECTED_FLWR_DATASETS_VERSION
    assert first.partition_sizes == EXPECTED_PARTITION_SIZES
    assert first.partition_fingerprints == EXPECTED_PARTITION_FINGERPRINTS
    assert first.membership_digests == EXPECTED_MEMBERSHIP_DIGESTS
    assert first.label_histogram_digests == EXPECTED_LABEL_HISTOGRAM_DIGESTS
    assert first.label_histograms[0] != first.label_histograms[1]
    assert first.total_variation_distance >= MIN_TOTAL_VARIATION_DISTANCE
    assert np.isclose(
        first.total_variation_distance,
        EXPECTED_TOTAL_VARIATION_DISTANCE,
        rtol=0.0,
        atol=1e-15,
    )
