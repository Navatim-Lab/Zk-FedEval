"""CNN task and deterministic non-IID CIFAR-100 data for the Flower app."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

DATASET_NAME = "uoft-cs/cifar100"
DATASET_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
NUM_PARTITIONS = 2
IMAGE_COLUMN = "img"
LABEL_COLUMN = "fine_label"
SAMPLE_ID_COLUMN = "sample_id"
NUM_CLASSES = 100
PARTITION_ALPHA = 0.5
PARTITION_MIN_SIZE = 10
PARTITION_SEED = 2026


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset
fds_config = None

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def create_partitioner(
    num_partitions: int,
    alpha: float,
    seed: int,
) -> DirichletPartitioner:
    """Create the deterministic fine-label-skew partitioner."""
    return DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by=LABEL_COLUMN,
        alpha=alpha,
        min_partition_size=PARTITION_MIN_SIZE,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )


def add_sample_ids(dataset: DatasetDict) -> DatasetDict:
    """Add stable row identifiers after the deterministic dataset shuffle."""
    train_split = dataset["train"]
    if SAMPLE_ID_COLUMN not in train_split.column_names:
        dataset["train"] = train_split.add_column(
            SAMPLE_ID_COLUMN,
            range(train_split.num_rows),
        )
    return dataset


def create_federated_dataset(
    num_partitions: int = NUM_PARTITIONS,
    partition_alpha: float = PARTITION_ALPHA,
    partition_seed: int = PARTITION_SEED,
    dataset_revision: str = DATASET_REVISION,
) -> FederatedDataset:
    """Build the revision-pinned, deterministically shuffled Flower dataset."""
    return FederatedDataset(
        dataset=DATASET_NAME,
        revision=dataset_revision,
        preprocessor=add_sample_ids,
        partitioners={
            "train": create_partitioner(
                num_partitions=num_partitions,
                alpha=partition_alpha,
                seed=partition_seed,
            )
        },
        shuffle=True,
        seed=partition_seed,
    )


def configure_determinism(seed: int) -> None:
    """Seed local RNGs and constrain PyTorch to deterministic CPU execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch[IMAGE_COLUMN] = [pytorch_transforms(image) for image in batch[IMAGE_COLUMN]]
    return batch


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partition_alpha: float = PARTITION_ALPHA,
    partition_seed: int = PARTITION_SEED,
):
    """Load one deterministic non-IID CIFAR-100 client partition."""
    global fds, fds_config
    requested_config = (num_partitions, partition_alpha, partition_seed)
    if fds is None or fds_config != requested_config:
        fds = create_federated_dataset(
            num_partitions=num_partitions,
            partition_alpha=partition_alpha,
            partition_seed=partition_seed,
        )
        fds_config = requested_config
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(
        test_size=0.2,
        seed=partition_seed,
    )
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    generator = torch.Generator().manual_seed(partition_seed + partition_id)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load the centralized CIFAR-100 test set and return a dataloader."""
    # Load entire test set
    test_dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split="test",
    )
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[IMAGE_COLUMN].to(device)
            labels = batch[LABEL_COLUMN].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[IMAGE_COLUMN].to(device)
            labels = batch[LABEL_COLUMN].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
