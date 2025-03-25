# ! pip install -r requirements.txt
# ! pre-commit install
# ! pre-commit run --all-files


# !jupyter nbconvert --to script .\VGG_transfer_learning.ipynb


import os
import random
from datetime import datetime
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import torch
import torch_directml
from ignite.metrics import Accuracy, Fbeta, MetricsLambda, Precision, Recall
from torch import device, nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.models import VGG, VGG19_Weights, vgg19
from torchvision.transforms import ToPILImage

from app.util.log_config import configure_logging


def disablePILDecompressionBombError() -> None:
    # Prevents PIL DecompressionBombError
    from PIL import Image, ImageFile

    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


# Needs to be here at top (before imports)
# as on Windows the function is otherwise non pickleable
# (Nasty to debug runtime errors incoming)
def worker_init_fn(worker_id: int) -> None:
    disablePILDecompressionBombError()
    logger = configure_logging("append")
    logger.info(f"Initialized worker {worker_id}")


# Setup logging before any torch module is imported
LOGGER = configure_logging(
    "append"
)  # append to existing log (relieas on log to be created in some calling module)

_BATCH_SIZE: int | None = None
_EPOCHS: int | None = None


def configure_training(
    batch_size: int = 8, epochs: int = 12, target_data_set: str = "Original dataset"
) -> None:
    global _BATCH_SIZE, _EPOCHS
    _BATCH_SIZE = batch_size
    _EPOCHS = epochs

    custom_seed = np.random.randint(2141403747)
    random.seed(custom_seed)  # apparently seed must be set at several places
    torch.manual_seed(custom_seed)


def _init_dataset_complete() -> ImageFolder:
    return ImageFolder(
        root=os.path.join(".", "archive", cast(str, _TARGET_SET)),
        # Img inference transformation piepeline: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
        transform=VGG19_Weights.IMAGENET1K_V1.transforms(),
    )


_DATASET_COMPLETE: ImageFolder | None = None

_TARGET_SET: Literal["Original dataset"] | None = (
    None  # target folder inside dataset (archive.zip)
)


def get_dataset_complete(
    target_set: Literal["Original dataset"] | None = None,
) -> ImageFolder:
    global _DATASET_COMPLETE
    if _DATASET_COMPLETE is None:
        global _TARGET_SET
        _TARGET_SET = target_set
        _DATASET_COMPLETE = _init_dataset_complete()

    return _DATASET_COMPLETE


_DEVICE: device | None = None


def _configure_device() -> torch.device:
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch_directml.device(torch_directml.default_device())
        if torch_directml.is_available()
        else torch.device("cpu")
    )
    global _DEVICE
    _DEVICE = device
    LOGGER.info(device)
    return device


def get_device() -> device:
    if _DEVICE is None:
        _configure_device()
    return _DEVICE


def configure_data_sets(source_dataset: Dataset) -> SimpleNamespace:
    return SimpleNamespace(
        **dict(
            zip(
                ["train", "validation", "test"],
                random_split(
                    dataset=source_dataset,
                    lengths=[
                        int(0.6 * len(source_dataset)),
                        int(0.3 * len(source_dataset)),
                        int(0.1 * len(source_dataset)),
                    ],
                ),
            )
        )
    )


def configure_data_loaders(
    data_sets: SimpleNamespace, workers: int | None = None
) -> SimpleNamespace:
    data_loader_kwargs = {
        "batch_size": _BATCH_SIZE,
        "persistent_workers": True if workers is None or workers >= 1 else False,
        "num_workers": max(2, ((os.cpu_count() or 0) // 2) - 1)
        if workers == None
        else workers,
        "shuffle": True,
        # "in_order": False, # Only introduced in PyTorch v2.6
        "drop_last": True,
        "worker_init_fn": worker_init_fn,
    }

    if "cuda" in get_device().type.lower():
        data_loader_kwargs.update(
            {
                "pin_memory": True,
                "pin_memory_device": get_device().type,
            }
        )

    return SimpleNamespace(
        train=DataLoader(data_sets.train, **data_loader_kwargs),
        validation=DataLoader(data_sets.validation, **data_loader_kwargs),
        test=DataLoader(data_sets.test, **data_loader_kwargs),
    )


_MODEL: VGG | None = None


def _configure_model() -> VGG:
    global _MODEL
    _MODEL = vgg19(
        weights=VGG19_Weights.DEFAULT
    )  # model initialization with pretrained default weights

    LOGGER.info("MODEL INIT INFO:")
    LOGGER.info(summary(_MODEL))  # use torchinfo for parameter info

    _MODEL.features.requires_grad_(False)  # freeze all the blocks of model
    _MODEL.avgpool.requires_grad_(False)
    _MODEL.classifier.requires_grad_(False)

    _MODEL.classifier[-1] = nn.Linear(
        in_features=4096, out_features=6, bias=True
    )  # replace the last classification layer
    # unfreeze new classification layer
    _MODEL.classifier[-1].requires_grad_(True)

    LOGGER.info("MODEL MODIFIED INFO:")
    LOGGER.info(summary(_MODEL))  # most params should be non trainable

    # send model with forzen layers (hopefully) to GPU
    _MODEL = _MODEL.to(get_device())
    return _MODEL


def get_model() -> VGG:
    if _MODEL is None:
        _configure_model()
    return _MODEL


_CRITERION: nn.CrossEntropyLoss | None = None


def _get_criterion() -> nn.CrossEntropyLoss:
    global _CRITERION
    if _CRITERION is None:
        _CRITERION = nn.CrossEntropyLoss()
    return _CRITERION


_PRECISION: Precision | None = None
_RECALL: Recall | None = None
_ACCURACY: Accuracy | None = None
_F1: MetricsLambda | None = None


def _configure_metrics(
    device: device,
) -> tuple[Precision, Recall, Accuracy, MetricsLambda]:
    precision = Precision(device=device)
    recall = Recall(device=device)
    accuracy = Accuracy(device=device)
    f1 = Fbeta(beta=1.0, precision=precision, recall=recall, device=device)
    return precision, recall, accuracy, f1


if any(metric is None for metric in (_PRECISION, _RECALL, _ACCURACY, _F1)):
    _PRECISION, _RECALL, _ACCURACY, _F1 = _configure_metrics(get_device())


def _get_precision_metric() -> Precision:
    return cast(Precision, _PRECISION)


def _get_recall_metric() -> Recall:
    return cast(Recall, _RECALL)


def _get_accuracy_metric() -> Accuracy:
    return cast(Accuracy, _ACCURACY)


def _get_f1_metric() -> MetricsLambda:
    return cast(MetricsLambda, _F1)


def reset_metrics():
    _get_precision_metric().reset()
    _get_recall_metric().reset()
    _get_accuracy_metric().reset()
    _get_f1_metric().reset()
    _SUMMARY_WRITER.flush()


def _train_one_epoch(
    model: VGG,
    epoch: int,
    optimizer: optim,
    scheduler: optim.lr_scheduler,
    data_loaders: SimpleNamespace,
    summary_writer: SummaryWriter,
) -> float:
    LOGGER.info(f"Starting training epoch: {epoch}")
    running_loss_training = 0.0
    last_loss = 0.0

    for i, data in enumerate(data_loaders.train):
        LOGGER.debug(f"iteration: {i}\tepoch-train: {epoch}")
        inputs, labels = data
        inputs, labels = inputs.to(get_device()), labels.to(get_device())
        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(inputs)  # Make predictions for this batch

        # Compute the loss and its gradients
        loss = _get_criterion()(outputs, labels)
        loss.backward()  # backpropagation

        optimizer.step()  # Adjust learning weights

        running_loss_training += loss.item()

        if i % cast(int, _BATCH_SIZE) == 0:
            last_loss = running_loss_training / (i + 1)  # loss per mini-batch
            tb_x = epoch * len(data_loaders.train) + i + 1

            summary_writer.add_scalar("Loss/train", last_loss, tb_x)

    scheduler.step()  # Adjust learning rate
    summary_writer.flush()

    return last_loss


_TIMESTAMP: str | None = None
if _TIMESTAMP is None:
    _TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

_SUMMARY_WRITER: SummaryWriter | None = None
if _SUMMARY_WRITER is None:
    _SUMMARY_WRITER = SummaryWriter(log_dir=f"runs/veggie_trainer_{_TIMESTAMP}")


def _get_summary_writer() -> SummaryWriter:
    return cast(SummaryWriter, _SUMMARY_WRITER)


def train(data_loaders: SimpleNamespace, model: VGG) -> VGG:
    model_path_best_epoch = ""
    optimizer = optim.SGD(
        filter(
            lambda model_parameter: model_parameter.requires_grad, model.parameters()
        ),  # optimize just the non frozen parameters
        lr=0.001,  # start learning rate
        momentum=0.9,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )  # adjust learning rate per epoch

    best_loss_validation = float("inf")

    for epoch in range(1, cast(int, _EPOCHS) + 1):
        model.train(True)
        avg_loss_train = _train_one_epoch(
            model, epoch, optimizer, scheduler, data_loaders, _get_summary_writer()
        )
        running_loss_test = 0.0
        avg_loss_validation = 0.0

        model.eval()  # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization

        LOGGER.info(f"Starting validation epoch: {epoch}")
        with (
            torch.no_grad()
        ):  # Disable gradient computation and reduce memory consumption.
            for i, data in enumerate(data_loaders.validation):
                LOGGER.debug(f"iteration: {i}\tepoch-validation: {epoch}")
                inputs, labels = data
                inputs, labels = inputs.to(get_device()), labels.to(get_device())
                outputs = model(inputs)
                loss = _get_criterion()(outputs, labels)
                running_loss_test += loss.item()

                # Log the running loss averaged per batch
                avg_loss_validation = running_loss_test / (i + 1)

                _get_precision_metric().update((outputs, labels))
                _get_recall_metric().update((outputs, labels))
                _get_accuracy_metric().update((outputs, labels))
                _get_f1_metric().update((outputs, labels))

        LOGGER.info(f"LOSS train {avg_loss_train} vs. validation {avg_loss_validation}")

        _get_summary_writer().add_scalar(
            "Loss/validation", avg_loss_validation, epoch + 1
        )
        _get_summary_writer().add_scalar(
            "Precision/validation",
            _get_precision_metric().compute().mean().item(),
            epoch + 1,
        )
        _get_summary_writer().add_scalar(
            "Recall/validation", _get_recall_metric().compute().mean().item(), epoch + 1
        )
        _get_summary_writer().add_scalar(
            "Accuracy/validation", _get_accuracy_metric().compute(), epoch + 1
        )
        _get_summary_writer().add_scalar(
            "F1/validation", _get_f1_metric().compute(), epoch + 1
        )

        reset_metrics()

        # Track best performance, and save the model's state
        if avg_loss_validation < best_loss_validation:
            best_loss_validation = avg_loss_validation

            model_path_best_epoch = os.path.join(
                ".", f"veggie-net-{_TIMESTAMP}-{epoch}.pth"
            )
            torch.save(model.state_dict(), model_path_best_epoch)

            model_path_previous_epoch = os.path.join(
                ".", f"veggie-net-{_TIMESTAMP}-{epoch - 1}.pth"
            )
            if os.path.exists(model_path_previous_epoch):
                os.remove(model_path_previous_epoch)

    LOGGER.info("Finished Training")
    # load best performing model (might be from a previous epoch)
    model.load_state_dict(torch.load(model_path_best_epoch))
    model = model.to(get_device())
    return model


def test(data_test: DataLoader, model: VGG):
    LOGGER.info("Running test")
    running_loss_test = 0.0
    avg_loss_test = 0.0

    with torch.no_grad():  # Disable gradient computation and reduce memory consumption.
        for i, data in enumerate(data_test):
            LOGGER.debug(rf"iteration: {i} of final test")
            inputs, labels = data
            inputs, labels = inputs.to(get_device()), labels.to(get_device())
            outputs = model(inputs)
            loss = _get_criterion()(outputs, labels)
            running_loss_test += loss.item()

            avg_loss_test = cast(float, running_loss_test / (i + 1))

            _get_precision_metric().update((outputs, labels))
            _get_recall_metric().update((outputs, labels))
            _get_accuracy_metric().update((outputs, labels))
            _get_f1_metric().update((outputs, labels))

        _get_summary_writer().add_scalar("Loss/test", avg_loss_test, _EPOCHS)
        _get_summary_writer().add_scalar(
            "Precision/test", _get_precision_metric().compute().mean().item(), _EPOCHS
        )
        _get_summary_writer().add_scalar(
            "Recall/test", _get_recall_metric().compute().mean().item(), _EPOCHS
        )

        avg_accuracy_test = _get_accuracy_metric().compute()
        _get_summary_writer().add_scalar("Accuracy/test", avg_accuracy_test, _EPOCHS)
        _get_summary_writer().add_scalar("F1/test", _get_f1_metric().compute(), _EPOCHS)

        reset_metrics()

    LOGGER.info(f"LOSS test {avg_loss_test}")
    LOGGER.info("Test complete")
    return avg_loss_test, avg_accuracy_test


# Load a previously trained model manually
def restore_model(model_iteration: str) -> VGG:
    global _MODEL
    _MODEL = _configure_model()
    _MODEL.load_state_dict(
        torch.load(os.path.join(".", model_iteration), weights_only=False)
    )
    _MODEL = _MODEL.to(device=get_device())
    return _MODEL


# manual (visual) inspection
def show_example_images(data_loader_test: DataLoader, model: VGG) -> None:
    disablePILDecompressionBombError()

    idx_to_class: dict[int, str] = {
        v: k for k, v in get_dataset_complete().class_to_idx.items()
    }

    random_subset_idx = random.randint(0, len(data_loader_test.dataset) - 1)

    sample, label = data_loader_test.dataset[random_subset_idx]
    sample_batch = sample.unsqueeze(0).to(get_device())

    from PIL import Image

    full_dataset = data_loader_test.dataset.dataset
    full_dataset_idx = data_loader_test.dataset.indices[random_subset_idx]

    sample_path = full_dataset.samples[full_dataset_idx][0]

    Image.open(sample_path).show()
    LOGGER.info(
        f"Prediction: {idx_to_class[model(sample_batch).argmax().item()]} | Actual Label: {idx_to_class[label]}"
    )


# usage: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# ! tensorboard --logdir=./runs/
# go to http://localhost:6006
