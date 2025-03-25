#!/usr/bin/env python


# Use this script for running the standalone/centralized PyTorch training
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.models import VGG

from app.transfer_learning import (
    configure_data_loaders,
    configure_data_sets,
    configure_training,
    get_dataset_complete,
    get_model,
    test,
    train,
)
from app.util.env_loader import APP_CONFIG
from app.util.log_config import configure_logging

if __name__ == "__main__":
    LOGGER = configure_logging(mode="overwrite", role="standalone")
    configure_training(
        batch_size=int(APP_CONFIG.get("BATCH_SIZE")),
        epochs=int(APP_CONFIG.get("EPOCHS")),
    )
    model: VGG = get_model()
    full_dataset: ImageFolder = get_dataset_complete(
        target_set=APP_CONFIG.get("TARGET_SET")
    )
    # mini_set = Subset(
    # dataset=full_dataset, indices=list(range(0, len(full_dataset), 16))
    # )  # test for quicker execution with a smaller subset
    # data_sets = configure_data_sets(mini_set)
    data_sets = configure_data_sets(full_dataset)
    data_loaders = configure_data_loaders(
        data_sets=data_sets,
        workers=int(APP_CONFIG.get("WORKERS_NUM"))
        if APP_CONFIG.get("WORKERS_NUM") is not None
        else None,
    )
    model = train(data_loaders, model)
    test(data_loaders.test, model)
