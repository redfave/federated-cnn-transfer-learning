# Use this script for running the standalone/centralized PyTorch training
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.models import VGG

from transfer_learning import (
    configure_data_loaders,
    configure_data_sets,
    configure_training,
    get_dataset_complete,
    get_model,
    test,
    train,
)
from util.logging import configure_logging

if __name__ == "__main__":
    LOGGER = configure_logging("overwrite")
    configure_training(batch_size=16, epochs=2)
    model: VGG = get_model()
    full_dataset: ImageFolder = get_dataset_complete(target_set="Original dataset")
    # mini_set = Subset(
    # dataset=full_dataset, indices=list(range(0, len(full_dataset), 16))
    # )  # test for quicker execution with a smaller subset
    # data_sets = configure_data_sets(mini_set)
    data_sets = configure_data_sets(full_dataset)
    data_loaders = configure_data_loaders(data_sets=data_sets, workers=4)
    model = train(data_loaders, model)
    test(data_loaders.test, model)
