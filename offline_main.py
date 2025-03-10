from torch.utils.data import Subset
from torchvision.models import VGG

from transfer_learning import (
    configure_data_loaders,
    configure_data_sets,
    configure_training,
    get_dataset_complete,
    get_model,
    restore_model,
    test,
    train,
)

if __name__ == "__main__":
    configure_training(batch_size=16, epochs=2)
    model: VGG = get_model()
    full_dataset = get_dataset_complete()
    mini_set = Subset(
        dataset=full_dataset, indices=list(range(0, len(full_dataset), 16))
    )  # test for quicker execution with a smaller subset
    data_sets = configure_data_sets(mini_set)

    data_loaders = configure_data_loaders(data_sets=data_sets, workers=4)
    model = train(data_loaders, model)
    # model: VGG = restore_model("veggie-net-20250312_123349-2.pth")
    test(data_loaders.test, model)
