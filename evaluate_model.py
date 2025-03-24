# Use this script to evaluate a trained model
from types import SimpleNamespace

from torchvision.datasets import ImageFolder
from torchvision.models import VGG

from transfer_learning import (
    configure_data_loaders,
    configure_data_sets,
    configure_training,
    get_dataset_complete,
    restore_model,
    show_example_images,
    test,
)

# set the model file to load here
_MODEL_NAME: str = "veggie-net-20250323_212224-1.pth"


def image_generator(model: VGG):
    while True:
        show_example_images(data_loaders.test, model)
        yield None


if __name__ == "__main__":
    configure_training(batch_size=8, epochs=0)
    full_dataset: ImageFolder = get_dataset_complete(target_set="Original dataset")
    data_sets: SimpleNamespace = configure_data_sets(full_dataset)
    data_loaders: SimpleNamespace = configure_data_loaders(
        data_sets=data_sets, workers=1
    )
    model: VGG = restore_model(_MODEL_NAME)
    # test(data_loaders.test, model)

    image_gen = image_generator(model=model)
    try:
        while True:
            next(image_gen)
            input("Press Enter to view next image (or Ctrl+C to exit)...")
    except KeyboardInterrupt:
        print("\nImage viewing stopped by user")
