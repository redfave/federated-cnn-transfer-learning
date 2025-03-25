#!/usr/bin/env python

# Use this script to evaluate a trained model
from types import SimpleNamespace

from torchvision.datasets import ImageFolder
from torchvision.models import VGG

from app.transfer_learning import (
    configure_data_loaders,
    configure_data_sets,
    configure_training,
    get_dataset_complete,
    restore_model,
    show_example_images,
    test,
)
from app.util.env_loader import APP_CONFIG


def image_generator(model: VGG):
    while True:
        show_example_images(data_loaders.test, model)
        yield None


if __name__ == "__main__":
    configure_training(batch_size=8, epochs=0)
    full_dataset: ImageFolder = get_dataset_complete(
        target_set=APP_CONFIG.get("TARGET_SET")
    )
    data_sets: SimpleNamespace = configure_data_sets(full_dataset)
    data_loaders: SimpleNamespace = configure_data_loaders(
        data_sets=data_sets,
        workers=int(APP_CONFIG.get("WORKERS_NUM"))
        if APP_CONFIG.get("WORKERS_NUM") is not None
        else None,
    )
    model: VGG = restore_model(APP_CONFIG.get("MODEL_ITERATION"))

    if APP_CONFIG.get("RUN_TEST") == "True":
        test(data_loaders.test, model)

    image_gen = image_generator(model=model)
    try:
        while True:
            next(image_gen)
            input("Press Enter to view next image (or Ctrl+C to exit)...")
    except KeyboardInterrupt:
        print("\nImage viewing stopped by user")
