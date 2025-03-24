import os
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from torch import tensor
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

os.environ["FLWR_LOG_LEVEL"] = "DEBUG"
os.environ["GRPC_VERBOSITY"] = "DEBUG"
# os.environ["GRPC_TRACE"] = "all"


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: VGG, data_loaders: SimpleNamespace):
        self.model = model
        self.data_loaders = data_loaders

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        LOGGER.info("Retrieving model parameters")
        self.model.train(True)
        LOGGER.info("Set model to training mode")
        parameters: List[np.ndarray] = [
            value.cpu().numpy() for _, value in self.model.state_dict().items()
        ]
        LOGGER.debug(f"Got {len(parameters)} parameters")
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        LOGGER.info("Setting model parameters")
        self.model.train(True)
        LOGGER.info("Set model to training mode")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict: OrderedDict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict=state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters=parameters)
        LOGGER.debug("Starting client training")
        train(data_loaders=self.data_loaders, model=self.model)
        return self.get_parameters(config={}), len(self.data_loaders.train), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters=parameters)
        LOGGER.debug("Starting model test")
        loss, accuracy = test(data_test=self.data_loaders.test, model=self.model)
        return loss, len(self.data_loaders.test), {"accuracy": accuracy}


if __name__ == "__main__":
    LOGGER = configure_logging("append")
    configure_training(
        batch_size=8, epochs=1
    )  # set epochs to one, as the "epochs" are the rounds configured by the server
    _MODEL: VGG = get_model()
    _DATA_SETS: SimpleNamespace = configure_data_sets(
        get_dataset_complete(target_set="Original dataset")
    )
    _DATA_LOADERS = configure_data_loaders(data_sets=_DATA_SETS, workers=None)
    _CLIENT = FlowerClient(model=_MODEL, data_loaders=_DATA_LOADERS).to_client()
    fl.client.start_client(
        server_address="localhost:8080",
        client=_CLIENT,
        grpc_max_message_length=1024**3,
        max_wait_time=60.0,
    )
