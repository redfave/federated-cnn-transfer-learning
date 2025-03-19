from collections import OrderedDict
import os
os.environ["FLWR_LOG_LEVEL"] = "DEBUG"
os.environ["GRPC_VERBOSITY"] = "DEBUG"
os.environ["GRPC_TRACE"] = "all"

import flwr as fl
from flwr.common.typing import NDArrays
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


def set_parameters(model: VGG, parameters: NDArrays) -> VGG:
    params: dict = zip(model.state_dict().keys(), parameters)
    state_dict: OrderedDict = OrderedDict({k: tensor(v) for k, v in params})
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [value.cpu().numpy() for _, value in _MODEL.state_dict().items()]

    def fit(self, parameters, config):
        global _MODEL
        _MODEL = set_parameters(model=_MODEL, parameters=parameters)
        _MODEL = train(data_loaders=_DATA_LOADERS, model=_MODEL)
        return self.get_parameters({}), len(_DATA_LOADERS.train), {}

    def evaluate(self, parameters, config):
        global _MODEL
        _MODEL = set_parameters(model=_MODEL, parameters=parameters)
        loss, accuracy = test(data_test=_DATA_LOADERS.test, model=_MODEL)
        return loss, len(_DATA_LOADERS.test), {"accuracy": accuracy}

if __name__ == "__main__":
    configure_training(batch_size=16, epochs=2)
    _MODEL: VGG = get_model()
    _DATA_SETS = configure_data_sets(get_dataset_complete())
    _DATA_LOADERS = configure_data_loaders(data_sets=_DATA_SETS, workers=4)
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient().to_client())
