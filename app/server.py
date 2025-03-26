#!/usr/bin/env python

import os
from collections import OrderedDict

import numpy as np
import torch
from flwr.common import FitRes
from torchvision.models import VGG

from app.transfer_learning import get_model
from app.util.env_loader import APP_CONFIG
from app.util.log_config import configure_logging

os.environ["FLWR_LOG_LEVEL"] = "DEBUG"
os.environ["GRPC_VERBOSITY"] = "DEBUG"

from typing import Any, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import Metrics


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, FitRes]],
        failures: list[tuple[Any, FitRes] | BaseException],
    ) -> tuple[Any | None, dict[str, Any]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            LOGGER.info(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            model: VGG = get_model()
            params_dict = zip(get_model().state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            torch.save(model.state_dict(), f"model_federated_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return (
        {"accuracy": sum(accuracies) / sum(examples)}
        if sum(examples) > 0
        else {"accuracy": 0.0}
    )


if __name__ == "__main__":
    LOGGER = configure_logging(mode="overwrite", role="server")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=int(APP_CONFIG.get("EPOCHS"))
        ),  # training rounds in federated learning are like epochs in centralized learning
        strategy=SaveModelStrategy(
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=int(APP_CONFIG.get("CLIENT_NUM")),
            min_fit_clients=int(APP_CONFIG.get("CLIENT_NUM")),
            min_evaluate_clients=int(APP_CONFIG.get("CLIENT_NUM")),
        ),
        grpc_max_message_length=1024**3,  # 1GB, should be enough for the model
    )
