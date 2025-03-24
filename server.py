import os

from util.logging import configure_logging

os.environ["FLWR_LOG_LEVEL"] = "DEBUG"
os.environ["GRPC_VERBOSITY"] = "DEBUG"

from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

_CLIENTS_NUM = 2  # The setup fails silently when there is only one available client (as 2 clients are the default value)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return (
        {"accuracy": sum(accuracies) / sum(examples)}
        if sum(examples) > 0
        else {"accuracy": 0.0}
    )


if __name__ == "__main__":
    LOGGER = configure_logging("overwrite")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=_CLIENTS_NUM,
            min_fit_clients=_CLIENTS_NUM,
            min_evaluate_clients=_CLIENTS_NUM,
        ),
        grpc_max_message_length=1024**3,  # 1GB, should be enough for the model
    )
