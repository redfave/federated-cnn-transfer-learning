import os
os.environ["FLWR_LOG_LEVEL"] = "DEBUG"
os.environ["GRPC_VERBOSITY"] = "DEBUG"
# os.environ["GRPC_TRACE"] = "all"

import subprocess
import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(),
)

# command = [
#     "flower-superlink",
#     "--insecure",
#     "--server-address", "0.0.0.0:8080",
#     "--num-rounds", "3",
#     "--strategy", "fedavg"
# ]
# subprocess.run(command)