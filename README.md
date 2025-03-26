# Federated learning with transfer learning - project showcase
The goal of this project is creating a working proof of concept for federated learning (i.e., decentralized learning).

## Federated aspect

In centralized (traditional) learning, the training data is sent to one central server. On the hand, this leads to a loss of control over the training data by handing them over to a third party.
On the other hand, in centralized learning vertical scaling (adding more powerful hardware to the server) is much easier to achieve than horizontal scaling (adding more compute nodes).

Federated learning aims to close the gap in those shortcomings. In federated learning the data remains on the edge/client devices, and the model is trained locally. After a training round, the local model is sent by the client to a server, where it gets aggregated with models by other clients. This allows privacy-friendly training, as the central server is never receiving the training data, as well as distributed training and training on the edge with noncontinuous internet connection.

The federated aspect is realized with the [Flower](https://flower.ai/) framework.

## CNN aspect

The base showcase is a transfer learning project in PyTorch, where a VGG19 CNN model is used as a base to retrain just the last output layer of the model to a new classification problem.

## Dataset

The dataset for this project consists of 4800 images of 6 vegetables. The goal is to retrain the VGG19 model to recognize those 6 vegetables accurately.

![Dataset examples](/readme_img/dataset.png)

## GPU support
Though usage of the [DirectML](https://github.com/microsoft/directml) interface, this model training can be accelerated even with AMD GPUs on Windows and inside the [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

CUDA is also supposed to work, due to a lack of access to Nvidia hardware, this could not be tested.

# Setup guide
## Dataset preparation

- Retrieve the [vegetables dataset](https://www.kaggle.com/datasets/jocelyndumlao/a-dataset-for-vegetable-identification) from Kaggle.
- Unzip the data into a folder called `archive` in the root folder of this project. The archive folder should contain at least the folder `Original dataset` as the project is tested on this. The directory structure should look like this:

```
[project root]
    +---app
    +---util
    +---archive
        +---Original dataset
            +---Beans
            +---Eggplant
            +---Ladies Finger
            +---Onion
            +---Pointed gourd
            +---Potato
```
You can test the project to your liking also with the `Augmented dataset` folder.

## Python venv setup (one time)
This project is tested under Windows 11 and Python 3.11. It's not fully compatible with newer versions of Python at the current time (Q1 2025).
```
cd .\federated-cnn-transfer-learning
python3 -m venv .venv
.\.venv\Scripts\activate
python3 -m pip install -vv -r .\requirements.txt
```
The code is supposed to work also under Linux as no Windows-specific functionalities or file paths were used, although this is neither guaranteed nor tested. 

## Venv activation
Before every execution of the project, your terminal session needs to activate the Python venv with all the required dependencies:
```
cd .\federated-cnn-transfer-learning
.\.venv\Scripts\activate
```
## Enabling code quality controls (for development only)
If you are making significant changes to the code, feel free to activate the pre-commit hook. It ensures aspects of code quality through:
- Static type checking with [mypy](https://mypy-lang.org/).
- Linting and formatting with [ruff](https://docs.astral.sh/ruff/).
- Consistent usage of syntax with [pyupgrade](https://github.com/asottile/pyupgrade).
```
cd .\federated-cnn-transfer-learning
.\.venv\Scripts\activate
pre-commit install
```

## Project structure and execution
The Python files inside the project root folder depicts one of the following aspects:
 - Centralized (regular) training:  `main_non_federated.py`
 - Federated training:              `server.py` and `client.py`
- Model performance evaluation:     `evaluate_model.py`

## Central config file
All aspects of the project are configured through the `config.env` residing inside the project root. Make sure to configure it to your needs before execution.

# Script execution
```
cd .\federated-cnn-transfer-learning
python3 -m app.[script_name (without extension)]
```
e.g: `python3 -m app.evaluate_model` or `python3 -m app.server`

# Model Files
Any training creates local files of the trained model.
- In centralized training, for every improved epoch performance, the created file name is: `veggie-net-XXXXXXXX_XXXXXX-X.pth`
- In federated training, the server creates for every round a file of the pattern: `model_federated_round_X.pth` 

# Federated setup
![Logical minimal Setup](/readme_img/logical_setup.png)

At least two physical machines are required for the intended setup:
- One machine with the role of server and a client, and the other machine is the other client.
- It is also possible to run the server on a separate machine.
- The data partitioning supports the addition of further clients.

## Federated Configuration
1. Start server with `python3 -m app.server`
2. Inside the `config.env`:
    - Set continuous client IDs.
    - Set the IP of the server for each client.
3. Start each client with  `python3 -m app.client`
    - The server needs to be available before any client goes online.
    - If any errors happen (e.g., Out of Memory), the whole setup needs to be restarted in the specific order.
    - If testing with just one client, the number of clients needs to be set to 1 for the server inside the `config.env`.