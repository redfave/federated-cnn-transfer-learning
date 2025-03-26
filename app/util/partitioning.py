from torch.utils.data import Dataset, Subset

from app.util.env_loader import APP_CONFIG


def partition_dataset(source_dataset: Dataset, client_id: int) -> Subset:
    total_clients = int(APP_CONFIG.get("CLIENT_NUM"))
    idx_source_dataset = list(range(len(source_dataset)))
    idx_client = [idx for idx in idx_source_dataset if idx % total_clients == client_id]
    return Subset(source_dataset, idx_client)
