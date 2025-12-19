import torch
import lightning as L
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class sampleGraphDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.sample_train, self.sample_val = create_sample_dataset(80), create_sample_dataset(20)

        if stage == "test":
            self.sample_test = create_sample_dataset(20)

        if stage == "predict":
            self.sample_predict = create_sample_dataset(20)

    def train_dataloader(self):
        return DataLoader(self.sample_train, batch_size = 8)

    def val_dataloader(self):
        return DataLoader(self.sample_val, batch_size = 8)

    def test_dataloader(self):
        return DataLoader(self.sample_test, batch_size = 8)

    def predict_dataloader(self):
        return DataLoader(self.sample_predict, batch_size = 8)

def create_sample_graph(N: int):
    feature_dim = 5
    output_dim = 3      # Needs to be 3 for x, y, z

    # Creates N nodes with random features
    x = torch.rand((N, feature_dim), dtype=torch.float)

    # Create random connectivity pattern
    # Creating N/2 edges
    source_nodes = torch.arange(0, N, 2).repeat(2)
    target_nodes = torch.arange(1, N, 2).repeat(2)
    
    # Make edges bidirectional
    edge_index = torch.stack([
        torch.cat([source_nodes, target_nodes]),
        torch.cat([target_nodes, source_nodes])
    ], dim=0)

    # Create sample target labels
    y_true = torch.rand((N, output_dim), dtype=torch.float) * 10

    return Data(x=x, edge_index=edge_index, y=y_true)

def create_sample_dataset(num_graphs: int):
    min_nodes = 5
    max_nodes = 16

    nodes_per_graph = torch.randint(min_nodes, max_nodes, (num_graphs,))

    return [
            create_sample_graph(nodes_per_graph[i].item()) 
            for i in range(num_graphs)
        ]
