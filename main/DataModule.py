import pandas as pd
import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from Dataset import jetRNA_v4_Dataset, PredictDataset

def collate_fn(batch):
    """To pad each sequence in batch to same length.

    """
    xs, edges, ys, y_centers, valid_coord_masks, id_list = zip(*batch)

    x_padded = pad_sequence(xs, batch_first=True, padding_value=-1.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=-1.0)
    y_padded = torch.nan_to_num(y_padded, nan=0.0, posinf=0.0, neginf=0.0)
    valid_coord_masks_batch = pad_sequence(valid_coord_masks, batch_first=True, padding_value=False)
    y_centers_stacked = torch.stack(y_centers, dim=0)

    # Offset edges for GCN batching
    batch_edges = []
    max_nodes = x_padded.shape[1]   # Length that sequences were padded to

    for i, edge_tensor in enumerate(edges):
        # Shift indices by batch * max_nodes
        offset = i * max_nodes
        batch_edges.append(edge_tensor + offset)

    combined_edges = torch.cat(batch_edges, dim=1)

    return x_padded, combined_edges, y_padded, y_centers_stacked, valid_coord_masks_batch, id_list

class jetRNA_v4_DataModule(L.LightningDataModule):
    """
    """
    def __init__(
            self,
            path_to_train_sequences_csv,
            path_to_train_embeddings,
            path_to_val_sequences_csv,
            path_to_val_embeddings,
            path_to_test_sequences_csv,
            path_to_test_embeddings,
            path_to_train_labels_csv,
            path_to_val_labels_csv,
            batch_size):
        super().__init__()
        self.train_sequences_df = pd.read_csv(path_to_train_sequences_csv)
        self.train_embeddings = path_to_train_embeddings
        self.train_labels_df = pd.read_csv(path_to_train_labels_csv)
        self.val_sequences_df = pd.read_csv(path_to_val_sequences_csv)
        self.val_embeddings = path_to_val_embeddings
        self.val_labels_df = pd.read_csv(path_to_val_labels_csv)
        self.test_sequences_df = pd.read_csv(path_to_test_sequences_csv)
        self.test_embeddings = path_to_test_embeddings

        self.batch_size = batch_size

    def setup(self, stage):
        self.train_data = jetRNA_v4_Dataset(self.train_sequences_df, self.train_labels_df, self.train_embeddings)
        self.val_data = jetRNA_v4_Dataset(self.val_sequences_df, self.val_labels_df, self.val_embeddings)
        self.test_data = PredictDataset(self.test_sequences_df, self.test_embeddings)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=15, persistent_workers=True, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=15, persistent_workers=True, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=collate_fn)

