import pandas as pd
import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from DataLoader import jetRNA_v4_DataLoader

def cnn_encoder_collate_fn(batch):
    """To pad each sequence in batch to same length.

    """
    xs, edges, ys, y_centers = zip(*batch)

    x_padded = pad_sequence(xs, batch_first=True, padding_value=-1.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=-1.0)

    y_centers_stacked = torch.stack(y_centers, dim=0)
    return x_padded, list(edges), y_padded, y_centers_stacked

class jetRNA_v4_DataModule(L.LightningDataModule):
    """
    """
    def __init__(
            self,
            path_to_train_sequences_csv,
            path_to_val_sequences_csv,
            path_to_test_sequences_csv,
            path_to_train_labels_csv,
            path_to_val_labels_csv,
            path_to_test_labels_csv,
            batch_size):
        super().__init__()
        self.path_to_train_sequences_csv = path_to_train_sequences_csv 
        self.path_to_train_labels_csv = path_to_train_labels_csv
        self.path_to_val_sequences_csv = path_to_val_sequences_csv
        self.path_to_val_labels_csv = path_to_val_labels_csv
        self.path_to_test_sequences_csv = path_to_test_sequences_csv
        self.path_to_test_labels_csv = path_to_test_labels_csv

        self.batch_size = batch_size

        self.train_sequences_df = None
        self.val_sequences_df = None
        self.test_sequences_df = None

        self.train_labels_df = None
        self.val_labels_df = None
        self.test_labels_df = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage):
        self.train_sequences_df = pd.read_csv(self.path_to_train_sequences_csv)
        self.val_sequences_df = pd.read_csv(self.path_to_val_sequences_csv)
        self.test_sequences_df = pd.read_csv(self.path_to_test_sequences_csv)

        self.train_labels_df = pd.read_csv(self.path_to_train_labels_csv)
        self.val_labels_df = pd.read_csv(self.path_to_val_labels_csv)
        self.test_labels_df = pd.read_csv(self.path_to_test_labels_csv)

        self.train_data = jetRNA_v4_DataLoader(self.train_sequences_df, self.train_labels_df)
        self.val_data = jetRNA_v4_DataLoader(self.val_sequences_df, self.val_labels_df)
        self.test_data = jetRNA_v4_DataLoader(self.test_sequences_df, self.test_labels_df)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=15, persistent_workers=True, shuffle=True, collate_fn=cnn_encoder_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=15, persistent_workers=True, collate_fn=cnn_encoder_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=cnn_encoder_collate_fn)


