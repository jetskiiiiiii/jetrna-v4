import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class jetRNA_v4_Dataset(Dataset):
    """
    Loads data from csv.
    """
    def __init__(self, sequences_df: pd.DataFrame, labels_df: pd.DataFrame, embeddings_dir: str):
        super().__init__()
        # Sequences
        self.sequences_df = sequences_df[["target_id"]]
        self.embeddings_dir = embeddings_dir

        # Labels
        labels_subset = labels_df[["ID", "x_1", "y_1", "z_1"]].copy()
        labels_subset["base_ID"] = labels_subset["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))

        # Group coords into dict of lists
        self.coord_map = {}
        for tid, group in labels_subset.groupby("base_ID"):
            self.coord_map[tid] = group[["x_1", "y_1", "z_1"]].values

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        row = self.sequences_df.iloc[idx]
        target_id = row["target_id"]

        x = np.load(f"{self.embeddings_dir}/{target_id}.npy")
        y = self.coord_map.get(target_id)

        if y is None:
            raise ValueError(f"No coordinates exist for {target_id}")

        sequence_length = len(y)

        edge_list = []
        for j in range(sequence_length - 1):
            edge_list.extend([[j, j + 1], [j + 1, j]])

        assert len(x) == sequence_length
        assert len(edge_list) == (sequence_length - 1) * 2

        # For pair representation
        # x.shape == 8xNxN
        #assert len(x[0]) == sequence_length
        #assert len(x[0, 0, :]) == 8 and len(y[0]) == 3

        x_tensor = torch.tensor(x, dtype=torch.float)

        y_tensor = torch.tensor(y, dtype=torch.float) 
        valid_coords_mask = (y_tensor > -1e15).all(dim=-1)

        y_safe = y_tensor.clone()
        y_safe[~valid_coords_mask] = 0.0

        # Zero-centering coordinates
        y_center = y_safe[valid_coords_mask].mean(dim=0) if valid_coords_mask.any() else torch.zeros(3)
        y_centered = (y_tensor - y_center) / 900.0         # Normalizing target values

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        assert edge_index.max() < x.shape[0]

        return x_tensor, edge_index, y_centered, y_center, valid_coords_mask, target_id   # Returning center to move coords back
     
class PredictDataset(Dataset):
    def __init__(self, sequences_df, embeddings_dir):
        self.sequences = sequences_df[['target_id']]
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        target_id = self.sequences.iloc[idx]['target_id']
        
        # Load your pre-calculated (N, 27) embedding
        x = np.load(f"{self.embeddings_dir}/{target_id}.npy")
        
        # Build edges as usual
        seq_len = len(x)
        edge_list = []
        for j in range(seq_len - 1):
            edge_list.extend([[j, j + 1], [j + 1, j]])
        
        # Since we have no labels, we return Zeros or None
        # We return a dummy y and y_center so the collate_fn doesn't break
        y_dummy = torch.zeros((seq_len, 3))
        center_dummy = torch.zeros(3)
        
        mask_dummy = torch.ones(seq_len, dtype=torch.bool)
        
        return torch.FloatTensor(x), torch.LongTensor(edge_list).t(), y_dummy, center_dummy, mask_dummy, target_id

