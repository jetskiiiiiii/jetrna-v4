import torch
import pandas as pd
#from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset

from pre_process_data.get_single_representation_from_sequence import get_single_representation_from_sequence
from pre_process_data.get_pair_representation_from_sequence import get_pair_representation_from_sequence
from pre_process_data.get_target_labels_from_df import get_target_labels_from_df

class jetRNA_v4_DataLoader(Dataset):
    """
    Loads data from csv.
    """
    def __init__(self, sequences_df: pd.DataFrame, labels_df: pd.DataFrame):
        super().__init__()
        # Sequences
        self.sequences_df = sequences_df[["target_id", "sequence"]]

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

        sequence = row["sequence"]

        #x = get_pair_representation_from_sequence(sequence)
        x = get_single_representation_from_sequence(sequence)
        y = self.coord_map.get(target_id)

        if y is None:
            raise ValueError(f"No coordinates exist for {target_id}")

        sequence_length = len(y)

        edge_list = []
        for j in range(sequence_length - 1):
            edge_list.extend([[j, j + 1], [j + 1, j]])

        # For single representation
        # Ensures x shape is N x 27 (for number of features in single embedding),
        # y shape is N x 3, and edge_list length is N
        assert len(x) == sequence_length
        assert len(x[0]) == 27 and len(y[0]) == 3

        # For pair representation
        # x.shape == 8xNxN
        #assert len(x[0]) == sequence_length
        #assert len(x[0, 0, :]) == 8 and len(y[0]) == 3

        assert len(edge_list) == (sequence_length - 1) * 2

        x_tensor = torch.tensor(x, dtype=torch.float)

        y_tensor = torch.tensor(y, dtype=torch.float) 
        # Zero-centering coordinates
        y_center = y_tensor.mean(dim=0)
        y_centered = (y_tensor - y_center) / 900.0         # Normalizing target values

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        #pd.DataFrame(x.cpu().numpy()).to_csv(f"./dataset/training_data/{sequence_name}_seq.csv")
        #pd.DataFrame(y.cpu().numpy()).to_csv(f"./dataset/training_data/{sequence_name}_lab.csv")
        #pd.DataFrame(edge_index.cpu().numpy()).to_csv(f"./dataset/training_data/{sequence_name}_edge.csv")
        
        return x_tensor, edge_index, y_centered, y_center   # Returning center to move coords back
     
