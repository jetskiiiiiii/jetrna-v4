import torch
import pandas as pd
#from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset

from pre_process_data.get_single_representation_from_sequence import get_single_representation_from_sequence
from pre_process_data.get_pair_representation_from_sequence import get_pair_representation_from_sequence
from pre_process_data.get_target_labels_from_df import get_target_labels_from_df

class jetRNA_v4_LazyDataset(Dataset):
    """
    Loads data from csv.
    """
    def __init__(self, sequences_df: pd.DataFrame, labels_df: pd.DataFrame):
        super().__init__()
        self.sequences_df = sequences_df
        self.labels_df = labels_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        sequences_row = self.sequences_df.iloc[idx]
        sequence_name = sequences_row["target_id"]

        sequence = sequences_row["sequence"]
        labels = self.labels_df[self.labels_df["ID"].str.startswith(sequence_name)]

        #x = get_pair_representation_from_sequence(sequence)
        x = get_single_representation_from_sequence(sequence)
        y = get_target_labels_from_df(labels)

        sequence_length = len(y)

        edge_list = []
        if sequence_length > 1:
            for j in range(sequence_length - 1):
                edge_list.extend([[j, j + 1], [j + 1, j]])

        # For single representation
        # Ensures x shape is N x 27 (for number of features in single embedding),
        # y shape is N x 3, and edge_list length is N
        print(len(x), sequence_length)
        assert len(x) == sequence_length
        assert len(x[0]) == 27 and len(y[0]) == 3

        # For pair representation
        # x.shape == 8xNxN
        #assert len(x[0]) == sequence_length
        #assert len(x[0, 0, :]) == 8 and len(y[0]) == 3

        assert len(edge_list) == (sequence_length - 1) * 2

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Normalizing target values
        y = y / 900

        pd.DataFrame(x.cpu().numpy()).to_csv(f"./dataset/training_data/{sequence_name}_seq.csv")
        pd.DataFrame(y.cpu().numpy()).to_csv(f"./dataset/training_data/{sequence_name}_lab.csv")
        pd.DataFrame(edge_index.cpu().numpy()).to_csv(f"./dataset/training_data/{sequence_name}_edge.csv")
        
        return x, edge_index, y
     
