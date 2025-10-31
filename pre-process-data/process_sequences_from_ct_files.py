import os
import pandas as pd
from get_data_from_ct_file import get_data_from_ct_file

def create_main_parquet():
    """
    Creates a parquet (for efficient memory) containing sequences and pairings from both RNAStrAlign and ArchiveII.
    No redundancy checking or train/val/test/splitting done.
    """
    archiveII_path = "./dataset/ct/archiveII"
    rnastralign_path = "./dataset/ct/rnastralign"

    rnastralign_filenames = os.listdir(rnastralign_path)
    archiveII_filenames = os.listdir(archiveII_path)

    ct_sequences_and_pairings_df = pd.DataFrame({"name": [], "source": [], "sequence": [], "pairing": []})

    for filename in rnastralign_filenames:
        root, _ = os.path.splitext(filename)
        try:
            sequence, pairing = get_data_from_ct_file(os.path.join(rnastralign_path, filename))
        except:
            continue
        new_row = pd.DataFrame({"name": root, "source": "RNAStrAlign", "sequence": sequence, "pairing": pairing})
        ct_sequences_and_pairings_df = pd.concat([ct_sequences_and_pairings_df, new_row], ignore_index=True) 

    for filename in archiveII_filenames:
        root, _ = os.path.splitext(filename)
        try:
            sequence, pairing = get_data_from_ct_file(os.path.join(archiveII_path, filename))
        except:
            continue
        new_row = pd.DataFrame({"name": root, "source": "ArchiveII", "sequence": sequence, "pairing": pairing})
        ct_sequences_and_pairings_df = pd.concat([ct_sequences_and_pairings_df, new_row], ignore_index=True) 

    ct_sequences_and_pairings_df.to_parquet("./dataset/rnastralign_archiveII_sequences_pairings.parquet", engine="pyarrow", compression="snappy")

if __name__ == "__main__":
    create_main_parquet()
