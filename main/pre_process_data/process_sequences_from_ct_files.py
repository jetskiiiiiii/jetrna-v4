import os
import pandas as pd
import sourmash as sm

from get_data_from_ct_file import get_data_from_ct_file
from datasets import load_dataset

ds = load_dataset("multimolecule/rnastralign")
print(type(ds))

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
        #print(root)
        try:
            sequence, pairing = get_data_from_ct_file(os.path.join(rnastralign_path, filename))
            new_row = pd.DataFrame({"name": [root], "source": ["RNAStrAlign"], "sequence": [sequence], "pairing": [pairing]})
            ct_sequences_and_pairings_df = pd.concat([ct_sequences_and_pairings_df, new_row], ignore_index=True) 
        except:
            continue

    for filename in archiveII_filenames:
        root, _ = os.path.splitext(filename)
        #print(root)
        try:
            sequence, pairing = get_data_from_ct_file(os.path.join(archiveII_path, filename))
            new_row = pd.DataFrame({"name": [root], "source": ["ArchiveII"], "sequence": [sequence], "pairing": [pairing]})
            ct_sequences_and_pairings_df = pd.concat([ct_sequences_and_pairings_df, new_row], ignore_index=True) 
        except:
            continue

    ct_sequences_and_pairings_df.to_parquet("./dataset/rnastralign_archiveII_sequences_pairings.parquet", engine="pyarrow", compression="snappy")


def remove_redundancies(path_to_parquet: str):
    """
    Using Nucleotide substitution matrices from NCBI database
    from biotite. Potential bottleneck because the matrix was
    calculated based on nucleotide frequency of protein data.

    To avoid having to find sequence identity for each pair of
    sequences, we assume similarly named files contain similar
    sequences. So, we first compare linearly through the
    directory.

    """
    df = pd.read_parquet(path_to_parquet, engine="pyarrow")
    df = df.sort_values(by="name")
    print(df.columns)

    removed_seqs = 0
    seq_a, seq_b, seq_to_compare = None, None, None

    # Range should be total_sequences - 1
    #for i in range(len(df) - 1):
    #for i in range(2):
    #    # Determining what seq_a should be
    #    if seq_to_compare is None:
    #        seq_a = df["sequence"][i]
    #        print(df["name"][i])
    #    else:
    #        seq_a = seq_to_compare
    #    seq_b = df["sequence"][i+1]
    #    print(df["name"][i+1])

    seq_a = df[df["name"] == "X77123"]["sequence"].item()
    seq_b = df[df["name"] == "X77122"]["sequence"].item()

    #seq_a = "GCUGCUGCUGCUGCUGCUGCUGCUGC"
    #seq_b = "UGCUCCUAGUACGAGAGGACCGGAGUG"
    print(seq_a)
    print("/n")
    print(seq_b)

    # Since algorithm is supposed to be for protein sequences,
    # we subsitute U for T
    seq_a = "".join(
        ["T" if s == "U" else s for s in seq_a]
    )

    seq_b = "".join(
        ["T" if s == "U" else s for s in seq_b]
    )

    mh1 = sm.MinHash(n=0, ksize=21, scaled=1000)
    mh2 = sm.MinHash(n=0, ksize=21, scaled=1000)
    mh1.add_sequence(seq_a)
    mh2.add_sequence(seq_b)

    sequence_identity = round(mh1.jaccard(mh2), 5)
    print(f"Sequence identity: {sequence_identity}")

    # If previous sequences are redundant, continue using seq_a to compare
    if sequence_identity > 0.75:
        len_a, len_b = len(seq_a), len(seq_b)
        shortest_seq = seq_a if len_a < len_b else seq_b

        removed_seqs += 1
        seq_to_compare = seq_a
    else:
        seq_to_compare = None

    print(removed_seqs)
    return removed_seqs

if __name__ == "__main__":
    #create_main_parquet()
    remove_redundancies("./dataset/rnastralign_archiveII_sequences_pairings.parquet")

