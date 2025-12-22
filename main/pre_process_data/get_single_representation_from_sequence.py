import itertools
import traceback
import numpy as np
import pandas as pd
from typing import List, Tuple

def get_single_representation_from_sequence(sequence: str) -> np.ndarray:
    """Convert nucleotide sequence into single representation.
    Features (current total: 27):
        - one-hot encoding (4 channels)
            - C, G, A, U
        - di-nucleotide frequencies (16 channels)
            - CC, CG, CA, CU, GC, GG, GA, GU, AC, AG, AA, AU, UC, UG, UA, UU
        - local content (4 channels)
            - 11 G/C, 21 G/C, 11 A/U, 11 A/U
        - potential inverted (3 channels)
            - 4 min, 5 min, 6 min

    Args:
        - sequence (str): Nucleotide sequence.

    Returns:
        - single_representation (nd.array): Array representating potential relationships between bases. 

    """
    length = len(sequence)
    num_channels = 28
    single_representation = np.zeros((length, num_channels))

    window_11_gc = []
    window_21_gc = []
    window_11_au = []
    window_21_au = []

    group_4, group_5, group_6 = get_potential_inversions_from_sequence(sequence)
    set_4, set_5, set_6 = set(group_4), set(group_5), set(group_6)

    for i in range(length):
        base = sequence[i]

        # Get one-hot sequence
        if base == "C":
            single_representation[i][0] = 1
        elif base == "G":
            single_representation[i][1] = 1
        elif base == "A":
            single_representation[i][2] = 1
        elif base == "U":
            single_representation[i][3] = 1
        else:
            single_representation[i][4] = 1     # To handle "-" and "N"

        # Get di-nucleotide
        if i < length - 1:     # Prevent indexing from going out of range
            next_base = sequence[i+1]
            bases = "GCAU"
            if base in bases and next_base in bases:
                idx_map = {"C": 0, "G": 1, "A": 2, "U": 3}
                di_idx = 5 + (idx_map[base] * 4) + idx_map[next_base]
                single_representation[i][di_idx] = 1

        # Get local g/c, a/u content
        ## We want context windows of size 11 and 21,
        
        # Sliding window
        for i in range(length):
            # Window boundaries
            w11_start, w11_end = max(0, i-5), min(length, i+6)
            w21_start, w21_end = max(0, i-10), min(length, i+11)

            seq_11 = sequence[w11_start:w11_end]
            seq_21 = sequence[w21_start:w21_end]

            # Calculating percentages
            single_representation[i, 21] = sum(1 for b in seq_11 if b in "GC") / len(seq_11)
            single_representation[i, 22] = sum(1 for b in seq_21 if b in "GC") / len(seq_21)
            single_representation[i, 23] = sum(1 for b in seq_11 if b in "AU") / len(seq_11)
            single_representation[i, 24] = sum(1 for b in seq_21 if b in "AU") / len(seq_21)

        # Get potential inversions
        if i in set_4: single_representation[i][25] = 1 
        if i in set_5: single_representation[i][26] = 1 
        if i in set_6: single_representation[i][27] = 1

    return single_representation

def get_potential_inversions_from_sequence(sequence: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Algorithm explanation:
    - Get all possible groupings of desired length (4, 5, or 6)
        - At the same time, invert the grouping and note index (for reference)
        - At the same time, check if current grouping matches with any previous inverted grouping
            - Account for GU, UG pairs
            - Account for grouping and inversion must not overlap
            - If yes, take note of indexes
                - Skip all indexes that have been noted (will result in non-numerical order of final lists)

    Args:
        - sequence (str)

    Returns: 
        - inversions_4
        - inversions_5
        - inversions_6

        * Note: indices not returned in numeric order. This is because
        if there are back to back inversions, only indices that haven't
        been added are then added.
    """
    length = len(sequence)
    group_4_inverted, group_5_inverted, group_6_inverted = {}, {}, {}
    # If index in these lists, skip
    group_4, group_5, group_6 = [], [], []
    group_4_potential_indices, group_5_potential_indices, group_6_potential_indices = [], [], []    # Keep track of indices involved in inverted groups
    inversions = {"C": ["G"],
                    "G": ["C", "U"],
                    "A": ["U"],
                    "U": ["A", "G"],
                    "-": [],
                    "N": [],
                    "X": []}
    for i in range(length-3): # Stop at 4th from end as that is the last possible grouping
        # Creating 4 base grouping
        # Every time grouping is created,
        # create the inversion, too.
        group = sequence[i:i+4]
        inverted_group = [inversions.get(base, []) for base in group]

        # If group exists, combinations have been included - only need the first instace of inversions
        if group not in group_4:
            inversion_combinations = list(itertools.product(*inverted_group))   # Get all combinations of inversions
            group_4_inverted[i] = ["".join(combination) for combination in inversion_combinations]
            group_4.append(group)
        
        # Check if grouping is an inversion of a previous grouping
        for i_inverted, key in enumerate(group_4_inverted):
            if (i_inverted + 3 < i) and (group in group_4_inverted[key]):
                #print(i_inverted, group_4[i_inverted], i, group)
                # Only need to include indices once
                for index in range(4):
                    if i_inverted+index not in group_4_potential_indices:
                        group_4_potential_indices.append(i_inverted+index)    # Append both indices of original and inverted group
                for index in range(4):
                    if i+index not in group_4_potential_indices:
                        group_4_potential_indices.append(i+index)
          
        if (i < length-4):
            group = sequence[i:i+5]
            inverted_group = [inversions[base] for base in group]

            # If group exists, combinations have been included - only need the first instace of inversions
            if group not in group_5:
                inversion_combinations = list(itertools.product(*inverted_group))   # Get all combinations of inversions
                group_5_inverted[i] = ["".join(combination) for combination in inversion_combinations]
                group_5.append(group)
            
            # Check if grouping is an inversion of a previous grouping
            for i_inverted, key in enumerate(group_5_inverted):
                if (i_inverted + 4 < i) and (group in group_5_inverted[key]):
                    #print(i_inverted, group_5[i_inverted], i, group)
                    # Only need to include indices once
                    for index in range(5):
                        if i_inverted+index not in group_5_potential_indices:
                            group_5_potential_indices.append(i_inverted+index)    # Append both indices of original and inverted group
                    for index in range(5):
                        if i+index not in group_5_potential_indices:
                            group_5_potential_indices.append(i+index)
          
        if (i < length-5):
            group = sequence[i:i+6]
            inverted_group = [inversions[base] for base in group]

            # If group exists, combinations have been included - only need the first instace of inversions
            if group not in group_6:
                inversion_combinations = list(itertools.product(*inverted_group))   # Get all combinations of inversions
                group_6_inverted[i] = ["".join(combination) for combination in inversion_combinations]
                group_6.append(group)
          
            # Check if grouping is an inversion of a previous grouping
            for i_inverted, key in enumerate(group_6_inverted):
                if (i_inverted + 5 < i) and (group in group_6_inverted[key]):
                    #print(i_inverted, group_6[i_inverted], i, group)
                    # Only need to include indices once
                    for index in range(6):
                        if i_inverted+index not in group_6_potential_indices:
                            group_6_potential_indices.append(i_inverted+index)    # Append both indices of original and inverted group
                    for index in range(6):
                        if i+index not in group_6_potential_indices:
                            group_6_potential_indices.append(i+index)


    return group_4_potential_indices, group_5_potential_indices, group_6_potential_indices

if __name__ == "__main__":
    x = get_single_representation_from_sequence("CGAUACGCUAUGCGCUAU")
    print(x.shape)
    #print(x[:, 26])
    #print(get_potential_inversions_from_sequence("CGAUACGCUAUGCGCUAU"))
    np.savetxt("example_single_embdding.csv", x, delimiter=",", fmt="%.2f")
    
    #test_sequences_path = "./dataset/stanford/test_sequences.csv"
    #val_sequences_path = "./dataset/stanford/validation_sequences.csv"
    #train_sequences_path = "./dataset/stanford/train_sequences.csv"

    #test_df = pd.read_csv(test_sequences_path)
    #val_df = pd.read_csv(val_sequences_path)
    #train_df = pd.read_csv(train_sequences_path)

    #for idx, row in val_df.iterrows():
    #    target_id = row["target_id"]
    #    sequence = row["sequence"]

    #    try:
    #        representation = get_single_representation_from_sequence(sequence)

    #        np.save(f"./dataset/stanford/single_representation/val/{target_id}.npy", representation)
    #    except Exception as e:
    #        print(f"Error processing {target_id}")
    #        print(f"Error Type: {type(e).__name__}")
    #        print(f"Error Message: {e}")
    #        traceback.print_exc()
    #
