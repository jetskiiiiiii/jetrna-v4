import itertools
import numpy as np
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
    num_channels = 27
    single_representation = np.zeros((length, num_channels))

    window_11_gc = []
    window_21_gc = []
    window_11_au = []
    window_21_au = []

    for i in range(length):
        # Get one-hot sequence
        if sequence[i] == "C":
            single_representation[i][0] = 1
        elif sequence[i] == "G":
            single_representation[i][1] = 1
        elif sequence[i] == "A":
            single_representation[i][2] = 1
        elif sequence[i] == "U":
            single_representation[i][3] = 1

        # Get di-nucleotide
        if i == length - 1:     # Prevent indexing from going out of range
            pass
        elif sequence[i] == "C" and sequence[i+1] == "C":
            single_representation[i][4] = 1
        elif sequence[i] == "C" and sequence[i+1] == "G":
            single_representation[i][5] = 1
        elif sequence[i] == "C" and sequence[i+1] == "A":
            single_representation[i][6] = 1
        elif sequence[i] == "C" and sequence[i+1] == "U":
            single_representation[i][7] = 1
        elif sequence[i] == "G" and sequence[i+1] == "C":
            single_representation[i][8] = 1
        elif sequence[i] == "G" and sequence[i+1] == "G":
            single_representation[i][9] = 1
        elif sequence[i] == "G" and sequence[i+1] == "A":
            single_representation[i][10] = 1
        elif sequence[i] == "G" and sequence[i+1] == "U":
            single_representation[i][11] = 1
        elif sequence[i] == "A" and sequence[i+1] == "C":
            single_representation[i][12] = 1
        elif sequence[i] == "A" and sequence[i+1] == "G":
            single_representation[i][13] = 1
        elif sequence[i] == "A" and sequence[i+1] == "A":
            single_representation[i][14] = 1
        elif sequence[i] == "A" and sequence[i+1] == "U":
            single_representation[i][15] = 1
        elif sequence[i] == "U" and sequence[i+1] == "C":
            single_representation[i][16] = 1
        elif sequence[i] == "U" and sequence[i+1] == "G":
            single_representation[i][17] = 1
        elif sequence[i] == "U" and sequence[i+1] == "A":
            single_representation[i][18] = 1
        elif sequence[i] == "U" and sequence[i+1] == "U":
            single_representation[i][19] = 1

        # Get local g/c, a/u content
        ## We want context windows of size 11 and 21,
        ## so we only need to iterate through 21 as the 11-window
        ## will always be inside of the 21 window.
        for j in range(21):
            window_idx = i-10+j
            if window_idx > 0:  # Indexing with negative number will grab from end of sequence
                try:            # Try block to capture indexing error when idx > length
                    if sequence[window_idx] in ["G", "C"]:      # If base is G/C
                        window_21_gc[i].append(sequence[window_idx])
                        if j > 4 and j < 16:    # If idx is inside of 11 window
                            window_11_gc[i].append(sequence[window_idx])
                    elif sequence[window_idx] in ["A", "U"]:    # If base is A/U
                        window_21_au[i].append(sequence[window_idx])
                        if j > 4 and j < 16:
                            window_11_gc[i].append(sequence[window_idx])
                except IndexError:
                    pass

        # To fit local contet into one channel, we get the average content in window
        single_representation[i][20] = len(window_11_gc)/length
        single_representation[i][21] = len(window_21_gc)/length
        single_representation[i][22] = len(window_11_au)/length
        single_representation[i][23] = len(window_21_au)/length

        # Get potential inversions
        group_4, group_5, group_6 = get_potential_inversions_from_sequence(sequence)
        if i in group_4:
            single_representation[i][24] = 1
            group_4.remove(i)
        else:
            single_representation[i][24] = 0

        if i in group_5:
            single_representation[i][25] = 1
            group_5.remove(i)
        else:
            single_representation[i][25] = 0

        if i in group_6:
            single_representation[i][26] = 1
            group_6.remove(i)
        else:
            single_representation[i][26] = 0


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
                    "U": ["A", "G"]}
    for i in range(length-3): # Stop at 4th from end as that is the last possible grouping
        # Creating 4 base grouping
        # Every time grouping is created,
        # create the inversion, too.
        group = sequence[i:i+4]
        inverted_group = [inversions[base] for base in group]

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
    #print(x.shape)
    #print(x[:, 26])
    #print(get_potential_inversions_from_sequence("CGAUACGCUAUGCGCUAU"))
