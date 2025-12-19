import numpy as np

def get_pair_representation_from_sequence(sequence: str) -> np.ndarray:
    """Convert nucleotide sequence into pair representation.
    Features/channels:
        - invalid (1 channel)
        - unpaired (1 channel)
        - potential pairings (6 channels)
        - same entity/base (1 channel)

    Args:
        - sequence (str): Nucleotide sequence.

    Returns:
        - pair_representation (nd.array): Array representating potential relationships between bases. 

    """
    length = len(sequence)

    # Channels to represent features in pair representation
    pair_representation = np.zeros((length, length, 8), dtype=np.int8)

    # Make sure sequence is all caps
    sequence = sequence.upper()

    for i in range(length):
        pair_representation[[i, i], [i, i], 1] = 1
        for j in range(i, length):
            # Invalid
            # Distance between base_2 and base_1 must be at least 3
            # A -> G, C
            # U -> C
            # G -> A
            # C -> A, U
            if (j - i  < 3
                or (sequence[i] == "A" and (sequence[j] == "G" or sequence[j] == "C"))
                or (sequence[i] == "U" and sequence[j] == "C")
                or (sequence[i] == "G" and sequence[j] == "A")
                or (sequence[i] == "C" and (sequence[j] == "A" or sequence[j] == "U"))
                    ):
                pair_representation[[i, j], [j, i], 0] = 1

            # Potential pairings
            elif sequence[i] == "C" and sequence[j] == "G":
                pair_representation[[i, j], [j, i], 2] = 1
            elif sequence[i] == "G" and sequence[j] == "C":
                pair_representation[[i, j], [j, i], 3] = 1
            elif sequence[i] == "A" and sequence[j] == "U":
                pair_representation[[i, j], [j, i], 4] = 1
            elif sequence[i] == "U" and sequence[j] == "A":
                pair_representation[[i, j], [j, i], 5] = 1
            elif sequence[i] == "G" and sequence[j] == "U":
                pair_representation[[i, j], [j, i], 6] = 1
            elif sequence[i] == "U" and sequence[j] == "G":
                pair_representation[[i, j], [j, i], 7] = 1

    return pair_representation

if __name__=="__main__":
    x = get_pair_representation_from_sequence("CGAUACGCUAUGCGCUAU")
    print(x)
