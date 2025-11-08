import numpy as np

def get_pair_representation_from_sequence(sequence: str) -> np.ndarray:
    """Convert nucleotide sequence into pair representation.
    Features/channels:
        - same entity/base (1 channel)
        - invalid (1 channel)
        - unpaired (1 channel)
        - potential pairings (6 channels)

    Args:
        - sequence (str): Nucleotide sequence.

    Returns:
        - pair_representation (nd.array): Array representating potential relationships between bases. 

    """
    length = len(sequence)

    # Channels to represent features in pair representation
    pair_representation = np.zeros((8, length, length), dtype=np.int8)

    # Make sure sequence is all caps
    sequence = sequence.upper()

    for i in range(length):
        pair_representation[1, [i, i], [i, i]] = 1
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
                pair_representation[0, [i, j], [j, i]] = 1

            # Potential pairings
            elif sequence[i] == "C" and sequence[j] == "G":
                pair_representation[2, [i, j], [j, i]] = 1
            elif sequence[i] == "G" and sequence[j] == "C":
                pair_representation[3, [i, j], [j, i]] = 1
            elif sequence[i] == "A" and sequence[j] == "U":
                pair_representation[4, [i, j], [j, i]] = 1
            elif sequence[i] == "U" and sequence[j] == "A":
                pair_representation[5, [i, j], [j, i]] = 1
            elif sequence[i] == "G" and sequence[j] == "U":
                pair_representation[6, [i, j], [j, i]] = 1
            elif sequence[i] == "U" and sequence[j] == "G":
                pair_representation[7, [i, j], [j, i]] = 1

    return pair_representation
