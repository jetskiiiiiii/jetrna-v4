import pandas as pd

def get_data_from_ct_file(filepath):
    """ Grabs 1) name, 2) length of sequence, 3) one hot sequence, and 4) pairings in DBN form.

    References:
        - https://rna.urmc.rochester.edu/Text/File_Formats.html

    Args:
        - filepath (str): Path to CT file

    Returns:
        - name (str): Name of molecule
        - length_of_sequence (int): Length of sequence
        - one_hot_sequence (nd.array): Array of bases in one hot sequence.
        - pairings (str): Pairings in DBN form.

    """
    name, length_of_sequence, one_hot_sequence, pairings = None, None, None, None

    with open(filepath, "r") as file:
        # Get name, length from first line
        first_line = file.readline().split()
        name = first_line[-1]
        length_of_sequence = int(first_line[0])

        # Arrays to store each nucleotide and its pairing
        sequence = [] 
        pairing = []

        next(file) # Starts for loop from 2nd line
        for line_number, line in enumerate(file, 1):
            parts = line.split()
            
            if len(parts) is not 6:
                return ValueError(f"Line {line_number} does not contain 6 values.")
            # Format of each column based on official CT documentation
            is_index_1_base = True if parts[1] in ["A", "C", "G", "U", "X"] else False
            is_index_2_index_minus_1 = True if parts[2] == int(parts[0]) - 1 else False
            is_index_3_index_plus_1 = True if parts[2] == int(parts[0]) + 1 else False
            is_index_4_within_pairing_range = True if int(parts[4]) in range(0, length_of_sequence+1) else False # Not strict check against pairing rules
            is_index_5_natural_numbering = True if parts[5] == parts[0] else False

            if (is_index_1_base
                    and is_index_2_index_minus_1
                    and is_index_3_index_plus_1
                    and is_index_4_within_pairing_range
                    and is_index_5_natural_numbering):
                sequence.append(parts[1])
                pairing.append(parts[4]) 

        for base in sequence:

        
        return name, length_of_sequence,  
