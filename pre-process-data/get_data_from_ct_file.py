import itertools
import pandas as pd

def get_data_from_ct_file(filepath):
    """ Grabs 1) name, 2) sequence, and 3) pairings. 

    References:
        - https://rna.urmc.rochester.edu/Text/File_Formats.html

    Args:
        - filepath (str): Path to CT file

    Returns:
        - sequence (str): Nucleotide base sequence
        - pairings (str): Pairings in DBN form

    """
    with open(filepath, "r") as file:
        length = 0
        # Get length of sequence
        for line_number, line in enumerate(file):
            parts = line.split()
            if parts[0].isdigit():
                length = int(parts[0])
                break

        # Get sequence and pairings
        sequence = ""
        pairing = []
        schema = [int, str, int, int, int, int] # Schema of what each line should be

        for line_number, line in enumerate(file):
            parts = line.split()
            parts_processed = []
            
            # Should have 6 parts, according to docs
            if len(parts) != len(schema):
                continue

            for i, (value, expected_type) in enumerate(zip(parts, schema)):
                if expected_type == int:
                    parts_processed.append(int(value))
                else:
                    parts_processed.append(value)

            # Format of each column based on official CT documentation
            is_index_1_base = True if parts_processed[1] in ["A", "C", "G", "U", "X"] else False
            is_index_2_index_minus_1 = True if parts_processed[2] == parts_processed[0] - 1 else False
            is_index_3_index_plus_1 = True if parts_processed[3] == parts_processed[0] + 1 else False
            is_index_4_within_pairing_range = True if parts_processed[4] > -1 and parts_processed[4] < length + 1 else False # Not strict check against pairing rules
            is_index_5_natural_numbering = True if parts_processed[5] == parts_processed[0] else False

            if (is_index_1_base
                    and is_index_2_index_minus_1
                    and is_index_3_index_plus_1
                    and is_index_4_within_pairing_range
                    and is_index_5_natural_numbering):
                sequence += parts_processed[1]
                pairing.append(parts_processed[4]) 

        return sequence, pairing

if __name__ == "__main__":
    sequence, pairing = get_data_from_ct_file("./dataset/bpRNA_CRW_1897.ct")
    print(sequence, pairing)
