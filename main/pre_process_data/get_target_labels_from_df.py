import pandas as pd
from typing import List

def get_target_labels_from_df(df) -> List[List[int]]:
    y_targets = []

    for row in df.itertuples(index=False):
        y_targets.append([row.x_1, row.y_1, row.z_1])

    return y_targets

