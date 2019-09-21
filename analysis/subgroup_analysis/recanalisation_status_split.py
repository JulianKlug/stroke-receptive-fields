import numpy as np
import pandas as pd

def split_dataset(data_set, recanalisation_status_path):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, param = data_set
    recanalisation_status_df = pd.read_excel(recanalisation_status_path)

    recanalised_indexes = []
    non_recanalised_indexes = []
    unknown_status_indexes = []

    for df_idx, row in recanalisation_status_df.iterrows():
        index = np.where(ids == row['anonymised_id'])[0]
        if index.size == 0: continue
        index = index[0]
        if row['iat_recanalized'] == 1:
            recanalised_indexes.append(index)
        elif row['iat_recanalized'] == 0:
            non_recanalised_indexes.append(index)
        else:
            unknown_status_indexes.append(index)

    return recanalised_indexes, non_recanalised_indexes, unknown_status_indexes