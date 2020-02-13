import numpy as np
import pandas as pd

def split_dataset(data_set, iat_status_path):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, param = data_set
    iat_status_df = pd.read_excel(iat_status_path)

    IAT_indexes = []
    IVT_indexes = []
    unknown_status_indexes = []

    for df_idx, row in iat_status_df.iterrows():
        index = np.where(ids == row['anonymised_id'])[0]
        if index.size == 0: continue
        index = index[0]
        if row['treat_iat'] == "yes":
            IAT_indexes.append(index)
        elif row['treat_iat'] == "no":
            IVT_indexes.append(index)
        else:
            unknown_status_indexes.append(index)

    return IAT_indexes, IVT_indexes, unknown_status_indexes