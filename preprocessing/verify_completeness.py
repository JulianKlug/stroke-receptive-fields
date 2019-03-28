import os
import numpy as np

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'extracted_test2')

def verify_completeness(data_dir):
    subject_folders = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for folder in subject_folders:
        folder_dir = os.path.join(data_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]

        hasT2 = 0
        hasMTT = 0
        hasTmax = 0
        hasCBF = 0
        hasCBV = 0
        hasSPC = 0
        hasVOI = 0

        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            studies = [f for f in os.listdir(modality_dir) if f.endswith(".nii")]
            for study in studies:
                if 't2_tse_tra' in study: hasT2 = 1
                if 'MTT' in study: hasMTT = 1
                if 'Tmax' in study: hasTmax = 1
                if 'CBF' in study: hasCBF = 1
                if 'CBV' in study: hasCBV = 1
                if 'SPC_301mm_Std' in study: hasSPC = 1

        # lesion files should be in subject dir
        nii_files = [f for f in os.listdir(folder_dir) if f.endswith(".nii")]
        for file in nii_files:
            if 'VOI' in file: hasVOI = 1

        conditions = [hasT2, hasMTT, hasTmax, hasCBF, hasCBV, hasSPC, hasVOI]
        condition_names = ['hasT2', 'hasMTT', 'hasTmax', 'hasCBF', 'hasCBV', 'hasSPC', 'hasVOI']

        if np.all(conditions):
            print(folder, 'is complete.')
        else:
            missing_files = np.array(condition_names)[np.where(np.array(conditions) < 1)[0]]
            print(folder, 'is missing', missing_files)
            return False
    return True

verify_completeness(data_dir)
