import os, sys
sys.path.insert(0, '../')
import pandas as pd
import numpy as np
import preprocessing.image_name_config as image_name_config

data_dir = '/Volumes/stroke_hdd1/stroke_db/2016/part2'
spc_ct_sequences = image_name_config.spc_ct_sequences

subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir,o))]

state_columns = ['patient', 'hasPCT', 'hasSPC', 'hasCompleteMRI', 'hasT2', 'hasDWI', 'lesionDrawn', 'hasAll', 'hasUnknown']
imaging_state_df = pd.DataFrame(columns=state_columns)

for subject in subjects:
    subject_dir = os.path.join(data_dir, subject)
    hasPCT = 0
    hasSPC = 0
    hasUnknown = 0
    hasDWI = 0
    hasT2 = 0
    lesionDrawn = 0
    hasAll = 0

    if os.path.isdir(subject_dir):
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]

        lesionDrawn = int(np.any(['VOI' in f or 'lesion' in f or 'Lesion' in f for f in os.listdir(subject_dir)]))

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)

            studies = os.listdir(modality_dir)

            if modality == 'study':
                hasUnknown = 1

            for study in studies:
                if study.startswith('VPCT_Perfusion_4D') or 'RAPID' in study:
                    hasPCT = 1
                if study in spc_ct_sequences:
                    hasSPC = 1
                if 'T2' in study or 't2' in study:
                    hasT2 = 1
                if 'ADC' in study or 'TRACE' in study or 'adc' in study \
                    or 'trace' in study or 'DWI' in study or 'dwi' in study:
                    hasDWI = 1
                if 'VOI' in study or 'lesion' in study or 'Lesion' in study:
                    lesionDrawn = 1
    if hasPCT and hasSPC and hasT2 and hasDWI and lesionDrawn:
        hasAll = 1
    imaging_state_df = imaging_state_df.append(
        pd.DataFrame([[subject, hasPCT, hasSPC, (hasT2 and hasDWI), hasT2, hasDWI, lesionDrawn, hasAll, hasUnknown]],
        columns = state_columns),
        ignore_index=True)

imaging_state_df.to_excel(os.path.join(data_dir, 'imaging_state.xlsx'))
