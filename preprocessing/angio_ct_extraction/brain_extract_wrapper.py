import os
import subprocess
from brain_extraction import extract_brain

main_dir = '/Users/julian/master/brain_and_donuts/data/multi_subj'
data_dir = os.path.join(main_dir, 'extracted_test2')

spc_start = 'SPC_301mm'
# Find default Angio file, no MIP projection
angio_start = 'DE_Angio_CT_075'

subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir,o))]

for subject in subjects:
    subject_dir = os.path.join(data_dir, subject)
    modalities = [o for o in os.listdir(subject_dir)
                    if os.path.isdir(os.path.join(subject_dir,o))]

    for modality in modalities:
        modality_dir = os.path.join(subject_dir, modality)
        studies = [o for o in os.listdir(modality_dir)
                        if os.path.isfile(os.path.join(modality_dir,o))]

        if modality.startswith('CT'):
            spc_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(spc_start) and i.endswith('.nii')]
            angio_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(angio_start) and i.endswith('.nii')]

            if len(spc_files) != 1:
                raise Exception('No SPC file found / or collision', subject, spc_files)
            if len(angio_files) != 1:
                raise Exception('No Angio file found / or collision', subject, angio_files)

            print('Extracting for', subject)
            extract_brain(angio_files[0], spc_files[0], modality_dir)
