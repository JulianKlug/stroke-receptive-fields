import os
import brain_extraction

main_dir = '/Users/julian/temp/extraction_bv40/trial1'
data_dir = os.path.join(main_dir, '')

spc_start = 'SPC_301mm'
# Find default Angio file, no MIP projection
angio_start = 'Angio_CT_075'

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

        if modality.startswith('pCT'):
            spc_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(spc_start) and i.endswith('.nii')]
            angio_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(angio_start) and i.endswith('.nii')]

            if len(spc_files) != 1:
                raise Exception('No SPC file found / or collision', subject, spc_files)
            if len(angio_files) != 1:
                raise Exception('No Angio file found / or collision', subject, angio_files)

            print('Extracting for', subject)
            brain_extraction.align_FOV(angio_files[0], spc_files[0], modality_dir)
            brain_extraction.extract_brain(angio_files[0], spc_files[0], modality_dir)
