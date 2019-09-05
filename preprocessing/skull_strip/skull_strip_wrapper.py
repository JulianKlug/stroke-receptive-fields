import os, subprocess
from csf_segmentation import createCSFMask

main_dir = '/Users/julian/temp/prepro'
data_dir = os.path.join(main_dir, '')
skull_strip_path = os.path.join(os.getcwd(), 'skull_strip.sh')
high_definition = True
print(skull_strip_path)

SPC_start = 'SPC'
CSF_mask_name = 'CSF_mask'
if high_definition:
    SPC_start = 'reor_SPC'
    CSF_mask_name = 'hd_CSF_mask'


subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir,o))]

for subject in subjects:
    subject_dir = os.path.join(data_dir, subject)
    modalities = [o for o in os.listdir(subject_dir)
                    if os.path.isdir(os.path.join(subject_dir,o))]

    for modality in modalities:
        modality_dir = os.path.join(subject_dir, modality)
        studies = [o for o in os.listdir(modality_dir)
                        if o.endswith(".nii")]

        for study in studies:
            study_path = os.path.join(modality_dir, study)
            if modality.startswith('pCT') & study.startswith(SPC_start):
                print(study)

                subprocess.run([skull_strip_path, '-i', study], cwd = modality_dir)

                createCSFMask(modality_dir, 'betted_' + study + '.gz', output_name = CSF_mask_name)
