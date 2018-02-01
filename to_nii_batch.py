import os
import subprocess

mount = '/run/media/jk/Elements'
main_dir = os.path.join(mount, 'MASTER/')
data_dir = os.path.join(main_dir, 'working_data')
output_dir = os.path.join(main_dir, 'extracted_data/')
dcm2niix_path = '/home/jk/master/dcm2niix/build/bin/dcm2niix'

subjects = os.listdir(data_dir)

for subject in subjects:
    subject_dir = os.path.join(data_dir, subject)
    modalities = [o for o in os.listdir(subject_dir)
                    if os.path.isdir(os.path.join(subject_dir,o))]

    for modality in modalities:
        modality_dir = os.path.join(subject_dir, modality)
        studies = [o for o in os.listdir(modality_dir)
                        if os.path.isdir(os.path.join(modality_dir,o))]


        for study in studies:
            study_dir = os.path.join(modality_dir, study)
            study_output_dir = os.path.join(output_dir, subject, modality, study)
            if not os.path.exists(study_output_dir):
                os.makedirs(study_output_dir)
            subprocess.run([dcm2niix_path, '-m', 'y', '-o', study_output_dir, study_dir], cwd = modality_dir)
