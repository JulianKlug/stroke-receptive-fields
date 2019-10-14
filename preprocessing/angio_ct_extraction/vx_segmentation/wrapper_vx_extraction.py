import os
import subprocess

extract_vx_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'extract_vx.sh')


main_dir = '/Users/julian/stroke_research/brain_and_donuts/data/multi_subj/extracted_angio_data'
data_dir = os.path.join(main_dir, '')

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
            angio_files = [i for i in os.listdir(modality_dir)
                                if os.path.isfile(os.path.join(modality_dir, i))
                                    and i.startswith('wbetted_' + angio_start) and i.endswith('.nii')]
            if len(angio_files) != 1:
                raise Exception('No Angio file found / or collision', subject, angio_files)

            print('Extracting Vessels for', subject)
            subprocess.run([extract_vx_path, '-i', angio_files[0]], cwd = modality_dir)
