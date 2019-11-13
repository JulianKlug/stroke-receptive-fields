import os
import subprocess
from hessian_vessels import segment_hessian_vessels

extract_vx_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'extract_vx.sh')


main_dir = '/Users/julian/temp/extraction_bv40/trial1'
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
                                    and i.startswith('betted_' + angio_start) and i.endswith('.nii.gz')]
            if len(angio_files) != 1:
                raise Exception('No Angio file found / or collision', subject, angio_files)

            print('Extracting Vessels for', subject)

            # Newer hessian tubular vessel filter
            segment_hessian_vessels(os.path.join(modality_dir, angio_files[0]), os.path.join(modality_dir, 'extracted_' + angio_files[0]))

            # Old thresholding extraction process
            # subprocess.run([extract_vx_path, '-i', angio_files[0]], cwd = modality_dir)
