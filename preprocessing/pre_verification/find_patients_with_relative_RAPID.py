import os

main_dir = '/Users/julian/stroke_research/data/working_data/'
data_dir = os.path.join(main_dir, '')

def find_relative_RAPID_output(data_dir):
    '''
    Verify if patient with perfusion data have relative instead of absolute RAPID outputs
    '''
    subject_folders = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for folder in subject_folders:
        folder_dir = os.path.join(data_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o))]

            hasRelativeRAPID = 0
            for study in studies:
                if 'rCBF' in study or 'rCBV' in study:
                    hasRelativeRAPID = 1
                    print(folder, 'has relative RAPID values for', study)

    return

find_relative_RAPID_output(data_dir)
