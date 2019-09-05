import os, sys
import nibabel as nib

def verify_TRACE_4D(data_dir):
    '''
    Verify if the TRACE image is 4D
    :param data_dir:
    :return:
    '''
    subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, o))]
    not_verified = []

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, o))]

        verified_TRACE = False
        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                       if o.endswith(".nii")]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if 'TRACE' in study and study.endswith('.nii'):
                    img = nib.load(study_path)
                    data = img.get_data()
                    if len(data.shape) == 4:
                        verified_TRACE = True
                        break
        if verified_TRACE:
            print(subject, 'verified.')
        else:
            print(subject, 'is missing correct TRACE.')
            not_verified.append(subject)

    print('Not verified TRACE:', not_verified)




if __name__ == '__main__':
    path = sys.argv[1]
    verify_TRACE_4D(path)
