import os
import pandas as pd
from mask_volume import mask_volume


def mask_volume_batch(dir):
    subjects = [o for o in os.listdir(dir)
               if os.path.isdir(os.path.join(dir, o))]

    df = pd.DataFrame(columns=['id', 'mask_volume_mm3', 'mask_volume_ml'])

    for subject in subjects:
        subject_folder = os.path.join(dir, subject)
        voi_files = [o for o in os.listdir(subject_folder)
               if os.path.isfile(os.path.join(subject_folder, o))]
        if any([f for f in voi_files if f.endswith('.nii.gz')]):
            voi_file = [f for f in voi_files if f.endswith('.nii.gz')][0]
            voi_file = os.path.join(subject_folder, voi_file)
        elif any([f for f in voi_files if f.endswith('.nii')]):
            voi_file = [f for f in voi_files if f.endswith('.nii')][0]
            voi_file = os.path.join(subject_folder, voi_file)
        else:
            voi_files = [f for f in voi_files if f.endswith('.voi')]
            if len(voi_files) == 0:
                print('No lesion found for subject', subject)
                df.append(pd.DataFrame([[subject, 'NaN', 'NaN']], columns=df.columns), ignore_index=True)
                continue
            initial_voi_file = voi_files[0]
            pre, ext = os.path.splitext(initial_voi_file)
            voi_file = os.path.join(subject_folder, pre + '.nii.gz')
            os.rename(os.path.join(subject_folder, initial_voi_file), voi_file)

        mask_volume_mm3, mask_volume_ml = mask_volume(voi_file)
        df = df.append(pd.DataFrame([[subject, mask_volume_mm3, mask_volume_ml]], columns=df.columns), ignore_index=True)
    df.to_excel(os.path.join(dir, 'mask_volumes.xlsx'))

mask_volume_batch('/Users/julian/stroke_datasets/WU-LESION-Drawing/WU_drawing_2018-2019-2020')

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Compute mask volume for each subject')
#     parser.add_argument('main_directory')
#     args = parser.parse_args()
#     mask_volume_batch(args.main_directory)

