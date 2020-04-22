import numpy as np
import os, argparse, sys
sys.path.insert(0, '../../../')
from analysis import data_loader as dl

def rescale_outliers(imgX, MASKS):
    '''
    Rescale outliers as some images from RAPID seem to be scaled x10
    Outliers are detected if their median exceeds 5 times the global median and are rescaled by dividing through 10
    :param imgX: image data (n, x, y, z, c)
    :return: rescaled_imgX
    '''

    for i in range(imgX.shape[0]):
        for channel in range(imgX.shape[-1]):
            median_channel = np.median(imgX[..., channel][MASKS])
            if np.median(imgX[i, ..., 0][MASKS[i]]) > 5 * median_channel:
                imgX[i, ..., 0] = imgX[i, ..., channel] / 10

    return imgX


def rescale_perfusion_maps_dataset(dataset_path, output_path=None):
    '''
    Rescale outliers in Perfusion Maps Dataset (needed for RAPID maps)
    :param dataset_path: path to dataset
    :return: save rescaled dataset
    '''
    data_dir, filename = os.path.split(dataset_path)
    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = dl.load_saved_data(data_dir, filename)

    ct_inputs = rescale_outliers(ct_inputs, brain_masks)

    dataset = (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params)
    if output_path is None:
        output_path = os.path.join(data_dir, f'rescaled_{filename}')
    out_dir, out_filename = os.path.split(output_path)
    dl.save_dataset(dataset, out_dir, out_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rescale outliers in Perfusion Maps Dataset (needed for RAPID maps)')
    parser.add_argument('dataset_path')
    args = parser.parse_args()
    rescale_perfusion_maps_dataset(args.dataset_path)

