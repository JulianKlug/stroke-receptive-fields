import os
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_closing, binary_erosion, binary_dilation

def createCSFMask(data_dir, betted_spc_name):
    """
    mean and standard deviation were determined of the intensity histogram within [0,400] HU.
    An upper intensity threshold μ − 1.96σ was applied to segment CSF
    followed by erosion with a structuring element with 1 voxel radius to remove potential false positive voxels due to noise,
    and connected region growing using the same thresholds.
    The CSF mask was subjected to a morphological closing operation with a structuring element with 1 voxel radius.
    """
    path = os.path.join(data_dir, betted_spc_name)
    img = nib.load(path)
    data = img.get_data()
    data[np.isnan(data)] = 0
    clipped_range_data = data[(data > 0) & (data < 40)]
    # intensity threshold μ − 1.96σ
    CSF_threshold = np.mean(clipped_range_data) - 1.96 * np.std(clipped_range_data)

    CSF_mask = np.zeros(data.shape)
    CSF_mask[(data < CSF_threshold)] = 1

    structure = np.ones((2,2,2), dtype=np.int)
    CSF_mask = binary_erosion(CSF_mask, structure)
    CSF_mask = binary_dilation(CSF_mask, structure)

    CSF_mask = binary_closing(CSF_mask, np.ones((4,4,4), dtype=np.int))

    coordinate_space = img.affine
    image_extension = '.nii'
    # MATLAB can not open NIFTI saved as int, thus float is necessary
    labeled_img = nib.Nifti1Image(CSF_mask.astype('float64'), affine=coordinate_space)
    nib.save(labeled_img, os.path.join(data_dir,  'CSF_mask' + image_extension))
