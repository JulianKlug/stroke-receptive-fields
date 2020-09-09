import numpy as np
import nibabel as nib

def mask_volume(mask_nifti_file):
    """
    Compute volume of mask from a nifti file (vanilla python).
    Equivalent to "fslstats /path/file.nii -V"
    CLI example: "python mask_volume /path/file.nii"
    :param mask_nifti_file: path to nifti file containing mask.
    All voxels of the mask should be of value 1, background should have value 0.
    :return: volume of mask in mm3
    """
    mask_image = nib.load(mask_nifti_file)
    header = mask_image.header
    _, vx, vy, vz, _, _, _, _ = header['pixdim']
    voxel_volume_mm3 = vx * vy * vz
    mask = mask_image.get_fdata()
    mask_volume_vx = np.sum(mask)
    mask_volume_mm3 = mask_volume_vx * voxel_volume_mm3
    mask_volume_ml = mask_volume_mm3 / 1000

    print(f'mask volume {mask_volume_vx} voxels \n{mask_volume_mm3} mm3 / {mask_volume_ml} ml')
    return mask_volume_mm3, mask_volume_ml


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute mask volume')
    parser.add_argument('mask_nifti_file')
    args = parser.parse_args()
    mask_volume(args.mask_nifti_file)
