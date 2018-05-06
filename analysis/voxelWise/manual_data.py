import os
import nibabel as nib
import numpy as np

def load(data_dir):
    input_dir = os.path.join(data_dir, 'pt1/Ct2_Cerebral_20160107')
    output_dir = os.path.join(data_dir, 'pt1/Neuro_Routine_20160111')

    input_image_path = os.path.join(input_dir, 'wcoreg_RAPID_TMax_[s]_pt1.nii')
    input_img = nib.load(input_image_path)
    input_data = input_img.get_data()

    input_matrix = np.zeros([input_data.shape[0],input_data.shape[1],input_data.shape[2],1])
    input_matrix[:,:,:,0] = input_data

    output_image_path = os.path.join(output_dir, 'wcoreg_VOI_lesion_pt1.nii')
    output_img = nib.load(output_image_path)
    output_data = output_img.get_data()

    IN = [input]
    OUT = [output_data]

    return IN, OUT
