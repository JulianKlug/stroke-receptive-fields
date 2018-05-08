import os
import nibabel as nib
import numpy as np

def load(data_dir):
    input_dir = os.path.join(data_dir, '448776/Ct2_Cerebrale/')
    output_dir = os.path.join(data_dir, '448776/Neuro_Cerebrale_64Ch')

    input_image_path = os.path.join(input_dir, 'wcoreg_RAPID_Tmax_448776.nii')
    input_img = nib.load(input_image_path)
    input_data = input_img.get_data()

    input_matrix = np.zeros([input_data.shape[0],input_data.shape[1],input_data.shape[2],1])
    input_matrix[:,:,:,0] = input_data

    output_image_path = os.path.join(output_dir, 'wcoreg_VOI_lesion_448776.nii')
    output_img = nib.load(output_image_path)
    output_data = output_img.get_data()

    IN = [input_matrix]
    OUT = [output_data]

    return IN, OUT
