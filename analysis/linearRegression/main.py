import sys
sys.path.insert(0, '../')

import os
import nibabel as nib
import numpy as np
import linReg1 as lr
import visual

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'analysis_test')
input_dir = os.path.join(data_dir, 'Barlovic_Radojka_19480907/Ct2_Cerebral_20160103')
output_dir = os.path.join(data_dir, 'Barlovic_Radojka_19480907/Neuro_Cerebrale_64Ch_20160106')

input_image_path = os.path.join(input_dir, 'wcoreg_RAPID_TMax_[s]_Barlovic_Radojka_19480907.nii')
input_img = nib.load(input_image_path)
input_data = input_img.get_data()

output_image_path = os.path.join(output_dir, 'wcoreg_VOI_lesion_Barlovic_Radojka_19480907.nii')
output_img = nib.load(output_image_path)
output_data = output_img.get_data()

IN = [input_data]
OUT = [output_data]

rf_dim = [1,1,1]

model = lr.receptive_field_log_model(IN, OUT, rf_dim)

predicted = lr.predict(input_data, model, rf_dim)

coordinate_space = input_img.affine
predicted_img = nib.Nifti1Image(predicted, affine=coordinate_space)
nib.save(predicted_img, os.path.join(data_dir,'predicted.nii'))


# visual.display(predicted)
