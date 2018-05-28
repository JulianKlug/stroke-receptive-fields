import sys
sys.path.insert(0, '../')

import os, timeit
import nibabel as nib
import numpy as np
from sklearn.externals import joblib
import receptiveField as rf
import visual
import data_loader

main_dir = '/Users/julian/master/data/'
data_dir = os.path.join(main_dir, 'analysis_test2')
model_dir = os.path.join(data_dir, 'temp_stride_test')
model_name = 'gl_shorter_new_test_rf0'
model_extension = '.pkl'
model_path = os.path.join(model_dir, model_name + model_extension)

input_dir = os.path.join(data_dir, '')

input_image_path = os.path.join(input_dir, 'patient/Ct2_Cerebral_20160103/wcoreg_RAPID_MTT_[s]_patient_19480907.nii')
input_img = nib.load(input_image_path)
# input_data = input_img.get_data()

# ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
# ct_sequences = ['wcoreg_RAPID_TMax_[s]']
mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load(input_dir, ct_sequences, mri_sequences)

input_data = IN[0]

homogenous_rf = 0
rf_dim = [homogenous_rf, homogenous_rf, homogenous_rf]

print('Predict output image with model: ', model_name)
model = joblib.load(model_path)

start = timeit.default_timer()
predicted = rf.predict(input_data, model, rf_dim)
end = timeit.default_timer()
print('Prediction time: ', end - start)


print('Predicted shape', predicted.shape)
print('Predicted lesion size', np.sum(predicted))

coordinate_space = input_img.affine
image_extension = '.nii'
predicted_img = nib.Nifti1Image(predicted, affine=coordinate_space)
nib.save(predicted_img, os.path.join(model_dir, model_name + '' + image_extension))


visual.display(predicted)
