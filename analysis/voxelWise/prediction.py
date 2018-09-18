import sys
sys.path.insert(0, '../')

import os, timeit
import nibabel as nib
import numpy as np
from sklearn.externals import joblib
import receptiveField as rf
import visual, scoring_utils
import data_loader

main_dir = '/Users/julian/master/server_output/test1/'
data_dir = os.path.join(main_dir, '')
model_dir = '/Users/julian/master/server_output/rf1_full_model'
model_name = 'rf1_full_model'
model_extension = '.pkl'
model_path = os.path.join(model_dir, model_name + model_extension)

input_dir = os.path.join(data_dir, 'LOO')

input_image_path = os.path.join(input_dir, '898729/Ct2_Cerebrale/wcoreg_RAPID_MTT_898729.nii')
input_img = nib.load(input_image_path)
# input_data = input_img.get_data()

ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
# ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
# ct_sequences = ['wcoreg_RAPID_TMax_[s]']
mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load_nifti(input_dir, ct_sequences, mri_sequences)

input_data = IN[0]
output_GT = OUT[0]

homogenous_rf = 1
rf_dim = [homogenous_rf, homogenous_rf, homogenous_rf]

print('Predict output image with model: ', model_name)
model = joblib.load(model_path)

start = timeit.default_timer()
predicted = rf.predict(input_data, input_dir, model, rf_dim, external_memory = True)
end = timeit.default_timer()
print('Prediction time: ', end - start)

print('Predicted shape', predicted.shape)

threshold = 0.5
lesion_volume = np.sum(predicted > threshold)
print('Predicted lesion volume', lesion_volume)

scoring_utils.validate(predicted.reshape(-1), output_GT.reshape(-1))

coordinate_space = input_img.affine
image_extension = '.nii'
predicted_img = nib.Nifti1Image(predicted, affine=coordinate_space)
nib.save(predicted_img, os.path.join(model_dir, model_name + '' + image_extension))


visual.display(predicted)
