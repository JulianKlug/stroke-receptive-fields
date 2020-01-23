import sys, os
sys.path.insert(0, '../')
from IAT_status_split import split_dataset
import data_loader
from voxelwise.vxl_glm.LogReg_glm import LogReg_glm
from voxelwise.vxl_threshold.Campbell_model import Campbell_Model_Generator
from voxelwise.vxl_threshold.customThresh  import customThreshold_Model_Generator
from voxelwise.wrapper_cv import rf_hyperopt


data_dir = '/home/klug/data/working_data/all_2016_2017'
main_save_dir = os.path.join(data_dir, 'IAT_subgroup_analysis')
IAT_status_path = '/home/klug/data/clinical_data/all_2016_2017/recanalisation_status.xlsx'
# data_dir = '/Users/julian/stroke_research/data/all2016_dataset'
# main_save_dir = os.path.join(data_dir, 'IAT_subgroup_analysis')
# IAT_status_path = '/Users/julian/temp/IAT_subgroup_analysis/recanalisation_status.xlsx'


data_set = data_loader.load_saved_data(data_dir)
clinical_inputs, ct_inputs, ct_label, mri_inputs, mri_lesion_GT, brain_masks, ids, param = data_set

# Feature can accelerate some algorithms
# should not be used if predetermined thresholds are used
feature_scaling = True
# Smoothing before training and testing (applied before thresholding by Campbell et al)
pre_smoothing = False
# Normalise channel by mean of contralateral side: can be used to obtain rCBF [1] and rCBV [3]
channels_to_normalise = False
# Add a normalization term to account for the number of voxels outside the defined brain ct_inputs a receptive field
undef_normalisation = False
# Use a flat receptive field (flat in z) - [rf_x, rf_y, 0]
flat_rf = False


#Model_Generator = customThreshold_Model_Generator(ct_inputs.shape, feature_scaling, fixed_threshold = 6)
#Model_Generator = Campbell_Model_Generator(ct_inputs.shape, feature_scaling, pre_smoothing)
Model_Generator = LogReg_glm
model_name = 'IVT_multiGLM'
rf_hyperopt_start = 0
rf_hyperopt_end = 5
n_repeats = 10
n_folds = 5

IAT_indexes, IVT_indexes, unknown_status_indexes = split_dataset(data_set, IAT_status_path)
selected = IVT_indexes

print(len(unknown_status_indexes), 'subjects with unknown status:', ids[unknown_status_indexes])


ct_inputs, ct_label, clinical_inputs, brain_masks, ids = ct_inputs[selected], ct_label[selected], None, brain_masks[selected], ids[selected]
rf_hyperopt(model_name, Model_Generator, ct_inputs, ct_label, clinical_inputs, brain_masks, ids,
            feature_scaling, pre_smoothing, channels_to_normalise, undef_normalisation, flat_rf,
                n_repeats, n_folds, main_save_dir, main_save_dir, rf_hyperopt_start, rf_hyperopt_end)
