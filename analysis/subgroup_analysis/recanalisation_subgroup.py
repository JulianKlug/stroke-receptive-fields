import sys, os
sys.path.insert(0, '../')
from random import shuffle
from subgroup_analysis import subgroup_evaluation_launcher
from recanalisation_status_split import split_dataset
import data_loader
from voxelwise.vxl_glm.LogReg_glm import LogReg_glm
from voxelwise.vxl_threshold.Campbell_model import Campbell_Model_Generator
from voxelwise.vxl_threshold.customThresh  import customThreshold_Model_Generator

# data_dir = '/home/klug/data/working_data/all_2016_2017'
# main_save_dir = os.path.join(data_dir, 'recanalised_subgroup_analysis')
# recanalisation_status_path = '/home/klug/data/clinical_data/all_2016_2017/recanalisation_status.xlsx'
data_dir = '/Users/julian/stroke_research/data/all2016_dataset'
main_save_dir = os.path.join(data_dir, 'recanalised_subgroup_analysis')
recanalisation_status_path = '/Users/julian/OneDrive - unige.ch/stroke_research/article/data/clinical_data_all_2016_2017/included_subjects_data_23072019/recanalisation_status.xlsx'


data_set = data_loader.load_saved_data(data_dir)
clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, param = data_set

# Feature can accelerate some algorithms
# should not be used if predetermined thresholds are used
feature_scaling = True

# Smoothing before training and testing (applied before thresholding by Campbell et al)
pre_smoothing = False

#Model_Generator = customThreshold_Model_Generator(ct_inputs.shape, feature_scaling, fixed_threshold = 6)
#Model_Generator = Campbell_Model_Generator(ct_inputs.shape, feature_scaling, pre_smoothing)
Model_Generator = LogReg_glm
model_name = 'multiGLM_test'
rf_hyperopt_start = 0
rf_hyperopt_end = 1

recanalised_indexes, non_recanalised_indexes, unknown_status_indexes = split_dataset(data_set,
                                                                                     recanalisation_status_path)
clinical_inputs = None

total_subjs = len(ids)
if len(recanalised_indexes) >= round(total_subjs / 5):
    size_test_group = round(total_subjs / 5)
else:
    size_test_group = len(recanalised_indexes)

shuffle(recanalised_indexes)
selected_recanalised_indexes = recanalised_indexes[:size_test_group]
non_selected_recanalised_indexes = recanalised_indexes[size_test_group:]
data_set = (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, param)
for rf in range(rf_hyperopt_start, rf_hyperopt_end):
    rf_dim = [rf, rf, rf]
    model_id = model_name + '_rf_' + str(rf)
    save_dir = os.path.join(main_save_dir, model_id)
    subgroup_evaluation_launcher(model_id, Model_Generator, data_set, selected_recanalised_indexes, save_dir,
                                       rf_dim, feature_scaling, pre_smoothing)