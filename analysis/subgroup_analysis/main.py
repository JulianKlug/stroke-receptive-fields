import sys, os
sys.path.insert(0, '../')
from recanalisation_status import recanalisation_evaluation_launcher
import data_loader
from voxelwise.vxl_glm.LogReg_glm import LogReg_glm

data_dir = '/Users/julian/stroke_research/data/all2016_dataset'
save_dir = os.path.join(data_dir, 'recanalised_subgroup_analysis')
recanalisation_status_path = ''

data_set = data_loader.load_saved_data(data_dir)
clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, param = data_set
clinical_inputs = None


# Feature can accelerate some algorithms
# should not be used if predetermined thresholds are used
feature_scaling = True

# Smoothing before training and testing (applied before thresholding by Campbell et al)
pre_smoothing = False

Model_Generator = LogReg_glm
model_name = 'multiGLM_test'

data_set = (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, param)
for rf in range(0, 1):
    rf_dim = [rf, rf, rf]
    model_id = model_name + '_rf_' + str(rf)
    save_dir = os.path.join(save_dir, model_id)
    recanalisation_evaluation_launcher(model_id, Model_Generator, data_set, recanalisation_status_path, save_dir,
                                       rf_dim, feature_scaling, pre_smoothing)