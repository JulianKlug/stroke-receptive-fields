import os, sys, random, string
sys.path.insert(0, '../')
import data_loader, manual_data
from vxl_glm.LogReg_glm import LogReg_glm
from vxl_threshold.RAPID_model import RAPID_Model_Generator
from vxl_threshold.Campbell_model import Campbell_Model_Generator
from wrapper_cv import launch_cv, rf_hyperopt

main_dir = '/Users/julian/master/data/all2016_dataset'
# main_dir = '/home/klug/data/working_data/2016_all'
data_dir = os.path.join(main_dir, '')
main_output_dir = os.path.join(main_dir, 'models')
main_save_dir = os.path.join(main_dir, 'temp_data')

CLIN, IN, OUT, MASKS, IDS, PARAMS = data_loader.load_saved_data(data_dir)
# Order: 'wcoreg_RAPID_Tmax', 'wcoreg_RAPID_rCBF', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV'

# Ignore clinical data for now
CLIN = None

# MASKS = numpy.full(OUT.shape, True) # do not use masks
# IN, OUT = manual_data.load(data_dir) # select data manually

# n_repeats = 10
# n_folds = 5
n_repeats = 1
n_folds = 2

# Feature can accelerate some algorithms
# should not be used if predetermined thresholds are used
feature_scaling = False

# Smoothing before training and testing (applied before thresholding by Campbell et al)
pre_smoothing = True

# Model_Generator = RAPID_Model_Generator(IN.shape, feature_scaling, threshold='train', post_smoothing=True)
Model_Generator = Campbell_Model_Generator(IN.shape, feature_scaling, pre_smoothing)
# Model_Generator = LogReg_glm

model_name = 'thresholdfree_presmooth_Campbell'
rf_hp_start = 0
rf_hp_end = 1
rf_hyperopt(model_name, Model_Generator, IN, OUT, CLIN, MASKS, IDS, feature_scaling, pre_smoothing,
                n_repeats, n_folds, main_save_dir, main_output_dir, rf_hp_start, rf_hp_end)
