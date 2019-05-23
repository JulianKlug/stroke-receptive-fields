import os, sys, random, string
sys.path.insert(0, '../')
import data_loader, manual_data
# from vxl_xgboost.default_ram_xgb import Default_ram_xgb
# from vxl_xgboost.ram_xgb import Ram_xgb
from vxl_glm.LogReg_glm import LogReg_glm
# from vxl_NN.LogReg_NN import LogReg_NN
# from vxl_NN.Keras_model import TwoLayerNetwork
# from vxl_threshold.Tmax6 import Tmax6_Model_Generator
from vxl_threshold.RAPID_model import RAPID_Model_Generator
from wrapper_cv import launch_cv, rf_hyperopt

main_dir = '/Users/julian/master/data/all2016_subset_prepro'
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

n_repeats = 10
n_folds = 5
# n_repeats = 1
# n_folds = 2

# Feature can accelerate some algorithms
# should not be used if predetermined thresholds are used
feature_scaling = True

# Model_Generator = RAPID_Model_Generator(IN.shape, feature_scaling)
Model_Generator = LogReg_glm

model_name = 'multi_logReg_Glm'
rf_hp_start = 5
rf_hp_end = 6
rf_hyperopt(model_name, Model_Generator, IN, OUT, CLIN, MASKS, IDS, feature_scaling,
                n_repeats, n_folds, main_save_dir, main_output_dir, rf_hp_start, rf_hp_end)
