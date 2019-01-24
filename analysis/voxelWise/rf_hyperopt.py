import os, sys, numpy
sys.path.insert(0, '../')
import data_loader, manual_data
from vxl_xgboost.default_ram_xgb import Default_ram_xgb
from vxl_xgboost.ram_xgb import Ram_xgb
from vxl_glm.LogReg_glm import LogReg_glm
from vxl_NN.LogReg_NN import LogReg_NN
from vxl_NN.Keras_model import TwoLayerNetwork
from wrapper_cv import launch_cv

main_dir = '/Users/julian/master/data/from_Server'
# main_dir = '/Users/julian/master/server_output'
# main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, '')
main_output_dir = os.path.join(main_dir, 'models')
main_save_dir = os.path.join(main_dir, 'temp_data')

CLIN, IN, OUT, MASKS = data_loader.load_saved_data(data_dir)
CLIN = None
# MASKS = numpy.full(OUT.shape, True)
# IN, OUT = manual_data.load(data_dir)

n_repeats = 10
n_folds = 5
feature_scaling = True

Model_Generator = TwoLayerNetwork

for rf in range(1):
    rf_dim = [rf, rf, rf]
    model_name = 'keras_test_' + str(rf)
    launch_cv(model_name, Model_Generator, rf_dim, IN, OUT, CLIN, MASKS, feature_scaling,
                    n_repeats, n_folds, main_save_dir, main_output_dir)

print('Hyperopt done.')
