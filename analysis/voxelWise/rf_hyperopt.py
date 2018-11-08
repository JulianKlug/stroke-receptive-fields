import os, sys
sys.path.insert(0, '../')
import data_loader, manual_data
from vxl_xgboost.external_mem_xgb import External_Memory_xgb
from vxl_xgboost.ram_xgb import Ram_xgb
from vxl_glm.LogReg_glm import LogReg_glm
from vxl_NN.LogReg_NN import LogReg_NN
from wrapper_cv import launch_cv

main_dir = '/Users/julian/master/data/clinical_data_test'
# main_dir = '/Users/julian/master/server_output'
# main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, '')
main_output_dir = os.path.join(main_dir, 'models')
main_save_dir = os.path.join(main_dir, 'temp_data')

CLIN, IN, OUT = data_loader.load_saved_data(data_dir)
CLIN = None
# IN, OUT = manual_data.load(data_dir)

n_repeats = 1
n_folds = 2

Model_Generator = LogReg_NN

for rf in range(1,2):
    rf_dim = [rf, rf, rf]
    model_name = 'NN_test' + str(rf)
    launch_cv(model_name, Model_Generator, rf_dim, IN, OUT, CLIN,
                    n_repeats, n_folds, main_save_dir, main_output_dir)

print('Hyperopt done.')
