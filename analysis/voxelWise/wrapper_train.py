import os, sys, numpy, torch, shutil
sys.path.insert(0, '../')
import data_loader
from vxl_xgboost.ram_xgb import Ram_xgb
from vxl_glm.LogReg_glm import LogReg_glm
from cv_framework import create_fold, standardise

main_dir = '/Users/julian/master/data/from_Server'
# main_dir = '/Users/julian/master/server_output'
# main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, '')
main_output_dir = os.path.join(main_dir, 'models')
main_save_dir = os.path.join(main_dir, 'temp_data')

CLIN, IN, OUT, MASKS = data_loader.load_saved_data(data_dir)
CLIN = None
MAKS = numpy.full(IN.shape, True)


rf = 1
model_name = 'test1_{}'.format(rf)
receptive_field_dimensions = [rf,rf,rf]
feature_scaling = True


save_dir = os.path.join(main_save_dir, model_name + '_data')
output_dir = os.path.join(main_output_dir, model_name + '_output')
if os.path.exists(save_dir) or os.path.exists(output_dir):
    print('This model already has saved output ', output_dir)
    validation = input('Type `yes` if you wish to delete your previous model:\t')
    if (validation != 'yes'):
        raise ValueError('Model already has saved data. Choose another model name or delete current model')
    else:
        shutil.rmtree(output_dir); shutil.rmtree(save_dir)
os.makedirs(save_dir); os.makedirs(output_dir)


model = Ram_xgb(save_dir, 'trained')
model.hello_world()
model_params = model.get_settings()

if len(IN.shape) < 5:
    IN = numpy.expand_dims(IN, axis=5)

if feature_scaling == True:
    IN, CLIN = standardise(IN, CLIN)

train = numpy.full(IN.shape[0], True)
test = numpy.full(IN.shape[0], False)
create_fold(model, IN, OUT, MASKS, receptive_field_dimensions, train, test, clinX = None)
trained_model, evals_result = model.train()

results = {
    'model_params': model_params,
    'train_evals': evals_result
}

torch.save(results, os.path.join(output_dir, 'params_' + model_name + '.npy'))
torch.save(trained_model, os.path.join(output_dir, 'trained_model_' + model_name + '.npy'))
