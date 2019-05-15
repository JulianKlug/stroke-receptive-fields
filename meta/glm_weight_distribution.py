import sys
sys.path.insert(0, '../')
import torch
import numpy as np
from analysis.visual import display
from scipy.stats import wilcoxon

# main_dir = '/Users/julian/master/server_output/selected_for_article1_13022019'
# output_dir = '/Users/julian/master/saved_results'

models_file = '/Users/julian/master/server_output/masked_scaled_cv_framework/LogRegGLM/masked_scaled_logRegGLM_rf_3_output/trained_models_masked_scaled_logRegGLM_rf_3.npy'
# Order: 'wcoreg_RAPID_Tmax', 'wcoreg_RAPID_rCBF', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV'


def plot_glm_spatial_weight_distribution(models_file):
    '''

    :param models_file: path to trained models file (works only with Tmax for now)
    :return:
    '''
    models = torch.load(models_file)
    weights = np.squeeze([m.coef_ for m in models])
    median_weights = np.median(weights, axis=0)
    std_weights = np.std(weights, axis=0)
    display(median_weights.reshape(7,7,7))


def glm_parameter_weight_distribution(models_file_path):
    models = torch.load(models_file_path)
    weights = np.squeeze([m.coef_ for m in models])
    median_weights = np.median(weights, axis=0).reshape(4, -1)
    median_parameter_weights = [np.median(median_weights[i, ...]) for i in range(4)]
    std_weights = np.std(median_weights, axis=1)

    print(std_weights)
    print(np.array(median_parameter_weights))
    print(wilcoxon(median_weights[3], median_weights[0]))


glm_parameter_weight_distribution(models_file)