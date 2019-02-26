import sys
sys.path.insert(0, '../')
import torch
import numpy as np
from analysis.visual import display

main_dir = '/Users/julian/master/server_output/selected_for_article1_13022019'
output_dir = '/Users/julian/master/saved_results'

models_file = '/Users/julian/master/server_output/selected_for_article1_13022019/Tmax0_logRegGLM/scaled_masked_Tmax0_logRegGLM_rf_3_output/trained_models_scaled_masked_Tmax0_logRegGLM_rf_3.npy'
models = torch.load(models_file)

weights = np.squeeze([m.coef_ for m in models])
median_weights = np.median(weights, axis=0)
std_weights = np.std(weights, axis=0)


display(median_weights.reshape(7,7,7))
