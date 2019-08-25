import sys, os
sys.path.insert(0, '../')
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from analysis.visual import display
from scipy.stats import wilcoxon
import pandas as pd

# main_dir = '/Users/julian/master/server_output/selected_for_article1_13022019'
# output_dir = '/Users/julian/master/saved_results'

models_file = '/Users/julian/stroke_research/all_2016_2017_results/selected_models/all_pCT_logReg/all_pCT_logReg_rf_3_output/trained_models_all_pCT_logReg_rf_3.npy'
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


def glm_parameter_weight_distribution(models_file_path, name):
    models = torch.load(models_file_path)
    weights = np.array([m.coef_.reshape(4, -1) for m in models])
    print(weights.shape)
    parameter_weights = np.array([np.median(weights[i, ...], axis=1) for i in range(len(models))]).T
    median_parameter_weights = np.median(parameter_weights, axis=1)
    std_weights = np.std(parameter_weights, axis=1)

    print(std_weights)
    print(np.array(median_parameter_weights))
    print(wilcoxon(parameter_weights[0], parameter_weights[1]))
    print(wilcoxon(parameter_weights[0], parameter_weights[2]))
    print(wilcoxon(parameter_weights[0], parameter_weights[3]))

    norm_median_parameters_weights = np.abs(median_parameter_weights) / np.sum(np.abs(median_parameter_weights))

    plot_parameters(['Tmax', 'CBF', 'MTT', 'CBV'], norm_median_parameters_weights, name)
    return median_parameter_weights, std_weights

def plot_parameters(score_names, scores, name):
    plt.figure()
    sns.barplot(x = score_names, y = scores, palette=sns.cubehelix_palette(4, start=0.7, rot=-.75))
    # axes formatting
    # plt.ylim(120, 160)
    sns.set()

    plt.title(name)
    plt.ylabel('Relative Parameter Weight')
    # plt.legend(loc="upper right")
    plt.show()
    fn = "parameter_weights_" + name
    plt.savefig(fn, format="svg")

def plot_weight_progression(Tmax, CBF, MTT, CBV):
    print(Tmax)
    ax= sns.lineplot('rf', 'median', data=Tmax)
    ax.errorbar(Tmax['rf'], Tmax['median'], yerr=Tmax['std'], fmt='-o')
    sns.lineplot('rf', 'median', data=CBF)
    sns.lineplot('rf', 'median', data=MTT)
    sns.lineplot('rf', 'median', data=CBV)
    plt.show()

def wrapper_comparative_model_weights(modality_dir):
    columns = ['rf', 'median', 'std']
    Tmax = pd.DataFrame(columns=columns)
    CBF = pd.DataFrame(columns=columns)
    MTT = pd.DataFrame(columns=columns)
    CBV = pd.DataFrame(columns=columns)

    evals = [o for o in os.listdir(modality_dir)
                        if os.path.isdir(os.path.join(modality_dir,o))]


    for eval_dir in evals:
        files = os.listdir(os.path.join(modality_dir, eval_dir))
        rf = None; mean_weights = None;
        for file in files:
            if (file.startswith('params_') and file.endswith('.npy')):
                params_path = os.path.join(modality_dir, eval_dir, file)
                param_obj = torch.load(params_path)
                rf = np.mean(param_obj['rf'])
        for file in files:
            if (file.startswith('trained_models_') and file.endswith('.npy')):
                models_path = os.path.join(modality_dir, eval_dir, file)
                mean_weights, std_weights = glm_parameter_weight_distribution(models_path, 'rf = ' + str(rf))
                mean_weights = np.abs(mean_weights)
                mean_weights = mean_weights / np.sum(mean_weights)
                std_weights = mean_weights / np.sum(mean_weights)

        if (rf is not None) and (mean_weights is not None):
            Tmax = Tmax.append(pd.DataFrame([[rf, mean_weights[0], std_weights[0]]], columns=columns), ignore_index=True)
            CBF = CBF.append(pd.DataFrame([[rf, mean_weights[1], std_weights[1]]], columns=columns), ignore_index=True)
            MTT = MTT.append(pd.DataFrame([[rf, mean_weights[2], std_weights[2]]], columns=columns), ignore_index=True)
            CBV = CBV.append(pd.DataFrame([[rf, mean_weights[3], std_weights[3]]], columns=columns), ignore_index=True)


    plot_weight_progression(Tmax, CBF, MTT, CBV)




wrapper_comparative_model_weights('/Users/julian/stroke_research/all_2016_2017_results/selected_models/all_pCT_logReg/')

# glm_parameter_weight_distribution(models_file)