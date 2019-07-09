import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    return [item for sublist in l for item in sublist]

def boxplot_sorted(df, by, column, rot=0, ax=None, hue=None):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    # return df2[meds.index].boxplot(rot=rot, hue=hue, return_type="axes", ax = ax)
    return sns.boxplot(x = 'model_tag', y = 'roc_auc', order = meds.index, data = df, hue = hue,
        ax = ax, dodge = False, palette="Pastel2")

columns = ['model_name', 'rf', 'model_class', 'model_tag', 'roc_auc']
def wrapper_plot_auc_roc_boxplot(modality_dirs):
    roc_auc_df = pd.DataFrame(columns = columns)

    for modality_dir in modality_dirs[:]:
        if os.path.isdir(modality_dir):
            rf_evaluations = [o for o in os.listdir(modality_dir)
                                if os.path.isdir(os.path.join(modality_dir,o))]

            for rf_eval in rf_evaluations:
                print('Reading', rf_eval)
                rf_eval_dir = os.path.join(modality_dir, rf_eval)

                result_files = [i for i in os.listdir(rf_eval_dir)
                                    if os.path.isfile(os.path.join(rf_eval_dir, i))
                                        and i.startswith('scores_') and i.endswith('.npy')]
                results = torch.load(os.path.join(rf_eval_dir, result_files[0]))

                params_files = [i for i in os.listdir(rf_eval_dir)
                                    if os.path.isfile(os.path.join(rf_eval_dir, i))
                                        and i.startswith('params_') and i.endswith('.npy')]
                params = torch.load(os.path.join(rf_eval_dir, params_files[0]))

                model_name = '_'.join(rf_eval.split('_')[:-1])
                if 'params' in results:
                    params = results['params']
                try:
                    rf = int(np.median(params['rf']))
                except (KeyError, TypeError):
                    rf = int(result_files[0].split('_')[-1].split('.')[0])

                if 'all_pCT' in model_name:
                    model_class = 'glm(multi)'
                if 'Tmax' in model_name:
                    model_class = 'glm(Tmax)'
                if 'Tmax_6' in model_name:
                    model_class = 'Tmax > 6s'
                if 'Tmax_trained' in model_name:
                    # this should be the same as glm(Tmax) at rf0
                    model_class = 'Tmax > t'
                    continue
                if 'CBV' in model_name:
                    model_class = 'glm(CBV)'
                if 'CBF' in model_name:
                    model_class = 'glm(CBF)'
                if 'MTT' in model_name:
                    model_class = 'glm(MTT)'
                if 'Campbell' in model_name and 'trained' in model_name:
                    model_class = 'Campbell'
                if 'Campbell' in model_name and not 'trained' in model_name:
                    model_class = 'nCBF'

                model_tag = model_class + ' at rf ' + str(rf)

                # leave out other models
                if not (rf == 0 or rf == 3):
                    continue
                scores = flatten(results['test_roc_auc'])
                new_entries = list(zip(
                    np.repeat(model_name,len(scores)),
                    np.repeat(rf,len(scores)),
                    np.repeat(model_class,len(scores)),
                    np.repeat(model_tag,len(scores)),
                    scores
                    ))
                roc_auc_df = roc_auc_df.append(pd.DataFrame(new_entries, columns = columns))


    fig, ax = plt.subplots()
    axes = boxplot_sorted(roc_auc_df, by=["model_tag"], column="roc_auc", hue='rf', ax = ax)
    labels = [l.get_text() for l in axes.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_title("Boxplots of the area under the ROC curve by model")
    plt.ylabel('Area under the ROC curve')
    plt.xlabel('')
    # Receptive field size (as voxels from center)q
    fig.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("/Users/julian/stroke_research/all_2016_2017_results/boxplot.svg", format="svg")

main_dir = '/Users/julian/stroke_research/all_2016_2017_results'
multiGLM = os.path.join(main_dir, 'all_pCT_logReg')
MTT = os.path.join(main_dir, 'MTT2_logReg')
Tmax = os.path.join(main_dir, 'Tmax0_logReg')
CBF = os.path.join(main_dir, 'CBF1_logReg')
CBV = os.path.join(main_dir, 'CBV3_logReg')
Tresh_Tmax = os.path.join(main_dir, 'Tmax_threshold')
normalised_CBF = os.path.join(main_dir, 'normalised_CBF')
Campbell_model = os.path.join(main_dir, 'Campbell_model')

# wrapper_plot_auc_roc_boxplot([normalised_CBF])
wrapper_plot_auc_roc_boxplot([multiGLM, MTT, Tmax, CBF, CBV, Tresh_Tmax, normalised_CBF, Campbell_model])
