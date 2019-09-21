import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_auc_roc_distribution(rf_dims, roc_auc_scores, settings_iterations, settings_folds):
    """
    Plot distribution of roc_auc scores for each value of rf (receptive field dimension)

    Args:
        rf_dims: list of receptiveField dimensions
        roc_auc_scores: list of roc_auc_scores coresponding to the rf_dim at the same index
        settings_iterations: list of number of iterations per rf_dim
        settings_folds : list of folds used for every rf_dim

    Returns:
        undefined
    """

    n_iterations = settings_iterations[0]
    n_folds = settings_folds[0]
    if not (len(set(settings_folds)) == 1 and len(set(settings_iterations)) == 1):
        print('!Settings used differ between experiments!')


    sns.set_palette(sns.color_palette("deep", 5))

    mean_roc_auc_scores = []
    mean_rf_dims = []

    # print(rf_dims, roc_auc_scores)
    roc_auc_scores = [x for _,x in sorted(zip(rf_dims, roc_auc_scores))]
    rf_dims.sort()

    for i in range(len(rf_dims)):

        if len(roc_auc_scores[i]) != 0:
            median_roc_auc_score = np.median(roc_auc_scores[i])
            mean_roc_auc = sum(roc_auc_scores[i]) / float(len(roc_auc_scores[i]))
            mean_roc_auc_scores.append(mean_roc_auc)
            mean_rf_dims.append(rf_dims[i])

            std_auc = np.std(roc_auc_scores[i], axis=0)

            sns.distplot(roc_auc_scores[i], bins=20, hist = True, kde=True, rug=False,
                label=r'ROC AUC for rf = %i' % (int(rf_dims[i][0])))

    print('means', mean_roc_auc_scores)
    print(mean_rf_dims)

    plt.ylabel('Number of simulations')
    plt.xlabel(r'Area under the ROC curve for %i-fold crossvalidation over %i iterations' % (int(n_folds), int(n_iterations)))
    # Receptive field size (as voxels from center)')
    plt.title('Distribution of ROC AUC scores')
    plt.legend(loc="upper right")

    plt.ion()
    plt.draw()
    plt.show()

def wrapper_plot_auc_roc_distribution(score_dir):
    roc_auc_scores = []
    rf_dims = []
    settings_iterations = []
    settings_folds = []
    files = os.listdir(score_dir)
    for file in files:
        if (file.startswith('scores_')):
            score_path = os.path.join(score_dir, file)
            score_obj = torch.load(score_path)

            # In older versions params were not seperated
            param_obj = score_obj
            if 'params' in score_obj:
                param_obj = score_obj['params']
            try:
                rf_dims.append(param_obj['rf'])
            except KeyError:
                rf_dims.append(file.split('_')[-1].split('.')[0])
            roc_auc_scores.append(score_obj['test_roc_auc'])
            settings_iterations.append(param_obj['settings_repeats'])
            settings_folds.append(param_obj['settings_folds'])

    plot_auc_roc_distribution(rf_dims, roc_auc_scores, settings_iterations, settings_folds)
