import os, torch, math, random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
import scipy.stats as stats

def plot_auc_roc(rf_dims, roc_auc_scores, model_name = 'model', color = 'C0', display_legend = True):
    """
    Plot roc_auc for each value of rf (receptive field dimension)

    Args:
        rf_dims: list of receptiveField dimensions
        roc_auc_scores: list of roc_auc_scores coresponding to the rf_dim at the same index

    Returns:
        undefined
    """
    z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*

    mean_roc_auc_scores = []
    mean_rf_dims = []
    auc_upper_limits = []
    auc_lower_limits = []

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
            margin_of_error = z_critical * (std_auc/math.sqrt(len(roc_auc_scores[i])))

            print(i, median_roc_auc_score, mean_roc_auc, std_auc, margin_of_error)
            # auc_upper_limits.append(np.minimum(mean_roc_auc_scores[i] + margin_of_error, 1))
            auc_upper_limits.append(mean_roc_auc_scores[i] + margin_of_error)
            auc_lower_limits.append(np.maximum(mean_roc_auc_scores[i] - margin_of_error, 0))

        for j in range(len(roc_auc_scores[i])):
            plt.plot(rf_dims[i], roc_auc_scores[i][j], 'k.', lw=1, alpha=0.3)

    print('means', mean_roc_auc_scores)
    print('Rf used:', mean_rf_dims)
    print('Using Z:', z_critical)
    print('low', auc_lower_limits)
    print('up', auc_upper_limits)

    if (display_legend):
        # Plot one additional point to have only one label
        plt.plot(0, 2, 'k.', lw=1, alpha=0.3, label=r'ROC AUC score')
        plt.fill_between(mean_rf_dims, auc_upper_limits, auc_lower_limits, color='grey', alpha=.2,
                         # label=r'$\pm$ 1 std. dev.')
                         label=r'$\pm$ 1 std. err.')
    else:
        plt.fill_between(mean_rf_dims, auc_upper_limits, auc_lower_limits, color='grey', alpha=.2)

    plt.plot(mean_rf_dims, mean_roc_auc_scores, color, label=r'Mean ROC AUC for %s' % (model_name))
    plt.ylim([-0.05, 1.05])
    plt.ylabel('ROC AUC')
    plt.xlabel('Receptive field size (as voxels from center)')
    plt.title('Area under the ROC curve')
    # plt.legend(loc="lower right")

    plt.ion()
    plt.draw()

def wrapper_plot_auc_roc(score_dir, model_name, color = 'C0', display_legend = True):
    roc_auc_scores = []
    rf_dims = []
    files = os.listdir(score_dir)
    for file in files:
        if (file.startswith('scores_') and file.endswith('.npy')):
            score_path = os.path.join(score_dir, file)
            score_obj = torch.load(score_path)
            try:
                rf_dims.append(np.mean(score_obj['rf']))
            except KeyError:
                rf_dims.append(int(file.split('_')[-1].split('.')[0]))
            roc_auc_scores.append(score_obj['test_roc_auc'])
    plot_auc_roc(rf_dims, roc_auc_scores, model_name, color, display_legend)
    # plt.show()

def compare(dir1, dir2, dir3, dir4 ):
    fig, ax = plt.subplots()
    wrapper_plot_auc_roc(dir1, 'multi-parameter glm', 'C0')
    wrapper_plot_auc_roc(dir2, 'MTT glm', 'C1', display_legend = False)
    wrapper_plot_auc_roc(dir3, 'Tmax glm', 'C2', display_legend = False)
    wrapper_plot_auc_roc(dir4, 'xgb', 'C3', display_legend = False)
    plt.legend(loc="lower right")
    plt.title('CV Framework')
    plt.show()
