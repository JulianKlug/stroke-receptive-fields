import seaborn as sns
import torch
import numpy as np

def plot_auc_roc_distribution(rf_dims, roc_auc_scores):
    """
    Plot distribution of roc_auc scores for each value of rf (receptive field dimension)

    Args:
        rf_dims: list of receptiveField dimensions
        roc_auc_scores: list of roc_auc_scores coresponding to the rf_dim at the same index

    Returns:
        undefined
    """

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

            sns.distplot(roc_auc_scores, bins=20, kde=False, rug=True)

    print('means', mean_roc_auc_scores)
    print(mean_rf_dims)

    # Plot one additional point to have only one label
    # plt.ylim([-0.05, 1.05])
    # plt.ylabel('ROC AUC')
    # plt.xlabel('Receptive field size (as voxels from center)')
    # plt.title('Area under the ROC curve')
    # plt.legend(loc="lower right")

    # plt.ion()
    # plt.draw()
    # plt.show()

def wrapper_plot_auc_roc_distribution(score_dir):
    roc_auc_scores = []
    rf_dims = []
    score_paths = []
    files = os.listdir(score_dir)
    for file in files:
        if (file.startswith('scores_repeat20_rf')):
            score_path = os.path.join(score_dir, file)
            rf_dims.append(file.split('_')[-1].split('.')[0])
            roc_auc_scores.append(torch.load(score_path)['test_roc_auc'])
    plot_auc_roc(rf_dims, roc_auc_scores)
