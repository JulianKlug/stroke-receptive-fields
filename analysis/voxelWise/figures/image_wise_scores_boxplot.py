import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

flatten = lambda l : [item for sublist in l for item in sublist]


def plot_image_wise_scores_boxplot(rf_dims, thresholded_volume_deltas, unthresholded_volume_deltas, image_wise_error_ratios):
    """
    Plot distribution of volume deltas for each value of rf (receptive field dimension)

    Args:
        rf_dims: list of receptiveField dimensions
        roc_auc_scores: list of roc_auc_scores coresponding to the rf_dim at the same index
        settings_iterations: list of number of iterations per rf_dim
        settings_folds : list of folds used for every rf_dim

    Returns:
        undefined
    """

    sns.set_palette(sns.color_palette("deep", 5))

    mean_roc_auc_scores = []
    mean_rf_dims = []

    # print(rf_dims, roc_auc_scores)
    thresholded_volume_deltas = [x for _,x in sorted(zip(rf_dims, thresholded_volume_deltas))]
    unthresholded_volume_deltas = [x for _,x in sorted(zip(rf_dims, unthresholded_volume_deltas))]
    image_wise_error_ratios = [x for _,x in sorted(zip(rf_dims, image_wise_error_ratios))]

    rf_dims.sort()

    # for i in range(len(rf_dims)):
    #
    #     if len(thresholded_volume_deltas[i]) != 0:
    #         median_roc_auc_score = np.median(roc_auc_scores[i])
    #         mean_roc_auc = sum(roc_auc_scores[i]) / float(len(roc_auc_scores[i]))
    #         mean_roc_auc_scores.append(mean_roc_auc)
    #         mean_rf_dims.append(rf_dims[i])
    #
    #         std_auc = np.std(roc_auc_scores[i], axis=0)
    #
    # print('means', mean_roc_auc_scores)
    # print(mean_rf_dims)

    ax1 = sns.boxplot(data=thresholded_volume_deltas, label=r'thresholded volume deltas')
    ax2 = sns.boxplot(data=unthresholded_volume_deltas, label=r'unthresholded volume deltas')
    ax3 = sns.boxplot(data=image_wise_error_ratios, label=r'image-wise error ratio')


    plt.ylabel('')
    plt.xlabel(r'Area under the ROC curve for')
    # Receptive field size (as voxels from center)')
    plt.title('Distribution of ROC AUC scores')
    plt.legend(loc="upper right")

    plt.ion()
    plt.draw()
    plt.show()

def wrapper_plot_image_wise_scores_boxplot(score_dir):
    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    rf_dims = []
    settings_iterations = []
    settings_folds = []
    files = os.listdir(score_dir)
    for file in files:
        if (file.startswith('scores_voxel_wise')):
            score_path = os.path.join(score_dir, file)
            score_obj = torch.load(score_path)
            try:
                rf_dims.append(score_obj['rf'])
            except KeyError:
                rf_dims.append(file.split('_')[-1].split('.')[0])

            thresh_v_deltas = flatten(score_obj['test_thresholded_volume_deltas'])
            thresholded_volume_deltas.append(thresh_v_deltas)
            unthresh_v_deltas = flatten(score_obj['test_unthresholded_volume_deltas'])
            unthresholded_volume_deltas.append(unthresh_v_deltas)
            image_wise_err_r = flatten(score_obj['test_image_wise_error_ratios'])
            image_wise_error_ratios.append(image_wise_err_r)

            settings_iterations.append(score_obj['settings_repeats'])
            settings_folds.append(score_obj['settings_folds'])
            print('For rf:', rf_dims[-1], 'found', settings_folds[-1], 'folds, repeated', settings_iterations[-1], 'times')

    plot_image_wise_scores_boxplot(rf_dims, thresholded_volume_deltas, unthresholded_volume_deltas, image_wise_error_ratios)
