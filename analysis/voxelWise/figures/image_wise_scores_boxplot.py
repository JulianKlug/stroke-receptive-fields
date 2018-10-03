import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

flatten = lambda l : [item for sublist in l for item in sublist]


def plot_image_wise_scores_boxplot(rf_dims, thresholded_volume_deltas, unthresholded_volume_deltas, image_wise_error_ratios, image_wise_jaccards):
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

    # print(rf_dims, roc_auc_scores)
    thresholded_volume_deltas = [x for _,x in sorted(zip(rf_dims, thresholded_volume_deltas))]
    unthresholded_volume_deltas = [x for _,x in sorted(zip(rf_dims, unthresholded_volume_deltas))]
    image_wise_error_ratios = [x for _,x in sorted(zip(rf_dims, image_wise_error_ratios))]
    image_wise_jaccards = [x for _,x in sorted(zip(rf_dims, image_wise_jaccards))]
    rf_dims.sort()

    plt.subplot(2, 2, 1)
    ax1 = sns.boxplot(data=thresholded_volume_deltas)#, label=r'thresholded volume deltas')
    plt.title('Distribution of thresholded volume deltas')
    ax1.set_xlabel('Receptive field size (as voxels from center)')

    plt.subplot(2, 2, 2)
    ax2 = sns.boxplot(data=unthresholded_volume_deltas)#, label=r'unthresholded volume deltas')
    plt.title('Distribution of unthresholded volume deltas')
    ax2.set_xlabel('Receptive field size (as voxels from center)')

    plt.subplot(2, 2, 3)
    ax3 = sns.boxplot(data=image_wise_error_ratios)#, label=r'image-wise error ratio')
    plt.title('Distribution of image-wise error ratios')
    ax3.set_xlabel('Receptive field size (as voxels from center)')

    plt.subplot(2, 2, 4)
    ax4 = sns.boxplot(data=image_wise_jaccards)#, label=r'image-wise jaccard scores')
    plt.title('Distribution of image-wise Jaccard scores')
    ax4.set_xlabel('Receptive field size (as voxels from center)')

    # plt.legend(loc="upper right")

    plt.ion()
    plt.draw()
    plt.show()

def wrapper_plot_image_wise_scores_boxplot(score_dir):
    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    image_wise_jaccards = []
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
            image_wise_j = flatten(score_obj['test_image_wise_jaccards'])
            image_wise_jaccards.append(image_wise_j)

            settings_iterations.append(score_obj['settings_repeats'])
            settings_folds.append(score_obj['settings_folds'])
            print('For rf:', rf_dims[-1], 'found', settings_folds[-1], 'folds, repeated', settings_iterations[-1], 'times')

    plot_image_wise_scores_boxplot(rf_dims, thresholded_volume_deltas, unthresholded_volume_deltas, image_wise_error_ratios, image_wise_jaccards)
