import os, torch, math, random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
import scipy.stats as stats

def plot_dice(rf_dims, dice_scores, model_name = 'model', color = 'C0', display_legend = True):
    """
    Plot dice score for each value of rf (receptive field dimension)

    Args:
        rf_dims: list of receptiveField dimensions
        dice_scores: list of dice_scores coresponding to the rf_dim at the same index

    Returns:
        undefined
    """
    z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*

    mean_dice_scores = []
    mean_rf_dims = []
    auc_upper_limits = []
    auc_lower_limits = []

    # print(rf_dims, dice_scores)
    dice_scores = [x for _,x in sorted(zip(rf_dims, dice_scores))]
    rf_dims.sort()

    for i in range(len(rf_dims)):
        rf_dice_scores = [item for sublist in dice_scores[i] for item in sublist]
        if len(rf_dice_scores) != 0:
            median_dice = np.median(rf_dice_scores)
            mean_dice = sum(rf_dice_scores) / float(len(rf_dice_scores))
            mean_dice_scores.append(mean_dice)
            mean_rf_dims.append(rf_dims[i])

            std_dice = np.std(rf_dice_scores, axis=0)
            margin_of_error = z_critical * (std_dice/math.sqrt(len(rf_dice_scores)))

            print(i, median_dice, mean_dice, std_dice, margin_of_error)
            # auc_upper_limits.append(np.minimum(mean_dice_scores[i] + margin_of_error, 1))
            auc_upper_limits.append(mean_dice_scores[i] + margin_of_error)
            auc_lower_limits.append(np.maximum(mean_dice_scores[i] - margin_of_error, 0))

        for j in range(len(rf_dice_scores)):
            plt.plot(rf_dims[i], rf_dice_scores[j], 'k.', lw=1, alpha=0.3)

    print(model_name)
    print('means', mean_dice_scores)
    print('Rf used:', mean_rf_dims)
    print('Using Z:', z_critical)
    print('low', auc_lower_limits)
    print('up', auc_upper_limits)

    if (display_legend):
        # Plot one additional point to have only one label
        plt.plot(0, 2, 'k.', lw=1, alpha=0.3, label=r'Dice score')
        plt.fill_between(mean_rf_dims, auc_upper_limits, auc_lower_limits, color='grey', alpha=.2,
                         # label=r'$\pm$ 1 std. dev.')
                         label=r'$\pm$ 1 std. err.')
    else:
        plt.fill_between(mean_rf_dims, auc_upper_limits, auc_lower_limits, color='grey', alpha=.2)

    plt.plot(mean_rf_dims, mean_dice_scores, color, label=r'Mean Dice for %s' % (model_name))
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Dice Coefficients')
    plt.xlabel('Receptive field size (as voxels from center)')
    plt.title('Dice')
    # plt.legend(loc="lower right")

    plt.ion()
    plt.draw()

def wrapper_plot_dice(score_dir, model_name, color = 'C0', display_legend = True):
    dice_scores = []
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
            dice_scores.append(score_obj['test_image_wise_dice'])
    plot_dice(rf_dims, dice_scores, model_name, color, display_legend)
    # plt.show()

def compare(dir1, dir2, dir3, dir4, dir5):
    fig, ax = plt.subplots()
    wrapper_plot_dice(dir1, 'multi-parameter glm', 'C0')
    wrapper_plot_dice(dir2, 'MTT glm', 'C1', display_legend = False)
    wrapper_plot_dice(dir3, 'Tmax glm', 'C2', display_legend = False)
    wrapper_plot_dice(dir4, 'xgb', 'C3', display_legend = False)
    wrapper_plot_dice(dir5, 'def xgb', 'C4', display_legend = False)
    plt.legend(loc="upper right")
    plt.title('CV Framework')
    plt.show()
