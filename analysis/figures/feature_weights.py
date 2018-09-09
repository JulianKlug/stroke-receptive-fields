import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

score_path = '/Users/julian/master/server_output/mixedRfHyperopt/scores_repeat20_rf_hyperopt_4.npy'

def get_feature_weights(path):
    model = torch.load(path)['trained_model']

    scores = model.get_score(importance_type='gain')

    rf = 4
    width = 2 * rf + 1
    n_features = width ** 3 * 4

    Tmax_score = 0
    CBF_score = 0
    MTT_score = 0
    CBV_score = 0
    score_total = 0

    scores_array = np.empty([n_features])

    # This is based on the fact that the loading order is Tmax, CBF, MTT, CBV
    for i in range(n_features):
        j = i + 1
        index = 'f' + str(j)
        try:
            score = scores[index]
            score_total += score
            scores_array[i] = score

            if (j % 4 == 0):
                CBV_score += score
            elif (j % 4 == 1):
                Tmax_score += score
            elif (j % 4 == 2):
                CBF_score += score
            elif (j % 4 == 3):
                MTT_score += score
            pass
        except KeyError as e:
            print('Feature does not exist', e)
            scores_array[i] = 0

    score_names = ['Tmax', 'MTT', 'CBV', 'CBF']
    scores = [Tmax_score/score_total, MTT_score/score_total, CBV_score/score_total, CBF_score/score_total]

    scores_matrix = scores_array.reshape(width, width, width, 4)
    return (score_names, scores, scores_matrix)

def plot_weights_3D(scores_4D):
    scores_3D = scores_4D.sum(axis = 3) #, keepdims = True)

    def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap='jet', origin="lower")

    n_i, n_j, n_k = scores_3D.shape
    center_i = (n_i - 1) // 2  # // for integer division
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    print('Image center: ', center_i, center_j, center_k)
    center_vox_value = scores_3D[center_i, center_j, center_k]
    print('Image center value: ', center_vox_value)

    slice_0 = scores_3D[center_i, :, :]
    slice_1 = scores_3D[:, center_j, :]
    slice_2 = scores_3D[:, :, center_k]

    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for weight distribution")

def plot_parameters(score_names, scores):
    plt.figure()
    sns.barplot(x = score_names, y = scores, palette=sns.cubehelix_palette(4, start=0.7, rot=-.75))
    # axes formatting
    # plt.ylim(120, 160)
    sns.set()
    sns.set_style("whitegrid")
    # sns.axes_style('white')
    # sns.set_style('white')
    # sns.despine(ax=ax, bottom=True, left=False)

    plt.title('Parameter Weights')
    plt.ylabel('Relative gain')
    # plt.legend(loc="upper right")


score_names, scores, scores_matrix = get_feature_weights(score_path)
plot_parameters(score_names, scores)
# plot_weights_3D(scores_matrix)

plt.ion()
plt.draw()
plt.show()
