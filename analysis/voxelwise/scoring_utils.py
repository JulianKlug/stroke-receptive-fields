import os, torch, math
from sklearn.metrics import f1_score, jaccard_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
from matplotlib import gridspec


def evaluate(probas_, y_test, mask_test, ids_test, n_subjects: int, n_x, n_y, n_z, model_threshold = 0.5):
    '''
    Evaluate performance of prediction
    :param probas_: probability of being of class 1 for every voxel - linear shape of all voxels [i]
    :param y_test: GT for every voxel [i]
    :param mask_test: mask of voxels used in original space for every subject - [n, x, y, z]
    :param ids_test: list of subj-ids [i]
    :param n_subjects: integer
    :param n_x: integer
    :param n_y: integer
    :param n_z: integer
    :param model_threshold : trained threshold used to binarize the probability
    :return:
    '''
    probas_ = np.squeeze(probas_)
    if probas_.shape != y_test.shape:
        print('PROBAS AND TEST IMAGE DO NOT HAVE THE SAME SHAPE', probas_.shape, y_test.shape)

    # Voxel-wise statistics
    # Compute ROC curve, area under the curve, f1, and accuracy
    fpr, tpr, roc_thresholds = roc_curve(y_test, probas_[:])
    roc_auc = auc(fpr, tpr)

    # threshold chosen to evaluate binary metrics of model
    threshold = model_threshold
    if np.isnan(threshold): threshold = 0.5
    print('Using threshold', str(threshold), 'for evaluation.')
    # get optimal cutOff on test data
    test_threshold = cutoff_youdens_j(fpr, tpr, roc_thresholds)
    print('Optimal threshold based on test data:', str(test_threshold))

    jaccard = jaccard_score(y_test, probas_[:] >= threshold)
    accuracy = accuracy_score(y_test, probas_[:] >= threshold)
    f1 = f1_score(y_test, probas_[:] >= threshold)
    # Positive predictive value : tp / (tp + fp)
    PPV = precision_score(y_test, probas_[:] >= threshold)

    # Image-wise statistics
    thresholded_predicted_volume_vox = []
    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    image_wise_jaccards = []
    image_wise_hausdorff = []
    image_wise_modified_hausdorff = []
    image_wise_dice = []
    image_wise_roc_auc = []
    image_wise_fpr = []
    image_wise_tpr = []
    image_wise_roc_thresholds = []
    # figure for visual evaluation

    plt.switch_backend('agg')
    ncol = 14
    nrow = 2 * (n_subjects // ncol) + 2
    figure = plt.figure(figsize=(ncol+1, nrow+1))
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.7, hspace=0.25,
             top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
             left=0.5/(ncol+1), right=1-0.5/(ncol+1))

    vxl_index = 0

    for subj in range(n_subjects):
        subj_n_vxl = np.sum(mask_test[subj])
        if ids_test is not None: subj_id = ids_test[subj]
        else : subj_id = None

        subj_image_wise_probas = probas_[vxl_index : vxl_index + subj_n_vxl]
        subj_image_wise_y_test = y_test[vxl_index : vxl_index + subj_n_vxl]
        vxl_index += subj_n_vxl

        img_fpr, img_tpr, img_roc_thresholds = roc_curve(subj_image_wise_y_test, subj_image_wise_probas[:])
        image_wise_roc_auc.append(auc(img_fpr, img_tpr))
        image_wise_fpr.append(img_fpr)
        image_wise_tpr.append(img_tpr)
        image_wise_roc_thresholds.append(img_roc_thresholds)

        # Record predicted volume (GT volume can be derived from predicted volume and delta)
        thresholded_predicted_volume_vox.append(np.sum(subj_image_wise_probas >= threshold))

        # Volume delta is defined as GT - predicted volume
        thresholded_volume_deltas.append(np.sum(subj_image_wise_y_test) - np.sum(subj_image_wise_probas >= threshold))
        unthresholded_volume_deltas.append(np.sum(subj_image_wise_y_test) - np.sum(subj_image_wise_probas))
        n_voxels = subj_image_wise_y_test.shape[0]

        # error ratio being defined as sum(FP + FN)/all
        image_wise_error_ratios.append(
            np.sum(abs(subj_image_wise_y_test - (subj_image_wise_probas >= threshold))) / n_voxels
        )
        image_wise_jaccards.append(jaccard_score(subj_image_wise_y_test, subj_image_wise_probas[:] >= threshold))
        image_wise_dice.append(dice(subj_image_wise_y_test, subj_image_wise_probas[:] >= threshold))

        # To calculate the hausdorff_distance, the image has to be rebuild as a 3D image
        subj_3D_probas = np.full(mask_test[subj].shape, 0, dtype = np.float64)
        subj_3D_probas[mask_test[subj]] = subj_image_wise_probas
        subj_3D_y_test = np.full(mask_test[subj].shape, 0)
        subj_3D_y_test[mask_test[subj]] = subj_image_wise_y_test

        visual_compare(subj_3D_y_test, subj_3D_probas, n_subjects, subj, n_z, gs, image_id = subj_id)

        hsd = hausdorff_distance(subj_3D_y_test, subj_3D_probas >= threshold, n_x, n_y, n_z)
        image_wise_hausdorff.append(hsd)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'evaluation_threshold': threshold,
        'optimal_threshold_on_test_data': test_threshold,
        'accuracy': accuracy,
        'jaccard': jaccard,
        'f1': f1,
        'roc_auc': roc_auc,
        'positive_predictive_value': PPV,
        'thresholded_predicted_volume_vox': thresholded_predicted_volume_vox,
        'thresholded_volume_deltas': thresholded_volume_deltas,
        'unthresholded_volume_deltas': unthresholded_volume_deltas,
        'image_wise_error_ratios': image_wise_error_ratios,
        'image_wise_jaccards': image_wise_jaccards,
        'image_wise_hausdorff': image_wise_hausdorff,
        'image_wise_modified_hausdorff': image_wise_modified_hausdorff,
        'image_wise_dice': image_wise_dice,
        'image_wise_roc_auc': image_wise_roc_auc,
        'image_wise_fpr': image_wise_fpr,
        'image_wise_tpr': image_wise_tpr,
        'image_wise_roc_thresholds': image_wise_roc_thresholds,
        'figure': figure
        }

def cutoff_youdens_j(fpr, tpr, thresholds):
    j_scores = tpr-fpr # J = sensivity + specificity - 1
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def hausdorff_distance(data1, data2, n_x, n_y, n_z):
    data1 = data1.reshape(n_x, n_y, n_z)
    data2 = data2.reshape(n_x, n_y, n_z)

    coordinates1 = np.array(np.where(data1 > 0)).transpose()
    coordinates2 = np.array(np.where(data2 > 0)).transpose()
    # modified_hausdorff =  ModHausdorffDist(coordinates1, coordinates2)[0]

    return directed_hausdorff(coordinates1, coordinates2)[0]

# Too calculation intensive for now
def ModHausdorffDist(A,B):
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)

# draw GT and test image on canvas
def visual_compare(GT, pred, n_images, i_image, n_z, gs, image_id = None):
    center_z = (n_z - 1) // 2
    i_line = 2 * (i_image // gs.get_geometry()[1])
    i_row = i_image % gs.get_geometry()[1]

    # plot GT image
    ax = plt.subplot(gs[i_line, i_row])
    if image_id is not None: ax.set_title(image_id, fontdict={'fontsize': 10})
    plt.imshow(-GT[:, :, center_z].T)
    plt.gca().invert_yaxis()
    plt.set_cmap('Greys')
    plt.clim(-1, 0)
    plt.axis('off')

    # plot reconstructed image
    ax = plt.subplot(gs[i_line + 1, i_row])
    plt.imshow(pred[:, :, center_z].T)
    plt.gca().invert_yaxis()
    plt.set_cmap('jet')
    plt.clim(0, 1)
    plt.axis('off')
