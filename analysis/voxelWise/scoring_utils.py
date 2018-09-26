import os, torch, math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
import scipy.stats as stats


def evaluate(probas_, y_test, n_subjects):
    # Voxel-wise statistics
    # Compute ROC curve, area under the curve, f1, and accuracy
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:])
    roc_auc = auc(fpr, tpr)
    # get optimal cutOff
    threshold = cutoff_youdens_j(fpr, tpr, thresholds)
    # threshold = 0.5 # threshold choosen to evaluate f1 and accuracy of model

    jaccard = jaccard_similarity_score(y_test, probas_[:] > threshold)
    accuracy = accuracy_score(y_test, probas_[:] > threshold)
    f1 = f1_score(y_test, probas_[:] > threshold)

    # Image-wise statistics
    # y_test and probas need to be in order of subjects
    image_wise_probas = probas_.reshape(n_subjects, -1)
    image_wise_y_test = y_test.reshape(n_subjects, -1)

    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    image_wise_jaccards = []

    for subj in range(n_subjects):
        # Volume delta is defined as GT - predicted volume
        thresholded_volume_deltas.append(image_wise_y_test[subj] - np.sum(image_wise_probas[subj] > threshold))
        unthresholded_volume_deltas.append(image_wise_y_test[subj] - np.sum(image_wise_probas[subj]))
        n_voxels = image_wise_y_test[subj].shape[0]
        # error ration being defined as sum(FP + FN)/all
        image_wise_error_ratios.append(
            np.sum(abs(image_wise_y_test[subj] - (image_wise_probas[subj] > threshold))) / n_voxels
        )
        image_wise_jaccards.append(jaccard_similarity_score(y_test, probas_[:] > threshold))


    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'accuracy': accuracy,
        'jaccard': jaccard,
        'f1': f1,
        'roc_auc': roc_auc,
        'thresholded_volume_deltas': thresholded_volume_deltas,
        'unthresholded_volume_deltas': unthresholded_volume_deltas,
        'image_wise_error_ratios': image_wise_error_ratios,
        'image_wise_jaccards': image_wise_jaccards
        }

def cutoff_youdens_j(fpr, tpr, thresholds):
    j_scores = tpr-fpr # J = sensivity + specificity - 1
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]
