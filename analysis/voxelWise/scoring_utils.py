import os, torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np


def plot_auc_roc(rf_dims, roc_auc_scores):
    """
    Plot roc_auc for each value of rf (receptive field dimension)

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
            mean_roc_auc_scores.append( sum(roc_auc_scores[i]) / float(len(roc_auc_scores[i])) )
            mean_rf_dims.append(rf_dims[i])

        for j in range(len(roc_auc_scores[i])):
            print(rf_dims[i], roc_auc_scores[i][j])
            plt.plot(rf_dims[i], roc_auc_scores[i][j], 'k.', lw=1, alpha=0.3)

    plt.plot(mean_rf_dims, mean_roc_auc_scores)

    plt.ylim([-0.05, 1.05])
    # plt.xlim([-0.05, 5.05])
    plt.ylabel('ROC AUC')
    plt.xlabel('Receptive field size (as voxels from center)')
    plt.title('Area under the ROC curve')
    # plt.legend(loc="lower right")

    plt.ion()
    plt.draw()
    plt.show()

def wrapper_plot_auc_roc(score_dir):
    roc_auc_scores = []
    rf_dims = []
    score_paths = []
    files = os.listdir(score_dir)
    for file in files:
        if (file.startswith('scores_rf_hyperopt_')):
            score_path = os.path.join(score_dir, file)
            rf_dims.append(file.split('_')[-1].split('.')[0])
            roc_auc_scores.append(torch.load(score_path)['test_roc_auc'])
    plot_auc_roc(rf_dims, roc_auc_scores)

def plot_roc(tprs, fprs):
    """
    Plot ROC curves

    Args:
        tprs: list of true positive rates
        fprs: list of false positive rates

    Returns:
        undefined
    """
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        aucs.append(roc_auc)
        tprs_interp.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=0.5, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.2, alpha=.8)

    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensibility (True Positive Rate)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.ion()
    plt.draw()
    plt.show()

def validate(y_pred, y_test):
    threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model

    jaccard = jaccard_similarity_score(y_test, y_pred > threshold)
    print('Jaccard similarity score: ', jaccard)

    roc_auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC score: ', roc_auc)

    accuracy = accuracy_score(y_test, y_pred > threshold)
    print('Accuracy score: ', accuracy)

    f1 = f1_score(y_test, y_pred > threshold)
    print('F1 score: ', f1)

    return accuracy, roc_auc, f1
