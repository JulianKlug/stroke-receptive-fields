import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np


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

def validate(model, X_test, y_test):
    score = model.accuracy(X_test, y_test)
    print('Voxel-wise accuracy: ', score)

    y_pred = model.predict(X_test)

    jaccard = jaccard_similarity_score(y_test, y_pred)
    print('Jaccard similarity score: ', jaccard)

    roc_auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC score: ', roc_auc)

    precision = precision_score(y_test, y_pred, average=None)
    print('Precision score: ', precision)

    f1 = f1_score(y_test, y_pred)
    print('F1 score: ', f1)

    return score, roc_auc, f1
