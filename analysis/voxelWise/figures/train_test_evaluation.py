import os, torch, math, random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
import scipy.stats as stats

def plot_train_evaluation(evals, model_name):
    """
    Plot evaluation during training

    Args:
        evals: evals object returned by xgb
        model_name: name of the model

    Returns:
        undefined
    """

    for i in range(len(evals)):
        train_score = evals[i]['train']['auc']
        test_tr_score = evals[i]['eval']['auc']
        iterations = np.array(range(len(train_score)))

        plt.plot(iterations, train_score, label=r'train score for fold %i' % (i))
        plt.plot(iterations, test_tr_score, label=r'test score for fold %i' % (i))

    plt.ylim([-0.05, 1.05])
    plt.ylabel('ROC AUC')
    plt.xlabel('iterations')
    plt.title(model_name)
    plt.legend(loc="lower right")

    plt.ion()
    plt.draw()
    plt.show()

def wrapper_plot_train_evaluation(score_path):
    evals = torch.load(score_path)['train_evals']
    plot_train_evaluation(evals, score_path)
