import os, torch, math, random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
import pandas as pd
import scipy.stats as stats

def plot_train_evaluation(evals, model_name, save_plot):
    """
    Plot evaluation during training

    Args:
        evals: evals object returned by xgb
        model_name: name of the model

    Returns:
        undefined
    """
    if save_plot:
        plt.ioff()

    train_scores = pd.DataFrame([x['train']['auc'] for x in evals])
    test_scores = pd.DataFrame([x['eval']['auc'] for x in evals])

    train_scores_mean = train_scores[:].mean().values
    train_scores_std = train_scores[:].std().values
    test_scores_mean = test_scores[:].mean().values
    test_scores_std = test_scores[:].std().values

    train_size = range(train_scores_mean.size)
    test_size = range(test_scores_mean.size)
    plt.grid()

    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_size, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(test_size, train_scores_mean, '-', color="r",
             label="Training score")
    plt.plot(test_size, test_scores_mean, '-', color="g",
             label="Cross-validation score")

    plt.ylim([-0.05, 1.05])
    plt.ylabel('ROC AUC')
    plt.xlabel('iterations')
    model_dir = model_name.split('/')[:-1]
    model_name = model_name.split('/')[-1]
    plt.title(model_name)
    plt.legend(loc="lower right")

    if save_plot:
        print(model_dir, model_name, model_name.split('.')[0] + '.png')
        plt.savefig(os.path.join('/', *model_dir, model_name.split('.')[0] + '.png'))
        plt.close()
    else:
        plt.ion()
        plt.draw()
        plt.show()

def wrapper_plot_train_evaluation(score_path, save_plot = False):
    evals = torch.load(score_path)['train_evals']
    if not evals:
        return
    plot_train_evaluation(evals, score_path, save_plot)
