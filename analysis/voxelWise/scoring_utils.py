import os, torch, math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
import numpy as np
import scipy.stats as stats

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
