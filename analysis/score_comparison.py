import os
import nibabel as nib
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'analysis_test_LOO')

y_true_path = os.path.join(data_dir, 'LOO/wcoreg_VOI_lesion.nii')
y_true_img = nib.load(y_true_path)
y_true = y_true_img.get_data().flatten()

predicted_name = 'xgb_Overbalanced_loo.nii'
predicted_path = os.path.join(data_dir, predicted_name)
predicted_img = nib.load(predicted_path)
y_pred = predicted_img.get_data().flatten()

print('Comparing for', predicted_name)

jaccard = jaccard_similarity_score(y_true, y_pred)
print('Jaccard similarity score: ', jaccard)

roc_auc = roc_auc_score(y_true, y_pred)
print('ROC AUC score: ', roc_auc)

precision = precision_score(y_true, y_pred, average=None)
print('Precision score: ', precision)

f1 = f1_score(y_true, y_pred)
print('F1 score: ', f1)
