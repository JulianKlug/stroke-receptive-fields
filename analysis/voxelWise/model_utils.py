import os
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score
from collections import Counter
import receptiveField as rf

def create(model_dir, model_name, input_data_list, output_data_list, receptive_field_dimensions):
    model_path = os.path.join(model_dir, model_name)
    rf_inputs, rf_outputs = rf.reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)

    # Create model object
    # model = linear_model.LogisticRegression(verbose = 1, max_iter = 1000000000)
    # model = RandomForestClassifier(verbose = 1)
    model = XGBClassifier(verbose_eval=True)

    # Reduce amount of data initially processed
    remaining_fraction = 0.3
    print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
    X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 0.7, random_state = 42)
    rf_inputs = X_retained
    rf_outputs = y_retained

    # Split off the test set
    # X_train, X_test, y_train, y_test = train_test_split(rf_inputs, rf_outputs, test_size=0.33, random_state=42)
    # X_train, y_train = balance(X_train, y_train)

    X_train = rf_inputs
    y_train = rf_outputs

    # Train the model using the training sets
    model.fit(X_train, y_train)

    print('Saving model as : ', model_path)
    joblib.dump(model, model_path)
    model_name_pure = model_name.split('.')[0]
    # np.save(os.path.join(model_dir, model_name_pure + '_X_test.npy'), X_test)
    # np.save(os.path.join(model_dir, model_name_pure + '_Y_test.npy'), y_test)

    return model

def evaluate_model(model_dir, model_name, input_data_list, output_data_list, receptive_field_dimensions):
    model_path = os.path.join(model_dir, model_name)
    rf_inputs, rf_outputs = rf.reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)

    model = XGBClassifier(verbose_eval=True)

    # Reduce amount of data initially processed
    remaining_fraction = 0.3
    print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
    X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 0.7, random_state = 42)
    X = X_retained
    y = y_retained

    print('Using Crossvalidation')
    kf = RepeatedKFold(n_splits = 5, n_repeats = 100, random_state = 42)
    scoring = ('accuracy', 'roc_auc', 'f1')
    results = cross_validate(model, X, y, cv = kf, n_jobs = 20, scoring = scoring)

    accuracy = np.median(results['test_accuracy'])
    roc_auc = np.median(results['test_roc_auc'])
    f1 = np.median(results['test_f1'])

    print('Voxel-wise accuracy: ', accuracy)
    print('ROC AUC score: ', roc_auc)
    print('F1 score: ', f1)

    model_name_pure = model_name.split('.')[0]
    np.save(os.path.join(model_dir, model_name_pure + '_cv_scores.npy'), results)

    return accuracy, roc_auc, f1

def validate(model, X_test, y_test):
    # The coefficients
    # print('Coefficients: \n', model.coef_)
    # print('Intercept: \n', model.intercept_)

    # Use score method to get accuracy of model
    score = model.score(X_test, y_test)
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

def balance(X, y):
    # from imblearn.under_sampling import RandomUnderSampler
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_sample(X, y)

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    print('Balancing Data.')
    print('Remaining data points after balancing: ', sorted(Counter(y_resampled).items()))
    return (X_resampled, y_resampled)
