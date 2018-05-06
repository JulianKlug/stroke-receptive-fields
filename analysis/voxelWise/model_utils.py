import os
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score
from collections import Counter
import receptiveField as rf

def create(model_dir, model_name, input_data_list, output_data_list, receptive_field_dimensions):
    model_path = os.path.join(model_dir, model_name)
    rf_inputs, rf_outputs = rf.reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)

    # Create linear regression object
    model = linear_model.LogisticRegression(verbose = 1, max_iter = 1000000000)
    # model = RandomForestClassifier(verbose = 1)

    X_train, X_test, y_train, y_test = train_test_split(rf_inputs, rf_outputs, test_size=0.33, random_state=42)
    X_train, y_train = balance(X_train, y_train)

    # Train the model using the training sets
    model.fit(X_train, y_train)

    print('Saving model as : ', model_path)
    joblib.dump(model, model_path)
    model_name_pure = model_name.split('.')[0]
    np.save(os.path.join(model_dir, model_name_pure + '_X_test.npy'), X_test)
    np.save(os.path.join(model_dir, model_name_pure + '_Y_test.npy'), y_test)

    return model, X_test, y_test

def stats(model, X_test, y_test):
    # The coefficients
    print('Coefficients: \n', model.coef_)
    print('Intercept: \n', model.intercept_)

    # Use score method to get accuracy of model
    score = model.score(X_test, y_test)
    print('Voxel-wise accuracy: ', score)

    y_pred = model.predict(X_test)

    jaccard = jaccard_similarity_score(y_test, y_pred)
    print('Jaccard similarity score: ', jaccard)

    roc_auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC score: ', roc_auc)

    f1 = f1_score(y_test, y_pred)
    print('F1 score: ', f1)

def balance(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)

    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X, y)
    print('Balancing Data.')
    print('Remaining data points after balancing: ', sorted(Counter(y_resampled).items()))
    return (X_resampled, y_resampled)
