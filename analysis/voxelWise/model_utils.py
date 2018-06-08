import os, timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
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

    # Balancing the data for training
    X_train, y_train = balance(X_retained, y_retained)


    # Train the model using the training sets
    model.fit(X_train, y_train)

    print('Saving model as : ', model_path)
    joblib.dump(model, model_path)
    model_name_pure = model_name.split('.')[0]

    return model

def train_CV(data_dir):
    model = XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')
    train_patient_wise_kfold_cv(model, data_dir)

def evaluate_model(data_dir, model_dir, model_name, input_data_array, output_data_array, receptive_field_dimensions):
    model_path = os.path.join(model_dir, model_name)

    model = XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')

    results = test_patient_wise_kfold_cv(data_dir, receptive_field_dimensions)
    # results = repeated_kfold_cv(model, input_data_array, output_data_array, receptive_field_dimensions)

    accuracy = np.median(results['test_accuracy'])
    roc_auc = np.median(results['test_roc_auc'])
    f1 = np.median(results['test_f1'])

    print('Voxel-wise accuracy: ', accuracy)
    print('ROC AUC score: ', roc_auc)
    print('F1 score: ', f1)

    model_name_pure = model_name.split('.')[0]
    np.save(os.path.join(model_dir, model_name_pure + '_' + str(receptive_field_dimensions[0]) + '_cv_scores.npy'), results)

    return accuracy, roc_auc, f1

def repeated_kfold_cv(model, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    """
    Patient wise Repeated KFold Crossvalidation
    Advantage over sklearns implementation: returns TP and FP rates (useful for plotting ROC curve)

    Args:
        model: model to crossvalidate (must implement sklearns interface)
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folfs in kfold (ie. k)

    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """

    print('Crossvalidation: ' + str(n_folds) +' fold, repeated : ' + str(n_repeats))
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []

    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration += 1
        print('Crossvalidation: Running ' + str(iteration) + ' of a total of ' + str(n_repeats))

        f = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(X, y):
            print('Evaluating split : ' + str(f))
            print('spit', X[train].shape, y[train].shape)

            start = timeit.default_timer()
            rf_inputs, rf_outputs = rf.reshape_to_receptive_field(X[train], y[train], receptive_field_dimensions)
            end = timeit.default_timer()
            print('Reshaped to receptive fields in: ', end - start)

            X_train, y_train = balance(rf_inputs, rf_outputs)

            # Reduce amount of data initially processed
            remaining_fraction = 0.3
            print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
            X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 1-remaining_fraction, random_state = 42)

            fittedModel = model.fit(X_retained, y_retained)

            X_test, y_test = rf.reshape_to_receptive_field(X[test], y[test], receptive_field_dimensions)

            probas_= fittedModel.predict_proba(X_test)
            # Compute ROC curve, area under the curve, f1, and accuracy
            threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            accuracies.append(accuracy_score(y_test, probas_[:, 1] > threshold))
            f1_scores.append(f1_score(y_test, probas_[:, 1] > threshold))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(tpr)
            fprs.append(fpr)
            f += 1

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'model': model,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
    }

def save_patient_wise_kfold_data_split(save_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(save_dir, 'iteration_' + str(iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)

        iteration += 1
        print('Crossvalidation: Running ' + str(iteration) + ' of a total of ' + str(n_repeats))

        f = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(X, y):
            print('Evaluating split : ' + str(f))

            start = timeit.default_timer()
            rf_inputs, rf_outputs = rf.reshape_to_receptive_field(X[train], y[train], receptive_field_dimensions)
            end = timeit.default_timer()
            print('Reshaped to receptive fields in: ', end - start)

            X_train, y_train = balance(rf_inputs, rf_outputs)
            X_test, y_test = X[test], y[test]
            np.savez_compressed(os.path.join(iteration_dir, 'fold_' + str(f)),
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test )

            f += 1

def train_patient_wise_kfold_cv(model, data_dir):

    iterations = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o)) and o.startswith('iteration')]

    for iteration in iterations:
        print(iteration)
        iteration_dir = os.path.join(data_dir, iteration)

        folds = [o for o in os.listdir(iteration_dir)
                        if o.startswith('fold')]
        for fold in folds:
            print('Training', fold, 'of', iteration)
            X_train = np.load(os.path.join(iteration_dir, fold))['X_train']
            y_train = np.load(os.path.join(iteration_dir, fold))['y_train']

            # Reduce amount of data initially processed
            remaining_fraction = 0.3
            print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
            X_retained, X_rest, y_retained, y_rest = train_test_split(X_train, y_train, test_size = 1-remaining_fraction, random_state = 42)

            trained_model = model.fit(X_retained, y_retained)

            fold_name_pure = fold.split('.')[0]
            model_extension = '.pkl'
            model_path = os.path.join(iteration_dir, 'model_' + fold_name_pure + model_extension)
            print('Saving model as : ', model_path)
            joblib.dump(trained_model, model_path)

def test_patient_wise_kfold_cv(data_dir, receptive_field_dimensions):
    print('Testing Crossvalidation')
    n_folds = 0;
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []

    iterations = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o)) and o.startswith('iteration')]
    n_repeats = len(iterations)

    for iteration in iterations:
        print(iteration)
        iteration_dir = os.path.join(data_dir, iteration)

        folds = [o for o in os.listdir(iteration_dir)
                        if o.startswith('fold')]
        n_folds = len(folds)
        for fold in folds:
            print('Testing', fold, 'of', iteration)
            X_test = np.load(os.path.join(iteration_dir, fold))['X_test']
            y_test = np.load(os.path.join(iteration_dir, fold))['y_test']
            fold_name_pure = fold.split('.')[0]
            model_extension = '.pkl'
            model = joblib.load(os.path.join(iteration_dir, 'model_' + fold_name_pure + model_extension))

            X_test, y_test = rf.reshape_to_receptive_field(X_test, y_test, receptive_field_dimensions)

            probas_= model.predict_proba(X_test)
            # Compute ROC curve, area under the curve, f1, and accuracy
            threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            accuracies.append(accuracy_score(y_test, probas_[:, 1] > threshold))
            f1_scores.append(f1_score(y_test, probas_[:, 1] > threshold))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(tpr)
            fprs.append(fpr)

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'model': model,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
    }


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
