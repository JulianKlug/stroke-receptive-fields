def ext_mem_repeated_kfold_cv(params, data_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5, create_folds = False, save_folds = True, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        data_dir: directory to use for saving the intermittent states
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folds in kfold (ie. k)
        create_folds (option, dafault False): boolean, if the folds should be created anew
        save_folds (optional, default True): boolean, if the created folds should be saved
        messaging (optional, defaults to None): instance of notification_system used to report errors

    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': params of the model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """
    print('Using external memory version for Crossvalidation')
    print('Using params:', params)

    if create_folds:
        if not save_folds:
            results, trained_models = ext_mem_continuous_repeated_kfold_cv(params, data_dir, X, y, receptive_field_dimensions, n_repeats, n_folds, messaging)
            results['rf'] = receptive_field_dimensions
            return (results, trained_models)

        external_save_patient_wise_kfold_data_split(data_dir, X, y, receptive_field_dimensions, n_repeats, n_folds)

    results, trained_models = external_evaluation_wrapper_patient_wise_kfold_cv(params, n_test_subjects, data_dir)

    results['rf'] = receptive_field_dimensions

    return (results, trained_models)


def ext_mem_continuous_repeated_kfold_cv(params, save_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    This function creates and evaluates k datafolds of n-iterations for crossvalidation,
    BUT erases the saved data after every fold evaluation
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        data_dir: directory to use for saving the intermittent states
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folds in kfold (ie. k)
        messaging (optional, defaults to None): instance of notification_system used to report errors


    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': params of the model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """
    print('CONTINOUS REPEATED KFOLD CV')
    print('Attention! Folds will not be saved.')

    # Initialising variables for evaluation
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []
    jaccards = []
    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    image_wise_jaccards = []
    trained_models = []
    train_evals = []
    failed_folds = 0

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
        print('This directory already exists: ', save_dir)
        validation = input('Type `yes` if you wish to delete your previous data:\t')
        if (validation != 'yes'):
            raise ValueError('Model already exists. Choose another model name or delete current model')
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    print('Saving repeated kfold data split to libsvm format', n_repeats, n_folds)

    ext_mem_extension = '.txt'
    start = timeit.default_timer()
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(save_dir, 'iteration_' + str(iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)

        iteration += 1
        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(X, y):
            print('Creating fold : ' + str(fold))
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            X_train, y_train = X[train], y[train]

            # Get balancing selector respecting population wide distribution
            balancing_selector = get_undersample_selector_array(y_train)
            all_subj_X_train = np.empty([np.sum(balancing_selector), receptive_field_size])
            all_subj_y_train = np.empty(np.sum(balancing_selector))
            all_subj_index = 0

            for subject in range(X_train.shape[0]):
                # reshape to rf expects data with n_subjects in first
                subj_X_train, subj_y_train = np.expand_dims(X_train[subject], axis=0), np.expand_dims(y_train[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)

                # Balance by using predefined balancing_selector
                subj_X_train, subj_y_train = rf_inputs[balancing_selector[subject].reshape(-1)], rf_outputs[balancing_selector[subject].reshape(-1)]

                train_data_path = os.path.join(fold_dir, 'fold_' + str(fold) + '_train' + ext_mem_extension)
                model.preload_train(subj_X_train, subj_y_train, train_data_path)
                # save_to_svmlight(subj_X_train, subj_y_train, train_data_path)
                all_subj_X_train[all_subj_index : all_subj_index + subj_X_train.shape[0], :] = subj_X_train
                all_subj_y_train[all_subj_index : all_subj_index + subj_y_train.shape[0]] = subj_y_train
                all_subj_index += subj_X_train.shape[0]

            X_test, y_test = X[test], y[test]
            n_test_subjects = X_test.shape[0]
            all_subj_X_test = np.empty([X_test.size / X_test.shape[-1], X_test.shape[-1] * (receptive_field_size[0]*2 + 1)**3])
            all_subj_y_test = np.empty(X_test.size / X_test.shape[-1])
            all_subj_index = 0
            # test_rf_inputs, test_rf_outputs = rf.reshape_to_receptive_field(X_test, y_test, receptive_field_dimensions)
            for subject in range(n_test_subjects):
                # reshape to rf expects data with n_subjects in first
                subj_X_test, subj_y_test = np.expand_dims(X_test[subject], axis=0), np.expand_dims(y_test[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_test, subj_y_test, receptive_field_dimensions)
                all_subj_X_test[all_subj_index : all_subj_index + rf_inputs.shape[0], :] = rf_inputs
                all_subj_y_test[all_subj_index : all_subj_index + rf_outputs.shape[0]] = rf_outputs
                all_subj_index += subj_X_train.shape[0]
                test_data_path = os.path.join(fold_dir, 'fold_' + str(fold) + '_test' + ext_mem_extension)
                model.preload_test(rf_inputs, rf_outputs, test_data_path)
                # save_to_svmlight(rf_inputs, rf_outputs, test_data_path)

            # Evaluate this fold
            print('Evaluating fold ' + str(fold) + ' of ' + str(n_folds - 1) + ' of iteration' + str(iteration))

            try:
                n_test_subjects = X_test.shape[0]
                fold_result = external_evaluate_fold_cv(params, n_test_subjects, fold_dir, 'fold_' + str(fold), ext_mem_extension)

                accuracies.append(fold_result['accuracy'])
                f1_scores.append(fold_result['f1'])
                aucs.append(fold_result['roc_auc'])
                tprs.append(fold_result['tpr'])
                fprs.append(fold_result['fpr'])
                jaccards.append(fold_result['jaccard'])
                thresholded_volume_deltas.append(fold_result['thresholded_volume_deltas'])
                unthresholded_volume_deltas.append(fold_result['unthresholded_volume_deltas'])
                image_wise_error_ratios.append(fold_result['image_wise_error_ratios'])
                image_wise_jaccards.append(fold_result['image_wise_jaccards'])
                train_evals.append(fold_result['train_evals'])
                trained_models.append(fold_result['trained_model'])
                pass
            except Exception as e:
                failed_folds += 1
                print('Evaluation of fold failed.')
                print(e)
                if (messaging):
                    title = 'Minor error upon rf_hyperopt at ' + str(receptive_field_dimensions)
                    tb = traceback.format_exc()
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            # Erase saved fold to free up space
            try:
                shutil.rmtree(fold_dir)
            except:
                print('No fold to clear.')

            fold += 1
            # End of fold iteration

        try:
            shutil.rmtree(iteration_dir)
        except:
            print('No iteration to clear.')
        # End of iteration iteration

    end = timeit.default_timer()
    print('Created, saved and evaluated splits in: ', end - start)

    return ({
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'failed_folds': failed_folds,
        'model_params': params,
        'train_evals': train_evals,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_jaccard': jaccards,
        'test_TPR': tprs,
        'test_FPR': fprs,
        'test_thresholded_volume_deltas': thresholded_volume_deltas,
        'test_unthresholded_volume_deltas': unthresholded_volume_deltas,
        'test_image_wise_error_ratios': image_wise_error_ratios,
        'test_image_wise_jaccards': image_wise_jaccards
    },
        trained_models
    )

def external_evaluate_fold_cv(params, n_test_subjects, fold_dir, fold, ext_mem_extension):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    This function evaluates a saved datafold
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        fold_dir: directory of the saved fold

    Returns: result dictionary
    """
    n_estimators = params['n_estimators']

    dtrain = xgb.DMatrix(os.path.join(fold_dir, fold + '_train' + ext_mem_extension)
        + '#' + os.path.join(fold_dir, 'dtrain.cache'))
    dtest = xgb.DMatrix(os.path.join(fold_dir, fold + '_test' + ext_mem_extension)
        + '#' + os.path.join(fold_dir, 'dtest.cache'))

    evals_result = {}

    trained_model = xgb.train(params, dtrain,
        num_boost_round = n_estimators,
        evals = [(dtest, 'eval'), (dtrain, 'train')],
        early_stopping_rounds = 30,
        evals_result=evals_result,
        verbose_eval = False)

    # Clean up cache files
    try:
        os.remove(os.path.join(fold_dir, 'dtrain.r0-1.cache'))
        os.remove(os.path.join(fold_dir, 'dtrain.r0-1.cache.row.page'))
        os.remove(os.path.join(fold_dir, 'dtest.r0-1.cache'))
        os.remove(os.path.join(fold_dir, 'dtest.r0-1.cache.row.page'))
    except:
        print('No cache to clear.')

    print('Testing', fold, 'in', fold_dir)
    y_test = dtest.get_label()
    probas_= trained_model.predict(dtest, ntree_limit=trained_model.best_ntree_limit)

    results = evaluate(probas_, y_test, n_test_subjects)
    results['trained_model'] = trained_model
    results['train_evals'] = evals_result

    return results
