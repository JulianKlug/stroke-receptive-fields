import os
import xgboost as xgb

params = {
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bytree': 1,
    'eval_metric': 'auc',
    'gamma': 0.84,
    'learning_rate': 0.4,
    'max_delta_step': 0,
    'max_depth': 1,
    'min_child_weight': 10.0,
    'n_estimators': 999,
    'n_jobs': -1,
    'objective': 'binary:logistic',
    'random_state': 0,
    'reg_alpha': 0.8671463346569078,
    'reg_lambda': 0.5916603334004378,
    'scale_pos_weight': 1,
    'silent': True,
    'subsample': 0.7925462136041614,
    'tree_method': 'hist'
}

n_estimators = params['n_estimators']


iteration_dir = ''
ext_mem_extension = '.txt'
fold = 4

fold_dir = os.path.join(iteration_dir, fold)

dtrain = xgb.DMatrix(os.path.join(fold_dir, fold + '_train' + ext_mem_extension)
    + '#' + os.path.join(fold_dir, 'dtrain.cache'))
dtest = xgb.DMatrix(os.path.join(fold_dir, fold + '_test' + ext_mem_extension)
    + '#' + os.path.join(fold_dir, 'dtest.cache'))

trained_model = xgb.train(params, dtrain,
    num_boost_round = n_estimators,
    evals = [(dtest, 'Test')],
    early_stopping_rounds = 30,
    verbose_eval = False)
