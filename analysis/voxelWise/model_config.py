
n_estimators = 999  # number of trees
obj = None
feval = None
evals = ()
early_stopping_rounds = None
evals_result = {}

parameters = {
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
    'n_estimators': n_estimators,
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
