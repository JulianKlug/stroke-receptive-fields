import os
import xgboost as xgb
from vxl_xgboost.xgb_params import XGB_PARAMS
from ext_mem_utils import save_to_svmlight


class External_Memory_xgb():
    """
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM
    """
    def __init__(self, fold_dir, fold_name):
        super(External_Memory_xgb, self).__init__()
        self.params = XGB_PARAMS
        self.n_estimators = self.params['n_estimators']
        self.evals_result = {}
        self.trained_model = None
        self.dtrain = None
        self.dtest = None

        self.ext_mem_extension = '.txt'
        self.fold_dir = fold_dir
        self.fold_name = fold_name

    @staticmethod
    def hello_world():
        print('External Memory XGB Model')
        print(XGB_PARAMS)
        print('Attention! Folds will not be saved.')

    @staticmethod
    def get_settings():
        return XGB_PARAMS

    def add_train_data(self, batch_X_train, batch_y_train):
        """
        Add a batch of training data to the whole training data pool
        All training data is saved in a svmlight file

        Args:
            batch_X_train: batch of training data
            batch_y_train: batch of training labels
        """
        train_data_path = os.path.join(self.fold_dir, 'fold_' + str(self.fold_name) + '_train' + self.ext_mem_extension)
        save_to_svmlight(batch_X_train, batch_y_train, train_data_path)

    def add_test_data(self, batch_X_test, batch_y_test):
        """
        Add a batch of testing data to the whole testing data pool
        All testing data is saved in a svmlight file
        """
        test_data_path = os.path.join(self.fold_dir, 'fold_' + str(self.fold_name) + '_test' + self.ext_mem_extension)
        save_to_svmlight(batch_X_test, batch_y_test, test_data_path)

    def train(self):
        self.dtrain = xgb.DMatrix(os.path.join(self.fold_dir, 'fold_' + str(self.fold_name) + '_train' + self.ext_mem_extension)
            + '#' + os.path.join(self.fold_dir, 'dtrain.cache'))
        self.dtest = xgb.DMatrix(os.path.join(self.fold_dir, 'fold_' + str(self.fold_name) + '_test' + self.ext_mem_extension)
            + '#' + os.path.join(self.fold_dir, 'dtest.cache'))

        self.trained_model = xgb.train(self.params, self.dtrain,
            num_boost_round = self.n_estimators,
            evals = [(self.dtest, 'eval'), (self.dtrain, 'train')],
            early_stopping_rounds = 30,
            evals_result = self.evals_result,
            verbose_eval = False)

        # Clean up cache files
        try:
            os.remove(os.path.join(self.fold_dir, 'dtrain.r0-1.cache'))
            os.remove(os.path.join(self.fold_dir, 'dtrain.r0-1.cache.row.page'))
            os.remove(os.path.join(self.fold_dir, 'dtest.r0-1.cache'))
            os.remove(os.path.join(self.fold_dir, 'dtest.r0-1.cache.row.page'))
        except:
            print('No cache to clear.')

        return self.trained_model, self.evals_result

    def predict(self, data):
        probas_ = self.trained_model.predict(data,
                    ntree_limit = self.trained_model.best_ntree_limit)
        return probas_

    def predict_test_data(self):
        probas_ = self.predict(self.dtest)
        return probas_

    def get_test_labels(self):
        y_test = self.dtest.get_label()
        return y_test
