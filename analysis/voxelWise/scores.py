import sys
sys.path.insert(0, '../')

import os
import numpy as np
from sklearn.externals import joblib
import model_utils

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'analysis_test_LOO')
model_dir = os.path.join(data_dir, 'models')

# Path to load the model from
model_name = 'LR1.pkl'
model_path = os.path.join(model_dir, model_name)
trained_model = joblib.load(model_path)

model_name_pure = model_name.split('.')[0]
X_test = np.load(os.path.join(model_dir, model_name_pure + '_X_test.npy'))
y_test = np.load(os.path.join(model_dir, model_name_pure + '_Y_test.npy'))

print('Stats for:', model_name)
model_utils.stats(trained_model, X_test, y_test)
