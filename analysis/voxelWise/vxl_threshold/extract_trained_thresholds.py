import os, torch, sys
sys.path.insert(0, '../')
import numpy as np
import pandas as pd

def extract_trained_thresholds(main_result_dir):
    threshold_df_columns = ['model', 'trained_threshold']
    threshold_df = pd.DataFrame([], columns=threshold_df_columns)

    result_folders = [o for o in os.listdir(main_result_dir)
                            if os.path.isdir(os.path.join(main_result_dir,o))]
    for model_result_folder in result_folders:
        trained_model_files = [o for o in os.listdir(os.path.join(main_result_dir, model_result_folder))
                            if o.startswith('trained_model') and o.endswith('.npy')]
        if not trained_model_files: continue
        trained_model_file = trained_model_files[0]
        trained_models = torch.load(os.path.join(main_result_dir, model_result_folder, trained_model_file))
        trained_threshold = np.median([model.train_threshold for model in trained_models])
        threshold_df = threshold_df.append({
            'model': model_result_folder,
            'trained_threshold': trained_threshold
        }, ignore_index = True)
    threshold_df.to_excel(os.path.join(main_result_dir, 'trained_thresholds_df.xlsx'))

extract_trained_thresholds('/Users/julian/master/all2016_results')