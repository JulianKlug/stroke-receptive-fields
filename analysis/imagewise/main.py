import os, sys
sys.path.insert(0, '../')
import data_loader
from evaluation_framework import train_test_evaluation
from models.TwoLayerNetwork import TwoLayerNetwork
from models.SegNet import SegNet

data_dir = '/Users/julian/stroke_research/data/all2016_subset_prepro'
save_dir = os.path.join(data_dir, 'imagewise_models')

CLIN, IN, OUT, MASKS, IDS, PARAMS = data_loader.load_saved_data(data_dir)

print(IN.shape)
model = SegNet(IN[0].shape)

experiment_prefix = 'test'

config = {
    'pre_smoothing': False,
}


train_test_evaluation(experiment_prefix, model, IN, OUT, MASKS, IDS, save_dir, config)
