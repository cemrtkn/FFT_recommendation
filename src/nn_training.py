import numpy as np
from config import hdf5_path
from preprocessing import DataPreprocessor
from model_utils import *

data_preprocessor = DataPreprocessor(hdf5_path)
#data_splits = data_preprocessor.prepare_data()
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
