import numpy as np
from config import hdf5_path, scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from preprocessing import DataPreprocessor
from model_utils import *

paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path_aug,
        'pca_path':pca_path_aug,
        'all_pprocessed_data_path':all_pprocessed_data_directory_aug,
    }

data_preprocessor = DataPreprocessor(paths,crop_augment_fold = 3)
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']

print("going into training")


'''param_grid = {
    'eta': np.arange(0.01, 1, 0.20),
    'alpha': np.arange(0, 10, 1),
}

# Fine-tune and fit
train_model(x_train, y_train, 'xgb', x_val, y_val, param_grid = param_grid)'''

#best_params = {'eta': 0.51, 'alpha': 6}
best_params_aug = {'alpha': 7, 'eta': 0.21000000000000002}
model = train_model(x_train, y_train,'xgb', params = best_params_aug)
evaluate_model(model, x_train, y_train, "training")
evaluate_model(model, x_val, y_val, "validation")

x_total = np.concatenate([x_train, x_val], axis = 0)
y_total = np.concatenate([y_train, y_val], axis = 0)
model = train_model(x_total, y_total,'xgb', params = best_params_aug)
evaluate_model(model, x_test, y_test, "test")
conf_matrix(model, x_test, y_test)



