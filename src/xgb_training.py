import numpy as np
from config import hdf5_path, scaler_path, pca_path_alt, all_pprocessed_data_directory ,scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from preprocessing import DataPreprocessor
from model_utils import *

paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path,
        'pca_path':pca_path_alt,
        'all_pprocessed_data_path':all_pprocessed_data_directory,
    }

data_preprocessor = DataPreprocessor(paths,crop_augment_fold = 1)
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']

'''x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)'''

print(y_train)
print("going into training")


param_grid = {
    'eta': np.arange(0.01, 1, 0.20),
    'alpha': np.arange(0, 10, 2),
}

# Fine-tune and fit
# loo validation
#train_model(x_train, y_train, 'xgb', x_val, y_val, param_grid = param_grid)
# kfold
#svm = train_model(x_train, y_train, 'xgb', param_grid = param_grid, folds=5)

# non-augmented loo validation
best_params = {'eta': 0.51, 'alpha': 6}
# kfold
#best_params = {'alpha': 0, 'eta': 0.21000000000000002}
# augmented loo validation
#best_params_aug = {'alpha': 7, 'eta': 0.21000000000000002}
model = train_model(x_train, y_train,'xgb', params = best_params)
evaluate_model(model, x_train, y_train, "training")
evaluate_model(model, x_val, y_val, "validation")

x_total = np.concatenate([x_train, x_val], axis = 0)
y_total = np.concatenate([y_train, y_val], axis = 0)
model = train_model(x_total, y_total,'xgb', params = best_params)
evaluate_model(model, x_test, y_test, "test")
conf_matrix(model, x_test, y_test)



