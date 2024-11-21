import numpy as np
from preprocessing import DataPreprocessor
from dummy_classifiers import Majority_Classifier, Sampling_Classifier
from config import hdf5_path, scaler_path, pca_path_alt, all_pprocessed_data_directory ,scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from model_utils import *



paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path,
        'pca_path':pca_path_alt,
        'all_pprocessed_data_path':all_pprocessed_data_directory,
    }

data_preprocessor = DataPreprocessor(paths, crop_augment_fold=1)
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']

# for k fold
'''x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)'''


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


majority_classifier = Majority_Classifier()
majority_classifier.fit(y_train)
accuracy = evaluate_model(majority_classifier, x_train, y_train, "training")
accuracy = evaluate_model(majority_classifier, x_val, y_val, "validation")

sampling_classifier = Sampling_Classifier()
sampling_classifier.fit(y_train)
accuracy = evaluate_model(sampling_classifier, x_train, y_train, "training")
accuracy = evaluate_model(sampling_classifier, x_val, y_val, "validation")

'''param_grid = {
    'alpha': [50, 100, 1000, 10000],
    'loss': ['hinge', 'log_loss'],
    'max_iter': [100, 1000, 10000],
    'random_state': list(range(0,20))
}
# tune and the fit
#svm = train_model(x_train, y_train, 'svm', x_val, y_val, param_grid = param_grid )
svm = train_model(x_train, y_train, 'svm', param_grid = param_grid )'''

# best params
# loo validation
params = {'alpha': 1000, 'loss': 'log_loss', 'max_iter': 1000, 'random_state': 4}
# kfold
#params = {'alpha': 50, 'loss': 'log_loss', 'max_iter': 10000, 'random_state': 11}
#params = {'alpha': 100, 'loss': 'log_loss', 'max_iter': 100, 'random_state': 14}
svm = train_model(x_train, y_train, 'svm',params = params )
accuracy = evaluate_model(svm, x_train, y_train, "training")
accuracy = evaluate_model(svm, x_val, y_val, "validation")

conf_matrix(svm, x_val, y_val)

x_total = np.concatenate([x_train, x_val], axis = 0)
y_total = np.concatenate([y_train, y_val], axis = 0)


svm = train_model(x_total, y_total, 'svm',params = params )
accuracy = evaluate_model(svm, x_total, y_total, "training")
accuracy = evaluate_model(svm, x_test, y_test, "test")

conf_matrix(svm, x_test, y_test)