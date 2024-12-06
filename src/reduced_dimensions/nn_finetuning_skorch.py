import numpy as np
from config import hdf5_path, scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from preprocessing import DataPreprocessor
from model_utils import *
import torch
import torch.optim as optim
from torchinfo import summary
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, make_scorer
from skorch.callbacks import EarlyStopping, Checkpoint, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from customizable_network import FCNetwork


def custom_scorer(y, y_pred):
    # y_pred is probabilities
    
    y_pred_label = np.argmax(y_pred, axis=1)
    # approximation of logits
    logits = torch.log(torch.tensor(y_pred))

    accuracy = accuracy_score(y, y_pred_label)

    validation_loss = torch.nn.CrossEntropyLoss()(logits, y.clone().detach()).item()
    print("Validation loss: ", validation_loss)

    return accuracy

acc_val_loss_scorer = make_scorer(custom_scorer, response_method ="predict_proba" ,greater_is_better=True)


lr_scheduler_callback = LRScheduler(
    policy=ReduceLROnPlateau,
    monitor='valid_loss', 
    patience=5,          
    factor=0.7,           
    threshold=0.00001,     
    min_lr=1e-6          
)

best_model_path = './saved_models/best_model_early_nn.pkl'
save_best_callback = Checkpoint(
            monitor='valid_loss_best',
            f_params=best_model_path,
            f_optimizer=None,
            f_criterion=None,
            f_history=None, 
            load_best=True
)

paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path_aug,
        'pca_path':pca_path_aug,
        'all_pprocessed_data_path':all_pprocessed_data_directory_aug,
    }

data_preprocessor = DataPreprocessor(paths,crop_augment_fold = 1)
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)


x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
valid_dataset = Dataset(x_val_tensor, y_val_tensor)


# Combine train and validation data for GridSearchCV compatibility
x_combined = np.concatenate([x_train, x_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)
x_combined_tensor = torch.tensor(x_combined, dtype=torch.float32).to(device)
y_combined_tensor = torch.tensor(y_combined, dtype=torch.long).to(device)

split_index = [-1] * len(x_train) + [0] * len(x_val)
predefined_split_grid_search = PredefinedSplit(test_fold=split_index)

net = NeuralNetClassifier(
    FCNetwork,
    module__X=x_train_tensor, # for input dimensions
    module__y=y_train_tensor, # for output dimensions
    max_epochs=200,
    lr=1e-4,
    iterator_train__shuffle=True,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    batch_size=64,
    callbacks=[
        ('early_stopping', EarlyStopping(monitor='valid_loss', lower_is_better=True)),
        ('lr_scheduler', lr_scheduler_callback),
        ('checkpoint', save_best_callback),
        ],  
    verbose=0,
    train_split=predefined_split(valid_dataset),
)

"""params = {
    'module__dims': ["1024-256", "1024-512-256"],
    'module__dropout_rate': [0.3, 0.4],
    'callbacks__early_stopping__patience': [10, 15],
    'callbacks__lr_scheduler__patience': [6,8],
}"""

# best out of gridsearch
params = {'callbacks__early_stopping__patience': [10], 'callbacks__lr_scheduler__patience': [8], 'module__dims': ['1024-256'], 'module__dropout_rate': [0.4]}


gs = GridSearchCV(net, params, refit=True, cv=predefined_split_grid_search, scoring=acc_val_loss_scorer, verbose=3, error_score="raise")

gs.fit(x_combined_tensor, y_combined_tensor)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))


x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

evaluate_model(gs.best_estimator_, x_test_tensor, y_test_tensor, "testing", logging=True)
conf_matrix(gs.best_estimator_, x_test_tensor, y_test_tensor)
