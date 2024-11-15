import numpy as np
from config import hdf5_path, scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from preprocessing import DataPreprocessor
from model_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score


def custom_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    
    y_pred_prob = estimator.predict_proba(X)
    validation_loss = torch.nn.CrossEntropyLoss()(y_pred_prob, y)
    return {'accuracy': accuracy, 'validation_loss': validation_loss.item()}

paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path_aug,
        'pca_path':pca_path_aug,
        'all_pprocessed_data_path':all_pprocessed_data_directory_aug,
    }

data_preprocessor = DataPreprocessor(paths,crop_augment_fold = 3)
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

class Network(nn.Module):
    def __init__(self, X, y, dims="1024-256", dropout_rate = 0.4 ,nonlin=nn.ReLU()):
        super().__init__()

        input_dim = X.shape[1]
        output_dim = len(torch.unique(y))

        self.dims = dims
        self.dropout_rate = dropout_rate
        self.nonlin = nonlin
        self.input_dim = input_dim
        self.output_dim = output_dim

        dim_list = list(map(int, self.dims.split('-')))
        dim_list = [input_dim] + dim_list + [output_dim]

        layers = []
        for i in range(len(dim_list) - 2):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            layers.append(self.nonlin)  
            layers.append(nn.Dropout(self.dropout_rate))  
        

        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(dim_list[-2], dim_list[-1])

    def forward(self, X, **kwargs):
        X = self.layers(X)  # Pass through all layers
        X = self.output(X)  # Apply the output layer and softmax
        return X

net = NeuralNetClassifier(
    Network,
    module__X=x_train_tensor,
    module__y=y_train_tensor,
    max_epochs=200,
    lr=0.0001,
    iterator_train__shuffle=True,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    batch_size=64,
    callbacks=[('early_stopping', EarlyStopping(monitor='valid_loss', lower_is_better=True))],  
    verbose=0,
    train_split=predefined_split(valid_dataset),
)
params = {
    'module__dims': ["1024-256", "1024-512-256", "2048-1024-512"],
    'module__dropout_rate': [0.2, 0.3, 0.4],
    'lr': [0.0001, 0.00007],
    'callbacks__early_stopping__patience': [5, 10, 15],
}

gs = GridSearchCV(net, params, refit=False, cv=predefined_split_grid_search, scoring=custom_scorer, verbose=3)

gs.fit(x_combined_tensor, y_combined_tensor)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

# Best parameters and score
print("Best Parameters:", gs.best_params_)
print("Best Score:", gs.best_score_)
