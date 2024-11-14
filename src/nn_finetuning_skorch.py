import numpy as np
from config import hdf5_path, scaler_path_aug, pca_path_aug, all_pprocessed_data_directory_aug
from preprocessing import DataPreprocessor
from model_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit




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

# merge splits and get predefined split for gridsearchcv
split_index = [-1] * len(x_train) + [0] * len(x_val)
predefined_split = PredefinedSplit(test_fold=split_index)

x_train = np.concatenate([x_train, x_val], axis = 0)
y_train = np.concatenate([y_train, y_val], axis = 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)


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
        summary(self)

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
)

net.set_params(train_split=False, verbose=0)
params = {
    'lr': [0.0001, 0.00007 ,0.00005],
    'max_epochs': [200, 300],
    'module__dims': ["1024-256", "2048-512"],
}

gs = GridSearchCV(net, params, refit=False, cv=predefined_split, scoring='accuracy', verbose=3)

gs.fit(x_train_tensor, y_train_tensor)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

# Best parameters and score
print("Best Parameters:", gs.best_params_)
print("Best Score:", gs.best_score_)
