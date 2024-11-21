import torch
import torch.nn as nn

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