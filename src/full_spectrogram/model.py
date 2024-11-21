import torch
import torch.nn as nn

class ConvNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)

        self._to_linear = self._calculate_flatten_size(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 64)  
        self.fc2 = nn.Linear(64, 8)  

    def _calculate_flatten_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x