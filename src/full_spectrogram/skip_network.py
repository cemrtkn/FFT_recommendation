
from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 4)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out
    

class SkipNetwork(nn.Module):
    def __init__(self, block, channels, input_shape):
        super(SkipNetwork, self).__init__()
        
        self.channels = channels
        self.conv_layers = nn.ModuleList()
        
        for layer_id, out_channels in enumerate(self.channels):
            conv_layer = self._make_conv_layer(block, out_channels, layer_id, stride = 1)
            self.conv_layers.append(conv_layer)
                
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=channels[-1], out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self._to_linear = self._calculate_flatten_size(input_shape)

        self.fc1 = nn.Sequential(
                nn.Linear(self._to_linear, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

        self.head = nn.Linear(1024, 8)


    def _calculate_flatten_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.outconv(x)
        #x = self.avgpool(x)
        return x.numel()

    def _make_conv_layer(self, block, out_channels, layer_id, stride=1):
        downsample = None
        if layer_id == 0:
            in_channels = 1 # first layer
        else:
            in_channels = self.channels[layer_id-1] 
            
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.outconv(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.head(x) 
        return x