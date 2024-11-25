import torch
import torch.nn as nn

class ConvNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.dout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.dout2 = nn.Dropout(0.2)


        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.dout3 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.dout4 = nn.Dropout(0.2)


        self._to_linear = self._calculate_flatten_size(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.act5 = nn.ReLU()
        self.dout5 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(64, 8)

        self.nonlin_acts = []

    def _calculate_flatten_size(self, input_shape):
        x = torch.zeros(1, *input_shape)

        x = self.act1(self.conv1(x))
        x = self.dout1(x)

        x = self.act2(self.conv2(x))
        x = self.dout2(x)

        torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.act3(self.conv3(x))
        x = self.dout3(x)

        x = self.act4(self.conv4(x))
        x = self.dout4(x)

        torch.max_pool2d(x, kernel_size=2, stride=2)

        return x.numel()



    def forward(self, x):

        self.nonlin_acts = []

        x = self.act1(self.conv1(x))
        self.nonlin_acts.append(x)
        x = self.dout1(x)

        x = self.act2(self.conv2(x))
        self.nonlin_acts.append(x)
        x = self.dout2(x)

        torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.act3(self.conv3(x))
        self.nonlin_acts.append(x)
        x = self.dout3(x)

        x = self.act4(self.conv4(x))
        self.nonlin_acts.append(x)
        x = self.dout4(x)

        torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = self.act5(self.fc1(x))
        self.nonlin_acts.append(x)
        x = self.dout5(x)

        x = self.fc2(x)  
        return x