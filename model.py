import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input shape: [batch_size, 1, 128, 1103]
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape: [batch_size, 16, 64, 551]

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output shape: [batch_size, 32, 32, 275]
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape: [batch_size, 32, 16, 137] -- Approximation

        # import ipdb;ipdb.set_trace()
        # Update the input size of fc1 to match the flattened output size: 32 channels * 12 height * 230 width
        self.fc1 = nn.Linear(39936, 64)  # Corrected input features to match actual output
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)  # Output classes 50 for ESC50

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x