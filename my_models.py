import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)
    

class LeNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*6*6, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)
    
    


class LeNetGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 12, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12*5*5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, 10)
            
        )

    def forward(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x),
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.linear_relu_stack(x)
        return x
    

class LeNetPadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(6, 16, 2, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)


class LeBigNet(nn.Module):
    def __init__(self):
        super.__init__()

        # input = 32x32x3
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        # input = 32x32x32
        self.second_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2)
        )

        # input = 16x16x32
        self.third_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        # input = 16x16x64
        self.fourth_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        # input = 8x8x64
        self.fifth_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        # input = 8x8x128
        self.sixth_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        # input = 4*4*128 (2^11)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.fourth_conv(x)
        x = self.fifth_conv(x)
        x = self.sixth_conv(x)
        x = nn.Flatten(x)
        x = self.linear_relu_stack(x)
        return x