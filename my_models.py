import torch.nn as nn
import torchvision
import torchvision.transforms.functional

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            nn.ReLU(nn.Conv2d(3, 6, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(nn.Conv2d(6, 16, 5)),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(nn.Linear(16*5*5, 120)),
            nn.ReLU(nn.Linear(120,84)),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)
    

class LeNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            nn.ReLU(nn.Conv2d(3, 6, 3)),
            nn.MaxPool2d(2),
            nn.ReLU(nn.Conv2d(6, 16, 4)),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(nn.Linear(16*6*6, 120)),
            nn.ReLU(nn.Linear(120,84)),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)
    
    


class LeNetGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            torchvision.transforms.functional.rgb_to_grayscale(),
            nn.ReLU(nn.Conv2d(1, 3, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(nn.Conv2d(3, 12, 5)),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(nn.Linear(12*5*5, 120)),
            nn.ReLU(nn.Linear(120,84)),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)
    

class LeNetPadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stack = nn.Sequential(
            nn.ReLU(nn.Conv2d(3, 6, 3, padding="same")),
            nn.MaxPool2d(4),
            nn.ReLU(nn.Conv2d(6, 16, 2, padding="same")),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(nn.Linear(16*4*4, 120)),
            nn.ReLU(nn.Linear(120,84)),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.training_stack(x)