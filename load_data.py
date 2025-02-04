import torch
from torchvision import datasets
import torchvision.transforms as transforms


tranform = (
    [transforms.ToTensor(),
     transforms.Normalize((.5, .5, .5),(.5, .5, .5))]
)

training_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=tranform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=tranform
)
