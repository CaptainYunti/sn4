import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

from load_data import training_data, test_data
from train_test import train, test

BATCH = 4

train_loader = DataLoader(training_data, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCH, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")




