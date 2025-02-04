import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from load_data import training_data, test_data
import my_models
import visualizer
from train_test import train, test

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH = 32
EPOCHS = 10

train_loader = DataLoader(training_data, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCH, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


print(f"Device: {device}")

visualizer.show_images(train_loader)


# model_gray = my_models.LeNetGray().to(device)
model_padding = my_models.LeNetPadding().to(device)
model_smaller = my_models.LeNet34().to(device)
model_lenet = my_models.LeNet5().to(device)

# visualizer.graph(model_gray, device)
visualizer.graph(model_padding, device)
visualizer.graph(model_smaller, device)
visualizer.graph(model_lenet, device)

# visualizer.projector(training_data)

models = [model_padding, model_smaller, model_lenet]

loss = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.parameters()) for model in models]


for indx, model in enumerate(models):
    print(f"Model: {print(model)}\n\n")
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_loader, model, loss, optimizers[indx], t, device)
        test(test_loader, model, loss, device)
        print("Done!\n\n")


# torch.save(model_gray.state_dict(), "./models/model_gray.pth")
torch.save(model_padding.state_dict(), "./models/model_padding.pth")
torch.save(model_smaller.state_dict(), "./models/model_smaller.pth")
torch.save(model_lenet.state_dict(), "./models/model_lenet.pth")




