import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter


# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((.5),(.5))]
)


training_set = torchvision.datasets.FashionMNIST(
    root="./data",
    download=True,
    train=True,
    transform=transform
)

validation_set = torchvision.datasets.FashionMNIST(
    root="./data",
    download=True,
    train=False,
    transform=transform
)


training_loader = torch.utils.data.DataLoader(training_set,batch_size=128,shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=128,shuffle=True)

classes = ("Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", 
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys") 
    else:
        plt.imshow(np.transpose(np.img, (1,20)))

dataiter = iter(training_loader)
images, labels = next(dataiter)

img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)

writer = SummaryWriter("runs/fashion_mnist_experiment_1")

writer.add_image("Four Fashion-MNIST Images", img_grid)
writer.flush()



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.to(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optmizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print(len(validation_loader))

for epoch in range(2):
    print(f"\nEpoch {epoch+1}")
    running_loss = 0.0

    for i, data in enumerate(training_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optmizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optmizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Batch {i+1}")
            running_vlos = 0.0

            net.eval()
            for j, vdata in enumerate(validation_loader):
                vinput, vlabels = vdata[0].to(device), vdata[1].to(device)
                voutputs = net(vinput)
                vloss = criterion(voutputs, vlabels)
                running_vlos += vloss.item()
            net.train()

            avg_loss = running_loss / 1000
            avg_vloss = running_vlos / len(validation_loader)

            writer.add_scalars("Training vs. Validation Loss", {"Training" : avg_loss, "Validation": avg_vloss},
                               epoch * len(training_loader) + i)
            
            running_loss = 0.0

print("Finished training")

writer.flush()


dataiter = iter(training_loader)
images, labels = next(dataiter)

writer.add_graph(net, images)
writer.flush()


# Select a random subset of data and corresponding labels
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# Extract a random subset of data
images, labels = select_n_random(training_set.data, training_set.targets)

# get the class labels for each image
class_labels = [classes[label] for label in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.flush()
writer.close()