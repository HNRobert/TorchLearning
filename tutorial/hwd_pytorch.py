import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Determine the objective

# set up the training data
training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False,
                           download=True, transform=ToTensor())

class_names = ["Zero", "One", "Two", "Three",
               "Four", "Five", "Six", "Seven", "Eight", "Nine"]

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(epoch_num=20):
    model.train()
    for i in range(epoch_num):
        loss_sum = 0
        for batch, (x, y) in enumerate(train_dataloader):
            # print(x.size())
            train_batch_size = x.size(0)
            x, y = x.to(device), y.to(device)
            x = x.reshape(train_batch_size, -1)
            # print(x.size())
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch:{i},loss_sum:{loss_sum}")


# train(30)

torch.save(model.state_dict(), "hwd_pytorch.pth")


def test():
    model.load_state_dict(torch.load("hwd_pytorch.pth"))
    model.eval()
    success_sum = 0
    test_loss = 0
    correct = 0
    length = len(test_dataloader.dataset)
    print(length)
    for batch, (x, y) in enumerate(test_dataloader):

        train_batch_size = x.size(0)
        x, y = x.to(device), y.to(device)
        x = x.reshape(train_batch_size, -1)
        y_pred = model(x)
        test_loss += loss_fn(y_pred, y).item()
        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= length
    print(f"Accuracy:{100*correct:>0.1f}%")


test()
