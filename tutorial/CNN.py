import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#Determine the objective

#set up the training data
training_data=datasets.MNIST(root="data",train=True,download=True,transform=ToTensor())
test_data=datasets.MNIST(root="data",train=False,download=True,transform=ToTensor())

class_names=["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

batch_size=64
train_dataloader=DataLoader(training_data,batch_size=batch_size)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(5,5),stride=1)
        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.relu1=nn.ReLU()

        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.relu2=nn.ReLU()
        self.fc1=nn.Linear(32*5*5,128)
        self.fc2=nn.Linear(128,10)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.relu2(x)

        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

device=("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

def train(epoch_num=20):
    model.train()
    for i in range(epoch_num):
        loss_sum=0
        for batch,(x,y) in enumerate(train_dataloader):
            # print(x.size())
            train_batch_size=x.size(0)
            x,y=x.to(device),y.to(device)
            # x=x.reshape(train_batch_size,-1)
            # print(x.size())
            y_pred=model(x)
            loss=loss_fn(y_pred,y)
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch:{i},loss_sum:{loss_sum}")


train()

torch.save(model.state_dict(),"hwd_CNN.pth")

def test():
    model.load_state_dict(torch.load("hwd_CNN.pth"))
    model.eval()
    success_sum=0
    test_loss=0
    correct=0
    length=len(test_dataloader.dataset)
    print(length)
    for batch,(x,y) in enumerate(test_dataloader):

        train_batch_size=x.size(0)
        x,y=x.to(device),y.to(device)
        # x=x.reshape(train_batch_size,-1)
        y_pred=model(x)
        test_loss+=loss_fn(y_pred,y).item()
        correct+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
    correct/=length
    print(f"Accuracy:{100*correct:>0.1f}%")