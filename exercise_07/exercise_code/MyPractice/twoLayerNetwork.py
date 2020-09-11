import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self, activation=nn.Sigmoid(), input_size=1*28*28, hidden_size=100, classes=10):
        super(Net, self).__init__()
        self.activation = activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        #flatten the input to 1-d torch array
        x = x.view(-1, self.input_size) #flatten the input 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

