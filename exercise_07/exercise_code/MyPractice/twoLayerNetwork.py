import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self, activation=nn.Sigmoid(), input_size=1*28*28, hidden_size=100, classes=10):
        super(Net, self).__init__()
        self.activation = activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes
        
        self.fc1 = nn.Linear(input_size, hidden_size, )
        self.fc2 = nn.Linear(hidden_size, hidden_size-20)
        self.fc3 = nn.Linear(hidden_size-20, hidden_size-40)
        self.fc4 = nn.Linear(hidden_size-40, hidden_size-80)
        sefl.fc5 = nn.Linear(hidden_size-80, hidden_size-160)
        self.fc6 = nn.Linear(hidden_size-160, hidden_size-80)
        self.fc7 = nn.Linear(hidden_size-80, hidden_size-40)
        self.fc8 = nn.Linear(hidden_size-40, hidden_size-20)
        self.fc9 = nn.Linear(hidden_size-20, hidden_size)
        self.fc10 = nn.Linear(hidden_size-10, classes)

    def forward(self, x):
        #flatten the input to 1-d torch array
        x = x.view(-1, self.input_size) #flatten the input 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

