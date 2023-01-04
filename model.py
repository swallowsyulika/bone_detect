import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=75, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=2)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc3(x)
        x = nn.functional.leaky_relu(x)
        output = self.fc4(x)
        # output = nn.functional.softmax(x)
        return output