# author: vipul vaibhaw
# Feel free to use this code for educational purposes

# In this code we will learn to use hooks in neural networks
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np 
import torch.nn.functional as F

# Let us start by building a very basic model in pytorch
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
         
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.m_pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=64*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, input_x):
        x = self.m_pool(F.relu(self.conv1(input_x)))
        x = self.m_pool(F.relu(self.conv2(x)))
        x = x.view(-1,64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x 


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

# dataloader 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])), batch_size=1, shuffle=True)

net = Net()
learning_rate = 0.001
momentum = 0.9
log_interval = 10
n_epochs = 1
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []

# register hooks on each layer
hookF = [Hook(layer[1]) for layer in list(net._modules.items())]
hookB = [Hook(layer[1],backward=True) for layer in list(net._modules.items())]

def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output = net(data)
        
        print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)
        for hook in hookF:
            print(hook.input)
            print(hook.output)
            print('---'*17)
        
        print('\n')

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print('***'*3+'  Backward Hooks Inputs & Outputs  '+'***'*3)
        for hook in hookB:
            print(hook.input)
            print(hook.output)
            print('---'*17)


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      
for epoch in range(1, n_epochs + 1):
    train(epoch)
                



