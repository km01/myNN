import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import load_iris
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def min_max(x_data):
    min = np.min(x_data,axis=0,keepdims=True)
    max = np.max(x_data,axis=0,keepdims=True)
    x_data = ((x_data - min)/(max - min)) - 0.5
    return x_data

class iris_dataset(Dataset):

    def __init__(self, sklearn_data):
        self.length = sklearn_data['data'].shape[0]
        self.x_data = torch.Tensor(min_max(sklearn_data['data']))
        self.y_data = torch.LongTensor(sklearn_data['target'])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length




IRIS = iris_dataset(load_iris())
IRIS_loader = DataLoader(dataset=IRIS, batch_size=20,shuffle=True,num_workers=0, drop_last=False)

model = nn.Sequential(nn.Linear(4,50), torch.nn.ReLU(),
                      torch.nn.Linear(50,30), torch.nn.ReLU(),
                      torch.nn.Linear(30,20),  torch.nn.ReLU(),
                      torch.nn.Linear(20,20),torch.nn.ReLU(),
                      torch.nn.Linear(20,10), torch.nn.ReLU(),
                      torch.nn.Linear(10,3), torch.nn.LogSoftmax())

optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss(reduction='sum')


for i in range(1000):
    mean_loss = 0.
    mean_acc = 0.
    nb_batch = 0
    for batch_order, (x, y) in enumerate(IRIS_loader):
        mini_x = Variable(x)
        mini_y = Variable(y)
        net_result = model.forward(mini_x)
        loss = loss_fn(net_result, mini_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, argmax = torch.max(net_result, 1)
        mean_loss += loss/len(net_result)
        mean_acc += np.sum(np.array(argmax == mini_y))/len(net_result)
        nb_batch += 1

    print('loss : ', mean_loss.item()/nb_batch, 'acc : ', mean_acc/nb_batch)


acc = 0.
for i in range(IRIS.length):
    x = Variable(IRIS.x_data[i])
    net_result = model.forward(x)
    a = torch.argmax(net_result)
    if(torch.argmax(net_result) == IRIS.y_data[i]):
        acc += 1.0

print(acc/IRIS.length)
