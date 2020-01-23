import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import load_iris
import torch.nn as nn
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



skiris = load_iris()
IRIS = iris_dataset(skiris)
IRIS_loader = DataLoader(dataset=IRIS, batch_size=8,shuffle=True,num_workers=0, drop_last=True)

weight = Variable(torch.Tensor(np.random.normal(0,0.5,12).reshape(4,3)), requires_grad=True)
bias = Variable(torch.zeros(3,dtype=torch.float),requires_grad=True)


sofmax = nn.LogSoftmax()
# loss_fn = nn.CrossEntropyLoss(reduction='sum')
loss_fn = nn.NLLLoss2d(reduction='sum')

for iter in range(1000):
    mean_loss = 0.
    mean_acc = 0.
    nb_batch = 0
    for batch_order, (x, y) in enumerate(IRIS_loader):
        mini_x = Variable(x)
        mini_y = Variable(y)

        net_result = torch.mm(mini_x, weight)
        net_result += bias
        net_result = sofmax(net_result)
        loss = loss_fn(net_result, mini_y)
        mean_loss += loss
        loss.backward()
        weight.data -= 0.01 * weight.grad
        weight.grad.zero_()
        bias.data -= 0.01 * bias.grad
        bias.grad.zero_()
        _, argmax = torch.max(net_result, 1)
        mean_acc += (argmax == mini_y).sum()
        nb_batch += 1
    print('loss : ', mean_loss.item()/IRIS.__len__(), 'acc : ', mean_acc.item()/150)


