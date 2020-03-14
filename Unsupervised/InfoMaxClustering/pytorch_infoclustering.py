import numpy as np
import torch.nn as nn
import torch.nn.init
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




class CustomDataSet(torch.utils.data.Dataset):

    def __init__(self, n_data):

        self.n_data = n_data
        _data = np.random.randn(n_data, 2)
        mu = [[1, -1], [0, np.sqrt(2.0)], [-1, -1]]
        sig = [[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]]
        for i in range(n_data):
            _data[i][0] = sig[i % 3][0] * _data[i][0] + mu[i % 3][0]
            _data[i][1] = sig[i % 3][1] * _data[i][1] + mu[i % 3][1]

        self.data = torch.from_numpy(_data).float()

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        return self.data[idx]


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=7, bias=True),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)


model = Discriminator()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

dataset = CustomDataSet(10000)
loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
n_epoch = 2000
n_batch = len(loader)


for ep in range(n_epoch):
    optimizer.zero_grad()
    p_y = model(dataset.data).sum(dim=0) / dataset.n_data
    p_y_bar_x = model(dataset.data)
    loss = (p_y_bar_x * torch.log(p_y_bar_x)).mean() - (p_y * torch.log(p_y/dataset.n_data)/dataset.n_data).mean()
    loss.backward()
    print('mutual info : ', -loss.item())
    optimizer.step()


label = torch.argmax(model(dataset.data), dim=1).numpy()
data = dataset.data.numpy()

group = [[], [], [], [], [], [], []]

for i in range(10000):
    group[label[i]].append(data[i])

for k in range(7):
    x = np.array(group[k])
    if len(x) != 0:
        plt.scatter(x[:, 0], x[:, 1], s=1)
plt.show()
