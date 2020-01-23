import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pyplot

def formular(x):
    return -x*x + 1.0

def formular2(x):
    return -1.2*x*x

x = np.linspace(-1,1,10,dtype=np.float).reshape(-1,1)
y = formular(x)


x_data1 = np.ndarray((1000),dtype=np.float)
y_data1 = np.ndarray((1000),dtype=np.float)
x_data2 = np.ndarray((1000),dtype=np.float)
y_data2 = np.ndarray((1000),dtype=np.float)

for i in range(10):
    for j in range(100):
        noise = np.random.normal(size=(2),loc=0.0,scale=0.3)
        x_data1[i*100 + j] = x[i] + noise[0]
        y_data1[i*100 + j] = formular(x_data1[i*100 + j]) + noise[1]
        noise = np.random.normal(size=(2),loc=0.0,scale=0.15)
        x_data2[i*100 + j] = x[i] + noise[0]
        y_data2[i*100 + j] = formular2(x_data2[i*100 + j]) + noise[1]

x_data = np.concatenate((x_data1, x_data2)).reshape(-1,1)
y_data = np.concatenate((y_data1, y_data2)).reshape(-1,1)
feature = np.concatenate((x_data,y_data),axis=1)
label = np.zeros(2000,dtype=np.int).reshape(-1,1)
for i in range(1000,2000):
    label[i][0] = 1

print(feature.shape)
print(label.shape)
class ConvexDataset(Dataset):
    def __init__(self, _x_data, _y_data):
        self.length = _x_data.shape[0]
        self.x_data = torch.Tensor(_x_data)
        self.y_data = torch.Tensor(_y_data)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.length

set = ConvexDataset(feature, label)
batch_loader = DataLoader(dataset=set, batch_size=5,shuffle=True,num_workers=0,drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.z1 = nn.Linear(2,50)
        self.a1 = nn.ReLU()
        self.z2 = nn.Linear(50,20)
        self.a2 = nn.ReLU()
        self.z3 = nn.Linear(20,1)
        self.a3 = nn.Sigmoid()
    def forward(self,x):
        x = self.a1(self.z1(x))
        x = self.a2(self.z2(x))
        x = self.a3(self.z3(x))
        return x
    
model = Net()

optimizer = optim.SGD(model.parameters(),lr=0.01)
loss_fn = nn.BCELoss()

half = 150
map = np.ones((2*half,2*half,3),dtype=np.float)
for i in range(2*half):
    map[half][i] = 0.0
    map[i][half] = 0.0

feature = (feature/4)*half + half

mean_loss = 0.0
for ep in range(100):
    for batch_idx, (x,y) in enumerate(batch_loader):
        mini_x = Variable(x)
        mini_y = Variable(y)
        net_res = model.forward(mini_x)
        loss = loss_fn(net_res, mini_y)
        mean_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mean_loss /= len(batch_loader)
    print(mean_loss)

shrink = 3.5
temp = np.zeros((1,2),dtype=np.float)
for i in range(2*half):
    for j in range(2*half):
        temp[0][0] = shrink*((i - half)/half)
        temp[0][1] = shrink*((j - half)/half)
        res = model(Variable(torch.Tensor(temp)))
        # print(temp[0][0], temp[0][1], res.data[0])
        if res.data[0] > 0.5:
            brightness = (res.data[0] - 0.5)
            map[j][i][0] = 0.5 * brightness
            map[j][i][1] = 1 * brightness
            map[j][i][2] = 0.5 * brightness
        else:
            brightness = (0.5 - res.data[0])
            map[j][i][0] = 1 * brightness
            map[j][i][1] = 0.5 * brightness
            map[j][i][2] = 0.5 * brightness
            

for i in range(feature.shape[0]):
    if label[i] == 0:
        x = int(round(feature[i][0]))
        y = int(round(feature[i][1]))
        map[y][x][0] = 1
        map[y][x][1] = 0
        map[y][x][2] = 0
    else:
        x = int(round(feature[i][0]))
        y = int(round(feature[i][1]))
        map[y][x][0] = 0
        map[y][x][1] = 1
        map[y][x][2] = 0


pyplot.imshow(map)
pyplot.show()
print(map)
print(map.shape)