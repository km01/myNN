import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

char_set = ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd', ';']
char_dic = {c: i for i, c in enumerate(char_set)}
ohv_size = len(char_dic)
ohv_list = np.eye(ohv_size)

input_sentence = 'hello world'
input_sentence_idx = [char_dic[c] for c in input_sentence]
input_data = np.array([[ohv_list[c] for c in input_sentence_idx]])
input_data = torch.from_numpy(input_data).float()

target_sentence = 'ello world;'
target_sentence_idx = [char_dic[c] for c in target_sentence]

target_data = np.array([target_sentence_idx])
target_data = torch.from_numpy(target_data).long()

rnn = nn.LSTM(input_size=ohv_size, hidden_size=ohv_size, batch_first=True)
fc = nn.Linear(ohv_size, ohv_size)

solver = optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=1e-1)
epoch = 20
for ep in range(epoch):
    solver.zero_grad()
    h, _ = rnn(input_data)
    print(h.shape)
    print(h[:, -1].shape)
    y = fc(h)
    loss = F.cross_entropy(y.view(-1, ohv_size), target_data.view(-1))
    loss.backward()
    solver.step()

    output = np.argmax(y.data.numpy(), axis=2)[0]
    output_sentence = ''.join([char_set[c] for c in output])
    print('{},  Loss:{},    out:{}'.format(ep, loss.item(), output_sentence))


