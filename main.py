#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:15:39 2020

@author: yang.kang
"""

import torch
from torch import nn
import torchvision.datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import os
from tqdm import tqdm
from torch.autograd import Variable

torch.manual_seed(1)    # reproducible

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,0,1,3,4,5"

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.bidirectional = True
        self.rnn_hid_size = 64
        self.rnn_type = 'LSTM'
        self.nlayers = 1
        self.rnn = nn.LSTM(     # LSTM 效果要比 nn.RNN() 好多了
            input_size = 28,      # 图片每行的数据像素点
            hidden_size = self.rnn_hid_size,     # rnn hidden unit
            num_layers = self.nlayers,       # 有几层 RNN layers
            batch_first = True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            bidirectional = self.bidirectional
        )
        
        self.out = nn.Linear(128, 10)    # 输出层

    def forward(self, x, hidden):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
    
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])
        
        r_out, (h_n, h_c) = self.rnn(x, hidden)   # None 表示 hidden state 会用全0的 state
        
    
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])      
        return out, h_n, h_c
    
            
    def init_hidden(self, batch_size):
        '''
        Initialize hidden state.
        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM
        Arguments:
            batch_size: batch size, an integer
        Returns:
            hidden: hidden state initialized
        '''
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            num_directions = 2 if self.bidirectional else 1
            
        if (torch.cuda.device_count() >= 1):
        
            hidden = (weight.new(batch_size, self.nlayers * num_directions, self.rnn_hid_size).zero_().cuda(),
                      weight.new(batch_size, self.nlayers * num_directions, self.rnn_hid_size).zero_().cuda())

        else:
            hidden = (weight.new(batch_size, self.n_layers * num_directions, self.n_hidden).zero_(),
                      weight.new(batch_size, self.n_layers * num_directions, self.n_hidden).zero_())

        return hidden


    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()


# Hyper Parameters
EPOCH = 2          # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 256
TIME_STEP = 28      # rnn 时间步数 / 图片高度
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
LR = 0.01           # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle


rnn = RNN()

if torch.cuda.device_count() > 1:
    rnn = nn.DataParallel(rnn, device_ids = [0, 1, 2, 3], dim = 0).cuda()

print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

"""
RNN (
  (rnn): LSTM(28, 64, batch_first=True)
  (out): Linear (64 -> 10)
)
"""

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root = './mnist/',    # 保存或者提取位置
    train = True,  # this is training data
    transform = torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download = DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


# training and testing
for epoch in range(EPOCH):
    for step, (x, b_y) in tqdm(enumerate(train_loader)):   # gives batch data
        
        b_x = x.view(-1, 28, 28).cuda(0)   # reshape x to (batch, time_step, input_size)
        try:
            hidden = rnn.init_hidden(b_x.size(0))
        except AttributeError:
            hidden = rnn.module.init_hidden(b_x.size(0))
        output, h_n, h_c = rnn(b_x, hidden)               # rnn output
        loss = loss_func(output, b_y.cuda(0))   # cross entropy loss   
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
    print(epoch)



"""
...
Epoch:  0 | train loss: 0.0945 | test accuracy: 0.94
Epoch:  0 | train loss: 0.0984 | test accuracy: 0.94
Epoch:  0 | train loss: 0.0332 | test accuracy: 0.95
Epoch:  0 | train loss: 0.1868 | test accuracy: 0.96
"""


test_output, h_n, h_c  = rnn(test_x[:20].view(-1, 28, 28).cuda(), rnn.module.init_hidden(20))
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:20].numpy(), 'real number')
"""
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
"""

