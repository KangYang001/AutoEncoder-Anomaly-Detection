#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:15:39 2020

@author: yang.kang
"""

import torch
from torch import nn
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.functional as F 
import os
from torch.autograd import Variable
import numpy as np
from pathlib import Path
import shutil
torch.manual_seed(1)    # reproducible

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,0,1,3,4,5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_corr(x, y):

    criterion = nn.MSELoss()
    vx = x - torch.mean(x, dim = 1, keepdim = True)
    vy = y - torch.mean(y, dim = 1, keepdim = True)
    corr = torch.mm(vx, vy.T)/torch.mm(torch.sqrt(torch.sum(vx ** 2, dim = 1, keepdim = True)), torch.sqrt(torch.sum(vy ** 2, dim = 1, keepdim = True)).T)
    cost = criterion(torch.diagonal(corr, 0).to(device), torch.from_numpy(np.ones(len(vx))).to(device).float())
    
    return cost

def loss_mee(x, y):
    
    sigma = 0.01
    error = (x - y).unsqueeze(1)
    error = error.repeat(1, error.size(2), 1)
    error = (error[:, 0, :].view(-1, 2000, 1) - error) ** 2
    dist = (1/(np.sqrt(2*np.pi) * sigma)) * torch.exp(-error/(2 * (sigma ** 2)))
    loss = -torch.log(torch.mean(dist * dist))
    
    print("MEE Loss")
    
    return loss

class AutoEncoder(nn.Module):
      
    def __init__(self, args, name_model):
        super(AutoEncoder, self).__init__()
        self.num_neurons = list(map(int, args.num_neurons[0]))
        self.name_model = name_model
        self.checkpoint_dir = Path(args.direction_workspace,'save', args.model, args.model_function, 
                              "sensors_" + str(args.sensors), str(args.batch_file), 
                              "ratio downsample " + str(args.ratio_downsample), self.name_model).joinpath('checkpoint')
        self.args = args       
        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(self.num_neurons[0], self.num_neurons[1]),
            nn.Dropout(args.dropout),
            nn.Tanh(),
            nn.Linear(self.num_neurons[1], self.num_neurons[2]),
            nn.Dropout(args.dropout),
            nn.Tanh(),
            nn.Linear(self.num_neurons[2], self.num_neurons[3]),            
            #nn.Dropout(args.dropout),
            #nn.Tanh(),
            #nn.Linear(self.num_neurons[3], self.num_neurons[4]),   # 压缩成3个特征, 进行 3D 图像可视化
            
        )
        # 解压
        self.decoder = nn.Sequential(
                
            #nn.Linear(self.num_neurons[4], self.num_neurons[3]),
            #nn.Tanh(),
            #nn.Dropout(args.dropout),
            nn.Linear(self.num_neurons[3], self.num_neurons[2]),
            nn.Dropout(args.dropout),
            nn.Tanh(),
            nn.Linear(self.num_neurons[2], self.num_neurons[1]),
            nn.Dropout(args.dropout),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(self.num_neurons[1], self.num_neurons[0]),
            #nn.Dropout(args.dropout),
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded                 

    def save_checkpoint(self, val_loss, best_val_loss, epoch, model, optimizer):
        
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        state = {'epoch': epoch,
                 'best_loss': best_val_loss,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'args': self.args}        

        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = self.checkpoint_dir
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.model_function).with_suffix('.pth')
        torch.save(state, checkpoint)
        
        if is_best:
            model_best_dir = Path(args.direction_workspace,'save', args.model, args.model_function, 
                                  "sensors_" + str(args.sensors), str(args.batch_file), 
                                  "ratio downsample " + str(args.ratio_downsample), self.name_model).joinpath('model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.model_function).with_suffix('.pth'))

        print('=> checkpoint saved.')

    def load_checkpoint(self, args, checkpoint):
        start_epoch = checkpoint['epoch'] +1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size = args.prediction_window_size
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss
    
    def locate_checkpoint(self):
        
        checkpoint_dir = self.checkpoint_dir
        checkpoint = checkpoint_dir.joinpath(self.args.model_function).with_suffix('.pth')
        checkpoint = torch.load(checkpoint)  
        
        return checkpoint
            
    
    

class MLP(nn.Module):     # 继承 torch 的 Module
    def __init__(self, args, n_feature, n_output):
        super(MLP, self).__init__()     # 继承 __init__ 功能     
        n_hidden = list(map(int, args.num_neurons_DNN[0]))
        self.input_layer = nn.Linear(n_feature, n_hidden[0]) # 隐藏层线性输出
        self.hidden_layer = nn.Sequential(
                nn.Linear(n_hidden[0], n_hidden[1]),
                nn.ReLU(),
                nn.Linear(n_hidden[1], n_hidden[2]),
                nn.ReLU(),
                nn.Linear(n_hidden[2], n_hidden[3]),
                nn.ReLU(),
                nn.Linear(n_hidden[3], n_hidden[4]),
                nn.ReLU()
                )
        self.output_layer = nn.Linear(n_hidden[-1], n_output) # 输出层线性输出
        
    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.input_layer(x))
        x = self.hidden_layer(x)      # 激励函数(隐藏层的线性值)
        x = self.output_layer(x)         # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

    def save_checkpoint(self, state, is_best):
        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = Path(args.direction_workspace,'save', args.model, args.model_function, "sensors_" + str(args.sensors), \
                              str(args.batch_file), "ratio downsample " + str(args.ratio_downsample)).joinpath('checkpoint')
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.model_function).with_suffix('.pth')
        torch.save(state, checkpoint)
        
        if is_best:
            model_best_dir = Path(args.direction_workspace,'save', args.model, args.model_function, "sensors_" + str(args.sensors), \
                              str(args.batch_file), "ratio downsample " + str(args.ratio_downsample)).joinpath('model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.model_function).with_suffix('.pth'))

        print('=> checkpoint saved.')

    def load_checkpoint(self, args, checkpoint):
        start_epoch = checkpoint['epoch'] +1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size = args.prediction_window_size
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss




'''
net = MLP(n_feature=2, n_hidden = [10,20,30,40,10], n_output=2) # 几个类别就几个 output
print(net)  # net 的结构
'''