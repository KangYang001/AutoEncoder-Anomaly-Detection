# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:05:24 2019

@author: Smartdatalab
"""

import argparse
import numpy as np
'''-------------------------------------------------------------------------'''
'''---------------------------- data information ---------------------------'''
'''-------------------------------------------------------------------------'''
parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset', conflict_handler = 'resolve')
parser.add_argument('--direction_workspace', type = str, default = "/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder",
                    help = 'type of the dataset')
parser.add_argument('--direction_matdata', type = str, default = "/home/UFAD/yang.kang/Ultrasonics/Kang/20180322_LongTermData_Mat", 
                    help = 'direction of the mat dataset')
parser.add_argument('--direction_data_weather', type = str, default = "/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA",
                    help = 'type of the dataset')
parser.add_argument('--folder_matfile', action='append', nargs='+')                   
parser.add_argument('--direction_mass_information', type = str, \
                    default = "/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA/code/file_with_mass.csv",
                    help = 'direction of mass information file')
parser.add_argument('--direction_label_weather', type = str, default = '/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA',
                    help = 'working direction')  
parser.add_argument('--ratio_downsample', type = int, default = 10, help = 'ratio between original dataset and downsampled dataset')
parser.add_argument('--threshold_data_type', type = float, default = 0, 
                    help = 'train data whose reconstruction coefficient are above threshold 0.975')
parser.add_argument('--batch_file', type = int, default = 50, metavar='N', help = 'the size of files in one data file')
parser.add_argument('--cuda_devices', type = int, default = 4, metavar='N', help = 'the number of cuda devices')
parser.add_argument('--sensors', type = str, default = '0', metavar='N', help = 'input features')
parser.add_argument('--signal_slice', action='append', nargs='+')  
parser.add_argument('--num_sensors', type = int, default = 8, metavar='N', help = 'input features')
parser.add_argument('--samples_signal', type = int, default = 2000, metavar='N', help = 'input features')
parser.add_argument('--type_input', type = str, default = '0, 1', metavar='N', help = 'input features : default(0, 1, 2, 3)')
parser.add_argument('--type_input_str',  type = str, default = '0123')
'''-------------------------------------------------------------------------'''
'''--------------------------- model information ---------------------------'''
'''-------------------------------------------------------------------------'''

parser.add_argument('--model', type = str, default = 'autoencoder',
                    help = 'type of net (autoencoder, DNN)')
parser.add_argument('--model_function', type = str, default = 'autoencoder',
                    help = 'type of net (autoencoder, detect_mass, detect_env)')
parser.add_argument('--cuda_visible_devices', type = str, default = '0, 1, 2, 3', help = 'CUDA ID used in model: default(0, 3, 4, 5)')
parser.add_argument('--num_neurons', action='append', nargs='+')
parser.add_argument('--num_neurons_DNN', action='append', nargs='+')
parser.add_argument('--divider_set', action='append', nargs='+') 
parser.add_argument('--lr', type = float, default = 0.00005, help = 'initial learning rate')
parser.add_argument('--weight_decay', type = float, default=1e-5, help = 'weight decay')
parser.add_argument('--dropout', type = float, default = 0.02, help = 'dropout applied to layers (0 = no dropout)')
parser.add_argument('--epochs', type = int, default = 40, help = 'upper epoch limit')  
parser.add_argument('--epochs_resume', type = int, default = 21, help = 'upper epoch limit') 
parser.add_argument('--clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--batch_size', type = int, default = 256, metavar='N', help = 'batch size')  
parser.add_argument('--device', type = str, default = 'cuda', help = 'cuda or cpu')


parser.add_argument('--tied', action = 'store_true', help = 'tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type = int, default = 1111, help = 'random seed')
parser.add_argument('--log_interval', type = int, default = 5, metavar = 'N', help = 'report interval')
parser.add_argument('--save_interval', type = int, default = 10, metavar = 'N', help = 'save interval')
parser.add_argument('--save_fig_interval', type = int, default = 40, metavar = 'N', help = 'save interval')
parser.add_argument('--save_fig', action = 'store_true', default = False, help = 'save figure')
parser.add_argument('--resume', '-r', help = 'use checkpoint model parameters as initial parameters (default: False)',
                    action = "store_true")
parser.add_argument('--pretrained','-p', default = False, help = 'use checkpoint model parameters and do not train anymore (default: False)',
                    action = "store_true")
parser.add_argument('--if_save_dataset','-s', default = True, help = 'if save data when creating data files (default: False)',
                    action = "store_true")

'''-------------------------------------------------------------------------'''
'''----------------------- anomaly detector information --------------------'''
'''-------------------------------------------------------------------------'''

parser.add_argument('--scatter_factor', type = float, default = 0.1, help = 'scatter factor')
parser.add_argument('--if_downsample','-s', default = False, help = 'if shuffle data when creating data files (default: False)',
                    action = "store_true")

args = parser.parse_args("--folder_matfile /MatData_00001_to_06000 \
                          --folder_matfile /MatData_06001_to_12000 \
                          --folder_matfile /MatData_12001_to_18000 \
                          --folder_matfile /MatData_18001_to_24000 \
                          --signal_slice 0 2000 \
                          --num_neurons 2000 512 128 32 3 \
                          --num_neurons_DNN 1000 512 256 64 4 \
                          --divider_set 0.001 0.99 1".split())


#args = parser.parse_args()
args.pretrained = False
args.resume = False
args.save_fig = True


type_input = np.array(eval(args.type_input))
type_input_str = str(type_input[0])
for i in range(1, len(type_input)):
    type_input_str = type_input_str + str(type_input[i])
      
    
args.type_input_str = type_input_str 