#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:07:02 2020

@author: yang.kang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:10:50 2020

@author: yang.kang
"""

import time
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from pathlib import Path
# import create_data_for_deeplearning
import numpy as np
import math
from tqdm import tqdm
import os
import sys
import pickle
import glob
import random
from datetime import datetime
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')

import simpleModel as Model
import configuration
import preprocess_data
import process_data
import create_dataset_for_autoencoder

args = configuration.args
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

'''-------------------------------------------------------------------------'''
'''--------------------------- hyperparameters -----------------------------'''
'''-------------------------------------------------------------------------'''
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if_generate_output = False
if_shuffle_data = False
keywords_dict = ['dataset_sonic', 'temperature', 'humidity', 'brightness', 'pressure']
cuda = torch.device('cuda:5')     # Default CUDA device
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)
type_input = np.array(eval(args.type_input))
sensor = 0
type_label_weather = 1

'''-------------------------------------------------------------------------'''
'''--------------------------- create dataset ------------------------------'''
'''-------------------------------------------------------------------------'''

load_dir = Path('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA', 'dataset', 'corrcoeff_PCA', '01234567', 
                '25').joinpath("combined dataset")
path_load = Path(load_dir).joinpath('datasets_PCA_weather.pickle')

if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as file:
        plate_ultrasonic_dataset = pickle.load(file)        
    print(path_load, "has been created\n") 
else:
    print(f"please create {path_load}")
   
corrcoef_T = plate_ultrasonic_dataset["corrcoef_T"][::10]
corrcoef = corrcoef_T[sensor][::10]

null_index = plate_ultrasonic_dataset['null_index']
index_10 =  np.where((null_index % args.ratio_downsample) == 0)
index_null_data = np.array(list(map(int, null_index[index_10]/args.ratio_downsample)))


filename_save = 'ultrasonic_orginial_downsample_'+ str(args.ratio_downsample) +'.pickle'
load_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" + str(args.sensors), \
                str(args.batch_file)).joinpath("ratio_downsample_" + str(args.ratio_downsample))
path_load = Path(load_dir).joinpath(filename_save)

if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as handle:
        plate_ultrasonic_dataset_T = pickle.load(handle)         
    print(path_load, "has been created\n") 
else:
    plate_ultrasonic_dataset_T = create_dataset_for_autoencoder.create_dataset(args)

norm_scale = [] 
dataset = plate_ultrasonic_dataset_T['dataset_sonic']
datatime = plate_ultrasonic_dataset_T['datatime']
dataset, temp_norm_scale = preprocess_data.normalize_data(dataset = dataset, normalization_type = 0)
norm_scale.append(temp_norm_scale)
print(args.model + " " + args.model_function)  
if args.model == "autoencoder" or args.model_function == "detect_mass":
    tag = plate_ultrasonic_dataset_T['tag'][np.newaxis:, ][:, np.newaxis] 
    dataset = np.concatenate((dataset, tag), axis = 1)
else:    
    temperature = plate_ultrasonic_dataset_T['temperature'][:, np.newaxis]
    pressure = plate_ultrasonic_dataset_T['pressure'][:, np.newaxis]
    brightness = plate_ultrasonic_dataset_T['brightness'][:, np.newaxis]
    humidity = plate_ultrasonic_dataset_T['humidity'][:, np.newaxis]
    dataset_env = np.concatenate((temperature, humidity, np.log(brightness), pressure), axis = 1)      
    index_null_env = np.array(list(set(np.argwhere(np.isnan(dataset_env))[:, 0])))
    dataset_env = np.delete(dataset_env, index_null_env, 0)  
    for i in range(dataset_env.shape[1]):
        dataset_env[:, i], temp_norm_scale = preprocess_data.normalize_data(dataset = dataset_env[:, i], normalization_type = 0)
        norm_scale.append(temp_norm_scale)
    norm_scale = np.array(norm_scale)
    dataset = np.delete(dataset, index_null_env, 0)
    
    dataset = np.concatenate((dataset, dataset_env[:, type_input]), axis = 1)

# dataset = np.delete(dataset, index_null_data, axis = 0)



datatime_train = [datetime(2018, 8, 5, 0, 1), datetime(2018, 8, 13, 0, 1)]
datatime_valid = [datetime(2018, 7, 29, 0, 1), datetime(2018, 8, 29, 0, 1)]

index_train = np.where((datatime < datatime_train[1]) & (datatime > datatime_train[0]))[0]
index_valid = np.where((datatime < datatime_valid[1]) & (datatime > datatime_valid[0]))[0]

num_data_T = int(len(index_train)/args.cuda_devices) * args.cuda_devices
dataset_train = torch.from_numpy(dataset[index_train[:num_data_T]])
num_data_T = int(len(index_valid)/args.cuda_devices) * args.cuda_devices
dataset_valid = dataset[index_valid[:num_data_T]]     

corrcoef_valid = corrcoef[index_valid[:num_data_T]]
datatime_valid = datatime[index_valid[:num_data_T]]
gen_dataset = {"dataset_valid": dataset_valid, 
               "corrcoef_valid": corrcoef_valid,
               "datatime_valid": datatime_valid}

dataset_valid = torch.from_numpy(dataset_valid)

if args.model == "autoencoder" or args.model_function == "detect_mass":
    if args.model == "autoencoder":
        dataset_train = dataset             
    train_data = torch.utils.data.TensorDataset(dataset_train[:, :args.samples_signal], dataset_train[:, args.samples_signal])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)    
    val_data = torch.utils.data.TensorDataset(dataset_valid[:, :args.samples_signal], dataset_valid[:, args.samples_signal])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, shuffle = True)
else:
    train_data = torch.utils.data.TensorDataset(dataset_train[:, :args.samples_signal], dataset_train[:, args.samples_signal:])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)    
    val_data = torch.utils.data.TensorDataset(dataset_valid[:, :args.samples_signal], dataset_valid[:, args.samples_signal:])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, shuffle = True)    

print("\n---------------------- complete loading data ---------------------------\n")

'''-------------------------------------------------------------------------'''
'''---------------------------- build model --------------------------------'''
'''-------------------------------------------------------------------------'''
#train_data.tensors
if args.model == "autoencoder":
    model = Model.AutoEncoder(args)  
    #optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr= args.lr)
    criterion = nn.MSELoss()
elif args.model_function == "detect_mass":
    model = Model.MLP(args = args, n_feature = args.samples_signal, n_output = 2)
    #optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    #criterion = nn.MSELoss() 
    criterion = nn.CrossEntropyLoss()
else:
    model = Model.MLP(args = args, n_feature = args.samples_signal, n_output = len(type_input))
    #optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    criterion = nn.MSELoss() 
  
model = nn.DataParallel(model, device_ids = list(eval(args.cuda_visible_devices))).cuda()   
print(model)
# Loop over epochs.
if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint_dir = Path(args.direction_workspace,'save', args.model, args.model_function, "sensors_" + str(args.sensors), \
                          str(args.batch_file), "ratio downsample " + str(args.ratio_downsample)).joinpath('checkpoint')
    checkpoint_dir.mkdir(parents = True, exist_ok = True)
    checkpoint = checkpoint_dir.joinpath(args.model_functiona).with_suffix('.pth')
    checkpoint = torch.load(checkpoint)
    args, start_epoch, best_val_loss = model.load_checkpoint(args, checkpoint)
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = math.inf
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)


'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''
if not args.pretrained:
    
    loss_val = []
    try:
        for epoch in range(start_epoch, args.epochs + 1):

            epoch_start_time = time.time()            

            train_loss = 0
            val_loss = 0
            if args.model == "autoencoder": 
                ''' input mode 1 '''                
                #process_control = tqdm(list(enumerate(train_loader)))
                for i, batch in tqdm(list(enumerate(train_loader))):
                    #print("*******8\n traindataset: batch[0].transpose_(0, 1).shape",batch[0].transpose_(0, 1).shape,"\n****** i: ", len(list(enumerate(train_loader)) ) )
                    cur_loss = process_data.train(args, model, optimizer, criterion, batch)
                    #print("\r | current epoch ", epo, "| cur_loss ", cur_loss )
                    train_loss = train_loss + cur_loss
                #process_control.set_description("| current epoch: %i | current loss %8.8f " %(epoch, train_loss))                
                train_loss = train_loss / (i+1)
                for i, batch in tqdm(list(enumerate(val_loader))):          
                    cur_val_loss = process_data.evaluate(args, model, criterion, batch)
                    val_loss = val_loss + cur_val_loss
                val_loss = val_loss / (i+1)
            else:
                '''input mode 2 '''
                for i, batch in tqdm(list(enumerate(train_loader))):
                    #print("*******8\n traindataset: batch[0].transpose_(0, 1).shape",batch[0].transpose_(0, 1).shape,"\n****** i: ", len(list(enumerate(train_loader)) ) )
                    cur_loss = process_data.train_DNN(args, model, optimizer, criterion, batch)
                    #print("\r | current epoch ", epo, "| cur_loss ", cur_loss )
                    train_loss = train_loss + cur_loss
                train_loss = train_loss / (i+1)
                for i, batch in tqdm(list(enumerate(val_loader))):          
                    cur_val_loss = process_data.evaluate_DNN(args, model, criterion, batch)
                    val_loss = val_loss + cur_val_loss
                val_loss = val_loss / (i+1)
            
            loss_val.append(val_loss) 
            print("| current epoch: %i | current loss %8.8f " %(epoch, train_loss))                                                       
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 89)                                           
            
            if (epoch % args.save_interval) == 0:
                # Save the model if the validation loss is the best we've seen so far.
                # is_best = val_loss > best_val_loss
                # best_val_loss = max(val_loss, best_val_loss)
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'args':args
                                    }
                model.module.save_checkpoint(model_dictionary, is_best)
        loss_val = np.array(loss_val)                    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if if_generate_output:
            
               
    process_data.generate_output_DNN_piece(args, epoch, model, criterion, device, gen_dataset, norm_scale)
        
  