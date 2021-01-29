#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:51:43 2020

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
from random import choices
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')

import simpleModel as Model
import configuration
import preprocess_data
import process_data
import create_dataset_for_autoencoder
from process_data import TrainAutoEncoder as TrainAutoEncoder
args = configuration.args
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
#os.environ["CUDA_VISIBLE_DEVICES"] = '2, 5'
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3, 4'

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
#cuda = torch.device('cuda:5')     # Default CUDA device
#cuda1 = torch.device('cuda:1')
#cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)
type_input = np.array(eval(args.type_input))

args.threshold_data_type = 0

'''-------------------------------------------------------------------------'''
'''--------------------------- create dataset ------------------------------'''
'''-------------------------------------------------------------------------'''

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

num_data_T = int(len(dataset)/args.cuda_devices) * args.cuda_devices
dataset = dataset[:num_data_T]

#np.argwhere(np.isnan(dataset))
'''-------------------------------------------------------------------------'''

filename_corr_weather = 'datasets_PCA_weather.pickle'
load_dir = Path(args.direction_data_weather, 'dataset', 'corrcoeff_PCA', '01234567', '25').joinpath("combined dataset")
path_load = Path(load_dir).joinpath(filename_corr_weather)
with open(path_load , 'rb') as file:
    plate_ultrasonic_dataset = pickle.load(file)
   
corrcoef_T = plate_ultrasonic_dataset["corrcoef_T"]
datatime = plate_ultrasonic_dataset["datatime"]
temperature = plate_ultrasonic_dataset["temperature"]
humidity = plate_ultrasonic_dataset["humidity"]
brightness = plate_ultrasonic_dataset["brightness"]
tag = plate_ultrasonic_dataset["tag"]
index_precipitation_all = plate_ultrasonic_dataset["index_precipitation_all"]
index_little_rain = plate_ultrasonic_dataset["index_little_rain"]
index_rain = plate_ultrasonic_dataset["index_rain"] 
index_snow = plate_ultrasonic_dataset["index_snow"]
index_mix = plate_ultrasonic_dataset["index_mix"]
null_index = plate_ultrasonic_dataset["null_index"]

corrcoef = corrcoef_T[0, ::10]
'''-------------------------------------------------------------------------'''

if args.threshold_data_type > 0:
    dataset = dataset[np.where(corrcoef > args.threshold_data_type)[0], :]    

index_shuffle = list(range(len(dataset)))
if if_shuffle_data:
    random.shuffle(index_shuffle)
index_shuffle = np.array(index_shuffle)
ratio_train = list(map(float, args.divider_set[0]))
divider_set = create_dataset_for_autoencoder.create_dataset_divider(args, index_shuffle, ratio_train = ratio_train)

dataset = torch.from_numpy(dataset)
dataset_train = dataset[index_shuffle[divider_set[1]: divider_set[2]]]
dataset_valid = dataset[index_shuffle[divider_set[0]: divider_set[1]]]     
dataset_test = dataset[index_shuffle[divider_set[2]: divider_set[3]]]

if args.model == "autoencoder" or args.model_function == "detect_mass":
    '''
    if args.model == "autoencoder":
        dataset_train = dataset  
    '''           
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

from process_data import TrainAutoEncoder as TrainAutoEncoder

if args.model == "autoencoder": 
    name_model_T = ["MEELoss"] # "MSELoss" "CorrLoss" "MEELoss"
    model_T = []; optimizer_T = []; criterion = []; train_model_T = []
    for i in range(len(name_model_T)):
        model_T.append(Model.AutoEncoder(args, name_model_T[i]))
        optimizer_T.append(optim.Adam(model_T[i].parameters(), lr = args.lr))
        #model_T[i] = nn.DataParallel(model_T[i], device_ids = list(eval(args.cuda_visible_devices))).cuda()
        model_T[i] = nn.DataParallel(model_T[i], device_ids = [0, 1]).cuda()
        print(model_T[i])             
    criterion_T = [Model.loss_mee] # nn.MSELoss(), Model.loss_corr, Model.loss_mee
    for i in range(len(name_model_T)):
        train_model_T.append(TrainAutoEncoder(args, model_T[i], optimizer_T[i], criterion_T[i], name_model_T[i]))
        
elif args.model_function == "detect_mass":
    model = Model.MLP(args = args, n_feature = args.samples_signal, n_output = 2)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()
else:
    model = Model.MLP(args = args, n_feature = args.samples_signal, n_output = len(type_input))
    #optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    criterion = nn.MSELoss() 

#model.module.encoder[0].weight   
#model = nn.DataParallel(model, device_ids = list(eval(args.cuda_visible_devices))).cuda()   
#print(model)

# Loop over epochs.
if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint = model.locate_checkpoint()
    args, start_epoch, best_val_loss = model.load_checkpoint(args, checkpoint)
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = []
    for i in range(len(train_model_T)):
        best_val_loss.append(math.inf)
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)

gen_dataset = plate_ultrasonic_dataset_T
#gen_dataset['data_sonic'] = gen_dataset.pop('dataset_sonic')
gen_dataset.update({'file_selected': ['Rawdata_data_17560.mat']})
gen_dataset['temperature'][np.where(gen_dataset['temperature'] < -50)] = gen_dataset['temperature'][np.where(gen_dataset['temperature'] < -50)[0]+1] 

'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''

if not args.pretrained:
    
    loss_val = []; reference = ["MSELoss"];
    try:
        for epoch in range(start_epoch, args.epochs + 1):

            epoch_start_time = time.time()            
            train_loss = np.zeros(len(train_model_T))
            val_loss = np.zeros(len(train_model_T))

            for i, batch in tqdm(list(enumerate(train_loader))):
                #print("*******8\n traindataset: batch[0].transpose_(0, 1).shape",batch[0].transpose_(0, 1).shape,"\n****** i: ", len(list(enumerate(train_loader)) ) )               
                #i, batch = list(enumerate(train_loader))[0]
                for k in range(len(train_model_T)):
                    train_model = train_model_T[k]
                    cur_loss = train_model.train(batch)
                    #print("\r | current epoch ", epo, "| cur_loss ", cur_loss )
                    train_loss[k] = train_loss[k] + cur_loss               
            train_loss = train_loss / (i+1)
            for i, batch in tqdm(list(enumerate(val_loader))):  
                for k in range(len(train_model_T)):
                    train_model = train_model_T[k]
                    cur_val_loss = train_model.evaluate(batch)
                    val_loss[k] = val_loss[k] + cur_val_loss
                val_loss = val_loss / (i+1)
            
            for k in range(len(train_model_T)):
                train_model  = train_model_T[k]
                print("| %s | current epoch: %i | current loss %8.8f " %(train_model.name_model, epoch, train_loss[k]))                                                       
                print('-' * 89)
                print('| %s | end of epoch %i  | time: %5.2f s | valid loss %6.5f | ' %(train_model.name_model, epoch, (time.time() - epoch_start_time), val_loss[k]))
                print('-' * 89)                                           
                   
            loss_val.append(val_loss)            
            if (epoch % args.save_interval) == 0:
                # Save the model if the validation loss is the best we've seen so far.
                # is_best = val_loss > best_val_loss
                # best_val_loss = max(val_loss, best_val_loss)
                for k in range(len(train_model_T)):
                    train_model  = train_model_T[k]; model = model_T[k]; optimizer = optimizer_T[k]
                    model.module.save_checkpoint(val_loss[k], best_val_loss[k], epoch, model, optimizer)               
                    train_model.generate_output_sample(epoch, gen_dataset, norm_scale, reference)
                
        loss_val = np.array(loss_val)                    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if if_generate_output:
    
    direction_ultrasonic_original = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", \
                                         "sensors_0_1").joinpath(str(args.batch_file))
    list_pickle_file = glob.glob(str(direction_ultrasonic_original) + '/*.pickle')
    list_pickle_file.sort()    
     
    try:
        for n in tqdm(range(len(list_pickle_file))):
            with open(list_pickle_file[n] , 'rb') as handle:
                gen_dataset = pickle.load(handle) 
            print(list_pickle_file[n])
                                  
            for k in range(len(train_model_T)):
                train_model  = train_model_T[k]         
                train_model.generate_output(epoch, gen_dataset, norm_scale)        
           
        train_model.generate_figure()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting early')
    
      