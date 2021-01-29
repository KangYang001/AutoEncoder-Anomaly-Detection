#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:43:57 2021

@author: yang.kang
"""
import h5py
import numpy as np

import scipy.io
import pickle
import os, os.path
import glob
from datetime import datetime, timedelta 
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')

import preprocess_data
import configuration

args = configuration.args



def create_dataset(args):
    
    ratio_downsample = 1
    '''
    startpoint = 0
    endpoint = 4000000
    '''
    filename_save = 'ultrasonic_orginial_downsample_'+ str(ratio_downsample) +'.pickle'
    load_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" + str(args.sensors), \
                    str(args.batch_file)).joinpath("ratio_downsample_" + str(ratio_downsample))
    path_load = Path(load_dir).joinpath(filename_save)
    
    if os.path.isfile(str(path_load)):  
        with open(path_load , 'rb') as handle:
            plate_ultrasonic_dataset_T = pickle.load(handle)         
        print(path_load, "has been created\n") 
        
        dataset_sonic = plate_ultrasonic_dataset_T['dataset_sonic'][::args.ratio_downsample, :]
        datatime =  plate_ultrasonic_dataset_T['datatime'][::args.ratio_downsample]
        temperature = plate_ultrasonic_dataset_T['temperature'][::args.ratio_downsample]
        pressure = plate_ultrasonic_dataset_T['pressure'][::args.ratio_downsample]
        brightness = plate_ultrasonic_dataset_T['brightness'][::args.ratio_downsample]
        humidity = plate_ultrasonic_dataset_T['humidity'][::args.ratio_downsample]
        tag = plate_ultrasonic_dataset_T['tag'][::args.ratio_downsample]      
        '''
        dataset_sonic = plate_ultrasonic_dataset_T['dataset_sonic'][startpoint:endpoint, :]
        datatime =  plate_ultrasonic_dataset_T['datatime'][startpoint:endpoint]
        temperature = plate_ultrasonic_dataset_T['temperature'][startpoint:endpoint]
        pressure = plate_ultrasonic_dataset_T['pressure'][startpoint:endpoint]
        brightness = plate_ultrasonic_dataset_T['brightness'][startpoint:endpoint]
        humidity = plate_ultrasonic_dataset_T['humidity'][startpoint:endpoint]
        tag = plate_ultrasonic_dataset_T['tag'][startpoint:endpoint]
        '''          
    else:
        direction_ultrasonic_original = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", \
                                         "sensors_" + str(args.sensors)).joinpath(str(args.batch_file))
        list_pickle_file = glob.glob(str(direction_ultrasonic_original) + '/*.pickle')
        list_pickle_file.sort()   
    
        dataset_sonic = np.empty((0, 2000))
        datatime = np.empty((0))
        temperature = np.empty((0))
        pressure = np.empty((0))
        brightness = np.empty((0))
        humidity = np.empty((0))
        tag = np.empty((0))
        
        for n in tqdm(range(len(list_pickle_file))):
            with open(list_pickle_file[n] , 'rb') as handle:
                plate_ultrasonic_dataset = pickle.load(handle) 
            print(list_pickle_file[n])
            temp_pickle_file = Path(list_pickle_file[n])
            temp_hdf5_file = Path(temp_pickle_file.parent, "HDF5").joinpath(temp_pickle_file.name.replace("pickle", "hdf5"))
           
            f = h5py.File(u'/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/dataset/ultrasonic_orginial/sensors_6/50/HDF5/ultrasonic_orginial_00050.hdf5', "w")    # mode = {'w', 'r', 'a'}
            #f.create_dataset('data_sonic', data = plate_ultrasonic_dataset['data_sonic'])
            #f.create_dataset('datatime', data = plate_ultrasonic_dataset['datatime'])
            f.create_dataset('temperature', data = plate_ultrasonic_dataset['temperature'])
            f.create_dataset('pressure', data = plate_ultrasonic_dataset['pressure'])
            f.create_dataset('brightness', data = plate_ultrasonic_dataset['brightness'])
            f.create_dataset('humidity', data = plate_ultrasonic_dataset['humidity'])
            f.create_dataset('tag', data = plate_ultrasonic_dataset['tag'])
            f.close()
            
            







            dataset_sonic = np.concatenate((dataset_sonic, plate_ultrasonic_dataset['data_sonic'][::args.ratio_downsample, :]), axis = 0)
            datatime = np.concatenate((datatime, plate_ultrasonic_dataset['datatime'][::args.ratio_downsample]), axis = 0)
            temperature = np.concatenate((temperature, plate_ultrasonic_dataset['temperature'][::args.ratio_downsample]), axis = 0)
            pressure = np.concatenate((pressure, plate_ultrasonic_dataset['pressure'][::args.ratio_downsample]), axis = 0)
            brightness = np.concatenate((brightness, plate_ultrasonic_dataset['brightness'][::args.ratio_downsample]), axis = 0)
            humidity = np.concatenate((humidity, plate_ultrasonic_dataset['humidity'][::args.ratio_downsample]), axis = 0)
            tag = np.concatenate((tag, plate_ultrasonic_dataset['tag'][::args.ratio_downsample]), axis = 0)
    
    plate_ultrasonic_dataset_T = {'dataset_sonic': dataset_sonic,\
                                  'datatime':  datatime, \
                                  'temperature': temperature, \
                                  'pressure': pressure, \
                                  'brightness': brightness, \
                                  'humidity': humidity, \
                                  'tag': tag}    
        
    filename_save = 'ultrasonic_orginial_downsample_'+ str(args.ratio_downsample) +'.pickle'
    #filename_save = 'ultrasonic_orginial_downsample_2018.pickle'
    save_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" +  str(args.sensors), \
                    str(args.batch_file)).joinpath("ratio_downsample_" + str(args.ratio_downsample))
    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = Path(save_dir).joinpath(filename_save)

    with open(path_save, 'wb') as handle:
        pickle.dump(plate_ultrasonic_dataset_T, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n") 
    
    return plate_ultrasonic_dataset_T

















def create_dataset(args):
    
    ratio_downsample = 1
    '''
    startpoint = 0
    endpoint = 4000000
    '''
    filename_save = 'ultrasonic_orginial_downsample_'+ str(ratio_downsample) +'.pickle'
    load_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" + str(args.sensors), \
                    str(args.batch_file)).joinpath("ratio_downsample_" + str(ratio_downsample))
    path_load = Path(load_dir).joinpath(filename_save)
    
    if os.path.isfile(str(path_load)):  
        with open(path_load , 'rb') as handle:
            plate_ultrasonic_dataset_T = pickle.load(handle)         
        print(path_load, "has been created\n") 
        
        dataset_sonic = plate_ultrasonic_dataset_T['dataset_sonic'][::args.ratio_downsample, :]
        datatime =  plate_ultrasonic_dataset_T['datatime'][::args.ratio_downsample]
        temperature = plate_ultrasonic_dataset_T['temperature'][::args.ratio_downsample]
        pressure = plate_ultrasonic_dataset_T['pressure'][::args.ratio_downsample]
        brightness = plate_ultrasonic_dataset_T['brightness'][::args.ratio_downsample]
        humidity = plate_ultrasonic_dataset_T['humidity'][::args.ratio_downsample]
        tag = plate_ultrasonic_dataset_T['tag'][::args.ratio_downsample]      
        '''
        dataset_sonic = plate_ultrasonic_dataset_T['dataset_sonic'][startpoint:endpoint, :]
        datatime =  plate_ultrasonic_dataset_T['datatime'][startpoint:endpoint]
        temperature = plate_ultrasonic_dataset_T['temperature'][startpoint:endpoint]
        pressure = plate_ultrasonic_dataset_T['pressure'][startpoint:endpoint]
        brightness = plate_ultrasonic_dataset_T['brightness'][startpoint:endpoint]
        humidity = plate_ultrasonic_dataset_T['humidity'][startpoint:endpoint]
        tag = plate_ultrasonic_dataset_T['tag'][startpoint:endpoint]
        '''          
    else:
        direction_ultrasonic_original = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", \
                                         "sensors_" + str(args.sensors)).joinpath(str(args.batch_file))
        list_pickle_file = glob.glob(str(direction_ultrasonic_original) + '/*.pickle')
        list_pickle_file.sort()   
    
        dataset_sonic = np.empty((0, 2000))
        datatime = np.empty((0))
        temperature = np.empty((0))
        pressure = np.empty((0))
        brightness = np.empty((0))
        humidity = np.empty((0))
        tag = np.empty((0))
        
        for n in tqdm(range(len(list_pickle_file))):
            with open(list_pickle_file[n] , 'rb') as handle:
                plate_ultrasonic_dataset = pickle.load(handle) 
            print(list_pickle_file[n])
            #gen_dataset = plate_ultrasonic_dataset
            #process_data.generate_output(args, epoch, model, gen_dataset, criterion, device)
            dataset_sonic = np.concatenate((dataset_sonic, plate_ultrasonic_dataset['data_sonic'][::args.ratio_downsample, :]), axis = 0)
            datatime = np.concatenate((datatime, plate_ultrasonic_dataset['datatime'][::args.ratio_downsample]), axis = 0)
            temperature = np.concatenate((temperature, plate_ultrasonic_dataset['temperature'][::args.ratio_downsample]), axis = 0)
            pressure = np.concatenate((pressure, plate_ultrasonic_dataset['pressure'][::args.ratio_downsample]), axis = 0)
            brightness = np.concatenate((brightness, plate_ultrasonic_dataset['brightness'][::args.ratio_downsample]), axis = 0)
            humidity = np.concatenate((humidity, plate_ultrasonic_dataset['humidity'][::args.ratio_downsample]), axis = 0)
            tag = np.concatenate((tag, plate_ultrasonic_dataset['tag'][::args.ratio_downsample]), axis = 0)
    
    plate_ultrasonic_dataset_T = {'dataset_sonic': dataset_sonic,\
                                  'datatime':  datatime, \
                                  'temperature': temperature, \
                                  'pressure': pressure, \
                                  'brightness': brightness, \
                                  'humidity': humidity, \
                                  'tag': tag}    
        
    filename_save = 'ultrasonic_orginial_downsample_'+ str(args.ratio_downsample) +'.pickle'
    #filename_save = 'ultrasonic_orginial_downsample_2018.pickle'
    save_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" +  str(args.sensors), \
                    str(args.batch_file)).joinpath("ratio_downsample_" + str(args.ratio_downsample))
    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = Path(save_dir).joinpath(filename_save)

    with open(path_save, 'wb') as handle:
        pickle.dump(plate_ultrasonic_dataset_T, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n") 
    
    return plate_ultrasonic_dataset_T
