#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:45:48 2020

@author: yang.kang
"""
import scipy.io
import numpy as np
import pickle
import os, os.path
import glob
from datetime import datetime, timedelta 
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')

import preprocess_data
import configuration

args = configuration.args

def create_dataset_divider(args, index_shuffle, ratio_train):
    
    divider_set = []
    num_data_T = int(int((len(index_shuffle)/args.cuda_devices)) * args.cuda_devices)
    temp_divider = 0; k = 0
    while temp_divider < num_data_T:
        divider_set.append(temp_divider)
        temp_divider = int(ratio_train[k] * num_data_T)
        while  (temp_divider % args.cuda_devices) != 0:
           temp_divider = temp_divider - 1    
        k = k + 1
    divider_set.append(temp_divider)               
    return divider_set


def create_scattering_dataset(args):
    
    save_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", 
                    "sensors_0_1").joinpath(str(args.batch_file))
    save_dir.mkdir(parents = True, exist_ok = True)     
    
    direction_ultrasonic_original1 = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", \
                                         "sensors_" + str(0)).joinpath(str(args.batch_file))
    list_pickle_file1 = glob.glob(str(direction_ultrasonic_original1) + '/*.pickle')
    list_pickle_file1.sort()  
    direction_ultrasonic_original2 = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", \
                                         "sensors_" + str(7)).joinpath(str(args.batch_file))
    list_pickle_file2 = glob.glob(str(direction_ultrasonic_original2) + '/*.pickle')
    list_pickle_file2.sort()  
    scatter_times_list = 10 * np.array([0, 0.2, 0, 0.4, 0, 0.6, 0, 0.8, 0, 1, 0, 0.1, 
                                        0, 0.3, 0, 0.5, 0, 0.7, 0, 0.9, 0])
    scatter_factor_T = []
    '''
    load_dir = Path(args.direction_workspace, 'dataset', "autoencoder_corrcoeff", args.sensors + " sensors", \
                    str(args.samples_signal), str(args.batch_file), "ratio_downsample_" + str(args.ratio_downsample))                  
    if args.scatter_factor > 0:
        load_dir = load_dir.joinpath(str(args.scatter_factor))
    if args.threshold_data_type > 0:
        load_dir = load_dir.joinpath(str(args.threshold_data_type))
    load_dir = load_dir.joinpath("combined dataset") 
    path_load = Path(load_dir).joinpath("scatter_factor.pickle")
    
    with open(path_load, 'rb') as file:
        scatter_factor_T = pickle.load(file)
        print(path_load, "has been created\n")
    '''
    k = 0 
    #startpoint = 0       
    for n in tqdm(range(len(list_pickle_file1))):
        with open(list_pickle_file1[n] , 'rb') as handle:
            plate_ultrasonic_dataset1 = pickle.load(handle) 
        print(list_pickle_file1[n])
        
        with open(list_pickle_file2[n] , 'rb') as handle:
            plate_ultrasonic_dataset2 = pickle.load(handle) 
        print(list_pickle_file2[n])            

        gen_dataset = plate_ultrasonic_dataset1
        gen_dataset['file_selected'] = gen_dataset.pop('file_selcted')
        if args.scatter_factor > 0:
            data_sonic = plate_ultrasonic_dataset1['data_sonic']
            
            #scatter_times = choices(scatter_times_list, k = 1)
            scatter_times = scatter_times_list[k]
            print(f"\n scatter_times: {scatter_times}\n")            
            scatter_factor = scatter_times * args.scatter_factor * np.ones(len(data_sonic))
            scatter_factor_T = scatter_factor_T + list(scatter_factor)
            
            #endpoint = startpoint + len(data_sonic)
            #scatter_factor = scatter_factor_T[startpoint:endpoint]
            #startpoint = endpoint

            data_sonic_scatter = scatter_factor[0] * plate_ultrasonic_dataset2['data_sonic']
            data_sonic = data_sonic + data_sonic_scatter
            print("\n", data_sonic[0, 0], plate_ultrasonic_dataset1['data_sonic'][0, 0], data_sonic_scatter[0, 0], "\n")
        
            gen_dataset['data_sonic'] = data_sonic
            gen_dataset.update( {'scatter_factor' : scatter_factor})
            gen_dataset.update( {'scatter_example' : [data_sonic[0, 0], plate_ultrasonic_dataset1['data_sonic'][0, 0], data_sonic_scatter[0, 0]]})

        if k >= len(scatter_times_list)-1:
            k = 0
        else:
            k = k+1            
        
        path_save = Path(save_dir).joinpath(list_pickle_file1[n].split('/')[-1])        
        with open(path_save, 'wb') as handle:
            pickle.dump(gen_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
        print(path_save, "has been created\n")  
        
    scatter_factor_T = np.array(scatter_factor_T)
    scatter_save_dir = save_dir.joinpath("scatter_factor")
    scatter_save_dir.mkdir(parents = True, exist_ok = True)      
    path_save = Path(scatter_save_dir).joinpath("scatter_factor.pickle")       
    with open(path_save, 'wb') as handle:
        pickle.dump(scatter_factor_T, handle, protocol = pickle.HIGHEST_PROTOCOL)        
        print(path_save, "has been created\n")  

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
    
# =============================================================================
#     plt.figure()
#     for i in range(0, len(plate_ultrasonic_dataset['datatime']), 50):
#         plt.plot(plate_ultrasonic_dataset['data_sonic'][i])
#         plt.title(str(i) + " " + str(plate_ultrasonic_dataset['datatime'][i]))
#         plt.show()
#         plt.pause(0.3)
#         plt.clf()        
#     
# =============================================================================
    
    
    
    return plate_ultrasonic_dataset_T


def create_batch_dataset(args):
    
    sensors = int(args.sensors)
    measurements_with_mass = preprocess_data.locate_measurements_with_mass(filename = args.direction_mass_information) 
    
    for index in range(len(args.folder_matfile)):
        
        try:
            direction_data_local =  args.direction_matdata + args.folder_matfile[index][0]
            list_mat_file = glob.glob(direction_data_local + '/*.mat')
            list_mat_file.sort()    
            start_series = 0
            
            for start_file in range(start_series, len(list_mat_file), args.batch_file):
                
                if (start_file + args.batch_file) < len(list_mat_file): 
                    end_file = start_file + args.batch_file
                else:
                    end_file = len(list_mat_file)
                        
                data_sonic, datatime, temperature, pressure, brightness, humidity, tag, file_selcted \
                = preprocess_data.read_files_batch(args, measurements_with_mass, list_mat_file, start_file, end_file)
                          
                if (np.shape(data_sonic)[0] != np.shape(datatime)[0]):
                    break
                    
                plate_ultrasonic_dataset = {'file_selcted': np.array(file_selcted), \
                                            'data_sonic': data_sonic[:, sensors, :],\
                                            'datatime':  datatime, \
                                            'temperature': temperature, \
                                            'pressure': pressure, \
                                            'brightness': brightness, \
                                            'humidity': humidity, \
                                            'tag': tag}
                
                number_file = list_mat_file[end_file - 1].split("_")[-1].split(".")[0]
                filename_save = 'ultrasonic_orginial_' + number_file + '.pickle'
                save_dir = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" + str(args.sensors)\
                                ).joinpath(str(args.batch_file))
                save_dir.mkdir(parents = True, exist_ok = True)
                path_save = Path(save_dir).joinpath(filename_save)
                
                with open(path_save, 'wb') as handle:
                    pickle.dump(plate_ultrasonic_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
                print(path_save, "has been created\n") 
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


if __name__ == '__main__':
    
    args = configuration.args
    
