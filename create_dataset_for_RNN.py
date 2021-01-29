#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:43:06 2020

@author: yang.kang
"""

import numpy as np
import pickle
import glob
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys
from tqdm import tqdm
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA/Eigenvector_Feature/code')

import configuration
import precipitation_2018_03_21_2019_03_21 as precipitation_2018
import precipitation_2019_03_22_2020_03_17 as precipitation_2019

'''-------------------------------------------------------------------------'''
'''------------------------------ function ---------------------------------'''
'''-------------------------------------------------------------------------'''
    
def create_measurement_dataset(direction_files):
    
    list_pickle_file = glob.glob(direction_files  + '/*.pickle')
    list_pickle_file.sort()
     
    corrcoef = []; measure_datatime = []; measure_temperature = []
    measure_humidity = []; measure_pressure = []; measure_brightness = [];
    eigenvector = []; tag = []; lable_eigenvector = []
    
    for i in range(len(list_pickle_file)):
        
        filename = list_pickle_file[i]
        print(f"{filename} has been loaded")
        with open(filename, 'rb') as file:
            plate_ultrasonic_dataset = pickle.load(file)
    
        corrcoef.append(plate_ultrasonic_dataset['correlation_coefficient'])
        measure_datatime.append(plate_ultrasonic_dataset['datetime'])
        measure_temperature.append(plate_ultrasonic_dataset['temperature'])
        measure_humidity.append(plate_ultrasonic_dataset['humidity'])
        measure_pressure.append(plate_ultrasonic_dataset['pressure'])
        measure_brightness.append(plate_ultrasonic_dataset['brightness'])
        tag.append(plate_ultrasonic_dataset['tag'])
        eigenvector.append(plate_ultrasonic_dataset['eigenvector'])
        if np.sum(plate_ultrasonic_dataset['tag']) > 1000:
            lable_eigenvector.append(1)
        else:
            lable_eigenvector.append(0)
    lable_eigenvector = np.array(lable_eigenvector)
        
    print("start to create measurement dataset\n")
    corrcoef1 = np.empty((8, 0)); measure_datatime1 = np.empty((0))
    measure_temperature1 = np.empty((0)); measure_humidity1 = np.empty((0)); 
    measure_pressure1 = np.empty((0)); measure_brightness1 = np.empty((0)); tag1 = np.empty((0))
    
    for i in tqdm(range(len(corrcoef))):
        corrcoef1 = np.concatenate((corrcoef1, corrcoef[i]), axis = 1)
        measure_datatime1 = np.concatenate((measure_datatime1, measure_datatime[i]), axis = 0)
        measure_temperature1 = np.concatenate((measure_temperature1, measure_temperature[i]), axis = 0)
        measure_humidity1 = np.concatenate((measure_humidity1, measure_humidity[i]), axis = 0)
        measure_pressure1 = np.concatenate((measure_pressure1, measure_pressure[i]), axis = 0)
        measure_brightness1 = np.concatenate((measure_brightness1, measure_brightness[i]), axis = 0)
        tag1 = np.concatenate((tag1, tag[i]), axis = 0)
    print("finish creating measurement dataset\n")
    
    eigenvector = np.array(eigenvector)
    return corrcoef1, measure_datatime1, measure_temperature1, measure_humidity1, measure_pressure1, measure_brightness1, tag1, eigenvector, lable_eigenvector


def delete_null_elements_dataset(corrcoef_T, measure_datatime, measure_temperature, measure_humidity, \
                                 measure_pressure, measure_brightness, tag):

    null_brightness_index = np.argwhere(np.isnan(measure_brightness))
    corrcoef_T = np.delete(corrcoef_T, null_brightness_index, 1)
    measure_datatime = np.delete(measure_datatime, null_brightness_index, 0)
    measure_temperature = np.delete(measure_temperature, null_brightness_index, 0)
    measure_humidity = np.delete(measure_humidity, null_brightness_index, 0)
    measure_pressure = np.delete(measure_pressure, null_brightness_index, 0)
    measure_brightness = np.delete(measure_brightness, null_brightness_index, 0)
    tag = np.delete(tag, null_brightness_index, 0)
    
    null_corrcoef_index = np.argwhere(np.isnan(corrcoef_T))
    null_corrcoef_index = np.array(list(set(null_corrcoef_index[:, 1])))
    corrcoef_T = np.delete(corrcoef_T, null_corrcoef_index, 1)
    measure_datatime = np.delete(measure_datatime, null_corrcoef_index, 0)
    measure_temperature = np.delete(measure_temperature, null_corrcoef_index, 0)
    measure_humidity = np.delete(measure_humidity, null_corrcoef_index, 0)
    measure_pressure = np.delete(measure_pressure, null_corrcoef_index, 0)
    measure_brightness = np.delete(measure_brightness, null_corrcoef_index, 0)
    tag = np.delete(tag, null_corrcoef_index, 0)
    
    measure_temperature[np.where(measure_temperature < -50)] = measure_temperature[np.where(measure_temperature < -50)[0] + 1]
  
    return corrcoef_T, measure_datatime, measure_temperature, measure_humidity, measure_pressure, measure_brightness, tag


def get_time_input(measure_datatime):
    
    measure_year = []
    measure_day = []
    measure_second = []
    for i in tqdm(range(len(measure_datatime))):
        measure_year.append(measure_datatime[i].year)
        measure_day.append(int(measure_datatime[i].strftime('%j')))
        hh, mm , ss = map(int, measure_datatime[i].strftime('%H:%M:%S').split(':'))
        measure_second.append(ss + 60*(mm + 60*hh))
    
    measure_year = np.array(measure_year)
    measure_day = np.array(measure_day)
    measure_second = np.array(measure_second)    
            
    return measure_year, measure_day, measure_second
    
def get_batch_dataset(args, source):
    
    i = 0
    seq_len = args.len_slice
    data = []
    while seq_len == args.len_slice:
        seq_len = min(args.len_slice, len(source) - i * args.len_slice) # sourcej即dataset   防止越界
        data.append(source[i * seq_len: (i+1) * seq_len]) # [ seq_len * batch_size * feature_size ]
        #target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
        i = i + 1
    data.pop()    
    data =  np.array(data)
    
    return data

def normalize_data(data):
    
    data_new = (data - np.min(data))/(np.max(data) - np.min(data))
    range_data = [np.max(data), np.min(data)]
    
    return data_new, range_data

def create_dataset(args, if_normalize_data = True, if_repeat_data = False):

    corrcoef_T, measure_datatime, measure_temperature, measure_humidity, measure_pressure, measure_brightness, tag, eigenvector_T, label_eigenvector = \
    create_measurement_dataset(args.direction_dataset)
    
    corrcoef_T, measure_datatime, measure_temperature, measure_humidity, measure_pressure, measure_brightness, tag = \
    delete_null_elements_dataset(corrcoef_T, measure_datatime, measure_temperature, measure_humidity, measure_pressure, \
                                 measure_brightness, tag)
    
    measure_year, measure_day, measure_second = get_time_input(measure_datatime)
    
    measure_brightness_log = np.log(measure_brightness)
    corrcoef = corrcoef_T[sensor]
    
    '''
    print("start to create precipitation datetime")
    index_precipitation_all = np.empty((0))
    for i in tqdm(range(len(precipitation_all_start))):
        index_precipitation_all  = np.concatenate((index_precipitation_all, np.where((measure_datatime > precipitation_all_start[i]) & \
                                                                            (measure_datatime < precipitation_all_end[i]))[0]), axis = 0)
    index_precipitation_all = np.array(list(map(int, index_precipitation_all))) 
    
    lable_precipitation = np.zeros(len(measure_datatime))
    lable_precipitation[index_precipitation_all] = 1
    '''
    direction_label_weather = Path(args.direction_label_weather).joinpath(str(len(measure_datatime)))
    path_load = Path(direction_label_weather).joinpath("label_weather.pickle")
    with open(path_load , 'rb') as file:
        label_weather_dict = pickle.load(file)
       
    label_weather = label_weather_dict["label_weather"]    
        
    dataset = np.array([corrcoef, measure_temperature, measure_humidity, measure_brightness_log, \
                        measure_second, measure_day,  measure_year, measure_datatime, label_weather])                

    dataset = dataset[:, ::args.ratio_downsample]
    
    range_data = []
    norm_scale = []
    if if_normalize_data:
        for i in range(1, 7):
            norm_scale.append([np.max(dataset[i]), np.min(dataset[i])])
            
        range_data.append([50, np.min(dataset[1])])
        dataset[1] = (dataset[1] - np.min(dataset[1]))/(50 - np.min(dataset[1]))
        range_data.append([100, np.min(dataset[2])])
        dataset[2] = (dataset[2] - np.min(dataset[2]))/(100 - np.min(dataset[2]))
        for i in range(3, 7):
            dataset[i], temp_range_data = normalize_data(dataset[i])
            range_data.append(temp_range_data)

    dataset = dataset.transpose((1, 0))
    range_data = np.array(range_data)
    norm_scale = np.array(norm_scale)
    dataset = get_batch_dataset(args, dataset)
    
    num_data_slice = len(dataset)
    flag = 0
    for i in tqdm(range(len(dataset))):
        if len(np.where(dataset[i, :, 8] == 2)[0]) > 0:
            tempdata = np.repeat(np.expand_dims(dataset[i, :, :], axis = 0), args.times_repeat_direct_solar_radiaiton, axis = 0)
            flag = 1                    
        elif len(np.where(dataset[i, :, 8] == 3)[0]) > 0:
            tempdata = np.repeat(np.expand_dims(dataset[i, :, :], axis = 0), args.times_repeat_subzero_temperature, axis = 0)
            flag = 1
        elif len(np.where(dataset[i, :, 8] == 4)[0]) > 0:
            tempdata = np.repeat(np.expand_dims(dataset[i, :, :], axis = 0), args.times_repeat_droplets, axis = 0)
            flag = 1
        elif len(np.where(dataset[i, :, 8] == 1)[0]) > 0:
            len_1 = len(np.where(dataset[i, :, 8] == 1)[0])
            if len_1 > int(len(dataset[i, :, 8])/2):
                tempdata = np.repeat(np.expand_dims(dataset[i, :, :], axis = 0), args.times_repeat_precipitation, axis = 0)            
            elif len_1 > int(len(dataset[i, :, 8])/4):    
                tempdata = np.repeat(np.expand_dims(dataset[i, :, :], axis = 0), 2, axis = 0)
            else:
                tempdata = np.repeat(np.expand_dims(dataset[i, :, :], axis = 0), 1, axis = 0)
            flag = 1
        if flag == 1:
            dataset = np.concatenate((dataset, tempdata), axis = 0)
            flag = 0        
    
    plate_ultrasonic_dataset = {"dataset": dataset,
                                "range_data": range_data,
                                "norm_scale": norm_scale,
                                "num_data_slice": num_data_slice}
    
    
    return plate_ultrasonic_dataset
    
'''-------------------------------------------------------------------------'''
'''-------------------------- hyperparameters ------------------------------'''
'''-------------------------------------------------------------------------'''

dict_keywords = ['file_selcted', 'datetime', 'temperature', 'pressure', 'brightness', 'humidity', \
                 'correlation_coefficient', 'dataset_feature', 'eigenvector', 'mean_data', 'tag']

# direction_files =  "/home/UFAD/yang.kang/Ultrasonics//Kang/Feature Data/PCA/Eigenvector_Feature/8 sensors/25"
sensor = 7
args = configuration.args

precipitation_all_start = np.concatenate((precipitation_2018.precipitation_all_start, precipitation_2019.precipitation_all_start), axis = 0)
precipitation_all_end = np.concatenate((precipitation_2018.precipitation_all_end, precipitation_2019.precipitation_all_end), axis = 0)
precipitation_little_rain_start = np.concatenate((precipitation_2018.precipitation_little_rain_start, precipitation_2019.precipitation_little_rain_start), axis = 0)
precipitation_little_rain_end = np.concatenate((precipitation_2018.precipitation_little_rain_end, precipitation_2019.precipitation_little_rain_end), axis = 0)
precipitation_rain_start = np.concatenate((precipitation_2018.precipitation_rain_start, precipitation_2019.precipitation_rain_start), axis = 0)
precipitation_rain_end = np.concatenate((precipitation_2018.precipitation_rain_end, precipitation_2019.precipitation_rain_end), axis = 0)
precipitation_snow_start = np.concatenate((precipitation_2018.precipitation_snow_start, precipitation_2019.precipitation_snow_start), axis = 0)
precipitation_snow_end = np.concatenate((precipitation_2018.precipitation_snow_end, precipitation_2019.precipitation_snow_end), axis = 0)
precipitation_mix_start = np.concatenate((precipitation_2018.precipitation_mix_start, precipitation_2019.precipitation_mix_start), axis = 0)
precipitation_mix_end = np.concatenate((precipitation_2018.precipitation_mix_end, precipitation_2019.precipitation_mix_end), axis = 0)

if_normalize_data = True
'''-------------------------------------------------------------------------'''
'''--------------------------- main function -------------------------------'''
'''-------------------------------------------------------------------------'''

if __name__ == "__main__":
    
    dataset = create_dataset(args, if_normalize_data)
    
    index_shuffle = list(range(len(dataset)))
    random.shuffle(index_shuffle)
    print(index_shuffle)    
    
    




