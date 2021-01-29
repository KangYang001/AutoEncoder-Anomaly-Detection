#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:39:48 2020

@author: yang.kang
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import scipy.io
import time
import csv
from datetime import datetime
###############################################################################
# user defined function
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def normalize_data(dataset, normalization_type = 0):
    
    print("start to normalize dataset")
    norm_scale = np.array([np.max(dataset), np.min(dataset)])
    if normalization_type == 0:
        dataset = (dataset - np.min(dataset))/(np.max(dataset) - np.min(dataset))
    else:            
        for i in tqdm(range(dataset.shape[0])):
            dataset[i,:] = (dataset[i,:] - min(dataset[i,:]))/(max(dataset[i,:]) - min(dataset[i,:]))
    print("complete to normalize dataset")    
    
    return dataset, norm_scale

def denormalize_data(dataset, norm_scale):
    
    print("start to denormalize dataset")
    dataset = dataset * (norm_scale[0]- norm_scale[1])+ norm_scale[1]
    print("complete to denormalize dataset")    
    
    return dataset

def reduce_matrix_dimension(measureDict, type_dataset):
    
    dict = ['Fs', 'd', 'e', 's', 't', 'y']
    
    if type_dataset == 0:
        
        measure_day = []
        measure_time = []
        measure_temperature = []
        measure_pressure = []
        measure_brightness = []
        measure_humidity = []
        measure_frequence = []
        
        n = len(measureDict[dict[1]][0])
    
        for i in range(n):
            measure_frequence.append(measureDict[dict[0]][0][i][0][0])
            measure_day.append(measureDict[dict[1]][0][i][0])
            measure_temperature.append(measureDict[dict[2]][0][i][0][0])
            measure_pressure.append(measureDict[dict[2]][0][i][0][1])
            measure_brightness.append(measureDict[dict[2]][0][i][0][2])
            measure_humidity.append(measureDict[dict[2]][0][i][0][3])
            measure_time.append(measureDict[dict[4]][0][i][0])
        
        if n < 400:
            for i in range(400 - n):
                measure_frequence.append(measureDict[dict[0]][0][n-1][0][0])
                measure_day.append(measureDict[dict[1]][0][n-1][0])
                measure_temperature.append(measureDict[dict[2]][0][n-1][0][0])
                measure_pressure.append(measureDict[dict[2]][0][n-1][0][1])
                measure_brightness.append(measureDict[dict[2]][0][n-1][0][2])
                measure_humidity.append(measureDict[dict[2]][0][n-1][0][3])
                measure_time.append(measureDict[dict[4]][0][n-1][0])
                
    elif type_dataset != 0:
        n = len(measureDict[dict[1]])
        
        measure_frequence = list(measureDict[dict[0]][0])
        measure_day = list(measureDict[dict[1]])
        measure_temperature = list(measureDict[dict[2]][:, 0])
        measure_pressure = list(measureDict[dict[2]][:, 1])
        measure_brightness = list(measureDict[dict[2]][:, 2])
        measure_humidity = list(measureDict[dict[2]][:, 3])
        measure_time = list(measureDict[dict[4]])
        
        if n < 400:
            for i in range(400 - n):
                measure_frequence.append(measureDict[dict[0]][0][n-1])
                measure_day.append(measureDict[dict[1]][n-1])
                measure_temperature.append(measureDict[dict[2]][n-1][0])
                measure_pressure.append(measureDict[dict[2]][n-1][1])
                measure_brightness.append(measureDict[dict[2]][n-1][2])
                measure_humidity.append(measureDict[dict[2]][n-1][3])
                measure_time.append(measureDict[dict[4]][n-1])        
            
    
    return np.array(measure_day), np.array(measure_time), np.array(measure_temperature), np.array(measure_pressure),\
    np.array(measure_brightness), np.array(measure_humidity), np.array(measure_frequence)

def locate_measurements_with_mass(filename):
    
    label_for_mass = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ' ')
        for row in csv_reader:
            if row == []:
                pass
            else:
                label_for_mass.append(list(map(int, row)))
                
    startpoint = label_for_mass[-6][0]
    endpoint = label_for_mass[-5][0]

    startpoint1 = label_for_mass[-4][0]
    endpoint1 = label_for_mass[-3][0]

    startpoint2 = label_for_mass[-2][0]
    endpoint2 = label_for_mass[-1][0]
    
    for i in range(startpoint + 1, endpoint):
        label_for_mass.append([i, 0, 399])    

    
    for i in range(startpoint1 + 1, endpoint1):
        label_for_mass.append([i, 0, 399])  
        
    for i in range(startpoint2 + 1, endpoint2):
        label_for_mass.append([i, 0, 399])      

    return np.array(label_for_mass)

def generate_timestamp(data_day, data_time):
    print("\nstart to generate time stamp\n")
    
    timestamp_T = []
    
    for i in range(np.shape(data_day)[0]):

        try:
            timestamp_T.append(datetime.strptime(data_day[i] + '/' + data_time[i].split(".")[0], '%Y/%m/%d/%H:%M:%S'))
        except ValueError:
            try:
                timestr = data_day[i] + '/' + data_time[i]
                timestamp = time.strptime(timestr.strip(), '%Y/%m/%d/%H:%M:%S.%f')
                timestamp = time.mktime(timestamp) + 0.0001
                timestamp_T.append(datetime.fromtimestamp(timestamp))
            except ValueError:
                timestr = data_day[i] + '/' + data_time[i]
                timestamp = time.strptime(timestr.strip(), '%Y/%m/%d/%H:%M:%S')
                timestamp = time.mktime(timestamp) + 0.0001
                timestamp_T.append(datetime.fromtimestamp(timestamp))                
                          
    print("\ncomplete generating time stamp\n")

    return np.array(timestamp_T)

def read_files_batch(args, measurements_with_mass, list_mat_file, start_file, end_file):
    
    type_dataset = 0
    signal_slice = list(map(int, args.signal_slice[0]))            
    measure_day = []; measure_time = []; measure_temperature = []; measure_pressure = []
    measure_brightness = []; measure_humidity = []; measure_frequence = []; file_selcted = []; tag = []
    dataset = np.empty((0, 8, (signal_slice[1] - signal_slice[0])))        
    for i in range(start_file, end_file):
        
        measureDict = scipy.io.loadmat(list_mat_file[i])
        file_selcted.append(list_mat_file[i])
        print(list_mat_file[i])
    
        temp_day, temp_time, temp_temperature, temp_pressure, temp_brightness, \
        temp_humidity, temp_frequene = reduce_matrix_dimension(measureDict, type_dataset)
        
        measure_day.append(temp_day)
        measure_time.append(temp_time)
        measure_temperature.append(temp_temperature)
        measure_pressure.append(temp_pressure)
        measure_brightness.append(temp_brightness)
        measure_humidity.append(temp_humidity)
        measure_frequence.append(temp_frequene)
        
        if type_dataset == 0:
            temp_data = measureDict['y'].transpose((2, 1, 0)) 
            for i_pad in range(400 - np.shape(temp_data)[0]):
                temp_data = np.concatenate((temp_data, np.expand_dims(temp_data[np.shape(temp_data)[0] - 1], axis = 0)), axis = 0)
        else:
            temp_data = measureDict['y']
            for i_pad in range(400 - np.shape(temp_data)[0]):
                temp_data = np.concatenate((temp_data, np.expand_dims(temp_data[np.shape(temp_data)[0] - 1], axis = 0)), axis = 0)
        
        dataset = np.concatenate((dataset, temp_data[:, :, signal_slice[0]:signal_slice[1]]), axis = 0)
        temp_tag = np.zeros(400)
        serial_file = int(list_mat_file[i].split("_")[-1].split(".")[0])

        if serial_file in measurements_with_mass[:, 0]:
            startpoint, endpoint = measurements_with_mass[np.where(measurements_with_mass[:, 0] == serial_file), 1:][0][0]
            temp_tag[startpoint :endpoint] = 1
        
        tag.append(temp_tag)

    measure_day = np.array(measure_day).reshape(-1)
    measure_time = np.array(measure_time).reshape(-1)
    measure_temperature = np.array(measure_temperature).reshape(-1)
    measure_pressure = np.array(measure_pressure).reshape(-1)
    measure_brightness = np.array(measure_brightness).reshape(-1)
    measure_humidity = np.array(measure_humidity).reshape(-1)
    measure_frequence = np.array(measure_frequence).reshape(-1)
    tag = np.array(tag).reshape(-1)
    
    measure_datetime = generate_timestamp(measure_day, measure_time)
     
    return dataset, measure_datetime, measure_temperature, measure_pressure, measure_brightness, measure_humidity, tag, file_selcted



    