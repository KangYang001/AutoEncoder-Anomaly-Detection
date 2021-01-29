#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 01:51:38 2020

@author: yang.kang
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import glob
from tqdm import tqdm
from pathlib import Path
import scipy.io
import os
from scipy import signal
import random 
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')
import configuration



'''-------------------------------------------------------------------------'''
'''--------------------------- hyperparameters -----------------------------'''
'''-------------------------------------------------------------------------'''

args = configuration.args
sensor = 0
type_label_weather = 2
if_load_dataset = False
if_reorganiza_dataset = False
if_plot_corrcoef_downsample = False
if_plot_raw_signal = True
if_plot_noise = False
if_plot_samples = False
if_plot_pulse_compression = False
'''-------------------------------------------------------------------------'''
'''--------------------------- create dataset ------------------------------'''
'''-------------------------------------------------------------------------'''

if if_load_dataset:
    
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
        print(f"please create {path_load}")
    
    dataset = plate_ultrasonic_dataset_T['dataset_sonic']
    dataset_env = np.array([plate_ultrasonic_dataset_T['temperature'], 
                            plate_ultrasonic_dataset_T['humidity'],
                            plate_ultrasonic_dataset_T['brightness'],
                            plate_ultrasonic_dataset_T['pressure']])

    
    samples_selected = random.sample(range(2000), 50)
    samples_selected.sort()
    samples_selected = np.array(samples_selected)
    '''
    dataset = dataset[:, samples_selected]
    partial_samples = {"dataset": dataset,
                       "samples_selected":samples_selected}
    
    path_save = Path(args.direction_workspace, "dataset", "ultrasonic_orginial", "sensors_" + str(args.sensors), \
                     str(args.batch_file)).joinpath("ratio_downsample_" + str(args.ratio_downsample))
    path_save = Path(path_save).joinpath("partial_samples_ultrasonics_data")
    with open(path_save, 'wb') as handle:
        pickle.dump(dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n") 
    '''

    '''-------------------------------------------------------------------------'''
    
    load_dir = Path(args.direction_data_weather, 'dataset', 'corrcoeff_PCA', '01234567', 
                    str(25)).joinpath("combined dataset")
    path_load = Path(load_dir).joinpath('datasets_PCA_weather.pickle')
    
    if os.path.isfile(str(path_load)):  
        print(path_load, "start to be loaded\n")
        with open(path_load , 'rb') as file:
            plate_ultrasonic_dataset = pickle.load(file)        
        print(path_load, "has been created\n") 
    else:
        print(f"please create {path_load}")
       
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
    null_index = plate_ultrasonic_dataset['null_index']
                 
    brightness_log = np.log(brightness)
    corrcoef = corrcoef_T[sensor]
    
    '''-------------------------------------------------------------------------'''
    
    label_weather = np.zeros(len(corrcoef)-len(null_index))
    direction_label_weather = Path(args.direction_label_weather, 'dataset', 'label_prediction').joinpath(str(len(label_weather)))
    path_load = Path(direction_label_weather).joinpath("label_weather.pickle")
    with open(path_load , 'rb') as file:
        label_weather_dict = pickle.load(file)
        print(path_load, "has been created")
    
    index_radiation_predict = label_weather_dict["index_radiation_predict"]
    index_icing_predict = label_weather_dict["index_icing_predict"]
    index_hydrops_predict = label_weather_dict["index_hydrops_predict"]
    
    index_precipitation_all = label_weather_dict["index_precipitation_all"]
    label_weather1 = np.zeros(len(label_weather))
    label_weather1[index_precipitation_all] = 1 
    if type_label_weather == 1:       
        label_weather[index_hydrops_predict] = 4
        label_weather[index_icing_predict] = 3
        label_weather[index_radiation_predict] = 2
        label_weather[index_precipitation_all] = 1  
    else:        
        index_pcpn_predict = label_weather_dict["index_pcpn_predict"]
        label_weather[index_pcpn_predict] = 1 
        label_weather[index_hydrops_predict] = 4
        label_weather[index_icing_predict] = 3
        label_weather[index_radiation_predict] = 2
    
if if_reorganiza_dataset:
    '''
    index_10 =  np.where((null_index % args.ratio_downsample) == 0)
    index_null_data = np.array(list(map(int, null_index[index_10]/args.ratio_downsample)))    
    dataset_10 = np.delete(dataset, index_null_data, axis = 0)
    data_env_10 = np.array([temperature[::10], humidity[::10], brightness[::10]])
    '''
    index_null_env = np.array(list(set(np.argwhere(np.isnan(dataset_env))[:,1])))
    dataset = np.delete(dataset, index_null_env, axis = 0)   
    
    corrcoef = corrcoef[::10]
    datatime = datatime[::10] 
    label_weather = label_weather[::10]
    label_weather1 = label_weather1[::10]
    
    corrcoef = np.delete(corrcoef, index_null_env, axis = 0)   
    datatime = np.delete(datatime, index_null_env, axis = 0)   
        
    index_radiation_predict = np.where(label_weather == 2)[0]
    index_icing_predict = np.where(label_weather == 3)[0]
    index_hydrops_predict = np.where(label_weather == 4)[0]
    index_pcpn_predict = np.where(label_weather == 1)[0]
    index_precipitation_all = np.where(label_weather1 == 1)[0]


    save_dir = Path(args.direction_workspace, 'result', 'raw_signals', 'sensor_0', str(args.ratio_downsample))
    index_predicted_fair = np.where(label_weather == 0)[0]
    i = 10000
    x = np.arange(1, len(dataset[i])+1)/1000
    plt.figure(figsize = (20, 5))
    plt.plot(x, dataset[index_predicted_fair[i]], \
             label = str(datatime[index_predicted_fair[i]]))          
    plt.xlim((0, 2))
    plt.ylim((-0.01, 0.01))
    plt.xticks(fontsize = 20, rotation = 0)
    plt.yticks(fontsize = 20, rotation = 0)
    plt.xlabel("time (ms)", fontsize = 20)
    plt.ylabel("magnitude", fontsize = 20)
    plt.legend(loc = 'upper right', scatterpoints = 10, fontsize = 20)      
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.15, wspace = 0.2, hspace = 0.3)
    plt.savefig(save_dir.joinpath("received guided wave").with_suffix('.png'))
    
'''-------------------------------------------------------------------------'''
if if_plot_pulse_compression:

    direction_data_local =  args.direction_matdata + args.folder_matfile[0][0]
    list_mat_file = glob.glob(direction_data_local + '/*.mat')
    list_mat_file.sort()    
    measureDict = scipy.io.loadmat(list_mat_file[1000])
    sig_trans = measureDict['s']
    plt.figure()
    for i in range(200, 400):
        plt.plot(sig_trans[:1010, 0, i])
        plt.title(str(i))
        plt.show()
        plt.pause(0.2)
        plt.clf()
        
    dict_weathers = ["fair", "precipitation", "sunlight", "freezing", "wetting"]
    index_predicted_fair = np.where(label_weather == 0)[0]
    index_predicted_weather = np.array([index_predicted_fair, index_pcpn_predict, index_radiation_predict, \
                                        index_icing_predict, index_hydrops_predict])
    n1 = 2
    i = np.where((corrcoef[index_predicted_weather[n1]] < 0.9) & 
                 (corrcoef[index_predicted_weather[n1]] > 0.8))[0][500] 

    corr1_signals = signal.correlate(dataset[index_predicted_fair[i]], sig_trans[:1000, 0, 0], mode='same') / 1000
    corr2_signals = signal.correlate(dataset[index_predicted_weather[n1][i]], sig_trans[:1000, 0, 0], mode='same') / 1000
    
    save_dir = Path(args.direction_workspace, 'result', 'matched_filter', 'sensor_0', str(args.ratio_downsample))  
    save_dir.mkdir(parents = True, exist_ok = True) 
    x = np.arange(1, 2001) * 1000 / 1e6     
    plt.figure(figsize = (20, 10))
    plt.subplot(211)
    plt.plot(x, dataset[index_predicted_fair[i]], \
             label = "predicted " + dict_weathers[0] + " " + str(datatime[index_predicted_fair[i]]) + \
             " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_fair[i]]))          
    plt.plot(x, dataset[index_predicted_weather[n1][i]], 
             label = "predicted " + dict_weathers[n1] + " " + str(datatime[index_predicted_weather[n1][i]]) + \
             " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_weather[n1][i]]),
             color = 'yellow') 
    plt.xlim((0, 2))
    plt.ylim((-0.01, 0.01))
    plt.xticks(fontsize = 20, rotation = 0)
    plt.yticks(fontsize = 20, rotation = 0)
    plt.xlabel("times(ms)", fontsize = 20)
    plt.ylabel("magnitude", fontsize = 20)
    plt.legend(loc = 'upper right', scatterpoints = 10, fontsize = 20)
    #plt.title("guided waves from fair VS guide waves from " + dict_weathers[n1] + " weather", fontsize = 24)
    plt.subplot(212)
    plt.plot(x, corr1_signals, \
             label = "predicted " + dict_weathers[0] + " " + str(datatime[index_predicted_fair[i]]) + \
             " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_fair[i]]))          
    plt.plot(x, corr2_signals, 
             label = "predicted " + dict_weathers[n1] + " " + str(datatime[index_predicted_weather[n1][i]]) + \
             " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_weather[n1][i]]),
             color = 'yellow') 
    plt.xlim((0, 2))
    #plt.ylim((-0.01, 0.01))
    plt.xticks(fontsize = 20, rotation = 0)
    plt.yticks(fontsize = 20, rotation = 0)
    plt.xlabel("times(ms)", fontsize = 20)
    plt.ylabel("magnitude", fontsize = 20)
    plt.legend(loc = 'upper right', scatterpoints = 10, fontsize = 20)
    #plt.title("pulse compression of guided waves from fair VS guide waves from " + dict_weathers[n1] + " weather", fontsize = 24)        
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
    plt.savefig(save_dir.joinpath("pulse compression of guided waves from fair VS guide waves from " + dict_weathers[n1] + " weather").with_suffix('.png'))
    
    

if if_plot_samples:
    
    dataset  = np.delete(dataset, null_index, axis = 0 )

    n1 = 10
    n2 = 20
    save_dir = Path(args.direction_workspace, 'result', 'samples_scatter', 'sensor_0', \
                    str(args.ratio_downsample), str(samples_selected[n1])  + ' Vs ' + str(samples_selected[n2]))    
    save_dir.mkdir(parents = True, exist_ok = True) 
    len_slice =  10000
    xlim = [np.min(dataset[:, n1]), np.max(dataset[:, n1])]
    ylim = [np.min(dataset[:, n2]), np.max(dataset[:, n2])]
    for startpoint in range(0, len(dataset), len_slice):
        middlepoint = int(startpoint + len_slice/2)
        endpoint = startpoint + len_slice
        
        temp_index_pcpn_p = np.where((index_pcpn_predict < endpoint) & (index_pcpn_predict > startpoint))[0]
        temp_index_radiation_p = np.where((index_radiation_predict < endpoint) & (index_radiation_predict > startpoint))[0]
        temp_index_icing_p = np.where((index_icing_predict < endpoint) & (index_icing_predict > startpoint))[0]
        temp_index_hydrops_p = np.where((index_hydrops_predict < endpoint) & (index_hydrops_predict > startpoint))[0]
        temp_index_precipitation = np.where((index_precipitation_all < endpoint) & (index_precipitation_all > startpoint))[0]
        
        plt.figure(figsize = (20, 10))
        plt.scatter(dataset[startpoint:middlepoint][:, n1], dataset[startpoint:middlepoint][:, n2],  
                    alpha = 0.9, s = 10, label = "predicted fair1")
        plt.scatter(dataset[middlepoint:endpoint][:, n1], dataset[middlepoint:endpoint][:, n2],  
                    alpha = 0.9, s = 10, label = "predicted fair2", color = 'blue')
        if temp_index_icing_p.size != 0:
            plt.scatter(dataset[index_icing_predict][temp_index_icing_p][:, n1], 
                        dataset[index_icing_predict][temp_index_icing_p][:, n2],  
                        label = "predicted freezing", alpha = 0.9, s = 10, color = 'gray')
        if temp_index_hydrops_p.size != 0:
            plt.scatter(dataset[index_hydrops_predict][temp_index_hydrops_p][:, n1], 
                        dataset[index_hydrops_predict][temp_index_hydrops_p][:, n2],  
                        label = "predicted hydrops", alpha = 0.9, s = 10, color = 'green')
        if temp_index_pcpn_p.size != 0:
            plt.scatter(dataset[index_pcpn_predict][temp_index_pcpn_p][:, n1], 
                        dataset[index_pcpn_predict][temp_index_pcpn_p][:, n2],  
                        label = "predicted precipitation", alpha = 0.9, s = 10, color = 'yellow')
        if temp_index_radiation_p.size != 0:
            plt.scatter(dataset[index_radiation_predict][temp_index_radiation_p][:, n1], 
                        dataset[index_radiation_predict][temp_index_radiation_p][:, n2],  
                        label = "predicted radiation", alpha = 0.9, s = 10, color = 'red')
        if temp_index_precipitation.size != 0:
            plt.scatter(dataset[index_precipitation_all][temp_index_precipitation][:, n1], 
                        dataset[index_precipitation_all][temp_index_precipitation][:, n2],  
                        label = "labelled precipitation", alpha = 0.9, s = 10, color = 'orange')
        plt.xticks(fontsize = 15, rotation = 0)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.yticks(fontsize = 15, rotation = 0)
        plt.xlabel(str(samples_selected[n1]), fontsize = 15,)
        plt.ylabel(str(samples_selected[n2]), fontsize = 15,)
    
        plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
        plt.legend(loc = 'lower right', scatterpoints = 10, fontsize = 16)
        plt.suptitle('the scatter between ' + str(samples_selected[n1])  + ' and ' + str(samples_selected[n2]) + ' sample ' + str(datatime[endpoint]), \
                     fontsize = 24, x = 0.5, y = 0.99)
        plt.savefig(save_dir.joinpath('file_batch_' + str(args.batch_file) + '_' + str(datatime[endpoint])).with_suffix('.png'))      
        plt.close()                

    
    











if if_plot_raw_signal:

    save_dir = Path(args.direction_workspace, 'result', 'raw_signals', 'sensor_0', str(args.ratio_downsample))
    
    dict_weathers = ["fair", "precipitation", "radiation", "freezing", "hydrops"]
    index_predicted_fair = np.where(label_weather == 0)[0]
    index_predicted_weather = np.array([index_predicted_fair, index_pcpn_predict, index_radiation_predict, \
                                        index_icing_predict, index_hydrops_predict])
    plt.figure(figsize = (15, 8))
    for i in range(0, len(index_pcpn_predict), 100):
        
        n1 = 1
        n2 = 4
        i = np.where((corrcoef[index_predicted_weather[n1]] < 0.9) & 
                     (corrcoef[index_predicted_weather[n1]] > 0.8))[0][100] 
        
        i1 = np.where((corrcoef[index_predicted_weather[n2]] < 0.8) & 
                      (corrcoef[index_predicted_weather[n2]] > 0.7))[0][100]
        plt.figure(figsize = (20, 10))
        plt.subplot(211)
        plt.plot(dataset[index_predicted_fair[i]], \
                 label = "predicted " + dict_weathers[0] + " " + str(datatime[index_predicted_fair[i]]) + \
                 " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_fair[i]]))          
        plt.plot(dataset[index_predicted_weather[n1][i]], 
                 label = "predicted " + dict_weathers[n1] + " " + str(datatime[index_predicted_weather[n1][i]]) + \
                 " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_weather[n1][i]]),
                 color = 'yellow') 
        plt.xlim((0, 2000))
        plt.ylim((-0.01, 0.01))
        plt.xticks(fontsize = 15, rotation = 0)
        plt.yticks(fontsize = 15, rotation = 0)
        plt.xlabel("samples", fontsize = 15)
        plt.ylabel("magnitude", fontsize = 15)
        plt.legend(loc = 'upper right', scatterpoints = 10, fontsize = 16)
        plt.title("guided waves from fair VS guide waves from " + dict_weathers[n1] + " weather", fontsize = 24)
        plt.subplot(212)
        plt.plot(dataset[index_predicted_fair[i]], \
                 label = "predicted " + dict_weathers[0] + " " + str(datatime[index_predicted_fair[i]]) + \
                 " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_fair[i]]))          
        plt.plot(dataset[index_predicted_weather[n2][i1]], 
                 label = "predicted " + dict_weathers[n2] + " " + str(datatime[index_predicted_weather[n2][i1]]) + \
                 " reconstruction coefficient: " + "{:.4f}".format(corrcoef[index_predicted_weather[n2][i1]]),
                 color = 'yellow') 
        plt.xlim((0, 2000))
        plt.ylim((-0.01, 0.01))
        plt.xticks(fontsize = 15, rotation = 0)
        plt.yticks(fontsize = 15, rotation = 0)
        plt.xlabel("samples", fontsize = 15)
        plt.ylabel("magnitude", fontsize = 15)
        plt.legend(loc = 'upper right', scatterpoints = 10, fontsize = 16)
        plt.title("guided waves from fair VS guide waves from " + dict_weathers[n2] + " weather", fontsize = 24)        
        plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
        #plt.show()
        #plt.pause(0.3)
        #plt.clf()
        plt.savefig(save_dir.joinpath("guided waves from fair VS guide waves from " + dict_weathers[n1] + " weather").with_suffix('.png'))
        

if if_plot_noise:

    keywords_noise = ['strucutred noise', 'unstrucutred noise']
    noise_gaussian = []
    mean_gaussian = np.array([0, 0])
    cov_gaussian = np.array([[10, 8], [8, 10]])    
    noise_gaussian.append(np.random.multivariate_normal(mean_gaussian, cov_gaussian, 1000))
    cov_gaussian = np.array([[10, 3], [3, 10]])
    noise_gaussian.append(np.random.multivariate_normal(mean_gaussian, cov_gaussian, 1000))
    
    save_dir = Path(args.direction_workspace, 'result', 'raw_signals', 'sensor_0', str(args.ratio_downsample))
    plt.figure(figsize = (20, 10))
    for i in range(2):
        plt.subplot(1,2, i+1)
        #plt.scatter(noise_gaussian[i][:,0], noise_gaussian[i][:,1], s = 5, label = keywords_noise[i])
        plt.scatter(noise_gaussian[i][:,0], noise_gaussian[i][:,1], s = 5)
        plt.xlim((-10, 10))
        plt.ylim((-10, 10))
        plt.xticks(fontsize = 24, rotation = 0)
        plt.yticks(fontsize = 24, rotation = 0)
        plt.xlabel("feature 1", fontsize = 24)
        plt.ylabel("feature 2", fontsize = 24)  
        #plt.title(keywords_noise[i], fontsize = 24)          
        plt.subplots_adjust(left = 0.08, right = 0.95, top = 0.95, bottom = 0.08, wspace = 0.2, hspace = 0.3)
        #plt.legend(loc = 'lower right', scatterpoints = 10, fontsize = 16)
    #plt.suptitle('strucutred noise VS unstrucutred noise', fontsize = 24, x = 0.5, y = 0.99)
    plt.savefig(save_dir.joinpath('strucutred noise VS unstrucutred noise').with_suffix('.png'))      
         

if if_plot_corrcoef_downsample:
    
    measurement_keywords = ['corrcoef', 'temperature', 'humidity', 'brightness']
    save_dir = Path(args.direction_workspace, 'result', 'raw_signals', 'sensor_0', str(args.ratio_downsample))
    save_dir.mkdir(parents = True, exist_ok = True)
    time_slice = int(400000/args.ratio_downsample)
    for startpoint in range(0, len(corrcoef), time_slice):
        if (startpoint + time_slice) < len(corrcoef):
            endpoint = startpoint + time_slice
        else:
            endpoint = len(corrcoef)
        print(endpoint)
              
        index_radiation_slice_p = np.where((datatime[index_radiation_predict] > datatime[startpoint]) & 
                                           (datatime[index_radiation_predict] < datatime[endpoint-1]))[0]             
        index_icing_slice_p = np.where((datatime[index_icing_predict] > datatime[startpoint]) & 
                                       (datatime[index_icing_predict] < datatime[endpoint-1]))[0]             
        index_hydrops_slice_p = np.where((datatime[index_hydrops_predict] > datatime[startpoint]) & 
                                         (datatime[index_hydrops_predict] < datatime[endpoint-1]))[0]                              
        index_precipitation_slice = np.where((datatime[index_precipitation_all] >  datatime[startpoint]) & 
                                             (datatime[index_precipitation_all] <  datatime[endpoint-1]))[0]          
        index_pcpn_slice_p = np.where((datatime[index_pcpn_predict] > datatime[startpoint]) & 
                                      (datatime[index_pcpn_predict] < datatime[endpoint-1]))[0]   
    
        plt.figure(figsize = (20, 10))
        for i in range(4):
            if i == 0:
                measurement_v = corrcoef
            else:
                measurement_v = dataset_env[i-1]
            plt.subplot(4, 1, i+1)
            plt.scatter(datatime[startpoint:endpoint], measurement_v[startpoint:endpoint], \
                        alpha = 1, s = 15, label = "predicted fair")        
            plt.scatter(datatime[index_hydrops_predict][index_hydrops_slice_p], 
                        measurement_v[index_hydrops_predict][index_hydrops_slice_p], 
                        label = "predicted hydrops", marker = ',', alpha = 0.9, s = 15, color = 'green')               
            plt.scatter(datatime[index_radiation_predict][index_radiation_slice_p], 
                        measurement_v[index_radiation_predict][index_radiation_slice_p],
                        label = "predicted radiation", marker = '>', alpha = 0.9, s = 15, color = 'red')
            plt.scatter(datatime[index_icing_predict][index_icing_slice_p], 
                        measurement_v[index_icing_predict][index_icing_slice_p], 
                        label = "predicted freezing", marker = ',', alpha = 0.9, s = 15, color = 'grey')                     
            plt.scatter(datatime[index_pcpn_predict][index_pcpn_slice_p], 
                        measurement_v[index_pcpn_predict][index_pcpn_slice_p], 
                        label = "predicted precipitation", alpha = 0.9, s = 15, color = 'yellow')  
            plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], 
                        measurement_v[index_precipitation_all][index_precipitation_slice], \
                        alpha = 1, s = 10, label = "labelled precipitation", color = 'orange')
            
            plt.xlim((datatime[startpoint], datatime[endpoint-1]))
            plt.xticks(fontsize = 15, rotation = 0)
            plt.yticks(fontsize = 15, rotation = 0)
            plt.ylabel(measurement_keywords[i], fontsize = 15)
            #plt.title(measurement_keywords[i].split('_')[1], fontsize = 28)     
            if i == 1:
                if np.mean(measurement_v[startpoint:endpoint]) < 10:
                    plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
        plt.xlabel("measurement time", fontsize = 15)
        plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
        plt.legend(loc = 'lower right', scatterpoints = 10, fontsize = 16)
        plt.suptitle('the change pattern among the reconstruction coefficient, temeprature, humidity and brightness', \
                     fontsize = 24, x = 0.5, y = 0.99)
        # plt.show()
        # plt.tight_layout()
        # plt.text(startPoint-500+10, target.min(), 'Epoch: '+str(epoch),fontsize=15) 
        plt.savefig(save_dir.joinpath('file_batch_' + str(args.batch_file) + '_' + str(datatime[startpoint]) 
        + '_' +'prediction').with_suffix('.png'))      
        plt.close()            





