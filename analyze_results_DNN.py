
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:39:32 2020

@author: yang.kang
"""
import numpy as np
import pickle
import os, os.path
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')
import configuration


'''-------------------------------------------------------------------------'''
'''------------------------------ function ---------------------------------'''
'''-------------------------------------------------------------------------'''

def combine_dataset_batch(args):
    
    if args.scatter_factor > 0:
        load_dir = Path(args.direction_workspace, 'dataset', args.model_function + "_predict", args.sensors + " sensors", str(args.batch_file), \
                        "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str, str(args.scatter_factor), str(args.samples_signal))
    else:
        load_dir = Path(args.direction_workspace, 'dataset', args.model_function + "_predict", args.sensors + " sensors", str(args.batch_file), \
                        "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str, str(args.samples_signal))

    list_pickle_file = glob.glob(str(load_dir) + '/*.pickle')
    list_pickle_file.sort()      

    if args.model_function == "detect_mass":
        y_predict = np.empty((0))
    else:
        type_input = list(eval(args.type_input))
        y_predict = np.empty((0, len(type_input)))
        scatter_factor = np.empty((0))
        
    file_selected = np.empty((0)) 
    for n in tqdm(range(len(list_pickle_file))):  
        with open(list_pickle_file[n] , 'rb') as file:
            y_predict_data = pickle.load(file)
        print(list_pickle_file[n])
        
        y_predict = np.concatenate((y_predict, y_predict_data['y_predict']), axis = 0)
        file_selected = np.concatenate((file_selected, y_predict_data['file_selcted']), axis = 0)
        if args.scatter_factor > 0:
            scatter_factor = np.concatenate((scatter_factor, y_predict_data['scatter_factor']), axis = 0)

    if args.scatter_factor > 0:
        y_predict_dataset = {"file_selcted": file_selected, \
                             "y_predict": y_predict, \
                             'scatter_factor': scatter_factor}          
    else:
        y_predict_dataset = {"file_selcted": file_selected, \
                             "y_predict": y_predict}          
    
    number_file = list_pickle_file[-1].split("_")[-1].split(".")[0]
    filename_save = 'y_predict_' + number_file + '.pickle'
    save_dir = load_dir.joinpath("combined dataset")
    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = Path(save_dir).joinpath(filename_save)
    
    with open(path_save, 'wb') as handle:
        pickle.dump(y_predict_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n")        
    
    return y_predict_dataset

'''-------------------------------------------------------------------------'''
'''-------------------------- hyperparameters ------------------------------'''
'''-------------------------------------------------------------------------'''

args = configuration.args
analyze_y_predict = True
if_plot_y_predict = True
sensor = 0
measurement_keywords = [ "temperature", "humidity", "brightness1", "pressure"]
type_input = np.array(eval(args.type_input))
'''-------------------------------------------------------------------------'''
'''--------------------------- main function -------------------------------'''
'''-------------------------------------------------------------------------'''

filename_save = 'y_predict_17560.pickle'
if args.scatter_factor > 0:
    load_dir =  Path(args.direction_workspace, 'dataset', args.model_function + "_predict", args.sensors + " sensors", \
                     str(args.batch_file),  "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str, \
                     str(args.scatter_factor), str(args.samples_signal)).joinpath("combined dataset")
else:
    load_dir =  Path(args.direction_workspace, 'dataset', args.model_function + "_predict", args.sensors + " sensors", \
                     str(args.batch_file), "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str, \
                     str(args.samples_signal)).joinpath("combined dataset")
        
path_load = Path(load_dir).joinpath(filename_save)
    
if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as handle:
        y_predict_dataset = pickle.load(handle)         
    print(path_load, "has been created\n") 
else:
    y_predict_dataset = combine_dataset_batch(args)

if analyze_y_predict:
    
    y_predict = y_predict_dataset['y_predict']
    if args.scatter_factor > 0:
        scatter_factor = y_predict_dataset['scatter_factor']
    filename_corr_weather = 'datasets_PCA_weather.pickle'
    load_dir = Path(args.direction_data_weather, 'dataset', 'corrcoeff_PCA', '01234567', str(25)).joinpath("combined dataset")
    path_load = Path(load_dir).joinpath(filename_corr_weather)
    with open(path_load , 'rb') as file:
        plate_ultrasonic_dataset = pickle.load(file)
       
    corrcoef_T = plate_ultrasonic_dataset["corrcoef_T"]
    datatime = plate_ultrasonic_dataset["datatime"]
    temperature = plate_ultrasonic_dataset["temperature"]
    humidity = plate_ultrasonic_dataset["humidity"]
    brightness = plate_ultrasonic_dataset["brightness"]
    pressure = plate_ultrasonic_dataset["pressure"]
    tag = plate_ultrasonic_dataset["tag"]
    index_precipitation_all = plate_ultrasonic_dataset["index_precipitation_all"]
    index_little_rain = plate_ultrasonic_dataset["index_little_rain"]
    index_rain = plate_ultrasonic_dataset["index_rain"] 
    index_snow = plate_ultrasonic_dataset["index_snow"]
    index_mix = plate_ultrasonic_dataset["index_mix"]
    null_index = plate_ultrasonic_dataset["null_index"]
                 
    brightness1 = np.log(brightness)
    corrcoef = corrcoef_T[sensor]
    
    temperature[np.where(temperature < -50)] = temperature[np.where(temperature < -50)[0]+1]
    #y_predict = np.delete(y_predict, null_index, axis = 0)    

    # print(f"confusion matrix for detecting mass: \n {confusion_matrix(tag, y_predict)}\n") 

if if_plot_y_predict:   
    
    if args.model_function == "detect_env":
        if args.scatter_factor > 0:
            save_dir = Path(args.direction_workspace, 'result', args.model_function + "_predict", args.sensors + " sensors", \
                            str(args.batch_file), "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str, str(args.scatter_factor),\
                            ).joinpath(str(args.samples_signal))            
        else:
            save_dir = Path(args.direction_workspace, 'result', args.model_function + "_predict", args.sensors + " sensors", \
                            str(args.batch_file), "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str\
                            ).joinpath(str(args.samples_signal))
        
        save_dir.mkdir(parents = True, exist_ok = True) 
        time_slice = 400000    
        for startpoint in range(0, len(corrcoef), time_slice):
            if (startpoint + time_slice) < len(corrcoef):
                endpoint = startpoint + time_slice
            else:
                endpoint = len(corrcoef)
    
            index_precipitation_slice = np.where((datatime[index_precipitation_all] >  datatime[startpoint]) & 
                                                 (datatime[index_precipitation_all] <  datatime[endpoint-1]))[0]                
            index_little_rain_slice = np.where((datatime[index_little_rain] > datatime[startpoint]) & 
                                               (datatime[index_little_rain] < datatime[endpoint-1]))[0]             
            index_rain_slice = np.where((datatime[index_rain] > datatime[startpoint]) & 
                                        (datatime[index_rain] < datatime[endpoint-1]))[0]             
            index_snow_slice = np.where((datatime[index_snow] > datatime[startpoint]) & 
                                        (datatime[index_snow] < datatime[endpoint-1]))[0]                                 
            index_mix_slice = np.where((datatime[index_mix] > datatime[startpoint]) & 
                                       (datatime[index_mix] < datatime[endpoint-1]))[0]   
                    
            plt.figure(figsize = (20, 10))
            if args.scatter_factor > 0:
                num_subplots = len(type_input)+2
            else:
                num_subplots = len(type_input)+1
            for i in range(num_subplots):
                plt.subplot(num_subplots, 1, i+1)
                if i == 0:
                    measurement_v = corrcoef
                    plt.ylabel("corrcoeff", fontsize = 15)
                elif (args.scatter_factor > 0) & (i == num_subplots - 1):
                    measurement_v = scatter_factor
                    plt.ylabel("scatter_factor", fontsize = 15)                    
                else:                        
                    measurement_v = eval(measurement_keywords[type_input[i-1]])
                                 
                plt.scatter(datatime[startpoint:endpoint], measurement_v[startpoint:endpoint], \
                            alpha = 1, s = 15, label = "unlabelled situation")        
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], 
                            measurement_v[index_precipitation_all][index_precipitation_slice], \
                            alpha = 1, s = 10, label = "labelled precipitation", color = 'orange')            
                plt.scatter(datatime[index_little_rain][index_little_rain_slice], 
                            measurement_v[index_little_rain][index_little_rain_slice], 
                            label = "little rain", marker = ',', alpha = 0.9, s = 15, color = 'green')               
                plt.scatter(datatime[index_rain][index_rain_slice], 
                            measurement_v[index_rain][index_rain_slice],
                            label = "rain", marker = '>', alpha = 0.9, s = 15, color = 'red')
                plt.scatter(datatime[index_snow][index_snow_slice], 
                            measurement_v[index_snow][index_snow_slice], 
                            label = "snow", marker = ',', alpha = 0.9, s = 15, color = 'grey')                        
                plt.scatter(datatime[index_mix][index_mix_slice], 
                            measurement_v[index_mix][index_mix_slice], 
                            label = "mix", alpha = 0.9, s = 15, color = 'yellow')  
                if (i > 0) & (i < num_subplots - 1):
                    plt.scatter(datatime[startpoint:endpoint], y_predict[startpoint:endpoint, i-1],
                                label = "predicted environment", alpha = 0.9, s = 5, color = 'black')   
                    plt.ylabel(measurement_keywords[i-1], fontsize = 15)
                plt.xlim((datatime[startpoint], datatime[endpoint-1]))
                plt.xticks(fontsize = 15, rotation = 0)
                plt.yticks(fontsize = 15, rotation = 0)

                #plt.title(measurement_keywords[i].split('_')[1], fontsize = 28)     
                if i == 1:
                    if np.mean(measurement_v[startpoint:endpoint]) < 10:
                        plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
            plt.xlabel("measurement time", fontsize = 15)
            plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
            plt.legend(loc = 'lower right', scatterpoints = 10, fontsize = 16)
            plt.suptitle('the predicted envrionment information', \
                         fontsize = 24, x = 0.5, y = 0.99)
            plt.savefig(save_dir.joinpath('ratio_downsample_' + str(args.ratio_downsample) + '_' + str(datatime[startpoint]) 
            + '_' + 'envrionment').with_suffix('.png'))
            plt.close()
   
    if args.model_function == "detect_mass":
        save_dir = Path(args.direction_workspace, 'result', args.model_function + "_predict", args.sensors + " sensors", str(args.batch_file), \
                        "ratio_downsample_" + str(args.ratio_downsample)).joinpath(str(args.samples_signal))
        save_dir.mkdir(parents = True, exist_ok = True)    
        measurement_keywords = ["tag", "temperature", "humidity", "brightness1"]
        time_slice = 400000    
        for startpoint in range(0, len(corrcoef), time_slice):
            if (startpoint + time_slice) < len(corrcoef):
                endpoint = startpoint + time_slice
            else:
                endpoint = len(corrcoef)
    
            index_precipitation_slice = np.where((datatime[index_precipitation_all] >  datatime[startpoint]) & 
                                                 (datatime[index_precipitation_all] <  datatime[endpoint-1]))[0]                
            index_little_rain_slice = np.where((datatime[index_little_rain] > datatime[startpoint]) & 
                                               (datatime[index_little_rain] < datatime[endpoint-1]))[0]             
            index_rain_slice = np.where((datatime[index_rain] > datatime[startpoint]) & 
                                        (datatime[index_rain] < datatime[endpoint-1]))[0]             
            index_snow_slice = np.where((datatime[index_snow] > datatime[startpoint]) & 
                                        (datatime[index_snow] < datatime[endpoint-1]))[0]                                 
            index_mix_slice = np.where((datatime[index_mix] > datatime[startpoint]) & 
                                       (datatime[index_mix] < datatime[endpoint-1]))[0]           
                    
            plt.figure(figsize = (20, 10))         
            for i in range(len(measurement_keywords)):       
                measurement_v = eval(measurement_keywords[i])
                plt.subplot(4, 1, i+1)
                plt.scatter(datatime[startpoint:endpoint], measurement_v[startpoint:endpoint], \
                            alpha = 1, s = 15, label = "unlabelled situation")        
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], 
                            measurement_v[index_precipitation_all][index_precipitation_slice], \
                            alpha = 1, s = 10, label = "labelled precipitation", color = 'orange')            
                plt.scatter(datatime[index_little_rain][index_little_rain_slice], 
                            measurement_v[index_little_rain][index_little_rain_slice], 
                            label = "little rain", marker = ',', alpha = 0.9, s = 15, color = 'green')               
                plt.scatter(datatime[index_rain][index_rain_slice], 
                            measurement_v[index_rain][index_rain_slice],
                            label = "rain", marker = '>', alpha = 0.9, s = 15, color = 'red')
                plt.scatter(datatime[index_snow][index_snow_slice], 
                            measurement_v[index_snow][index_snow_slice], 
                            label = "snow", marker = ',', alpha = 0.9, s = 15, color = 'grey')                        
                plt.scatter(datatime[index_mix][index_mix_slice], 
                            measurement_v[index_mix][index_mix_slice], 
                            label = "mix", alpha = 0.9, s = 15, color = 'yellow')                
                plt.xlim((datatime[startpoint], datatime[endpoint-1]))
                plt.xticks(fontsize = 15, rotation = 0)
                plt.yticks(fontsize = 15, rotation = 0)
                plt.ylabel(measurement_keywords[i], fontsize = 15)
                #plt.title(measurement_keywords[i].split('_')[1], fontsize = 28)     
                if i == 1:
                    if np.mean(measurement_v[startpoint:endpoint]) < 10:
                        plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
                if i == 0:
                    plt.scatter(datatime[startpoint:endpoint], y_predict[startpoint:endpoint], \
                                alpha = 1, s = 1, label = "predicted mass", color = 'black')                         
            plt.xlabel("measurement time", fontsize = 15)
            plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
            plt.legend(loc = 'lower right', scatterpoints = 10, fontsize = 16)
            plt.suptitle('the predicted mass information', \
                         fontsize = 24, x = 0.5, y = 0.99)
            plt.savefig(save_dir.joinpath('ratio_downsample_' + str(args.ratio_downsample) + '_' + str(datatime[startpoint]) 
            + '_' + 'mass').with_suffix('.png'))
            plt.close()            
            
