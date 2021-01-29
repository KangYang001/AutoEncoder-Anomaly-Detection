#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:39:32 2020

@author: yang.kang
"""
import numpy as np
import pickle
import os, os.path
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/autoencoder/code')
import configuration
import create_dataset_for_autoencoder

'''-------------------------------------------------------------------------'''
'''------------------------------ function ---------------------------------'''
'''-------------------------------------------------------------------------'''


'''-------------------------------------------------------------------------'''
'''-------------------------- hyperparameters ------------------------------'''
'''-------------------------------------------------------------------------'''

args = configuration.args

sensor = 0
#measurement_keywords = ["corrcoef", "corrcoef_autoencoder", "temperature", "humidity"]
measurement_keywords = ["corrcoef_autoencoder", "scatter_factor", "temperature", "humidity"]
len_signal = 2000

'''-------------------------------------------------------------------------'''
'''--------------------------- main function -------------------------------'''
'''-------------------------------------------------------------------------'''

filename_save = 'corrcoef_autoenc_17560.pickle'

load_dir = Path(args.direction_workspace, 'dataset', "autoencoder_corrcoeff", args.sensors + " sensors", 
                str(len_signal), str(args.batch_file), "ratio_downsample_" + str(args.ratio_downsample))
if args.scatter_factor > 0:
    load_dir = load_dir.joinpath(str(args.scatter_factor))      
if args.if_downsample:
    load_dir = load_dir.joinpath(str(args.ratio_downsample)) 
if args.threshold_data_type > 0:
    load_dir = load_dir.joinpath(str(args.threshold_data_type))        
load_dir = load_dir.joinpath("combined dataset")        


path_load = Path(load_dir).joinpath("corr loss 2000 512 128 32 0.7").joinpath(filename_save)
    
if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as handle:
        corrcoef_autencoder_dataset_CORR = pickle.load(handle)         
    print(path_load, "has been created\n") 
else:
    print("please create dataset")

path_load = Path(load_dir).joinpath("MSE 2000 512 128 32 0.7").joinpath(filename_save)
    
if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as handle:
        corrcoef_autencoder_dataset_MSE = pickle.load(handle)         
    print(path_load, "has been created\n") 
else:
    print("please create dataset")

if not args.if_downsample:  
    corrcoef_autoenc_CORR = corrcoef_autencoder_dataset_CORR['corrcoef']
    scatter_factor = corrcoef_autencoder_dataset_CORR["scatter_factor"]
    corrcoef_autoenc_MSE = corrcoef_autencoder_dataset_MSE['corrcoef']
else:
    corrcoef_autoenc = corrcoef_autencoder_dataset_CORR['corrcoef']

'''-------------------------------------------------------------------------'''

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
tag = plate_ultrasonic_dataset["tag"]
index_precipitation_all = plate_ultrasonic_dataset["index_precipitation_all"]
index_little_rain = plate_ultrasonic_dataset["index_little_rain"]
index_rain = plate_ultrasonic_dataset["index_rain"] 
index_snow = plate_ultrasonic_dataset["index_snow"]
index_mix = plate_ultrasonic_dataset["index_mix"]
null_index = plate_ultrasonic_dataset["null_index"]
             
brightness1 = np.log(brightness)
corrcoef = corrcoef_T[sensor]
corr_PCA = corrcoef
#corrcoef_autoencoder = np.delete(corrcoef_autoencoder, null_index, axis = 0)    


index_shuffle = list(range(len(datatime)))
index_shuffle = np.array(index_shuffle)
ratio_train = list(map(float, args.divider_set[0]))
divider_set = create_dataset_for_autoencoder.create_dataset_divider(args, index_shuffle, ratio_train = ratio_train)

'''-------------------------------------------------------------------------'''

    
save_dir = Path(args.direction_workspace, 'result', "autoencoder_corrcoeff", args.sensors + " sensors", \
                str(args.samples_signal), str(args.batch_file), "ratio_downsample_" + str(args.ratio_downsample))

if args.if_downsample:
    save_dir = save_dir.joinpath(str(args.ratio_downsample))
if args.scatter_factor > 0:
    save_dir = save_dir.joinpath(str(args.scatter_factor))
if args.threshold_data_type > 0:
    save_dir = save_dir.joinpath(str(args.threshold_data_type))  

save_dir.mkdir(parents = True, exist_ok = True)      


index_precipitation_all = np.concatenate((index_precipitation_all, index_little_rain, index_rain), axis = 0)
index_precipitation_all = list(set(index_precipitation_all))
index_precipitation_all.sort()
index_precipitation_all = np.array(index_precipitation_all)  

if args.scatter_factor > 0:
    measurement_keywords = ["corrcoef", "scatter_factor", "temperature", "humidity"]
    measurement_labels = ['RC','Scatter' ,'T ['+u'\u2103]', 'H [%]']
else:
    measurement_keywords = ["corrcoef", "temperature", "humidity", "brightness"]
    measurement_labels = ['RC', 'T ['+u'\u2103]', 'H [%]', 'ln(B) [ln(CD)]']


def plotReCoefNNCP(save_dir, fontsize = [20, 20]):
    
    save_dir.mkdir(parents = True, exist_ok = True)           
    time_slice = 300000  
    for startpoint in range(0, len(datatime), time_slice):
        if (startpoint + time_slice) < len(datatime):
            endpoint = startpoint + time_slice
        else:
            endpoint = len(datatime)       
        '''
        index_precipitation_slice = np.where((datatime[index_precipitation_all] >  datatime[startpoint]) & 
                                             (datatime[index_precipitation_all] <  datatime[endpoint-1]))[0]  
        '''                        
        index_xticks = np.percentile(np.arange(startpoint, endpoint), np.arange(0, 99, 24))
        index_xticks = np.array(list(map(int, index_xticks)))        
        plt.figure(figsize = (20, 20))
        for i in range(4):       
            measurement_v = measurement_keywords[i]
            plt.subplot(4, 1, i+1)
            if measurement_keywords[i] == "corrcoef":
                if endpoint < divider_set[2]:               
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_autoenc_CORR[startpoint:endpoint], \
                                alpha = 0.9, s = 15, label = "Reconstruction Coefficient(Train Data) -- CorrLoss")
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_autoenc_MSE[startpoint:endpoint], \
                                alpha = 0.9, s = 15, label = "Reconstruction Coefficient(Train Data) -- MSE")
                elif startpoint > divider_set[2]:
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_autoenc_CORR[startpoint:endpoint], \
                                alpha = 0.9, s = 15, c = 'c', label = "Reconstruction Coefficient(Test Data) -- CorrLoss")
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_autoenc_MSE[startpoint:endpoint], \
                                alpha = 0.9, s = 15, c = 'yellow', label = "Reconstruction Coefficient(Test Data) -- MSE") 
                else:
                    plt.scatter(datatime[startpoint:divider_set[2]], corrcoef_autoenc_CORR[startpoint:divider_set[2]], \
                                alpha = 0.9, s = 15, label = "Reconstruction Coefficient(Train Data) -- CorrLoss")
                    plt.scatter(datatime[startpoint:divider_set[2]], corrcoef_autoenc_MSE[startpoint:divider_set[2]], \
                                alpha = 0.9, s = 15, label = "Reconstruction Coefficient(Train Data) -- MSE")                                                
                    plt.scatter(datatime[divider_set[2]:endpoint], corrcoef_autoenc_CORR[divider_set[2]:endpoint], \
                                alpha = 0.9, s = 15, c = 'c', label = "Reconstruction Coefficient(Test Data) -- CorrLoss")
                    plt.scatter(datatime[divider_set[2]:endpoint], corrcoef_autoenc_MSE[divider_set[2]:endpoint], \
                                alpha = 0.9, s = 15, c = 'yellow', label = "Reconstruction Coefficient(Test Data) -- MSE")     
                        
                '''
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], \
                            corrcoef_autoenc_CORR[index_precipitation_all][index_precipitation_slice], \
                            alpha = 0.9, s = 15, c = 'black', label = "Recorded Precipitation--CorrLoss")                   
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], \
                            corrcoef_autoenc_MSE[index_precipitation_all][index_precipitation_slice], \
                            alpha = 0.9, s = 15, c = 'black', label = "Recorded Precipitation--MSE") 
                '''       
                plt.legend(loc = 'upper center', scatterpoints = 10, fontsize = fontsize[1], bbox_to_anchor=(0.4, -4.26),
                           fancybox = False, shadow = False, ncol = 1, frameon = False)                 
            else:
                plt.scatter(datatime[startpoint:endpoint], \
                            corrcoef_autencoder_dataset_CORR[measurement_v][startpoint:endpoint], \
                            alpha = 0.9, s = 15, label = "Fair")
                '''
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], 
                            corrcoef_autencoder_dataset_CORR[measurement_v][index_precipitation_all][index_precipitation_slice], \
                            alpha = 1, s = 15, c = 'black', label = "Recorded Precipitation")
                '''                                     
            plt.xlim((datatime[startpoint], datatime[endpoint-1]))
            plt.xticks(datatime[index_xticks], fontsize = fontsize[0], rotation = 0)
            plt.yticks(fontsize = fontsize[0], rotation = 0)
            plt.ylabel(measurement_labels[i], fontsize = fontsize[1])  
            if measurement_keywords[i] == "temperature":
                if np.mean(corrcoef_autencoder_dataset_CORR[measurement_v][startpoint:endpoint]) < 10:
                    plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
        plt.xlabel("Measurement Time [YY:MM:DD]", fontsize = fontsize[1])       
        plt.subplots_adjust(left = 0.11, right = 0.95, top = 0.99, bottom = 0.14, wspace = 0.3, hspace = 0.3)
        plt.savefig(save_dir.joinpath('scatter_' + '_ratio_downsample_' + str(args.ratio_downsample) + \
                                      ' ' + str(datatime[startpoint])).with_suffix('.png'), bbox_inches = 'tight')      
        plt.close() 


plotReCoefNNCP(save_dir = save_dir.joinpath("CorrLoss VS MSE 0.7"), fontsize = [40, 40])


'''-------------------------------------------------------------------------'''

measurement_keywords = ["corrcoef", "corr_PCA", "temperature", "humidity"]
measurement_labels = ['RC_NN','RC_PCA' ,'T ['+u'\u2103]', 'H [%]']

def plotReCoefCP(save_dir, fontsize = [20, 20]):  

    save_dir.mkdir(parents = True, exist_ok = True)         
    time_slice = 300000  
    for startpoint in range(0, len(datatime), time_slice):
        if (startpoint + time_slice) < len(datatime):
            endpoint = startpoint + time_slice
        else:
            endpoint = len(datatime)       

        index_precipitation_slice = np.where((datatime[index_precipitation_all] >  datatime[startpoint]) & 
                                             (datatime[index_precipitation_all] <  datatime[endpoint-1]))[0]  
                                
        index_xticks = np.percentile(np.arange(startpoint, endpoint), np.arange(0, 99, 24))
        index_xticks = np.array(list(map(int, index_xticks)))        
        plt.figure(figsize = (20, 20))
        for i in range(4):       
            measurement_v = measurement_keywords[i]
            plt.subplot(4, 1, i+1)
            if measurement_keywords[i] == "corrcoef":
                plt.scatter(datatime[startpoint:endpoint], corrcoef_autoenc_CORR[startpoint:endpoint], \
                            alpha = 0.9, s = 15, label = "Fair")
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], \
                            corrcoef_autoenc_CORR[index_precipitation_all][index_precipitation_slice], \
                            alpha = 0.9, s = 15, label = "Recorded Precipitation")                   
                plt.scatter(datatime[startpoint:endpoint], scatter_factor[startpoint:endpoint], \
                            alpha = 0.9, s = 15, color = 'blue', label = "scatter factor")  
                plt.legend(loc = 'upper center', scatterpoints = 10, fontsize = fontsize[1], bbox_to_anchor=(0.4, -4.26),
                           fancybox = False, shadow = False, ncol = 1, frameon = False)                        
            elif measurement_keywords[i] == "corr_PCA":
                plt.scatter(datatime[startpoint:endpoint], corr_PCA[startpoint:endpoint], \
                            alpha = 0.9, s = 15, label = "Fair--PCA")
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], \
                            corr_PCA[index_precipitation_all][index_precipitation_slice], \
                            alpha = 0.9, s = 15, label = "Recorded Precipitation--PCA")                  
            else:
                plt.scatter(datatime[startpoint:endpoint], \
                            corrcoef_autencoder_dataset_CORR[measurement_v][startpoint:endpoint], \
                            alpha = 0.9, s = 15, label = "Fair")
                plt.scatter(datatime[index_precipitation_all][index_precipitation_slice], 
                            corrcoef_autencoder_dataset_CORR[measurement_v][index_precipitation_all][index_precipitation_slice], \
                            alpha = 1, s = 15, label = "Recorded Precipitation")                                     
            plt.xlim((datatime[startpoint], datatime[endpoint-1]))
            plt.xticks(datatime[index_xticks], fontsize = fontsize[0], rotation = 0)
            plt.yticks(fontsize = fontsize[0], rotation = 0)
            plt.ylabel(measurement_labels[i], fontsize = fontsize[1])  
            if measurement_keywords[i] == "temperature":
                if np.mean(corrcoef_autencoder_dataset_CORR[measurement_v][startpoint:endpoint]) < 10:
                    plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
        plt.xlabel("Measurement Time [YY:MM:DD]", fontsize = fontsize[1])
    
        plt.subplots_adjust(left = 0.11, right = 0.95, top = 0.99, bottom = 0.14, wspace = 0.3, hspace = 0.3)
        plt.savefig(save_dir.joinpath('scatter_' + '_ratio_downsample_' + str(args.ratio_downsample) + \
                                      ' ' + str(datatime[startpoint])).with_suffix('.png'), bbox_inches = 'tight')      
        plt.close() 


plotReCoefCP(save_dir = save_dir.joinpath("CorrLoss VS PCA"), fontsize = [40, 40])


'''-------------------------------------------------------------------------'''


path_load = Path(load_dir).joinpath("corr loss 2000 512 128 32").joinpath(filename_save)
    
if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as handle:
        corrcoef_autencoder_dataset_CORR = pickle.load(handle)         
    print(path_load, "has been created\n") 
else:
    print("please create dataset")

path_load = Path(load_dir).joinpath("MSE 2000 512 128 32").joinpath(filename_save)
    
if os.path.isfile(str(path_load)):  
    print(path_load, "start to be loaded\n")
    with open(path_load , 'rb') as handle:
        corrcoef_autencoder_dataset_MSE = pickle.load(handle)         
    print(path_load, "has been created\n") 
else:
    print("please create dataset")


if not args.if_downsample:  
    corrcoef_autoenc_CORR = corrcoef_autencoder_dataset_CORR['corrcoef']
    scatter_factor = corrcoef_autencoder_dataset_CORR["scatter_factor"]
    corrcoef_autoenc_MSE = corrcoef_autencoder_dataset_MSE['corrcoef']
else:
    corrcoef_autoenc = corrcoef_autencoder_dataset_CORR['corrcoef']


def plotHistReconCoeff(corrcoef_autoenc_diff, ratio_RR_high, save_dir, fontsize = 40):
    
    save_dir.mkdir(parents = True, exist_ok = True) 
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    ax.yaxis.get_offset_text().set_fontsize(fontsize)     
    #ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    #ax.xaxis.get_offset_text().set_fontsize(fontsize)      
    plt.hist(corrcoef_autoenc_diff, bins = 75, label = 'The Ratio of Reconstruction Coefficient Difference\n (> 0): ' + '{:.3f}'.format(ratio_RR_high))
    plt.xlim([-0.3, 0.4])
    plt.xticks(np.arange(-0.3, 0.4, 0.1), fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xlabel("Reconstruction Coefficient", fontsize = fontsize)
    plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.15),
               fancybox = False, shadow = False, ncol = 1, fontsize = fontsize, frameon = False)
    plt.subplots_adjust(left = 0.08, right = 0.95, top = 0.95, bottom = 0.35, wspace = 0.2, hspace = 0.3)
    plt.savefig(save_dir.joinpath('The Ratio of High Reconstruction Coefficient Difference').with_suffix('.pdf'), bbox_inches = 'tight')
    plt.close()

corrcoef_autoenc_diff = corrcoef_autoenc_CORR - corrcoef_autoenc_MSE
ratio_bigger_0 = len(np.where(corrcoef_autoenc_diff > 0)[0]) / len(corrcoef_autoenc_diff)
plotHistReconCoeff(corrcoef_autoenc_diff = corrcoef_autoenc_diff, ratio_RR_high = ratio_bigger_0, \
                   save_dir = save_dir.joinpath("Paper"), fontsize = 40)

    
corr_PCA[np.argwhere(np.isnan(corr_PCA))[0][0]] = 0.99    
  
labels_pseudo_damage_0 = np.ones(len(scatter_factor))
labels_pseudo_damage_0[np.where(scatter_factor > 0)] = 2
fpr_0, tpr_0, thresholds_0 = metrics.roc_curve(labels_pseudo_damage_0, corrcoef_autoenc_CORR, pos_label = 2)

labels_pseudo_damage_2 = np.ones(len(scatter_factor))
labels_pseudo_damage_2[np.where(scatter_factor > 0.2)] = 2
fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(labels_pseudo_damage_2, corrcoef_autoenc_CORR, pos_label = 2)

labels_pseudo_damage_0_PCA = np.ones(len(scatter_factor))
labels_pseudo_damage_0_PCA[np.where((scatter_factor > 0) & (corr_PCA > 0.975))] = 2
fpr_0_PCA, tpr_0_PCA, thresholds_0_PCA = metrics.roc_curve(labels_pseudo_damage_0_PCA, corrcoef_autoenc_CORR, pos_label = 2)

labels_pseudo_damage_2_PCA = np.ones(len(scatter_factor))
labels_pseudo_damage_2_PCA[np.where((scatter_factor > 0.2) & (corr_PCA > 0.975))] = 2
fpr_2_PCA, tpr_2_PCA, thresholds_2_PCA = metrics.roc_curve(labels_pseudo_damage_2_PCA, corrcoef_autoenc_CORR, pos_label = 2)

plt.figure(1)
plt.plot(fpr_0, tpr_0, label = "damage: scatter > 0")
plt.plot(fpr_2, tpr_2, label = "damage: scatter > 0.2")
plt.plot(fpr_0_PCA, tpr_0_PCA, label = "damage: scatter > 0 and PCA RR > 0.975")
plt.plot(fpr_2_PCA, tpr_2_PCA, label = "damage: scatter > 0.2 and PCA RR > 0.975")
plt.legend(loc = "lower right")
plt.show()


'''-------------------------------------------------------------------------'''


reference = ["MSELoss"]; name_model = "CorrLoss"
   
load_dir = Path(args.direction_workspace, 'dataset', "autoencoder_corrcoeff", 
                args.sensors + " sensors", str(args.samples_signal), str(args.batch_file), 
                "ratio_downsample_" + str(args.ratio_downsample), str(args.ratio_downsample))

save_fig_dir = Path(args.direction_workspace, 'result', "autoencoder_corrcoeff", 
                    args.sensors + " sensors", str(args.samples_signal), str(args.batch_file), 
                    "ratio_downsample_" + str(args.ratio_downsample), str(args.ratio_downsample))  

if args.scatter_factor > 0:  
    load_dir = load_dir.joinpath(str(args.scatter_factor))
    save_fig_dir = save_fig_dir.joinpath(str(args.scatter_factor))               

#if self.args.threshold_data_type > 0:
load_dir = load_dir.joinpath(str(args.threshold_data_type))        
save_fig_dir = save_fig_dir.joinpath(str(args.threshold_data_type))

load_dir = load_dir.joinpath('_'.join(args.num_neurons[0]) + '_'.join(args.divider_set[0]))   
save_fig_dir = save_fig_dir.joinpath('_'.join(args.num_neurons[0]) + '_'.join(args.divider_set[0]))        

filename_save = 'CorrLoss_corrcoeff_autoenc_17560.pickle'

path_load = Path(load_dir).joinpath(filename_save)  
with open(path_load, 'rb') as handle:
    corrcoeff_dataset = pickle.load(handle)      
print(path_load, "has been loaded\n")   



try:
    corrcoef_dict = []
    for n in range(len(reference)):
        path_load = load_dir.joinpath(reference[n] + '_corrcoeff_autoenc_17560.pickle')
        print(path_load, "start to be loaded\n")
        with open(str(path_load), 'rb') as handle:
            corrcoef_autencoder_dataset = pickle.load(handle)         
            print(path_load, "has been created\n") 
        corrcoef_dict.append(corrcoef_autencoder_dataset['corrcoef'])
except TypeError:
    pass
                        
time_slice = 30000; fontsize =[40, 40]; epoch = 20
measurement_keywords = ["corrcoef", "temperature", "humidity", "brightness"]
measurement_labels = ['RC', 'T ['+u'\u2103]', 'H [%]', 'ln(B) [ln(CD)]']   
    
datatime = corrcoeff_dataset['datatime']; brightness = np.log(corrcoeff_dataset['brightness'])
temperature = corrcoeff_dataset['temperature']; humidity = corrcoeff_dataset['humidity']
corrcoef = corrcoeff_dataset['corrcoef']
 
for startpoint in range(0, len(corrcoef), time_slice):
    if (startpoint + time_slice) < len(corrcoef):
        endpoint = startpoint + time_slice
    else:
        endpoint = len(corrcoef)                                       
    index_xticks = np.percentile(np.arange(startpoint, endpoint), np.arange(0, 99, 24))
    index_xticks = np.array(list(map(int, index_xticks)))        
    plt.figure(figsize = (20, 20))
    for i in range(4):       
        measurement_v = eval(measurement_keywords[i])
        plt.subplot(4, 1, i+1)
        if i == 0:
            plt.scatter(datatime[startpoint:endpoint], measurement_v[startpoint:endpoint], \
                        alpha = 0.9, s = 15, label = name_model)
            try:
                for n in range(len(reference)):
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_dict[n][startpoint:endpoint], \
                                alpha = 0.9, s = 15, label = reference[n])
                ncol = len(reference) + 1
            except TypeError:
                ncol = 1  
            plt.legend(loc = 'upper center', scatterpoints = 10, fontsize = fontsize[1], bbox_to_anchor=(0.4, -4.26),
                       fancybox = False, shadow = False, ncol = ncol, frameon = False)                    
        else:
            plt.scatter(datatime[startpoint:endpoint], measurement_v[startpoint:endpoint], \
                        alpha = 0.9, s = 15)                    
                    
        plt.xlim((datatime[startpoint], datatime[endpoint-1]))
        plt.xticks(datatime[index_xticks], fontsize = fontsize[0], rotation = 0)
        plt.yticks(fontsize = fontsize[0], rotation = 0)
        plt.ylabel(measurement_labels[i], fontsize = fontsize[1])
        #plt.title(measurement_keywords[i].split('_')[1], fontsize = 28)     
        if i == 1:
            if np.mean(measurement_v[startpoint:endpoint]) < 10:
                plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
    plt.xlabel("Measurement Time [YY:MM:DD]", fontsize = fontsize[1])            
    plt.subplots_adjust(left = 0.11, right = 0.95, top = 0.99, bottom = 0.14, wspace = 0.3, hspace = 0.3)
    plt.savefig(save_fig_dir.joinpath(name_model + '_' + str(epoch) + '_ratio_downsample_' + str(args.ratio_downsample) + \
                                      ' ' + str(datatime[startpoint])).with_suffix('.png'), bbox_inches = 'tight')      
    plt.close()   




