# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:11:55 2019

@author: Administrator
"""

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch import optim
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F  
import pickle
import sys
import glob
sys.path.append('/home/UFAD/yang.kang/Ultrasonics/Kang/Feature Data/PCA/autoencoder/code')
import random
import os 
import create_dataset_for_autoencoder

###############################################################################
# user defined function
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combine_dataset_batch(load_dir, name_model, args):
     
    list_pickle_file = glob.glob(str(load_dir) + '/*.pickle')
    list_pickle_file.sort()      

    index_file = []
    for i in range(len(list_pickle_file)):
        if name_model in list_pickle_file[i]:
            index_file.append(i)
    index_file = np.array(index_file)           
    list_pickle_file = np.array(list_pickle_file)[index_file]

    corrcoef = np.empty((0))
    scatter_factor = np.empty((0)); file_selected = np.empty((0)); datatime = np.empty((0))
    temperature = np.empty((0)); pressure = np.empty((0)); brightness = np.empty((0))
    humidity = np.empty((0)); tag = np.empty((0))
    
    for n in tqdm(range(len(list_pickle_file))):
    
        with open(list_pickle_file[n] , 'rb') as file:
            corrcoeff_dataset = pickle.load(file)
        print(list_pickle_file[n])
        
        corrcoef = np.concatenate((corrcoef, corrcoeff_dataset['corrcoef']), axis = 0)
        file_selected = np.concatenate((file_selected, corrcoeff_dataset['file_selected']), axis = 0)
        datatime = np.concatenate((datatime, corrcoeff_dataset['datatime']), axis = 0)
        temperature = np.concatenate((temperature, corrcoeff_dataset['temperature']), axis = 0) 
        pressure = np.concatenate((pressure, corrcoeff_dataset['pressure']), axis = 0)
        brightness = np.concatenate((brightness, corrcoeff_dataset['brightness']), axis = 0)
        humidity = np.concatenate((humidity, corrcoeff_dataset['humidity']), axis = 0) 
        tag = np.concatenate((tag, corrcoeff_dataset['tag']), axis = 0)      
        if args.scatter_factor > 0:
            scatter_factor = np.concatenate((scatter_factor, corrcoeff_dataset['scatter_factor']), axis = 0)
       
    corrcoef_autencoder_dataset = {'file_selected': file_selected, \
                                   'corrcoef': corrcoef,\
                                   'datatime': datatime,\
                                   'temperature':temperature,\
                                   'pressure':pressure,\
                                   'brightness':brightness,\
                                   'humidity':humidity,\
                                   'tag':tag}
    if args.scatter_factor > 0:
         corrcoef_autencoder_dataset.update( {'scatter_factor': scatter_factor} )
    
    number_file = list_pickle_file[-1].split("_")[-1].split(".")[0]
    filename_save = name_model + '_corrcoef_autoenc_' + number_file + '.pickle'
    save_dir = load_dir.joinpath("combined dataset")
    
    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = Path(save_dir).joinpath(filename_save)
    
    with open(path_save, 'wb') as handle:
        pickle.dump(corrcoef_autencoder_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
        print(path_save, "has been created\n")        
    
    return corrcoef_autencoder_dataset


def cal_corrcoeff(dataset, dataset_reconstr):
    
    corrcoef = np.corrcoef(dataset, dataset_reconstr, rowvar = True)
    corrcoef = np.diagonal(corrcoef[dataset.shape[0]:, :dataset.shape[0]])
    
    return corrcoef


class TrainAutoEncoder(nn.Module):
      
    def __init__(self, args, model, optimizer, criterion, name_model):
        super(TrainAutoEncoder, self).__init__()
        
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.name_model = name_model
                
    def train(self, batch):
        
        with torch.enable_grad():
            # Turn on training mode which enables dropout.
            self.model.train()
            b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
            encoded, decoded = self.model(b_x)
            #encoded, decoded = model(b_x)
            loss = self.criterion(decoded, b_x).float()
            self.optimizer.zero_grad()               # clear gradients for this training step        
            loss.backward()                     # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()                    # apply gradients                
    
        return loss.detach().cpu().numpy() 
    
    def evaluate(self, batch):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        with torch.no_grad():
            b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
            encoded, decoded = self.model(b_x)
            loss = self.criterion(decoded, b_x)
        
        return loss.detach().cpu().numpy() 

    def cal_corrcoeff(self, gen_dataset, norm_scale):
        
        try:
            dataset = gen_dataset['data_sonic']
        except KeyError:
            dataset = gen_dataset['dataset_sonic']
            
        batch_size_generate = 1024
        num_data_T = int(len(dataset)/self.args.cuda_devices) * self.args.cuda_devices
        dataset = dataset[:num_data_T]; dataset = (dataset - norm_scale[0][0])/( norm_scale[0][1]- norm_scale[0][0])
        data_input = torch.from_numpy(dataset); data_label = torch.from_numpy(gen_dataset['tag'])             
        genout_data = torch.utils.data.TensorDataset(data_input, data_label)
        genout_loader = torch.utils.data.DataLoader(genout_data, batch_size = batch_size_generate, shuffle = False)        

        corrcoef = np.empty((0))         
        self.model.eval()
        with torch.no_grad():       
            for i, batch in tqdm(list(enumerate(genout_loader))):          
                b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
                encoded, decoded = self.model(b_x)         
                temp_corrcoef = np.corrcoef(b_x.detach().cpu().numpy(), decoded.detach().cpu().numpy(), rowvar = True)                
                temp_corrcoef = np.diagonal(temp_corrcoef[b_x.shape[0]:, :b_x.shape[0]])
                corrcoef = np.concatenate((corrcoef, temp_corrcoef), axis = 0)
 
        np.set_printoptions(precision = 5)        
        print(f"\n correlation coefficents between orginal data and reconstructed data \n {corrcoef[:100]}")

        corrcoeff_dataset = {'datatime': gen_dataset['datatime'],
                             'temperature': gen_dataset['temperature'],
                             'pressure': gen_dataset['pressure'],
                             'brightness': gen_dataset['brightness'],
                             'humidity': gen_dataset['humidity'],
                             'tag': gen_dataset['tag'], 
                             'corrcoef': corrcoef, 
                             'file_selected':gen_dataset['file_selected']}
        try: 
            corrcoeff_dataset['scatter_factor'] = gen_dataset['scatter_factor']
            print(gen_dataset['scatter_example'])
        except KeyError:
            print("no scatter factor")
        
        return corrcoeff_dataset


    def generate_output_sample(self, epoch, gen_dataset, norm_scale, reference):
     
        corrcoeff_dataset = self.cal_corrcoeff(gen_dataset, norm_scale)
   
        save_dir = Path(self.args.direction_workspace, 'dataset', "autoencoder_corrcoeff", 
                        self.args.sensors + " sensors", str(self.args.samples_signal), str(self.args.batch_file), 
                        "ratio_downsample_" + str(self.args.ratio_downsample), str(self.args.ratio_downsample))
        
        save_fig_dir = Path(self.args.direction_workspace, 'result', "autoencoder_corrcoeff", 
                            self.args.sensors + " sensors", str(self.args.samples_signal), str(self.args.batch_file), 
                            "ratio_downsample_" + str(self.args.ratio_downsample), str(self.args.ratio_downsample))  
    
        if self.args.scatter_factor > 0:  
            save_dir = save_dir.joinpath(str(self.args.scatter_factor))
            save_fig_dir = save_fig_dir.joinpath(str(self.args.scatter_factor))               
    
        #if self.args.threshold_data_type > 0:
        save_dir = save_dir.joinpath(str(self.args.threshold_data_type))        
        save_fig_dir = save_fig_dir.joinpath(str(self.args.threshold_data_type))
    
        save_dir = save_dir.joinpath('_'.join(self.args.num_neurons[0]) + '_'.join(self.args.divider_set[0]))   
        save_fig_dir = save_fig_dir.joinpath('_'.join(self.args.num_neurons[0]) + '_'.join(self.args.divider_set[0]))        
    
        number_file = gen_dataset['file_selected'][-1].split("_")[-1].split(".")[0]
        filename_save = self.name_model + '_corrcoeff_autoenc_' + number_file + '.pickle'
    
        save_dir.mkdir(parents = True, exist_ok = True)
        path_save = Path(save_dir).joinpath(filename_save)  
        with open(path_save, 'wb') as handle:
            pickle.dump(corrcoeff_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
        print(path_save, "has been created\n")     
     
        save_fig_dir.mkdir(parents = True, exist_ok = True)
        self.plotReCoeff(epoch = epoch, save_dir = save_dir, save_fig_dir = save_fig_dir, corrcoeff_dataset = corrcoeff_dataset, reference = reference, fontsize = [40, 40])   
    

    def generate_output(self, epoch, gen_dataset, norm_scale):
        
        corrcoeff_dataset = self.cal_corrcoeff(gen_dataset, norm_scale)   
        
        save_dir = Path(self.args.direction_workspace, 'dataset', "autoencoder_corrcoeff", self.args.sensors + " sensors", 
                        str(self.args.samples_signal), str(self.args.batch_file), 
                        "ratio_downsample_" + str(self.args.ratio_downsample))
        if self.args.scatter_factor > 0:
            save_dir = save_dir.joinpath(str(self.args.scatter_factor))
        #if self.args.threshold_data_type > 0:
        save_dir = save_dir.joinpath(str(self.args.threshold_data_type)).joinpath('_'.join(self.args.num_neurons[0]) + '_'.join(self.args.divider_set[0]))          
                   
        number_file = gen_dataset['file_selected'][-1].split("_")[-1].split(".")[0]
        filename_save = self.name_model + '_corrcoef_autoencoder_' + number_file + '.pickle'
    
        save_dir.mkdir(parents = True, exist_ok = True)
        path_save = Path(save_dir).joinpath(filename_save)
        
        with open(path_save, 'wb') as handle:
            pickle.dump(corrcoeff_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
        print(path_save, "has been created\n")    


    def plotReCoeff(self, epoch, save_dir, save_fig_dir, corrcoeff_dataset, reference = None, fontsize = [20, 20]):
        
        try:
            corrcoef_dict = []
            for n in range(len(reference)):
                path_load = save_dir.joinpath(reference[n] + '_corrcoeff_autoenc_17560.pickle')
                print(path_load, "start to be loaded\n")
                with open(path_load , 'rb') as handle:
                    corrcoef_autencoder_dataset = pickle.load(handle)         
                    print(path_load, "has been created\n") 
                corrcoef_dict.append(corrcoef_autencoder_dataset['corrcoef'])
        except TypeError:
            pass
                                
        time_slice = 30000
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
                                alpha = 0.9, s = 15, label = self.name_model)
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
            plt.savefig(save_fig_dir.joinpath(self.name_model + '_' + str(epoch) + '_ratio_downsample_' + str(self.args.ratio_downsample) + \
                                              ' ' + str(datatime[startpoint])).with_suffix('.png'), bbox_inches = 'tight')      
            plt.close()                

    def generate_figure(self):
    
        filename_save = self.name_model + '_corrcoef_autoenc_17560.pickle'  
        load_dir = Path(self.args.direction_workspace, 'dataset', "autoencoder_corrcoeff", 
                        self.args.sensors + " sensors", str(self.args.samples_signal), str(self.args.batch_file), 
                        "ratio_downsample_" + str(self.args.ratio_downsample))
        save_dir = Path(self.args.direction_workspace, 'result', "autoencoder_corrcoeff", self.args.sensors + " sensors", \
                        str(self.args.samples_signal), str(self.args.batch_file), "ratio_downsample_" + str(self.args.ratio_downsample))
    
        if self.args.scatter_factor > 0:
            load_dir = load_dir.joinpath(str(self.args.scatter_factor))
            save_dir = save_dir.joinpath(str(self.args.scatter_factor))
        #if self.args.threshold_data_type > 0:
        load_dir = load_dir.joinpath(str(self.args.threshold_data_type)) 
        save_dir = save_dir.joinpath(str(self.args.threshold_data_type))  
    
        load_dir = load_dir.joinpath('_'.join(self.args.num_neurons[0]) + '_'.join(self.args.divider_set[0])).joinpath("combined dataset")
        path_load = Path(load_dir).joinpath(filename_save)
        save_dir = save_dir.joinpath('_'.join(self.args.num_neurons[0]) + '_'.join(self.args.divider_set[0]))
        save_dir.mkdir(parents = True, exist_ok = True)  
            
        if os.path.isfile(str(path_load)):  
            print(path_load, "start to be loaded\n")
            with open(path_load , 'rb') as handle:
                corrcoef_autencoder_dataset = pickle.load(handle)         
            print(path_load, "has been created\n") 
        else:
            load_dir = load_dir.parent
            corrcoef_autencoder_dataset = combine_dataset_batch(load_dir, self.name_model, self.args)            
            
        if self.args.scatter_factor > 0:
            measurement_keywords = ["corrcoef", "scatter_factor", "temperature", "humidity"]
            measurement_labels = ['RC','Scatter' ,'T ['+u'\u2103]', 'H [%]']
        else:
            measurement_keywords = ["corrcoef", "temperature", "humidity", "brightness"]
            measurement_labels = ['RC', 'T ['+u'\u2103]', 'H [%]', 'ln(B) [ln(CD)]']
        
        datatime = corrcoef_autencoder_dataset["datatime"]
        index_shuffle = list(range(len(datatime))); index_shuffle = np.array(index_shuffle)
        ratio_train = list(map(float, self.args.divider_set[0]))
        divider_set = create_dataset_for_autoencoder.create_dataset_divider(self.args, index_shuffle, ratio_train = ratio_train)
    
        time_slice = 300000
        fontsize = [40, 40]    
        for startpoint in range(0, len(datatime), time_slice):
            if (startpoint + time_slice) < len(datatime):
                endpoint = startpoint + time_slice
            else:
                endpoint = len(datatime)              
                                     
            index_xticks = np.percentile(np.arange(startpoint, endpoint), np.arange(0, 99, 24))
            index_xticks = np.array(list(map(int, index_xticks)))        
            plt.figure(figsize = (20, 20))
            for i in range(4):       
                measurement_v = measurement_keywords[i]
                plt.subplot(4, 1, i+1)
                
                if endpoint < divider_set[2]:
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_autencoder_dataset[measurement_v][startpoint:endpoint], \
                                alpha = 0.9, s = 15, label = "Fair--Train Data")
                elif startpoint > divider_set[2]:
                    plt.scatter(datatime[startpoint:endpoint], corrcoef_autencoder_dataset[measurement_v][startpoint:endpoint], \
                                alpha = 0.9, s = 15, c = 'c', label = "Fair--Test Data") 
                else:
                    plt.scatter(datatime[startpoint: divider_set[2]], corrcoef_autencoder_dataset[measurement_v][startpoint: divider_set[2]], \
                                alpha = 0.9, s = 15, label = "Fair--Train Data") 
                    plt.scatter(datatime[divider_set[2]: endpoint], corrcoef_autencoder_dataset[measurement_v][divider_set[2]: endpoint], \
                                alpha = 0.9, s = 15, c = 'c', label = "Fair--Test Data") 
                                                     
                plt.xlim((datatime[startpoint], datatime[endpoint-1]))
                plt.xticks(datatime[index_xticks], fontsize = fontsize[0], rotation = 0)
                plt.yticks(fontsize = fontsize[0], rotation = 0)
                plt.ylabel(measurement_labels[i], fontsize = fontsize[1])  
                if measurement_keywords[i] == "temperature":
                    if np.mean(corrcoef_autencoder_dataset[measurement_v][startpoint:endpoint]) < 10:
                        plt.scatter(datatime[startpoint:endpoint], np.zeros(len(datatime[startpoint:endpoint])), color = 'blue', alpha = 0.5, s = 0.5)
            plt.xlabel("Measurement Time [YY:MM:DD]", fontsize = fontsize[1])
            plt.legend(loc = 'upper center', scatterpoints = 10, fontsize = fontsize[1], bbox_to_anchor=(0.4, -0.26),
                       fancybox = False, shadow = False, ncol = 1, frameon = False)
            
            plt.subplots_adjust(left = 0.11, right = 0.95, top = 0.99, bottom = 0.14, wspace = 0.3, hspace = 0.3)
            plt.savefig(save_dir.joinpath(self.name_model + '_scatter_' + '_ratio_downsample_' + str(self.args.ratio_downsample) + \
                                          ' ' + str(datatime[startpoint])).with_suffix('.png'), bbox_inches = 'tight')      
            plt.close()        
            
 
    
 
def train_DNN(args, model, optimizer, criterion, batch):

    with torch.enable_grad(): # Turn on training mode which enables dropout.
        model.train()
        b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
        if args.model_function == "detect_mass":
            b_y = batch[1].type(torch.LongTensor).to(device)
        else:
            b_y = batch[1].to(device).float()
        out = model(b_x)
        loss = criterion(out, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step        
        loss.backward()                     # backpropagation, compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()                    # apply gradients
           
    cur_loss = loss.detach().cpu().numpy()    
    return cur_loss

def evaluate_DNN(args, model, criterion, batch):
    
    model.eval()
    with torch.no_grad():
        b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
        if args.model_function == "detect_mass":
            b_y = batch[1].type(torch.LongTensor).to(device)
        else:
            b_y = batch[1].to(device).float()
        out = model(b_x)
        loss = criterion(out, b_y)      # mean square error   
        #print(out.detach().cpu().numpy().T)
    
    for i in range(out.shape[1]):
        temp_corrcoef = np.corrcoef(b_y[:, i].cpu().data.numpy().squeeze(), out[:, i].cpu().data.numpy().squeeze())[0][1]    
        print(f"{i+1}: correlation coefficents between outputs and labels \n {temp_corrcoef}")  
    
    return loss.detach().cpu().numpy() 
                     
def generate_output_DNN(args, epoch, model, criterion, device, gen_dataset, norm_scale):

    type_input = list(eval(args.type_input))
    batch_size_generate = 1024    
    data_input = gen_dataset['data_sonic'] 
    data_input = torch.from_numpy((data_input - norm_scale[0][1])/(norm_scale[0][0] - norm_scale[0][1]))
    num_data_T = int(len(data_input)/args.cuda_devices) * args.cuda_devices
    data_input = data_input[:num_data_T]
    
    if args.model_function == "detect_mass":   
        data_label = gen_dataset['tag']
    else:
        data_label = np.array([gen_dataset['temperature'][:num_data_T], gen_dataset['humidity'][:num_data_T], \
                               gen_dataset['brightness'][:num_data_T], gen_dataset['pressure'][:num_data_T]]) 
        data_label[2] = np.log(data_label[2])  
        for i in range(len(type_input)):
            data_label[type_input[i]] = (data_label[type_input[i]] - norm_scale[type_input[i]+1][1])/(norm_scale[type_input[i]+1][0] - norm_scale[type_input[i]+1][1])
    data_label = data_label.transpose(1, 0)
    data_label = torch.from_numpy(data_label)  
    
    genout_data = torch.utils.data.TensorDataset(data_input, data_label)
    genout_loader = torch.utils.data.DataLoader(genout_data, batch_size = batch_size_generate, shuffle = False)

    model.eval()
    with torch.no_grad():            
        if args.model_function == "detect_mass":  
            y_predict = np.empty((0)) 
            for i, batch in tqdm(list(enumerate(genout_loader))):          
                b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
                out = model(b_x)
                b_y_predict = torch.max(F.softmax(out), 1)[1]
                b_y_predict = b_y_predict.cpu().data.numpy().squeeze()
                b_y = batch[1].numpy()
                y_predict = np.concatenate((y_predict, b_y_predict), axis = 0)
            print(f"confusion matrix for detecting mass: \n {confusion_matrix(b_y, b_y_predict)}\n")                                          
        else:
            y_predict = np.empty((0, len(type_input))) 
            for i, batch in tqdm(list(enumerate(genout_loader))):          
                b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
                out = model(b_x)             
                b_y_predict = out.cpu().data.numpy().squeeze()
                b_y = batch[1].numpy()
                y_predict = np.concatenate((y_predict, b_y_predict), axis = 0)
                
            for i in range(b_y_predict.shape[1]):
                temp_corrcoef = np.corrcoef(b_y[:, i], b_y_predict[:, i])[0][1]    
                print(f"\n{i+1}: correlation coefficents between outputs and labels \n {temp_corrcoef}")   
                y_predict[:, i] =  y_predict[:, i] * (norm_scale[type_input[i]+1][0]- norm_scale[type_input[i]+1][1]) + norm_scale[type_input[i]+1][1]

    if args.scatter_factor > 0:
        y_predict_dataset = {'file_selcted': gen_dataset['file_selcted'],
                             'y_predict': y_predict,
                             'scatter_factor': gen_dataset['scatter_factor']}
        save_dir = Path(args.direction_workspace, 'dataset', args.model_function + "_predict", args.sensors + " sensors", str(args.batch_file), \
                        "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str, str(args.scatter_factor)).joinpath(str(batch[0].shape[1]))

    else:
        y_predict_dataset = {'file_selcted': gen_dataset['file_selcted'],
                             'y_predict': y_predict}  
        save_dir = Path(args.direction_workspace, 'dataset', args.model_function + "_predict", args.sensors + " sensors", str(args.batch_file), \
                        "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str).joinpath(str(batch[0].shape[1]))
            
    number_file = gen_dataset['file_selcted'][-1].split("_")[-1].split(".")[0]
    filename_save = 'y_predict_' + number_file + '.pickle'
       
    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = Path(save_dir).joinpath(filename_save)
    
    with open(path_save, 'wb') as handle:
        pickle.dump(y_predict_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n")   
        
def generate_output_DNN_piece(args, epoch, model, criterion, device, gen_dataset, norm_scale):

    batch_size_generate = 512
    type_input = np.array(eval(args.type_input))
    dataset_valid = gen_dataset["dataset_valid"] 
    datatime_valid = gen_dataset["datatime_valid"]
    corrcoef_valid = gen_dataset["corrcoef_valid"]
    label_valid = dataset_valid[:, args.samples_signal:]
    
    dataset_valid = torch.from_numpy(dataset_valid)
    val_data = torch.utils.data.TensorDataset(dataset_valid[:, :args.samples_signal], dataset_valid[:, args.samples_signal:])
    genout_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size_generate, shuffle = False)    
    measurement_keywords = ['corrcoeff', 'temperature', 'humidity', 'brightness', 'pressure']
    model.eval()
    with torch.no_grad():            
        if args.model_function == "detect_mass":  
            y_predict = np.empty((0)) 
            for i, batch in tqdm(list(enumerate(genout_loader))):          
                b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
                out = model(b_x)
                b_y_predict = torch.max(F.softmax(out), 1)[1]
                b_y_predict = b_y_predict.cpu().data.numpy().squeeze()
                y_predict = np.concatenate((y_predict, b_y_predict), axis = 0)
            b_y = batch[1].numpy()
            print(f"confusion matrix for detecting mass: \n {confusion_matrix(b_y, b_y_predict)}\n")                                          
        else:
            y_predict = np.empty((0, len(type_input))) 
            for i, batch in tqdm(list(enumerate(genout_loader))):          
                b_x = batch[0].view(-1, batch[0].shape[1]).to(device).float()   # batch x, shape (batch, 28*28)
                out = model(b_x)             
                b_y_predict = out.cpu().data.numpy().squeeze()
                y_predict = np.concatenate((y_predict, b_y_predict), axis = 0)
            
            b_y = batch[1].numpy()
            for i in range(b_y_predict.shape[1]):
                temp_corrcoef = np.corrcoef(b_y[:, i], b_y_predict[:, i])[0][1]    
                print(f"correlation coefficents between outputs and labels \n {temp_corrcoef}")   
                y_predict[:, i] =  y_predict[:, i] * (norm_scale[type_input[i]+1][0]- norm_scale[type_input[i]+1][1]) + norm_scale[type_input[i]+1][1]
                label_valid[:, i] =  label_valid[:, i] * (norm_scale[type_input[i]+1][0]- norm_scale[type_input[i]+1][1]) + norm_scale[type_input[i]+1][1]

    save_dir = Path(args.direction_workspace, 'result', args.model_function + "_predict", args.sensors + " sensors", 'time_fragment', \
                    "ratio_downsample_" + str(args.ratio_downsample), args.type_input_str).joinpath(str(args.samples_signal))
    save_dir.mkdir(parents = True, exist_ok = True)  
    
    plt.figure(figsize = (20, 10))
    for i in range(len(type_input)+1):       
        plt.subplot(4, 1, i+1)
        if i == 0:
            plt.scatter(datatime_valid, corrcoef_valid, alpha = 1, s = 15)
            plt.ylabel(measurement_keywords[i], fontsize = 15)
        else:
            plt.scatter(datatime_valid, label_valid[:, i-1], \
                        alpha = 1, s = 15, label = "orginal")        
            plt.scatter(datatime_valid, y_predict[:, i-1], \
                        alpha = 1, s = 10, label = "predicted", color = 'orange')            
            plt.ylabel(measurement_keywords[type_input[i-1]+1], fontsize = 15)
        plt.xlim((datatime_valid[0], datatime_valid[-1]))
        plt.xticks(fontsize = 15, rotation = 0)
        plt.yticks(fontsize = 15, rotation = 0)        
        #plt.title(measurement_keywords[i].split('_')[1], fontsize = 28)     
        plt.xlabel("measurement time", fontsize = 15)
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.93, bottom = 0.07, wspace = 0.2, hspace = 0.3)
    plt.legend(loc = 'lower right', scatterpoints = 10, fontsize = 16)
    plt.suptitle('the predicted envrionment information', \
                 fontsize = 24, x = 0.5, y = 0.99)
    plt.savefig(save_dir.joinpath('ratio_downsample_' + str(args.ratio_downsample) + '_' + str(datatime_valid[0]) 
    + '_' + 'envrionment').with_suffix('.png'))
    plt.close() 